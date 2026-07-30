"""Re-test every previously rejected method with a statistic that can measure it.

All eleven earlier rejections used the worst adequately powered fold on the
1,103-record set. That statistic was later shown to move by 0.030 under
perturbations carrying no information at all, a leaked timing column and a
StandardScaler in front of a tree model, while the mean across powered folds
moved by 0.0014. Every one of those methods was therefore judged with an
instrument coarser than the effects it was measuring.

This re-runs them on 6,600 records, where all four folds are powered, and ranks
on the mean. The mean is also the better estimator for this competition: the
hidden set is one unseen site, so expected performance on a random held-out
site is the quantity of interest rather than a pessimistic floor.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.feature_io import load_features
from src.eval import challenge_metrics as cm
from src.eval import protocol
from src.models.age_adjust import AgeResidualizer
from src.models.matched_weights import matched_pair_weights

df, cols = load_features('data/processed/features_large_v9.pkl', drop_cols=['bmi'])
X = df[cols].to_numpy(float); y = df.label.to_numpy(float)
sites = df.site_id.to_numpy(); ages = df.age.to_numpy(float)
prev = cm.prevalence_by_age(ages, y, ages, gap=2)
age_idx = cols.index('age') if 'age' in cols else None
print(f'{len(df)} records, {len(cols)} features, age included\n', flush=True)

def gbm():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=50, colsample_bytree=0.5,
                              reg_lambda=1.0, random_state=42, verbose=-1)

def base(model_factory, residualize=False, matched=False, temper=1.0,
         site_balanced=True, scale=True):
    def f(Xtr, ytr, str_, atr, Xte, ate):
        if residualize:
            r = AgeResidualizer().fit(Xtr, atr, ytr)
            Xtr, Xte = r.transform(Xtr, atr), r.transform(Xte, ate)
        steps = [('i', SimpleImputer(strategy='median'))]
        if scale: steps.append(('s', StandardScaler()))
        steps.append(('m', model_factory()))
        p = Pipeline(steps)
        w = None
        if matched: w = matched_pair_weights(ytr, atr, temper=temper)
        if site_balanced:
            c = pd.Series(str_).value_counts()
            sw = np.array([1.0/c[s] for s in str_]); sw = sw/sw.mean()
            w = sw if w is None else w*sw
        kw = {'m__sample_weight': w/w.mean()} if w is not None else {}
        p.fit(Xtr, ytr, **kw)
        return p.predict_proba(Xte)[:, 1]
    return f

def drop_col(fn, idx):
    def f(Xtr, ytr, str_, atr, Xte, ate):
        keep = [i for i in range(Xtr.shape[1]) if i != idx]
        return fn(Xtr[:, keep], ytr, str_, atr, Xte[:, keep], ate)
    return f

def site_covariate(fn):
    def f(Xtr, ytr, str_, atr, Xte, ate):
        codes = {s: i for i, s in enumerate(sorted(pd.unique(str_)))}
        tr = np.column_stack([Xtr, [codes[s] for s in str_]])
        te = np.column_stack([Xte, np.full(len(Xte), len(codes))])
        return fn(tr, ytr, str_, atr, te, ate)
    return f

lr = lambda C: (lambda: LogisticRegression(C=C, max_iter=2000, random_state=42))
methods = {
  'lgbm (reference)':            base(gbm),
  'lgbm, age dropped':           drop_col(base(gbm), age_idx) if age_idx is not None else None,
  'lgbm, age residualized':      base(gbm, residualize=True),
  'lgbm, matched weights a=1':   base(gbm, matched=True, temper=1.0),
  'lgbm, matched weights a=.5':  base(gbm, matched=True, temper=0.5),
  'lgbm, site as covariate':     site_covariate(base(gbm)),
  'lgbm, no site balancing':     base(gbm, site_balanced=False),
  'lr C=0.005':                  base(lr(0.005)),
  'lr C=0.005, age dropped':     drop_col(base(lr(0.005)), age_idx) if age_idx is not None else None,
  'lr C=0.05':                   base(lr(0.05)),
  'lr C=0.001':                  base(lr(0.001)),
}
methods = {k: v for k, v in methods.items() if v is not None}

print(f"{'method':32s} {'mean':>7s} {'worst':>7s} {'delta_mean':>11s}  {'secs':>5s}")
ref = None
rows = []
for name, fn in methods.items():
    t = time.time()
    try:
        folds = protocol.evaluate(fn, X, y, sites, ages, prevalence=prev,
                                  n_boot=200, verbose=False)
    except Exception as e:
        print(f'{name:32s} FAILED {type(e).__name__}: {str(e)[:50]}', flush=True); continue
    m = protocol.mean_powered(folds)
    w = folds[folds.powered].auroc_age.min()
    if ref is None: ref = m
    rows.append((name, m, w))
    print(f'{name:32s} {m:7.4f} {w:7.4f} {m-ref:+11.4f}  {time.time()-t:5.0f}',
          flush=True)

print('\nranked by mean (stable to ~0.0014):')
for name, m, w in sorted(rows, key=lambda r: -r[1]):
    print(f'  {name:32s} {m:.4f}')
