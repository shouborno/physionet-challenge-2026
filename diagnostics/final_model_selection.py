"""The comparison that decides what we submit.

Every earlier TabFM result was measured against site-balanced LightGBM, which
the re-test showed costs 0.0177. So TabFM's apparent +0.028 was partly a
handicap on its opponent, and the honest comparison is against the unbalanced
baseline with site as a covariate, which is the strongest configuration found.

Also tests the blend. TabFM and LightGBM disagree by fold in a way that usually
averages well, and rank-averaging is the right combiner since the two are
calibrated differently and the metric only reads order.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/scratch/simran/pn26/tabfm')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, torch, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.feature_io import load_features
from src.eval import challenge_metrics as cm
from src.eval import protocol

df, allcols = load_features('data/processed/features_large_v9.pkl', drop_cols=['bmi'])
X = df[allcols].to_numpy(float); y = df.label.to_numpy(float)
sites = df.site_id.to_numpy(); ages = df.age.to_numpy(float)
prev = cm.prevalence_by_age(ages, y, ages, gap=2)
print(f'{len(df)} records, {len(allcols)} features\n', flush=True)

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as hub
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = hub.load('classification', device=dev)
print(f'TabFM on {dev}\n', flush=True)

def add_site(Xtr, Xte, str_):
    codes = {s: i for i, s in enumerate(sorted(pd.unique(str_)))}
    return (np.column_stack([Xtr, [codes[s] for s in str_]]),
            np.column_stack([Xte, np.full(len(Xte), len(codes))]))

def lgbm(site_cov=False, balanced=False):
    def f(Xtr, ytr, str_, atr, Xte, ate):
        if site_cov: Xtr, Xte = add_site(Xtr, Xte, str_)
        p = Pipeline([('i', SimpleImputer(strategy='median')),
                      ('s', StandardScaler()),
                      ('m', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                          num_leaves=15, min_child_samples=50, colsample_bytree=0.5,
                          reg_lambda=1.0, random_state=42, verbose=-1))])
        kw = {}
        if balanced:
            c = pd.Series(str_).value_counts()
            w = np.array([1.0/c[s] for s in str_]); kw['m__sample_weight'] = w/w.mean()
        p.fit(Xtr, ytr, **kw)
        return p.predict_proba(Xte)[:, 1]
    return f

def tabfm(n_estimators=32, site_cov=False):
    def f(Xtr, ytr, str_, atr, Xte, ate):
        if site_cov: Xtr, Xte = add_site(Xtr, Xte, str_)
        imp = SimpleImputer(strategy='median')
        Ztr = imp.fit_transform(Xtr); Zte = imp.transform(Xte)
        c = TabFMClassifier(MODEL, n_estimators=n_estimators, max_num_features=500,
                            n_svd_features='sqrt', random_state=42)
        c.fit(Ztr, ytr.astype(int))
        return c.predict_proba(Zte)[:, 1]
    return f

def blend(a, b, wa):
    def f(*args):
        ra = rankdata(a(*args)); rb = rankdata(b(*args)); n = len(ra)
        return wa*ra/n + (1-wa)*rb/n
    return f

best_lgbm = lgbm(site_cov=True, balanced=False)
best_tabfm = tabfm(32, site_cov=True)

cases = {
  'lgbm balanced (old baseline)':  lgbm(False, True),
  'lgbm unbalanced':               lgbm(False, False),
  'lgbm unbalanced + site cov':    best_lgbm,
  'tabfm n32':                     tabfm(32, False),
  'tabfm n32 + site cov':          best_tabfm,
  'blend 50/50 (best of each)':    blend(best_tabfm, best_lgbm, 0.5),
  'blend 70/30 tabfm-heavy':       blend(best_tabfm, best_lgbm, 0.7),
  'blend 30/70 lgbm-heavy':        blend(best_tabfm, best_lgbm, 0.3),
}
print(f"{'model':34s} {'mean':>7s} {'worst':>7s}  {'per-fold':<44s} {'s':>4s}")
for name, fn in cases.items():
    t = time.time()
    try:
        folds = protocol.evaluate(fn, X, y, sites, ages, prevalence=prev,
                                  n_boot=300, verbose=False)
    except Exception as e:
        print(f'{name:34s} FAILED {type(e).__name__}: {str(e)[:50]}', flush=True); continue
    per = ' '.join(f'{r.auroc_age:.3f}' for _, r in folds.iterrows())
    print(f'{name:34s} {protocol.mean_powered(folds):7.4f} '
          f'{folds[folds.powered].auroc_age.min():7.4f}  {per:<44s} {time.time()-t:4.0f}',
          flush=True)
