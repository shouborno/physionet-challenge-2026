"""Improve TabFM, and test whether it complements LightGBM rather than replacing it.

Three things worth trying.

The first is undoing a handicap of mine: n_estimators defaults to 32 and the
earlier runs used 8, chosen for speed. TabFM's ensembling is over feature
shuffles and normalizations, so that quarter-strength setting plausibly cost
real accuracy.

The second is its built-in preprocessing levers, which exist precisely for wide
inputs: SVD reduction to the 500-feature cap, feature crosses, and the
normalization method.

The third is complementarity. TabFM and LightGBM have opposite error profiles:
TabFM wins on I0002, I0006 and the inverted fold while LightGBM wins on S0001,
the largest. Models that disagree that cleanly usually average well. Scores are
combined by rank rather than probability, since the two are on different scales
and AUROC only cares about order.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/scratch/simran/pn26/tabfm')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, torch, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from src.data.feature_io import load_features
from src.eval import challenge_metrics as cm
from src.eval import protocol

df, cols = load_features('data/processed/features_large_v9.pkl',
                         drop_age=True, drop_cols=['bmi'])
X = df[cols].to_numpy(float); y = df.label.to_numpy(float)
sites = df.site_id.to_numpy(); ages = df.age.to_numpy(float)
prev = cm.prevalence_by_age(ages, y, ages, gap=2)
print(f'{len(df)} records, {len(cols)} features\n', flush=True)

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as hub
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = hub.load('classification', device=dev)
print(f'TabFM loaded on {dev}\n', flush=True)

def tabfm_fit(**kw):
    def f(Xtr, ytr, str_, atr, Xte, ate):
        imp = SimpleImputer(strategy='median')
        Ztr = imp.fit_transform(Xtr); Zte = imp.transform(Xte)
        clf = TabFMClassifier(MODEL, random_state=42, **kw)
        clf.fit(Ztr, ytr.astype(int))
        return clf.predict_proba(Zte)[:, 1]
    return f

def lgbm_fit():
    def f(Xtr, ytr, str_, atr, Xte, ate):
        imp = SimpleImputer(strategy='median')
        Ztr = imp.fit_transform(Xtr); Zte = imp.transform(Xte)
        c = pd.Series(str_).value_counts(); w = np.array([1.0/c[s] for s in str_])
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                               min_child_samples=50, colsample_bytree=0.5,
                               reg_lambda=1.0, random_state=42, verbose=-1)
        m.fit(Ztr, ytr, sample_weight=w/w.mean())
        return m.predict_proba(Zte)[:, 1]
    return f

def blend_fit(a, b, wa=0.5):
    """Rank-average two scorers; AUROC depends only on order, so ranks are the
    right common scale when the two models are calibrated differently."""
    def f(*args):
        sa = rankdata(a(*args)); sb = rankdata(b(*args))
        n = len(sa)
        return wa * sa/n + (1-wa) * sb/n
    return f

configs = {
  'tabfm n_est=8 (previous)':   tabfm_fit(n_estimators=8,  max_num_features=500, n_svd_features='sqrt'),
  'tabfm n_est=32 (default)':   tabfm_fit(n_estimators=32, max_num_features=500, n_svd_features='sqrt'),
  'tabfm n_est=32, no svd':     tabfm_fit(n_estimators=32, max_num_features=500, n_svd_features=0),
  'tabfm n_est=32, crosses':    tabfm_fit(n_estimators=32, max_num_features=500,
                                          n_svd_features='sqrt', n_feature_crosses='sqrt'),
  'tabfm n_est=32, nnls':       tabfm_fit(n_estimators=32, max_num_features=500,
                                          n_svd_features='sqrt', enable_nnls=True),
  'lgbm':                       lgbm_fit(),
}
configs['blend tabfm+lgbm 50/50'] = blend_fit(
    tabfm_fit(n_estimators=32, max_num_features=500, n_svd_features='sqrt'), lgbm_fit(), 0.5)
configs['blend tabfm+lgbm 70/30'] = blend_fit(
    tabfm_fit(n_estimators=32, max_num_features=500, n_svd_features='sqrt'), lgbm_fit(), 0.7)

print(f"{'config':30s} {'worst':>7s} {'mean':>7s} {'select':>7s}  {'secs':>5s}")
for name, fn in configs.items():
    t = time.time()
    try:
        folds = protocol.evaluate(fn, X, y, sites, ages, prevalence=prev,
                                  n_boot=300, verbose=False)
    except Exception as e:
        print(f'{name:30s} FAILED {type(e).__name__}: {str(e)[:60]}', flush=True); continue
    pw = folds[folds.powered]
    print(f'{name:30s} {pw.auroc_age.min():7.4f} {pw.auroc_age.mean():7.4f} '
          f'{protocol.selection_statistic(folds):7.4f}  {time.time()-t:5.0f}', flush=True)
