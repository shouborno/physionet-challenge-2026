"""The configuration we would actually submit, measured as such.

The model comparison ran on 952 features while the shippable-subset test ran on
159, so the winning model has never been evaluated on the features that can go
in the container. The 342 bytecode-only features cannot ship: the container is
Python 3.10 and those extractors exist only as 3.11 bytecode. Coherence can
ship but was measured as worthless twice.

This closes the loop: TabFM, LightGBM and their blend on exactly the 159
features with clean source, versus the 952-feature results for reference.
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

import team_code
from helper_code import find_patients
R = '/scratch/simran/pn26/raw/training_set_small'
probe = team_code._extract_one(R, find_patients(R + '/demographics.csv')[0],
                               R + '/demographics.csv', 'channel_table.csv')
inline = set(probe[0].keys())

df, cols = load_features('data/processed/features_large_v10.pkl', drop_cols=['bmi'])
tp = [c for c in cols if c.startswith('tp_')]
ship = sorted(set(cols) & (inline | set(tp) | {'rec_year'}))
print(f'shippable feature set: {len(ship)} '
      f'({len(set(cols) & inline)} inline, {len(tp)} temporal, rec_year)\n', flush=True)

y = df.label.to_numpy(float); sites = df.site_id.to_numpy(); ages = df.age.to_numpy(float)
prev = cm.prevalence_by_age(ages, y, ages, gap=2)
X = df[ship].to_numpy(float)

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as hub
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = hub.load('classification', device=dev)
print(f'TabFM on {dev}\n', flush=True)

def add_site(Xtr, Xte, s):
    codes = {v: i for i, v in enumerate(sorted(pd.unique(s)))}
    return (np.column_stack([Xtr, [codes[v] for v in s]]),
            np.column_stack([Xte, np.full(len(Xte), len(codes))]))

def lgbm(site_cov=False):
    def f(Xtr, ytr, s, atr, Xte, ate):
        if site_cov: Xtr, Xte = add_site(Xtr, Xte, s)
        p = Pipeline([('i', SimpleImputer(strategy='median')), ('s', StandardScaler()),
                      ('m', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                          num_leaves=15, min_child_samples=50, colsample_bytree=0.5,
                          reg_lambda=1.0, random_state=42, verbose=-1))])
        p.fit(Xtr, ytr)
        return p.predict_proba(Xte)[:, 1]
    return f

def tabfm(site_cov=True):
    def f(Xtr, ytr, s, atr, Xte, ate):
        if site_cov: Xtr, Xte = add_site(Xtr, Xte, s)
        imp = SimpleImputer(strategy='median')
        c = TabFMClassifier(MODEL, n_estimators=32, max_num_features=500,
                            n_svd_features='sqrt', random_state=42)
        c.fit(imp.fit_transform(Xtr), ytr.astype(int))
        return c.predict_proba(imp.transform(Xte))[:, 1]
    return f

def blend(a, b, wa):
    def f(*args):
        ra, rb = rankdata(a(*args)), rankdata(b(*args)); n = len(ra)
        return wa*ra/n + (1-wa)*rb/n
    return f

cases = {
  'lgbm':                    lgbm(False),
  'tabfm + site cov':        tabfm(True),
  'blend 70/30':             blend(tabfm(True), lgbm(False), 0.7),
  'blend 50/50':             blend(tabfm(True), lgbm(False), 0.5),
}
print(f"{'model (159 shippable features)':32s} {'mean':>7s} {'worst':>7s}  per-fold")
for name, fn in cases.items():
    t = time.time()
    folds = protocol.evaluate(fn, X, y, sites, ages, prevalence=prev,
                              n_boot=300, verbose=False)
    per = ' '.join(f'{r.auroc_age:.3f}' for _, r in folds.iterrows())
    print(f'{name:32s} {protocol.mean_powered(folds):7.4f} '
          f'{folds[folds.powered].auroc_age.min():7.4f}  {per}  ({time.time()-t:.0f}s)',
          flush=True)
print('\n952-feature reference: tabfm+cov 0.7693/0.6674, blend70 0.7677/0.6806, lgbm 0.7404/0.6877')
