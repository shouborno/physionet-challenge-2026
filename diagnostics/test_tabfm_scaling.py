"""Is TabFM's advantage really about context size, or about which site is held out?

Across folds TabFM's score tracks training-set size with Spearman 1.000, which
would suggest in-context learning benefits from more context and that the real
submission, training on all 6,600 records, would do better than any fold shows.

But every fold changes two things at once: how many training records there are
and which site is held out. This fixes the fold and varies only the training
size, which separates the two explanations.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/scratch/simran/pn26/tabfm')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, torch
from sklearn.impute import SimpleImputer
from src.data.feature_io import load_features
from src.eval import challenge_metrics as cm

df, cols = load_features('data/processed/features_large_v8.pkl',
                         drop_age=True, drop_cols=['bmi'])
X = df[cols].to_numpy(float); y = df.label.to_numpy(float)
sites = df.site_id.to_numpy(); ages = df.age.to_numpy(float)

# Fixed fold: train on S0001, test on the pooled remainder. Largest training
# pool, so it can be subsampled over the widest range.
tr = sites == 'S0001'; te = ~tr
Xtr_all, ytr_all, atr_all = X[tr], y[tr], ages[tr]
Xte, yte, ate = X[te], y[te], ages[te]
print(f'fixed fold: train S0001 ({tr.sum()}), test rest ({te.sum()}, '
      f'{int(yte.sum())} positive)', flush=True)

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as hub
import lightgbm as lgb
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = hub.load('classification', device=dev)
print('TabFM weights loaded', flush=True)

rng = np.random.default_rng(42)
imp = SimpleImputer(strategy='median')
Zte = None

print(f'\n{"n_train":>8s} {"pos":>5s} {"TabFM":>8s} {"LGBM":>8s}')
for n in (1461, 2500, 3500, 4500, 5139):
    n = min(n, tr.sum())
    # Stratified subsample so prevalence is held constant across sizes.
    pos = np.flatnonzero(ytr_all == 1); neg = np.flatnonzero(ytr_all == 0)
    frac = n / len(ytr_all)
    keep = np.concatenate([
        rng.choice(pos, max(2, int(round(len(pos) * frac))), replace=False),
        rng.choice(neg, n - max(2, int(round(len(pos) * frac))), replace=False)])
    Xs, ys, as_ = Xtr_all[keep], ytr_all[keep], atr_all[keep]

    Ztr = imp.fit_transform(Xs)
    if Zte is None or True:
        Zte = imp.transform(Xte)

    clf = TabFMClassifier(MODEL, n_estimators=8, max_num_features=500,
                          n_svd_features='sqrt', random_state=42)
    clf.fit(Ztr, ys.astype(int))
    s_tab = clf.predict_proba(Zte)[:, 1]

    g = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                           min_child_samples=50, colsample_bytree=0.5,
                           reg_lambda=1.0, random_state=42, verbose=-1)
    g.fit(Ztr, ys)
    s_lgb = g.predict_proba(Zte)[:, 1]

    print(f'{n:8d} {int(ys.sum()):5d} '
          f'{cm.auroc_age(yte, s_tab, ate, gap=2):8.4f} '
          f'{cm.auroc_age(yte, s_lgb, ate, gap=2):8.4f}', flush=True)
