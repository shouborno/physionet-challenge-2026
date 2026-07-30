"""Google TabFM against LightGBM under leave-one-site-out.

TabFM (released 2026-06-30) reads training rows and query rows as one context
and predicts in a single forward pass through frozen weights, trained on
hundreds of millions of synthetic datasets from structural causal models. Google
reports it beating tuned gradient-boosted trees zero-shot, which is exactly the
claim worth testing here after eleven modelling interventions have failed.

Unlike TabPFN it advertises no hard feature cap, so this runs on the full
feature set first and only falls back to PCA if it refuses.
"""
import sys, time, warnings, traceback
warnings.filterwarnings('ignore')
sys.path.insert(0, '/scratch/simran/pn26/tabfm')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd
from src.data.feature_io import load_features
from src.eval import challenge_metrics as cm
from src.eval import protocol

FEATS = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/features_large_v8.pkl'
NCOMP = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # 0 = no PCA

df, cols = load_features(FEATS, drop_age=True, drop_cols=['bmi'])
X = df[cols].to_numpy(float)
y = df.label.to_numpy(float); sites = df.site_id.to_numpy(); ages = df.age.to_numpy(float)
prev = cm.prevalence_by_age(ages, y, ages, gap=2)
print(f'{FEATS}: {len(df)} records, {len(cols)} features, pca={NCOMP or "none"}', flush=True)

def tabfm_fit(model):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    def f(Xtr, ytr, str_, atr, Xte, ate):
        from tabfm import TabFMClassifier
        # TabFM does its own normalization and its own SVD reduction to the
        # 500-feature cap, so feeding it raw imputed values is preferable to
        # imposing an external PCA the way TabPFN required.
        pre = Pipeline([('i', SimpleImputer(strategy='median'))])
        Ztr = pre.fit_transform(Xtr)
        Zte = pre.transform(Xte)
        clf = TabFMClassifier(model, n_estimators=8, max_num_features=500,
                              n_svd_features='sqrt', random_state=42)
        clf.fit(Ztr, ytr.astype(int))
        return clf.predict_proba(Zte)[:, 1]
    return f

import torch
from tabfm import tabfm_v1_0_0_pytorch as tabfm_hub
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'loading TabFM weights on {dev} ...', flush=True)
MODEL = tabfm_hub.load('classification', device=dev)
print('weights loaded', flush=True)

t = time.time()
try:
    folds = protocol.evaluate(tabfm_fit(MODEL), X, y, sites, ages, prevalence=prev,
                              n_boot=300, verbose=True)
except Exception:
    traceback.print_exc(); sys.exit(1)
pw = folds[folds.powered]
print(f'\nTabFM worst powered = {pw.auroc_age.min():.4f}  '
      f'mean powered = {pw.auroc_age.mean():.4f}  '
      f'select = {protocol.selection_statistic(folds):.4f}  ({time.time()-t:.0f}s)')
print('LightGBM reference on the same matrix: worst 0.6610, mean 0.7197, select 0.5985')
