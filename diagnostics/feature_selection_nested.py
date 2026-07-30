"""Select features without selecting on the evaluation statistic.

159 was never chosen for performance: it is simply everything with clean Python
source, since 307 features exist only as 3.11 bytecode and cannot enter a 3.10
container. Actual selection has not been tried, and it now matters because
TabFM caps at 500 features with SVD reduction, so feature count is causally
relevant to it rather than incidental.

The danger is the mechanism that turned 0.780 into 0.644 in the unofficial
phase: greedy forward selection scored against the same statistic used to
report results. Roughly 7,000 evaluations against a noisy statistic buys
0.05-0.10 of optimism for free.

So selection happens strictly inside the training folds here. For each
leave-one-site-out split, importance is computed on the training sites only and
the top-k are chosen there; the held-out site never participates. The ranking
therefore differs per fold, which is the point.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.feature_io import load_features
from src.eval import challenge_metrics as cm
from src.eval import protocol

df, cols = load_features('data/processed/features_large_v10.pkl', drop_cols=['bmi'])
y=df.label.to_numpy(float); sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)
X=df[cols].to_numpy(float)
print(f'{len(df)} records, {len(cols)} features\n', flush=True)

def select_in_fold(Xtr, ytr, k):
    """Rank by LightGBM gain on the training sites only."""
    imp=SimpleImputer(strategy='median')
    Z=imp.fit_transform(Xtr)
    r=lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                         min_child_samples=50, colsample_bytree=0.5,
                         random_state=42, verbose=-1)
    r.fit(Z, ytr)
    return np.argsort(r.booster_.feature_importance('gain'))[::-1][:k]

def fit_topk(k):
    def f(Xtr,ytr,s,atr,Xte,ate):
        idx=select_in_fold(Xtr,ytr,k)
        p=Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),
                    ('m',lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,
                        num_leaves=15,min_child_samples=50,colsample_bytree=0.5,
                        reg_lambda=1.0,random_state=42,verbose=-1))])
        p.fit(Xtr[:,idx],ytr)
        return p.predict_proba(Xte[:,idx])[:,1]
    return f

def fit_all():
    def f(Xtr,ytr,s,atr,Xte,ate):
        p=Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),
                    ('m',lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,
                        num_leaves=15,min_child_samples=50,colsample_bytree=0.5,
                        reg_lambda=1.0,random_state=42,verbose=-1))])
        p.fit(Xtr,ytr); return p.predict_proba(Xte)[:,1]
    return f

print(f"{'k (selected in-fold)':26s} {'mean':>7s} {'worst':>7s}")
folds=protocol.evaluate(fit_all(),X,y,sites,ages,prevalence=prev,n_boot=150,verbose=False)
print(f'{"all 1015":26s} {protocol.mean_powered(folds):7.4f} '
      f'{folds[folds.powered].auroc_age.min():7.4f}', flush=True)
for k in (50, 100, 200, 400, 500):
    folds=protocol.evaluate(fit_topk(k),X,y,sites,ages,prevalence=prev,n_boot=150,verbose=False)
    print(f'{"top-"+str(k):26s} {protocol.mean_powered(folds):7.4f} '
          f'{folds[folds.powered].auroc_age.min():7.4f}', flush=True)
