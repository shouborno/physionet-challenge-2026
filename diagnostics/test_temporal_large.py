"""Does temporal pooling add anything the nightly summaries miss?

Every existing feature collapses a night to one number, usually a mean. These
63 take the 0.88 quantile of per-epoch band power instead, following the winner
of the closest analogous challenge, plus dispersion, drift and how concentrated
the worst stretches are. Two nights with the same mean look different here if
one is uniformly mediocre and the other alternates between normal and severely
slowed.

Measured on the mean across powered folds, which is stable to about 0.0014,
against the current best configuration of unbalanced LightGBM without BMI.
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

df, allcols = load_features('data/processed/features_large_v10.pkl', drop_cols=['bmi'])
y=df.label.to_numpy(float); sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)
tp=[c for c in allcols if c.startswith('tp_')]
coh=[c for c in allcols if c.startswith(('coh_','iaf_','taf_','alpha_ratio_hi_lo'))]
print(f'{len(df)} records, {len(allcols)} features ({len(tp)} temporal, {len(coh)} coherence)\n')

def fit():
    def f(Xtr,ytr,str_,atr,Xte,ate):
        p=Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),
                    ('m',lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,
                        num_leaves=15,min_child_samples=50,colsample_bytree=0.5,
                        reg_lambda=1.0,random_state=42,verbose=-1))])
        p.fit(Xtr,ytr)   # unbalanced: site weighting costs 0.018
        return p.predict_proba(Xte)[:,1]
    return f

variants={
 'without temporal (previous best)': [c for c in allcols if c not in tp],
 'with temporal':                    allcols,
 'temporal only':                    tp,
 'temporal, no coherence':           [c for c in allcols if c not in coh],
}
print(f"{'variant':34s} {'n':>5s} {'mean':>7s} {'worst':>7s}")
for name, cc in variants.items():
    X=df[cc].to_numpy(float)
    folds=protocol.evaluate(fit(),X,y,sites,ages,prevalence=prev,n_boot=300,verbose=False)
    print(f'{name:34s} {len(cc):5d} {protocol.mean_powered(folds):7.4f} '
          f'{folds[folds.powered].auroc_age.min():7.4f}', flush=True)
