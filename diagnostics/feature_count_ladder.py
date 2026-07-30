"""Does feature count matter, or is the ordering noise?

Three numbers were quoted at each other: 0.7404 for 952 features on v9, 0.7383
for 1,015 on v10, and 0.7491 for the 159 shippable ones. The first two come
from different matrices, so only the v10 pair is comparable, and even there the
spread is about 0.011 against a stability of roughly 0.0014 measured for
*pipeline* perturbations rather than feature-set changes.

This measures the ladder on one matrix with one pipeline, and adds a seed sweep
so the spread across LightGBM seeds can be compared against the spread across
feature counts. If they are the same size, feature count is not doing anything.
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
import team_code
from helper_code import find_patients

R='/scratch/simran/pn26/raw/training_set_small'
inline=set(team_code._extract_one(R, find_patients(R+'/demographics.csv')[0],
                                  R+'/demographics.csv','channel_table.csv')[0].keys())
df, cols = load_features('data/processed/features_large_v10.pkl', drop_cols=['bmi'])
tp=[c for c in cols if c.startswith('tp_')]
coh=[c for c in cols if c.startswith(('coh_','iaf_','taf_','alpha_ratio_hi_lo'))]
ship=sorted(set(cols)&(inline|set(tp)|{'rec_year'}))
byte=[c for c in cols if c not in set(ship)|set(coh)]

y=df.label.to_numpy(float); sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)

def fit(seed=42):
    def f(Xtr,ytr,s,atr,Xte,ate):
        p=Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),
                    ('m',lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,
                        num_leaves=15,min_child_samples=50,colsample_bytree=0.5,
                        reg_lambda=1.0,random_state=seed,verbose=-1))])
        p.fit(Xtr,ytr); return p.predict_proba(Xte)[:,1]
    return f

def score(cc, seed=42):
    folds=protocol.evaluate(fit(seed), df[cc].to_numpy(float), y, sites, ages,
                            prevalence=prev, n_boot=150, verbose=False)
    return protocol.mean_powered(folds)

print('same matrix, same pipeline, seed 42:\n')
ladder=[('shippable 159', ship), ('shippable + bytecode', ship+byte),
        ('shippable + coherence', ship+coh), ('everything', list(cols))]
for name, cc in ladder:
    print(f'  {name:26s} {len(cc):5d} feats   {score(cc):.4f}', flush=True)

print('\nseed sweep on the shippable set (same features, only seed varies):')
vals=[score(ship, s) for s in (1,7,13,42,99)]
print('  ' + '  '.join(f'{v:.4f}' for v in vals))
print(f'  spread across seeds: {max(vals)-min(vals):.4f}')
print('\nIf the seed spread is comparable to the ladder spread, feature count is noise.')
