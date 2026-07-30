"""How much do we lose if we only ship features we have clean source for?

The 272-feature research extractor exists only as Python 3.11 bytecode. It
cannot go in the submission: the container is 3.10, and shipping compiled
artifacts would undermine the open-source requirement. What we can ship is
team_code.py's inline extractor, plus features_coherence.py and
features_temporal.py, both written from scratch here.

If the shippable subset performs comparably, the bytecode never needs
recovering. If not, the extractors have to be rewritten before submission.
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

# Exactly the names team_code.py's inline extractor emits.
import team_code
from helper_code import find_patients, HEADERS
R='/scratch/simran/pn26/raw/training_set_small'
rec=find_patients(R+'/demographics.csv')[0]
out=team_code._extract_one(R, rec, R+'/demographics.csv', 'channel_table.csv')
inline=set(out[0].keys()) if out else set()
print(f'team_code inline extractor emits {len(inline)} features')

df, cols = load_features('data/processed/features_large_v10.pkl', drop_cols=['bmi'])
coh=[c for c in cols if c.startswith(('coh_','iaf_','taf_','alpha_ratio_hi_lo'))]
tp =[c for c in cols if c.startswith('tp_')]
shippable=sorted(set(cols) & (inline | set(coh) | set(tp) | {'rec_year'}))
print(f'shippable: {len(shippable)} of {len(cols)}')

y=df.label.to_numpy(float); sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)

def fit():
    def f(Xtr,ytr,str_,atr,Xte,ate):
        p=Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),
                    ('m',lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,
                        num_leaves=15,min_child_samples=50,colsample_bytree=0.5,
                        reg_lambda=1.0,random_state=42,verbose=-1))])
        p.fit(Xtr,ytr)
        return p.predict_proba(Xte)[:,1]
    return f

print(f"\n{'feature set':34s} {'n':>5s} {'mean':>7s} {'worst':>7s}")
for name, cc in (('everything (needs bytecode)', cols),
                 ('shippable only', shippable),
                 ('shippable, no coherence', [c for c in shippable if c not in coh])):
    X=df[cc].to_numpy(float)
    folds=protocol.evaluate(fit(),X,y,sites,ages,prevalence=prev,n_boot=300,verbose=False)
    print(f'{name:34s} {len(cc):5d} {protocol.mean_powered(folds):7.4f} '
          f'{folds[folds.powered].auroc_age.min():7.4f}', flush=True)
