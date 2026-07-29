"""Measure what the recording-date artifact is worth on top of the real features.

The label definition forces it: a negative needs at least six years of clean
follow-up so must be an older recording, while a positive needs only one to six
years to diagnosis. Positives therefore average 2015.2 against 2013.2. It is a
construction artifact rather than physiology, and CreationTime is supplied for
the hidden sites as well, so the question is purely how much it is worth.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd
from src.data.self_norm import META
from src.eval import challenge_metrics as cm
from src.eval import protocol
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def fit():
    def f(Xtr,ytr,str_,atr,Xte,ate):
        p=Pipeline([('i',SimpleImputer(strategy='median')),
                    ('m',lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,
                        num_leaves=15,min_child_samples=50,subsample=0.8,
                        colsample_bytree=0.5,reg_lambda=1.0,random_state=42,verbose=-1))])
        c=pd.Series(str_).value_counts(); w=np.array([1.0/c[s] for s in str_])
        p.fit(Xtr,ytr,m__sample_weight=w/w.mean())
        return p.predict_proba(Xte)[:,1]
    return f

df=pd.read_pickle('data/processed/features_small_v9.pkl')
df=df[df.label.notna()].copy()
demo=pd.read_csv('/scratch/simran/pn26/raw/training_set_small/demographics.csv')
t=pd.to_datetime(demo['CreationTime'],format='mixed')
demo['rec_year']=t.dt.year + t.dt.dayofyear/366.0
df=df.merge(demo[['BidsFolder','SessionID','rec_year']]
            .rename(columns={'BidsFolder':'patient_id','SessionID':'session_id'}),
            on=['patient_id','session_id'],how='left')

y=df.label.to_numpy(float); sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)
base=[c for c in df.columns if c not in META and c!='rec_year'
      and pd.api.types.is_numeric_dtype(df[c]) and c!='age']

for name, cols in (('features only (no age, no year)', base),
                   ('features + recording year', base+['rec_year'])):
    X=df[cols].to_numpy(float)
    folds=protocol.evaluate(fit(),X,y,sites,ages,prevalence=prev,n_boot=500,verbose=False)
    pw=folds[folds.powered]
    print(f'{name:34s} worst={pw.auroc_age.min():.4f}  select={protocol.selection_statistic(folds):.4f}')
    for _,r in folds.iterrows():
        print(f'    {r.fold:22s} {r.auroc_age:.4f}  pairs={int(r.n_pairs):6d} powered={bool(r.powered)}')
