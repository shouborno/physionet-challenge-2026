"""Do explicit missingness indicators help, or are they a site-specific trap?

BMI is 75.9% missing and whether it was recorded predicts the label hard: at
S0001 prevalence is 36.9% when BMI is present against 3.2% when absent. That is
almost certainly healthcare-contact intensity, since a fuller clinical workup
both records BMI and raises the chance of later diagnosis. It is available at
inference, so the question is whether it transfers across sites or is a trap.

Leave-one-site-out answers it directly: the indicator is fitted on two sites and
tested on a third, which is the same generalization the leaderboard demands.
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
                        num_leaves=15,min_child_samples=50,colsample_bytree=0.5,
                        reg_lambda=1.0,random_state=42,verbose=-1))])
        c=pd.Series(str_).value_counts(); w=np.array([1.0/c[s] for s in str_])
        p.fit(Xtr,ytr,m__sample_weight=w/w.mean())
        return p.predict_proba(Xte)[:,1]
    return f

df=pd.read_pickle('data/processed/features_small_v11.pkl'); df=df[df.label.notna()].copy()
base=[c for c in df.columns if c not in META and pd.api.types.is_numeric_dtype(df[c]) and c!='age']
y=df.label.to_numpy(float); sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)

# Indicators for the columns whose absence is most label-associated.
nan_rate=df[base].isna().mean()
cands=[c for c in base if 0.02 < nan_rate[c] < 0.98]
df['miss_bmi']=df['bmi'].isna().astype(float)
extra=['miss_bmi']
for c in ['mean_cycle_duration_min','first_cycle_n3_pct']:
    if c in df.columns:
        df[f'miss_{c}']=df[c].isna().astype(float); extra.append(f'miss_{c}')

for name, cols in (('baseline', base),
                   ('+ bmi missingness', base+['miss_bmi']),
                   ('+ all missingness indicators', base+extra),
                   ('bmi dropped entirely', [c for c in base if c!='bmi'])):
    X=df[cols].to_numpy(float)
    folds=protocol.evaluate(fit(),X,y,sites,ages,prevalence=prev,n_boot=400,verbose=False)
    pw=folds[folds.powered]
    per={r.fold.replace('holdout_','').replace('train_','t_'):round(r.auroc_age,4)
         for _,r in folds.iterrows()}
    print(f'{name:30s} worst={pw.auroc_age.min():.4f}  {per}')
