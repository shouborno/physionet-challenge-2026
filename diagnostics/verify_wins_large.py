"""Do the four small-set wins survive on 6,600 records?

Coherence looked worth +0.024 on the small set and is a null on the large one,
which means the other three gains were measured under the same wide error bar
and deserve the same re-test. This is the arbiter: all four folds are powered
here, and the subject-level interval is roughly a third of what it was.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.data.feature_io import load_features
from src.eval import challenge_metrics as cm
from src.eval import protocol

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

FE='data/processed/features_large_v9.pkl'
df, allcols = load_features(FE)          # everything, age and bmi included
y=df.label.to_numpy(float); sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)
coh=[c for c in allcols if c.startswith(('coh_','iaf_','taf_','alpha_ratio_hi_lo'))]
print(f'{len(df)} records, {len(allcols)} features ({len(coh)} coherence)\n')

base=[c for c in allcols if c not in ('age','bmi')]
variants={
 'reference (no age, no bmi, +coh)': base,
 'add age back':                     base+['age'],
 'add bmi back':                     base+['bmi'],
 'remove coherence':                 [c for c in base if c not in coh],
 'remove recording year':            [c for c in base if c!='rec_year'],
 'everything (age+bmi+coh+year)':    allcols,
}
print(f"{'variant':34s} {'worst':>7s} {'mean':>7s} {'select':>7s}")
for name, cc in variants.items():
    X=df[cc].to_numpy(float)
    folds=protocol.evaluate(fit(),X,y,sites,ages,prevalence=prev,n_boot=400,verbose=False)
    pw=folds[folds.powered]
    print(f'{name:34s} {pw.auroc_age.min():7.4f} {pw.auroc_age.mean():7.4f} '
          f'{protocol.selection_statistic(folds):7.4f}', flush=True)
