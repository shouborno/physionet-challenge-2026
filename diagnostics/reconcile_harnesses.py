"""Why do two of my own harnesses disagree by 0.03 on the same configuration?

run_baselines.py reports 0.6556 for LightGBM on features_large_v9 with age and
bmi withheld; verify_wins_large.py reports 0.6254 for what should be the same
thing. That gap is larger than most effects reported this session, so it has to
be found before any of these numbers are trusted.

Differences between the two call paths, isolated one at a time:
  A  feature list construction (--drop-age plus --drop-cols vs a manual filter)
  B  StandardScaler present in run_baselines, absent in the verifier
  C  subsample=0.8 present in run_baselines, absent in the verifier
  D  site-balanced sample weights, present in both but worth confirming
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.feature_io import load_features, EXCLUDE
from src.eval import challenge_metrics as cm
from src.eval import protocol

FE='data/processed/features_large_v9.pkl'

# Path A: exactly what run_baselines.py builds.
raw = pd.read_pickle(FE); raw = raw[raw.label.notna()]
META = {'patient_id','site_id','session_id','label','extract_time_sec',
        'demo_age','demo_sex','Time_to_Event','Time_to_Last_Visit'}
cols_rb = [c for c in raw.columns if c not in META and pd.api.types.is_numeric_dtype(raw[c])]
cols_rb = [c for c in cols_rb if c != 'age' and c != 'bmi']

# Path B: exactly what the verifier builds.
df, allcols = load_features(FE)
cols_vf = [c for c in allcols if c not in ('age','bmi')]

print(f'run_baselines path: {len(cols_rb)} features')
print(f'verifier path     : {len(cols_vf)} features')
only_rb = sorted(set(cols_rb) - set(cols_vf)); only_vf = sorted(set(cols_vf) - set(cols_rb))
print(f'only in run_baselines: {len(only_rb)} {only_rb[:8]}')
print(f'only in verifier     : {len(only_vf)} {only_vf[:8]}')

y=raw.label.to_numpy(float); sites=raw.site_id.to_numpy(); ages=raw.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)

def make(scaler, subsample):
    def f(Xtr,ytr,str_,atr,Xte,ate):
        steps=[('i',SimpleImputer(strategy='median'))]
        if scaler: steps.append(('s',StandardScaler()))
        kw=dict(n_estimators=300,learning_rate=0.03,num_leaves=15,
                min_child_samples=50,colsample_bytree=0.5,reg_lambda=1.0,
                random_state=42,verbose=-1)
        if subsample: kw['subsample']=0.8
        steps.append(('m',lgb.LGBMClassifier(**kw)))
        p=Pipeline(steps)
        c=pd.Series(str_).value_counts(); w=np.array([1.0/c[s] for s in str_])
        p.fit(Xtr,ytr,m__sample_weight=w/w.mean())
        return p.predict_proba(Xte)[:,1]
    return f

print(f"\n{'config':46s} {'worst':>7s} {'mean':>7s}")
for cname, cc in (('run_baselines features', cols_rb), ('verifier features', cols_vf)):
    X=raw[cc].to_numpy(float)
    for sname, scaler, sub in (('scaler+subsample (run_baselines)',True,True),
                               ('no scaler, no subsample (verifier)',False,False)):
        folds=protocol.evaluate(make(scaler,sub),X,y,sites,ages,prevalence=prev,
                                n_boot=200,verbose=False)
        pw=folds[folds.powered]
        print(f'{cname[:20]:20s} | {sname:24s} {pw.auroc_age.min():7.4f} '
              f'{pw.auroc_age.mean():7.4f}', flush=True)
