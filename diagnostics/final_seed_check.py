"""Multi-seed comparison of the two submission candidates.

Every number driving the submission choice is single-seed, and the seed sweep
on this feature set spans 0.0058. TabFM and the 70/30 blend differ by 0.0007,
which is inside that, so the honest question is not which scores higher but
whether they are separable at all and which has the steadier floor.

LightGBM's seed enters both the blend and the standalone arm; TabFM's
random_state controls its feature-shuffle ensembling. Both are varied together.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/scratch/simran/pn26/tabfm')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, torch, lightgbm as lgb
from scipy.stats import rankdata
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
ship=sorted(set(cols)&(inline|set(tp)|{'rec_year'}))
X=df[ship].to_numpy(float); y=df.label.to_numpy(float)
sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)
print(f'{len(ship)} features, {len(df)} records\n', flush=True)

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as hub
MODEL = hub.load('classification', device='cuda' if torch.cuda.is_available() else 'cpu')

def add_site(Xtr, Xte, s):
    c={v:i for i,v in enumerate(sorted(pd.unique(s)))}
    return (np.column_stack([Xtr,[c[v] for v in s]]),
            np.column_stack([Xte,np.full(len(Xte),len(c))]))

def lgbm(seed):
    def f(Xtr,ytr,s,atr,Xte,ate):
        p=Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),
                    ('m',lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,
                        num_leaves=15,min_child_samples=50,colsample_bytree=0.5,
                        reg_lambda=1.0,random_state=seed,verbose=-1))])
        p.fit(Xtr,ytr); return p.predict_proba(Xte)[:,1]
    return f

def tabfm(seed):
    def f(Xtr,ytr,s,atr,Xte,ate):
        Xtr2,Xte2=add_site(Xtr,Xte,s)
        imp=SimpleImputer(strategy='median')
        c=TabFMClassifier(MODEL,n_estimators=32,max_num_features=500,
                          n_svd_features='sqrt',random_state=seed)
        c.fit(imp.fit_transform(Xtr2), ytr.astype(int))
        return c.predict_proba(imp.transform(Xte2))[:,1]
    return f

def blend(seed, wa=0.7):
    a,b=tabfm(seed),lgbm(seed)
    def f(*args):
        ra,rb=rankdata(a(*args)),rankdata(b(*args)); n=len(ra)
        return wa*ra/n+(1-wa)*rb/n
    return f

results={'tabfm':[], 'blend70':[], 'lgbm':[]}
worsts={'tabfm':[], 'blend70':[], 'lgbm':[]}
for seed in (7, 42, 99):
    for name, fn in (('tabfm',tabfm(seed)), ('blend70',blend(seed)), ('lgbm',lgbm(seed))):
        t=time.time()
        folds=protocol.evaluate(fn,X,y,sites,ages,prevalence=prev,n_boot=150,verbose=False)
        m=protocol.mean_powered(folds); w=folds[folds.powered].auroc_age.min()
        results[name].append(m); worsts[name].append(w)
        print(f'  seed {seed:3d} {name:9s} mean={m:.4f} worst={w:.4f} ({time.time()-t:.0f}s)',
              flush=True)

print(f"\n{'model':10s} {'mean':>18s} {'worst':>18s}")
for k in results:
    m=np.array(results[k]); w=np.array(worsts[k])
    print(f'{k:10s} {m.mean():.4f} +/- {m.std():.4f}   {w.mean():.4f} +/- {w.std():.4f}')
d=np.array(results['tabfm'])-np.array(results['blend70'])
print(f'\ntabfm minus blend70, paired by seed: {d.mean():+.4f} +/- {d.std():.4f}')
dw=np.array(worsts['blend70'])-np.array(worsts['tabfm'])
print(f'blend70 minus tabfm on worst fold : {dw.mean():+.4f} +/- {dw.std():.4f}')
