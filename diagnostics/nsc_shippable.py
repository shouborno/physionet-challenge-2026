"""NSC compression on the shippable features only, multi-seed.

Family-wise segment PCA reached 0.7881 on all 1,015 features, the best result
in this project, but 307 of those exist only as bytecode and cannot ship. The
other 708 can: the inline extractor's 95, the 550 coherence features and the 63
temporal ones all have clean source in this repo.

That matters because coherence is most of what compression rescued. Raw, its
550 columns diluted everything else and it measured as a null three times.
Compressed to one component per family it occupies about thirty slots and
contributes. No earlier test asked that question; they all compared raw
coherence against no coherence.

Three seeds, since the gap being chased is around 0.01 and TabFM's seed spread
is 0.0006. PCA is fitted inside training folds only.
"""
import sys, re, time, warnings, collections
warnings.filterwarnings('ignore')
sys.path.insert(0, '/scratch/simran/pn26/tabfm')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, torch, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
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
ship=sorted(set(cols)&(inline|set(tp)|set(coh)|{'rec_year'}))
print(f'shippable: {len(ship)} of {len(cols)} '
      f'({len(set(cols)&inline)} inline, {len(coh)} coherence, {len(tp)} temporal)',
      flush=True)

y=df.label.to_numpy(float); sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)

def family_of(name):
    if name.startswith('coh_'):
        m=re.match(r'coh_[a-z0-9]+_[a-z0-9]+_([a-z0-9]+)_([a-z0-9]+)',name)
        return f'coh_{m.group(1)}_{m.group(2)}' if m else 'coh_other'
    if name.startswith('tp_'):
        m=re.match(r'tp_([a-z0-9]+)_',name)
        return f'tp_{m.group(1)}' if m else 'tp_other'
    m=re.match(r'^([a-z]+(?:_[a-z0-9]+)?)',name)
    return m.group(1) if m else 'other'

def make_groups(colnames):
    g=collections.defaultdict(list)
    for i,c in enumerate(colnames): g[family_of(c)].append(i)
    return g

def nsc(groups, n_comp=1, min_group=3):
    def build(Xtr,Xte):
        imp=SimpleImputer(strategy='median'); sc=StandardScaler()
        Ztr=sc.fit_transform(imp.fit_transform(Xtr)); Zte=sc.transform(imp.transform(Xte))
        a,b=[],[]
        for fam,idx in sorted(groups.items()):
            if len(idx)<min_group:
                a.append(Ztr[:,idx]); b.append(Zte[:,idx]); continue
            k=min(n_comp,len(idx),Ztr.shape[0]-1)
            p=PCA(n_components=k,random_state=42).fit(Ztr[:,idx])
            a.append(p.transform(Ztr[:,idx])); b.append(p.transform(Zte[:,idx]))
        return np.hstack(a), np.hstack(b)
    return build

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as hub
MODEL=hub.load('classification', device='cuda' if torch.cuda.is_available() else 'cpu')

def tabfm_fn(builder, seed):
    def f(Xtr,ytr,s,atr,Xte,ate):
        A,B=builder(Xtr,Xte) if builder else (
            SimpleImputer(strategy='median').fit(Xtr).transform(Xtr),
            SimpleImputer(strategy='median').fit(Xtr).transform(Xte))
        codes={v:i for i,v in enumerate(sorted(pd.unique(s)))}
        A=np.column_stack([A,[codes[v] for v in s]])
        B=np.column_stack([B,np.full(len(B),len(codes))])
        c=TabFMClassifier(MODEL,n_estimators=32,max_num_features=500,
                          n_svd_features='sqrt',random_state=seed)
        c.fit(A,ytr.astype(int)); return c.predict_proba(B)[:,1]
    return f

def lgbm_fn(builder, seed):
    def f(Xtr,ytr,s,atr,Xte,ate):
        A,B=builder(Xtr,Xte) if builder else (Xtr,Xte)
        m=lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,
                             min_child_samples=50,colsample_bytree=0.5,
                             reg_lambda=1.0,random_state=seed,verbose=-1)
        imp=SimpleImputer(strategy='median')
        m.fit(imp.fit_transform(A),ytr)
        return m.predict_proba(imp.transform(B))[:,1]
    return f

def blend_fn(builder, seed, wa=0.7):
    a,b=tabfm_fn(builder,seed),lgbm_fn(builder,seed)
    def f(*args):
        ra,rb=rankdata(a(*args)),rankdata(b(*args)); n=len(ra)
        return wa*ra/n+(1-wa)*rb/n
    return f

Xs=df[ship].to_numpy(float)
gs=make_groups(ship)
print(f'{len(gs)} families in the shippable set\n', flush=True)

def run(label, fn_factory, X, seeds=(7,42,99)):
    ms,ws=[],[]
    for sd in seeds:
        folds=protocol.evaluate(fn_factory(sd),X,y,sites,ages,prevalence=prev,
                                n_boot=100,verbose=False)
        ms.append(protocol.mean_powered(folds))
        ws.append(folds[folds.powered].auroc_age.min())
    print(f'{label:38s} {np.mean(ms):.4f} +/- {np.std(ms):.4f}   worst {np.mean(ws):.4f}',
          flush=True)
    return np.mean(ms)

b1=nsc(gs,1)
run('tabfm + NSC1 (shippable 709)', lambda sd: tabfm_fn(b1,sd), Xs)
run('blend70 + NSC1 (shippable 709)', lambda sd: blend_fn(b1,sd), Xs)
run('lgbm + NSC1 (shippable 709)',   lambda sd: lgbm_fn(b1,sd), Xs)
print('\nreference: tabfm on 159 raw = 0.7737, tabfm+NSC1 on all 1015 = 0.7881')
