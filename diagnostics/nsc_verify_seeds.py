"""Re-measure the 0.7881 at three seeds before believing it.

Family-wise compression on all 1,015 features scored 0.7881 in a single-seed
run, the best number in this project, and that result drove an afternoon of
reasoning about which block it came from. The decomposition then failed to
reproduce it from any block: coherence compressed gives 0.7694 and bytecode
compressed 0.7598, both below the 0.7737 the plain 159 features reach.

Either the gain needs every block together, or the single-seed number was not
real. This runs the same configuration at three seeds against the two
references, all measured the same way.
"""
import sys, re, warnings, collections
warnings.filterwarnings('ignore')
sys.path.insert(0, '/scratch/simran/pn26/tabfm')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, torch
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
base=sorted(set(cols)&(inline|set(tp)|{'rec_year'}))
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

def nsc_builder(colnames, n_comp=1, min_group=3, pca_seed=42):
    g=collections.defaultdict(list)
    for i,c in enumerate(colnames): g[family_of(c)].append(i)
    def build(Xtr,Xte):
        imp=SimpleImputer(strategy='median'); sc=StandardScaler()
        Ztr=sc.fit_transform(imp.fit_transform(Xtr)); Zte=sc.transform(imp.transform(Xte))
        a,b=[],[]
        for fam,idx in sorted(g.items()):
            if len(idx)<min_group:
                a.append(Ztr[:,idx]); b.append(Zte[:,idx]); continue
            k=min(n_comp,len(idx),Ztr.shape[0]-1)
            p=PCA(n_components=k,random_state=pca_seed).fit(Ztr[:,idx])
            a.append(p.transform(Ztr[:,idx])); b.append(p.transform(Zte[:,idx]))
        return np.hstack(a), np.hstack(b)
    return build

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as hub
MODEL=hub.load('classification', device='cuda' if torch.cuda.is_available() else 'cpu')

def tabfm_fn(builder, seed):
    def f(Xtr,ytr,s,atr,Xte,ate):
        if builder is None:
            imp=SimpleImputer(strategy='median')
            A,B=imp.fit_transform(Xtr),imp.transform(Xte)
        else:
            A,B=builder(Xtr,Xte)
        codes={v:i for i,v in enumerate(sorted(pd.unique(s)))}
        A=np.column_stack([A,[codes[v] for v in s]])
        B=np.column_stack([B,np.full(len(B),len(codes))])
        c=TabFMClassifier(MODEL,n_estimators=32,max_num_features=500,
                          n_svd_features='sqrt',random_state=seed)
        c.fit(A,ytr.astype(int)); return c.predict_proba(B)[:,1]
    return f

def run(label, cc, compress, seeds=(7,42,99)):
    X=df[cc].to_numpy(float)
    b=nsc_builder(cc) if compress else None
    ms=[]
    for sd in seeds:
        folds=protocol.evaluate(tabfm_fn(b,sd),X,y,sites,ages,prevalence=prev,
                                n_boot=100,verbose=False)
        ms.append(protocol.mean_powered(folds))
    print(f'{label:34s} {np.mean(ms):.4f} +/- {np.std(ms):.4f}   '
          f'seeds {" ".join(f"{m:.4f}" for m in ms)}', flush=True)

run('NSC1 all 1015 (claimed 0.7881)', list(cols), True)
run('raw all 1015 (claimed 0.7789)',  list(cols), False)
run('raw 159 base (known 0.7737)',    base, False)
