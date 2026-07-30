"""Compress the wide blocks, keep the base features raw.

The earlier decomposition compressed every column it was given, including the
159 base features, so 'base + bytecode' really meant 'base compressed from 159
columns to about 40 family components, plus bytecode compressed'. Every row of
it fell below the raw-159 baseline, which is what destroying the base would
predict, and it made a real effect look absent: NSC1 over all 1,015 features
reproduces at 0.7865 +/- 0.0019 against 0.7737 for raw 159.

Here the base passes through untouched and only the wide blocks are reduced to
one component per family, which is what the compression is actually for.
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
coh=[c for c in cols if c.startswith(('coh_','iaf_','taf_','alpha_ratio_hi_lo'))]
selfref=[c for c in cols if '_vs_' in c or '_z_' in c or c.endswith('_stagerange')]
base=sorted(set(cols)&(inline|set(tp)|{'rec_year'}))
byte=[c for c in cols if c not in set(base)|set(coh)|set(selfref)]
y=df.label.to_numpy(float); sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)
print(f'base {len(base)} (kept raw), coherence {len(coh)}, self-ref {len(selfref)}, '
      f'bytecode {len(byte)}\n', flush=True)

def family_of(name):
    if name.startswith('coh_'):
        m=re.match(r'coh_[a-z0-9]+_[a-z0-9]+_([a-z0-9]+)_([a-z0-9]+)',name)
        return f'coh_{m.group(1)}_{m.group(2)}' if m else 'coh_other'
    m=re.match(r'^([a-z]+(?:_[a-z0-9]+)?)',name)
    return m.group(1) if m else 'other'

def build_mixed(raw_cols, comp_cols, n_comp=1, min_group=3):
    """Raw columns pass through; compressed ones become family components."""
    ri=[cols.index(c) for c in raw_cols]
    g=collections.defaultdict(list)
    for c in comp_cols: g[family_of(c)].append(cols.index(c))
    def build(Xtr,Xte):
        imp=SimpleImputer(strategy='median'); sc=StandardScaler()
        Ztr=sc.fit_transform(imp.fit_transform(Xtr)); Zte=sc.transform(imp.transform(Xte))
        a=[Ztr[:,ri]]; b=[Zte[:,ri]]
        for fam,idx in sorted(g.items()):
            if len(idx)<min_group:
                a.append(Ztr[:,idx]); b.append(Zte[:,idx]); continue
            k=min(n_comp,len(idx),Ztr.shape[0]-1)
            p=PCA(n_components=k,random_state=42).fit(Ztr[:,idx])
            a.append(p.transform(Ztr[:,idx])); b.append(p.transform(Zte[:,idx]))
        return np.hstack(a), np.hstack(b)
    return build, len(g)

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as hub
MODEL=hub.load('classification', device='cuda' if torch.cuda.is_available() else 'cpu')

def tabfm_fn(builder, seed):
    def f(Xtr,ytr,s,atr,Xte,ate):
        A,B=builder(Xtr,Xte)
        codes={v:i for i,v in enumerate(sorted(pd.unique(s)))}
        A=np.column_stack([A,[codes[v] for v in s]])
        B=np.column_stack([B,np.full(len(B),len(codes))])
        c=TabFMClassifier(MODEL,n_estimators=32,max_num_features=500,
                          n_svd_features='sqrt',random_state=seed)
        c.fit(A,ytr.astype(int)); return c.predict_proba(B)[:,1]
    return f

X=df[cols].to_numpy(float)
def run(label, comp_cols, seeds=(7,42,99)):
    b,nf=build_mixed(base, comp_cols)
    ms=[]
    for sd in seeds:
        folds=protocol.evaluate(tabfm_fn(b,sd),X,y,sites,ages,prevalence=prev,
                                n_boot=100,verbose=False)
        ms.append(protocol.mean_powered(folds))
    print(f'{label:44s} {np.mean(ms):.4f} +/- {np.std(ms):.4f}  ({nf} fams)', flush=True)

run('base raw + coherence compressed', coh)
run('base raw + bytecode compressed', byte)
run('base raw + self-ref compressed', selfref)
run('base raw + coherence + self-ref compressed', coh+selfref)
run('base raw + everything compressed', coh+byte+selfref)
print('\nreference: raw 159 = 0.7737, everything compressed incl. base = 0.7865')
