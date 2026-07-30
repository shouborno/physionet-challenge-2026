"""Which block does family-wise compression actually rescue?

NSC1 on all 1,015 features reached 0.7881, the best result here. On the 709
shippable ones it reaches 0.7694, below the 0.7737 that raw 159 features get.
So the gain is not coming from compressed coherence, which the 709 set already
contains. It has to come from the 306 features that are missing there: 176
bytecode-only extractor outputs and 130 self-referential columns derived from
them.

That distinction decides real work. If the gain lives in the bytecode block,
those extractors have to be rewritten in clean source before any of it can
ship, which is a substantial job justified only by a real effect. If it lives
in the self-referential block, that code is already clean (src/data/self_norm.py)
and only its inputs are bytecode.

Each block is added to the 159-feature base, compressed, and measured at three
seeds.
"""
import sys, re, time, warnings, collections
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
print(f'base {len(base)}, coherence {len(coh)}, self-ref {len(selfref)}, '
      f'bytecode-only {len(byte)}\n', flush=True)

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

def nsc_builder(colnames, n_comp=1, min_group=3):
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
            p=PCA(n_components=k,random_state=42).fit(Ztr[:,idx])
            a.append(p.transform(Ztr[:,idx])); b.append(p.transform(Zte[:,idx]))
        return np.hstack(a), np.hstack(b)
    return build, len(g)

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

def run(label, cc, compress=True, seeds=(7,42,99)):
    X=df[cc].to_numpy(float)
    builder, nfam = nsc_builder(cc) if compress else (None, 0)
    ms,ws=[],[]
    for sd in seeds:
        folds=protocol.evaluate(tabfm_fn(builder,sd),X,y,sites,ages,
                                prevalence=prev,n_boot=100,verbose=False)
        ms.append(protocol.mean_powered(folds))
        ws.append(folds[folds.powered].auroc_age.min())
    tag=f'{len(cc)}f' + (f'/{nfam}fam' if compress else ' raw')
    print(f'{label:36s} {tag:12s} {np.mean(ms):.4f} +/- {np.std(ms):.4f}  '
          f'worst {np.mean(ws):.4f}', flush=True)

run('base 159, raw', base, compress=False)
run('base + bytecode, NSC1', base+byte)
run('base + self-ref, NSC1', base+selfref)
run('base + bytecode + self-ref, NSC1', base+byte+selfref)
run('base + coherence, NSC1', base+coh)
print('\nreference: NSC1 on all 1015 = 0.7881')
