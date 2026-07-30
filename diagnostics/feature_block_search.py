"""Which combination of shippable feature blocks is best?

Everything today used one fixed set of 159: the 95 inline features plus the 63
temporal ones plus rec_year. Those blocks have never been varied against each
other. Temporal in particular was only ever judged as an addition to a large
matrix, where it read as a null; it has never been tried on its own or against
the inline block.

This is a search over five blocks rather than over individual features, so it
is eight comparisons rather than thousands. Feature-level selection was tested
separately and lost to using everything at every k, which is consistent with
the unofficial-phase collapse; block-level choices are few enough to be honest
about.

Each candidate is run at three seeds, because the differences that matter here
are around 0.005 and single-seed numbers have already misled twice today.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.feature_io import load_features
from src.eval import challenge_metrics as cm
from src.eval import protocol
import team_code
from helper_code import find_patients

R='/scratch/simran/pn26/raw/training_set_small'
inline_names=set(team_code._extract_one(R, find_patients(R+'/demographics.csv')[0],
                                        R+'/demographics.csv','channel_table.csv')[0].keys())
df, cols = load_features('data/processed/features_large_v10.pkl', drop_cols=['bmi'])
inline=sorted(set(cols)&inline_names)
tp=[c for c in cols if c.startswith('tp_')]
coh=[c for c in cols if c.startswith(('coh_','iaf_','taf_','alpha_ratio_hi_lo'))]
yr=['rec_year'] if 'rec_year' in cols else []

y=df.label.to_numpy(float); sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)
print(f'blocks: inline {len(inline)}, temporal {len(tp)}, coherence {len(coh)}, '
      f'rec_year {len(yr)}\n', flush=True)

def score(cc, seed):
    def f(Xtr,ytr,s,atr,Xte,ate):
        p=Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),
                    ('m',lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,
                        num_leaves=15,min_child_samples=50,colsample_bytree=0.5,
                        reg_lambda=1.0,random_state=seed,verbose=-1))])
        p.fit(Xtr,ytr); return p.predict_proba(Xte)[:,1]
    folds=protocol.evaluate(f, df[cc].to_numpy(float), y, sites, ages,
                            prevalence=prev, n_boot=100, verbose=False)
    return protocol.mean_powered(folds), folds[folds.powered].auroc_age.min()

cands={
 'inline+temporal+year (current)': inline+tp+yr,
 'inline+year':                    inline+yr,
 'temporal+year':                  tp+yr,
 'inline+temporal':                inline+tp,
 'year only':                      yr,
 'inline+temporal+coh+year':       inline+tp+coh+yr,
 'temporal+coh+year':              tp+coh+yr,
}
print(f"{'blocks':34s} {'n':>5s} {'mean':>16s} {'worst':>8s}")
for name, cc in cands.items():
    if not cc: continue
    ms, ws = zip(*[score(cc, s) for s in (7,42,99)])
    m=np.array(ms)
    print(f'{name:34s} {len(cc):5d} {m.mean():.4f} +/- {m.std():.4f}  '
          f'{np.mean(ws):8.4f}', flush=True)
