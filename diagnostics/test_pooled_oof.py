"""Does pooled out-of-fold scoring resolve differences the per-fold statistic cannot?

Every gain measured this session is smaller than the +/-0.039 subject-level
bootstrap SE on the single adequately powered fold. If pooling all four folds'
out-of-fold predictions roughly halves that, the same comparisons become
decidable.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.data.feature_io import load_features
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

df,cols=load_features('data/processed/features_small_v11.pkl',drop_age=True,drop_cols=['bmi'])
y=df.label.to_numpy(float); sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)

# The same three variants measured per-fold, now pooled.
variants={
 'full (951 feats)': cols,
 'no coherence':     [c for c in cols if not c.startswith(('coh_','iaf_','taf_','alpha_ratio_hi_lo'))],
 'no recording year':[c for c in cols if c!='rec_year'],
}
for name,cc in variants.items():
    X=df[cc].to_numpy(float)
    folds=protocol.evaluate(fit(),X,y,sites,ages,n_boot=400,verbose=False)
    pw=folds[folds.powered]
    pooled=protocol.pooled_out_of_fold(fit(),X,y,sites,ages,n_boot=800)
    print(f'{name:20s} ({len(cc):4d} feats)')
    print(f'    per-fold worst  {pw.auroc_age.min():.4f}  '
          f'pairs={int(pw.n_pairs.min()):5d}  sd={pw.boot_sd.max():.4f}')
    print(f'    pooled OOF      {pooled["pooled_auroc_age"]:.4f}  '
          f'pairs={pooled["n_pairs"]:5d}  sd={pooled["boot_sd"]:.4f}  '
          f'[{pooled["ci_low"]:.3f}, {pooled["ci_high"]:.3f}]')
