"""Do better feature-selection methods help, or is selection itself the problem?

Only one method has been tried: top-k by LightGBM gain, which lost to using
everything at every k from 50 to 500. But that is a weak selector. It scores
features independently, so it keeps whole clusters of near-duplicates and
cannot see a feature that matters only alongside another.

Five methods with different failure modes, all fitted strictly inside the
training folds so the held-out site never influences what is chosen:

  permutation importance   measures the drop from shuffling a feature, so it
                           reflects what the model actually uses rather than
                           how often it split
  L1 / elastic-net         selects jointly, dropping redundant columns because
                           the penalty is shared
  Boruta-style shadow      keeps a feature only if it beats a randomly
                           permuted copy of itself, which sets the bar at
                           chance rather than at an arbitrary k
  correlation pruning      removes near-duplicates first, then ranks, which
                           targets redundancy directly
  variance pruning         removes near-constant columns, the cheapest sanity
                           filter

The baseline is all 1,015 features, and the reference is the 159 shippable
ones, which no selection over the full set has yet matched.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.feature_io import load_features
from src.eval import challenge_metrics as cm
from src.eval import protocol

df, cols = load_features('data/processed/features_large_v10.pkl', drop_cols=['bmi'])
X=df[cols].to_numpy(float); y=df.label.to_numpy(float)
sites=df.site_id.to_numpy(); ages=df.age.to_numpy(float)
prev=cm.prevalence_by_age(ages,y,ages,gap=2)
print(f'{len(df)} records, {len(cols)} features\n', flush=True)

def base_model(seed=42):
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=50, colsample_bytree=0.5,
                              reg_lambda=1.0, random_state=seed, verbose=-1)

def run(selector, seed=42):
    """selector(Xtr, ytr) -> column indices, chosen inside the fold only."""
    def f(Xtr,ytr,s,atr,Xte,ate):
        idx = selector(Xtr, ytr) if selector else np.arange(Xtr.shape[1])
        if len(idx)==0: idx=np.arange(Xtr.shape[1])
        p=Pipeline([('i',SimpleImputer(strategy='median')),('s',StandardScaler()),
                    ('m',base_model(seed))])
        p.fit(Xtr[:,idx],ytr)
        return p.predict_proba(Xte[:,idx])[:,1]
    folds=protocol.evaluate(f,X,y,sites,ages,prevalence=prev,n_boot=100,verbose=False)
    return protocol.mean_powered(folds), folds[folds.powered].auroc_age.min()

def _prep(Xtr):
    return SimpleImputer(strategy='median').fit_transform(Xtr)

def sel_permutation(k):
    from sklearn.inspection import permutation_importance
    def s(Xtr,ytr):
        Z=_prep(Xtr)
        m=base_model(); m.fit(Z,ytr)
        r=permutation_importance(m,Z,ytr,n_repeats=3,random_state=42,
                                 scoring='roc_auc',n_jobs=8)
        return np.argsort(r.importances_mean)[::-1][:k]
    return s

def sel_l1(C):
    def s(Xtr,ytr):
        Z=StandardScaler().fit_transform(_prep(Xtr))
        m=LogisticRegression(penalty='l1',solver='liblinear',C=C,max_iter=2000)
        m.fit(Z,ytr)
        idx=np.flatnonzero(np.abs(m.coef_[0])>1e-8)
        return idx
    return s

def sel_shadow():
    """Keep features beating the best of their own permuted copies."""
    def s(Xtr,ytr):
        Z=_prep(Xtr)
        rng=np.random.default_rng(42)
        shadow=np.column_stack([rng.permutation(Z[:,j]) for j in range(Z.shape[1])])
        both=np.column_stack([Z,shadow])
        m=base_model(); m.fit(both,ytr)
        imp=m.booster_.feature_importance('gain')
        n=Z.shape[1]
        threshold=np.percentile(imp[n:],95)
        return np.flatnonzero(imp[:n]>threshold)
    return s

def sel_decorrelate(thresh, k):
    def s(Xtr,ytr):
        Z=_prep(Xtr)
        m=base_model(); m.fit(Z,ytr)
        order=np.argsort(m.booster_.feature_importance('gain'))[::-1]
        keep=[]
        for j in order:
            if len(keep)>=k: break
            if not keep or np.max(np.abs(np.corrcoef(
                    np.column_stack([Z[:,j],Z[:,keep]]),rowvar=False)[0,1:]))<thresh:
                keep.append(j)
        return np.array(keep)
    return s

def sel_variance(frac):
    def s(Xtr,ytr):
        Z=_prep(Xtr)
        v=Z.std(axis=0)
        return np.flatnonzero(v>np.quantile(v,frac))
    return s

methods=[('all 1015 (no selection)', None),
         ('decorrelate 0.9, top-200', sel_decorrelate(0.9,200)),
         ('decorrelate 0.95, top-400', sel_decorrelate(0.95,400)),
         ('decorrelate 0.98, all', sel_decorrelate(0.98,1015))]

print(f"{'method':30s} {'mean':>8s} {'worst':>8s}  {'secs':>5s}")
for name, sel in methods:
    t=time.time()
    try:
        m,w=run(sel)
        print(f'{name:30s} {m:8.4f} {w:8.4f}  {time.time()-t:5.0f}', flush=True)
    except Exception as e:
        print(f'{name:30s} FAILED {type(e).__name__}: {str(e)[:40]}', flush=True)
print('\nreference: 159 shippable features = 0.7496 +/- 0.0014')
