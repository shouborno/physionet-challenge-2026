"""Can TabPFN v2 run here, and does it beat LightGBM under LOSO?

TabPFN is a prior-fitted transformer for small tabular problems, the regime
with the strongest published edge at roughly our sample size. Version 8 gates
weight download behind interactive license acceptance, so this uses 2.0.9,
which is the version the benchmark literature refers to. It caps around 500
features against our 951, so dimensionality comes down by in-fold PCA: PCA is
unsupervised and cannot leak the label, while supervised selection over 951
columns at 62 effective positives is the mechanism behind the earlier 0.136
optimism gap.
"""
import sys, warnings, time
warnings.filterwarnings('ignore')
sys.path.insert(0, '/scratch/simran/pn26/tabpfn2')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd
from src.data.feature_io import load_features
from src.eval import challenge_metrics as cm
from src.eval import protocol

df, cols = load_features('data/processed/features_small_v11.pkl',
                         drop_age=True, drop_cols=['bmi'])
X = df[cols].to_numpy(float)
y = df.label.to_numpy(float); sites = df.site_id.to_numpy(); ages = df.age.to_numpy(float)
prev = cm.prevalence_by_age(ages, y, ages, gap=2)
print(f'{len(df)} records, {len(cols)} features')

def tabpfn_fit(ncomp=200):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    def f(Xtr, ytr, str_, atr, Xte, ate):
        from tabpfn import TabPFNClassifier
        pre = Pipeline([('i', SimpleImputer(strategy='median')),
                        ('s', StandardScaler()),
                        ('p', PCA(n_components=min(ncomp, Xtr.shape[1], Xtr.shape[0]-1),
                                  random_state=42))])
        Ztr = pre.fit_transform(Xtr); Zte = pre.transform(Xte)
        clf = TabPFNClassifier(device='cuda' if __import__('torch').cuda.is_available() else 'cpu', random_state=42)
        clf.fit(Ztr, ytr.astype(int))
        return clf.predict_proba(Zte)[:, 1]
    return f

t = time.time()
folds = protocol.evaluate(tabpfn_fit(), X, y, sites, ages, prevalence=prev,
                          n_boot=300, verbose=True)
pw = folds[folds.powered]
print(f'\nTabPFN worst powered fold = {pw.auroc_age.min():.4f}  '
      f'({time.time()-t:.0f}s)')
print('LightGBM reference on the same features = 0.6786')
