"""Combine the two site findings, and re-test what site balancing was masking.

Two results from the re-test point the same way. Removing equal-site weighting
gains 0.0177 and adding site as a model covariate gains 0.0135, so the model
does better when allowed to see and use site structure rather than being forced
to ignore it. They are independent changes and should compose.

This also re-runs the coherence and recording-year questions without site
balancing, since that weighting was applied to every earlier feature comparison
and may have been suppressing effects.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.data.feature_io import load_features
from src.eval import challenge_metrics as cm
from src.eval import protocol

df, allcols = load_features('data/processed/features_large_v9.pkl')
y = df.label.to_numpy(float); sites = df.site_id.to_numpy(); ages = df.age.to_numpy(float)
prev = cm.prevalence_by_age(ages, y, ages, gap=2)
coh = [c for c in allcols if c.startswith(('coh_','iaf_','taf_','alpha_ratio_hi_lo'))]
print(f'{len(df)} records, {len(allcols)} features\n', flush=True)

def make(site_balanced=False, site_covariate=False):
    def f(Xtr, ytr, str_, atr, Xte, ate):
        if site_covariate:
            codes = {s: i for i, s in enumerate(sorted(pd.unique(str_)))}
            Xtr = np.column_stack([Xtr, [codes[s] for s in str_]])
            Xte = np.column_stack([Xte, np.full(len(Xte), len(codes))])
        p = Pipeline([('i', SimpleImputer(strategy='median')),
                      ('s', StandardScaler()),
                      ('m', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                          num_leaves=15, min_child_samples=50, colsample_bytree=0.5,
                          reg_lambda=1.0, random_state=42, verbose=-1))])
        kw = {}
        if site_balanced:
            c = pd.Series(str_).value_counts()
            w = np.array([1.0/c[s] for s in str_]); kw['m__sample_weight'] = w/w.mean()
        p.fit(Xtr, ytr, **kw)
        return p.predict_proba(Xte)[:, 1]
    return f

no_bmi = [c for c in allcols if c != 'bmi']
cases = [
  ('balanced, no covariate (old default)', no_bmi, True,  False),
  ('unbalanced',                           no_bmi, False, False),
  ('unbalanced + site covariate',          no_bmi, False, True),
  ('balanced + site covariate',            no_bmi, True,  True),
  ('unbalanced+cov, no coherence',         [c for c in no_bmi if c not in coh], False, True),
  ('unbalanced+cov, no rec_year',          [c for c in no_bmi if c != 'rec_year'], False, True),
  ('unbalanced+cov, with bmi',             allcols, False, True),
]
print(f"{'variant':38s} {'mean':>7s} {'worst':>7s}")
for name, cc, bal, cov in cases:
    X = df[cc].to_numpy(float)
    folds = protocol.evaluate(make(bal, cov), X, y, sites, ages,
                              prevalence=prev, n_boot=300, verbose=False)
    print(f'{name:38s} {protocol.mean_powered(folds):7.4f} '
          f'{folds[folds.powered].auroc_age.min():7.4f}', flush=True)
