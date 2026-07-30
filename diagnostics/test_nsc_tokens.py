"""GOTabPFN's compression idea, applied to TabFM with family-based grouping.

GOTabPFN (ICML 2026) makes TabPFN-style models work on wide tables by ordering
features so correlated ones are adjacent, cutting the ordered axis at
low-correlation boundaries, and replacing each segment with its first principal
component. One scalar per correlated group, so the model sees compact tokens
instead of a wide raw table.

That matters here because TabFM caps at 500 features and applies a *global* SVD
above the cap, which is the blunt reduction the paper argues against. At 1,015
features it is compressing nearly half the input with no regard for structure.

The adaptation: their graph-ordering step exists to discover which features
belong together, and our column names already say. coh_c3_c4_alpha_n2 is a
coherence feature, tp_delta_q88 is temporal, eeg_n2_delta is spectral. So the
grouping is free, and only the per-group PCA is needed.

Fitted inside training folds only, so the held-out site never influences the
components.
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

df, cols = load_features('data/processed/features_large_v10.pkl', drop_cols=['bmi'])
X = df[cols].to_numpy(float); y = df.label.to_numpy(float)
sites = df.site_id.to_numpy(); ages = df.age.to_numpy(float)
prev = cm.prevalence_by_age(ages, y, ages, gap=2)

def family_of(name):
    """Group by semantic family, which the column names already encode."""
    if name.startswith('coh_'):
        m = re.match(r'coh_[a-z0-9]+_[a-z0-9]+_([a-z0-9]+)_([a-z0-9]+)', name)
        return f'coh_{m.group(1)}_{m.group(2)}' if m else 'coh_other'
    if name.startswith('tp_'):
        m = re.match(r'tp_([a-z0-9]+)_', name)
        return f'tp_{m.group(1)}' if m else 'tp_other'
    m = re.match(r'^([a-z]+(?:_[a-z0-9]+)?)', name)
    return m.group(1) if m else 'other'

groups = collections.defaultdict(list)
for i, c in enumerate(cols):
    groups[family_of(c)].append(i)
sizes = sorted((len(v) for v in groups.values()), reverse=True)
print(f'{len(cols)} features in {len(groups)} families; largest {sizes[:6]}\n', flush=True)

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as hub
MODEL = hub.load('classification', device='cuda' if torch.cuda.is_available() else 'cpu')

def nsc_tokens(n_comp_per_group, min_group=3):
    """Segment-PCA: each family of >= min_group columns becomes n components."""
    def build(Xtr, Xte):
        imp = SimpleImputer(strategy='median'); sc = StandardScaler()
        Ztr = sc.fit_transform(imp.fit_transform(Xtr))
        Zte = sc.transform(imp.transform(Xte))
        tr_parts, te_parts = [], []
        for fam, idx in sorted(groups.items()):
            if len(idx) < min_group:
                tr_parts.append(Ztr[:, idx]); te_parts.append(Zte[:, idx]); continue
            k = min(n_comp_per_group, len(idx), Ztr.shape[0] - 1)
            p = PCA(n_components=k, random_state=42).fit(Ztr[:, idx])
            tr_parts.append(p.transform(Ztr[:, idx]))
            te_parts.append(p.transform(Zte[:, idx]))
        return np.hstack(tr_parts), np.hstack(te_parts)
    return build

def tabfm_fit(builder=None):
    def f(Xtr, ytr, s, atr, Xte, ate):
        if builder is None:
            imp = SimpleImputer(strategy='median')
            A, B = imp.fit_transform(Xtr), imp.transform(Xte)
        else:
            A, B = builder(Xtr, Xte)
        codes = {v: i for i, v in enumerate(sorted(pd.unique(s)))}
        A = np.column_stack([A, [codes[v] for v in s]])
        B = np.column_stack([B, np.full(len(B), len(codes))])
        c = TabFMClassifier(MODEL, n_estimators=32, max_num_features=500,
                            n_svd_features='sqrt', random_state=42)
        c.fit(A, ytr.astype(int))
        return c.predict_proba(B)[:, 1]
    return f

cases = [('tabfm raw 1015 (global SVD)', None),
         ('NSC 1 comp/family', nsc_tokens(1)),
         ('NSC 2 comp/family', nsc_tokens(2)),
         ('NSC 3 comp/family', nsc_tokens(3))]
print(f"{'variant':30s} {'dims':>6s} {'mean':>8s} {'worst':>8s}  {'secs':>5s}")
for name, b in cases:
    t = time.time()
    if b is not None:
        d = b(X[:100], X[:10])[0].shape[1]
    else:
        d = X.shape[1]
    folds = protocol.evaluate(tabfm_fit(b), X, y, sites, ages,
                              prevalence=prev, n_boot=100, verbose=False)
    print(f'{name:30s} {d:6d} {protocol.mean_powered(folds):8.4f} '
          f'{folds[folds.powered].auroc_age.min():8.4f}  {time.time()-t:5.0f}', flush=True)
print('\nreference: tabfm on 159 shippable = 0.7737 +/- 0.0006')
