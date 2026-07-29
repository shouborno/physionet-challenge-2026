#!/usr/bin/env python
"""Join the coherence pass onto an existing feature matrix.

Coherence runs as a separate extraction so it can be added without recomputing
the 272 base features. This merges the per-record pickles in and reports how
site-identifying the new columns are relative to the old, since the argument
for coherence is partly that it is scale-invariant and so should leak less.
"""
import argparse, glob, os, sys
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.data.self_norm import META, site_predictiveness  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--features', required=True)
ap.add_argument('--coherence-parts', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.coherence_parts, '*.pkl')))
if not files:
    raise SystemExit(f'no coherence pickles in {a.coherence_parts}')
coh = pd.DataFrame([pd.read_pickle(f) for f in files])
print(f'coherence records={len(coh)} columns={coh.shape[1]}')

base = pd.read_pickle(a.features)
merged = base.merge(coh, on=['patient_id', 'session_id'], how='left')
if len(merged) != len(base):
    raise SystemExit(f'join changed row count {len(base)} -> {len(merged)}')

new_cols = [c for c in coh.columns
            if c not in ('patient_id', 'session_id', 'coh_time_sec')]
old_cols = [c for c in base.columns
            if c not in META and pd.api.types.is_numeric_dtype(base[c])]

matched = merged[new_cols].notna().all(axis=1).sum()
print(f'records with complete coherence: {matched}/{len(merged)}')
print(f'features {len(old_cols)} -> {len(old_cols) + len(new_cols)}')

sites = merged['site_id'].to_numpy()
def leak(cols):
    s = np.array([site_predictiveness(merged[c].to_numpy(dtype=float), sites)
                  for c in cols])
    s = s[np.isfinite(s)]
    return (s.mean(), np.median(s), (s > 0.65).mean()) if s.size else (np.nan,)*3

for name, cols in (('existing', old_cols), ('coherence', new_cols)):
    m, md, hi = leak(cols)
    print(f'  {name:10s} site-predictiveness mean={m:.4f} median={md:.4f} '
          f'>0.65: {100*hi:.1f}%')

merged.to_pickle(a.out)
print(f'wrote {a.out} shape={merged.shape}')
