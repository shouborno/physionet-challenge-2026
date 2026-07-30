#!/usr/bin/env python
"""Join the temporal pooling pass onto a feature matrix."""
import argparse, glob, os, sys
import pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--features', required=True)
ap.add_argument('--parts', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.parts, '*.pkl')))
tp = pd.DataFrame([pd.read_pickle(f) for f in files])
print(f'temporal records={len(tp)} columns={tp.shape[1]}')
base = pd.read_pickle(a.features)
merged = base.merge(tp, on=['patient_id', 'session_id'], how='left')
if len(merged) != len(base):
    raise SystemExit(f'join changed rows {len(base)} -> {len(merged)}')
new = [c for c in tp.columns if c not in ('patient_id', 'session_id')]
print(f'complete: {merged[new].notna().all(axis=1).sum()}/{len(merged)}')
merged.to_pickle(a.out)
print(f'wrote {a.out} shape={merged.shape}')
