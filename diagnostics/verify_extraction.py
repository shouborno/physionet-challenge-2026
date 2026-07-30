"""Confirm the shipped extractor emits exactly the features the model expects.

The 159-feature set is 95 from the inline extractor, 63 temporal pooling
features and recording year. Temporal pooling lived only in the offline
scripts, so it had to be wired into team_code.py; this checks the wiring
produces the same names the trained model was fitted on, since a silent name
mismatch would surface as a column of NaNs at inference rather than an error.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, pandas as pd
import team_code
from helper_code import find_patients
from src.data.feature_io import load_features

R='/scratch/simran/pn26/raw/training_set_small'
recs=find_patients(R+'/demographics.csv')
out=team_code._extract_one(R, recs[0], R+'/demographics.csv', 'channel_table.csv')
if out is None:
    raise SystemExit('extraction returned None')
fd, label, age = out[0], out[1], out[2]
tp=[k for k in fd if k.startswith('tp_')]
print(f'extractor emits {len(fd)} features ({len(tp)} temporal)')
finite_tp=sum(1 for k in tp if np.isfinite(fd[k]))
print(f'temporal finite: {finite_tp}/{len(tp)}')

# Names must match what the offline matrix used, or the model sees NaN columns.
df, cols = load_features('data/processed/features_large_v10.pkl', drop_cols=['bmi'])
offline_tp=set(c for c in cols if c.startswith('tp_'))
emitted_tp=set(tp)
missing=sorted(offline_tp-emitted_tp); extra=sorted(emitted_tp-offline_tp)
print(f'\nnames in training matrix but not emitted: {len(missing)}')
if missing: print('  ', missing[:6])
print(f'names emitted but not in training matrix: {len(extra)}')
if extra: print('  ', extra[:6])

import team_code as tc
inline=set(fd)-emitted_tp
in_matrix=len(inline & set(cols))
print(f'\ninline features also in training matrix: {in_matrix}/{len(inline)}')
shippable=sorted((inline & set(cols)) | offline_tp | {'rec_year'})
print(f'shippable feature count: {len(shippable)}')
print('OK' if not missing else 'MISMATCH')
