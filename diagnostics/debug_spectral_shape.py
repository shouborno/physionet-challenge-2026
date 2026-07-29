"""Find why extract_spectral_shape_features returns all-NaN.

Six features (aperiodic exponents, alpha peak parameters) came back 100% missing
across all 1,103 records, which is a silent failure rather than a data property.
"""
import sys, warnings, traceback, inspect
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')

import numpy as np
import pandas as pd

from src.data.pyc_bridge import load_feature_modules
from helper_code import load_signal_data

m = load_feature_modules()
R = '/scratch/simran/pn26/raw/training_set_small'
row = pd.read_csv(R + '/demographics.csv').to_dict('records')[0]
p = f"{R}/physiological_data/{row['SiteID']}/{row['BidsFolder']}_ses-{row['SessionID']}.edf"
a = (f"{R}/algorithmic_annotations/{row['SiteID']}/{row['BidsFolder']}"
     f"_ses-{row['SessionID']}_caisr_annotations.edf")

ch, fs = load_signal_data(p)
algo, _ = load_signal_data(a)

fe = m['features_eeg']
print('signature:', inspect.signature(fe.extract_spectral_shape_features))

stage = algo.get('stage_caisr')
print('stage_caisr:', type(stage).__name__,
      getattr(stage, 'shape', None), 'unique:',
      np.unique(stage)[:8] if stage is not None else None)

cache = fe.precompute_eeg_cache(ch, fs, stage)
print('cache type:', type(cache))
if hasattr(cache, 'keys'):
    print('cache keys:', list(cache)[:12])
    for k in list(cache)[:12]:
        v = cache[k]
        print(f'   {k}: {type(v).__name__}'
              + (f' len={len(v)}' if hasattr(v, "__len__") else ''))

try:
    out = fe.extract_spectral_shape_features(ch, fs, stage, cache)
    print('\noutput:')
    for k, v in out.items():
        print(f'   {k} = {v}')
except Exception:
    traceback.print_exc()
