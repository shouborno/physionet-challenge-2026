"""Can TabFM run inside the submission container?

Three constraints decide it. The evaluation GPU is optional, so CPU has to
work. Inference is in-context, meaning every prediction carries the whole
6,600-row training context through a forward pass, so per-record cost is what
matters rather than a one-off fit. And the container allows 96 hours without a
GPU for training plus 48 for inference.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/scratch/simran/pn26/tabfm')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, torch
from sklearn.impute import SimpleImputer
from src.data.feature_io import load_features

df, cols = load_features('data/processed/features_large_v9.pkl',
                         drop_age=True, drop_cols=['bmi'])
X = SimpleImputer(strategy='median').fit_transform(df[cols].to_numpy(float))
y = df.label.to_numpy(int)
print(f'{X.shape[0]} rows, {X.shape[1]} features', flush=True)

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as hub
print(f'torch threads: {torch.get_num_threads()}', flush=True)

for dev in ('cpu',):
    t = time.time(); model = hub.load('classification', device=dev)
    print(f'\n[{dev}] weights loaded in {time.time()-t:.0f}s', flush=True)
    clf = TabFMClassifier(model, n_estimators=8, max_num_features=500,
                          n_svd_features='sqrt', random_state=42)
    t = time.time(); clf.fit(X, y); fit_s = time.time()-t
    print(f'[{dev}] fit on {len(y)} rows: {fit_s:.1f}s', flush=True)
    for n in (5, 25, 100):
        t = time.time(); clf.predict_proba(X[:n]); el = time.time()-t
        print(f'[{dev}] predict {n:4d} rows: {el:6.1f}s  '
              f'({el/n:.2f}s/record)', flush=True)
        if el/n > 0.5:
            print(f'[{dev}] projected for 1,000 validation records: '
                  f'{el/n*1000/3600:.1f}h', flush=True)
            break
    else:
        el_per = el/n
        print(f'[{dev}] projected 1,000 records: {el_per*1000/3600:.2f}h, '
              f'10,000: {el_per*10000/3600:.1f}h', flush=True)
