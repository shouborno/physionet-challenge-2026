"""How fast is TabFM inference when queries are batched?

The submission harness calls run_model once per record, but nothing requires
predicting one at a time: the first call can predict every record in the folder
and serve the rest from cache. That is ordinary batching, not transductive
learning, since no statistic of the test set enters the model.

It matters because TabFM is in-context. Every forward pass carries the whole
6,600-row training context, so one-at-a-time inference pays that cost per
record while batching amortizes it. On CPU with 16 threads a 5-row batch did
not finish in 16 minutes; this measures the GPU path we would actually request,
and the CPU fallback at a batch size worth using.
"""
import sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/scratch/simran/pn26/tabfm')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')
import numpy as np, torch
from sklearn.impute import SimpleImputer
from src.data.feature_io import load_features

df, cols = load_features('data/processed/features_large_v9.pkl', drop_cols=['bmi'])
X = SimpleImputer(strategy='median').fit_transform(df[cols].to_numpy(float))
y = df.label.to_numpy(int)
print(f'{X.shape[0]} context rows, {X.shape[1]} features', flush=True)

from tabfm import TabFMClassifier, tabfm_v1_0_0_pytorch as hub
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device={dev}', flush=True)
model = hub.load('classification', device=dev)

clf = TabFMClassifier(model, n_estimators=32, max_num_features=500,
                      n_svd_features='sqrt', random_state=42)
t = time.time(); clf.fit(X, y)
print(f'fit {len(y)} rows: {time.time()-t:.1f}s', flush=True)

for n in (1, 10, 100, 1000):
    n = min(n, len(X))
    t = time.time(); clf.predict_proba(X[:n]); el = time.time() - t
    print(f'predict {n:5d}: {el:7.1f}s  ({el/n*1000:7.1f} ms/record)', flush=True)

# The validation set is one site; the test set is larger. Project both.
print('\nprojection at the largest measured batch:', flush=True)
t = time.time(); clf.predict_proba(X[:1000]); per = (time.time()-t)/1000
for size, label in ((1000, 'validation'), (5000, 'test if 5x larger')):
    print(f'  {label:20s} {size:5d} records: {per*size/3600:.2f}h '
          f'(limit 48h)', flush=True)
