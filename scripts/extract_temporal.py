#!/usr/bin/env python
"""Temporal pooling extraction pass.

Separate from the other extractors so it can be added without recomputing
anything. Measured at about 0.5s per record, so a full pass over 6,600 records
is well under an hour even at modest parallelism.
"""
import argparse, os, sys, time, traceback, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from helper_code import (ALGORITHMIC_ANNOTATIONS_SUBFOLDER, DEMOGRAPHICS_FILE,
                         HEADERS, PHYSIOLOGICAL_DATA_SUBFOLDER, load_signal_data)
from src.data.features_coherence import standardize_channels
from src.data.features_temporal import extract_temporal_pooling_features

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--data-folder', required=True)
ap.add_argument('--outdir', required=True)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--n-shards', type=int, default=1)
ap.add_argument('--csv-path', default=os.path.join(REPO, 'channel_table.csv'))
ap.add_argument('--overwrite', action='store_true')
a = ap.parse_args()

os.makedirs(a.outdir, exist_ok=True)
rows = pd.read_csv(os.path.join(a.data_folder, DEMOGRAPHICS_FILE)).to_dict('records')
mine = [r for i, r in enumerate(rows) if i % a.n_shards == a.shard]
print(f'shard {a.shard}/{a.n_shards}: {len(mine)} of {len(rows)}', flush=True)

done = failed = skipped = 0
t0 = time.time()
for i, row in enumerate(mine):
    bids = row[HEADERS['bids_folder']]; sess = row[HEADERS['session_id']]
    site = row[HEADERS['site_id']]
    out = os.path.join(a.outdir, f'{bids}_ses-{sess}.pkl')
    if os.path.exists(out) and not a.overwrite:
        skipped += 1; continue
    try:
        phys = os.path.join(a.data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site,
                            f'{bids}_ses-{sess}.edf')
        algo_p = os.path.join(a.data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                              site, f'{bids}_ses-{sess}_caisr_annotations.edf')
        if not os.path.exists(phys):
            failed += 1; continue
        ch, fs = load_signal_data(phys)
        algo = {}
        if os.path.exists(algo_p):
            algo, _ = load_signal_data(algo_p)
        std, sfs = standardize_channels(ch, fs, a.csv_path)
        feats = extract_temporal_pooling_features(std, sfs, algo.get('stage_caisr'))
        pd.to_pickle({'patient_id': bids, 'session_id': sess, **feats}, out)
        del ch, std, algo
        done += 1
    except Exception:
        failed += 1
        print(f'FAILED {bids}_ses-{sess}', flush=True)
        traceback.print_exc()
    if (i + 1) % 50 == 0:
        print(f'  {i+1}/{len(mine)} done={done} skip={skipped} fail={failed} '
              f'{(time.time()-t0)/max(done,1):.1f}s/rec', flush=True)

print(f'shard {a.shard} complete: done={done} skipped={skipped} failed={failed} '
      f'elapsed={time.time()-t0:.0f}s', flush=True)
sys.exit(1 if failed and not done else 0)
