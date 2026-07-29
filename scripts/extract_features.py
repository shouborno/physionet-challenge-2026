#!/usr/bin/env python
"""Extract the full research feature pool, as a resumable SLURM array worker.

Writes one pickle per record under `--outdir`, so a task that dies takes only
its own shard with it and a rerun skips whatever is already on disk. The pool
will grow as self-referential features are added, and this gets rerun each time.

Extraction is ~13 s of CPU per record, so 6,600 records is roughly 25 core-hours
in total. The real cost is EDF I/O over Ceph, not compute, which is why the
array is sized for parallel reads rather than for cores.

Usage (single shard):
    python scripts/extract_features.py --data-folder /scratch/.../training_set_small \
        --outdir data/processed/v7_parts --shard 3 --n-shards 128
"""

import argparse
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from helper_code import (  # noqa: E402
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER, DEMOGRAPHICS_FILE, HEADERS,
    PHYSIOLOGICAL_DATA_SUBFOLDER, load_signal_data,
)


def load_extractors():
    """Recover the research extractors from bytecode (see src/data/pyc_bridge)."""
    from src.data.pyc_bridge import load_feature_modules
    mods = load_feature_modules()
    return mods


def record_paths(data_folder, site_id, bids_folder, session_id):
    phys = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER, site_id,
                        f'{bids_folder}_ses-{session_id}.edf')
    algo = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id,
                        f'{bids_folder}_ses-{session_id}_caisr_annotations.edf')
    return phys, algo


def extract_record(mods, row, data_folder, csv_path):
    """Run the full extractor stack on one record."""
    site_id = row[HEADERS['site_id']]
    bids = row[HEADERS['bids_folder']]
    session = row[HEADERS['session_id']]

    phys_path, algo_path = record_paths(data_folder, site_id, bids, session)

    phys_channels, phys_fs = ({}, {})
    if os.path.exists(phys_path):
        phys_channels, phys_fs = load_signal_data(phys_path)
    algo_data = {}
    if os.path.exists(algo_path):
        algo_data, _ = load_signal_data(algo_path)

    features = mods['features']
    feats = features.extract_all_features(
        row, phys_channels, phys_fs, algo_data,
        compute_eeg=True, compute_hrv=True)

    del phys_channels, algo_data
    return feats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-folder', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--csv-path', default=os.path.join(REPO, 'channel_table.csv'))
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    demo = pd.read_csv(os.path.join(args.data_folder, DEMOGRAPHICS_FILE))
    rows = demo.to_dict('records')
    if args.limit:
        rows = rows[:args.limit]

    mine = [r for i, r in enumerate(rows) if i % args.n_shards == args.shard]
    print(f'shard {args.shard}/{args.n_shards}: {len(mine)} of {len(rows)} records',
          flush=True)

    mods = load_extractors()
    print(f'loaded extractor modules: {sorted(mods)}', flush=True)

    done = failed = skipped = 0
    t0 = time.time()

    for i, row in enumerate(mine):
        bids = row[HEADERS['bids_folder']]
        session = row[HEADERS['session_id']]
        out_path = os.path.join(args.outdir, f'{bids}_ses-{session}.pkl')

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        start = time.time()
        try:
            feats = extract_record(mods, row, args.data_folder, args.csv_path)
            record = {
                'patient_id': bids,
                'site_id': row[HEADERS['site_id']],
                'session_id': session,
                'extract_time_sec': time.time() - start,
                **feats,
            }
            # Labels stay out of the feature cache: they are regenerated from
            # demographics at training time so a label revision does not
            # silently invalidate 25 core-hours of extraction.
            pd.to_pickle(record, out_path)
            done += 1
        except Exception:
            failed += 1
            print(f'FAILED {bids}_ses-{session}', flush=True)
            traceback.print_exc()

        if (i + 1) % 10 == 0:
            rate = (time.time() - t0) / max(done, 1)
            print(f'  {i + 1}/{len(mine)}  done={done} skip={skipped} '
                  f'fail={failed}  {rate:.1f}s/record', flush=True)

    print(f'shard {args.shard} complete: done={done} skipped={skipped} '
          f'failed={failed} elapsed={time.time() - t0:.0f}s', flush=True)
    return 1 if failed and not done else 0


if __name__ == '__main__':
    sys.exit(main())
