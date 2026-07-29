#!/usr/bin/env python
"""Extract coherence and alpha-frequency features as a separate pass.

Kept apart from the main extractor so it can be run and re-run without
recomputing the 272 existing features. Coherence costs about 6 s per record
against 9 s for everything else, so a standalone pass over 1,103 records is
roughly two core-hours.

Usage:
    python scripts/extract_coherence.py --data-folder ... --outdir ... \
        --shard 0 --n-shards 16
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
from src.data.features_coherence import (  # noqa: E402
    extract_all_connectivity_features, standardize_channels,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-folder', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--csv-path', default=os.path.join(REPO, 'channel_table.csv'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = pd.read_csv(os.path.join(args.data_folder,
                                    DEMOGRAPHICS_FILE)).to_dict('records')
    mine = [r for i, r in enumerate(rows) if i % args.n_shards == args.shard]
    print(f'shard {args.shard}/{args.n_shards}: {len(mine)} of {len(rows)}',
          flush=True)

    done = failed = skipped = 0
    t0 = time.time()

    for i, row in enumerate(mine):
        bids = row[HEADERS['bids_folder']]
        session = row[HEADERS['session_id']]
        site = row[HEADERS['site_id']]
        out_path = os.path.join(args.outdir, f'{bids}_ses-{session}.pkl')
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        start = time.time()
        try:
            phys = os.path.join(args.data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                                site, f'{bids}_ses-{session}.edf')
            algo_path = os.path.join(
                args.data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site,
                f'{bids}_ses-{session}_caisr_annotations.edf')
            if not os.path.exists(phys):
                failed += 1
                continue

            channels, fs_dict = load_signal_data(phys)
            algo = {}
            if os.path.exists(algo_path):
                algo, _ = load_signal_data(algo_path)

            std, std_fs = standardize_channels(channels, fs_dict, args.csv_path)
            feats = extract_all_connectivity_features(
                std, std_fs, algo.get('stage_caisr'))

            pd.to_pickle({'patient_id': bids, 'session_id': session,
                          'coh_time_sec': time.time() - start, **feats},
                         out_path)
            del channels, std, algo
            done += 1
        except Exception:
            failed += 1
            print(f'FAILED {bids}_ses-{session}', flush=True)
            traceback.print_exc()

        if (i + 1) % 20 == 0:
            print(f'  {i + 1}/{len(mine)} done={done} skip={skipped} '
                  f'fail={failed} {(time.time() - t0) / max(done, 1):.1f}s/rec',
                  flush=True)

    print(f'shard {args.shard} complete: done={done} skipped={skipped} '
          f'failed={failed} elapsed={time.time() - t0:.0f}s', flush=True)
    return 1 if failed and not done else 0


if __name__ == '__main__':
    sys.exit(main())
