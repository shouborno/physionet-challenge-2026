#!/usr/bin/env python
"""Extract Philosopher's Stone latents for Challenge records.

Cost is entirely in preprocessing: the generalized Morse wavelet transform runs
about 80 s per night on CPU, while the forward pass is roughly 0.1 s on GPU.
So each shard computes its own spectrograms and reuses one loaded model, and
the array is sized for CPU throughput rather than GPU count.

Each record is embedded twice. With the true age covariate, as the model
intends, and with age pinned to the pretraining cohort mean. The model's first
regression target is age_z, and the challenge metric only compares patients
within two years of each other, so any component of the latent that merely
encodes age earns nothing. Holding both lets that be measured instead of
assumed.

Usage:
    python scripts/extract_ps_latents.py --data-folder ... --outdir ... \
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
    DEMOGRAPHICS_FILE, HEADERS, PHYSIOLOGICAL_DATA_SUBFOLDER,
    load_rename_rules, load_signal_data, standardize_channel_names_rename_only,
)
from src.features import philosopher as ps  # noqa: E402

BIPOLAR = [('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
           ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
           ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1'])]


def standardize(channels, fs_dict, csv_path):
    """Rename to canonical labels and derive the bipolar montages."""
    rules = load_rename_rules(csv_path)
    rename_map, drop = standardize_channel_names_rename_only(
        list(channels.keys()), rules)

    std, std_fs = {}, {}
    for old in channels:
        if old in drop:
            continue
        new = rename_map.get(old, old.lower())
        std[new] = channels[old]
        if old in fs_dict:
            std_fs[new] = fs_dict[old]

    # I0006 records unipolar, so the derivations the model expects have to be
    # constructed rather than read.
    for target, pos, negs in BIPOLAR:
        if target in std or pos not in std:
            continue
        for neg in negs:
            if neg in std and len(std[neg]) == len(std[pos]):
                std[target] = (np.asarray(std[pos], dtype=float)
                               - np.asarray(std[neg], dtype=float))
                std_fs[target] = std_fs.get(pos)
                break
    return std, std_fs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-folder', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--n-shards', type=int, default=1)
    parser.add_argument('--csv-path', default=os.path.join(REPO, 'channel_table.csv'))
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rows = pd.read_csv(os.path.join(args.data_folder,
                                    DEMOGRAPHICS_FILE)).to_dict('records')
    if args.limit:
        rows = rows[:args.limit]
    mine = [r for i, r in enumerate(rows) if i % args.n_shards == args.shard]
    print(f'shard {args.shard}/{args.n_shards}: {len(mine)} of {len(rows)}',
          flush=True)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, cfg = ps.load_ps_model(device=device)
    print(f'model ready on {device}', flush=True)

    done = failed = skipped = 0
    t0 = time.time()

    for i, row in enumerate(mine):
        bids = row[HEADERS['bids_folder']]
        session = row[HEADERS['session_id']]
        out_path = os.path.join(args.outdir, f'{bids}_ses-{session}.npz')
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        start = time.time()
        try:
            edf = os.path.join(args.data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                               row[HEADERS['site_id']], f'{bids}_ses-{session}.edf')
            if not os.path.exists(edf):
                failed += 1
                continue

            channels, fs_dict = load_signal_data(edf)
            std, std_fs = standardize(channels, fs_dict, args.csv_path)
            signal, fs, name = ps.pick_channel(std, std_fs)
            if signal is None:
                print(f'no EEG derivation for {bids}', flush=True)
                failed += 1
                continue

            specs = ps.make_spectrogram(ps.resample_to(signal, fs), cfg,
                                        channel_name=name)
            age = row.get(HEADERS['age'])
            sex = ps.sex_to_numeric(row.get(HEADERS['sex']))

            latent_true, heads_true = ps.infer_latent(
                model, specs, age, sex, device=device, pin_age=False)
            latent_pin, _ = ps.infer_latent(
                model, specs, age, sex, device=device, pin_age=True,
                want_heads=False)

            head_names = sorted(heads_true)
            head_values = np.concatenate(
                [np.ravel(heads_true[k]) for k in head_names]) if head_names \
                else np.array([])

            np.savez_compressed(
                out_path,
                latent_true=latent_true.astype(np.float32),
                latent_pinned=latent_pin.astype(np.float32),
                head_names=np.array(head_names),
                head_values=head_values.astype(np.float32),
                channel=name,
                elapsed=time.time() - start,
            )
            del channels, std, specs
            done += 1
        except Exception:
            failed += 1
            print(f'FAILED {bids}_ses-{session}', flush=True)
            traceback.print_exc()

        if (i + 1) % 5 == 0:
            rate = (time.time() - t0) / max(done, 1)
            print(f'  {i + 1}/{len(mine)} done={done} skip={skipped} '
                  f'fail={failed} {rate:.0f}s/record', flush=True)

    print(f'shard {args.shard} complete: done={done} skipped={skipped} '
          f'failed={failed} elapsed={time.time() - t0:.0f}s', flush=True)
    return 1 if failed and not done else 0


if __name__ == '__main__':
    sys.exit(main())
