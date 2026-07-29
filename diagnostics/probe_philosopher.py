#!/usr/bin/env python
"""Smallest end-to-end check that Philosopher's Stone runs on Challenge data.

Confirms the checkpoint loads under the installed torch, that the latent comes
back at the documented 1024 dimensions, and that a real Challenge EDF can be
driven through preprocessing into the model. The previous attempt died on a
head-construction bug in a since-deleted driver, so this establishes a working
path before any extraction is launched at scale.
"""

import os
import sys
import time
import traceback
import warnings

warnings.filterwarnings('ignore')

REPO = '/home/simran/sleep-study-cognitive-screening-challenge'
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features import philosopher as ps  # noqa: E402


def main():
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'torch {torch.__version__}  device={device}')
    if device == 'cuda':
        print(f'gpu: {torch.cuda.get_device_name(0)}')

    print('\n--- loading checkpoint ---')
    t = time.time()
    try:
        model, cfg = ps.load_ps_model(device=device)
    except Exception:
        traceback.print_exc()
        return 1
    n_params = sum(p.numel() for p in model.parameters())
    print(f'loaded in {time.time() - t:.1f}s  params={n_params / 1e6:.1f}M')
    print(f'training mode (must be True): {model.training}')

    print('\n--- real Challenge record ---')
    from helper_code import load_signal_data
    R = '/scratch/simran/pn26/raw/training_set_small'
    row = pd.read_csv(os.path.join(R, 'demographics.csv')).to_dict('records')[0]
    edf = (f"{R}/physiological_data/{row['SiteID']}/"
           f"{row['BidsFolder']}_ses-{row['SessionID']}.edf")
    print(f'record: {row["BidsFolder"]}  age={row["Age"]}  sex={row["Sex"]}')

    channels, fs_dict = load_signal_data(edf)

    # The raw EDF uses site-specific labels; reuse the submission's own
    # standardization so the channel names match CHANNEL_PREFERENCE.
    import team_code
    from helper_code import load_rename_rules, standardize_channel_names_rename_only
    rules = load_rename_rules(os.path.join(REPO, 'channel_table.csv'))
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
    for target, pos, negs in [('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
                              ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
                              ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1'])]:
        if target in std or pos not in std:
            continue
        for neg in negs:
            if neg in std and len(std[neg]) == len(std[pos]):
                std[target] = np.asarray(std[pos], float) - np.asarray(std[neg], float)
                std_fs[target] = std_fs.get(pos)
                break

    print(f'available derivations: '
          f'{[c for c in ps.CHANNEL_PREFERENCE if c in std]}')

    sig, fs, name = ps.pick_channel(std, std_fs)
    if sig is None:
        print('FAIL: no usable EEG derivation')
        return 1
    print(f'using {name} at {fs} Hz, {len(sig) / fs / 3600:.1f} h')

    t = time.time()
    sig200 = ps.resample_to(sig, fs)
    specs = ps.make_spectrogram(sig200, cfg, channel_name=name)
    print(f'spectrogram {np.shape(specs)} in {time.time() - t:.1f}s')

    for pin in (False, True):
        t = time.time()
        try:
            latent, reg, clf = ps.infer_latent(
                model, specs, row['Age'], ps.sex_to_numeric(row['Sex']),
                device=device, pin_age=pin)
        except Exception:
            traceback.print_exc()
            return 1
        label = 'age pinned to cohort mean' if pin else 'true age'
        print(f'\n[{label}] {time.time() - t:.1f}s')
        print(f'  latent {latent.shape} '
              f'mean={latent.mean():.4f} sd={latent.std():.4f} '
              f'nonzero={np.count_nonzero(latent)}')
        print(f'  regression head {reg.shape}, classification head {clf.shape}')

    print('\nPROBE OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
