#!/usr/bin/env python
"""Verify the bytecode-recovered extractors reproduce the cached feature matrix.

`data/processed/features_training_v6.pkl` was produced by the original sources
before they were deleted. If the modules loaded from bytecode reproduce it
value-for-value on the same records, the recovery is exact, and that cache
becomes the oracle for rewriting clean source later.

Needs the unofficial-phase EDFs, so it skips when they are not mounted.
Run under SLURM, not on the login node: each record is ~13 s of CPU.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

CACHE = os.path.join(REPO, 'data/processed/features_training_v6.pkl')
DATA = os.path.join(REPO, 'data/raw/training_set')

pytestmark = pytest.mark.skipif(
    not (os.path.exists(CACHE) and os.path.isdir(DATA)),
    reason='v6 cache or unofficial-phase EDFs unavailable',
)

N_RECORDS = int(os.environ.get('ORACLE_N_RECORDS', '3'))
TOLERANCE = 1e-9

# The one feature expected to differ from the cache. The March helper_code
# returned 0.0 for a missing BMI; the July one returns NaN. The cache was built
# under the old behavior, so a missing BMI reads 0.0 there and NaN now.
#
# The new behavior is the correct one and the difference matters: BMI is 75.9%
# missing in the official training set, so under the old code three quarters of
# patients carried a nonsense 0.0 that shifted the scaler and gave the model a
# de facto missingness indicator. Now they are NaN and get median-imputed.
EXPECTED_DIFFERENCES = {'bmi'}


@pytest.fixture(scope='module')
def cached():
    return pd.read_pickle(CACHE)


@pytest.fixture(scope='module')
def extractors():
    from src.data.pyc_bridge import load_feature_modules
    return load_feature_modules()


def test_cache_shape(cached):
    assert len(cached) == 622
    meta = {'patient_id', 'site_id', 'session_id', 'label', 'extract_time_sec'}
    assert len(set(cached.columns) - meta) == 272


def test_recovered_extractors_reproduce_cache(cached, extractors):
    """Re-extract a few records and compare against the cached values."""
    from helper_code import HEADERS, load_signal_data
    from scripts.extract_features import record_paths

    demo = pd.read_csv(os.path.join(DATA, 'demographics.csv'))
    by_bids = {r[HEADERS['bids_folder']]: r for r in demo.to_dict('records')}

    meta = {'patient_id', 'site_id', 'session_id', 'label', 'extract_time_sec'}
    feature_cols = [c for c in cached.columns if c not in meta]

    checked = 0
    mismatches = []
    seen_expected = set()

    for _, row in cached.iterrows():
        if checked >= N_RECORDS:
            break
        bids = row['patient_id']
        if bids not in by_bids:
            continue
        meta_row = by_bids[bids]

        phys_path, algo_path = record_paths(
            DATA, meta_row[HEADERS['site_id']], bids,
            meta_row[HEADERS['session_id']])
        if not os.path.exists(phys_path):
            continue

        phys_channels, phys_fs = load_signal_data(phys_path)
        algo_data = {}
        if os.path.exists(algo_path):
            algo_data, _ = load_signal_data(algo_path)

        fresh = extractors['features'].extract_all_features(
            meta_row, phys_channels, phys_fs, algo_data,
            compute_eeg=True, compute_hrv=True)

        for col in feature_cols:
            if col not in fresh:
                mismatches.append(f'{bids}: {col} missing from re-extraction')
                continue
            old, new = row[col], fresh[col]
            if pd.isna(old) and pd.isna(new):
                continue
            differs = (pd.isna(old) != pd.isna(new)) or not np.isclose(
                float(old), float(new), rtol=TOLERANCE, atol=TOLERANCE)
            if not differs:
                continue
            if col in EXPECTED_DIFFERENCES:
                seen_expected.add(col)
                continue
            mismatches.append(f'{bids}: {col} {old} vs {new}')

        checked += 1

    if checked == 0:
        pytest.skip('no cached record had its EDF available')

    assert not mismatches, (
        f'{len(mismatches)} unexpected mismatches over {checked} records; '
        f'first 10:\n' + '\n'.join(mismatches[:10]))
    print(f'\n{checked} records reproduced exactly across '
          f'{len(feature_cols) - len(seen_expected)} of {len(feature_cols)} '
          f'features; expected differences seen: {sorted(seen_expected) or "none"}')


def test_missing_bmi_is_nan_not_zero():
    """Pin the helper_code change that the oracle comparison surfaced.

    Guards against a future re-vendor silently restoring the 0.0 default, which
    would put a nonsense value into three quarters of the BMI column.
    """
    from helper_code import load_bmi
    assert np.isnan(load_bmi({}))
    assert np.isnan(load_bmi({'BMI': ''}))
    assert load_bmi({'BMI': 27.5}) == 27.5
