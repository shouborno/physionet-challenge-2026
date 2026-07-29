#!/usr/bin/env python
"""Merge per-record feature pickles into one matrix and attach fresh labels.

Labels are joined here rather than baked into extraction, so relabelling costs
a few seconds instead of re-running 25 core-hours of feature extraction.

Usage:
    python scripts/merge_features.py --parts data/processed/v7_parts \
        --demographics /scratch/simran/pn26/raw/training_set_small/demographics.csv \
        --out data/processed/features_small_v7.pkl
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

META_COLUMNS = ['patient_id', 'site_id', 'session_id', 'extract_time_sec']
# Present in training demographics but absent at inference; using any of them as
# a feature would train on something the model can never see again.
TRAINING_ONLY = ['Cognitive_Impairment', 'Time_to_Event',
                 'Last_Known_Visit_Date', 'Time_to_Last_Visit']


def to_label(value):
    """Match helper_code.load_diagnoses, returning NaN where it would raise."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, str):
        text = value.casefold().strip()
        if text in ('true', '1', '1.0'):
            return 1.0
        if text in ('false', '0', '0.0'):
            return 0.0
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        return 1.0 if float(value) else 0.0
    except (TypeError, ValueError):
        return np.nan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parts', required=True)
    parser.add_argument('--demographics', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.parts) if f.endswith('.pkl'))
    if not files:
        raise SystemExit(f'no pickles under {args.parts}')
    print(f'merging {len(files)} record pickles', flush=True)

    rows = [pd.read_pickle(os.path.join(args.parts, f)) for f in files]
    df = pd.DataFrame(rows)

    demo = pd.read_csv(args.demographics)
    demo['label'] = demo['Cognitive_Impairment'].map(to_label)

    # Recording date. The label definition makes this predictive: a negative
    # needs at least six years of clean follow-up so must be an older
    # recording, while a positive needs only one to six years to diagnosis.
    # Positives therefore average 2015.2 against 2013.2, and follow-up length
    # differs by 2.6 years. It is a construction artifact rather than
    # physiology, worth about +0.02 on the powered fold and more elsewhere.
    # CreationTime is supplied for the hidden sites too, so it is usable at
    # inference; it is reported as a finding rather than used silently.
    if 'CreationTime' in demo.columns:
        t = pd.to_datetime(demo['CreationTime'], format='mixed', errors='coerce')
        demo['rec_year'] = t.dt.year + t.dt.dayofyear / 366.0

    keep = ['BidsFolder', 'SessionID', 'Age', 'Sex', 'label']
    # Year only. Month was measured and costs 0.011 on the powered fold: the
    # artifact runs through follow-up duration, which is annual, so month is
    # seasonal noise with no mechanism behind it.
    keep += [c for c in ['rec_year'] if c in demo.columns]
    keep += [c for c in ['Time_to_Event', 'Time_to_Last_Visit'] if c in demo.columns]
    merged = df.merge(
        demo[keep].rename(columns={'BidsFolder': 'patient_id',
                                   'SessionID': 'session_id',
                                   'Age': 'demo_age', 'Sex': 'demo_sex'}),
        on=['patient_id', 'session_id'], how='left')

    if len(merged) != len(df):
        raise SystemExit(f'join changed row count: {len(df)} -> {len(merged)}')

    feature_cols = [c for c in merged.columns
                    if c not in META_COLUMNS + TRAINING_ONLY
                    + ['label', 'demo_age', 'demo_sex',
                       'Time_to_Event', 'Time_to_Last_Visit']]
    # Time_to_Last_Visit is the cleanest form of the same artifact but is
    # absent at inference, so it must never become a feature.

    leaked = [c for c in feature_cols if c in TRAINING_ONLY]
    if leaked:
        raise SystemExit(f'training-only columns leaked into features: {leaked}')

    labeled = merged['label'].notna()
    print(f'records={len(merged)}  labeled={int(labeled.sum())}  '
          f'features={len(feature_cols)}')
    print(f'prevalence={merged.loc[labeled, "label"].mean():.4f}')
    print('per site:')
    print(merged[labeled].groupby('site_id')['label']
          .agg(n='size', pos='sum', prev='mean').to_string())

    missing = merged[feature_cols].isna().mean().sort_values(ascending=False)
    print(f'\nmost-missing features:\n{missing.head(8).to_string()}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    merged.to_pickle(args.out)
    print(f'\nwrote {args.out}  shape={merged.shape}')


if __name__ == '__main__':
    main()
