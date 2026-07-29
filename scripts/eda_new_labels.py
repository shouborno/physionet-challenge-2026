#!/usr/bin/env python
"""Characterize the official-phase labels before any modeling.

These measurements decide where the next three weeks go. The critical one is how
strongly age alone predicts the label: the primary metric only compares patients
within 2 years of each other, so any signal age carries is neutralized. If age
alone is a strong plain-AUROC predictor, age-residualization and the pairwise
ranking objective become the dominant lever; if it is near chance, that effort
belongs elsewhere.

Usage:
    python scripts/eda_new_labels.py --demographics data/raw/training_set_small/demographics.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.eval import challenge_metrics as cm  # noqa: E402

GAP = 2


def section(title):
    print(f'\n{"=" * 70}\n{title}\n{"=" * 70}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demographics', required=True)
    parser.add_argument('--compare', default=None,
                        help='a second demographics.csv (e.g. large) to compare against')
    parser.add_argument('--old', default=None,
                        help='the old 622-record demographics.csv, for label drift')
    args = parser.parse_args()

    df = pd.read_csv(args.demographics)

    section('1. SHAPE AND COLUMNS')
    print(f'rows={len(df)}  columns={len(df.columns)}')
    print(f'columns: {list(df.columns)}')
    print(f'\nCognitive_Impairment dtype: {df["Cognitive_Impairment"].dtype}')
    print(f'raw value counts:\n{df["Cognitive_Impairment"].value_counts(dropna=False)}')

    # Normalize the label the way helper_code.load_diagnoses does.
    def to_label(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, str):
            return 1.0 if v.casefold().strip() == 'true' else 0.0
        return 1.0 if v else 0.0

    df['y'] = df['Cognitive_Impairment'].map(to_label)
    labeled = df[df['y'].notna()].copy()

    section('2. PREVALENCE')
    print(f'labeled records: {len(labeled)} / {len(df)}')
    print(f'overall prevalence: {labeled["y"].mean():.4f} '
          f'({int(labeled["y"].sum())} positive)')
    print('\nper site:')
    per_site = labeled.groupby('SiteID').agg(
        n=('y', 'size'), positive=('y', 'sum'), prevalence=('y', 'mean'),
        age_mean=('Age', 'mean'), age_sd=('Age', 'std'))
    print(per_site.to_string())

    section('3. AGE ALONE AS A PREDICTOR  [the steering measurement]')
    sub = labeled[labeled['Age'].notna()]
    ages = sub['Age'].to_numpy(dtype=float)
    y = sub['y'].to_numpy(dtype=float)

    from sklearn.metrics import roc_auc_score
    plain = roc_auc_score(y, ages)
    conditioned, pairs = cm.auroc_age(y, ages, ages, gap=GAP, return_pairs=True)
    print(f'plain AUROC of age alone      : {plain:.4f}')
    print(f'age-conditioned AUROC of age  : {conditioned:.4f}  ({pairs} pairs)')
    print(f'\ninterpretation: the metric removes {plain - 0.5:.4f} of ranking '
          f'advantage that age carries under plain AUROC.')
    if plain >= 0.60:
        print('=> age is a STRONG confound. Age-residualization and the pairwise '
              'ranking objective are the dominant lever.')
    elif plain >= 0.55:
        print('=> age is a MODERATE confound. Age-aware modeling is worth doing.')
    else:
        print('=> age is a WEAK confound. Prioritize features and domain '
              'generalization over age-aware modeling.')

    section('4. AGE-MATCHED PAIR BUDGET (effective sample size)')
    total_pairs = int((y == 1).sum() * (y == 0).sum())
    print(f'all (pos, neg) pairs      : {total_pairs}')
    print(f'age-matched at gap={GAP}      : {pairs}  ({100 * pairs / max(total_pairs, 1):.1f}%)')
    print('\nper site, held out alone:')
    for site, g in sub.groupby('SiteID'):
        gy = g['y'].to_numpy(dtype=float)
        ga = g['Age'].to_numpy(dtype=float)
        n_pairs = cm.n_matched_pairs(gy, ga, gap=GAP)
        flag = '' if n_pairs >= 1000 else '   <-- UNDERPOWERED, do not select on this fold'
        print(f'  {site}: n={len(g):5d}  pos={int(gy.sum()):4d}  '
              f'age-matched pairs={n_pairs:8d}{flag}')

    section('5. PREVALENCE BY AGE (the organizers\' prevalence file)')
    prev = cm.prevalence_by_age(ages, y, ages, gap=GAP)
    for lo in range(int(np.nanmin(ages)) // 10 * 10, int(np.nanmax(ages)) + 10, 10):
        band = [p for a, p in prev.items() if lo <= a < lo + 10]
        n_band = int(((ages >= lo) & (ages < lo + 10)).sum())
        if band:
            print(f'  age {lo}-{lo + 9}: n={n_band:5d}  '
                  f'mean p_a={np.mean(band):.4f}  range {min(band):.3f}-{max(band):.3f}')

    section('6. AGE DISTRIBUTION AND THE 89-YEAR CAP')
    print(f'age: min={np.nanmin(ages):.0f} max={np.nanmax(ages):.0f} '
          f'mean={np.nanmean(ages):.1f} sd={np.nanstd(ages):.1f}')
    top = pd.Series(ages).value_counts().head(5)
    print(f'\nmost common ages:\n{top.to_string()}')
    capped = int((ages >= 89).sum())
    print(f'\nrecords at age >= 89: {capped} ({100 * capped / len(ages):.1f}%)')
    if capped > 0.02 * len(ages):
        print('=> HIPAA top-coding creates a pileup; that single band contributes '
              'a disproportionate share of age-matched pairs.')

    section('7. MISSINGNESS')
    for col in ['Age', 'Sex', 'Race', 'Ethnicity', 'BMI', 'Cognitive_Impairment']:
        if col in df.columns:
            miss = df[col].isna().mean()
            print(f'  {col:22s} {100 * miss:5.1f}% missing')

    section('8. TIME TO EVENT (auxiliary target, never a feature)')
    for col in ['Time_to_Event', 'Time_to_Last_Visit']:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(v):
                print(f'  {col}: n={len(v)} median={v.median():.0f}d '
                      f'p10={v.quantile(.1):.0f} p90={v.quantile(.9):.0f}')
    if 'Time_to_Event' in df.columns:
        tte = pd.to_numeric(labeled.loc[labeled['y'] == 1, 'Time_to_Event'],
                            errors='coerce').dropna()
        if len(tte):
            print(f'  positives only: median={tte.median():.0f}d '
                  f'({tte.median() / 365.25:.1f}y), '
                  f'range {tte.min():.0f}-{tte.max():.0f}')

    section('9. EXCLUDED ("Other") GROUP')
    unlabeled = df[df['y'].isna()]
    print(f'records with no usable label: {len(unlabeled)}')
    if len(unlabeled):
        print('=> these are extra unlabeled recordings, usable for the auxiliary '
              'head and for domain-adaptation experiments.')
        print(unlabeled.groupby('SiteID').size().to_string())

    if args.compare and os.path.exists(args.compare):
        section('10. SMALL vs LARGE OVERLAP')
        other = pd.read_csv(args.compare)
        a = set(df['SiteID'].astype(str) + '_' + df['BDSPPatientID'].astype(str))
        b = set(other['SiteID'].astype(str) + '_' + other['BDSPPatientID'].astype(str))
        print(f'this={len(a)} other={len(b)} intersection={len(a & b)}')
        print(f'this is a subset of other: {a <= b}')
        if a <= b:
            print('=> features extracted on the small set are reusable verbatim.')

    if args.old and os.path.exists(args.old):
        section('11. LABEL DRIFT vs THE UNOFFICIAL PHASE')
        old = pd.read_csv(args.old)
        old['y_old'] = old['Cognitive_Impairment'].map(to_label)
        key = ['SiteID', 'BDSPPatientID']
        merged = df.merge(old[key + ['y_old']], on=key, how='inner')
        both = merged[merged['y'].notna() & merged['y_old'].notna()]
        if len(both):
            flipped = int((both['y'] != both['y_old']).sum())
            print(f'records in both: {len(both)}  labels flipped: {flipped} '
                  f'({100 * flipped / len(both):.1f}%)')
            print(f'old prevalence={both["y_old"].mean():.3f}  '
                  f'new prevalence={both["y"].mean():.3f}')
            print('=> any result carried over from the unofficial phase is stale '
                  'by this much.')

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
