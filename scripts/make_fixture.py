#!/usr/bin/env python
"""Build a small train/holdout fixture from a Challenge data folder.

The holdout side is written in the shape the evaluation containers actually see:
a demographics.csv with no label columns, mirroring the supplementary set. Signal
files are symlinked, so the fixture costs almost no disk.

Usage:
    python scripts/make_fixture.py --src data/raw/training_set \
        --out data/fixtures/smoke --holdout-frac 0.25
"""

import argparse
import os
import shutil

import pandas as pd

SUBFOLDERS = ['physiological_data', 'algorithmic_annotations', 'human_annotations']
LABEL_COLUMNS = ['Cognitive_Impairment', 'Time_to_Event',
                 'Last_Known_Visit_Date', 'Time_to_Last_Visit']


def link_records(src, dst, rows):
    """Symlink every signal file belonging to `rows` from src into dst."""
    for sub in SUBFOLDERS:
        src_sub = os.path.join(src, sub)
        if not os.path.isdir(src_sub):
            continue
        for _, row in rows.iterrows():
            site = row['SiteID']
            src_dir = os.path.join(src_sub, site)
            if not os.path.isdir(src_dir):
                continue
            dst_dir = os.path.join(dst, sub, site)
            os.makedirs(dst_dir, exist_ok=True)
            prefix = f"{row['BidsFolder']}_ses-{row['SessionID']}"
            for name in os.listdir(src_dir):
                if name.startswith(prefix):
                    link = os.path.join(dst_dir, name)
                    if not os.path.exists(link):
                        os.symlink(os.path.abspath(os.path.join(src_dir, name)), link)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--holdout-frac', type=float, default=0.25)
    parser.add_argument('--limit', type=int, default=None,
                        help='cap the total number of records, for a fast smoke test')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    demo = pd.read_csv(os.path.join(args.src, 'demographics.csv'))
    if args.limit:
        # Stratify the cap by site so every site stays represented.
        demo = (demo.groupby('SiteID', group_keys=False)
                    .apply(lambda g: g.sample(min(len(g),
                                                  max(2, args.limit // demo['SiteID'].nunique())),
                                              random_state=args.seed)))

    holdout = demo.sample(frac=args.holdout_frac, random_state=args.seed)
    train = demo.drop(holdout.index)

    train_dir = os.path.join(args.out, 'training_data')
    holdout_dir = os.path.join(args.out, 'holdout_data')
    for d in (train_dir, holdout_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    train.to_csv(os.path.join(train_dir, 'demographics.csv'), index=False)
    link_records(args.src, train_dir, train)

    # The holdout demographics must not carry labels, matching validation/test.
    holdout.drop(columns=[c for c in LABEL_COLUMNS if c in holdout.columns]) \
           .to_csv(os.path.join(holdout_dir, 'demographics.csv'), index=False)
    link_records(args.src, holdout_dir, holdout)

    # Labels for scoring, and the training set as the prevalence reference —
    # the organizers state they compute prevalence from the training set.
    holdout.to_csv(os.path.join(args.out, 'holdout_labels.csv'), index=False)
    train.to_csv(os.path.join(args.out, 'prevalence.csv'), index=False)

    print(f'train={len(train)} holdout={len(holdout)}')
    print(f'train prevalence={train["Cognitive_Impairment"].mean():.3f} '
          f'holdout prevalence={holdout["Cognitive_Impairment"].mean():.3f}')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
