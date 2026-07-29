#!/usr/bin/env python
"""Tune LightGBM against the challenge metric, not a generic objective.

LightGBM leads every comparison so far, but only one configuration has been
tried. This searches the parameters that matter at 84 positives, where the risk
is overfitting the trees to a handful of cases rather than underfitting.

Selection is deliberately conservative. Scoring is the worst adequately powered
fold, and the search reports how far the best configuration sits above the
median so the gain can be read against the spread of the search itself. With
one powered fold and roughly 5,000 age-matched pairs, a search over dozens of
configurations can manufacture 0.02-0.05 of apparent improvement from noise
alone, which is exactly the mechanism that turned 0.780 into 0.644 in the
unofficial phase.

Usage:
    python scripts/tune_lgbm.py --features data/processed/features_small_v8.pkl
"""

import argparse
import itertools
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.data.self_norm import META  # noqa: E402
from src.eval import challenge_metrics as cm  # noqa: E402
from src.eval import protocol  # noqa: E402

GRID = {
    'num_leaves': [7, 15, 31],
    'min_child_samples': [20, 50, 100],
    'learning_rate': [0.02, 0.05],
    'n_estimators': [200, 500],
    'colsample_bytree': [0.3, 0.6],
    'reg_lambda': [1.0, 10.0],
}


def make_fit(params, site_balanced=True):
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    def fit_predict(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', lgb.LGBMClassifier(random_state=42, verbose=-1,
                                         subsample=0.8, subsample_freq=1,
                                         **params)),
        ])
        kwargs = {}
        if site_balanced:
            counts = pd.Series(sites_tr).value_counts()
            w = np.array([1.0 / counts[s] for s in sites_tr])
            kwargs['model__sample_weight'] = w / w.mean()
        pipe.fit(X_tr, y_tr, **kwargs)
        return pipe.predict_proba(X_te)[:, 1]

    return fit_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', default='results/lgbm_tuning.json')
    parser.add_argument('--max-configs', type=int, default=48)
    parser.add_argument('--n-boot', type=int, default=400)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    df = pd.read_pickle(args.features)
    df = df[df['label'].notna()].copy()
    cols = [c for c in df.columns
            if c not in META and pd.api.types.is_numeric_dtype(df[c])]

    X = df[cols].to_numpy(dtype=float)
    y = df['label'].to_numpy(dtype=float)
    sites = df['site_id'].to_numpy()
    ages = df['age'].to_numpy(dtype=float)
    prevalence = cm.prevalence_by_age(ages, y, ages, gap=2)

    keys = list(GRID)
    everything = [dict(zip(keys, combo))
                  for combo in itertools.product(*(GRID[k] for k in keys))]
    rng = np.random.default_rng(args.seed)
    if len(everything) > args.max_configs:
        pick = rng.choice(len(everything), args.max_configs, replace=False)
        configs = [everything[i] for i in sorted(pick)]
    else:
        configs = everything

    print(f'records={len(df)} features={len(cols)} prevalence={y.mean():.4f}')
    print(f'searching {len(configs)} of {len(everything)} configurations\n')

    scored = []
    for i, params in enumerate(configs, 1):
        folds = protocol.evaluate(make_fit(params), X, y, sites, ages,
                                  prevalence=prevalence, n_boot=args.n_boot,
                                  verbose=False)
        powered = folds[folds['powered']]
        if powered.empty:
            continue
        worst = float(powered['auroc_age'].min())
        scored.append({'params': params, 'worst': worst,
                       'select': protocol.selection_statistic(folds),
                       'mean_powered': float(powered['auroc_age'].mean())})
        print(f'  [{i:3d}/{len(configs)}] worst={worst:.4f}  {params}',
              flush=True)

    scored.sort(key=lambda d: -d['worst'])
    values = np.array([s['worst'] for s in scored])

    print('\n' + '=' * 74)
    print('TOP CONFIGURATIONS')
    print('=' * 74)
    for s in scored[:5]:
        print(f'  worst={s["worst"]:.4f}  select={s["select"]:.4f}  {s["params"]}')

    print(f'\nsearch spread: median={np.median(values):.4f}  '
          f'best={values.max():.4f}  worst={values.min():.4f}  '
          f'sd={values.std():.4f}')
    print(f'best sits {values.max() - np.median(values):+.4f} above the median.')
    print('With one powered fold, a search this size can produce a gain of that '
          'order from noise. Treat it as a candidate, not a result, until it '
          'holds on the larger dataset.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(scored, fh, indent=2, default=float)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
