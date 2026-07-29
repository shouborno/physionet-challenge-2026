#!/usr/bin/env python
"""Compare feature subsets under the challenge metric.

Adding the 130 self-referential features on top of the original 272 made things
slightly worse, which does not by itself condemn them: with 84 positives, 402
columns may simply dilute. This separates the two explanations by scoring the
derived features alone, the originals alone, and site-filtered subsets of each.

Usage:
    python scripts/feature_set_ablation.py --features data/processed/features_small_v8.pkl
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.data.self_norm import META, find_stage_families, site_predictiveness  # noqa: E402
from src.eval import challenge_metrics as cm  # noqa: E402
from src.eval import protocol  # noqa: E402


def lgbm_fit(site_balanced=True):
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_predict(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=15,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.5,
                reg_lambda=1.0, random_state=42, verbose=-1)),
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
    parser.add_argument('--out', default='results/feature_set_ablation.json')
    parser.add_argument('--site-auc-max', type=float, default=0.65,
                        help='drop features whose site-predictiveness exceeds this')
    args = parser.parse_args()

    df = pd.read_pickle(args.features)
    df = df[df['label'].notna()].copy()

    numeric = [c for c in df.columns
               if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    # Derived columns carry the naming produced by self_norm.
    derived = [c for c in numeric
               if '_vs_' in c or '_z_' in c or c.endswith('_stagerange')]
    original = [c for c in numeric if c not in derived]

    y = df['label'].to_numpy(dtype=float)
    sites = df['site_id'].to_numpy()
    ages = df['age'].to_numpy(dtype=float)
    prevalence = cm.prevalence_by_age(ages, y, ages, gap=2)

    leak = {c: site_predictiveness(df[c].to_numpy(), sites) for c in numeric}

    def keep_low_leak(cols):
        return [c for c in cols
                if np.isfinite(leak.get(c, np.nan))
                and leak[c] <= args.site_auc_max]

    subsets = {
        'original_272': original,
        'derived_only': derived,
        'original_plus_derived': original + derived,
        'original_lowleak': keep_low_leak(original),
        'derived_lowleak': keep_low_leak(derived),
        'all_lowleak': keep_low_leak(numeric),
    }

    print(f'records={len(df)} prevalence={y.mean():.4f}')
    for name, cols in subsets.items():
        print(f'  {name:26s} {len(cols):4d} features')

    fit = lgbm_fit()
    results, summary = {}, []
    for name, cols in subsets.items():
        if len(cols) < 5:
            print(f'\n--- {name}: too few features, skipped ---')
            continue
        print(f'\n--- {name} ({len(cols)} features) ---')
        X = df[cols].to_numpy(dtype=float)
        folds = protocol.evaluate(fit, X, y, sites, ages, prevalence=prevalence,
                                  n_boot=1000)
        results[name] = folds.to_dict('records')
        powered = folds[folds['powered']]
        summary.append((name, len(cols),
                        protocol.selection_statistic(folds),
                        float(powered['auroc_age'].min()) if not powered.empty
                        else float('nan')))

    print('\n' + '=' * 72)
    print('FEATURE SET RANKING (worst powered fold)')
    print('=' * 72)
    for name, n, sel, worst in sorted(summary, key=lambda t: -t[3]):
        print(f'  {name:26s} n={n:4d}  worst={worst:.4f}  select={sel:.4f}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
