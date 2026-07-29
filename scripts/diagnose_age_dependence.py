#!/usr/bin/env python
"""Ask whether age is actually a problem for our models.

Three separate methods built to exploit the metric's age-matched structure have
now failed to beat a plain classifier: a within-stratum pairwise ranking loss,
spline age residualization, and exact conditional logistic regression. Three
failures with one shared premise is worth testing directly, because the premise
may simply be false.

The premise is that our models rank substantially on age and that removing that
dependence will help. If instead the fitted models barely use age, then all
three methods are solving a problem we do not have, and every one of them pays
a variance cost for nothing.

Measurements, in increasing order of directness:

  1. Correlation between the model's score and age. If near zero, there is
     nothing to remove.
  2. Plain AUROC minus age-conditioned AUROC. Age alone shows a gap of 0.234
     (0.772 against 0.538), so a model with a small gap is already ranking on
     something other than age.
  3. Simply dropping age and its close proxies from the feature set. This is
     the cheapest possible intervention and needs no estimated trend.

Usage:
    python scripts/diagnose_age_dependence.py --features data/processed/features_small_v8.pkl
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

from sklearn.metrics import roc_auc_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.data.self_norm import META  # noqa: E402
from src.eval import challenge_metrics as cm  # noqa: E402
from src.eval import protocol  # noqa: E402


def lgbm_fit():
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    def fit_predict(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=15,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.5,
                reg_lambda=1.0, random_state=42, verbose=-1)),
        ])
        counts = pd.Series(sites_tr).value_counts()
        w = np.array([1.0 / counts[s] for s in sites_tr])
        pipe.fit(X_tr, y_tr, model__sample_weight=w / w.mean())
        return pipe.predict_proba(X_te)[:, 1]

    return fit_predict


def out_of_fold_scores(fit, X, y, sites, ages):
    """Collect held-out scores from every leave-one-site-out fold."""
    scores = np.full(len(y), np.nan)
    for _name, tr, te in protocol.site_folds(sites):
        if not _name.startswith('holdout_'):
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        scores[te] = fit(X[tr], y[tr], sites[tr], ages[tr], X[te], ages[te])
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', default='results/age_dependence.json')
    args = parser.parse_args()

    df = pd.read_pickle(args.features)
    df = df[df['label'].notna()].copy()

    all_cols = [c for c in df.columns
                if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    y = df['label'].to_numpy(dtype=float)
    sites = df['site_id'].to_numpy()
    ages = df['age'].to_numpy(dtype=float)
    prevalence = cm.prevalence_by_age(ages, y, ages, gap=2)

    # Which features are close proxies for age?
    proxies = []
    for c in all_cols:
        v = df[c].to_numpy(dtype=float)
        ok = np.isfinite(v) & np.isfinite(ages)
        if ok.sum() < 50:
            continue
        r = abs(spearmanr(v[ok], ages[ok]).statistic)
        if np.isfinite(r) and r >= 0.35:
            proxies.append((c, float(r)))
    proxies.sort(key=lambda t: -t[1])

    print(f'records={len(df)} features={len(all_cols)} prevalence={y.mean():.4f}')
    print(f'\nfeatures correlated with age at |rho| >= 0.35: {len(proxies)}')
    for c, r in proxies[:12]:
        print(f'   {c:38s} rho={r:.3f}')

    fit = lgbm_fit()
    variants = {
        'all_features': all_cols,
        'without_age': [c for c in all_cols if c != 'age'],
        'without_age_and_proxies': [c for c in all_cols
                                    if c != 'age'
                                    and c not in {p for p, _ in proxies}],
    }

    report = {'n_proxies': len(proxies),
              'proxies': proxies[:40], 'variants': {}}

    print('\n' + '=' * 74)
    for name, cols in variants.items():
        if len(cols) < 5:
            continue
        X = df[cols].to_numpy(dtype=float)
        oof = out_of_fold_scores(fit, X, y, sites, ages)
        ok = np.isfinite(oof)

        plain = roc_auc_score(y[ok], oof[ok])
        conditioned, pairs = cm.auroc_age(y[ok], oof[ok], ages[ok], gap=2,
                                          return_pairs=True)
        rho = spearmanr(oof[ok], ages[ok]).statistic

        folds = protocol.evaluate(fit, X, y, sites, ages, prevalence=prevalence,
                                  n_boot=500, verbose=False)
        powered = folds[folds['powered']]
        worst = float(powered['auroc_age'].min()) if not powered.empty else np.nan

        print(f'{name}  ({len(cols)} features)')
        print(f'   pooled out-of-fold: plain AUROC={plain:.4f}  '
              f'age-conditioned={conditioned:.4f}  gap={plain - conditioned:+.4f}')
        print(f'   score-age Spearman rho = {rho:+.4f}')
        print(f'   worst powered fold     = {worst:.4f}')
        print()

        report['variants'][name] = {
            'n_features': len(cols), 'plain_auroc': float(plain),
            'auroc_age': float(conditioned), 'gap': float(plain - conditioned),
            'score_age_rho': float(rho), 'worst_powered': worst,
            'n_pairs': int(pairs),
        }

    ref_plain = roc_auc_score(y, ages)
    ref_cond = cm.auroc_age(y, ages, ages, gap=2)
    print(f'reference, age alone: plain={ref_plain:.4f}  '
          f'age-conditioned={ref_cond:.4f}  gap={ref_plain - ref_cond:+.4f}')

    base = report['variants'].get('all_features', {})
    if base:
        print('\nINTERPRETATION')
        if abs(base['score_age_rho']) < 0.25:
            print('  The fitted model barely ranks on age, so there is little '
                  'age dependence to remove. That would explain why three '
                  'separate age-aware methods each cost accuracy without '
                  'buying anything: they pay a variance price to solve a '
                  'problem this model does not have.')
        else:
            print('  The model does rank substantially on age, so age-aware '
                  'methods target something real and their failure is about '
                  'execution rather than premise.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
