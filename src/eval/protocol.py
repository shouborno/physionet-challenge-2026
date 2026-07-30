#!/usr/bin/env python
"""Leave-one-site-out evaluation that does not flatter itself.

In the unofficial phase, LOSO cross-validation reported 0.780 while the
leaderboard returned 0.644, and the configuration chosen on worst-site LOSO
scored *worse* on validation than the one it replaced. Two causes:

1. Greedy forward selection over 272 features maximizing the worst-site LOSO
   statistic is a search over the evaluation statistic itself. Roughly 7,000
   evaluations against a statistic with bootstrap sd ~0.025 buys 0.05-0.10 of
   optimistic bias for free. The fix is regularization, not selection.
2. Folds were treated as interchangeable. On the official small training set,
   holding out I0002 leaves 48 age-matched pairs and I0006 leaves 441, against
   5,124 for S0001. A fold resting on 48 comparisons is noise, and averaging it
   into a single "mean LOSO" hides that.

So: every fold reports its age-matched pair count and a patient-level bootstrap
interval, folds below a pair threshold are reported but never selected on, and
the selection statistic is a lower confidence bound rather than a point estimate.
"""

import numpy as np
import pandas as pd

from . import challenge_metrics as cm

MIN_PAIRS_FOR_SELECTION = 1000
DEFAULT_GAP = 2


def site_folds(sites, primary_only=False):
    """Yield (name, train_mask, test_mask) for each evaluation fold.

    Beyond the standard leave-one-site-out folds, this adds the inverse of the
    dominant-site fold: train on the largest site alone and test on the pooled
    remainder. That is the closest structural analogue of the real setup, where
    a large training corpus is scored against one unseen site.
    """
    sites = np.asarray(sites)
    unique = sorted(pd.unique(sites))

    largest = max(unique, key=lambda s: int((sites == s).sum()))

    folds = []
    for site in unique:
        test = sites == site
        folds.append((f'holdout_{site}', ~test, test))

    if len(unique) > 1:
        test = sites != largest
        folds.append((f'train_{largest}_only', ~test, test))

    if primary_only:
        keep = {f'holdout_{largest}', f'train_{largest}_only'}
        folds = [f for f in folds if f[0] in keep]

    return folds


def evaluate(fit_predict, X, y, sites, ages, gap=DEFAULT_GAP, n_boot=2000,
             seed=0, prevalence=None, verbose=True):
    """Run every fold and return a per-fold DataFrame.

    `fit_predict(X_train, y_train, sites_train, ages_train, X_test, ages_test)`
    must return a probability (or any monotone score) per test row.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    sites = np.asarray(sites)
    ages = np.asarray(ages, dtype=float)

    rows = []
    for name, tr, te in site_folds(sites):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            if verbose:
                print(f'  {name}: skipped (single class)')
            continue

        scores = fit_predict(X[tr], y[tr], sites[tr], ages[tr], X[te], ages[te])
        scores = np.asarray(scores, dtype=float).ravel()

        binary = None
        if prevalence is not None:
            binary = cm.optimal_binary(
                scores, ages[te], prevalence,
                prior_train=float(y[tr].mean()), prior_target=float(y[te].mean()))

        report = cm.score_fold(y[te], scores, ages[te], gap=gap, n_boot=n_boot,
                               seed=seed, prevalence=prevalence, binary=binary)
        report['fold'] = name
        report['n_train'] = int(tr.sum())
        report['powered'] = report['n_pairs'] >= MIN_PAIRS_FOR_SELECTION
        rows.append(report)

        if verbose:
            flag = '' if report['powered'] else '  [underpowered]'
            print(f'  {name:22s} auroc_age={report["auroc_age"]:.4f} '
                  f'[{report["ci_low"]:.3f}, {report["ci_high"]:.3f}]  '
                  f'pairs={report["n_pairs"]:6d}{flag}')

    cols = ['fold', 'auroc_age', 'ci_low', 'ci_high', 'boot_sd', 'n_pairs',
            'powered', 'n', 'n_pos', 'n_train']
    df = pd.DataFrame(rows)
    if 'reward' in df.columns:
        cols.append('reward')
    return df[[c for c in cols if c in df.columns]]


def pooled_out_of_fold(fit_predict, X, y, sites, ages, gap=DEFAULT_GAP,
                       n_boot=2000, seed=0, rank_normalize=True):
    """Score every record once, from a model that never saw its site.

    The per-fold statistic is the honest one but it is starved: the only
    adequately powered fold carries 5,124 age-matched pairs and a subject-level
    bootstrap SE near 0.039, which is wider than any gain measured so far. This
    predicts each record from the leave-one-site-out model for its own site and
    scores all of them together, recovering the pairs that fall across folds and
    roughly halving the interval.

    The catch is that predictions then come from different models, whose score
    scales need not agree, and a pair spanning two folds compares two scales.
    Rank-normalizing within fold removes that. It is not free: per-fold
    normalization is not a global monotone transform, so it does change the
    metric slightly, which is why this is reported alongside the per-fold
    numbers rather than replacing them.

    It is also optimistic relative to deployment, where one model scores every
    record. Use it to rank candidates, not to predict the leaderboard.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    sites = np.asarray(sites)
    ages = np.asarray(ages, dtype=float)

    scores = np.full(len(y), np.nan)
    for name, tr, te in site_folds(sites):
        if not name.startswith('holdout_'):
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        fold_scores = np.asarray(
            fit_predict(X[tr], y[tr], sites[tr], ages[tr], X[te], ages[te]),
            dtype=float).ravel()
        if rank_normalize and len(fold_scores) > 1:
            order = np.argsort(np.argsort(fold_scores))
            fold_scores = (order + 0.5) / len(fold_scores)
        scores[te] = fold_scores

    ok = np.isfinite(scores)
    if ok.sum() < 50:
        return {}

    point, pairs = cm.auroc_age(y[ok], scores[ok], ages[ok], gap=gap,
                                return_pairs=True)
    lo, hi, sd = cm.bootstrap_ci(y[ok], scores[ok], ages[ok], gap=gap,
                                 n_boot=n_boot, seed=seed)
    return {'pooled_auroc_age': point, 'n_pairs': pairs, 'ci_low': lo,
            'ci_high': hi, 'boot_sd': sd, 'n_scored': int(ok.sum())}


def selection_statistic(folds_df):
    """The single number to rank models by.

    The lower 95% bound of the worst adequately-powered fold. Using a lower
    bound rather than a point estimate penalizes models whose apparent
    advantage rests on few comparisons, which is what let selection overfit
    last time.

    Read this alongside `mean_powered`, not instead of it. A controlled
    experiment on the full dataset varied two things that carry no information
    at all, one leaked timing column and a StandardScaler in front of a tree
    model, and moved the worst fold by 0.030 while the mean across powered
    folds moved by 0.0014. The bootstrap interval captures resampling variance
    but not that pipeline sensitivity, so the worst fold is the conservative
    statistic and the mean is the stable one.
    """
    powered = folds_df[folds_df['powered']]
    if powered.empty:
        return float('nan')
    return float(powered['ci_low'].min())


def mean_powered(folds_df):
    """Mean age-conditioned AUROC across adequately powered folds.

    Empirically an order of magnitude more stable than the worst fold under
    pipeline perturbations, and closer to what a single hidden site measures.
    """
    powered = folds_df[folds_df['powered']]
    if powered.empty:
        return float('nan')
    return float(powered['auroc_age'].mean())


def summarize(folds_df, label=''):
    """One-line summary plus the honest caveats."""
    powered = folds_df[folds_df['powered']]
    stat = selection_statistic(folds_df)
    mean_powered = powered['auroc_age'].mean() if not powered.empty else float('nan')
    worst = powered['auroc_age'].min() if not powered.empty else float('nan')
    n_under = int((~folds_df['powered']).sum())

    out = (f'{label:28s} selection={stat:.4f}  worst_powered={worst:.4f}  '
           f'mean_powered={mean_powered:.4f}')
    if n_under:
        out += f'  ({n_under} underpowered fold(s) excluded)'
    return out


def noise_floor(labels, predictions, ages, target_prevalence=0.10, gap=DEFAULT_GAP,
                n_draws=300, seed=0):
    """Estimate the leaderboard's resolution by subsampling to a target prevalence.

    Age-conditioned AUROC is prevalence-invariant in expectation, so this is not
    a model-selection tool: resampling corrects no bias and only inflates
    variance. Its one legitimate use is measuring how large a difference has to
    be before it means anything. Measured on the unofficial data, sd rose from
    0.000 at full prevalence to 0.076 at 5%, which is why differences below
    ~0.05 should never cost a submission slot.
    """
    labels = np.asarray(labels, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    ages = np.asarray(ages, dtype=float)
    rng = np.random.default_rng(seed)

    pos_idx = np.flatnonzero(labels == 1)
    neg_idx = np.flatnonzero(labels == 0)
    n_pos_target = max(1, int(round(target_prevalence * len(neg_idx)
                                    / (1 - target_prevalence))))
    n_pos_target = min(n_pos_target, len(pos_idx))

    draws = []
    for _ in range(n_draws):
        keep = np.concatenate([rng.choice(pos_idx, n_pos_target, replace=False),
                               neg_idx])
        draws.append(cm.auroc_age(labels[keep], predictions[keep], ages[keep], gap=gap))

    draws = np.array([d for d in draws if np.isfinite(d)])
    if draws.size == 0:
        return {}
    return {
        'target_prevalence': target_prevalence,
        'mean': float(draws.mean()),
        'sd': float(draws.std()),
        'p2.5': float(np.quantile(draws, 0.025)),
        'p97.5': float(np.quantile(draws, 0.975)),
        'n_pos_kept': int(n_pos_target),
    }
