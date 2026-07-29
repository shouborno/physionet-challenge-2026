#!/usr/bin/env python
"""Challenge 2026 scoring, vectorized, with uncertainty.

The primary metric is age-conditioned AUROC at gap=2: among (positive, negative)
pairs whose ages differ by at most 2 years, the fraction where the positive is
scored higher, counting ties as a half. The reference implementation in
`evaluate_model.py` is an O(n_pos * n_neg) Python double loop, which is roughly
10^7 iterations at 6,600 records and unusable inside a cross-validation or
bootstrap loop. The vectorized versions here are checked against the reference
by `tests/test_challenge_metrics.py`.

Every estimate is reported with the number of age-matched pairs behind it.
Only about 16% of pairs are age-matched at gap=2, so the effective sample size
is far smaller than the record count suggests, and a fold can look precise while
resting on a few dozen comparisons.
"""

import os
import sys

import numpy as np

DEFAULT_GAP = 2

# Import the organizers' reference implementations so the local harness and the
# leaderboard cannot drift apart. evaluate_model.py lives at the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from evaluate_model import (  # noqa: F401
        compute_auroc_age as reference_auroc_age,
        compute_auroc_weighted as reference_auroc_weighted,
        compute_prevalence as reference_prevalence,
        compute_reward as reference_reward,
    )
    HAVE_REFERENCE = True
except ImportError:  # pragma: no cover - only when run outside the repo
    HAVE_REFERENCE = False


def _as_arrays(labels, predictions, ages):
    labels = np.asarray(labels, dtype=float).ravel()
    predictions = np.asarray(predictions, dtype=float).ravel()
    ages = np.asarray(ages, dtype=float).ravel()
    if not (labels.size == predictions.size == ages.size):
        raise ValueError('labels, predictions and ages must have equal length')
    return labels, predictions, ages


def auroc_age(labels, predictions, ages, gap=DEFAULT_GAP, return_pairs=False):
    """Age-conditioned AUROC. Matches evaluate_model.compute_auroc_age.

    Pairs a positive with a negative only when their ages are within `gap`
    years. Ties in prediction score a half. Returns NaN when no age-matched
    pair exists, where the reference would divide by zero.
    """
    labels, predictions, ages = _as_arrays(labels, predictions, ages)

    pos = labels == 1
    neg = labels == 0
    if not pos.any() or not neg.any():
        return (float('nan'), 0) if return_pairs else float('nan')

    p_scores = predictions[pos][:, None]
    n_scores = predictions[neg][None, :]
    p_ages = ages[pos][:, None]
    n_ages = ages[neg][None, :]

    # NaN ages never satisfy the comparison, which matches the reference: it
    # evaluates abs(nan) <= gap as False and skips the pair.
    matched = np.abs(p_ages - n_ages) <= gap
    denom = int(matched.sum())
    if denom == 0:
        return (float('nan'), 0) if return_pairs else float('nan')

    wins = (p_scores > n_scores) & matched
    ties = (p_scores == n_scores) & matched
    numer = float(wins.sum()) + 0.5 * float(ties.sum())

    score = numer / denom
    return (score, denom) if return_pairs else score


def n_matched_pairs(labels, ages, gap=DEFAULT_GAP):
    """Number of age-matched (positive, negative) pairs: the effective sample size."""
    labels = np.asarray(labels, dtype=float).ravel()
    ages = np.asarray(ages, dtype=float).ravel()
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return 0
    return int((np.abs(ages[pos][:, None] - ages[neg][None, :]) <= gap).sum())


def auroc_age_weighted_pairs(labels, predictions, ages, age_weight_fn,
                             gap=DEFAULT_GAP):
    """Age-conditioned AUROC with each pair weighted by `age_weight_fn(age)`.

    The unweighted metric weights each age band by its local pair count, which
    is flat in balanced training data but concentrates on older bands when
    prevalence rises with age, as it will at the target site. Weighting pairs by
    the target-to-training age-prevalence ratio corrects that bias without the
    variance cost of resampling.
    """
    labels, predictions, ages = _as_arrays(labels, predictions, ages)
    pos, neg = labels == 1, labels == 0
    if not pos.any() or not neg.any():
        return float('nan')

    p_ages, n_ages = ages[pos][:, None], ages[neg][None, :]
    matched = np.abs(p_ages - n_ages) <= gap
    if not matched.any():
        return float('nan')

    # Weight a pair by the mean of its two endpoint weights.
    w = 0.5 * (np.asarray(age_weight_fn(ages[pos]), dtype=float)[:, None]
               + np.asarray(age_weight_fn(ages[neg]), dtype=float)[None, :])
    w = np.where(matched, w, 0.0)

    p_scores, n_scores = predictions[pos][:, None], predictions[neg][None, :]
    numer = float((w * (p_scores > n_scores)).sum()) \
        + 0.5 * float((w * (p_scores == n_scores)).sum())
    denom = float(w.sum())
    return numer / denom if denom > 0 else float('nan')


def prevalence_by_age(eval_ages, prevalence_labels, prevalence_ages, gap=DEFAULT_GAP):
    """Positive-class prevalence near each evaluated age.

    Mirrors evaluate_model.compute_prevalence, including its floor of 0.5 on the
    numerator. The organizers state they compute this from the *training* set,
    so `prevalence_labels`/`prevalence_ages` should come from training
    demographics even when scoring a held-out site.
    """
    eval_ages = np.asarray(eval_ages, dtype=float).ravel()
    prevalence_labels = np.asarray(prevalence_labels, dtype=float).ravel()
    prevalence_ages = np.asarray(prevalence_ages, dtype=float).ravel()

    out = {}
    for age in np.unique(eval_ages[np.isfinite(eval_ages)]):
        window = np.abs(prevalence_ages - age) <= gap
        n = int(window.sum())
        if n == 0:
            continue
        out[age] = max(float(prevalence_labels[window].sum()), 0.5) / n
    return out


def reward(labels, binary_predictions, ages, age_to_prevalence):
    """Prevalence-weighted reward. Matches evaluate_model.compute_reward.

    Note this consumes the *binary* output, not the probability.
    """
    labels, binary_predictions, ages = _as_arrays(labels, binary_predictions, ages)
    n = labels.size

    finite = np.isfinite(ages)
    if not finite.any():
        return float('nan')

    p = np.array([age_to_prevalence.get(a, np.nan) for a in ages], dtype=float)
    p = np.clip(p, 0.5 / n, 1 - 0.5 / n)

    scores = np.zeros(n, dtype=float)
    tp = (labels == 1) & (binary_predictions == 1)
    tn = (labels == 0) & (binary_predictions == 0)
    wrong = ((labels == 1) & (binary_predictions == 0)) | \
            ((labels == 0) & (binary_predictions == 1))

    scores[tp] = 1.0 / p[tp] - 1.0
    scores[tn] = 1.0 / (1.0 - p[tn]) - 1.0
    scores[wrong] = -1.0
    scores[~finite] = 0.0

    return float(scores.sum() / int(finite.sum()))


def optimal_binary(probabilities, ages, age_to_prevalence,
                   prior_train=None, prior_target=None):
    """Expected-reward-optimal binary decisions.

    With reward 1/p-1 for a true positive, 1/(1-p)-1 for a true negative and -1
    otherwise, comparing the two expected values gives: predict positive exactly
    when the calibrated posterior q exceeds the local age prevalence p_a. If
    `prior_train` and `prior_target` are given, q is prior-shifted first, which
    matters because the training set is far more balanced than validation/test.
    """
    probabilities = np.asarray(probabilities, dtype=float).ravel()
    ages = np.asarray(ages, dtype=float).ravel()

    q = probabilities
    if prior_train is not None and prior_target is not None \
            and 0 < prior_train < 1 and 0 < prior_target < 1:
        pos = q * (prior_target / prior_train)
        neg = (1.0 - q) * ((1.0 - prior_target) / (1.0 - prior_train))
        with np.errstate(invalid='ignore', divide='ignore'):
            q = np.where(pos + neg > 0, pos / (pos + neg), q)

    default_p = prior_target if prior_target is not None else 0.10
    p = np.array([age_to_prevalence.get(a, default_p) for a in ages], dtype=float)
    return (q > p).astype(int)


def bootstrap_ci(labels, predictions, ages, gap=DEFAULT_GAP, n_boot=2000,
                 alpha=0.05, seed=0):
    """Patient-level bootstrap CI for age-conditioned AUROC.

    Resamples patients, not pairs, because pairs are not independent.
    """
    labels, predictions, ages = _as_arrays(labels, predictions, ages)
    rng = np.random.default_rng(seed)
    n = labels.size

    draws = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = auroc_age(labels[idx], predictions[idx], ages[idx], gap=gap)

    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return float('nan'), float('nan'), float('nan')
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), float(draws.std())


def score_fold(labels, probabilities, ages, gap=DEFAULT_GAP, n_boot=2000,
               seed=0, prevalence=None, binary=None):
    """Full report for one held-out fold.

    Returns the point estimate alongside the pair count and bootstrap interval,
    so a fold resting on 60 comparisons cannot be mistaken for a solid estimate.
    """
    point, pairs = auroc_age(labels, probabilities, ages, gap=gap, return_pairs=True)
    lo, hi, sd = bootstrap_ci(labels, probabilities, ages, gap=gap,
                              n_boot=n_boot, seed=seed)

    out = {
        'auroc_age': point,
        'n_pairs': pairs,
        'ci_low': lo,
        'ci_high': hi,
        'boot_sd': sd,
        'n': int(np.size(labels)),
        'n_pos': int(np.sum(np.asarray(labels) == 1)),
    }

    if prevalence is not None and binary is not None:
        out['reward'] = reward(labels, binary, ages, prevalence)

    return out
