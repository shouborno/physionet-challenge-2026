#!/usr/bin/env python
"""Sample weights that make a pointwise model target the matched metric.

The metric counts each (positive, negative) pair whose ages fall within two
years, so a subject's contribution is proportional to how many opposite-class
subjects sit inside their caliper:

    w_i = #{ j : y_j != y_i, |age_j - age_i| <= 2 }

Weighting a plain classifier this way is not a heuristic. Kotlowski,
Dembczynski and Hullermeier (ICML 2011), restated as Theorem 6 in Agarwal
(JMLR 15:1653, 2014), bound the ranking regret of a *balanced* pointwise
logistic model by 2*sqrt(its own regret). Unbalanced, the constant at 7.6%
prevalence is about 10. Applying the balance per stratum and combining by
Jensen gives exactly the weights above: positive and negative mass are equal
inside every age band, and bands are weighted by their matched-pair count,
which is the metric's own implicit weighting.

This corrects a real misallocation that a single scale_pos_weight cannot see.
Prevalence here runs from about 2% at ages 50-59 to 33% at 80-89, so one global
positive weight over-weights the oldest band several-fold while under-weighting
the youngest by as much again.

Unlike age residualization, nothing is estimated and no feature is modified, so
there is no nuisance-parameter noise to pay for. Residualization was measured
to perturb the pairs the metric scores five times *less* than the pairs it
ignores, which is why it could only ever lose.
"""

import numpy as np

DEFAULT_GAP = 2


def matched_pair_weights(y, ages, gap=DEFAULT_GAP, temper=1.0,
                         normalize=True, floor=1e-3):
    """Weight each subject by its count of eligible opposite-class partners.

    `temper` raises the raw counts to a power. Full balancing (1.0) can overfit
    at 84 positives, so 0.5 and 0.75 are worth sweeping alongside it.
    """
    y = np.asarray(y).ravel()
    ages = np.asarray(ages, dtype=float).ravel()

    pos = y == 1
    neg = y == 0
    finite = np.isfinite(ages)

    counts = np.zeros(len(y), dtype=float)
    if pos.any() and neg.any():
        pos_ages = ages[pos & finite]
        neg_ages = ages[neg & finite]
        # For each subject, how many opposite-class subjects lie in the caliper.
        within_pos = np.abs(ages[:, None] - neg_ages[None, :]) <= gap
        within_neg = np.abs(ages[:, None] - pos_ages[None, :]) <= gap
        counts[pos] = within_pos[pos].sum(axis=1)
        counts[neg] = within_neg[neg].sum(axis=1)

    # A subject with no eligible partner contributes nothing to the metric, but
    # zeroing it outright discards a usable training example, so floor it.
    counts = np.maximum(counts, floor)
    w = counts ** float(temper)

    if normalize:
        # Equalize total mass between classes so neither dominates the loss,
        # then scale to mean 1 so learning rates stay comparable.
        for mask in (pos, neg):
            if mask.any() and w[mask].sum() > 0:
                w[mask] = w[mask] / w[mask].sum() * mask.sum()
        w = w / w.mean()

    return w


def effective_sample_size(y, ages, gap=DEFAULT_GAP):
    """Kish effective sample size of the positives under matched weighting.

    The metric's precision is bounded by subjects, not by the ~10,000 pairs
    they generate, and an uneven age distribution pushes this materially below
    the raw positive count.
    """
    y = np.asarray(y).ravel()
    ages = np.asarray(ages, dtype=float).ravel()
    pos = y == 1
    neg = y == 0
    if not pos.any() or not neg.any():
        return 0.0

    neg_ages = ages[neg & np.isfinite(ages)]
    m = np.array([np.sum(np.abs(neg_ages - a) <= gap) for a in ages[pos]],
                 dtype=float)
    if m.sum() == 0:
        return 0.0
    return float(m.sum() ** 2 / np.sum(m ** 2))


def age_band_report(y, ages, gap=DEFAULT_GAP, band=10):
    """Prevalence and matched-pair mass per age band, for diagnosis."""
    y = np.asarray(y).ravel()
    ages = np.asarray(ages, dtype=float).ravel()
    w = matched_pair_weights(y, ages, gap=gap)

    rows = []
    lo = int(np.floor(np.nanmin(ages) / band) * band)
    hi = int(np.ceil(np.nanmax(ages) / band) * band)
    for start in range(lo, hi, band):
        mask = (ages >= start) & (ages < start + band)
        if not mask.any():
            continue
        rows.append({
            'band': f'{start}-{start + band - 1}',
            'n': int(mask.sum()),
            'positives': int(y[mask].sum()),
            'prevalence': float(y[mask].mean()),
            'weight_share': float(w[mask].sum() / w.sum()),
            'naive_share': float(mask.sum() / len(y)),
        })
    return rows
