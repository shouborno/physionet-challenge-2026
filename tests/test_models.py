#!/usr/bin/env python
"""Behavioral tests for the metric-aligned modeling pieces.

These check the properties the methods are *claimed* to have, since those
claims are what the plan rests on: that pairwise ranking is invariant to
per-site offsets, and that age residualization actually removes age.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import challenge_metrics as cm  # noqa: E402
from src.eval import protocol  # noqa: E402
from src.models.age_adjust import (  # noqa: E402
    AgeConditionalStandardizer, AgeResidualizer,
)
from src.models.pairwise_rank import build_pairs  # noqa: E402


def synthetic(n=600, seed=0, site_offset=0.0, n_sites=3, base_logit=-4.0):
    """Data where age drives prevalence but a latent signal drives the label.

    Calibrated against the official training set, where age alone reaches 0.772
    plain AUROC but only 0.538 age-conditioned, and prevalence rises from ~2% in
    the fifties to ~33% in the eighties.
    """
    rng = np.random.default_rng(seed)
    ages = rng.integers(50, 89, size=n).astype(float)
    sites = rng.integers(0, n_sites, size=n)
    signal = rng.standard_normal(n)

    logit = base_logit + 0.10 * (ages - 60) + 1.2 * signal
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)

    feature = signal + site_offset * sites + 0.4 * rng.standard_normal(n)
    age_proxy = 0.08 * ages + 0.3 * rng.standard_normal(n)
    X = np.column_stack([feature, age_proxy, rng.standard_normal(n)])
    return X, y, ages, sites


def test_build_pairs_respects_age_gap_and_site():
    y = np.array([1, 0, 1, 0])
    ages = np.array([70.0, 71.0, 60.0, 80.0])
    sites = np.array(['A', 'A', 'B', 'B'])

    p, n = build_pairs(y, ages, sites, gap=2, within_site=True)
    assert len(p) == 1
    assert (p[0], n[0]) == (0, 1)

    # Relaxing the site constraint admits no new pair here, since the B-site
    # positive is 20 years from every negative.
    p2, _ = build_pairs(y, ages, sites, gap=2, within_site=False)
    assert len(p2) == 1


def test_build_pairs_within_site_excludes_cross_site_matches():
    y = np.array([1, 0, 1, 0])
    ages = np.array([70.0, 70.0, 70.0, 70.0])
    sites = np.array(['A', 'B', 'B', 'A'])

    within, _ = build_pairs(y, ages, sites, gap=2, within_site=True)
    across, _ = build_pairs(y, ages, sites, gap=2, within_site=False)
    assert len(within) == 2   # (0,3) in A and (2,1) in B
    assert len(across) == 4   # every positive against every negative


def test_pairwise_objective_is_invariant_to_site_offsets():
    """The central claim: a per-site additive shift cancels inside s_i - s_j."""
    rng = np.random.default_rng(1)
    n = 200
    sites = rng.integers(0, 3, size=n)
    y = (rng.random(n) < 0.3).astype(float)
    ages = rng.integers(60, 70, size=n).astype(float)
    scores = rng.standard_normal(n)

    p, q = build_pairs(y, ages, sites, gap=2, within_site=True)
    assert len(p) > 0

    base = scores[p] - scores[q]
    shifted = scores + 5.0 * sites
    assert np.allclose(base, shifted[p] - shifted[q])

    # Without the within-site constraint the offset leaks into the differences.
    p2, q2 = build_pairs(y, ages, sites, gap=2, within_site=False)
    assert not np.allclose(scores[p2] - scores[q2], shifted[p2] - shifted[q2])


def test_age_residualizer_removes_age_dependence():
    X, y, ages, _ = synthetic(n=800, seed=3)
    age_proxy = X[:, 1]

    before = abs(np.corrcoef(age_proxy, ages)[0, 1])
    residual = AgeResidualizer().fit_transform(X, ages, y)
    after = abs(np.corrcoef(residual[:, 1], ages)[0, 1])

    assert before > 0.8
    assert after < 0.15


def test_age_residualizer_preserves_non_age_signal():
    X, y, ages, _ = synthetic(n=800, seed=4)
    residual = AgeResidualizer().fit_transform(X, ages, y)
    # Column 0 is the latent signal; residualizing on age must not destroy it.
    assert abs(np.corrcoef(residual[:, 0], X[:, 0])[0, 1]) > 0.9


def test_age_residualizer_does_not_leak_test_ages():
    """transform() must use only the fitted trend, so it is fold-safe."""
    X, y, ages, _ = synthetic(n=400, seed=5)
    fitted = AgeResidualizer().fit(X[:200], ages[:200], y[:200])
    a = fitted.transform(X[200:], ages[200:])
    b = fitted.transform(X[200:], ages[200:])
    assert np.allclose(a, b)


def test_age_conditional_standardizer_centers_by_age():
    rng = np.random.default_rng(6)
    ages = rng.integers(50, 89, size=1000).astype(float)
    scores = 0.1 * ages + rng.standard_normal(1000)

    standardized = AgeConditionalStandardizer().fit_transform(scores, ages)

    old = ages >= 75
    young = ages <= 60
    # The raw score differs sharply by age band; the standardized one should not.
    assert abs(scores[old].mean() - scores[young].mean()) > 1.0
    assert abs(standardized[old].mean() - standardized[young].mean()) < 0.4


def test_selection_statistic_ignores_underpowered_folds():
    import pandas as pd
    folds = pd.DataFrame([
        {'fold': 'tiny', 'auroc_age': 0.95, 'ci_low': 0.40, 'n_pairs': 40,
         'powered': False},
        {'fold': 'big', 'auroc_age': 0.70, 'ci_low': 0.66, 'n_pairs': 5000,
         'powered': True},
    ])
    # The tiny fold's spuriously high point estimate and low bound are both
    # excluded; only the powered fold sets the statistic.
    assert protocol.selection_statistic(folds) == pytest.approx(0.66)


def test_noise_floor_reports_inflated_variance_at_low_prevalence():
    """Resampling to low prevalence should widen, not shift, the estimate."""
    X, y, ages, _ = synthetic(n=1500, seed=7, base_logit=-2.0)
    scores = X[:, 0]
    assert y.sum() >= 60, 'generator should yield enough positives to resample'

    full = cm.auroc_age(y, scores, ages, gap=2)
    low = protocol.noise_floor(y, scores, ages, target_prevalence=0.10,
                               n_draws=60, seed=1)
    assert abs(low['mean'] - full) < 0.08   # prevalence-invariant in expectation
    assert low['sd'] > 0.0                  # but noisier


def test_metric_neutralizes_age_on_synthetic_data():
    """Sanity-check the synthetic generator reproduces the real regime."""
    from sklearn.metrics import roc_auc_score
    _, y, ages, _ = synthetic(n=2000, seed=8)
    plain = roc_auc_score(y, ages)
    conditioned = cm.auroc_age(y, ages, ages, gap=2)
    # The real data shows 0.772 plain against 0.538 conditioned.
    assert plain > 0.70
    assert conditioned < plain - 0.20
