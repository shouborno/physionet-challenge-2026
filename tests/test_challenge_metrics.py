#!/usr/bin/env python
"""Assert our vectorized metrics equal the organizers' reference implementations.

Every model decision rides on these numbers, so drift here is silent and fatal.
Run with: python -m pytest tests/test_challenge_metrics.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import challenge_metrics as cm  # noqa: E402

pytestmark = pytest.mark.skipif(
    not cm.HAVE_REFERENCE,
    reason='evaluate_model.py reference implementation not importable',
)

GAPS = [0, 1, 2, 5]


def random_case(rng, n=60, prevalence=0.3, age_lo=45, age_hi=90, tie_rate=0.0):
    labels = (rng.random(n) < prevalence).astype(float)
    predictions = rng.random(n)
    if tie_rate:
        # Round to force exact ties, which the metric counts as a half.
        predictions = np.round(predictions, 1)
    ages = rng.integers(age_lo, age_hi, size=n).astype(float)
    return labels, predictions, ages


@pytest.mark.parametrize('seed', range(12))
@pytest.mark.parametrize('gap', GAPS)
def test_auroc_age_matches_reference(seed, gap):
    rng = np.random.default_rng(seed)
    labels, predictions, ages = random_case(rng)
    if labels.sum() == 0 or labels.sum() == len(labels):
        pytest.skip('degenerate single-class draw')

    expected = cm.reference_auroc_age(labels, predictions, ages, gap=gap)
    actual = cm.auroc_age(labels, predictions, ages, gap=gap)
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('seed', range(8))
def test_auroc_age_with_ties(seed):
    """Ties score 0.5 and are easy to get wrong; exercise them explicitly."""
    rng = np.random.default_rng(100 + seed)
    labels, predictions, ages = random_case(rng, tie_rate=1.0)
    if labels.sum() == 0 or labels.sum() == len(labels):
        pytest.skip('degenerate single-class draw')

    expected = cm.reference_auroc_age(labels, predictions, ages, gap=2)
    actual = cm.auroc_age(labels, predictions, ages, gap=2)
    assert actual == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize('seed', range(8))
def test_prevalence_matches_reference(seed):
    rng = np.random.default_rng(200 + seed)
    labels, _, ages = random_case(rng, n=80)
    eval_ages = rng.integers(45, 90, size=40).astype(float)

    expected = cm.reference_prevalence(eval_ages, labels, ages, gap=2)
    actual = cm.prevalence_by_age(eval_ages, labels, ages, gap=2)

    assert set(actual) == set(expected)
    for age in expected:
        assert actual[age] == pytest.approx(expected[age], abs=1e-12)


@pytest.mark.parametrize('seed', range(8))
def test_reward_matches_reference(seed):
    rng = np.random.default_rng(300 + seed)
    labels, _, ages = random_case(rng, n=70)
    binary = (rng.random(70) < 0.4).astype(float)

    prevalence = cm.reference_prevalence(ages, labels, ages, gap=2)
    expected = cm.reference_reward(labels, binary, ages, prevalence)
    actual = cm.reward(labels, binary, ages, prevalence)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_auroc_age_handles_no_matched_pairs():
    """Widely separated ages leave no comparable pair; reference divides by zero."""
    labels = np.array([1.0, 0.0])
    predictions = np.array([0.9, 0.1])
    ages = np.array([50.0, 80.0])
    score, pairs = cm.auroc_age(labels, predictions, ages, gap=2, return_pairs=True)
    assert pairs == 0
    assert np.isnan(score)


def test_auroc_age_handles_single_class():
    labels = np.ones(5)
    assert np.isnan(cm.auroc_age(labels, np.random.rand(5), np.full(5, 70.0)))


def test_nan_ages_are_excluded():
    """A NaN age can never be within gap of anything, matching the reference."""
    labels = np.array([1.0, 0.0, 1.0, 0.0])
    predictions = np.array([0.9, 0.1, 0.8, 0.2])
    ages = np.array([70.0, 71.0, np.nan, np.nan])
    score, pairs = cm.auroc_age(labels, predictions, ages, gap=2, return_pairs=True)
    assert pairs == 1
    assert score == pytest.approx(1.0)


def test_optimal_binary_threshold_is_local_prevalence():
    """The expected-reward-optimal rule is q > p_a; verify it beats a 0.5 cut.

    At 10% prevalence a 0.5 threshold is far too conservative and forfeits most
    of the reward metric.
    """
    rng = np.random.default_rng(7)
    n = 4000
    ages = rng.integers(60, 80, size=n).astype(float)
    labels = (rng.random(n) < 0.10).astype(float)
    # A usefully-but-imperfectly informative posterior.
    q = np.clip(0.10 + 0.25 * labels + 0.05 * rng.standard_normal(n), 0.001, 0.999)

    prevalence = cm.prevalence_by_age(ages, labels, ages, gap=2)

    optimal = cm.optimal_binary(q, ages, prevalence)
    naive = (q >= 0.5).astype(int)

    assert cm.reward(labels, optimal, ages, prevalence) > \
        cm.reward(labels, naive, ages, prevalence)


def test_prior_shift_moves_posterior_toward_target():
    """Calibrating from a balanced training set to 10% must lower the posterior."""
    q = np.array([0.5, 0.8])
    ages = np.array([70.0, 70.0])
    shifted = cm.optimal_binary(q, ages, {70.0: 0.99},
                                prior_train=0.5, prior_target=0.10)
    # With p_a pinned at 0.99 nothing should fire once q is deflated to ~0.1/0.3.
    assert shifted.tolist() == [0, 0]


@pytest.mark.parametrize('seed', range(4))
def test_bootstrap_ci_brackets_point_estimate(seed):
    rng = np.random.default_rng(400 + seed)
    labels, predictions, ages = random_case(rng, n=200)
    point = cm.auroc_age(labels, predictions, ages, gap=2)
    lo, hi, sd = cm.bootstrap_ci(labels, predictions, ages, gap=2,
                                 n_boot=300, seed=seed)
    assert lo <= point <= hi
    assert sd > 0
