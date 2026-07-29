#!/usr/bin/env python
"""Remove age from features and from scores.

Age alone reaches 0.772 plain AUROC on the official training set but only 0.538
under the age-conditioned metric, so most of what a naive model learns from age
earns nothing. Worse, the training age-prevalence gradient (2% at 50-59 rising
to 33% at 80-89) will not match the target site's, so leaning on it is actively
harmful under distribution shift.

Two independent tools:

`AgeResidualizer` strips the smooth age trend from each feature, fitted on
training folds only. What remains is how unusual a patient looks *for their
age*, which is the only thing the metric can reward.

`AgeConditionalStandardizer` centers and scales the final score by age. It is
locally order-preserving within a couple of years, but the metric compares
patients up to 4 years apart, so it genuinely moves the number. Fitted on
training data only, so it raises no transductive question.
"""

import numpy as np


def _spline_basis(age, knots, degree=3):
    """Truncated power basis for a natural-ish cubic spline.

    Small and dependency-free; scipy/patsy would work but this keeps the model
    file self-contained for the submission container.
    """
    age = np.asarray(age, dtype=float).ravel()
    cols = [np.ones_like(age)]
    for d in range(1, degree + 1):
        cols.append(age ** d)
    for k in knots:
        cols.append(np.clip(age - k, 0, None) ** degree)
    return np.column_stack(cols)


class AgeResidualizer:
    """Subtract a per-feature smooth function of age.

    Fitted on the negatives by default: the age trend among people who never
    develop impairment is the healthy-aging baseline, so residuals measure
    departure from it. Fitting on everyone would partly absorb the signal,
    since prevalence itself rises with age.
    """

    def __init__(self, n_knots=4, degree=3, fit_on_negatives=True, ridge=1e-6):
        self.n_knots = n_knots
        self.degree = degree
        self.fit_on_negatives = fit_on_negatives
        self.ridge = ridge
        self.knots_ = None
        self.coefs_ = None
        self.fallback_ = None

    def fit(self, X, ages, y=None):
        X = np.asarray(X, dtype=float)
        ages = np.asarray(ages, dtype=float).ravel()

        mask = np.isfinite(ages)
        if self.fit_on_negatives and y is not None:
            neg = np.asarray(y).ravel() == 0
            if neg.sum() >= 50:
                mask = mask & neg

        quantiles = np.linspace(0, 100, self.n_knots + 2)[1:-1]
        self.knots_ = np.percentile(ages[np.isfinite(ages)], quantiles)

        basis = _spline_basis(ages[mask], self.knots_, self.degree)
        gram = basis.T @ basis + self.ridge * np.eye(basis.shape[1])

        self.coefs_ = np.zeros((basis.shape[1], X.shape[1]))
        self.fallback_ = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            col = X[mask, j]
            ok = np.isfinite(col)
            self.fallback_[j] = np.nanmedian(X[:, j]) if np.isfinite(X[:, j]).any() else 0.0
            if ok.sum() < basis.shape[1] + 5:
                continue
            b = basis[ok]
            g = b.T @ b + self.ridge * np.eye(b.shape[1])
            try:
                self.coefs_[:, j] = np.linalg.solve(g, b.T @ col[ok])
            except np.linalg.LinAlgError:
                self.coefs_[:, j] = np.linalg.lstsq(b, col[ok], rcond=None)[0]
        return self

    def transform(self, X, ages):
        X = np.asarray(X, dtype=float)
        ages = np.asarray(ages, dtype=float).ravel()
        safe_ages = np.where(np.isfinite(ages), ages, np.nanmedian(ages))
        predicted = _spline_basis(safe_ages, self.knots_, self.degree) @ self.coefs_
        return X - predicted

    def fit_transform(self, X, ages, y=None):
        return self.fit(X, ages, y).transform(X, ages)


class AgeConditionalStandardizer:
    """Standardize a score against its age-conditional mean and spread."""

    def __init__(self, n_knots=4, degree=3, ridge=1e-6, min_scale=1e-6):
        self.n_knots = n_knots
        self.degree = degree
        self.ridge = ridge
        self.min_scale = min_scale
        self.knots_ = None
        self.mu_coefs_ = None
        self.sd_coefs_ = None

    def _solve(self, basis, target):
        gram = basis.T @ basis + self.ridge * np.eye(basis.shape[1])
        try:
            return np.linalg.solve(gram, basis.T @ target)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(basis, target, rcond=None)[0]

    def fit(self, scores, ages):
        scores = np.asarray(scores, dtype=float).ravel()
        ages = np.asarray(ages, dtype=float).ravel()
        ok = np.isfinite(scores) & np.isfinite(ages)

        quantiles = np.linspace(0, 100, self.n_knots + 2)[1:-1]
        self.knots_ = np.percentile(ages[ok], quantiles)

        basis = _spline_basis(ages[ok], self.knots_, self.degree)
        self.mu_coefs_ = self._solve(basis, scores[ok])

        residual = np.abs(scores[ok] - basis @ self.mu_coefs_)
        self.sd_coefs_ = self._solve(basis, residual)
        return self

    def transform(self, scores, ages):
        scores = np.asarray(scores, dtype=float).ravel()
        ages = np.asarray(ages, dtype=float).ravel()
        safe = np.where(np.isfinite(ages), ages, np.nanmedian(ages))
        basis = _spline_basis(safe, self.knots_, self.degree)
        mu = basis @ self.mu_coefs_
        sd = np.maximum(np.abs(basis @ self.sd_coefs_), self.min_scale)
        return (scores - mu) / sd

    def fit_transform(self, scores, ages):
        return self.fit(scores, ages).transform(scores, ages)


def age_prevalence_weights(ages, train_labels, train_ages,
                           target_curve=None, gap=2):
    """Weight each patient so the pair distribution matches the target site.

    The metric weights each age band by its local pair count. That is roughly
    flat in the training data but concentrates on older bands when prevalence
    rises with age, as it will at the target. Reweighting corrects that bias
    using all the data, unlike prevalence resampling, which corrects no bias and
    triples the variance.

    `target_curve` maps age to expected target prevalence. Without one this
    returns unit weights, so the caller can always apply it safely.
    """
    ages = np.asarray(ages, dtype=float).ravel()
    if target_curve is None:
        return np.ones_like(ages)

    train_labels = np.asarray(train_labels, dtype=float).ravel()
    train_ages = np.asarray(train_ages, dtype=float).ravel()

    weights = np.ones_like(ages)
    for i, age in enumerate(ages):
        if not np.isfinite(age):
            continue
        window = np.abs(train_ages - age) <= gap
        if not window.any():
            continue
        p_train = max(train_labels[window].mean(), 1e-6)
        p_target = max(float(target_curve(age)), 1e-6)
        weights[i] = p_target / p_train

    return weights / weights.mean()
