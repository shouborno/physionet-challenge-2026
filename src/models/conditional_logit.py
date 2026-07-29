#!/usr/bin/env python
"""Conditional logistic regression over age-matched strata.

The challenge metric is
    P(s_i > s_j | y_i=1, y_j=0, |age_i - age_j| <= 2),
which is precisely a matched case-control comparison. Epidemiology has an exact
estimator for that design: conditional logistic regression, which conditions on
the number of cases in each matched set so the stratum intercept drops out of
the likelihood entirely. Nothing that is constant within a stratum, including
age itself and anything monotone in it, can influence the fit.

That is a stronger and cleaner guarantee than either thing already tried here.
Age residualization subtracted an estimated age trend from every feature and
consistently lost 0.02-0.05, which suggests the estimated trend was carrying
real within-band signal away with it. Conditional likelihood removes the
stratum effect by construction, estimating nothing and so discarding nothing.

The earlier pairwise ranker was a related idea that failed its own test, but it
differed in two ways that plausibly mattered at 84 positives: it was a
two-layer MLP rather than a linear model, and it sampled pairs stochastically
rather than using the exact within-stratum likelihood. This is the low-variance
version of the same intuition.

One honest wrinkle: the metric's window slides continuously, while strata are
disjoint bins. Two patients three years apart can share a bin, and two patients
one year apart can fall either side of a boundary. Averaging several models
fitted at shifted bin offsets approximates the sliding window.
"""

import numpy as np

try:
    import torch
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


def make_strata(ages, width=4.0, offset=0.0):
    """Disjoint age bins of the given width, shifted by `offset`.

    A width of 4 is the natural choice: the metric compares patients up to 2
    years apart in either direction.
    """
    ages = np.asarray(ages, dtype=float)
    safe = np.where(np.isfinite(ages), ages, np.nanmedian(ages))
    return np.floor((safe - offset) / width).astype(int)


class ConditionalLogit:
    """Exact conditional likelihood over strata, fitted with L2 regularization.

    For a stratum with cases C and controls, the conditional log-likelihood
    contribution used here is the standard 1:m form summed over cases:

        sum_{i in C} [ s_i - logsumexp over {i} union controls of s ]

    which is the Cox partial likelihood with all events tied at one time, and
    equals conditional logistic regression for matched sets. Strata containing
    only cases or only controls contribute nothing and are dropped, exactly as
    in the classical estimator.
    """

    def __init__(self, alpha=1.0, width=4.0, n_offsets=4, max_iter=500,
                 lr=0.05, seed=42, verbose=False):
        if not HAVE_TORCH:
            raise ImportError('ConditionalLogit requires torch')
        self.alpha = alpha
        self.width = width
        self.n_offsets = n_offsets
        self.max_iter = max_iter
        self.lr = lr
        self.seed = seed
        self.verbose = verbose
        self.models_ = []
        self.mean_ = None
        self.std_ = None

    def _prepare(self, X, fit=False):
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        if fit:
            self.mean_ = np.nanmedian(X, axis=0)
            self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        filled = np.where(np.isnan(X), self.mean_, X)
        if fit:
            sd = filled.std(axis=0)
            self.std_ = np.where(sd > 1e-8, sd, 1.0)
        return ((filled - self.mean_) / self.std_).astype(np.float32)

    def _fit_one(self, Xs, y, strata):
        """Fit a single linear scorer under the conditional likelihood."""
        torch.manual_seed(self.seed)
        n_features = Xs.shape[1]
        w = torch.zeros(n_features, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.LBFGS([w], lr=self.lr, max_iter=self.max_iter,
                                line_search_fn='strong_wolfe')
        Xt = torch.from_numpy(Xs)

        groups = []
        for s in np.unique(strata):
            mask = strata == s
            yy = y[mask]
            # A stratum with no case or no control is uninformative about the
            # within-stratum contrast, which is the whole point of the design.
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            idx = torch.from_numpy(np.flatnonzero(mask))
            groups.append((idx, torch.from_numpy((yy == 1).astype(np.bool_))))

        if not groups:
            return None

        def closure():
            opt.zero_grad()
            scores = Xt @ w
            total = 0.0
            for idx, is_case in groups:
                s = scores[idx]
                # Each case is compared against the whole matched set.
                total = total - (s[is_case] - torch.logsumexp(s, dim=0)).sum()
            loss = total / len(groups) + self.alpha * (w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return w.detach().numpy().copy()

    def fit(self, X, y, ages):
        Xs = self._prepare(X, fit=True)
        y = np.asarray(y).ravel()
        ages = np.asarray(ages, dtype=float).ravel()

        self.models_ = []
        for k in range(self.n_offsets):
            offset = self.width * k / self.n_offsets
            strata = make_strata(ages, self.width, offset)
            w = self._fit_one(Xs, y, strata)
            if w is not None:
                self.models_.append(w)
            if self.verbose:
                print(f'  offset {offset:.1f}: '
                      f'{"fitted" if w is not None else "no usable strata"}')

        if not self.models_:
            raise ValueError('no stratum contained both a case and a control')
        return self

    def decision_function(self, X):
        Xs = self._prepare(X)
        # Averaging the offset models approximates the metric's sliding window.
        return np.mean([Xs @ w for w in self.models_], axis=0)

    def predict_proba(self, X):
        s = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-s))
        return np.column_stack([1 - p, p])
