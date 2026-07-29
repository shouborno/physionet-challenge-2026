#!/usr/bin/env python
"""Within-site, age-matched pairwise ranking.

The primary metric is
    P(s_i > s_j | y_i=1, y_j=0, |a_i - a_j| <= 2),
so the natural surrogate is a logistic loss over exactly that pair set:
    L = mean softplus(-(s_i - s_j))

Two constraints on which pairs enter the loss, each doing real work.

*Age-matched.* On the official training set age alone reaches 0.772 plain
AUROC but only 0.538 age-conditioned: the metric neutralizes nearly all of it.
A pointwise classifier spends most of its capacity on that neutralized axis. A
loss defined only over age-matched pairs cannot.

*Within-site.* If the scorer picks up a per-site additive offset, which is what
site batch effects produce and what ComBat tries to remove after the fact, then
s_i - s_j cancels it exactly. The loss is structurally unable to be rewarded
for fitting it, and evaluation happens on a single hidden site where a global
offset is irrelevant anyway. So one constraint delivers both the right
objective and domain generalization, without the instability of a gradient
reversal layer.

The objective is also prevalence-invariant by construction: it samples pairs,
not instances, so the 7.6% positive rate needs no reweighting.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

DEFAULT_GAP = 2


def build_pairs(y, ages, sites=None, gap=DEFAULT_GAP, within_site=True,
                max_pairs=None, rng=None):
    """Index every (positive, negative) pair eligible for the loss.

    Returns (pos_idx, neg_idx) arrays. Pairs are enumerated per site to keep the
    cross product small; the full cross product over 6,600 records would be
    ~10^7 entries before filtering.
    """
    y = np.asarray(y).ravel()
    ages = np.asarray(ages, dtype=float).ravel()
    if sites is None or not within_site:
        sites = np.zeros(len(y), dtype=int)
    sites = np.asarray(sites).ravel()

    pos_all, neg_all = [], []
    for site in np.unique(sites):
        mask = np.flatnonzero(sites == site)
        p = mask[y[mask] == 1]
        n = mask[y[mask] == 0]
        if len(p) == 0 or len(n) == 0:
            continue
        close = np.abs(ages[p][:, None] - ages[n][None, :]) <= gap
        pi, ni = np.nonzero(close)
        pos_all.append(p[pi])
        neg_all.append(n[ni])

    if not pos_all:
        return np.array([], dtype=int), np.array([], dtype=int)

    pos_idx = np.concatenate(pos_all)
    neg_idx = np.concatenate(neg_all)

    if max_pairs is not None and len(pos_idx) > max_pairs:
        rng = rng or np.random.default_rng(0)
        take = rng.choice(len(pos_idx), max_pairs, replace=False)
        pos_idx, neg_idx = pos_idx[take], neg_idx[take]

    return pos_idx, neg_idx


class PairwiseRanker:
    """MLP scorer trained on age-matched, within-site pairs.

    Emits a score, not a probability. Calibrate separately when a probability
    is needed for the binary decision rule.
    """

    def __init__(self, hidden=64, dropout=0.3, lr=1e-3, weight_decay=1e-4,
                 epochs=200, pairs_per_epoch=100_000, gap=DEFAULT_GAP,
                 within_site=True, site_balanced=True, vrex_beta=0.0,
                 patience=25, seed=42, verbose=False):
        if not HAVE_TORCH:
            raise ImportError('PairwiseRanker requires torch')
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.pairs_per_epoch = pairs_per_epoch
        self.gap = gap
        self.within_site = within_site
        self.site_balanced = site_balanced
        self.vrex_beta = vrex_beta
        self.patience = patience
        self.seed = seed
        self.verbose = verbose
        self.model_ = None
        self.mean_ = None
        self.std_ = None

    def _prepare(self, X):
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        if self.mean_ is None:
            self.mean_ = np.nanmedian(X, axis=0)
            self.mean_ = np.where(np.isfinite(self.mean_), self.mean_, 0.0)
        filled = np.where(np.isnan(X), self.mean_, X)
        if self.std_ is None:
            sd = filled.std(axis=0)
            self.std_ = np.where(sd > 1e-8, sd, 1.0)
        return ((filled - self.mean_) / self.std_).astype(np.float32)

    def fit(self, X, y, sites=None, ages=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        Xs = self._prepare(X)
        y = np.asarray(y).ravel()
        ages = np.asarray(ages, dtype=float).ravel()
        sites = np.zeros(len(y)) if sites is None else np.asarray(sites).ravel()

        # Enumerate eligible pairs per site so we can weight sites equally: the
        # dominant site holds ~78% of records and would otherwise define the loss.
        per_site = {}
        for site in np.unique(sites):
            m = sites == site
            idx = np.flatnonzero(m)
            p, n = build_pairs(y[m], ages[m], gap=self.gap, within_site=False)
            if len(p):
                per_site[site] = (idx[p], idx[n])

        if not per_site:
            raise ValueError('no age-matched within-site pairs available')

        if not self.within_site:
            allp = np.concatenate([v[0] for v in per_site.values()])
            alln = np.concatenate([v[1] for v in per_site.values()])
            per_site = {'_all': (allp, alln)}

        n_features = Xs.shape[1]
        self.model_ = nn.Sequential(
            nn.Linear(n_features, self.hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden, 1),
        )
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        Xt = torch.from_numpy(Xs)

        sites_list = list(per_site)
        per_site_budget = max(1, self.pairs_per_epoch // len(sites_list))

        best, best_state, bad = float('inf'), None, 0
        for epoch in range(self.epochs):
            self.model_.train()
            opt.zero_grad()

            losses = []
            for site in sites_list:
                p_idx, n_idx = per_site[site]
                budget = per_site_budget if self.site_balanced else \
                    max(1, int(self.pairs_per_epoch * len(p_idx)
                               / sum(len(v[0]) for v in per_site.values())))
                take = rng.integers(0, len(p_idx), size=min(budget, len(p_idx)))
                sp = self.model_(Xt[p_idx[take]]).squeeze(-1)
                sn = self.model_(Xt[n_idx[take]]).squeeze(-1)
                losses.append(nn.functional.softplus(-(sp - sn)).mean())

            stacked = torch.stack(losses)
            loss = stacked.mean()
            if self.vrex_beta > 0 and len(losses) > 1:
                # V-REx: penalize disagreement in per-site risk, pushing toward
                # a solution that holds across sites rather than on average.
                loss = loss + self.vrex_beta * stacked.var()

            loss.backward()
            opt.step()

            value = float(loss.item())
            if value < best - 1e-5:
                best, bad = value, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

            if self.verbose and epoch % 25 == 0:
                print(f'  epoch {epoch:4d} loss={value:.5f}')

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def decision_function(self, X):
        self.model_.eval()
        with torch.no_grad():
            return self.model_(
                torch.from_numpy(self._prepare(X))).squeeze(-1).numpy()

    def predict_proba(self, X):
        """Sigmoid of the score. Monotone in the score, so AUROC is unchanged.

        This is a convenience for code expecting a probability; it is not
        calibrated, and the binary decision rule needs a calibrated posterior.
        """
        s = self.decision_function(X)
        p = 1.0 / (1.0 + np.exp(-s))
        return np.column_stack([1 - p, p])
