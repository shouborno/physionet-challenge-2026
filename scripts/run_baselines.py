#!/usr/bin/env python
"""Compare candidate models under the official metric and honest folds.

Produces the table the CinC paper is built on. Every row is scored with
age-conditioned AUROC at gap=2, per fold, with the age-matched pair count and a
patient-level bootstrap interval. Ranking uses the lower bound of the worst
adequately-powered fold, never a point estimate, because point-estimate
selection over many candidates is what turned 0.780 LOSO into 0.644 on the
leaderboard last time.

Plain AUROC is reported alongside purely for contrast: age alone scores 0.772
plain against 0.538 conditioned, so a model can look strong on the old metric
while earning almost nothing on the new one.

Usage:
    python scripts/run_baselines.py --features data/processed/features_small_v7.pkl
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

from src.eval import challenge_metrics as cm  # noqa: E402
from src.eval import protocol  # noqa: E402
from src.models.age_adjust import AgeConditionalStandardizer, AgeResidualizer  # noqa: E402

META = {'patient_id', 'site_id', 'session_id', 'label', 'extract_time_sec',
        'demo_age', 'demo_sex', 'Time_to_Event', 'Time_to_Last_Visit'}


def make_sklearn_fit(estimator_factory, residualize=False, standardize=False,
                     site_balanced=False, matched_weights=False, temper=1.0):
    """Wrap a scikit-learn estimator as a fit_predict for the protocol."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_predict(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
        if residualize:
            res = AgeResidualizer().fit(X_tr, ages_tr, y_tr)
            X_tr, X_te = res.transform(X_tr, ages_tr), res.transform(X_te, ages_te)

        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', estimator_factory()),
        ])

        kwargs = {}
        weights = None
        if matched_weights:
            # Weight each subject by its count of eligible opposite-class
            # partners inside the 2-year caliper: the per-stratum balancing
            # that makes a pointwise logistic model AUC-consistent with
            # constant 2 rather than ~10 at 7.6% prevalence.
            from src.models.matched_weights import matched_pair_weights
            weights = matched_pair_weights(y_tr, ages_tr, temper=temper)
        if site_balanced:
            # Weight each site equally. The dominant site holds ~78% of records
            # and would otherwise define the fit on its own; this is most of
            # what group DRO buys, without the implementation risk.
            counts = pd.Series(sites_tr).value_counts()
            w = np.array([1.0 / counts[s] for s in sites_tr])
            w = w / w.mean()
            weights = w if weights is None else weights * w
        if weights is not None:
            kwargs['model__sample_weight'] = weights / weights.mean()

        pipe.fit(X_tr, y_tr, **kwargs)
        scores = pipe.predict_proba(X_te)[:, 1]

        if standardize:
            std = AgeConditionalStandardizer().fit(
                pipe.predict_proba(X_tr)[:, 1], ages_tr)
            scores = std.transform(scores, ages_te)
        return scores

    return fit_predict


def make_pairwise_fit(residualize=False, **kwargs):
    from src.models.pairwise_rank import PairwiseRanker

    def fit_predict(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
        Xtr, Xte = X_tr, X_te
        if residualize:
            res = AgeResidualizer().fit(X_tr, ages_tr, y_tr)
            Xtr, Xte = res.transform(X_tr, ages_tr), res.transform(X_te, ages_te)
        ranker = PairwiseRanker(**kwargs)
        ranker.fit(Xtr, y_tr, sites_tr, ages_tr)
        return ranker.decision_function(Xte)

    return fit_predict


def make_clogit_fit(alpha=1.0, width=4.0, n_offsets=4):
    """Conditional logistic regression over age-matched strata.

    The metric is a matched case-control comparison, and this is the exact
    estimator for that design: conditioning on each stratum removes its
    intercept from the likelihood, so age cannot influence the fit without
    estimating and subtracting an age trend first.
    """
    from src.models.conditional_logit import ConditionalLogit

    def fit_predict(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
        model = ConditionalLogit(alpha=alpha, width=width, n_offsets=n_offsets)
        model.fit(X_tr, y_tr, ages_tr)
        return model.decision_function(X_te)

    return fit_predict


def make_tabpfn_fit(n_components=300):
    """TabPFN v2 with in-fold PCA to respect its feature limit.

    TabPFN is a prior-fitted transformer designed for small tabular problems,
    the regime with the strongest published edge at roughly our sample size. It
    caps at about 500 features and we now carry 953, so dimensionality has to
    come down. PCA rather than feature selection: it is unsupervised, so it
    cannot leak the label, whereas supervised selection over 953 columns at 62
    effective positives is the exact mechanism that produced the 0.136 optimism
    gap in the unofficial phase.
    """
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_predict(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
        from tabpfn import TabPFNClassifier
        pre = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=min(n_components, X_tr.shape[1],
                                         X_tr.shape[0] - 1), random_state=42)),
        ])
        Ztr = pre.fit_transform(X_tr)
        Zte = pre.transform(X_te)
        clf = TabPFNClassifier(device='cpu', random_state=42)
        clf.fit(Ztr, y_tr.astype(int))
        return clf.predict_proba(Zte)[:, 1]

    return fit_predict


def make_site_covariate_fit(estimator_factory):
    """Include site as a model input, with an unknown-site code at inference.

    Removing site failed here three ways (ComBat, rank normalization, filtering
    site-predictive features), and there is a theoretical reason: driving the
    feature-marginal divergence to zero leaves error floored by the divergence
    between label marginals, which is real since site prevalence runs 6.5% to
    14.8%. The recommended alternative is to model site rather than subtract it.
    Held-out folds see an unseen code, which is exactly the inference condition.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_predict(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
        codes = {s: i for i, s in enumerate(sorted(pd.unique(sites_tr)))}
        unknown = len(codes)
        tr = np.column_stack([X_tr, [codes[s] for s in sites_tr]])
        # Every test record is from an unseen site under leave-one-site-out,
        # matching how the model will be used.
        te = np.column_stack([X_te, np.full(len(X_te), unknown)])
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', estimator_factory()),
        ])
        pipe.fit(tr, y_tr)
        return pipe.predict_proba(te)[:, 1]

    return fit_predict


def constant_age_fit(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
    """Age as the only predictor: the floor the metric is designed to remove."""
    return ages_te


def build_candidates(n_features):
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb

    def lr(C):
        return lambda: LogisticRegression(C=C, max_iter=2000, penalty='l2',
                                          random_state=42)

    def elasticnet(l1_ratio):
        return lambda: LogisticRegression(
            C=0.05, max_iter=4000, penalty='elasticnet', solver='saga',
            l1_ratio=l1_ratio, random_state=42)

    def gbm():
        return lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.5,
            reg_lambda=1.0, random_state=42, verbose=-1)

    candidates = {
        'age_only': constant_age_fit,
        'lgbm_sitecov': make_site_covariate_fit(gbm),
        'tabpfn_pca300': make_tabpfn_fit(300),
        'lgbm_matchedw': make_sklearn_fit(gbm, matched_weights=True),
        'lgbm_matchedw_t075': make_sklearn_fit(gbm, matched_weights=True, temper=0.75),
        'lgbm_matchedw_t05': make_sklearn_fit(gbm, matched_weights=True, temper=0.5),
        'lgbm_matchedw_siteb': make_sklearn_fit(gbm, matched_weights=True,
                                                site_balanced=True),
        'lr_C0.005_shipped': make_sklearn_fit(lr(0.005)),
        'lr_C0.005_siteweighted': make_sklearn_fit(lr(0.005), site_balanced=True),
        'lr_C0.005_ageresid': make_sklearn_fit(lr(0.005), residualize=True),
        'lr_C0.005_ageresid_agestd': make_sklearn_fit(
            lr(0.005), residualize=True, standardize=True),
        'elasticnet_l1_0.5': make_sklearn_fit(elasticnet(0.5), site_balanced=True),
        'lgbm': make_sklearn_fit(gbm, site_balanced=True),
        'lgbm_ageresid': make_sklearn_fit(gbm, residualize=True, site_balanced=True),
    }

    try:
        import torch  # noqa: F401
        candidates['clogit_a1'] = make_clogit_fit(alpha=1.0)
        candidates['clogit_a10'] = make_clogit_fit(alpha=10.0)
        candidates['clogit_a100'] = make_clogit_fit(alpha=100.0)
        candidates['pairwise_withinsite'] = make_pairwise_fit(
            hidden=64, epochs=300, seed=42)
        candidates['pairwise_withinsite_ageresid'] = make_pairwise_fit(
            residualize=True, hidden=64, epochs=300, seed=42)
        candidates['pairwise_crosssite'] = make_pairwise_fit(
            hidden=64, epochs=300, within_site=False, seed=42)
        candidates['pairwise_vrex'] = make_pairwise_fit(
            hidden=64, epochs=300, vrex_beta=1.0, seed=42)
    except ImportError:
        print('torch unavailable; skipping pairwise candidates')

    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', default='results/baselines.json')
    parser.add_argument('--only', nargs='*', default=None)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--drop-cols', nargs='*', default=None,
                        help='additional feature columns to exclude')
    parser.add_argument('--drop-age', action='store_true',
                        help='exclude age as a model input; it still drives the '
                             'metric and the decision threshold')
    args = parser.parse_args()

    df = pd.read_pickle(args.features)
    df = df[df['label'].notna()].copy()

    feature_cols = [c for c in df.columns
                    if c not in META and pd.api.types.is_numeric_dtype(df[c])]
    if args.drop_age:
        # Age contributes nothing the metric rewards, and measurably displaces
        # signal that does: the fitted score correlates with age at rho=0.33
        # and the worst powered fold improves by 0.025 once it is removed.
        feature_cols = [c for c in feature_cols if c != 'age']
        print('dropping age from model inputs')
    if args.drop_cols:
        # BMI is 75.9% missing and whether it was recorded is a
        # healthcare-contact proxy that does not transfer: at S0001 prevalence
        # is 36.9% when present against 3.2% when absent, while I0006 shows
        # nothing. Making the missingness explicit collapsed the inverted fold
        # from 0.697 to 0.574.
        before = len(feature_cols)
        feature_cols = [c for c in feature_cols if c not in set(args.drop_cols)]
        print(f'dropping {before - len(feature_cols)} column(s): '
              f'{sorted(set(args.drop_cols))}')
    X = df[feature_cols].to_numpy(dtype=float)
    y = df['label'].to_numpy(dtype=float)
    sites = df['site_id'].to_numpy()
    ages = df['age'].to_numpy(dtype=float)

    print(f'records={len(df)}  features={len(feature_cols)}  '
          f'prevalence={y.mean():.4f}')
    print(f'sites: {dict(pd.Series(sites).value_counts())}')

    prevalence = cm.prevalence_by_age(ages, y, ages, gap=2)

    candidates = build_candidates(len(feature_cols))
    if args.only:
        candidates = {k: v for k, v in candidates.items() if k in args.only}

    results = {}
    summaries = []
    for name, fit_predict in candidates.items():
        print(f'\n--- {name} ---')
        try:
            folds = protocol.evaluate(
                fit_predict, X, y, sites, ages,
                n_boot=args.n_boot, prevalence=prevalence)
        except Exception as exc:
            print(f'  FAILED: {type(exc).__name__}: {exc}')
            continue
        results[name] = folds.to_dict('records')
        summaries.append((name, protocol.selection_statistic(folds), folds))
        print(protocol.summarize(folds, name))

    print('\n' + '=' * 78)
    print('RANKING  (lower bound of the worst adequately-powered fold)')
    print('=' * 78)
    for name, stat, folds in sorted(summaries, key=lambda t: -t[1]):
        powered = folds[folds['powered']]
        worst = powered['auroc_age'].min() if not powered.empty else float('nan')
        mean = powered['auroc_age'].mean() if not powered.empty else float('nan')
        print(f'  {name:34s} select={stat:.4f}  worst={worst:.4f}  mean={mean:.4f}')

    print('\nNote: differences below ~0.05 are inside the leaderboard noise '
          'floor at 5-15% prevalence and should not cost a submission slot.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
