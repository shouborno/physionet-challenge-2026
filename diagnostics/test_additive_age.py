#!/usr/bin/env python
"""Tune how much age the model is allowed to use, instead of choosing all or none.

Withholding age entirely gained 0.025, but that is a blunt instrument. Age is
not worth zero under this metric: because the caliper is two years wide and
prevalence climbs steeply with age, the positive in a matched pair is
systematically slightly older, and age alone still scores 0.5375 rather than
0.5000. Dropping the column throws that away.

The refinement is to force additive separability and then scale the age term.
LightGBM interaction constraints, with age in its own group, give

    f(x, a) = h(x) + q(a)

and the two parts can be separated without refitting. Predicting once with the
true age and once with age pinned to a reference value gives

    s_full = h(x) + q(a),  s_ref = h(x) + q(a_ref)

so the tempered score is just an interpolation:

    s_lambda = s_ref + lambda * (s_full - s_ref)

lambda = 0 is fully age-neutral, lambda = 1 is the unconstrained model, and the
optimum is a one-dimensional sweep rather than a binary choice. Scores are
combined on the log-odds scale, where the model is additive; interpolating raw
probabilities would not respect the constraint.
"""

import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/simran/sleep-study-cognitive-screening-challenge')

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402

from src.data.feature_io import load_features  # noqa: E402
from src.eval import challenge_metrics as cm  # noqa: E402
from src.eval import protocol  # noqa: E402

LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]


def make_fit(lam):
    def fit_predict(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
        # Age occupies the final column; constraining it to its own group
        # forbids any interaction, so its contribution is a pure additive term.
        age_idx = X_tr.shape[1] - 1
        others = list(range(age_idx))

        imp = SimpleImputer(strategy='median')
        Ztr = imp.fit_transform(X_tr)
        Zte = imp.transform(X_te)

        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15,
            min_child_samples=50, colsample_bytree=0.5, reg_lambda=1.0,
            random_state=42, verbose=-1,
            interaction_constraints=[[age_idx], others])
        counts = pd.Series(sites_tr).value_counts()
        w = np.array([1.0 / counts[s] for s in sites_tr])
        model.fit(Ztr, y_tr, sample_weight=w / w.mean())

        # Reference age: the training median, so the pinned prediction stays
        # inside the range the model was fitted on.
        ref = float(np.nanmedian(ages_tr))
        Zte_ref = Zte.copy()
        Zte_ref[:, age_idx] = ref

        full = model.predict_proba(Zte)[:, 1]
        base = model.predict_proba(Zte_ref)[:, 1]

        def logit(p):
            p = np.clip(p, 1e-9, 1 - 1e-9)
            return np.log(p / (1 - p))

        # Interpolate where the model is additive.
        return logit(base) + lam * (logit(full) - logit(base))

    return fit_predict


def main():
    df, cols = load_features('data/processed/features_small_v11.pkl',
                             drop_age=True, drop_cols=['bmi'])
    # Age last, so its index is known to the constraint.
    X = np.column_stack([df[cols].to_numpy(dtype=float),
                         df['age'].to_numpy(dtype=float)])
    y = df['label'].to_numpy(dtype=float)
    sites = df['site_id'].to_numpy()
    ages = df['age'].to_numpy(dtype=float)
    prevalence = cm.prevalence_by_age(ages, y, ages, gap=2)

    print(f'{len(df)} records, {len(cols)} features plus age')
    print('reference: age withheld entirely = 0.6786\n')

    for lam in LAMBDAS:
        folds = protocol.evaluate(make_fit(lam), X, y, sites, ages,
                                  prevalence=prevalence, n_boot=400,
                                  verbose=False)
        powered = folds[folds['powered']]
        per = {r.fold.replace('holdout_', '').replace('train_', 't_'):
               round(r.auroc_age, 4) for _, r in folds.iterrows()}
        print(f'lambda={lam:<5} worst={powered.auroc_age.min():.4f}  '
              f'select={protocol.selection_statistic(folds):.4f}  {per}')


if __name__ == '__main__':
    main()
