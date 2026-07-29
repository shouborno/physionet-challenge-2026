#!/usr/bin/env python
"""Evaluate Philosopher's Stone latents under the challenge metric.

Gates, stated before the numbers are seen so the decision is not fitted to them:

  G1  Best PS-alone variant reaches at least 0.58 on the worst adequately
      powered fold. Below that the track is dropped.
  G2  Hand features combined with PS beat hand features alone by at least 0.02
      on the worst powered fold, and lose on no fold.

Everything is scored with age-conditioned AUROC at gap=2. Plain AUROC is
printed only as a contrast: this model's first regression target is age_z, so a
strong plain-AUROC number from these latents would be largely age, which the
challenge metric discounts entirely. The gap between the two columns is itself
the measurement of how much of the model's apparent skill survives.

Three latent variants are compared for the same reason: embedded with the true
age covariate, with age pinned to the pretraining cohort mean, and true-age
embeddings with the age trend splined out afterwards.

Usage:
    python scripts/eval_ps_latents.py --ps-dir /scratch/.../ps_small \
        --features data/processed/features_small_v8.pkl
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from sklearn.metrics import roc_auc_score  # noqa: E402

from src.data.self_norm import META  # noqa: E402
from src.eval import challenge_metrics as cm  # noqa: E402
from src.eval import protocol  # noqa: E402
from src.models.age_adjust import AgeResidualizer  # noqa: E402

G1_MIN_ALONE = 0.58
G2_MIN_GAIN = 0.02


def load_latents(ps_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(ps_dir, '*.npz'))):
        base = os.path.basename(path)[:-4]
        patient, _, session = base.rpartition('_ses-')
        with np.load(path, allow_pickle=True) as z:
            rows.append({
                'patient_id': patient,
                'session_id': int(session),
                'latent_true': z['latent_true'],
                'latent_pinned': z['latent_pinned'],
                'head_values': z['head_values'] if 'head_values' in z else None,
                'head_names': list(z['head_names']) if 'head_names' in z else [],
            })
    return pd.DataFrame(rows)


def ridge_fit(alpha=10.0):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_predict(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=1.0 / alpha, max_iter=4000,
                                         penalty='l2', random_state=42)),
        ])
        pipe.fit(X_tr, y_tr)
        return pipe.predict_proba(X_te)[:, 1]

    return fit_predict


def lgbm_fit():
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    def fit_predict(X_tr, y_tr, sites_tr, ages_tr, X_te, ages_te):
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=15,
                min_child_samples=50, subsample=0.8, colsample_bytree=0.5,
                reg_lambda=1.0, random_state=42, verbose=-1)),
        ])
        counts = pd.Series(sites_tr).value_counts()
        w = np.array([1.0 / counts[s] for s in sites_tr])
        pipe.fit(X_tr, y_tr, model__sample_weight=w / w.mean())
        return pipe.predict_proba(X_te)[:, 1]

    return fit_predict


def residualize_matrix(X, ages, y):
    return AgeResidualizer().fit_transform(X, ages, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ps-dir', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', default='results/ps_eval.json')
    parser.add_argument('--n-boot', type=int, default=1000)
    args = parser.parse_args()

    ps_df = load_latents(args.ps_dir)
    print(f'loaded {len(ps_df)} PS latents')
    if ps_df.empty:
        raise SystemExit('no latents found')

    feats = pd.read_pickle(args.features)
    feats = feats[feats['label'].notna()].copy()

    merged = feats.merge(ps_df, on=['patient_id', 'session_id'], how='inner')
    print(f'matched {len(merged)} of {len(feats)} labelled records')
    if len(merged) < 100:
        raise SystemExit('too few matched records to evaluate')

    y = merged['label'].to_numpy(dtype=float)
    sites = merged['site_id'].to_numpy()
    ages = merged['age'].to_numpy(dtype=float)
    prevalence = cm.prevalence_by_age(ages, y, ages, gap=2)

    hand_cols = [c for c in merged.columns
                 if c not in META and c not in
                 ('latent_true', 'latent_pinned', 'head_values', 'head_names')
                 and pd.api.types.is_numeric_dtype(merged[c])]
    X_hand = merged[hand_cols].to_numpy(dtype=float)
    L_true = np.vstack(merged['latent_true'].to_numpy())
    L_pin = np.vstack(merged['latent_pinned'].to_numpy())
    L_resid = residualize_matrix(L_true, ages, y)

    heads = None
    if merged['head_values'].iloc[0] is not None:
        try:
            heads = np.vstack(merged['head_values'].to_numpy())
        except ValueError:
            heads = None

    print(f'hand features={X_hand.shape[1]}  latent={L_true.shape[1]}'
          + (f'  heads={heads.shape[1]}' if heads is not None else ''))

    # How much of the latent is simply age? Answering this first frames
    # everything that follows.
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_predict
    age_hat = cross_val_predict(Ridge(alpha=10.0), L_true, ages, cv=5)
    r2 = 1 - np.sum((ages - age_hat) ** 2) / np.sum((ages - ages.mean()) ** 2)
    print(f'\nage recoverable from the true-age latent: R^2={r2:.3f}')
    age_hat_pin = cross_val_predict(Ridge(alpha=10.0), L_pin, ages, cv=5)
    r2p = 1 - np.sum((ages - age_hat_pin) ** 2) / np.sum((ages - ages.mean()) ** 2)
    print(f'age recoverable from the pinned latent  : R^2={r2p:.3f}')

    candidates = {
        'hand_only_lgbm': (X_hand, lgbm_fit()),
        'ps_true_ridge': (L_true, ridge_fit()),
        'ps_pinned_ridge': (L_pin, ridge_fit()),
        'ps_ageresid_ridge': (L_resid, ridge_fit()),
        'ps_true_lgbm': (L_true, lgbm_fit()),
        'hand_plus_ps_true': (np.hstack([X_hand, L_true]), lgbm_fit()),
        'hand_plus_ps_resid': (np.hstack([X_hand, L_resid]), lgbm_fit()),
    }
    if heads is not None:
        candidates['ps_heads_ridge'] = (heads, ridge_fit())
        candidates['hand_plus_heads'] = (np.hstack([X_hand, heads]), lgbm_fit())

    results, summary = {}, []
    for name, (X, fit) in candidates.items():
        print(f'\n--- {name} ({X.shape[1]} features) ---')
        try:
            folds = protocol.evaluate(fit, X, y, sites, ages,
                                      prevalence=prevalence, n_boot=args.n_boot)
        except Exception as exc:  # noqa: BLE001
            print(f'  FAILED: {type(exc).__name__}: {exc}')
            continue
        results[name] = folds.to_dict('records')
        powered = folds[folds['powered']]
        worst = float(powered['auroc_age'].min()) if not powered.empty else np.nan
        summary.append((name, worst, protocol.selection_statistic(folds)))

    print('\n' + '=' * 74)
    print('RANKING (worst adequately powered fold, age-conditioned AUROC)')
    print('=' * 74)
    for name, worst, sel in sorted(summary, key=lambda t: -(t[1] if np.isfinite(t[1]) else -1)):
        print(f'  {name:26s} worst={worst:.4f}  select={sel:.4f}')

    table = {n: w for n, w, _ in summary}
    hand = table.get('hand_only_lgbm', np.nan)
    ps_alone = max((table.get(k, np.nan) for k in
                    ('ps_true_ridge', 'ps_pinned_ridge', 'ps_ageresid_ridge',
                     'ps_true_lgbm', 'ps_heads_ridge')), default=np.nan)
    combined = max((table.get(k, np.nan) for k in
                    ('hand_plus_ps_true', 'hand_plus_ps_resid',
                     'hand_plus_heads')), default=np.nan)

    print('\nGATES')
    g1 = np.isfinite(ps_alone) and ps_alone >= G1_MIN_ALONE
    print(f'  G1 PS alone >= {G1_MIN_ALONE}: {ps_alone:.4f} -> {"PASS" if g1 else "FAIL"}')
    g2 = np.isfinite(combined) and np.isfinite(hand) and \
        (combined - hand) >= G2_MIN_GAIN
    print(f'  G2 combined beats hand by >= {G2_MIN_GAIN}: '
          f'{combined:.4f} vs {hand:.4f} (gain {combined - hand:+.4f}) '
          f'-> {"PASS" if g2 else "FAIL"}')

    if not (g1 or g2):
        print('\nBoth gates fail. On this evidence the pretrained model adds '
              'nothing the hand features do not already carry.')
    elif g2:
        print('\nG2 passes: the latents contribute signal beyond the hand '
              'features. Worth carrying into a submission.')
    else:
        print('\nG1 passes but G2 does not: the latents are informative alone '
              'yet redundant with the hand features.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump({'folds': results,
                   'age_r2_true': float(r2), 'age_r2_pinned': float(r2p),
                   'gates': {'g1': bool(g1), 'g2': bool(g2)}},
                  fh, indent=2, default=float)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
