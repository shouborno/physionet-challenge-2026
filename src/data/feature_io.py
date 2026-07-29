#!/usr/bin/env python
"""Loading feature matrices with the label-revealing columns removed.

`Time_to_Event` is the days from PSG to the first cognitive-impairment ICD
code, so it is finite for exactly the positives and missing for every negative;
median imputation turns "is missing" into a perfect label indicator, and it
alone gives held-out AUROC 1.0. `Time_to_Last_Visit` leaks more subtly, since
every negative needs at least six years of follow-up while positives start much
lower.

Both live inside the feature pickles, and until now the protection lived in the
consumers: every analysis script filters through `META`. That works only for as
long as every future script remembers to. A collaborating agent hit exactly this
by hand-rolling its own drop list, and reported 0.999 AUROC before catching it.

This loader makes the guarantee structural. Use it instead of `pd.read_pickle`
and the leak cannot reach a model.
"""

import numpy as np
import pandas as pd

# Present in training demographics, absent at inference. Using any of them as a
# feature trains the model on something it can never see again.
TRAINING_ONLY = ('Time_to_Event', 'Last_Known_Visit_Date', 'Time_to_Last_Visit',
                 'Cognitive_Impairment')

# Identifiers and targets: not leaks, but not features either.
NON_FEATURES = ('patient_id', 'site_id', 'session_id', 'label',
                'extract_time_sec', 'coh_time_sec', 'demo_age', 'demo_sex')

EXCLUDE = frozenset(TRAINING_ONLY + NON_FEATURES)


def load_features(path, drop_age=False, drop_cols=(), labelled_only=True):
    """Return (frame, feature_columns) with leaking columns already excluded."""
    df = pd.read_pickle(path)
    if labelled_only and 'label' in df.columns:
        df = df[df['label'].notna()].copy()

    cols = [c for c in df.columns
            if c not in EXCLUDE and pd.api.types.is_numeric_dtype(df[c])]
    if drop_age:
        cols = [c for c in cols if c != 'age']
    if drop_cols:
        cols = [c for c in cols if c not in set(drop_cols)]

    assert_no_leaks(df, cols)
    return df, cols


def assert_no_leaks(df, feature_cols, y=None, max_univariate_auroc=0.95):
    """Fail loudly on a leaking column, by name and by behaviour.

    The name check catches the known columns. The behaviour check catches
    anything new that separates the classes almost perfectly, which no real
    physiological feature does here: the strongest legitimate one reaches 0.74.
    """
    named = [c for c in feature_cols if c in TRAINING_ONLY]
    if named:
        raise ValueError(f'training-only columns in feature list: {named}')

    if y is None and 'label' in df.columns:
        y = df['label'].to_numpy(dtype=float)
    if y is None or len(np.unique(y[np.isfinite(y)])) < 2:
        return

    from sklearn.metrics import roc_auc_score

    suspects = []
    for c in feature_cols:
        v = df[c].to_numpy(dtype=float)
        ok = np.isfinite(v) & np.isfinite(y)
        if ok.sum() < 50 or len(np.unique(y[ok])) < 2:
            continue
        auc = roc_auc_score(y[ok], v[ok])
        auc = max(auc, 1 - auc)
        if auc >= max_univariate_auroc:
            suspects.append((c, round(float(auc), 4)))
        # Missingness can leak even when the values do not.
        if df[c].isna().any():
            miss_auc = roc_auc_score(y, df[c].isna().astype(int))
            if max(miss_auc, 1 - miss_auc) >= max_univariate_auroc:
                suspects.append((f'{c} (missingness)',
                                 round(float(max(miss_auc, 1 - miss_auc)), 4)))

    if suspects:
        raise ValueError(f'columns separate the classes almost perfectly, '
                         f'which no real feature here does: {suspects}')
