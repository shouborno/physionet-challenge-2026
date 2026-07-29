"""
TabPFN fold evaluation: mRMR ranking + TabPFN fit + permutation importance.

Usage:
    python run_fold.py --fold-site S0001 --data /path/to/features_training_v3.pkl --outdir /path/to/output
"""

import argparse
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from mrmr import mrmr_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

warnings.filterwarnings("ignore")

META_COLS = {"patient_id", "site_id", "session_id", "label", "extract_time_sec"}


def load_data(path):
    df = pickle.load(open(path, "rb"))
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = df[feature_cols].copy()
    y = df["label"].values.astype(int)
    sites = df["site_id"].values
    return X, y, sites, feature_cols


def run_fold(fold_site, X, y, sites, feature_cols, n_repeats=10):
    test_mask = sites == fold_site
    train_mask = ~test_mask

    X_train, y_train = X.iloc[train_mask].copy(), y[train_mask]
    X_test, y_test = X.iloc[test_mask].copy(), y[test_mask]

    print(f"Fold: hold out {fold_site}")
    print(f"  Train: {train_mask.sum()} samples, Test: {test_mask.sum()} samples")
    print(f"  Train sites: {np.unique(sites[train_mask])}")
    print(f"  Features: {len(feature_cols)}")

    # --- Handle NaN/Inf ---
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Drop columns that are all NaN in training
    valid_cols = X_train.columns[X_train.notna().any()].tolist()
    X_train = X_train[valid_cols]
    X_test = X_test[valid_cols]
    print(f"  Valid features (non-all-NaN): {len(valid_cols)}")

    # --- mRMR ranking ---
    print("  Computing mRMR ranking...")
    t0 = time.time()
    # mRMR needs a DataFrame for X and a Series for y
    X_train_filled = X_train.fillna(X_train.median())
    y_train_series = pd.Series(y_train, index=X_train_filled.index, name="label")
    mrmr_ranked = mrmr_classif(
        X_train_filled, y_train_series,
        K=len(valid_cols),
        show_progress=False,
    )
    mrmr_time = time.time() - t0
    print(f"  mRMR done in {mrmr_time:.1f}s, top 10: {mrmr_ranked[:10]}")

    # --- TabPFN on all features ---
    print("  Fitting TabPFN on all features...")
    t0 = time.time()
    clf = TabPFNClassifier(device="cuda", n_estimators=8)
    # X_train_filled already created above for mRMR
    X_test_filled = X_test.fillna(X_train.median())
    clf.fit(X_train_filled, y_train)
    y_prob = clf.predict_proba(X_test_filled)[:, 1]
    auroc_all = roc_auc_score(y_test, y_prob)
    fit_time = time.time() - t0
    print(f"  TabPFN all-features AUROC: {auroc_all:.4f} ({fit_time:.1f}s)")

    # --- Permutation importance ---
    print(f"  Computing permutation importance ({n_repeats} repeats)...")
    t0 = time.time()
    perm_result = permutation_importance(
        clf, X_test_filled, y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
    )
    perm_time = time.time() - t0
    print(f"  Permutation importance done in {perm_time:.1f}s")

    perm_imp_mean = perm_result.importances_mean
    perm_imp_std = perm_result.importances_std

    # --- K-sweep with mRMR-ranked features ---
    print("  K-sweep with mRMR ranking...")
    k_values = [10, 20, 30, 40, 50, 60, 80, len(valid_cols)]
    k_values = sorted(set(k for k in k_values if k <= len(valid_cols)))
    k_results = {}
    for k in k_values:
        top_k = mrmr_ranked[:k]
        X_tr_k = X_train_filled[top_k]
        X_te_k = X_test_filled[top_k]
        clf_k = TabPFNClassifier(device="cuda", n_estimators=8)
        clf_k.fit(X_tr_k, y_train)
        y_prob_k = clf_k.predict_proba(X_te_k)[:, 1]
        auroc_k = roc_auc_score(y_test, y_prob_k)
        k_results[k] = auroc_k
        print(f"    K={k:3d}: AUROC={auroc_k:.4f}")

    return {
        "fold_site": fold_site,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "valid_features": valid_cols,
        "mrmr_ranking": mrmr_ranked,
        "auroc_all_features": auroc_all,
        "perm_importance_mean": dict(zip(valid_cols, perm_imp_mean.tolist())),
        "perm_importance_std": dict(zip(valid_cols, perm_imp_std.tolist())),
        "k_sweep_auroc": {str(k): v for k, v in k_results.items()},
        "mrmr_time_sec": mrmr_time,
        "fit_time_sec": fit_time,
        "perm_time_sec": perm_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-site", required=True)
    parser.add_argument("--data", default="data/processed/features_training_v3.pkl")
    parser.add_argument("--outdir", default="results/tabpfn_selection")
    parser.add_argument("--n-repeats", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y, sites, feature_cols = load_data(args.data)
    result = run_fold(args.fold_site, X, y, sites, feature_cols, n_repeats=args.n_repeats)

    outpath = os.path.join(args.outdir, f"fold_{args.fold_site}.json")
    with open(outpath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
