"""
Combine per-fold TabPFN results into a unified feature ranking and K-sweep summary.

Usage:
    python combine_folds.py --indir results/tabpfn_selection --outdir results/tabpfn_selection
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


SITES = ["S0001", "I0002", "I0006"]


def load_folds(indir):
    folds = {}
    for site in SITES:
        path = os.path.join(indir, f"fold_{site}.json")
        with open(path) as f:
            folds[site] = json.load(f)
    return folds


def build_feature_table(folds):
    """Build a unified feature ranking table."""
    # Collect all features across folds
    all_features = set()
    for fold in folds.values():
        all_features.update(fold["valid_features"])
    all_features = sorted(all_features)

    rows = []
    for feat in all_features:
        row = {"feature": feat}

        # Permutation importance per fold
        perm_vals = []
        for site in SITES:
            val = folds[site]["perm_importance_mean"].get(feat, np.nan)
            row[f"perm_imp_{site}"] = val
            if not np.isnan(val):
                perm_vals.append(val)

        row["perm_imp_mean"] = np.mean(perm_vals) if perm_vals else np.nan
        row["perm_imp_min"] = np.min(perm_vals) if perm_vals else np.nan

        # mRMR rank per fold
        mrmr_ranks = []
        for site in SITES:
            ranking = folds[site]["mrmr_ranking"]
            if feat in ranking:
                rank = ranking.index(feat) + 1
            else:
                rank = len(ranking) + 1  # Not present
            row[f"mrmr_rank_{site}"] = rank
            mrmr_ranks.append(rank)

        row["mrmr_rank_mean"] = np.mean(mrmr_ranks)
        row["mrmr_rank_max"] = np.max(mrmr_ranks)  # Worst rank across folds

        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by mean permutation importance (descending)
    df = df.sort_values("perm_imp_mean", ascending=False).reset_index(drop=True)
    return df


def build_ksweep_table(folds):
    """Build K-sweep AUROC summary."""
    rows = []
    all_ks = set()
    for fold in folds.values():
        all_ks.update(fold["k_sweep_auroc"].keys())
    all_ks = sorted(all_ks, key=int)

    for k in all_ks:
        row = {"K": int(k)}
        vals = []
        for site in SITES:
            auroc = folds[site]["k_sweep_auroc"].get(k, np.nan)
            row[f"auroc_{site}"] = auroc
            if not np.isnan(auroc):
                vals.append(auroc)
        row["auroc_mean"] = np.mean(vals)
        row["auroc_min"] = np.min(vals)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", default="results/tabpfn_selection")
    parser.add_argument("--outdir", default="results/tabpfn_selection")
    args = parser.parse_args()

    folds = load_folds(args.indir)

    # Per-fold summary
    print("=" * 70)
    print("PER-FOLD SUMMARY")
    print("=" * 70)
    for site in SITES:
        fold = folds[site]
        print(f"\n  Hold-out {site}: AUROC(all features) = {fold['auroc_all_features']:.4f}")
        print(f"    Train: {fold['n_train']}, Test: {fold['n_test']}")
        print(f"    mRMR top 5: {fold['mrmr_ranking'][:5]}")
        print(f"    Times: mRMR={fold['mrmr_time_sec']:.1f}s, fit={fold['fit_time_sec']:.1f}s, perm={fold['perm_time_sec']:.1f}s")

    # All-features AUROC
    aurocs = [folds[s]["auroc_all_features"] for s in SITES]
    print(f"\n  LOSO mean AUROC (all features): {np.mean(aurocs):.4f}")
    print(f"  LOSO min AUROC (all features):  {np.min(aurocs):.4f}")

    # K-sweep table
    print("\n" + "=" * 70)
    print("K-SWEEP (mRMR top-K features)")
    print("=" * 70)
    ksweep = build_ksweep_table(folds)
    print(ksweep.to_string(index=False, float_format="%.4f"))

    best_mean_k = ksweep.loc[ksweep["auroc_mean"].idxmax(), "K"]
    best_min_k = ksweep.loc[ksweep["auroc_min"].idxmax(), "K"]
    print(f"\n  Best K (mean AUROC): {best_mean_k}")
    print(f"  Best K (min-site AUROC): {best_min_k}")

    # Feature ranking table
    print("\n" + "=" * 70)
    print("FEATURE RANKING (by mean permutation importance)")
    print("=" * 70)
    feat_df = build_feature_table(folds)
    # Show top 30
    cols_display = ["feature", "perm_imp_mean", "perm_imp_min",
                    "mrmr_rank_mean", "mrmr_rank_max"]
    print(feat_df[cols_display].head(30).to_string(index=False, float_format="%.4f"))

    # Features with positive importance across ALL folds
    perm_cols = [f"perm_imp_{s}" for s in SITES]
    consistently_positive = feat_df[feat_df[perm_cols].min(axis=1) > 0]
    print(f"\n  Features with positive importance in ALL folds: {len(consistently_positive)}")
    if len(consistently_positive) > 0:
        print("   ", consistently_positive["feature"].tolist())

    # Save outputs
    feat_df.to_csv(os.path.join(args.outdir, "feature_ranking.csv"), index=False)
    ksweep.to_csv(os.path.join(args.outdir, "k_sweep_summary.csv"), index=False)

    summary = {
        "loso_auroc_all_features": {s: folds[s]["auroc_all_features"] for s in SITES},
        "loso_mean_auroc_all": float(np.mean(aurocs)),
        "loso_min_auroc_all": float(np.min(aurocs)),
        "best_k_mean": int(best_mean_k),
        "best_k_minsite": int(best_min_k),
        "n_consistently_positive_features": len(consistently_positive),
        "consistently_positive_features": consistently_positive["feature"].tolist() if len(consistently_positive) > 0 else [],
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: feature_ranking.csv, k_sweep_summary.csv, summary.json")


if __name__ == "__main__":
    main()
