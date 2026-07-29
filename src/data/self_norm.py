#!/usr/bin/env python
"""Within-recording self-referential features.

Site effects in PSG are mostly multiplicative or additive per recording:
amplifier gain, electrode impedance, filter settings, reference montage. A
feature expressed relative to another feature *from the same night* is immune to
all of them by construction. That is a stronger guarantee than estimating a
correction after the fact, which is why ComBat and rank normalization failed
here: there is no correct way to apply a learned per-site correction to a site
never seen.

Twenty-six feature families are measured in two or more sleep stages, so the
other stages of the same night provide the reference. Three derived forms:

  contrast   value(stage) - value(reference stage). Cancels any per-recording
             additive offset.
  z-stage    (value(stage) - mean over stages) / sd over stages. Cancels
             additive *and* multiplicative per-recording effects, the strongest
             of the three, at the cost of being noisy over few stages.
  range      max - min across stages. A within-recording dynamic range, which
             carries information the individual stages do not.

Everything is computed from an existing feature matrix, so adding these costs
no re-extraction.
"""

import re

import numpy as np
import pandas as pd

STAGES = ['wake', 'n1', 'n2', 'n3', 'rem']

# Preference order for the contrast reference. Wake is the natural baseline: it
# is present in every recording and is the state least affected by sleep
# architecture differences between patients.
REFERENCE_PREFERENCE = ['wake', 'n2', 'n3', 'rem', 'n1']

META = {'patient_id', 'site_id', 'session_id', 'label', 'extract_time_sec',
        'demo_age', 'demo_sex', 'Time_to_Event', 'Time_to_Last_Visit'}


def find_stage_families(columns):
    """Map family name -> {stage: column} for anything measured in 2+ stages."""
    families = {}
    for col in columns:
        for stage in STAGES:
            suffix = f'_{stage}'
            if col.endswith(suffix):
                families.setdefault(col[:-len(suffix)], {})[stage] = col
                break
    return {k: v for k, v in families.items() if len(v) >= 2}


def add_self_referential(df, feature_cols=None, contrasts=True, zstage=True,
                         ranges=True):
    """Append within-recording derived features. Returns (df, new_column_names).

    The input frame is not modified.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns
                        if c not in META and pd.api.types.is_numeric_dtype(df[c])]

    families = find_stage_families(feature_cols)
    # Accumulate then concat once. Assigning ~200 columns one at a time
    # fragments the frame badly enough that pandas warns about it.
    derived = {}
    new_cols = []

    for family, by_stage in sorted(families.items()):
        present = [s for s in REFERENCE_PREFERENCE if s in by_stage]
        if len(present) < 2:
            continue
        ref_stage = present[0]
        ref = df[by_stage[ref_stage]]

        if contrasts:
            for stage in present[1:]:
                name = f'{family}_{stage}_vs_{ref_stage}'
                derived[name] = df[by_stage[stage]] - ref
                new_cols.append(name)

        block = df[[by_stage[s] for s in present]]

        if zstage:
            mean = block.mean(axis=1)
            sd = block.std(axis=1)
            # A zero spread means the stages are identical; the z-score is
            # undefined there rather than zero, so leave it missing.
            sd = sd.where(sd > 1e-12)
            for stage in present:
                name = f'{family}_z_{stage}'
                derived[name] = (df[by_stage[stage]] - mean) / sd
                new_cols.append(name)

        if ranges:
            name = f'{family}_stagerange'
            derived[name] = block.max(axis=1) - block.min(axis=1)
            new_cols.append(name)

    out = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
    return out, new_cols


def site_predictiveness(values, sites):
    """Max one-vs-rest AUROC for predicting site from a single feature.

    A domain-adversarial score computed per feature: how well does this feature
    alone identify the recording site? Lower is better for generalization.
    """
    values = np.asarray(values, dtype=float)
    sites = np.asarray(sites)
    ok = np.isfinite(values)
    if ok.sum() < 20:
        return np.nan

    v, s = values[ok], sites[ok]
    order = np.argsort(v)
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = np.arange(1, len(v) + 1)

    best = 0.5
    for site in np.unique(s):
        pos = s == site
        n_pos, n_neg = int(pos.sum()), int((~pos).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        auc = (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        best = max(best, max(auc, 1 - auc))
    return best


def compare_site_leakage(df, original_cols, derived_cols, site_col='site_id'):
    """Check the derived features are less site-identifying than their sources.

    This is the claim the whole approach rests on, so it is worth measuring
    rather than assuming.
    """
    sites = df[site_col].to_numpy()

    def summarize(cols):
        scores = [site_predictiveness(df[c].to_numpy(), sites) for c in cols]
        scores = np.array([s for s in scores if np.isfinite(s)])
        if scores.size == 0:
            return {}
        return {
            'n': int(scores.size),
            'mean': float(scores.mean()),
            'median': float(np.median(scores)),
            'frac_over_0.65': float((scores > 0.65).mean()),
            'frac_over_0.80': float((scores > 0.80).mean()),
        }

    return {'original': summarize(original_cols),
            'derived': summarize(derived_cols)}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    df = pd.read_pickle(args.features)
    base = [c for c in df.columns
            if c not in META and pd.api.types.is_numeric_dtype(df[c])]

    families = find_stage_families(base)
    print(f'{len(base)} base features, {len(families)} multi-stage families')

    aug, new = add_self_referential(df, base)
    print(f'added {len(new)} self-referential features -> {len(base) + len(new)} total')

    report = compare_site_leakage(aug, base, new)
    print('\nsite-predictiveness (max one-vs-rest AUROC per feature; lower is better)')
    for group, stats in report.items():
        if stats:
            print(f'  {group:9s} n={stats["n"]:4d}  mean={stats["mean"]:.4f}  '
                  f'median={stats["median"]:.4f}  '
                  f'>0.65: {100 * stats["frac_over_0.65"]:.1f}%  '
                  f'>0.80: {100 * stats["frac_over_0.80"]:.1f}%')

    if args.out:
        aug.to_pickle(args.out)
        print(f'\nwrote {args.out}  shape={aug.shape}')
