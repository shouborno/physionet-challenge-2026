# Official phase: results ledger

Team `bashlab_wpi`. Scores are age-conditioned AUROC at gap=2, on the full
6,600-record training set, reported as the **mean across adequately powered
leave-one-site-out folds** unless stated otherwise.

## Why the mean, and not the worst fold

Most of this session ranked models on the worst powered fold. That was a
mistake, and a controlled experiment shows why. Two changes that carry no
information at all — one leaked timing column (`coh_time_sec`, which tracks
recording length and site) and a StandardScaler in front of a tree model —
moved the worst fold from 0.6254 to 0.6556. Across the same four
configurations the mean moved from 0.7204 to 0.7218.

| statistic | range under pure noise |
|---|---|
| worst powered fold | **0.030** |
| mean across powered folds | **0.0014** |

Bootstrap intervals do not capture this: they measure resampling variance, not
pipeline sensitivity. Every effect reported earlier at +0.011 to +0.032 was
inside the noise of the statistic used to measure it.

The mean is also the better estimator for this competition. The hidden set is a
single unseen site, so expected performance on a random held-out site is the
quantity of interest, not a pessimistic floor.

## The bar

| | age-conditioned AUROC |
|---|---|
| Leaderboard top five | 0.748 - 0.773 |
| Roughly top twenty | ~0.70 |
| **Our best configuration** | **0.7404** |
| Age alone (floor) | 0.5308 |

## What actually works

Unbalanced LightGBM, BMI withheld, recording year included, 952 features,
6,600 records.

| change | delta on mean | verdict |
|---|---|---|
| recording year | **+0.090** | real, and dominant |
| no site weighting | **+0.018** | real |
| drop BMI | **+0.012** | real |
| coherence features | +0.002 | noise |
| age as an input | -0.002 | noise |
| site as a covariate | +0.014 vs balanced, -0.004 vs unbalanced | substitutes for the weighting, does not compose |

The dominant effect is five times larger than anything else and is not
physiology. The label definition requires six or more years of clean follow-up
for a negative, so negatives must be older recordings: positives average 2015.2
against 2013.2, with follow-up 2.6 years shorter. Removing recording year costs
0.090. This is legitimate under the rules and `CreationTime` is supplied for
the hidden sites, but any claim about sleep physiology has to be made net of it.

## Corrections to earlier conclusions in this repo

Four claims were reported as findings and are wrong. All were measured on the
1,103-record set using the worst-fold statistic.

1. **"Dropping age gains +0.025."** On 6,600 records it is -0.002. The
   supporting diagnostics were real (score-age correlation fell 0.329 to 0.136)
   but did not imply the performance claim.
2. **"Coherence gains +0.024."** It is +0.002. A mechanism was written for an
   effect that did not exist.
3. **"Site handling is settled in both directions; neither helps."** Removing
   site weighting is the single best model-level change found (+0.018), and
   site as a covariate also helps relative to the balanced default.
4. **"Site balancing is most of what group DRO buys, at lower risk."** It was
   applied to nearly every model this session and cost 0.018 throughout.

Two further corrections concern interpretation rather than numbers:

- The pairwise ranker was recorded as cleanly failing a pre-declared test. Its
  implementation used roughly 300 full-batch updates and early-stopped on
  training loss, so the result says nothing about the objective.
- Coherence was predicted to be *less* site-identifying because dividing by
  both auto-spectra cancels channel gain. It is far more so, 0.7052 against
  0.5890, because gain cancels but reference montage does not, and I0006's
  montages are derived rather than native.

## Models

| model | mean | note |
|---|---|---|
| LightGBM, unbalanced | **0.7404** | current best |
| LightGBM, balanced (old default) | 0.7227 | |
| TabFM, n_estimators=8 | 0.7479 | measured against the handicapped baseline |
| LR C=0.001 | 0.6781 | |
| LR C=0.005 (the shipped model) | 0.6769 | |
| LR C=0.05 | 0.6557 | |

TabFM (Google, released 2026-06-30, Apache 2.0) is the only model to beat
LightGBM, but every TabFM comparison so far used the site-balanced baseline,
which is worth 0.018 on its own. A corrected comparison at full strength, with
blends, is in progress. On CPU it loads in 17s and fits 6,600 rows in 11s;
per-record in-context inference cost is the open question for deployability.

## Settled negatives

Philosopher's Stone fails both pre-registered gates: G1 wanted 0.58 alone and
got 0.5753, G2 wanted +0.02 combined and lost 0.029. Age is recoverable from
its 1024-d latent at R^2 = 0.998, so the embedding is close to a re-encoding of
the one axis this metric values at zero. Contradicting the published
explanation for such failures, site decodes from the latent at 0.9175 against
0.9637 from our own hand features, so here the problem is the age axis, not
site encoding.

TabPFN v2 with PCA to 200 components reached 0.5931 against LightGBM's 0.6786
on the worst fold, though the PCA confound means this is not a clean test.
TabPFN v3 is now licensed and loads; untested.

Age residualization, matched-pair weighting, conditional logistic regression
and the additive-age lambda sweep all land within noise or below on the mean.
Three independent protocols agree that methods exploiting the metric's matched
structure do not beat a plain classifier, which Clemencon's result predicts
since the Bayes scorer is already optimal within every stratum.

## Data

6,600 records, 7.55% prevalence, 498 positives. S0001 5,139 (6.5%), I0006 1,142
(9.8%), I0002 319 (16.3%). All four folds are adequately powered on the large
set; on the small set only one was.

| predictor | plain AUROC | age-conditioned |
|---|---|---|
| age alone | 0.7719 | 0.5375 |

Age alone would have won the unofficial phase, whose leaderboard topped out at
0.725 on plain AUROC. Only 11.8% of (positive, negative) pairs are age-matched
at gap=2.

## Bugs found

- **The submission could not have scored.** `team_code.py` called
  `get_standardized_race`, deleted upstream in favour of `load_race`. The
  organizers substitute their own `helper_code.py`, so this was invisible
  locally and fatal remotely.
- **Six features silently dead**: specparam 1.x attribute names against
  2.0.0rc6, with the AttributeError swallowed by a broad except.
- **Stale `channel_table.csv`** missing the `-e1` aliases added upstream
  mid-phase, which look like the hidden sites' naming.
- **`load_bmi`** changed from 0.0 to NaN upstream; BMI is 75.9% missing.
- **`coh_time_sec` leaked into a feature list**, worth +0.013 on the worst
  fold. Instrumentation is now excluded by suffix and asserted against.

## Entry log

| # | date | model | local mean | leaderboard | reward |
|---|---|---|---|---|---|
| - | - | none submitted yet | - | - | - |

Nothing calibrates the local estimate against the leaderboard. Ten entries
remain and the phase closes in late August.
