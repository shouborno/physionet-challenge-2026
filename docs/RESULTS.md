# Official phase: results ledger

Team `bashlab_wpi`. Scores are age-conditioned AUROC at gap=2, the metric that
decides the Challenge, on the worst adequately powered leave-one-site-out fold
unless stated otherwise.

## The bar

| | age-conditioned AUROC |
|---|---|
| Leaderboard top five | 0.748 - 0.773 |
| Roughly top twenty | ~0.70 |
| **Our best so far** | **0.6553** |
| Age alone (floor) | 0.5401 |

157 submissions on the board.

**Published ceiling for this problem.** Ye et al., "Dementia detection from
brain activity during sleep" (SLEEP 2023), built ~1,071 hand-crafted features on
the same BDSP data lineage and reached 0.78 AUROC for dementia against normal,
age and sex matched, with linear models. The leaderboard's top five sit exactly
on that, so the leaders are plausibly reproducing that feature bank rather than
doing anything exotic. Their code is public.

## Progress

| change | worst powered fold | delta |
|---|---|---|
| LR C=0.005, 272 features (the shipped model) | 0.5937 | - |
| LightGBM, 402 features | 0.6066 | +0.013 |
| age withheld as a model input | 0.6311 | +0.025 |
| plus 550 coherence features (952 total) | **0.6553** | +0.024 |

Both real gains changed *what information the model has*. Everything that
rearranged the objective or the weighting landed within noise.

## The measurement floor

Subject-level bootstrap SE is **0.0389**. Resampling pairs instead understates
it 8.1x, so any pair-level interval is wrong by an order of magnitude.

Effective positive sample size is **62.1 of 84** (Kish), because the age
distribution is uneven. Differences below roughly 0.03 to 0.05 are not
resolvable, and a candidate list longer than five to ten is ordered by noise.

Each individual gain above is therefore at the edge of significance. What
supports the age result independently of its point estimate is the mechanism
check: score-age correlation fell from 0.329 to 0.136 and the plain-minus-
conditioned gap from 0.095 to 0.035.

## Fold power

| held-out fold | n | positives | age-matched pairs | usable |
|---|---|---|---|---|
| S0001 | 857 | 56 | 5,124 | yes |
| inverted (train S0001, test rest) | 246 | 28 | 791 | no |
| I0006 | 192 | 20 | 441 | no |
| I0002 | 54 | 8 | 48 | no |

One fold in four can support a decision, and it trains on only 246 records
because holding out S0001 leaves just the two small sites. The 6,600-record set
repairs this: that fold would train on 1,461 and the inverted fold on 5,139.

## The defining property of this task

| predictor | plain AUROC | age-conditioned |
|---|---|---|
| age alone | **0.7719** | **0.5375** |

Age alone would have won the unofficial phase, whose leaderboard topped out at
0.725 on plain AUROC. Only 11.8% of (positive, negative) pairs are age-matched
at gap=2. Prevalence runs 2% at ages 50-59 to 33% at 80-89.

## Falsified hypotheses

Six methods that rearrange the objective, all within noise of each other and
none beating plain LightGBM:

| method | worst fold | note |
|---|---|---|
| conditional logistic regression | 0.5824 | exact estimator for a matched design |
| matched-pair sample weights | 0.5818 | monotone harm: 0.6311, 0.6181, 0.5997, 0.5818 as strength rises |
| pairwise ranking, within-site | 0.5868 | see correction below |
| age residualization | -0.02 to -0.05 | consistent across three model families |
| site filtering | 0.5974 vs 0.6118 | removing site-predictive features costs accuracy |
| hyperparameter tuning | 0.6310 | +0.015 over search median, sd 0.0193 |

**Why residualization cannot work.** Measured on our own matrix, it perturbs
age-matched pair differences by 2.9% and unmatched ones by 15.7%. Within-stratum
AUC is translation-invariant, so subtracting a common function of age cannot
move the estimand; it only injects roughly 2,000 estimated nuisance parameters
of noise. Upside bounded at zero.

**Correction.** The pairwise ranker was recorded as cleanly failing a
pre-declared test. Its implementation takes one optimizer step per epoch, about
300 full-batch updates, and early-stops on training loss. An undertrained MLP
losing to a GBM on 84 positives says nothing about the objective. The
measurement stands; the interpretation does not.

**Correction.** I predicted coherence would be *less* site-identifying, since
dividing by both auto-spectra cancels per-channel gain. It is far more so: mean
0.7052 against 0.5890, with 66.4% above 0.65 against 16.9%. I0006 records
unipolar and its montages are derived here, so pairs sharing a reference
electrode carry correlated noise. Coherence cancels gain but not reference
montage. The features help anyway.

## Bugs found

- **The submission could not have scored.** `team_code.py` called
  `get_standardized_race`, deleted upstream in favour of `load_race`. The
  organizers substitute their own `helper_code.py`, so this was invisible
  locally and fatal remotely.
- **Six features silently dead.** Alpha peak parameters and aperiodic exponents
  were NaN on all 1,103 records: specparam 1.x attribute names against 2.0.0rc6,
  with the AttributeError swallowed by a broad except.
- **Stale `channel_table.csv`** missing the `-e1` aliases added upstream
  mid-phase, which look like the hidden sites' naming. ECG would not have been
  found there.
- **`load_bmi`** changed from 0.0 to NaN upstream. BMI is 75.9% missing.

## Open question worth one submission

Nothing calibrates our leave-one-site-out estimate against the leaderboard. Our
folds hold out entire sites that may differ from each other more than I0004
differs from the training pool, which would make the estimate pessimistic; in
the unofficial phase greedy selection made it optimistic by 0.136. One scored
entry resolves direction and magnitude, and ten are available.

## Entry log

| # | date | model | local worst fold | leaderboard | reward |
|---|---|---|---|---|---|
| - | - | none submitted yet | - | - | - |
