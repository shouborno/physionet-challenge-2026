# Official phase: results ledger

Team `bashlab_wpi`. All scores are age-conditioned AUROC at gap=2, the metric
that decides the Challenge, on the worst adequately powered leave-one-site-out
fold unless stated otherwise.

## Where the bar is

| | age-conditioned AUROC |
|---|---|
| Leaderboard top five | 0.748 - 0.773 |
| Roughly top twenty | ~0.70 |
| Our target | 0.75 |
| Submissions on the board | 157 |

## Data

Official small training set: 1,103 records, **7.62% prevalence** (84 positive),
across S0001 (857, 6.5%), I0006 (192, 10.4%) and I0002 (54, 14.8%). The large
set adds up to 6,600 records and is downloading.

The single most important property of this task:

| predictor | plain AUROC | age-conditioned |
|---|---|---|
| age alone | **0.7719** | **0.5375** |

Age alone would have won the unofficial phase, whose leaderboard topped out at
0.725 on plain AUROC. The new metric removes almost all of that, which is
plainly why it was changed. Any model that ranks partly on age is spending
capacity on an axis now worth nothing.

Only 11.8% of (positive, negative) pairs are age-matched at gap=2, so the
effective sample size is far below the record count.

## Fold power, and why most results cannot be trusted

| held-out fold | n | positives | age-matched pairs | usable |
|---|---|---|---|---|
| S0001 | 857 | 56 | 5,124 | yes |
| inverted (train S0001, test rest) | 246 | 28 | 791 | no |
| I0006 | 192 | 20 | 441 | no |
| I0002 | 54 | 8 | **48** | no |

One fold in four can support a decision. Results are therefore reported with
pair counts and bootstrap intervals, ranked on the lower bound of the worst
powered fold, and folds under 1,000 pairs are shown but never selected on.

**The powered fold trains on only 246 records.** Holding out S0001 leaves just
the two small sites, so 0.61 there is a data-starved estimate rather than an
expected leaderboard score. Folds trained on 857 records reach 0.607-0.636
cross-site and 0.703 within-site, so the realistic figure is nearer 0.62-0.65.

## What has been measured

| approach | worst powered fold | note |
|---|---|---|
| LightGBM, 402 features | **0.6118** | best so far |
| LightGBM, 272 features | 0.6044 | |
| all features, site-filtered | 0.5974 | filtering **hurts** |
| LR C=0.005 (the shipped model) | 0.5937 | |
| pairwise ranker, within-site | 0.5868 | failed its pre-declared test |
| pairwise ranker, cross-site | 0.5796 | ablation of the above |
| self-referential features only | 0.5773 | |
| age alone | 0.5401 | floor |

Hand-crafted features are saturated near **0.61**. Twelve model and feature
variations move it by less than 0.01 in total.

### Falsified, and stated as such

**The age-matched pairwise ranking objective.** Declared before running: beat
the plain classifier by 0.02 on the powered fold's lower bound or drop it. It
came in 0.0004 *below*. Its within-site constraint, the whole
domain-generalization argument, beat the cross-site ablation by 0.007, well
inside noise. It may have been data-starved at 84 positives and gets one
re-test on the large set, but it has no measured advantage.

**Age residualization.** Consistently harmful: -0.054 for LightGBM, -0.030 for
the LR, -0.018 for the pairwise ranker. Three model families, same direction.
The age-dependent part of these features evidently carries within-band
information rather than pure confound.

**Site filtering.** Per-feature site-predictiveness fell sharply with the
self-referential features (mean 0.6024 to 0.5610; the fraction above 0.65 from
22.8% to 4.6%; nothing above 0.80 against 1.8% before), yet cross-site accuracy
did **not** improve, and explicitly dropping site-predictive features made it
worse. Reducing site leakage is not sufficient here, which also explains why
ComBat (0.743 to 0.492) and rank normalization (to 0.634) failed in the
unofficial phase.

## Bugs found

- **The submission could not have scored.** `team_code.py` called
  `get_standardized_race`, deleted upstream in favour of `load_race`. Since the
  organizers substitute their own `helper_code.py`, this was invisible locally
  and fatal remotely. Reproduced, then fixed.
- **Six features were silently dead.** All four alpha peak parameters and three
  aperiodic exponents were NaN on all 1,103 records: the extractor reads
  specparam 1.x attribute names and the installed version is 2.0.0rc6, with the
  AttributeError swallowed by a broad except. Fixed; it changed no scores.
- **Stale `channel_table.csv`** was missing the `-e1` aliases upstream added
  mid-phase, which look like the hidden sites' naming. ECG would not have been
  found there, quietly NaN-ing four shipped features on validation and test only.
- **`load_bmi` semantics changed** from 0.0 to NaN. BMI is 75.9% missing, so the
  old default put a nonsense value in three quarters of the column.

## Open question worth one submission

Nothing calibrates our leave-one-site-out estimate against the leaderboard. Our
folds hold out entire sites that may differ from each other far more than
I0004 differs from the training pool, which would make the estimate pessimistic;
in the unofficial phase, by contrast, greedy selection made it optimistic by
0.136. One scored entry resolves the direction and magnitude, and ten are
available. Until then every number here is uncalibrated.

## Entry log

| # | date | model | local worst fold | leaderboard | reward |
|---|---|---|---|---|---|
| - | - | none submitted yet | - | - | - |
