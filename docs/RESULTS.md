# Official phase: results ledger

Team `bashlab_wpi`. Age-conditioned AUROC at gap=2 on the full 6,600-record
training set, reported as the **mean across adequately powered
leave-one-site-out folds**, averaged over three seeds.

## How things are measured here, and why it changed twice

Most of this project ranked models on the worst powered fold. That was wrong. A
controlled experiment varied two things carrying no information at all, a
leaked timing column and a StandardScaler in front of a tree model, and moved
the worst fold by 0.030 while the mean moved by 0.0014. Bootstrap intervals do
not capture this: they measure resampling variance, not pipeline sensitivity.

The mean is also the better estimator for this competition, since the hidden
set is a single unseen site.

Single-seed measurement then caused a second round of errors, so every figure
below is the mean of three seeds. Seed spread is about 0.0006 for TabFM and the
blend, 0.0007 for LightGBM, and up to 0.0058 for LightGBM on some feature sets.
**Differences under 0.005 are not read as real.**

## The bar

| | age-conditioned AUROC |
|---|---|
| Leaderboard top five | 0.748 - 0.773 |
| Roughly top twenty | ~0.70 |
| **Submission candidate** | **0.7723 +/- 0.0005** |
| Recording year alone | 0.7103 |
| Age alone | 0.5308 |

## Models, all on the 159 shipped features

| model | mean | worst fold |
|---|---|---|
| TabFM + site covariate | 0.7737 +/- 0.0006 | 0.6788 |
| **blend, 0.7 TabFM / 0.3 LightGBM** | **0.7723 +/- 0.0005** | **0.6812** |
| blend 50/50 | 0.7674 | 0.6824 |
| LightGBM, unbalanced | 0.7487 +/- 0.0007 | 0.6634 |
| LR C=0.005 (the previously shipped model) | 0.6769 | - |

TabFM and the blend are tied: 0.0014 apart against a seed spread of 0.0006, and
the blend holds the better worst fold. The blend ships because a single unseen
site makes worst-case behaviour matter, and because it still predicts if TabFM
fails to load in the container.

## What moves the number

| change | delta | note |
|---|---|---|
| recording year | **+0.084** | label artifact, not physiology |
| TabFM over LightGBM | +0.025 | needs a GPU |
| temporal pooling block | +0.018 | strongest physiological block |
| no site weighting | +0.018 | removal of something added earlier |
| drop BMI | +0.012 | missingness is a site-specific care proxy |
| site covariate, with TabFM | +0.006 | helps TabFM, hurts unbalanced LightGBM |
| coherence, 550 features | -0.006 | null raw, null compressed |
| bytecode-only block, 307 features | -0.011 | cannot ship, and does not help |

## Recording date

Recording year alone reaches 0.7103, within 0.04 of the full model and enough
for roughly top twenty. All 158 physiological features together add 0.039 on
top of it. Removing it costs 0.084, four times any other effect.

The label definition requires six or more years of clean follow-up for a
negative but only one to six years to diagnosis for a positive, so negatives are
systematically older recordings: positives average 2015.2 against 2013.2, with
follow-up 2.6 years shorter. `CreationTime` ships for the hidden sites and the
same labelling code generates their labels, so the mechanism should carry. Used
deliberately and reported; any physiological claim must be stated net of it.

## Feature selection does not work here

Seven methods, nine configurations, four mechanisms. All lose to using every
feature collected.

| method | mean |
|---|---|
| none, all 1,015 | **0.7456** |
| decorrelate r>0.98 | 0.7438 |
| permutation importance, top 400 | 0.7408 |
| L1, C=0.1 | 0.7399 |
| decorrelate r>0.95, top 400 | 0.7399 |
| variance prune 25% | 0.7329 |
| permutation importance, top 200 | 0.7247 |
| Boruta-style shadow | 0.7203 |
| L1, C=0.01 | 0.7195 |

The ordering tracks how much was removed, not how it was chosen: milder cuts
lose less, aggressive cuts lose more. Selection is removing signal, not noise.
With 498 positives across three sites, a selector fitted on two sites appears to
pick features separating those two sites, and nesting cannot fix that. Nesting
prevents optimistic scoring; it does not make the choice transfer.

This is a second, independent explanation for the unofficial phase collapsing
from 0.780 to 0.644, alongside selecting on the evaluation statistic.

What does help is deciding not to collect a block at all, which is a judgement
about mechanism rather than a data-driven cut.

## Corrections

Ten claims in this repo were wrong. All but the last two came from ranking on
the worst fold or from a single seed.

1. "Dropping age gains +0.025." It is -0.002.
2. "Coherence gains +0.024." It is -0.006.
3. "Site handling is settled; neither direction helps." Removing site weighting
   is worth +0.018.
4. "Site balancing is most of what group DRO buys, at lower risk." It cost
   0.018 across nearly every experiment here.
5. "Temporal pooling is a null." It is the strongest physiological block:
   0.7429 with recording year against 0.7248 for the 95 classical features.
   The null came from adding 63 good columns to a 952-feature matrix that was
   mostly noise.
6. "TabFM cannot ship, CPU inference is too slow." The measurement ran
   single-threaded under a wrapper that pins OMP_NUM_THREADS=1. It is slow on
   CPU for a different reason, and the fix is batching plus a GPU.
7. "Family-wise compression reaches 0.7881, the best result here." Single seed,
   and no block reproduces it: coherence compressed gives 0.7694, bytecode
   0.7598, self-referential 0.7595, all below the 0.7737 baseline.
8. "Compression rescued coherence." Asserted from one number without running
   the test that would distinguish causes. The 709-feature shippable set
   contains all coherence compressed and still loses to 159 without it.
9. The pairwise ranker was recorded as cleanly failing a pre-declared test. It
   used roughly 300 full-batch updates and early-stopped on training loss, so
   the result says nothing about the objective.
10. Coherence was predicted to be less site-identifying because dividing by both
    auto-spectra cancels channel gain. It is far more so, 0.7052 against 0.5890:
    gain cancels but reference montage does not.

## Settled negatives

Philosopher's Stone fails both pre-registered gates: 0.5753 alone against a
0.58 gate, and -0.029 combined. Age is recoverable from its 1024-d latent at
R^2 = 0.998, so the embedding is close to a re-encoding of the one axis this
metric values at zero. Site decodes from it at 0.9175 against 0.9637 from our
hand features, so the problem is the age axis, not site encoding.

Age residualization, matched-pair weighting, conditional logistic regression
and the additive-age sweep all land within noise. Three protocols agree that
methods exploiting the metric's matched structure do not beat a plain
classifier, as Clemencon's result predicts.

## Deployment

GPU on request: A30 or RTX 6000 Ada, 72h training, 48h inference, 16 vCPU, 60
GiB. TabFM is Apache 2.0, 6.2 GB of weights, baked into the image at build time.

In-context inference carries the whole training set through every forward pass,
so cost is nearly fixed per call: 230 s for one prediction, 270 s for a
thousand. One at a time that is 64 hours against a 48-hour limit; batched it is
four and a half minutes. `run_model` therefore scores the whole folder on its
first call and serves the rest from cache. This is ordinary batching, not a
transductive trick: no test-set statistic enters the model and the predictions
are identical.

This cluster has no docker, podman or apptainer, so the image cannot be built
or tested here. A clean-virtualenv install of the pinned requirements plus the
weight-bake step is the closest available proxy.

## Data

6,600 records, 7.55% prevalence, 498 positives. S0001 5,139 (6.5%), I0006 1,142
(9.8%), I0002 319 (16.3%). Age alone scores 0.7719 plain against 0.5375
conditioned, and would have won the unofficial phase, whose leaderboard topped
out at 0.725 on plain AUROC. Only 11.8% of pairs are age-matched at gap=2.

## Bugs found

- **The previous submission could not have scored.** `team_code.py` called
  `get_standardized_race`, deleted upstream. The organizers substitute their own
  `helper_code.py`, so this was invisible locally and fatal remotely.
- **Six features silently dead**: specparam 1.x attribute names against 2.0.0rc6,
  with the AttributeError swallowed by a broad except.
- **`coh_time_sec` leaked into a feature list**, worth +0.013 on the worst fold.
  Instrumentation is now excluded by suffix and asserted against.
- **Stale `channel_table.csv`** missing the `-e1` aliases added upstream.
- **`load_bmi`** changed from 0.0 to NaN upstream; BMI is 75.9% missing.

## Entry log

| # | date | model | local mean | leaderboard | reward |
|---|---|---|---|---|---|
| - | - | none submitted yet | - | - | - |

Nothing calibrates the local estimate against the leaderboard. In the
unofficial phase the equivalent gap was 0.136 in the optimistic direction. Ten
entries remain and the phase closes in late August.
