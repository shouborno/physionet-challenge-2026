# PhysioNet Challenge 2026: Screening for Cognitive Impairment During Sleep Studies

Entry by S. A. I. Shouborno and Bashima Islam, Worcester Polytechnic Institute.

## Approach

A rank blend of two models over 159 features, weighted 0.7 to TabFM and 0.3 to
LightGBM.

TabFM is Google's tabular foundation model, released June 2026 under Apache
2.0. It is used in context: the training set is carried through every forward
pass rather than distilled into weights. That is what makes it strong here, and
also why inference is batched, since the cost is nearly fixed per call rather
than per record. LightGBM is a plain gradient-boosted tree on the same
features. The two are combined by rank, because the metric reads only order and
the models are calibrated differently.

**Features** are 159 columns from three sources:

- 95 classical polysomnography features: CAISR annotation summaries, sleep
  architecture and stage transitions, HRV, EEG band powers by stage, slow-wave
  morphology, spindles, SpO2 and demographics.
- 63 temporal pooling features. Every classical feature collapses a night into
  one number, almost always a mean. These take high quantiles, dispersion,
  drift and excursion rates of per-epoch band power instead, so a night that
  alternates between normal and severely slowed is distinguishable from one
  that is uniformly mediocre. Adapted from the winning entry of the 2023
  I-CARE Challenge, the closest analogous problem.
- Recording year, discussed below.

Preprocessing is median imputation and standard scaling. Site is supplied to
TabFM as a covariate, with an unseen-site code at inference, which is the real
deployment condition.

## Recording date

The largest single effect in this model is not physiology. The label definition
requires at least six years of clean follow-up for a negative but only one to
six years to diagnosis for a positive, so negatives are systematically older
recordings: positives average 2015.2 against 2013.2, with follow-up 2.6 years
shorter. Recording year alone reaches 0.7103 age-conditioned AUROC. All 158
physiological features together add 0.039 on top of it.

`CreationTime` is supplied for every site including the hidden ones, and the
same labelling code generates their labels, so the artifact is expected to
carry. It is used deliberately and reported here rather than relied on quietly.
Any claim about sleep physiology from this work has to be stated net of it.

## What did not work

Recorded because the negatives were as expensive to establish as the positives.

- **Feature selection of any kind.** Seven methods across four mechanisms,
  ranking by gain, permutation importance, L1, Boruta-style shadow features,
  correlation pruning and variance filtering, all scored below simply using
  every feature collected. The ordering tracked how much was removed rather
  than how it was chosen. With 498 positives across three sites, a selector
  fitted on two sites appears to choose features that separate those two sites.
- **Inter-channel coherence**, 550 features across derivation pairs, bands and
  stages. Null raw and null compressed.
- **A sleep-EEG foundation model** trained on 36,000 recordings scored 0.5753
  alone against a 0.58 pre-registered gate, and lost 0.029 in combination. Age
  is recoverable from its latent at R-squared 0.998, so the embedding is close
  to a re-encoding of the one axis this metric values at zero.
- **Methods exploiting the metric's matched structure**, including pairwise
  ranking, conditional logistic regression over age strata, matched-pair
  weighting and age residualization. All within noise. Clemencon's result
  predicts this: the Bayes scorer is already optimal within every stratum.
- **Equal-site weighting**, which seemed obviously right when one site holds
  78% of the records, and cost 0.018.

## Evaluation

Leave-one-site-out over the training sites, plus an inverted fold that trains on
the dominant site alone. Models are ranked on the mean across adequately
powered folds rather than the worst fold: a controlled experiment showed the
worst fold moves by 0.030 under changes carrying no information at all, while
the mean moves by 0.0014. Every reported figure is the average of three seeds.

## Running

`python train_model.py -d <data> -m <model> -v` then
`python run_model.py -d <data> -m <model> -o <output> -v`.

A GPU is requested on the submission form. TabFM carries the whole training set
through each forward pass, which takes about 230 seconds per batch on an A30
and does not complete in useful time on CPU. If TabFM cannot be loaded the
entry falls back to LightGBM alone, which scores lower but still scores.
