# PhysioNet Challenge 2026: Screening for Cognitive Impairment During Sleep Studies

Entry by S. A. I. Shouborno and Bashima Islam, University of Massachusetts Amherst.

## Approach

A rank blend of two models over 151 features, weighted 0.7 to TabFM and 0.3 to
LightGBM, with the training features adapted to the target site before either
model is fitted.

TabFM is Google's tabular foundation model, released June 2026 under Apache
2.0. It is used in context: the training set is carried through every forward
pass rather than distilled into weights. That is what makes it strong here, and
also why inference is batched, since the cost is nearly fixed per call rather
than per record. LightGBM is a plain gradient-boosted tree on the same
features. The two are combined by rank, because the metric reads only order and
the models are calibrated differently.

**Features** are 151 columns from three sources:

- 86 classical polysomnography features: CAISR annotation summaries, sleep
  architecture and stage transitions, HRV, EEG band powers by stage, SpO2 and
  demographics.
- 63 temporal pooling features. Every classical feature collapses a night into
  one number, almost always a mean. These take high quantiles, dispersion,
  drift and excursion rates of per-epoch band power instead, so a night that
  alternates between normal and severely slowed is distinguishable from one
  that is uniformly mediocre. Adapted from the winning entry of the 2023
  I-CARE Challenge, the closest analogous problem.
- Recording year and its position within the site's collection window,
  discussed below. The hidden sets carried no dates, so both were empty at
  scoring.

Preprocessing is median imputation and standard scaling. Site is supplied to
TabFM as a covariate, with an unseen-site code at inference, which is the real
deployment condition.

## Domain adaptation

The test set is a single hospital never seen in training, and its recordings
are available unlabelled before any prediction is made. CORAL (Sun and Saenko)
whitens the training covariance and recolours it to the target's, so
second-order structure is matched without any label. Both models are then
fitted on the adapted features at inference.

This is transductive: training data is transformed using a statistic of the set
being scored. Nothing in the Challenge rules restricts it and `run_model`
receives the whole data folder, but it is a stronger use of the test set than
batching for speed and is stated here rather than buried.

With recording date among the features, CORAL raised the worst training site,
S0001, by 0.020 in leave-one-site-out validation and left the mean unchanged.
Without the date it does not help. It moves the mean by 0.004, lowers S0001 by
0.016, and with the held-out site's dates blanked, the condition the hidden sets
imposed, it costs 0.019 on the mean. It stays in the code because it is part of
the scored entry.

Adaptation is skipped below 200 target records, where a 151x151 covariance
cannot be estimated well. With the date present, three other adaptation methods
were tried: target-mean scaling, importance weighting by an estimated density
ratio, and pseudo-labelling target records into TabFM's context. All landed
within noise or below. The Bures-Wasserstein map, which is the theoretically
better-motivated transport, measured worse.

## Recording date

The largest single effect in the training data is not physiology. The label
definition requires at least six years of clean follow-up for a negative but
only one to six years to diagnosis for a positive, so no negative can be recent.
From 2020 onward the training set holds 38 positives and no negatives, and
negative recordings are older by a median of 3.3 years. Recording date alone
reaches 0.723 age-conditioned AUROC across the leave-one-site-out folds, more
where a site's collection window is short, 0.811 at I0002, which spans 6.7
years, against 0.660 at S0001, which spans 14.7.

The organizers withheld recording dates from the hidden validation and test
sets, judging them not to generalise, and rescored every entry without them.
`_add_date_features` sets both date features to NaN when `CreationTime` is
missing, and they are then imputed to the training median, so the scored model
had no date information.

Cross-validated the same way, with the held-out site's dates blanked and
imputed, the submitted pipeline averages 0.687 (0.701 at I0002, 0.730 at I0006,
0.629 at S0001) against 0.678 on the hidden test set. With the date available it
averages 0.786, so the date inflated site-held-out validation by 0.099. Refitted
without the date, the temporal pooling block adds 0.035 to the tree and the
TabFM blend a further 0.032.

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
- **Slow-wave and spindle morphology.** Removing these nine columns is worth
  0.005. YASA's detectors fail frequently on this data, so the columns encode
  whether the detector fired as much as they encode morphology, and detector
  failure tracks amplifier and montage rather than physiology.
- **A sleep-EEG foundation model** trained on 36,000 recordings scored 0.5753
  alone against a 0.58 pre-registered gate, and lost 0.029 in combination. Age
  is recoverable from its latent at R-squared 0.998, so the embedding is close
  to a re-encoding of the one axis this metric values at zero.
- **Methods exploiting the metric's matched structure**, including pairwise
  ranking, conditional logistic regression over age strata, matched-pair
  weighting and age residualization. All within noise. Clemencon's result
  predicts this: the Bayes scorer is already optimal within every stratum, and
  Meisner et al. report the same null for direct maximization of exactly this
  objective.
- **Time to diagnosis as a graded target.** All 498 positives carry days to
  diagnosis, from 374 to 2191. Weighting positives by imminence, regressing on
  a graded target, and weighting negatives by follow-up length all land within
  0.002 of the plain binary label.
- **Equal-site weighting**, which seemed obviously right when one site holds
  78% of the records, and cost 0.018.

## Evaluation

Leave-one-site-out over the training sites, plus an inverted fold that trains on
the dominant site alone. Models are ranked on the mean across adequately
powered folds rather than the worst fold: a controlled experiment showed the
worst fold moves by 0.030 under changes carrying no information at all, while
the mean moves by 0.0014. Reported figures are averages of three seeds, except
those in the Domain adaptation and Recording date sections, which are single
runs.

The metric keeps only the 11 to 14 percent of pairs that fall within the
two-year age caliper, and sampled metrics of this kind are known not to
preserve relative orderings even in expectation, so differences below a few
hundredths should be read as unresolved rather than real.

## Running

`python train_model.py -d <data> -m <model> -v` then
`python run_model.py -d <data> -m <model> -o <output> -v`.

A GPU is requested on the submission form. TabFM carries the whole training set
through each forward pass, which takes about 230 seconds per batch on an A30
and does not complete in useful time on CPU. If TabFM cannot be loaded the
entry falls back to LightGBM alone, which scores lower but still scores.
