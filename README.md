# PhysioNet Challenge 2026: Screening for Cognitive Impairment During Sleep Studies

Entry by S. A. I. Shouborno and Bashima Islam, Worcester Polytechnic Institute.

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
  discussed below.

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

It is worth 0.013 in leave-one-site-out validation, and should be worth more in
deployment. The supplementary recordings put the median standardized mean
difference from the hidden site to the training sites at 0.399, against 0.136
to 0.247 between the training sites themselves, so validation exercises
adaptation across roughly half the real gap. Widening the gap synthetically,
CORAL's advantage grows from 0.008 to 0.040 while the unadapted model degrades.

Adaptation is skipped below 200 target records, where a 151x151 covariance
cannot be estimated well enough to help. Of four adaptation methods tried, only
CORAL helped: target-mean scaling, importance weighting by an estimated density
ratio, and pseudo-labelling target records into TabFM's context all landed
within noise or below. The Bures-Wasserstein map, which is the theoretically
better-motivated transport, measured worse.

## Recording date

The largest single effect in this model is not physiology. The label definition
requires at least six years of clean follow-up for a negative but only one to
six years to diagnosis for a positive, so negatives are systematically older
recordings: positives average 2015.2 against 2013.2, with follow-up 2.6 years
shorter. Recording year alone reaches 0.7103 age-conditioned AUROC. All the
physiological features together add 0.039 on top of it.

The model also uses the recording's position within its own site's collection
window, which is worth a further 0.018. The same calendar year sits early in a
site collecting from 2007 to 2022 and late in one collecting from 2011 to 2020,
and the artifact runs through position in the window rather than through the
calendar, so the relative form transfers where the absolute one does not.

`CreationTime` is supplied for every site including the hidden ones, and the
same labelling code generates their labels, so the artifact is expected to
carry. Held out, it does: trained on two sites and scored on the third it
reaches 0.827, 0.744 and 0.617, tracking each site's prevalence. It is used
deliberately and reported here rather than relied on quietly. Any claim about
sleep physiology from this work has to be stated net of it.

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
the mean moves by 0.0014. Every reported figure is the average of three seeds.

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
