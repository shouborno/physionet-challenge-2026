# PhysioNet Challenge 2026: Screening for Cognitive Impairment During Sleep Studies

Entry by S. A. I. Shouborno and Bashima Islam, Worcester Polytechnic Institute.

## Approach

Dual-model probability-average ensemble of two logistic regression classifiers trained on curated sleep features extracted from PSG recordings and CAISR algorithmic annotations.

**Model A** (31 features, "overall maximizer"): Greedy forward-selected features maximizing mean LOSO AUROC. Includes sleep architecture (stage percentages, bout statistics, transition probabilities), CAISR softmax probability statistics, HRV (RMSSD), SpO2 statistics, EEG Hjorth mobility and kurtosis, slow wave density, and SO-spindle coupling.

**Model B** (12 features, "site-robust hedger"): Greedy forward-selected features maximizing worst-case site AUROC. Emphasizes features with lower site variance: arousal probability variability, HRV pNN50, EEG nonlinear dynamics (DFA), slow wave density, and SO-spindle coupling.

**Ensemble**: P(CI) = 0.6 × P_A + 0.4 × P_B. The weighting balances overall accuracy with robustness to unseen sites.

Both models use L2-regularized logistic regression (C=0.005) with median imputation and standard scaling. Features are extracted from CAISR algorithmic annotations, raw EEG (via spectral analysis, YASA slow wave/spindle detection, and DFA via antropy), raw ECG (via NeuroKit2 R-peak detection), SpO2, and demographics.
