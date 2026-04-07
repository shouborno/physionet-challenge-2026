# PhysioNet Challenge 2026: Screening for Cognitive Impairment During Sleep Studies

Entry by S. A. I. Shouborno and Bashima Islam, Worcester Polytechnic Institute.

## Approach

Single L2-regularized logistic regression (C=0.005) with 28 features selected via
greedy forward search optimizing worst-site Leave-One-Site-Out (LOSO) AUROC.
A domain-adversarial feature stability filter removes features that predict
recording site (AUC > 0.65) or whose coefficient sign is inconsistent across
LOSO folds, ensuring the model relies only on site-invariant biomarkers.

**Features** span six categories:
- CAISR annotations: arousal probability variability, stage probability entropy, N2 confidence
- Sleep architecture: stage transition probabilities (R→W, N3→W, N2→N3), transition persistence
- HRV: REM-specific RMSSD, LF/HF ratio, sample entropy
- EEG spectral: N3 relative theta, wake relative beta, REM alpha, overall alpha, REM theta/alpha ratio
- Slow-wave morphology: N3 slope, frequency, duration, negative-peak/PTP amplitude ratio (site-invariant)
- Nonlinear dynamics: DFA (wake), N3 kurtosis, SpO2 ODI, spindle density (N3)

Preprocessing: median imputation → standard scaling. All features extracted from
CAISR algorithmic annotations, raw EEG (spectral analysis, YASA slow-wave/spindle
detection, DFA via antropy), raw ECG (NeuroKit2 R-peak detection), SpO2, and demographics.
