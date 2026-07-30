#!/usr/bin/env python
"""Temporal pooling features: how bad the night gets, not just its average.

Every feature we have summarizes a whole night with a single number, usually a
mean over epochs. The winner of the closest analogous challenge (I-CARE 2023,
predicting neurological outcome from multi-hour post-arrest EEG) pooled
differently: they took the 0.88 or 0.89 *quantile* of each feature over
multi-hour blocks, plus a positional encoding of block time, and fed that to a
gradient-boosted tree. A high quantile is a soft maximum, capturing "the worst
part of the night" while staying robust to a single bad epoch in a way a true
maximum is not.

That is a different axis from anything here. Two nights can share a mean delta
power while one is uniformly mediocre and the other alternates between normal
and severely slowed, and only the second pattern is visible to a quantile or a
dispersion statistic. Cognitive impairment plausibly shows up as intermittent
disruption rather than a uniform shift, and a mean averages exactly that away.

Computed on per-epoch band powers from the best available EEG derivation, so it
needs one pass over the signal but no new detectors.
"""

from collections import OrderedDict

import numpy as np

BANDS = OrderedDict([
    ('delta', (0.5, 4.0)),
    ('theta', (4.0, 8.0)),
    ('alpha', (8.0, 12.0)),
    ('sigma', (12.0, 16.0)),
    ('beta', (16.0, 30.0)),
])

# CAISR: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake.
SLEEP_CODES = (1, 2, 3, 4)

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.88, 0.95)
EPOCH_SECONDS = 30.0
CHANNELS = ['c4-m1', 'c3-m2', 'f4-m1', 'f3-m2', 'o2-m1', 'o1-m2']


def _epoch_band_powers(signal, fs, n_epochs):
    """Relative band power per 30-second epoch. Returns (n_epochs, n_bands)."""
    from scipy.signal import welch

    per_epoch = int(round(EPOCH_SECONDS * fs))
    if per_epoch < 8 or n_epochs < 4:
        return None

    usable = min(n_epochs, len(signal) // per_epoch)
    if usable < 4:
        return None

    out = np.full((usable, len(BANDS)), np.nan)
    nperseg = min(per_epoch, int(4 * fs))
    for i in range(usable):
        seg = signal[i * per_epoch:(i + 1) * per_epoch]
        if len(seg) < nperseg or not np.isfinite(seg).all():
            continue
        freqs, psd = welch(seg, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        total = psd[(freqs >= 0.5) & (freqs <= 30.0)].sum()
        if total <= 0:
            continue
        for j, (lo, hi) in enumerate(BANDS.values()):
            out[i, j] = psd[(freqs >= lo) & (freqs < hi)].sum() / total
    return out


def extract_temporal_pooling_features(channels, fs_dict, stage_signal):
    """Quantiles, dispersion and drift of per-epoch band power across the night."""
    feat = OrderedDict()

    band_names = list(BANDS)
    placeholders = []
    for b in band_names:
        placeholders += [f'tp_{b}_q{int(q * 100)}' for q in QUANTILES]
        placeholders += [f'tp_{b}_iqr', f'tp_{b}_range80', f'tp_{b}_cv',
                         f'tp_{b}_drift', f'tp_{b}_worst_hour_frac',
                         f'tp_{b}_excursion_rate']

    channel = next((c for c in CHANNELS
                    if c in channels and channels[c] is not None
                    and len(channels[c]) > 0), None)
    if channel is None or stage_signal is None or len(stage_signal) == 0:
        return OrderedDict((k, np.nan) for k in placeholders)

    fs = fs_dict.get(channel)
    if not fs:
        return OrderedDict((k, np.nan) for k in placeholders)

    stages = np.asarray(stage_signal)
    powers = _epoch_band_powers(np.asarray(channels[channel], dtype=float),
                                float(fs), len(stages))
    if powers is None:
        return OrderedDict((k, np.nan) for k in placeholders)

    # Restrict to scored sleep: wake epochs are dominated by movement and eye
    # artifact, which would drive the high quantiles for the wrong reason.
    stages = stages[:len(powers)]
    keep = np.isin(stages, SLEEP_CODES)
    if keep.sum() < 20:
        return OrderedDict((k, np.nan) for k in placeholders)
    powers = powers[keep]

    n = len(powers)
    for j, band in enumerate(band_names):
        v = powers[:, j]
        v = v[np.isfinite(v)]
        if v.size < 20:
            for k in placeholders:
                if k.startswith(f'tp_{band}_'):
                    feat[k] = np.nan
            continue

        qs = np.quantile(v, QUANTILES)
        for q, val in zip(QUANTILES, qs):
            feat[f'tp_{band}_q{int(q * 100)}'] = float(val)

        median = float(np.median(v))
        feat[f'tp_{band}_iqr'] = float(np.quantile(v, 0.75) - np.quantile(v, 0.25))
        feat[f'tp_{band}_range80'] = float(np.quantile(v, 0.90)
                                           - np.quantile(v, 0.10))
        # Coefficient of variation: dispersion relative to level, so it is not
        # simply a restatement of the mean.
        feat[f'tp_{band}_cv'] = float(v.std() / median) if median > 1e-12 else np.nan

        # Drift across the night, first third against last third.
        third = max(1, len(v) // 3)
        feat[f'tp_{band}_drift'] = float(v[-third:].mean() - v[:third].mean())

        # Concentration: does the extreme sit in one stretch or scatter through
        # the night? A contiguous bad hour means something different from the
        # same epochs spread evenly.
        hi = v >= np.quantile(v, 0.88)
        window = max(2, int(3600 / EPOCH_SECONDS))
        if len(v) >= window:
            counts = np.convolve(hi.astype(float), np.ones(window), 'valid')
            feat[f'tp_{band}_worst_hour_frac'] = float(counts.max() / window)
        else:
            feat[f'tp_{band}_worst_hour_frac'] = float(hi.mean())

        # How often the signal crosses into its own tail, a rate rather than a
        # level, so it is insensitive to per-recording gain.
        feat[f'tp_{band}_excursion_rate'] = float(
            np.mean(np.diff(hi.astype(int)) == 1) * (3600 / EPOCH_SECONDS))

    # Cross-band ratios at the high quantile: the slowing signature evaluated
    # during the worst part of the night rather than on average.
    for num, den in (('delta', 'alpha'), ('theta', 'alpha'), ('delta', 'beta')):
        a = feat.get(f'tp_{num}_q88')
        b = feat.get(f'tp_{den}_q88')
        feat[f'tp_ratio_{num}_{den}_q88'] = (
            float(a / b) if a is not None and b not in (None, 0)
            and np.isfinite(a) and np.isfinite(b) and b > 1e-12 else np.nan)

    return feat
