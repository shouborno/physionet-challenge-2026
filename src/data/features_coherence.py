#!/usr/bin/env python
"""Inter-channel EEG coherence features, by channel pair, stage and band.

The largest identified gap against the published feature bank for this exact
problem. Ye et al., "Dementia detection from brain activity during sleep"
(SLEEP 2023), built ~1,071 features on the same BDSP data lineage and reached
0.78 AUROC for dementia against normal, age and sex matched, using only
hand-crafted features and linear models. Roughly 300 of those features are
inter-channel coherence per pair, stage and band. Our 272-feature bank has
none.

Two reasons this family is worth adding beyond simply closing the gap.

Coherence measures functional connectivity between cortical regions, which is
disrupted in cognitive impairment through a different mechanism than the
regional power changes the existing features capture. It is new information,
not a rearrangement of what is already there.

And it is scale-invariant by construction. Magnitude-squared coherence divides
the cross-spectrum by both auto-spectra, so any per-channel gain cancels
exactly. Amplifier gain, electrode impedance and recording scale, which are
the dominant site effects in multi-site PSG and which defeated ComBat and rank
normalization here, cannot move it. That matters especially given the finding
that the aperiodic exponent is heavily device-contaminated in multi-site data
(Purcell group, eNeuro 2022, where Compumedics amplifiers drove the wake
spectral slope from about -1 to about -8 on one derivation but not another).

Also computes the individual alpha frequency and the theta-alpha transition
frequency, both in the published bank and both absent here, and both robust to
gain because they are locations on the frequency axis rather than magnitudes.
"""

from collections import OrderedDict

import numpy as np

# CAISR stage encoding: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=unscored.
STAGE_CODES = {'wake': 5, 'n2': 2, 'n3': 1, 'rem': 4}

BANDS = OrderedDict([
    ('delta', (0.5, 4.0)),
    ('theta', (4.0, 8.0)),
    ('alpha', (8.0, 12.0)),
    ('sigma', (12.0, 16.0)),
    ('beta', (16.0, 30.0)),
])

# Alpha sub-bands, also in the published bank. The sub-band balance shifts with
# cognitive decline even when total alpha power does not.
ALPHA_SUBBANDS = OrderedDict([
    ('alpha1', (8.0, 9.5)),
    ('alpha2', (9.5, 11.0)),
    ('alpha3', (11.0, 12.5)),
])

CHANNELS = ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 'o1-m2', 'o2-m1']

# Cap the signal fed to each coherence estimate. Whole-night segments make the
# Welch estimate no more accurate, only slower.
MAX_SECONDS_PER_STAGE = 1800


def _stage_segments(signal, fs, stage_signal, stage_value, max_seconds):
    """Concatenate the parts of `signal` scored as `stage_value`."""
    if signal is None or stage_signal is None or len(stage_signal) == 0:
        return None
    n = len(signal)
    n_epochs = len(stage_signal)
    if n == 0 or n_epochs == 0:
        return None

    samples_per_epoch = n / n_epochs
    idx = np.flatnonzero(np.asarray(stage_signal) == stage_value)
    if idx.size == 0:
        return None

    pieces = []
    total = 0
    limit = int(max_seconds * fs)
    for i in idx:
        start = int(i * samples_per_epoch)
        end = min(int((i + 1) * samples_per_epoch), n)
        if end <= start:
            continue
        pieces.append(signal[start:end])
        total += end - start
        if total >= limit:
            break

    if not pieces:
        return None
    out = np.concatenate(pieces)
    return out if len(out) >= int(4 * fs) else None


def _band_coherence(x, y, fs):
    """Mean magnitude-squared coherence in each band, plus the alpha sub-bands."""
    from scipy.signal import coherence

    nperseg = int(min(4 * fs, len(x)))
    if nperseg < 8:
        return {}
    freqs, cxy = coherence(x, y, fs=fs, nperseg=nperseg,
                           noverlap=nperseg // 2)

    out = {}
    for name, (lo, hi) in list(BANDS.items()) + list(ALPHA_SUBBANDS.items()):
        mask = (freqs >= lo) & (freqs < hi)
        out[name] = float(np.mean(cxy[mask])) if mask.any() else np.nan
    return out


def extract_coherence_features(channels, fs_dict, stage_signal,
                               max_seconds=MAX_SECONDS_PER_STAGE):
    """Coherence for every available derivation pair, stage and band."""
    feat = OrderedDict()

    available = [c for c in CHANNELS
                 if c in channels and channels[c] is not None
                 and len(channels[c]) > 0]

    pairs = [(a, b) for i, a in enumerate(available) for b in available[i + 1:]]

    for stage_name, stage_code in STAGE_CODES.items():
        cache = {}
        for pair in pairs:
            fs = fs_dict.get(pair[0]) or fs_dict.get(pair[1])
            if not fs:
                continue

            segs = []
            usable = True
            for ch in pair:
                if ch not in cache:
                    cache[ch] = _stage_segments(
                        np.asarray(channels[ch], dtype=float), fs,
                        stage_signal, stage_code, max_seconds)
                if cache[ch] is None:
                    usable = False
                    break
                segs.append(cache[ch])

            short = f'{pair[0].split("-")[0]}_{pair[1].split("-")[0]}'
            if not usable:
                for band in list(BANDS) + list(ALPHA_SUBBANDS):
                    feat[f'coh_{short}_{band}_{stage_name}'] = np.nan
                continue

            n = min(len(segs[0]), len(segs[1]))
            try:
                values = _band_coherence(segs[0][:n], segs[1][:n], fs)
            except Exception:  # noqa: BLE001 - a bad pair must not lose the record
                values = {}
            for band in list(BANDS) + list(ALPHA_SUBBANDS):
                feat[f'coh_{short}_{band}_{stage_name}'] = values.get(band, np.nan)

    # Summaries across pairs: interhemispheric and anterior-posterior contrasts
    # are more interpretable and less noisy than any single pair.
    for stage_name in STAGE_CODES:
        for band in list(BANDS) + list(ALPHA_SUBBANDS):
            vals = [v for k, v in feat.items()
                    if k.endswith(f'_{band}_{stage_name}') and np.isfinite(v)]
            feat[f'coh_mean_{band}_{stage_name}'] = (
                float(np.mean(vals)) if vals else np.nan)
            feat[f'coh_std_{band}_{stage_name}'] = (
                float(np.std(vals)) if len(vals) > 1 else np.nan)

    return feat


def extract_alpha_frequency_features(channels, fs_dict, stage_signal,
                                     max_seconds=MAX_SECONDS_PER_STAGE):
    """Individual alpha frequency and the theta-alpha transition frequency.

    Both are positions on the frequency axis rather than magnitudes, so a
    per-channel gain cannot shift them. Alpha slowing is one of the most
    replicated EEG correlates of cognitive decline.
    """
    from scipy.signal import welch

    feat = OrderedDict()
    channel = next((c for c in CHANNELS
                    if c in channels and channels[c] is not None
                    and len(channels[c]) > 0), None)
    if channel is None:
        for stage_name in ('wake', 'rem'):
            feat[f'iaf_{stage_name}'] = np.nan
            feat[f'taf_{stage_name}'] = np.nan
            feat[f'alpha_ratio_hi_lo_{stage_name}'] = np.nan
        return feat

    fs = fs_dict.get(channel)
    signal = np.asarray(channels[channel], dtype=float)

    for stage_name in ('wake', 'rem'):
        seg = _stage_segments(signal, fs, stage_signal,
                              STAGE_CODES[stage_name], max_seconds) if fs else None
        iaf = taf = ratio = np.nan
        if seg is not None:
            nperseg = int(min(8 * fs, len(seg)))
            if nperseg >= 16:
                freqs, psd = welch(seg, fs=fs, nperseg=nperseg,
                                   noverlap=nperseg // 2)
                band = (freqs >= 6.0) & (freqs <= 14.0)
                if band.any():
                    f_b, p_b = freqs[band], psd[band]
                    # Individual alpha frequency: the spectral centroid over the
                    # extended alpha range, which is more stable than argmax.
                    if p_b.sum() > 0:
                        iaf = float(np.sum(f_b * p_b) / np.sum(p_b))
                    # Theta-alpha transition frequency: the spectral minimum
                    # separating the theta and alpha peaks.
                    window = (f_b >= 6.0) & (f_b <= 9.0)
                    if window.any():
                        taf = float(f_b[window][np.argmin(p_b[window])])
                lo = (freqs >= 8.0) & (freqs < 10.0)
                hi = (freqs >= 10.0) & (freqs < 12.5)
                if lo.any() and hi.any() and psd[lo].sum() > 0:
                    ratio = float(psd[hi].sum() / psd[lo].sum())

        feat[f'iaf_{stage_name}'] = iaf
        feat[f'taf_{stage_name}'] = taf
        feat[f'alpha_ratio_hi_lo_{stage_name}'] = ratio

    return feat


BIPOLAR = [('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
           ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
           ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1'])]


def standardize_channels(channels, fs_dict, csv_path):
    """Canonical channel names plus derived bipolar montages.

    Coherence needs named derivations, and I0006 records unipolar, so the
    montages have to be constructed rather than read.
    """
    import os

    from helper_code import (load_rename_rules,
                             standardize_channel_names_rename_only)

    rules = load_rename_rules(os.path.abspath(csv_path))
    rename_map, drop = standardize_channel_names_rename_only(
        list(channels.keys()), rules)

    std, std_fs = {}, {}
    for old in channels:
        if old in drop:
            continue
        new = rename_map.get(old, old.lower())
        std[new] = channels[old]
        if old in fs_dict:
            std_fs[new] = fs_dict[old]

    for target, pos, negs in BIPOLAR:
        if target in std or pos not in std:
            continue
        for neg in negs:
            if neg in std and len(std[neg]) == len(std[pos]):
                std[target] = (np.asarray(std[pos], dtype=float)
                               - np.asarray(std[neg], dtype=float))
                std_fs[target] = std_fs.get(pos)
                break
    return std, std_fs


def extract_all_connectivity_features(channels, fs_dict, stage_signal):
    feat = OrderedDict()
    feat.update(extract_coherence_features(channels, fs_dict, stage_signal))
    feat.update(extract_alpha_frequency_features(channels, fs_dict, stage_signal))
    return feat
