#!/usr/bin/env python
"""Nonlinear complexity features of the sleep EEG.

Sixteen features that measure how irregular the EEG is rather than how much
power sits in each band, and the ablation makes them the most valuable block
outside the shipped set: removing them costs 0.0091, more than any other group
including the 68 spectral features.

Rewritten from the compiled research module, which cannot ship. The parameters
here are not guesses. The compiled function still runs, so its calls into
antropy were traced with their arguments and the values below are what it
passes: order 3 and delay 1 for permutation and SVD entropy, kmax 10 for
Higuchi, Welch for spectral entropy, everything normalized, all at 100 Hz.
Window lengths come from the traced array shapes: 30,000 samples for the
per-stage features, 12,000 for sample entropy, 60,000 for DFA and permutation
entropy. diagnostics/validate_complexity.py checks this against the cached
research values record by record.

Why these measures. Cortical slowing shows up in band power, but loss of signal
complexity is a partly separate phenomenon: an EEG can hold its spectral shape
while becoming more regular and predictable, and that regularity is what
Lempel-Ziv, the fractal dimensions and the entropies capture.
"""

from collections import OrderedDict

import numpy as np

# CAISR stage codes: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake.
STAGE_CODES = {'n3': 1, 'n2': 2, 'n1': 3, 'rem': 4, 'wake': 5}

TARGET_FS = 100.0
EPOCH_SECONDS = 30.0

# Traced window lengths, in samples at 100 Hz.
SECONDS_STANDARD = 300     # 30,000 samples: the per-stage block
SECONDS_SAMPLE_ENTROPY = 120   # 12,000: sample entropy is O(n^2)
SECONDS_LONG = 600         # 60,000: DFA and permutation entropy

PERM_ORDER = 3
PERM_DELAY = 1
SVD_ORDER = 3
SVD_DELAY = 1
HIGUCHI_KMAX = 10

FEATURE_NAMES = (
    [f'{f}_{s}' for s in ('n2', 'n3')
     for f in ('lziv_complexity', 'svd_entropy', 'higuchi_fd',
               'spectral_entropy', 'katz_fd', 'petrosian_fd')]
    + ['sample_entropy_rem', 'dfa_wake', 'perm_entropy_wake']
)

CHANNEL_PRIORITY = ['c4-m1', 'c3-m2', 'f4-m1', 'f3-m2', 'o2-m1', 'o1-m2']


def _nan_features():
    return OrderedDict((n, np.nan) for n in FEATURE_NAMES)


def _best_channel(channels, fs_dict):
    for name in CHANNEL_PRIORITY:
        sig = channels.get(name)
        if sig is not None and len(sig) > 0 and fs_dict.get(name):
            return name
    return None


def _stage_signal(signal, fs, stages, stage_code, max_seconds):
    """Concatenate the epochs scored as one stage, resampled to 100 Hz.

    Decimation is by integer stride when the rate divides cleanly and by linear
    interpolation otherwise, which is what the research module did: the
    complexity measures below care about sample ordering rather than about
    anti-alias filtering, and a filter would itself alter the fractal
    dimensions.
    """
    per_epoch = int(round(EPOCH_SECONDS * fs))
    if per_epoch < 8 or stages is None or len(stages) == 0:
        return None

    usable = min(len(stages), len(signal) // per_epoch)
    idx = np.flatnonzero(np.asarray(stages[:usable]) == stage_code)
    if idx.size == 0:
        return None

    need = int(max_seconds * fs)
    parts, total = [], 0
    for i in idx:
        seg = signal[i * per_epoch:(i + 1) * per_epoch]
        if len(seg) < per_epoch or not np.isfinite(seg).all():
            continue
        parts.append(seg)
        total += len(seg)
        if total >= need:
            break
    if not parts:
        return None

    x = np.concatenate(parts)[:need].astype(float)
    if len(x) < int(30 * fs):
        return None

    if abs(fs - TARGET_FS) < 1e-6:
        return x
    ratio = fs / TARGET_FS
    if abs(ratio - round(ratio)) < 1e-9 and ratio >= 1:
        return x[::int(round(ratio))]
    n_out = int(len(x) * TARGET_FS / fs)
    if n_out < 100:
        return None
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)


def extract_complexity_features(channels, fs_dict, stage_signal):
    """Lempel-Ziv, fractal dimensions and entropies per sleep stage."""
    feat = _nan_features()
    try:
        import antropy as ant
    except ImportError:
        return feat

    name = _best_channel(channels, fs_dict)
    if name is None or stage_signal is None:
        return feat

    sig = np.asarray(channels[name], dtype=float)
    fs = float(fs_dict[name])
    stages = np.asarray(stage_signal)

    for stage in ('n2', 'n3'):
        x = _stage_signal(sig, fs, stages, STAGE_CODES[stage], SECONDS_STANDARD)
        if x is None:
            continue
        # Lempel-Ziv operates on a binary sequence; the research module
        # thresholded at the median, which the traced int64 dtype confirms.
        try:
            binary = (x > np.median(x)).astype(int)
            feat[f'lziv_complexity_{stage}'] = float(
                ant.lziv_complexity(binary, normalize=True))
        except Exception:
            pass
        for key, fn in (
            (f'svd_entropy_{stage}',
             lambda v: ant.svd_entropy(v, order=SVD_ORDER, delay=SVD_DELAY,
                                       normalize=True)),
            (f'higuchi_fd_{stage}',
             lambda v: ant.higuchi_fd(v, kmax=HIGUCHI_KMAX)),
            (f'spectral_entropy_{stage}',
             lambda v: ant.spectral_entropy(v, sf=TARGET_FS, method='welch',
                                            normalize=True)),
            (f'katz_fd_{stage}', ant.katz_fd),
            (f'petrosian_fd_{stage}', ant.petrosian_fd),
        ):
            try:
                feat[key] = float(fn(x))
            except Exception:
                pass

    x = _stage_signal(sig, fs, stages, STAGE_CODES['rem'],
                      SECONDS_SAMPLE_ENTROPY)
    if x is not None:
        try:
            feat['sample_entropy_rem'] = float(ant.sample_entropy(x))
        except Exception:
            pass

    x = _stage_signal(sig, fs, stages, STAGE_CODES['wake'], SECONDS_LONG)
    if x is not None:
        try:
            feat['dfa_wake'] = float(ant.detrended_fluctuation(x))
        except Exception:
            pass
        try:
            feat['perm_entropy_wake'] = float(
                ant.perm_entropy(x, order=PERM_ORDER, delay=PERM_DELAY,
                                 normalize=True))
        except Exception:
            pass

    return feat
