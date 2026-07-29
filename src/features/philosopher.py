#!/usr/bin/env python
"""Philosopher's Stone latent extraction for Challenge records.

A brain-health model pretrained on 36,000 sleep recordings, promoted by an
organizer on the Challenge forum on 2026-07-07 and drawn from the same BDSP
lineage as the Challenge data. It maps one channel of overnight EEG to a
1024-dimensional latent plus cognition, disease and mortality heads.

Two things to hold onto while using it.

Its first regression target is `age_z` and age is fed in as a covariate, so its
dominant latent axis is age, which the age-conditioned metric assigns exactly
zero value. Every evaluation here therefore uses age-conditioned AUROC from the
first run; a strong plain-AUROC number from these latents would be mostly age
and would mean nothing. Three covariate variants are extracted so the age
contribution can be measured rather than assumed.

Batch normalization must stay in train mode. The reference implementation is
explicit that per-sample batchnorm statistics are part of the method, so
calling .eval() silently changes the outputs.

The earlier failure on this model was a head-construction bug in a deleted
driver script, not an environment incompatibility: the checkpoint loads under
the installed torch.
"""

import os
import sys

import numpy as np

PS_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'external', 'philosophers-stone')
VENDORED_TIMM = os.path.join(PS_ROOT, 'pytorch-image-models')

# Preference order matches the model's documented input: single-channel C4-M1,
# with the other central/frontal derivations as fallbacks.
CHANNEL_PREFERENCE = ['c4-m1', 'c3-m2', 'f4-m1', 'f3-m2', 'o2-m1', 'o1-m2']

TARGET_FS = 200
PRETRAIN_AGE_MEAN = 59.6   # cohort mean the model was trained against
PRETRAIN_AGE_SD = 15.0


def _ensure_paths():
    for p in (PS_ROOT, VENDORED_TIMM):
        if p not in sys.path:
            sys.path.insert(0, p)


def load_ps_model(device='cuda'):
    """Load the pretrained model. Leaves batchnorm in train mode deliberately."""
    _ensure_paths()
    from phi_utils.philosopher_utils import load_model, DefaultConfig

    cfg = DefaultConfig()
    cfg.device = device
    model = load_model(cfg)
    model.to(device)
    # Not model.eval(): see the module docstring.
    model.train()
    return model, cfg


def pick_channel(channels, fs_dict):
    """Choose the best available EEG derivation and its sampling rate."""
    for name in CHANNEL_PREFERENCE:
        sig = channels.get(name)
        if sig is not None and len(sig) > 0:
            return np.asarray(sig, dtype=np.float64), float(fs_dict.get(name, 0) or 0), name
    return None, None, None


def resample_to(signal, fs_in, fs_out=TARGET_FS):
    if not fs_in or abs(fs_in - fs_out) < 1e-6:
        return signal
    from scipy.signal import resample_poly
    from math import gcd
    a, b = int(round(fs_out)), int(round(fs_in))
    g = gcd(a, b)
    return resample_poly(signal, a // g, b // g)


def make_spectrogram(signal, cfg, fs=TARGET_FS, channel_name='c4-m1'):
    """Filter and convert to the wavelet spectrogram the model expects.

    Uses the reference pipeline's own `_compute_wavelet_specs` rather than
    reimplementing it: that function downsamples 200 -> 100 Hz to match
    training, applies the generalized Morse wavelet, interpolates onto a fixed
    100-bin frequency grid and pads to 11 hours. Reproducing any of that by
    hand would be a silent source of drift from the pretraining distribution.

    Both it and `preprocess_filter` take a DataFrame and select behaviour from
    the column name, so the channel is relabelled to cfg.channel.
    """
    _ensure_paths()
    import pandas as pd
    from phi_utils.philosopher_utils import _compute_wavelet_specs
    from phi_utils.preprocessing_and_spectrograms import preprocess_filter

    # cfg.channel drives column lookup inside _compute_wavelet_specs; whichever
    # derivation we actually picked is filtered identically, so relabel it.
    column = cfg.channel if cfg.channel in CHANNEL_PREFERENCE else 'c4-m1'
    frame = pd.DataFrame({column: np.asarray(signal, dtype=np.float64)})
    filtered = preprocess_filter(frame, Fs=fs)

    specs = _compute_wavelet_specs(filtered, fs, cfg)
    expected = (cfg.hours_pad * 3600 * cfg.fs_time, cfg.n_freqs)
    if tuple(specs.shape) != expected:
        raise ValueError(f'spectrogram {specs.shape} != expected {expected}')
    return specs


def age_z(age, pin_to_cohort_mean=False):
    """Standardized age covariate; optionally pinned to the pretraining mean.

    Pinning removes the explicit age input, which isolates how much of the
    latent's usefulness is age the metric would discount anyway.
    """
    if pin_to_cohort_mean or age is None or not np.isfinite(age):
        return 0.0
    return float((age - PRETRAIN_AGE_MEAN) / PRETRAIN_AGE_SD)


def infer_latent(model, specs, age_value, sex_value, device='cuda',
                 pin_age=False, want_heads=True):
    """Return (latent[1024], head_outputs dict).

    Bypasses `forward()` deliberately. That method ends with
    `torch.cat(yp_classification)`, and this checkpoint loads with no
    classification entries in `task_types`, so the concatenation raises on an
    empty list. This is the failure that stalled the track in March: a head
    assembly bug, not an environment problem. The latent is produced before
    that line, so calling stem -> maxvit -> final latent space reaches it
    directly, and skipping the sleep-stage decoder makes it cheaper too.
    """
    import torch

    x = torch.tensor(specs, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    cov = torch.tensor([[age_z(age_value, pin_age), float(sex_value)]],
                       dtype=torch.float32).to(device)

    with torch.no_grad():
        x1, x2, x3, h = model.stem(x)
        h = model.forward_maxvit(h)
        latent = model.forward_final_latent_space(h, cov)

        heads = {}
        if want_heads:
            for name, head in model.heads.items():
                try:
                    heads[name.replace('final_head_', '')] = (
                        head(latent).squeeze(0).float().cpu().numpy())
                except Exception:  # noqa: BLE001 - a broken head must not
                    continue      # cost us the latent

    out = latent.squeeze(0).float().cpu().numpy()
    del x, cov, h, x1, x2, x3
    return out, heads


def sex_to_numeric(sex):
    """The manifest encodes sex as 0=female, 1=male."""
    s = str(sex).strip().casefold()
    if s.startswith('m'):
        return 1.0
    if s.startswith('f'):
        return 0.0
    return 0.5
