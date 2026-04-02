#!/usr/bin/env python
"""
PhysioNet Challenge 2026: Screening for Cognitive Impairment During Sleep Studies.

Dual-model probability average ensemble:
  Model A: 31-feature LR (v3 greedy — maximizes overall AUROC)
  Model B: 12-feature LR (min-site greedy — maximizes worst-case site AUROC)
  P(CI) = alpha * P_A + (1-alpha) * P_B,  alpha = 0.6

LOSO CV: Mean AUROC ~0.740, worst-site ~0.585
"""

import joblib
import numpy as np
import os
import warnings
from collections import OrderedDict

from helper_code import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# NumPy 2.x compat
_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# ============================================================
# Feature Lists & Hyperparameters
# ============================================================

MODEL_A_FEATURES = [
    'pct_rem', 'pct_wake', 'sleep_efficiency', 'arousal_index', 'ahi_auto',
    'n_awakenings', 'total_sleep_time_min', 'bout_mean_R', 'bout_std_R',
    'caisr_prob_arous_std', 'stage_prob_entropy_std', 'hrv_rmssd',
    'trans_R_W', 'trans_R_R', 'caisr_prob_r_std', 'eeg_n3_mobility',
    'caisr_prob_w_min', 'age', 'spo2_pct_below90', 'sex_female',
    'bout_std_W', 'eeg_n2_kurtosis', 'eeg_n3_kurtosis', 'spo2_std',
    'bout_mean_W', 'so_count_n2', 'sw_density_n2', 'so_spindle_coupling_n2',
    'spo2_mean', 'trans_W_N1', 'bout_mean_N3',
]

MODEL_B_FEATURES = [
    'caisr_prob_arous_std', 'hrv_pnn50', 'caisr_prob_n2_max',
    'eeg_rem_activity', 'sw_density_n2', 'trans_R_W',
    'so_spindle_coupling_n2', 'total_recording_min', 'caisr_prob_n1_max',
    'trans_R_N2', 'eeg_n3_rel_delta', 'dfa_n3',
]

ENSEMBLE_ALPHA = 0.6
MODEL_A_C = 0.005
MODEL_B_C = 0.005

EEG_CANDIDATES = ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 'o1-m2', 'o2-m1']


# ============================================================
# Helpers
# ============================================================

def _event_index(signal, total_hours):
    """Count discrete events (rising edges) per hour."""
    if signal is None or len(signal) == 0 or total_hours <= 0:
        return np.nan
    binary = (signal > 0).astype(int)
    rising = np.diff(binary, prepend=0) == 1
    return float(np.sum(rising)) / total_hours


def _find_best_eeg(channels, fs_dict):
    """Find best available EEG channel from priority list."""
    for ch in EEG_CANDIDATES:
        if ch in channels and channels[ch] is not None and len(channels[ch]) > 0:
            return channels[ch].astype(float), fs_dict.get(ch)
    return None, None


def _get_stage_eeg(eeg_sig, eeg_fs, stage_signal, stage_val):
    """Concatenate EEG segments for a given sleep stage value."""
    if stage_signal is None or len(stage_signal) == 0:
        return np.array([])
    n_eeg = len(eeg_sig)
    n_stages = len(stage_signal)
    if n_stages == 0 or n_eeg == 0:
        return np.array([])
    samples_per_epoch = n_eeg / n_stages
    stage_indices = np.where(stage_signal == stage_val)[0]
    segments = []
    for idx in stage_indices:
        start = int(idx * samples_per_epoch)
        end = int((idx + 1) * samples_per_epoch)
        end = min(end, n_eeg)
        if start < n_eeg and end > start:
            segments.append(eeg_sig[start:end])
    return np.concatenate(segments) if segments else np.array([])


def _simple_rpeak_detect(ecg_sig, fs):
    """Fallback R-peak detection using scipy."""
    try:
        from scipy.signal import find_peaks
        min_dist = int(0.4 * fs)
        peaks, _ = find_peaks(ecg_sig, distance=min_dist,
                              height=np.percentile(ecg_sig, 70))
        return peaks.tolist()
    except Exception:
        return []


def _bout_stats(valid_stages, stage_val):
    """Compute bout mean and std for a given stage value."""
    bout_mask = (valid_stages == stage_val).astype(int)
    diffs = np.diff(bout_mask, prepend=0, append=0)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    bout_lengths = ends - starts
    if len(bout_lengths) > 0:
        return float(np.mean(bout_lengths)), float(np.std(bout_lengths))
    return np.nan, np.nan


# ============================================================
# Feature Extractors
# ============================================================

def extract_caisr_features(algo_data):
    """Extract CAISR annotation features (sleep architecture, transitions, probabilities)."""
    feat = OrderedDict()

    stages = algo_data.get('stage_caisr', np.array([]))
    valid_mask = (stages < 9.0) & (stages >= 0) if len(stages) > 0 else np.array([], dtype=bool)
    valid_stages = stages[valid_mask] if len(stages) > 0 else np.array([])
    total_epochs = len(valid_stages)

    # Total recording duration from resp_caisr (1 Hz)
    resp = algo_data.get('resp_caisr', np.array([]))
    total_hours = len(resp) / 3600.0 if len(resp) > 0 else 0
    if total_hours > 0:
        feat['total_recording_min'] = len(resp) / 60.0
    else:
        feat['total_recording_min'] = float(len(stages) * 30 / 60.0) if len(stages) > 0 else np.nan

    if total_epochs > 0:
        feat['pct_wake'] = float(np.mean(valid_stages == 5))
        feat['pct_rem'] = float(np.mean(valid_stages == 4))
        sleep_mask = (valid_stages >= 1) & (valid_stages <= 4)
        total_sleep_epochs = np.sum(sleep_mask)
        feat['sleep_efficiency'] = float(np.mean(sleep_mask))
        feat['total_sleep_time_min'] = float(total_sleep_epochs * 30 / 60.0)

        # Awakenings
        sleep_indices = np.where(sleep_mask)[0]
        if len(sleep_indices) > 1:
            first_sleep = sleep_indices[0]
            last_sleep = sleep_indices[-1]
            in_period = valid_stages[first_sleep:last_sleep + 1]
            feat['n_awakenings'] = float(np.sum(np.diff((in_period == 5).astype(int)) == 1))
        else:
            feat['n_awakenings'] = 0.0

        # Transition matrix (5x5: W=0, N1=1, N2=2, N3=3, R=4)
        stage_map = {5: 0, 3: 1, 2: 2, 1: 3, 4: 4}
        mapped = np.array([stage_map.get(int(s), -1) for s in valid_stages])
        valid_trans = mapped[mapped >= 0]
        trans_matrix = np.zeros((5, 5))
        for i in range(len(valid_trans) - 1):
            trans_matrix[valid_trans[i], valid_trans[i + 1]] += 1
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans_prob = trans_matrix / row_sums
        stage_names = ['W', 'N1', 'N2', 'N3', 'R']
        for i, sn in enumerate(stage_names):
            for j, dn in enumerate(stage_names):
                feat[f'trans_{sn}_{dn}'] = float(trans_prob[i, j])

        # Bout statistics for R, W, N3
        feat['bout_mean_R'], feat['bout_std_R'] = _bout_stats(valid_stages, 4)
        feat['bout_mean_W'], feat['bout_std_W'] = _bout_stats(valid_stages, 5)
        feat['bout_mean_N3'], _ = _bout_stats(valid_stages, 1)
    else:
        for k in ['pct_wake', 'pct_rem', 'sleep_efficiency', 'total_sleep_time_min',
                   'n_awakenings', 'bout_mean_R', 'bout_std_R', 'bout_mean_W',
                   'bout_std_W', 'bout_mean_N3']:
            feat[k] = np.nan
        for sn in ['W', 'N1', 'N2', 'N3', 'R']:
            for dn in ['W', 'N1', 'N2', 'N3', 'R']:
                feat[f'trans_{sn}_{dn}'] = np.nan

    # Event indices
    feat['ahi_auto'] = _event_index(resp, total_hours)
    feat['arousal_index'] = _event_index(algo_data.get('arousal_caisr', np.array([])), total_hours)

    # CAISR probability features
    prob_channels = {
        'caisr_prob_w': ['std', 'min'],
        'caisr_prob_r': ['std'],
        'caisr_prob_arous': ['std'],
        'caisr_prob_n1': ['max'],
        'caisr_prob_n2': ['max'],
    }
    for ch_name, stats in prob_channels.items():
        sig = algo_data.get(ch_name, np.array([]))
        valid = sig[sig < 2.0] if len(sig) > 0 else np.array([])
        for stat in stats:
            if len(valid) > 0:
                if stat == 'std':
                    feat[f'{ch_name}_{stat}'] = float(np.std(valid))
                elif stat == 'min':
                    feat[f'{ch_name}_{stat}'] = float(np.min(valid))
                elif stat == 'max':
                    feat[f'{ch_name}_{stat}'] = float(np.max(valid))
            else:
                feat[f'{ch_name}_{stat}'] = np.nan

    # Stage probability entropy
    stage_prob_keys = ['caisr_prob_w', 'caisr_prob_n1', 'caisr_prob_n2',
                       'caisr_prob_n3', 'caisr_prob_r']
    prob_arrays = [algo_data.get(ch, np.array([])) for ch in stage_prob_keys]
    if all(len(p) > 0 for p in prob_arrays):
        min_len = min(len(p) for p in prob_arrays)
        probs = np.stack([p[:min_len] for p in prob_arrays], axis=1)
        valid_epoch_mask = np.all(probs < 2.0, axis=1)
        valid_probs = probs[valid_epoch_mask]
        if len(valid_probs) > 0:
            row_sums = np.maximum(valid_probs.sum(axis=1, keepdims=True), 1e-10)
            norm_probs = np.clip(valid_probs / row_sums, 1e-10, 1.0)
            entropy = -np.sum(norm_probs * np.log2(norm_probs), axis=1)
            feat['stage_prob_entropy_std'] = float(np.std(entropy))
        else:
            feat['stage_prob_entropy_std'] = np.nan
    else:
        feat['stage_prob_entropy_std'] = np.nan

    return feat


def extract_spo2_features(channels, fs_dict):
    """Extract SpO2 features: mean, std, pct_below90."""
    feat = OrderedDict()
    spo2_sig = None
    for ch in ['spo2', 'sao2']:
        if ch in channels and channels[ch] is not None:
            spo2_sig = channels[ch].astype(float)
            break
    if spo2_sig is None or len(spo2_sig) == 0:
        feat['spo2_pct_below90'] = np.nan
        feat['spo2_mean'] = np.nan
        feat['spo2_std'] = np.nan
        return feat

    if np.nanmax(spo2_sig) <= 1.5:
        spo2_sig = spo2_sig * 100.0

    valid = spo2_sig[(spo2_sig > 10) & (spo2_sig <= 100)]
    if len(valid) == 0:
        feat['spo2_pct_below90'] = np.nan
        feat['spo2_mean'] = np.nan
        feat['spo2_std'] = np.nan
    else:
        feat['spo2_pct_below90'] = float(np.mean(valid < 90))
        feat['spo2_mean'] = float(np.mean(valid))
        feat['spo2_std'] = float(np.std(valid))
    return feat


def extract_hrv_features(channels, fs_dict):
    """Extract HRV features: RMSSD and pNN50."""
    feat = OrderedDict()
    ecg_sig, ecg_fs = None, None
    for ch in ['ecg', 'ekg']:
        if ch in channels and ch in fs_dict:
            ecg_sig = channels[ch]
            ecg_fs = fs_dict[ch]
            break
    if ecg_sig is None or ecg_fs is None or ecg_fs <= 0:
        feat['hrv_rmssd'] = np.nan
        feat['hrv_pnn50'] = np.nan
        return feat

    max_samples = int(2 * 3600 * ecg_fs)
    ecg_sig = ecg_sig[:max_samples].astype(float)

    try:
        import neurokit2 as nk
        cleaned = nk.ecg_clean(ecg_sig, sampling_rate=int(ecg_fs))
        _, info = nk.ecg_peaks(cleaned, sampling_rate=int(ecg_fs))
        r_peaks = info.get('ECG_R_Peaks', [])
    except Exception:
        r_peaks = _simple_rpeak_detect(ecg_sig, ecg_fs)

    if len(r_peaks) < 10:
        feat['hrv_rmssd'] = np.nan
        feat['hrv_pnn50'] = np.nan
        return feat

    rr = np.diff(np.array(r_peaks)) / ecg_fs * 1000.0  # ms
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 10:
        feat['hrv_rmssd'] = np.nan
        feat['hrv_pnn50'] = np.nan
        return feat

    feat['hrv_rmssd'] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    feat['hrv_pnn50'] = float(np.mean(np.abs(np.diff(rr)) > 50))
    return feat


def _clipped_kurtosis(eeg_segment, eeg_fs):
    """Kurtosis of EEG segment with 1-99th percentile clipping."""
    from scipy.stats import kurtosis as scipy_kurtosis
    if len(eeg_segment) <= int(4 * eeg_fs):
        return np.nan
    p1, p99 = np.percentile(eeg_segment, [1, 99])
    return float(scipy_kurtosis(np.clip(eeg_segment, p1, p99), fisher=True))


def _replace_inf(X):
    """Replace inf with NaN in feature array."""
    return np.where(np.isinf(X), np.nan, X)


def extract_eeg_features(eeg_sig, eeg_fs, n2_eeg, n3_eeg, rem_eeg):
    """Extract EEG features: mobility, kurtosis, activity, relative delta.

    Takes pre-computed stage-specific EEG segments to avoid redundant extraction.
    """
    from scipy.signal import welch

    feat = OrderedDict()

    nan_keys = ['eeg_n3_mobility', 'eeg_n2_kurtosis', 'eeg_n3_kurtosis',
                'eeg_rem_activity', 'eeg_n3_rel_delta']

    if eeg_sig is None or eeg_fs is None or eeg_fs <= 0:
        for k in nan_keys:
            feat[k] = np.nan
        return feat

    # N3 mobility (Hjorth)
    if len(n3_eeg) > int(4 * eeg_fs):
        activity = np.var(n3_eeg)
        if activity > 0:
            feat['eeg_n3_mobility'] = float(np.sqrt(np.var(np.diff(n3_eeg)) / activity))
        else:
            feat['eeg_n3_mobility'] = np.nan
    else:
        feat['eeg_n3_mobility'] = np.nan

    feat['eeg_n2_kurtosis'] = _clipped_kurtosis(n2_eeg, eeg_fs)
    feat['eeg_n3_kurtosis'] = _clipped_kurtosis(n3_eeg, eeg_fs)

    # REM activity (Hjorth activity = variance)
    if len(rem_eeg) > int(4 * eeg_fs):
        feat['eeg_rem_activity'] = float(np.var(rem_eeg))
    else:
        feat['eeg_rem_activity'] = np.nan

    # N3 relative delta power
    if len(n3_eeg) > int(4 * eeg_fs):
        try:
            nperseg = min(int(4 * eeg_fs), len(n3_eeg))
            freqs, psd = welch(n3_eeg, fs=eeg_fs, nperseg=nperseg, noverlap=nperseg // 2)
            delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
            total_mask = (freqs >= 0.5) & (freqs <= 30.0)
            total_power = _trapz(psd[total_mask], freqs[total_mask])
            if total_power > 0:
                delta_power = _trapz(psd[delta_mask], freqs[delta_mask])
                feat['eeg_n3_rel_delta'] = float(delta_power / total_power)
            else:
                feat['eeg_n3_rel_delta'] = np.nan
        except Exception:
            feat['eeg_n3_rel_delta'] = np.nan
    else:
        feat['eeg_n3_rel_delta'] = np.nan

    return feat


def extract_nonlinear_features(eeg_sig, eeg_fs, n2_eeg, n3_eeg):
    """Extract nonlinear EEG features: DFA, slow wave density, SO-spindle coupling.

    Takes pre-computed stage-specific EEG segments to avoid redundant extraction.
    """
    feat = OrderedDict()

    nan_keys = ['dfa_n3', 'sw_density_n2', 'so_count_n2', 'so_spindle_coupling_n2']

    if eeg_sig is None or eeg_fs is None or eeg_fs <= 0:
        for k in nan_keys:
            feat[k] = np.nan
        return feat

    # --- DFA in N3 ---
    feat['dfa_n3'] = np.nan
    if len(n3_eeg) > int(30 * eeg_fs):
        try:
            from scipy.signal import resample
            target_fs = 100.0
            if eeg_fs > target_fs * 1.5:
                ds_eeg = resample(n3_eeg, int(len(n3_eeg) * target_fs / eeg_fs))
            else:
                ds_eeg = n3_eeg
                target_fs = eeg_fs
            ds_eeg = ds_eeg[:int(600 * target_fs)]

            import antropy as ant
            feat['dfa_n3'] = float(ant.detrended_fluctuation(ds_eeg))
        except Exception:
            feat['dfa_n3'] = np.nan

    # --- Slow waves and SO-spindle coupling in N2 ---
    feat['sw_density_n2'] = np.nan
    feat['so_count_n2'] = np.nan
    feat['so_spindle_coupling_n2'] = np.nan

    # Cap N2 to 60 minutes for yasa performance
    max_n2_samples = int(60 * 60 * eeg_fs)
    n2_capped = n2_eeg[:max_n2_samples] if len(n2_eeg) > max_n2_samples else n2_eeg

    if len(n2_capped) > int(30 * eeg_fs):
        try:
            import yasa

            sw = yasa.sw_detect(n2_capped, sf=eeg_fs, verbose=False)
            if sw is not None:
                sw_summary = sw.summary()
                n2_duration_min = len(n2_capped) / eeg_fs / 60.0
                feat['sw_density_n2'] = float(len(sw_summary) / n2_duration_min) if n2_duration_min > 0 else np.nan
                feat['so_count_n2'] = float(len(sw_summary))

                sp = yasa.spindles_detect(n2_capped, sf=eeg_fs, verbose=False)
                if sp is not None and len(sw_summary) > 0:
                    sp_summary = sp.summary()
                    if len(sp_summary) > 0:
                        sw_starts = sw_summary['Start'].values
                        sp_starts = np.sort(sp_summary['Start'].values)
                        coupled = 0
                        for sw_start in sw_starts:
                            idx = np.searchsorted(sp_starts, sw_start)
                            if idx < len(sp_starts) and (sp_starts[idx] - sw_start) < 1.5:
                                coupled += 1
                        feat['so_spindle_coupling_n2'] = float(coupled / len(sw_starts))
                    else:
                        feat['so_spindle_coupling_n2'] = 0.0
                else:
                    feat['so_spindle_coupling_n2'] = 0.0
            else:
                feat['sw_density_n2'] = 0.0
                feat['so_count_n2'] = 0.0
                feat['so_spindle_coupling_n2'] = 0.0

        except Exception as e:
            warnings.warn(f"Nonlinear feature extraction failed: {e}", stacklevel=2)

    return feat


# ============================================================
# Full Feature Extraction
# ============================================================

def extract_all_features_for_record(patient_data, phys_channels, phys_fs,
                                     algo_data, csv_path=DEFAULT_CSV_PATH):
    """Extract all features for one record. Returns a dict of {name: value}."""
    feat = OrderedDict()

    # 1. Demographics
    feat['age'] = float(load_age(patient_data))
    sex = load_sex(patient_data)
    feat['sex_female'] = 1.0 if sex == 'Female' else 0.0

    # 2. Standardize channels and derive bipolar signals
    if phys_channels:
        rename_rules = load_rename_rules(os.path.abspath(csv_path))
        original_labels = list(phys_channels.keys())
        rename_map, cols_to_drop = standardize_channel_names_rename_only(original_labels, rename_rules)

        std_channels = {}
        std_fs = {}
        for old_label in phys_channels:
            if old_label in cols_to_drop:
                continue
            new_label = rename_map.get(old_label, old_label.lower())
            std_channels[new_label] = phys_channels[old_label]
            if old_label in phys_fs:
                std_fs[new_label] = phys_fs[old_label]

        bipolar_configs = [
            ('c3-m2', 'c3', ['m2']), ('c4-m1', 'c4', ['m1']),
            ('f3-m2', 'f3', ['m2']), ('f4-m1', 'f4', ['m1']),
            ('o1-m2', 'o1', ['m2']), ('o2-m1', 'o2', ['m1']),
        ]
        for target, pos, neg_list in bipolar_configs:
            if target in std_channels or pos not in std_channels:
                continue
            if not all(n in std_channels for n in neg_list):
                continue
            all_fs = [std_fs.get(ch) for ch in [pos] + neg_list]
            if len(set(all_fs)) > 1 or all_fs[0] is None:
                continue
            derived = derive_bipolar_signal(std_channels[pos], std_channels[neg_list[0]])
            if derived is not None:
                std_channels[target] = derived
                std_fs[target] = std_fs[pos]
    else:
        std_channels = {}
        std_fs = {}

    # 3. CAISR features
    if algo_data:
        feat.update(extract_caisr_features(algo_data))
    else:
        # Fill all CAISR-derived features with NaN
        for k in ['pct_rem', 'pct_wake', 'sleep_efficiency', 'arousal_index', 'ahi_auto',
                   'n_awakenings', 'total_sleep_time_min', 'total_recording_min',
                   'bout_mean_R', 'bout_std_R', 'bout_mean_W', 'bout_std_W', 'bout_mean_N3',
                   'caisr_prob_arous_std', 'stage_prob_entropy_std', 'caisr_prob_r_std',
                   'caisr_prob_w_min', 'caisr_prob_n1_max', 'caisr_prob_n2_max',
                   'trans_R_W', 'trans_R_R', 'trans_W_N1', 'trans_R_N2']:
            feat[k] = np.nan

    # 4. SpO2
    feat.update(extract_spo2_features(std_channels, std_fs))

    # 5. HRV
    feat.update(extract_hrv_features(std_channels, std_fs))

    # 6. EEG features — compute stage segments once, share across extractors
    stage_signal = algo_data.get('stage_caisr', None) if algo_data else None
    eeg_sig, eeg_fs = _find_best_eeg(std_channels, std_fs)
    if eeg_sig is not None and eeg_fs is not None and stage_signal is not None:
        n2_eeg = _get_stage_eeg(eeg_sig, eeg_fs, stage_signal, 2)
        n3_eeg = _get_stage_eeg(eeg_sig, eeg_fs, stage_signal, 1)
        rem_eeg = _get_stage_eeg(eeg_sig, eeg_fs, stage_signal, 4)
    else:
        n2_eeg, n3_eeg, rem_eeg = np.array([]), np.array([]), np.array([])

    feat.update(extract_eeg_features(eeg_sig, eeg_fs, n2_eeg, n3_eeg, rem_eeg))
    feat.update(extract_nonlinear_features(eeg_sig, eeg_fs, n2_eeg, n3_eeg))

    del eeg_sig, n2_eeg, n3_eeg, rem_eeg

    return feat


def _feat_dict_to_vector(feat_dict, feature_list):
    """Extract ordered feature vector from dict."""
    return np.array([feat_dict.get(f, np.nan) for f in feature_list], dtype=np.float64)


def _load_record_signals(data_folder, site_id, patient_id, session_id):
    """Load physiological and algorithmic annotation signals for a record."""
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                             site_id, f"{patient_id}_ses-{session_id}.edf")
    if os.path.exists(phys_file):
        phys_channels, phys_fs = load_signal_data(phys_file)
    else:
        phys_channels, phys_fs = {}, {}

    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
    else:
        algo_data = {}

    return phys_channels, phys_fs, algo_data


# ============================================================
# Required Functions
# ============================================================

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    """Train dual-model ensemble on training data."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from tqdm import tqdm

    if verbose:
        print('Finding the Challenge data...')

    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    if verbose:
        print(f'Extracting features from {num_records} records...')

    feat_dicts = []
    labels = []

    pbar = tqdm(range(num_records), desc="Extracting", unit="rec", disable=not verbose)
    for i in pbar:
        try:
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            patient_data = load_demographics(demo_file, patient_id, session_id)

            label = load_diagnoses(demo_file, patient_id)
            if label != 0 and label != 1:
                continue

            phys_channels, phys_fs, algo_data = _load_record_signals(
                data_folder, site_id, patient_id, session_id)

            fd = extract_all_features_for_record(
                patient_data, phys_channels, phys_fs, algo_data, csv_path)
            feat_dicts.append(fd)
            labels.append(label)

            del phys_channels, algo_data

        except Exception as e:
            if verbose:
                tqdm.write(f"  Error on record {i}: {e}")
            continue

    pbar.close()

    y = np.array(labels, dtype=int)
    if verbose:
        print(f'Training on {len(y)} records ({y.sum()} positive, {len(y)-y.sum()} negative)')

    # Build feature matrices
    X_a = np.array([_feat_dict_to_vector(fd, MODEL_A_FEATURES) for fd in feat_dicts])
    X_b = np.array([_feat_dict_to_vector(fd, MODEL_B_FEATURES) for fd in feat_dicts])

    X_a = _replace_inf(X_a)
    X_b = _replace_inf(X_b)

    if verbose:
        print(f'Model A features: {X_a.shape[1]}, Model B features: {X_b.shape[1]}')

    def _make_pipeline(C):
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=C, max_iter=1000, random_state=42, penalty='l2')),
        ])

    model_a = _make_pipeline(MODEL_A_C)
    model_a.fit(X_a, y)

    model_b = _make_pipeline(MODEL_B_C)
    model_b.fit(X_b, y)

    if verbose:
        print(f'Ensemble alpha = {ENSEMBLE_ALPHA} (Model A weight)')
        print('Model A top features:')
        coefs_a = model_a.named_steps['lr'].coef_[0]
        for fname, c in sorted(zip(MODEL_A_FEATURES, coefs_a), key=lambda x: abs(x[1]), reverse=True)[:10]:
            print(f'  {fname:30s} {c:+.4f}')
        print('Model B top features:')
        coefs_b = model_b.named_steps['lr'].coef_[0]
        for fname, c in sorted(zip(MODEL_B_FEATURES, coefs_b), key=lambda x: abs(x[1]), reverse=True)[:10]:
            print(f'  {fname:30s} {c:+.4f}')

    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model_a, model_b)

    if verbose:
        print('Done training.')


def load_model(model_folder, verbose):
    """Load the trained dual-model ensemble."""
    filename = os.path.join(model_folder, 'model.sav')
    return joblib.load(filename)


def run_model(model, record, data_folder, verbose):
    """Run dual-model ensemble on a single record."""
    model_a = model['model_a']
    model_b = model['model_b']
    alpha = model['alpha']

    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(demo_file, patient_id, session_id)

    phys_channels, phys_fs, algo_data = _load_record_signals(
        data_folder, site_id, patient_id, session_id)

    feat_dict = extract_all_features_for_record(
        patient_data, phys_channels, phys_fs, algo_data, DEFAULT_CSV_PATH)

    features_a = model.get('features_a', MODEL_A_FEATURES)
    features_b = model.get('features_b', MODEL_B_FEATURES)
    vec_a = _replace_inf(_feat_dict_to_vector(feat_dict, features_a))
    vec_b = _replace_inf(_feat_dict_to_vector(feat_dict, features_b))

    prob_a = float(model_a.predict_proba(vec_a.reshape(1, -1))[0, 1])
    prob_b = float(model_b.predict_proba(vec_b.reshape(1, -1))[0, 1])

    probability_output = alpha * prob_a + (1 - alpha) * prob_b
    binary_output = int(probability_output >= 0.5)

    return binary_output, probability_output


# ============================================================
# Save Helper
# ============================================================

def save_model(model_folder, model_a, model_b):
    """Save the dual-model ensemble."""
    d = {
        'model_a': model_a,
        'model_b': model_b,
        'alpha': ENSEMBLE_ALPHA,
        'features_a': MODEL_A_FEATURES,
        'features_b': MODEL_B_FEATURES,
    }
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
