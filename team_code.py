#!/usr/bin/env python
"""
PhysioNet Challenge 2026: Screening for Cognitive Impairment During Sleep Studies.

Tier 1 model: Logistic regression with 20 curated, site-invariant features.
LOSO AUROC ~0.728 on training data (3 sites).
"""

import joblib
import numpy as np
import os
import sys
from collections import OrderedDict

from helper_code import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV_PATH = os.path.join(SCRIPT_DIR, 'channel_table.csv')

# NumPy 2.x compat
_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# The 20 curated features selected via LOSO-aware forward selection.
OPTIMAL_FEATURES = [
    'pct_rem', 'pct_wake', 'sleep_efficiency', 'arousal_index',
    'ahi_auto', 'n_awakenings', 'total_sleep_time_min',
    'bout_mean_R', 'bout_std_R', 'caisr_prob_arous_std',
    'stage_prob_entropy_std', 'hrv_rmssd', 'trans_R_W',
    'trans_R_R', 'caisr_prob_r_std', 'eeg_n3_mobility',
    'caisr_prob_w_min', 'age', 'spo2_pct_below90', 'sex_female',
]

OPTIMAL_C = 0.005

# ============================================================
# Feature Extraction
# ============================================================

def _event_index(signal, total_hours):
    """Count discrete events (rising edges) per hour."""
    if signal is None or len(signal) == 0 or total_hours <= 0:
        return np.nan
    binary = (signal > 0).astype(int)
    rising = np.diff(binary, prepend=0) == 1
    return float(np.sum(rising)) / total_hours


def extract_caisr_features(algo_data):
    """Extract CAISR annotation features needed for the 20-feature model."""
    feat = OrderedDict()

    stages = algo_data.get('stage_caisr', np.array([]))
    valid_mask = (stages < 9.0) & (stages >= 0) if len(stages) > 0 else np.array([], dtype=bool)
    valid_stages = stages[valid_mask] if len(stages) > 0 else np.array([])
    total_epochs = len(valid_stages)

    if total_epochs > 0:
        feat['pct_wake'] = float(np.mean(valid_stages == 5))
        feat['pct_rem'] = float(np.mean(valid_stages == 4))
        sleep_mask = (valid_stages >= 1) & (valid_stages <= 4)
        total_sleep_epochs = np.sum(sleep_mask)
        feat['sleep_efficiency'] = float(np.mean(sleep_mask))
        feat['total_sleep_time_min'] = float(total_sleep_epochs * 30 / 60.0)

        # Awakenings
        first_sleep = np.argmax(sleep_mask) if np.any(sleep_mask) else 0
        last_sleep = len(valid_stages) - 1 - np.argmax(sleep_mask[::-1]) if np.any(sleep_mask) else 0
        if first_sleep < last_sleep:
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

        # Bout stats for REM
        stage_val = 4  # REM
        is_stage = (valid_stages == stage_val).astype(int)
        diffs = np.diff(is_stage, prepend=0, append=0)
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        bouts = ends - starts if len(starts) > 0 else np.array([])
        feat['bout_mean_R'] = float(np.mean(bouts)) if len(bouts) > 0 else np.nan
        feat['bout_std_R'] = float(np.std(bouts)) if len(bouts) > 0 else np.nan
    else:
        for k in ['pct_wake', 'pct_rem', 'sleep_efficiency', 'total_sleep_time_min',
                   'n_awakenings', 'bout_mean_R', 'bout_std_R']:
            feat[k] = np.nan
        for sn in ['W', 'N1', 'N2', 'N3', 'R']:
            for dn in ['W', 'N1', 'N2', 'N3', 'R']:
                feat[f'trans_{sn}_{dn}'] = np.nan

    # Event indices
    resp = algo_data.get('resp_caisr', np.array([]))
    total_hours = len(resp) / 3600.0 if len(resp) > 0 else 0
    feat['ahi_auto'] = _event_index(resp, total_hours)
    feat['arousal_index'] = _event_index(algo_data.get('arousal_caisr', np.array([])), total_hours)

    # CAISR probability features
    for ch_name in ['caisr_prob_w', 'caisr_prob_r', 'caisr_prob_arous']:
        sig = algo_data.get(ch_name, np.array([]))
        if len(sig) > 0:
            valid = sig[sig < 2.0]  # filter filler values (9.0)
            suffix = ch_name  # e.g. 'caisr_prob_w'
            feat[f'{suffix}_std'] = float(np.std(valid)) if len(valid) > 0 else np.nan
            feat[f'{suffix}_min'] = float(np.min(valid)) if len(valid) > 0 else np.nan
        else:
            feat[f'{ch_name}_std'] = np.nan
            feat[f'{ch_name}_min'] = np.nan

    # Stage probability entropy
    prob_channels = ['caisr_prob_w', 'caisr_prob_n1', 'caisr_prob_n2', 'caisr_prob_n3', 'caisr_prob_r']
    prob_arrays = [algo_data.get(ch, np.array([])) for ch in prob_channels]
    if all(len(p) > 0 for p in prob_arrays):
        min_len = min(len(p) for p in prob_arrays)
        probs = np.stack([p[:min_len] for p in prob_arrays], axis=1)
        # Filter out filler epochs (any prob > 2.0)
        valid_epoch_mask = np.all(probs < 2.0, axis=1)
        valid_probs = probs[valid_epoch_mask]
        if len(valid_probs) > 0:
            row_sums = valid_probs.sum(axis=1, keepdims=True)
            row_sums = np.maximum(row_sums, 1e-10)
            norm_probs = valid_probs / row_sums
            norm_probs = np.clip(norm_probs, 1e-10, 1.0)
            entropy = -np.sum(norm_probs * np.log2(norm_probs), axis=1)
            feat['stage_prob_entropy_std'] = float(np.std(entropy))
        else:
            feat['stage_prob_entropy_std'] = np.nan
    else:
        feat['stage_prob_entropy_std'] = np.nan

    return feat


def extract_spo2_features(channels, fs_dict):
    """Extract SpO2 features."""
    feat = OrderedDict()
    spo2_sig = None
    for ch in ['spo2', 'sao2']:
        if ch in channels and channels[ch] is not None:
            spo2_sig = channels[ch].astype(float)
            break
    if spo2_sig is None or len(spo2_sig) == 0:
        feat['spo2_pct_below90'] = np.nan
        return feat

    # Auto-detect 0-1 vs 0-100 scale
    if np.nanmax(spo2_sig) <= 1.5:
        spo2_sig = spo2_sig * 100.0

    valid = spo2_sig[(spo2_sig > 10) & (spo2_sig <= 100)]
    feat['spo2_pct_below90'] = float(np.mean(valid < 90)) if len(valid) > 0 else np.nan
    return feat


def extract_hrv_features(channels, fs_dict):
    """Extract HRV features from ECG using R-peak detection."""
    feat = OrderedDict()
    ecg_sig, ecg_fs = None, None
    for ch in ['ecg', 'ekg']:
        if ch in channels and ch in fs_dict:
            ecg_sig = channels[ch]
            ecg_fs = fs_dict[ch]
            break
    if ecg_sig is None or ecg_fs is None or ecg_fs <= 0:
        feat['hrv_rmssd'] = np.nan
        return feat

    # Use first 2 hours to save time
    max_samples = int(2 * 3600 * ecg_fs)
    ecg_sig = ecg_sig[:max_samples].astype(float)

    try:
        import neurokit2 as nk
        cleaned = nk.ecg_clean(ecg_sig, sampling_rate=int(ecg_fs))
        _, info = nk.ecg_peaks(cleaned, sampling_rate=int(ecg_fs))
        r_peaks = info.get('ECG_R_Peaks', [])
    except Exception:
        r_peaks = _simple_rpeak_detect(ecg_sig, ecg_fs)

    if len(r_peaks) < 5:
        feat['hrv_rmssd'] = np.nan
        return feat

    rr = np.diff(np.array(r_peaks)) / ecg_fs * 1000.0  # ms
    rr = rr[(rr > 300) & (rr < 2000)]
    if len(rr) < 3:
        feat['hrv_rmssd'] = np.nan
        return feat

    feat['hrv_rmssd'] = float(np.sqrt(np.mean(np.diff(rr) ** 2)))
    return feat


def _simple_rpeak_detect(ecg_sig, fs):
    """Fallback R-peak detection using scipy if neurokit2 unavailable."""
    try:
        from scipy.signal import find_peaks
        # Bandpass filter approximation: just find peaks in raw signal
        min_dist = int(0.4 * fs)  # min 0.4s between beats
        peaks, _ = find_peaks(ecg_sig, distance=min_dist, height=np.percentile(ecg_sig, 70))
        return peaks.tolist()
    except Exception:
        return []


def extract_eeg_n3_mobility(channels, fs_dict, stage_signal):
    """Extract EEG Hjorth mobility during N3 sleep."""
    feat = OrderedDict()
    eeg_sig, eeg_fs = None, None
    for ch in ['c3-m2', 'c4-m1', 'f3-m2', 'f4-m1', 'o1-m2', 'o2-m1']:
        if ch in channels and ch in fs_dict:
            eeg_sig = channels[ch]
            eeg_fs = fs_dict[ch]
            break
    if eeg_sig is None or eeg_fs is None or stage_signal is None:
        feat['eeg_n3_mobility'] = np.nan
        return feat

    # Get N3 segments (stage == 1)
    valid_stages = stage_signal[stage_signal < 9]
    n3_mask = (valid_stages == 1)
    if np.sum(n3_mask) < 2:
        feat['eeg_n3_mobility'] = np.nan
        return feat

    # Concatenate N3 EEG segments
    epoch_dur = 30
    samples_per_epoch = int(epoch_dur * eeg_fs)
    n3_indices = np.where(n3_mask)[0]
    segments = []
    for idx in n3_indices:
        start = idx * samples_per_epoch
        end = start + samples_per_epoch
        if end <= len(eeg_sig):
            segments.append(eeg_sig[start:end])
    if not segments:
        feat['eeg_n3_mobility'] = np.nan
        return feat

    concat = np.concatenate(segments).astype(float)
    activity = np.var(concat)
    if activity <= 0:
        feat['eeg_n3_mobility'] = np.nan
        return feat
    var_d1 = np.var(np.diff(concat))
    feat['eeg_n3_mobility'] = float(np.sqrt(var_d1 / activity))
    return feat


def extract_all_features_for_record(patient_data, phys_channels, phys_fs,
                                     algo_data, csv_path=DEFAULT_CSV_PATH):
    """Extract all 20 features for one record."""
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

        # Bipolar derivations
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
        caisr_feat = extract_caisr_features(algo_data)
        feat.update(caisr_feat)
    else:
        for k in ['pct_rem', 'pct_wake', 'sleep_efficiency', 'arousal_index', 'ahi_auto',
                   'n_awakenings', 'total_sleep_time_min', 'bout_mean_R', 'bout_std_R',
                   'caisr_prob_arous_std', 'stage_prob_entropy_std', 'trans_R_W', 'trans_R_R',
                   'caisr_prob_r_std', 'caisr_prob_w_min']:
            feat[k] = np.nan

    # 4. SpO2
    spo2_feat = extract_spo2_features(std_channels, std_fs)
    feat.update(spo2_feat)

    # 5. HRV
    hrv_feat = extract_hrv_features(std_channels, std_fs)
    feat.update(hrv_feat)

    # 6. EEG N3 mobility
    stage_signal = algo_data.get('stage_caisr', None) if algo_data else None
    eeg_feat = extract_eeg_n3_mobility(std_channels, std_fs, stage_signal)
    feat.update(eeg_feat)

    # Build feature vector in correct order
    return np.array([feat.get(f, np.nan) for f in OPTIMAL_FEATURES], dtype=np.float64)


# ============================================================
# Required functions
# ============================================================

def train_model(data_folder, model_folder, verbose, csv_path=DEFAULT_CSV_PATH):
    """Train the model on training data."""
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

    features_list = []
    labels_list = []

    pbar = tqdm(range(num_records), desc="Extracting", unit="rec", disable=not verbose)
    for i in pbar:
        try:
            record = patient_metadata_list[i]
            patient_id = record[HEADERS['bids_folder']]
            site_id = record[HEADERS['site_id']]
            session_id = record[HEADERS['session_id']]

            # Load demographics
            demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
            patient_data = load_demographics(demo_file, patient_id, session_id)

            # Load label
            label = load_diagnoses(demo_file, patient_id)
            if label != 0 and label != 1:
                continue

            # Load physiological data
            phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                                     site_id, f"{patient_id}_ses-{session_id}.edf")
            if os.path.exists(phys_file):
                phys_channels, phys_fs = load_signal_data(phys_file)
            else:
                phys_channels, phys_fs = {}, {}

            # Load CAISR annotations
            algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                                     site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
            if os.path.exists(algo_file):
                algo_data, _ = load_signal_data(algo_file)
            else:
                algo_data = {}

            # Extract features
            feat_vec = extract_all_features_for_record(
                patient_data, phys_channels, phys_fs, algo_data, csv_path)

            features_list.append(feat_vec)
            labels_list.append(label)

            # Free memory
            del phys_channels, algo_data

        except Exception as e:
            if verbose:
                tqdm.write(f"  Error on record {i}: {e}")
            continue

    pbar.close()

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels_list, dtype=int)

    if verbose:
        print(f'Training on {len(y)} records ({y.sum()} positive, {len(y)-y.sum()} negative)')
        print(f'Feature shape: {X.shape}')

    # Replace inf
    X = np.where(np.isinf(X), np.nan, X)

    # Train pipeline
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=OPTIMAL_C, max_iter=1000, random_state=42, penalty='l2')),
    ])
    model.fit(X, y)

    if verbose:
        coefs = model.named_steps['lr'].coef_[0]
        print('Model coefficients:')
        for fname, c in sorted(zip(OPTIMAL_FEATURES, coefs), key=lambda x: abs(x[1]), reverse=True):
            print(f'  {fname:30s} {c:+.4f}')

    # Save
    os.makedirs(model_folder, exist_ok=True)
    save_model(model_folder, model)

    if verbose:
        print('Done training.')


def load_model(model_folder, verbose):
    """Load the trained model."""
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model


def run_model(model, record, data_folder, verbose):
    """Run the model on a single record."""
    model_pipeline = model['model']

    patient_id = record[HEADERS['bids_folder']]
    site_id = record[HEADERS['site_id']]
    session_id = record[HEADERS['session_id']]

    # Load demographics
    demo_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_data = load_demographics(demo_file, patient_id, session_id)

    # Load physiological data
    phys_file = os.path.join(data_folder, PHYSIOLOGICAL_DATA_SUBFOLDER,
                             site_id, f"{patient_id}_ses-{session_id}.edf")
    if os.path.exists(phys_file):
        phys_channels, phys_fs = load_signal_data(phys_file)
    else:
        phys_channels, phys_fs = {}, {}

    # Load CAISR annotations
    algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
                             site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
    if os.path.exists(algo_file):
        algo_data, _ = load_signal_data(algo_file)
    else:
        algo_data = {}

    # Extract features
    feat_vec = extract_all_features_for_record(
        patient_data, phys_channels, phys_fs, algo_data)

    # Replace inf
    feat_vec = np.where(np.isinf(feat_vec), np.nan, feat_vec)
    X = feat_vec.reshape(1, -1)

    # Predict
    probability_output = float(model_pipeline.predict_proba(X)[0, 1])
    binary_output = int(probability_output >= 0.5)

    return binary_output, probability_output


# ============================================================
# Helper
# ============================================================

def save_model(model_folder, model):
    """Save the model."""
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)
