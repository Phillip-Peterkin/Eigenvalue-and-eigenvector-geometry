"""Seizure EEG preprocessing for CHB-MIT: bandpass, notch, PCA on baseline.

Scientific context
------------------
Prepares CHB-MIT bipolar EEG data for VAR(1) Jacobian estimation in the
seizure prediction analysis. The CHB-MIT BIDS dataset uses 18 bipolar
channels (double banana montage), so CSD (Surface Laplacian) is NOT
applicable — bipolar channels are already spatially differentiated.

Preprocessing pipeline:
1. Bad channel detection and interpolation
2. Notch filter at 60 Hz + harmonics (US recording)
3. Bandpass filter 0.5–45 Hz
4. NO downsampling (already at 256 Hz)
5. NO CSD (bipolar montage)
6. PCA fitted on interictal baseline, projected onto all periods

PCA strategy: fit on the interictal baseline (−30 to −10 min relative to
seizure onset) to establish a fixed coordinate system. Project pre-ictal,
ictal, and post-ictal data into this subspace. This is the shared-subspace
approach — eliminates coordinate-system confounds by design.

Units
-----
- EEG voltages in microvolts (uV).
- Sampling rate: 256 Hz (native CHB-MIT).
- PCA components: arbitrary units (linear mixtures of bipolar channels).

Assumptions
-----------
- 18 bipolar channels from the double banana montage.
- Line noise at 60 Hz (US recording site).
- Bandpass 0.5–45 Hz removes DC drift and high-frequency muscle artifacts.
- PCA retains 15 components for dimensional comparability with propofol
  and sleep analyses (15×15 VAR matrices).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import mne
from scipy.stats import kurtosis as sp_kurtosis
from sklearn.decomposition import PCA


def preprocess_chbmit_raw(
    raw: mne.io.Raw,
    line_freq: float = 60.0,
    bandpass: tuple[float, float] = (0.5, 45.0),
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Preprocess CHB-MIT EEG: bad channels, notch, bandpass. No CSD.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (18 bipolar channels, 256 Hz).
    line_freq : float
        Line noise frequency in Hz (60 for US).
    bandpass : tuple[float, float]
        (low_freq, high_freq) for bandpass filter.

    Returns
    -------
    data : np.ndarray, shape (n_channels, n_samples)
        Preprocessed multi-channel data.
    sfreq : float
        Sampling rate (unchanged from input).
    info : dict
        Preprocessing provenance.
    """
    from cmcc.preprocess.qc import detect_bad_channels

    if not raw.preload:
        raw.load_data()

    info: dict[str, Any] = {
        "n_channels_original": len(raw.ch_names),
        "sfreq_original": raw.info["sfreq"],
        "montage": "bipolar_double_banana",
        "csd_applied": False,
    }

    bad = detect_bad_channels(raw)
    raw.info["bads"] = bad
    info["bad_channels"] = bad
    if bad:
        raw.interpolate_bads(reset_bads=True)
    info["n_channels_after_qc"] = len(raw.ch_names)

    freqs = [line_freq * i for i in range(1, 4)]
    valid_freqs = [f for f in freqs if f < raw.info["sfreq"] / 2.0]
    if valid_freqs:
        raw.notch_filter(valid_freqs, verbose=False)
    info["notch_freqs"] = valid_freqs

    if bandpass is not None:
        raw.filter(
            l_freq=bandpass[0], h_freq=bandpass[1],
            verbose=False,
        )
    info["bandpass"] = list(bandpass) if bandpass else None
    info["sfreq_final"] = raw.info["sfreq"]

    data = raw.get_data()
    return data, raw.info["sfreq"], info


def fit_baseline_pca(
    data: np.ndarray,
    sfreq: float,
    baseline_start_sec: float,
    baseline_end_sec: float,
    n_components: int = 15,
) -> tuple[PCA, dict[str, Any]]:
    """Fit PCA on the interictal baseline segment.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Full preprocessed recording.
    sfreq : float
        Sampling rate in Hz.
    baseline_start_sec : float
        Start of baseline in seconds from recording start.
    baseline_end_sec : float
        End of baseline in seconds from recording start.
    n_components : int
        Number of PCA components to retain.

    Returns
    -------
    pca : PCA
        Fitted PCA object.
    info : dict
        Provenance: n_components, explained_variance_ratio, etc.
    """
    n_ch, n_samples = data.shape
    start_samp = max(0, int(baseline_start_sec * sfreq))
    end_samp = min(n_samples, int(baseline_end_sec * sfreq))

    if end_samp <= start_samp:
        raise ValueError(
            f"Invalid baseline: start={baseline_start_sec}s, end={baseline_end_sec}s "
            f"(samples: {start_samp}–{end_samp})"
        )

    baseline_data = data[:, start_samp:end_samp]
    n_use = min(n_components, n_ch)

    pca = PCA(n_components=n_use, svd_solver="full")
    pca.fit(baseline_data.T)

    info = {
        "n_components": n_use,
        "n_channels_input": n_ch,
        "baseline_start_sec": baseline_start_sec,
        "baseline_end_sec": baseline_end_sec,
        "baseline_samples": end_samp - start_samp,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
    }

    return pca, info


def project_to_pca(data: np.ndarray, pca: PCA) -> np.ndarray:
    """Project data into a fitted PCA subspace.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Multi-channel data to project.
    pca : PCA
        Fitted PCA object.

    Returns
    -------
    np.ndarray, shape (n_components, n_samples)
        PCA-projected data.
    """
    return pca.transform(data.T).T


def reject_artifact_windows(
    data_pca: np.ndarray,
    sfreq: float,
    window_sec: float = 0.5,
    step_sec: float = 0.1,
    variance_threshold_sd: float = 5.0,
    kurtosis_threshold: float = 10.0,
) -> np.ndarray:
    """Flag artifact-contaminated windows via variance and kurtosis.

    Parameters
    ----------
    data_pca : np.ndarray, shape (n_components, n_samples)
        PCA-projected data.
    sfreq : float
        Sampling rate in Hz.
    window_sec : float
        Window duration in seconds.
    step_sec : float
        Step size in seconds.
    variance_threshold_sd : float
        Windows with mean variance exceeding this many SDs above
        the global mean variance are flagged as artifacts.
    kurtosis_threshold : float
        Windows with mean excess kurtosis above this threshold
        are flagged.

    Returns
    -------
    good_mask : np.ndarray, shape (n_windows,), dtype bool
        True = clean window, False = artifact.
    """
    n_ch, n_samples = data_pca.shape
    window_samples = int(window_sec * sfreq)
    step_samples = max(1, int(step_sec * sfreq))

    starts = list(range(0, n_samples - window_samples, step_samples))
    n_windows = len(starts)

    if n_windows == 0:
        return np.array([], dtype=bool)

    win_var = np.zeros(n_windows)
    win_kurt = np.zeros(n_windows)

    for i, s in enumerate(starts):
        seg = data_pca[:, s:s + window_samples]
        win_var[i] = np.mean(np.var(seg, axis=1))
        win_kurt[i] = np.mean(sp_kurtosis(seg, axis=1, fisher=True))

    var_mean = np.mean(win_var)
    var_std = np.std(win_var)

    good_mask = np.ones(n_windows, dtype=bool)
    if var_std > 0:
        good_mask &= win_var < (var_mean + variance_threshold_sd * var_std)
    good_mask &= win_kurt < kurtosis_threshold

    return good_mask
