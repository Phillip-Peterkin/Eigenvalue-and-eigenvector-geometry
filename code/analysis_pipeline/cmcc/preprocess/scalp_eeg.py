"""Scalp EEG preprocessing: CSD spatial filter, PCA dimensionality reduction,
and BIDS data loading for the ds005620 propofol sedation dataset.

Scientific context
------------------
Scalp EEG signals are spatially blurred by volume conduction through the skull.
The Current Source Density (CSD / Surface Laplacian) transform acts as a
high-pass spatial filter, sharpening signals toward local cortical sources.
After CSD, PCA reduces the remaining redundancy to a small number of
components suitable for VAR(1) Jacobian estimation.

Cross-modal analysis note
--------------------------
The iEEG pipeline estimates Jacobians on the high-gamma envelope (70-150 Hz),
a proxy for local cortical spiking activity. This scalp EEG pipeline estimates
Jacobians on broadband CSD-PCA components (0.5-45 Hz after bandpass), which
reflect synaptic currents and mesoscale oscillatory dynamics — a fundamentally
different observable. Results should be framed as a cross-modal generalization
test, not a direct replication.

PCA is fitted independently per condition (awake, sedation). The resulting
component spaces differ across conditions. However, all cross-condition
comparisons use basis-invariant metrics (eigenvalues, spectral radius,
eigenvalue gaps, singular values, effective rank), so this does not
invalidate the analysis.

Units
-----
- EEG voltages in microvolts (uV) from BrainVision files.
- CSD output in uV/m^2 (current source density).
- PCA components are in arbitrary units (linear mixtures of CSD channels).
- Sampling rates in Hz.

Assumptions
-----------
- Standard 10-20/10-05 electrode montage with known 3D positions.
- VEOG, HEOG, and EMG channels are non-EEG and must be removed before CSD.
- Bandpass filtering (0.5-45 Hz) removes DC drift and muscle artifacts before
  CSD and PCA, so the Jacobian captures neural oscillatory dynamics only.
- PCA is applied to the time-domain CSD data (no frequency-domain transform).
- Downsampling uses MNE's resample with built-in anti-aliasing filter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import mne
from sklearn.decomposition import PCA


DS005620_NON_EEG = ["VEOG", "HEOG", "EMG"]

DS005620_MONTAGE_NAME = "standard_1005"


def apply_csd(
    raw: mne.io.Raw,
    lambda2: float = 1e-5,
    stiffness: int = 4,
) -> mne.io.Raw:
    """Apply Current Source Density (Surface Laplacian) transform.

    Requires a montage with 3D electrode positions. If no montage is set,
    attempts to set a standard 10-05 montage automatically.

    Parameters
    ----------
    raw : mne.io.Raw
        EEG data with montage set. Must be preloaded.
    lambda2 : float
        Regularization parameter for the spherical spline interpolation.
    stiffness : int
        Stiffness of the spline (typically 4).

    Returns
    -------
    mne.io.Raw
        CSD-transformed data. Channel units become uV/m^2.

    Raises
    ------
    ValueError
        If no montage is set and automatic montage matching fails.
    """
    if raw.get_montage() is None:
        montage = mne.channels.make_standard_montage(DS005620_MONTAGE_NAME)
        try:
            raw.set_montage(montage, on_missing="warn")
        except Exception as exc:
            raise ValueError(
                "No montage set on Raw and automatic montage matching failed. "
                "Set a montage with raw.set_montage() before calling apply_csd."
            ) from exc

    if not raw.preload:
        raw.load_data()

    csd_raw = mne.preprocessing.compute_current_source_density(
        raw, lambda2=lambda2, stiffness=stiffness,
    )
    return csd_raw


def pca_reduce(
    data: np.ndarray,
    n_components: int = 15,
    return_pca: bool = False,
) -> np.ndarray | tuple[np.ndarray, PCA]:
    """Reduce multi-channel data to top PCA components.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Multi-channel time series.
    n_components : int
        Number of principal components to retain.
    return_pca : bool
        If True, also return the fitted PCA object.

    Returns
    -------
    data_pca : np.ndarray, shape (n_components_used, n_samples)
        Component time series.
    pca : PCA
        Fitted PCA object (only if return_pca=True).

    Notes
    -----
    If n_channels < n_components, all components are kept (n_components
    is clamped to n_channels).
    """
    n_ch, n_samp = data.shape
    n_use = min(n_components, n_ch)

    pca = PCA(n_components=n_use, svd_solver="full")
    data_pca = pca.fit_transform(data.T).T

    if return_pca:
        return data_pca, pca
    return data_pca


def load_ds005620_subject(
    subject_id: str,
    data_root: str | Path,
    task: str = "awake",
    acq: str = "EC",
    run: int | None = None,
) -> mne.io.Raw:
    """Load a BrainVision file from ds005620 BIDS layout.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g. "1074" or "sub-1074").
    data_root : str or Path
        Root directory of ds005620.
    task : str
        Task label: "awake", "sed", or "sed2".
    acq : str
        Acquisition label: "EC", "EO", "rest", "tms".
    run : int or None
        Run number (required for sed/sed2 tasks, None for awake).

    Returns
    -------
    mne.io.Raw
        Loaded raw data with VEOG/HEOG dropped and standard montage set.
    """
    data_root = Path(data_root)
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"

    if run is not None:
        fname = f"{sid}_task-{task}_acq-{acq}_run-{run}_eeg.vhdr"
    else:
        fname = f"{sid}_task-{task}_acq-{acq}_eeg.vhdr"

    vhdr_path = data_root / sid / "eeg" / fname
    if not vhdr_path.exists():
        raise FileNotFoundError(f"BrainVision file not found: {vhdr_path}")

    raw = mne.io.read_raw_brainvision(str(vhdr_path), preload=True, verbose=False)

    drop_chs = [ch for ch in DS005620_NON_EEG if ch in raw.ch_names]
    if drop_chs:
        raw.drop_channels(drop_chs)

    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    montage = mne.channels.make_standard_montage(DS005620_MONTAGE_NAME)
    raw.set_montage(montage, on_missing="warn")

    return raw


def preprocess_to_csd(
    raw: mne.io.Raw,
    line_freq: float = 50.0,
    downsample_to: float = 500.0,
    bandpass: tuple[float, float] = (0.5, 45.0),
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Preprocess scalp EEG through CSD, stopping before PCA.

    Pipeline stages: bad channel detection -> notch filter -> bandpass ->
    downsample -> CSD. Returns the CSD-transformed multi-channel data
    suitable for subsequent PCA (either per-state or shared-subspace).

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (must have montage set, VEOG/HEOG already dropped).
    line_freq : float
        Line noise frequency in Hz.
    downsample_to : float
        Target sampling rate in Hz.
    bandpass : tuple[float, float]
        (low_freq, high_freq) for bandpass filter. None to skip.

    Returns
    -------
    data_csd : np.ndarray, shape (n_channels, n_samples)
        CSD-transformed multi-channel data.
    sfreq : float
        Sampling rate after downsampling.
    info : dict
        Preprocessing provenance.
    """
    from cmcc.preprocess.qc import detect_bad_channels

    if not raw.preload:
        raw.load_data()

    info: dict[str, Any] = {
        "n_channels_original": len(raw.ch_names),
        "sfreq_original": raw.info["sfreq"],
    }

    bad = detect_bad_channels(raw)
    raw.info["bads"] = bad
    info["bad_channels"] = bad
    if bad:
        raw.interpolate_bads(reset_bads=True)
    info["n_channels_after_qc"] = len(raw.ch_names)

    freqs = [line_freq * i for i in range(1, 4)]
    raw.notch_filter(freqs, verbose=False)

    if bandpass is not None:
        raw.filter(
            l_freq=bandpass[0], h_freq=bandpass[1],
            verbose=False,
        )
    info["bandpass"] = list(bandpass) if bandpass is not None else None

    if raw.info["sfreq"] > downsample_to:
        raw.resample(downsample_to, verbose=False)
    info["sfreq_final"] = raw.info["sfreq"]

    raw = apply_csd(raw)

    data_csd = raw.get_data()
    return data_csd, raw.info["sfreq"], info


def pca_reduce_shared(
    state_data: dict[str, np.ndarray],
    n_components: int = 15,
) -> tuple[dict[str, np.ndarray], PCA, dict[str, Any]]:
    """Fit PCA on pooled multi-state data and project each state separately.

    This implements shared-subspace PCA: instead of fitting PCA independently
    per state (which yields different component spaces), we pool all states'
    CSD data, fit PCA once, then project each state into the common basis.
    This ensures between-state geometry comparisons are made in the same
    coordinate system.

    Parameters
    ----------
    state_data : dict[str, np.ndarray]
        Mapping from state label to CSD data array, each shape
        (n_channels, n_samples_state). All arrays must have the same
        n_channels.
    n_components : int
        Number of principal components to retain.

    Returns
    -------
    projected : dict[str, np.ndarray]
        Per-state PCA-projected data, each shape (n_components, n_samples_state).
    pca : PCA
        The fitted PCA object (fit on pooled data).
    info : dict
        Provenance: n_states, samples_per_state, total_samples_pooled,
        explained_variance_ratio, cumulative_variance.

    Raises
    ------
    ValueError
        If state_data is empty or channel counts are inconsistent.
    """
    if not state_data:
        raise ValueError("state_data must contain at least one state.")

    n_channels_list = [v.shape[0] for v in state_data.values()]
    if len(set(n_channels_list)) != 1:
        raise ValueError(
            f"All states must have the same number of channels. "
            f"Got: {dict(zip(state_data.keys(), n_channels_list))}"
        )
    n_ch = n_channels_list[0]
    n_use = min(n_components, n_ch)

    pooled = np.concatenate(list(state_data.values()), axis=1)

    pca = PCA(n_components=n_use, svd_solver="full")
    pca.fit(pooled.T)

    projected = {}
    samples_per_state = {}
    for label, data in state_data.items():
        projected[label] = pca.transform(data.T).T
        samples_per_state[label] = data.shape[1]

    info = {
        "n_states": len(state_data),
        "state_labels": list(state_data.keys()),
        "samples_per_state": samples_per_state,
        "total_samples_pooled": pooled.shape[1],
        "n_components": n_use,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": float(np.sum(pca.explained_variance_ratio_)),
    }

    return projected, pca, info


def preprocess_scalp_eeg(
    raw: mne.io.Raw,
    line_freq: float = 50.0,
    downsample_to: float = 500.0,
    n_components: int = 15,
    bandpass: tuple[float, float] = (0.5, 45.0),
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Full scalp EEG preprocessing pipeline.

    Pipeline stages:
    1. Bad channel detection and interpolation
    2. Line noise removal (notch filter at line_freq + harmonics)
    3. Bandpass filter (removes DC drift and high-frequency muscle artifacts)
    4. Downsample (with anti-aliasing)
    5. CSD / Surface Laplacian transform
    6. PCA dimensionality reduction

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (must have montage set, VEOG/HEOG already dropped).
    line_freq : float
        Line noise frequency in Hz (50 for Europe, 60 for North America).
    downsample_to : float
        Target sampling rate in Hz.
    n_components : int
        Number of PCA components to retain.
    bandpass : tuple[float, float]
        (low_freq, high_freq) in Hz for bandpass filter. Removes DC drift
        (< low_freq) and muscle/EMG artifacts (> high_freq). Applied before
        downsampling to ensure clean anti-aliasing. Set to None to skip.

    Returns
    -------
    data_pca : np.ndarray, shape (n_components, n_samples)
        PCA-reduced component time series.
    sfreq : float
        Sampling rate after downsampling.
    info : dict
        Preprocessing provenance: n_channels_original, bad_channels,
        n_channels_after_qc, sfreq_original, sfreq_final, bandpass,
        n_components, explained_variance_ratio, cumulative_variance.
    """
    from cmcc.preprocess.qc import detect_bad_channels

    if not raw.preload:
        raw.load_data()

    info: dict[str, Any] = {
        "n_channels_original": len(raw.ch_names),
        "sfreq_original": raw.info["sfreq"],
    }

    bad = detect_bad_channels(raw)
    raw.info["bads"] = bad
    info["bad_channels"] = bad
    if bad:
        raw.interpolate_bads(reset_bads=True)
    info["n_channels_after_qc"] = len(raw.ch_names)

    freqs = [line_freq * i for i in range(1, 4)]
    raw.notch_filter(freqs, verbose=False)

    if bandpass is not None:
        raw.filter(
            l_freq=bandpass[0], h_freq=bandpass[1],
            verbose=False,
        )
    info["bandpass"] = list(bandpass) if bandpass is not None else None

    if raw.info["sfreq"] > downsample_to:
        raw.resample(downsample_to, verbose=False)
    info["sfreq_final"] = raw.info["sfreq"]

    raw = apply_csd(raw)

    data = raw.get_data()
    data_pca, pca_obj = pca_reduce(data, n_components=n_components, return_pca=True)

    info["n_components"] = data_pca.shape[0]
    info["explained_variance_ratio"] = pca_obj.explained_variance_ratio_.tolist()
    info["cumulative_variance"] = float(
        np.sum(pca_obj.explained_variance_ratio_)
    )

    return data_pca, raw.info["sfreq"], info
