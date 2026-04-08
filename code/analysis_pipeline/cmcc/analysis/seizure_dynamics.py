"""Time-resolved eigenvalue spacing dynamics for seizure prediction.

Scientific context
------------------
Tests whether pre-ictal dynamics show eigenvalue spacing narrowing
(approaching exceptional-point geometry), whether seizure onset
corresponds to a spacing minimum, and whether post-ictal recovery shows
spacing widening. This is a time-resolved trajectory analysis aligned
to annotated seizure onsets — fundamentally different from the discrete
state comparisons (awake vs sedation) in the propofol and sleep analyses.

The analysis computes sliding-window VAR(1) Jacobian eigenvalue metrics
across the peri-ictal period, smooths and z-scores them to an interictal
baseline, and aligns all seizures to a common time axis (t=0 at onset).

Controls include:
- Sham-onset null (random interictal timepoints)
- Phase-randomized surrogates (spectral-matched noise)
- Alpha/delta spectral power partial correlations

Units
-----
- Time in seconds relative to seizure onset (negative = pre-ictal).
- Spacing metrics in z-score units relative to interictal baseline.
- Power in arbitrary units (mean squared amplitude per window).

Assumptions
-----------
- Input is PCA-projected data (n_components x n_samples) at 256 Hz.
- PCA was fitted on interictal baseline only (shared-subspace approach).
- Seizure onset and offset are annotated in seconds from recording start.
- Muscle artifact contaminates ictal period; primary analysis excludes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt

from cmcc.analysis.dynamical_systems import (
    estimate_jacobian,
    detect_exceptional_points,
)


@dataclass
class SeizureTrajectory:
    """Time-resolved eigenvalue spacing trajectory for one seizure.

    Attributes
    ----------
    subject_id : str
    seizure_idx : int
    time_sec : np.ndarray
        Time relative to onset (negative = pre-ictal).
    min_spacing_raw : np.ndarray
        Raw minimum eigenvalue spacing (unsmoothed, un-z-scored).
    min_spacing_z : np.ndarray
        Z-scored, smoothed minimum spacing.
    median_nns_z : np.ndarray
        Z-scored, smoothed median nearest-neighbor spacing.
    p10_nns_z : np.ndarray
        Z-scored, smoothed 10th percentile NNS.
    spectral_radius_z : np.ndarray
        Z-scored, smoothed spectral radius.
    ep_score_z : np.ndarray
        Z-scored, smoothed EP score.
    alpha_power_z : np.ndarray
        Z-scored alpha power.
    delta_power_z : np.ndarray
        Z-scored delta power.
    artifact_mask : np.ndarray
        True = clean window.
    period_label : np.ndarray
        String label per window.
    baseline_mean : float
        Mean spacing during baseline.
    baseline_std : float
        Std during baseline.
    preictal_slope : float
        Linear slope of spacing vs time in pre-ictal window.
    seizure_duration : float
        Seizure duration in seconds.
    event_type : str
    """
    subject_id: str
    seizure_idx: int
    time_sec: np.ndarray
    min_spacing_raw: np.ndarray
    min_spacing_z: np.ndarray
    median_nns_z: np.ndarray
    p10_nns_z: np.ndarray
    spectral_radius_z: np.ndarray
    ep_score_z: np.ndarray
    alpha_power_z: np.ndarray
    delta_power_z: np.ndarray
    artifact_mask: np.ndarray
    period_label: np.ndarray
    baseline_mean: float
    baseline_std: float
    preictal_slope: float
    seizure_duration: float
    event_type: str


def _compute_nns_metrics(eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute nearest-neighbor spacing metrics from eigenvalue arrays.

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (n_windows, n_channels)
        Complex eigenvalues per window.

    Returns
    -------
    min_spacing : np.ndarray, shape (n_windows,)
    median_nns : np.ndarray, shape (n_windows,)
    p10_nns : np.ndarray, shape (n_windows,)
    """
    n_windows, n_ch = eigenvalues.shape
    min_spacing = np.zeros(n_windows)
    median_nns = np.zeros(n_windows)
    p10_nns = np.zeros(n_windows)

    for w in range(n_windows):
        evals = eigenvalues[w]
        n_eval = min(len(evals), 20)
        gaps = []
        for i in range(n_eval):
            for j in range(i + 1, n_eval):
                gaps.append(abs(evals[i] - evals[j]))
        if gaps:
            gaps_arr = np.array(gaps)
            min_spacing[w] = np.min(gaps_arr)

            nns = np.zeros(n_eval)
            for i in range(n_eval):
                dists = [abs(evals[i] - evals[j]) for j in range(n_eval) if j != i]
                nns[i] = min(dists) if dists else 0.0
            median_nns[w] = np.median(nns)
            p10_nns[w] = np.percentile(nns, 10)

    return min_spacing, median_nns, p10_nns


def _smooth_timeseries(x: np.ndarray, window_size: int) -> np.ndarray:
    """Apply centered moving average smoothing."""
    if window_size <= 1 or len(x) <= window_size:
        return x.copy()
    kernel = np.ones(window_size) / window_size
    padded = np.pad(x, window_size // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(x)]


def _zscore_to_baseline(
    x: np.ndarray, baseline_mask: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Z-score a time series to a baseline period.

    Returns
    -------
    z : np.ndarray
    baseline_mean : float
    baseline_std : float
    """
    baseline_vals = x[baseline_mask]
    if len(baseline_vals) < 2:
        return np.zeros_like(x), 0.0, 1.0
    mu = float(np.mean(baseline_vals))
    sigma = float(np.std(baseline_vals))
    if sigma == 0:
        sigma = 1.0
    return (x - mu) / sigma, mu, sigma


def compute_spectral_power_per_window(
    data: np.ndarray,
    sfreq: float,
    window_centers: np.ndarray,
    window_samples: int,
    band: tuple[float, float] = (8.0, 12.0),
) -> np.ndarray:
    """Compute band-limited power per sliding window.

    Parameters
    ----------
    data : np.ndarray, shape (n_components, n_samples)
    sfreq : float
    window_centers : np.ndarray, shape (n_windows,)
    window_samples : int
    band : tuple[float, float]
        (low_freq, high_freq) in Hz.

    Returns
    -------
    np.ndarray, shape (n_windows,)
    """
    nyq = sfreq / 2.0
    lo = band[0] / nyq
    hi = min(band[1] / nyq, 0.99)
    if lo >= hi or lo <= 0:
        return np.zeros(len(window_centers))

    sos = butter(4, [lo, hi], btype="band", output="sos")
    data_filt = sosfiltfilt(sos, data, axis=1)

    n_ch, n_total = data.shape
    n_windows = len(window_centers)
    half_w = window_samples // 2
    power = np.zeros(n_windows)

    for i, c in enumerate(window_centers):
        c = int(c)
        start = max(0, c - half_w)
        end = min(n_total, c + half_w)
        if end > start:
            power[i] = np.mean(data_filt[:, start:end] ** 2)

    return power


def compute_seizure_trajectory(
    data_pca: np.ndarray,
    sfreq: float,
    seizure_onset_sec: float,
    seizure_offset_sec: float,
    baseline_start_sec: float,
    baseline_end_sec: float,
    window_sec: float = 0.5,
    step_sec: float = 0.1,
    regularization: float = 1e-4,
    smoothing_sec: float = 30.0,
    subject_id: str = "",
    seizure_idx: int = 0,
    seizure_duration: float = 0.0,
    event_type: str = "sz",
    artifact_mask: np.ndarray | None = None,
) -> SeizureTrajectory:
    """Compute time-resolved eigenvalue spacing trajectory for one seizure.

    Parameters
    ----------
    data_pca : np.ndarray, shape (n_components, n_samples)
        PCA-projected peri-ictal data (full recording or extracted segment).
    sfreq : float
        Sampling rate in Hz.
    seizure_onset_sec : float
        Seizure onset in seconds from start of data_pca.
    seizure_offset_sec : float
        Seizure offset in seconds from start of data_pca.
    baseline_start_sec : float
        Interictal baseline start in seconds from start of data_pca.
    baseline_end_sec : float
        Interictal baseline end in seconds from start of data_pca.
    window_sec : float
        VAR(1) window duration in seconds.
    step_sec : float
        Step size in seconds.
    regularization : float
        Ridge parameter for VAR(1).
    smoothing_sec : float
        Moving average window in seconds for smoothing.
    subject_id : str
    seizure_idx : int
    seizure_duration : float
    event_type : str
    artifact_mask : np.ndarray or None
        Pre-computed artifact mask. If None, all windows are clean.

    Returns
    -------
    SeizureTrajectory
    """
    n_ch, n_samples = data_pca.shape

    ch_mean = data_pca.mean(axis=1, keepdims=True)
    ch_std = data_pca.std(axis=1, keepdims=True)
    ch_std[ch_std == 0] = 1.0
    data_z = (data_pca - ch_mean) / ch_std

    window_samples = int(window_sec * sfreq)
    step_samples = max(1, int(step_sec * sfreq))
    window_samples = max(window_samples, n_ch + 10)

    jac_result = estimate_jacobian(
        data_z,
        window_size=window_samples,
        step_size=step_samples,
        regularization=regularization,
    )

    ep_result = detect_exceptional_points(jac_result)

    time_sec = (jac_result.window_centers / sfreq) - seizure_onset_sec

    min_spacing, median_nns, p10_nns = _compute_nns_metrics(jac_result.eigenvalues)

    alpha_power = compute_spectral_power_per_window(
        data_pca, sfreq, jac_result.window_centers, window_samples,
        band=(8.0, 12.0),
    )
    delta_power = compute_spectral_power_per_window(
        data_pca, sfreq, jac_result.window_centers, window_samples,
        band=(0.5, 4.0),
    )

    smoothing_windows = max(1, int(smoothing_sec / step_sec))
    min_spacing_s = _smooth_timeseries(min_spacing, smoothing_windows)
    median_nns_s = _smooth_timeseries(median_nns, smoothing_windows)
    p10_nns_s = _smooth_timeseries(p10_nns, smoothing_windows)
    spec_radius_s = _smooth_timeseries(jac_result.spectral_radius, smoothing_windows)
    ep_score_s = _smooth_timeseries(ep_result.ep_scores, smoothing_windows)
    alpha_s = _smooth_timeseries(alpha_power, smoothing_windows)
    delta_s = _smooth_timeseries(delta_power, smoothing_windows)

    baseline_time_start = baseline_start_sec - seizure_onset_sec
    baseline_time_end = baseline_end_sec - seizure_onset_sec
    baseline_mask = (time_sec >= baseline_time_start) & (time_sec < baseline_time_end)

    min_spacing_z, bl_mean, bl_std = _zscore_to_baseline(min_spacing_s, baseline_mask)
    median_nns_z, _, _ = _zscore_to_baseline(median_nns_s, baseline_mask)
    p10_nns_z, _, _ = _zscore_to_baseline(p10_nns_s, baseline_mask)
    spec_radius_z, _, _ = _zscore_to_baseline(spec_radius_s, baseline_mask)
    ep_score_z, _, _ = _zscore_to_baseline(ep_score_s, baseline_mask)
    alpha_z, _, _ = _zscore_to_baseline(alpha_s, baseline_mask)
    delta_z, _, _ = _zscore_to_baseline(delta_s, baseline_mask)

    n_windows = len(time_sec)
    onset_rel = 0.0
    offset_rel = seizure_offset_sec - seizure_onset_sec
    postictal_skip_end = offset_rel + 120.0

    period_label = np.array(["interictal_baseline"] * n_windows)
    for i in range(n_windows):
        t = time_sec[i]
        if baseline_time_start <= t < baseline_time_end:
            period_label[i] = "interictal_baseline"
        elif baseline_time_end <= t < onset_rel:
            period_label[i] = "pre_ictal"
        elif onset_rel <= t < offset_rel:
            period_label[i] = "ictal"
        elif offset_rel <= t < postictal_skip_end:
            period_label[i] = "post_ictal_skip"
        elif t >= postictal_skip_end:
            period_label[i] = "post_ictal"

    if artifact_mask is None:
        artifact_mask = np.ones(n_windows, dtype=bool)
    elif len(artifact_mask) != n_windows:
        artifact_mask = np.ones(n_windows, dtype=bool)

    preictal_slope = compute_preictal_slope(time_sec, min_spacing_z)

    return SeizureTrajectory(
        subject_id=subject_id,
        seizure_idx=seizure_idx,
        time_sec=time_sec,
        min_spacing_raw=min_spacing,
        min_spacing_z=min_spacing_z,
        median_nns_z=median_nns_z,
        p10_nns_z=p10_nns_z,
        spectral_radius_z=spec_radius_z,
        ep_score_z=ep_score_z,
        alpha_power_z=alpha_z,
        delta_power_z=delta_z,
        artifact_mask=artifact_mask,
        period_label=period_label,
        baseline_mean=bl_mean,
        baseline_std=bl_std,
        preictal_slope=preictal_slope,
        seizure_duration=seizure_duration,
        event_type=event_type,
    )


def compute_preictal_slope(
    time_sec: np.ndarray,
    spacing_z: np.ndarray,
    window_minutes: tuple[float, float] = (-10.0, 0.0),
) -> float:
    """Compute linear regression slope of spacing vs time in pre-ictal window.

    Parameters
    ----------
    time_sec : np.ndarray
        Time relative to onset.
    spacing_z : np.ndarray
        Z-scored spacing.
    window_minutes : tuple[float, float]
        (start, end) in minutes relative to onset.

    Returns
    -------
    float
        Slope (units per second). Negative = narrowing.
    """
    t_start = window_minutes[0] * 60.0
    t_end = window_minutes[1] * 60.0
    mask = (time_sec >= t_start) & (time_sec < t_end)
    if mask.sum() < 3:
        return float("nan")

    t = time_sec[mask]
    y = spacing_z[mask]
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return float("nan")

    t = t[valid]
    y = y[valid]
    coeffs = np.polyfit(t, y, 1)
    return float(coeffs[0])


def phase_randomize_surrogate(
    data: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Generate a phase-randomized surrogate preserving power spectrum.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (n_channels, n_samples)
    """
    n_ch, n_samp = data.shape
    fft_data = np.fft.rfft(data, axis=1)
    n_freq = fft_data.shape[1]
    phases = rng.uniform(0, 2 * np.pi, n_freq)
    phases[0] = 0.0
    if n_samp % 2 == 0:
        phases[-1] = 0.0
    phase_shift = np.exp(1j * phases)
    fft_surrogate = fft_data * phase_shift[np.newaxis, :]
    surrogate = np.fft.irfft(fft_surrogate, n=n_samp, axis=1)
    return surrogate


def compute_sham_trajectories(
    data_pca: np.ndarray,
    sfreq: float,
    n_shams: int,
    seizure_onsets_sec: list[float],
    recording_duration_sec: float,
    rng: np.random.Generator,
    window_sec: float = 0.5,
    step_sec: float = 0.1,
    regularization: float = 1e-4,
    smoothing_sec: float = 30.0,
    analysis_window_sec: float = 2400.0,
    exclusion_radius_sec: float = 3600.0,
) -> list[SeizureTrajectory]:
    """Generate sham-onset trajectories from interictal periods.

    Selects random timepoints far from real seizures and computes
    identical spacing trajectories to serve as null distribution.

    Parameters
    ----------
    data_pca : np.ndarray, shape (n_components, n_samples)
    sfreq : float
    n_shams : int
    seizure_onsets_sec : list[float]
        Real seizure onsets to avoid.
    recording_duration_sec : float
    rng : np.random.Generator
    window_sec, step_sec, regularization, smoothing_sec : float
    analysis_window_sec : float
        Total analysis window around sham onset (pre + post).
    exclusion_radius_sec : float
        Minimum distance from any real seizure onset.

    Returns
    -------
    list[SeizureTrajectory]
    """
    half_window = analysis_window_sec / 2.0
    min_onset = half_window
    max_onset = recording_duration_sec - half_window

    if max_onset <= min_onset:
        return []

    sham_trajectories: list[SeizureTrajectory] = []
    attempts = 0
    max_attempts = n_shams * 20

    while len(sham_trajectories) < n_shams and attempts < max_attempts:
        attempts += 1
        sham_onset = float(rng.uniform(min_onset, max_onset))

        too_close = any(
            abs(sham_onset - real_onset) < exclusion_radius_sec
            for real_onset in seizure_onsets_sec
        )
        if too_close:
            continue

        onset_sample = int(sham_onset * sfreq)
        start_sample = max(0, onset_sample - int(half_window * sfreq))
        end_sample = min(data_pca.shape[1], onset_sample + int(half_window * sfreq))

        if end_sample - start_sample < int(half_window * sfreq):
            continue

        seg = data_pca[:, start_sample:end_sample]
        local_onset = sham_onset - (start_sample / sfreq)
        baseline_start = 0.0
        baseline_end = local_onset - 600.0

        if baseline_end <= baseline_start + 60:
            continue

        try:
            traj = compute_seizure_trajectory(
                seg, sfreq,
                seizure_onset_sec=local_onset,
                seizure_offset_sec=local_onset + 30.0,
                baseline_start_sec=baseline_start,
                baseline_end_sec=baseline_end,
                window_sec=window_sec,
                step_sec=step_sec,
                regularization=regularization,
                smoothing_sec=smoothing_sec,
                subject_id="sham",
                seizure_idx=len(sham_trajectories),
                seizure_duration=0.0,
                event_type="sham",
            )
            sham_trajectories.append(traj)
        except (ValueError, np.linalg.LinAlgError):
            continue

    return sham_trajectories


def compute_surrogate_baseline(
    data_interictal: np.ndarray,
    sfreq: float,
    n_surrogates: int,
    rng: np.random.Generator,
    window_sec: float = 0.5,
    step_sec: float = 0.1,
    regularization: float = 1e-4,
) -> dict[str, Any]:
    """Compute spacing statistics on phase-randomized surrogates.

    Parameters
    ----------
    data_interictal : np.ndarray, shape (n_components, n_samples)
        Interictal data segment.
    sfreq : float
    n_surrogates : int
    rng : np.random.Generator
    window_sec, step_sec, regularization : float

    Returns
    -------
    dict with keys: surrogate_mean_spacings, surrogate_std_spacing,
    real_mean_spacing, real_std_spacing.
    """
    n_ch, n_samples = data_interictal.shape
    window_samples = max(int(window_sec * sfreq), n_ch + 10)
    step_samples = max(1, int(step_sec * sfreq))

    if n_samples < window_samples + 1:
        return {
            "surrogate_mean_spacings": [],
            "surrogate_std_spacing": float("nan"),
            "real_mean_spacing": float("nan"),
            "real_std_spacing": float("nan"),
        }

    ch_mean = data_interictal.mean(axis=1, keepdims=True)
    ch_std = data_interictal.std(axis=1, keepdims=True)
    ch_std[ch_std == 0] = 1.0
    data_z = (data_interictal - ch_mean) / ch_std

    jac = estimate_jacobian(data_z, window_size=window_samples,
                            step_size=step_samples, regularization=regularization)
    ep = detect_exceptional_points(jac)
    real_mean = float(np.mean(ep.min_eigenvalue_gaps))
    real_std = float(np.std(ep.min_eigenvalue_gaps))

    surr_means = []
    for _ in range(n_surrogates):
        surr = phase_randomize_surrogate(data_interictal, rng)
        surr_z = (surr - ch_mean) / ch_std
        try:
            jac_s = estimate_jacobian(surr_z, window_size=window_samples,
                                      step_size=step_samples, regularization=regularization)
            ep_s = detect_exceptional_points(jac_s)
            surr_means.append(float(np.mean(ep_s.min_eigenvalue_gaps)))
        except (ValueError, np.linalg.LinAlgError):
            continue

    return {
        "surrogate_mean_spacings": surr_means,
        "surrogate_std_spacing": float(np.std(surr_means)) if surr_means else float("nan"),
        "real_mean_spacing": real_mean,
        "real_std_spacing": real_std,
    }


def partial_correlation_control(
    time_sec: np.ndarray,
    spacing_z: np.ndarray,
    alpha_power_z: np.ndarray,
    delta_power_z: np.ndarray,
    preictal_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Test whether spacing-time relationship survives spectral power control.

    Computes partial correlation of spacing vs time, controlling for
    alpha power and delta power changes. This rules out spectral
    confounds in the pre-ictal spacing narrowing.

    Parameters
    ----------
    time_sec : np.ndarray
    spacing_z : np.ndarray
    alpha_power_z : np.ndarray
    delta_power_z : np.ndarray
    preictal_mask : np.ndarray or None
        If provided, restrict to pre-ictal windows only.

    Returns
    -------
    dict with keys: r_raw, p_raw, r_partial, p_partial, n_windows, method.
    """
    from cmcc.analysis.ep_advanced import _effective_n, _adjusted_correlation_p

    if preictal_mask is not None:
        t = time_sec[preictal_mask]
        s = spacing_z[preictal_mask]
        a = alpha_power_z[preictal_mask]
        d = delta_power_z[preictal_mask]
    else:
        t = time_sec
        s = spacing_z
        a = alpha_power_z
        d = delta_power_z

    valid = np.isfinite(t) & np.isfinite(s) & np.isfinite(a) & np.isfinite(d)
    t = t[valid]
    s = s[valid]
    a = a[valid]
    d = d[valid]

    n = len(t)
    if n < 5:
        return {
            "r_raw": float("nan"),
            "p_raw": float("nan"),
            "r_partial": float("nan"),
            "p_partial": float("nan"),
            "n_windows": n,
            "method": "partial_correlation",
        }

    from scipy import stats as sp_stats

    r_raw, _ = sp_stats.pearsonr(t, s)
    n_eff_raw = _effective_n(t, s)
    p_raw = _adjusted_correlation_p(r_raw, n_eff_raw)

    X = np.column_stack([t, a, d])
    X_mean = X.mean(axis=0)
    X_centered = X - X_mean
    s_mean = s.mean()
    s_centered = s - s_mean

    try:
        XtX = X_centered.T @ X_centered
        Xts = X_centered.T @ s_centered
        beta = np.linalg.solve(XtX, Xts)
        s_resid = s_centered - X_centered @ beta
        t_resid = X_centered[:, 0] - X_centered[:, 1:] @ np.linalg.solve(
            X_centered[:, 1:].T @ X_centered[:, 1:],
            X_centered[:, 1:].T @ X_centered[:, 0],
        )

        ss_s = np.sum(s_resid ** 2)
        ss_t = np.sum(t_resid ** 2)
        if ss_s > 0 and ss_t > 0:
            r_partial = float(np.sum(t_resid * s_resid) / np.sqrt(ss_t * ss_s))
        else:
            r_partial = float("nan")
    except np.linalg.LinAlgError:
        r_partial = float("nan")

    n_eff_partial = max(3, n_eff_raw - 2)
    p_partial = _adjusted_correlation_p(r_partial, n_eff_partial)

    return {
        "r_raw": float(r_raw),
        "p_raw": float(p_raw),
        "r_partial": float(r_partial),
        "p_partial": float(p_partial),
        "n_windows": n,
        "n_eff": n_eff_raw,
        "method": "partial_correlation",
    }
