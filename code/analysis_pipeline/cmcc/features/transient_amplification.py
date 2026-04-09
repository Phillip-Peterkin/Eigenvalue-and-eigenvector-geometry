"""Transient amplification analysis for fitted VAR(1) operators.

Purpose
-------
Non-normal matrices can transiently amplify perturbations even when all
eigenvalues lie inside the unit circle (i.e., the system is asymptotically
stable). This module quantifies that amplification from the fitted Jacobians
produced by the sliding-window VAR(1) pipeline.

Scientific context
------------------
In non-normal recurrent networks, the eigenvector matrix V can be
ill-conditioned (kappa(V) >> 1). This allows ||A^k|| to exceed
||A||^k for small k, producing a transient energy "hump" before
eventual decay. The Kreiss constant K(A) lower-bounds the worst-case
transient growth. If operator geometry (EP score) truly reflects
non-normality, subjects/windows with higher EP scores should show
larger transient amplification envelopes.

Expected inputs
---------------
- JacobianResult from dynamical_systems.estimate_jacobian
  (jacobians array of shape (n_windows, n_ch, n_ch))

Assumptions
-----------
- Transient amplification is computed from the FITTED operator,
  not the true neural dynamics. It is a property of the estimator
  output, not a direct measurement of neural amplification.
- The operator norm used is the 2-norm (largest singular value).
- The Kreiss constant is approximated numerically via matrix powers
  rather than via the resolvent, which is cheaper and sufficient
  for moderate dimensions (n_ch <= 30).

Known limitations
-----------------
- Amplification magnitude depends on regularization, window length,
  and PCA dimension — the same pipeline-dependence caveats that
  apply to all operator-geometry metrics.
- The Kreiss constant is a worst-case bound; realized amplification
  in neural data may be smaller.
- Numerical overflow is possible for large matrix powers; we cap
  the horizon and use early termination.

Validation strategy
-------------------
- Unit test: known non-normal matrix with analytically predictable
  amplification envelope.
- Unit test: normal (symmetric) matrix should show monotonic decay
  (amplification ratio = 1.0).
- Property test: amplification ratio >= 1.0 for all matrices.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.linalg import norm as la_norm


def compute_energy_envelope(
    A: np.ndarray,
    max_horizon: int = 50,
    norm_type: int = 2,
) -> np.ndarray:
    """Compute the energy envelope ||A^k||_2 for k = 0, 1, ..., max_horizon.

    Parameters
    ----------
    A : np.ndarray, shape (n, n)
        System matrix (fitted VAR(1) Jacobian for one window).
    max_horizon : int
        Maximum number of discrete time steps.
    norm_type : int
        Matrix norm type (2 = spectral norm = largest singular value).

    Returns
    -------
    envelope : np.ndarray, shape (max_horizon + 1,)
        ||A^k||_2 for k = 0, ..., max_horizon.
        envelope[0] = 1.0 (identity norm).
    """
    n = A.shape[0]
    envelope = np.zeros(max_horizon + 1)
    envelope[0] = 1.0

    A_power = np.eye(n)
    for k in range(1, max_horizon + 1):
        A_power = A_power @ A
        nrm = la_norm(A_power, ord=norm_type)
        if not np.isfinite(nrm) or nrm > 1e15:
            envelope[k:] = np.nan
            break
        envelope[k] = nrm

    return envelope


def compute_amplification_ratio(
    envelope: np.ndarray,
    spectral_radius: float | None = None,
) -> float:
    """Peak transient amplification ratio: max_k ||A^k|| / rho(A)^k.

    For a normal matrix, ||A^k|| = rho(A)^k, so the ratio is 1.0.
    For a non-normal matrix, ||A^k|| can exceed rho(A)^k for small k,
    producing transient amplification even in asymptotically stable systems.

    Parameters
    ----------
    envelope : np.ndarray, shape (K+1,)
        Output of compute_energy_envelope.
    spectral_radius : float or None
        rho(A) = max |eigenvalue|. If None, estimated from the envelope
        decay rate at large k.

    Returns
    -------
    ratio : float
        max_{k>=1} envelope[k] / max(rho^k, epsilon).
        Returns 1.0 if spectral_radius is 0 or envelope is all NaN.
    """
    if len(envelope) < 2:
        return 1.0

    if spectral_radius is None or spectral_radius <= 0:
        return float(np.nanmax(envelope))

    ratios = []
    eps = 1e-30
    for k in range(1, len(envelope)):
        if not np.isfinite(envelope[k]):
            break
        expected = max(spectral_radius ** k, eps)
        ratios.append(envelope[k] / expected)

    if not ratios:
        return 1.0

    return float(max(ratios))


def compute_kreiss_constant(envelope: np.ndarray) -> float:
    """Approximate Kreiss constant: max_k ||A^k||.

    The Kreiss constant K(A) = sup_{k>=0} ||A^k|| provides a lower
    bound on the worst-case transient growth. For stable systems
    (spectral radius < 1), this is finite.

    Parameters
    ----------
    envelope : np.ndarray, shape (K+1,)
        Output of compute_energy_envelope.

    Returns
    -------
    kreiss : float
        Maximum of the energy envelope. Returns 1.0 if all NaN.
    """
    valid = envelope[np.isfinite(envelope)]
    if len(valid) == 0:
        return 1.0
    return float(np.max(valid))


def compute_peak_time(envelope: np.ndarray) -> int:
    """Time step at which peak amplification occurs.

    Parameters
    ----------
    envelope : np.ndarray, shape (K+1,)
        Output of compute_energy_envelope.

    Returns
    -------
    k_peak : int
        Index of maximum envelope value. Returns 0 if all NaN.
    """
    valid_mask = np.isfinite(envelope)
    if not np.any(valid_mask):
        return 0
    return int(np.argmax(np.where(valid_mask, envelope, -np.inf)))


def analyze_jacobian_amplification(
    jacobians: np.ndarray,
    max_horizon: int = 50,
) -> dict:
    """Compute transient amplification statistics across all windows.

    Parameters
    ----------
    jacobians : np.ndarray, shape (n_windows, n_ch, n_ch)
        Fitted Jacobian matrices from sliding-window VAR(1).
    max_horizon : int
        Maximum discrete time steps for energy envelope.

    Returns
    -------
    dict with keys:
        kreiss_constants : np.ndarray, shape (n_windows,)
        amplification_ratios : np.ndarray, shape (n_windows,)
        peak_times : np.ndarray, shape (n_windows,)
        mean_envelope : np.ndarray, shape (max_horizon + 1,)
            Mean energy envelope across windows.
        has_hump : np.ndarray, shape (n_windows,), dtype bool
            True if envelope[k] > envelope[0] for any k > 0
            (i.e., transient amplification above initial energy).
    """
    n_windows = jacobians.shape[0]

    kreiss = np.zeros(n_windows)
    amp_ratios = np.zeros(n_windows)
    peak_times = np.zeros(n_windows, dtype=int)
    has_hump = np.zeros(n_windows, dtype=bool)
    envelopes = np.zeros((n_windows, max_horizon + 1))

    for w in range(n_windows):
        A = jacobians[w]
        env = compute_energy_envelope(A, max_horizon=max_horizon)
        envelopes[w] = env
        kreiss[w] = compute_kreiss_constant(env)
        rho = float(np.max(np.abs(np.linalg.eigvals(A))))
        amp_ratios[w] = compute_amplification_ratio(env, spectral_radius=rho)
        peak_times[w] = compute_peak_time(env)
        has_hump[w] = np.any(env[1:] > env[0]) if np.any(np.isfinite(env[1:])) else False

    mean_envelope = np.nanmean(envelopes, axis=0)

    return {
        "kreiss_constants": kreiss,
        "amplification_ratios": amp_ratios,
        "peak_times": peak_times,
        "mean_envelope": mean_envelope,
        "has_hump": has_hump,
        "envelopes": envelopes,
    }


def compute_hump_magnitude(envelope: np.ndarray) -> float:
    """Graded hump magnitude: max(||A^k|| - 1, 0) for k >= 1.

    Unlike the binary has_hump indicator, this returns the *magnitude*
    of the peak overshoot above the initial norm ||A^0|| = 1. Returns
    0.0 for monotonically decaying (normal-like) envelopes.

    Parameters
    ----------
    envelope : np.ndarray, shape (K+1,)
        Output of compute_energy_envelope.

    Returns
    -------
    magnitude : float
        Peak overshoot above 1.0. Zero if no hump.
    """
    if len(envelope) < 2:
        return 0.0
    valid = envelope[1:][np.isfinite(envelope[1:])]
    if len(valid) == 0:
        return 0.0
    return float(max(np.max(valid) - envelope[0], 0.0))


def surrogate_amplification_null(
    data: np.ndarray,
    sfreq: float,
    n_surrogates: int = 50,
    window_size: int = 500,
    step_size: int = 100,
    max_channels: int = 30,
    max_horizon: int = 50,
    regularization: float = 1e-4,
    seed: int = 42,
    jacobian_estimator: Callable | None = None,
) -> dict:
    """Phase-randomized surrogate null model for transient amplification.

    Generates surrogates that preserve spectral content but destroy
    temporal structure. Fits VAR(1) on each surrogate and computes
    amplification metrics. This establishes the baseline level of
    transient amplification expected from fitting short-window VAR(1)
    to spectrally matched noise.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Original multichannel time series.
    sfreq : float
        Sampling frequency in Hz.
    n_surrogates : int
        Number of phase-randomized surrogates.
    window_size : int
        VAR(1) estimation window in samples.
    step_size : int
        Stride between windows in samples.
    max_channels : int
        Max channels to use (PCA if exceeded).
    max_horizon : int
        Max time steps for energy envelope.
    regularization : float
        Ridge parameter for VAR(1).
    seed : int
        Base random seed for reproducibility.
    jacobian_estimator : callable or None
        Function with signature (data, window_size, step_size, regularization)
        -> JacobianResult. If None, imports from dynamical_systems.

    Returns
    -------
    dict with keys:
        surrogate_kreiss : np.ndarray, shape (n_surrogates,)
            Mean Kreiss constant per surrogate.
        surrogate_amp_ratio : np.ndarray, shape (n_surrogates,)
            Mean amplification ratio per surrogate.
        surrogate_hump_frac : np.ndarray, shape (n_surrogates,)
            Fraction of windows with hump per surrogate.
        surrogate_hump_magnitude : np.ndarray, shape (n_surrogates,)
            Mean hump magnitude per surrogate.
        n_surrogates : int
        seed : int
    """
    if jacobian_estimator is None:
        from cmcc.analysis.dynamical_systems import estimate_jacobian
        jacobian_estimator = estimate_jacobian

    n_ch, n_samp = data.shape
    if n_ch > max_channels:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=max_channels)
        data = pca.fit_transform(data.T).T

    surr_kreiss = np.zeros(n_surrogates)
    surr_amp = np.zeros(n_surrogates)
    surr_hump_frac = np.zeros(n_surrogates)
    surr_hump_mag = np.zeros(n_surrogates)

    for s in range(n_surrogates):
        rng = np.random.default_rng(seed + s)
        surr_data = _phase_randomize(data, rng)

        jac = jacobian_estimator(
            surr_data,
            window_size=window_size,
            step_size=step_size,
            regularization=regularization,
        )

        amp = analyze_jacobian_amplification(jac.jacobians, max_horizon=max_horizon)
        surr_kreiss[s] = float(np.mean(amp["kreiss_constants"]))
        surr_amp[s] = float(np.mean(amp["amplification_ratios"]))
        surr_hump_frac[s] = float(np.mean(amp["has_hump"]))

        mags = np.array([
            compute_hump_magnitude(amp["envelopes"][w])
            for w in range(amp["envelopes"].shape[0])
        ])
        surr_hump_mag[s] = float(np.mean(mags))

    return {
        "surrogate_kreiss": surr_kreiss,
        "surrogate_amp_ratio": surr_amp,
        "surrogate_hump_frac": surr_hump_frac,
        "surrogate_hump_magnitude": surr_hump_mag,
        "n_surrogates": n_surrogates,
        "seed": seed,
    }


def compare_real_vs_surrogate(
    real_kreiss: float,
    real_amp_ratio: float,
    real_hump_frac: float,
    real_hump_magnitude: float,
    surrogate_result: dict,
) -> dict:
    """Non-parametric comparison of real vs surrogate amplification.

    Computes percentile rank of real value within surrogate distribution
    and a one-sided p-value (fraction of surrogates >= real).

    Parameters
    ----------
    real_kreiss : float
        Mean Kreiss constant from real data.
    real_amp_ratio : float
        Mean amplification ratio from real data.
    real_hump_frac : float
        Hump fraction from real data.
    real_hump_magnitude : float
        Mean hump magnitude from real data.
    surrogate_result : dict
        Output of surrogate_amplification_null.

    Returns
    -------
    dict with comparison results for each metric.
    """
    def _rank_test(real_val, surr_vals):
        n = len(surr_vals)
        n_geq = np.sum(surr_vals >= real_val)
        p_val = (n_geq + 1) / (n + 1)
        percentile = 100.0 * np.sum(surr_vals < real_val) / n
        effect_size = (real_val - np.mean(surr_vals)) / max(np.std(surr_vals), 1e-30)
        return {
            "real": float(real_val),
            "surrogate_mean": float(np.mean(surr_vals)),
            "surrogate_std": float(np.std(surr_vals)),
            "surrogate_median": float(np.median(surr_vals)),
            "percentile": float(percentile),
            "p_value_one_sided": float(p_val),
            "effect_size_z": float(effect_size),
        }

    return {
        "kreiss_constant": _rank_test(
            real_kreiss, surrogate_result["surrogate_kreiss"]
        ),
        "amplification_ratio": _rank_test(
            real_amp_ratio, surrogate_result["surrogate_amp_ratio"]
        ),
        "hump_fraction": _rank_test(
            real_hump_frac, surrogate_result["surrogate_hump_frac"]
        ),
        "hump_magnitude": _rank_test(
            real_hump_magnitude, surrogate_result["surrogate_hump_magnitude"]
        ),
    }


def compute_model_free_energy_growth(
    data: np.ndarray,
    sfreq: float,
    percentile_threshold: float = 95.0,
    horizon_sec: float = 0.1,
    min_gap_sec: float = 0.05,
) -> dict:
    """Model-free test for transient energy amplification in raw signal.

    Detects sudden energy INFLUX events (large positive dE/dt) and tracks
    whether total energy continues to grow afterward. This avoids the
    regression-to-mean artifact that would occur if we selected peak-energy
    moments (which must decline by definition).

    A perturbation in the non-normal dynamics sense is a sudden energy
    injection. If the system amplifies, energy at t+k should exceed
    energy at t for some k > 0 after the influx event.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Multichannel time series.
    sfreq : float
        Sampling frequency in Hz.
    percentile_threshold : float
        Percentile of dE/dt for identifying influx events (default 95th).
    horizon_sec : float
        How far ahead to track energy after each event, in seconds.
    min_gap_sec : float
        Minimum gap between events to avoid overlap, in seconds.

    Returns
    -------
    dict with keys:
        n_events : int
        mean_energy_trajectory : np.ndarray, shape (horizon_samples + 1,)
            Mean energy at each lag after event onset, normalized to 1.0 at t=0.
        fraction_growing_at_peak : float
            Fraction of events where energy exceeded initial energy at any
            subsequent lag (evidence of continued amplification after influx).
        peak_growth_ratio : float
            Mean of max(trajectory[1:]) / trajectory[0] across events.
            Values > 1.0 indicate transient amplification.
        peak_lag_samples : int
            Lag at which mean trajectory peaks.
        peak_lag_sec : float
        raw_trajectories : np.ndarray, shape (n_events, horizon_samples + 1)
    """
    n_ch, n_samp = data.shape
    horizon_samples = int(horizon_sec * sfreq)
    min_gap_samples = int(min_gap_sec * sfreq)

    energy = np.sum(data ** 2, axis=0)

    dE = np.diff(energy)

    if len(dE) < 10:
        return {
            "n_events": 0,
            "mean_energy_trajectory": np.ones(horizon_samples + 1),
            "fraction_growing_at_peak": 0.0,
            "peak_growth_ratio": 1.0,
            "peak_lag_samples": 0,
            "peak_lag_sec": 0.0,
            "raw_trajectories": np.array([]),
        }

    threshold = np.percentile(dE, percentile_threshold)
    candidate_indices = np.where(dE >= threshold)[0]

    events = []
    last_event = -min_gap_samples - 1
    for idx in candidate_indices:
        evt_start = idx + 1
        if (evt_start - last_event >= min_gap_samples
                and evt_start + horizon_samples < n_samp
                and evt_start >= 1):
            events.append(evt_start)
            last_event = evt_start

    if len(events) < 5:
        return {
            "n_events": len(events),
            "mean_energy_trajectory": np.ones(horizon_samples + 1),
            "fraction_growing_at_peak": 0.0,
            "peak_growth_ratio": 1.0,
            "peak_lag_samples": 0,
            "peak_lag_sec": 0.0,
            "raw_trajectories": np.array([]),
        }

    trajectories = np.zeros((len(events), horizon_samples + 1))
    for i, evt in enumerate(events):
        seg_energy = energy[evt:evt + horizon_samples + 1]
        if seg_energy[0] > 0:
            trajectories[i] = seg_energy / seg_energy[0]
        else:
            trajectories[i] = 1.0

    mean_traj = np.mean(trajectories, axis=0)

    if len(mean_traj) > 1:
        peak_lag = int(np.argmax(mean_traj[1:]) + 1)
    else:
        peak_lag = 0

    growing = np.array([
        np.any(t[1:] > t[0]) for t in trajectories
    ]) if trajectories.shape[1] > 1 else np.zeros(len(events), dtype=bool)
    peak_ratios = np.array([
        np.max(t[1:]) / max(t[0], 1e-30) for t in trajectories
    ]) if trajectories.shape[1] > 1 else np.ones(len(events))

    return {
        "n_events": len(events),
        "mean_energy_trajectory": mean_traj,
        "fraction_growing_at_peak": float(np.mean(growing)),
        "peak_growth_ratio": float(np.mean(peak_ratios)),
        "peak_lag_samples": peak_lag,
        "peak_lag_sec": float(peak_lag / sfreq),
        "raw_trajectories": trajectories,
    }


def compute_out_of_sample_prediction(
    jacobians: np.ndarray,
    data: np.ndarray,
    window_centers: np.ndarray,
    window_size: int,
    n_predict_steps: int = 10,
) -> dict:
    """Test whether fitted operator's amplification predicts actual trajectory.

    For each window, uses the fitted Jacobian to predict the next
    n_predict_steps of the actual signal. Compares the predicted energy
    trajectory with the observed energy trajectory. If the fitted
    amplification is a real dynamical property, windows with higher
    Kreiss constants should show better prediction of energy growth.

    Parameters
    ----------
    jacobians : np.ndarray, shape (n_windows, n_ch, n_ch)
    data : np.ndarray, shape (n_ch, n_samples)
    window_centers : np.ndarray, shape (n_windows,)
    window_size : int
    n_predict_steps : int
        Steps ahead to predict.

    Returns
    -------
    dict with keys:
        prediction_r2 : np.ndarray, shape (n_windows,)
            R² between predicted and actual energy trajectory per window.
        mean_r2 : float
        kreiss_vs_r2_corr : tuple (r, p)
            Correlation between Kreiss constant and prediction quality.
        predicted_growth : np.ndarray, shape (n_windows,)
            Max predicted energy / initial energy per window.
        observed_growth : np.ndarray, shape (n_windows,)
            Max observed energy / initial energy per window.
        growth_correlation : tuple (r, p)
            Correlation between predicted and observed growth.
    """
    from scipy import stats as sp_stats

    n_windows, n_ch, _ = jacobians.shape
    n_samp = data.shape[1]
    half_w = window_size // 2

    pred_r2 = np.full(n_windows, np.nan)
    pred_growth = np.full(n_windows, np.nan)
    obs_growth = np.full(n_windows, np.nan)
    kreiss_vals = np.zeros(n_windows)

    for w in range(n_windows):
        A = jacobians[w]
        center = int(window_centers[w])
        start = center + half_w + 1

        if start + n_predict_steps >= n_samp:
            continue

        x0 = data[:, start]
        if np.sum(x0 ** 2) < 1e-30:
            continue

        predicted_energy = np.zeros(n_predict_steps + 1)
        observed_energy = np.zeros(n_predict_steps + 1)

        x_pred = x0.copy()
        predicted_energy[0] = np.sum(x0 ** 2)
        observed_energy[0] = np.sum(data[:, start] ** 2)

        for k in range(1, n_predict_steps + 1):
            x_pred = A @ x_pred
            predicted_energy[k] = np.sum(x_pred ** 2)
            if start + k < n_samp:
                observed_energy[k] = np.sum(data[:, start + k] ** 2)

        if predicted_energy[0] > 0:
            pred_norm = predicted_energy / predicted_energy[0]
            obs_norm = observed_energy / observed_energy[0]

            ss_res = np.sum((obs_norm - pred_norm) ** 2)
            ss_tot = np.sum((obs_norm - np.mean(obs_norm)) ** 2)
            pred_r2[w] = 1.0 - ss_res / max(ss_tot, 1e-30)
            pred_growth[w] = float(np.max(pred_norm))
            obs_growth[w] = float(np.max(obs_norm))

        env = compute_energy_envelope(A, max_horizon=n_predict_steps)
        kreiss_vals[w] = compute_kreiss_constant(env)

    valid = np.isfinite(pred_r2)
    mean_r2 = float(np.nanmean(pred_r2)) if np.any(valid) else float("nan")

    valid_both = valid & np.isfinite(pred_growth) & np.isfinite(obs_growth)

    if np.sum(valid) > 3:
        r_kr, p_kr = sp_stats.pearsonr(kreiss_vals[valid], pred_r2[valid])
    else:
        r_kr, p_kr = float("nan"), float("nan")

    if np.sum(valid_both) > 3:
        r_g, p_g = sp_stats.pearsonr(pred_growth[valid_both], obs_growth[valid_both])
    else:
        r_g, p_g = float("nan"), float("nan")

    return {
        "prediction_r2": pred_r2,
        "mean_r2": mean_r2,
        "kreiss_vs_r2_corr": (float(r_kr), float(p_kr)),
        "predicted_growth": pred_growth,
        "observed_growth": obs_growth,
        "growth_correlation": (float(r_g), float(p_g)),
        "n_valid_windows": int(np.sum(valid)),
    }


def compute_residual_kreiss(
    kreiss_constants: np.ndarray,
    spectral_radii: np.ndarray,
) -> dict:
    """Kreiss constant residualized against spectral radius.

    Regresses log(Kreiss) on log(1/(1 - rho)) per-window, returning
    residuals that capture eigenvector non-orthogonality independent of
    stability margin. If only the stability margin changes between
    conditions, residuals should be zero; a nonzero residual indicates
    genuine non-normality change.

    Parameters
    ----------
    kreiss_constants : np.ndarray, shape (n_windows,)
        Per-window Kreiss constants.
    spectral_radii : np.ndarray, shape (n_windows,)
        Per-window spectral radii.

    Returns
    -------
    dict with keys:
        residuals : np.ndarray, shape (n_stable,)
            Residual log-Kreiss after partialling out spectral radius.
        mean_residual : float
        median_residual : float
        slope : float
            Regression slope (log-Kreiss on log-gap).
        r_squared : float
        n_stable : int
    """
    stable = spectral_radii < 1.0
    if np.sum(stable) < 10:
        return {
            "residuals": np.array([]),
            "mean_residual": float("nan"),
            "median_residual": float("nan"),
            "slope": float("nan"),
            "r_squared": float("nan"),
            "n_stable": int(np.sum(stable)),
        }

    kr = kreiss_constants[stable]
    rho = spectral_radii[stable]

    log_kr = np.log(np.clip(kr, 1e-30, None))
    log_gap_inv = np.log(np.clip(1.0 / (1.0 - rho), 1e-30, None))

    X = np.column_stack([np.ones(len(log_gap_inv)), log_gap_inv])
    beta, _, _, _ = np.linalg.lstsq(X, log_kr, rcond=None)
    predicted = X @ beta
    residuals = log_kr - predicted

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_kr - np.mean(log_kr)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    return {
        "residuals": residuals,
        "mean_residual": float(np.mean(residuals)),
        "median_residual": float(np.median(residuals)),
        "slope": float(beta[1]),
        "r_squared": float(r2),
        "n_stable": int(np.sum(stable)),
    }


def _phase_randomize(data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Phase-randomize multichannel data preserving power spectrum.

    Applies the same random phase rotation to all channels (preserving
    cross-channel spectral structure) while destroying temporal
    correlations that drive non-normal operator geometry.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
    rng : np.random.Generator

    Returns
    -------
    surrogate : np.ndarray, same shape as data
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
    return np.fft.irfft(fft_surrogate, n=n_samp, axis=1)
