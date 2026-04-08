"""Dynamical systems analysis: Jacobian estimation, exceptional point
detection, and chiral phase flux measurement.

Maps multi-channel neural data into a high-dimensional state space,
estimates the system Jacobian via windowed VAR(1) regression, detects
eigenvalue degeneracies (exceptional points) where both eigenvalues
and eigenvectors coalesce, and measures the directional (chiral) phase
rotation of coalescing eigenvectors.

Exceptional points (EPs) occur in non-Hermitian systems — neural
dynamics are dissipative, hence the effective Jacobian is generically
non-Hermitian. Near an EP, the system shows extreme sensitivity to
perturbation (square-root topology in parameter space), which has
implications for neural criticality distinct from branching-process
criticality.

Chirality: When eigenvectors coalesce near an EP, encircling the EP
in parameter space causes eigenvalue swapping with a definite
topological winding. In the time domain, this manifests as a
consistent (chiral) phase rotation of the coalescing eigenvectors.
The hypothesis is that conscious processing produces chiral
(directed) phase rotation, while noise/unconsciousness produces
random phase jitter even near an EP.

Units: state vectors are z-scored channel amplitudes (dimensionless).
Time is in samples; convert to seconds via 1/sfreq.
Phase angles are in radians.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import linalg


@dataclass
class JacobianResult:
    """Result of windowed Jacobian estimation.

    Attributes
    ----------
    jacobians : np.ndarray, shape (n_windows, n_channels, n_channels)
        Estimated Jacobian (VAR(1) coefficient matrix) per window.
    eigenvalues : np.ndarray, shape (n_windows, n_channels)
        Eigenvalues of each Jacobian, sorted by descending magnitude.
    eigenvectors : np.ndarray, shape (n_windows, n_channels, n_channels)
        Right eigenvector matrices (columns = eigenvectors).
    window_centers : np.ndarray, shape (n_windows,)
        Sample index of each window center.
    spectral_radius : np.ndarray, shape (n_windows,)
        Max |eigenvalue| per window (stability indicator).
    condition_numbers : np.ndarray, shape (n_windows,)
        Condition number of eigenvector matrix per window.
        Diverges near exceptional points.
    residual_variance : np.ndarray, shape (n_windows,)
        Fraction of variance unexplained by VAR(1) fit per window.
    regularization : float
        Ridge regularization parameter used.
    """

    jacobians: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    window_centers: np.ndarray
    spectral_radius: np.ndarray
    condition_numbers: np.ndarray
    residual_variance: np.ndarray
    regularization: float


@dataclass
class ExceptionalPointResult:
    """Result of exceptional point detection.

    Attributes
    ----------
    min_eigenvalue_gaps : np.ndarray, shape (n_windows,)
        Minimum |lambda_i - lambda_j| across all eigenvalue pairs per window.
    gap_pair_indices : np.ndarray, shape (n_windows, 2)
        Indices (i, j) of the closest eigenvalue pair per window.
    eigenvector_overlaps : np.ndarray, shape (n_windows,)
        |<v_i|v_j>| for the closest eigenvalue pair. Approaches 1.0 at an EP.
    petermann_factors : np.ndarray, shape (n_windows,)
        Petermann excess noise factor for the closest pair.
        K = 1 / |<v_L|v_R>|^2 for a given mode. Diverges at an EP.
    ep_scores : np.ndarray, shape (n_windows,)
        Composite EP proximity score: high when gap is small AND
        eigenvectors are nearly parallel. Range [0, inf).
    ep_candidates : list[dict]
        Windows where ep_score exceeds threshold, with metadata.
    threshold : float
        EP score threshold used for candidate detection.
    """

    min_eigenvalue_gaps: np.ndarray
    gap_pair_indices: np.ndarray
    eigenvector_overlaps: np.ndarray
    petermann_factors: np.ndarray
    ep_scores: np.ndarray
    ep_candidates: list[dict] = field(default_factory=list)
    threshold: float = 0.0


@dataclass
class NonHermitianDecomposition:
    """Result of Hermitian / anti-Hermitian decomposition of the Jacobian.

    For a real Jacobian J, the symmetric part S = (J + J^T)/2 captures
    mutual (undirected) coupling, and the anti-symmetric part A = (J - J^T)/2
    captures directional (chiral) neural flow. The ratio ||A||_F / ||S||_F
    quantifies genuine non-Hermiticity — high values indicate active,
    directed dynamics that cannot arise from passive noise or symmetric
    oscillations.

    Attributes
    ----------
    asymmetry_ratio : np.ndarray, shape (n_windows,)
        ||A||_F / ||S||_F per window. 0 = purely symmetric, >1 = anti-
        symmetric dominates.
    symmetric_power : np.ndarray, shape (n_windows,)
        ||S||_F^2 per window (Frobenius norm squared of symmetric part).
    antisymmetric_power : np.ndarray, shape (n_windows,)
        ||A||_F^2 per window (Frobenius norm squared of anti-symmetric part).
    asymmetry_eigenvalues : np.ndarray, shape (n_windows, n_channels)
        Eigenvalues of A (purely imaginary for real J; stored as imaginary
        parts). These give the intrinsic rotation frequencies of the
        directional component.
    mean_asymmetry_ratio : float
        Time-averaged asymmetry ratio.
    std_asymmetry_ratio : float
        Temporal variability of asymmetry ratio.
    max_rotation_frequency : np.ndarray, shape (n_windows,)
        Max |imag eigenvalue of A| per window — the fastest directional
        rotation rate in the anti-symmetric component.
    kurtosis_asymmetry_ratio : float
        Excess kurtosis of asymmetry_ratio time series. High values
        indicate heavy-tailed bursts of directional flow.
    max_asymmetry_ratio : float
        Maximum asymmetry ratio across all windows.
    p95_asymmetry_ratio : float
        95th percentile of asymmetry ratio.
    p99_asymmetry_ratio : float
        99th percentile of asymmetry ratio.
    dynamic_range : float
        p99 / p01 ratio — how much the asymmetry ratio fluctuates.
        High values indicate the system can transiently enter strongly
        directional states even if the mean is low.
    """

    asymmetry_ratio: np.ndarray
    symmetric_power: np.ndarray
    antisymmetric_power: np.ndarray
    asymmetry_eigenvalues: np.ndarray
    mean_asymmetry_ratio: float
    std_asymmetry_ratio: float
    max_rotation_frequency: np.ndarray
    kurtosis_asymmetry_ratio: float
    max_asymmetry_ratio: float
    p95_asymmetry_ratio: float
    p99_asymmetry_ratio: float
    dynamic_range: float


def decompose_jacobian_hermiticity(
    jac_result: JacobianResult,
) -> NonHermitianDecomposition:
    """Decompose each windowed Jacobian into symmetric and anti-symmetric parts.

    J = S + A where S = (J + J^T)/2 (symmetric) and A = (J - J^T)/2
    (anti-symmetric / skew-symmetric).

    For a real matrix:
    - S captures undirected mutual coupling (noise, passive oscillations).
    - A captures directional flow (active, chiral neural dynamics).
    - ||A||_F / ||S||_F is a gauge-invariant measure of non-Hermiticity
      that does not depend on eigenvector tracking or eigenvalue sorting.

    Parameters
    ----------
    jac_result : JacobianResult
        Output of estimate_jacobian.

    Returns
    -------
    NonHermitianDecomposition
    """
    n_windows, n_ch, _ = jac_result.jacobians.shape

    asymmetry_ratio = np.zeros(n_windows)
    sym_power = np.zeros(n_windows)
    asym_power = np.zeros(n_windows)
    asym_evals = np.zeros((n_windows, n_ch))
    max_rot_freq = np.zeros(n_windows)

    for w in range(n_windows):
        J = jac_result.jacobians[w]
        S = 0.5 * (J + J.T)
        A = 0.5 * (J - J.T)

        s_norm = np.linalg.norm(S, "fro")
        a_norm = np.linalg.norm(A, "fro")

        sym_power[w] = s_norm ** 2
        asym_power[w] = a_norm ** 2
        asymmetry_ratio[w] = a_norm / s_norm if s_norm > 0 else 0.0

        evals_A = linalg.eigvals(A)
        asym_evals[w] = np.sort(evals_A.imag)[::-1]
        max_rot_freq[w] = np.max(np.abs(evals_A.imag))

    from scipy.stats import kurtosis as sp_kurtosis
    p01 = float(np.percentile(asymmetry_ratio, 1))
    p95 = float(np.percentile(asymmetry_ratio, 95))
    p99 = float(np.percentile(asymmetry_ratio, 99))
    dyn_range = p99 / p01 if p01 > 0 else 0.0

    return NonHermitianDecomposition(
        asymmetry_ratio=asymmetry_ratio,
        symmetric_power=sym_power,
        antisymmetric_power=asym_power,
        asymmetry_eigenvalues=asym_evals,
        mean_asymmetry_ratio=float(np.mean(asymmetry_ratio)),
        std_asymmetry_ratio=float(np.std(asymmetry_ratio)),
        max_rotation_frequency=max_rot_freq,
        kurtosis_asymmetry_ratio=float(sp_kurtosis(asymmetry_ratio, fisher=True)),
        max_asymmetry_ratio=float(np.max(asymmetry_ratio)),
        p95_asymmetry_ratio=p95,
        p99_asymmetry_ratio=p99,
        dynamic_range=dyn_range,
    )


def estimate_jacobian(
    data: np.ndarray,
    window_size: int = 500,
    step_size: int = 100,
    regularization: float = 1e-4,
) -> JacobianResult:
    """Estimate time-varying Jacobian via windowed VAR(1) with ridge regression.

    The state vector x(t) is the vector of channel values at time t.
    We fit x(t+1) = A * x(t) + noise in each window, where A is
    the Jacobian of the discrete-time dynamical system.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Multi-channel time series (should be z-scored per channel).
    window_size : int
        Number of samples per estimation window. Must exceed n_channels
        for the regression to be well-conditioned.
    step_size : int
        Stride between consecutive windows.
    regularization : float
        Ridge parameter (lambda * I added to X @ X.T). Prevents
        ill-conditioning when channels are correlated.

    Returns
    -------
    JacobianResult
    """
    n_ch, n_samples = data.shape

    if window_size <= n_ch:
        raise ValueError(
            f"window_size ({window_size}) must exceed n_channels ({n_ch}) "
            f"for VAR(1) regression to be well-conditioned."
        )

    if n_samples < window_size + 1:
        raise ValueError(
            f"Data too short ({n_samples} samples) for window_size={window_size}."
        )

    starts = list(range(0, n_samples - window_size, step_size))
    n_windows = len(starts)

    jacobians = np.zeros((n_windows, n_ch, n_ch))
    eigenvalues = np.zeros((n_windows, n_ch), dtype=complex)
    eigenvectors = np.zeros((n_windows, n_ch, n_ch), dtype=complex)
    window_centers = np.zeros(n_windows)
    spectral_radius = np.zeros(n_windows)
    condition_numbers = np.zeros(n_windows)
    residual_variance = np.zeros(n_windows)

    reg_matrix = regularization * np.eye(n_ch)

    for i, start in enumerate(starts):
        end = start + window_size
        X = data[:, start:end - 1]
        Y = data[:, start + 1:end]

        XXT = X @ X.T + reg_matrix
        YXT = Y @ X.T

        try:
            A = linalg.solve(XXT.T, YXT.T, assume_a="pos").T
        except linalg.LinAlgError:
            A = YXT @ np.linalg.pinv(XXT)

        jacobians[i] = A
        window_centers[i] = start + window_size / 2

        evals, evecs = linalg.eig(A)
        sort_idx = np.argsort(-np.abs(evals))
        eigenvalues[i] = evals[sort_idx]
        eigenvectors[i] = evecs[:, sort_idx]

        spectral_radius[i] = np.max(np.abs(evals))

        try:
            condition_numbers[i] = np.linalg.cond(evecs[:, sort_idx])
        except np.linalg.LinAlgError:
            condition_numbers[i] = np.inf

        Y_pred = A @ X
        total_var = np.sum(Y ** 2)
        resid_var = np.sum((Y - Y_pred) ** 2)
        residual_variance[i] = resid_var / total_var if total_var > 0 else 1.0

    return JacobianResult(
        jacobians=jacobians,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        window_centers=window_centers,
        spectral_radius=spectral_radius,
        condition_numbers=condition_numbers,
        residual_variance=residual_variance,
        regularization=regularization,
    )


def detect_exceptional_points(
    jac_result: JacobianResult,
    ep_threshold: float = 10.0,
) -> ExceptionalPointResult:
    """Detect exceptional point candidates from Jacobian eigenvalue spectra.

    An exceptional point occurs when two eigenvalues AND their eigenvectors
    coalesce. We detect this via:
    1. Minimum eigenvalue gap: min_{i!=j} |lambda_i - lambda_j|
    2. Eigenvector overlap: |<v_i|v_j>| for the closest pair (-> 1 at EP)
    3. Petermann factor: 1/|<v_L|v_R>|^2 (-> infinity at EP)
    4. Composite EP score: overlap / (gap + epsilon)

    Parameters
    ----------
    jac_result : JacobianResult
        Output of estimate_jacobian.
    ep_threshold : float
        Minimum EP score to flag a window as a candidate.

    Returns
    -------
    ExceptionalPointResult
    """
    n_windows = len(jac_result.eigenvalues)
    n_ch = jac_result.eigenvalues.shape[1]

    min_gaps = np.zeros(n_windows)
    gap_pairs = np.zeros((n_windows, 2), dtype=int)
    overlaps = np.zeros(n_windows)
    petermann = np.zeros(n_windows)
    ep_scores = np.zeros(n_windows)

    for w in range(n_windows):
        evals = jac_result.eigenvalues[w]
        evecs = jac_result.eigenvectors[w]

        best_gap = np.inf
        best_i, best_j = 0, 1

        for ii in range(min(n_ch, 20)):
            for jj in range(ii + 1, min(n_ch, 20)):
                gap = abs(evals[ii] - evals[jj])
                if gap < best_gap:
                    best_gap = gap
                    best_i, best_j = ii, jj

        min_gaps[w] = best_gap
        gap_pairs[w] = [best_i, best_j]

        v_i = evecs[:, best_i]
        v_j = evecs[:, best_j]
        norm_i = np.linalg.norm(v_i)
        norm_j = np.linalg.norm(v_j)
        if norm_i > 0 and norm_j > 0:
            overlap = abs(np.dot(np.conj(v_i), v_j)) / (norm_i * norm_j)
        else:
            overlap = 0.0
        overlaps[w] = overlap

        try:
            A = jac_result.jacobians[w]
            evals_l, evecs_l = linalg.eig(A.T)
            sort_l = np.argsort(-np.abs(evals_l))
            v_L = evecs_l[:, sort_l[best_i]]
            v_R = evecs[:, best_i]
            inner = abs(np.dot(np.conj(v_L), v_R))
            if inner > 0:
                petermann[w] = 1.0 / (inner ** 2)
            else:
                petermann[w] = np.inf
        except Exception:
            petermann[w] = np.nan

        epsilon = 1e-10
        ep_scores[w] = overlap / (best_gap + epsilon)

    candidates = []
    for w in range(n_windows):
        if ep_scores[w] >= ep_threshold:
            candidates.append({
                "window_idx": int(w),
                "window_center_sample": float(jac_result.window_centers[w]),
                "ep_score": float(ep_scores[w]),
                "eigenvalue_gap": float(min_gaps[w]),
                "eigenvector_overlap": float(overlaps[w]),
                "petermann_factor": float(petermann[w]),
                "spectral_radius": float(jac_result.spectral_radius[w]),
                "lambda_i": complex(jac_result.eigenvalues[w, gap_pairs[w, 0]]),
                "lambda_j": complex(jac_result.eigenvalues[w, gap_pairs[w, 1]]),
            })

    return ExceptionalPointResult(
        min_eigenvalue_gaps=min_gaps,
        gap_pair_indices=gap_pairs,
        eigenvector_overlaps=overlaps,
        petermann_factors=petermann,
        ep_scores=ep_scores,
        ep_candidates=candidates,
        threshold=ep_threshold,
    )


def compute_ep_proximity_timecourse(
    data: np.ndarray,
    sfreq: float,
    window_sec: float = 0.5,
    step_sec: float = 0.1,
    regularization: float = 1e-4,
    max_channels: int = 30,
    seed: int = 42,
) -> dict:
    """Full EP analysis pipeline: state space -> Jacobian -> EP detection.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Multi-channel neural data (e.g., high-gamma envelope).
    sfreq : float
        Sampling frequency in Hz.
    window_sec : float
        Jacobian estimation window in seconds.
    step_sec : float
        Step size in seconds.
    regularization : float
        Ridge parameter for VAR(1).
    max_channels : int
        Subsample to this many channels if n_channels exceeds it.
    seed : int
        Random seed for channel subsampling.

    Returns
    -------
    dict with keys:
        jac_result : JacobianResult
        ep_result : ExceptionalPointResult
        n_channels_used : int
        sfreq : float
        window_sec : float
        step_sec : float
    """
    n_ch, n_samples = data.shape

    rng = np.random.default_rng(seed)
    if n_ch > max_channels:
        ch_idx = np.sort(rng.choice(n_ch, max_channels, replace=False))
        data_sub = data[ch_idx]
    else:
        ch_idx = np.arange(n_ch)
        data_sub = data

    ch_mean = data_sub.mean(axis=1, keepdims=True)
    ch_std = data_sub.std(axis=1, keepdims=True)
    ch_std[ch_std == 0] = 1.0
    data_z = (data_sub - ch_mean) / ch_std

    window_samples = int(window_sec * sfreq)
    step_samples = max(1, int(step_sec * sfreq))

    n_ch_used = data_z.shape[0]
    window_samples = max(window_samples, n_ch_used + 10)

    jac_result = estimate_jacobian(
        data_z,
        window_size=window_samples,
        step_size=step_samples,
        regularization=regularization,
    )

    ep_result = detect_exceptional_points(jac_result)

    return {
        "jac_result": jac_result,
        "ep_result": ep_result,
        "n_channels_used": int(n_ch_used),
        "channel_indices": ch_idx.tolist(),
        "sfreq": sfreq,
        "window_sec": window_sec,
        "step_sec": step_sec,
    }


@dataclass
class ChiralityResult:
    """Result of chiral phase flux measurement around the EP.

    Attributes
    ----------
    phase_velocities : np.ndarray, shape (n_windows - 1,)
        Instantaneous phase rotation rate (rad/step) of the tracked
        coalescing eigenvector between consecutive windows.
        Positive = one chirality, negative = other.
    cumulative_phase : np.ndarray, shape (n_windows,)
        Accumulated geometric phase over time (rad).
    winding_number : float
        Total phase / (2*pi). Non-integer = incomplete winding.
    berry_phase : float
        Gauge-invariant geometric phase accumulated over the
        full trajectory: gamma = -Im sum ln(<v(w)|v(w+1)>).
    mean_phase_velocity : float
        Mean angular velocity (rad/step). Nonzero = chiral.
    phase_velocity_std : float
        Std of angular velocity. Low = consistent chirality.
    chirality_index : float
        |mean_velocity| / std_velocity. High = strong chirality.
        Analogous to a signal-to-noise ratio for directionality.
    circular_mean_direction : float
        Circular mean of phase velocity direction (rad).
    circular_variance : float
        Circular variance of phase velocities. 0 = perfectly
        consistent direction, 1 = uniformly random.
    tracking_quality : np.ndarray, shape (n_windows - 1,)
        |<v(w)|v(w+1)>| after tracking. Should be close to 1
        if eigenvector tracking is working correctly.
    tracked_eigenvalue_indices : np.ndarray, shape (n_windows,)
        Which eigenvalue index was tracked at each window.
    """

    phase_velocities: np.ndarray
    cumulative_phase: np.ndarray
    winding_number: float
    berry_phase: float
    mean_phase_velocity: float
    phase_velocity_std: float
    chirality_index: float
    circular_mean_direction: float
    circular_variance: float
    tracking_quality: np.ndarray
    tracked_eigenvalue_indices: np.ndarray


def _track_eigenvector(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    start_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track a single eigenvector across windows via maximum overlap.

    Resolves the gauge ambiguity: eigenvectors have arbitrary global
    phase at each window. We fix this by choosing the phase that
    maximizes Re(<v(w)|v(w+1)>), ensuring smooth continuation.

    Parameters
    ----------
    eigenvectors : np.ndarray, shape (n_windows, n_channels, n_channels)
        Right eigenvector matrices (columns = eigenvectors).
    eigenvalues : np.ndarray, shape (n_windows, n_channels)
        Complex eigenvalues sorted by descending magnitude.
    start_idx : int
        Which eigenvalue index to start tracking from (at window 0).

    Returns
    -------
    tracked_vectors : np.ndarray, shape (n_windows, n_channels)
        The tracked eigenvector at each window, with gauge fixed.
    tracked_indices : np.ndarray, shape (n_windows,)
        Which column index was selected at each window.
    overlaps : np.ndarray, shape (n_windows - 1,)
        |<v(w)|v(w+1)>| at each transition.
    """
    n_windows, n_ch, _ = eigenvectors.shape

    tracked_vectors = np.zeros((n_windows, n_ch), dtype=complex)
    tracked_indices = np.zeros(n_windows, dtype=int)
    overlaps = np.zeros(n_windows - 1)

    tracked_vectors[0] = eigenvectors[0, :, start_idx]
    norm = np.linalg.norm(tracked_vectors[0])
    if norm > 0:
        tracked_vectors[0] /= norm
    tracked_indices[0] = start_idx

    for w in range(1, n_windows):
        v_prev = tracked_vectors[w - 1]

        best_overlap = -1.0
        best_j = 0
        best_sign = 1.0

        for j in range(min(n_ch, 20)):
            v_candidate = eigenvectors[w, :, j]
            norm_c = np.linalg.norm(v_candidate)
            if norm_c == 0:
                continue
            v_candidate = v_candidate / norm_c

            inner = np.dot(np.conj(v_prev), v_candidate)
            overlap_mag = abs(inner)

            if overlap_mag > best_overlap:
                best_overlap = overlap_mag
                best_j = j
                best_sign = inner / overlap_mag if overlap_mag > 0 else 1.0

        v_selected = eigenvectors[w, :, best_j].copy()
        norm_s = np.linalg.norm(v_selected)
        if norm_s > 0:
            v_selected /= norm_s

        phase_correction = np.conj(best_sign) / abs(best_sign) if abs(best_sign) > 0 else 1.0
        v_selected *= phase_correction

        tracked_vectors[w] = v_selected
        tracked_indices[w] = best_j
        overlaps[w - 1] = best_overlap

    return tracked_vectors, tracked_indices, overlaps


def measure_chirality(
    jac_result: JacobianResult,
    ep_result: ExceptionalPointResult,
) -> ChiralityResult:
    """Measure the chiral phase flux of the coalescing eigenvalue pair.

    Tracks the eigenvalue SPLITTING (Delta_lambda = lambda_i - lambda_j)
    of the closest eigenvalue pair across windows. The phase of this
    complex splitting is the quantity that winds around an exceptional
    point.

    For a real Jacobian, individual eigenvalue phases are trivially 0 or
    pi, or flip randomly between conjugate branches. The splitting phase
    arg(Delta_lambda) is gauge-invariant and directly measures the
    topological winding around the EP.

    The Berry phase is computed from gauge-fixed (tracked) eigenvectors
    using the Pancharatnam connection.

    A high chirality index indicates consistent rotational direction —
    the system orbits the EP with a definite handedness. A low index
    indicates random jitter — no topological winding.

    Parameters
    ----------
    jac_result : JacobianResult
        Output of estimate_jacobian.
    ep_result : ExceptionalPointResult
        Output of detect_exceptional_points.

    Returns
    -------
    ChiralityResult
    """
    n_windows = len(jac_result.eigenvalues)

    most_common_i = int(np.median(ep_result.gap_pair_indices[:, 0]))

    tracked_vecs, tracked_idx, track_quality = _track_eigenvector(
        jac_result.eigenvectors,
        jac_result.eigenvalues,
        start_idx=most_common_i,
    )

    pair_tuples = [tuple(ep_result.gap_pair_indices[w]) for w in range(n_windows)]
    pair_counts = {}
    for pt in pair_tuples:
        pair_counts[pt] = pair_counts.get(pt, 0) + 1
    stable_pair = max(pair_counts, key=pair_counts.get)
    pair_i, pair_j = stable_pair

    delta_lambda = np.zeros(n_windows, dtype=complex)
    for w in range(n_windows):
        lam_i = jac_result.eigenvalues[w, pair_i]
        lam_j = jac_result.eigenvalues[w, pair_j]
        delta_lambda[w] = lam_i - lam_j

    splitting_phase = np.angle(delta_lambda)

    phase_velocities = np.diff(splitting_phase)
    phase_velocities = (phase_velocities + np.pi) % (2 * np.pi) - np.pi

    cumulative_phase = np.zeros(n_windows)
    cumulative_phase[1:] = np.cumsum(phase_velocities)

    winding_number = cumulative_phase[-1] / (2 * np.pi)

    berry_terms = np.zeros(n_windows - 1)
    for w in range(n_windows - 1):
        inner = np.dot(np.conj(tracked_vecs[w]), tracked_vecs[w + 1])
        if abs(inner) > 0:
            berry_terms[w] = -np.imag(np.log(inner))
    berry_phase = float(np.sum(berry_terms))

    mean_pv = float(np.mean(phase_velocities))
    std_pv = float(np.std(phase_velocities))
    chirality_index = abs(mean_pv) / std_pv if std_pv > 0 else 0.0

    unit_phases = np.exp(1j * phase_velocities)
    R = np.abs(np.mean(unit_phases))
    circular_mean_dir = float(np.angle(np.mean(unit_phases)))
    circular_var = float(1.0 - R)

    return ChiralityResult(
        phase_velocities=phase_velocities,
        cumulative_phase=cumulative_phase,
        winding_number=float(winding_number),
        berry_phase=berry_phase,
        mean_phase_velocity=mean_pv,
        phase_velocity_std=std_pv,
        chirality_index=float(chirality_index),
        circular_mean_direction=circular_mean_dir,
        circular_variance=circular_var,
        tracking_quality=track_quality,
        tracked_eigenvalue_indices=tracked_idx,
    )


def measure_chirality_epochs(
    data: np.ndarray,
    sfreq: float,
    epoch_intervals: list[tuple[int, int]],
    epoch_labels: list[str],
    window_sec: float = 0.5,
    step_sec: float = 0.1,
    regularization: float = 1e-4,
    max_channels: int = 30,
    seed: int = 42,
) -> dict:
    """Measure chirality separately for each epoch (e.g., task-relevant vs irrelevant).

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Full continuous data.
    sfreq : float
        Sampling frequency.
    epoch_intervals : list of (start_sample, end_sample)
        Time intervals for each epoch.
    epoch_labels : list of str
        Label for each epoch (e.g., "relevant", "irrelevant").
    window_sec, step_sec, regularization, max_channels, seed :
        Passed to compute_ep_proximity_timecourse.

    Returns
    -------
    dict with per-epoch chirality results and group comparison.
    """
    n_ch = data.shape[0]
    rng = np.random.default_rng(seed)
    if n_ch > max_channels:
        ch_idx = np.sort(rng.choice(n_ch, max_channels, replace=False))
    else:
        ch_idx = np.arange(n_ch)

    ch_mean = data[ch_idx].mean(axis=1, keepdims=True)
    ch_std = data[ch_idx].std(axis=1, keepdims=True)
    ch_std[ch_std == 0] = 1.0

    epoch_results = []
    for i, (start, end) in enumerate(epoch_intervals):
        seg = data[ch_idx, start:end]
        if seg.shape[1] < max_channels + 20:
            epoch_results.append(None)
            continue

        seg_z = (seg - ch_mean) / ch_std

        window_samples = max(int(window_sec * sfreq), len(ch_idx) + 10)
        step_samples = max(1, int(step_sec * sfreq))

        if seg_z.shape[1] < window_samples + 1:
            epoch_results.append(None)
            continue

        try:
            jac = estimate_jacobian(seg_z, window_size=window_samples,
                                    step_size=step_samples,
                                    regularization=regularization)
            ep = detect_exceptional_points(jac)
            chiral = measure_chirality(jac, ep)
            epoch_results.append({
                "label": epoch_labels[i],
                "start_sample": int(start),
                "end_sample": int(end),
                "n_windows": len(jac.window_centers),
                "chirality_index": float(chiral.chirality_index),
                "mean_phase_velocity": float(chiral.mean_phase_velocity),
                "phase_velocity_std": float(chiral.phase_velocity_std),
                "winding_number": float(chiral.winding_number),
                "berry_phase": float(chiral.berry_phase),
                "circular_variance": float(chiral.circular_variance),
                "circular_mean_direction": float(chiral.circular_mean_direction),
                "mean_tracking_quality": float(np.mean(chiral.tracking_quality)),
                "mean_ep_score": float(np.mean(ep.ep_scores)),
                "mean_spectral_radius": float(np.mean(jac.spectral_radius)),
            })
        except (ValueError, np.linalg.LinAlgError):
            epoch_results.append(None)

    return {
        "n_epochs": len(epoch_intervals),
        "n_valid": sum(1 for r in epoch_results if r is not None),
        "channel_indices": ch_idx.tolist(),
        "epoch_results": epoch_results,
    }
