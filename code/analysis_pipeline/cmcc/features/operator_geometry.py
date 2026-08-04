"""Pure operator-geometry metric helpers with known synthetic behavior.

Scientific rationale
--------------------
Fitted first-order vector autoregressive (VAR(1)) operators yield eigenvalues
and eigenvectors. The manuscript summarizes those spectra with:

1. Spectral radius (maximum absolute eigenvalue) — a stability-margin summary.
2. Minimum eigenvalue gap — how crowded the spectrum is.
3. Eigenvector overlap of the closest pair — non-orthogonality of that pair.
4. Legacy proximity score (`ep_score` in JSON): overlap / (gap + epsilon).
5. Manuscript near-degeneracy (ND) score: within-subject average of
   z-scored -log10(gap) and z-scored log10(eigenvector condition number).
6. Singular-value concentration summaries (participation ratio and effective
   rank), which behave like inequality / concentration indices on the singular
   spectrum (high when energy is concentrated in few modes).

These helpers are intentionally free of input/output and dataset loading so
unit tests can verify the math on synthetic inputs with known answers.
"""
from __future__ import annotations

import numpy as np

PROXIMITY_SCORE_EPSILON = 1e-10
ND_SCORE_EPSILON = 1e-12


def spectral_radius_from_eigenvalues(eigenvalues: np.ndarray) -> float:
    """Return max |lambda| for a one-window eigenvalue vector.

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (n_modes,)
        Complex or real eigenvalues. Units: dimensionless (VAR(1) map).

    Returns
    -------
    float
        Spectral radius. Failure mode: empty input raises ValueError.
    """
    evals = np.asarray(eigenvalues)
    if evals.size == 0:
        raise ValueError("eigenvalues must be non-empty")
    return float(np.max(np.abs(evals)))


def minimum_eigenvalue_gap(
    eigenvalues: np.ndarray,
    max_modes: int = 20,
) -> tuple[float, int, int]:
    """Return the smallest pairwise |lambda_i - lambda_j| among leading modes.

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (n_modes,)
        Eigenvalues for one window (any order).
    max_modes : int
        Cap on how many leading modes (by current array order) are compared.
        Matches ``detect_exceptional_points`` in ``dynamical_systems``.

    Returns
    -------
    gap : float
        Minimum absolute pairwise difference. Units: dimensionless.
    index_i, index_j : int
        Indices of the closest pair in the supplied array.

    Failure modes
    -------------
    Fewer than two modes raises ValueError.
    """
    evals = np.asarray(eigenvalues)
    n_modes = int(evals.shape[0])
    if n_modes < 2:
        raise ValueError("Need at least two eigenvalues to compute a gap")

    n_use = min(n_modes, max_modes)
    best_gap = np.inf
    best_i, best_j = 0, 1
    for ii in range(n_use):
        for jj in range(ii + 1, n_use):
            gap = abs(evals[ii] - evals[jj])
            if gap < best_gap:
                best_gap = float(gap)
                best_i, best_j = ii, jj
    return float(best_gap), best_i, best_j


def eigenvector_overlap(vector_i: np.ndarray, vector_j: np.ndarray) -> float:
    """Return absolute cosine similarity |<v_i|v_j>| / (||v_i|| ||v_j||).

    Parameters
    ----------
    vector_i, vector_j : np.ndarray, shape (n_modes,)
        Right eigenvectors (may be complex). Units: dimensionless.

    Returns
    -------
    float
        Overlap in [0, 1]. Zero-norm vectors return 0.0.
    """
    v_i = np.asarray(vector_i)
    v_j = np.asarray(vector_j)
    norm_i = np.linalg.norm(v_i)
    norm_j = np.linalg.norm(v_j)
    if norm_i <= 0.0 or norm_j <= 0.0:
        return 0.0
    return float(abs(np.dot(np.conj(v_i), v_j)) / (norm_i * norm_j))


def geometry_proximity_score(
    overlap: float,
    gap: float,
    epsilon: float = PROXIMITY_SCORE_EPSILON,
) -> float:
    """Legacy per-window proximity score stored as ``ep_score`` in JSON.

    Definition used by ``detect_exceptional_points``:
    score = overlap / (gap + epsilon).

    Parameters
    ----------
    overlap : float
        Eigenvector overlap in [0, 1].
    gap : float
        Minimum eigenvalue gap (non-negative).
    epsilon : float
        Numerical floor preventing division by zero.

    Returns
    -------
    float
        Proximity score in [0, inf). Larger means closer to coalescence.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float(overlap) / (float(gap) + float(epsilon))


def _nan_zscore(values: np.ndarray) -> np.ndarray:
    """Within-vector z-score ignoring non-finite entries."""
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() == 0:
        return out
    mu = float(np.mean(values[finite]))
    sigma = float(np.std(values[finite]))
    if sigma == 0.0:
        out[finite] = 0.0
        return out
    out[finite] = (values[finite] - mu) / sigma
    return out


def compute_nd_score(
    gaps: np.ndarray,
    condition_numbers: np.ndarray,
    epsilon: float = ND_SCORE_EPSILON,
) -> np.ndarray:
    """Manuscript near-degeneracy (ND) score (Methods composite).

    ND = 0.5 * [ z(-log10(gap + eps)) + z(log10(kappa + eps)) ]

    where z is a within-subject (within-array) z-score. This matches the
    manuscript equation; it is distinct from the legacy JSON ``ep_score``
    proximity ratio ``overlap / (gap + eps)``.

    Parameters
    ----------
    gaps : np.ndarray, shape (n_windows,)
        Minimum eigenvalue gaps. Units: dimensionless.
    condition_numbers : np.ndarray, shape (n_windows,)
        Eigenvector-matrix condition numbers kappa(V). Dimensionless.
    epsilon : float
        Numerical floor inside the logarithms (manuscript uses 1e-12).

    Returns
    -------
    np.ndarray, shape (n_windows,)
        Window-level ND scores. Non-finite inputs propagate as NaN.

    Failure modes
    -------------
    Length mismatch raises ValueError. epsilon <= 0 raises ValueError.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    gap_arr = np.asarray(gaps, dtype=float)
    kappa_arr = np.asarray(condition_numbers, dtype=float)
    if gap_arr.shape != kappa_arr.shape:
        raise ValueError(
            f"gaps and condition_numbers must share shape; got {gap_arr.shape} vs {kappa_arr.shape}"
        )

    crowding = -np.log10(np.maximum(gap_arr, 0.0) + epsilon)
    nonorthogonality = np.log10(np.maximum(kappa_arr, 0.0) + epsilon)
    return 0.5 * (_nan_zscore(crowding) + _nan_zscore(nonorthogonality))


def participation_ratio(singular_values: np.ndarray) -> float:
    """Participation ratio (concentration index) of a singular spectrum.

    PR = (sum sigma_i)^2 / sum(sigma_i^2)

    Behavior (analogous to an inverse-concentration / anti-Gini summary):
    - Uniform spectrum over k equal singular values -> PR = k (delocalized).
    - Single nonzero singular value -> PR = 1 (fully concentrated).

    Parameters
    ----------
    singular_values : np.ndarray, shape (n_modes,)
        Non-negative singular values preferred; absolutes are taken.

    Returns
    -------
    float
        Participation ratio in [1, n_modes] for nonzero spectra; 0 if all zero.
    """
    sigma = np.abs(np.asarray(singular_values, dtype=float))
    s_sum = float(sigma.sum())
    s_sq_sum = float((sigma ** 2).sum())
    if s_sq_sum <= 0.0:
        return 0.0
    return (s_sum ** 2) / s_sq_sum


def effective_rank(singular_values: np.ndarray) -> float:
    """Shannon effective rank of a singular spectrum.

    erank = exp(-sum p_i log p_i) with p_i = sigma_i / sum(sigma).

    Uniform spectrum over k modes -> erank = k.
    Single nonzero mode -> erank = 1.
    """
    sigma = np.abs(np.asarray(singular_values, dtype=float))
    s_sum = float(sigma.sum())
    if s_sum <= 0.0:
        return 0.0
    p = sigma / s_sum
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))
