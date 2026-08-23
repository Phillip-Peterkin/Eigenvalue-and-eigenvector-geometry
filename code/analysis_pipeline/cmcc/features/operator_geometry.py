"""Pure operator-geometry metric helpers with known synthetic behavior.

Scientific rationale
--------------------
Fitted first-order vector autoregressive (VAR(1)) operators yield eigenvalues
and eigenvectors. The public analysis summarizes those spectra with:

1. Spectral radius (maximum absolute eigenvalue), a stability-margin summary.
2. Minimum eigenvalue gap, describing local spectral crowding.
3. Eigenvector overlap of the closest pair, describing pairwise non-orthogonality.
4. Historical proximity score (`ep_score` in JSON): overlap / (gap + epsilon).
5. Current manuscript Near-Degeneracy (ND) score: first-principal-component
   projection of within-array standardized eigenvalue crowding and log
   eigenvector condition number, with sign normalized to positive loadings.
6. Singular-value concentration summaries such as participation ratio and
   effective rank.

The historical proximity score and current manuscript ND score are distinct
statistics. Backward-compatible names do not imply mathematical equivalence.
"""
from __future__ import annotations

import numpy as np

PROXIMITY_SCORE_EPSILON = 1e-10
ND_SCORE_EPSILON = 1e-12


def spectral_radius_from_eigenvalues(eigenvalues: np.ndarray) -> float:
    """Return the maximum absolute eigenvalue for one fitted window."""
    evals = np.asarray(eigenvalues)
    if evals.size == 0:
        raise ValueError("eigenvalues must be non-empty")
    return float(np.max(np.abs(evals)))


def minimum_eigenvalue_gap(
    eigenvalues: np.ndarray,
    max_modes: int = 20,
) -> tuple[float, int, int]:
    """Return the smallest pairwise eigenvalue spacing among leading modes."""
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
    """Return absolute cosine similarity between two real or complex vectors."""
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
    """Return the historical proximity score stored as `ep_score` in JSON.

    score = overlap / (gap + epsilon)

    This is retained for provenance. It is not the current manuscript ND score.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    return float(overlap) / (float(gap) + float(epsilon))


def _nan_zscore(values: np.ndarray) -> np.ndarray:
    """Within-vector z-score while preserving non-finite entries as NaN."""
    values = np.asarray(values, dtype=float)
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
    """Compute the current manuscript window-level Near-Degeneracy score.

    For each valid window, define eigenvalue crowding and eigenvector
    non-orthogonality as::

        c = -log10(gap + epsilon)
        k =  log10(condition_number + epsilon)

    A window contributes to the analysis only when both metric inputs are
    finite and non-negative. Unpaired or invalid windows remain ``NaN`` and are
    excluded before either feature is standardized, so they cannot influence
    valid-window scores indirectly. The paired feature series are z-scored
    within the supplied analysis unit, stacked as an ``n_windows x 2`` matrix,
    and projected onto the first principal component of their covariance.

    The manuscript ND construct assumes a common direction in which greater
    crowding and greater non-orthogonality load positively. Principal-component
    sign is arbitrary, so same-sign loadings are oriented positive. If the
    first component has opposite-sign or zero loadings, no global sign flip can
    satisfy that scientific definition and this function fails loudly rather
    than silently redefining ND.

    This implementation intentionally differs from the historical `ep_score`
    ratio. It also means the mean score within the same standardized analysis
    unit is approximately zero by construction, which is important when
    designing subject-level aggregation.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    gap_arr = np.asarray(gaps, dtype=float)
    kappa_arr = np.asarray(condition_numbers, dtype=float)
    if gap_arr.shape != kappa_arr.shape:
        raise ValueError(
            f"gaps and condition_numbers must share shape; got {gap_arr.shape} vs {kappa_arr.shape}"
        )
    if gap_arr.ndim != 1:
        raise ValueError("gaps and condition_numbers must be one-dimensional window series")

    crowding = np.full(gap_arr.shape, np.nan, dtype=float)
    nonorthogonality = np.full(kappa_arr.shape, np.nan, dtype=float)

    valid_pair = (
        np.isfinite(gap_arr)
        & (gap_arr >= 0.0)
        & np.isfinite(kappa_arr)
        & (kappa_arr >= 0.0)
    )
    crowding[valid_pair] = -np.log10(gap_arr[valid_pair] + epsilon)
    nonorthogonality[valid_pair] = np.log10(kappa_arr[valid_pair] + epsilon)

    z_c = _nan_zscore(crowding)
    z_k = _nan_zscore(nonorthogonality)

    scores = np.full(gap_arr.shape, np.nan, dtype=float)
    finite = np.isfinite(z_c) & np.isfinite(z_k)
    if finite.sum() == 0:
        return scores

    x = np.column_stack([z_c[finite], z_k[finite]])
    if x.shape[0] == 1 or np.allclose(x, 0.0):
        scores[finite] = 0.0
        return scores

    covariance = np.cov(x, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    loading = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)

    if loading[0] * loading[1] <= 0.0:
        raise ValueError(
            "Near-Degeneracy PC1 loadings must share a nonzero sign so both can be oriented positive"
        )
    if np.all(loading < 0.0):
        loading = -loading

    scores[finite] = x @ loading
    return scores


def participation_ratio(singular_values: np.ndarray) -> float:
    """Return the participation ratio of a singular-value spectrum.

    ``PR = (sum sigma_i)^2 / sum(sigma_i^2)``.
    Uniform nonzero spectra approach the number of modes; a rank-one spectrum
    returns one.
    """
    sigma = np.abs(np.asarray(singular_values, dtype=float))
    s_sum = float(sigma.sum())
    s_sq_sum = float((sigma**2).sum())
    if s_sq_sum <= 0.0:
        return 0.0
    return (s_sum**2) / s_sq_sum


def effective_rank(singular_values: np.ndarray) -> float:
    """Return Shannon effective rank of a singular-value spectrum."""
    sigma = np.abs(np.asarray(singular_values, dtype=float))
    s_sum = float(sigma.sum())
    if s_sum <= 0.0:
        return 0.0
    p = sigma / s_sum
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))
