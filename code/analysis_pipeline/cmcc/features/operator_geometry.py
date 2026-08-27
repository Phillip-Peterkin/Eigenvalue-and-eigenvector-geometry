"""Pure operator-geometry metric helpers with explicit numerical contracts.

The historical proximity score and current Near-Degeneracy (ND) score are
mathematically distinct. Backward-compatible names do not imply equivalence.
"""
from __future__ import annotations

import numpy as np

PROXIMITY_SCORE_EPSILON = 1e-10
ND_SCORE_EPSILON = 1e-12


def _as_finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def spectral_radius_from_eigenvalues(eigenvalues: np.ndarray) -> float:
    """Return the maximum absolute eigenvalue for one fitted window."""
    evals = _as_finite_vector(eigenvalues, name="eigenvalues")
    return float(np.max(np.abs(evals)))


def minimum_eigenvalue_gap(
    eigenvalues: np.ndarray,
    max_modes: int = 20,
) -> tuple[float, int, int]:
    """Return the minimum spacing among the leading-magnitude eigenmodes.

    `leading` is defined here, not by caller ordering: eigenvalues are ranked by
    descending magnitude before the `max_modes` truncation. Returned indices
    always refer to the caller's original array. This makes the result invariant
    to input ordering while preserving index provenance.
    """
    evals = _as_finite_vector(eigenvalues, name="eigenvalues")
    if evals.size < 2:
        raise ValueError("Need at least two eigenvalues to compute a gap")
    if not isinstance(max_modes, int) or isinstance(max_modes, bool) or max_modes < 2:
        raise ValueError("max_modes must be an integer >= 2")

    order = np.argsort(-np.abs(evals), kind="stable")
    selected = order[: min(evals.size, max_modes)]

    best_gap = np.inf
    best_i = int(selected[0])
    best_j = int(selected[1])
    for pos_i in range(len(selected)):
        ii = int(selected[pos_i])
        for pos_j in range(pos_i + 1, len(selected)):
            jj = int(selected[pos_j])
            gap = float(abs(evals[ii] - evals[jj]))
            if gap < best_gap:
                best_gap = gap
                best_i, best_j = ii, jj
    return float(best_gap), best_i, best_j


def eigenvector_overlap(vector_i: np.ndarray, vector_j: np.ndarray) -> float:
    """Return absolute cosine similarity between two real or complex vectors."""
    v_i = _as_finite_vector(vector_i, name="vector_i")
    v_j = _as_finite_vector(vector_j, name="vector_j")
    if v_i.shape != v_j.shape:
        raise ValueError(f"eigenvectors must share shape; got {v_i.shape} vs {v_j.shape}")
    norm_i = float(np.linalg.norm(v_i))
    norm_j = float(np.linalg.norm(v_j))
    if norm_i == 0.0 or norm_j == 0.0:
        return 0.0
    value = float(abs(np.vdot(v_i, v_j)) / (norm_i * norm_j))
    return float(np.clip(value, 0.0, 1.0))


def geometry_proximity_score(
    overlap: float,
    gap: float,
    epsilon: float = PROXIMITY_SCORE_EPSILON,
) -> float:
    """Return the historical proximity score stored as `ep_score` in JSON."""
    overlap = float(overlap)
    gap = float(gap)
    epsilon = float(epsilon)
    if not np.isfinite(overlap) or overlap < 0.0:
        raise ValueError("overlap must be finite and non-negative")
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError("gap must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    return overlap / (gap + epsilon)


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

    Valid paired windows are transformed as

    `crowding = -log10(gap + epsilon)` and
    `nonorthogonality = log10(condition_number + epsilon)`.

    Both series are standardized within the supplied analysis unit and projected
    onto PC1. Same-sign loadings are oriented positive; opposite-sign or zero
    loadings fail loudly because they do not satisfy the declared ND construct.
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

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
    """Return `(sum sigma)^2 / sum(sigma^2)` for a finite singular-value vector."""
    sigma = np.abs(_as_finite_vector(np.asarray(singular_values, dtype=float), name="singular_values"))
    s_sum = float(sigma.sum())
    s_sq_sum = float((sigma**2).sum())
    if s_sq_sum == 0.0:
        return 0.0
    return (s_sum**2) / s_sq_sum


def effective_rank(singular_values: np.ndarray) -> float:
    """Return Shannon effective rank of a finite singular-value spectrum."""
    sigma = np.abs(_as_finite_vector(np.asarray(singular_values, dtype=float), name="singular_values"))
    s_sum = float(sigma.sum())
    if s_sum == 0.0:
        return 0.0
    p = sigma / s_sum
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))
