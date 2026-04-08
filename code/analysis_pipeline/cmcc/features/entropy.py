"""Multi-scale entropy computation for neural time series."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MSEResult:
    """Multi-scale entropy result.

    Attributes
    ----------
    scales : list[int]
        Coarse-graining scale factors.
    entropy_values : list[float]
        Sample entropy at each scale.
    complexity_index : float
        Area under the MSE curve (sum of entropy values).
    m : int
        Embedding dimension used.
    r : float
        Tolerance used (absolute, not relative).
    """

    scales: list[int]
    entropy_values: list[float]
    complexity_index: float
    m: int
    r: float


def _coarse_grain(data: np.ndarray, scale: int) -> np.ndarray:
    """Coarse-grain a time series by averaging non-overlapping windows."""
    n = len(data)
    n_new = n // scale
    if n_new == 0:
        return np.array([])
    return np.mean(data[: n_new * scale].reshape(-1, scale), axis=1)


def _sample_entropy(data: np.ndarray, m: int, r: float) -> float:
    """Compute sample entropy of a 1D time series.

    Parameters
    ----------
    data : np.ndarray
        1D time series.
    m : int
        Embedding dimension.
    r : float
        Tolerance (absolute).

    Returns
    -------
    float
        Sample entropy value. Returns NaN if undefined.
    """
    n = len(data)
    if n < m + 2:
        return float("nan")

    try:
        from antropy import sample_entropy
        return float(sample_entropy(data, order=m, metric="chebyshev"))
    except (ImportError, Exception):
        pass

    def _count_matches(templates: np.ndarray, tol: float) -> int:
        count = 0
        n_t = len(templates)
        for i in range(n_t):
            for j in range(i + 1, n_t):
                if np.max(np.abs(templates[i] - templates[j])) <= tol:
                    count += 1
        return count

    templates_m = np.array([data[i : i + m] for i in range(n - m)])
    templates_m1 = np.array([data[i : i + m + 1] for i in range(n - m - 1)])

    B = _count_matches(templates_m, r)
    A = _count_matches(templates_m1, r)

    if B == 0:
        return float("nan")

    return -np.log(A / B) if A > 0 else float("inf")


def compute_mse(
    data: np.ndarray,
    scales: range | list[int] | None = None,
    m: int = 2,
    r_factor: float = 0.15,
) -> MSEResult:
    """Compute multi-scale entropy of a 1D time series.

    Parameters
    ----------
    data : np.ndarray
        1D time series (single channel).
    scales : range or list[int] or None
        Coarse-graining scale factors. Default: range(1, 21).
    m : int
        Embedding dimension.
    r_factor : float
        Tolerance as fraction of data standard deviation.

    Returns
    -------
    MSEResult
    """
    if scales is None:
        scales = list(range(1, 21))
    else:
        scales = list(scales)

    r = r_factor * np.std(data)
    if r == 0:
        r = 1e-10

    entropy_values = []
    for scale in scales:
        cg = _coarse_grain(data, scale)
        if len(cg) < m + 2:
            entropy_values.append(float("nan"))
        else:
            se = _sample_entropy(cg, m, r)
            entropy_values.append(se)

    valid = [v for v in entropy_values if not np.isnan(v) and not np.isinf(v)]
    ci = float(np.sum(valid)) if valid else float("nan")

    return MSEResult(
        scales=scales,
        entropy_values=entropy_values,
        complexity_index=ci,
        m=m,
        r=float(r),
    )
