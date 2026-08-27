"""Validated entry point for sliding-window VAR(1) fitting."""
from __future__ import annotations

import numpy as np

from cmcc.analysis.dynamical_systems import JacobianResult, estimate_jacobian


def estimate_var_operator(
    data: np.ndarray,
    window_size: int = 500,
    step_size: int = 100,
    regularization: float = 1e-4,
) -> JacobianResult:
    """Validate inputs and delegate to the retained VAR(1) implementation."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"data must be 2-D (channels, samples); got ndim={array.ndim}")
    if array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError(f"data must contain channels and samples; got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("data must contain only finite values")
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 1:
        raise ValueError("window_size must be an integer > 1")
    if not isinstance(step_size, int) or isinstance(step_size, bool) or step_size <= 0:
        raise ValueError("step_size must be a positive integer")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and non-negative")
    if window_size <= array.shape[0]:
        raise ValueError("window_size must exceed the channel count")
    if array.shape[1] < window_size + 1:
        raise ValueError("data is too short for the requested window_size")

    result = estimate_jacobian(
        array,
        window_size=window_size,
        step_size=step_size,
        regularization=float(regularization),
    )
    if not np.all(np.isfinite(result.jacobians)):
        raise FloatingPointError("VAR fitting produced non-finite coefficient matrices")
    if not np.all(np.isfinite(result.spectral_radius)):
        raise FloatingPointError("VAR fitting produced non-finite spectral radii")
    if not np.all(np.isfinite(result.residual_variance)):
        raise FloatingPointError("VAR fitting produced non-finite residual variances")
    return result
