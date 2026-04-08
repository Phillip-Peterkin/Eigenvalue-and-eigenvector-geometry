"""Detrended Fluctuation Analysis for neural time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DFAResult:
    """DFA result.

    Attributes
    ----------
    alpha : float
        DFA scaling exponent.
        - alpha ~ 0.5: white noise (uncorrelated)
        - alpha ~ 1.0: 1/f noise (long-range correlations)
        - alpha ~ 1.5: Brownian motion (integrated white noise)
    scales : np.ndarray
        Window sizes used.
    fluctuations : np.ndarray
        RMS fluctuation at each scale.
    r_squared : float
        R-squared of the log-log linear fit.
    """

    alpha: float
    scales: np.ndarray
    fluctuations: np.ndarray
    r_squared: float


def compute_dfa(
    data: np.ndarray,
    scales: np.ndarray | None = None,
    min_scale: int = 4,
    max_scale_fraction: float = 0.1,
) -> DFAResult:
    """Compute DFA scaling exponent for a 1D time series.

    Parameters
    ----------
    data : np.ndarray
        1D time series.
    scales : np.ndarray or None
        Window sizes to use. If None, auto-generated as log-spaced
        integers from min_scale to max_scale_fraction * len(data).
    min_scale : int
        Minimum window size.
    max_scale_fraction : float
        Maximum window size as fraction of data length.

    Returns
    -------
    DFAResult
    """
    try:
        import antropy
        alpha_val = antropy.detrended_fluctuation(data)
        return DFAResult(
            alpha=float(alpha_val),
            scales=np.array([]),
            fluctuations=np.array([]),
            r_squared=float("nan"),
        )
    except (ImportError, Exception):
        pass

    n = len(data)
    if n < min_scale * 4:
        return DFAResult(
            alpha=float("nan"),
            scales=np.array([]),
            fluctuations=np.array([]),
            r_squared=float("nan"),
        )

    if scales is None:
        max_scale = max(min_scale + 1, int(max_scale_fraction * n))
        scales = np.unique(
            np.logspace(
                np.log10(min_scale),
                np.log10(max_scale),
                num=20,
            ).astype(int)
        )

    cumsum = np.cumsum(data - np.mean(data))

    fluctuations = []
    valid_scales = []

    for s in scales:
        n_segments = n // s
        if n_segments < 1:
            continue

        rms_values = []
        for seg in range(n_segments):
            start = seg * s
            end = start + s
            segment = cumsum[start:end]

            x = np.arange(s)
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            residual = segment - trend
            rms_values.append(np.sqrt(np.mean(residual**2)))

        if rms_values:
            fluctuations.append(np.mean(rms_values))
            valid_scales.append(s)

    if len(valid_scales) < 3:
        return DFAResult(
            alpha=float("nan"),
            scales=np.array(valid_scales),
            fluctuations=np.array(fluctuations),
            r_squared=float("nan"),
        )

    log_scales = np.log(valid_scales)
    log_fluct = np.log(fluctuations)

    coeffs = np.polyfit(log_scales, log_fluct, 1)
    alpha_val = coeffs[0]

    predicted = np.polyval(coeffs, log_scales)
    ss_res = np.sum((log_fluct - predicted) ** 2)
    ss_tot = np.sum((log_fluct - np.mean(log_fluct)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return DFAResult(
        alpha=float(alpha_val),
        scales=np.array(valid_scales),
        fluctuations=np.array(fluctuations),
        r_squared=r_squared,
    )
