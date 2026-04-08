"""Filtering operations: line noise removal and high-gamma extraction."""

from __future__ import annotations

import mne
import numpy as np


SITE_LINE_FREQ = {
    "CE": 60.0,
    "CF": 50.0,
    "CG": 50.0,
}


def get_line_freq(site: str, override: float | None = None) -> float:
    """Get line noise frequency for a given site.

    Parameters
    ----------
    site : str
        Site prefix ("CE", "CF", "CG").
    override : float or None
        If provided, use this value instead of auto-detection.

    Returns
    -------
    float
        Line noise frequency in Hz.
    """
    if override is not None:
        return override
    return SITE_LINE_FREQ.get(site, 60.0)


def remove_line_noise(
    raw: mne.io.Raw,
    line_freq: float = 60.0,
    n_harmonics: int = 3,
) -> mne.io.Raw:
    """Remove line noise using notch filtering.

    Uses MNE's notch_filter at the fundamental and harmonics.
    If the `ieeg` package is available, uses its multitaper line_filter
    for better spectral precision.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data (must be preloaded).
    line_freq : float
        Fundamental line noise frequency in Hz (50 or 60).
    n_harmonics : int
        Number of harmonics to remove (including fundamental).

    Returns
    -------
    mne.io.Raw
        Filtered data (modified in-place).
    """
    try:
        from ieeg.mt_filter import line_filter
        line_filter(raw, raw.info["sfreq"], verbose=False)
    except (ImportError, Exception):
        freqs = [line_freq * (i + 1) for i in range(n_harmonics)]
        nyquist = raw.info["sfreq"] / 2.0
        freqs = [f for f in freqs if f < nyquist]
        if freqs:
            raw.notch_filter(freqs, verbose=False)

    return raw


def extract_high_gamma(
    raw: mne.io.Raw,
    passband: tuple[int, int] = (70, 150),
) -> mne.io.Raw:
    """Extract high-gamma envelope using filterbank Hilbert method.

    Attempts to use the ieeg toolbox's optimized gamma extraction.
    Falls back to MNE bandpass + Hilbert if ieeg is not available.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data (must be preloaded).
    passband : tuple[int, int]
        Frequency band in Hz, default (70, 150).

    Returns
    -------
    mne.io.Raw
        Raw object containing the high-gamma analytic amplitude envelope.
    """
    try:
        from ieeg.timefreq.gamma import extract
        result = extract(raw, passband=passband, copy=True, verbose=False)
        return result
    except (ImportError, Exception):
        raw_copy = raw.copy()
        raw_copy.filter(
            l_freq=passband[0],
            h_freq=passband[1],
            method="iir",
            verbose=False,
        )
        raw_copy.apply_hilbert(envelope=True, verbose=False)
        return raw_copy
