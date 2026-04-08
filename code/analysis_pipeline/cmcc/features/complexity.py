"""Lempel-Ziv complexity computation for multi-channel neural data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LZCResult:
    """Lempel-Ziv complexity result.

    Attributes
    ----------
    lzc_raw : float
        Raw LZ complexity count.
    lzc_normalized : float
        LZc normalized by expected complexity of random sequence.
    surrogate_mean : float
        Mean LZc of surrogate (shuffled) sequences.
    surrogate_std : float
        Std of surrogate LZc values.
    n_channels : int
        Number of channels used.
    n_samples : int
        Number of time samples.
    """

    lzc_raw: float
    lzc_normalized: float
    surrogate_mean: float
    surrogate_std: float
    n_channels: int
    n_samples: int


def _lz76(sequence: np.ndarray) -> int:
    """Compute Lempel-Ziv complexity (LZ76 / Kaspar-Schuster algorithm).

    Uses the scanning approach from Kaspar & Schuster (1987) which converts
    the sequence to a string and leverages Python's optimized ``str.find``
    (Boyer-Moore / two-way hybrid in CPython) for substring search, yielding
    O(n log n) expected-time performance instead of the naive O(n^2) loop.

    Parameters
    ----------
    sequence : np.ndarray
        1D binary array (values 0 or 1).

    Returns
    -------
    int
        Number of distinct substrings (complexity count).
    """
    n = len(sequence)
    if n == 0:
        return 0
    if n == 1:
        return 1

    s = sequence.astype(np.uint8).tobytes()

    complexity = 1
    start = 0
    length = 1

    while start + length <= n:
        window_end = start + length
        sub = s[start:window_end]
        search_end = window_end - 1
        if search_end <= 0:
            complexity += 1
            start = window_end
            length = 1
            continue

        pos = s[:search_end].find(sub)
        if pos == -1:
            complexity += 1
            start = window_end
            length = 1
        else:
            length += 1

    return complexity


def _binarize_multichannel(data: np.ndarray) -> np.ndarray:
    """Binarize each channel by median threshold, then concatenate.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Multi-channel data.

    Returns
    -------
    np.ndarray
        1D binary array of length n_channels * n_samples.
    """
    medians = np.median(data, axis=1, keepdims=True)
    binary = (data > medians).astype(np.int8)
    return binary.ravel()


def compute_lzc(
    data: np.ndarray,
    normalize: bool = True,
    n_surrogates: int = 100,
    seed: int = 42,
) -> LZCResult:
    """Compute Lempel-Ziv complexity for multi-channel neural data.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Multi-channel neural data.
    normalize : bool
        If True, normalize by surrogate LZc.
    n_surrogates : int
        Number of shuffled surrogates for normalization.
    seed : int
        Random seed for surrogates.

    Returns
    -------
    LZCResult
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_channels, n_samples = data.shape
    binary_seq = _binarize_multichannel(data)
    lzc_raw = _lz76(binary_seq)

    surrogate_mean = float("nan")
    surrogate_std = float("nan")
    lzc_normalized = float(lzc_raw)

    if normalize and n_surrogates > 0:
        rng = np.random.default_rng(seed)
        surrogate_values = []
        for _ in range(n_surrogates):
            shuffled = binary_seq.copy()
            rng.shuffle(shuffled)
            surrogate_values.append(_lz76(shuffled))

        surrogate_mean = float(np.mean(surrogate_values))
        surrogate_std = float(np.std(surrogate_values))
        if surrogate_mean > 0:
            lzc_normalized = float(lzc_raw) / surrogate_mean
        else:
            lzc_normalized = float("nan")

    return LZCResult(
        lzc_raw=float(lzc_raw),
        lzc_normalized=lzc_normalized,
        surrogate_mean=surrogate_mean,
        surrogate_std=surrogate_std,
        n_channels=n_channels,
        n_samples=n_samples,
    )
