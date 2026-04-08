"""Neuronal avalanche detection from multi-channel neural data.

Implements the Beggs & Plenz (2003) discretized avalanche detection
adapted for ECoG/iEEG high-gamma envelope data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Avalanche:
    """A single neuronal avalanche event.

    Attributes
    ----------
    onset_bin : int
        Index of the first active bin.
    duration_bins : int
        Number of contiguous active bins.
    size : int
        Total number of supra-threshold channel-sample events.
    channels : set[int]
        Set of unique channel indices involved.
    onset_time_s : float
        Onset time in seconds (if known), else NaN.
    """

    onset_bin: int
    duration_bins: int
    size: int
    channels: set[int] = field(default_factory=set)
    onset_time_s: float = float("nan")


def _zscore_channels(data: np.ndarray) -> np.ndarray:
    means = np.mean(data, axis=1, keepdims=True)
    stds = np.std(data, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    return (data - means) / stds


def _compute_mean_iei(binary_activity: np.ndarray) -> float:
    """Compute mean inter-event interval across all channels.

    Parameters
    ----------
    binary_activity : np.ndarray
        Binary matrix (n_channels, n_samples). 1 = supra-threshold.

    Returns
    -------
    float
        Mean inter-event interval in samples. Returns 1.0 if insufficient events.
    """
    all_ieis = []
    for ch_idx in range(binary_activity.shape[0]):
        event_indices = np.where(binary_activity[ch_idx] == 1)[0]
        if len(event_indices) > 1:
            ieis = np.diff(event_indices)
            all_ieis.extend(ieis.tolist())

    if len(all_ieis) == 0:
        return 1.0

    return float(np.mean(all_ieis))


def detect_avalanches(
    data: np.ndarray,
    sfreq: float,
    threshold_sd: float = 3.0,
    bin_width_factor: float = 1.0,
) -> list[Avalanche]:
    """Detect neuronal avalanches from multi-channel data.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
        Multi-channel neural data (e.g., high-gamma envelope).
    sfreq : float
        Sampling frequency in Hz.
    threshold_sd : float
        Threshold in standard deviations for event detection.
    bin_width_factor : float
        Multiplier of mean inter-event interval for temporal binning.

    Returns
    -------
    list[Avalanche]
        Detected avalanches sorted by onset time.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data (n_channels, n_samples), got shape {data.shape}")

    n_channels, n_samples = data.shape
    if n_channels == 0 or n_samples == 0:
        return []

    zscored = _zscore_channels(data)
    binary = (np.abs(zscored) > threshold_sd).astype(np.int8)

    mean_iei = _compute_mean_iei(binary)
    bin_width_samples = max(1, int(round(bin_width_factor * mean_iei)))

    n_bins = n_samples // bin_width_samples
    if n_bins == 0:
        return []

    avalanches = []
    in_avalanche = False
    current_onset = 0
    current_size = 0
    current_channels: set[int] = set()

    for b in range(n_bins):
        start = b * bin_width_samples
        end = min(start + bin_width_samples, n_samples)
        bin_slice = binary[:, start:end]

        bin_events = int(bin_slice.sum())
        active_channels = set(np.where(bin_slice.any(axis=1))[0].tolist())

        if bin_events > 0:
            if not in_avalanche:
                current_onset = b
                current_size = 0
                current_channels = set()
                in_avalanche = True

            current_size += bin_events
            current_channels.update(active_channels)
        else:
            if in_avalanche:
                duration = b - current_onset
                onset_time = current_onset * bin_width_samples / sfreq
                avalanches.append(Avalanche(
                    onset_bin=current_onset,
                    duration_bins=duration,
                    size=current_size,
                    channels=current_channels,
                    onset_time_s=onset_time,
                ))
                in_avalanche = False

    if in_avalanche:
        duration = n_bins - current_onset
        onset_time = current_onset * bin_width_samples / sfreq
        avalanches.append(Avalanche(
            onset_bin=current_onset,
            duration_bins=duration,
            size=current_size,
            channels=current_channels,
            onset_time_s=onset_time,
        ))

    return avalanches


def avalanche_sizes(avalanches: list[Avalanche]) -> np.ndarray:
    """Extract avalanche sizes as an array."""
    if not avalanches:
        return np.array([], dtype=int)
    return np.array([a.size for a in avalanches])


def avalanche_durations(avalanches: list[Avalanche]) -> np.ndarray:
    """Extract avalanche durations (in bins) as an array."""
    if not avalanches:
        return np.array([], dtype=int)
    return np.array([a.duration_bins for a in avalanches])
