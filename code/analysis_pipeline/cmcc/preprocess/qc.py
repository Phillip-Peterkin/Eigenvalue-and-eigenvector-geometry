"""Bad channel detection and artifact rejection for iEEG data."""

from __future__ import annotations

from typing import Any

import numpy as np
import mne


def detect_bad_channels(
    raw: mne.io.Raw,
    flat_threshold: float = 1e-7,
    amplitude_threshold_sd: float = 5.0,
    min_variance_ratio: float = 0.01,
) -> list[str]:
    """Detect bad channels using amplitude, flatness, and variance heuristics.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw ECoG data (must be preloaded or will be loaded temporarily).
    flat_threshold : float
        Peak-to-peak amplitude below this value (in V) flags a flat channel.
    amplitude_threshold_sd : float
        Channels with mean absolute amplitude exceeding this many SDs
        above the cross-channel mean are flagged.
    min_variance_ratio : float
        Channels with variance below this fraction of the median
        cross-channel variance are flagged.

    Returns
    -------
    list[str]
        Names of detected bad channels.
    """
    was_preloaded = raw.preload
    if not was_preloaded:
        raw = raw.copy().load_data()

    eeg_picks = mne.pick_types(raw.info, eeg=True, ecog=True, seeg=True, exclude=[])
    if len(eeg_picks) == 0:
        eeg_picks = np.arange(len(raw.ch_names))

    data = raw.get_data(picks=eeg_picks)
    ch_names = [raw.ch_names[i] for i in eeg_picks]
    bad_channels: set[str] = set()

    ptp = np.ptp(data, axis=1)
    flat_mask = ptp < flat_threshold
    for i in np.where(flat_mask)[0]:
        bad_channels.add(ch_names[i])

    mean_abs = np.mean(np.abs(data), axis=1)
    global_mean = np.mean(mean_abs)
    global_std = np.std(mean_abs)
    if global_std > 0:
        high_mask = mean_abs > global_mean + amplitude_threshold_sd * global_std
        for i in np.where(high_mask)[0]:
            bad_channels.add(ch_names[i])

    variances = np.var(data, axis=1)
    median_var = np.median(variances)
    if median_var > 0:
        low_var_mask = variances < min_variance_ratio * median_var
        for i in np.where(low_var_mask)[0]:
            bad_channels.add(ch_names[i])

    return sorted(bad_channels)


def mark_bad_channels(raw: mne.io.Raw, bad_channels: list[str]) -> mne.io.Raw:
    """Mark channels as bad in the Raw object.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object.
    bad_channels : list[str]
        Channel names to mark as bad.

    Returns
    -------
    mne.io.Raw
        The same Raw object with bads updated (in-place).
    """
    existing_bads = set(raw.info["bads"])
    raw.info["bads"] = sorted(existing_bads | set(bad_channels))
    return raw
