"""Laplace re-referencing for iEEG data using provided mapping."""

from __future__ import annotations

import numpy as np
import mne


def apply_laplace(
    raw: mne.io.Raw,
    laplace_map: dict[str, dict],
) -> mne.io.Raw:
    """Apply Laplace re-referencing using the provided channel mapping.

    For each channel, the Laplace reference is computed as:
        channel_laplace = channel - mean(ref_1, ref_2)

    If only one reference is available (ref_1 or ref_2 is None),
    bipolar referencing is used (subtract the single reference).
    If both references are None, the channel is left unchanged.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data (must be preloaded).
    laplace_map : dict[str, dict]
        Channel -> {"ref_1": str|None, "ref_2": str|None} mapping.

    Returns
    -------
    mne.io.Raw
        Re-referenced data (modified in-place).
    """
    if not raw.preload:
        raw.load_data()

    available_channels = set(raw.ch_names)
    original_data = raw.get_data().copy()
    new_data = original_data.copy()

    ch_index = {name: i for i, name in enumerate(raw.ch_names)}

    for ch_name, refs in laplace_map.items():
        if ch_name not in available_channels:
            continue

        ch_idx = ch_index[ch_name]
        ref_1 = refs.get("ref_1")
        ref_2 = refs.get("ref_2")

        valid_refs = []
        if ref_1 is not None and ref_1 in available_channels:
            valid_refs.append(ch_index[ref_1])
        if ref_2 is not None and ref_2 in available_channels:
            valid_refs.append(ch_index[ref_2])

        if len(valid_refs) == 0:
            continue

        ref_mean = np.mean(original_data[valid_refs, :], axis=0)
        new_data[ch_idx, :] = original_data[ch_idx, :] - ref_mean

    raw._data = new_data
    return raw
