"""Trigger alignment and trial epoching from behavioral event logs."""

from __future__ import annotations

from typing import Any

import numpy as np
import mne
import pandas as pd


def extract_trigger_events(
    raw: mne.io.Raw,
    trigger_channels: list[str],
    threshold_sd: float = 3.0,
) -> np.ndarray:
    """Extract trigger event times from DC trigger channels.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data containing trigger channels.
    trigger_channels : list[str]
        Names of trigger channels (e.g., ["DC5", "DC6"]).
    threshold_sd : float
        Threshold in standard deviations for trigger detection.

    Returns
    -------
    np.ndarray
        Array of trigger onset times in seconds, sorted.
    """
    if not raw.preload:
        raw = raw.copy().load_data()

    available = [ch for ch in trigger_channels if ch in raw.ch_names]
    if not available:
        return np.array([])

    trigger_data = raw.get_data(picks=available)
    combined = np.sum(np.abs(trigger_data), axis=0)

    mean_val = np.mean(combined)
    std_val = np.std(combined)
    if std_val == 0:
        return np.array([])

    threshold = mean_val + threshold_sd * std_val
    above = combined > threshold

    onsets = np.where(np.diff(above.astype(int)) == 1)[0] + 1
    times = onsets / raw.info["sfreq"]

    return times


def align_triggers(
    trigger_times: np.ndarray,
    behavior_times: np.ndarray,
    max_offset_s: float = 5.0,
) -> float:
    """Compute clock offset between neural and behavioral timestamps.

    Uses cross-correlation of inter-event intervals to find the best
    alignment, then computes median offset.

    Parameters
    ----------
    trigger_times : np.ndarray
        Trigger onset times from the neural recording (seconds).
    behavior_times : np.ndarray
        Event times from the behavioral log (seconds).
    max_offset_s : float
        Maximum allowable clock offset in seconds.

    Returns
    -------
    float
        Clock offset such that: neural_time = behavior_time + offset.

    Raises
    ------
    ValueError
        If alignment fails (too few events or offset exceeds max).
    """
    if len(trigger_times) < 3 or len(behavior_times) < 3:
        raise ValueError(
            f"Too few events for alignment: "
            f"{len(trigger_times)} triggers, {len(behavior_times)} behavioral events"
        )

    trig_iei = np.diff(trigger_times)
    beh_iei = np.diff(behavior_times)

    min_len = min(len(trig_iei), len(beh_iei))
    if min_len < 2:
        raise ValueError("Not enough inter-event intervals for alignment")

    best_rmse = np.inf
    best_shift = 0

    max_shift = min(len(trig_iei), len(beh_iei)) // 2
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            t_slice = trig_iei[shift : shift + min_len]
            b_slice = beh_iei[:min_len]
        else:
            t_slice = trig_iei[:min_len]
            b_slice = beh_iei[-shift : -shift + min_len]

        actual_len = min(len(t_slice), len(b_slice))
        if actual_len < 2:
            continue

        t_s = t_slice[:actual_len]
        b_s = b_slice[:actual_len]

        rmse = np.sqrt(np.mean((t_s - b_s) ** 2))
        if rmse < best_rmse or (rmse == best_rmse and abs(shift) < abs(best_shift)):
            best_rmse = rmse
            best_shift = shift

    median_iei = np.median(np.concatenate([trig_iei, beh_iei]))
    if median_iei > 0 and best_rmse > 0.5 * median_iei:
        raise ValueError(
            f"Poor trigger alignment (best RMSE = {best_rmse:.3f}s, "
            f"median IEI = {median_iei:.3f}s). "
            f"Check trigger channels and behavioral log."
        )

    if best_shift >= 0:
        matched_trig = trigger_times[best_shift : best_shift + min_len + 1]
        matched_beh = behavior_times[: min_len + 1]
    else:
        matched_trig = trigger_times[: min_len + 1]
        matched_beh = behavior_times[-best_shift : -best_shift + min_len + 1]

    actual_len = min(len(matched_trig), len(matched_beh))
    offsets = matched_trig[:actual_len] - matched_beh[:actual_len]
    offset = float(np.median(offsets))

    if abs(offset) > max_offset_s:
        raise ValueError(
            f"Clock offset ({offset:.3f}s) exceeds maximum ({max_offset_s}s)"
        )

    return offset


def create_epochs(
    raw: mne.io.Raw,
    behavior: pd.DataFrame,
    tmin: float = -0.5,
    tmax: float = 2.0,
    baseline: tuple[float, float] | None = None,
    clock_offset: float = 0.0,
) -> mne.Epochs:
    """Create MNE Epochs from Raw data using behavioral event log.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed raw data (should be preloaded).
    behavior : pd.DataFrame
        Behavioral log with columns: eventType, time, plndStimulusDur,
        miniBlockType, event, dsrdResponse, trial, block, miniBlock.
    tmin : float
        Start of epoch relative to stimulus onset (seconds).
    tmax : float
        End of epoch relative to stimulus onset (seconds).
    baseline : tuple or None
        Baseline correction window (tmin, tmax) or None.
    clock_offset : float
        Offset to add to behavioral times to align with neural clock.

    Returns
    -------
    mne.Epochs
        Epoched data with metadata.
    """
    stim_events = behavior[behavior["eventType"] == "Stimulus"].copy()
    if stim_events.empty:
        raise ValueError("No stimulus events found in behavioral log")

    stim_events = stim_events.reset_index(drop=True)

    neural_times = stim_events["time"].values + clock_offset
    sfreq = raw.info["sfreq"]
    sample_indices = (neural_times * sfreq).astype(int)

    n_samples = raw.n_times
    valid_mask = (sample_indices >= 0) & (sample_indices < n_samples)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        import warnings
        warnings.warn(
            f"Dropping {n_invalid} events outside recording boundaries",
            stacklevel=2,
        )
        stim_events = stim_events[valid_mask].reset_index(drop=True)
        sample_indices = sample_indices[valid_mask]

    events_array = np.column_stack([
        sample_indices,
        np.zeros(len(sample_indices), dtype=int),
        np.ones(len(sample_indices), dtype=int),
    ])

    metadata = _build_metadata(stim_events)

    epochs = mne.Epochs(
        raw,
        events=events_array,
        event_id={"stimulus": 1},
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        metadata=metadata,
        preload=True,
        verbose=False,
    )

    return epochs


def _build_metadata(stim_events: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.DataFrame()
    metadata["trial"] = stim_events["trial"].values
    metadata["block"] = stim_events["block"].values
    metadata["miniBlock"] = stim_events["miniBlock"].values
    metadata["duration"] = stim_events["plndStimulusDur"].values
    metadata["miniBlockType"] = stim_events["miniBlockType"].values
    metadata["event_code"] = stim_events["event"].values
    metadata["is_target"] = (stim_events["dsrdResponse"].values > 0).astype(int)

    metadata["category"] = _infer_category(stim_events["event"].values)

    target_cats = stim_events["miniBlockType"].values
    inferred_cats = metadata["category"].values
    metadata["task_relevant"] = _determine_task_relevance(
        inferred_cats, target_cats
    ).astype(int)

    return metadata


def _infer_category(event_codes: np.ndarray) -> np.ndarray:
    """Infer stimulus category from event codes.

    Cogitate event code convention (approximate):
    - 1xxx: Faces
    - 2xxx: Objects
    - 3xxx: Letters
    - 4xxx: False fonts / Symbols

    This is a heuristic; actual mapping should be verified per dataset.
    """
    categories = []
    for code in event_codes:
        code_int = int(abs(code))
        first_digit = code_int // 1000
        if first_digit == 1:
            categories.append("face")
        elif first_digit == 2:
            categories.append("object")
        elif first_digit == 3:
            categories.append("letter")
        elif first_digit == 4:
            categories.append("false_font")
        else:
            categories.append("unknown")
    return np.array(categories)


def _determine_task_relevance(
    stimulus_categories: np.ndarray,
    miniblock_types: np.ndarray,
) -> np.ndarray:
    """Determine if each stimulus is task-relevant based on miniBlock target categories."""
    relevance = np.zeros(len(stimulus_categories), dtype=bool)
    for i, (cat, mb_type) in enumerate(zip(stimulus_categories, miniblock_types)):
        if isinstance(mb_type, str):
            targets = [t.strip().lower() for t in mb_type.split("&")]
            relevance[i] = cat.lower() in targets
    return relevance
