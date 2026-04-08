"""Schema validation for Cogitate iEEG behavioral, electrode, and Laplace data."""

from __future__ import annotations

from typing import Any

import pandas as pd


BEHAVIOR_REQUIRED_COLUMNS = [
    "expName",
    "block",
    "miniBlock",
    "trial",
    "miniBlockType",
    "targ1",
    "targ2",
    "plndStimulusDur",
    "plndJitterDur",
    "dsrdResponse",
    "event",
    "time",
    "eventType",
]

VALID_EVENT_TYPES = {"Fixation", "Jitter", "Stimulus"}
VALID_DURATIONS = {0.5, 1.0, 1.5}

ELECTRODE_REQUIRED_COLUMNS = ["name", "x", "y", "z"]


def validate_behavior(df: pd.DataFrame) -> list[str]:
    """Validate behavioral CSV structure and value ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Behavioral data loaded from CSV.

    Returns
    -------
    list[str]
        List of validation error messages. Empty if valid.
    """
    errors: list[str] = []

    missing_cols = set(BEHAVIOR_REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {sorted(missing_cols)}")
        return errors

    invalid_types = set(df["eventType"].dropna().unique()) - VALID_EVENT_TYPES
    if invalid_types:
        errors.append(f"Invalid eventType values: {invalid_types}")

    stim_rows = df[df["eventType"] == "Stimulus"]
    if not stim_rows.empty:
        durations = set(stim_rows["plndStimulusDur"].dropna().unique())
        invalid_durs = durations - VALID_DURATIONS
        if invalid_durs:
            errors.append(f"Unexpected plndStimulusDur values: {invalid_durs}")

    time_col = df["time"].dropna()
    if len(time_col) > 1:
        diffs = time_col.diff().iloc[1:]
        if (diffs < 0).any():
            n_violations = (diffs < 0).sum()
            errors.append(
                f"Timestamps not monotonically increasing: {n_violations} violations"
            )

    if df.empty:
        errors.append("Behavioral DataFrame is empty")

    return errors


def validate_electrodes(df: pd.DataFrame) -> list[str]:
    """Validate electrode coordinate table.

    Parameters
    ----------
    df : pd.DataFrame
        Electrode data loaded from TSV.

    Returns
    -------
    list[str]
        List of validation error messages. Empty if valid.
    """
    errors: list[str] = []

    missing_cols = set(ELECTRODE_REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing electrode columns: {sorted(missing_cols)}")
        return errors

    if df["name"].duplicated().any():
        dups = df["name"][df["name"].duplicated()].tolist()
        errors.append(f"Duplicate electrode names: {dups}")

    for coord in ["x", "y", "z"]:
        col = pd.to_numeric(df[coord], errors="coerce")
        n_nan = col.isna().sum()
        if n_nan > 0:
            errors.append(f"Non-numeric values in electrode '{coord}': {n_nan} entries")

    if df.empty:
        errors.append("Electrode DataFrame is empty")

    return errors


def validate_laplace_map(
    mapping: dict[str, dict], channel_names: list[str] | None = None
) -> list[str]:
    """Validate Laplace re-referencing mapping.

    Parameters
    ----------
    mapping : dict
        Channel -> {"ref_1": str|None, "ref_2": str|None}.
    channel_names : list[str] or None
        If provided, check that referenced channels exist.

    Returns
    -------
    list[str]
        List of validation error messages. Empty if valid.
    """
    errors: list[str] = []

    if not mapping:
        errors.append("Laplace mapping is empty")
        return errors

    for ch_name, refs in mapping.items():
        if not isinstance(refs, dict):
            errors.append(f"Channel '{ch_name}': refs must be a dict, got {type(refs)}")
            continue

        if "ref_1" not in refs or "ref_2" not in refs:
            errors.append(f"Channel '{ch_name}': missing ref_1 or ref_2 keys")

    if channel_names is not None:
        ch_set = set(channel_names)
        for ch_name, refs in mapping.items():
            if not isinstance(refs, dict):
                continue
            for ref_key in ("ref_1", "ref_2"):
                ref_val = refs.get(ref_key)
                if ref_val is not None and ref_val not in ch_set:
                    errors.append(
                        f"Channel '{ch_name}': {ref_key}='{ref_val}' not in channel list"
                    )

    return errors
