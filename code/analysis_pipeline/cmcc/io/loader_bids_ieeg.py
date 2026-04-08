"""Load BIDS-format iEEG data (ds004752): EDF recordings, events, electrodes.

Scientific context
------------------
ds004752 contains SEEG depth electrode recordings from 15 epilepsy patients
performing a verbal working memory (Sternberg) task at Schweizerische
Epilepsie-Klinik, Zurich. Each subject has 2-8 sessions of ~50 trials.
Sampling rates are 2000 or 4096 Hz depending on subject/system.

This loader reads standard BIDS iEEG layout and returns data structures
compatible with the CMCC analysis pipeline, enabling a modality-matched
independent replication test on data from a different lab, paradigm,
patient population, and recording system.

Units
-----
- iEEG voltages in microvolts (uV) from EDF files.
- Electrode coordinates in MNI space (mm).
- Sampling rates in Hz.
- Event onsets in seconds; event samples in integer sample indices.

Assumptions
-----------
- All channels labeled SEEG or ECOG in channels.tsv are neural channels.
- No Laplace re-referencing is applied (SEEG depth electrodes have
  different spatial geometry than ECoG grids; bipolar re-referencing
  along the shaft is the standard alternative but is optional here).
- Line noise is 50 Hz (European recording site).
- Each EDF file contains one session of concatenated trials.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd


@dataclass
class BidsIeegSubject:
    """Container for a single ds004752 subject's BIDS iEEG data.

    Attributes
    ----------
    subject_id : str
        BIDS subject identifier (e.g., "sub-01").
    sessions : list[str]
        Available session identifiers (e.g., ["ses-01", "ses-02"]).
    raw : dict[str, mne.io.Raw]
        Session ID -> MNE Raw object mapping.
    events : dict[str, pd.DataFrame]
        Session ID -> BIDS events DataFrame.
    electrodes : pd.DataFrame
        Electrode coordinates (MNI) with anatomical labels.
    channels_info : dict[str, pd.DataFrame]
        Session ID -> BIDS channels DataFrame.
    sidecar : dict[str, dict]
        Session ID -> iEEG JSON sidecar metadata.
    """

    subject_id: str
    sessions: list[str] = field(default_factory=list)
    raw: dict[str, mne.io.Raw] = field(default_factory=dict)
    events: dict[str, pd.DataFrame] = field(default_factory=dict)
    electrodes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    channels_info: dict[str, pd.DataFrame] = field(default_factory=dict)
    sidecar: dict[str, dict] = field(default_factory=dict)


def discover_sessions(data_root: Path, subject_id: str) -> list[str]:
    """Find all session directories for a subject."""
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    subj_dir = data_root / sid
    if not subj_dir.is_dir():
        raise FileNotFoundError(f"Subject directory not found: {subj_dir}")
    sessions = sorted(
        d.name for d in subj_dir.iterdir()
        if d.is_dir() and d.name.startswith("ses-")
    )
    if not sessions:
        raise FileNotFoundError(f"No session directories found in {subj_dir}")
    return sessions


def _bids_stem(subject_id: str, session: str) -> str:
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    return f"{sid}_{session}_task-verbalWM_run-01"


def load_bids_edf(
    data_root: Path, subject_id: str, session: str,
) -> mne.io.Raw:
    """Load a single BIDS iEEG EDF file."""
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    stem = _bids_stem(subject_id, session)
    edf_path = data_root / sid / session / "ieeg" / f"{stem}_ieeg.edf"
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    return raw


def load_bids_events(
    data_root: Path, subject_id: str, session: str,
) -> pd.DataFrame:
    """Load BIDS events TSV for a session."""
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    stem = _bids_stem(subject_id, session)
    tsv_path = data_root / sid / session / "ieeg" / f"{stem}_events.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Events TSV not found: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    return df


def load_bids_electrodes(
    data_root: Path, subject_id: str, session: str = "ses-01",
) -> pd.DataFrame:
    """Load electrode coordinates (MNI) with anatomical labels."""
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    stem = _bids_stem(subject_id, session)
    tsv_path = data_root / sid / session / "ieeg" / f"{stem}_electrodes.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Electrodes TSV not found: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    return df


def load_bids_channels(
    data_root: Path, subject_id: str, session: str,
) -> pd.DataFrame:
    """Load BIDS channels TSV."""
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    stem = _bids_stem(subject_id, session)
    tsv_path = data_root / sid / session / "ieeg" / f"{stem}_channels.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Channels TSV not found: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    return df


def load_bids_sidecar(
    data_root: Path, subject_id: str, session: str,
) -> dict[str, Any]:
    """Load iEEG JSON sidecar metadata."""
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    stem = _bids_stem(subject_id, session)
    json_path = data_root / sid / session / "ieeg" / f"{stem}_ieeg.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r") as f:
        return json.load(f)


def get_neural_channels(channels_df: pd.DataFrame) -> list[str]:
    """Return channel names that are SEEG or ECOG (neural recording channels)."""
    neural_types = {"SEEG", "ECOG"}
    mask = channels_df["type"].isin(neural_types)
    return channels_df.loc[mask, "name"].tolist()


def load_bids_ieeg_subject(
    subject_id: str,
    data_root: str | Path,
    sessions: list[str] | None = None,
    preload: bool = False,
) -> BidsIeegSubject:
    """Load all data for a single ds004752 subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., "sub-01" or "01").
    data_root : str or Path
        Root directory of ds004752.
    sessions : list[str] or None
        Session IDs to load. If None, discovers all sessions.
    preload : bool
        Whether to preload EDF data into memory.

    Returns
    -------
    BidsIeegSubject
    """
    data_root = Path(data_root)
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"

    if sessions is None:
        sessions = discover_sessions(data_root, sid)

    raw_dict: dict[str, mne.io.Raw] = {}
    events_dict: dict[str, pd.DataFrame] = {}
    channels_dict: dict[str, pd.DataFrame] = {}
    sidecar_dict: dict[str, dict] = {}

    import warnings
    for ses in sessions:
        try:
            raw = load_bids_edf(data_root, sid, ses)
            if preload:
                raw.load_data()
            raw_dict[ses] = raw
        except FileNotFoundError as e:
            warnings.warn(f"Skipping {ses}: {e}", stacklevel=2)

        try:
            events_dict[ses] = load_bids_events(data_root, sid, ses)
        except FileNotFoundError:
            pass

        try:
            channels_dict[ses] = load_bids_channels(data_root, sid, ses)
        except FileNotFoundError:
            pass

        try:
            sidecar_dict[ses] = load_bids_sidecar(data_root, sid, ses)
        except FileNotFoundError:
            pass

    try:
        electrodes = load_bids_electrodes(data_root, sid, sessions[0])
    except FileNotFoundError:
        electrodes = pd.DataFrame()

    return BidsIeegSubject(
        subject_id=sid,
        sessions=list(raw_dict.keys()),
        raw=raw_dict,
        events=events_dict,
        electrodes=electrodes,
        channels_info=channels_dict,
        sidecar=sidecar_dict,
    )
