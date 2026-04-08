"""Load BIDS-format CHB-MIT Scalp EEG data: EDF recordings, seizure annotations.

Scientific context
------------------
The CHB-MIT Scalp EEG Database contains continuous long-term EEG monitoring
from 23 pediatric subjects with intractable epilepsy, recorded at Boston
Children's Hospital. The BIDS version reorganizes the original PhysioNet
dataset into standard BIDS layout with 18 bipolar channels (double banana
montage from the 10-20 system), sampled at 256 Hz.

This loader reads the BIDS layout, parses seizure annotations from events
TSV files, and builds a structured seizure catalog that maps each seizure
to its recording file with temporal context (pre-ictal availability,
inter-seizure intervals). The catalog enforces configurable inclusion
criteria for the seizure prediction analysis.

Units
-----
- EEG voltages in microvolts (uV) from EDF files.
- Seizure onsets and durations in seconds from recording start.
- Sampling rate: 256 Hz.
- Recording durations in seconds.

Assumptions
-----------
- BIDS layout: sub-XX/ses-XX/eeg/sub-XX_ses-XX_task-szMonitoring_run-XX_eeg.edf
- Events TSV contains columns: onset, duration, eventType, recordingDuration.
- eventType starting with 'sz' indicates a seizure; 'bckg' is background.
- Data is already in a double banana bipolar montage (18 channels).
  CSD is NOT applicable to bipolar data.
- Line noise is 60 Hz (US recording site).
- Some recordings contain multiple seizures; some contain none.
- sub-01 has two sessions (ses-01, ses-02); all others have one.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import mne
import numpy as np
import pandas as pd


@dataclass
class SeizureEvent:
    """Single annotated seizure with temporal context.

    Attributes
    ----------
    subject_id : str
        BIDS subject ID (e.g., "sub-01").
    session : str
        BIDS session ID (e.g., "ses-01").
    run : str
        BIDS run label (e.g., "run-03").
    onset_sec : float
        Seizure onset in seconds from recording start.
    duration_sec : float
        Seizure duration in seconds.
    offset_sec : float
        Seizure offset (onset + duration) in seconds.
    event_type : str
        BIDS eventType (e.g., "sz", "sz_foc_ia").
    recording_duration : float
        Total recording length in seconds.
    edf_path : Path
        Absolute path to the EDF file.
    pre_ictal_available : float
        Seconds of data before seizure onset in this recording.
    post_ictal_available : float
        Seconds of data after seizure offset in this recording.
    prev_seizure_offset : float | None
        Offset (seconds) of the previous seizure in this recording, or None.
    """
    subject_id: str
    session: str
    run: str
    onset_sec: float
    duration_sec: float
    offset_sec: float
    event_type: str
    recording_duration: float
    edf_path: Path
    pre_ictal_available: float
    post_ictal_available: float
    prev_seizure_offset: float | None
    recording_start_datetime: str | None = None


@dataclass
class SubjectCatalog:
    """Per-subject summary of seizure data.

    Attributes
    ----------
    subject_id : str
        BIDS subject ID.
    sessions : list[str]
        Available sessions.
    seizures : list[SeizureEvent]
        All discovered seizures (before inclusion filtering).
    n_seizures : int
        Total seizures discovered.
    n_eligible : int
        Seizures meeting inclusion criteria.
    eligible_seizures : list[SeizureEvent]
        Seizures meeting inclusion criteria.
    total_recording_hours : float
        Sum of all recording durations in hours.
    """
    subject_id: str
    sessions: list[str] = field(default_factory=list)
    seizures: list[SeizureEvent] = field(default_factory=list)
    n_seizures: int = 0
    n_eligible: int = 0
    eligible_seizures: list[SeizureEvent] = field(default_factory=list)
    total_recording_hours: float = 0.0


def discover_subjects(data_root: str | Path) -> list[str]:
    """Find all subject directories in the BIDS root."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    subjects = sorted(
        d.name for d in data_root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )
    return subjects


def discover_sessions(data_root: str | Path, subject_id: str) -> list[str]:
    """Find all session directories for a subject."""
    data_root = Path(data_root)
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


def discover_runs(
    data_root: str | Path, subject_id: str, session: str,
) -> list[str]:
    """Find all run labels for a subject/session by scanning EDF files."""
    data_root = Path(data_root)
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    eeg_dir = data_root / sid / session / "eeg"
    if not eeg_dir.is_dir():
        raise FileNotFoundError(f"EEG directory not found: {eeg_dir}")
    runs = sorted(set(
        _extract_run_label(f.name)
        for f in eeg_dir.glob(f"{sid}_{session}_task-szMonitoring_run-*_eeg.edf")
    ))
    return runs


def _extract_run_label(filename: str) -> str:
    """Extract run-XX from a BIDS filename."""
    for part in filename.split("_"):
        if part.startswith("run-"):
            return part
    raise ValueError(f"No run label found in filename: {filename}")


def _bids_stem(subject_id: str, session: str, run: str) -> str:
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    return f"{sid}_{session}_task-szMonitoring_{run}"


def load_events_tsv(
    data_root: str | Path, subject_id: str, session: str, run: str,
) -> pd.DataFrame:
    """Load BIDS events TSV for a specific run."""
    data_root = Path(data_root)
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    stem = _bids_stem(sid, session, run)
    tsv_path = data_root / sid / session / "eeg" / f"{stem}_events.tsv"
    if not tsv_path.exists():
        raise FileNotFoundError(f"Events TSV not found: {tsv_path}")
    df = pd.read_csv(tsv_path, sep="\t")
    return df


def load_eeg_sidecar(
    data_root: str | Path, subject_id: str, session: str, run: str,
) -> dict[str, Any]:
    """Load EEG JSON sidecar metadata."""
    data_root = Path(data_root)
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    stem = _bids_stem(sid, session, run)
    json_path = data_root / sid / session / "eeg" / f"{stem}_eeg.json"
    if not json_path.exists():
        return {}
    with open(json_path, "r") as f:
        return json.load(f)


def load_raw_edf(
    data_root: str | Path, subject_id: str, session: str, run: str,
    preload: bool = False,
) -> mne.io.Raw:
    """Load a single EDF file from the BIDS layout."""
    data_root = Path(data_root)
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    stem = _bids_stem(sid, session, run)
    edf_path = data_root / sid / session / "eeg" / f"{stem}_eeg.edf"
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=preload, verbose=False)
    return raw


def _get_edf_path(
    data_root: Path, subject_id: str, session: str, run: str,
) -> Path:
    sid = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    stem = _bids_stem(sid, session, run)
    return data_root / sid / session / "eeg" / f"{stem}_eeg.edf"


def _parse_seizures_from_events(
    events_df: pd.DataFrame,
    subject_id: str,
    session: str,
    run: str,
    edf_path: Path,
) -> list[SeizureEvent]:
    """Extract seizure events from an events DataFrame."""
    sz_mask = events_df["eventType"].str.startswith("sz", na=False)
    sz_rows = events_df[sz_mask].sort_values("onset").reset_index(drop=True)

    if sz_rows.empty:
        return []

    rec_dur_col = "recordingDuration"
    if rec_dur_col in events_df.columns:
        rec_dur = float(events_df[rec_dur_col].dropna().iloc[0])
    else:
        rec_dur = float("nan")

    rec_dt: str | None = None
    if "dateTime" in events_df.columns:
        dt_vals = events_df["dateTime"].dropna()
        dt_vals = dt_vals[dt_vals != "n/a"]
        if len(dt_vals) > 0:
            rec_dt = str(dt_vals.iloc[0])

    seizures: list[SeizureEvent] = []
    for i, row in sz_rows.iterrows():
        onset = float(row["onset"])
        duration = float(row["duration"])
        if not np.isfinite(onset) or not np.isfinite(duration):
            continue
        offset = onset + duration

        if i == 0 or len(seizures) == 0:
            prev_offset = None
        else:
            prev_offset = seizures[-1].offset_sec

        post_avail = rec_dur - offset if np.isfinite(rec_dur) else 0.0

        seizures.append(SeizureEvent(
            subject_id=subject_id,
            session=session,
            run=run,
            onset_sec=onset,
            duration_sec=duration,
            offset_sec=offset,
            event_type=str(row["eventType"]),
            recording_duration=rec_dur,
            edf_path=edf_path,
            pre_ictal_available=onset,
            post_ictal_available=max(0.0, post_avail),
            prev_seizure_offset=prev_offset,
            recording_start_datetime=rec_dt,
        ))

    return seizures


def _apply_inclusion_criteria(
    seizures: list[SeizureEvent],
    min_preictal_sec: float = 1200.0,
    min_inter_seizure_sec: float = 3600.0,
) -> list[SeizureEvent]:
    """Filter seizures by inclusion criteria.

    Criteria:
    1. Seizure onset >= min_preictal_sec (sufficient pre-ictal data).
    2. If a previous seizure exists in the same recording, its offset must be
       >= min_inter_seizure_sec before the current onset.
    """
    eligible: list[SeizureEvent] = []
    for sz in seizures:
        if sz.pre_ictal_available < min_preictal_sec:
            continue
        if sz.prev_seizure_offset is not None:
            gap = sz.onset_sec - sz.prev_seizure_offset
            if gap < min_inter_seizure_sec:
                continue
        eligible.append(sz)
    return eligible


def _absolute_onset_sec(sz: SeizureEvent) -> float | None:
    """Parse recording_start_datetime + onset_sec into a Unix-like timestamp."""
    if sz.recording_start_datetime is None:
        return None
    try:
        rec_start = datetime.strptime(sz.recording_start_datetime, "%Y-%m-%d %H:%M:%S")
        epoch = datetime(2000, 1, 1)
        return (rec_start - epoch).total_seconds() + sz.onset_sec
    except (ValueError, TypeError):
        return None


def _enforce_cross_recording_gaps(seizures: list[SeizureEvent]) -> list[SeizureEvent]:
    """Update prev_seizure_offset to account for inter-seizure gaps across runs.

    When consecutive seizures fall in different EDF files, the per-run parser
    cannot compute their temporal gap. This function uses
    recording_start_datetime to establish absolute ordering and sets
    prev_seizure_offset so that _apply_inclusion_criteria correctly
    excludes seizures that are too close to a prior seizure in a
    different recording.
    """
    timed = [(sz, _absolute_onset_sec(sz)) for sz in seizures]
    timed_valid = [(sz, t) for sz, t in timed if t is not None]

    if len(timed_valid) < 2:
        return seizures

    timed_valid.sort(key=lambda x: x[1])

    for i in range(1, len(timed_valid)):
        sz_curr, t_curr_onset = timed_valid[i]
        sz_prev, t_prev_onset = timed_valid[i - 1]

        if sz_curr.edf_path == sz_prev.edf_path:
            continue

        t_prev_offset = t_prev_onset + sz_prev.duration_sec
        true_gap = t_curr_onset - t_prev_offset

        virtual_offset = sz_curr.onset_sec - true_gap

        if sz_curr.prev_seizure_offset is None:
            sz_curr.prev_seizure_offset = virtual_offset
        else:
            existing_gap = sz_curr.onset_sec - sz_curr.prev_seizure_offset
            if true_gap < existing_gap:
                sz_curr.prev_seizure_offset = virtual_offset

    return seizures


def build_seizure_catalog(
    data_root: str | Path,
    min_preictal_sec: float = 1200.0,
    min_inter_seizure_sec: float = 3600.0,
    subjects: list[str] | None = None,
) -> dict[str, SubjectCatalog]:
    """Build a complete seizure catalog from the BIDS CHB-MIT dataset.

    Parameters
    ----------
    data_root : str or Path
        Root directory of BIDS_CHB-MIT.
    min_preictal_sec : float
        Minimum seconds of pre-ictal data required (default 1200 = 20 min).
    min_inter_seizure_sec : float
        Minimum seconds between previous seizure offset and current onset
        (default 3600 = 60 min).
    subjects : list[str] or None
        Specific subjects to include. If None, discover all.

    Returns
    -------
    dict[str, SubjectCatalog]
        Mapping from subject_id to their catalog.
    """
    data_root = Path(data_root)

    if subjects is None:
        subject_ids = discover_subjects(data_root)
    else:
        subject_ids = [
            s if s.startswith("sub-") else f"sub-{s}" for s in subjects
        ]

    catalogs: dict[str, SubjectCatalog] = {}

    for sid in subject_ids:
        try:
            sessions = discover_sessions(data_root, sid)
        except FileNotFoundError:
            warnings.warn(f"Skipping {sid}: no sessions found", stacklevel=2)
            continue

        all_seizures: list[SeizureEvent] = []
        total_rec_sec = 0.0

        for ses in sessions:
            try:
                runs = discover_runs(data_root, sid, ses)
            except FileNotFoundError:
                continue

            for run in runs:
                edf_path = _get_edf_path(data_root, sid, ses, run)
                try:
                    events_df = load_events_tsv(data_root, sid, ses, run)
                except FileNotFoundError:
                    continue

                if "recordingDuration" in events_df.columns:
                    rd = events_df["recordingDuration"].dropna()
                    if len(rd) > 0:
                        total_rec_sec += float(rd.iloc[0])

                run_seizures = _parse_seizures_from_events(
                    events_df, sid, ses, run, edf_path,
                )
                all_seizures.extend(run_seizures)

        _enforce_cross_recording_gaps(all_seizures)

        eligible = _apply_inclusion_criteria(
            all_seizures, min_preictal_sec, min_inter_seizure_sec,
        )

        catalogs[sid] = SubjectCatalog(
            subject_id=sid,
            sessions=sessions,
            seizures=all_seizures,
            n_seizures=len(all_seizures),
            n_eligible=len(eligible),
            eligible_seizures=eligible,
            total_recording_hours=total_rec_sec / 3600.0,
        )

    return catalogs


def catalog_summary(catalogs: dict[str, SubjectCatalog]) -> dict[str, Any]:
    """Generate a human-readable summary of the seizure catalog."""
    n_subjects = len(catalogs)
    total_sz = sum(c.n_seizures for c in catalogs.values())
    total_eligible = sum(c.n_eligible for c in catalogs.values())
    total_hours = sum(c.total_recording_hours for c in catalogs.values())
    subjects_with_eligible = sum(
        1 for c in catalogs.values() if c.n_eligible > 0
    )
    return {
        "n_subjects": n_subjects,
        "total_seizures": total_sz,
        "total_eligible_seizures": total_eligible,
        "subjects_with_eligible_seizures": subjects_with_eligible,
        "total_recording_hours": round(total_hours, 1),
        "per_subject": {
            sid: {
                "n_seizures": c.n_seizures,
                "n_eligible": c.n_eligible,
                "recording_hours": round(c.total_recording_hours, 1),
            }
            for sid, c in catalogs.items()
        },
    }
