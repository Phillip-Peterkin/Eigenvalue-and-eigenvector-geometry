"""Load Cogitate iEEG data: EDF recordings, behavioral CSVs, electrode metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mne
import pandas as pd


@dataclass
class SubjectData:
    """Container for all data associated with a single Cogitate subject.

    Attributes
    ----------
    subject_id : str
        Subject identifier (e.g., "CE103").
    site : str
        Recording site derived from subject_id prefix ("CE", "CF", "CG").
    raw : dict[str, mne.io.Raw]
        Run ID -> MNE Raw object mapping.
    behavior : dict[str, pd.DataFrame]
        Run ID -> behavioral log DataFrame mapping.
    electrodes : pd.DataFrame
        Electrode coordinates in fsaverage space (columns: name, x, y, z, size).
    laplace_map : dict[str, dict]
        Channel -> {"ref_1": str|None, "ref_2": str|None} mapping.
    metadata : dict[str, Any]
        Combined CRF and EXQU metadata.
    """

    subject_id: str
    site: str
    raw: dict[str, mne.io.Raw] = field(default_factory=dict)
    behavior: dict[str, pd.DataFrame] = field(default_factory=dict)
    electrodes: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    laplace_map: dict[str, dict] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _derive_site(subject_id: str) -> str:
    prefix = subject_id[:2].upper()
    if prefix not in ("CE", "CF", "CG"):
        raise ValueError(
            f"Unknown site prefix '{prefix}' from subject_id '{subject_id}'. "
            f"Expected CE, CF, or CG."
        )
    return prefix


def _find_subject_dir(data_root: Path, subject_id: str) -> Path:
    ecog_dir = data_root / f"{subject_id}_ECOG_1"
    if not ecog_dir.is_dir():
        raise FileNotFoundError(
            f"Subject directory not found: {ecog_dir}"
        )
    return ecog_dir


def load_edf(subject_dir: Path, subject_id: str, run_id: str) -> mne.io.Raw:
    """Load a single EDF file for a given run.

    Parameters
    ----------
    subject_dir : Path
        Path to the subject's ECOG directory (e.g., CE103_ECOG_1/).
    subject_id : str
        Subject identifier.
    run_id : str
        Run identifier (e.g., "DurR1").

    Returns
    -------
    mne.io.Raw
        Raw EDF data (not preloaded).

    Raises
    ------
    FileNotFoundError
        If the EDF file does not exist.
    """
    edf_dir = subject_dir / f"{run_id}_" / "EDF"
    if not edf_dir.is_dir():
        edf_dir = subject_dir / run_id / "EDF"
    edf_pattern = f"{subject_id}_ECoG_1_{run_id}.EDF"
    edf_path = edf_dir / edf_pattern

    if not edf_path.exists():
        candidates = list(edf_dir.glob("*.EDF")) + list(edf_dir.glob("*.edf"))
        if len(candidates) == 1:
            edf_path = candidates[0]
        elif len(candidates) > 1:
            raise FileNotFoundError(
                f"Multiple EDF files found in {edf_dir}: {candidates}. "
                f"Expected {edf_pattern}."
            )
        else:
            raise FileNotFoundError(
                f"EDF file not found: {edf_path}"
            )

    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    return raw


def load_behavior(subject_dir: Path, subject_id: str, run_id: str) -> pd.DataFrame:
    """Load behavioral CSV for a given run.

    Parameters
    ----------
    subject_dir : Path
        Path to the subject's ECOG directory.
    subject_id : str
        Subject identifier.
    run_id : str
        Run identifier (e.g., "DurR1"). Maps to CSV suffix "RawDurR{N}".

    Returns
    -------
    pd.DataFrame
        Behavioral log with columns: expName, block, miniBlock, trial,
        miniBlockType, targ1, targ2, plndStimulusDur, plndJitterDur,
        dsrdResponse, event, time, eventType.
    """
    run_num = run_id.replace("DurR", "")
    beh_dir = subject_dir / "BEH"
    csv_pattern = f"{subject_id}_Beh_1_RawDurR{run_num}.csv"
    csv_path = beh_dir / csv_pattern

    if not csv_path.exists():
        candidates = list(beh_dir.glob(f"*RawDurR{run_num}*"))
        if len(candidates) == 1:
            csv_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Behavioral CSV not found: {csv_path}"
            )

    df = pd.read_csv(csv_path)
    return df


def load_electrodes(subject_dir: Path, subject_id: str) -> pd.DataFrame:
    """Load electrode coordinates in fsaverage space.

    Parameters
    ----------
    subject_dir : Path
        Path to the subject's ECOG directory.
    subject_id : str
        Subject identifier.

    Returns
    -------
    pd.DataFrame
        Electrode table with columns: name, x, y, z, size.
    """
    elec_dir = subject_dir / "METADATA" / "electrode_coordinates"
    tsv_pattern = f"sub-{subject_id}_ses-1_space-fsaverage_electrodes.tsv"
    tsv_path = elec_dir / tsv_pattern

    if not tsv_path.exists():
        candidates = list(elec_dir.glob("*electrodes.tsv"))
        if len(candidates) == 1:
            tsv_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Electrode TSV not found: {tsv_path}"
            )

    df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8-sig")
    return df


def load_laplace_map(subject_dir: Path, subject_id: str) -> dict[str, dict]:
    """Load Laplace re-referencing mapping.

    Parameters
    ----------
    subject_dir : Path
        Path to the subject's ECOG directory.
    subject_id : str
        Subject identifier.

    Returns
    -------
    dict[str, dict]
        Channel name -> {"ref_1": str|None, "ref_2": str|None}.
    """
    elec_dir = subject_dir / "METADATA" / "electrode_coordinates"
    json_pattern = f"sub-{subject_id}_ses-1_laplace_mapping_ieeg.json"
    json_path = elec_dir / json_pattern

    if not json_path.exists():
        candidates = list(elec_dir.glob("*laplace_mapping*"))
        if len(candidates) == 1:
            json_path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Laplace mapping JSON not found: {json_path}"
            )

    with open(json_path, "r") as f:
        mapping = json.load(f)

    return mapping


def _load_metadata(subject_dir: Path, subject_id: str) -> dict[str, Any]:
    meta_dir = subject_dir / "METADATA"
    combined: dict[str, Any] = {}

    crf_path = meta_dir / f"{subject_id}_CRF.json"
    if crf_path.exists():
        with open(crf_path, "r") as f:
            combined["crf"] = json.load(f)

    exqu_path = meta_dir / f"{subject_id}_EXQU.json"
    if exqu_path.exists():
        with open(exqu_path, "r") as f:
            combined["exqu"] = json.load(f)

    return combined


def load_subject(
    subject_id: str,
    data_root: str | Path,
    runs: list[str] | None = None,
    preload_edf: bool = False,
) -> SubjectData:
    """Load all data for a single Cogitate iEEG subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g., "CE103").
    data_root : str or Path
        Root directory of the Cogitate iEEG dataset.
    runs : list[str] or None
        Run IDs to load. If None, loads DurR1-DurR5.
    preload_edf : bool
        Whether to preload EDF data into memory.

    Returns
    -------
    SubjectData
        Complete subject data container.
    """
    data_root = Path(data_root)
    if runs is None:
        runs = [f"DurR{i}" for i in range(1, 6)]

    site = _derive_site(subject_id)
    subject_dir = _find_subject_dir(data_root, subject_id)

    raw_dict: dict[str, mne.io.Raw] = {}
    beh_dict: dict[str, pd.DataFrame] = {}

    for run_id in runs:
        try:
            raw = load_edf(subject_dir, subject_id, run_id)
            if preload_edf:
                raw.load_data()
            raw_dict[run_id] = raw
        except FileNotFoundError as e:
            import warnings
            warnings.warn(f"Skipping {run_id}: {e}", stacklevel=2)

        try:
            beh = load_behavior(subject_dir, subject_id, run_id)
            beh_dict[run_id] = beh
        except FileNotFoundError as e:
            import warnings
            warnings.warn(f"Skipping behavior for {run_id}: {e}", stacklevel=2)

    electrodes = load_electrodes(subject_dir, subject_id)
    laplace_map = load_laplace_map(subject_dir, subject_id)
    metadata = _load_metadata(subject_dir, subject_id)

    return SubjectData(
        subject_id=subject_id,
        site=site,
        raw=raw_dict,
        behavior=beh_dict,
        electrodes=electrodes,
        laplace_map=laplace_map,
        metadata=metadata,
    )
