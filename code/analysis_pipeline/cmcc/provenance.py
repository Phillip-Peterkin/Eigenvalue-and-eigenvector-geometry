"""Run provenance logging: timestamps, versions, config snapshots, and hashes."""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

REFERENCE_DISTRIBUTIONS = (
    "operator-geometry",
    "mne",
    "numpy",
    "scipy",
    "pandas",
    "powerlaw",
    "antropy",
    "neurokit2",
    "scikit-learn",
    "matplotlib",
    "seaborn",
    "h5py",
    "pyyaml",
    "statsmodels",
)


@dataclass
class RunProvenance:
    timestamp: str
    config_hash: str
    config_snapshot: dict[str, Any]
    git_commit: str | None
    python_version: str
    platform: str
    package_versions: dict[str, str]
    random_seed: int | None
    subject_ids: list[str]
    cohort_manifest: str | None


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_package_versions() -> dict[str, str]:
    """Return installed distribution versions using package metadata names.

    Distribution metadata avoids import-name mismatches such as
    `scikit-learn` -> `sklearn` and `pyyaml` -> `yaml`.
    """
    versions: dict[str, str] = {}
    for distribution in REFERENCE_DISTRIBUTIONS:
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def _config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log_run(config: dict[str, Any], results_dir: str | Path) -> RunProvenance:
    """Create and save a provenance record for the current run."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_config = config.get("data", {})

    prov = RunProvenance(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_hash=_config_hash(config),
        config_snapshot=config,
        git_commit=_get_git_commit(),
        python_version=sys.version,
        platform=platform.platform(),
        package_versions=_get_package_versions(),
        random_seed=config.get("random_seed"),
        subject_ids=list(data_config.get("subjects", [])),
        cohort_manifest=data_config.get("cohort_manifest"),
    )

    prov_path = results_dir / f"provenance_{prov.config_hash[:16]}.json"
    with prov_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(prov), handle, indent=2, default=str)
        handle.write("\n")
    return prov


def save_config_snapshot(config: dict[str, Any], results_dir: str | Path) -> Path:
    """Save the exact configuration used for a run."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    digest = _config_hash(config)
    path = results_dir / f"config_snapshot_{digest[:16]}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, default_flow_style=False, sort_keys=True)
    return path


def save_results_hdf5(
    results: dict[str, Any],
    results_dir: str | Path,
    filename: str = "results.h5",
) -> Path:
    """Save nested numeric results to HDF5."""
    import h5py
    import numpy as np

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename

    def _write_group(group: h5py.Group, data: dict) -> None:
        for key, value in data.items():
            key_str = str(key)
            if isinstance(value, dict):
                sub = group.create_group(key_str)
                _write_group(sub, value)
            elif isinstance(value, np.ndarray):
                group.create_dataset(key_str, data=value)
            elif isinstance(value, (int, float)):
                group.create_dataset(key_str, data=value)
            elif isinstance(value, str):
                group.attrs[key_str] = value
            elif isinstance(value, list):
                try:
                    group.create_dataset(key_str, data=np.asarray(value))
                except (ValueError, TypeError):
                    group.attrs[key_str] = json.dumps(value)

    with h5py.File(path, "w") as handle:
        _write_group(handle, results)
    return path


def save_summary_json(
    summary: dict[str, Any],
    results_dir: str | Path,
    filename: str = "summary.json",
) -> Path:
    """Save a human-readable JSON summary of analysis results."""
    import numpy as np

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename

    def _make_serializable(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {str(key): _make_serializable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(value) for value in obj]
        if isinstance(obj, set):
            return sorted(obj)
        return obj

    with path.open("w", encoding="utf-8") as handle:
        json.dump(_make_serializable(summary), handle, indent=2, default=str)
        handle.write("\n")
    return path
