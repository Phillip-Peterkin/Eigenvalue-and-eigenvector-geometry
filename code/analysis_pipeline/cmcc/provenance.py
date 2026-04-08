"""Run provenance logging: timestamps, versions, config snapshots."""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RunProvenance:
    timestamp: str
    config_hash: str
    config_snapshot: dict[str, Any]
    git_commit: str | None
    python_version: str
    package_versions: dict[str, str]
    random_seed: int | None
    subject_ids: list[str]


def _get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _get_package_versions() -> dict[str, str]:
    packages = [
        "mne", "numpy", "scipy", "pandas", "powerlaw",
        "antropy", "neurokit2", "scikit-learn", "matplotlib",
        "seaborn", "h5py", "pyyaml",
    ]
    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg.replace("-", "_"))
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def _config_hash(config: dict[str, Any]) -> str:
    serialized = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def log_run(config: dict[str, Any], results_dir: str | Path) -> RunProvenance:
    """Create and save a provenance record for the current run.

    Parameters
    ----------
    config : dict
        The full configuration used for this run.
    results_dir : str or Path
        Directory where provenance file will be saved.

    Returns
    -------
    RunProvenance
        The provenance record.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    prov = RunProvenance(
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        config_hash=_config_hash(config),
        config_snapshot=config,
        git_commit=_get_git_commit(),
        python_version=f"{sys.version} ({platform.platform()})",
        package_versions=_get_package_versions(),
        random_seed=config.get("random_seed"),
        subject_ids=config.get("data", {}).get("subjects", []),
    )

    prov_path = results_dir / f"provenance_{prov.config_hash}.json"
    with open(prov_path, "w") as f:
        json.dump(asdict(prov), f, indent=2, default=str)

    return prov


def save_config_snapshot(config: dict[str, Any], results_dir: str | Path) -> Path:
    """Save a YAML snapshot of the config used for this run."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    h = _config_hash(config)
    path = results_dir / f"config_snapshot_{h}.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)
    return path


def save_results_hdf5(
    results: dict[str, Any],
    results_dir: str | Path,
    filename: str = "results.h5",
) -> Path:
    """Save numeric results to HDF5.

    Parameters
    ----------
    results : dict
        Nested dict of results. Leaf values must be numeric arrays or scalars.
    results_dir : str or Path
        Output directory.
    filename : str
        Output filename.

    Returns
    -------
    Path
        Path to the saved HDF5 file.
    """
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
                    arr = np.array(value)
                    group.create_dataset(key_str, data=arr)
                except (ValueError, TypeError):
                    group.attrs[key_str] = json.dumps(value)

    with h5py.File(path, "w") as f:
        _write_group(f, results)

    return path


def save_summary_json(
    summary: dict[str, Any],
    results_dir: str | Path,
    filename: str = "summary.json",
) -> Path:
    """Save a human-readable JSON summary of analysis results."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / filename

    def _make_serializable(obj: Any) -> Any:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_serializable(v) for v in obj]
        if isinstance(obj, set):
            return sorted(list(obj))
        return obj

    with open(path, "w") as f:
        json.dump(_make_serializable(summary), f, indent=2, default=str)

    return path
