"""Configuration loading, validation, and defaults for CMCC pipeline."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "data": {
        "root": "./data/Cogitate_IEEG_EXP1",
        "subjects": ["CE103"],
        "runs": ["DurR1", "DurR2", "DurR3", "DurR4", "DurR5"],
    },
    "preprocessing": {
        "line_freq": 60.0,
        "high_gamma_passband": [70, 150],
        "epoch_tmin": -0.5,
        "epoch_tmax": 2.0,
        "baseline": [-0.5, 0.0],
    },
    "avalanche": {
        "threshold_sd": 3.0,
        "bin_width_factor": 1.0,
        "sensitivity": {
            "threshold_sd": [2.0, 2.5, 3.0, 3.5, 4.0],
            "bin_width_factor": [0.5, 1.0, 2.0, 4.0],
        },
    },
    "powerlaw": {
        "discrete": True,
        "n_bootstrap": 2500,
        "xmin_method": "clauset",
        "compare_distributions": [
            "exponential",
            "lognormal",
            "truncated_power_law",
        ],
    },
    "complexity": {
        "lzc_n_surrogates": 100,
        "mse_scales": [1, 20],
        "mse_m": 2,
        "mse_r_factor": 0.15,
        "dfa_scales": None,
    },
    "statistics": {
        "n_perm": 5000,
        "alpha": 0.05,
        "correction": "fdr_bh",
        "effect_size": "hedges_g",
    },
    "decoding": {
        "classifier": "lda",
        "cv_strategy": "leave_one_block_out",
        "n_channels_list": [5, 10, 20],
    },
    "random_seed": 42,
    "output": {
        "results_dir": "./results",
        "save_intermediates": True,
        "format": "hdf5",
    },
}

_REQUIRED_SECTIONS = [
    "data",
    "preprocessing",
    "avalanche",
    "powerlaw",
    "complexity",
    "statistics",
    "random_seed",
    "output",
]


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_config(config: dict[str, Any]) -> list[str]:
    """Return list of validation error messages. Empty list means valid."""
    errors: list[str] = []

    for section in _REQUIRED_SECTIONS:
        if section not in config:
            errors.append(f"Missing required section: '{section}'")

    data = config.get("data", {})
    if not data.get("root"):
        errors.append("data.root must be a non-empty path")
    elif not Path(data["root"]).exists():
        errors.append(f"data.root path does not exist: {data['root']}")

    if not data.get("subjects"):
        errors.append("data.subjects must be a non-empty list")

    if not data.get("runs"):
        errors.append("data.runs must be a non-empty list")

    pp = config.get("preprocessing", {})
    passband = pp.get("high_gamma_passband", [])
    if len(passband) != 2 or passband[0] >= passband[1]:
        errors.append(
            "preprocessing.high_gamma_passband must be [low, high] with low < high"
        )

    av = config.get("avalanche", {})
    if av.get("threshold_sd", 0) <= 0:
        errors.append("avalanche.threshold_sd must be positive")

    seed = config.get("random_seed")
    if seed is not None and not isinstance(seed, int):
        errors.append("random_seed must be an integer or null")

    return errors


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load config from YAML, merge with defaults, and validate.

    Parameters
    ----------
    path : str or Path or None
        Path to YAML config file. If None, returns defaults only.

    Returns
    -------
    dict
        Validated configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If validation fails.
    """
    if path is None:
        config = copy.deepcopy(DEFAULTS)
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(DEFAULTS, user_config)

    errors = validate_config(config)
    if errors:
        raise ValueError(
            "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return config
