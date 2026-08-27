"""Configuration loading, validation, and canonical defaults for the pipeline.

`code/config.yaml` is the human-readable canonical configuration. The defaults
below mirror that file so the installed package remains usable without a source
checkout. Regression tests require the two representations to stay synchronized.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

CANONICAL_COGITATE_SUBJECTS = [
    "CE103",
    "CE110",
    "CF102",
    "CF103",
    "CF104",
    "CF105",
    "CF106",
    "CF109",
    "CF110",
    "CF113",
    "CF119",
    "CF121",
    "CF122",
    "CF124",
    "CF125",
    "CF126",
    "CG103",
    "CG104",
]

DEFAULTS: dict[str, Any] = {
    "data": {
        "root": "./data/Cogitate_IEEG_EXP1",
        "cohort_manifest": "cohorts/cogitate_primary.json",
        "subjects": CANONICAL_COGITATE_SUBJECTS.copy(),
        "runs": ["DurR1", "DurR2", "DurR3", "DurR4", "DurR5"],
        "strict_reproduction": True,
    },
    "preprocessing": {
        "line_freq": 60.0,
        "high_gamma_passband": [70, 150],
        "broadband_passband": [1, 200],
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
        "n_random": 10,
        "target_tau": 1.5,
        "n_perm_comparison": 1000,
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
    "decoding",
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


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_passband(name: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        errors.append(f"{name} must contain exactly two numeric bounds")
        return
    low, high = value
    if not _is_finite_number(low) or not _is_finite_number(high):
        errors.append(f"{name} bounds must be finite numbers")
        return
    if low < 0 or low >= high:
        errors.append(f"{name} must satisfy 0 <= low < high")


def validate_config(config: dict[str, Any]) -> list[str]:
    """Return validation errors. An empty list means the config is executable."""
    errors: list[str] = []

    if not isinstance(config, dict):
        return ["configuration root must be a mapping"]

    for section in _REQUIRED_SECTIONS:
        if section not in config:
            errors.append(f"Missing required section: '{section}'")

    data = config.get("data", {})
    if not isinstance(data, dict):
        errors.append("data must be a mapping")
        data = {}
    if not isinstance(data.get("root"), str) or not data.get("root", "").strip():
        errors.append("data.root must be a non-empty path string")
    manifest = data.get("cohort_manifest")
    if not isinstance(manifest, str) or not manifest.strip():
        errors.append("data.cohort_manifest must be a non-empty path string")
    subjects = data.get("subjects")
    if not isinstance(subjects, list) or not subjects or not all(isinstance(x, str) and x for x in subjects):
        errors.append("data.subjects must be a non-empty list of subject IDs")
    elif len(subjects) != len(set(subjects)):
        errors.append("data.subjects must not contain duplicates")
    runs = data.get("runs")
    if not isinstance(runs, list) or not runs or not all(isinstance(x, str) and x for x in runs):
        errors.append("data.runs must be a non-empty list of run IDs")
    elif len(runs) != len(set(runs)):
        errors.append("data.runs must not contain duplicates")
    if not isinstance(data.get("strict_reproduction"), bool):
        errors.append("data.strict_reproduction must be boolean")

    pp = config.get("preprocessing", {})
    if not isinstance(pp, dict):
        errors.append("preprocessing must be a mapping")
        pp = {}
    _validate_passband("preprocessing.high_gamma_passband", pp.get("high_gamma_passband"), errors)
    _validate_passband("preprocessing.broadband_passband", pp.get("broadband_passband"), errors)
    if not _is_finite_number(pp.get("line_freq")) or pp.get("line_freq", 0) <= 0:
        errors.append("preprocessing.line_freq must be finite and positive")
    if not _is_finite_number(pp.get("epoch_tmin")) or not _is_finite_number(pp.get("epoch_tmax")):
        errors.append("preprocessing epoch bounds must be finite")
    elif pp["epoch_tmin"] >= pp["epoch_tmax"]:
        errors.append("preprocessing.epoch_tmin must be less than epoch_tmax")
    baseline = pp.get("baseline")
    if baseline is not None:
        if not isinstance(baseline, (list, tuple)) or len(baseline) != 2:
            errors.append("preprocessing.baseline must be null or [start, end]")
        elif not all(_is_finite_number(x) for x in baseline) or baseline[0] > baseline[1]:
            errors.append("preprocessing.baseline must contain finite ordered bounds")

    av = config.get("avalanche", {})
    if not isinstance(av, dict):
        errors.append("avalanche must be a mapping")
        av = {}
    if not _is_finite_number(av.get("threshold_sd")) or av.get("threshold_sd", 0) <= 0:
        errors.append("avalanche.threshold_sd must be finite and positive")
    if not _is_finite_number(av.get("bin_width_factor")) or av.get("bin_width_factor", 0) <= 0:
        errors.append("avalanche.bin_width_factor must be finite and positive")

    powerlaw = config.get("powerlaw", {})
    if not isinstance(powerlaw, dict):
        errors.append("powerlaw must be a mapping")
        powerlaw = {}
    if not isinstance(powerlaw.get("n_bootstrap"), int) or powerlaw.get("n_bootstrap", 0) <= 0:
        errors.append("powerlaw.n_bootstrap must be a positive integer")

    complexity = config.get("complexity", {})
    if not isinstance(complexity, dict):
        errors.append("complexity must be a mapping")
        complexity = {}
    if not isinstance(complexity.get("lzc_n_surrogates"), int) or complexity.get("lzc_n_surrogates", -1) < 0:
        errors.append("complexity.lzc_n_surrogates must be a non-negative integer")

    statistics = config.get("statistics", {})
    if not isinstance(statistics, dict):
        errors.append("statistics must be a mapping")
        statistics = {}
    if not isinstance(statistics.get("n_perm"), int) or statistics.get("n_perm", 0) <= 0:
        errors.append("statistics.n_perm must be a positive integer")
    alpha = statistics.get("alpha")
    if not _is_finite_number(alpha) or not 0 < alpha < 1:
        errors.append("statistics.alpha must satisfy 0 < alpha < 1")

    decoding = config.get("decoding", {})
    if not isinstance(decoding, dict):
        errors.append("decoding must be a mapping")
        decoding = {}
    channels = decoding.get("n_channels_list")
    if not isinstance(channels, list) or not channels or not all(isinstance(x, int) and x > 0 for x in channels):
        errors.append("decoding.n_channels_list must be a non-empty list of positive integers")
    for key in ("n_random", "n_perm_comparison"):
        if not isinstance(decoding.get(key), int) or decoding.get(key, 0) <= 0:
            errors.append(f"decoding.{key} must be a positive integer")
    if not _is_finite_number(decoding.get("target_tau")):
        errors.append("decoding.target_tau must be finite")

    seed = config.get("random_seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        errors.append("random_seed must be an integer or null")

    output = config.get("output", {})
    if not isinstance(output, dict):
        errors.append("output must be a mapping")
        output = {}
    if not isinstance(output.get("results_dir"), str) or not output.get("results_dir", "").strip():
        errors.append("output.results_dir must be a non-empty path string")
    if not isinstance(output.get("save_intermediates"), bool):
        errors.append("output.save_intermediates must be boolean")
    if output.get("format") not in {"hdf5", "json", "csv"}:
        errors.append("output.format must be one of: hdf5, json, csv")

    return errors


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML override, merge with canonical defaults, and validate it."""
    if path is None:
        config = copy.deepcopy(DEFAULTS)
    else:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            user_config = yaml.safe_load(handle) or {}
        if not isinstance(user_config, dict):
            raise ValueError("Config validation failed:\n  - YAML root must be a mapping")
        config = _deep_merge(DEFAULTS, user_config)

    errors = validate_config(config)
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    return config
