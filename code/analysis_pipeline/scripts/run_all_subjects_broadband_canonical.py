"""Canonical broadband runner for the public reproduction path.

Canonical broadband reproduction requires an explicit broadband passband and a
versioned cohort. Canonical mode is intentionally strict: the scientific cohort
comes from the versioned configuration/manifest, never from whatever directories
happen to be present on a reviewer's machine. A best-effort mode exists for
exploratory batch processing, but it is opt-in and is not a valid release-
reproduction path.
"""
from __future__ import annotations

import argparse
import gc
import time
import traceback
from pathlib import Path

import numpy as np
from run_all_subjects_broadband import RUNS, run_single_subject

from cmcc.config import load_config
from cmcc.data_roots import require_data_root, resolve_data_root
from cmcc.io.loader import load_edf
from cmcc.provenance import save_summary_json

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
ANALYSIS_TYPE = "canonical_broadband"


def _broadband_config(config: dict) -> dict:
    """Return a copy configured explicitly for broadband analysis."""
    copied = dict(config)
    copied["preprocessing"] = dict(config["preprocessing"])
    broadband = [float(value) for value in copied["preprocessing"]["broadband_passband"]]
    if (
        len(broadband) != 2
        or not np.all(np.isfinite(broadband))
        or broadband[0] < 0.0
        or broadband[0] >= broadband[1]
    ):
        raise ValueError(
            "preprocessing.broadband_passband must contain finite "
            "[low_hz, high_hz] bounds with 0 <= low_hz < high_hz"
        )

    copied["preprocessing"]["broadband_passband"] = broadband
    # Historical implementation accepts one generic passband slot. This adapter
    # is the only location where the compatibility translation is permitted.
    copied["preprocessing"]["high_gamma_passband"] = broadband.copy()
    return copied


def _effective_passband(configured: list[float], sampling_frequency_hz: float) -> list[float]:
    """Return the exact passband used after the historical Nyquist adjustment."""
    if not np.isfinite(sampling_frequency_hz) or sampling_frequency_hz <= 0.0:
        raise ValueError("sampling frequency must be finite and positive")

    low_hz, high_hz = (float(configured[0]), float(configured[1]))
    if not np.isfinite(low_hz) or not np.isfinite(high_hz):
        raise ValueError("configured passband bounds must be finite")

    nyquist_hz = sampling_frequency_hz / 2.0
    if high_hz >= nyquist_hz:
        high_hz = float(int(nyquist_hz - 1.0))
    if low_hz >= high_hz:
        low_hz = max(1.0, high_hz - 10.0)

    if not np.isfinite(low_hz) or not np.isfinite(high_hz) or low_hz < 0.0 or low_hz >= high_hz:
        raise ValueError(
            "effective broadband passband is invalid after Nyquist adjustment: "
            f"[{low_hz}, {high_hz}] Hz at sampling frequency {sampling_frequency_hz} Hz"
        )
    return [low_hz, high_hz]


def _subject_effective_passbands(
    data_root: Path,
    subject_id: str,
    configured: list[float],
    *,
    require_all_runs: bool,
) -> dict[str, list[float]]:
    """Read run headers and record the effective passband for each expected run."""
    subject_dir = data_root / f"{subject_id}_ECOG_1"
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"Missing expected subject directory: {subject_dir}")

    effective: dict[str, list[float]] = {}
    missing_runs: list[str] = []
    for run_id in RUNS:
        try:
            raw = load_edf(subject_dir, subject_id, run_id)
        except FileNotFoundError:
            missing_runs.append(run_id)
            continue
        try:
            effective[run_id] = _effective_passband(configured, float(raw.info["sfreq"]))
        finally:
            raw.close()

    if not effective:
        raise FileNotFoundError(f"No readable EDF runs found for {subject_id}")
    if require_all_runs and missing_runs:
        raise FileNotFoundError(
            f"Strict reproduction requires all expected runs for {subject_id}; "
            f"missing: {', '.join(missing_runs)}"
        )
    return effective


def _configured_subjects(config: dict) -> list[str]:
    subjects = list(config["data"]["subjects"])
    if not subjects:
        raise ValueError("Canonical reproduction requires a non-empty configured subject cohort")
    if len(subjects) != len(set(subjects)):
        raise ValueError("Canonical subject cohort contains duplicate IDs")
    return subjects


def _discover_subjects(data_root: Path) -> list[str]:
    """Filesystem discovery for explicit best-effort exploratory mode only."""
    return sorted(
        {
            path.name.split("_")[0]
            for path in data_root.iterdir()
            if path.is_dir() and "ECOG" in path.name
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best-effort",
        action="store_true",
        help=(
            "Explore all locally discoverable subjects and tolerate partial failures. "
            "This mode is not valid for canonical release reproduction."
        ),
    )
    return parser.parse_args()


def main(*, best_effort: bool | None = None) -> None:
    """Run strict canonical reproduction unless best-effort is explicitly requested."""
    if best_effort is None:
        best_effort = bool(_parse_args().best_effort)

    start = time.time()
    config = _broadband_config(load_config(str(CONFIG_PATH)))
    strict = bool(config["data"].get("strict_reproduction", True)) and not best_effort

    data_root = require_data_root("IEEG_DATA_ROOT")
    config["data"] = dict(config["data"])
    config["data"]["root"] = str(data_root)

    results_root = resolve_data_root("RESULTS_ROOT")
    if results_root is None:
        results_dir = Path(config["output"]["results_dir"]) / "broadband_canonical"
    else:
        results_dir = results_root / "broadband_canonical"
    results_dir.mkdir(parents=True, exist_ok=True)

    configured_passband = list(config["preprocessing"]["broadband_passband"])
    subjects = _configured_subjects(config) if strict else _discover_subjects(data_root)
    if not subjects:
        raise RuntimeError("No subjects selected for broadband analysis")

    summaries: list[dict] = []
    for index, subject_id in enumerate(subjects, start=1):
        print(f"[{index}/{len(subjects)}] broadband: {subject_id}", flush=True)
        effective_passbands: dict[str, list[float]] = {}
        try:
            effective_passbands = _subject_effective_passbands(
                data_root,
                subject_id,
                configured_passband,
                require_all_runs=strict,
            )
            summary = run_single_subject(subject_id, config, results_dir)
            if strict and summary.get("status") != "OK":
                raise RuntimeError(
                    f"Strict reproduction subject {subject_id} returned status "
                    f"{summary.get('status')!r}: {summary.get('error', 'no error detail')}"
                )
        except Exception as exc:
            traceback.print_exc()
            summary = {"subject": subject_id, "status": "FAILED", "error": str(exc)}

        summary = dict(summary)
        summary["analysis_type"] = ANALYSIS_TYPE
        summary["execution_mode"] = "strict_reproduction" if strict else "best_effort"
        summary["configured_broadband_passband_hz"] = configured_passband
        summary["effective_broadband_passband_by_run_hz"] = effective_passbands
        summary["data_root_contract"] = "IEEG_DATA_ROOT"
        summary["configured_cohort_manifest"] = config["data"].get("cohort_manifest")
        summaries.append(summary)

        save_summary_json(summary, results_dir, f"summary_{subject_id}_pooled.json")
        gc.collect()

    save_summary_json(summaries, results_dir, "group_all_subjects_broadband_canonical.json")

    successful = [item for item in summaries if item.get("status") == "OK"]
    failed = [item for item in summaries if item.get("status") != "OK"]
    if strict and failed:
        failures = "; ".join(f"{item['subject']}: {item.get('error', item.get('status'))}" for item in failed)
        raise RuntimeError(f"Strict canonical reproduction incomplete: {failures}")
    if strict and len(successful) != len(subjects):
        raise RuntimeError(
            f"Strict canonical reproduction expected {len(subjects)} successful subjects, "
            f"got {len(successful)}"
        )
    if not successful:
        raise RuntimeError("Broadband analysis produced no successful subject summaries")

    sigma = np.asarray([item["branching_sigma"] for item in successful], dtype=float)
    print(
        f"Broadband complete ({'strict' if strict else 'best-effort'}): "
        f"n={len(successful)}, mean branching statistic={np.mean(sigma):.6f}",
        flush=True,
    )
    print(f"Elapsed seconds: {time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
