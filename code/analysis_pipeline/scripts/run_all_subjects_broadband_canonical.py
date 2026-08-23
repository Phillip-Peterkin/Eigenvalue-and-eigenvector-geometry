"""Canonical broadband runner for the public reproduction path.

This wrapper exists because the historical broadband runner reused the
`high_gamma_passband` configuration key even though its scientific intent was
to analyze broadband data. That behavior made a clean checkout ambiguous.

The canonical runner requires an explicit `preprocessing.broadband_passband`
entry, copies that value into the historical runner's expected passband slot in
memory, records the effective passband after per-recording Nyquist adjustment,
and then executes the same subject-level analysis code. The original historical
runner is retained for provenance.
"""
from __future__ import annotations

import gc
import os
import time
import traceback
from pathlib import Path

import numpy as np

from cmcc.config import load_config
from cmcc.data_roots import require_data_root, resolve_data_root
from cmcc.io.loader import load_edf
from cmcc.provenance import save_summary_json

from run_all_subjects_broadband import RUNS, run_single_subject

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
ANALYSIS_TYPE = "canonical_broadband"


def _broadband_config(config: dict) -> dict:
    """Return a copy configured explicitly for broadband analysis.

    The passband is expressed in Hertz. Bounds must be finite, non-negative,
    and strictly increasing. Per-recording Nyquist adjustment is validated
    separately by :func:`_effective_passband`.
    """
    copied = dict(config)
    copied["preprocessing"] = dict(config["preprocessing"])
    try:
        broadband = [float(value) for value in copied["preprocessing"]["broadband_passband"]]
    except KeyError as exc:
        raise KeyError(
            "Canonical broadband reproduction requires "
            "preprocessing.broadband_passband in code/config.yaml"
        ) from exc

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
    copied["preprocessing"]["high_gamma_passband"] = broadband.copy()
    return copied


def _effective_passband(configured: list[float], sampling_frequency_hz: float) -> list[float]:
    """Return the exact passband used after the historical Nyquist adjustment.

    Parameters are in Hertz. Invalid or non-finite sampling frequencies and
    any effective interval with ``low_hz >= high_hz`` raise ``ValueError`` so a
    malformed recording cannot silently become a successful reproduction.
    """
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
) -> dict[str, list[float]]:
    """Read run headers and record the effective broadband band for each run."""
    subject_dir = data_root / f"{subject_id}_ECOG_1"
    effective: dict[str, list[float]] = {}
    for run_id in RUNS:
        try:
            raw = load_edf(subject_dir, subject_id, run_id)
        except FileNotFoundError:
            continue
        try:
            effective[run_id] = _effective_passband(configured, float(raw.info["sfreq"]))
        finally:
            raw.close()

    if not effective:
        raise FileNotFoundError(f"No readable EDF runs found for {subject_id}")
    return effective


def main() -> None:
    """Run the canonical COGITATE broadband reproduction.

    The configured and effective passbands are recorded in Hertz for every
    subject. The dataset root must be supplied through ``IEEG_DATA_ROOT``.
    Invalid passbands, unreadable subjects, or a batch with zero successful
    subjects cause a nonzero failure after an audit summary is written.
    """
    start = time.time()
    config = _broadband_config(load_config(str(CONFIG_PATH)))

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
    subjects = sorted(
        {
            directory.split("_")[0]
            for directory in os.listdir(data_root)
            if "ECOG" in directory and (data_root / directory).is_dir()
        }
    )

    summaries = []
    for index, subject_id in enumerate(subjects, start=1):
        print(f"[{index}/{len(subjects)}] broadband: {subject_id}", flush=True)
        effective_passbands: dict[str, list[float]] = {}
        try:
            effective_passbands = _subject_effective_passbands(
                data_root,
                subject_id,
                configured_passband,
            )
            summary = run_single_subject(subject_id, config, results_dir)
        except Exception as exc:  # preserve a complete batch audit trail
            traceback.print_exc()
            summary = {"subject": subject_id, "status": "FAILED", "error": str(exc)}

        summary = dict(summary)
        summary["analysis_type"] = ANALYSIS_TYPE
        summary["configured_broadband_passband_hz"] = configured_passband
        summary["effective_broadband_passband_by_run_hz"] = effective_passbands
        summary["data_root_contract"] = "IEEG_DATA_ROOT"
        summaries.append(summary)

        save_summary_json(summary, results_dir, f"summary_{subject_id}_pooled.json")
        gc.collect()

    save_summary_json(summaries, results_dir, "group_all_subjects_broadband_canonical.json")

    successful = [item for item in summaries if item.get("status") == "OK"]
    if not successful:
        print(f"Elapsed seconds: {time.time() - start:.1f}", flush=True)
        raise RuntimeError("Canonical broadband analysis produced no successful subject summaries")

    sigma = np.asarray([item["branching_sigma"] for item in successful], dtype=float)
    print(
        f"Broadband canonical complete: n={len(successful)}, "
        f"mean branching statistic={np.mean(sigma):.6f}",
        flush=True,
    )
    print(f"Elapsed seconds: {time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
