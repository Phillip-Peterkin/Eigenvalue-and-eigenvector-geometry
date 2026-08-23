"""Canonical broadband runner for the public reproduction path.

This wrapper exists because the historical broadband runner reused the
`high_gamma_passband` configuration key even though its scientific intent was
to analyze broadband data. That behavior made a clean checkout ambiguous.

The canonical runner requires an explicit `preprocessing.broadband_passband`
entry, copies that value into the historical runner's expected passband slot in
memory, and then executes the same subject-level analysis code. The original
historical runner is retained for provenance.
"""
from __future__ import annotations

import gc
import os
import time
import traceback
from pathlib import Path

import numpy as np

from cmcc.config import load_config
from cmcc.provenance import save_summary_json

from run_all_subjects_broadband import run_single_subject

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def _broadband_config(config: dict) -> dict:
    """Return a copy configured explicitly for the broadband analysis.

    The historical subject-level implementation reads the key
    `high_gamma_passband`. To preserve the historical analysis code while
    removing public ambiguity, this wrapper replaces that in-memory value with
    the explicitly declared broadband passband before any subject is processed.
    """
    copied = dict(config)
    copied["preprocessing"] = dict(config["preprocessing"])
    try:
        broadband = list(copied["preprocessing"]["broadband_passband"])
    except KeyError as exc:
        raise KeyError(
            "Canonical broadband reproduction requires "
            "preprocessing.broadband_passband in code/config.yaml"
        ) from exc
    if len(broadband) != 2 or broadband[0] >= broadband[1]:
        raise ValueError(
            "preprocessing.broadband_passband must be [low_hz, high_hz] with low_hz < high_hz"
        )
    copied["preprocessing"]["high_gamma_passband"] = broadband
    return copied


def main() -> None:
    start = time.time()
    config = _broadband_config(load_config(str(CONFIG_PATH)))
    data_root = Path(config["data"]["root"])
    results_dir = Path(config["output"]["results_dir"]) / "broadband_canonical"
    results_dir.mkdir(parents=True, exist_ok=True)

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
        try:
            summary = run_single_subject(subject_id, config, results_dir)
        except Exception as exc:  # preserve a complete batch audit trail
            traceback.print_exc()
            summary = {"subject": subject_id, "status": "FAILED", "error": str(exc)}
        summaries.append(summary)
        gc.collect()

    save_summary_json(summaries, results_dir, "group_all_subjects_broadband_canonical.json")

    successful = [item for item in summaries if item.get("status") == "OK"]
    if successful:
        sigma = np.asarray([item["branching_sigma"] for item in successful], dtype=float)
        print(
            f"Broadband canonical complete: n={len(successful)}, "
            f"mean branching statistic={np.mean(sigma):.6f}",
            flush=True,
        )
    print(f"Elapsed seconds: {time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
