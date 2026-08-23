"""Public entry point for the historical geometry brain-state test battery.

The historical runner is retained in ``_geometry_brain_states_legacy.py`` for
computational provenance. That runner populated its feature named ``nd_score``
from ``mean_ep_score``; numerically the field is the legacy overlap/gap proximity
statistic, not the current PC1-based Near-Degeneracy (ND) score.

This entry point delegates the historical analysis while overlaying current
public contracts: semantic feature labels, corrected subject-level uncertainty,
wrapped angular consistency, repository-only amplification provenance, stable
progress labels, serialized execution, and no-overwrite protection for the
locked historical JSON.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Support direct execution from the scripts directory without requiring an
# editable install first, matching the historical runner's behavior.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import _geometry_brain_states_legacy as _legacy  # noqa: E402

from cmcc.analysis.geometry_embedding import (  # noqa: E402
    analyze_geometric_structure,
    classify_states_loso,
    compare_geometry_vs_power,
    extract_propofol_features_semantic,
    extract_sleep_features_semantic,
)

HISTORICAL_ARTIFACT = "geometry_brain_states.json"
CORRECTED_ARTIFACT = "geometry_brain_states_legacy_proximity_corrected_inference.json"
AMPLIFICATION_ARTIFACT = "transient_amplification.json"
LOCK_ARTIFACT = ".geometry_brain_states_public.lock"
_historical_log = _legacy.log


def _public_log(message: str) -> None:
    """Normalize historical progress counters while preserving log destination."""
    normalized = message
    for index in range(1, 6):
        normalized = normalized.replace(f"[{index}/7]", f"[{index}/8]")
    _historical_log(normalized)


def _load_repository_amplification_r() -> float | None:
    """Load amplification evidence only from the checked-in repository artifact."""
    path = (_legacy.RESULTS_JSON / AMPLIFICATION_ARTIFACT).resolve()
    if not path.exists():
        _public_log(f"  Amplification source missing: {path}")
        return None
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        value = data["cross_subject_correlations"]["condition_number_vs_kreiss"]["r"]
    except (KeyError, TypeError):
        _public_log(f"  Amplification correlation missing from {path}")
        return None
    _public_log(f"  Amplification r={float(value):.4f} loaded from {path}")
    return float(value)


def _run_preserving_historical_artifact() -> Path:
    """Run the delegated battery under an exclusive process-safe file lock.

    The lock covers the full delegated run because the historical implementation
    temporarily writes the locked historical result path before this wrapper
    restores it. A stale lock can remain after a hard process termination; that
    condition fails loudly and requires explicit operator removal rather than an
    unsafe automatic takeover.
    """
    historical_path = _legacy.RESULTS_JSON / HISTORICAL_ARTIFACT
    corrected_path = _legacy.RESULTS_JSON / CORRECTED_ARTIFACT
    lock_path = _legacy.RESULTS_JSON / LOCK_ARTIFACT

    lock_fd: int | None = None
    lock_acquired = False
    try:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            raise RuntimeError(
                f"Another public geometry-state rerun appears active: {lock_path}. "
                "If no process is active, remove the stale lock explicitly."
            ) from exc

        lock_acquired = True
        os.write(lock_fd, f"pid={os.getpid()}\n".encode("utf-8"))
        os.close(lock_fd)
        lock_fd = None

        if corrected_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite {corrected_path}. "
                "Archive or remove it explicitly first."
            )

        original_bytes = historical_path.read_bytes() if historical_path.exists() else None
        try:
            _legacy.main()
            if not historical_path.exists():
                raise RuntimeError(
                    "Delegated geometry-state runner completed without producing its expected JSON"
                )
            payload = json.loads(historical_path.read_text(encoding="utf-8"))
            payload["public_rerun_contract"] = {
                "feature_semantics": "legacy_proximity_score",
                "historical_artifact_preserved": HISTORICAL_ARTIFACT,
                "bootstrap": "subject_block_with_replacement_preserving_multiplicity",
                "finite_permutation_p": "(exceedances + 1) / (B + 1)",
                "unscorable_loso_folds": "excluded_from_point_metrics",
                "angular_consistency": "wrapped_to_minus_pi_plus_pi",
                "amplification_source": str(
                    (_legacy.RESULTS_JSON / AMPLIFICATION_ARTIFACT).resolve()
                ),
                "concurrency_guard": "exclusive_lock_file_and_exclusive_output_create",
                "current_nd_validation": False,
            }
            with corrected_path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
                handle.write("\n")
        finally:
            if original_bytes is None:
                if historical_path.exists():
                    historical_path.unlink()
            else:
                historical_path.write_bytes(original_bytes)

        return corrected_path
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_acquired:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def main() -> None:
    """Run the battery with corrected semantics, inference, and provenance."""
    _legacy.log = _public_log
    _legacy.extract_propofol_features = extract_propofol_features_semantic
    _legacy.extract_sleep_features = extract_sleep_features_semantic
    _legacy.classify_states_loso = classify_states_loso
    _legacy.compare_geometry_vs_power = compare_geometry_vs_power
    _legacy.analyze_geometric_structure = analyze_geometric_structure
    _legacy._load_amplification_r = _load_repository_amplification_r
    output_path = _run_preserving_historical_artifact()
    print(f"Corrected public rerun written to {output_path}", flush=True)


if __name__ == "__main__":
    main()
