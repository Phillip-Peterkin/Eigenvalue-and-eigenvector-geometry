"""Pretty-print a group-level subject disposition JSON.

Usage:
    set RESULTS_ROOT=./results
    python -m analysis path...   # or:
    python _group_summary.py path/to/group_summaries.json

The input JSON must be a list of per-subject summary dicts with a ``status`` field.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        path = Path(argv[0])
    else:
        results_root = Path(os.environ.get("RESULTS_ROOT", "./results"))
        candidates = [
            results_root / "group_summaries.json",
            results_root / "json_results" / "group_summaries.json",
            results_root / "all_subjects.json",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise SystemExit(
                "Provide a group summary JSON path, or place group_summaries.json "
                "under RESULTS_ROOT."
            )

    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON list of subject summaries in {path}")

    all_subjects = data
    ok = [s for s in all_subjects if s.get("status") == "OK"]
    skip = [s for s in all_subjects if s.get("status") == "SKIP"]
    fail = [s for s in all_subjects if s.get("status") == "FAILED"]

    print("=" * 78)
    print("CMCC PIPELINE — COMPLETE GROUP RESULTS")
    print("=" * 78)
    print(f"\nSource: {path}")
    print(f"\n## SUBJECT DISPOSITION (N={len(all_subjects)})")
    print(f"  OK:      {len(ok)}")
    print(f"  SKIP:    {len(skip)}")
    for subject in skip:
        print(f"           {subject.get('subject')}: {subject.get('error', '')}")
    print(f"  FAILED:  {len(fail)}")
    for subject in fail:
        print(f"           {subject.get('subject')}: {str(subject.get('error', ''))[:60]}")


if __name__ == "__main__":
    main()
