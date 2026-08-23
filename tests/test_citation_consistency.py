"""Guard against citation drift across public-facing files."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_YEAR = "2026"
CANONICAL_TITLE = (
    "Fitted-Operator Geometry as a Complementary Descriptive Axis for "
    "Brain-State Discrimination in Human iEEG and Scalp EEG"
)


def _citation_metadata() -> dict:
    return yaml.safe_load((REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8"))


def test_readme_title_matches_citation_metadata() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    first_line = readme.splitlines()[0]
    assert first_line == f"# {CANONICAL_TITLE}"
    assert _citation_metadata()["title"] == CANONICAL_TITLE


def test_readme_points_to_canonical_citation_file() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Citation metadata are provided in `CITATION.cff`" in readme


def test_citation_cff_release_year_is_canonical() -> None:
    metadata = _citation_metadata()
    release_date = str(metadata["date-released"])
    assert release_date.startswith(f"{CANONICAL_YEAR}-")


def test_no_stale_peterkin_2025_citation_in_readme() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"Peterkin(?:2025|\s*\(2025\))", readme)
