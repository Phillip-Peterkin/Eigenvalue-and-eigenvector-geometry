"""Guard against citation year drift across public-facing files."""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_YEAR = "2026"


def test_readme_intro_year_is_canonical() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"Peterkin\s*\((\d{4})\)", readme)
    assert match is not None
    assert match.group(1) == CANONICAL_YEAR


def test_readme_bibtex_year_is_canonical() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    year = re.search(r"year\s*=\s*\{(\d{4})\}", readme)
    key = re.search(r"@article\{Peterkin(\d{4})", readme)
    assert year is not None and key is not None
    assert year.group(1) == CANONICAL_YEAR
    assert key.group(1) == CANONICAL_YEAR


def test_citation_cff_release_year_is_canonical() -> None:
    text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'date-released:\s*"(\d{4})-', text)
    assert match is not None
    assert match.group(1) == CANONICAL_YEAR


def test_no_stale_peterkin_2025_citation_in_readme() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"Peterkin(?:2025|\s*\(2025\))", readme)
