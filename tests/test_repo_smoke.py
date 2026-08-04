"""Repository hygiene smoke checks for interview / reviewer first glance."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_license_exists() -> None:
    text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in text
    assert "Phillip Peterkin" in text


def test_ci_workflow_exists() -> None:
    workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow.is_file()
    text = workflow.read_text(encoding="utf-8")
    assert "pytest" in text
    assert '".[dev]"' in text or ".[dev]" in text


def test_pyproject_defines_package() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "operator-geometry"' in text
    assert "cmcc" in text


def test_cmcc_package_imports() -> None:
    import cmcc

    assert cmcc.__version__
