"""Shared pytest fixtures for repository-level tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def preregistration(repo_root: Path) -> dict:
    path = repo_root / "preregistration_spec.json"
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
