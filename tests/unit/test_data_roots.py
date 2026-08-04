"""Unit tests for canonical dataset-root resolution."""
from __future__ import annotations

from pathlib import Path

import pytest

from cmcc.data_roots import DataRootError, require_data_root, resolve_data_root


def test_resolve_canonical_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IEEG_DATA_ROOT", "/tmp/ieeg")
    assert resolve_data_root("IEEG_DATA_ROOT") == Path("/tmp/ieeg")


def test_resolve_legacy_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROPOFOL_DATA_ROOT", raising=False)
    monkeypatch.setenv("DS005620_ROOT", "/tmp/propofol")
    assert resolve_data_root("PROPOFOL_DATA_ROOT") == Path("/tmp/propofol")


def test_require_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLEEP_DATA_ROOT", raising=False)
    monkeypatch.delenv("ANPHY_SLEEP_ROOT", raising=False)
    with pytest.raises(DataRootError, match="SLEEP_DATA_ROOT"):
        require_data_root("SLEEP_DATA_ROOT")


def test_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="Unknown data-root key"):
        resolve_data_root("NOT_A_REAL_ROOT")
