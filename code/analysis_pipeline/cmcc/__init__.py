"""Fitted-operator geometry analysis package for human iEEG and scalp EEG."""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("operator-geometry")
except metadata.PackageNotFoundError:  # source-tree import before installation
    __version__ = "0+uninstalled"
