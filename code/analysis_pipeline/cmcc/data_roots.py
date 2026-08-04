"""Canonical dataset root resolution for the operator-geometry pipeline.

Scientific rationale:
    Reproducibility requires a single documented contract for where local
    datasets live. Reviewers and collaborators should never need a machine-
    specific absolute path baked into source.

Public environment variables (documented in README):
    IEEG_DATA_ROOT       Cogitate iEEG Experiment 1
    DS004752_DATA_ROOT   Zurich SEEG (OpenNeuro ds004752)
    PROPOFOL_DATA_ROOT   Cambridge propofol EEG (OpenNeuro ds005620)
    SLEEP_DATA_ROOT      ANPHY-Sleep polysomnography
    RESULTS_ROOT         Optional override for analysis outputs

Legacy aliases are accepted for one release so older shell scripts keep working,
but new documentation should use the canonical names only.
"""
from __future__ import annotations

import os
from pathlib import Path

# Canonical name -> accepted legacy aliases
_ENV_ALIASES: dict[str, tuple[str, ...]] = {
    "IEEG_DATA_ROOT": ("COGITATE_IEEG_ROOT",),
    "DS004752_DATA_ROOT": ("DS004752_ROOT",),
    "PROPOFOL_DATA_ROOT": ("DS005620_ROOT",),
    "SLEEP_DATA_ROOT": ("ANPHY_SLEEP_ROOT",),
    "RESULTS_ROOT": (),
}


class DataRootError(RuntimeError):
    """Raised when a required dataset root cannot be resolved."""


def resolve_data_root(
    canonical_name: str,
    *,
    default: str | Path | None = None,
    required: bool = False,
) -> Path | None:
    """Resolve a dataset root from the environment.

    Parameters
    ----------
    canonical_name:
        One of the README-documented variables (for example, ``IEEG_DATA_ROOT``).
    default:
        Optional relative fallback used only when the variable is unset.
        Prefer ``None`` for scripts that must not invent a local path.
    required:
        If True, raise ``DataRootError`` when unset and no default is provided.

    Returns
    -------
    pathlib.Path or None
        Resolved path, or None when optional and unset.
    """
    if canonical_name not in _ENV_ALIASES:
        raise ValueError(
            f"Unknown data-root key {canonical_name!r}. "
            f"Expected one of: {', '.join(sorted(_ENV_ALIASES))}."
        )

    candidates = (canonical_name, *_ENV_ALIASES[canonical_name])
    for name in candidates:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser()

    if default is not None:
        return Path(default)

    if required:
        alias_note = ""
        aliases = _ENV_ALIASES[canonical_name]
        if aliases:
            alias_note = f" (legacy aliases also accepted: {', '.join(aliases)})"
        raise DataRootError(
            f"Set {canonical_name} to your local dataset directory{alias_note}."
        )
    return None


def require_data_root(canonical_name: str) -> Path:
    """Resolve a required dataset root or raise a clear error."""
    path = resolve_data_root(canonical_name, required=True)
    assert path is not None
    return path
