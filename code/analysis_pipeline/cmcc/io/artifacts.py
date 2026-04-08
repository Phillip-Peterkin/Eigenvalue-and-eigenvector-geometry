"""Save and load intermediate results as HDF5 artifacts with metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


def save_array(
    path: str | Path,
    key: str,
    data: np.ndarray,
    attrs: dict[str, Any] | None = None,
) -> None:
    """Save a numpy array to an HDF5 file with optional metadata attributes.

    Parameters
    ----------
    path : str or Path
        HDF5 file path.
    key : str
        Dataset key within the file.
    data : np.ndarray
        Array to save.
    attrs : dict or None
        Metadata attributes to attach to the dataset.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "a") as f:
        if key in f:
            del f[key]
        ds = f.create_dataset(key, data=data)
        if attrs:
            for k, v in attrs.items():
                if isinstance(v, (dict, list)):
                    ds.attrs[k] = json.dumps(v)
                elif v is None:
                    ds.attrs[k] = "null"
                else:
                    ds.attrs[k] = v


def load_array(path: str | Path, key: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a numpy array and its attributes from an HDF5 file.

    Parameters
    ----------
    path : str or Path
        HDF5 file path.
    key : str
        Dataset key within the file.

    Returns
    -------
    tuple[np.ndarray, dict]
        The array and its metadata attributes.
    """
    with h5py.File(path, "r") as f:
        data = f[key][:]
        attrs = dict(f[key].attrs)
    for k, v in attrs.items():
        if isinstance(v, str):
            if v == "null":
                attrs[k] = None
            else:
                try:
                    attrs[k] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    pass
    return data, attrs


def save_dataframe(path: str | Path, key: str, df: pd.DataFrame) -> None:
    """Save a pandas DataFrame to HDF5.

    Parameters
    ----------
    path : str or Path
        HDF5 file path.
    key : str
        Group key within the file.
    df : pd.DataFrame
        DataFrame to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(str(path), key=key, mode="a")


def load_dataframe(path: str | Path, key: str) -> pd.DataFrame:
    """Load a pandas DataFrame from HDF5.

    Parameters
    ----------
    path : str or Path
        HDF5 file path.
    key : str
        Group key within the file.

    Returns
    -------
    pd.DataFrame
    """
    return pd.read_hdf(str(path), key=key)
