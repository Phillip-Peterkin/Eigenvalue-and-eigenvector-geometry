"""Spatial ROI definition and anterior/posterior analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ROIDefinition:
    """Region of interest definition.

    Attributes
    ----------
    posterior_channels : list[str]
        Channels in the posterior ROI.
    anterior_channels : list[str]
        Channels in the anterior ROI.
    unclassified_channels : list[str]
        Channels that could not be classified.
    y_threshold : float
        Y-coordinate threshold used for classification.
    """

    posterior_channels: list[str]
    anterior_channels: list[str]
    unclassified_channels: list[str]
    y_threshold: float


def define_posterior_roi(
    electrodes: pd.DataFrame,
    y_threshold: float = 0.0,
) -> ROIDefinition:
    """Define posterior and anterior ROIs from electrode coordinates.

    In MNI/fsaverage space, y < 0 is posterior to the central sulcus.

    Parameters
    ----------
    electrodes : pd.DataFrame
        Electrode table with columns: name, x, y, z.
    y_threshold : float
        Y-coordinate threshold. Channels with y < threshold are posterior.

    Returns
    -------
    ROIDefinition
    """
    posterior = []
    anterior = []
    unclassified = []

    for _, row in electrodes.iterrows():
        name = str(row["name"])
        try:
            y_val = float(row["y"])
        except (ValueError, TypeError):
            unclassified.append(name)
            continue

        if np.isnan(y_val):
            unclassified.append(name)
        elif y_val < y_threshold:
            posterior.append(name)
        else:
            anterior.append(name)

    return ROIDefinition(
        posterior_channels=posterior,
        anterior_channels=anterior,
        unclassified_channels=unclassified,
        y_threshold=y_threshold,
    )
