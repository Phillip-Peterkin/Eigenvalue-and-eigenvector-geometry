"""Spatial electrode metric maps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_electrode_metric_map(
    electrodes: pd.DataFrame,
    metric_values: dict[str, float],
    metric_name: str = "Metric",
    title: str | None = None,
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Plot metric values at electrode locations (2D x-y projection).

    Parameters
    ----------
    electrodes : pd.DataFrame
        Electrode table with columns: name, x, y, z.
    metric_values : dict[str, float]
        Channel name -> metric value.
    metric_name : str
        Colorbar label.
    title : str or None
        Plot title.
    cmap : str
        Colormap name.
    ax : Axes or None
        Pre-existing axes.
    vmin : float or None
        Minimum colorbar value. If None, auto-scaled from data.
    vmax : float or None
        Maximum colorbar value. If None, auto-scaled from data.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    x_coords = []
    y_coords = []
    values = []
    labels = []

    for _, row in electrodes.iterrows():
        name = str(row["name"])
        if name in metric_values and not np.isnan(metric_values[name]):
            x_coords.append(float(row["x"]))
            y_coords.append(float(row["y"]))
            values.append(metric_values[name])
            labels.append(name)

    if not values:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        return fig

    sc = ax.scatter(x_coords, y_coords, c=values, cmap=cmap,
                    s=60, edgecolors="black", linewidths=0.5,
                    vmin=vmin, vmax=vmax)
    fig.colorbar(sc, ax=ax, label=metric_name)
    ax.set_xlabel("x (fsaverage)")
    ax.set_ylabel("y (fsaverage)")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
