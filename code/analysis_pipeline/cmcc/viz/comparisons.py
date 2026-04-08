"""Condition comparison violin/swarm plots for criticality metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_condition_comparison(
    data: dict[str, np.ndarray],
    metric_name: str = "Metric",
    title: str | None = None,
    ax: plt.Axes | None = None,
    kind: str = "violin",
) -> plt.Figure:
    """Plot metric values across conditions.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Condition name -> array of metric values.
    metric_name : str
        Y-axis label.
    title : str or None
        Plot title.
    ax : Axes or None
        Pre-existing axes.
    kind : str
        "violin" or "box".

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(4, len(data) * 1.5), 5))
    else:
        fig = ax.figure

    conditions = list(data.keys())
    all_vals = []
    all_labels = []
    for cond in conditions:
        vals = np.asarray(data[cond], dtype=float)
        vals = vals[~np.isnan(vals)]
        all_vals.extend(vals.tolist())
        all_labels.extend([cond] * len(vals))

    if not all_vals:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    import pandas as pd
    df = pd.DataFrame({"Condition": all_labels, metric_name: all_vals})

    if kind == "violin":
        sns.violinplot(data=df, x="Condition", y=metric_name, ax=ax, inner="box")
    else:
        sns.boxplot(data=df, x="Condition", y=metric_name, ax=ax)

    sns.stripplot(data=df, x="Condition", y=metric_name, ax=ax,
                  color="black", alpha=0.3, size=3)

    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig
