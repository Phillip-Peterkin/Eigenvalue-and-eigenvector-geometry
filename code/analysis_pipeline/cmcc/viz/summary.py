"""MSE curves, DFA log-log plots, branching ratio distributions."""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmcc.features.entropy import MSEResult
from cmcc.features.dfa import DFAResult


def plot_mse_curves(
    results: dict[str, MSEResult],
    title: str = "Multi-Scale Entropy",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot MSE curves for multiple conditions.

    Parameters
    ----------
    results : dict[str, MSEResult]
        Condition name -> MSEResult.
    title : str
        Plot title.
    ax : Axes or None
        Pre-existing axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure

    for label, mse in results.items():
        valid_idx = [i for i, v in enumerate(mse.entropy_values)
                     if not np.isnan(v) and not np.isinf(v)]
        scales = [mse.scales[i] for i in valid_idx]
        vals = [mse.entropy_values[i] for i in valid_idx]
        ax.plot(scales, vals, "o-", label=f"{label} (CI={mse.complexity_index:.2f})")

    ax.set_xlabel("Scale factor")
    ax.set_ylabel("Sample entropy")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_dfa(
    result: DFAResult,
    title: str = "DFA",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot DFA log-log fluctuation vs. scale.

    Parameters
    ----------
    result : DFAResult
        DFA result with scales and fluctuations.
    title : str
        Plot title.
    ax : Axes or None
        Pre-existing axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    if len(result.scales) > 0 and len(result.fluctuations) > 0:
        ax.loglog(result.scales, result.fluctuations, "o", ms=5)
        if not np.isnan(result.alpha) and len(result.scales) >= 2:
            log_s = np.log(result.scales)
            y_fit = np.exp(result.alpha * log_s +
                           (np.log(result.fluctuations[0]) -
                            result.alpha * log_s[0]))
            ax.loglog(result.scales, y_fit, "-", lw=2,
                      label=f"α={result.alpha:.3f}")
            ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f"α = {result.alpha:.3f}",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)

    ax.set_xlabel("Window size")
    ax.set_ylabel("Fluctuation F(n)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_branching_distribution(
    sigma_values: dict[str, list[float]],
    title: str = "Branching Ratio",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot branching ratio distributions per condition.

    Parameters
    ----------
    sigma_values : dict[str, list[float]]
        Condition name -> list of sigma values.
    title : str
        Plot title.
    ax : Axes or None
        Pre-existing axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    import seaborn as sns
    import pandas as pd

    all_vals = []
    all_labels = []
    for label, vals in sigma_values.items():
        clean = [v for v in vals if not np.isnan(v)]
        all_vals.extend(clean)
        all_labels.extend([label] * len(clean))

    if all_vals:
        df = pd.DataFrame({"Condition": all_labels, "σ": all_vals})
        sns.violinplot(data=df, x="Condition", y="σ", ax=ax, inner="box")
        ax.axhline(1.0, ls="--", color="red", alpha=0.5, label="σ=1 (critical)")
        ax.legend(fontsize=8)

    ax.set_title(title)
    fig.tight_layout()
    return fig
