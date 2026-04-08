"""Avalanche distribution visualization with publication-quality annotations."""

from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cmcc.features.powerlaw_fit import PowerLawResult


def plot_avalanche_distributions(
    sizes: np.ndarray,
    durations: np.ndarray,
    fit_result: PowerLawResult | None = None,
    title: str = "Avalanche Distributions",
    ax_size: plt.Axes | None = None,
    ax_dur: plt.Axes | None = None,
) -> plt.Figure:
    """Plot log-log avalanche size and duration distributions.

    Overlays maximum-likelihood power-law fits with GOF statistics and
    comparator distribution annotations (exponential, lognormal) when
    comparison_results are available in the fit_result.

    Parameters
    ----------
    sizes : np.ndarray
        Avalanche sizes.
    durations : np.ndarray
        Avalanche durations (bins).
    fit_result : PowerLawResult or None
        If provided, overlay power-law fit lines and annotate statistics.
    title : str
        Figure title.
    ax_size, ax_dur : Axes or None
        Pre-existing axes. If None, a new figure is created.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if ax_size is None or ax_dur is None:
        fig, (ax_size, ax_dur) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig = ax_size.figure

    _plot_single_distribution(
        ax_size, sizes, fit_result,
        exponent_attr="tau", xmin_attr="tau_xmin",
        ks_attr="tau_ks_distance", pval_attr="tau_p_value",
        comp_key="size",
        xlabel="Size", ylabel="P(S)", dist_title="Size distribution",
    )

    _plot_single_distribution(
        ax_dur, durations, fit_result,
        exponent_attr="alpha", xmin_attr="alpha_xmin",
        ks_attr="alpha_ks_distance", pval_attr="alpha_p_value",
        comp_key="duration",
        xlabel="Duration (bins)", ylabel="P(D)", dist_title="Duration distribution",
    )

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _plot_single_distribution(
    ax: plt.Axes,
    data: np.ndarray,
    fit_result: PowerLawResult | None,
    exponent_attr: str,
    xmin_attr: str,
    ks_attr: str,
    pval_attr: str,
    comp_key: str,
    xlabel: str,
    ylabel: str,
    dist_title: str,
) -> None:
    """Plot one distribution panel with fit, annotations, and comparator info."""
    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(dist_title)
        return

    vals, counts = np.unique(data, return_counts=True)
    probs = counts / counts.sum()
    ax.loglog(vals, probs, "o", ms=4, alpha=0.7, label="data")

    annotation_lines = []

    if fit_result is not None:
        exponent = getattr(fit_result, exponent_attr, float("nan"))
        xmin = getattr(fit_result, xmin_attr, float("nan"))
        ks_d = getattr(fit_result, ks_attr, float("nan"))
        p_val = getattr(fit_result, pval_attr, float("nan"))

        if not np.isnan(exponent):
            xmin_val = xmin if not np.isnan(xmin) else vals.min()
            x_fit = vals[vals >= xmin_val]
            if len(x_fit) > 1:
                y_fit = x_fit.astype(float) ** (-exponent)
                y_fit *= probs[vals >= xmin_val][0] / y_fit[0]
                ax.loglog(x_fit, y_fit, "-", lw=2, color="C1",
                          label=f"PL fit ({exponent_attr}={exponent:.2f})")

            annotation_lines.append(f"{exponent_attr}={exponent:.3f}")

        if not np.isnan(ks_d):
            annotation_lines.append(f"KS={ks_d:.3f}")

        if not np.isnan(p_val):
            annotation_lines.append(f"GOF p={p_val:.3f}")

        comparisons = getattr(fit_result, "comparison_results", {})
        comp_dict = comparisons.get(comp_key, {})
        for alt_name, alt_stats in comp_dict.items():
            if isinstance(alt_stats, dict):
                R = alt_stats.get("log_likelihood_ratio", float("nan"))
                p = alt_stats.get("p_value", float("nan"))
                if not np.isnan(R):
                    direction = "PL" if R > 0 else alt_name
                    annotation_lines.append(
                        f"vs {alt_name}: R={R:.2f}, p={p:.3f} ({direction})"
                    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(dist_title)
    ax.legend(fontsize=8)

    if annotation_lines:
        annotation_text = "\n".join(annotation_lines)
        ax.text(
            0.98, 0.98, annotation_text,
            transform=ax.transAxes, fontsize=7,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
        )
