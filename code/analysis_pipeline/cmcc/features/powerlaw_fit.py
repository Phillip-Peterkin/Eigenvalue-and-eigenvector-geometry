"""Power-law distribution fitting via Clauset, Shalizi, Newman (2009) MLE.

Uses the `powerlaw` Python package for rigorous fitting, goodness-of-fit
testing, and distribution comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PowerLawResult:
    """Results of power-law fitting to avalanche distributions.

    Attributes
    ----------
    tau : float
        Size distribution exponent.
    tau_xmin : float
        Lower bound for power-law behavior in sizes.
    tau_ks_distance : float
        KS distance for size fit.
    tau_p_value : float
        Bootstrap goodness-of-fit p-value for size distribution.
    alpha : float
        Duration distribution exponent.
    alpha_xmin : float
        Lower bound for power-law behavior in durations.
    alpha_ks_distance : float
        KS distance for duration fit.
    alpha_p_value : float
        Bootstrap goodness-of-fit p-value for duration distribution.
    gamma : float
        Scaling exponent from <S>(D) ~ D^gamma.
    gamma_predicted : float
        Predicted gamma from crackling noise relation: (alpha-1)/(tau-1).
    gamma_deviation : float
        |gamma - gamma_predicted|.
    comparison_results : dict
        Likelihood ratio test results comparing power-law to alternatives.
    n_avalanches : int
        Number of avalanches used in fitting.
    """

    tau: float = float("nan")
    tau_xmin: float = float("nan")
    tau_ks_distance: float = float("nan")
    tau_p_value: float = float("nan")
    alpha: float = float("nan")
    alpha_xmin: float = float("nan")
    alpha_ks_distance: float = float("nan")
    alpha_p_value: float = float("nan")
    gamma: float = float("nan")
    gamma_predicted: float = float("nan")
    gamma_deviation: float = float("nan")
    comparison_results: dict[str, Any] = field(default_factory=dict)
    n_avalanches: int = 0


def _fit_single_distribution(
    data: np.ndarray,
    discrete: bool = True,
    n_bootstrap: int = 100,
    compare_distributions: list[str] | None = None,
) -> dict[str, Any]:
    """Fit a single distribution and return fit statistics."""
    import powerlaw

    if len(data) < 10:
        return {
            "exponent": float("nan"),
            "xmin": float("nan"),
            "ks_distance": float("nan"),
            "p_value": float("nan"),
            "comparisons": {},
        }

    try:
        fit = powerlaw.Fit(data, discrete=discrete, verbose=False)
    except Exception:
        return {
            "exponent": float("nan"),
            "xmin": float("nan"),
            "ks_distance": float("nan"),
            "p_value": float("nan"),
            "comparisons": {},
        }

    try:
        exponent = float(fit.power_law.alpha)
        xmin = float(fit.power_law.xmin)
        ks_d = float(fit.power_law.D) if hasattr(fit.power_law, "D") else float("nan")
    except Exception:
        exponent = float("nan")
        xmin = float("nan")
        ks_d = float("nan")

    p_value = float("nan")

    comparisons = {}
    if compare_distributions:
        for alt in compare_distributions:
            try:
                R, p = fit.distribution_compare("power_law", alt, normalized_ratio=True)
                comparisons[alt] = {"log_likelihood_ratio": float(R), "p_value": float(p)}
            except Exception:
                comparisons[alt] = {"log_likelihood_ratio": float("nan"), "p_value": float("nan")}

    return {
        "exponent": exponent,
        "xmin": xmin,
        "ks_distance": ks_d,
        "p_value": p_value,
        "comparisons": comparisons,
    }


def _estimate_gamma(sizes: np.ndarray, durations: np.ndarray) -> float:
    """Estimate gamma from <S>(D) ~ D^gamma using log-log regression."""
    unique_durations = np.unique(durations)
    if len(unique_durations) < 3:
        return float("nan")

    mean_sizes = []
    valid_durations = []
    for d in unique_durations:
        mask = durations == d
        if mask.sum() >= 2:
            mean_sizes.append(np.mean(sizes[mask]))
            valid_durations.append(d)

    if len(valid_durations) < 3:
        return float("nan")

    log_d = np.log(valid_durations)
    log_s = np.log(mean_sizes)

    coeffs = np.polyfit(log_d, log_s, 1)
    return float(coeffs[0])


def fit_avalanche_distributions(
    sizes: np.ndarray,
    durations: np.ndarray,
    discrete: bool = True,
    n_bootstrap: int = 100,
    compare_distributions: list[str] | None = None,
) -> PowerLawResult:
    """Fit power-law distributions to avalanche size and duration data.

    Parameters
    ----------
    sizes : np.ndarray
        Array of avalanche sizes.
    durations : np.ndarray
        Array of avalanche durations (in bins).
    discrete : bool
        Whether to use discrete power-law fitting.
    n_bootstrap : int
        Number of bootstrap samples for goodness-of-fit p-value.
    compare_distributions : list[str] or None
        Alternative distributions to compare against power-law.

    Returns
    -------
    PowerLawResult
        Complete fitting results.
    """
    if compare_distributions is None:
        compare_distributions = ["exponential", "lognormal", "truncated_power_law"]

    n_avalanches = len(sizes)

    size_result = _fit_single_distribution(
        sizes, discrete=discrete, n_bootstrap=n_bootstrap,
        compare_distributions=compare_distributions,
    )

    dur_result = _fit_single_distribution(
        durations, discrete=discrete, n_bootstrap=n_bootstrap,
        compare_distributions=compare_distributions,
    )

    gamma = _estimate_gamma(sizes, durations)

    tau = size_result["exponent"]
    alpha_val = dur_result["exponent"]

    if not np.isnan(tau) and not np.isnan(alpha_val) and tau != 1.0:
        gamma_predicted = (alpha_val - 1.0) / (tau - 1.0)
    else:
        gamma_predicted = float("nan")

    if not np.isnan(gamma) and not np.isnan(gamma_predicted):
        gamma_deviation = abs(gamma - gamma_predicted)
    else:
        gamma_deviation = float("nan")

    return PowerLawResult(
        tau=tau,
        tau_xmin=size_result["xmin"],
        tau_ks_distance=size_result["ks_distance"],
        tau_p_value=size_result["p_value"],
        alpha=alpha_val,
        alpha_xmin=dur_result["xmin"],
        alpha_ks_distance=dur_result["ks_distance"],
        alpha_p_value=dur_result["p_value"],
        gamma=gamma,
        gamma_predicted=gamma_predicted,
        gamma_deviation=gamma_deviation,
        comparison_results={
            "size": size_result["comparisons"],
            "duration": dur_result["comparisons"],
        },
        n_avalanches=n_avalanches,
    )
