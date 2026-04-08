"""Within-subject condition contrasts using permutation tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class ContrastResult:
    """Result of a condition contrast.

    Attributes
    ----------
    condition_a : str
        Name of condition A.
    condition_b : str
        Name of condition B.
    mean_a : float
        Mean of metric in condition A.
    mean_b : float
        Mean of metric in condition B.
    effect_size : float
        Hedge's g effect size.
    p_value : float
        Permutation test p-value.
    n_a : int
        Number of observations in A.
    n_b : int
        Number of observations in B.
    """

    condition_a: str
    condition_b: str
    mean_a: float
    mean_b: float
    effect_size: float
    p_value: float
    n_a: int
    n_b: int


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Hedge's g effect size (bias-corrected Cohen's d).

    Parameters
    ----------
    a, b : np.ndarray
        Two groups of observations.

    Returns
    -------
    float
        Hedge's g. Positive means a > b.
    """
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")

    mean_diff = np.mean(a) - np.mean(b)
    pooled_var = ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (
        n_a + n_b - 2
    )
    pooled_sd = np.sqrt(pooled_var)

    if pooled_sd == 0:
        return float("nan")

    d = mean_diff / pooled_sd
    correction = 1 - 3 / (4 * (n_a + n_b) - 9)
    return float(d * correction)


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
    statistic: str = "mean_diff",
) -> float:
    """Two-sample permutation test.

    Parameters
    ----------
    a, b : np.ndarray
        Two groups.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.
    statistic : str
        Test statistic: "mean_diff" or "t".

    Returns
    -------
    float
        Two-tailed p-value.
    """
    rng = np.random.default_rng(seed)
    combined = np.concatenate([a, b])
    n_a = len(a)

    if statistic == "mean_diff":
        observed = abs(np.mean(a) - np.mean(b))
    else:
        t, _ = stats.ttest_ind(a, b, equal_var=False)
        observed = abs(t)

    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]

        if statistic == "mean_diff":
            perm_stat = abs(np.mean(perm_a) - np.mean(perm_b))
        else:
            t, _ = stats.ttest_ind(perm_a, perm_b, equal_var=False)
            perm_stat = abs(t)

        if perm_stat >= observed:
            count += 1

    return (count + 1) / (n_perm + 1)


def condition_contrast(
    values_a: np.ndarray,
    values_b: np.ndarray,
    condition_a: str = "A",
    condition_b: str = "B",
    n_perm: int = 5000,
    seed: int = 42,
) -> ContrastResult:
    """Compare a metric between two conditions.

    Parameters
    ----------
    values_a, values_b : np.ndarray
        Metric values for each condition.
    condition_a, condition_b : str
        Condition labels.
    n_perm : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    ContrastResult
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]

    return ContrastResult(
        condition_a=condition_a,
        condition_b=condition_b,
        mean_a=float(np.mean(a)) if len(a) > 0 else float("nan"),
        mean_b=float(np.mean(b)) if len(b) > 0 else float("nan"),
        effect_size=hedges_g(a, b),
        p_value=permutation_test(a, b, n_perm=n_perm, seed=seed),
        n_a=len(a),
        n_b=len(b),
    )


def fdr_correction(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : list[float]
        Uncorrected p-values.
    alpha : float
        Significance level.

    Returns
    -------
    list[bool]
        Whether each test is significant after FDR correction.
    """
    n = len(p_values)
    if n == 0:
        return []

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    thresholds = alpha * np.arange(1, n + 1) / n
    significant = np.zeros(n, dtype=bool)

    max_sig_idx = -1
    for i in range(n):
        if sorted_p[i] <= thresholds[i]:
            max_sig_idx = i

    if max_sig_idx >= 0:
        significant[sorted_indices[: max_sig_idx + 1]] = True

    return significant.tolist()
