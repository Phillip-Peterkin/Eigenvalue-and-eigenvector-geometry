"""Branching ratio computation for criticality assessment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BranchingResult:
    """Branching ratio result.

    Attributes
    ----------
    sigma : float
        Branching ratio: sigma = <n_{t+1}> / <n_t>.
        - sigma < 1: subcritical
        - sigma ~ 1: critical
        - sigma > 1: supercritical
    sigma_std : float
        Standard deviation of per-timestep ratios.
    n_active_mean : float
        Mean number of active channels per time bin.
    n_timesteps : int
        Number of time bins used in computation.
    """

    sigma: float
    sigma_std: float
    n_active_mean: float
    n_timesteps: int


def compute_branching_ratio(
    binary_activity: np.ndarray,
) -> BranchingResult:
    """Compute branching ratio from binary activity matrix.

    Parameters
    ----------
    binary_activity : np.ndarray, shape (n_channels, n_timesteps)
        Binary matrix where 1 = active, 0 = inactive.

    Returns
    -------
    BranchingResult
    """
    if binary_activity.ndim != 2:
        raise ValueError(
            f"Expected 2D binary activity matrix, got shape {binary_activity.shape}"
        )

    n_active = binary_activity.sum(axis=0)
    n_timesteps = len(n_active)

    if n_timesteps < 2:
        return BranchingResult(
            sigma=float("nan"),
            sigma_std=float("nan"),
            n_active_mean=float(np.mean(n_active)),
            n_timesteps=n_timesteps,
        )

    ancestors = n_active[:-1]
    descendants = n_active[1:]

    valid = ancestors > 0
    if valid.sum() == 0:
        return BranchingResult(
            sigma=float("nan"),
            sigma_std=float("nan"),
            n_active_mean=float(np.mean(n_active)),
            n_timesteps=n_timesteps,
        )

    ratios = descendants[valid].astype(float) / ancestors[valid].astype(float)
    sigma = float(np.mean(ratios))
    sigma_std = float(np.std(ratios))

    return BranchingResult(
        sigma=sigma,
        sigma_std=sigma_std,
        n_active_mean=float(np.mean(n_active)),
        n_timesteps=n_timesteps,
    )
