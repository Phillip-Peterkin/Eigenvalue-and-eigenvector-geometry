"""Advanced exceptional-point analyses for non-Hermitian neural criticality.

Four analyses that extend the base EP detection pipeline:

1. State-Switch Contrast — compares EP metrics between task-relevant and
   task-irrelevant stimulus windows to distinguish state from trait.
   Uses circular-shift permutation to preserve temporal autocorrelation.
2. Spectral Radius Sensitivity — correlates the system gain (rho) with
   eigenvalue gap tightening to test whether criticality drives EP approach.
   Reports autocorrelation-adjusted degrees of freedom.
3. SVD Dimension (Topological Dimension) — measures effective rank collapse
   of the Jacobian near EPs as evidence of macro-mode formation.
   Includes random-matrix null model to control for mathematical coupling
   between EP score and effective rank.
4. Petermann Noise Prediction — tests whether the Petermann excess noise
   factor predicts high-gamma power bursts via lag/lead analysis.
   Uses NaN-aware cross-correlation and non-overlapping Granger stride.

All functions are pure computation (no I/O). They accept JacobianResult
and ExceptionalPointResult from cmcc.analysis.dynamical_systems.

Units
-----
- Window centers are in sample indices.
- Stimulus intervals are (start_sample, end_sample) tuples.
- step_sec is the Jacobian window step in seconds (for lag conversion).
- High-gamma power is in arbitrary units (mean squared amplitude).
- Petermann factors are dimensionless; log-transformed for correlations.

Scientific validity notes
-------------------------
- All within-subject correlations report n_eff (effective sample size after
  Bartlett's autocorrelation correction) and p_adjusted alongside p_nominal.
- State contrast uses circular-shift permutation to preserve temporal
  autocorrelation structure (Fix I2).
- SVD dimension includes a random-matrix null model to distinguish neural
  effects from mathematical coupling between EP score and effective rank (Fix C3).
- Petermann analysis uses NaN-aware cross-correlation instead of median-fill
  (Fix M1), and non-overlapping stride for Granger test (Fix I3).
- Multiple comparisons within state contrast are FDR-corrected (Fix I1).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats

from cmcc.analysis.contrasts import ContrastResult, condition_contrast, fdr_correction
from cmcc.analysis.dynamical_systems import (
    ExceptionalPointResult,
    JacobianResult,
    estimate_jacobian,
    detect_exceptional_points,
)


def _effective_n(x: np.ndarray, y: np.ndarray | None = None) -> int:
    """Estimate effective sample size under autocorrelation (Bartlett's formula).

    For a single series: n_eff = n * (1 - rho1) / (1 + rho1)
    For bivariate correlation: n_eff = n * (1 - rho_x * rho_y) / (1 + rho_x * rho_y)
    where rho1 is the lag-1 autocorrelation.

    Parameters
    ----------
    x : np.ndarray, shape (n,)
    y : np.ndarray, shape (n,), optional
        If provided, compute bivariate effective N.

    Returns
    -------
    int
        Effective sample size, clamped to [3, n].
    """
    n = len(x)
    if n < 4:
        return n

    x_c = x - np.mean(x)
    var_x = np.sum(x_c ** 2)
    rho_x = np.sum(x_c[:-1] * x_c[1:]) / var_x if var_x > 0 else 0.0
    rho_x = np.clip(rho_x, -0.99, 0.99)

    if y is not None:
        y_c = y - np.mean(y)
        var_y = np.sum(y_c ** 2)
        rho_y = np.sum(y_c[:-1] * y_c[1:]) / var_y if var_y > 0 else 0.0
        rho_y = np.clip(rho_y, -0.99, 0.99)
        product = rho_x * rho_y
    else:
        product = rho_x

    denom = 1.0 + product
    if denom <= 0:
        return n
    n_eff = n * (1.0 - product) / denom
    return max(3, min(n, int(round(n_eff))))


def _adjusted_correlation_p(r: float, n_eff: int) -> float:
    """Compute p-value for Pearson r using autocorrelation-adjusted df.

    Uses t = r * sqrt(n_eff - 2) / sqrt(1 - r^2) with n_eff - 2 df.
    """
    if n_eff < 3 or not np.isfinite(r):
        return float("nan")
    r2 = r * r
    if r2 >= 1.0:
        return 0.0
    t_stat = r * np.sqrt(n_eff - 2) / np.sqrt(1.0 - r2)
    return float(2.0 * sp_stats.t.sf(abs(t_stat), df=n_eff - 2))


def _circular_shift_permutation_test(
    values: np.ndarray,
    condition_mask: np.ndarray,
    n_perm: int = 500,
    seed: int = 42,
) -> float:
    """Circular-shift permutation test preserving temporal autocorrelation.

    Instead of shuffling individual window labels (which destroys temporal
    structure), circularly shift the condition labels relative to the values
    by a random offset on each permutation.

    Parameters
    ----------
    values : np.ndarray, shape (n,)
        Time-ordered metric values for ALL windows (both conditions + excluded).
    condition_mask : np.ndarray, shape (n,), dtype int
        0 = excluded, 1 = condition A (relevant), -1 = condition B (irrelevant).
    n_perm : int
    seed : int

    Returns
    -------
    float
        Two-tailed p-value.
    """
    rng = np.random.default_rng(seed)
    n = len(values)

    a_vals = values[condition_mask == 1]
    b_vals = values[condition_mask == -1]
    if len(a_vals) == 0 or len(b_vals) == 0:
        return float("nan")

    observed = abs(np.mean(a_vals) - np.mean(b_vals))
    count = 0
    for _ in range(n_perm):
        shift = rng.integers(1, n)
        shifted_mask = np.roll(condition_mask, shift)
        a_perm = values[shifted_mask == 1]
        b_perm = values[shifted_mask == -1]
        if len(a_perm) == 0 or len(b_perm) == 0:
            continue
        perm_stat = abs(np.mean(a_perm) - np.mean(b_perm))
        if perm_stat >= observed:
            count += 1

    return (count + 1) / (n_perm + 1)


@dataclass
class StateContrastResult:
    """Result of state-switch contrast between task conditions.

    Attributes
    ----------
    n_relevant_windows : int
        Number of Jacobian windows falling in task-relevant intervals.
    n_irrelevant_windows : int
        Number of Jacobian windows falling in task-irrelevant intervals.
    n_eff_relevant : int
        Effective (autocorrelation-adjusted) N for relevant windows.
    n_eff_irrelevant : int
        Effective (autocorrelation-adjusted) N for irrelevant windows.
    gap_contrast : ContrastResult
        Min eigenvalue gap contrast (Relevant vs Irrelevant).
    condition_number_contrast : ContrastResult
        Condition number contrast.
    ep_score_contrast : ContrastResult
        EP score contrast.
    spectral_radius_contrast : ContrastResult
        Spectral radius contrast.
    fdr_significant : dict[str, bool]
        Whether each metric's contrast survives FDR correction at alpha=0.05.
    circular_shift_p : dict[str, float]
        P-values from circular-shift permutation (temporally valid).

    Notes
    -----
    The permutation p-values in the ContrastResult objects use simple
    label shuffling (nominal, anti-conservative with autocorrelated windows).
    The circular_shift_p values preserve temporal structure and are the
    scientifically valid p-values for inference.
    """

    n_relevant_windows: int
    n_irrelevant_windows: int
    n_eff_relevant: int
    n_eff_irrelevant: int
    gap_contrast: ContrastResult
    condition_number_contrast: ContrastResult
    ep_score_contrast: ContrastResult
    spectral_radius_contrast: ContrastResult
    fdr_significant: dict = field(default_factory=dict)
    circular_shift_p: dict = field(default_factory=dict)


@dataclass
class SVDDimensionResult:
    """Result of SVD-based dimensionality analysis of the Jacobian.

    Attributes
    ----------
    participation_ratio : np.ndarray, shape (n_windows,)
        (sum sigma_i)^2 / sum(sigma_i^2). Equals n_channels for full-rank
        uniform spectrum, 1 for rank-1.
    effective_rank : np.ndarray, shape (n_windows,)
        exp(entropy of normalized singular values). Model-free measure
        of the number of active modes.
    pr_vs_ep_score : dict
        Pearson and Spearman correlation of PR with EP score.
        Includes n_eff and p_adjusted for autocorrelation correction.
    erank_vs_ep_score : dict
        Pearson and Spearman correlation of effective rank with EP score.
        Includes n_eff and p_adjusted for autocorrelation correction.
    mean_pr : float
        Time-averaged participation ratio.
    mean_erank : float
        Time-averaged effective rank.
    null_r_mean : float
        Mean erank-vs-EP r from random-matrix null model.
    null_r_std : float
        Std of erank-vs-EP r from random-matrix null model.
    null_p : float
        Fraction of null surrogates with |r| >= |r_observed|.
    """

    participation_ratio: np.ndarray
    effective_rank: np.ndarray
    pr_vs_ep_score: dict
    erank_vs_ep_score: dict
    mean_pr: float
    mean_erank: float
    null_r_mean: float = float("nan")
    null_r_std: float = float("nan")
    null_p: float = float("nan")


@dataclass
class PetermannNoiseResult:
    """Result of Petermann factor vs high-gamma power analysis.

    Attributes
    ----------
    correlation_r : float
        Pearson r between log(K) and HG power.
    correlation_p : float
        Nominal p-value for the Pearson correlation (assumes independence).
    p_adjusted : float
        Autocorrelation-adjusted p-value for the Pearson correlation.
    n_eff : int
        Effective sample size after autocorrelation correction.
    cross_correlation : np.ndarray
        NaN-aware normalized cross-correlation at each lag.
    lags : np.ndarray
        Lag indices (negative = K leads HG).
    peak_lag : int
        Lag of maximum cross-correlation.
    peak_lag_seconds : float
        Peak lag converted to seconds.
    granger_f : float
        F-statistic for incremental predictive power of K(t-1) on HG(t).
        Uses non-overlapping window stride.
    granger_p : float
        P-value for the Granger F-test.
    granger_stride : int
        Window stride used for Granger test (1 = overlapping, >1 = non-overlapping).
    n_valid_windows : int
        Number of windows with finite Petermann factors used.
    """

    correlation_r: float
    correlation_p: float
    p_adjusted: float
    n_eff: int
    cross_correlation: np.ndarray
    lags: np.ndarray
    peak_lag: int
    peak_lag_seconds: float
    granger_f: float
    granger_p: float
    granger_stride: int
    n_valid_windows: int


def _classify_windows(
    window_centers: np.ndarray,
    intervals: list[tuple[int, int]],
) -> np.ndarray:
    """Return boolean mask of windows whose center falls in any interval.

    Parameters
    ----------
    window_centers : np.ndarray, shape (n_windows,)
        Sample index of each window center.
    intervals : list of (start_sample, end_sample)
        Time intervals.

    Returns
    -------
    np.ndarray, shape (n_windows,), dtype bool
    """
    mask = np.zeros(len(window_centers), dtype=bool)
    for start, end in intervals:
        mask |= (window_centers >= start) & (window_centers < end)
    return mask


def compute_state_contrast(
    jac_result: JacobianResult,
    ep_result: ExceptionalPointResult,
    stim_intervals_rel: list[tuple[int, int]],
    stim_intervals_irr: list[tuple[int, int]],
    n_perm: int = 500,
    seed: int = 42,
) -> StateContrastResult:
    """Compare EP metrics between task-relevant and task-irrelevant windows.

    A Jacobian window is classified as 'relevant' or 'irrelevant' if its
    center sample falls within a stimulus presentation interval of that type.
    Windows outside all stimulus intervals are excluded.

    Uses circular-shift permutation to preserve temporal autocorrelation
    structure. Also applies FDR correction across the 4 metric tests.

    Parameters
    ----------
    jac_result : JacobianResult
    ep_result : ExceptionalPointResult
    stim_intervals_rel : list of (start_sample, end_sample)
        Task-relevant stimulus intervals.
    stim_intervals_irr : list of (start_sample, end_sample)
        Task-irrelevant stimulus intervals.
    n_perm : int
        Permutations for contrast tests.
    seed : int

    Returns
    -------
    StateContrastResult
    """
    centers = jac_result.window_centers
    rel_mask = _classify_windows(centers, stim_intervals_rel)
    irr_mask = _classify_windows(centers, stim_intervals_irr)

    gap_contrast = condition_contrast(
        ep_result.min_eigenvalue_gaps[rel_mask],
        ep_result.min_eigenvalue_gaps[irr_mask],
        "Relevant", "Irrelevant", n_perm=n_perm, seed=seed,
    )
    cond_contrast = condition_contrast(
        jac_result.condition_numbers[rel_mask],
        jac_result.condition_numbers[irr_mask],
        "Relevant", "Irrelevant", n_perm=n_perm, seed=seed + 1,
    )
    ep_contrast = condition_contrast(
        ep_result.ep_scores[rel_mask],
        ep_result.ep_scores[irr_mask],
        "Relevant", "Irrelevant", n_perm=n_perm, seed=seed + 2,
    )
    sr_contrast = condition_contrast(
        jac_result.spectral_radius[rel_mask],
        jac_result.spectral_radius[irr_mask],
        "Relevant", "Irrelevant", n_perm=n_perm, seed=seed + 3,
    )

    n_eff_rel = _effective_n(ep_result.min_eigenvalue_gaps[rel_mask]) if rel_mask.sum() > 3 else int(rel_mask.sum())
    n_eff_irr = _effective_n(ep_result.min_eigenvalue_gaps[irr_mask]) if irr_mask.sum() > 3 else int(irr_mask.sum())

    p_values = [
        gap_contrast.p_value,
        cond_contrast.p_value,
        ep_contrast.p_value,
        sr_contrast.p_value,
    ]
    metric_names = ["gap", "condition_number", "ep_score", "spectral_radius"]
    fdr_sig = fdr_correction(p_values, alpha=0.05)
    fdr_significant = dict(zip(metric_names, fdr_sig))

    condition_label = np.zeros(len(centers), dtype=int)
    condition_label[rel_mask] = 1
    condition_label[irr_mask] = -1

    metrics_arrays = {
        "gap": ep_result.min_eigenvalue_gaps,
        "condition_number": jac_result.condition_numbers,
        "ep_score": ep_result.ep_scores,
        "spectral_radius": jac_result.spectral_radius,
    }
    circular_shift_p = {}
    for i, name in enumerate(metric_names):
        circular_shift_p[name] = _circular_shift_permutation_test(
            metrics_arrays[name], condition_label,
            n_perm=n_perm, seed=seed + 10 + i,
        )

    return StateContrastResult(
        n_relevant_windows=int(rel_mask.sum()),
        n_irrelevant_windows=int(irr_mask.sum()),
        n_eff_relevant=n_eff_rel,
        n_eff_irrelevant=n_eff_irr,
        gap_contrast=gap_contrast,
        condition_number_contrast=cond_contrast,
        ep_score_contrast=ep_contrast,
        spectral_radius_contrast=sr_contrast,
        fdr_significant=fdr_significant,
        circular_shift_p=circular_shift_p,
    )


def compute_spectral_radius_sensitivity(
    jac_result: JacobianResult,
    ep_result: ExceptionalPointResult,
) -> dict:
    """Correlate spectral radius with min eigenvalue gap across windows.

    Tests whether pushing gain toward stability edge (rho -> 1) tightens
    eigenvalue modes (gap -> 0).

    Reports both nominal p-values (assuming independence) and
    autocorrelation-adjusted p-values using effective degrees of freedom.

    Parameters
    ----------
    jac_result : JacobianResult
    ep_result : ExceptionalPointResult

    Returns
    -------
    dict with keys: r, p_nominal, p_adjusted, rho, p_spearman, n_windows, n_eff
    """
    sr = jac_result.spectral_radius
    gaps = ep_result.min_eigenvalue_gaps
    n = len(sr)

    if n < 3:
        return {
            "r": float("nan"), "p_nominal": float("nan"),
            "p_adjusted": float("nan"),
            "rho": float("nan"), "p_spearman": float("nan"),
            "n_windows": n, "n_eff": n,
        }

    r, p = sp_stats.pearsonr(sr, gaps)
    rho, p_sp = sp_stats.spearmanr(sr, gaps)
    n_eff = _effective_n(sr, gaps)
    p_adj = _adjusted_correlation_p(float(r), n_eff)

    return {
        "r": float(r),
        "p_nominal": float(p),
        "p_adjusted": float(p_adj),
        "rho": float(rho),
        "p_spearman": float(p_sp),
        "n_windows": n,
        "n_eff": n_eff,
    }


def _compute_svd_null(
    n_windows: int,
    n_channels: int,
    spectral_radii: np.ndarray,
    n_surrogates: int = 200,
    seed: int = 42,
) -> tuple[float, float]:
    """Random-matrix null model for erank-vs-EP-score coupling.

    Generates random matrices with matched spectral radius per window,
    computes EP score and effective rank, and returns the null distribution
    of their Pearson correlation.

    Returns
    -------
    null_r_mean : float
    null_r_std : float
    """
    rng = np.random.default_rng(seed)
    null_rs = np.zeros(n_surrogates)

    for s in range(n_surrogates):
        eranks_null = np.zeros(n_windows)
        ep_scores_null = np.zeros(n_windows)

        for w in range(n_windows):
            J = rng.standard_normal((n_channels, n_channels))
            evals = np.linalg.eigvals(J)
            max_abs = np.max(np.abs(evals))
            if max_abs > 0:
                J = J * (spectral_radii[w] / max_abs)
                evals = evals * (spectral_radii[w] / max_abs)

            sigma = np.linalg.svd(J, compute_uv=False)
            sigma = np.abs(sigma)
            s_sum = sigma.sum()
            if s_sum > 0:
                p = sigma / s_sum
                p = p[p > 0]
                eranks_null[w] = np.exp(-np.sum(p * np.log(p)))

            gaps = np.inf
            best_overlap = 0.0
            _, evecs = np.linalg.eig(J)
            for ii in range(min(n_channels, 10)):
                for jj in range(ii + 1, min(n_channels, 10)):
                    g = abs(evals[ii] - evals[jj])
                    if g < gaps:
                        gaps = g
                        v_i = evecs[:, ii]
                        v_j = evecs[:, jj]
                        ni = np.linalg.norm(v_i)
                        nj = np.linalg.norm(v_j)
                        if ni > 0 and nj > 0:
                            best_overlap = abs(np.dot(np.conj(v_i), v_j)) / (ni * nj)
            ep_scores_null[w] = best_overlap / (gaps + 1e-10)

        if n_windows >= 3:
            try:
                r_null, _ = sp_stats.pearsonr(eranks_null, ep_scores_null)
                null_rs[s] = r_null
            except Exception:
                null_rs[s] = float("nan")
        else:
            null_rs[s] = float("nan")

    valid = null_rs[np.isfinite(null_rs)]
    if len(valid) > 0:
        return float(np.mean(valid)), float(np.std(valid))
    return float("nan"), float("nan")


def compute_svd_dimension(
    jac_result: JacobianResult,
    ep_result: ExceptionalPointResult,
    run_null: bool = True,
    n_null_surrogates: int = 200,
    null_seed: int = 42,
) -> SVDDimensionResult:
    """Compute SVD-based dimensionality metrics for each Jacobian window.

    Participation Ratio: PR = (sum sigma_i)^2 / sum(sigma_i^2)
    Effective Rank: erank = exp(-sum(p_i * log(p_i))) where p_i = sigma_i / sum(sigma)

    Includes autocorrelation-adjusted p-values and a random-matrix null model
    to control for mathematical coupling between EP score and effective rank.

    Parameters
    ----------
    jac_result : JacobianResult
    ep_result : ExceptionalPointResult
    run_null : bool
        If True, compute random-matrix null model for erank-EP coupling.
    n_null_surrogates : int
        Number of surrogate iterations for null model.
    null_seed : int

    Returns
    -------
    SVDDimensionResult
    """
    n_windows = jac_result.jacobians.shape[0]
    n_channels = jac_result.jacobians.shape[1]
    pr = np.zeros(n_windows)
    erank = np.zeros(n_windows)

    for w in range(n_windows):
        J = jac_result.jacobians[w]
        sigma = np.linalg.svd(J, compute_uv=False)
        sigma = np.abs(sigma)

        s_sum = sigma.sum()
        s_sq_sum = (sigma ** 2).sum()
        pr[w] = (s_sum ** 2) / s_sq_sum if s_sq_sum > 0 else 0.0

        if s_sum > 0:
            p = sigma / s_sum
            p = p[p > 0]
            erank[w] = np.exp(-np.sum(p * np.log(p)))
        else:
            erank[w] = 0.0

    ep_scores = ep_result.ep_scores

    if n_windows >= 3:
        r_pr, p_pr = sp_stats.pearsonr(pr, ep_scores)
        rho_pr, psp_pr = sp_stats.spearmanr(pr, ep_scores)
        r_er, p_er = sp_stats.pearsonr(erank, ep_scores)
        rho_er, psp_er = sp_stats.spearmanr(erank, ep_scores)
        n_eff_pr = _effective_n(pr, ep_scores)
        n_eff_er = _effective_n(erank, ep_scores)
        p_adj_pr = _adjusted_correlation_p(float(r_pr), n_eff_pr)
        p_adj_er = _adjusted_correlation_p(float(r_er), n_eff_er)
    else:
        r_pr = p_pr = rho_pr = psp_pr = float("nan")
        r_er = p_er = rho_er = psp_er = float("nan")
        n_eff_pr = n_eff_er = n_windows
        p_adj_pr = p_adj_er = float("nan")

    null_r_mean = float("nan")
    null_r_std = float("nan")
    null_p = float("nan")
    if run_null and n_windows >= 3:
        null_r_mean, null_r_std = _compute_svd_null(
            n_windows, n_channels, jac_result.spectral_radius,
            n_surrogates=n_null_surrogates, seed=null_seed,
        )
        if np.isfinite(null_r_mean) and np.isfinite(null_r_std) and null_r_std > 0:
            z_obs = (float(r_er) - null_r_mean) / null_r_std
            null_p = float(2.0 * sp_stats.norm.sf(abs(z_obs)))

    return SVDDimensionResult(
        participation_ratio=pr,
        effective_rank=erank,
        pr_vs_ep_score={
            "r": float(r_pr), "p_nominal": float(p_pr),
            "p_adjusted": float(p_adj_pr), "n_eff": n_eff_pr,
            "rho": float(rho_pr), "p_spearman": float(psp_pr),
        },
        erank_vs_ep_score={
            "r": float(r_er), "p_nominal": float(p_er),
            "p_adjusted": float(p_adj_er), "n_eff": n_eff_er,
            "rho": float(rho_er), "p_spearman": float(psp_er),
        },
        mean_pr=float(np.mean(pr)),
        mean_erank=float(np.mean(erank)),
        null_r_mean=null_r_mean,
        null_r_std=null_r_std,
        null_p=null_p,
    )


def compute_petermann_noise(
    ep_result: ExceptionalPointResult,
    hg_power_per_window: np.ndarray,
    step_sec: float = 0.1,
    window_sec: float = 0.5,
    max_lag: int = 20,
) -> PetermannNoiseResult:
    """Correlate Petermann factor with high-gamma power and test temporal precedence.

    Uses log(K) to handle the extreme dynamic range of K near EPs.
    Infinite / NaN K values are excluded.

    Cross-correlation convention: negative lag means K leads HG power.
    Uses NaN-aware cross-correlation (no median-fill).

    Granger-like test uses non-overlapping window stride to avoid inflated
    F-statistics from data overlap between adjacent windows.

    Parameters
    ----------
    ep_result : ExceptionalPointResult
    hg_power_per_window : np.ndarray, shape (n_windows,)
        Mean high-gamma power per Jacobian window.
    step_sec : float
        Time step between consecutive Jacobian windows (seconds).
    window_sec : float
        Duration of each Jacobian window (seconds). Used to compute
        non-overlapping stride for Granger test.
    max_lag : int
        Maximum lag (in window steps) for cross-correlation.

    Returns
    -------
    PetermannNoiseResult
    """
    K = ep_result.petermann_factors.copy()
    hg = np.asarray(hg_power_per_window, dtype=float)
    granger_stride = max(1, round(window_sec / step_sec))

    finite_mask = np.isfinite(K) & (K > 0) & np.isfinite(hg)
    n_valid = int(finite_mask.sum())

    lags = np.arange(-max_lag, max_lag + 1)

    if n_valid < 10:
        return PetermannNoiseResult(
            correlation_r=float("nan"),
            correlation_p=float("nan"),
            p_adjusted=float("nan"),
            n_eff=0,
            cross_correlation=np.full(len(lags), np.nan),
            lags=lags,
            peak_lag=0,
            peak_lag_seconds=0.0,
            granger_f=float("nan"),
            granger_p=float("nan"),
            granger_stride=granger_stride,
            n_valid_windows=n_valid,
        )

    logK = np.full_like(K, np.nan)
    logK[finite_mask] = np.log(K[finite_mask])

    valid_logK = logK[finite_mask]
    valid_hg = hg[finite_mask]
    r_corr, p_corr = sp_stats.pearsonr(valid_logK, valid_hg)
    n_eff = _effective_n(valid_logK, valid_hg)
    p_adj = _adjusted_correlation_p(float(r_corr), n_eff)

    logK_mean = np.nanmean(logK)
    logK_std = np.nanstd(logK[finite_mask])
    hg_mean = np.mean(hg[np.isfinite(hg)])
    hg_std = np.std(hg[np.isfinite(hg)])

    n_total = len(logK)
    xcorr = np.full(len(lags), np.nan)
    for i, lag in enumerate(lags):
        if lag < 0:
            x_idx = np.arange(0, n_total + lag)
            y_idx = np.arange(-lag, n_total)
        elif lag > 0:
            x_idx = np.arange(lag, n_total)
            y_idx = np.arange(0, n_total - lag)
        else:
            x_idx = np.arange(n_total)
            y_idx = np.arange(n_total)

        x_vals = logK[x_idx]
        y_vals = hg[y_idx]
        pair_valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        n_pairs = int(pair_valid.sum())

        if n_pairs >= 10 and logK_std > 0 and hg_std > 0:
            x_z = (x_vals[pair_valid] - logK_mean) / logK_std
            y_z = (y_vals[pair_valid] - hg_mean) / hg_std
            xcorr[i] = np.mean(x_z * y_z)

    finite_xcorr = np.where(np.isfinite(xcorr))[0]
    if len(finite_xcorr) > 0:
        peak_idx = finite_xcorr[np.argmax(xcorr[finite_xcorr])]
    else:
        peak_idx = len(lags) // 2
    peak_lag = int(lags[peak_idx])
    peak_lag_sec = peak_lag * step_sec

    both_valid = np.isfinite(logK) & np.isfinite(hg)
    valid_idx = np.where(both_valid)[0]
    valid_idx = valid_idx[valid_idx >= granger_stride]
    valid_idx_sub = valid_idx[::granger_stride]

    granger_f = float("nan")
    granger_p = float("nan")
    if len(valid_idx_sub) >= 10:
        lagged_idx = valid_idx_sub - granger_stride
        lagged_valid = np.isfinite(hg[lagged_idx]) & np.isfinite(logK[lagged_idx])
        valid_idx_sub = valid_idx_sub[lagged_valid]
        lagged_idx = valid_idx_sub - granger_stride

        if len(valid_idx_sub) >= 10:
            y = hg[valid_idx_sub]
            x_restricted = hg[lagged_idx].reshape(-1, 1)
            x_full = np.column_stack([hg[lagged_idx], logK[lagged_idx]])

            try:
                from numpy.linalg import lstsq
                X_r = np.column_stack([x_restricted, np.ones(len(y))])
                X_f = np.column_stack([x_full, np.ones(len(y))])
                beta_r, _, _, _ = lstsq(X_r, y, rcond=None)
                beta_f, _, _, _ = lstsq(X_f, y, rcond=None)

                ss_r = np.sum((y - X_r @ beta_r) ** 2)
                ss_f = np.sum((y - X_f @ beta_f) ** 2)

                df_diff = 1
                df_resid = len(y) - 3
                if df_resid > 0 and ss_f > 0:
                    f_stat = ((ss_r - ss_f) / df_diff) / (ss_f / df_resid)
                    granger_f = float(f_stat)
                    granger_p = float(1.0 - sp_stats.f.cdf(f_stat, df_diff, df_resid))
            except Exception:
                pass

    return PetermannNoiseResult(
        correlation_r=float(r_corr),
        correlation_p=float(p_corr),
        p_adjusted=float(p_adj),
        n_eff=n_eff,
        cross_correlation=xcorr,
        lags=lags,
        peak_lag=peak_lag,
        peak_lag_seconds=peak_lag_sec,
        granger_f=granger_f,
        granger_p=granger_p,
        granger_stride=granger_stride,
        n_valid_windows=n_valid,
    )


def compute_petermann_noise_surrogate(
    data: np.ndarray,
    hg_power_per_window: np.ndarray,
    observed_r: float,
    sfreq: float,
    window_sec: float = 0.5,
    step_sec: float = 0.1,
    max_channels: int = 30,
    n_surrogates: int = 200,
    seed: int = 42,
) -> dict:
    """Phase-randomized surrogate control for Petermann-HG correlation.

    Tests whether the observed log(K)-vs-HG-power correlation exceeds
    what would be expected from estimation artifacts. For each surrogate:
    1. Phase-randomize the multi-channel data (preserving amplitude spectrum
       and cross-channel structure).
    2. Re-estimate Jacobian and EP on the surrogate data.
    3. Compute log(K)-vs-HG-power Pearson r.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
    hg_power_per_window : np.ndarray, shape (n_windows,)
    observed_r : float
        The observed Pearson r from the real data.
    sfreq : float
    window_sec, step_sec : float
    max_channels : int
    n_surrogates : int
    seed : int

    Returns
    -------
    dict with keys: surrogate_rs, surrogate_mean_r, surrogate_std_r, surrogate_p
    """
    rng = np.random.default_rng(seed)
    n_ch, n_samp = data.shape
    surrogate_rs = np.full(n_surrogates, np.nan)

    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)

    ch_use = min(n_ch, max_channels)
    if n_ch > max_channels:
        data_sub = data[:max_channels]
    else:
        data_sub = data

    for s in range(n_surrogates):
        fft_data = np.fft.rfft(data_sub, axis=1)
        n_freq = fft_data.shape[1]
        random_phases = rng.uniform(0, 2 * np.pi, n_freq)
        random_phases[0] = 0
        if n_samp % 2 == 0 and n_freq > 1:
            random_phases[-1] = 0
        phase_shift = np.exp(1j * random_phases)
        surr_fft = fft_data * phase_shift[np.newaxis, :]
        surr_data = np.fft.irfft(surr_fft, n=n_samp, axis=1)

        try:
            jac_surr = estimate_jacobian(
                surr_data, window_size=window_samples,
                step_size=step_samples, regularization=1e-4,
            )
            ep_surr = detect_exceptional_points(jac_surr)

            K_surr = ep_surr.petermann_factors
            finite = np.isfinite(K_surr) & (K_surr > 0)
            if finite.sum() >= 10:
                logK_surr = np.log(K_surr[finite])
                hg_surr = hg_power_per_window[finite] if len(hg_power_per_window) == len(K_surr) else hg_power_per_window[:finite.sum()]
                if len(logK_surr) == len(hg_surr):
                    r_surr, _ = sp_stats.pearsonr(logK_surr, hg_surr)
                    surrogate_rs[s] = r_surr
        except Exception:
            continue

    valid_rs = surrogate_rs[np.isfinite(surrogate_rs)]
    if len(valid_rs) > 0:
        surrogate_p = float(np.mean(np.abs(valid_rs) >= abs(observed_r)))
        return {
            "surrogate_rs": surrogate_rs,
            "surrogate_mean_r": float(np.mean(valid_rs)),
            "surrogate_std_r": float(np.std(valid_rs)),
            "surrogate_p": surrogate_p,
            "n_valid_surrogates": len(valid_rs),
        }
    return {
        "surrogate_rs": surrogate_rs,
        "surrogate_mean_r": float("nan"),
        "surrogate_std_r": float("nan"),
        "surrogate_p": float("nan"),
        "n_valid_surrogates": 0,
    }


def compute_alpha_power_per_window(
    data: np.ndarray,
    sfreq: float,
    window_centers: np.ndarray,
    window_samples: int,
    alpha_band: tuple[float, float] = (8.0, 12.0),
) -> np.ndarray:
    """Compute alpha-band power per Jacobian window.

    Bandpass-filters each PCA component to the alpha band (8-12 Hz default)
    using a zero-phase 4th-order Butterworth filter, then computes mean
    squared amplitude across components in each window.

    Alpha power from PCA components reflects the projection of scalp alpha
    onto the dominant variance axes. This is valid because PCA is a linear
    transform preserving frequency content, but the spatial interpretation
    is mixed.

    Parameters
    ----------
    data : np.ndarray, shape (n_components, n_samples)
        PCA-reduced time series (broadband, 0.5-45 Hz after preprocessing).
    sfreq : float
        Sampling rate in Hz.
    window_centers : np.ndarray, shape (n_windows,)
        Center sample indices for each Jacobian window.
    window_samples : int
        Number of samples per window.
    alpha_band : tuple[float, float]
        (low, high) frequency in Hz for alpha band.

    Returns
    -------
    np.ndarray, shape (n_windows,)
        Mean alpha-band power per window.
    """
    from scipy.signal import butter, sosfiltfilt

    nyq = sfreq / 2.0
    lo = alpha_band[0] / nyq
    hi = alpha_band[1] / nyq
    if hi >= 1.0:
        hi = 0.99
    sos = butter(4, [lo, hi], btype="band", output="sos")

    data_alpha = sosfiltfilt(sos, data, axis=1)

    n_ch, n_total = data.shape
    n_windows = len(window_centers)
    half_w = window_samples // 2
    alpha_power = np.zeros(n_windows)
    for i, c in enumerate(window_centers):
        c = int(c)
        start = max(0, c - half_w)
        end = min(n_total, c + half_w)
        if end > start:
            alpha_power[i] = np.mean(data_alpha[:, start:end] ** 2)

    return alpha_power


def compute_condition_number_alpha_sensitivity(
    jac_result: JacobianResult,
    alpha_power_per_window: np.ndarray,
) -> dict:
    """Correlate eigenvector condition number with alpha-band power.

    Tests whether EP proximity (high condition number = near-defective
    eigenvector matrix) couples with alpha oscillatory power, especially
    under propofol where GABAergic enhancement produces alpha hypersynchrony.

    Uses log(condition_number) to handle the extreme dynamic range of
    condition numbers near EPs.

    Parameters
    ----------
    jac_result : JacobianResult
        Contains condition_numbers array from Jacobian estimation.
    alpha_power_per_window : np.ndarray, shape (n_windows,)

    Returns
    -------
    dict with keys: r, p_nominal, p_adjusted, rho, p_spearman, n_windows, n_eff
    """
    cond = jac_result.condition_numbers
    alpha = np.asarray(alpha_power_per_window, dtype=float)
    n = min(len(cond), len(alpha))
    cond = cond[:n]
    alpha = alpha[:n]

    valid = np.isfinite(cond) & np.isfinite(alpha) & (cond > 0) & (alpha > 0)
    n_valid = int(valid.sum())

    if n_valid < 10:
        return {
            "r": float("nan"), "p_nominal": float("nan"),
            "p_adjusted": float("nan"),
            "rho": float("nan"), "p_spearman": float("nan"),
            "n_windows": n_valid, "n_eff": 0,
        }

    log_cond = np.log(cond[valid])
    alpha_v = alpha[valid]

    r, p = sp_stats.pearsonr(log_cond, alpha_v)
    rho, p_sp = sp_stats.spearmanr(log_cond, alpha_v)
    n_eff = _effective_n(log_cond, alpha_v)
    p_adj = _adjusted_correlation_p(float(r), n_eff)

    return {
        "r": float(r),
        "p_nominal": float(p),
        "p_adjusted": float(p_adj),
        "rho": float(rho),
        "p_spearman": float(p_sp),
        "n_windows": n_valid,
        "n_eff": n_eff,
    }


def compute_alternative_spacing(
    eigenvalues: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute alternative nearest-neighbor eigenvalue spacing summaries.

    The standard pipeline uses the minimum nearest-neighbor gap, which is
    a fragile extreme statistic. This function computes additional summaries
    (median and 10th percentile of all pairwise nearest-neighbor distances)
    to test whether state patterns are robust to spacing metric choice.

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (n_windows, n_channels)
        Complex eigenvalues per window (from JacobianResult.eigenvalues).

    Returns
    -------
    dict with keys:
        min_gap : np.ndarray, shape (n_windows,)
        median_nn_gap : np.ndarray, shape (n_windows,)
        p10_nn_gap : np.ndarray, shape (n_windows,)
    """
    n_windows, n_ch = eigenvalues.shape
    min_gap = np.zeros(n_windows)
    median_nn_gap = np.zeros(n_windows)
    p10_nn_gap = np.zeros(n_windows)

    for w in range(n_windows):
        eigs = eigenvalues[w]
        diffs = np.abs(eigs[:, None] - eigs[None, :])
        np.fill_diagonal(diffs, np.inf)
        nn_dists = np.min(diffs, axis=1)
        min_gap[w] = float(np.min(nn_dists))
        median_nn_gap[w] = float(np.median(nn_dists))
        p10_nn_gap[w] = float(np.percentile(nn_dists, 10))

    return {
        "min_gap": min_gap,
        "median_nn_gap": median_nn_gap,
        "p10_nn_gap": p10_nn_gap,
    }
