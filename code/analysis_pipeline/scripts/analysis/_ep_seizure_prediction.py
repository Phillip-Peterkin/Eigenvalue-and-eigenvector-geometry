"""Seizure Prediction via Eigenvalue Spacing Dynamics (CHB-MIT).

Tests whether pre-ictal dynamics show eigenvalue spacing narrowing
(approaching exceptional-point geometry), seizure onset corresponds
to a spacing minimum, and post-ictal recovery shows spacing widening.

Dataset: BIDS CHB-MIT Scalp EEG Database — 23 pediatric subjects,
~198 annotated seizures, 18 bipolar channels (double banana montage),
256 Hz.

Preprocessing: Bad channel detection -> notch 60 Hz -> bandpass
0.5-45 Hz -> PCA (18 -> 15 components, fitted on interictal baseline).
No CSD (bipolar montage).

Primary analysis:
1. Sliding-window VAR(1) Jacobian -> eigenvalue spacing trajectory
2. Align to seizure onset (t=0), z-score to interictal baseline
3. Test: is mean z-scored spacing at -5, -2, -1 min below zero?
4. Cluster-based permutation test across pre-ictal time axis
5. Sham-onset null distribution
6. Phase-randomized surrogate control
7. Alpha/delta spectral power partial correlation

Scientific framing: Within-subject, within-seizure analysis.
Cannot compare to healthy controls (epileptic baseline != normal).
All comparisons are relative to each patient's own interictal baseline.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

import yaml
from cmcc.io.loader_chbmit import (
    build_seizure_catalog,
    catalog_summary,
    load_raw_edf,
)
from cmcc.preprocess.seizure_eeg import (
    fit_baseline_pca,
    preprocess_chbmit_raw,
    project_to_pca,
    reject_artifact_windows,
)
from cmcc.analysis.seizure_dynamics import (
    SeizureTrajectory,
    compute_preictal_slope,
    compute_seizure_trajectory,
    compute_sham_trajectories,
    compute_surrogate_baseline,
    partial_correlation_control,
    phase_randomize_surrogate,
)
from cmcc.provenance import log_run, save_summary_json

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "chbmit.yaml"


def log(msg):
    print(msg, flush=True)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def process_single_seizure(seizure_event, cfg, rng):
    """Load, preprocess, and compute trajectory for one seizure."""
    pp = cfg["preprocessing"]
    var = cfg["var"]
    sm = cfg["smoothing"]
    art = cfg["artifact"]
    sz_cfg = cfg["seizure"]

    try:
        raw = load_raw_edf(
            cfg["data"]["root"],
            seizure_event.subject_id,
            seizure_event.session,
            seizure_event.run,
            preload=True,
        )
    except Exception as e:
        log(f"      SKIP: {e}")
        return None

    try:
        data, sfreq, preproc_info = preprocess_chbmit_raw(
            raw,
            line_freq=pp["line_freq"],
            bandpass=tuple(pp["bandpass"]),
        )
    except Exception as e:
        log(f"      SKIP preprocessing: {e}")
        return None

    del raw
    gc.collect()

    bl_window = sz_cfg["baseline_window"]
    baseline_start = seizure_event.onset_sec + bl_window[0]
    baseline_end = seizure_event.onset_sec + bl_window[1]
    baseline_start = max(0.0, baseline_start)

    if baseline_end <= baseline_start + 60:
        log(f"      SKIP: insufficient baseline")
        return None

    try:
        pca, pca_info = fit_baseline_pca(
            data, sfreq,
            baseline_start_sec=baseline_start,
            baseline_end_sec=baseline_end,
            n_components=pp["n_components"],
        )
    except Exception as e:
        log(f"      SKIP PCA: {e}")
        return None

    data_pca = project_to_pca(data, pca)
    del data
    gc.collect()

    artifact_mask = reject_artifact_windows(
        data_pca, sfreq,
        window_sec=var["window_sec"],
        step_sec=var["step_sec"],
        variance_threshold_sd=art["variance_threshold_sd"],
        kurtosis_threshold=art["kurtosis_threshold"],
    )

    try:
        traj = compute_seizure_trajectory(
            data_pca, sfreq,
            seizure_onset_sec=seizure_event.onset_sec,
            seizure_offset_sec=seizure_event.offset_sec,
            baseline_start_sec=baseline_start,
            baseline_end_sec=baseline_end,
            window_sec=var["window_sec"],
            step_sec=var["step_sec"],
            regularization=var["regularization"],
            smoothing_sec=sm["moving_average_sec"],
            subject_id=seizure_event.subject_id,
            seizure_idx=0,
            seizure_duration=seizure_event.duration_sec,
            event_type=seizure_event.event_type,
            artifact_mask=artifact_mask,
        )
    except Exception as e:
        log(f"      SKIP trajectory: {e}")
        return None

    del data_pca
    gc.collect()

    return traj


def interpolate_to_common_time(trajectories, time_grid):
    """Interpolate trajectories onto a common time grid."""
    n_grid = len(time_grid)
    fields = [
        "min_spacing_z", "median_nns_z", "p10_nns_z",
        "spectral_radius_z", "ep_score_z",
        "alpha_power_z", "delta_power_z",
    ]

    interpolated = {f: [] for f in fields}

    for traj in trajectories:
        for f in fields:
            values = getattr(traj, f)
            interp = np.interp(time_grid, traj.time_sec, values,
                               left=np.nan, right=np.nan)
            interpolated[f].append(interp)

    result = {}
    for f in fields:
        arr = np.array(interpolated[f])
        result[f] = arr

    return result


def group_statistics(subject_means, time_grid, cfg):
    """Compute group-level statistics on subject-mean trajectories."""
    stats_cfg = cfg["statistics"]

    n_subjects, n_time = subject_means.shape
    if n_subjects < 3:
        return {"error": "insufficient subjects"}

    results = {}

    for t_min in [-5, -2, -1]:
        t_sec = t_min * 60.0
        idx = np.argmin(np.abs(time_grid - t_sec))
        values = subject_means[:, idx]
        valid = values[np.isfinite(values)]
        if len(valid) < 3:
            results[f"t{t_min}min"] = {"n": len(valid), "error": "insufficient"}
            continue

        t_stat, p_val = sp_stats.ttest_1samp(valid, 0.0)
        d = float(np.mean(valid) / np.std(valid)) if np.std(valid) > 0 else 0.0

        results[f"t{t_min}min"] = {
            "mean_z": float(np.mean(valid)),
            "sem": float(sp_stats.sem(valid)),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "cohens_d": d,
            "n_subjects": len(valid),
        }

    try:
        import statsmodels.api as sm
        from statsmodels.regression.mixed_linear_model import MixedLM

        preictal_mask = (time_grid >= -600) & (time_grid < 0)
        preictal_idx = np.where(preictal_mask)[0]

        if len(preictal_idx) > 0:
            rows = []
            for subj_i in range(n_subjects):
                for t_idx in preictal_idx[::10]:
                    val = subject_means[subj_i, t_idx]
                    if np.isfinite(val):
                        rows.append({
                            "subject": subj_i,
                            "time": time_grid[t_idx],
                            "spacing_z": val,
                        })

            if len(rows) > 10:
                import pandas as pd
                df_lme = pd.DataFrame(rows)
                model = MixedLM.from_formula(
                    "spacing_z ~ time",
                    groups="subject",
                    data=df_lme,
                )
                fit = model.fit(reml=True, method="lbfgs")
                results["mixed_effects"] = {
                    "time_coef": float(fit.params.get("time", np.nan)),
                    "time_pvalue": float(fit.pvalues.get("time", np.nan)),
                    "intercept": float(fit.params.get("Intercept", np.nan)),
                    "n_observations": len(df_lme),
                    "n_groups": df_lme["subject"].nunique(),
                }
    except Exception as e:
        results["mixed_effects_error"] = str(e)

    return results


def _vectorized_ttest_1samp(data):
    """Vectorized one-sample t-test across columns (axis=0).

    Handles NaN by computing per-column valid counts. Returns t-statistics
    as a 1D array. Much faster than calling scipy.stats.ttest_1samp in a loop.
    """
    n_subj, n_time = data.shape
    col_mean = np.nanmean(data, axis=0)
    col_std = np.nanstd(data, axis=0, ddof=1)
    col_n = np.sum(np.isfinite(data), axis=0)
    col_n_safe = np.where(col_n >= 3, col_n, 1).astype(float)
    col_std_safe = np.where(col_std > 0, col_std, 1.0)
    t_vals = col_mean / (col_std_safe / np.sqrt(col_n_safe))
    t_vals[col_n < 3] = 0.0
    return t_vals


def cluster_permutation_test(subject_means, time_grid, n_perm=5000, alpha=0.05, seed=42):
    """Cluster-based permutation test across time axis (vectorized)."""
    rng = np.random.default_rng(seed)
    n_subjects, n_time = subject_means.shape

    preictal_mask = (time_grid >= -600) & (time_grid < 0)
    if preictal_mask.sum() == 0:
        return {"error": "no pre-ictal windows"}

    data = subject_means[:, preictal_mask]
    time_sub = time_grid[preictal_mask]

    valid_count = np.sum(np.isfinite(data), axis=0)
    enough_mask = valid_count >= 3
    if enough_mask.sum() == 0:
        return {"error": "insufficient data"}

    t_obs = _vectorized_ttest_1samp(data)

    threshold = sp_stats.t.ppf(1 - alpha / 2, df=max(1, n_subjects - 1))

    def _find_clusters(t_vals, thresh):
        sig = np.abs(t_vals) > thresh
        clusters = []
        in_cluster = False
        start = 0
        for i in range(len(sig)):
            if sig[i] and not in_cluster:
                start = i
                in_cluster = True
            elif not sig[i] and in_cluster:
                clusters.append((start, i, float(np.sum(t_vals[start:i]))))
                in_cluster = False
        if in_cluster:
            clusters.append((start, len(sig), float(np.sum(t_vals[start:]))))
        return clusters

    obs_clusters = _find_clusters(t_obs, threshold)
    if not obs_clusters:
        return {
            "n_clusters": 0,
            "time_range": [float(time_sub[0]), float(time_sub[-1])],
        }

    obs_max_stat = max(abs(c[2]) for c in obs_clusters)

    null_max_stats = np.zeros(n_perm)
    for perm in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_subjects)
        perm_data = data * signs[:, np.newaxis]
        t_perm = _vectorized_ttest_1samp(perm_data)
        perm_clusters = _find_clusters(t_perm, threshold)
        if perm_clusters:
            null_max_stats[perm] = max(abs(c[2]) for c in perm_clusters)

    p_cluster = float(np.mean(null_max_stats >= obs_max_stat))

    sig_clusters = []
    for start, end, stat in obs_clusters:
        c_p = float(np.mean(null_max_stats >= abs(stat)))
        if c_p < alpha:
            sig_clusters.append({
                "start_sec": float(time_sub[start]),
                "end_sec": float(time_sub[min(end - 1, len(time_sub) - 1)]),
                "cluster_stat": float(stat),
                "p_value": c_p,
            })

    return {
        "n_clusters": len(obs_clusters),
        "n_significant_clusters": len(sig_clusters),
        "significant_clusters": sig_clusters,
        "max_cluster_stat": float(obs_max_stat),
        "p_max_cluster": p_cluster,
        "n_perm": n_perm,
    }


def plot_grand_average(subject_means, time_grid, fig_dir, sham_means=None):
    """Plot grand-average spacing trajectory with CI shading."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    preictal_mask = (time_grid >= -1800) & (time_grid < 600)
    t = time_grid[preictal_mask] / 60.0
    data = subject_means[:, preictal_mask]

    mean_traj = np.nanmean(data, axis=0)
    sem_traj = np.nanstd(data, axis=0) / np.sqrt(np.sum(np.isfinite(data), axis=0).clip(1))

    ax.plot(t, mean_traj, "b-", lw=2, label="Real seizures")
    ax.fill_between(t, mean_traj - 1.96 * sem_traj, mean_traj + 1.96 * sem_traj,
                    alpha=0.2, color="blue")

    if sham_means is not None and sham_means.shape[0] > 0:
        sham_data = sham_means[:, preictal_mask] if sham_means.shape[1] == subject_means.shape[1] else None
        if sham_data is not None:
            sham_mean = np.nanmean(sham_data, axis=0)
            sham_sem = np.nanstd(sham_data, axis=0) / np.sqrt(
                np.sum(np.isfinite(sham_data), axis=0).clip(1))
            ax.plot(t, sham_mean, "gray", lw=2, ls="--", label="Sham onsets")
            ax.fill_between(t, sham_mean - 1.96 * sham_sem, sham_mean + 1.96 * sham_sem,
                            alpha=0.15, color="gray")

    ax.axvline(0, color="red", ls="--", lw=1.5, label="Seizure onset")
    ax.axhline(0, color="black", ls=":", lw=0.8)
    ax.axvspan(-30, -10, alpha=0.05, color="green")

    ax.set_xlabel("Time relative to seizure onset (min)")
    ax.set_ylabel("Minimum eigenvalue spacing (z-score)")
    ax.set_title("Grand-Average Pre-Ictal Eigenvalue Spacing Trajectory")
    ax.legend(loc="upper right")
    ax.set_xlim(-30, 10)

    fig.tight_layout()
    fig.savefig(fig_dir / "grand_average_trajectory.png", dpi=200)
    fig.savefig(fig_dir / "grand_average_trajectory.pdf")
    plt.close(fig)


def plot_per_subject_heatmap(subject_means, subject_ids, time_grid, fig_dir):
    """Plot per-subject spacing heatmap."""
    preictal_mask = (time_grid >= -1800) & (time_grid < 600)
    t = time_grid[preictal_mask] / 60.0
    data = subject_means[:, preictal_mask]

    fig, ax = plt.subplots(1, 1, figsize=(14, max(4, len(subject_ids) * 0.4)))
    vmax = np.nanpercentile(np.abs(data), 95) if data.size > 0 else 2.0
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[t[0], t[-1], len(subject_ids) - 0.5, -0.5])
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Time relative to onset (min)")
    ax.set_ylabel("Subject")
    ax.set_yticks(range(len(subject_ids)))
    ax.set_yticklabels(subject_ids, fontsize=7)
    ax.set_title("Per-Subject Pre-Ictal Spacing (z-score)")
    plt.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(fig_dir / "per_subject_heatmap.png", dpi=200)
    plt.close(fig)


def plot_slope_histogram(slopes, fig_dir):
    """Plot histogram of per-seizure pre-ictal slopes."""
    slopes = np.array([s for s in slopes if np.isfinite(s)])
    if len(slopes) == 0:
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(slopes, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", ls="--", lw=1.5)
    pct_negative = float(np.mean(slopes < 0) * 100)
    ax.set_xlabel("Pre-ictal spacing slope (z-score / sec)")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Seizure Pre-Ictal Slopes ({pct_negative:.0f}% negative)")
    fig.tight_layout()
    fig.savefig(fig_dir / "preictal_slope_histogram.png", dpi=200)
    plt.close(fig)


def plot_dose_response(slopes, durations, fig_dir):
    """Plot dose-response: slope vs seizure duration."""
    valid = np.isfinite(slopes) & np.isfinite(durations) & (durations > 0)
    if valid.sum() < 5:
        return

    s = slopes[valid]
    d = durations[valid]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.scatter(d, s, alpha=0.5, edgecolors="black", linewidths=0.5)
    r, p = sp_stats.pearsonr(d, s)
    ax.set_xlabel("Seizure duration (sec)")
    ax.set_ylabel("Pre-ictal spacing slope (z-score / sec)")
    ax.set_title(f"Dose-Response: Slope vs Duration (r={r:.3f}, p={p:.3f})")
    ax.axhline(0, color="red", ls=":", lw=0.8)

    if len(d) > 2:
        z = np.polyfit(d, s, 1)
        d_range = np.linspace(d.min(), d.max(), 100)
        ax.plot(d_range, np.polyval(z, d_range), "r-", lw=1.5, alpha=0.7)

    fig.tight_layout()
    fig.savefig(fig_dir / "dose_response.png", dpi=200)
    plt.close(fig)


def plot_seizure_level_heatmap(all_trajectories, time_grid, fig_dir):
    """Plot per-seizure spacing heatmap showing individual trajectory heterogeneity."""
    if not all_trajectories:
        return

    n_grid = len(time_grid)
    n_sz = len(all_trajectories)
    data = np.full((n_sz, n_grid), np.nan)
    labels = []

    for i, traj in enumerate(all_trajectories):
        interp = np.interp(time_grid, traj.time_sec, traj.min_spacing_z,
                           left=np.nan, right=np.nan)
        data[i] = interp
        labels.append(f"{traj.subject_id}_sz{traj.seizure_idx}")

    preictal_mask = (time_grid >= -1800) & (time_grid < 600)
    t = time_grid[preictal_mask] / 60.0
    data_sub = data[:, preictal_mask]

    fig, ax = plt.subplots(1, 1, figsize=(14, max(4, n_sz * 0.15)))
    vmax = np.nanpercentile(np.abs(data_sub), 95) if data_sub.size > 0 else 2.0
    vmax = max(vmax, 0.5)
    im = ax.imshow(data_sub, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[t[0], t[-1], n_sz - 0.5, -0.5])
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_xlabel("Time relative to onset (min)")
    ax.set_ylabel("Seizure")
    ax.set_title(f"Per-Seizure Spacing Trajectories (n={n_sz})")
    plt.colorbar(im, ax=ax, label="z-score")
    fig.tight_layout()
    fig.savefig(fig_dir / "seizure_level_heatmap.png", dpi=200)
    plt.close(fig)


def compute_ictal_minimum_timing(trajectories, time_grid):
    """Find time of minimum spacing relative to onset for each seizure.

    Returns dict with per-seizure min-timing, and group statistics testing
    whether the minimum clusters near onset (t=0).
    """
    min_times = []
    for traj in trajectories:
        analysis_mask = (traj.time_sec >= -600) & (traj.time_sec <= 300)
        if analysis_mask.sum() < 10:
            min_times.append(float("nan"))
            continue
        t_sub = traj.time_sec[analysis_mask]
        s_sub = traj.min_spacing_z[analysis_mask]
        valid = np.isfinite(s_sub)
        if valid.sum() < 5:
            min_times.append(float("nan"))
            continue
        idx = np.nanargmin(s_sub)
        min_times.append(float(t_sub[idx]))

    min_times_arr = np.array(min_times)
    finite = min_times_arr[np.isfinite(min_times_arr)]

    result = {
        "per_seizure_min_time_sec": [float(x) for x in min_times],
        "n_seizures": len(finite),
    }

    if len(finite) >= 3:
        result["mean_min_time_sec"] = float(np.mean(finite))
        result["median_min_time_sec"] = float(np.median(finite))
        result["std_min_time_sec"] = float(np.std(finite))
        t_stat, p_val = sp_stats.ttest_1samp(finite, 0.0)
        result["t_stat_vs_zero"] = float(t_stat)
        result["p_value_vs_zero"] = float(p_val)
        pct_before_onset = float(np.mean(finite < 0) * 100)
        pct_within_60s = float(np.mean(np.abs(finite) < 60) * 100)
        result["pct_before_onset"] = pct_before_onset
        result["pct_within_60s_of_onset"] = pct_within_60s

    return result


def compute_postictal_recovery(trajectories):
    """Compute post-ictal recovery slope and pre-ictal vs post-ictal contrast.

    Tests whether spacing widens after seizure termination.
    """
    postictal_slopes = []
    preictal_means = []
    postictal_means = []

    for traj in trajectories:
        post_mask = traj.period_label == "post_ictal"
        pre_mask = traj.period_label == "pre_ictal"

        if post_mask.sum() >= 10:
            t_post = traj.time_sec[post_mask]
            s_post = traj.min_spacing_z[post_mask]
            valid = np.isfinite(s_post)
            if valid.sum() >= 5:
                coeffs = np.polyfit(t_post[valid], s_post[valid], 1)
                postictal_slopes.append(float(coeffs[0]))

        if post_mask.sum() >= 5:
            postictal_means.append(float(np.nanmean(traj.min_spacing_z[post_mask])))
        if pre_mask.sum() >= 5:
            preictal_means.append(float(np.nanmean(traj.min_spacing_z[pre_mask])))

    slopes_arr = np.array(postictal_slopes)
    finite_slopes = slopes_arr[np.isfinite(slopes_arr)]

    result = {
        "n_seizures_with_postictal": len(finite_slopes),
    }

    if len(finite_slopes) >= 3:
        result["mean_postictal_slope"] = float(np.mean(finite_slopes))
        result["pct_positive_slope"] = float(np.mean(finite_slopes > 0) * 100)
        t_stat, p_val = sp_stats.ttest_1samp(finite_slopes, 0.0)
        result["t_stat_slope_vs_zero"] = float(t_stat)
        result["p_value_slope_vs_zero"] = float(p_val)

    pre_arr = np.array(preictal_means)
    post_arr = np.array(postictal_means)
    n_paired = min(len(pre_arr), len(post_arr))
    if n_paired >= 3:
        pre_p = pre_arr[:n_paired]
        post_p = post_arr[:n_paired]
        valid = np.isfinite(pre_p) & np.isfinite(post_p)
        if valid.sum() >= 3:
            t_stat, p_val = sp_stats.ttest_rel(post_p[valid], pre_p[valid])
            result["pre_vs_post_contrast"] = {
                "mean_preictal_z": float(np.mean(pre_p[valid])),
                "mean_postictal_z": float(np.mean(post_p[valid])),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "n_pairs": int(valid.sum()),
                "direction": "recovery" if np.mean(post_p[valid]) > np.mean(pre_p[valid]) else "no_recovery",
            }

    return result


def run_multi_metric_statistics(subject_trajectories, time_grid, cfg):
    """Run group statistics on all spacing and dynamical metrics for robustness.

    Returns dict with results for min_spacing_z, median_nns_z, p10_nns_z,
    spectral_radius_z, and ep_score_z.
    """
    metrics = ["min_spacing_z", "median_nns_z", "p10_nns_z", "spectral_radius_z", "ep_score_z"]
    seed = cfg["random_seed"]
    results = {}

    for metric in metrics:
        subject_means_list = []
        for sid, trajs in sorted(subject_trajectories.items()):
            interp = interpolate_to_common_time(trajs, time_grid)
            subj_mean = np.nanmean(interp[metric], axis=0)
            subject_means_list.append(subj_mean)

        if not subject_means_list:
            results[metric] = {"error": "no data"}
            continue

        subject_means = np.array(subject_means_list)
        gs = group_statistics(subject_means, time_grid, cfg)
        cp = cluster_permutation_test(
            subject_means, time_grid,
            n_perm=cfg["statistics"]["n_perm"],
            alpha=cfg["statistics"]["alpha"],
            seed=seed,
        )
        results[metric] = {
            "group_statistics": gs,
            "cluster_permutation": cp,
        }

    return results


def main():
    t0 = time.time()
    log("=" * 70)
    log("SEIZURE PREDICTION VIA EIGENVALUE SPACING (CHB-MIT)")
    log("=" * 70)

    cfg = load_config()
    seed = cfg["random_seed"]
    rng = np.random.default_rng(seed)

    results_dir = Path(cfg["output"]["results_dir"]) / "analysis"
    fig_dir = Path(cfg["output"]["results_dir"]) / "figures" / "ep_seizure"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log("\n1. Building seizure catalog...")
    sz_cfg = cfg["seizure"]
    catalogs = build_seizure_catalog(
        cfg["data"]["root"],
        min_preictal_sec=sz_cfg["min_preictal_sec"],
        min_inter_seizure_sec=sz_cfg["min_inter_seizure_sec"],
    )
    summary = catalog_summary(catalogs)
    log(f"   Subjects: {summary['n_subjects']}")
    log(f"   Total seizures: {summary['total_seizures']}")
    log(f"   Eligible seizures: {summary['total_eligible_seizures']}")
    log(f"   Subjects with eligible: {summary['subjects_with_eligible_seizures']}")
    log(f"   Total recording hours: {summary['total_recording_hours']}")

    save_summary_json(summary, results_dir, "seizure_catalog_summary.json")

    log("\n2. Processing seizures per subject...")
    all_trajectories: list[SeizureTrajectory] = []
    subject_trajectories: dict[str, list[SeizureTrajectory]] = {}
    all_slopes: list[float] = []
    all_durations: list[float] = []
    for sid, cat in sorted(catalogs.items()):
        if cat.n_eligible == 0:
            continue

        log(f"\n  {sid}: {cat.n_eligible} eligible seizures")
        subj_trajs: list[SeizureTrajectory] = []

        for i, sz in enumerate(cat.eligible_seizures):
            log(f"    Seizure {i+1}/{cat.n_eligible}: "
                f"onset={sz.onset_sec:.0f}s, dur={sz.duration_sec:.0f}s, "
                f"type={sz.event_type}")

            traj = process_single_seizure(sz, cfg, rng)
            if traj is not None:
                traj.seizure_idx = i
                subj_trajs.append(traj)
                all_trajectories.append(traj)
                all_slopes.append(traj.preictal_slope)
                all_durations.append(traj.seizure_duration)
                log(f"      OK: {len(traj.time_sec)} windows, "
                    f"slope={traj.preictal_slope:.6f}")

        if subj_trajs:
            subject_trajectories[sid] = subj_trajs

    log(f"\n   Total trajectories: {len(all_trajectories)}")
    log(f"   Subjects with data: {len(subject_trajectories)}")

    if len(all_trajectories) == 0:
        log("\nERROR: No trajectories computed. Exiting.")
        return

    log("\n3. Building common time grid and interpolating...")
    time_grid = np.arange(-1800, 600, cfg["var"]["step_sec"])

    subject_means_list = []
    subject_ids_list = []

    for sid, trajs in sorted(subject_trajectories.items()):
        interp = interpolate_to_common_time(trajs, time_grid)
        subj_mean = np.nanmean(interp["min_spacing_z"], axis=0)
        subject_means_list.append(subj_mean)
        subject_ids_list.append(sid)

    subject_means = np.array(subject_means_list)
    log(f"   Subject means shape: {subject_means.shape}")

    log("\n4. Group-level statistics (min_spacing_z)...")
    group_stats = group_statistics(subject_means, time_grid, cfg)
    for key, val in group_stats.items():
        if isinstance(val, dict) and "mean_z" in val:
            log(f"   {key}: mean_z={val['mean_z']:.4f}, p={val['p_value']:.4f}, "
                f"d={val['cohens_d']:.3f}, n={val['n_subjects']}")

    log("\n5. Cluster-based permutation test...")
    cluster_results = cluster_permutation_test(
        subject_means, time_grid,
        n_perm=cfg["statistics"]["n_perm"],
        alpha=cfg["statistics"]["alpha"],
        seed=seed,
    )
    log(f"   Clusters found: {cluster_results.get('n_clusters', 0)}")
    log(f"   Significant clusters: {cluster_results.get('n_significant_clusters', 0)}")
    if "significant_clusters" in cluster_results:
        for cl in cluster_results["significant_clusters"]:
            log(f"     [{cl['start_sec']/60:.1f} to {cl['end_sec']/60:.1f} min] "
                f"p={cl['p_value']:.4f}")

    log("\n5b. Multi-metric robustness (median_nns_z, p10_nns_z)...")
    multi_metric_results = run_multi_metric_statistics(
        subject_trajectories, time_grid, cfg,
    )
    for metric, mres in multi_metric_results.items():
        gs = mres.get("group_statistics", {})
        for key, val in gs.items():
            if isinstance(val, dict) and "mean_z" in val:
                log(f"   {metric} {key}: mean_z={val['mean_z']:.4f}, "
                    f"p={val['p_value']:.4f}")

    log("\n6. Partial correlation controls...")
    partial_results = {}
    for sid, trajs in subject_trajectories.items():
        for traj in trajs:
            preictal = traj.period_label == "pre_ictal"
            if preictal.sum() > 10:
                pc = partial_correlation_control(
                    traj.time_sec, traj.min_spacing_z,
                    traj.alpha_power_z, traj.delta_power_z,
                    preictal_mask=preictal,
                )
                partial_results[f"{sid}_sz{traj.seizure_idx}"] = pc

    r_raw_vals: list[float] = []
    r_part_vals: list[float] = []
    if partial_results:
        r_raw_vals = [v["r_raw"] for v in partial_results.values() if np.isfinite(v["r_raw"])]
        r_part_vals = [v["r_partial"] for v in partial_results.values() if np.isfinite(v["r_partial"])]
        log(f"   Mean r_raw: {np.mean(r_raw_vals):.4f} (n={len(r_raw_vals)})")
        log(f"   Mean r_partial: {np.mean(r_part_vals):.4f} (n={len(r_part_vals)})")

    log("\n6b. Ictal minimum timing...")
    ictal_min_results = compute_ictal_minimum_timing(all_trajectories, time_grid)
    if "mean_min_time_sec" in ictal_min_results:
        log(f"   Mean minimum time: {ictal_min_results['mean_min_time_sec']:.1f}s "
            f"(median: {ictal_min_results['median_min_time_sec']:.1f}s)")
        log(f"   % before onset: {ictal_min_results.get('pct_before_onset', 0):.1f}%")
        log(f"   % within 60s of onset: {ictal_min_results.get('pct_within_60s_of_onset', 0):.1f}%")

    log("\n6c. Post-ictal recovery...")
    postictal_results = compute_postictal_recovery(all_trajectories)
    if "mean_postictal_slope" in postictal_results:
        log(f"   Mean post-ictal slope: {postictal_results['mean_postictal_slope']:.6f}")
        log(f"   % positive (recovery): {postictal_results['pct_positive_slope']:.1f}%")
        log(f"   p-value slope > 0: {postictal_results.get('p_value_slope_vs_zero', float('nan')):.4f}")
    if "pre_vs_post_contrast" in postictal_results:
        c = postictal_results["pre_vs_post_contrast"]
        log(f"   Pre vs post contrast: pre={c['mean_preictal_z']:.3f}, "
            f"post={c['mean_postictal_z']:.3f}, p={c['p_value']:.4f}, "
            f"direction={c['direction']}")

    log("\n7. Sham-onset null distribution...")
    stats_cfg = cfg["statistics"]
    n_sham = stats_cfg.get("n_sham_per_subject", 20)
    sham_all: list[SeizureTrajectory] = []

    for sid, cat in sorted(catalogs.items()):
        if sid not in subject_trajectories:
            continue

        real_onsets = [sz.onset_sec for sz in cat.eligible_seizures]

        for sz in cat.eligible_seizures[:1]:
            try:
                raw = load_raw_edf(
                    cfg["data"]["root"], sz.subject_id,
                    sz.session, sz.run, preload=True,
                )
                data, sfreq, _ = preprocess_chbmit_raw(
                    raw,
                    line_freq=cfg["preprocessing"]["line_freq"],
                    bandpass=tuple(cfg["preprocessing"]["bandpass"]),
                )
                del raw

                bl_window = sz_cfg["baseline_window"]
                bl_start = max(0.0, sz.onset_sec + bl_window[0])
                bl_end = sz.onset_sec + bl_window[1]
                if bl_end > bl_start + 60:
                    pca, _ = fit_baseline_pca(
                        data, sfreq, bl_start, bl_end,
                        n_components=cfg["preprocessing"]["n_components"],
                    )
                    data_pca = project_to_pca(data, pca)
                    del data

                    shams = compute_sham_trajectories(
                        data_pca, sfreq,
                        n_shams=n_sham,
                        seizure_onsets_sec=real_onsets,
                        recording_duration_sec=data_pca.shape[1] / sfreq,
                        rng=rng,
                        window_sec=cfg["var"]["window_sec"],
                        step_sec=cfg["var"]["step_sec"],
                        regularization=cfg["var"]["regularization"],
                        smoothing_sec=cfg["smoothing"]["moving_average_sec"],
                    )
                    sham_all.extend(shams)
                    log(f"   {sid}: {len(shams)} sham trajectories")
                    del data_pca
                else:
                    del data
            except Exception as e:
                log(f"   {sid}: sham SKIP — {e}")

            gc.collect()
            break

    log(f"   Total sham trajectories: {len(sham_all)}")

    sham_means = None
    if sham_all:
        sham_interp_list = []
        for traj in sham_all:
            interp = np.interp(time_grid, traj.time_sec, traj.min_spacing_z,
                               left=np.nan, right=np.nan)
            sham_interp_list.append(interp)
        sham_means = np.array(sham_interp_list)

    log("\n8. Phase-randomized surrogate control...")
    n_surr = stats_cfg.get("n_surrogates", 10)
    n_surr_subj = stats_cfg.get("surrogate_subjects", 5)
    surrogate_results_list = []
    surr_count = 0

    for sid, cat in sorted(catalogs.items()):
        if sid not in subject_trajectories:
            continue
        if surr_count >= n_surr_subj:
            break

        for sz in cat.eligible_seizures[:1]:
            try:
                raw = load_raw_edf(
                    cfg["data"]["root"], sz.subject_id,
                    sz.session, sz.run, preload=True,
                )
                data, sfreq, _ = preprocess_chbmit_raw(
                    raw,
                    line_freq=cfg["preprocessing"]["line_freq"],
                    bandpass=tuple(cfg["preprocessing"]["bandpass"]),
                )
                del raw

                bl_window = sz_cfg["baseline_window"]
                bl_start = max(0.0, sz.onset_sec + bl_window[0])
                bl_end = sz.onset_sec + bl_window[1]
                if bl_end > bl_start + 60:
                    pca, _ = fit_baseline_pca(
                        data, sfreq, bl_start, bl_end,
                        n_components=cfg["preprocessing"]["n_components"],
                    )
                    data_pca = project_to_pca(data, pca)
                    del data

                    bl_start_samp = int(bl_start * sfreq)
                    bl_end_samp = int(bl_end * sfreq)
                    interictal_seg = data_pca[:, bl_start_samp:bl_end_samp]

                    if interictal_seg.shape[1] > 500:
                        surr_res = compute_surrogate_baseline(
                            interictal_seg, sfreq,
                            n_surrogates=n_surr,
                            rng=rng,
                            window_sec=cfg["var"]["window_sec"],
                            step_sec=cfg["var"]["step_sec"],
                            regularization=cfg["var"]["regularization"],
                        )
                        surr_res["subject_id"] = sid
                        surrogate_results_list.append(surr_res)
                        log(f"   {sid}: real_mean={surr_res['real_mean_spacing']:.6f}, "
                            f"surr_mean={np.mean(surr_res['surrogate_mean_spacings']):.6f}")
                        surr_count += 1
                    del data_pca
                else:
                    del data
            except Exception as e:
                log(f"   {sid}: surrogate SKIP — {e}")

            gc.collect()
            break

    log(f"   Subjects with surrogate data: {len(surrogate_results_list)}")

    log("\n9. Heterogeneity analysis...")
    slopes_arr = np.array(all_slopes)
    durations_arr = np.array(all_durations)
    finite_slopes = slopes_arr[np.isfinite(slopes_arr)]
    if len(finite_slopes) > 0:
        pct_neg = float(np.mean(finite_slopes < 0) * 100)
        log(f"   Seizures with negative slope: {pct_neg:.1f}%")
        log(f"   Mean slope: {np.mean(finite_slopes):.6f}")
        log(f"   Median slope: {np.median(finite_slopes):.6f}")

        if len(finite_slopes) >= 5:
            valid_dr = np.isfinite(slopes_arr) & np.isfinite(durations_arr) & (durations_arr > 0)
            if valid_dr.sum() >= 5:
                r_dr, p_dr = sp_stats.pearsonr(slopes_arr[valid_dr], durations_arr[valid_dr])
                log(f"   Dose-response (slope vs duration): r={r_dr:.4f}, p={p_dr:.4f}")

    log("\n10. Generating figures...")
    plot_grand_average(subject_means, time_grid, fig_dir, sham_means=sham_means)
    plot_per_subject_heatmap(subject_means, subject_ids_list, time_grid, fig_dir)
    plot_seizure_level_heatmap(all_trajectories, time_grid, fig_dir)
    plot_slope_histogram(slopes_arr, fig_dir)
    plot_dose_response(slopes_arr, durations_arr, fig_dir)

    log("\n11. Saving results...")
    final_results = {
        "catalog_summary": summary,
        "n_trajectories": len(all_trajectories),
        "n_subjects_analyzed": len(subject_trajectories),
        "group_statistics": group_stats,
        "cluster_permutation": cluster_results,
        "multi_metric_robustness": multi_metric_results,
        "ictal_minimum_timing": ictal_min_results,
        "postictal_recovery": postictal_results,
        "sham_onset_null": {
            "n_sham_trajectories": len(sham_all),
            "n_subjects_with_shams": sum(
                1 for s in sham_all if s.subject_id != "sham"
            ) if sham_all else 0,
        },
        "surrogate_control": {
            "n_subjects": len(surrogate_results_list),
            "per_subject": [
                {
                    "subject_id": r.get("subject_id", ""),
                    "real_mean_spacing": r["real_mean_spacing"],
                    "surrogate_mean_spacing": float(np.mean(r["surrogate_mean_spacings"]))
                    if r["surrogate_mean_spacings"] else float("nan"),
                    "real_exceeds_surrogate": r["real_mean_spacing"] > float(np.mean(r["surrogate_mean_spacings"]))
                    if r["surrogate_mean_spacings"] else False,
                }
                for r in surrogate_results_list
            ],
        },
        "heterogeneity": {
            "n_seizures": len(finite_slopes),
            "pct_negative_slope": float(np.mean(finite_slopes < 0) * 100) if len(finite_slopes) > 0 else 0,
            "mean_slope": float(np.mean(finite_slopes)) if len(finite_slopes) > 0 else float("nan"),
            "median_slope": float(np.median(finite_slopes)) if len(finite_slopes) > 0 else float("nan"),
        },
        "partial_correlation_summary": {
            "n_seizures_tested": len(partial_results),
            "mean_r_raw": float(np.mean(r_raw_vals)) if r_raw_vals else float("nan"),
            "mean_r_partial": float(np.mean(r_part_vals)) if r_part_vals else float("nan"),
        },
        "config": cfg,
    }

    save_summary_json(final_results, results_dir, "seizure_prediction_results.json")
    log_run(cfg, results_dir)

    elapsed = time.time() - t0
    log(f"\n{'=' * 70}")
    log(f"COMPLETE in {elapsed/60:.1f} min")
    log(f"Results: {results_dir}")
    log(f"Figures: {fig_dir}")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
