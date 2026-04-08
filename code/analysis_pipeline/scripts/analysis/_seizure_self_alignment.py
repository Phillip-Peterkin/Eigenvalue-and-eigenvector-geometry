"""Self-alignment analysis: align seizures to their own minimum spacing.

Instead of aligning all seizures to clinical onset (t=0), aligns each
seizure to its own minimum eigenvalue spacing timepoint. If pre-ictal
effects occur at variable lead times across seizures, fixed-onset
alignment washes them out. Self-alignment can reveal signals hidden
by timing jitter.

Loads trajectory cache from _seizure_stratification.py output.
"""
from __future__ import annotations

import json
import sys
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

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "chbmit.yaml"


def log(msg):
    print(msg, flush=True)


def load_trajectory_cache(cache_path):
    data = np.load(cache_path, allow_pickle=True)
    trajectories = {}
    for key in data.files:
        traj = json.loads(str(data[key]))
        for field in ["time_sec", "min_spacing_z", "min_spacing_raw",
                      "spectral_radius_z", "ep_score_z", "alpha_power_z",
                      "delta_power_z", "median_nns_z", "p10_nns_z"]:
            if field in traj:
                traj[field] = np.array(traj[field])
        trajectories[key] = traj
    return trajectories


def find_minimum_spacing_time(traj, search_window=(-600, 300)):
    t = traj["time_sec"]
    s = traj["min_spacing_z"]
    mask = (t >= search_window[0]) & (t <= search_window[1])
    if mask.sum() < 5:
        return float("nan")
    s_sub = s[mask]
    valid = np.isfinite(s_sub)
    if valid.sum() < 3:
        return float("nan")
    idx = np.nanargmin(s_sub)
    return float(t[mask][idx])


def self_align_trajectories(trajectories, align_window_sec=(-600, 300)):
    aligned = {}
    min_times = {}

    for key, traj in trajectories.items():
        min_t = find_minimum_spacing_time(traj, search_window=align_window_sec)
        if np.isnan(min_t):
            continue
        min_times[key] = min_t
        aligned_t = traj["time_sec"] - min_t
        aligned[key] = {
            "time_sec_original": traj["time_sec"],
            "time_sec_aligned": aligned_t,
            "min_spacing_z": traj["min_spacing_z"],
            "spectral_radius_z": traj["spectral_radius_z"],
            "alpha_power_z": traj["alpha_power_z"],
            "delta_power_z": traj["delta_power_z"],
            "subject_id": traj["subject_id"],
            "min_time_original": min_t,
        }

    return aligned, min_times


def compute_aligned_average(aligned_trajs, time_grid):
    n_grid = len(time_grid)
    fields = ["min_spacing_z", "spectral_radius_z", "alpha_power_z", "delta_power_z"]
    all_interp = {f: [] for f in fields}

    for key, traj in aligned_trajs.items():
        for f in fields:
            interp = np.interp(time_grid, traj["time_sec_aligned"], traj[f],
                               left=np.nan, right=np.nan)
            all_interp[f].append(interp)

    results = {}
    for f in fields:
        arr = np.array(all_interp[f])
        results[f] = {
            "mean": np.nanmean(arr, axis=0),
            "sem": np.nanstd(arr, axis=0) / np.sqrt(np.sum(np.isfinite(arr), axis=0).clip(1)),
            "n_valid": np.sum(np.isfinite(arr), axis=0),
        }

    return results


def compute_onset_aligned_average(trajectories, time_grid):
    fields = ["min_spacing_z", "spectral_radius_z", "alpha_power_z", "delta_power_z"]
    all_interp = {f: [] for f in fields}

    for key, traj in trajectories.items():
        for f in fields:
            interp = np.interp(time_grid, traj["time_sec"], traj[f],
                               left=np.nan, right=np.nan)
            all_interp[f].append(interp)

    results = {}
    for f in fields:
        arr = np.array(all_interp[f])
        results[f] = {
            "mean": np.nanmean(arr, axis=0),
            "sem": np.nanstd(arr, axis=0) / np.sqrt(np.sum(np.isfinite(arr), axis=0).clip(1)),
            "n_valid": np.sum(np.isfinite(arr), axis=0),
        }

    return results


def plot_alignment_comparison(onset_avg, self_avg, onset_grid, self_grid, fig_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    fields = [
        ("min_spacing_z", "Minimum Eigenvalue Spacing"),
        ("spectral_radius_z", "Spectral Radius"),
        ("alpha_power_z", "Alpha Power (8-12 Hz)"),
        ("delta_power_z", "Delta Power (1-4 Hz)"),
    ]

    for ax, (field, title) in zip(axes.flatten(), fields):
        t_onset = onset_grid / 60.0
        m_onset = onset_avg[field]["mean"]
        s_onset = onset_avg[field]["sem"]
        ax.plot(t_onset, m_onset, "b-", lw=2, label="Onset-aligned")
        ax.fill_between(t_onset, m_onset - 1.96 * s_onset, m_onset + 1.96 * s_onset,
                        alpha=0.15, color="blue")

        t_self = self_grid / 60.0
        m_self = self_avg[field]["mean"]
        s_self = self_avg[field]["sem"]
        ax.plot(t_self, m_self, "r-", lw=2, label="Self-aligned (min spacing)")
        ax.fill_between(t_self, m_self - 1.96 * s_self, m_self + 1.96 * s_self,
                        alpha=0.15, color="red")

        ax.axvline(0, color="black", ls="--", lw=1, alpha=0.7)
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("z-score")
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle("Onset-Aligned vs Self-Aligned Grand Averages", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "alignment_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "alignment_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_min_time_distribution(min_times, fig_dir):
    vals = [v for v in min_times.values() if np.isfinite(v)]
    if not vals:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(np.array(vals) / 60.0, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", ls="--", lw=2, label="Seizure onset")
    ax.set_xlabel("Time of minimum spacing relative to onset (min)")
    ax.set_ylabel("Count")
    pct_before = float(np.mean(np.array(vals) < 0) * 100)
    ax.set_title(f"Distribution of Minimum Spacing Timing ({pct_before:.0f}% pre-ictal)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "min_time_distribution.png", dpi=200)
    plt.close(fig)


def quantify_sharpness(avg_data, time_grid, window_min=(-5, 0)):
    t_min = time_grid / 60.0
    mask = (t_min >= window_min[0]) & (t_min <= window_min[1])
    if mask.sum() < 3:
        return float("nan"), float("nan")

    mean_vals = avg_data["min_spacing_z"]["mean"][mask]
    valid = np.isfinite(mean_vals)
    if valid.sum() < 3:
        return float("nan"), float("nan")

    t_sub = (time_grid[mask])[valid]
    y_sub = mean_vals[valid]
    slope = np.polyfit(t_sub, y_sub, 1)[0]
    min_val = float(np.nanmin(mean_vals[valid]))
    return float(slope), min_val


def main():
    log("=" * 70)
    log("SELF-ALIGNMENT ANALYSIS")
    log("=" * 70)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["output"]["results_dir"]) / "analysis"
    fig_dir = Path(cfg["output"]["results_dir"]) / "figures" / "self_alignment"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cache_path = results_dir / "trajectory_cache.npz"
    if not cache_path.exists():
        log(f"ERROR: Trajectory cache not found at {cache_path}")
        log("Run _seizure_stratification.py first.")
        return

    log("\n1. Loading trajectory cache...")
    trajectories = load_trajectory_cache(cache_path)
    log(f"   Loaded {len(trajectories)} seizure trajectories")

    log("\n2. Self-aligning trajectories...")
    aligned, min_times = self_align_trajectories(trajectories)
    log(f"   Successfully aligned: {len(aligned)} / {len(trajectories)}")

    min_vals = [v for v in min_times.values() if np.isfinite(v)]
    if min_vals:
        log(f"   Min-time range: {np.min(min_vals)/60:.1f} to {np.max(min_vals)/60:.1f} min")
        log(f"   Min-time median: {np.median(min_vals)/60:.1f} min")
        log(f"   % before onset: {np.mean(np.array(min_vals) < 0)*100:.1f}%")

    log("\n3. Computing averages...")
    onset_grid = np.arange(-1800, 600, 0.5)
    self_grid = np.arange(-600, 600, 0.5)

    onset_avg = compute_onset_aligned_average(trajectories, onset_grid)
    self_avg = compute_aligned_average(aligned, self_grid)

    log("\n4. Comparing sharpness...")
    onset_slope, onset_min = quantify_sharpness(onset_avg, onset_grid, window_min=(-5, 0))
    self_slope, self_min = quantify_sharpness(self_avg, self_grid, window_min=(-5, 0))

    log(f"   Onset-aligned: slope={onset_slope:.6f}, min={onset_min:.3f}")
    log(f"   Self-aligned:  slope={self_slope:.6f}, min={self_min:.3f}")

    if np.isfinite(onset_slope) and np.isfinite(self_slope) and onset_slope != 0:
        improvement = abs(self_slope) / abs(onset_slope)
        log(f"   Self-alignment slope magnitude ratio: {improvement:.2f}x")

    log("\n5. Generating figures...")
    plot_alignment_comparison(onset_avg, self_avg, onset_grid, self_grid, fig_dir)
    plot_min_time_distribution(min_times, fig_dir)

    log("\n6. Saving results...")
    results = {
        "n_trajectories": len(trajectories),
        "n_aligned": len(aligned),
        "min_time_stats": {
            "mean_sec": float(np.mean(min_vals)) if min_vals else None,
            "median_sec": float(np.median(min_vals)) if min_vals else None,
            "std_sec": float(np.std(min_vals)) if min_vals else None,
            "pct_before_onset": float(np.mean(np.array(min_vals) < 0) * 100) if min_vals else None,
        },
        "sharpness_comparison": {
            "onset_aligned_slope": onset_slope if np.isfinite(onset_slope) else None,
            "onset_aligned_min": onset_min if np.isfinite(onset_min) else None,
            "self_aligned_slope": self_slope if np.isfinite(self_slope) else None,
            "self_aligned_min": self_min if np.isfinite(self_min) else None,
        },
        "per_seizure_min_times": {k: float(v) for k, v in min_times.items()},
    }

    with open(results_dir / "self_alignment_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    log(f"\n{'=' * 70}")
    log("SELF-ALIGNMENT COMPLETE")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
