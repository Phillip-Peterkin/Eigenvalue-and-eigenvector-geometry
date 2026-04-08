"""Individual seizure multi-metric timeline plots.

For 10-15 representative seizures (mix of narrowing/widening, different
subjects), plots eigenvalue spacing, spectral radius, alpha power, and
delta power on a shared timeline. This allows visual assessment of whether
spacing narrowing corresponds to increasing oscillatory structure (mechanism)
or is an estimator artifact uncorrelated with spectral changes.

Loads trajectory cache and per-seizure features from _seizure_stratification.py.
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def load_features_csv(csv_path):
    features = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row:
                try:
                    row[key] = float(row[key])
                except (ValueError, TypeError):
                    pass
            features.append(row)
    return features


def select_representative_seizures(features, trajectories, n_total=15):
    narrowing = [f for f in features if f.get("group") == "narrowing"]
    widening = [f for f in features if f.get("group") == "widening"]

    narrowing.sort(key=lambda f: f.get("raw_spacing_change", 0))
    widening.sort(key=lambda f: f.get("raw_spacing_change", 0), reverse=True)

    selected = []
    seen_subjects = set()

    for source, label in [(narrowing, "narrowing"), (widening, "widening")]:
        n_target = n_total // 2 if label == "narrowing" else n_total - n_total // 2
        for f in source:
            if len(selected) >= n_total:
                break
            sid = f["subject_id"]
            key = f"{sid}_sz{int(f['seizure_idx_global'])}"
            if key in trajectories:
                if sid not in seen_subjects or len([s for s in selected if s["group"] == label]) < n_target:
                    selected.append({
                        "key": key,
                        "subject_id": sid,
                        "group": label,
                        "raw_spacing_change": f.get("raw_spacing_change", 0),
                        "preictal_slope": f.get("preictal_slope", 0),
                    })
                    seen_subjects.add(sid)
            if len([s for s in selected if s["group"] == label]) >= n_target:
                break

    return selected[:n_total]


def plot_individual_timeline(traj, metadata, fig_dir):
    key = metadata["key"]
    sid = metadata["subject_id"]
    group = metadata["group"]

    t_min = traj["time_sec"] / 60.0
    seizure_dur = traj.get("seizure_duration", 0)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    metrics = [
        ("min_spacing_z", "Eigenvalue Spacing (z)", "steelblue"),
        ("spectral_radius_z", "Spectral Radius (z)", "darkorange"),
        ("alpha_power_z", "Alpha Power 8-12Hz (z)", "green"),
        ("delta_power_z", "Delta Power 1-4Hz (z)", "purple"),
    ]

    for ax, (field, ylabel, color) in zip(axes, metrics):
        y = traj[field]
        valid = np.isfinite(y)
        ax.plot(t_min[valid], y[valid], color=color, lw=1.2, alpha=0.8)
        ax.axvline(0, color="red", ls="--", lw=1.5, alpha=0.7, label="Onset")
        if seizure_dur > 0:
            ax.axvline(seizure_dur / 60.0, color="red", ls=":", lw=1, alpha=0.5, label="Offset")
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.axvspan(-30, -10, alpha=0.03, color="green")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlim(-30, 10)

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Time relative to seizure onset (min)")

    slope = metadata.get("preictal_slope", 0)
    change = metadata.get("raw_spacing_change", 0)
    fig.suptitle(
        f"{sid} — {group.upper()} | slope={slope:.5f} | raw_change={change:.6f}",
        fontsize=12, y=1.01,
    )

    fig.tight_layout()
    safe_key = key.replace("/", "_").replace("\\", "_")
    fig.savefig(fig_dir / f"timeline_{safe_key}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_grid(selected, trajectories, fig_dir):
    n = len(selected)
    if n == 0:
        return

    n_cols = min(5, n)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx, meta in enumerate(selected):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        traj = trajectories[meta["key"]]

        t_min = traj["time_sec"] / 60.0
        valid_s = np.isfinite(traj["min_spacing_z"])
        valid_r = np.isfinite(traj["spectral_radius_z"])

        ax.plot(t_min[valid_s], traj["min_spacing_z"][valid_s], "b-", lw=1, alpha=0.8, label="Spacing")
        ax.plot(t_min[valid_r], traj["spectral_radius_z"][valid_r], "r-", lw=1, alpha=0.6, label="Spec. radius")
        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.axhline(0, color="gray", ls=":", lw=0.5)
        ax.set_xlim(-15, 5)

        color = "blue" if meta["group"] == "narrowing" else "red"
        ax.set_title(f"{meta['subject_id']} [{meta['group'][0].upper()}]", fontsize=9, color=color)

        if row == n_rows - 1:
            ax.set_xlabel("Time (min)", fontsize=8)
        if col == 0:
            ax.set_ylabel("z-score", fontsize=8)

    for idx in range(len(selected), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Individual Seizure Timelines: Spacing vs Spectral Radius", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / "summary_grid.png", dpi=200, bbox_inches="tight")
    fig.savefig(fig_dir / "summary_grid.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    log("=" * 70)
    log("INDIVIDUAL SEIZURE MULTI-METRIC TIMELINES")
    log("=" * 70)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["output"]["results_dir"]) / "analysis"
    fig_dir = Path(cfg["output"]["results_dir"]) / "figures" / "individual_seizures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cache_path = results_dir / "trajectory_cache.npz"
    csv_path = results_dir / "per_seizure_features.csv"

    if not cache_path.exists():
        log(f"ERROR: Trajectory cache not found at {cache_path}")
        log("Run _seizure_stratification.py first.")
        return

    if not csv_path.exists():
        log(f"ERROR: Features CSV not found at {csv_path}")
        log("Run _seizure_stratification.py first.")
        return

    log("\n1. Loading data...")
    trajectories = load_trajectory_cache(cache_path)
    features = load_features_csv(csv_path)
    log(f"   {len(trajectories)} trajectories, {len(features)} feature records")

    log("\n2. Selecting representative seizures...")
    selected = select_representative_seizures(features, trajectories, n_total=15)
    log(f"   Selected {len(selected)} seizures:")
    for s in selected:
        log(f"     {s['key']}: {s['group']}, change={s['raw_spacing_change']:.6f}")

    log("\n3. Generating individual timeline plots...")
    for meta in selected:
        traj = trajectories[meta["key"]]
        plot_individual_timeline(traj, meta, fig_dir)
        log(f"   Plotted {meta['key']}")

    log("\n4. Generating summary grid...")
    plot_summary_grid(selected, trajectories, fig_dir)

    log(f"\n{'=' * 70}")
    log("INDIVIDUAL TIMELINES COMPLETE")
    log(f"Figures: {fig_dir}")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
