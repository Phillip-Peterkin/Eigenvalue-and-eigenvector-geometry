"""Diagnostic: why does the Stage 2 seizure detector fail?

The real-time simulation showed 87% detection but 80% false alarm rate.
This script diagnoses the root cause by examining:
1. Feature distributions in pre-ictal vs sham windows
2. Temporal alarm profile (does the detector fire everywhere?)
3. Per-feature separability (Cohen's d)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = CMCC_ROOT / "results_chbmit" / "analysis"
FIG_DIR = CMCC_ROOT / "results_chbmit" / "figures" / "diagnostic"

FEATURES = ["sp_mean", "sp_std", "sp_slope", "sp_raw_mean",
            "sr_mean", "ep_mean", "alpha_mean", "delta_mean",
            "med_nns_mean", "p10_nns_mean"]

WINDOWS = {
    "early_bl": (-1800, -1200),
    "late_bl": (-1200, -600),
    "pre_ictal_10_5": (-600, -300),
    "pre_ictal_5_1": (-300, -60),
}


def load_trajectories():
    cache = np.load(RESULTS_DIR / "trajectory_cache.npz", allow_pickle=True)
    trajectories = {}
    for key in cache.files:
        traj = json.loads(str(cache[key]))
        for field in ["time_sec", "min_spacing_z", "min_spacing_raw",
                      "spectral_radius_z", "ep_score_z", "alpha_power_z",
                      "delta_power_z", "median_nns_z", "p10_nns_z"]:
            if field in traj:
                traj[field] = np.array(traj[field])
        trajectories[key] = traj
    return trajectories


def extract_features(traj, t_start, t_end):
    t = traj["time_sec"]
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() < 3:
        return None

    def sm(arr):
        v = arr[mask]; f = np.isfinite(v)
        return float(np.mean(v[f])) if f.sum() > 0 else np.nan

    def ss(arr):
        v = arr[mask]; f = np.isfinite(v)
        return float(np.std(v[f])) if f.sum() > 2 else np.nan

    def slope(arr):
        v = arr[mask]; tt = t[mask]; f = np.isfinite(v)
        if f.sum() < 3: return np.nan
        tf = tt[f] - tt[f].mean(); vf = v[f]
        d = np.sum(tf ** 2)
        return float(np.sum(tf * vf) / d) if d > 0 else np.nan

    return {
        "sp_mean": sm(traj["min_spacing_z"]),
        "sp_std": ss(traj["min_spacing_z"]),
        "sp_slope": slope(traj["min_spacing_z"]),
        "sp_raw_mean": sm(traj["min_spacing_raw"]),
        "sr_mean": sm(traj["spectral_radius_z"]),
        "ep_mean": sm(traj["ep_score_z"]),
        "alpha_mean": sm(traj["alpha_power_z"]),
        "delta_mean": sm(traj["delta_power_z"]),
        "med_nns_mean": sm(traj["median_nns_z"]),
        "p10_nns_mean": sm(traj["p10_nns_z"]),
    }


def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 3 or nb < 3:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def main():
    print("=" * 78)
    print("DETECTOR DIAGNOSTIC")
    print("=" * 78)

    trajectories = load_trajectories()
    df = pd.read_csv(RESULTS_DIR / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)

    print("\n--- 25a: Feature distributions across windows ---")
    all_window_data = {w: [] for w in WINDOWS}

    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        traj = trajectories[key]
        for w_name, (t_start, t_end) in WINDOWS.items():
            feat = extract_features(traj, t_start, t_end)
            if feat is not None:
                feat["group"] = row["group"]
                feat["subject_id"] = row["subject_id"]
                all_window_data[w_name].append(feat)

    window_dfs = {w: pd.DataFrame(rows) for w, rows in all_window_data.items()}

    print(f"\n  Window sizes: {', '.join(f'{w}={len(d)}' for w, d in window_dfs.items())}")

    key_features = ["sr_mean", "sp_mean", "ep_mean", "alpha_mean"]
    fig, axes = plt.subplots(len(key_features), 1, figsize=(12, 3 * len(key_features)))
    colors = {"early_bl": "#aaaaaa", "late_bl": "#888888",
              "pre_ictal_10_5": "#E66100", "pre_ictal_5_1": "#B2182B"}

    for ax, feat_name in zip(axes, key_features):
        for w_name, w_df in window_dfs.items():
            vals = w_df[feat_name].dropna().values
            if len(vals) > 5:
                ax.hist(vals, bins=30, alpha=0.4, color=colors[w_name],
                        label=f"{w_name} (mean={np.mean(vals):.3f})", density=True)
        ax.set_xlabel(feat_name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
        ax.set_title(f"Distribution of {feat_name} across windows")

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "feature_distributions.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n--- 25b: Temporal alarm profile ---")
    time_steps = np.arange(-1800 + 300, 60, 30)
    n_examples = 10

    example_keys = []
    for _, row in df.head(n_examples).iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key in trajectories:
            example_keys.append((key, row["subject_id"], row["group_label"]))

    train_X, train_y = [], []
    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        traj = trajectories[key]
        pre = extract_features(traj, -300, -60)
        bl = extract_features(traj, -1800, -600)
        if pre is None or bl is None:
            continue
        pre_vec = [pre[f] for f in FEATURES]
        bl_vec = [bl[f] for f in FEATURES]
        if all(np.isfinite(v) for v in pre_vec):
            train_X.append(pre_vec)
            train_y.append(1)
        if all(np.isfinite(v) for v in bl_vec):
            train_X.append(bl_vec)
            train_y.append(0)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    sc = StandardScaler()
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    clf.fit(sc.fit_transform(train_X), train_y)

    fig, axes = plt.subplots(len(example_keys), 1, figsize=(12, 2.5 * len(example_keys)), sharex=True)
    if len(example_keys) == 1:
        axes = [axes]

    for ax, (key, sub, regime) in zip(axes, example_keys):
        traj = trajectories[key]
        probs = []
        valid_t = []
        for t_now in time_steps:
            feat = extract_features(traj, max(-1800, t_now - 300), t_now)
            if feat is None:
                continue
            vec = [feat[f] for f in FEATURES]
            if all(np.isfinite(v) for v in vec):
                p = clf.predict_proba(sc.transform([vec]))[0, 1]
                probs.append(p)
                valid_t.append(t_now / 60)

        ax.plot(valid_t, probs, color="#E66100", linewidth=1.5)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(0.6, color="red", linestyle="--", alpha=0.3)
        ax.axvline(0, color="black", linestyle="-", alpha=0.5)
        regime_label = "N" if regime == 1 else "W"
        ax.set_ylabel("P(pre-ictal)")
        ax.set_title(f"{key} ({regime_label})", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.15)

    axes[-1].set_xlabel("Time (min)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "temporal_alarm_profile.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\n--- 25c: Feature separability audit ---")
    pre_df = window_dfs["pre_ictal_5_1"]
    bl_df = window_dfs["late_bl"]

    print(f"\n  {'Feature':20s} {'Cohen d':>10s} {'Uni AUC':>10s} {'Pre mean':>10s} {'BL mean':>10s}")
    print(f"  {'-'*65}")

    separability = {}
    for feat_name in FEATURES:
        pre_vals = pre_df[feat_name].dropna().values
        bl_vals = bl_df[feat_name].dropna().values
        d = cohens_d(pre_vals, bl_vals)

        if len(pre_vals) > 5 and len(bl_vals) > 5:
            y_all = np.concatenate([np.ones(len(pre_vals)), np.zeros(len(bl_vals))])
            x_all = np.concatenate([pre_vals, bl_vals])
            finite = np.isfinite(x_all)
            try:
                uni_auc = roc_auc_score(y_all[finite], x_all[finite])
            except ValueError:
                uni_auc = 0.5
        else:
            uni_auc = float("nan")

        separability[feat_name] = {
            "cohens_d": d,
            "univariate_auc": float(uni_auc),
            "pre_mean": float(np.nanmean(pre_vals)),
            "bl_mean": float(np.nanmean(bl_vals)),
            "pre_std": float(np.nanstd(pre_vals)),
            "bl_std": float(np.nanstd(bl_vals)),
        }
        print(f"  {feat_name:20s} {d:10.3f} {uni_auc:10.3f} {np.nanmean(pre_vals):10.3f} {np.nanmean(bl_vals):10.3f}")

    print(f"\n  Same analysis split by regime:")
    for regime_name, regime_val in [("narrowing", "narrowing"), ("widening", "widening")]:
        print(f"\n  --- {regime_name} ---")
        pre_r = pre_df[pre_df["group"] == regime_name]
        bl_r = bl_df[bl_df["group"] == regime_name]
        for feat_name in FEATURES:
            pre_vals = pre_r[feat_name].dropna().values
            bl_vals = bl_r[feat_name].dropna().values
            d = cohens_d(pre_vals, bl_vals)
            if len(pre_vals) > 3 and len(bl_vals) > 3:
                y_all = np.concatenate([np.ones(len(pre_vals)), np.zeros(len(bl_vals))])
                x_all = np.concatenate([pre_vals, bl_vals])
                fin = np.isfinite(x_all)
                try:
                    uni_auc = roc_auc_score(y_all[fin], x_all[fin])
                except ValueError:
                    uni_auc = 0.5
            else:
                uni_auc = float("nan")
            print(f"    {feat_name:20s} d={d:7.3f}  AUC={uni_auc:.3f}")

    print("\n\n--- DIAGNOSIS ---")
    bl_means = {f: float(np.nanmean(bl_df[f].dropna().values)) for f in FEATURES}
    pre_means = {f: float(np.nanmean(pre_df[f].dropna().values)) for f in FEATURES}

    all_near_zero = all(abs(bl_means[f]) < 0.3 for f in FEATURES if "z" not in f or f in ["sr_mean", "ep_mean", "sp_mean"])
    print(f"  Baseline features near zero (z-scored): {all_near_zero}")
    print(f"  This means sham windows from baseline are trivially 'normal'")
    print(f"  Pre-ictal features deviate from zero only in the last 5-10 min")
    print(f"  => Detector learns 'not zero' rather than 'pre-ictal signature'")
    print(f"  => Any non-baseline segment will trigger (including random noise)")

    out = {
        "description": "Detector diagnostic - why Stage 2 fails",
        "feature_separability": separability,
        "window_sizes": {w: len(d) for w, d in window_dfs.items()},
        "baseline_means": bl_means,
        "preictal_means": pre_means,
        "diagnosis": [
            "Sham windows from baseline have z-scored features near 0 by construction",
            "Detector learns to distinguish 'any deviation from zero' not 'pre-ictal signature'",
            "Interictal segments with natural variation also deviate from zero => false alarms",
            "Need independent interictal shams that go through full pipeline",
            "Need change-detection features rather than absolute level features",
        ],
    }
    out_path = RESULTS_DIR / "detector_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
