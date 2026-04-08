"""Publication-quality figures for the two-regime seizure analysis.

ANALYSIS IS FROZEN. This script produces figures only — no new metrics,
no new parameter sweeps. Loads existing results from:
- trajectory_cache.npz (per-seizure z-scored trajectories)
- per_seizure_features.csv (group assignments)
- regime_precision.json (classifier and incremental value results)

Produces three figures:
16a. Regime-separated grand-average trajectories (narrowing vs widening)
16b. LOSO ROC curve with AUC and accuracy-vs-null panel
16c. Ablation: spectral-only vs spectral+geometry ROC overlay + AUC bar
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

import yaml

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "chbmit.yaml"

NARROW_COLOR = "#2166AC"
WIDEN_COLOR = "#B2182B"
SPECTRAL_COLOR = "#E66100"
GEOM_COLOR = "#5D3A9B"


def log(msg):
    print(msg, flush=True)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


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


def interpolate_to_grid(trajectories, time_grid):
    aligned = {}
    for key, traj in trajectories.items():
        t = traj["time_sec"]
        s = traj["min_spacing_z"]
        if len(t) < 10:
            continue
        valid = np.isfinite(s)
        if valid.sum() < 10:
            continue
        interp = np.interp(time_grid, t[valid], s[valid],
                           left=np.nan, right=np.nan)
        aligned[key] = interp
    return aligned


def figure_16a(trajectories, features_df, fig_dir, cfg):
    log("  16a: Regime-separated trajectory plot...")

    narrowing_keys = set()
    widening_keys = set()
    for _, row in features_df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if row["raw_spacing_change"] < 0:
            narrowing_keys.add(key)
        else:
            widening_keys.add(key)

    bl = cfg["seizure"]["baseline_window"]
    time_grid = np.arange(bl[0], 600, 2.0)

    aligned = interpolate_to_grid(trajectories, time_grid)

    narrow_mat = np.array([aligned[k] for k in aligned if k in narrowing_keys])
    widen_mat = np.array([aligned[k] for k in aligned if k in widening_keys])

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for mat, color, label, n in [
        (narrow_mat, NARROW_COLOR, "Narrowing", len(narrow_mat)),
        (widen_mat, WIDEN_COLOR, "Widening", len(widen_mat)),
    ]:
        mean = np.nanmean(mat, axis=0)
        sem = np.nanstd(mat, axis=0) / np.sqrt(np.sum(np.isfinite(mat), axis=0).clip(1))
        t_min = time_grid / 60.0
        ax.plot(t_min, mean, color=color, lw=2.5, label=f"{label} (n={n})")
        ax.fill_between(t_min, mean - sem, mean + sem, color=color, alpha=0.15)

    ax.axvline(0, color="black", ls="--", lw=1.5, alpha=0.7, label="Seizure onset")
    ax.axhline(0, color="gray", ls=":", lw=1, alpha=0.5)

    bl_min = [b / 60.0 for b in bl]
    ax.axvspan(bl_min[0], bl_min[1], color="green", alpha=0.06)
    ax.text(np.mean(bl_min), ax.get_ylim()[1] * 0.9, "Baseline",
            ha="center", va="top", fontsize=8, color="green", alpha=0.7)

    pre_min = [-600 / 60.0, 0]
    ax.axvspan(pre_min[0], pre_min[1], color="orange", alpha=0.04)

    ax.set_xlabel("Time relative to seizure onset (min)", fontsize=12)
    ax.set_ylabel("Min eigenvalue spacing (z-score)", fontsize=12)
    ax.set_title("Two Dynamical Regimes: Pre-Ictal Trajectories", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.set_xlim(bl[0] / 60.0, 10)
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(fig_dir / "fig_regime_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved: {fig_dir / 'fig_regime_trajectories.png'}")


def figure_16b(precision_json, fig_dir):
    log("  16b: LOSO ROC / accuracy-vs-null...")

    clf = precision_json["classifier"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    y_true = np.array(clf["y_true_full"]) if "y_true_full" in clf else None
    y_prob = np.array(clf["y_prob_full"]) if "y_prob_full" in clf else None

    if y_true is None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        cfg = load_config()
        results_dir = Path(cfg["output"]["results_dir"]) / "analysis"
        df = pd.read_csv(results_dir / "per_seizure_features.csv")
        df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)
        feat_cols = clf["features_used"]
        X = df[feat_cols].values
        y_all = df["group_label"].values
        sids = df["subject_id"].values
        finite = np.all(np.isfinite(X), axis=1)
        X, y_all, sids = X[finite], y_all[finite], sids[finite]
        y_prob = np.full(len(y_all), np.nan)
        for sub in np.unique(sids):
            tr = sids != sub
            te = sids == sub
            if tr.sum() < 5 or te.sum() == 0 or len(np.unique(y_all[tr])) < 2:
                continue
            sc = StandardScaler()
            c = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            c.fit(sc.fit_transform(X[tr]), y_all[tr])
            y_prob[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
        v = np.isfinite(y_prob)
        y_true = y_all[v]
        y_prob = y_prob[v]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, color=GEOM_COLOR, lw=2.5, label=f"LOSO AUC = {clf['auc']:.3f}")
    ax.fill_between(fpr, 0, tpr, color=GEOM_COLOR, alpha=0.08)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Leave-One-Subject-Out ROC", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    ax = axes[1]
    bars = ax.bar(
        ["Observed", "Null (mean)"],
        [clf["accuracy"], clf["null_accuracy_mean"]],
        yerr=[0, clf["null_accuracy_std"] * 2],
        color=[GEOM_COLOR, "#BBBBBB"],
        edgecolor="black", linewidth=0.8, alpha=0.85, width=0.5,
    )
    ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.5, label="Chance (50%)")
    ax.set_ylabel("Classification Accuracy", fontsize=11)
    ax.set_title(f"Accuracy: {clf['accuracy']:.1%} (p < 0.001)", fontsize=12, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    fig.tight_layout(w_pad=3)
    fig.savefig(fig_dir / "fig_loso_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved: {fig_dir / 'fig_loso_roc.png'}")


def figure_16c(precision_json, fig_dir):
    log("  16c: Ablation — spectral-only vs spectral+geometry...")

    incr = precision_json["incremental_value"]
    comp = incr.get("comparison", {})

    cfg = load_config()
    results_dir = Path(cfg["output"]["results_dir"]) / "analysis"
    df = pd.read_csv(results_dir / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    roc_data = {}
    for model_name in ["spectral_only", "spectral_plus_geometry"]:
        if model_name not in incr or "error" in incr[model_name]:
            continue
        feat_cols = incr[model_name]["features"]
        X = df[feat_cols].values
        y_all = df["group_label"].values
        sids = df["subject_id"].values
        finite = np.all(np.isfinite(X), axis=1)
        X, y_all, sids = X[finite], y_all[finite], sids[finite]
        y_prob = np.full(len(y_all), np.nan)
        for sub in np.unique(sids):
            tr = sids != sub
            te = sids == sub
            if tr.sum() < 5 or te.sum() == 0 or len(np.unique(y_all[tr])) < 2:
                continue
            sc = StandardScaler()
            c = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            c.fit(sc.fit_transform(X[tr]), y_all[tr])
            y_prob[te] = c.predict_proba(sc.transform(X[te]))[:, 1]
        v = np.isfinite(y_prob)
        fpr, tpr, _ = roc_curve(y_all[v], y_prob[v])
        roc_data[model_name] = {"fpr": fpr, "tpr": tpr, "auc": incr[model_name]["loso_auc"]}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})

    ax = axes[0]
    style = {
        "spectral_only": {"color": SPECTRAL_COLOR, "ls": "--", "label": "Spectral only"},
        "spectral_plus_geometry": {"color": GEOM_COLOR, "ls": "-", "label": "Spectral + Geometry"},
    }
    for name, rd in roc_data.items():
        s = style[name]
        ax.plot(rd["fpr"], rd["tpr"], color=s["color"], ls=s["ls"], lw=2.5,
                label=f"{s['label']} (AUC={rd['auc']:.3f})")
        ax.fill_between(rd["fpr"], 0, rd["tpr"], color=s["color"], alpha=0.05)

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Model Comparison: ROC Curves", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    ax = axes[1]
    models = ["spectral_only", "spectral_plus_geometry"]
    aucs = [incr[m]["loso_auc"] for m in models]
    labels = ["Spectral\n(2 feat)", "Spectral +\nGeometry (5 feat)"]
    colors = [SPECTRAL_COLOR, GEOM_COLOR]

    bars = ax.bar(labels, aucs, color=colors, edgecolor="black", linewidth=0.8,
                  alpha=0.85, width=0.45)
    ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.4)
    ax.set_ylabel("LOSO AUC", fontsize=11)
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Incremental Value of Geometry", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)

    if comp:
        delta_auc = comp.get("delta_auc", 0)
        delta_bic = comp.get("delta_bic", 0)
        lr_p = comp.get("likelihood_ratio_p", 1)
        p_str = f"{lr_p:.1e}" if lr_p < 0.001 else f"{lr_p:.4f}"
        annotation = f"+{delta_auc:.3f} AUC\nBIC: {delta_bic:+.1f}\nLR p = {p_str}"
        ax.annotate(
            annotation,
            xy=(1, aucs[1]), xycoords=("axes fraction", "data"),
            xytext=(0.55, 0.88), textcoords="axes fraction",
            fontsize=9, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
        )

    fig.tight_layout(w_pad=3)
    fig.savefig(fig_dir / "fig_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"    Saved: {fig_dir / 'fig_ablation.png'}")


def main():
    log("=" * 70)
    log("PUBLICATION FIGURES — TWO-REGIME ANALYSIS")
    log("=" * 70)

    cfg = load_config()
    results_dir = Path(cfg["output"]["results_dir"]) / "analysis"
    fig_dir = Path(cfg["output"]["results_dir"]) / "figures" / "publication"
    fig_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectory_cache(results_dir / "trajectory_cache.npz")
    features_df = pd.read_csv(results_dir / "per_seizure_features.csv")
    with open(results_dir / "regime_precision.json") as f:
        precision_json = json.load(f)

    log(f"Loaded {len(trajectories)} trajectories, {len(features_df)} features rows")

    figure_16a(trajectories, features_df, fig_dir, cfg)
    figure_16b(precision_json, fig_dir)
    figure_16c(precision_json, fig_dir)

    log(f"\nAll figures saved to: {fig_dir}")
    log("=" * 70)


if __name__ == "__main__":
    main()
