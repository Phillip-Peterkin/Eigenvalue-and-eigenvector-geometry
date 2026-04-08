"""Sliding-window regime onset detection.

Pinpoints the exact minute where the two seizure regimes (narrowing vs
widening) become separable.  Tests both 1-minute sliding windows and
cumulative windows expanding from -30 min.

Uses LOSO logistic regression with permutation testing at each window.
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
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = CMCC_ROOT / "results_chbmit" / "analysis"
FIG_DIR = CMCC_ROOT / "results_chbmit" / "figures" / "regime_onset"

FEATURE_SETS = {
    "geometry_no_spacing": ["sr_mean", "ep_mean", "med_nns_mean", "p10_nns_mean"],
    "no_spacing": ["sr_mean", "ep_mean", "alpha_mean", "delta_mean",
                   "med_nns_mean", "p10_nns_mean"],
    "all_features": [
        "sp_mean", "sp_std", "sp_slope", "sp_raw_mean",
        "sr_mean", "ep_mean", "alpha_mean", "delta_mean",
        "med_nns_mean", "p10_nns_mean",
    ],
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
        v = arr[mask]
        f = np.isfinite(v)
        return float(np.mean(v[f])) if f.sum() > 0 else np.nan

    def ss(arr):
        v = arr[mask]
        f = np.isfinite(v)
        return float(np.std(v[f])) if f.sum() > 2 else np.nan

    def slope(arr):
        v = arr[mask]
        tt = t[mask]
        f = np.isfinite(v)
        if f.sum() < 3:
            return np.nan
        tf = tt[f] - tt[f].mean()
        vf = v[f]
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


def loso_auc(X, y, sids, rng, n_perm=200):
    subjects = np.unique(sids)
    if len(subjects) < 3 or len(np.unique(y)) < 2:
        return None

    def _run(X_all, y_all, sid_all):
        y_prob = np.full(len(y_all), np.nan)
        y_pred = np.full(len(y_all), -1)
        for sub in subjects:
            tr = sid_all != sub
            te = sid_all == sub
            if tr.sum() < 5 or te.sum() == 0 or len(np.unique(y_all[tr])) < 2:
                continue
            sc = StandardScaler()
            clf = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs",
                max_iter=1000, random_state=42,
            )
            clf.fit(sc.fit_transform(X_all[tr]), y_all[tr])
            y_prob[te] = clf.predict_proba(sc.transform(X_all[te]))[:, 1]
            y_pred[te] = clf.predict(sc.transform(X_all[te]))
        valid = y_pred >= 0
        return y_prob, y_pred, valid

    y_prob, y_pred, valid = _run(X, y, sids)
    if valid.sum() < 10:
        return None

    acc = float(accuracy_score(y[valid], y_pred[valid]))
    try:
        auc = float(roc_auc_score(y[valid], y_prob[valid]))
    except ValueError:
        auc = float("nan")

    null_aucs = np.zeros(n_perm)
    for p in range(n_perm):
        y_shuf = y.copy()
        for sub in subjects:
            sm = sids == sub
            y_shuf[sm] = rng.permutation(y_shuf[sm])
        yp_n, _, v_n = _run(X, y_shuf, sids)
        if v_n.sum() >= 10:
            try:
                null_aucs[p] = roc_auc_score(y_shuf[v_n], yp_n[v_n])
            except ValueError:
                null_aucs[p] = 0.5

    p_val = float(np.mean(null_aucs >= auc))
    return {"auc": auc, "accuracy": acc, "p_value": p_val, "n_valid": int(valid.sum())}


def build_dataset(trajectories, df, t_start, t_end):
    rows = []
    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        feat = extract_features(trajectories[key], t_start, t_end)
        if feat is None:
            continue
        feat["subject_id"] = row["subject_id"]
        feat["group_label"] = row["group_label"]
        rows.append(feat)
    return pd.DataFrame(rows)


def run_window(data, feat_cols, rng, n_perm=200):
    available = [c for c in feat_cols if c in data.columns]
    if len(available) < len(feat_cols):
        return None
    X = data[available].values.copy()
    y = data["group_label"].values.copy()
    sids = data["subject_id"].values.copy()
    finite = np.all(np.isfinite(X), axis=1)
    X, y, sids = X[finite], y[finite], sids[finite]
    if len(X) < 15 or len(np.unique(y)) < 2:
        return None
    return loso_auc(X, y, sids, rng, n_perm=n_perm)


def main():
    print("=" * 78)
    print("SLIDING-WINDOW REGIME ONSET DETECTION")
    print("=" * 78)

    trajectories = load_trajectories()
    df = pd.read_csv(RESULTS_DIR / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)
    rng = np.random.default_rng(42)

    sliding_minutes = list(range(-15, -1))
    cumul_end_minutes = list(range(-15, 0))

    results = {"sliding_1min": {}, "cumulative": {}}

    print("\n--- 1-minute sliding windows ---")
    for fs_name, feat_cols in FEATURE_SETS.items():
        print(f"\n  Feature set: {fs_name}")
        fs_results = {}
        for m in sliding_minutes:
            t_start = m * 60
            t_end = (m + 1) * 60
            label = f"[{m},{m+1}]"
            data = build_dataset(trajectories, df, t_start, t_end)
            if len(data) < 15:
                print(f"    {label:12s}  SKIP (n={len(data)})")
                continue
            res = run_window(data, feat_cols, rng, n_perm=200)
            if res is None:
                print(f"    {label:12s}  FAILED")
                continue
            sig = "*" if res["p_value"] < 0.05 else " "
            print(f"    {label:12s}  AUC={res['auc']:.3f}  p={res['p_value']:.4f}{sig}  n={res['n_valid']}")
            fs_results[label] = {**res, "t_start_min": m, "t_end_min": m + 1}
        results["sliding_1min"][fs_name] = fs_results

    print("\n--- Cumulative windows [-30, t] ---")
    for fs_name, feat_cols in FEATURE_SETS.items():
        print(f"\n  Feature set: {fs_name}")
        fs_results = {}
        for m in cumul_end_minutes:
            t_start = -1800
            t_end = m * 60
            label = f"[-30,{m}]"
            data = build_dataset(trajectories, df, t_start, t_end)
            if len(data) < 15:
                print(f"    {label:12s}  SKIP (n={len(data)})")
                continue
            res = run_window(data, feat_cols, rng, n_perm=200)
            if res is None:
                print(f"    {label:12s}  FAILED")
                continue
            sig = "*" if res["p_value"] < 0.05 else " "
            print(f"    {label:12s}  AUC={res['auc']:.3f}  p={res['p_value']:.4f}{sig}  n={res['n_valid']}")
            fs_results[label] = {**res, "t_start_min": -30, "t_end_min": m}
        results["cumulative"][fs_name] = fs_results

    thresholds = {}
    for window_type in ["sliding_1min", "cumulative"]:
        for fs_name, fs_results in results[window_type].items():
            key = f"{window_type}/{fs_name}"
            earliest = None
            for label, r in sorted(fs_results.items(), key=lambda x: x[1].get("t_start_min", 0)):
                if r["auc"] > 0.75 and r["p_value"] < 0.05:
                    if window_type == "sliding_1min":
                        t = r["t_start_min"]
                    else:
                        t = r["t_end_min"]
                    if earliest is None or t < earliest:
                        earliest = t
            thresholds[key] = earliest

    print("\n\n" + "=" * 78)
    print("REGIME EMERGENCE TIMES (earliest window with AUC > 0.75, p < 0.05)")
    print("=" * 78)
    for key, t in sorted(thresholds.items()):
        t_str = f"{t} min" if t is not None else "never"
        print(f"  {key:45s}  {t_str}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = {"geometry_no_spacing": "#5D3A9B", "no_spacing": "#E66100", "all_features": "#333333"}

    for ax, window_type, title in [
        (axes[0], "sliding_1min", "1-minute sliding windows"),
        (axes[1], "cumulative", "Cumulative windows [-30, t]"),
    ]:
        for fs_name, fs_results in results[window_type].items():
            if not fs_results:
                continue
            if window_type == "sliding_1min":
                x = [r["t_start_min"] + 0.5 for r in fs_results.values()]
            else:
                x = [r["t_end_min"] for r in fs_results.values()]
            y_auc = [r["auc"] for r in fs_results.values()]
            sig = [r["p_value"] < 0.05 for r in fs_results.values()]

            ax.plot(x, y_auc, "o-", color=colors.get(fs_name, "#666"),
                    label=fs_name, markersize=4)
            sig_x = [xi for xi, s in zip(x, sig) if s]
            sig_y = [yi for yi, s in zip(y_auc, sig) if s]
            if sig_x:
                ax.scatter(sig_x, sig_y, color=colors.get(fs_name, "#666"),
                           s=60, zorder=5, edgecolors="black", linewidths=0.8)

        ax.axhline(0.75, color="red", linestyle="--", alpha=0.5, label="AUC=0.75")
        ax.axhline(0.50, color="gray", linestyle=":", alpha=0.5, label="chance")
        ax.axvline(-10, color="blue", linestyle=":", alpha=0.3)
        ax.set_xlabel("Time relative to onset (min)")
        ax.set_ylabel("LOSO AUC")
        ax.set_title(title)
        ax.set_ylim(0.15, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "regime_onset_auc_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {FIG_DIR / 'regime_onset_auc_curves.png'}")

    out = {
        "description": "Sliding-window regime onset detection",
        "feature_sets": {k: v for k, v in FEATURE_SETS.items()},
        "results": results,
        "emergence_times": thresholds,
    }
    out_path = RESULTS_DIR / "regime_onset_detection.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
