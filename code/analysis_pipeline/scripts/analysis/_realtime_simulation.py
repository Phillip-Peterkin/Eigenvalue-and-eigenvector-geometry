"""Real-time causal simulation of two-stage seizure detection.

Steps through each seizure from -30 min to onset in 30-second increments.
At each step, computes cumulative features using ONLY data seen so far,
predicts regime, and applies the regime-stratified seizure detector.

All classifiers are trained on other subjects only (LOSO). No future
data, no batch normalization across time, no look-ahead.

For false alarm estimation, runs the same simulation on interictal
sham segments.
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
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = CMCC_ROOT / "results_chbmit" / "analysis"
FIG_DIR = CMCC_ROOT / "results_chbmit" / "figures" / "realtime"

REGIME_FEATURES = ["sr_mean", "ep_mean", "med_nns_mean", "p10_nns_mean"]
DETECTION_FEATURES = [
    "sp_mean", "sp_std", "sp_slope", "sp_raw_mean",
    "sr_mean", "ep_mean", "alpha_mean", "delta_mean",
    "med_nns_mean", "p10_nns_mean",
]

TIME_STEP_SEC = 30
START_SEC = -1800
END_SEC = 0
TIME_POINTS = np.arange(START_SEC + 300, END_SEC + TIME_STEP_SEC, TIME_STEP_SEC)


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


def train_models_loso(table, test_subject):
    train = table[table["subject_id"] != test_subject]
    if len(train) < 5:
        return None

    regime_cols = [f"regime_{f}" for f in REGIME_FEATURES]
    det_cols = [f"det_{f}" for f in DETECTION_FEATURES]
    sham_cols = [f"sham_{f}" for f in DETECTION_FEATURES]

    X_r = train[regime_cols].values
    y_r = train["true_regime"].values
    fin_r = np.all(np.isfinite(X_r), axis=1)
    if fin_r.sum() < 5 or len(np.unique(y_r[fin_r])) < 2:
        return None

    sc_r = StandardScaler()
    clf_r = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                max_iter=1000, random_state=42)
    clf_r.fit(sc_r.fit_transform(X_r[fin_r]), y_r[fin_r])

    models = {"regime_scaler": sc_r, "regime_clf": clf_r, "det_models": {}}

    for regime_val in [0, 1]:
        r_mask = train["true_regime"].values == regime_val
        if r_mask.sum() < 3:
            continue
        r_train = train[r_mask]
        X_pre = r_train[det_cols].values
        X_sham = r_train[sham_cols].values
        X_d = np.vstack([X_pre, X_sham])
        y_d = np.concatenate([np.ones(len(r_train)), np.zeros(len(r_train))])
        fin_d = np.all(np.isfinite(X_d), axis=1)
        if fin_d.sum() < 6 or len(np.unique(y_d[fin_d])) < 2:
            continue
        sc_d = StandardScaler()
        clf_d = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                    max_iter=1000, random_state=42)
        clf_d.fit(sc_d.fit_transform(X_d[fin_d]), y_d[fin_d])
        models["det_models"][regime_val] = {"scaler": sc_d, "clf": clf_d}

    if len(models["det_models"]) < 2:
        return None
    return models


def build_training_table(trajectories, df, rng):
    rows = []
    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        traj = trajectories[key]

        regime_feat = extract_features(traj, -1800, -300)
        if regime_feat is None:
            continue
        det_feat = extract_features(traj, -300, -60)
        if det_feat is None:
            continue

        sham_t_start = rng.uniform(-1800, -660)
        sham_feat = extract_features(traj, sham_t_start, sham_t_start + 240)
        if sham_feat is None:
            continue

        r = {"subject_id": row["subject_id"], "seizure_key": key,
             "true_regime": row["group_label"]}
        for f in REGIME_FEATURES:
            r[f"regime_{f}"] = regime_feat[f]
        for f in DETECTION_FEATURES:
            r[f"det_{f}"] = det_feat[f]
            r[f"sham_{f}"] = sham_feat[f]
        rows.append(r)
    return pd.DataFrame(rows)


def simulate_seizure(traj, models, time_points):
    regime_probs = []
    alarm_probs = []

    for t_now in time_points:
        cum_feat = extract_features(traj, -1800, t_now)
        if cum_feat is None:
            regime_probs.append(0.5)
            alarm_probs.append(0.5)
            continue

        X_r = np.array([[cum_feat[f] for f in REGIME_FEATURES]])
        if not np.all(np.isfinite(X_r)):
            regime_probs.append(0.5)
            alarm_probs.append(0.5)
            continue

        p_narrow = models["regime_clf"].predict_proba(
            models["regime_scaler"].transform(X_r))[0, 1]
        regime_probs.append(float(p_narrow))

        recent_start = max(-1800, t_now - 300)
        det_feat = extract_features(traj, recent_start, t_now)
        if det_feat is None:
            alarm_probs.append(0.5)
            continue

        X_d = np.array([[det_feat[f] for f in DETECTION_FEATURES]])
        if not np.all(np.isfinite(X_d)):
            alarm_probs.append(0.5)
            continue

        p_alarm = 0.0
        for regime_val in [0, 1]:
            if regime_val not in models["det_models"]:
                continue
            dm = models["det_models"][regime_val]
            p_det = dm["clf"].predict_proba(dm["scaler"].transform(X_d))[0, 1]
            w = p_narrow if regime_val == 1 else (1 - p_narrow)
            p_alarm += w * p_det

        alarm_probs.append(float(p_alarm))

    return np.array(regime_probs), np.array(alarm_probs)


def main():
    print("=" * 78)
    print("REAL-TIME CAUSAL SIMULATION")
    print(f"Stepping from {START_SEC}s to {END_SEC}s in {TIME_STEP_SEC}s increments")
    print("=" * 78)

    trajectories = load_trajectories()
    df = pd.read_csv(RESULTS_DIR / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)
    rng = np.random.default_rng(42)

    training_table = build_training_table(trajectories, df, rng)
    print(f"Training table: {len(training_table)} seizures, {training_table.subject_id.nunique()} subjects")

    subjects = df["subject_id"].unique()
    per_seizure = []

    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        traj = trajectories[key]
        sub = row["subject_id"]

        models = train_models_loso(training_table, sub)
        if models is None:
            continue

        regime_probs, alarm_probs = simulate_seizure(traj, models, TIME_POINTS)

        alarm_threshold = 0.6
        above = alarm_probs >= alarm_threshold
        if above.any():
            first_alarm_idx = np.argmax(above)
            warning_time_sec = float(END_SEC - TIME_POINTS[first_alarm_idx])
            warning_time_min = warning_time_sec / 60
        else:
            warning_time_sec = float("nan")
            warning_time_min = float("nan")

        regime_stable_time = float("nan")
        pred_regime_final = 1 if regime_probs[-1] >= 0.5 else 0
        n_consec = 4
        for i in range(len(regime_probs) - n_consec + 1):
            window = regime_probs[i:i + n_consec]
            if all(p >= 0.5 for p in window) or all(p < 0.5 for p in window):
                regime_stable_time = float(TIME_POINTS[i]) / 60
                break

        per_seizure.append({
            "seizure_key": key,
            "subject_id": sub,
            "true_regime": int(row["group_label"]),
            "pred_regime_final": pred_regime_final,
            "regime_correct": pred_regime_final == row["group_label"],
            "warning_time_min": warning_time_min,
            "regime_stable_time_min": regime_stable_time,
            "final_alarm_prob": float(alarm_probs[-1]),
            "max_alarm_prob": float(np.max(alarm_probs)),
        })

    ps = pd.DataFrame(per_seizure)
    print(f"\nSimulated {len(ps)} seizures")
    print(f"Regime accuracy: {ps.regime_correct.mean():.3f}")

    warned = ps[np.isfinite(ps.warning_time_min)]
    print(f"Seizures with alarm (threshold=0.6): {len(warned)}/{len(ps)} ({100*len(warned)/len(ps):.0f}%)")
    if len(warned) > 0:
        print(f"Median warning time: {warned.warning_time_min.median():.1f} min")
        print(f"Mean warning time:   {warned.warning_time_min.mean():.1f} min")
        print(f"Min warning time:    {warned.warning_time_min.min():.1f} min")

    stable = ps[np.isfinite(ps.regime_stable_time_min)]
    if len(stable) > 0:
        print(f"\nRegime stabilization (2-min consecutive):")
        print(f"  Median time: {stable.regime_stable_time_min.median():.1f} min")
        print(f"  Mean time:   {stable.regime_stable_time_min.mean():.1f} min")

    for regime_name, regime_val in [("narrowing", 1), ("widening", 0)]:
        sub = ps[ps.true_regime == regime_val]
        sub_warned = sub[np.isfinite(sub.warning_time_min)]
        print(f"\n  {regime_name} (n={len(sub)}):")
        print(f"    Regime accuracy: {sub.regime_correct.mean():.3f}")
        print(f"    Alarm rate: {len(sub_warned)}/{len(sub)} ({100*len(sub_warned)/len(sub):.0f}%)")
        if len(sub_warned) > 0:
            print(f"    Median warning: {sub_warned.warning_time_min.median():.1f} min")

    sham_alarms = []
    n_sham_per_seizure = 3
    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        traj = trajectories[key]
        sub = row["subject_id"]
        models = train_models_loso(training_table, sub)
        if models is None:
            continue

        for _ in range(n_sham_per_seizure):
            offset = rng.uniform(-1800, -600)
            shifted_times = TIME_POINTS + offset
            _, sham_alarm = simulate_seizure(traj, models, shifted_times)
            max_sham = float(np.max(sham_alarm))
            sham_alarms.append(max_sham)

    sham_alarms = np.array(sham_alarms)
    print(f"\nFalse alarm analysis ({len(sham_alarms)} sham segments):")
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        fa_rate = float(np.mean(sham_alarms >= threshold))
        real_rate = float(np.mean(ps.max_alarm_prob >= threshold))
        print(f"  Threshold={threshold:.1f}:  FA rate={fa_rate:.3f}  Detection rate={real_rate:.3f}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    n_examples = min(8, len(ps))
    example_keys = []
    for regime_val in [1, 0]:
        sub = ps[ps.true_regime == regime_val].sort_values("max_alarm_prob", ascending=False)
        example_keys.extend(sub.head(n_examples // 2)["seizure_key"].tolist())

    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 2.5 * n_examples), sharex=True)
    if n_examples == 1:
        axes = [axes]

    for ax_idx, sz_key in enumerate(example_keys[:n_examples]):
        ax = axes[ax_idx]
        traj = trajectories[sz_key]
        sub = ps[ps.seizure_key == sz_key].iloc[0]["subject_id"]

        models = train_models_loso(training_table, sub)
        if models is None:
            continue
        regime_probs, alarm_probs = simulate_seizure(traj, models, TIME_POINTS)

        t_min = TIME_POINTS / 60
        ax.plot(t_min, regime_probs, color="#5D3A9B", linewidth=1.5, label="Regime P(narrow)")
        ax.plot(t_min, alarm_probs, color="#E66100", linewidth=1.5, label="Alarm prob")
        ax.axhline(0.6, color="red", linestyle="--", alpha=0.3, linewidth=0.8)
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
        ax.axvline(0, color="black", linestyle="-", alpha=0.5, linewidth=0.8)

        true_r = ps[ps.seizure_key == sz_key].iloc[0]["true_regime"]
        regime_label = "N" if true_r == 1 else "W"
        ax.set_ylabel("Probability")
        ax.set_title(f"{sz_key} (true={regime_label})", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.15)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    axes[-1].set_xlabel("Time relative to onset (min)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "example_timelines.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    warned_times = warned.warning_time_min.values
    if len(warned_times) > 0:
        ax.hist(warned_times, bins=20, color="#5D3A9B", edgecolor="black", alpha=0.7)
        ax.axvline(np.median(warned_times), color="red", linestyle="--",
                   label=f"Median={np.median(warned_times):.1f} min")
    ax.set_xlabel("Warning time (min before onset)")
    ax.set_ylabel("Count")
    ax.set_title("Warning Time Distribution (threshold=0.6)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIG_DIR / "warning_time_histogram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigures saved to {FIG_DIR}")

    out_path = RESULTS_DIR / "realtime_simulation.json"
    summary = {
        "description": "Real-time causal simulation of two-stage detection",
        "time_step_sec": TIME_STEP_SEC,
        "n_time_points": len(TIME_POINTS),
        "n_seizures_simulated": len(ps),
        "regime_accuracy": float(ps.regime_correct.mean()),
        "alarm_threshold": 0.6,
        "n_seizures_warned": int(len(warned)),
        "detection_rate": float(len(warned) / len(ps)) if len(ps) > 0 else 0,
        "median_warning_time_min": float(warned.warning_time_min.median()) if len(warned) > 0 else float("nan"),
        "mean_warning_time_min": float(warned.warning_time_min.mean()) if len(warned) > 0 else float("nan"),
        "false_alarm_rates": {
            str(t): float(np.mean(sham_alarms >= t))
            for t in [0.5, 0.6, 0.7, 0.8]
        },
        "per_regime": {
            "narrowing": {
                "n": int((ps.true_regime == 1).sum()),
                "regime_accuracy": float(ps[ps.true_regime == 1].regime_correct.mean()),
                "alarm_rate": float(np.isfinite(ps[ps.true_regime == 1].warning_time_min).mean()),
            },
            "widening": {
                "n": int((ps.true_regime == 0).sum()),
                "regime_accuracy": float(ps[ps.true_regime == 0].regime_correct.mean()),
                "alarm_rate": float(np.isfinite(ps[ps.true_regime == 0].warning_time_min).mean()),
            },
        },
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
