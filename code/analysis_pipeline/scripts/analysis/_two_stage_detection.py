"""Two-stage prospective seizure detection pipeline.

Stage 1: Identify seizure regime (narrowing vs widening) from cumulative
         features [-30, t] using geometry_no_spacing features.
Stage 2: Apply regime-stratified seizure detector (pre-ictal vs sham)
         using all available features at the detection horizon.

Both stages use LOSO cross-validation with no data leakage: the test
subject's data is excluded from BOTH the regime classifier AND the
seizure detector training.

Comparison:
  - Unstratified: single population detector (from Task 19)
  - Two-stage (hard): use predicted regime to select detector
  - Two-stage (soft): weight both detectors by regime probability
  - Oracle: use true regime label to select detector (upper bound)
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
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

warnings.filterwarnings("ignore")

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = CMCC_ROOT / "results_chbmit" / "analysis"
FIG_DIR = CMCC_ROOT / "results_chbmit" / "figures" / "two_stage"

DETECTION_HORIZONS = {
    "H3_-10_-5": (-600, -300),
    "H4_-5_-1": (-300, -60),
}

REGIME_FEATURES = ["sr_mean", "ep_mean", "med_nns_mean", "p10_nns_mean"]
DETECTION_FEATURES = [
    "sp_mean", "sp_std", "sp_slope", "sp_raw_mean",
    "sr_mean", "ep_mean", "alpha_mean", "delta_mean",
    "med_nns_mean", "p10_nns_mean",
]


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


def build_seizure_table(trajectories, df, det_t_start, det_t_end, rng):
    bl_start, bl_end = -1800, -600
    regime_t_end = det_t_start

    rows = []
    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        traj = trajectories[key]

        regime_feat = extract_features(traj, -1800, regime_t_end)
        if regime_feat is None:
            continue
        det_feat = extract_features(traj, det_t_start, det_t_end)
        if det_feat is None:
            continue

        duration = det_t_end - det_t_start
        sham_pool_end = min(bl_end, det_t_start - 30)
        if sham_pool_end - bl_start < duration:
            sham_t_start = rng.uniform(bl_start, bl_end - duration)
        else:
            sham_t_start = rng.uniform(bl_start, sham_pool_end - duration)
        sham_t_end = sham_t_start + duration
        sham_feat = extract_features(traj, sham_t_start, sham_t_end)
        if sham_feat is None:
            continue

        r = {
            "subject_id": row["subject_id"],
            "seizure_key": key,
            "true_regime": row["group_label"],
        }
        for f in REGIME_FEATURES:
            r[f"regime_{f}"] = regime_feat[f]
        for f in DETECTION_FEATURES:
            r[f"pre_{f}"] = det_feat[f]
            r[f"sham_{f}"] = sham_feat[f]
        rows.append(r)

    return pd.DataFrame(rows)


def run_two_stage(table, rng, n_perm=200):
    subjects = table["subject_id"].unique()
    n_sz = len(table)

    regime_cols = [f"regime_{f}" for f in REGIME_FEATURES]
    pre_det_cols = [f"pre_{f}" for f in DETECTION_FEATURES]
    sham_det_cols = [f"sham_{f}" for f in DETECTION_FEATURES]

    unstrat_probs = np.full(n_sz, np.nan)
    hard_probs = np.full(n_sz, np.nan)
    soft_probs = np.full(n_sz, np.nan)
    oracle_probs = np.full(n_sz, np.nan)
    regime_pred_probs = np.full(n_sz, np.nan)
    true_labels = np.ones(n_sz)

    unstrat_sham = np.full(n_sz, np.nan)
    hard_sham = np.full(n_sz, np.nan)
    soft_sham = np.full(n_sz, np.nan)
    oracle_sham = np.full(n_sz, np.nan)

    for test_sub in subjects:
        test_mask = table["subject_id"].values == test_sub
        train_mask = ~test_mask
        if train_mask.sum() < 5 or test_mask.sum() == 0:
            continue

        train = table[train_mask]
        test = table[test_mask]
        test_idx = np.where(test_mask)[0]

        X_regime_train = train[regime_cols].values
        y_regime_train = train["true_regime"].values
        X_regime_test = test[regime_cols].values

        finite_r = np.all(np.isfinite(X_regime_train), axis=1)
        if finite_r.sum() < 5 or len(np.unique(y_regime_train[finite_r])) < 2:
            continue

        sc_r = StandardScaler()
        clf_r = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                    max_iter=1000, random_state=42)
        clf_r.fit(sc_r.fit_transform(X_regime_train[finite_r]),
                  y_regime_train[finite_r])

        finite_r_test = np.all(np.isfinite(X_regime_test), axis=1)
        pred_regime_prob = np.full(test_mask.sum(), 0.5)
        if finite_r_test.any():
            pred_regime_prob[finite_r_test] = clf_r.predict_proba(
                sc_r.transform(X_regime_test[finite_r_test]))[:, 1]

        for i, idx in enumerate(test_idx):
            regime_pred_probs[idx] = pred_regime_prob[i]

        X_det_pre_train = train[pre_det_cols].values
        X_det_sham_train = train[sham_det_cols].values
        X_det_train = np.vstack([X_det_pre_train, X_det_sham_train])
        y_det_train = np.concatenate([np.ones(len(train)), np.zeros(len(train))])

        X_det_pre_test = test[pre_det_cols].values
        X_det_sham_test = test[sham_det_cols].values

        finite_d = np.all(np.isfinite(X_det_train), axis=1)
        if finite_d.sum() < 10 or len(np.unique(y_det_train[finite_d])) < 2:
            continue

        sc_d_all = StandardScaler()
        clf_d_all = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                        max_iter=1000, random_state=42)
        clf_d_all.fit(sc_d_all.fit_transform(X_det_train[finite_d]),
                      y_det_train[finite_d])

        for i, idx in enumerate(test_idx):
            pre_row = X_det_pre_test[i:i+1]
            sham_row = X_det_sham_test[i:i+1]
            if np.all(np.isfinite(pre_row)):
                unstrat_probs[idx] = clf_d_all.predict_proba(
                    sc_d_all.transform(pre_row))[0, 1]
            if np.all(np.isfinite(sham_row)):
                unstrat_sham[idx] = clf_d_all.predict_proba(
                    sc_d_all.transform(sham_row))[0, 1]

        for regime_val, regime_name in [(1, "narrowing"), (0, "widening")]:
            r_mask = train["true_regime"].values == regime_val
            if r_mask.sum() < 3:
                continue
            r_train = train[r_mask]
            X_r_pre = r_train[pre_det_cols].values
            X_r_sham = r_train[sham_det_cols].values
            X_r = np.vstack([X_r_pre, X_r_sham])
            y_r = np.concatenate([np.ones(len(r_train)), np.zeros(len(r_train))])
            fin_r = np.all(np.isfinite(X_r), axis=1)
            if fin_r.sum() < 6 or len(np.unique(y_r[fin_r])) < 2:
                continue

            sc_r_det = StandardScaler()
            clf_r_det = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                            max_iter=1000, random_state=42)
            clf_r_det.fit(sc_r_det.fit_transform(X_r[fin_r]), y_r[fin_r])

            for i, idx in enumerate(test_idx):
                pre_row = X_det_pre_test[i:i+1]
                sham_row = X_det_sham_test[i:i+1]
                if not np.all(np.isfinite(pre_row)):
                    continue

                p_det = clf_r_det.predict_proba(sc_r_det.transform(pre_row))[0, 1]
                p_sham_det = clf_r_det.predict_proba(sc_r_det.transform(sham_row))[0, 1] if np.all(np.isfinite(sham_row)) else 0.5

                true_r = table.iloc[idx]["true_regime"]
                pred_r_prob = pred_regime_prob[i]
                pred_r = 1 if pred_r_prob >= 0.5 else 0

                if true_r == regime_val:
                    oracle_probs[idx] = p_det
                    oracle_sham[idx] = p_sham_det

                if pred_r == regime_val:
                    hard_probs[idx] = p_det
                    hard_sham[idx] = p_sham_det

                if regime_val == 1:
                    w = pred_r_prob
                else:
                    w = 1 - pred_r_prob

                if np.isnan(soft_probs[idx]):
                    soft_probs[idx] = w * p_det
                    soft_sham[idx] = w * p_sham_det
                else:
                    soft_probs[idx] += w * p_det
                    soft_sham[idx] += w * p_sham_det

    def eval_method(pre_probs, sham_probs, name):
        valid_pre = np.isfinite(pre_probs)
        valid_sham = np.isfinite(sham_probs)
        valid = valid_pre & valid_sham

        if valid.sum() < 10:
            return {"method": name, "error": "insufficient data", "n_valid": int(valid.sum())}

        y_all = np.concatenate([np.ones(valid.sum()), np.zeros(valid.sum())])
        p_all = np.concatenate([pre_probs[valid], sham_probs[valid]])

        try:
            auc = float(roc_auc_score(y_all, p_all))
        except ValueError:
            auc = float("nan")

        acc = float(np.mean((p_all >= 0.5) == y_all))

        fpr, tpr, _ = roc_curve(y_all, p_all)
        sens_5 = float(np.interp(0.05, fpr, tpr))
        sens_10 = float(np.interp(0.10, fpr, tpr))

        return {
            "method": name,
            "auc": auc,
            "accuracy": acc,
            "sensitivity_at_5pct_far": sens_5,
            "sensitivity_at_10pct_far": sens_10,
            "n_valid": int(valid.sum()),
        }

    results = {
        "unstratified": eval_method(unstrat_probs, unstrat_sham, "unstratified"),
        "two_stage_hard": eval_method(hard_probs, hard_sham, "two_stage_hard"),
        "two_stage_soft": eval_method(soft_probs, soft_sham, "two_stage_soft"),
        "oracle": eval_method(oracle_probs, oracle_sham, "oracle"),
    }

    regime_acc = np.nanmean((regime_pred_probs >= 0.5).astype(int) == table["true_regime"].values)
    results["regime_classifier"] = {
        "accuracy": float(regime_acc) if np.isfinite(regime_acc) else float("nan"),
        "n_valid": int(np.isfinite(regime_pred_probs).sum()),
    }

    return results


def main():
    print("=" * 78)
    print("TWO-STAGE PROSPECTIVE SEIZURE DETECTION")
    print("Stage 1: regime identification | Stage 2: stratified detection")
    print("=" * 78)

    trajectories = load_trajectories()
    df = pd.read_csv(RESULTS_DIR / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)
    rng = np.random.default_rng(42)

    all_results = {}

    for h_key, (det_start, det_end) in DETECTION_HORIZONS.items():
        print(f"\n{'='*78}")
        print(f"DETECTION HORIZON: {h_key}  [{det_start}s, {det_end}s)")
        print(f"Regime features from: [-1800s, {det_start}s)")
        print(f"{'='*78}")

        table = build_seizure_table(trajectories, df, det_start, det_end, rng)
        print(f"  Seizures with complete data: {len(table)}")
        print(f"  Narrowing: {(table.true_regime == 1).sum()}")
        print(f"  Widening:  {(table.true_regime == 0).sum()}")
        print(f"  Subjects:  {table.subject_id.nunique()}")

        results = run_two_stage(table, rng, n_perm=200)

        print(f"\n  {'Method':25s} {'AUC':>7s} {'Acc':>7s} {'Sens@5%':>9s} {'Sens@10%':>9s}")
        print(f"  {'-'*60}")
        for name in ["unstratified", "two_stage_hard", "two_stage_soft", "oracle"]:
            r = results[name]
            if "error" in r:
                print(f"  {name:25s} {r['error']}")
            else:
                print(f"  {name:25s} {r['auc']:7.3f} {r['accuracy']:7.3f} "
                      f"{r['sensitivity_at_5pct_far']:9.3f} {r['sensitivity_at_10pct_far']:9.3f}")

        rc = results["regime_classifier"]
        print(f"\n  Regime classifier accuracy: {rc['accuracy']:.3f} (n={rc['n_valid']})")

        all_results[h_key] = results

    print(f"\n\n{'='*78}")
    print("COMPARISON SUMMARY")
    print(f"{'='*78}")

    for h_key in DETECTION_HORIZONS:
        print(f"\n  {h_key}:")
        r = all_results[h_key]
        unstrat_auc = r["unstratified"].get("auc", np.nan)
        hard_auc = r["two_stage_hard"].get("auc", np.nan)
        soft_auc = r["two_stage_soft"].get("auc", np.nan)
        oracle_auc = r["oracle"].get("auc", np.nan)

        print(f"    Unstratified:    AUC = {unstrat_auc:.3f}")
        print(f"    Two-stage hard:  AUC = {hard_auc:.3f}  (delta = {hard_auc - unstrat_auc:+.3f})")
        print(f"    Two-stage soft:  AUC = {soft_auc:.3f}  (delta = {soft_auc - unstrat_auc:+.3f})")
        print(f"    Oracle:          AUC = {oracle_auc:.3f}  (delta = {oracle_auc - unstrat_auc:+.3f})")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(DETECTION_HORIZONS), figsize=(7 * len(DETECTION_HORIZONS), 6))
    if len(DETECTION_HORIZONS) == 1:
        axes = [axes]

    colors = {"unstratified": "#888888", "two_stage_hard": "#E66100",
              "two_stage_soft": "#5D3A9B", "oracle": "#2166AC"}

    for ax, h_key in zip(axes, DETECTION_HORIZONS):
        r = all_results[h_key]
        methods = ["unstratified", "two_stage_hard", "two_stage_soft", "oracle"]
        aucs = [r[m].get("auc", 0) for m in methods]
        labels = ["Unstratified", "2-Stage\n(hard)", "2-Stage\n(soft)", "Oracle"]
        bars = ax.bar(labels, aucs, color=[colors[m] for m in methods], edgecolor="black", linewidth=0.5)
        for bar, auc_val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{auc_val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_ylim(0.3, 1.05)
        ax.set_ylabel("AUC")
        ax.set_title(f"Detection at {h_key}")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "two_stage_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {FIG_DIR / 'two_stage_comparison.png'}")

    out_path = RESULTS_DIR / "two_stage_detection.json"
    with open(out_path, "w") as f:
        json.dump({
            "description": "Two-stage prospective seizure detection",
            "stage1_features": REGIME_FEATURES,
            "stage2_features": DETECTION_FEATURES,
            "detection_horizons": {k: list(v) for k, v in DETECTION_HORIZONS.items()},
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
