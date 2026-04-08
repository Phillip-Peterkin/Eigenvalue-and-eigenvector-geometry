"""Improved Stage 2 detector with change-detection features and SPRT.

Key improvements over the original detector:
1. Sham source: early baseline [-30,-20] instead of full baseline [-30,-10]
   (avoids overlap with z-score normalization window edge)
2. Change-detection features: slopes, accelerations, cumulative deviations
3. Regime-aware detection: separate classifiers per regime
4. Sequential Probability Ratio Test for alarm accumulation

Diagnostic finding: population-level features cancel because the two
regimes move in opposite directions. Within-regime separability is
strong (d > 0.9 for key features). The fix is regime-conditioned
detection with change features.
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
FIG_DIR = CMCC_ROOT / "results_chbmit" / "figures" / "improved_detection"

REGIME_FEATURES = ["sr_mean", "ep_mean", "med_nns_mean", "p10_nns_mean"]

BASE_FEATURES = ["sp_mean", "sr_mean", "ep_mean", "alpha_mean", "delta_mean",
                 "med_nns_mean", "p10_nns_mean"]


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


def _window_mean(traj_field, t, t_start, t_end):
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() < 2:
        return np.nan
    v = traj_field[mask]
    f = np.isfinite(v)
    return float(np.mean(v[f])) if f.sum() > 0 else np.nan


def _window_slope(traj_field, t, t_start, t_end):
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() < 3:
        return np.nan
    v = traj_field[mask]
    tt = t[mask]
    f = np.isfinite(v)
    if f.sum() < 3:
        return np.nan
    tf = tt[f] - tt[f].mean()
    vf = v[f]
    d = np.sum(tf ** 2)
    return float(np.sum(tf * vf) / d) if d > 0 else np.nan


def _window_std(traj_field, t, t_start, t_end):
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() < 3:
        return np.nan
    v = traj_field[mask]
    f = np.isfinite(v)
    return float(np.std(v[f])) if f.sum() > 2 else np.nan


FIELD_MAP = {
    "sp": "min_spacing_z",
    "sr": "spectral_radius_z",
    "ep": "ep_score_z",
    "alpha": "alpha_power_z",
    "delta": "delta_power_z",
    "med_nns": "median_nns_z",
    "p10_nns": "p10_nns_z",
}

SHORT_NAMES = ["sp", "sr", "ep", "alpha", "delta", "med_nns", "p10_nns"]


def extract_change_features(traj, t_now, baseline_end=-600):
    t = traj["time_sec"]
    bl_start = -1800

    feats = {}

    for sn in SHORT_NAMES:
        field = FIELD_MAP[sn]
        arr = traj[field]

        bl_mean = _window_mean(arr, t, bl_start, baseline_end)
        recent_mean = _window_mean(arr, t, t_now - 300, t_now)
        feats[f"{sn}_recent_mean"] = recent_mean

        feats[f"{sn}_deviation"] = recent_mean - bl_mean if np.isfinite(bl_mean) and np.isfinite(recent_mean) else np.nan

        feats[f"{sn}_recent_slope"] = _window_slope(arr, t, t_now - 300, t_now)

        slope_recent = _window_slope(arr, t, t_now - 300, t_now)
        slope_earlier = _window_slope(arr, t, t_now - 600, t_now - 300)
        if np.isfinite(slope_recent) and np.isfinite(slope_earlier):
            feats[f"{sn}_acceleration"] = slope_recent - slope_earlier
        else:
            feats[f"{sn}_acceleration"] = np.nan

        cum_mean = _window_mean(arr, t, bl_start, t_now)
        if np.isfinite(cum_mean) and np.isfinite(bl_mean):
            feats[f"{sn}_cum_deviation"] = cum_mean - bl_mean
        else:
            feats[f"{sn}_cum_deviation"] = np.nan

        mask_cum = (t >= bl_start) & (t < t_now)
        if mask_cum.sum() > 5:
            v = arr[mask_cum]
            f = np.isfinite(v)
            if f.sum() > 5:
                diffs = np.diff(v[f])
                feats[f"{sn}_jitter"] = float(np.std(diffs))
            else:
                feats[f"{sn}_jitter"] = np.nan
        else:
            feats[f"{sn}_jitter"] = np.nan

        bl_std = _window_std(arr, t, bl_start, baseline_end)
        if np.isfinite(bl_std) and bl_std > 0 and np.isfinite(recent_mean) and np.isfinite(bl_mean):
            feats[f"{sn}_zscore_from_bl"] = (recent_mean - bl_mean) / bl_std
        else:
            feats[f"{sn}_zscore_from_bl"] = np.nan

    return feats


def extract_regime_features(traj, t_end):
    t = traj["time_sec"]
    feats = {}
    for sn in ["sr", "ep", "med_nns", "p10_nns"]:
        field = FIELD_MAP[sn]
        feats[f"regime_{sn}_mean"] = _window_mean(traj[field], t, -1800, t_end)
    return feats


def build_improved_dataset(trajectories, df, det_time, rng):
    rows = []
    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        traj = trajectories[key]

        pre_feat = extract_change_features(traj, det_time)
        if pre_feat is None or all(np.isnan(v) for v in pre_feat.values()):
            continue

        sham_time = rng.uniform(-1800 + 300, -1200)
        sham_feat = extract_change_features(traj, sham_time, baseline_end=sham_time - 300)

        regime_feat = extract_regime_features(traj, det_time)

        pre_row = {**pre_feat, **regime_feat,
                   "subject_id": row["subject_id"], "seizure_key": key,
                   "true_regime": row["group_label"], "label": 1, "segment": "pre_ictal"}
        sham_row = {**sham_feat, **regime_feat,
                    "subject_id": row["subject_id"], "seizure_key": key,
                    "true_regime": row["group_label"], "label": 0, "segment": "sham"}
        rows.append(pre_row)
        rows.append(sham_row)

    return pd.DataFrame(rows)


def get_feature_cols(df):
    exclude = {"subject_id", "seizure_key", "true_regime", "label", "segment"}
    return [c for c in df.columns if c not in exclude and c.startswith(("sp_", "sr_", "ep_", "alpha_", "delta_", "med_", "p10_", "regime_"))]


def loso_classify_regime_aware(df, feat_cols, n_perm=200, seed=42):
    rng = np.random.default_rng(seed)
    subjects = df["subject_id"].unique()

    y_prob_all = np.full(len(df), np.nan)
    y_true_all = df["label"].values.copy()
    regimes = df["true_regime"].values.copy()

    for test_sub in subjects:
        test_mask = df["subject_id"].values == test_sub
        train_mask = ~test_mask

        for regime_val in [0, 1]:
            train_regime = train_mask & (regimes == regime_val)
            test_regime = test_mask & (regimes == regime_val)

            if train_regime.sum() < 6 or test_regime.sum() == 0:
                continue

            X_train = df.loc[train_regime, feat_cols].values
            y_train = y_true_all[train_regime]
            X_test = df.loc[test_regime, feat_cols].values

            fin_tr = np.all(np.isfinite(X_train), axis=1)
            fin_te = np.all(np.isfinite(X_test), axis=1)

            if fin_tr.sum() < 6 or len(np.unique(y_train[fin_tr])) < 2:
                continue

            sc = StandardScaler()
            clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                      max_iter=1000, random_state=42)
            clf.fit(sc.fit_transform(X_train[fin_tr]), y_train[fin_tr])

            test_idx = np.where(test_regime)[0]
            for i, idx in enumerate(test_idx):
                if fin_te[i]:
                    y_prob_all[idx] = clf.predict_proba(
                        sc.transform(X_test[i:i+1]))[0, 1]

    valid = np.isfinite(y_prob_all)
    if valid.sum() < 20:
        return {"error": "insufficient valid predictions"}

    auc = float(roc_auc_score(y_true_all[valid], y_prob_all[valid]))
    acc = float(np.mean((y_prob_all[valid] >= 0.5) == y_true_all[valid]))

    fpr, tpr, _ = roc_curve(y_true_all[valid], y_prob_all[valid])
    sens_5 = float(np.interp(0.05, fpr, tpr))
    sens_10 = float(np.interp(0.10, fpr, tpr))

    null_aucs = np.zeros(n_perm)
    for p in range(n_perm):
        y_shuf = y_true_all.copy()
        for sub in subjects:
            sm = df["subject_id"].values == sub
            y_shuf[sm] = rng.permutation(y_shuf[sm])
        y_prob_shuf = np.full(len(df), np.nan)
        for test_sub in subjects:
            test_m = df["subject_id"].values == test_sub
            train_m = ~test_m
            for regime_val in [0, 1]:
                tr_r = train_m & (regimes == regime_val)
                te_r = test_m & (regimes == regime_val)
                if tr_r.sum() < 6 or te_r.sum() == 0:
                    continue
                X_tr = df.loc[tr_r, feat_cols].values
                y_tr = y_shuf[tr_r]
                X_te = df.loc[te_r, feat_cols].values
                fin_tr2 = np.all(np.isfinite(X_tr), axis=1)
                fin_te2 = np.all(np.isfinite(X_te), axis=1)
                if fin_tr2.sum() < 6 or len(np.unique(y_tr[fin_tr2])) < 2:
                    continue
                sc2 = StandardScaler()
                clf2 = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                           max_iter=1000, random_state=42)
                clf2.fit(sc2.fit_transform(X_tr[fin_tr2]), y_tr[fin_tr2])
                te_idx = np.where(te_r)[0]
                for i, idx in enumerate(te_idx):
                    if fin_te2[i]:
                        y_prob_shuf[idx] = clf2.predict_proba(
                            sc2.transform(X_te[i:i+1]))[0, 1]
        v2 = np.isfinite(y_prob_shuf)
        if v2.sum() >= 20:
            try:
                null_aucs[p] = roc_auc_score(y_shuf[v2], y_prob_shuf[v2])
            except ValueError:
                null_aucs[p] = 0.5

    p_val = float(np.mean(null_aucs >= auc))

    return {
        "auc": auc, "accuracy": acc, "p_value": p_val,
        "sensitivity_at_5pct_far": sens_5,
        "sensitivity_at_10pct_far": sens_10,
        "n_valid": int(valid.sum()),
        "n_preictal": int((y_true_all[valid] == 1).sum()),
        "n_sham": int((y_true_all[valid] == 0).sum()),
    }


def run_sprt_simulation(trajectories, df, rng, feat_cols_for_sprt):
    subjects = df["subject_id"].unique()
    time_steps = np.arange(-1800 + 300, 60, 30)

    training_data = build_improved_dataset(trajectories, df, -60, rng)

    per_seizure_results = []

    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        traj = trajectories[key]
        sub = row["subject_id"]
        true_regime = row["group_label"]

        train = training_data[training_data["subject_id"] != sub]
        regime_train = train[train["true_regime"] == true_regime]

        if len(regime_train) < 6:
            continue

        X_tr = regime_train[feat_cols_for_sprt].values
        y_tr = regime_train["label"].values
        fin = np.all(np.isfinite(X_tr), axis=1)
        if fin.sum() < 6 or len(np.unique(y_tr[fin])) < 2:
            continue

        sc = StandardScaler()
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                  max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(X_tr[fin]), y_tr[fin])

        cum_llr = 0.0
        alarm_time = None
        llr_trace = []
        prob_trace = []
        forget_factor = 0.95

        for t_now in time_steps:
            feat = extract_change_features(traj, t_now)
            if feat is None:
                llr_trace.append(cum_llr)
                prob_trace.append(0.5)
                continue

            x_vec = np.array([[feat.get(f, np.nan) for f in feat_cols_for_sprt]])
            if not np.all(np.isfinite(x_vec)):
                llr_trace.append(cum_llr)
                prob_trace.append(0.5)
                continue

            p = clf.predict_proba(sc.transform(x_vec))[0, 1]
            prob_trace.append(float(p))

            p_clipped = np.clip(p, 0.01, 0.99)
            llr = np.log(p_clipped / (1 - p_clipped))
            cum_llr = forget_factor * cum_llr + llr
            cum_llr = max(cum_llr, 0)
            llr_trace.append(cum_llr)

            if alarm_time is None and cum_llr > 3.0:
                alarm_time = float(t_now)

        warning_min = float((0 - alarm_time) / 60) if alarm_time is not None else float("nan")

        per_seizure_results.append({
            "seizure_key": key,
            "subject_id": sub,
            "true_regime": true_regime,
            "alarm_time_sec": alarm_time,
            "warning_time_min": warning_min,
            "max_cum_llr": float(max(llr_trace)) if llr_trace else 0,
            "final_prob": prob_trace[-1] if prob_trace else 0.5,
        })

    sham_alarms = []
    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        traj = trajectories[key]
        sub = row["subject_id"]
        true_regime = row["group_label"]

        train = training_data[training_data["subject_id"] != sub]
        regime_train = train[train["true_regime"] == true_regime]
        if len(regime_train) < 6:
            continue

        X_tr = regime_train[feat_cols_for_sprt].values
        y_tr = regime_train["label"].values
        fin = np.all(np.isfinite(X_tr), axis=1)
        if fin.sum() < 6 or len(np.unique(y_tr[fin])) < 2:
            continue

        sc = StandardScaler()
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                  max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(X_tr[fin]), y_tr[fin])

        for trial in range(3):
            offset = rng.uniform(-1800, -900)
            shifted_steps = time_steps + offset
            cum_llr = 0.0
            max_llr = 0.0
            for t_now in shifted_steps:
                feat = extract_change_features(traj, t_now, baseline_end=t_now - 300)
                if feat is None:
                    continue
                x_vec = np.array([[feat.get(f, np.nan) for f in feat_cols_for_sprt]])
                if not np.all(np.isfinite(x_vec)):
                    continue
                p = clf.predict_proba(sc.transform(x_vec))[0, 1]
                p_clipped = np.clip(p, 0.01, 0.99)
                llr = np.log(p_clipped / (1 - p_clipped))
                cum_llr = 0.95 * cum_llr + llr
                cum_llr = max(cum_llr, 0)
                max_llr = max(max_llr, cum_llr)
            sham_alarms.append(max_llr)

    return per_seizure_results, sham_alarms


def main():
    print("=" * 78)
    print("IMPROVED DETECTOR: Change Features + Regime-Aware + SPRT")
    print("=" * 78)

    trajectories = load_trajectories()
    df = pd.read_csv(RESULTS_DIR / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)
    rng = np.random.default_rng(42)

    print("\n--- Task 26: Building improved dataset ---")
    for det_time_label, det_time in [("t=-5min", -300), ("t=-1min", -60)]:
        dataset = build_improved_dataset(trajectories, df, det_time, rng)
        feat_cols = get_feature_cols(dataset)
        print(f"\n  Detection at {det_time_label}: {len(dataset)} rows, {len(feat_cols)} features")

        print(f"\n--- Task 27: Feature separability at {det_time_label} ---")
        pre = dataset[dataset["label"] == 1]
        sham = dataset[dataset["label"] == 0]
        top_features = []
        for fc in feat_cols:
            pre_vals = pre[fc].dropna().values
            sham_vals = sham[fc].dropna().values
            if len(pre_vals) < 5 or len(sham_vals) < 5:
                continue
            y_all = np.concatenate([np.ones(len(pre_vals)), np.zeros(len(sham_vals))])
            x_all = np.concatenate([pre_vals, sham_vals])
            fin = np.isfinite(x_all)
            if fin.sum() < 10:
                continue
            try:
                uni_auc = roc_auc_score(y_all[fin], x_all[fin])
            except ValueError:
                uni_auc = 0.5
            top_features.append((fc, abs(uni_auc - 0.5), uni_auc))

        top_features.sort(key=lambda x: -x[1])
        print(f"  Top 15 features by |AUC-0.5|:")
        for fc, delta, auc in top_features[:15]:
            print(f"    {fc:35s}  AUC={auc:.3f}")

        print(f"\n--- Task 28a: Regime-aware LOSO classifier at {det_time_label} ---")
        top_feat_names = [f[0] for f in top_features[:20]]
        result = loso_classify_regime_aware(dataset, top_feat_names, n_perm=200, seed=42)
        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  AUC={result['auc']:.3f}  Acc={result['accuracy']:.3f}  "
                  f"p={result['p_value']:.4f}  "
                  f"Sens@5%FAR={result['sensitivity_at_5pct_far']:.3f}  "
                  f"Sens@10%FAR={result['sensitivity_at_10pct_far']:.3f}")

    print(f"\n\n--- Task 28b: SPRT simulation ---")
    dataset_full = build_improved_dataset(trajectories, df, -60, rng)
    all_feat_cols = get_feature_cols(dataset_full)
    top_features_full = []
    pre = dataset_full[dataset_full["label"] == 1]
    sham = dataset_full[dataset_full["label"] == 0]
    for fc in all_feat_cols:
        pre_vals = pre[fc].dropna().values
        sham_vals = sham[fc].dropna().values
        if len(pre_vals) < 5 or len(sham_vals) < 5:
            continue
        y_all = np.concatenate([np.ones(len(pre_vals)), np.zeros(len(sham_vals))])
        x_all = np.concatenate([pre_vals, sham_vals])
        fin = np.isfinite(x_all)
        if fin.sum() < 10:
            continue
        try:
            uni_auc = roc_auc_score(y_all[fin], x_all[fin])
        except ValueError:
            uni_auc = 0.5
        top_features_full.append((fc, abs(uni_auc - 0.5), uni_auc))
    top_features_full.sort(key=lambda x: -x[1])
    sprt_feat_cols = [f[0] for f in top_features_full[:15]]

    print(f"  Using {len(sprt_feat_cols)} features for SPRT")
    per_seizure, sham_alarms = run_sprt_simulation(trajectories, df, rng, sprt_feat_cols)

    ps = pd.DataFrame(per_seizure)
    sham_alarms = np.array(sham_alarms)
    print(f"  Simulated {len(ps)} seizures, {len(sham_alarms)} sham segments")

    warned = ps[np.isfinite(ps.warning_time_min)]
    print(f"  Seizures with alarm (SPRT threshold=3.0): {len(warned)}/{len(ps)} ({100*len(warned)/len(ps):.0f}%)")
    if len(warned) > 0:
        print(f"  Median warning time: {warned.warning_time_min.median():.1f} min")
        print(f"  Mean warning time:   {warned.warning_time_min.mean():.1f} min")

    print(f"\n  SPRT false alarm analysis:")
    for threshold in [2.0, 3.0, 4.0, 5.0]:
        fa_rate = float(np.mean(sham_alarms >= threshold))
        det_rate = float(np.mean(ps.max_cum_llr >= threshold))
        print(f"    LLR threshold={threshold:.1f}:  FA={fa_rate:.3f}  Det={det_rate:.3f}  "
              f"Det-FA={det_rate-fa_rate:+.3f}")

    for regime_name, regime_val in [("narrowing", 1), ("widening", 0)]:
        sub = ps[ps.true_regime == regime_val]
        sub_warned = sub[np.isfinite(sub.warning_time_min)]
        print(f"\n  {regime_name} (n={len(sub)}):")
        print(f"    Alarm rate: {len(sub_warned)}/{len(sub)} ({100*len(sub_warned)/len(sub):.0f}%)")
        if len(sub_warned) > 0:
            print(f"    Median warning: {sub_warned.warning_time_min.median():.1f} min")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    thresholds = np.arange(0, 8, 0.1)
    det_rates = [float(np.mean(ps.max_cum_llr >= t)) for t in thresholds]
    fa_rates = [float(np.mean(sham_alarms >= t)) for t in thresholds]
    ax1.plot(fa_rates, det_rates, color="#5D3A9B", linewidth=2)
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax1.set_xlabel("False Alarm Rate")
    ax1.set_ylabel("Detection Rate")
    ax1.set_title("SPRT ROC (max cumulative LLR)")
    ax1.grid(True, alpha=0.2)

    for target_fa in [0.5, 0.25, 0.1, 0.05]:
        for i, fa in enumerate(fa_rates):
            if fa <= target_fa:
                ax1.annotate(f"FA={target_fa}: Det={det_rates[i]:.2f}",
                            xy=(fa, det_rates[i]),
                            fontsize=7, ha="left")
                break

    warned_times = warned.warning_time_min.values
    if len(warned_times) > 0:
        ax2.hist(warned_times, bins=20, color="#5D3A9B", edgecolor="black", alpha=0.7)
        ax2.axvline(np.median(warned_times), color="red", linestyle="--",
                    label=f"Median={np.median(warned_times):.1f} min")
    ax2.set_xlabel("Warning time (min)")
    ax2.set_ylabel("Count")
    ax2.set_title("SPRT Warning Time Distribution")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(FIG_DIR / "improved_detection_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n--- Comparison with old detector ---")
    print(f"  Old:      FA@0.6 = 0.796  Det@0.6 = 0.874")
    old_net = 0.874 - 0.796
    for t in [3.0, 4.0, 5.0]:
        fa = float(np.mean(sham_alarms >= t))
        det = float(np.mean(ps.max_cum_llr >= t))
        net = det - fa
        improvement = "BETTER" if net > old_net else "WORSE"
        print(f"  SPRT@{t}: FA={fa:.3f}  Det={det:.3f}  Net={net:+.3f}  ({improvement})")

    out = {
        "description": "Improved detector with change features + regime-aware + SPRT",
        "n_seizures": len(ps),
        "sprt_features_used": sprt_feat_cols,
        "sprt_threshold_analysis": {
            str(t): {"fa_rate": float(np.mean(sham_alarms >= t)),
                     "detection_rate": float(np.mean(ps.max_cum_llr >= t))}
            for t in [2.0, 3.0, 4.0, 5.0, 6.0]
        },
        "detection_rate_at_default_threshold": float(len(warned) / len(ps)) if len(ps) > 0 else 0,
        "median_warning_time_min": float(warned.warning_time_min.median()) if len(warned) > 0 else float("nan"),
        "old_detector": {"fa_rate": 0.796, "detection_rate": 0.874, "net": 0.078},
    }
    out_path = RESULTS_DIR / "improved_detection.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
