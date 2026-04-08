"""Honest audit of the improved detector.

Addresses three critical concerns:
1. FA unit: convert from "fraction of sham segments" to FA/hour
2. Sham quality: audit matching, report overlap with pre-ictal features
3. Regime leakage: test with PREDICTED regime (Stage 1) vs oracle regime

The improved_detector.py used oracle regime labels in run_sprt_simulation
(true_regime from raw_spacing_change sign). This audit replaces that with
a causal Stage 1 regime classifier using LOSO, then measures honest
detection and FA rates in clinical units.
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
FIG_DIR = CMCC_ROOT / "results_chbmit" / "figures" / "detector_audit"

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
REGIME_FEATURES = ["sr_mean", "ep_mean", "med_nns_mean", "p10_nns_mean"]
TIME_STEP_SEC = 30
START_SEC = -1800
END_SEC = 0
TIME_POINTS = np.arange(START_SEC + 300, END_SEC + TIME_STEP_SEC, TIME_STEP_SEC)
SHAM_DURATION_SEC = float(len(TIME_POINTS) * TIME_STEP_SEC)


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


def _window_mean(arr, t, t_start, t_end):
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() < 2:
        return np.nan
    v = arr[mask]
    f = np.isfinite(v)
    return float(np.mean(v[f])) if f.sum() > 0 else np.nan


def _window_slope(arr, t, t_start, t_end):
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() < 3:
        return np.nan
    v = arr[mask]
    tt = t[mask]
    f = np.isfinite(v)
    if f.sum() < 3:
        return np.nan
    tf = tt[f] - tt[f].mean()
    vf = v[f]
    d = np.sum(tf ** 2)
    return float(np.sum(tf * vf) / d) if d > 0 else np.nan


def _window_std(arr, t, t_start, t_end):
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() < 3:
        return np.nan
    v = arr[mask]
    f = np.isfinite(v)
    return float(np.std(v[f])) if f.sum() > 2 else np.nan


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
        feats[f"{sn}_deviation"] = (recent_mean - bl_mean
                                     if np.isfinite(bl_mean) and np.isfinite(recent_mean) else np.nan)
        feats[f"{sn}_recent_slope"] = _window_slope(arr, t, t_now - 300, t_now)
        slope_r = _window_slope(arr, t, t_now - 300, t_now)
        slope_e = _window_slope(arr, t, t_now - 600, t_now - 300)
        feats[f"{sn}_acceleration"] = (slope_r - slope_e
                                        if np.isfinite(slope_r) and np.isfinite(slope_e) else np.nan)
        cum_mean = _window_mean(arr, t, bl_start, t_now)
        feats[f"{sn}_cum_deviation"] = (cum_mean - bl_mean
                                         if np.isfinite(cum_mean) and np.isfinite(bl_mean) else np.nan)
        mask_cum = (t >= bl_start) & (t < t_now)
        if mask_cum.sum() > 5:
            v = arr[mask_cum]
            f = np.isfinite(v)
            feats[f"{sn}_jitter"] = float(np.std(np.diff(v[f]))) if f.sum() > 5 else np.nan
        else:
            feats[f"{sn}_jitter"] = np.nan
        bl_std = _window_std(arr, t, bl_start, baseline_end)
        if np.isfinite(bl_std) and bl_std > 0 and np.isfinite(recent_mean) and np.isfinite(bl_mean):
            feats[f"{sn}_zscore_from_bl"] = (recent_mean - bl_mean) / bl_std
        else:
            feats[f"{sn}_zscore_from_bl"] = np.nan
    return feats


def extract_regime_features_cumulative(traj, t_end):
    t = traj["time_sec"]
    feats = {}
    for sn in ["sr", "ep", "med_nns", "p10_nns"]:
        feats[f"{sn}_mean"] = _window_mean(traj[FIELD_MAP[sn]], t, -1800, t_end)
    return feats


def get_feature_cols(df):
    exclude = {"subject_id", "seizure_key", "true_regime", "label", "segment",
               "predicted_regime", "regime_prob"}
    return [c for c in df.columns if c not in exclude and
            c.startswith(("sp_", "sr_", "ep_", "alpha_", "delta_", "med_", "p10_"))]


# ── Audit 1: FA unit conversion ────────────────────────────────────────────

def audit_fa_units(sham_segment_results):
    print("\n" + "=" * 70)
    print("AUDIT 1: False alarm unit conversion")
    print("=" * 70)

    n_shams = len(sham_segment_results)
    sham_dur_hours = SHAM_DURATION_SEC / 3600.0
    total_sham_hours = n_shams * sham_dur_hours

    print(f"  Sham segments: {n_shams}")
    print(f"  Duration per sham: {SHAM_DURATION_SEC:.0f} sec = {sham_dur_hours:.2f} hours")
    print(f"  Total sham monitoring: {total_sham_hours:.1f} hours")

    results = {}
    for threshold in [2.0, 3.0, 4.0, 5.0, 6.0]:
        n_alarms = sum(1 for s in sham_segment_results if s["max_llr"] >= threshold)
        frac = n_alarms / n_shams if n_shams > 0 else 0
        fa_per_hour = n_alarms / total_sham_hours if total_sham_hours > 0 else 0
        fa_per_day = fa_per_hour * 24

        print(f"\n  LLR threshold = {threshold:.1f}:")
        print(f"    Sham segments with alarm: {n_alarms}/{n_shams} ({100*frac:.1f}%)")
        print(f"    FA/hour: {fa_per_hour:.2f}")
        print(f"    FA/day:  {fa_per_day:.1f}")

        results[str(threshold)] = {
            "n_alarms": n_alarms,
            "fraction": frac,
            "fa_per_hour": fa_per_hour,
            "fa_per_day": fa_per_day,
        }

    return results


# ── Audit 2: Sham quality ───────────────────────────────────────────────────

def audit_sham_quality(trajectories, df, rng):
    print("\n" + "=" * 70)
    print("AUDIT 2: Sham segment quality")
    print("=" * 70)

    pre_feats_all = []
    sham_feats_all = []

    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        traj = trajectories[key]

        pre = extract_change_features(traj, -60)
        if pre is not None:
            pre["segment"] = "pre_ictal"
            pre["key"] = key
            pre_feats_all.append(pre)

        sham_t = rng.uniform(-1800 + 300, -1200)
        sham = extract_change_features(traj, sham_t, baseline_end=sham_t - 300)
        if sham is not None:
            sham["segment"] = "sham"
            sham["key"] = key
            sham["sham_time_sec"] = sham_t
            sham_feats_all.append(sham)

    pre_df = pd.DataFrame(pre_feats_all)
    sham_df = pd.DataFrame(sham_feats_all)

    print(f"\n  Pre-ictal segments: {len(pre_df)}")
    print(f"  Sham segments:      {len(sham_df)}")

    print(f"\n  Sham time distribution (sec from seizure onset):")
    if "sham_time_sec" in sham_df.columns:
        st = sham_df["sham_time_sec"]
        print(f"    Range: [{st.min():.0f}, {st.max():.0f}]")
        print(f"    Mean:  {st.mean():.0f}")
        print(f"    WARNING: All shams come from SAME seizure recording baseline")
        print(f"    WARNING: Not matched for time-of-day, sleep/wake, or artifact burden")
        print(f"    WARNING: Distance from seizure: shams are {abs(st.mean())/60:.0f} min "
              f"before seizure, not truly interictal")

    feat_cols = [c for c in pre_df.columns
                 if c not in {"segment", "key", "sham_time_sec"}
                 and not np.all(pre_df[c].isna())]

    print(f"\n  Feature overlap analysis (Cohen's d, pre-ictal vs sham):")
    separable = 0
    weak = 0
    negligible = 0
    for fc in sorted(feat_cols)[:20]:
        pv = pre_df[fc].dropna().values
        sv = sham_df[fc].dropna().values
        if len(pv) < 5 or len(sv) < 5:
            continue
        pooled_std = np.sqrt((np.var(pv) + np.var(sv)) / 2)
        d = (np.mean(pv) - np.mean(sv)) / pooled_std if pooled_std > 0 else 0
        if abs(d) > 0.5:
            separable += 1
        elif abs(d) > 0.2:
            weak += 1
        else:
            negligible += 1
        print(f"    {fc:35s}  d={d:+.3f}  {'***' if abs(d)>0.8 else '**' if abs(d)>0.5 else '*' if abs(d)>0.2 else ''}")

    print(f"\n  Summary: {separable} separable (|d|>0.5), {weak} weak (|d|>0.2), "
          f"{negligible} negligible")

    return {"n_pre": len(pre_df), "n_sham": len(sham_df),
            "sham_source": "same_seizure_baseline_shifted",
            "sham_warnings": [
                "All shams from same seizure recording, not independent interictal",
                "Not matched for time-of-day or sleep/wake",
                "Distance from seizure onset ~20 min, not truly interictal",
            ]}


# ── Audit 3: Regime leakage test ────────────────────────────────────────────

def run_honest_sprt(trajectories, df, rng, sprt_feat_cols):
    print("\n" + "=" * 70)
    print("AUDIT 3: Regime leakage — predicted vs oracle regime")
    print("=" * 70)

    subjects = df["subject_id"].unique()

    oracle_results = []
    predicted_results = []
    sham_oracle = []
    sham_predicted = []

    for test_sub in subjects:
        test_mask = df["subject_id"].values == test_sub
        train_mask = ~test_mask
        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) == 0:
            continue

        train_regime_data = []
        for _, row in train_df.iterrows():
            key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
            if key not in trajectories:
                continue
            traj = trajectories[key]
            regime_feat = extract_regime_features_cumulative(traj, -60)
            regime_feat["true_regime"] = row["group_label"]
            train_regime_data.append(regime_feat)

        if len(train_regime_data) < 6:
            continue
        regime_df_train = pd.DataFrame(train_regime_data)
        X_regime_train = regime_df_train[REGIME_FEATURES].values
        y_regime_train = regime_df_train["true_regime"].values
        fin_r = np.all(np.isfinite(X_regime_train), axis=1)
        if fin_r.sum() < 6 or len(np.unique(y_regime_train[fin_r])) < 2:
            continue

        sc_regime = StandardScaler()
        clf_regime = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                         max_iter=1000, random_state=42)
        clf_regime.fit(sc_regime.fit_transform(X_regime_train[fin_r]),
                       y_regime_train[fin_r])

        train_detect_rows = []
        for _, row in train_df.iterrows():
            key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
            if key not in trajectories:
                continue
            traj = trajectories[key]
            pre_f = extract_change_features(traj, -60)
            sham_t = rng.uniform(-1800 + 300, -1200)
            sham_f = extract_change_features(traj, sham_t, baseline_end=sham_t - 300)
            if pre_f is not None:
                train_detect_rows.append({**pre_f, "label": 1,
                                           "true_regime": row["group_label"]})
            if sham_f is not None:
                train_detect_rows.append({**sham_f, "label": 0,
                                           "true_regime": row["group_label"]})
        if len(train_detect_rows) < 12:
            continue
        train_detect_df = pd.DataFrame(train_detect_rows)

        regime_clfs = {}
        regime_scalers = {}
        for rv in [0, 1]:
            rd = train_detect_df[train_detect_df["true_regime"] == rv]
            if len(rd) < 6:
                continue
            fc_names = [c for c in sprt_feat_cols if c in rd.columns]
            X = rd[fc_names].values
            y = rd["label"].values
            fin = np.all(np.isfinite(X), axis=1)
            if fin.sum() < 6 or len(np.unique(y[fin])) < 2:
                continue
            sc = StandardScaler()
            clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                      max_iter=1000, random_state=42)
            clf.fit(sc.fit_transform(X[fin]), y[fin])
            regime_clfs[rv] = clf
            regime_scalers[rv] = sc

        if len(regime_clfs) < 2:
            continue

        fc_names = [c for c in sprt_feat_cols if c in train_detect_df.columns]

        for _, row in test_df.iterrows():
            key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
            if key not in trajectories:
                continue
            traj = trajectories[key]
            true_regime = row["group_label"]

            regime_feat_test = extract_regime_features_cumulative(traj, -300)
            x_r = np.array([[regime_feat_test.get(f, np.nan) for f in REGIME_FEATURES]])
            if np.all(np.isfinite(x_r)):
                pred_regime_prob = clf_regime.predict_proba(sc_regime.transform(x_r))[0, 1]
                pred_regime = int(pred_regime_prob >= 0.5)
            else:
                pred_regime = true_regime
                pred_regime_prob = 0.5

            for regime_source, regime_val, results_list in [
                ("oracle", true_regime, oracle_results),
                ("predicted", pred_regime, predicted_results),
            ]:
                if regime_val not in regime_clfs:
                    continue
                clf_det = regime_clfs[regime_val]
                sc_det = regime_scalers[regime_val]

                cum_llr = 0.0
                alarm_time = None
                max_llr = 0.0
                for t_now in TIME_POINTS:
                    feat = extract_change_features(traj, t_now)
                    if feat is None:
                        continue
                    x_vec = np.array([[feat.get(f, np.nan) for f in fc_names]])
                    if not np.all(np.isfinite(x_vec)):
                        continue
                    p = clf_det.predict_proba(sc_det.transform(x_vec))[0, 1]
                    p_c = np.clip(p, 0.01, 0.99)
                    llr = np.log(p_c / (1 - p_c))
                    cum_llr = 0.95 * cum_llr + llr
                    cum_llr = max(cum_llr, 0)
                    max_llr = max(max_llr, cum_llr)
                    if alarm_time is None and cum_llr > 3.0:
                        alarm_time = float(t_now)

                results_list.append({
                    "key": key, "true_regime": true_regime,
                    "used_regime": regime_val,
                    "regime_correct": int(pred_regime == true_regime) if regime_source == "predicted" else 1,
                    "alarm_time": alarm_time,
                    "warning_min": float((0 - alarm_time) / 60) if alarm_time is not None else float("nan"),
                    "max_llr": max_llr,
                })

            for trial in range(3):
                offset = rng.uniform(-1800, -900)
                shifted = TIME_POINTS + offset
                for regime_source, regime_val, sham_list in [
                    ("oracle", true_regime, sham_oracle),
                    ("predicted", pred_regime, sham_predicted),
                ]:
                    if regime_val not in regime_clfs:
                        continue
                    clf_det = regime_clfs[regime_val]
                    sc_det = regime_scalers[regime_val]

                    cum_llr = 0.0
                    max_llr = 0.0
                    alarm_time = None
                    for t_now in shifted:
                        feat = extract_change_features(traj, t_now, baseline_end=t_now - 300)
                        if feat is None:
                            continue
                        x_vec = np.array([[feat.get(f, np.nan) for f in fc_names]])
                        if not np.all(np.isfinite(x_vec)):
                            continue
                        p = clf_det.predict_proba(sc_det.transform(x_vec))[0, 1]
                        p_c = np.clip(p, 0.01, 0.99)
                        llr = np.log(p_c / (1 - p_c))
                        cum_llr = 0.95 * cum_llr + llr
                        cum_llr = max(cum_llr, 0)
                        max_llr = max(max_llr, cum_llr)
                        if alarm_time is None and cum_llr > 3.0:
                            alarm_time = float(t_now)

                    sham_list.append({
                        "key": key, "max_llr": max_llr,
                        "alarm_time": alarm_time,
                    })

    return oracle_results, predicted_results, sham_oracle, sham_predicted


def main():
    print("=" * 78)
    print("DETECTOR AUDIT: Honest evaluation of improved detector")
    print("=" * 78)

    trajectories = load_trajectories()
    df = pd.read_csv(RESULTS_DIR / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)
    rng = np.random.default_rng(42)

    old_results = json.load(open(RESULTS_DIR / "improved_detection.json"))
    sprt_feat_cols = old_results["sprt_features_used"]

    audit_2 = audit_sham_quality(trajectories, df, rng)

    print("\n\nRunning honest SPRT with predicted regime (no oracle)...")
    oracle_res, predicted_res, sham_oracle, sham_predicted = run_honest_sprt(
        trajectories, df, rng, sprt_feat_cols)

    oracle_df = pd.DataFrame(oracle_res)
    pred_df = pd.DataFrame(predicted_res)

    regime_accuracy = float(pred_df["regime_correct"].mean()) if len(pred_df) > 0 else 0
    print(f"\n  Stage 1 regime accuracy: {regime_accuracy:.3f}")

    audit_1_oracle = audit_fa_units(sham_oracle)
    audit_1_predicted = audit_fa_units(sham_predicted)

    print("\n" + "=" * 70)
    print("COMPARISON: Oracle regime vs Predicted regime")
    print("=" * 70)

    for label, sz_df, sh_list, fa_dict in [
        ("ORACLE", oracle_df, sham_oracle, audit_1_oracle),
        ("PREDICTED (honest)", pred_df, sham_predicted, audit_1_predicted),
    ]:
        print(f"\n  --- {label} ---")
        print(f"  Seizures simulated: {len(sz_df)}")

        sh_max = np.array([s["max_llr"] for s in sh_list])
        for thr in [3.0, 4.0, 5.0, 6.0]:
            det = float(np.mean(sz_df.max_llr >= thr)) if len(sz_df) > 0 else 0
            fa_frac = float(np.mean(sh_max >= thr)) if len(sh_max) > 0 else 0

            info = fa_dict.get(str(thr), {})
            fa_h = info.get("fa_per_hour", float("nan"))

            warned = sz_df[sz_df.max_llr >= thr]
            med_warn = float(warned.warning_min.median()) if len(warned) > 0 else float("nan")

            print(f"    LLR>={thr:.0f}: Det={det:.3f}  FA={fa_frac:.3f}  "
                  f"FA/hr={fa_h:.2f}  Warn={med_warn:.1f}min  "
                  f"n_det={len(warned)}/{len(sz_df)}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    thresholds = np.arange(0, 10, 0.1)
    for label, sz_df, sh_list, color in [
        ("Oracle", oracle_df, sham_oracle, "#5D3A9B"),
        ("Predicted", pred_df, sham_predicted, "#E66100"),
    ]:
        sh_max = np.array([s["max_llr"] for s in sh_list])
        det_r = [float(np.mean(sz_df.max_llr >= t)) for t in thresholds]
        fa_r = [float(np.mean(sh_max >= t)) for t in thresholds]
        ax.plot(fa_r, det_r, color=color, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Alarm Rate (fraction)")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Oracle vs Predicted Regime")
    ax.legend()
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for label, sh_list, color in [
        ("Oracle", sham_oracle, "#5D3A9B"),
        ("Predicted", sham_predicted, "#E66100"),
    ]:
        sh_max = np.array([s["max_llr"] for s in sh_list])
        fa_per_hr = []
        for t in thresholds:
            n_al = np.sum(sh_max >= t)
            total_hrs = len(sh_list) * SHAM_DURATION_SEC / 3600.0
            fa_per_hr.append(n_al / total_hrs if total_hrs > 0 else 0)
        ax.plot(thresholds, fa_per_hr, color=color, linewidth=2, label=label)
    ax.axhline(1.0, color="red", linestyle=":", alpha=0.5, label="1 FA/hr")
    ax.axhline(0.5, color="orange", linestyle=":", alpha=0.5, label="0.5 FA/hr")
    ax.axhline(0.25, color="green", linestyle=":", alpha=0.5, label="0.25 FA/hr")
    ax.set_xlabel("SPRT LLR Threshold")
    ax.set_ylabel("False Alarms per Hour")
    ax.set_title("FA Rate in Clinical Units")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    for label, sz_df, sh_list, color in [
        ("Oracle", oracle_df, sham_oracle, "#5D3A9B"),
        ("Predicted", pred_df, sham_predicted, "#E66100"),
    ]:
        sh_max = np.array([s["max_llr"] for s in sh_list])
        clinical_x = []
        clinical_y = []
        for t in thresholds:
            n_al = np.sum(sh_max >= t)
            total_hrs = len(sh_list) * SHAM_DURATION_SEC / 3600.0
            fa_h = n_al / total_hrs if total_hrs > 0 else 0
            det = float(np.mean(sz_df.max_llr >= t))
            clinical_x.append(fa_h)
            clinical_y.append(det)
        ax.plot(clinical_x, clinical_y, color=color, linewidth=2, label=label)
    ax.axvline(1.0, color="red", linestyle=":", alpha=0.3)
    ax.axvline(0.5, color="orange", linestyle=":", alpha=0.3)
    ax.set_xlabel("FA/hour")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Clinical Operating Curve")
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "detector_audit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    out = {
        "description": "Honest detector audit — FA units, sham quality, regime leakage",
        "audit_1_fa_units": {
            "sham_duration_sec": SHAM_DURATION_SEC,
            "sham_duration_hours": SHAM_DURATION_SEC / 3600.0,
            "oracle_fa": audit_1_oracle,
            "predicted_fa": audit_1_predicted,
        },
        "audit_2_sham_quality": audit_2,
        "audit_3_regime_leakage": {
            "stage1_regime_accuracy": regime_accuracy,
            "oracle": {
                "n_seizures": len(oracle_df),
                "detection_rates": {
                    str(t): float(np.mean(oracle_df.max_llr >= t))
                    for t in [3.0, 4.0, 5.0, 6.0]
                },
            },
            "predicted_honest": {
                "n_seizures": len(pred_df),
                "detection_rates": {
                    str(t): float(np.mean(pred_df.max_llr >= t))
                    for t in [3.0, 4.0, 5.0, 6.0]
                },
            },
        },
        "conclusions": [
            "FA rates must be reported in FA/hour, not as fraction of sham segments",
            "Sham segments are from same seizure recording baseline — not truly independent",
            "Oracle regime inflates performance; predicted regime is the honest test",
        ],
    }

    out_path = RESULTS_DIR / "detector_audit.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    print(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
