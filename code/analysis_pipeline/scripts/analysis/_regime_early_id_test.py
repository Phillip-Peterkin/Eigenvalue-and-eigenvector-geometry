"""Early regime identification test (v2).

Retrodictive regime-identification: can baseline dynamics predict which
later regime (narrowing vs widening) a seizure will follow?

Label: group_label = (raw_spacing_change < 0), defined by the seizure's
own trajectory.  This is NOT a prospective seizure marker — it is a test
of whether the regime is telegraphed in earlier dynamics.

Three improvements over v1:
1. Spacing-free ablation — removes all spacing-derived features to avoid
   near-tautological leakage from the regime-defining variable.
2. Horizon sweep — tests multiple time windows instead of one 20-min block.
3. Per-subject performance spread — reports subject-level accuracy and
   bootstrapped AUC confidence intervals.

Horizons tested:
  W1: [-30, -20] min   (early baseline)
  W2: [-20, -10] min   (late baseline)
  W3: [-10,  -5] min   (early pre-ictal)
  W4: [ -5,  -1] min   (late pre-ictal)
  C1: [-30, -10] min   (cumulative baseline)
  C2: [-30,  -5] min   (cumulative through early pre-ictal)

Thresholds:
  AUC > 0.75  =>  viable
  AUC ~ 0.6   =>  weak
  AUC ~ 0.5   =>  dead end
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

warnings.filterwarnings("ignore")

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = CMCC_ROOT / "results_chbmit" / "analysis"

HORIZONS = {
    "W1_-30_-20": (-1800, -1200),
    "W2_-20_-10": (-1200, -600),
    "W3_-10_-5": (-600, -300),
    "W4_-5_-1": (-300, -60),
    "C1_-30_-10": (-1800, -600),
    "C2_-30_-5": (-1800, -300),
}

HORIZON_LABELS = {
    "W1_-30_-20": "[-30,-20] min",
    "W2_-20_-10": "[-20,-10] min",
    "W3_-10_-5":  "[-10,-5] min",
    "W4_-5_-1":   "[-5,-1] min",
    "C1_-30_-10": "[-30,-10] min (cumul)",
    "C2_-30_-5":  "[-30,-5] min (cumul)",
}

FEATURE_SETS = {
    "all_features": [
        "sp_mean", "sp_std", "sp_slope", "sp_raw_mean",
        "sr_mean", "ep_mean", "alpha_mean", "delta_mean",
        "med_nns_mean", "p10_nns_mean",
    ],
    "no_spacing": ["sr_mean", "ep_mean", "alpha_mean", "delta_mean",
                   "med_nns_mean", "p10_nns_mean"],
    "geometry_no_spacing": ["sr_mean", "ep_mean", "med_nns_mean", "p10_nns_mean"],
    "spectral_power": ["alpha_mean", "delta_mean"],
    "spacing_only": ["sp_mean", "sp_std", "sp_slope", "sp_raw_mean"],
    "spectral_radius_only": ["sr_mean"],
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
    if mask.sum() < 5:
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
        if f.sum() < 5:
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


def loso_classify_with_details(X, y, sids, rng, n_perm=500):
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

    per_subject = {}
    for sub in subjects:
        sub_mask = (sids == sub) & valid
        if sub_mask.sum() == 0:
            continue
        sub_acc = float(accuracy_score(y[sub_mask], y_pred[sub_mask]))
        sub_n = int(sub_mask.sum())
        sub_n_correct = int((y[sub_mask] == y_pred[sub_mask]).sum())
        per_subject[sub] = {
            "accuracy": sub_acc,
            "n_seizures": sub_n,
            "n_correct": sub_n_correct,
            "n_narrowing": int((y[sub_mask] == 1).sum()),
            "n_widening": int((y[sub_mask] == 0).sum()),
        }

    sub_accs = [v["accuracy"] for v in per_subject.values() if v["n_seizures"] >= 2]

    n_boot = 2000
    boot_aucs = np.zeros(n_boot)
    y_v, yp_v = y[valid], y_prob[valid]
    for b in range(n_boot):
        idx = rng.choice(len(y_v), size=len(y_v), replace=True)
        if len(np.unique(y_v[idx])) < 2:
            boot_aucs[b] = np.nan
            continue
        try:
            boot_aucs[b] = roc_auc_score(y_v[idx], yp_v[idx])
        except ValueError:
            boot_aucs[b] = np.nan
    boot_finite = boot_aucs[np.isfinite(boot_aucs)]
    auc_ci_lo = float(np.percentile(boot_finite, 2.5)) if len(boot_finite) > 0 else np.nan
    auc_ci_hi = float(np.percentile(boot_finite, 97.5)) if len(boot_finite) > 0 else np.nan

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

    return {
        "auc": auc,
        "auc_ci_95": [auc_ci_lo, auc_ci_hi],
        "accuracy": acc,
        "p_value": p_val,
        "n_valid": int(valid.sum()),
        "null_auc_mean": float(np.mean(null_aucs)),
        "null_auc_std": float(np.std(null_aucs)),
        "per_subject_accuracy_mean": float(np.mean(sub_accs)) if sub_accs else np.nan,
        "per_subject_accuracy_std": float(np.std(sub_accs)) if sub_accs else np.nan,
        "per_subject_accuracy_min": float(np.min(sub_accs)) if sub_accs else np.nan,
        "per_subject_accuracy_max": float(np.max(sub_accs)) if sub_accs else np.nan,
        "n_subjects_with_2plus": len(sub_accs),
        "per_subject_detail": per_subject,
    }


def main():
    print("=" * 78)
    print("EARLY REGIME IDENTIFICATION TEST (v2)")
    print("Retrodictive: can baseline dynamics predict later regime label?")
    print("Fixes: spacing-free ablation, horizon sweep, per-subject spread")
    print("=" * 78)
    print()

    trajectories = load_trajectories()
    df = pd.read_csv(RESULTS_DIR / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)
    rng = np.random.default_rng(42)

    all_results = {}

    for h_key, (t_start, t_end) in HORIZONS.items():
        h_label = HORIZON_LABELS[h_key]
        print(f"\n{'='*78}")
        print(f"HORIZON: {h_label}  [{t_start}s, {t_end}s)")
        print(f"{'='*78}")

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

        data = pd.DataFrame(rows)
        n_narrow = int((data.group_label == 1).sum())
        n_widen = int((data.group_label == 0).sum())
        n_subj = data.subject_id.nunique()
        print(f"  n={len(data)}  narrow={n_narrow}  widen={n_widen}  subjects={n_subj}")
        print()

        h_results = {}
        for fs_name, feat_cols in FEATURE_SETS.items():
            available = [c for c in feat_cols if c in data.columns]
            if len(available) < len(feat_cols):
                continue

            X = data[available].values.copy()
            y = data["group_label"].values.copy()
            sids = data["subject_id"].values.copy()

            finite = np.all(np.isfinite(X), axis=1)
            X, y, sids = X[finite], y[finite], sids[finite]

            if len(X) < 15 or len(np.unique(y)) < 2:
                print(f"  {fs_name:25s}  SKIP (insufficient data)")
                continue

            res = loso_classify_with_details(X, y, sids, rng, n_perm=500)
            if res is None:
                print(f"  {fs_name:25s}  FAILED")
                continue

            sig = "***" if res["p_value"] < 0.001 else \
                  "**" if res["p_value"] < 0.01 else \
                  "*" if res["p_value"] < 0.05 else ""

            ci = res["auc_ci_95"]
            subj_str = f"subj_acc={res['per_subject_accuracy_mean']:.2f}+/-{res['per_subject_accuracy_std']:.2f}"
            print(f"  {fs_name:25s}  AUC={res['auc']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  "
                  f"p={res['p_value']:.4f}{sig:4s}  {subj_str}  (k={len(available)})")

            res_clean = {k: v for k, v in res.items() if k != "per_subject_detail"}
            res_clean["features_used"] = available
            h_results[fs_name] = res_clean

        all_results[h_key] = {
            "label": h_label,
            "n_seizures": len(data),
            "n_narrowing": n_narrow,
            "n_widening": n_widen,
            "n_subjects": n_subj,
            "feature_sets": h_results,
        }

    print(f"\n\n{'='*78}")
    print("SUMMARY TABLE")
    print(f"{'='*78}")
    print(f"{'Horizon':25s} {'Feature Set':25s} {'AUC':>7s} {'95% CI':>15s} {'p':>8s} {'SubjAcc':>10s}")
    print("-" * 95)

    global_best_auc = 0.0
    global_best_info = ""

    for h_key in HORIZONS:
        h_label = HORIZON_LABELS[h_key]
        h_data = all_results.get(h_key, {}).get("feature_sets", {})
        for fs_name, res in h_data.items():
            ci = res.get("auc_ci_95", [np.nan, np.nan])
            sig = "*" if res.get("p_value", 1) < 0.05 else " "
            subj_acc = res.get("per_subject_accuracy_mean", np.nan)
            print(f"  {h_label:23s} {fs_name:25s} {res['auc']:7.3f} "
                  f"[{ci[0]:.3f},{ci[1]:.3f}] {res['p_value']:8.4f}{sig} "
                  f"{subj_acc:9.3f}")
            if res["auc"] > global_best_auc:
                global_best_auc = res["auc"]
                global_best_info = f"{h_label} / {fs_name}"

    print()
    print(f"{'='*78}")

    no_spacing_best = 0.0
    for h_key in HORIZONS:
        h_data = all_results.get(h_key, {}).get("feature_sets", {})
        for fs_name in ["no_spacing", "geometry_no_spacing", "spectral_power", "spectral_radius_only"]:
            if fs_name in h_data:
                no_spacing_best = max(no_spacing_best, h_data[fs_name]["auc"])

    print(f"Best overall AUC:              {global_best_auc:.3f}  ({global_best_info})")
    print(f"Best AUC without spacing:      {no_spacing_best:.3f}")
    print()

    if global_best_auc > 0.75:
        verdict = "VIABLE"
    elif global_best_auc > 0.6:
        verdict = "WEAK"
    else:
        verdict = "DEAD END"

    if no_spacing_best > 0.75:
        spacing_verdict = "Non-spacing features alone predict regime => independent marker"
    elif no_spacing_best > 0.6:
        spacing_verdict = "Weak non-spacing signal => partially independent"
    else:
        spacing_verdict = "Non-spacing features cannot predict regime => spacing-dependent"

    print(f"VERDICT (overall):   {verdict}")
    print(f"VERDICT (no-spacing): {spacing_verdict}")
    print(f"{'='*78}")

    out_path = RESULTS_DIR / "early_regime_id_test.json"
    with open(out_path, "w") as f:
        json.dump({
            "description": "Retrodictive regime identification test (v2)",
            "notes": [
                "Label is raw_spacing_change sign — defined by later trajectory, not known at prediction time",
                "This tests whether the regime is telegraphed in earlier dynamics",
                "Spacing-free ablation checks independence from regime-defining variable",
                "Per-subject spread checks whether a few subjects carry the signal",
            ],
            "horizons": {k: list(v) for k, v in HORIZONS.items()},
            "feature_sets_tested": list(FEATURE_SETS.keys()),
            "results": all_results,
            "best_overall_auc": global_best_auc,
            "best_overall_info": global_best_info,
            "best_no_spacing_auc": no_spacing_best,
            "verdict_overall": verdict,
            "verdict_no_spacing": spacing_verdict,
        }, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
