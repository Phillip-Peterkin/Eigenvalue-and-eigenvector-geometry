"""Subject-level normalization predictor test.

Redesigns the seizure predictor to use SUBJECT-LEVEL normalization
instead of per-event z-scoring. The per-event approach makes each
event's baseline = 0 by construction, so the detector learned
"anything not zero" rather than a genuine pre-ictal signature.

Subject-level normalization uses pooled interictal data from each
subject's seizure-free recordings as the reference distribution.
Both pre-ictal and interictal test segments are measured against
this shared reference.

Available raw data:
- min_spacing_raw: available in both seizure and sham caches
- All other features: only z-scored (per-event) values in cache

Approach:
A. min_spacing_raw: full subject-level normalization test
B. z-scored features: within-subject paired comparison
   (controls for between-subject variance without needing raw values)
C. LOSO classifier + SPRT under both normalization schemes
D. Snyder-criteria evaluation if separability exists
"""
from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = CMCC_ROOT / "results_chbmit" / "analysis"
FIG_DIR = CMCC_ROOT / "results_chbmit" / "figures" / "subject_norm"

FIELD_MAP = {
    "sp": "min_spacing_z", "sr": "spectral_radius_z", "ep": "ep_score_z",
    "alpha": "alpha_power_z", "delta": "delta_power_z",
    "med_nns": "median_nns_z", "p10_nns": "p10_nns_z",
}
SHORT_NAMES = list(FIELD_MAP.keys())
TIME_STEP_SEC = 30
TIME_POINTS = np.arange(-1800 + 300, 60, TIME_STEP_SEC)


def log(msg):
    print(msg, flush=True)


def load_trajectories(cache_path):
    cache = np.load(cache_path, allow_pickle=True)
    trajs = {}
    for key in cache.files:
        traj = json.loads(str(cache[key]))
        for field in ["time_sec", "min_spacing_z", "min_spacing_raw",
                      "spectral_radius_z", "ep_score_z", "alpha_power_z",
                      "delta_power_z", "median_nns_z", "p10_nns_z"]:
            if field in traj:
                traj[field] = np.array(traj[field])
        trajs[key] = traj
    return trajs


def _smooth(x, window=60):
    if len(x) <= window:
        return x.copy()
    kernel = np.ones(window) / window
    padded = np.pad(x, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(x)]


def _window_vals(arr, t, t_start, t_end):
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() < 2:
        return np.array([])
    v = arr[mask]
    return v[np.isfinite(v)]


def _window_mean(arr, t, t_start, t_end):
    v = _window_vals(arr, t, t_start, t_end)
    return float(np.mean(v)) if len(v) > 0 else np.nan


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


def build_subject_norms(sham_trajs):
    """Build per-subject feature distributions from interictal trajectories.

    For min_spacing_raw: uses actual raw values.
    For z-scored features: pools all windows to get the subject's typical
    z-scored feature range. This is NOT ideal (z-scores are relative to
    each event's baseline) but is the best available without reprocessing.
    """
    subject_pools = {}

    for key, traj in sham_trajs.items():
        sub = key.split("_sham")[0]
        if sub not in subject_pools:
            subject_pools[sub] = {f: [] for f in ["min_spacing_raw"] + list(FIELD_MAP.values())}

        t = traj["time_sec"]
        for field in ["min_spacing_raw"] + list(FIELD_MAP.values()):
            if field in traj:
                vals = traj[field]
                finite = vals[np.isfinite(vals)]
                subject_pools[sub][field].extend(finite.tolist())

    norms = {}
    for sub, pools in subject_pools.items():
        norms[sub] = {}
        for field, vals in pools.items():
            arr = np.array(vals)
            if len(arr) > 10:
                norms[sub][field] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "p5": float(np.percentile(arr, 5)),
                    "p95": float(np.percentile(arr, 95)),
                    "n": len(arr),
                }
            else:
                norms[sub][field] = {"mean": 0.0, "std": 1.0, "p5": 0.0, "p95": 1.0, "n": 0}
    return norms


def subject_normalize(arr, norm_stats):
    mu = norm_stats["mean"]
    sigma = norm_stats["std"]
    if sigma == 0 or not np.isfinite(sigma):
        sigma = 1.0
    return (arr - mu) / sigma


def extract_subject_norm_features(traj, t_now, sub_norms, baseline_end=-600):
    """Extract features using subject-level normalization.

    For min_spacing_raw: normalize against subject interictal distribution.
    For z-scored features: re-normalize the z-scored values against the
    subject's typical z-score range (corrects for baseline drift).
    """
    t = traj["time_sec"]
    bl_start = t.min()
    feats = {}

    raw_sp = traj.get("min_spacing_raw")
    if raw_sp is not None:
        sp_norm = sub_norms.get("min_spacing_raw")
        if sp_norm and sp_norm["n"] > 10:
            sp_subj = subject_normalize(raw_sp, sp_norm)
            sp_smooth = _smooth(sp_subj, 60)

            bl_mean = _window_mean(sp_smooth, t, bl_start, baseline_end)
            recent_mean = _window_mean(sp_smooth, t, t_now - 300, t_now)
            feats["sp_raw_subj_level"] = recent_mean
            feats["sp_raw_subj_deviation"] = (recent_mean - bl_mean
                                              if np.isfinite(bl_mean) and np.isfinite(recent_mean)
                                              else np.nan)
            feats["sp_raw_subj_slope"] = _window_slope(sp_smooth, t, t_now - 300, t_now)
            cum_mean = _window_mean(sp_smooth, t, bl_start, t_now)
            feats["sp_raw_subj_cum_dev"] = (cum_mean - bl_mean
                                            if np.isfinite(cum_mean) and np.isfinite(bl_mean)
                                            else np.nan)

    for sn, field in FIELD_MAP.items():
        if field not in traj:
            for suffix in ["_subj_level", "_subj_deviation", "_subj_slope"]:
                feats[f"{sn}{suffix}"] = np.nan
            continue

        arr = traj[field]
        fn = sub_norms.get(field)
        if fn and fn["n"] > 10:
            arr_renorm = subject_normalize(arr, fn)
        else:
            arr_renorm = arr

        bl_mean = _window_mean(arr_renorm, t, bl_start, baseline_end)
        recent_mean = _window_mean(arr_renorm, t, t_now - 300, t_now)
        feats[f"{sn}_subj_level"] = recent_mean
        feats[f"{sn}_subj_deviation"] = (recent_mean - bl_mean
                                         if np.isfinite(bl_mean) and np.isfinite(recent_mean)
                                         else np.nan)
        feats[f"{sn}_subj_slope"] = _window_slope(arr_renorm, t, t_now - 300, t_now)

    return feats


def within_subject_paired_test(seizure_trajs, sham_trajs, df):
    """Within-subject comparison: for each subject, compare their
    pre-ictal feature values against their interictal feature values.

    This controls for between-subject variance without needing raw values.
    The z-scored features are comparable WITHIN subject because each
    event's z-scoring removes between-event mean shifts.
    """
    subjects_with_both = set()
    sham_subs = {k.split("_sham")[0] for k in sham_trajs}
    sz_subs = set(df["subject_id"].unique())
    subjects_with_both = sham_subs & sz_subs

    results = {}

    for field_name, field_key in [("min_spacing_raw", "min_spacing_raw")] + \
                                  [(sn, FIELD_MAP[sn]) for sn in SHORT_NAMES]:
        sub_diffs = []
        pre_vals_all = []
        inter_vals_all = []

        for sub in sorted(subjects_with_both):
            sz_keys = [k for k in seizure_trajs if k.startswith(sub + "_sz")]
            sh_keys = [k for k in sham_trajs if k.startswith(sub + "_sham")]

            if not sz_keys or not sh_keys:
                continue

            pre_means = []
            for sk in sz_keys:
                traj = seizure_trajs[sk]
                if field_key not in traj:
                    continue
                t = traj["time_sec"]
                v = _window_vals(traj[field_key], t, -300, -60)
                if len(v) > 2:
                    pre_means.append(float(np.mean(v)))

            inter_means = []
            for shk in sh_keys:
                straj = sham_trajs[shk]
                if field_key not in straj:
                    continue
                t = straj["time_sec"]
                mid = float(np.median(t))
                v = _window_vals(straj[field_key], t, mid - 300, mid)
                if len(v) > 2:
                    inter_means.append(float(np.mean(v)))

            if pre_means and inter_means:
                sub_pre = np.mean(pre_means)
                sub_inter = np.mean(inter_means)
                sub_diffs.append(sub_pre - sub_inter)
                pre_vals_all.append(sub_pre)
                inter_vals_all.append(sub_inter)

        if len(sub_diffs) >= 5:
            diffs = np.array(sub_diffs)
            t_stat, p_val = stats.ttest_1samp(diffs, 0)
            pooled_sd = np.std(diffs)
            cohens_d = float(np.mean(diffs) / pooled_sd) if pooled_sd > 0 else 0.0
            w_stat, w_p = stats.wilcoxon(diffs, alternative="two-sided") if len(diffs) >= 6 else (np.nan, np.nan)

            results[field_name] = {
                "n_subjects": len(sub_diffs),
                "mean_diff": float(np.mean(diffs)),
                "std_diff": float(np.std(diffs)),
                "cohens_d": cohens_d,
                "t_stat": float(t_stat),
                "p_ttest": float(p_val),
                "wilcoxon_p": float(w_p),
                "pre_mean": float(np.mean(pre_vals_all)),
                "inter_mean": float(np.mean(inter_vals_all)),
            }

    return results


def loso_subject_norm_classifier(seizure_trajs, sham_trajs, df, sub_norms, seed=42):
    """LOSO classifier using subject-normalized features."""
    rng = np.random.default_rng(seed)

    pre_rows = []
    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in seizure_trajs:
            continue
        sub = row["subject_id"]
        if sub not in sub_norms:
            continue
        traj = seizure_trajs[key]
        feats = extract_subject_norm_features(traj, -60, sub_norms[sub])
        feats["subject_id"] = sub
        feats["label"] = 1
        feats["key"] = key
        pre_rows.append(feats)

    sham_rows = []
    for skey, straj in sham_trajs.items():
        sub = skey.split("_sham")[0]
        if sub not in sub_norms:
            continue
        t = straj["time_sec"]
        mid = float(np.median(t))
        feats = extract_subject_norm_features(straj, mid, sub_norms[sub],
                                              baseline_end=mid - 600)
        feats["subject_id"] = sub
        feats["label"] = 0
        feats["key"] = skey
        sham_rows.append(feats)

    all_rows = pre_rows + sham_rows
    if len(all_rows) < 20:
        return {"error": "insufficient data"}

    full_df = pd.DataFrame(all_rows)
    feat_cols = [c for c in full_df.columns
                 if c not in {"subject_id", "label", "key"}
                 and full_df[c].notna().sum() > len(full_df) * 0.5]

    if not feat_cols:
        return {"error": "no valid features"}

    subjects = full_df["subject_id"].unique()
    y_true = full_df["label"].values.copy()
    y_prob = np.full(len(full_df), np.nan)

    for test_sub in subjects:
        test_mask = full_df["subject_id"].values == test_sub
        train_mask = ~test_mask

        X_train = full_df.loc[train_mask, feat_cols].values
        y_train = y_true[train_mask]
        X_test = full_df.loc[test_mask, feat_cols].values

        fin_tr = np.all(np.isfinite(X_train), axis=1)
        fin_te = np.all(np.isfinite(X_test), axis=1)

        if fin_tr.sum() < 10 or len(np.unique(y_train[fin_tr])) < 2:
            continue

        sc = StandardScaler()
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(X_train[fin_tr]), y_train[fin_tr])

        test_idx = np.where(test_mask)[0]
        for i, idx in enumerate(test_idx):
            if fin_te[i]:
                y_prob[idx] = clf.predict_proba(
                    sc.transform(X_test[i:i+1]))[0, 1]

    valid = np.isfinite(y_prob)
    if valid.sum() < 20:
        return {"error": "insufficient valid predictions", "n_valid": int(valid.sum())}

    auc = float(roc_auc_score(y_true[valid], y_prob[valid]))
    acc = float(np.mean((y_prob[valid] >= 0.5) == y_true[valid]))

    fpr, tpr, _ = roc_curve(y_true[valid], y_prob[valid])
    sens_5 = float(np.interp(0.05, fpr, tpr))
    sens_10 = float(np.interp(0.10, fpr, tpr))

    n_perm = 200
    null_aucs = np.zeros(n_perm)
    for p in range(n_perm):
        y_shuf = y_true.copy()
        for sub in subjects:
            sm = full_df["subject_id"].values == sub
            y_shuf[sm] = rng.permutation(y_shuf[sm])
        y_prob_s = np.full(len(full_df), np.nan)
        for test_sub in subjects:
            test_m = full_df["subject_id"].values == test_sub
            train_m = ~test_m
            X_tr = full_df.loc[train_m, feat_cols].values
            y_tr = y_shuf[train_m]
            X_te = full_df.loc[test_m, feat_cols].values
            fin_tr2 = np.all(np.isfinite(X_tr), axis=1)
            fin_te2 = np.all(np.isfinite(X_te), axis=1)
            if fin_tr2.sum() < 10 or len(np.unique(y_tr[fin_tr2])) < 2:
                continue
            sc2 = StandardScaler()
            clf2 = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                       max_iter=1000, random_state=42)
            clf2.fit(sc2.fit_transform(X_tr[fin_tr2]), y_tr[fin_tr2])
            te_idx = np.where(test_m)[0]
            for i, idx in enumerate(te_idx):
                if fin_te2[i]:
                    y_prob_s[idx] = clf2.predict_proba(
                        sc2.transform(X_te[i:i+1]))[0, 1]
        v2 = np.isfinite(y_prob_s)
        if v2.sum() >= 20:
            try:
                null_aucs[p] = roc_auc_score(y_shuf[v2], y_prob_s[v2])
            except ValueError:
                null_aucs[p] = 0.5

    p_val = float(np.mean(null_aucs >= auc))

    top_d = []
    for fc in feat_cols:
        pv = full_df.loc[full_df["label"] == 1, fc].dropna().values
        sv = full_df.loc[full_df["label"] == 0, fc].dropna().values
        if len(pv) > 3 and len(sv) > 3:
            pooled = np.sqrt((np.var(pv) + np.var(sv)) / 2)
            d = (np.mean(pv) - np.mean(sv)) / pooled if pooled > 0 else 0
            try:
                uni_auc = roc_auc_score(
                    np.concatenate([np.ones(len(pv)), np.zeros(len(sv))]),
                    np.concatenate([pv, sv]))
            except ValueError:
                uni_auc = 0.5
            top_d.append({"feature": fc, "cohens_d": float(d), "uni_auc": float(uni_auc)})
    top_d.sort(key=lambda x: -abs(x["cohens_d"]))

    return {
        "auc": auc,
        "accuracy": acc,
        "p_value": p_val,
        "sensitivity_at_5pct_far": sens_5,
        "sensitivity_at_10pct_far": sens_10,
        "n_valid": int(valid.sum()),
        "n_preictal": int((y_true[valid] == 1).sum()),
        "n_sham": int((y_true[valid] == 0).sum()),
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "top_features": top_d[:20],
        "null_auc_mean": float(np.mean(null_aucs)),
        "null_auc_std": float(np.std(null_aucs)),
    }


def run_subject_norm_sprt(seizure_trajs, sham_trajs, df, sub_norms, seed=42):
    """SPRT using subject-normalized features."""
    rng = np.random.default_rng(seed)
    subjects = df["subject_id"].unique()

    train_rows = []
    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in seizure_trajs:
            continue
        sub = row["subject_id"]
        if sub not in sub_norms:
            continue
        traj = seizure_trajs[key]

        pre_f = extract_subject_norm_features(traj, -60, sub_norms[sub])
        pre_f["subject_id"] = sub
        pre_f["label"] = 1
        train_rows.append(pre_f)

        sham_t = rng.uniform(-1800 + 300, -1200)
        sham_f = extract_subject_norm_features(traj, sham_t, sub_norms[sub],
                                               baseline_end=sham_t - 300)
        sham_f["subject_id"] = sub
        sham_f["label"] = 0
        train_rows.append(sham_f)

    train_df = pd.DataFrame(train_rows)
    feat_cols = [c for c in train_df.columns
                 if c not in {"subject_id", "label"}
                 and train_df[c].notna().sum() > len(train_df) * 0.5]

    seizure_results = []
    sham_results = []

    for test_sub in subjects:
        if test_sub not in sub_norms:
            continue

        train_mask = train_df["subject_id"].values != test_sub
        X_tr = train_df.loc[train_mask, feat_cols].values
        y_tr = train_df.loc[train_mask, "label"].values
        fin = np.all(np.isfinite(X_tr), axis=1)
        if fin.sum() < 10 or len(np.unique(y_tr[fin])) < 2:
            continue

        sc = StandardScaler()
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(X_tr[fin]), y_tr[fin])

        test_szs = df[df["subject_id"] == test_sub]
        for _, row in test_szs.iterrows():
            key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
            if key not in seizure_trajs:
                continue
            traj = seizure_trajs[key]

            cum_llr = 0.0
            alarm_time = None
            max_llr = 0.0
            for t_now in TIME_POINTS:
                feat = extract_subject_norm_features(traj, t_now, sub_norms[test_sub])
                x_vec = np.array([[feat.get(f, np.nan) for f in feat_cols]])
                if not np.all(np.isfinite(x_vec)):
                    continue
                p = clf.predict_proba(sc.transform(x_vec))[0, 1]
                p_c = np.clip(p, 0.01, 0.99)
                llr = np.log(p_c / (1 - p_c))
                cum_llr = 0.95 * cum_llr + llr
                cum_llr = max(cum_llr, 0)
                max_llr = max(max_llr, cum_llr)
                if alarm_time is None and cum_llr > 3.0:
                    alarm_time = float(t_now)

            seizure_results.append({
                "key": key, "subject_id": test_sub,
                "alarm_time": alarm_time,
                "warning_min": float((0 - alarm_time) / 60) if alarm_time is not None else float("nan"),
                "max_llr": max_llr,
            })

        test_sham_keys = [k for k in sham_trajs if k.startswith(test_sub + "_sham")]
        for skey in test_sham_keys:
            straj = sham_trajs[skey]
            t = straj["time_sec"]
            t_range = t.max() - t.min()
            if t_range < 600:
                continue

            sim_start = t.min() + 300
            sim_end = t.max() - 60
            sim_steps = np.arange(sim_start, sim_end, TIME_STEP_SEC)

            cum_llr = 0.0
            max_llr = 0.0
            alarm_time = None
            for t_now in sim_steps:
                feat = extract_subject_norm_features(straj, t_now, sub_norms[test_sub],
                                                     baseline_end=t_now - 600)
                x_vec = np.array([[feat.get(f, np.nan) for f in feat_cols]])
                if not np.all(np.isfinite(x_vec)):
                    continue
                p = clf.predict_proba(sc.transform(x_vec))[0, 1]
                p_c = np.clip(p, 0.01, 0.99)
                llr = np.log(p_c / (1 - p_c))
                cum_llr = 0.95 * cum_llr + llr
                cum_llr = max(cum_llr, 0)
                max_llr = max(max_llr, cum_llr)
                if alarm_time is None and cum_llr > 3.0:
                    alarm_time = float(t_now)

            sham_duration_sec = float(len(sim_steps) * TIME_STEP_SEC)
            sham_results.append({
                "key": skey, "subject_id": test_sub,
                "max_llr": max_llr, "alarm_time": alarm_time,
                "duration_sec": sham_duration_sec,
            })

    return seizure_results, sham_results, feat_cols


def compute_sprt_table(sz_results, sh_results):
    sz_df = pd.DataFrame(sz_results)
    sh_df = pd.DataFrame(sh_results)

    if len(sh_df) == 0:
        return {}, sz_df, sh_df

    total_sham_hours = sh_df["duration_sec"].sum() / 3600.0
    sh_max = sh_df["max_llr"].values

    table = {}
    for thr in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        det = float(np.mean(sz_df.max_llr >= thr)) if len(sz_df) > 0 else 0
        n_fa = int(np.sum(sh_max >= thr))
        fa_frac = float(np.mean(sh_max >= thr))
        fa_hr = n_fa / total_sham_hours if total_sham_hours > 0 else 0

        warned = sz_df[sz_df.max_llr >= thr]
        med_warn = float(warned.warning_min.median()) if len(warned) > 0 else float("nan")

        table[str(thr)] = {
            "detection_rate": det, "fa_fraction": fa_frac,
            "fa_per_hour": fa_hr, "fa_per_day": fa_hr * 24,
            "n_detected": int((sz_df.max_llr >= thr).sum()),
            "n_false_alarms": n_fa, "median_warning_min": med_warn,
        }

    return table, sz_df, sh_df


def raw_spacing_level_test(seizure_trajs, sham_trajs, df, sub_norms):
    """Test whether absolute LEVEL of raw spacing differs between
    pre-ictal and interictal, using subject-level normalization.

    This is the key test: per-event z-scoring removes level differences.
    Subject-level normalization preserves them.
    """
    pre_levels = []
    inter_levels = []

    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in seizure_trajs:
            continue
        sub = row["subject_id"]
        if sub not in sub_norms or sub_norms[sub]["min_spacing_raw"]["n"] < 10:
            continue

        traj = seizure_trajs[key]
        t = traj["time_sec"]
        raw = traj["min_spacing_raw"]

        norm = sub_norms[sub]["min_spacing_raw"]
        raw_subj = (raw - norm["mean"]) / max(norm["std"], 1e-10)

        pre_vals = _window_vals(raw_subj, t, -300, -60)
        if len(pre_vals) > 2:
            pre_levels.append(float(np.mean(pre_vals)))

    for skey, straj in sham_trajs.items():
        sub = skey.split("_sham")[0]
        if sub not in sub_norms or sub_norms[sub]["min_spacing_raw"]["n"] < 10:
            continue

        t = straj["time_sec"]
        raw = straj["min_spacing_raw"]

        norm = sub_norms[sub]["min_spacing_raw"]
        raw_subj = (raw - norm["mean"]) / max(norm["std"], 1e-10)

        mid = float(np.median(t))
        inter_vals = _window_vals(raw_subj, t, mid - 300, mid)
        if len(inter_vals) > 2:
            inter_levels.append(float(np.mean(inter_vals)))

    pre_arr = np.array(pre_levels)
    inter_arr = np.array(inter_levels)

    if len(pre_arr) < 5 or len(inter_arr) < 5:
        return {"error": "insufficient data"}

    pooled = np.sqrt((np.var(pre_arr) + np.var(inter_arr)) / 2)
    d = (np.mean(pre_arr) - np.mean(inter_arr)) / pooled if pooled > 0 else 0
    u_stat, u_p = stats.mannwhitneyu(pre_arr, inter_arr, alternative="two-sided")
    t_stat, t_p = stats.ttest_ind(pre_arr, inter_arr)

    try:
        auc = roc_auc_score(
            np.concatenate([np.ones(len(pre_arr)), np.zeros(len(inter_arr))]),
            np.concatenate([pre_arr, inter_arr]))
    except ValueError:
        auc = 0.5

    return {
        "n_preictal": len(pre_arr),
        "n_interictal": len(inter_arr),
        "pre_mean": float(np.mean(pre_arr)),
        "pre_std": float(np.std(pre_arr)),
        "inter_mean": float(np.mean(inter_arr)),
        "inter_std": float(np.std(inter_arr)),
        "cohens_d": float(d),
        "mann_whitney_p": float(u_p),
        "ttest_p": float(t_p),
        "univariate_auc": float(auc),
    }


def make_figures(paired_results, level_test, classifier_result,
                 sprt_table, sz_df, sh_df, old_sprt):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    ax = axes[0, 0]
    fields = []
    ds = []
    ps = []
    for fname, r in sorted(paired_results.items()):
        fields.append(fname)
        ds.append(r["cohens_d"])
        ps.append(r["p_ttest"])
    colors = ["#5D3A9B" if abs(d) > 0.5 else "#E66100" if abs(d) > 0.2 else "#999999"
              for d in ds]
    ax.barh(range(len(fields)), ds, color=colors, alpha=0.8)
    ax.set_yticks(range(len(fields)))
    ax.set_yticklabels(fields, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.axvline(0.5, color="red", linestyle=":", alpha=0.4)
    ax.axvline(-0.5, color="red", linestyle=":", alpha=0.4)
    ax.set_xlabel("Cohen's d (within-subject)")
    ax.set_title("Within-Subject Paired Comparison\n(pre-ictal minus interictal)")

    ax = axes[0, 1]
    if "error" not in level_test:
        labels = ["Pre-ictal", "Interictal"]
        means = [level_test["pre_mean"], level_test["inter_mean"]]
        stds = [level_test["pre_std"], level_test["inter_std"]]
        bars = ax.bar(labels, means, yerr=stds, color=["#5D3A9B", "#E66100"],
                      alpha=0.7, capsize=5)
        ax.set_ylabel("Subject-normalized min_spacing")
        ax.set_title(f"Raw Spacing Level Test\nd={level_test['cohens_d']:.3f}, "
                     f"p={level_test['mann_whitney_p']:.4f}")
    else:
        ax.text(0.5, 0.5, level_test.get("error", "error"), transform=ax.transAxes,
                ha="center")
        ax.set_title("Raw Spacing Level Test")

    ax = axes[0, 2]
    if "error" not in classifier_result:
        ax.text(0.05, 0.9, f"LOSO AUC: {classifier_result['auc']:.3f}",
                transform=ax.transAxes, fontsize=14, fontweight="bold")
        ax.text(0.05, 0.78, f"Accuracy: {classifier_result['accuracy']:.3f}",
                transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.66, f"Perm p-value: {classifier_result['p_value']:.3f}",
                transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.54, f"Sens@5%FAR: {classifier_result['sensitivity_at_5pct_far']:.3f}",
                transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.42, f"Sens@10%FAR: {classifier_result['sensitivity_at_10pct_far']:.3f}",
                transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.30, f"Null AUC: {classifier_result['null_auc_mean']:.3f}±{classifier_result['null_auc_std']:.3f}",
                transform=ax.transAxes, fontsize=12)
        ax.text(0.05, 0.18, f"N features: {classifier_result['n_features']}",
                transform=ax.transAxes, fontsize=10)
        ax.text(0.05, 0.06, f"N pre: {classifier_result['n_preictal']}, N sham: {classifier_result['n_sham']}",
                transform=ax.transAxes, fontsize=10)
        verdict = "POSITIVE" if classifier_result["auc"] > 0.65 and classifier_result["p_value"] < 0.05 else "NULL"
        color = "green" if verdict == "POSITIVE" else "red"
        ax.text(0.5, 0.95, verdict, transform=ax.transAxes, fontsize=18,
                fontweight="bold", ha="center", color=color)
    else:
        ax.text(0.5, 0.5, str(classifier_result), transform=ax.transAxes, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Subject-Norm LOSO Classifier")

    ax = axes[1, 0]
    if sprt_table and len(sz_df) > 0 and len(sh_df) > 0:
        total_sham_hours = sh_df["duration_sec"].sum() / 3600.0
        sh_max = sh_df["max_llr"].values
        thresholds = np.arange(0, 12, 0.1)
        det_r = [float(np.mean(sz_df.max_llr >= t)) for t in thresholds]
        fa_hr_r = [int(np.sum(sh_max >= t)) / total_sham_hours for t in thresholds]
        ax.plot(fa_hr_r, det_r, color="#5D3A9B", linewidth=2, label="Subj-norm")

        if old_sprt:
            old_det = [old_sprt.get(str(float(t)), {}).get("detection_rate", np.nan)
                       for t in [2, 3, 4, 5, 6, 7, 8]]
            old_fa = [old_sprt.get(str(float(t)), {}).get("fa_per_hour", np.nan)
                      for t in [2, 3, 4, 5, 6, 7, 8]]
            ax.plot(old_fa, old_det, "o--", color="#E66100", label="Per-event z (fair sham)")

        ax.axline((0, 0), (1, 1), color="gray", linestyle=":", alpha=0.3, label="Chance")
        ax.set_xlabel("FA/hour")
        ax.set_ylabel("Detection Rate")
        ax.set_title("SPRT: Subject-Norm vs Per-Event Z")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, "Insufficient SPRT data", transform=ax.transAxes, ha="center")
        ax.set_title("SPRT Comparison")

    ax = axes[1, 1]
    if sprt_table:
        thrs = sorted(sprt_table.keys(), key=float)
        det_vals = [sprt_table[t]["detection_rate"] for t in thrs]
        fa_vals = [sprt_table[t]["fa_per_hour"] for t in thrs]
        x = np.arange(len(thrs))
        width = 0.35
        ax.bar(x - width/2, det_vals, width, label="Detection", color="#5D3A9B", alpha=0.7)
        ax.bar(x + width/2, fa_vals, width, label="FA/hr", color="#E66100", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"LLR≥{t}" for t in thrs], fontsize=7, rotation=45)
        ax.set_ylabel("Rate")
        ax.set_title("SPRT Threshold Analysis (Subj-Norm)")
        ax.legend()
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)

    ax = axes[1, 2]
    if "error" not in classifier_result and classifier_result.get("top_features"):
        top = classifier_result["top_features"][:10]
        names = [t["feature"] for t in top]
        d_vals = [t["cohens_d"] for t in top]
        colors = ["#5D3A9B" if abs(d) > 0.5 else "#E66100" if abs(d) > 0.2 else "#999"
                  for d in d_vals]
        ax.barh(range(len(names)), d_vals, color=colors, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=7)
        ax.axvline(0.5, color="red", linestyle=":", alpha=0.4)
        ax.axvline(-0.5, color="red", linestyle=":", alpha=0.4)
        ax.set_xlabel("Cohen's d")
        ax.set_title("Top Classifier Features (Subj-Norm)")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "subject_norm_predictor.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    log("=" * 78)
    log("SUBJECT-LEVEL NORMALIZATION PREDICTOR TEST")
    log("=" * 78)

    df = pd.read_csv(RESULTS_DIR / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)

    log("\n--- Loading trajectory caches ---")
    seizure_trajs = load_trajectories(RESULTS_DIR / "trajectory_cache.npz")
    sham_trajs = load_trajectories(RESULTS_DIR / "fair_sham_cache.npz")
    log(f"  Seizure trajectories: {len(seizure_trajs)}")
    log(f"  Fair sham trajectories: {len(sham_trajs)}")

    log("\n--- Step A: Build subject-level normalization reference ---")
    sub_norms = build_subject_norms(sham_trajs)
    log(f"  Subjects with norms: {len(sub_norms)}")
    for sub in sorted(list(sub_norms.keys())[:5]):
        sp = sub_norms[sub]["min_spacing_raw"]
        log(f"    {sub}: raw spacing mean={sp['mean']:.6f} std={sp['std']:.6f} n={sp['n']}")

    log("\n--- Step B: Within-subject paired comparison ---")
    paired = within_subject_paired_test(seizure_trajs, sham_trajs, df)
    for fname, r in sorted(paired.items(), key=lambda x: -abs(x[1].get("cohens_d", 0))):
        sig = "*" if r["p_ttest"] < 0.05 else ""
        log(f"  {fname:25s}  d={r['cohens_d']:+.3f}  p={r['p_ttest']:.4f}  "
            f"n={r['n_subjects']}  {sig}")

    log("\n--- Step C: Raw spacing level test (subject-normalized) ---")
    level_test = raw_spacing_level_test(seizure_trajs, sham_trajs, df, sub_norms)
    if "error" not in level_test:
        log(f"  Pre-ictal mean: {level_test['pre_mean']:+.3f} ± {level_test['pre_std']:.3f}")
        log(f"  Interictal mean: {level_test['inter_mean']:+.3f} ± {level_test['inter_std']:.3f}")
        log(f"  Cohen's d: {level_test['cohens_d']:+.4f}")
        log(f"  AUC: {level_test['univariate_auc']:.3f}")
        log(f"  Mann-Whitney p: {level_test['mann_whitney_p']:.4f}")
    else:
        log(f"  {level_test}")

    log("\n--- Step D: LOSO classifier with subject-normalized features ---")
    classifier_result = loso_subject_norm_classifier(
        seizure_trajs, sham_trajs, df, sub_norms, seed=42)
    if "error" not in classifier_result:
        log(f"  AUC: {classifier_result['auc']:.3f}")
        log(f"  Accuracy: {classifier_result['accuracy']:.3f}")
        log(f"  Perm p-value: {classifier_result['p_value']:.3f}")
        log(f"  Sens@5%FAR: {classifier_result['sensitivity_at_5pct_far']:.3f}")
        log(f"  Sens@10%FAR: {classifier_result['sensitivity_at_10pct_far']:.3f}")
        log(f"  Null AUC: {classifier_result['null_auc_mean']:.3f}±{classifier_result['null_auc_std']:.3f}")
        log(f"  N features: {classifier_result['n_features']}")
        if classifier_result.get("top_features"):
            log(f"  Top features by |d|:")
            for tf in classifier_result["top_features"][:5]:
                log(f"    {tf['feature']:35s} d={tf['cohens_d']:+.3f} AUC={tf['uni_auc']:.3f}")
    else:
        log(f"  {classifier_result}")

    log("\n--- Step E: SPRT with subject-normalized features ---")
    sz_res, sh_res, sprt_feat_cols = run_subject_norm_sprt(
        seizure_trajs, sham_trajs, df, sub_norms, seed=42)
    sprt_table, sz_df, sh_df = compute_sprt_table(sz_res, sh_res)

    if sprt_table:
        total_sham_hours = sh_df["duration_sec"].sum() / 3600.0
        log(f"  Total sham monitoring: {total_sham_hours:.1f} hours")
        log(f"  {'Threshold':>10s} {'Det':>8s} {'FA/hr':>8s} {'Net':>8s}")
        log(f"  {'-'*36}")
        for thr_s, info in sorted(sprt_table.items(), key=lambda x: float(x[0])):
            net = info["detection_rate"] - info["fa_per_hour"]
            log(f"  LLR>={float(thr_s):5.1f} {info['detection_rate']:>8.3f} "
                f"{info['fa_per_hour']:>8.2f} {net:>+8.3f}")

    old_sprt = {}
    fair_path = RESULTS_DIR / "fair_sham_evaluation.json"
    if fair_path.exists():
        old_data = json.load(open(fair_path))
        old_sprt = old_data.get("sprt_results", {})

    log("\n--- Step F: Comparison with per-event z-score results ---")
    if old_sprt:
        log(f"  {'Threshold':>10s} {'Old Det':>9s} {'Old FA/hr':>10s} {'New Det':>9s} {'New FA/hr':>10s} {'Delta':>8s}")
        log(f"  {'-'*58}")
        for thr_s in sorted(old_sprt.keys(), key=float):
            old = old_sprt[thr_s]
            new = sprt_table.get(thr_s, {})
            if new:
                delta_det = new["detection_rate"] - old["detection_rate"]
                log(f"  LLR>={float(thr_s):5.1f} {old['detection_rate']:>9.3f} "
                    f"{old['fa_per_hour']:>10.2f} {new['detection_rate']:>9.3f} "
                    f"{new['fa_per_hour']:>10.2f} {delta_det:>+8.3f}")

    log("\n--- Step G: Verdict ---")
    if "error" not in classifier_result:
        auc = classifier_result["auc"]
        p = classifier_result["p_value"]
        if auc > 0.65 and p < 0.05:
            log("  VERDICT: Subject-level normalization reveals separability")
            log(f"  AUC={auc:.3f} (p={p:.3f})")
            log("  Proceeding to Snyder-criteria evaluation would be warranted")
        elif auc > 0.55:
            log("  VERDICT: Weak signal detected but not significant")
            log(f"  AUC={auc:.3f} (p={p:.3f})")
            log("  Insufficient evidence for prospective prediction")
        else:
            log("  VERDICT: NULL RESULT — subject-level normalization does not rescue prediction")
            log(f"  AUC={auc:.3f} (p={p:.3f})")
            log("  Pre-ictal dynamics are within normal interictal variability")
            log("  even when measured against a stable subject-level reference")
    else:
        log(f"  VERDICT: Could not evaluate — {classifier_result}")

    log("\n--- Saving results ---")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    make_figures(paired, level_test, classifier_result,
                 sprt_table, sz_df, sh_df, old_sprt)
    log(f"  Figures saved to {FIG_DIR}")

    norm_stats_export = {}
    for sub, norms in sub_norms.items():
        norm_stats_export[sub] = {
            field: {k: v for k, v in stats_dict.items()}
            for field, stats_dict in norms.items()
        }

    out = {
        "description": "Subject-level normalization predictor test",
        "normalization": "subject-level interictal reference from seizure-free recordings",
        "n_subjects_with_norms": len(sub_norms),
        "n_seizures": len(seizure_trajs),
        "n_fair_shams": len(sham_trajs),
        "within_subject_paired_test": paired,
        "raw_spacing_level_test": level_test,
        "loso_classifier": {k: v for k, v in classifier_result.items()
                            if k != "feature_cols"},
        "sprt_results": sprt_table,
        "comparison_with_per_event_z": {
            "per_event_z_best_auc": "see fair_sham_evaluation.json",
            "subject_norm_auc": classifier_result.get("auc", "N/A"),
        },
        "subject_norm_stats_sample": {
            sub: {
                "min_spacing_raw": sub_norms[sub]["min_spacing_raw"],
            }
            for sub in sorted(list(sub_norms.keys()))[:3]
        },
    }

    out_path = RESULTS_DIR / "subject_norm_predictor.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"  Results saved to {out_path}")

    log("\n" + "=" * 78)
    log("DONE")
    log("=" * 78)


if __name__ == "__main__":
    main()
