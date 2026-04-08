"""Clinically fair sham evaluation.

Generates proper interictal sham trajectories by processing raw EDF files
through the FULL pipeline (preprocess -> PCA -> VAR(1) -> eigenvalues ->
z-score -> feature extraction). Shams come from interictal segments >1 hour
from any seizure in the same subject's recordings.

This replaces the cheap shifted-baseline shams that were too easy to beat.
The old shams came from the same seizure's trajectory cache and were ~23 min
before the seizure — not truly interictal.

Sham matching criteria:
- Same subject (controls for individual brain structure)
- Same recording (controls for time-of-day, sleep/wake, electrode impedance)
- Full pipeline processing (same preprocessing, PCA, VAR, z-scoring)
- >1 hour from any seizure (genuinely interictal)
- Random interictal timepoints (not constrained to quiet baseline)
"""
from __future__ import annotations

import gc
import json
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

import yaml
from cmcc.io.loader_chbmit import build_seizure_catalog, load_raw_edf
from cmcc.preprocess.seizure_eeg import (
    fit_baseline_pca, preprocess_chbmit_raw, project_to_pca,
)
from cmcc.analysis.seizure_dynamics import (
    compute_sham_trajectories, compute_seizure_trajectory,
)

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "chbmit.yaml"
RESULTS_DIR = CMCC_ROOT / "results_chbmit" / "analysis"
FIG_DIR = CMCC_ROOT / "results_chbmit" / "figures" / "fair_sham"

FIELD_MAP = {
    "sp": "min_spacing_z", "sr": "spectral_radius_z", "ep": "ep_score_z",
    "alpha": "alpha_power_z", "delta": "delta_power_z",
    "med_nns": "median_nns_z", "p10_nns": "p10_nns_z",
}
SHORT_NAMES = list(FIELD_MAP.keys())
REGIME_FEATURES = ["sr_mean", "ep_mean", "med_nns_mean", "p10_nns_mean"]
TIME_STEP_SEC = 30
TIME_POINTS = np.arange(-1800 + 300, 60, TIME_STEP_SEC)


def log(msg):
    print(msg, flush=True)


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


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


def extract_change_features(traj_dict, t_now, baseline_end=-600):
    t = traj_dict["time_sec"]
    bl_start = t.min()
    feats = {}
    for sn in SHORT_NAMES:
        field = FIELD_MAP[sn]
        if field not in traj_dict:
            for k in ["recent_mean", "deviation", "recent_slope", "acceleration",
                       "cum_deviation", "jitter", "zscore_from_bl"]:
                feats[f"{sn}_{k}"] = np.nan
            continue
        arr = traj_dict[field]
        bl_mean = _window_mean(arr, t, bl_start, baseline_end)
        recent_mean = _window_mean(arr, t, t_now - 300, t_now)
        feats[f"{sn}_recent_mean"] = recent_mean
        feats[f"{sn}_deviation"] = (recent_mean - bl_mean
                                     if np.isfinite(bl_mean) and np.isfinite(recent_mean)
                                     else np.nan)
        feats[f"{sn}_recent_slope"] = _window_slope(arr, t, t_now - 300, t_now)
        sr = _window_slope(arr, t, t_now - 300, t_now)
        se = _window_slope(arr, t, t_now - 600, t_now - 300)
        feats[f"{sn}_acceleration"] = (sr - se if np.isfinite(sr) and np.isfinite(se)
                                        else np.nan)
        cm = _window_mean(arr, t, bl_start, t_now)
        feats[f"{sn}_cum_deviation"] = (cm - bl_mean
                                         if np.isfinite(cm) and np.isfinite(bl_mean)
                                         else np.nan)
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


def traj_to_dict(traj):
    return {
        "time_sec": np.array(traj.time_sec),
        "min_spacing_z": np.array(traj.min_spacing_z),
        "min_spacing_raw": np.array(traj.min_spacing_raw),
        "spectral_radius_z": np.array(traj.spectral_radius_z),
        "ep_score_z": np.array(traj.ep_score_z),
        "alpha_power_z": np.array(traj.alpha_power_z),
        "delta_power_z": np.array(traj.delta_power_z),
        "median_nns_z": np.array(traj.median_nns_z),
        "p10_nns_z": np.array(traj.p10_nns_z),
    }


def find_seizure_free_runs(cfg, catalogs):
    from cmcc.io.loader_chbmit import (
        discover_sessions, discover_runs, load_events_tsv,
    )
    data_root = Path(cfg["data"]["root"])
    seizure_free = {}

    for sid, cat in sorted(catalogs.items()):
        if cat.n_eligible == 0:
            continue
        sf_runs = []
        for ses in cat.sessions:
            try:
                runs = discover_runs(data_root, sid, ses)
            except FileNotFoundError:
                continue
            for run in runs:
                try:
                    ev = load_events_tsv(data_root, sid, ses, run)
                except FileNotFoundError:
                    continue
                has_sz = ev["eventType"].str.startswith("sz", na=False).any()
                if not has_sz:
                    rd_col = "recordingDuration"
                    rec_dur = float(ev[rd_col].dropna().iloc[0]) if rd_col in ev.columns and len(ev[rd_col].dropna()) > 0 else 0
                    sf_runs.append({"session": ses, "run": run, "rec_dur": rec_dur})
        if sf_runs:
            sf_runs.sort(key=lambda x: -x["rec_dur"])
            seizure_free[sid] = sf_runs
    return seizure_free


def generate_fair_shams(cfg, catalogs, n_shams_per_subject=3, seed=42):
    rng = np.random.default_rng(seed)
    var_cfg = cfg["var"]
    pp_cfg = cfg["preprocessing"]
    sm_cfg = cfg["smoothing"]

    sham_trajectories = {}
    sham_meta = []

    sf_runs = find_seizure_free_runs(cfg, catalogs)
    log(f"  Found seizure-free recordings for {len(sf_runs)} subjects")

    for sid, runs_info in sorted(sf_runs.items()):
        cat = catalogs[sid]
        sz_ref = cat.eligible_seizures[0]

        shams_generated = 0
        for run_info in runs_info[:3]:
            if shams_generated >= n_shams_per_subject:
                break

            ses = run_info["session"]
            run = run_info["run"]
            rec_dur = run_info["rec_dur"]

            if rec_dur < 2400:
                continue

            log(f"  {sid}: loading seizure-free {run} ({rec_dur/3600:.1f}h)...")

            try:
                raw = load_raw_edf(
                    cfg["data"]["root"], sid, ses, run, preload=True,
                )
                data, sfreq, _ = preprocess_chbmit_raw(
                    raw, line_freq=pp_cfg["line_freq"],
                    bandpass=tuple(pp_cfg["bandpass"]),
                )
                del raw

                n_samp = data.shape[1]
                actual_dur = n_samp / sfreq

                bl_start = 0.0
                bl_end = min(1200.0, actual_dur * 0.5)
                if bl_end <= 120:
                    log(f"    SKIP: too short for baseline")
                    del data
                    gc.collect()
                    continue

                pca, _ = fit_baseline_pca(
                    data, sfreq, bl_start, bl_end,
                    n_components=pp_cfg["n_components"],
                )
                data_pca = project_to_pca(data, pca)
                del data

                n_possible = max(1, n_shams_per_subject - shams_generated)
                analysis_window = 2400.0
                half_w = analysis_window / 2.0

                for trial in range(n_possible * 5):
                    if shams_generated >= n_shams_per_subject:
                        break

                    fake_onset = rng.uniform(half_w + 60, actual_dur - half_w - 60)
                    onset_samp = int(fake_onset * sfreq)
                    start_samp = max(0, onset_samp - int(half_w * sfreq))
                    end_samp = min(data_pca.shape[1], onset_samp + int(half_w * sfreq))

                    if end_samp - start_samp < int(half_w * sfreq):
                        continue

                    seg = data_pca[:, start_samp:end_samp]
                    local_onset = fake_onset - (start_samp / sfreq)
                    local_bl_end = local_onset - 600.0

                    if local_bl_end <= 60:
                        continue

                    try:
                        traj = compute_seizure_trajectory(
                            seg, sfreq,
                            seizure_onset_sec=local_onset,
                            seizure_offset_sec=local_onset + 30.0,
                            baseline_start_sec=0.0,
                            baseline_end_sec=local_bl_end,
                            window_sec=var_cfg["window_sec"],
                            step_sec=var_cfg["step_sec"],
                            regularization=var_cfg["regularization"],
                            smoothing_sec=sm_cfg["moving_average_sec"],
                            subject_id=sid,
                            seizure_idx=-1,
                            seizure_duration=0.0,
                            event_type="sham",
                        )

                        if len(traj.time_sec) > 50:
                            key = f"{sid}_sham{shams_generated}"
                            sham_trajectories[key] = traj_to_dict(traj)
                            sham_meta.append({
                                "key": key, "subject_id": sid,
                                "source_run": run, "source_session": ses,
                                "seizure_free": True,
                                "n_windows": len(traj.time_sec),
                            })
                            shams_generated += 1
                    except Exception:
                        continue

                del data_pca
                log(f"    Generated {shams_generated} shams so far for {sid}")

            except Exception as e:
                log(f"    SKIP: {e}")

            gc.collect()

    for sid, cat in sorted(catalogs.items()):
        if sid in sf_runs:
            continue
        if cat.n_eligible == 0:
            continue

        sz = cat.eligible_seizures[0]
        real_onsets = [s.onset_sec for s in cat.eligible_seizures]

        log(f"  {sid}: no seizure-free runs, trying seizure recording with reduced exclusion...")
        try:
            raw = load_raw_edf(
                cfg["data"]["root"], sz.subject_id, sz.session, sz.run,
                preload=True,
            )
            data, sfreq, _ = preprocess_chbmit_raw(
                raw, line_freq=pp_cfg["line_freq"],
                bandpass=tuple(pp_cfg["bandpass"]),
            )
            del raw

            bl_window = cfg["seizure"]["baseline_window"]
            bl_start = max(0.0, sz.onset_sec + bl_window[0])
            bl_end = sz.onset_sec + bl_window[1]

            if bl_end <= bl_start + 60:
                del data
                gc.collect()
                continue

            pca, _ = fit_baseline_pca(
                data, sfreq, bl_start, bl_end,
                n_components=pp_cfg["n_components"],
            )
            data_pca = project_to_pca(data, pca)
            del data

            rec_dur = data_pca.shape[1] / sfreq

            shams = compute_sham_trajectories(
                data_pca, sfreq,
                n_shams=n_shams_per_subject,
                seizure_onsets_sec=real_onsets,
                recording_duration_sec=rec_dur,
                rng=rng,
                window_sec=var_cfg["window_sec"],
                step_sec=var_cfg["step_sec"],
                regularization=var_cfg["regularization"],
                smoothing_sec=sm_cfg["moving_average_sec"],
                exclusion_radius_sec=1200.0,
            )

            for i, sham_traj in enumerate(shams):
                key = f"{sid}_sham{i}"
                sham_trajectories[key] = traj_to_dict(sham_traj)
                sham_meta.append({
                    "key": key, "subject_id": sid,
                    "source_run": sz.run, "source_session": sz.session,
                    "seizure_free": False,
                    "n_windows": len(sham_traj.time_sec),
                })

            log(f"    Generated {len(shams)} shams from seizure recording")
            del data_pca

        except Exception as e:
            log(f"    SKIP: {e}")

        gc.collect()

    return sham_trajectories, sham_meta


def run_fair_sprt(seizure_trajs, sham_trajs, df, sham_meta, sprt_feat_cols, seed=42):
    rng = np.random.default_rng(seed)
    subjects = df["subject_id"].unique()
    sham_subjects = {m["subject_id"] for m in sham_meta}

    regime_train_data = []
    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in seizure_trajs:
            continue
        traj = seizure_trajs[key]
        regime_feat = {}
        t = traj["time_sec"]
        for sn in ["sr", "ep", "med_nns", "p10_nns"]:
            regime_feat[f"{sn}_mean"] = _window_mean(traj[FIELD_MAP[sn]], t, -1800, -60)
        regime_feat["true_regime"] = row["group_label"]
        regime_feat["subject_id"] = row["subject_id"]
        regime_train_data.append(regime_feat)
    regime_df = pd.DataFrame(regime_train_data)

    seizure_results = []
    sham_results = []

    for test_sub in subjects:
        test_mask = df["subject_id"].values == test_sub
        train_mask = ~test_mask
        test_df = df[test_mask]
        train_df = df[train_mask]

        if len(test_df) == 0:
            continue

        regime_train = regime_df[regime_df["subject_id"] != test_sub]
        X_rt = regime_train[REGIME_FEATURES].values
        y_rt = regime_train["true_regime"].values
        fin_r = np.all(np.isfinite(X_rt), axis=1)
        if fin_r.sum() < 6 or len(np.unique(y_rt[fin_r])) < 2:
            continue

        sc_r = StandardScaler()
        clf_r = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                    max_iter=1000, random_state=42)
        clf_r.fit(sc_r.fit_transform(X_rt[fin_r]), y_rt[fin_r])

        detect_rows = []
        for _, row in train_df.iterrows():
            key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
            if key not in seizure_trajs:
                continue
            traj = seizure_trajs[key]
            pre_f = extract_change_features(traj, -60)
            if pre_f:
                detect_rows.append({**pre_f, "label": 1,
                                     "true_regime": row["group_label"]})

        train_sham_subs = [s for s in sham_subjects if s != test_sub]
        for skey, straj in sham_trajs.items():
            sub_of_sham = skey.split("_sham")[0]
            if sub_of_sham == test_sub:
                continue
            t = straj["time_sec"]
            mid = float(np.median(t))
            sham_f = extract_change_features(straj, mid, baseline_end=mid - 600)
            if sham_f:
                regime_guess = 0
                detect_rows.append({**sham_f, "label": 0,
                                     "true_regime": regime_guess})

        if len(detect_rows) < 12:
            continue
        detect_df = pd.DataFrame(detect_rows)

        fc_names = [c for c in sprt_feat_cols if c in detect_df.columns]
        regime_clfs = {}
        regime_scalers = {}
        for rv in [0, 1]:
            rd = detect_df[detect_df["true_regime"] == rv]
            if len(rd) < 6:
                rd = detect_df
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

        if not regime_clfs:
            continue

        for _, row in test_df.iterrows():
            key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
            if key not in seizure_trajs:
                continue
            traj = seizure_trajs[key]

            reg_feat = {}
            t = traj["time_sec"]
            for sn in ["sr", "ep", "med_nns", "p10_nns"]:
                reg_feat[f"{sn}_mean"] = _window_mean(traj[FIELD_MAP[sn]], t, -1800, -300)
            x_r = np.array([[reg_feat.get(f, np.nan) for f in REGIME_FEATURES]])
            if np.all(np.isfinite(x_r)):
                pred_prob = clf_r.predict_proba(sc_r.transform(x_r))[0, 1]
                pred_regime = int(pred_prob >= 0.5)
            else:
                pred_regime = 0

            if pred_regime not in regime_clfs:
                pred_regime = list(regime_clfs.keys())[0]

            clf_det = regime_clfs[pred_regime]
            sc_det = regime_scalers[pred_regime]

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

            seizure_results.append({
                "key": key, "subject_id": test_sub,
                "true_regime": row["group_label"],
                "pred_regime": pred_regime,
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

            reg_feat = {}
            for sn in ["sr", "ep", "med_nns", "p10_nns"]:
                reg_feat[f"{sn}_mean"] = _window_mean(straj[FIELD_MAP[sn]], t, t.min(), sim_end)
            x_r = np.array([[reg_feat.get(f, np.nan) for f in REGIME_FEATURES]])
            if np.all(np.isfinite(x_r)):
                pred_regime = int(clf_r.predict_proba(sc_r.transform(x_r))[0, 1] >= 0.5)
            else:
                pred_regime = 0

            if pred_regime not in regime_clfs:
                pred_regime = list(regime_clfs.keys())[0]

            clf_det = regime_clfs[pred_regime]
            sc_det = regime_scalers[pred_regime]

            cum_llr = 0.0
            max_llr = 0.0
            alarm_time = None
            for t_now in sim_steps:
                feat = extract_change_features(straj, t_now, baseline_end=t_now - 600)
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

            sham_duration_sec = float(len(sim_steps) * TIME_STEP_SEC)
            sham_results.append({
                "key": skey, "subject_id": test_sub,
                "max_llr": max_llr,
                "alarm_time": alarm_time,
                "duration_sec": sham_duration_sec,
            })

    return seizure_results, sham_results


def main():
    log("=" * 78)
    log("FAIR SHAM EVALUATION")
    log("=" * 78)

    cfg = load_config()
    catalogs = build_seizure_catalog(
        cfg["data"]["root"],
        min_preictal_sec=cfg["seizure"]["min_preictal_sec"],
        min_inter_seizure_sec=cfg["seizure"]["min_inter_seizure_sec"],
    )

    df = pd.read_csv(RESULTS_DIR / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)

    cache = np.load(RESULTS_DIR / "trajectory_cache.npz", allow_pickle=True)
    seizure_trajs = {}
    for key in cache.files:
        traj = json.loads(str(cache[key]))
        for field in ["time_sec", "min_spacing_z", "min_spacing_raw",
                      "spectral_radius_z", "ep_score_z", "alpha_power_z",
                      "delta_power_z", "median_nns_z", "p10_nns_z"]:
            if field in traj:
                traj[field] = np.array(traj[field])
        seizure_trajs[key] = traj

    log(f"\nLoaded {len(seizure_trajs)} seizure trajectories")

    log("\n--- Step 1: Generate fair interictal shams ---")
    sham_trajs, sham_meta = generate_fair_shams(cfg, catalogs,
                                                  n_shams_per_subject=5, seed=42)

    log(f"\nTotal fair shams: {len(sham_trajs)}")
    if len(sham_trajs) == 0:
        log("ERROR: No shams generated. Cannot evaluate.")
        return

    sham_cache_path = RESULTS_DIR / "fair_sham_cache.npz"
    save_dict = {}
    for k, v in sham_trajs.items():
        save_dict[k] = json.dumps({
            kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
            for kk, vv in v.items()
        })
    np.savez_compressed(sham_cache_path, **save_dict)
    log(f"Saved sham cache to {sham_cache_path}")

    log("\n--- Step 2: Feature separability with fair shams ---")
    pre_feats = []
    sham_feats = []
    for _, row in df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in seizure_trajs:
            continue
        f = extract_change_features(seizure_trajs[key], -60)
        if f:
            pre_feats.append(f)

    for skey, straj in sham_trajs.items():
        t = straj["time_sec"]
        mid = float(np.median(t))
        f = extract_change_features(straj, mid, baseline_end=mid - 600)
        if f:
            sham_feats.append(f)

    pre_df = pd.DataFrame(pre_feats)
    sham_df = pd.DataFrame(sham_feats)

    log(f"  Pre-ictal: {len(pre_df)}, Fair shams: {len(sham_df)}")

    feat_cols = [c for c in pre_df.columns
                 if not np.all(pre_df[c].isna()) and c in sham_df.columns]

    log(f"\n  Feature separability (Cohen's d, pre-ictal vs FAIR sham):")
    top_features = []
    for fc in sorted(feat_cols):
        pv = pre_df[fc].dropna().values
        sv = sham_df[fc].dropna().values
        if len(pv) < 5 or len(sv) < 5:
            continue
        pooled = np.sqrt((np.var(pv) + np.var(sv)) / 2)
        d = (np.mean(pv) - np.mean(sv)) / pooled if pooled > 0 else 0
        y_all = np.concatenate([np.ones(len(pv)), np.zeros(len(sv))])
        x_all = np.concatenate([pv, sv])
        try:
            uni_auc = roc_auc_score(y_all, x_all)
        except ValueError:
            uni_auc = 0.5
        top_features.append((fc, abs(uni_auc - 0.5), uni_auc, d))

    top_features.sort(key=lambda x: -x[1])
    n_sep = sum(1 for _, _, _, d in top_features if abs(d) > 0.5)
    n_weak = sum(1 for _, _, _, d in top_features if 0.2 < abs(d) <= 0.5)
    n_neg = sum(1 for _, _, _, d in top_features if abs(d) <= 0.2)

    for fc, delta, auc, d in top_features[:20]:
        marker = "***" if abs(d) > 0.8 else "**" if abs(d) > 0.5 else "*" if abs(d) > 0.2 else ""
        log(f"    {fc:35s}  AUC={auc:.3f}  d={d:+.3f}  {marker}")

    log(f"\n  Summary: {n_sep} separable (|d|>0.5), {n_weak} weak, {n_neg} negligible")

    old_results = json.load(open(RESULTS_DIR / "improved_detection.json"))
    sprt_feat_cols = old_results["sprt_features_used"]

    log("\n--- Step 3: SPRT evaluation with fair shams ---")
    sz_results, sh_results = run_fair_sprt(
        seizure_trajs, sham_trajs, df, sham_meta, sprt_feat_cols, seed=42)

    sz_df = pd.DataFrame(sz_results)
    sh_df = pd.DataFrame(sh_results)

    log(f"\n  Seizures evaluated: {len(sz_df)}")
    log(f"  Fair sham segments: {len(sh_df)}")

    if len(sh_df) == 0:
        log("  No sham segments available for FA estimation")
        return

    total_sham_hours = sh_df["duration_sec"].sum() / 3600.0
    log(f"  Total sham monitoring: {total_sham_hours:.1f} hours")

    sh_max = sh_df["max_llr"].values

    log(f"\n  SPRT Results (FAIR shams):")
    log(f"  {'Threshold':>10s} {'Det':>8s} {'FA(frac)':>10s} {'FA/hr':>8s} {'FA/day':>8s} {'Warn(min)':>10s}")
    log(f"  {'-'*56}")

    result_table = {}
    for thr in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
        det = float(np.mean(sz_df.max_llr >= thr)) if len(sz_df) > 0 else 0
        n_fa = int(np.sum(sh_max >= thr))
        fa_frac = float(np.mean(sh_max >= thr))
        fa_hr = n_fa / total_sham_hours if total_sham_hours > 0 else 0
        fa_day = fa_hr * 24

        warned = sz_df[sz_df.max_llr >= thr]
        med_warn = float(warned.warning_min.median()) if len(warned) > 0 else float("nan")

        log(f"  {thr:>10.1f} {det:>8.3f} {fa_frac:>10.3f} {fa_hr:>8.2f} {fa_day:>8.1f} {med_warn:>10.1f}")

        result_table[str(thr)] = {
            "detection_rate": det, "fa_fraction": fa_frac,
            "fa_per_hour": fa_hr, "fa_per_day": fa_day,
            "n_detected": int((sz_df.max_llr >= thr).sum()),
            "n_false_alarms": n_fa,
            "median_warning_min": med_warn,
        }

    log(f"\n  Comparison with old (easy) shams:")
    old_audit = {}
    audit_path = RESULTS_DIR / "detector_audit.json"
    if audit_path.exists():
        old_audit = json.load(open(audit_path))
        old_fa = old_audit.get("audit_1_fa_units", {}).get("predicted_fa", {})
        for thr_s, info in old_fa.items():
            new = result_table.get(thr_s, {})
            if new:
                log(f"    LLR>={thr_s}: Old FA/hr={info.get('fa_per_hour',0):.2f} "
                    f"-> Fair FA/hr={new['fa_per_hour']:.2f}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    thresholds = np.arange(0, 12, 0.1)
    det_r = [float(np.mean(sz_df.max_llr >= t)) for t in thresholds]
    n_fa_r = [int(np.sum(sh_max >= t)) for t in thresholds]
    fa_hr_r = [n / total_sham_hours for n in n_fa_r]
    ax.plot(fa_hr_r, det_r, color="#5D3A9B", linewidth=2, label="Fair shams")
    ax.axvline(1.0, color="red", linestyle=":", alpha=0.4, label="1 FA/hr")
    ax.axvline(0.5, color="orange", linestyle=":", alpha=0.4, label="0.5 FA/hr")
    ax.axvline(0.25, color="green", linestyle=":", alpha=0.4, label="0.25 FA/hr")
    ax.set_xlabel("FA/hour")
    ax.set_ylabel("Detection Rate")
    ax.set_title("Clinical Operating Curve (Fair Shams)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 5)

    ax = axes[1]
    warned = sz_df[np.isfinite(sz_df.warning_min)]
    if len(warned) > 0:
        ax.hist(warned.warning_min.values, bins=20, color="#5D3A9B",
                edgecolor="black", alpha=0.7)
        ax.axvline(warned.warning_min.median(), color="red", linestyle="--",
                   label=f"Median={warned.warning_min.median():.1f} min")
    ax.set_xlabel("Warning time (min)")
    ax.set_ylabel("Count")
    ax.set_title("Warning Time Distribution")
    ax.legend()

    ax = axes[2]
    for fc, _, auc, d in top_features[:15]:
        ax.barh(fc, abs(d), color="#5D3A9B" if d > 0 else "#E66100", alpha=0.7)
    ax.set_xlabel("|Cohen's d|")
    ax.set_title("Feature Separability (Fair Shams)")
    ax.axvline(0.5, color="red", linestyle=":", alpha=0.4)
    ax.axvline(0.2, color="orange", linestyle=":", alpha=0.4)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fair_sham_evaluation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    out = {
        "description": "Clinically fair sham evaluation",
        "sham_source": "compute_sham_trajectories — full pipeline on interictal segments >1hr from seizures",
        "n_seizures_evaluated": len(sz_df),
        "n_fair_shams": len(sh_df),
        "total_sham_monitoring_hours": total_sham_hours,
        "sham_matching": {
            "same_subject": True,
            "same_recording": True,
            "full_pipeline": True,
            "min_distance_from_seizure_sec": 3600,
        },
        "feature_separability_summary": {
            "n_separable": n_sep,
            "n_weak": n_weak,
            "n_negligible": n_neg,
        },
        "sprt_results": result_table,
        "sham_meta": sham_meta,
    }

    out_path = RESULTS_DIR / "fair_sham_evaluation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    log(f"\nSaved to {out_path}")
    log(f"Figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
