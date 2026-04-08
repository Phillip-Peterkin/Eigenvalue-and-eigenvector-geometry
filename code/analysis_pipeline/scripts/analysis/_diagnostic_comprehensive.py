"""Comprehensive diagnostic: 6-point investigation of spacing sign reversal.

1. Baseline normalization comparison (same seizures, both configs)
2. Raw + z-scored + spectral dump for 5 seizures
3. Paired per-seizure comparison (pre-ictal slope, timing, raw means)
4. One-change-at-a-time parameter grid on 10 seizures
5. Condition number, residual variance, near-singularity fraction
6. Onset spike artifact investigation (channel saturation, spectral redistribution)
"""
from __future__ import annotations

import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

import yaml
from cmcc.io.loader_chbmit import build_seizure_catalog, load_raw_edf
from cmcc.preprocess.seizure_eeg import (
    fit_baseline_pca, preprocess_chbmit_raw, project_to_pca,
)
from cmcc.analysis.seizure_dynamics import (
    compute_seizure_trajectory, compute_preictal_slope,
)
from cmcc.analysis.dynamical_systems import estimate_jacobian

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "chbmit.yaml"
FIG_DIR = CMCC_ROOT / "results_chbmit" / "figures" / "diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)


PARAM_GRID = [
    {"label": "w0.5_c15_r1e-4", "window": 0.5, "step": 0.1, "n_comp": 15, "reg": 1e-4},
    {"label": "w2.0_c15_r1e-4", "window": 2.0, "step": 0.5, "n_comp": 15, "reg": 1e-4},
    {"label": "w0.5_c10_r1e-4", "window": 0.5, "step": 0.1, "n_comp": 10, "reg": 1e-4},
    {"label": "w2.0_c10_r1e-4", "window": 2.0, "step": 0.5, "n_comp": 10, "reg": 1e-4},
    {"label": "w0.5_c15_r1e-3", "window": 0.5, "step": 0.1, "n_comp": 15, "reg": 1e-3},
    {"label": "w2.0_c15_r1e-3", "window": 2.0, "step": 0.5, "n_comp": 15, "reg": 1e-3},
    {"label": "w0.5_c10_r1e-3", "window": 0.5, "step": 0.1, "n_comp": 10, "reg": 1e-3},
    {"label": "w2.0_c10_r1e-3", "window": 2.0, "step": 0.5, "n_comp": 10, "reg": 1e-3},
]


def compute_trajectory_and_jacobian(data, sfreq, sz, bl_start, bl_end, p):
    pca, _ = fit_baseline_pca(data, sfreq, bl_start, bl_end, n_components=p["n_comp"])
    data_pca = project_to_pca(data, pca)

    traj = compute_seizure_trajectory(
        data_pca, sfreq,
        seizure_onset_sec=sz.onset_sec,
        seizure_offset_sec=sz.offset_sec,
        baseline_start_sec=bl_start,
        baseline_end_sec=bl_end,
        window_sec=p["window"],
        step_sec=p["step"],
        regularization=p["reg"],
        smoothing_sec=30.0,
        subject_id=sz.subject_id,
    )

    ch_mean = data_pca.mean(axis=1, keepdims=True)
    ch_std = data_pca.std(axis=1, keepdims=True)
    ch_std[ch_std == 0] = 1.0
    data_z = (data_pca - ch_mean) / ch_std

    window_samples = max(int(p["window"] * sfreq), p["n_comp"] + 10)
    step_samples = max(1, int(p["step"] * sfreq))

    jac = estimate_jacobian(
        data_z, window_size=window_samples,
        step_size=step_samples, regularization=p["reg"],
    )

    del data_pca, data_z
    return traj, jac


def extract_period_stats(traj, jac, sfreq):
    bl_mask = (traj.time_sec >= -1800) & (traj.time_sec < -600)
    pre_mask = (traj.time_sec >= -600) & (traj.time_sec < 0)
    peri_mask = (traj.time_sec >= -60) & (traj.time_sec < 0)
    ictal_mask = (traj.time_sec >= 0) & (traj.time_sec < traj.seizure_duration)
    post_mask = (traj.time_sec >= traj.seizure_duration + 120)

    jac_n = len(jac.spectral_radius)

    def safe_mean(arr, mask):
        m = mask[:len(arr)] if len(mask) > len(arr) else mask
        vals = arr[m]
        return float(np.nanmean(vals)) if len(vals) > 0 else float("nan")

    def safe_std(arr, mask):
        m = mask[:len(arr)] if len(mask) > len(arr) else mask
        vals = arr[m]
        return float(np.nanstd(vals)) if len(vals) > 0 else float("nan")

    def safe_frac_above(arr, mask, threshold):
        m = mask[:len(arr)] if len(mask) > len(arr) else mask
        vals = arr[m]
        if len(vals) == 0:
            return float("nan")
        return float(np.mean(vals > threshold))

    min_time = float("nan")
    analysis_mask = (traj.time_sec >= -600) & (traj.time_sec <= 300)
    if analysis_mask.sum() > 5:
        s = traj.min_spacing_z[analysis_mask]
        valid = np.isfinite(s)
        if valid.sum() > 3:
            idx = np.nanargmin(s)
            min_time = float(traj.time_sec[analysis_mask][idx])

    return {
        "bl_raw_mean": safe_mean(traj.min_spacing_raw, bl_mask),
        "bl_raw_std": safe_std(traj.min_spacing_raw, bl_mask),
        "pre_raw_mean": safe_mean(traj.min_spacing_raw, pre_mask),
        "peri_raw_mean": safe_mean(traj.min_spacing_raw, peri_mask),
        "ictal_raw_mean": safe_mean(traj.min_spacing_raw, ictal_mask),
        "post_raw_mean": safe_mean(traj.min_spacing_raw, post_mask),
        "pre_z_mean": safe_mean(traj.min_spacing_z, pre_mask),
        "peri_z_mean": safe_mean(traj.min_spacing_z, peri_mask),
        "bl_z_mean": safe_mean(traj.min_spacing_z, bl_mask),
        "pre_spec_radius": safe_mean(traj.spectral_radius_z, pre_mask),
        "pre_alpha_z": safe_mean(traj.alpha_power_z, pre_mask),
        "pre_delta_z": safe_mean(traj.delta_power_z, pre_mask),
        "slope": traj.preictal_slope,
        "min_time_sec": min_time,
        "baseline_mean": traj.baseline_mean,
        "baseline_std": traj.baseline_std,
        "cond_bl_median": safe_mean(jac.condition_numbers, bl_mask),
        "cond_pre_median": safe_mean(jac.condition_numbers, pre_mask),
        "resid_bl": safe_mean(jac.residual_variance, bl_mask),
        "resid_pre": safe_mean(jac.residual_variance, pre_mask),
        "resid_ictal": safe_mean(jac.residual_variance, ictal_mask),
        "spec_radius_bl": safe_mean(jac.spectral_radius, bl_mask),
        "spec_radius_pre": safe_mean(jac.spectral_radius, pre_mask),
        "spec_radius_ictal": safe_mean(jac.spectral_radius, ictal_mask),
        "frac_cond_above_100_bl": safe_frac_above(jac.condition_numbers, bl_mask, 100),
        "frac_cond_above_100_pre": safe_frac_above(jac.condition_numbers, pre_mask, 100),
        "n_windows": len(traj.time_sec),
    }


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    log("Building catalog...")
    catalogs = build_seizure_catalog(
        cfg["data"]["root"],
        min_preictal_sec=cfg["seizure"]["min_preictal_sec"],
        min_inter_seizure_sec=cfg["seizure"]["min_inter_seizure_sec"],
    )

    target_subjects = ["sub-01", "sub-03", "sub-04", "sub-05", "sub-08",
                       "sub-10", "sub-12", "sub-16", "sub-17", "sub-22"]
    bl_window = cfg["seizure"]["baseline_window"]

    all_grid_results = {}

    for sid in target_subjects:
        cat = catalogs[sid]
        if cat.n_eligible == 0:
            continue
        sz = cat.eligible_seizures[0]
        log(f"\n{'='*60}")
        log(f"{sid}: onset={sz.onset_sec:.0f}s, dur={sz.duration_sec:.0f}s")
        log(f"{'='*60}")

        try:
            raw = load_raw_edf(cfg["data"]["root"], sz.subject_id, sz.session, sz.run, preload=True)
            data, sfreq, _ = preprocess_chbmit_raw(raw, line_freq=60.0, bandpass=(0.5, 45.0))
            del raw
        except Exception as e:
            log(f"  SKIP load: {e}")
            continue

        bl_start = max(0.0, sz.onset_sec + bl_window[0])
        bl_end = sz.onset_sec + bl_window[1]

        seizure_results = {}

        for p in PARAM_GRID:
            try:
                traj, jac = compute_trajectory_and_jacobian(data, sfreq, sz, bl_start, bl_end, p)
                stats = extract_period_stats(traj, jac, sfreq)
                seizure_results[p["label"]] = stats

                raw_dir = "WIDER" if stats["pre_raw_mean"] > stats["bl_raw_mean"] else "NARROWER"
                log(f"\n  {p['label']}:")
                log(f"    bl_raw={stats['bl_raw_mean']:.6f} (std={stats['bl_raw_std']:.6f})")
                log(f"    pre_raw={stats['pre_raw_mean']:.6f} => {raw_dir}")
                log(f"    z_pre={stats['pre_z_mean']:+.4f}, slope={stats['slope']:+.6f}")
                log(f"    baseline_mean={stats['baseline_mean']:.6f}, baseline_std={stats['baseline_std']:.6f}")
                log(f"    cond_bl={stats['cond_bl_median']:.1f}, cond_pre={stats['cond_pre_median']:.1f}")
                log(f"    resid_bl={stats['resid_bl']:.4f}, resid_pre={stats['resid_pre']:.4f}")
                log(f"    frac_cond>100: bl={stats['frac_cond_above_100_bl']:.3f}, pre={stats['frac_cond_above_100_pre']:.3f}")
                log(f"    spec_radius: bl={stats['spec_radius_bl']:.4f}, pre={stats['spec_radius_pre']:.4f}, ictal={stats['spec_radius_ictal']:.4f}")
                log(f"    min_time={stats['min_time_sec']:.1f}s")

                del traj, jac
                gc.collect()

            except Exception as e:
                log(f"  {p['label']}: SKIP — {e}")

        all_grid_results[sid] = seizure_results
        del data
        gc.collect()

    log("\n\n" + "=" * 80)
    log("ANALYSIS 1: BASELINE NORMALIZATION COMPARISON")
    log("=" * 80)
    old_key = "w0.5_c15_r1e-4"
    new_key = "w2.0_c10_r1e-3"
    for sid, res in sorted(all_grid_results.items()):
        if old_key in res and new_key in res:
            o = res[old_key]
            n = res[new_key]
            log(f"\n{sid}:")
            log(f"  OLD bl_mean={o['baseline_mean']:.6f}, bl_std={o['baseline_std']:.6f}")
            log(f"  NEW bl_mean={n['baseline_mean']:.6f}, bl_std={n['baseline_std']:.6f}")
            log(f"  bl_mean ratio (new/old): {n['baseline_mean']/o['baseline_mean']:.3f}" if o['baseline_mean'] != 0 else "  bl_mean: old=0")
            log(f"  bl_std ratio (new/old):  {n['baseline_std']/o['baseline_std']:.3f}" if o['baseline_std'] != 0 else "  bl_std: old=0")

    log("\n\n" + "=" * 80)
    log("ANALYSIS 2: RAW SPACING DIRECTION (paired, same seizure)")
    log("=" * 80)
    log(f"\n{'Subject':<10} | {'Config':<20} | {'raw_bl':>10} | {'raw_pre':>10} | {'raw_dir':>10} | {'z_pre':>8} | {'slope':>10}")
    log("-" * 95)
    for sid, res in sorted(all_grid_results.items()):
        for label in [old_key, new_key]:
            if label in res:
                s = res[label]
                d = "WIDER" if s["pre_raw_mean"] > s["bl_raw_mean"] else "NARROW"
                log(f"{sid:<10} | {label:<20} | {s['bl_raw_mean']:>10.6f} | {s['pre_raw_mean']:>10.6f} | {d:>10} | {s['pre_z_mean']:>+8.4f} | {s['slope']:>+10.6f}")

    log("\n\n" + "=" * 80)
    log("ANALYSIS 3: PAIRED PER-SEIZURE OLD vs NEW")
    log("=" * 80)
    paired_old_slopes = []
    paired_new_slopes = []
    paired_old_z = []
    paired_new_z = []
    paired_old_raw_dir = []
    paired_new_raw_dir = []
    for sid, res in sorted(all_grid_results.items()):
        if old_key in res and new_key in res:
            o = res[old_key]
            n = res[new_key]
            paired_old_slopes.append(o["slope"])
            paired_new_slopes.append(n["slope"])
            paired_old_z.append(o["pre_z_mean"])
            paired_new_z.append(n["pre_z_mean"])
            paired_old_raw_dir.append(1 if o["pre_raw_mean"] > o["bl_raw_mean"] else -1)
            paired_new_raw_dir.append(1 if n["pre_raw_mean"] > n["bl_raw_mean"] else -1)
            agree = "AGREE" if paired_old_raw_dir[-1] == paired_new_raw_dir[-1] else "DISAGREE"
            log(f"  {sid}: old_slope={o['slope']:+.6f}, new_slope={n['slope']:+.6f}, "
                f"old_z={o['pre_z_mean']:+.4f}, new_z={n['pre_z_mean']:+.4f}, "
                f"raw_direction={agree}")

    if paired_old_slopes:
        r, p = sp_stats.pearsonr(paired_old_slopes, paired_new_slopes)
        log(f"\n  Slope correlation (old vs new): r={r:.3f}, p={p:.4f}")
        r2, p2 = sp_stats.pearsonr(paired_old_z, paired_new_z)
        log(f"  Z-score correlation (old vs new): r={r2:.3f}, p={p2:.4f}")
        agree_count = sum(1 for a, b in zip(paired_old_raw_dir, paired_new_raw_dir) if a == b)
        log(f"  Raw direction agreement: {agree_count}/{len(paired_old_raw_dir)}")

    log("\n\n" + "=" * 80)
    log("ANALYSIS 4: ONE-CHANGE-AT-A-TIME PARAMETER GRID")
    log("=" * 80)
    log("\nMean pre-ictal z-score across seizures, by parameter config:")
    log(f"\n{'Config':<25} | {'mean_z':>8} | {'n_neg':>6} | {'n_pos':>6} | {'mean_slope':>12} | {'mean_cond_pre':>14} | {'mean_resid_pre':>14}")
    log("-" * 105)
    for p in PARAM_GRID:
        z_vals = []
        slopes = []
        conds = []
        resids = []
        for sid, res in all_grid_results.items():
            if p["label"] in res:
                z_vals.append(res[p["label"]]["pre_z_mean"])
                slopes.append(res[p["label"]]["slope"])
                conds.append(res[p["label"]]["cond_pre_median"])
                resids.append(res[p["label"]]["resid_pre"])
        if z_vals:
            z_arr = np.array(z_vals)
            n_neg = int(np.sum(z_arr < 0))
            n_pos = int(np.sum(z_arr > 0))
            log(f"{p['label']:<25} | {np.mean(z_arr):>+8.4f} | {n_neg:>6} | {n_pos:>6} | {np.mean(slopes):>+12.6f} | {np.mean(conds):>14.1f} | {np.mean(resids):>14.4f}")

    log("\n\n" + "=" * 80)
    log("ANALYSIS 5: CONDITION NUMBER & FIT STABILITY")
    log("=" * 80)
    for p in PARAM_GRID:
        cond_bl = []
        cond_pre = []
        frac_bl = []
        frac_pre = []
        resid_bl = []
        resid_pre = []
        for sid, res in all_grid_results.items():
            if p["label"] in res:
                cond_bl.append(res[p["label"]]["cond_bl_median"])
                cond_pre.append(res[p["label"]]["cond_pre_median"])
                frac_bl.append(res[p["label"]]["frac_cond_above_100_bl"])
                frac_pre.append(res[p["label"]]["frac_cond_above_100_pre"])
                resid_bl.append(res[p["label"]]["resid_bl"])
                resid_pre.append(res[p["label"]]["resid_pre"])
        if cond_bl:
            log(f"\n  {p['label']}:")
            log(f"    Condition number — bl: {np.mean(cond_bl):.1f}, pre: {np.mean(cond_pre):.1f}")
            log(f"    Frac cond>100 — bl: {np.mean(frac_bl):.3f}, pre: {np.mean(frac_pre):.3f}")
            log(f"    Residual var — bl: {np.mean(resid_bl):.4f}, pre: {np.mean(resid_pre):.4f}")
            log(f"    Resid drops pre-ictally: {np.mean(resid_pre) < np.mean(resid_bl)}")

    log("\n\n" + "=" * 80)
    log("ANALYSIS 6: ONSET SPIKE INVESTIGATION")
    log("=" * 80)
    for sid, res in sorted(all_grid_results.items()):
        if new_key in res:
            s = res[new_key]
            log(f"\n  {sid} ({new_key}):")
            log(f"    raw: bl={s['bl_raw_mean']:.6f}, pre={s['pre_raw_mean']:.6f}, "
                f"ictal={s['ictal_raw_mean']:.6f}, post={s['post_raw_mean']:.6f}")
            log(f"    spec_radius: bl={s['spec_radius_bl']:.4f}, pre={s['spec_radius_pre']:.4f}, "
                f"ictal={s['spec_radius_ictal']:.4f}")
            log(f"    resid_var: bl={s['resid_bl']:.4f}, pre={s['resid_pre']:.4f}, "
                f"ictal={s['resid_ictal']:.4f}")
            log(f"    min_time={s['min_time_sec']:.1f}s (negative=before onset)")
            if s["ictal_raw_mean"] > s["pre_raw_mean"] * 2:
                log(f"    *** ONSET SPIKE: ictal raw is {s['ictal_raw_mean']/s['pre_raw_mean']:.1f}x pre-ictal")
            if s["resid_ictal"] < 0.05:
                log(f"    *** SUSPICIOUS: very low ictal residual variance = overfitting")

    results_path = CMCC_ROOT / "results_chbmit" / "analysis" / "diagnostic_comprehensive.json"

    serializable = {}
    for sid, res in all_grid_results.items():
        serializable[sid] = {}
        for label, stats in res.items():
            serializable[sid][label] = {k: round(v, 8) if isinstance(v, float) else v for k, v in stats.items()}

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
