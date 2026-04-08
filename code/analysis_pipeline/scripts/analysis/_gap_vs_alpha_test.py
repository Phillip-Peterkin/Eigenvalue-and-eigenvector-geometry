import os
"""Discriminating test: Is the eigenvalue gap tightening under propofol
just alpha hypersynchrony, or a topological signature independent of frequency?

For each subject's sedation run-1 AND awake condition:
1. Load and preprocess EEG -> CSD -> PCA (15 components)
2. Fit Jacobian -> extract per-window min eigenvalue gap
3. Bandpass PCA data to alpha (8-13 Hz) -> extract per-window alpha power
4. Correlate gap vs alpha power across windows (with autocorrelation correction)

Decision criterion:
- If per-subject gap-alpha |r| > 0.8 consistently: tightening IS alpha hypersynchrony
- If gap-alpha r is weak or inconsistent: topological signature independent of alpha

This is a LEAN script — only computes gap, alpha power, and their correlation.
No SVD, no Petermann, no Granger, no spectral sensitivity.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

from cmcc.preprocess.scalp_eeg import load_ds005620_subject, preprocess_scalp_eeg
from cmcc.analysis.dynamical_systems import compute_ep_proximity_timecourse
from cmcc.analysis.ep_advanced import (
    _effective_n,
    _adjusted_correlation_p,
    compute_alpha_power_per_window,
)

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("PROPOFOL_DATA_ROOT", "./data/ds005620"))
FIG_DIR = CMCC_ROOT / "results" / "figures" / "ep_propofol"
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 0.5
STEP_SEC = 0.1
N_COMPONENTS = 15
DOWNSAMPLE_TO = 500.0
LINE_FREQ = 50.0
SEED = 42
ALPHA_BAND = (8.0, 13.0)

EXCLUDED_SUBJECTS = ["sub-1037"]
SUBJECT_IDS = [
    "1010", "1016", "1017", "1022", "1024", "1033", "1036",
    "1045", "1046", "1054", "1055", "1057", "1060", "1061",
    "1062", "1064", "1067", "1068", "1071", "1074",
]


def log(msg):
    print(msg, flush=True)


def analyze_gap_alpha(subject_id, condition):
    if condition == "awake":
        task, acq, run = "awake", "EC", None
    else:
        task, acq, run = "sed", "rest", 1

    try:
        raw = load_ds005620_subject(subject_id, DATA_ROOT, task=task, acq=acq, run=run)
    except FileNotFoundError:
        return None

    data_pca, sfreq, info = preprocess_scalp_eeg(
        raw, line_freq=LINE_FREQ, downsample_to=DOWNSAMPLE_TO,
        n_components=N_COMPONENTS,
    )

    ep_tc = compute_ep_proximity_timecourse(
        data_pca, sfreq=sfreq,
        window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        max_channels=N_COMPONENTS, seed=SEED,
    )

    jac = ep_tc["jac_result"]
    ep = ep_tc["ep_result"]
    n_windows = len(jac.window_centers)
    window_samples = int(WINDOW_SEC * sfreq)

    gaps = ep.min_eigenvalue_gaps

    alpha_power = compute_alpha_power_per_window(
        data_pca, sfreq, jac.window_centers, window_samples,
        alpha_band=ALPHA_BAND,
    )

    valid = np.isfinite(gaps) & np.isfinite(alpha_power) & (alpha_power > 0)
    n_valid = int(valid.sum())

    if n_valid < 20:
        return None

    g = gaps[valid]
    a = alpha_power[valid]

    r, p_nom = sp_stats.pearsonr(g, a)
    rho, p_sp = sp_stats.spearmanr(g, a)
    n_eff = _effective_n(g, a)
    p_adj = _adjusted_correlation_p(float(r), n_eff)

    return {
        "subject": subject_id,
        "condition": condition,
        "n_windows": n_valid,
        "n_eff": n_eff,
        "pearson_r": float(r),
        "p_nominal": float(p_nom),
        "p_adjusted": float(p_adj),
        "spearman_rho": float(rho),
        "p_spearman": float(p_sp),
        "mean_gap": float(np.mean(g)),
        "mean_alpha": float(np.mean(a)),
        "duration_sec": float(data_pca.shape[1] / sfreq),
        "cumulative_var": info["cumulative_variance"],
    }


def main():
    log("=" * 70)
    log("GAP vs ALPHA POWER — Discriminating Test")
    log(f"Alpha band: {ALPHA_BAND[0]}-{ALPHA_BAND[1]} Hz")
    log(f"If |r| > 0.8: tightening = alpha hypersynchrony")
    log(f"If |r| < 0.3: topological signature independent of alpha")
    log("=" * 70)

    subjects = [s for s in SUBJECT_IDS if f"sub-{s}" not in EXCLUDED_SUBJECTS]
    all_results = []

    for subj in subjects:
        for cond in ["sed_run1", "awake"]:
            t0 = time.time()
            log(f"\n  {subj} [{cond}]...")
            try:
                result = analyze_gap_alpha(subj, cond)
                if result:
                    log(f"    r={result['pearson_r']:.4f} (p_adj={result['p_adjusted']:.4f}, "
                        f"n_eff={result['n_eff']}) "
                        f"rho={result['spearman_rho']:.4f} "
                        f"[{result['n_windows']} windows, {time.time()-t0:.0f}s]")
                    all_results.append(result)
                else:
                    log(f"    SKIP: insufficient data")
            except Exception as e:
                log(f"    ERROR: {e}")

            del result
            gc.collect()

    sed_results = [r for r in all_results if r["condition"] == "sed_run1"]
    awake_results = [r for r in all_results if r["condition"] == "awake"]

    log(f"\n{'='*70}")
    log(f"SUMMARY")
    log(f"{'='*70}")

    group = {}

    for label, subset in [("sedation_run1", sed_results), ("awake", awake_results)]:
        if len(subset) < 3:
            continue
        rs = [r["pearson_r"] for r in subset]
        rhos = [r["spearman_rho"] for r in subset]

        t_r, p_r = sp_stats.ttest_1samp(rs, 0.0)

        group[label] = {
            "n": len(subset),
            "mean_r": float(np.mean(rs)),
            "std_r": float(np.std(rs)),
            "median_r": float(np.median(rs)),
            "min_r": float(np.min(rs)),
            "max_r": float(np.max(rs)),
            "mean_rho": float(np.mean(rhos)),
            "t_vs_zero": float(t_r),
            "p_vs_zero": float(p_r),
            "n_above_0.8": int(sum(1 for r in rs if abs(r) > 0.8)),
            "n_above_0.5": int(sum(1 for r in rs if abs(r) > 0.5)),
            "n_above_0.3": int(sum(1 for r in rs if abs(r) > 0.3)),
        }

        log(f"\n  {label} (N={len(subset)}):")
        log(f"    Mean r = {np.mean(rs):.4f} +/- {np.std(rs):.4f}")
        log(f"    Median r = {np.median(rs):.4f}")
        log(f"    Range: [{np.min(rs):.4f}, {np.max(rs):.4f}]")
        log(f"    Mean rho = {np.mean(rhos):.4f}")
        log(f"    t vs 0: t={t_r:.3f}, p={p_r:.6f}")
        log(f"    |r| > 0.8: {group[label]['n_above_0.8']}/{len(subset)}")
        log(f"    |r| > 0.5: {group[label]['n_above_0.5']}/{len(subset)}")
        log(f"    |r| > 0.3: {group[label]['n_above_0.3']}/{len(subset)}")

    if sed_results and awake_results:
        paired_subs = set(r["subject"] for r in sed_results) & set(r["subject"] for r in awake_results)
        sed_map = {r["subject"]: r["pearson_r"] for r in sed_results}
        awake_map = {r["subject"]: r["pearson_r"] for r in awake_results}
        paired_sed = [sed_map[s] for s in sorted(paired_subs)]
        paired_awake = [awake_map[s] for s in sorted(paired_subs)]

        if len(paired_sed) >= 3:
            t_p, p_p = sp_stats.ttest_rel(paired_sed, paired_awake)
            diff = np.array(paired_sed) - np.array(paired_awake)
            d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
            group["sed_vs_awake_paired"] = {
                "mean_diff": float(np.mean(diff)),
                "t": float(t_p),
                "p": float(p_p),
                "cohens_d": d,
                "n": len(paired_sed),
            }
            log(f"\n  Sed vs Awake (paired, N={len(paired_sed)}):")
            log(f"    Mean diff (sed-awake): {np.mean(diff):.4f}")
            log(f"    t={t_p:.3f}, p={p_p:.6f}, d={d:.3f}")

    # ── VERDICT ──
    if sed_results:
        mean_sed_r = np.mean([r["pearson_r"] for r in sed_results])
        n_high = sum(1 for r in sed_results if abs(r["pearson_r"]) > 0.8)
        n_total = len(sed_results)

        log(f"\n{'='*70}")
        log(f"VERDICT")
        log(f"{'='*70}")
        if n_high > n_total * 0.5:
            log(f"  |r| > 0.8 in {n_high}/{n_total} subjects.")
            log(f"  --> Gap tightening IS alpha hypersynchrony.")
            log(f"      The 'pathological tightening' is driven by propofol-induced")
            log(f"      alpha oscillations, not an independent topological feature.")
            group["verdict"] = "alpha_hypersynchrony"
        elif abs(mean_sed_r) < 0.3 and n_high == 0:
            log(f"  Mean |r| = {abs(mean_sed_r):.3f}, 0/{n_total} subjects above 0.8.")
            log(f"  --> TOPOLOGICAL SIGNATURE independent of alpha band.")
            log(f"      The eigenvalue gap dynamics under propofol represent a")
            log(f"      geometric property of the Jacobian NOT reducible to alpha power.")
            group["verdict"] = "topological_signature"
        else:
            log(f"  Mean |r| = {abs(mean_sed_r):.3f}, {n_high}/{n_total} above 0.8.")
            log(f"  --> MIXED: partial alpha coupling. Further band decomposition needed.")
            group["verdict"] = "mixed"

    # ── Figures ──
    if sed_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        sed_rs = [r["pearson_r"] for r in sed_results]
        awake_rs = [r["pearson_r"] for r in awake_results] if awake_results else []
        if awake_rs:
            ax.hist(awake_rs, bins=15, alpha=0.6, color="steelblue", label="Awake", edgecolor="white")
        ax.hist(sed_rs, bins=15, alpha=0.6, color="coral", label="Sedation", edgecolor="white")
        ax.axvline(0, color="black", linestyle=":", alpha=0.5)
        ax.axvline(0.8, color="red", linestyle="--", alpha=0.7, label="|r|=0.8 threshold")
        ax.axvline(-0.8, color="red", linestyle="--", alpha=0.7)
        ax.set_xlabel("Gap-Alpha Pearson r (per subject)")
        ax.set_ylabel("Count")
        ax.set_title("Gap vs Alpha Power Correlation Distribution")
        ax.legend()

        ax = axes[1]
        if awake_rs and len(paired_sed) >= 3:
            for s_r, a_r in zip(paired_sed, paired_awake):
                ax.plot([0, 1], [a_r, s_r], "o-", color="gray", alpha=0.4, markersize=4)
            ax.plot(0, np.mean(paired_awake), "D", color="steelblue", markersize=10, zorder=5)
            ax.plot(1, np.mean(paired_sed), "D", color="coral", markersize=10, zorder=5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Awake", "Sedation"])
            ax.set_ylabel("Gap-Alpha r")
            p_str = f"p={group.get('sed_vs_awake_paired',{}).get('p',float('nan')):.4f}"
            ax.set_title(f"Gap-Alpha Coupling: Awake vs Sedation ({p_str})")
        else:
            ax.bar(range(len(sed_rs)), sorted(sed_rs), color="coral", alpha=0.7)
            ax.axhline(0.8, color="red", linestyle="--")
            ax.axhline(-0.8, color="red", linestyle="--")
            ax.set_ylabel("Gap-Alpha r")
            ax.set_title("Per-Subject Gap-Alpha r (Sedation)")

        plt.tight_layout()
        fig_path = FIG_DIR / "gap_vs_alpha_discriminating_test.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"\n  Figure: {fig_path}")

    out = {
        "test": "gap_vs_alpha_discriminating",
        "alpha_band_hz": list(ALPHA_BAND),
        "parameters": {
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "n_components": N_COMPONENTS,
        },
        "per_subject": all_results,
        "group_statistics": group,
    }

    out_path = RESULTS_DIR / "gap_vs_alpha_test.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"  Results: {out_path}")
    log(f"\n{'='*70}")
    log("DONE")


if __name__ == "__main__":
    main()
