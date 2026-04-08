import os
"""Cross-modal generalization test of EP/criticality dynamics (ds005620).

Tests whether the spectral-radius-to-eigenvalue-gap coupling found in iEEG
high-gamma (r~0.10, t=13.6) generalizes to broadband scalp EEG dynamics,
and whether propofol sedation abolishes it.

IMPORTANT: This is NOT a direct replication. The iEEG pipeline uses
high-gamma envelope (70-150 Hz, a proxy for cortical spiking), while this
pipeline uses broadband CSD-PCA components (0.5-45 Hz, reflecting synaptic
currents and mesoscale oscillatory dynamics). These are fundamentally
different neural observables. A positive result is a stronger claim
(modality-independent); a negative result is ambiguous (could be
modality-specific or a sensitivity issue).

Dataset: ds005620 — 20 subjects, 65-channel scalp EEG at 5000 Hz.
Conditions: awake (eyes-closed) vs propofol sedation (rest runs 1-3).

Preprocessing: VEOG/HEOG/EMG removal -> bad channel detection ->
notch filter (50 Hz) -> bandpass (0.5-45 Hz) -> downsample (5000->500 Hz)
-> CSD -> PCA (15 components).

PCA is fitted independently per condition. All cross-condition comparisons
use basis-invariant metrics (eigenvalues, spectral radius, eigenvalue gaps,
effective rank).

Primary sedation comparison uses run-1 only (deepest sedation). Pooled
average across runs 1-3 is reported as secondary for robustness.
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

from cmcc.preprocess.scalp_eeg import (
    load_ds005620_subject,
    preprocess_scalp_eeg,
)
from cmcc.analysis.dynamical_systems import compute_ep_proximity_timecourse
from cmcc.analysis.contrasts import fdr_correction
from cmcc.analysis.ep_advanced import (
    compute_spectral_radius_sensitivity,
    compute_svd_dimension,
    compute_petermann_noise,
    compute_alpha_power_per_window,
    compute_condition_number_alpha_sensitivity,
)

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("PROPOFOL_DATA_ROOT", "./data/ds005620"))
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"
FIG_DIR = CMCC_ROOT / "results" / "figures" / "ep_propofol"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 0.5
STEP_SEC = 0.1
N_COMPONENTS = 15
DOWNSAMPLE_TO = 500.0
LINE_FREQ = 50.0
SEED = 42

EXCLUDED_SUBJECTS = ["sub-1037"]

SUBJECT_IDS = [
    "1010", "1016", "1017", "1022", "1024", "1033", "1036",
    "1045", "1046", "1054", "1055", "1057", "1060", "1061",
    "1062", "1064", "1067", "1068", "1071", "1074",
]


def log(msg):
    print(msg, flush=True)


def load_subject_condition(subject_id, data_root, condition):
    """Load and preprocess one condition for a subject.

    Parameters
    ----------
    subject_id : str
        e.g. "1010"
    data_root : Path
        Root of ds005620.
    condition : str
        "awake" or "sed_run-1", "sed_run-2", "sed_run-3"

    Returns
    -------
    data_pca : np.ndarray, shape (n_components, n_samples) or None
    sfreq : float
    preprocess_info : dict
    """
    if condition == "awake":
        task, acq, run = "awake", "EC", None
    elif condition.startswith("sed_run-"):
        run_num = int(condition.split("-")[1])
        task, acq, run = "sed", "rest", run_num
    else:
        raise ValueError(f"Unknown condition: {condition}")

    try:
        raw = load_ds005620_subject(subject_id, data_root, task=task, acq=acq, run=run)
    except FileNotFoundError:
        return None, None, None

    data_pca, sfreq, info = preprocess_scalp_eeg(
        raw, line_freq=LINE_FREQ, downsample_to=DOWNSAMPLE_TO,
        n_components=N_COMPONENTS,
    )
    info["condition"] = condition
    info["subject"] = subject_id
    return data_pca, sfreq, info


def analyze_condition(data_pca, sfreq, condition_label):
    """Run EP analyses on one condition's PCA-reduced data.

    Uses broadband power (mean squared PCA component amplitude) rather
    than high-gamma envelope for the Petermann noise test. This is the
    scalp EEG equivalent — not the same observable as iEEG high-gamma.

    Returns
    -------
    dict with spectral_sensitivity, svd_dimension, mean EP metrics
    """
    ep_tc = compute_ep_proximity_timecourse(
        data_pca, sfreq=sfreq,
        window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        max_channels=N_COMPONENTS, seed=SEED,
    )

    jac = ep_tc["jac_result"]
    ep = ep_tc["ep_result"]
    n_windows = len(jac.window_centers)

    srs = compute_spectral_radius_sensitivity(jac, ep)

    svd = compute_svd_dimension(jac, ep, run_null=True)

    window_samples = max(int(WINDOW_SEC * sfreq), data_pca.shape[0] + 10)
    n_ch, n_total = data_pca.shape
    half_w = window_samples // 2
    broadband_power = np.zeros(n_windows)
    for i, c in enumerate(jac.window_centers):
        c = int(c)
        start = max(0, c - half_w)
        end = min(n_total, c + half_w)
        if end > start:
            broadband_power[i] = np.mean(data_pca[:, start:end] ** 2)

    pet = compute_petermann_noise(
        ep, broadband_power, step_sec=STEP_SEC, window_sec=WINDOW_SEC,
    )

    alpha_power = compute_alpha_power_per_window(
        data_pca, sfreq, jac.window_centers, window_samples,
    )
    alpha_sens = compute_condition_number_alpha_sensitivity(jac, alpha_power)

    return {
        "condition": condition_label,
        "n_windows": n_windows,
        "n_components_used": ep_tc["n_channels_used"],
        "mean_spectral_radius": float(np.mean(jac.spectral_radius)),
        "std_spectral_radius": float(np.std(jac.spectral_radius)),
        "mean_eigenvalue_gap": float(np.mean(ep.min_eigenvalue_gaps)),
        "std_eigenvalue_gap": float(np.std(ep.min_eigenvalue_gaps)),
        "mean_ep_score": float(np.mean(ep.ep_scores)),
        "mean_petermann": float(np.nanmean(ep.petermann_factors[np.isfinite(ep.petermann_factors)])) if np.any(np.isfinite(ep.petermann_factors)) else float("nan"),
        "spectral_sensitivity": srs,
        "svd_dimension": {
            "mean_pr": svd.mean_pr,
            "mean_erank": svd.mean_erank,
            "pr_vs_ep": svd.pr_vs_ep_score,
            "erank_vs_ep": svd.erank_vs_ep_score,
            "null_r_mean": svd.null_r_mean,
            "null_r_std": svd.null_r_std,
            "null_p": svd.null_p,
        },
        "petermann_noise": {
            "r": pet.correlation_r,
            "p_nominal": pet.correlation_p,
            "p_adjusted": pet.p_adjusted,
            "n_eff": pet.n_eff,
            "peak_lag": pet.peak_lag,
            "peak_lag_sec": pet.peak_lag_seconds,
            "granger_f": pet.granger_f,
            "granger_p": pet.granger_p,
            "granger_stride": pet.granger_stride,
            "n_valid_windows": pet.n_valid_windows,
        },
        "alpha_sensitivity": alpha_sens,
        "mean_alpha_power": float(np.mean(alpha_power)),
    }


def analyze_single_subject(subject_id, data_root):
    """Process awake + sedation conditions for one subject."""
    t0 = time.time()
    log(f"\n  {subject_id}...")

    awake_data, awake_sfreq, awake_info = load_subject_condition(
        subject_id, data_root, "awake",
    )
    if awake_data is None:
        log(f"    SKIP: no awake data")
        return None

    log(f"    awake: {awake_data.shape[0]} comp, {awake_sfreq} Hz, "
        f"{awake_data.shape[1]/awake_sfreq:.1f}s, "
        f"var={awake_info['cumulative_variance']:.3f}")

    awake_result = analyze_condition(awake_data, awake_sfreq, "awake")
    log(f"    awake spectral_sensitivity: r={awake_result['spectral_sensitivity']['r']:.4f} "
        f"p_adj={awake_result['spectral_sensitivity']['p_adjusted']:.4f} "
        f"n_eff={awake_result['spectral_sensitivity']['n_eff']}")
    log(f"    awake alpha_sensitivity: r={awake_result['alpha_sensitivity']['r']:.4f} "
        f"p_adj={awake_result['alpha_sensitivity']['p_adjusted']:.4f}")

    del awake_data
    gc.collect()

    sed_results = []
    sed_preprocess = []
    for run in [1, 2, 3]:
        cond = f"sed_run-{run}"
        sed_data, sed_sfreq, sed_info = load_subject_condition(
            subject_id, data_root, cond,
        )
        if sed_data is None:
            log(f"    {cond}: not found, skipping")
            continue

        log(f"    {cond}: {sed_data.shape[0]} comp, {sed_sfreq} Hz, "
            f"{sed_data.shape[1]/sed_sfreq:.1f}s, "
            f"var={sed_info['cumulative_variance']:.3f}")

        sed_result = analyze_condition(sed_data, sed_sfreq, cond)
        log(f"    {cond} spectral_sensitivity: r={sed_result['spectral_sensitivity']['r']:.4f} "
            f"p_adj={sed_result['spectral_sensitivity']['p_adjusted']:.4f}")
        sed_results.append(sed_result)
        sed_preprocess.append(sed_info)

        del sed_data
        gc.collect()

    if not sed_results:
        log(f"    SKIP: no sedation data")
        return None

    sed_pooled = {
        "mean_spectral_radius": float(np.mean([s["mean_spectral_radius"] for s in sed_results])),
        "mean_eigenvalue_gap": float(np.mean([s["mean_eigenvalue_gap"] for s in sed_results])),
        "mean_ep_score": float(np.mean([s["mean_ep_score"] for s in sed_results])),
        "mean_erank": float(np.mean([s["svd_dimension"]["mean_erank"] for s in sed_results])),
        "spectral_sensitivity_r": float(np.mean([s["spectral_sensitivity"]["r"] for s in sed_results])),
        "alpha_sensitivity_r": float(np.mean([s["alpha_sensitivity"]["r"] for s in sed_results])),
    }

    run1 = [s for s in sed_results if s["condition"] == "sed_run-1"]
    if run1:
        sed_run1 = {
            "mean_spectral_radius": run1[0]["mean_spectral_radius"],
            "mean_eigenvalue_gap": run1[0]["mean_eigenvalue_gap"],
            "mean_ep_score": run1[0]["mean_ep_score"],
            "mean_erank": run1[0]["svd_dimension"]["mean_erank"],
            "spectral_sensitivity_r": run1[0]["spectral_sensitivity"]["r"],
            "alpha_sensitivity_r": run1[0]["alpha_sensitivity"]["r"],
        }
    else:
        sed_run1 = sed_pooled

    summary = {
        "subject": subject_id,
        "awake": awake_result,
        "awake_preprocess": awake_info,
        "sedation_runs": sed_results,
        "sedation_preprocess": sed_preprocess,
        "sedation_run1": sed_run1,
        "sedation_pooled": sed_pooled,
        "delta_spectral_radius_run1": awake_result["mean_spectral_radius"] - sed_run1["mean_spectral_radius"],
        "delta_eigenvalue_gap_run1": awake_result["mean_eigenvalue_gap"] - sed_run1["mean_eigenvalue_gap"],
        "delta_ep_score_run1": awake_result["mean_ep_score"] - sed_run1["mean_ep_score"],
        "delta_erank_run1": awake_result["svd_dimension"]["mean_erank"] - sed_run1["mean_erank"],
        "delta_spectral_sensitivity_r_run1": awake_result["spectral_sensitivity"]["r"] - sed_run1["spectral_sensitivity_r"],
        "delta_spectral_radius_pooled": awake_result["mean_spectral_radius"] - sed_pooled["mean_spectral_radius"],
        "delta_eigenvalue_gap_pooled": awake_result["mean_eigenvalue_gap"] - sed_pooled["mean_eigenvalue_gap"],
        "delta_ep_score_pooled": awake_result["mean_ep_score"] - sed_pooled["mean_ep_score"],
        "delta_erank_pooled": awake_result["svd_dimension"]["mean_erank"] - sed_pooled["mean_erank"],
        "delta_spectral_sensitivity_r_pooled": awake_result["spectral_sensitivity"]["r"] - sed_pooled["spectral_sensitivity_r"],
        "elapsed_s": time.time() - t0,
    }

    log(f"    deltas (run1): rho={summary['delta_spectral_radius_run1']:.4f} "
        f"gap={summary['delta_eigenvalue_gap_run1']:.6f} "
        f"spec_sens_r={summary['delta_spectral_sensitivity_r_run1']:.4f}")

    return summary


def _cohens_d_paired(a, b):
    """Cohen's d_z for paired samples: mean(diff) / std(diff)."""
    diff = np.array(a) - np.array(b)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def _paired_test(awake_vals, sed_vals, label, group, all_group_p, all_group_p_labels):
    """Run paired t-test with Cohen's d and append to group dict."""
    if len(awake_vals) < 3:
        return
    t_val, p_val = sp_stats.ttest_rel(awake_vals, sed_vals)
    d = _cohens_d_paired(awake_vals, sed_vals)
    group[label] = {
        "mean_awake": float(np.mean(awake_vals)),
        "mean_sed": float(np.mean(sed_vals)),
        "mean_diff": float(np.mean(np.array(awake_vals) - np.array(sed_vals))),
        "t": float(t_val),
        "p": float(p_val),
        "cohens_d": d,
        "n": len(awake_vals),
    }
    all_group_p.append(float(p_val))
    all_group_p_labels.append(label)


def compute_group_statistics(all_results):
    valid = [s for s in all_results if s is not None]
    if len(valid) < 3:
        return {}

    group = {}
    all_group_p = []
    all_group_p_labels = []

    awake_rs = [s["awake"]["spectral_sensitivity"]["r"] for s in valid]
    sed_rs_run1 = [s["sedation_run1"]["spectral_sensitivity_r"] for s in valid]
    sed_rs_pooled = [s["sedation_pooled"]["spectral_sensitivity_r"] for s in valid]

    if len(awake_rs) >= 3:
        t_awake, p_awake = sp_stats.ttest_1samp(awake_rs, 0.0)
        group["spectral_sensitivity_awake"] = {
            "mean_r": float(np.mean(awake_rs)),
            "std_r": float(np.std(awake_rs)),
            "t": float(t_awake),
            "p": float(p_awake),
            "n": len(awake_rs),
        }
        all_group_p.append(float(p_awake))
        all_group_p_labels.append("spectral_sensitivity_awake")

    if len(sed_rs_run1) >= 3:
        t_sed, p_sed = sp_stats.ttest_1samp(sed_rs_run1, 0.0)
        group["spectral_sensitivity_sedation_run1"] = {
            "mean_r": float(np.mean(sed_rs_run1)),
            "std_r": float(np.std(sed_rs_run1)),
            "t": float(t_sed),
            "p": float(p_sed),
            "n": len(sed_rs_run1),
        }
        all_group_p.append(float(p_sed))
        all_group_p_labels.append("spectral_sensitivity_sedation_run1")

    if len(awake_rs) >= 3 and len(sed_rs_run1) >= 3:
        t_paired, p_paired = sp_stats.ttest_rel(awake_rs, sed_rs_run1)
        d = _cohens_d_paired(awake_rs, sed_rs_run1)
        group["spectral_sensitivity_awake_vs_sed_run1"] = {
            "mean_diff": float(np.mean(np.array(awake_rs) - np.array(sed_rs_run1))),
            "t": float(t_paired),
            "p": float(p_paired),
            "cohens_d": d,
            "n": len(awake_rs),
            "comparison": "primary (run-1 only)",
        }
        all_group_p.append(float(p_paired))
        all_group_p_labels.append("spectral_sensitivity_awake_vs_sed_run1")

    if len(awake_rs) >= 3 and len(sed_rs_pooled) >= 3:
        t_pooled, p_pooled = sp_stats.ttest_rel(awake_rs, sed_rs_pooled)
        d_pooled = _cohens_d_paired(awake_rs, sed_rs_pooled)
        group["spectral_sensitivity_awake_vs_sed_pooled"] = {
            "mean_diff": float(np.mean(np.array(awake_rs) - np.array(sed_rs_pooled))),
            "t": float(t_pooled),
            "p": float(p_pooled),
            "cohens_d": d_pooled,
            "n": len(awake_rs),
            "comparison": "secondary (pooled runs 1-3)",
        }

    awake_rho = [s["awake"]["mean_spectral_radius"] for s in valid]
    sed_rho_r1 = [s["sedation_run1"]["mean_spectral_radius"] for s in valid]
    _paired_test(awake_rho, sed_rho_r1, "spectral_radius_awake_vs_sed_run1",
                 group, all_group_p, all_group_p_labels)

    awake_gap = [s["awake"]["mean_eigenvalue_gap"] for s in valid]
    sed_gap_r1 = [s["sedation_run1"]["mean_eigenvalue_gap"] for s in valid]
    _paired_test(awake_gap, sed_gap_r1, "eigenvalue_gap_awake_vs_sed_run1",
                 group, all_group_p, all_group_p_labels)

    awake_erank = [s["awake"]["svd_dimension"]["mean_erank"] for s in valid]
    sed_erank_r1 = [s["sedation_run1"]["mean_erank"] for s in valid]
    _paired_test(awake_erank, sed_erank_r1, "effective_rank_awake_vs_sed_run1",
                 group, all_group_p, all_group_p_labels)

    if len(awake_gap) >= 3 and len(sed_gap_r1) >= 3:
        t_gw, p_gw = sp_stats.ttest_rel(sed_gap_r1, awake_gap, alternative="greater")
        d_gw = _cohens_d_paired(sed_gap_r1, awake_gap)
        group["gap_widening"] = {
            "mean_awake": float(np.mean(awake_gap)),
            "mean_sed_run1": float(np.mean(sed_gap_r1)),
            "mean_diff_sed_minus_awake": float(np.mean(np.array(sed_gap_r1) - np.array(awake_gap))),
            "t": float(t_gw),
            "p_onesided": float(p_gw),
            "cohens_d": d_gw,
            "n": len(awake_gap),
            "hypothesis": "sed_run1 gap > awake gap (modes decouple under propofol)",
        }
        all_group_p.append(float(p_gw))
        all_group_p_labels.append("gap_widening")

    sed_rho_pool = [s["sedation_pooled"]["mean_spectral_radius"] for s in valid]
    sed_gap_pool = [s["sedation_pooled"]["mean_eigenvalue_gap"] for s in valid]
    sed_erank_pool = [s["sedation_pooled"]["mean_erank"] for s in valid]
    group["pooled_secondary"] = {}
    for label, a, b in [
        ("spectral_radius", awake_rho, sed_rho_pool),
        ("eigenvalue_gap", awake_gap, sed_gap_pool),
        ("effective_rank", awake_erank, sed_erank_pool),
    ]:
        if len(a) >= 3:
            t_v, p_v = sp_stats.ttest_rel(a, b)
            group["pooled_secondary"][label] = {
                "mean_diff": float(np.mean(np.array(a) - np.array(b))),
                "t": float(t_v),
                "p": float(p_v),
                "cohens_d": _cohens_d_paired(a, b),
                "n": len(a),
            }

    alpha_pairs = [(s["awake"]["alpha_sensitivity"]["r"],
                     s["sedation_run1"]["alpha_sensitivity_r"]) for s in valid
                    if np.isfinite(s["awake"]["alpha_sensitivity"]["r"])
                    and np.isfinite(s["sedation_run1"]["alpha_sensitivity_r"])]
    if alpha_pairs:
        awake_alpha_r = [p[0] for p in alpha_pairs]
        sed_alpha_r = [p[1] for p in alpha_pairs]
    else:
        awake_alpha_r, sed_alpha_r = [], []

    if len(sed_alpha_r) >= 3:
        t_as, p_as = sp_stats.ttest_1samp(sed_alpha_r, 0.0)
        group["alpha_sensitivity_sedation_run1"] = {
            "mean_r": float(np.mean(sed_alpha_r)),
            "std_r": float(np.std(sed_alpha_r)),
            "t": float(t_as),
            "p": float(p_as),
            "n": len(sed_alpha_r),
        }
        all_group_p.append(float(p_as))
        all_group_p_labels.append("alpha_sensitivity_sedation_run1")

    if len(awake_alpha_r) >= 3:
        t_aa, p_aa = sp_stats.ttest_1samp(awake_alpha_r, 0.0)
        group["alpha_sensitivity_awake"] = {
            "mean_r": float(np.mean(awake_alpha_r)),
            "std_r": float(np.std(awake_alpha_r)),
            "t": float(t_aa),
            "p": float(p_aa),
            "n": len(awake_alpha_r),
        }
        all_group_p.append(float(p_aa))
        all_group_p_labels.append("alpha_sensitivity_awake")

    if len(awake_alpha_r) >= 3:
        t_ap, p_ap = sp_stats.ttest_rel(sed_alpha_r, awake_alpha_r)
        d_ap = _cohens_d_paired(sed_alpha_r, awake_alpha_r)
        group["alpha_sensitivity_sed_vs_awake"] = {
            "mean_diff": float(np.mean(np.array(sed_alpha_r) - np.array(awake_alpha_r))),
            "t": float(t_ap),
            "p": float(p_ap),
            "cohens_d": d_ap,
            "n": len(awake_alpha_r),
            "hypothesis": "propofol alpha hypersynchrony creates different EP-alpha coupling than wakefulness",
        }
        all_group_p.append(float(p_ap))
        all_group_p_labels.append("alpha_sensitivity_sed_vs_awake")

    delta_r = [s["delta_spectral_sensitivity_r_run1"] for s in valid]
    delta_gap = [s["delta_eigenvalue_gap_run1"] for s in valid]
    if len(delta_r) >= 5:
        r_dd, p_dd = sp_stats.pearsonr(delta_r, delta_gap)
        rho_dd, p_sp_dd = sp_stats.spearmanr(delta_r, delta_gap)
        group["delta_delta_correlation"] = {
            "r": float(r_dd),
            "p": float(p_dd),
            "rho": float(rho_dd),
            "p_spearman": float(p_sp_dd),
            "n_subjects": len(delta_r),
            "description": "cross-subject correlation of delta_r (awake-sed spec_sens) vs delta_gap (awake-sed eigenvalue gap). Deltas are awake minus sed: positive delta_r = sensitivity loss, negative delta_gap = gap widening. A NEGATIVE r supports the hypothesis (more sensitivity loss -> more gap widening).",
        }
        all_group_p.append(float(p_dd))
        all_group_p_labels.append("delta_delta_correlation")

    if len(all_group_p) >= 2:
        fdr_sig = fdr_correction(all_group_p, alpha=0.05)
        group["fdr_correction"] = {
            label: {"p": float(p), "fdr_significant": bool(sig)}
            for label, p, sig in zip(all_group_p_labels, all_group_p, fdr_sig)
        }

    return group


def plot_propofol_summary(all_results, group_stats, output_dir):
    valid = [s for s in all_results if s is not None]
    if len(valid) < 3:
        log("  Too few subjects for summary plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EP Cross-Modal Generalization — ds005620 Scalp EEG (Run-1 Primary)",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    awake_rho = [s["awake"]["mean_spectral_radius"] for s in valid]
    sed_rho = [s["sedation_run1"]["mean_spectral_radius"] for s in valid]
    for a, s in zip(awake_rho, sed_rho):
        ax.plot([0, 1], [a, s], "o-", color="gray", alpha=0.4, markersize=4)
    ax.plot(0, np.mean(awake_rho), "D", color="steelblue", markersize=10, zorder=5)
    ax.plot(1, np.mean(sed_rho), "D", color="coral", markersize=10, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Awake", "Sedation (run-1)"])
    ax.set_ylabel("Mean Spectral Radius (ρ)")
    gs = group_stats.get("spectral_radius_awake_vs_sed_run1", {})
    ax.set_title(f"Spectral Radius (t={gs.get('t', float('nan')):.2f}, p={gs.get('p', float('nan')):.4f}, d={gs.get('cohens_d', float('nan')):.2f})")

    ax = axes[0, 1]
    awake_gap = [s["awake"]["mean_eigenvalue_gap"] for s in valid]
    sed_gap = [s["sedation_run1"]["mean_eigenvalue_gap"] for s in valid]
    for a, s in zip(awake_gap, sed_gap):
        ax.plot([0, 1], [a, s], "o-", color="gray", alpha=0.4, markersize=4)
    ax.plot(0, np.mean(awake_gap), "D", color="steelblue", markersize=10, zorder=5)
    ax.plot(1, np.mean(sed_gap), "D", color="coral", markersize=10, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Awake", "Sedation (run-1)"])
    ax.set_ylabel("Mean Min Eigenvalue Gap")
    gs = group_stats.get("eigenvalue_gap_awake_vs_sed_run1", {})
    ax.set_title(f"Eigenvalue Gap (t={gs.get('t', float('nan')):.2f}, p={gs.get('p', float('nan')):.4f}, d={gs.get('cohens_d', float('nan')):.2f})")

    ax = axes[1, 0]
    awake_r = [s["awake"]["spectral_sensitivity"]["r"] for s in valid]
    sed_r = [s["sedation_run1"]["spectral_sensitivity_r"] for s in valid]
    for a, s in zip(awake_r, sed_r):
        ax.plot([0, 1], [a, s], "o-", color="gray", alpha=0.4, markersize=4)
    ax.plot(0, np.mean(awake_r), "D", color="steelblue", markersize=10, zorder=5)
    ax.plot(1, np.mean(sed_r), "D", color="coral", markersize=10, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Awake", "Sedation (run-1)"])
    ax.set_ylabel("Spectral Sensitivity r")
    gs = group_stats.get("spectral_sensitivity_awake_vs_sed_run1", {})
    ax.set_title(f"Spectral Sensitivity (t={gs.get('t', float('nan')):.2f}, p={gs.get('p', float('nan')):.4f}, d={gs.get('cohens_d', float('nan')):.2f})")

    ax = axes[1, 1]
    awake_er = [s["awake"]["svd_dimension"]["mean_erank"] for s in valid]
    sed_er = [s["sedation_run1"]["mean_erank"] for s in valid]
    for a, s in zip(awake_er, sed_er):
        ax.plot([0, 1], [a, s], "o-", color="gray", alpha=0.4, markersize=4)
    ax.plot(0, np.mean(awake_er), "D", color="steelblue", markersize=10, zorder=5)
    ax.plot(1, np.mean(sed_er), "D", color="coral", markersize=10, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Awake", "Sedation (run-1)"])
    ax.set_ylabel("Effective Rank")
    gs = group_stats.get("effective_rank_awake_vs_sed_run1", {})
    ax.set_title(f"Effective Rank (t={gs.get('t', float('nan')):.2f}, p={gs.get('p', float('nan')):.4f}, d={gs.get('cohens_d', float('nan')):.2f})")

    plt.tight_layout()
    plt.savefig(output_dir / "ep_propofol_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure: {output_dir / 'ep_propofol_summary.png'}")


def plot_gap_widening_histogram(all_results, group_stats, output_dir):
    valid = [s for s in all_results if s is not None]
    if len(valid) < 3:
        return

    awake_gaps = [s["awake"]["mean_eigenvalue_gap"] for s in valid]
    sed_gaps = [s["sedation_run1"]["mean_eigenvalue_gap"] for s in valid]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.linspace(
        min(min(awake_gaps), min(sed_gaps)) * 0.9,
        max(max(awake_gaps), max(sed_gaps)) * 1.1,
        15,
    )
    ax.hist(awake_gaps, bins=bins, alpha=0.6, color="steelblue", label="Awake", edgecolor="white")
    ax.hist(sed_gaps, bins=bins, alpha=0.6, color="coral", label="Sedation (run-1)", edgecolor="white")
    ax.axvline(np.mean(awake_gaps), color="steelblue", linestyle="--", linewidth=2)
    ax.axvline(np.mean(sed_gaps), color="coral", linestyle="--", linewidth=2)
    ax.set_xlabel("Mean Min Eigenvalue Gap")
    ax.set_ylabel("Count")

    gw = group_stats.get("gap_widening", {})
    ax.set_title(
        f"Test A: Gap Widening Under Propofol\n"
        f"one-sided p={gw.get('p_onesided', float('nan')):.4f}, "
        f"d={gw.get('cohens_d', float('nan')):.2f}"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gap_widening_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure: {output_dir / 'gap_widening_histogram.png'}")


def plot_delta_delta_scatter(all_results, group_stats, output_dir):
    valid = [s for s in all_results if s is not None]
    if len(valid) < 5:
        log("  Too few subjects for delta-delta scatter")
        return

    delta_r = [s["delta_spectral_sensitivity_r_run1"] for s in valid]
    delta_gap = [s["delta_eigenvalue_gap_run1"] for s in valid]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.scatter(delta_r, delta_gap, s=50, c="steelblue", edgecolors="white", zorder=5)

    if len(delta_r) >= 3:
        z = np.polyfit(delta_r, delta_gap, 1)
        x_line = np.linspace(min(delta_r), max(delta_r), 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="coral", linewidth=2)

    ax.set_xlabel("Δ Spectral Sensitivity r (awake - sed)")
    ax.set_ylabel("Δ Eigenvalue Gap (awake - sed)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    dd = group_stats.get("delta_delta_correlation", {})
    ax.set_title(
        f"Test C: Delta-Delta Correlation\n"
        f"r={dd.get('r', float('nan')):.3f}, p={dd.get('p', float('nan')):.4f}, "
        f"N={dd.get('n_subjects', 0)}"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "delta_delta_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure: {output_dir / 'delta_delta_scatter.png'}")


def default_ser(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    log("=" * 70)
    log("PROPOFOL EEG REPLICATION — EP/Criticality Dynamics")
    log(f"Dataset: ds005620 ({DATA_ROOT})")
    log(f"Parameters: window={WINDOW_SEC}s, step={STEP_SEC}s, "
        f"PCA={N_COMPONENTS} components, downsample={DOWNSAMPLE_TO} Hz")
    log("=" * 70)

    subjects = [s for s in SUBJECT_IDS if f"sub-{s}" not in EXCLUDED_SUBJECTS]
    log(f"\nSubjects: {len(subjects)} (excluded: {EXCLUDED_SUBJECTS})")

    all_results = []
    for subj in subjects:
        try:
            result = analyze_single_subject(subj, DATA_ROOT)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            log(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    log(f"\n{'='*70}")
    log(f"RESULTS: {len(all_results)} subjects analyzed")

    group_stats = compute_group_statistics(all_results)

    if group_stats:
        log(f"\n  Group statistics:")
        for key, val in group_stats.items():
            if key != "fdr_correction":
                log(f"    {key}: {val}")
        if "fdr_correction" in group_stats:
            log(f"    FDR correction:")
            for label, info in group_stats["fdr_correction"].items():
                log(f"      {label}: p={info['p']:.6f} sig={info['fdr_significant']}")

    plot_propofol_summary(all_results, group_stats, FIG_DIR)
    plot_gap_widening_histogram(all_results, group_stats, FIG_DIR)
    plot_delta_delta_scatter(all_results, group_stats, FIG_DIR)

    out = {
        "n_subjects": len(all_results),
        "parameters": {
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "n_components": N_COMPONENTS,
            "downsample_to": DOWNSAMPLE_TO,
            "line_freq": LINE_FREQ,
            "seed": SEED,
        },
        "excluded_subjects": EXCLUDED_SUBJECTS,
        "subjects": all_results,
        "group_statistics": group_stats,
    }

    out_path = RESULTS_DIR / "ep_propofol_eeg.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=default_ser)
    log(f"\n  Results: {out_path}")
    log(f"\n{'='*70}")
    log("DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()
