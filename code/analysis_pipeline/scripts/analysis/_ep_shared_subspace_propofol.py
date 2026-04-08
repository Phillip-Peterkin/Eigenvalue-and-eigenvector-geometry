import os
"""Shared-subspace robustness test for propofol EP geometry.

Addresses the single biggest vulnerability in the per-state PCA pipeline:
geometry differences between states could partly reflect fitting PCA
independently per state. Here we pool awake + sedation data within each
subject before PCA, fit once, then project both states into the common
subspace. If the key effects (gap tightening, spectral radius shift)
survive, the result is not an artifact of state-specific coordinate systems.

Also computes alternative spacing metrics (median NN, 10th percentile NN)
to test whether the state pattern depends on the fragile minimum-gap
extreme statistic.
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
    preprocess_to_csd,
    pca_reduce_shared,
)
from cmcc.analysis.dynamical_systems import compute_ep_proximity_timecourse
from cmcc.analysis.ep_advanced import (
    compute_spectral_radius_sensitivity,
    compute_alternative_spacing,
)
from cmcc.analysis.contrasts import fdr_correction

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("PROPOFOL_DATA_ROOT", "./data/ds005620"))
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"
FIG_DIR = CMCC_ROOT / "results" / "figures" / "ep_shared_subspace"
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


def load_csd(subject_id, data_root, condition):
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

    data_csd, sfreq, info = preprocess_to_csd(
        raw, line_freq=LINE_FREQ, downsample_to=DOWNSAMPLE_TO,
    )
    info["condition"] = condition
    info["subject"] = subject_id
    return data_csd, sfreq, info


def analyze_condition_shared(data_pca, sfreq, label):
    ep_tc = compute_ep_proximity_timecourse(
        data_pca, sfreq=sfreq,
        window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        max_channels=N_COMPONENTS, seed=SEED,
    )

    jac = ep_tc["jac_result"]
    ep = ep_tc["ep_result"]

    srs = compute_spectral_radius_sensitivity(jac, ep)
    alt_spacing = compute_alternative_spacing(jac.eigenvalues)

    return {
        "condition": label,
        "n_windows": len(jac.window_centers),
        "mean_spectral_radius": float(np.mean(jac.spectral_radius)),
        "mean_eigenvalue_gap": float(np.mean(ep.min_eigenvalue_gaps)),
        "mean_median_nn_gap": float(np.mean(alt_spacing["median_nn_gap"])),
        "mean_p10_nn_gap": float(np.mean(alt_spacing["p10_nn_gap"])),
        "mean_ep_score": float(np.mean(ep.ep_scores)),
        "spectral_sensitivity": srs,
    }


def analyze_single_subject(subject_id, data_root):
    t0 = time.time()
    log(f"\n  {subject_id}...")

    awake_csd, awake_sfreq, awake_info = load_csd(subject_id, data_root, "awake")
    if awake_csd is None:
        log(f"    SKIP: no awake data")
        return None

    sed_csd, sed_sfreq, sed_info = load_csd(subject_id, data_root, "sed_run-1")
    if sed_csd is None:
        log(f"    SKIP: no sed_run-1 data")
        return None

    n_ch_awake = awake_csd.shape[0]
    n_ch_sed = sed_csd.shape[0]
    if n_ch_awake != n_ch_sed:
        shared_ch = min(n_ch_awake, n_ch_sed)
        log(f"    WARNING: channel mismatch ({n_ch_awake} vs {n_ch_sed}), using first {shared_ch}")
        awake_csd = awake_csd[:shared_ch]
        sed_csd = sed_csd[:shared_ch]

    log(f"    Pooling: awake {awake_csd.shape}, sed {sed_csd.shape}")

    state_data = {"awake": awake_csd, "sed_run1": sed_csd}
    projected, pca_obj, pca_info = pca_reduce_shared(state_data, n_components=N_COMPONENTS)

    log(f"    Shared PCA: {pca_info['n_components']} comp, "
        f"cumvar={pca_info['cumulative_variance']:.3f}, "
        f"pooled={pca_info['total_samples_pooled']} samples")

    del awake_csd, sed_csd
    gc.collect()

    awake_result = analyze_condition_shared(projected["awake"], awake_sfreq, "awake")
    log(f"    awake: rho={awake_result['mean_spectral_radius']:.4f} "
        f"gap={awake_result['mean_eigenvalue_gap']:.6f} "
        f"spec_sens_r={awake_result['spectral_sensitivity']['r']:.4f}")

    sed_result = analyze_condition_shared(projected["sed_run1"], sed_sfreq, "sed_run1")
    log(f"    sed:   rho={sed_result['mean_spectral_radius']:.4f} "
        f"gap={sed_result['mean_eigenvalue_gap']:.6f} "
        f"spec_sens_r={sed_result['spectral_sensitivity']['r']:.4f}")

    del projected
    gc.collect()

    summary = {
        "subject": subject_id,
        "pca_mode": "shared_subspace",
        "shared_pca_info": pca_info,
        "awake": awake_result,
        "sedation_run1": sed_result,
        "delta_spectral_radius": awake_result["mean_spectral_radius"] - sed_result["mean_spectral_radius"],
        "delta_eigenvalue_gap": awake_result["mean_eigenvalue_gap"] - sed_result["mean_eigenvalue_gap"],
        "delta_median_nn_gap": awake_result["mean_median_nn_gap"] - sed_result["mean_median_nn_gap"],
        "delta_p10_nn_gap": awake_result["mean_p10_nn_gap"] - sed_result["mean_p10_nn_gap"],
        "delta_spectral_sensitivity_r": awake_result["spectral_sensitivity"]["r"] - sed_result["spectral_sensitivity"]["r"],
        "elapsed_s": time.time() - t0,
    }

    return summary


def _cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def _paired_test(awake_vals, sed_vals, label, group, all_p, all_labels):
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
    all_p.append(float(p_val))
    all_labels.append(label)


def compute_group_statistics(all_results, original_json_path=None):
    valid = [s for s in all_results if s is not None]
    if len(valid) < 3:
        return {}

    group = {}
    all_p = []
    all_labels = []

    awake_rho = [s["awake"]["mean_spectral_radius"] for s in valid]
    sed_rho = [s["sedation_run1"]["mean_spectral_radius"] for s in valid]
    _paired_test(awake_rho, sed_rho, "spectral_radius", group, all_p, all_labels)

    awake_gap = [s["awake"]["mean_eigenvalue_gap"] for s in valid]
    sed_gap = [s["sedation_run1"]["mean_eigenvalue_gap"] for s in valid]
    _paired_test(awake_gap, sed_gap, "min_eigenvalue_gap", group, all_p, all_labels)

    awake_med = [s["awake"]["mean_median_nn_gap"] for s in valid]
    sed_med = [s["sedation_run1"]["mean_median_nn_gap"] for s in valid]
    _paired_test(awake_med, sed_med, "median_nn_gap", group, all_p, all_labels)

    awake_p10 = [s["awake"]["mean_p10_nn_gap"] for s in valid]
    sed_p10 = [s["sedation_run1"]["mean_p10_nn_gap"] for s in valid]
    _paired_test(awake_p10, sed_p10, "p10_nn_gap", group, all_p, all_labels)

    awake_r = [s["awake"]["spectral_sensitivity"]["r"] for s in valid]
    sed_r = [s["sedation_run1"]["spectral_sensitivity"]["r"] for s in valid]
    _paired_test(awake_r, sed_r, "spectral_sensitivity", group, all_p, all_labels)

    if len(all_p) >= 2:
        fdr_sig = fdr_correction(all_p, alpha=0.05)
        group["fdr_correction"] = {
            label: {"p": float(p), "fdr_significant": bool(sig)}
            for label, p, sig in zip(all_labels, all_p, fdr_sig)
        }

    if original_json_path and Path(original_json_path).exists():
        try:
            with open(original_json_path) as f:
                orig = json.load(f)
            orig_gs = orig.get("group_statistics", {})
            comparison = {}
            metric_map = {
                "spectral_radius": "spectral_radius_awake_vs_sed_run1",
                "min_eigenvalue_gap": "eigenvalue_gap_awake_vs_sed_run1",
            }
            for shared_key, orig_key in metric_map.items():
                if shared_key in group and orig_key in orig_gs:
                    comparison[shared_key] = {
                        "original_d": orig_gs[orig_key].get("cohens_d"),
                        "original_p": orig_gs[orig_key].get("p"),
                        "shared_d": group[shared_key]["cohens_d"],
                        "shared_p": group[shared_key]["p"],
                    }
            group["comparison_to_original"] = comparison
        except Exception as e:
            group["comparison_to_original"] = {"error": str(e)}

    return group


def plot_comparison(all_results, group_stats, output_dir):
    valid = [s for s in all_results if s is not None]
    if len(valid) < 3:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Shared-Subspace PCA — Propofol State Comparisons",
                 fontsize=14, fontweight="bold")

    metrics = [
        ("mean_spectral_radius", "Spectral Radius", "spectral_radius"),
        ("mean_eigenvalue_gap", "Min Eigenvalue Gap", "min_eigenvalue_gap"),
        ("mean_median_nn_gap", "Median NN Gap", "median_nn_gap"),
        ("mean_p10_nn_gap", "P10 NN Gap", "p10_nn_gap"),
    ]

    for idx, (key, ylabel, gs_key) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        awake_vals = [s["awake"][key] for s in valid]
        sed_vals = [s["sedation_run1"][key] for s in valid]
        for a, s in zip(awake_vals, sed_vals):
            ax.plot([0, 1], [a, s], "o-", color="gray", alpha=0.4, markersize=4)
        ax.plot(0, np.mean(awake_vals), "D", color="steelblue", markersize=10, zorder=5)
        ax.plot(1, np.mean(sed_vals), "D", color="coral", markersize=10, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Awake", "Sedation"])
        ax.set_ylabel(ylabel)
        gs = group_stats.get(gs_key, {})
        ax.set_title(f"{ylabel}\nt={gs.get('t', 0):.2f}, p={gs.get('p', 1):.4f}, d={gs.get('cohens_d', 0):.2f}")

    ax = axes[1, 1]
    awake_r = [s["awake"]["spectral_sensitivity"]["r"] for s in valid]
    sed_r = [s["sedation_run1"]["spectral_sensitivity"]["r"] for s in valid]
    for a, s in zip(awake_r, sed_r):
        ax.plot([0, 1], [a, s], "o-", color="gray", alpha=0.4, markersize=4)
    ax.plot(0, np.mean(awake_r), "D", color="steelblue", markersize=10, zorder=5)
    ax.plot(1, np.mean(sed_r), "D", color="coral", markersize=10, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Awake", "Sedation"])
    ax.set_ylabel("Spectral Sensitivity r")
    gs = group_stats.get("spectral_sensitivity", {})
    ax.set_title(f"Spectral Sensitivity\nt={gs.get('t', 0):.2f}, p={gs.get('p', 1):.4f}, d={gs.get('cohens_d', 0):.2f}")

    axes[1, 2].axis("off")
    comp = group_stats.get("comparison_to_original", {})
    if comp and "error" not in comp:
        text = "Original vs Shared-Subspace:\n\n"
        for metric, vals in comp.items():
            text += f"{metric}:\n"
            text += f"  orig d={vals['original_d']:.2f}, p={vals['original_p']:.4f}\n"
            text += f"  shared d={vals['shared_d']:.2f}, p={vals['shared_p']:.4f}\n\n"
        axes[1, 2].text(0.1, 0.5, text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment="center", fontfamily="monospace")

    plt.tight_layout()
    plt.savefig(output_dir / "shared_subspace_propofol.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure: {output_dir / 'shared_subspace_propofol.png'}")


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
    log("SHARED-SUBSPACE PCA — Propofol EP Robustness Test")
    log(f"Dataset: ds005620 ({DATA_ROOT})")
    log(f"Parameters: window={WINDOW_SEC}s, step={STEP_SEC}s, "
        f"PCA={N_COMPONENTS} (SHARED), downsample={DOWNSAMPLE_TO} Hz")
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

    orig_json = RESULTS_DIR / "ep_propofol_eeg.json"
    group_stats = compute_group_statistics(all_results, original_json_path=orig_json)

    if group_stats:
        log(f"\n  Group statistics (shared subspace):")
        for key, val in group_stats.items():
            if key not in ("fdr_correction", "comparison_to_original"):
                log(f"    {key}: {val}")
        if "fdr_correction" in group_stats:
            log(f"    FDR correction:")
            for label, info in group_stats["fdr_correction"].items():
                log(f"      {label}: p={info['p']:.6f} sig={info['fdr_significant']}")
        if "comparison_to_original" in group_stats:
            comp = group_stats["comparison_to_original"]
            if "error" not in comp:
                log(f"\n  Comparison to per-state PCA:")
                for metric, vals in comp.items():
                    log(f"    {metric}: orig d={vals['original_d']:.2f} -> shared d={vals['shared_d']:.2f}")

    plot_comparison(all_results, group_stats, FIG_DIR)

    out = {
        "analysis": "shared_subspace_propofol",
        "n_subjects": len(all_results),
        "parameters": {
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "n_components": N_COMPONENTS,
            "pca_mode": "shared_subspace",
            "downsample_to": DOWNSAMPLE_TO,
            "line_freq": LINE_FREQ,
            "seed": SEED,
        },
        "excluded_subjects": EXCLUDED_SUBJECTS,
        "subjects": all_results,
        "group_statistics": group_stats,
    }

    out_path = RESULTS_DIR / "ep_shared_subspace_propofol.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=default_ser)
    log(f"\n  Results: {out_path}")
    log(f"\n{'='*70}")
    log("DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()
