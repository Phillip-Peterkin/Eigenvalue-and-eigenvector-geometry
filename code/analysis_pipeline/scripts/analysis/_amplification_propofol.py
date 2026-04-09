"""Propofol state-dependent transient amplification analysis.

Scientific question
-------------------
Does transient amplification (Kreiss constant, hump magnitude) change
systematically between awake and propofol-sedated states? If amplification
tracks pharmacological state, then the amplification is tied to circuit
state, not a fixed fitting artifact. Note: the direction of change is
not assumed a priori — propofol increases slow-wave power and
autocorrelation, which may increase operator non-normality.

What this analysis CAN establish
---------------------------------
- Whether fitted operator amplification varies with pharmacological state.
- Whether the direction of change is consistent with GABAergic stabilization.
- Whether amplification is a state-dependent property, not a fixed artifact.

What this analysis CANNOT establish
------------------------------------
- That propofol directly causes reduced amplification (confounded by
  arousal, metabolic rate, and other co-varying factors).
- That the fitted operator's amplification reflects true neural amplification
  (same caveat as the iEEG analysis).

Pipeline stage: feature construction + statistical inference
Dataset: ds005620 — 20 subjects, scalp EEG, awake vs propofol sedation
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

from cmcc.preprocess.scalp_eeg import load_ds005620_subject, preprocess_scalp_eeg
from cmcc.analysis.dynamical_systems import estimate_jacobian
from cmcc.features.transient_amplification import (
    analyze_jacobian_amplification,
    compute_hump_magnitude,
    compute_model_free_energy_growth,
    compute_out_of_sample_prediction,
)

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("PROPOFOL_DATA_ROOT", "./data/ds005620"))
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"
FIG_DIR = CMCC_ROOT / "results" / "figures" / "amplification_propofol"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 0.5
STEP_SEC = 0.1
N_COMPONENTS = 15
DOWNSAMPLE_TO = 500.0
LINE_FREQ = 50.0
MAX_HORIZON = 50
SEED = 42

EXCLUDED_SUBJECTS = ["sub-1037"]
SUBJECT_IDS = [
    "1010", "1016", "1017", "1022", "1024", "1033", "1036",
    "1045", "1046", "1054", "1055", "1057", "1060", "1061",
    "1062", "1064", "1067", "1068", "1071", "1074",
]


def log(msg):
    print(msg, flush=True)


def load_condition(subject_id, condition):
    if condition == "awake":
        task, acq, run = "awake", "EC", None
    elif condition.startswith("sed_run-"):
        run_num = int(condition.split("-")[1])
        task, acq, run = "sed", "rest", run_num
    else:
        raise ValueError(f"Unknown condition: {condition}")

    try:
        raw = load_ds005620_subject(subject_id, DATA_ROOT, task=task, acq=acq, run=run)
    except FileNotFoundError:
        return None, None

    data_pca, sfreq, info = preprocess_scalp_eeg(
        raw, line_freq=LINE_FREQ, downsample_to=DOWNSAMPLE_TO,
        n_components=N_COMPONENTS,
    )
    return data_pca, sfreq


OOS_PREDICT_STEPS = 5


def analyze_condition_amplification(data_pca, sfreq, condition_label):
    window_samples = int(WINDOW_SEC * sfreq)
    step_samples = int(STEP_SEC * sfreq)

    jac_result = estimate_jacobian(
        data_pca, window_size=window_samples, step_size=step_samples,
    )
    n_windows = len(jac_result.window_centers)

    amp = analyze_jacobian_amplification(jac_result.jacobians, max_horizon=MAX_HORIZON)

    hump_mags = np.array([
        compute_hump_magnitude(amp["envelopes"][w])
        for w in range(n_windows)
    ])

    stable_mask = jac_result.spectral_radius < 1.0
    n_stable = int(np.sum(stable_mask))
    frac_stable = float(n_stable / n_windows) if n_windows > 0 else 0.0

    if n_stable > 0:
        stable_kreiss = amp["kreiss_constants"][stable_mask]
        stable_amp_ratio = amp["amplification_ratios"][stable_mask]
        stable_hump_mag = hump_mags[stable_mask]
        stable_cond = jac_result.condition_numbers[stable_mask]
        stable_rho = jac_result.spectral_radius[stable_mask]
        stable_envelopes = amp["envelopes"][stable_mask]
        stable_mean_env = np.nanmean(stable_envelopes, axis=0)
    else:
        stable_kreiss = np.array([np.nan])
        stable_amp_ratio = np.array([np.nan])
        stable_hump_mag = np.array([np.nan])
        stable_cond = np.array([np.nan])
        stable_rho = np.array([np.nan])
        stable_mean_env = amp["mean_envelope"]

    mfeg = compute_model_free_energy_growth(
        data_pca, sfreq=sfreq,
        percentile_threshold=95.0,
        horizon_sec=0.1,
    )

    oos = compute_out_of_sample_prediction(
        jac_result.jacobians, data_pca,
        jac_result.window_centers, window_samples,
        n_predict_steps=OOS_PREDICT_STEPS,
    )

    return {
        "condition": condition_label,
        "n_windows": n_windows,
        "n_stable_windows": n_stable,
        "fraction_stable": frac_stable,
        "kreiss_mean": float(np.mean(amp["kreiss_constants"])),
        "kreiss_std": float(np.std(amp["kreiss_constants"])),
        "kreiss_median": float(np.median(amp["kreiss_constants"])),
        "amplification_ratio_mean": float(np.mean(amp["amplification_ratios"])),
        "amplification_ratio_std": float(np.std(amp["amplification_ratios"])),
        "amplification_ratio_median": float(np.median(amp["amplification_ratios"])),
        "hump_fraction": float(np.mean(amp["has_hump"])),
        "hump_magnitude_mean": float(np.mean(hump_mags)),
        "hump_magnitude_std": float(np.std(hump_mags)),
        "hump_magnitude_median": float(np.median(hump_mags)),
        "peak_time_mean": float(np.mean(amp["peak_times"])),
        "condition_number_mean": float(np.mean(jac_result.condition_numbers)),
        "condition_number_median": float(np.median(jac_result.condition_numbers)),
        "spectral_radius_mean": float(np.mean(jac_result.spectral_radius)),
        "spectral_radius_median": float(np.median(jac_result.spectral_radius)),
        "mean_envelope": amp["mean_envelope"].tolist(),
        "stable_only": {
            "n_windows": n_stable,
            "kreiss_mean": float(np.nanmean(stable_kreiss)),
            "kreiss_median": float(np.nanmedian(stable_kreiss)),
            "amplification_ratio_mean": float(np.nanmean(stable_amp_ratio)),
            "amplification_ratio_median": float(np.nanmedian(stable_amp_ratio)),
            "hump_magnitude_mean": float(np.nanmean(stable_hump_mag)),
            "hump_magnitude_median": float(np.nanmedian(stable_hump_mag)),
            "condition_number_mean": float(np.nanmean(stable_cond)),
            "spectral_radius_mean": float(np.nanmean(stable_rho)),
            "mean_envelope": stable_mean_env.tolist(),
        },
        "model_free_energy": {
            "n_events": mfeg["n_events"],
            "fraction_growing": mfeg["fraction_growing_at_peak"],
            "peak_growth_ratio": mfeg["peak_growth_ratio"],
            "peak_lag_sec": mfeg["peak_lag_sec"],
            "mean_trajectory": mfeg["mean_energy_trajectory"].tolist(),
        },
        "out_of_sample": {
            "mean_r2": oos["mean_r2"],
            "kreiss_vs_r2": {"r": oos["kreiss_vs_r2_corr"][0], "p": oos["kreiss_vs_r2_corr"][1]},
            "growth_correlation": {"r": oos["growth_correlation"][0], "p": oos["growth_correlation"][1]},
            "n_valid_windows": oos["n_valid_windows"],
            "n_predict_steps": OOS_PREDICT_STEPS,
        },
    }


def analyze_subject(subject_id):
    t0 = time.time()
    log(f"  {subject_id}...")

    awake_data, awake_sfreq = load_condition(subject_id, "awake")
    if awake_data is None:
        log(f"    SKIP: no awake data")
        return None

    log(f"    awake: {awake_data.shape[0]} comp, {awake_sfreq} Hz, "
        f"{awake_data.shape[1]/awake_sfreq:.1f}s")

    awake_result = analyze_condition_amplification(awake_data, awake_sfreq, "awake")
    log(f"    awake: kreiss={awake_result['kreiss_mean']:.3f}, "
        f"hump_mag={awake_result['hump_magnitude_mean']:.4f}, "
        f"mf_growth={awake_result['model_free_energy']['peak_growth_ratio']:.3f}")

    del awake_data
    gc.collect()

    sed_data, sed_sfreq = load_condition(subject_id, "sed_run-1")
    if sed_data is None:
        log(f"    SKIP: no sedation data")
        return None

    log(f"    sed: {sed_data.shape[0]} comp, {sed_sfreq} Hz, "
        f"{sed_data.shape[1]/sed_sfreq:.1f}s")

    sed_result = analyze_condition_amplification(sed_data, sed_sfreq, "sedation_run1")
    log(f"    sed: kreiss={sed_result['kreiss_mean']:.3f}, "
        f"hump_mag={sed_result['hump_magnitude_mean']:.4f}, "
        f"mf_growth={sed_result['model_free_energy']['peak_growth_ratio']:.3f}")

    del sed_data
    gc.collect()

    elapsed = time.time() - t0
    log(f"    delta_kreiss={awake_result['kreiss_mean'] - sed_result['kreiss_mean']:.3f}, "
        f"{elapsed:.1f}s")

    return {
        "subject": subject_id,
        "awake": awake_result,
        "sedation_run1": sed_result,
        "delta_kreiss": awake_result["kreiss_mean"] - sed_result["kreiss_mean"],
        "delta_kreiss_median": awake_result["kreiss_median"] - sed_result["kreiss_median"],
        "delta_kreiss_stable": (
            awake_result["stable_only"]["kreiss_median"] -
            sed_result["stable_only"]["kreiss_median"]
        ),
        "delta_amplification_ratio": awake_result["amplification_ratio_mean"] - sed_result["amplification_ratio_mean"],
        "delta_amplification_ratio_median": awake_result["amplification_ratio_median"] - sed_result["amplification_ratio_median"],
        "delta_hump_magnitude": awake_result["hump_magnitude_mean"] - sed_result["hump_magnitude_mean"],
        "delta_hump_magnitude_median": awake_result["hump_magnitude_median"] - sed_result["hump_magnitude_median"],
        "delta_condition_number": awake_result["condition_number_mean"] - sed_result["condition_number_mean"],
        "delta_spectral_radius": awake_result["spectral_radius_mean"] - sed_result["spectral_radius_mean"],
        "delta_model_free_growth": (
            awake_result["model_free_energy"]["peak_growth_ratio"] -
            sed_result["model_free_energy"]["peak_growth_ratio"]
        ),
        "awake_fraction_stable": awake_result["fraction_stable"],
        "sed_fraction_stable": sed_result["fraction_stable"],
        "elapsed_s": elapsed,
    }


def compute_group_statistics(subjects_data):
    n = len(subjects_data)
    deltas = {
        "kreiss": [s["delta_kreiss"] for s in subjects_data],
        "kreiss_median": [s["delta_kreiss_median"] for s in subjects_data],
        "kreiss_stable": [s["delta_kreiss_stable"] for s in subjects_data],
        "amplification_ratio": [s["delta_amplification_ratio"] for s in subjects_data],
        "amplification_ratio_median": [s["delta_amplification_ratio_median"] for s in subjects_data],
        "hump_magnitude": [s["delta_hump_magnitude"] for s in subjects_data],
        "hump_magnitude_median": [s["delta_hump_magnitude_median"] for s in subjects_data],
        "condition_number": [s["delta_condition_number"] for s in subjects_data],
        "spectral_radius": [s["delta_spectral_radius"] for s in subjects_data],
        "model_free_growth": [s["delta_model_free_growth"] for s in subjects_data],
    }

    group = {}
    for key, vals in deltas.items():
        arr = np.array(vals)
        t_stat, p_val = sp_stats.ttest_1samp(arr, 0.0)
        wilcox_stat, wilcox_p = sp_stats.wilcoxon(arr)
        effect_d = float(np.mean(arr) / max(np.std(arr, ddof=1), 1e-30))
        group[f"delta_{key}"] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "median": float(np.median(arr)),
            "t_stat": float(t_stat),
            "p_ttest": float(p_val),
            "wilcoxon_stat": float(wilcox_stat),
            "p_wilcoxon": float(wilcox_p),
            "cohens_d": effect_d,
            "n": n,
            "n_positive": int(np.sum(arr > 0)),
            "n_negative": int(np.sum(arr < 0)),
        }

    awake_kreiss = [s["awake"]["kreiss_mean"] for s in subjects_data]
    sed_kreiss = [s["sedation_run1"]["kreiss_mean"] for s in subjects_data]
    awake_mf = [s["awake"]["model_free_energy"]["peak_growth_ratio"] for s in subjects_data]
    sed_mf = [s["sedation_run1"]["model_free_energy"]["peak_growth_ratio"] for s in subjects_data]
    awake_oos = [s["awake"]["out_of_sample"]["mean_r2"] for s in subjects_data]
    sed_oos = [s["sedation_run1"]["out_of_sample"]["mean_r2"] for s in subjects_data]

    awake_frac_stable = [s["awake_fraction_stable"] for s in subjects_data]
    sed_frac_stable = [s["sed_fraction_stable"] for s in subjects_data]
    group["stability_summary"] = {
        "awake_fraction_stable_mean": float(np.mean(awake_frac_stable)),
        "awake_fraction_stable_min": float(np.min(awake_frac_stable)),
        "sed_fraction_stable_mean": float(np.mean(sed_frac_stable)),
        "sed_fraction_stable_min": float(np.min(sed_frac_stable)),
    }

    group["awake_kreiss_summary"] = {
        "mean": float(np.mean(awake_kreiss)),
        "std": float(np.std(awake_kreiss)),
        "median": float(np.median(awake_kreiss)),
    }
    group["sedation_kreiss_summary"] = {
        "mean": float(np.mean(sed_kreiss)),
        "std": float(np.std(sed_kreiss)),
        "median": float(np.median(sed_kreiss)),
    }
    group["awake_model_free_growth_summary"] = {
        "mean": float(np.mean(awake_mf)),
        "std": float(np.std(awake_mf)),
    }
    group["sedation_model_free_growth_summary"] = {
        "mean": float(np.mean(sed_mf)),
        "std": float(np.std(sed_mf)),
    }
    group["awake_oos_r2_summary"] = {
        "mean": float(np.nanmean(awake_oos)),
        "std": float(np.nanstd(awake_oos)),
    }
    group["sedation_oos_r2_summary"] = {
        "mean": float(np.nanmean(sed_oos)),
        "std": float(np.nanstd(sed_oos)),
    }

    return group


def make_summary_figure(subjects_data, group_stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(subjects_data)
    awake_k = [s["awake"]["stable_only"]["kreiss_median"] for s in subjects_data]
    sed_k = [s["sedation_run1"]["stable_only"]["kreiss_median"] for s in subjects_data]
    awake_mf = [s["awake"]["model_free_energy"]["peak_growth_ratio"] for s in subjects_data]
    sed_mf = [s["sedation_run1"]["model_free_energy"]["peak_growth_ratio"] for s in subjects_data]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    for i in range(n):
        ax.plot([0, 1], [awake_k[i], sed_k[i]], "o-", alpha=0.4, color="gray")
    ax.plot([0, 1], [np.median(awake_k), np.median(sed_k)], "s-", color="red", linewidth=2, markersize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Awake", "Sedation"])
    ax.set_ylabel("Median Kreiss constant (stable windows)")
    gs = group_stats.get("delta_kreiss_stable", {})
    ax.set_title(f"Kreiss (stable, median): awake vs sedation\np(Wilcoxon)={gs.get('p_wilcoxon', 1):.4f}, d={gs.get('cohens_d', 0):.2f}")

    ax = axes[0, 1]
    for i in range(n):
        ax.plot([0, 1], [awake_mf[i], sed_mf[i]], "o-", alpha=0.4, color="gray")
    ax.plot([0, 1], [np.median(awake_mf), np.median(sed_mf)], "s-", color="blue", linewidth=2, markersize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Awake", "Sedation"])
    ax.set_ylabel("Peak growth ratio (model-free)")
    gs = group_stats.get("delta_model_free_growth", {})
    ax.set_title(f"Model-free energy growth\np(Wilcoxon)={gs.get('p_wilcoxon', 1):.4f}, d={gs.get('cohens_d', 0):.2f}")

    ax = axes[1, 0]
    for s in subjects_data:
        env = np.array(s["awake"]["stable_only"]["mean_envelope"])
        ax.plot(env, alpha=0.3, color="blue")
    for s in subjects_data:
        env = np.array(s["sedation_run1"]["stable_only"]["mean_envelope"])
        ax.plot(env, alpha=0.3, color="red")
    mean_awake = np.mean([s["awake"]["stable_only"]["mean_envelope"] for s in subjects_data], axis=0)
    mean_sed = np.mean([s["sedation_run1"]["stable_only"]["mean_envelope"] for s in subjects_data], axis=0)
    ax.plot(mean_awake, "b-", linewidth=2, label="Awake (mean)")
    ax.plot(mean_sed, "r-", linewidth=2, label="Sedation (mean)")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Time step k")
    ax.set_ylabel("||A^k||_2")
    ax.set_title("Energy envelopes (stable windows only)")
    ax.legend()

    ax = axes[1, 1]
    deltas = [s["delta_kreiss_stable"] for s in subjects_data]
    ax.hist(deltas, bins=10, edgecolor="black", alpha=0.7)
    ax.axvline(x=0, color="red", linestyle="--")
    ax.set_xlabel("Delta Kreiss stable-median (awake - sedation)")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-subject Kreiss change (stable)\n{sum(1 for d in deltas if d > 0)}/{n} positive")

    fig.suptitle("Propofol Amplification Analysis\n(Quasi-causal: does amplification track pharmacological state?)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = FIG_DIR / "amplification_propofol_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Figure saved: {out_path}")


def main():
    log("=" * 60)
    log("PROPOFOL TRANSIENT AMPLIFICATION ANALYSIS")
    log("Quasi-causal test: does amplification track pharmacological state?")
    log("=" * 60)

    subjects_data = []
    for subj_id in SUBJECT_IDS:
        if f"sub-{subj_id}" in EXCLUDED_SUBJECTS:
            continue
        result = analyze_subject(subj_id)
        if result is not None:
            subjects_data.append(result)
        gc.collect()

    log(f"\n{len(subjects_data)} subjects completed")

    group = compute_group_statistics(subjects_data)

    log("\n--- Group statistics (all windows) ---")
    for key in ["delta_kreiss", "delta_amplification_ratio", "delta_hump_magnitude",
                "delta_condition_number", "delta_model_free_growth"]:
        g = group.get(key, {})
        log(f"  {key}: mean={g.get('mean', 0):.4f}, median={g.get('median', 0):.4f}, "
            f"d={g.get('cohens_d', 0):.3f}, p(wilcox)={g.get('p_wilcoxon', 1):.4f}, "
            f"{g.get('n_positive', 0)}/{g.get('n', 0)} positive")

    log("\n--- Group statistics (stable-only & median) ---")
    for key in ["delta_kreiss_median", "delta_kreiss_stable",
                "delta_amplification_ratio_median", "delta_hump_magnitude_median"]:
        g = group.get(key, {})
        log(f"  {key}: mean={g.get('mean', 0):.4f}, median={g.get('median', 0):.4f}, "
            f"d={g.get('cohens_d', 0):.3f}, p(wilcox)={g.get('p_wilcoxon', 1):.4f}, "
            f"{g.get('n_positive', 0)}/{g.get('n', 0)} positive")

    stab = group.get("stability_summary", {})
    log(f"\n  Stability: awake {stab.get('awake_fraction_stable_mean', 0):.3f} "
        f"(min {stab.get('awake_fraction_stable_min', 0):.3f}), "
        f"sed {stab.get('sed_fraction_stable_mean', 0):.3f} "
        f"(min {stab.get('sed_fraction_stable_min', 0):.3f})")

    output = {
        "analysis": "amplification_propofol",
        "description": (
            "Propofol state-dependent transient amplification analysis. "
            "Tests whether fitted operator amplification (Kreiss constant, "
            "hump magnitude) changes between awake and propofol-sedated states. "
            "Includes model-free energy growth (no model fitting) and "
            "out-of-sample trajectory prediction as complementary evidence. "
            "All fitted-operator results are properties of the estimator. "
            "Model-free results are measured directly from the signal."
        ),
        "parameters": {
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "n_components": N_COMPONENTS,
            "max_horizon": MAX_HORIZON,
            "downsample_to": DOWNSAMPLE_TO,
            "oos_predict_steps": OOS_PREDICT_STEPS,
        },
        "n_subjects": len(subjects_data),
        "group_statistics": group,
        "subjects": subjects_data,
        "guardrail": (
            "This is a quasi-causal analysis. Propofol changes multiple "
            "physiological parameters simultaneously. A significant state "
            "effect is consistent with circuit-level amplification changes "
            "but does not prove direct causation. Model-free energy growth "
            "is the strongest evidence because it involves no model fitting."
        ),
    }

    out_path = RESULTS_DIR / "amplification_propofol.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nResults saved: {out_path}")

    try:
        make_summary_figure(subjects_data, group)
    except Exception as e:
        log(f"  Figure generation failed: {e}")

    log("\nDone.")


if __name__ == "__main__":
    main()
