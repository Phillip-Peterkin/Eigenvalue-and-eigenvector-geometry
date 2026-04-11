"""
Propofol amplification hardening analyses.

Three complementary tests that strengthen the propofol transient
amplification finding from _amplification_propofol.py:

1. Dose-response across sedation runs
   - Analyze all 3 sedation runs per subject
   - Test for monotonic Kreiss trend (Jonckheere-Terpstra or Page trend)
   - Prediction: if amplification tracks pharmacological depth, Kreiss
     should increase monotonically from awake -> run-1 -> run-2 -> run-3

2. Spectral-radius-controlled Kreiss
   - Regress out spectral radius from Kreiss per-window
   - Compare residual between conditions
   - Tests whether eigenvector non-orthogonality changes, not just
     stability margin

What this analysis CAN establish
---------------------------------
- Whether the awake-sedation Kreiss difference is a dose-dependent trend
  (not a binary artifact of comparing two different recording sessions)
- Whether the Kreiss difference survives control for spectral radius
  (non-normality change, not just stability change)

What this analysis CANNOT establish
------------------------------------
- Pharmacological mechanism (propofol confounds remain)
- That fitted operator amplification = true neural amplification

Pipeline stage: statistical inference (hardening layer)
Dataset: ds005620 — 20 subjects, scalp EEG, awake + 3 sedation runs
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
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*toeplitz.*", category=RuntimeWarning)

from cmcc.preprocess.scalp_eeg import load_ds005620_subject, preprocess_scalp_eeg
from cmcc.analysis.dynamical_systems import estimate_jacobian
from cmcc.features.transient_amplification import (
    analyze_jacobian_amplification,
    compute_hump_magnitude,
    compute_residual_kreiss,
)

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("DS005620_ROOT", r"c:\openneuro\ds005620"))
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

CONDITIONS = ["awake", "sed_run-1", "sed_run-2", "sed_run-3"]


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


def analyze_condition(data_pca, sfreq):
    window_samples = int(WINDOW_SEC * sfreq)
    step_samples = int(STEP_SEC * sfreq)

    jac_result = estimate_jacobian(
        data_pca, window_size=window_samples, step_size=step_samples,
    )
    n_windows = len(jac_result.window_centers)

    amp = analyze_jacobian_amplification(jac_result.jacobians, max_horizon=MAX_HORIZON)

    stable_mask = jac_result.spectral_radius < 1.0
    n_stable = int(np.sum(stable_mask))

    kreiss_median = float(np.median(amp["kreiss_constants"]))

    if n_stable > 0:
        stable_kreiss_median = float(np.median(amp["kreiss_constants"][stable_mask]))
    else:
        stable_kreiss_median = float("nan")

    return {
        "n_windows": n_windows,
        "n_stable": n_stable,
        "fraction_stable": float(n_stable / n_windows) if n_windows > 0 else 0.0,
        "kreiss_median": kreiss_median,
        "kreiss_mean": float(np.mean(amp["kreiss_constants"])),
        "stable_kreiss_median": stable_kreiss_median,
        "spectral_radius_median": float(np.median(jac_result.spectral_radius)),
        "spectral_radius_mean": float(np.mean(jac_result.spectral_radius)),
        "condition_number_median": float(np.median(jac_result.condition_numbers)),
        "kreiss_array": amp["kreiss_constants"],
        "spectral_radius_array": jac_result.spectral_radius,
    }


def analyze_subject_dose_response(subject_id):
    t0 = time.time()
    log(f"  {subject_id}...")

    cond_results = {}
    for cond in CONDITIONS:
        data, sfreq = load_condition(subject_id, cond)
        if data is None:
            log(f"    SKIP {cond}: no data")
            cond_results[cond] = None
            continue

        log(f"    {cond}: {data.shape[0]} comp, {sfreq} Hz, "
            f"{data.shape[1]/sfreq:.1f}s")

        cond_results[cond] = analyze_condition(data, sfreq)
        log(f"    {cond}: kreiss_med={cond_results[cond]['kreiss_median']:.3f}, "
            f"rho_med={cond_results[cond]['spectral_radius_median']:.4f}")

        del data
        gc.collect()

    if cond_results["awake"] is None:
        return None

    all_kreiss = []
    all_rho = []
    all_labels = []
    for cond in CONDITIONS:
        if cond_results.get(cond) is not None:
            kr = cond_results[cond]["kreiss_array"]
            rh = cond_results[cond]["spectral_radius_array"]
            all_kreiss.append(kr)
            all_rho.append(rh)
            all_labels.append(np.full(len(kr), CONDITIONS.index(cond)))

    if len(all_kreiss) >= 2:
        pooled_kr = np.concatenate(all_kreiss)
        pooled_rho = np.concatenate(all_rho)
        pooled_labels = np.concatenate(all_labels)

        pooled_resid = compute_residual_kreiss(pooled_kr, pooled_rho)

        if len(pooled_resid["residuals"]) > 0:
            stable_mask = pooled_rho < 1.0
            residuals = pooled_resid["residuals"]
            labels_stable = pooled_labels[stable_mask]

            for cond in CONDITIONS:
                cond_idx = CONDITIONS.index(cond)
                if cond_results.get(cond) is not None:
                    cond_mask = labels_stable == cond_idx
                    if np.sum(cond_mask) > 0:
                        cond_results[cond]["pooled_residual_kreiss_mean"] = float(np.mean(residuals[cond_mask]))
                        cond_results[cond]["pooled_residual_kreiss_median"] = float(np.median(residuals[cond_mask]))
                    else:
                        cond_results[cond]["pooled_residual_kreiss_mean"] = float("nan")
                        cond_results[cond]["pooled_residual_kreiss_median"] = float("nan")

            log(f"    pooled residual-kreiss slope={pooled_resid['slope']:.3f}, R²={pooled_resid['r_squared']:.3f}")
            for cond in CONDITIONS:
                if cond_results.get(cond) is not None and "pooled_residual_kreiss_mean" in cond_results[cond]:
                    log(f"    {cond} pooled_resid_mean={cond_results[cond]['pooled_residual_kreiss_mean']:.4f}")

    for cond in CONDITIONS:
        if cond_results.get(cond) is not None:
            cond_results[cond].pop("kreiss_array", None)
            cond_results[cond].pop("spectral_radius_array", None)

    elapsed = time.time() - t0
    log(f"    {elapsed:.1f}s")

    return {
        "subject": subject_id,
        "conditions": cond_results,
        "elapsed_s": elapsed,
    }


def page_trend_test(ranks_matrix, n_permutations=10000, seed=42):
    """Page's L test for monotonic trend across ordered conditions.

    Parameters
    ----------
    ranks_matrix : np.ndarray, shape (n_subjects, k_conditions)
        Ranks within each subject row.
    n_permutations : int
        Number of permutations for p-value estimation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        L : float
            Page's L statistic.
        p_value : float
            Permutation-based one-sided p-value (fraction of
            permuted L >= observed L).
    """
    n, k = ranks_matrix.shape
    weights = np.arange(1, k + 1)
    col_rank_sums = np.sum(ranks_matrix, axis=0)
    L_obs = float(np.sum(weights * col_rank_sums))

    rng = np.random.default_rng(seed)
    n_geq = 0
    for _ in range(n_permutations):
        perm = ranks_matrix.copy()
        for i in range(n):
            rng.shuffle(perm[i])
        col_sums_perm = np.sum(perm, axis=0)
        L_perm = float(np.sum(weights * col_sums_perm))
        if L_perm >= L_obs:
            n_geq += 1

    p_value = (n_geq + 1) / (n_permutations + 1)

    return {"L": L_obs, "p_value": p_value}


def compute_dose_response_statistics(subjects_data):
    valid_all = []
    for s in subjects_data:
        conds = s["conditions"]
        if all(conds.get(c) is not None for c in CONDITIONS):
            valid_all.append(s)

    log(f"\n  Dose-response: {len(valid_all)} subjects with all 4 conditions")

    if len(valid_all) < 5:
        return {"n_subjects_all_runs": len(valid_all), "insufficient_data": True}

    kreiss_matrix = np.array([
        [s["conditions"][c]["stable_kreiss_median"] for c in CONDITIONS]
        for s in valid_all
    ])

    resid_matrix = np.array([
        [s["conditions"][c].get("pooled_residual_kreiss_mean", float("nan")) for c in CONDITIONS]
        for s in valid_all
    ])

    rho_matrix = np.array([
        [s["conditions"][c]["spectral_radius_median"] for c in CONDITIONS]
        for s in valid_all
    ])

    n_subj = kreiss_matrix.shape[0]
    ranks = np.zeros_like(kreiss_matrix)
    for i in range(n_subj):
        ranks[i] = sp_stats.rankdata(kreiss_matrix[i])

    page_result = page_trend_test(ranks)

    n_monotonic = 0
    for i in range(n_subj):
        row = kreiss_matrix[i]
        if all(row[j] <= row[j+1] for j in range(len(row)-1)):
            n_monotonic += 1

    pairwise = {}
    for j in range(1, len(CONDITIONS)):
        delta = kreiss_matrix[:, j] - kreiss_matrix[:, 0]
        t, p = sp_stats.ttest_1samp(delta, 0.0)
        w, pw = sp_stats.wilcoxon(delta)
        sd = np.std(delta, ddof=1)
        if sd < 1e-12:
            log(f"    WARNING: near-zero SD for {CONDITIONS[j]} Cohen's d")
        d = float(np.mean(delta) / max(sd, 1e-30))
        pairwise[f"awake_vs_{CONDITIONS[j]}"] = {
            "delta_mean": float(np.mean(delta)),
            "delta_median": float(np.median(delta)),
            "cohens_d": d,
            "p_ttest": float(p),
            "p_wilcoxon": float(pw),
            "n_positive": int(np.sum(delta > 0)),
        }

    resid_pairwise = {}
    for j in range(1, len(CONDITIONS)):
        delta = resid_matrix[:, j] - resid_matrix[:, 0]
        if np.std(delta) > 1e-30:
            t, p = sp_stats.ttest_1samp(delta, 0.0)
            w, pw = sp_stats.wilcoxon(delta)
            d = float(np.mean(delta) / max(np.std(delta, ddof=1), 1e-30))
        else:
            t, p, w, pw, d = 0, 1.0, 0, 1.0, 0.0
        resid_pairwise[f"awake_vs_{CONDITIONS[j]}"] = {
            "delta_mean": float(np.mean(delta)),
            "delta_median": float(np.median(delta)),
            "cohens_d": float(d),
            "p_ttest": float(p),
            "p_wilcoxon": float(pw),
            "n_positive": int(np.sum(delta > 0)),
        }

    return {
        "n_subjects_all_runs": len(valid_all),
        "kreiss_by_condition": {
            c: {
                "mean": float(np.mean(kreiss_matrix[:, i])),
                "median": float(np.median(kreiss_matrix[:, i])),
                "std": float(np.std(kreiss_matrix[:, i])),
            }
            for i, c in enumerate(CONDITIONS)
        },
        "spectral_radius_by_condition": {
            c: {
                "mean": float(np.mean(rho_matrix[:, i])),
                "median": float(np.median(rho_matrix[:, i])),
            }
            for i, c in enumerate(CONDITIONS)
        },
        "page_L_statistic": page_result["L"],
        "page_L_p_value": page_result["p_value"],
        "n_strictly_monotonic": n_monotonic,
        "fraction_monotonic": float(n_monotonic / len(valid_all)),
        "pairwise_kreiss": pairwise,
        "pairwise_residual_kreiss": resid_pairwise,
        "residual_kreiss_by_condition": {
            c: {
                "mean": float(np.mean(resid_matrix[:, i])),
                "median": float(np.median(resid_matrix[:, i])),
            }
            for i, c in enumerate(CONDITIONS)
        },
    }


def compute_rho_controlled_statistics(subjects_data):
    valid = [s for s in subjects_data
             if s["conditions"].get("awake") is not None
             and s["conditions"].get("sed_run-1") is not None]

    if len(valid) < 5:
        return {"n_subjects": len(valid), "insufficient_data": True}

    awake_resid = [s["conditions"]["awake"].get("pooled_residual_kreiss_mean", float("nan")) for s in valid]
    sed_resid = [s["conditions"]["sed_run-1"].get("pooled_residual_kreiss_mean", float("nan")) for s in valid]
    delta_resid = np.array(sed_resid) - np.array(awake_resid)

    nan_mask = np.isnan(delta_resid)
    if np.any(nan_mask):
        log(f"  rho_controlled: dropping {int(np.sum(nan_mask))} subjects with NaN residuals")
        delta_resid = delta_resid[~nan_mask]
        awake_resid = [v for v, m in zip(awake_resid, nan_mask) if not m]
        sed_resid = [v for v, m in zip(sed_resid, nan_mask) if not m]
        valid = [v for v, m in zip(valid, nan_mask) if not m]

    if len(delta_resid) < 5:
        return {"n_subjects": len(delta_resid), "insufficient_data": True}

    t, p = sp_stats.ttest_1samp(delta_resid, 0.0)
    w, pw = sp_stats.wilcoxon(delta_resid)
    sd = np.std(delta_resid, ddof=1)
    if sd < 1e-12:
        log("  WARNING: near-zero SD in rho-controlled Cohen's d")
    d = float(np.mean(delta_resid) / max(sd, 1e-30))

    awake_rho = [s["conditions"]["awake"]["spectral_radius_median"] for s in valid]
    sed_rho = [s["conditions"]["sed_run-1"]["spectral_radius_median"] for s in valid]

    return {
        "n_subjects": len(valid),
        "delta_residual_kreiss": {
            "mean": float(np.mean(delta_resid)),
            "median": float(np.median(delta_resid)),
            "std": float(np.std(delta_resid, ddof=1)),
            "cohens_d": d,
            "t_stat": float(t),
            "p_ttest": float(p),
            "p_wilcoxon": float(pw),
            "n_positive": int(np.sum(delta_resid > 0)),
            "n_negative": int(np.sum(delta_resid < 0)),
        },
        "spectral_radius_summary": {
            "awake_median": float(np.median(awake_rho)),
            "sed_median": float(np.median(sed_rho)),
        },
        "interpretation": (
            "Positive residual delta = sedation has higher non-normality "
            "AFTER controlling for spectral radius (stability margin). "
            "This means the Kreiss increase is not merely from the system "
            "being closer to instability."
        ),
    }


def make_hardening_figure(subjects_data, dose_stats, rho_stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid_all = [
        s for s in subjects_data
        if all(s["conditions"].get(c) is not None for c in CONDITIONS)
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    if len(valid_all) >= 3:
        for s in valid_all:
            vals = [s["conditions"][c]["stable_kreiss_median"] for c in CONDITIONS]
            ax.plot(range(4), vals, "o-", alpha=0.3, color="gray")
        medians = [
            np.median([s["conditions"][c]["stable_kreiss_median"] for s in valid_all])
            for c in CONDITIONS
        ]
        ax.plot(range(4), medians, "s-", color="red", linewidth=2, markersize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Awake", "Sed R1", "Sed R2", "Sed R3"], fontsize=9)
    ax.set_ylabel("Stable Kreiss median")
    n_mono = dose_stats.get("n_strictly_monotonic", 0)
    n_all = dose_stats.get("n_subjects_all_runs", 0)
    ax.set_title(f"Dose-response: Kreiss across runs\n{n_mono}/{n_all} monotonic")

    ax = axes[0, 1]
    if len(valid_all) >= 3:
        for s in valid_all:
            vals = [s["conditions"][c]["spectral_radius_median"] for c in CONDITIONS]
            ax.plot(range(4), vals, "o-", alpha=0.3, color="gray")
        medians = [
            np.median([s["conditions"][c]["spectral_radius_median"] for s in valid_all])
            for c in CONDITIONS
        ]
        ax.plot(range(4), medians, "s-", color="blue", linewidth=2, markersize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Awake", "Sed R1", "Sed R2", "Sed R3"], fontsize=9)
    ax.set_ylabel("Spectral radius median")
    ax.set_title("Dose-response: Spectral radius")

    ax = axes[1, 0]
    if len(valid_all) >= 3:
        for s in valid_all:
            vals = [s["conditions"][c].get("pooled_residual_kreiss_mean", float("nan")) for c in CONDITIONS]
            ax.plot(range(4), vals, "o-", alpha=0.3, color="gray")
        medians = [
            np.median([s["conditions"][c].get("pooled_residual_kreiss_mean", 0.0) for s in valid_all])
            for c in CONDITIONS
        ]
        ax.plot(range(4), medians, "s-", color="green", linewidth=2, markersize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Awake", "Sed R1", "Sed R2", "Sed R3"], fontsize=9)
    ax.set_ylabel("Residual log-Kreiss (rho-controlled)")
    ax.set_title("Dose-response: Residual Kreiss\n(non-normality independent of stability)")

    ax = axes[1, 1]
    rs = rho_stats.get("delta_residual_kreiss", {})
    valid_aw_sed = [
        s for s in subjects_data
        if s["conditions"].get("awake") is not None
        and s["conditions"].get("sed_run-1") is not None
    ]
    valid_aw_sed = [
        s for s in valid_aw_sed
        if "pooled_residual_kreiss_mean" in s["conditions"].get("awake", {})
        and "pooled_residual_kreiss_mean" in s["conditions"].get("sed_run-1", {})
    ]
    if len(valid_aw_sed) >= 3:
        deltas = [
            s["conditions"]["sed_run-1"]["pooled_residual_kreiss_mean"]
            - s["conditions"]["awake"]["pooled_residual_kreiss_mean"]
            for s in valid_aw_sed
        ]
        ax.hist(deltas, bins=10, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--")
    ax.set_xlabel("Delta residual Kreiss (sed - awake)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Rho-controlled Kreiss: sed vs awake\n"
        f"d={rs.get('cohens_d', 0):.2f}, p={rs.get('p_wilcoxon', 1):.4f}"
    )

    fig.suptitle(
        "Propofol Amplification Hardening\n"
        "(Dose-response + spectral-radius control)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = FIG_DIR / "amplification_propofol_hardening.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Figure saved: {out_path}")


def main():
    log("=" * 60)
    log("PROPOFOL AMPLIFICATION HARDENING")
    log("Dose-response + spectral-radius-controlled Kreiss")
    log("=" * 60)

    subjects_data = []
    for subj_id in SUBJECT_IDS:
        if f"sub-{subj_id}" in EXCLUDED_SUBJECTS:
            continue
        result = analyze_subject_dose_response(subj_id)
        if result is not None:
            subjects_data.append(result)
        gc.collect()

    log(f"\n{len(subjects_data)} subjects completed")

    dose_stats = compute_dose_response_statistics(subjects_data)
    rho_stats = compute_rho_controlled_statistics(subjects_data)

    log("\n--- Dose-response results ---")
    ds = dose_stats
    if not ds.get("insufficient_data"):
        log(f"  N with all 4 conditions: {ds['n_subjects_all_runs']}")
        log(f"  Page L statistic: {ds['page_L_statistic']:.1f}, p={ds['page_L_p_value']:.4f}")
        log(f"  Strictly monotonic: {ds['n_strictly_monotonic']}/{ds['n_subjects_all_runs']}")
        for cond in CONDITIONS:
            ks = ds["kreiss_by_condition"][cond]
            log(f"  {cond}: kreiss median={ks['median']:.3f} (mean={ks['mean']:.3f})")
        for key, val in ds["pairwise_kreiss"].items():
            log(f"  {key}: d={val['cohens_d']:.3f}, p(w)={val['p_wilcoxon']:.4f}, "
                f"{val['n_positive']}/{ds['n_subjects_all_runs']} positive")

    log("\n--- Residual Kreiss (rho-controlled) results ---")
    for key, val in ds.get("pairwise_residual_kreiss", {}).items():
        log(f"  {key}: d={val['cohens_d']:.3f}, p(w)={val['p_wilcoxon']:.4f}, "
            f"{val['n_positive']}/{ds.get('n_subjects_all_runs', 0)} positive")

    log("\n--- Awake vs Sed-1 rho-controlled ---")
    rs = rho_stats.get("delta_residual_kreiss", {})
    log(f"  Delta residual Kreiss: mean={rs.get('mean', 0):.4f}, "
        f"d={rs.get('cohens_d', 0):.3f}, p(w)={rs.get('p_wilcoxon', 1):.4f}, "
        f"{rs.get('n_positive', 0)}/{rho_stats.get('n_subjects', 0)} positive")

    output = {
        "analysis": "amplification_propofol_hardening",
        "description": (
            "Two hardening analyses for the propofol amplification finding: "
            "(1) dose-response across all 3 sedation runs testing monotonic "
            "trend; (2) spectral-radius-controlled Kreiss testing whether "
            "non-normality (not just stability margin) changes with state."
        ),
        "parameters": {
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "n_components": N_COMPONENTS,
            "max_horizon": MAX_HORIZON,
            "downsample_to": DOWNSAMPLE_TO,
            "conditions": CONDITIONS,
        },
        "n_subjects": len(subjects_data),
        "dose_response": dose_stats,
        "rho_controlled": rho_stats,
        "subjects": subjects_data,
        "guardrail": (
            "Dose-response assumes sedation runs reflect increasing "
            "pharmacological depth. Residual-Kreiss assumes the log-linear "
            "relationship between Kreiss and spectral radius captures the "
            "stability-margin contribution. Deviations from log-linearity "
            "could produce spurious residuals."
        ),
    }

    out_path = RESULTS_DIR / "amplification_propofol_hardening.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nResults saved: {out_path}")

    try:
        make_hardening_figure(subjects_data, dose_stats, rho_stats)
    except Exception as e:
        log(f"  Figure generation failed: {e}")

    log("\nDone.")


if __name__ == "__main__":
    main()
