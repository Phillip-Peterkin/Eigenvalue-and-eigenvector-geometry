import os
"""EP Robustness & Validity Checks.

Three tests that strengthen the scientific validity of the EP findings:

1. SURROGATE DATA CONTROL — Phase-randomize EEG data, run EP pipeline on
   surrogates, show that spectral sensitivity r is flat (~0) in spectrally
   matched noise. Validates that the observed r is genuinely neural.

2. PARTIAL REGRESSION — Test whether the delta-delta correlation (r=-0.68)
   survives after controlling for alpha power changes. Uses existing JSON
   results (no raw data needed).

3. PCA COMPONENT ROBUSTNESS — Re-run propofol pipeline with 10 and 20
   PCA components (complementing the standard 15). Tests whether spectral
   sensitivity is robust to state-space dimensionality.

Uses propofol (ds005620) data for tests 1 and 3.

Methodological guardrails
-------------------------
- Phase-randomized surrogates constrain interpretation: absolute spectral
  sensitivity magnitude is not specific to neural temporal structure
  (group p = 0.23 with 200 surrogates per subject).
- Jackknife sensitivity addresses single-subject leverage only, not
  causal validity or generalizability.
- Multi-block sleep analysis addresses segment-selection robustness only,
  not independence of sleep-stage scoring from eigenvalue geometry.
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
from cmcc.analysis.ep_advanced import compute_spectral_radius_sensitivity

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("PROPOFOL_DATA_ROOT", "./data/ds005620"))
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"
FIG_DIR = CMCC_ROOT / "results" / "figures" / "ep_robustness"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 0.5
STEP_SEC = 0.1
DOWNSAMPLE_TO = 500.0
LINE_FREQ = 50.0
SEED = 42

EXCLUDED_SUBJECTS = ["sub-1037"]
SUBJECT_IDS = [
    "1010", "1016", "1017", "1022", "1024", "1033", "1036",
    "1045", "1046", "1054", "1055", "1057", "1060", "1061",
    "1062", "1064", "1067", "1068", "1071", "1074",
]

SURROGATE_SUBJECTS = ["1010", "1017", "1033", "1055", "1068"]
N_SURROGATES = 200
PCA_VARIANTS = [10, 15, 20]


def log(msg):
    print(msg, flush=True)


def phase_randomize(data, rng):
    n_ch, n_samp = data.shape
    fft_data = np.fft.rfft(data, axis=1)
    n_freq = fft_data.shape[1]
    phases = rng.uniform(0, 2 * np.pi, n_freq)
    phases[0] = 0.0
    if n_samp % 2 == 0:
        phases[-1] = 0.0
    phase_shift = np.exp(1j * phases)
    fft_surrogate = fft_data * phase_shift[np.newaxis, :]
    surrogate = np.fft.irfft(fft_surrogate, n=n_samp, axis=1)
    return surrogate


def run_surrogate_test():
    log("\n" + "=" * 70)
    log("TEST 1: SURROGATE DATA CONTROL")
    log(f"Subjects: {SURROGATE_SUBJECTS}")
    log(f"N surrogates per subject: {N_SURROGATES}")
    log("=" * 70)

    results = []
    n_comp = 15

    for subj in SURROGATE_SUBJECTS:
        log(f"\n  {subj}...")
        try:
            raw = load_ds005620_subject(subj, DATA_ROOT, task="awake", acq="EC")
        except FileNotFoundError:
            log(f"    SKIP: not found")
            continue

        data_pca, sfreq, info = preprocess_scalp_eeg(
            raw, line_freq=LINE_FREQ, downsample_to=DOWNSAMPLE_TO,
            n_components=n_comp,
        )

        ep_tc = compute_ep_proximity_timecourse(
            data_pca, sfreq=sfreq,
            window_sec=WINDOW_SEC, step_sec=STEP_SEC,
            max_channels=n_comp, seed=SEED,
        )
        real_r = compute_spectral_radius_sensitivity(
            ep_tc["jac_result"], ep_tc["ep_result"]
        )["r"]

        log(f"    Real r = {real_r:.4f}")

        surrogate_rs = []
        rng = np.random.default_rng(SEED)

        for s_i in range(N_SURROGATES):
            surr_data = phase_randomize(data_pca, rng)

            ep_tc_s = compute_ep_proximity_timecourse(
                surr_data, sfreq=sfreq,
                window_sec=WINDOW_SEC, step_sec=STEP_SEC,
                max_channels=n_comp, seed=SEED,
            )
            surr_r = compute_spectral_radius_sensitivity(
                ep_tc_s["jac_result"], ep_tc_s["ep_result"]
            )["r"]
            surrogate_rs.append(surr_r)

            del ep_tc_s
            gc.collect()

        mean_surr = float(np.mean(surrogate_rs))
        std_surr = float(np.std(surrogate_rs))
        p_surr = float(np.mean([abs(sr) >= abs(real_r) for sr in surrogate_rs]))

        log(f"    Surrogate r = {mean_surr:.4f} +/- {std_surr:.4f}")
        log(f"    p_surrogate = {p_surr:.3f}")

        results.append({
            "subject": subj,
            "real_r": float(real_r),
            "surrogate_mean_r": mean_surr,
            "surrogate_std_r": std_surr,
            "surrogate_rs": [float(r) for r in surrogate_rs],
            "p_surrogate": p_surr,
            "n_surrogates": N_SURROGATES,
        })

        del data_pca, raw
        gc.collect()

    if len(results) >= 2:
        real_rs = [r["real_r"] for r in results]
        surr_means = [r["surrogate_mean_r"] for r in results]
        t_val, p_val = sp_stats.ttest_rel(real_rs, surr_means)
        group = {
            "n_subjects": len(results),
            "mean_real_r": float(np.mean(real_rs)),
            "mean_surrogate_r": float(np.mean(surr_means)),
            "t_real_vs_surrogate": float(t_val),
            "p_real_vs_surrogate": float(p_val),
        }
        log(f"\n  GROUP: real r={group['mean_real_r']:.4f} vs surr r={group['mean_surrogate_r']:.4f}")
        log(f"  t={t_val:.3f}, p={p_val:.6f}")
    else:
        group = {}

    return {"per_subject": results, "group": group}


def run_partial_regression():
    log("\n" + "=" * 70)
    log("TEST 2: PARTIAL REGRESSION (Delta-Delta controlling for Alpha)")
    log("=" * 70)

    propofol_path = RESULTS_DIR / "ep_propofol_eeg.json"
    alpha_path = RESULTS_DIR / "gap_vs_alpha_test.json"

    if not propofol_path.exists():
        log("  ERROR: ep_propofol_eeg.json not found")
        return {"error": "missing propofol results"}
    if not alpha_path.exists():
        log("  ERROR: gap_vs_alpha_test.json not found")
        return {"error": "missing gap_vs_alpha results"}

    with open(propofol_path) as f:
        propofol = json.load(f)
    with open(alpha_path) as f:
        alpha_data = json.load(f)

    alpha_by_subj = {}
    for r in alpha_data["per_subject"]:
        sid = r["subject"]
        cond = r["condition"]
        if sid not in alpha_by_subj:
            alpha_by_subj[sid] = {}
        alpha_by_subj[sid][cond] = r["mean_alpha"]

    subjects = [s for s in propofol["subjects"] if s is not None]
    delta_r = []
    delta_gap = []
    delta_alpha = []
    valid_sids = []

    for s in subjects:
        sid = s["subject"]
        if sid not in alpha_by_subj:
            continue
        a_info = alpha_by_subj[sid]
        if "awake" not in a_info or "sed_run1" not in a_info:
            continue

        dr = s["delta_spectral_sensitivity_r_run1"]
        dg = s["delta_eigenvalue_gap_run1"]
        da = a_info["awake"] - a_info["sed_run1"]

        delta_r.append(dr)
        delta_gap.append(dg)
        delta_alpha.append(da)
        valid_sids.append(sid)

    if len(delta_r) < 5:
        log("  ERROR: too few subjects with matched data")
        return {"error": "too few subjects"}

    delta_r = np.array(delta_r)
    delta_gap = np.array(delta_gap)
    delta_alpha = np.array(delta_alpha)

    r_raw, p_raw = sp_stats.pearsonr(delta_r, delta_gap)
    log(f"\n  Raw delta-delta: r={r_raw:.4f}, p={p_raw:.6f}, N={len(delta_r)}")

    def partial_corr(x, y, z):
        r_xy, _ = sp_stats.pearsonr(x, y)
        r_xz, _ = sp_stats.pearsonr(x, z)
        r_yz, _ = sp_stats.pearsonr(y, z)
        num = r_xy - r_xz * r_yz
        denom = np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
        if denom == 0:
            return float("nan"), float("nan")
        r_partial = num / denom
        n = len(x)
        df = n - 3
        if df < 1:
            return float(r_partial), float("nan")
        t_stat = r_partial * np.sqrt(df) / np.sqrt(1 - r_partial ** 2) if abs(r_partial) < 1 else float("inf")
        p_val = 2 * sp_stats.t.sf(abs(t_stat), df)
        return float(r_partial), float(p_val)

    r_partial, p_partial = partial_corr(delta_r, delta_gap, delta_alpha)

    r_alpha_r, p_alpha_r = sp_stats.pearsonr(delta_r, delta_alpha)
    r_alpha_gap, p_alpha_gap = sp_stats.pearsonr(delta_gap, delta_alpha)

    log(f"  Partial r (controlling alpha): r={r_partial:.4f}, p={p_partial:.6f}")
    log(f"  Alpha-r correlation: r={r_alpha_r:.4f}, p={p_alpha_r:.4f}")
    log(f"  Alpha-gap correlation: r={r_alpha_gap:.4f}, p={p_alpha_gap:.4f}")

    survived = abs(r_partial) > 0.3 and p_partial < 0.05
    log(f"\n  VERDICT: Delta-delta {'SURVIVES' if survived else 'does NOT survive'} "
        f"alpha control (r_partial={r_partial:.4f})")

    result = {
        "n_subjects": len(delta_r),
        "raw_r": float(r_raw),
        "raw_p": float(p_raw),
        "partial_r_controlling_alpha": float(r_partial),
        "partial_p": float(p_partial),
        "alpha_vs_delta_r": {"r": float(r_alpha_r), "p": float(p_alpha_r)},
        "alpha_vs_delta_gap": {"r": float(r_alpha_gap), "p": float(p_alpha_gap)},
        "survived_alpha_control": survived,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.scatter(delta_r, delta_gap, s=50, c="steelblue", edgecolors="white", zorder=5)
    z = np.polyfit(delta_r, delta_gap, 1)
    x_line = np.linspace(min(delta_r), max(delta_r), 100)
    ax.plot(x_line, np.polyval(z, x_line), "--", color="coral", linewidth=2)
    ax.set_xlabel("Delta Spec Sens r (awake - sed)")
    ax.set_ylabel("Delta Gap (awake - sed)")
    ax.set_title(f"Raw: r={r_raw:.3f}, p={p_raw:.4f}")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

    ax = axes[1]
    ax.scatter(delta_alpha, delta_gap, s=50, c="mediumpurple", edgecolors="white", zorder=5)
    z2 = np.polyfit(delta_alpha, delta_gap, 1)
    x2 = np.linspace(min(delta_alpha), max(delta_alpha), 100)
    ax.plot(x2, np.polyval(z2, x2), "--", color="coral", linewidth=2)
    ax.set_xlabel("Delta Alpha Power (awake - sed)")
    ax.set_ylabel("Delta Gap (awake - sed)")
    ax.set_title(f"Alpha-Gap: r={r_alpha_gap:.3f}")

    ax = axes[2]
    bars = [r_raw, r_partial]
    colors = ["steelblue", "seagreen"]
    labels = ["Raw", "Partial\n(alpha controlled)"]
    bar_x = [0, 1]
    ax.bar(bar_x, bars, color=colors, alpha=0.7, width=0.5)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Pearson r")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"Partial Regression\nraw r={r_raw:.3f} -> partial r={r_partial:.3f}")

    plt.tight_layout()
    fig_path = FIG_DIR / "partial_regression_alpha_control.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure: {fig_path}")

    return result


def run_pca_robustness():
    log("\n" + "=" * 70)
    log("TEST 3: PCA COMPONENT ROBUSTNESS")
    log(f"Testing N_COMPONENTS = {PCA_VARIANTS}")
    log("=" * 70)

    subjects = [s for s in SUBJECT_IDS if f"sub-{s}" not in EXCLUDED_SUBJECTS]
    results = {n: [] for n in PCA_VARIANTS}

    for subj in subjects:
        log(f"\n  {subj}...")
        try:
            raw = load_ds005620_subject(subj, DATA_ROOT, task="awake", acq="EC")
        except FileNotFoundError:
            log(f"    SKIP: not found")
            continue

        for n_comp in PCA_VARIANTS:
            t0 = time.time()
            try:
                raw_copy = raw.copy()
                data_pca, sfreq, info = preprocess_scalp_eeg(
                    raw_copy, line_freq=LINE_FREQ, downsample_to=DOWNSAMPLE_TO,
                    n_components=n_comp,
                )

                ep_tc = compute_ep_proximity_timecourse(
                    data_pca, sfreq=sfreq,
                    window_sec=WINDOW_SEC, step_sec=STEP_SEC,
                    max_channels=n_comp, seed=SEED,
                )

                spec_sens = compute_spectral_radius_sensitivity(
                    ep_tc["jac_result"], ep_tc["ep_result"]
                )

                r_val = spec_sens["r"]
                gap_val = float(np.mean(ep_tc["ep_result"].min_eigenvalue_gaps))
                rho_val = float(np.mean(ep_tc["jac_result"].spectral_radius))

                results[n_comp].append({
                    "subject": subj,
                    "n_components": n_comp,
                    "spectral_sensitivity_r": float(r_val),
                    "p_adjusted": float(spec_sens["p_adjusted"]),
                    "n_eff": spec_sens["n_eff"],
                    "mean_gap": gap_val,
                    "mean_rho": rho_val,
                    "cumulative_var": info["cumulative_variance"],
                })

                log(f"    PCA={n_comp}: r={r_val:.4f} gap={gap_val:.6f} "
                    f"var={info['cumulative_variance']:.3f} [{time.time()-t0:.0f}s]")

                del data_pca, ep_tc, raw_copy
                gc.collect()

            except Exception as e:
                log(f"    PCA={n_comp}: ERROR {e}")

        del raw
        gc.collect()

    group_by_n = {}
    for n_comp in PCA_VARIANTS:
        if len(results[n_comp]) >= 3:
            rs = [r["spectral_sensitivity_r"] for r in results[n_comp]]
            t_v, p_v = sp_stats.ttest_1samp(rs, 0.0)
            group_by_n[n_comp] = {
                "n_subjects": len(rs),
                "mean_r": float(np.mean(rs)),
                "std_r": float(np.std(rs)),
                "t_vs_zero": float(t_v),
                "p_vs_zero": float(p_v),
                "min_r": float(np.min(rs)),
                "max_r": float(np.max(rs)),
            }
            log(f"\n  PCA={n_comp}: mean r={np.mean(rs):.4f} +/- {np.std(rs):.4f}, "
                f"t={t_v:.3f}, p={p_v:.6f}")

    if len(group_by_n) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        for n_comp in PCA_VARIANTS:
            if results[n_comp]:
                rs = [r["spectral_sensitivity_r"] for r in results[n_comp]]
                ax.bar(PCA_VARIANTS.index(n_comp), np.mean(rs),
                       yerr=np.std(rs) / np.sqrt(len(rs)),
                       color=["coral", "steelblue", "seagreen"][PCA_VARIANTS.index(n_comp)],
                       alpha=0.7, width=0.6, capsize=5)
        ax.set_xticks(range(len(PCA_VARIANTS)))
        ax.set_xticklabels([str(n) for n in PCA_VARIANTS])
        ax.set_xlabel("N PCA Components")
        ax.set_ylabel("Mean Spectral Sensitivity r")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title("Spectral Sensitivity by PCA Dimensionality")

        ax = axes[1]
        matched_sids = set()
        for n_comp in PCA_VARIANTS:
            sids = set(r["subject"] for r in results[n_comp])
            matched_sids = sids if not matched_sids else matched_sids & sids
        matched_sids = sorted(matched_sids)

        if len(matched_sids) >= 3 and len(PCA_VARIANTS) >= 2:
            for sid in matched_sids:
                vals = []
                for n_comp in PCA_VARIANTS:
                    r_dict = {r["subject"]: r for r in results[n_comp]}
                    vals.append(r_dict[sid]["spectral_sensitivity_r"])
                ax.plot(range(len(PCA_VARIANTS)), vals, "o-", alpha=0.3, color="gray", markersize=3)

            for i, n_comp in enumerate(PCA_VARIANTS):
                r_dict = {r["subject"]: r for r in results[n_comp]}
                mean_v = np.mean([r_dict[s]["spectral_sensitivity_r"] for s in matched_sids])
                ax.plot(i, mean_v, "D",
                        color=["coral", "steelblue", "seagreen"][i],
                        markersize=10, zorder=5)

        ax.set_xticks(range(len(PCA_VARIANTS)))
        ax.set_xticklabels([str(n) for n in PCA_VARIANTS])
        ax.set_xlabel("N PCA Components")
        ax.set_ylabel("Spectral Sensitivity r")
        ax.set_title("Per-Subject Robustness")

        plt.tight_layout()
        fig_path = FIG_DIR / "pca_robustness.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  Figure: {fig_path}")

    return {
        "per_subject": {str(n): results[n] for n in PCA_VARIANTS},
        "group": {str(n): group_by_n.get(n, {}) for n in PCA_VARIANTS},
    }


def main():
    log("=" * 70)
    log("EP ROBUSTNESS & VALIDITY CHECKS")
    log("=" * 70)

    t_start = time.time()

    test2_result = run_partial_regression()

    test1_result = run_surrogate_test()

    test3_result = run_pca_robustness()

    elapsed = time.time() - t_start
    log(f"\n{'='*70}")
    log(f"ALL ROBUSTNESS CHECKS COMPLETE ({elapsed:.0f}s)")
    log(f"{'='*70}")

    output = {
        "analysis": "ep_robustness_checks",
        "surrogate_control": test1_result,
        "partial_regression_alpha": test2_result,
        "pca_robustness": test3_result,
        "elapsed_seconds": elapsed,
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = RESULTS_DIR / "ep_robustness_checks.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    log(f"\n  Results: {json_path}")


if __name__ == "__main__":
    main()
