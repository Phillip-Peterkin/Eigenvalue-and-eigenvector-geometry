import os
"""Advanced EP Analyses for CMCC Pipeline.

Four mechanistic tests of non-Hermitian criticality:
1. State-Switch Contrast — EP metrics in task-relevant vs irrelevant windows
2. Spectral Radius Sensitivity — rho vs eigenvalue gap correlation
3. SVD Dimension — effective rank collapse near EPs
4. Petermann Noise — K factor predicting high-gamma power bursts

Runs on all QC-passed subjects using DurR1 high-gamma data.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

from cmcc.config import load_config
from cmcc.preprocess.filter import SITE_LINE_FREQ
from cmcc.analysis.dynamical_systems import compute_ep_proximity_timecourse
from cmcc.analysis.contrasts import fdr_correction
from cmcc.analysis.ep_advanced import (
    compute_state_contrast,
    compute_spectral_radius_sensitivity,
    compute_svd_dimension,
    compute_petermann_noise,
    compute_petermann_noise_surrogate,
)

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "default.yaml"
RESULTS_HG = CMCC_ROOT / "results"
FIG_DIR = RESULTS_HG / "figures" / "ep_advanced"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path(os.environ.get("IEEG_DATA_ROOT", "./data/Cogitate_IEEG_EXP1"))

MAX_CHANNELS = 30
WINDOW_SEC = 0.5
STEP_SEC = 0.1


def log(msg):
    print(msg, flush=True)


def load_group_results():
    with open(RESULTS_HG / "group_all_subjects.json") as f:
        data = json.load(f)
    return {s["subject"]: s for s in data if s.get("status") == "OK" and s.get("qc_pass", False)}


def load_and_preprocess(subject_id, config):
    from cmcc.io.loader import load_subject
    from cmcc.preprocess.qc import detect_bad_channels, mark_bad_channels
    from cmcc.preprocess.filter import remove_line_noise, extract_high_gamma
    from cmcc.preprocess.reference import apply_laplace

    site = subject_id[:2].upper()
    line_freq = SITE_LINE_FREQ.get(site, 60.0)

    subject_data = load_subject(subject_id, config["data"]["root"], runs=["DurR1"], preload_edf=True)
    if "DurR1" not in subject_data.raw:
        return None, None, None

    raw = subject_data.raw["DurR1"]
    n_samples_raw = raw.n_times

    non_ecog = [ch for ch in raw.ch_names
                if ch.startswith("DC") or ch.startswith("C12")
                or ch.startswith("EKG") or ch.startswith("EMG")
                or ch == "C128"]
    if non_ecog:
        raw.drop_channels(non_ecog)

    bad = detect_bad_channels(raw)
    mark_bad_channels(raw, bad)
    good = [ch for ch in raw.ch_names if ch not in bad]
    if len(good) < 5:
        return None, None, None
    raw.pick(good)

    raw = remove_line_noise(raw, line_freq=line_freq)
    if subject_data.laplace_map:
        raw = apply_laplace(raw, subject_data.laplace_map)

    passband = tuple(config["preprocessing"]["high_gamma_passband"])
    gamma_raw = extract_high_gamma(raw, passband=passband)

    return gamma_raw.get_data(), gamma_raw.info["sfreq"], n_samples_raw


def load_events_for_run(data_root, subject_id, sfreq, n_samples_raw):
    meta_dir = Path(data_root) / f"{subject_id}_ECOG_1" / "METADATA" / "electrode_coordinates"
    tsv_path = meta_dir / f"sub-{subject_id}_ses-1_task-Dur_events.tsv"
    if not tsv_path.exists():
        return None, None

    events_df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8-sig")
    stim = events_df[events_df["trial_type"].str.contains("stimulus onset", na=False)].copy()
    stim = stim[stim["sample"] < n_samples_raw].copy()

    parts = stim["trial_type"].str.split("/", expand=True)
    stim["duration_ms"] = parts[6].str.replace("ms", "").astype(int)
    stim["relevance"] = parts[7]
    stim["task_relevant"] = stim["relevance"].str.contains("Relevant").astype(int)

    rel_intervals = []
    irr_intervals = []
    for _, row in stim.iterrows():
        onset = int(row["sample"])
        dur_samples = int(row["duration_ms"] * sfreq / 1000.0)
        interval = (onset, onset + dur_samples)
        if row["task_relevant"] == 1:
            rel_intervals.append(interval)
        else:
            irr_intervals.append(interval)

    return rel_intervals, irr_intervals


def compute_hg_power_per_window(data, window_centers, window_samples):
    n_ch, n_total = data.shape
    half_w = window_samples // 2
    power = np.zeros(len(window_centers))
    for i, c in enumerate(window_centers):
        c = int(c)
        start = max(0, c - half_w)
        end = min(n_total, c + half_w)
        if end > start:
            power[i] = np.mean(data[:, start:end] ** 2)
    return power


def analyze_single_subject(subject_id, config, hg_results):
    t0 = time.time()
    log(f"\n  {subject_id}...")

    data, sfreq, n_samples_raw = load_and_preprocess(subject_id, config)
    if data is None:
        log(f"    SKIP: no data")
        return None

    n_ch, n_samples = data.shape
    log(f"    {n_ch} ch, {sfreq} Hz, {n_samples/sfreq:.1f}s")

    ep_tc = compute_ep_proximity_timecourse(
        data, sfreq=sfreq,
        window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        max_channels=MAX_CHANNELS, seed=42,
    )

    jac = ep_tc["jac_result"]
    ep = ep_tc["ep_result"]
    n_windows = len(jac.window_centers)
    window_samples = max(int(WINDOW_SEC * sfreq), ep_tc["n_channels_used"] + 10)

    hg = hg_results.get(subject_id, {})

    summary = {
        "subject": subject_id,
        "n_channels_used": ep_tc["n_channels_used"],
        "sfreq": sfreq,
        "n_windows": n_windows,
        "hg_sigma": hg.get("branching_sigma"),
        "hg_tau": hg.get("tau"),
        "hg_lzc": hg.get("lzc_normalized"),
    }

    rel_intervals, irr_intervals = load_events_for_run(
        config["data"]["root"], subject_id, sfreq, n_samples_raw or n_samples,
    )
    if rel_intervals is not None and len(rel_intervals) > 0:
        sc = compute_state_contrast(
            jac, ep, rel_intervals, irr_intervals, n_perm=500, seed=42,
        )
        summary["state_contrast"] = {
            "n_relevant_windows": sc.n_relevant_windows,
            "n_irrelevant_windows": sc.n_irrelevant_windows,
            "n_eff_relevant": sc.n_eff_relevant,
            "n_eff_irrelevant": sc.n_eff_irrelevant,
            "fdr_significant": sc.fdr_significant,
            "circular_shift_p": sc.circular_shift_p,
            "gap": {
                "mean_rel": sc.gap_contrast.mean_a,
                "mean_irr": sc.gap_contrast.mean_b,
                "g": sc.gap_contrast.effect_size,
                "p_nominal": sc.gap_contrast.p_value,
                "p_circular_shift": sc.circular_shift_p.get("gap", float("nan")),
                "fdr_significant": sc.fdr_significant.get("gap", False),
            },
            "condition_number": {
                "mean_rel": sc.condition_number_contrast.mean_a,
                "mean_irr": sc.condition_number_contrast.mean_b,
                "g": sc.condition_number_contrast.effect_size,
                "p_nominal": sc.condition_number_contrast.p_value,
                "p_circular_shift": sc.circular_shift_p.get("condition_number", float("nan")),
                "fdr_significant": sc.fdr_significant.get("condition_number", False),
            },
            "ep_score": {
                "mean_rel": sc.ep_score_contrast.mean_a,
                "mean_irr": sc.ep_score_contrast.mean_b,
                "g": sc.ep_score_contrast.effect_size,
                "p_nominal": sc.ep_score_contrast.p_value,
                "p_circular_shift": sc.circular_shift_p.get("ep_score", float("nan")),
                "fdr_significant": sc.fdr_significant.get("ep_score", False),
            },
            "spectral_radius": {
                "mean_rel": sc.spectral_radius_contrast.mean_a,
                "mean_irr": sc.spectral_radius_contrast.mean_b,
                "g": sc.spectral_radius_contrast.effect_size,
                "p_nominal": sc.spectral_radius_contrast.p_value,
                "p_circular_shift": sc.circular_shift_p.get("spectral_radius", float("nan")),
                "fdr_significant": sc.fdr_significant.get("spectral_radius", False),
            },
        }
        log(f"    state_contrast: rel={sc.n_relevant_windows} irr={sc.n_irrelevant_windows} "
            f"n_eff=({sc.n_eff_relevant},{sc.n_eff_irrelevant}) "
            f"gap_g={sc.gap_contrast.effect_size:.3f} ep_g={sc.ep_score_contrast.effect_size:.3f}")
    else:
        summary["state_contrast"] = None
        log(f"    state_contrast: SKIP (no events)")

    srs = compute_spectral_radius_sensitivity(jac, ep)
    summary["spectral_sensitivity"] = srs
    log(f"    spectral_sensitivity: r={srs['r']:.3f} p_adj={srs['p_adjusted']:.4f} n_eff={srs['n_eff']}")

    svd = compute_svd_dimension(jac, ep, run_null=True)
    summary["svd_dimension"] = {
        "mean_pr": svd.mean_pr,
        "mean_erank": svd.mean_erank,
        "pr_vs_ep": svd.pr_vs_ep_score,
        "erank_vs_ep": svd.erank_vs_ep_score,
        "null_r_mean": svd.null_r_mean,
        "null_r_std": svd.null_r_std,
        "null_p": svd.null_p,
    }
    log(f"    svd_dimension: mean_erank={svd.mean_erank:.2f} "
        f"erank_vs_ep r={svd.erank_vs_ep_score['r']:.3f} "
        f"p_adj={svd.erank_vs_ep_score.get('p_adjusted', float('nan')):.4f} "
        f"null_p={svd.null_p:.3f}")

    hg_power = compute_hg_power_per_window(data, jac.window_centers, window_samples)
    pet = compute_petermann_noise(
        ep, hg_power, step_sec=STEP_SEC, window_sec=WINDOW_SEC,
    )
    summary["petermann_noise"] = {
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
    }
    log(f"    petermann_noise: r={pet.correlation_r:.3f} p_adj={pet.p_adjusted:.4f} "
        f"n_eff={pet.n_eff} peak_lag={pet.peak_lag} "
        f"granger_p={pet.granger_p:.4f}" if np.isfinite(pet.granger_p) else
        f"    petermann_noise: r={pet.correlation_r:.3f} peak_lag={pet.peak_lag}")

    summary["elapsed_s"] = time.time() - t0

    del data, ep_tc
    gc.collect()

    return summary


def compute_group_statistics(all_results):
    valid = [s for s in all_results if s is not None]
    if len(valid) < 3:
        return {}

    group = {}
    all_group_p = []
    all_group_p_labels = []

    sc_valid = [s for s in valid if s.get("state_contrast") is not None]
    if len(sc_valid) >= 3:
        sc_stats = {}
        for metric in ["gap", "condition_number", "ep_score", "spectral_radius"]:
            gs = [s["state_contrast"][metric]["g"] for s in sc_valid
                  if s["state_contrast"][metric]["g"] is not None
                  and np.isfinite(s["state_contrast"][metric]["g"])]
            n_effs = [s["state_contrast"].get("n_eff_relevant", 0) for s in sc_valid]
            if len(gs) >= 3:
                t_val, p_val = sp_stats.ttest_1samp(gs, 0.0)
                sc_stats[metric] = {
                    "mean_g": float(np.mean(gs)),
                    "std_g": float(np.std(gs)),
                    "t": float(t_val),
                    "p": float(p_val),
                    "n": len(gs),
                    "mean_n_eff_relevant": float(np.mean(n_effs)),
                    "min_n_eff_relevant": int(np.min(n_effs)),
                }
                all_group_p.append(p_val)
                all_group_p_labels.append(f"state_contrast_{metric}")
        group["state_contrast"] = sc_stats

    rs = [s["spectral_sensitivity"]["r"] for s in valid
          if np.isfinite(s["spectral_sensitivity"]["r"])]
    n_effs_sr = [s["spectral_sensitivity"]["n_eff"] for s in valid
                 if np.isfinite(s["spectral_sensitivity"]["r"])]
    if len(rs) >= 3:
        t_val, p_val = sp_stats.ttest_1samp(rs, 0.0)
        group["spectral_sensitivity"] = {
            "mean_r": float(np.mean(rs)),
            "std_r": float(np.std(rs)),
            "t": float(t_val),
            "p": float(p_val),
            "n": len(rs),
            "mean_n_eff": float(np.mean(n_effs_sr)),
            "min_n_eff": int(np.min(n_effs_sr)),
        }
        all_group_p.append(p_val)
        all_group_p_labels.append("spectral_sensitivity")

    erank_rs = [s["svd_dimension"]["erank_vs_ep"]["r"] for s in valid
                if np.isfinite(s["svd_dimension"]["erank_vs_ep"]["r"])]
    erank_n_effs = [s["svd_dimension"]["erank_vs_ep"].get("n_eff", 0) for s in valid
                    if np.isfinite(s["svd_dimension"]["erank_vs_ep"]["r"])]
    null_ps = [s["svd_dimension"].get("null_p", float("nan")) for s in valid
               if np.isfinite(s["svd_dimension"].get("null_p", float("nan")))]
    if len(erank_rs) >= 3:
        t_val, p_val = sp_stats.ttest_1samp(erank_rs, 0.0)
        group["svd_dimension"] = {
            "mean_erank_vs_ep_r": float(np.mean(erank_rs)),
            "t": float(t_val),
            "p": float(p_val),
            "n": len(erank_rs),
            "mean_n_eff": float(np.mean(erank_n_effs)) if erank_n_effs else float("nan"),
            "min_n_eff": int(np.min(erank_n_effs)) if erank_n_effs else 0,
            "mean_null_p": float(np.mean(null_ps)) if null_ps else float("nan"),
        }
        all_group_p.append(p_val)
        all_group_p_labels.append("svd_dimension")

    pet_valid = [s for s in valid if s.get("petermann_noise") is not None
                 and np.isfinite(s["petermann_noise"]["peak_lag"])]
    if len(pet_valid) >= 3:
        lags = [s["petermann_noise"]["peak_lag"] for s in pet_valid]
        rs_pet = [s["petermann_noise"]["r"] for s in pet_valid
                  if np.isfinite(s["petermann_noise"]["r"])]
        pet_n_effs = [s["petermann_noise"].get("n_eff", 0) for s in pet_valid]
        t_lag, p_lag = sp_stats.ttest_1samp(lags, 0.0)
        group["petermann_noise"] = {
            "mean_peak_lag": float(np.mean(lags)),
            "std_peak_lag": float(np.std(lags)),
            "t_lag": float(t_lag),
            "p_lag": float(p_lag),
            "mean_r": float(np.mean(rs_pet)) if rs_pet else float("nan"),
            "n": len(lags),
            "mean_n_eff": float(np.mean(pet_n_effs)),
            "min_n_eff": int(np.min(pet_n_effs)) if pet_n_effs else 0,
        }
        all_group_p.append(p_lag)
        all_group_p_labels.append("petermann_noise_lag")

    if len(all_group_p) >= 2:
        fdr_sig = fdr_correction(all_group_p, alpha=0.05)
        group["fdr_correction"] = {
            label: {"p": float(p), "fdr_significant": bool(sig)}
            for label, p, sig in zip(all_group_p_labels, all_group_p, fdr_sig)
        }

    return group


def plot_ep_advanced_summary(all_results, group_stats, output_dir):
    valid = [s for s in all_results if s is not None]
    if len(valid) < 3:
        log("  Too few subjects for summary plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Advanced EP Analyses — CMCC Cohort", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    sc_valid = [s for s in valid if s.get("state_contrast") is not None]
    if sc_valid:
        metrics = ["gap", "ep_score", "spectral_radius", "condition_number"]
        labels = ["Eig Gap", "EP Score", "Spec Radius", "Cond Num"]
        gs_per_metric = []
        for m in metrics:
            gs = [s["state_contrast"][m]["g"] for s in sc_valid
                  if np.isfinite(s["state_contrast"][m]["g"])]
            gs_per_metric.append(gs)

        bp = ax.boxplot(gs_per_metric, labels=labels, patch_artist=True)
        colors = ["steelblue", "coral", "green", "purple"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
        ax.set_ylabel("Hedges' g (Relevant - Irrelevant)")
        ax.set_title("State-Switch Contrast (circular-shift + FDR)")
    else:
        ax.text(0.5, 0.5, "No state contrast data", ha="center", va="center", transform=ax.transAxes)

    ax = axes[0, 1]
    rs_sr = [s["spectral_sensitivity"]["r"] for s in valid
             if np.isfinite(s["spectral_sensitivity"]["r"])]
    if rs_sr:
        ax.hist(rs_sr, bins=15, color="steelblue", edgecolor="k", alpha=0.7)
        ax.axvline(np.mean(rs_sr), color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Pearson r (spectral radius vs min gap)")
        ax.set_ylabel("Count")
        gs = group_stats.get("spectral_sensitivity", {})
        t_val = gs.get("t", float("nan"))
        p_val = gs.get("p", float("nan"))
        ax.set_title(f"Spectral Radius Sensitivity (t={t_val:.2f}, p={p_val:.3f})")

    ax = axes[1, 0]
    erank_rs = [s["svd_dimension"]["erank_vs_ep"]["r"] for s in valid
                if np.isfinite(s["svd_dimension"]["erank_vs_ep"]["r"])]
    if erank_rs:
        ax.hist(erank_rs, bins=15, color="green", edgecolor="k", alpha=0.7)
        ax.axvline(np.mean(erank_rs), color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Pearson r (effective rank vs EP score)")
        ax.set_ylabel("Count")
        gs = group_stats.get("svd_dimension", {})
        t_val = gs.get("t", float("nan"))
        p_val = gs.get("p", float("nan"))
        ax.set_title(f"SVD Dimension Collapse (t={t_val:.2f}, p={p_val:.3f})")

    ax = axes[1, 1]
    pet_valid = [s for s in valid if s.get("petermann_noise") is not None
                 and np.isfinite(s["petermann_noise"]["peak_lag"])]
    if pet_valid:
        lags = [s["petermann_noise"]["peak_lag"] for s in pet_valid]
        ax.hist(lags, bins=15, color="coral", edgecolor="k", alpha=0.7)
        ax.axvline(0, color="k", linestyle="--", linewidth=0.5)
        ax.axvline(np.mean(lags), color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Peak lag (windows, negative = K leads HG)")
        ax.set_ylabel("Count")
        gs = group_stats.get("petermann_noise", {})
        t_val = gs.get("t_lag", float("nan"))
        p_val = gs.get("p_lag", float("nan"))
        ax.set_title(f"Petermann Noise Lead/Lag (t={t_val:.2f}, p={p_val:.3f})")

    plt.tight_layout()
    plt.savefig(output_dir / "ep_advanced_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure: {output_dir / 'ep_advanced_summary.png'}")


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
    log("ADVANCED EP ANALYSES — CMCC iEEG Cohort")
    log("=" * 70)

    config = load_config(str(CONFIG_PATH))
    hg_results = load_group_results()
    subjects = sorted(hg_results.keys())
    log(f"\nSubjects: {len(subjects)} QC-passed")

    all_results = []
    for subj in subjects:
        try:
            result = analyze_single_subject(subj, config, hg_results)
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
            log(f"    {key}: {val}")

    plot_ep_advanced_summary(all_results, group_stats, FIG_DIR)

    out = {
        "n_subjects": len(all_results),
        "parameters": {
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "max_channels": MAX_CHANNELS,
        },
        "subjects": all_results,
        "group_statistics": group_stats,
    }

    out_path = RESULTS_HG / "analysis" / "ep_advanced.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=default_ser)
    log(f"\n  Results: {out_path}")
    log(f"\n{'='*70}")
    log("DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()
