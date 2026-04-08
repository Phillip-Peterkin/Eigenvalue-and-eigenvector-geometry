import os
"""Exceptional Point Analysis for CMCC Pipeline.

Maps iEEG channels into a high-dimensional state space, estimates
the time-varying Jacobian via windowed VAR(1), and detects eigenvalue
degeneracies (exceptional points) where eigenvalues AND eigenvectors
coalesce — a signature of non-Hermitian criticality distinct from
branching-process criticality.

Runs on all QC-passed subjects using DurR1 high-gamma data.
Correlates EP proximity metrics with existing criticality measures
(sigma, tau, LZc).
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

from cmcc.config import load_config
from cmcc.preprocess.filter import SITE_LINE_FREQ
from cmcc.analysis.dynamical_systems import compute_ep_proximity_timecourse

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "default.yaml"
RESULTS_HG = CMCC_ROOT / "results"
FIG_DIR = RESULTS_HG / "figures" / "exceptional_points"
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
        return None, None

    raw = subject_data.raw["DurR1"]

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
        return None, None
    raw.pick(good)

    raw = remove_line_noise(raw, line_freq=line_freq)
    if subject_data.laplace_map:
        raw = apply_laplace(raw, subject_data.laplace_map)

    passband = tuple(config["preprocessing"]["high_gamma_passband"])
    gamma_raw = extract_high_gamma(raw, passband=passband)

    return gamma_raw.get_data(), gamma_raw.info["sfreq"]


def analyze_single_subject(subject_id, config, hg_results):
    t0 = time.time()
    log(f"\n  {subject_id}...")

    data, sfreq = load_and_preprocess(subject_id, config)
    if data is None:
        log(f"    SKIP: no data")
        return None

    n_ch, n_samples = data.shape
    log(f"    {n_ch} ch, {sfreq} Hz, {n_samples/sfreq:.1f}s")

    result = compute_ep_proximity_timecourse(
        data, sfreq=sfreq,
        window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        max_channels=MAX_CHANNELS, seed=42,
    )

    jac = result["jac_result"]
    ep = result["ep_result"]

    hg = hg_results.get(subject_id, {})

    summary = {
        "subject": subject_id,
        "n_channels_original": int(n_ch),
        "n_channels_used": result["n_channels_used"],
        "sfreq": sfreq,
        "n_samples": int(n_samples),
        "n_windows": len(jac.window_centers),
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "spectral_radius_mean": float(np.mean(jac.spectral_radius)),
        "spectral_radius_std": float(np.std(jac.spectral_radius)),
        "spectral_radius_max": float(np.max(jac.spectral_radius)),
        "condition_number_mean": float(np.mean(jac.condition_numbers)),
        "condition_number_max": float(np.max(jac.condition_numbers)),
        "residual_variance_mean": float(np.mean(jac.residual_variance)),
        "min_eigenvalue_gap_mean": float(np.mean(ep.min_eigenvalue_gaps)),
        "min_eigenvalue_gap_min": float(np.min(ep.min_eigenvalue_gaps)),
        "eigenvector_overlap_mean": float(np.mean(ep.eigenvector_overlaps)),
        "eigenvector_overlap_max": float(np.max(ep.eigenvector_overlaps)),
        "petermann_factor_mean": float(np.nanmean(ep.petermann_factors[np.isfinite(ep.petermann_factors)])) if np.any(np.isfinite(ep.petermann_factors)) else float("nan"),
        "petermann_factor_max": float(np.nanmax(ep.petermann_factors[np.isfinite(ep.petermann_factors)])) if np.any(np.isfinite(ep.petermann_factors)) else float("nan"),
        "ep_score_mean": float(np.mean(ep.ep_scores)),
        "ep_score_max": float(np.max(ep.ep_scores)),
        "n_ep_candidates": len(ep.ep_candidates),
        "hg_sigma": hg.get("branching_sigma"),
        "hg_tau": hg.get("tau"),
        "hg_lzc": hg.get("lzc_normalized"),
        "elapsed_s": time.time() - t0,
    }

    log(f"    spectral_radius={summary['spectral_radius_mean']:.4f} "
        f"min_gap={summary['min_eigenvalue_gap_min']:.6f} "
        f"max_overlap={summary['eigenvector_overlap_max']:.4f} "
        f"EP_candidates={summary['n_ep_candidates']} "
        f"[{summary['elapsed_s']:.0f}s]")

    del data, result
    gc.collect()

    return summary


def plot_summary(subjects_data, output_path):
    valid = [s for s in subjects_data if s is not None
             and s.get("hg_sigma") is not None]
    if len(valid) < 3:
        log("  Too few subjects for summary plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Exceptional Point Analysis — CMCC Cohort", fontsize=14, fontweight="bold")

    sigmas = [s["hg_sigma"] for s in valid]
    taus = [s["hg_tau"] for s in valid]
    lzcs = [s["hg_lzc"] for s in valid]
    ep_scores = [s["ep_score_mean"] for s in valid]
    min_gaps = [s["min_eigenvalue_gap_min"] for s in valid]
    overlaps = [s["eigenvector_overlap_max"] for s in valid]
    spec_rad = [s["spectral_radius_mean"] for s in valid]
    cond_nums = [s["condition_number_mean"] for s in valid]

    ax = axes[0, 0]
    ax.scatter(sigmas, ep_scores, c="steelblue", s=50, edgecolor="k", linewidth=0.5)
    r, p = sp_stats.pearsonr(sigmas, ep_scores)
    ax.set_xlabel("Branching ratio (sigma)")
    ax.set_ylabel("Mean EP score")
    ax.set_title(f"Sigma vs EP score (r={r:.3f}, p={p:.3f})")

    ax = axes[0, 1]
    ax.scatter(taus, ep_scores, c="coral", s=50, edgecolor="k", linewidth=0.5)
    r, p = sp_stats.pearsonr(taus, ep_scores)
    ax.set_xlabel("Power-law exponent (tau)")
    ax.set_ylabel("Mean EP score")
    ax.set_title(f"Tau vs EP score (r={r:.3f}, p={p:.3f})")

    ax = axes[0, 2]
    ax.scatter(lzcs, ep_scores, c="green", s=50, edgecolor="k", linewidth=0.5)
    r, p = sp_stats.pearsonr(lzcs, ep_scores)
    ax.set_xlabel("LZc (normalized)")
    ax.set_ylabel("Mean EP score")
    ax.set_title(f"LZc vs EP score (r={r:.3f}, p={p:.3f})")

    ax = axes[1, 0]
    ax.scatter(sigmas, min_gaps, c="steelblue", s=50, edgecolor="k", linewidth=0.5)
    r, p = sp_stats.pearsonr(sigmas, min_gaps)
    ax.set_xlabel("Branching ratio (sigma)")
    ax.set_ylabel("Min eigenvalue gap")
    ax.set_title(f"Sigma vs min gap (r={r:.3f}, p={p:.3f})")

    ax = axes[1, 1]
    ax.scatter(spec_rad, overlaps, c="purple", s=50, edgecolor="k", linewidth=0.5)
    r_val, p_val = sp_stats.pearsonr(spec_rad, overlaps)
    ax.set_xlabel("Mean spectral radius")
    ax.set_ylabel("Max eigenvector overlap")
    ax.set_title(f"Spectral radius vs overlap (r={r_val:.3f}, p={p_val:.3f})")

    ax = axes[1, 2]
    n_eps = [s["n_ep_candidates"] for s in valid]
    ax.bar(range(len(valid)), n_eps, color="orange", edgecolor="k", linewidth=0.5)
    ax.set_xlabel("Subject index")
    ax.set_ylabel("# EP candidates")
    ax.set_title(f"EP candidates per subject (total={sum(n_eps)})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure: {output_path}")


def main():
    log("=" * 70)
    log("EXCEPTIONAL POINT ANALYSIS — CMCC iEEG Cohort")
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

    log(f"\n{'='*70}")
    log(f"RESULTS: {len(all_results)} subjects analyzed")

    valid = [s for s in all_results if s.get("hg_sigma") is not None]
    if len(valid) >= 3:
        sigmas = [s["hg_sigma"] for s in valid]
        ep_scores = [s["ep_score_mean"] for s in valid]
        min_gaps = [s["min_eigenvalue_gap_min"] for s in valid]
        overlaps = [s["eigenvector_overlap_max"] for s in valid]
        spec_rads = [s["spectral_radius_mean"] for s in valid]
        taus = [s["hg_tau"] for s in valid]

        log(f"\n  Group statistics (n={len(valid)}):")
        log(f"    Spectral radius:  {np.mean(spec_rads):.4f} +/- {np.std(spec_rads):.4f}")
        log(f"    Min eig gap:      {np.mean(min_gaps):.6f} +/- {np.std(min_gaps):.6f}")
        log(f"    Max overlap:      {np.mean(overlaps):.4f} +/- {np.std(overlaps):.4f}")
        log(f"    Mean EP score:    {np.mean(ep_scores):.4f} +/- {np.std(ep_scores):.4f}")

        log(f"\n  Correlations with criticality:")
        for name, vals in [("sigma", sigmas), ("tau", taus)]:
            r_ep, p_ep = sp_stats.pearsonr(vals, ep_scores)
            r_gap, p_gap = sp_stats.pearsonr(vals, min_gaps)
            log(f"    {name} vs EP_score:    r={r_ep:.3f}, p={p_ep:.3f}")
            log(f"    {name} vs min_gap:     r={r_gap:.3f}, p={p_gap:.3f}")

    plot_summary(all_results, FIG_DIR / "ep_summary.png")

    out = {
        "n_subjects": len(all_results),
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "max_channels": MAX_CHANNELS,
        "subjects": all_results,
    }

    if len(valid) >= 3:
        sigmas = [s["hg_sigma"] for s in valid]
        taus = [s["hg_tau"] for s in valid]
        lzcs = [s["hg_lzc"] for s in valid]
        ep_scores = [s["ep_score_mean"] for s in valid]
        min_gaps = [s["min_eigenvalue_gap_min"] for s in valid]

        correlations = {}
        for name, vals in [("sigma", sigmas), ("tau", taus), ("lzc", lzcs)]:
            r_ep, p_ep = sp_stats.pearsonr(vals, ep_scores)
            r_gap, p_gap = sp_stats.pearsonr(vals, min_gaps)
            correlations[f"{name}_vs_ep_score"] = {"r": r_ep, "p": p_ep, "n": len(valid)}
            correlations[f"{name}_vs_min_gap"] = {"r": r_gap, "p": p_gap, "n": len(valid)}
        out["correlations"] = correlations

    out_path = RESULTS_HG / "analysis" / "exceptional_points.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=default_ser)
    log(f"\n  Results: {out_path}")
    log(f"\n{'='*70}")
    log("DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()
