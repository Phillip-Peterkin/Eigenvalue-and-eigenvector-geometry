"""Chiral phase flux analysis for CMCC iEEG cohort.

Measures the directional (chiral) phase rotation of eigenvalues near
exceptional points. Tests the hypothesis that conscious processing
produces consistent (chiral) eigenvalue phase rotation, while
noise/unconsciousness produces random phase jitter.

Uses eigenvalue phase tracking from the time-varying Jacobian estimated
in the EP analysis pipeline.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*toeplitz.*", category=RuntimeWarning)

from cmcc.config import load_config
from cmcc.preprocess.filter import SITE_LINE_FREQ
from cmcc.analysis.dynamical_systems import (
    compute_ep_proximity_timecourse,
    detect_exceptional_points,
    measure_chirality,
    decompose_jacobian_hermiticity,
)

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "default.yaml"
RESULTS_HG = CMCC_ROOT / "results"
FIG_DIR = RESULTS_HG / "figures" / "chirality"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path(os.environ.get("COGITATE_IEEG_ROOT", r"c:\openneuro\Cogitate_IEEG_EXP1"))

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
    chiral = measure_chirality(jac, ep)
    nh = decompose_jacobian_hermiticity(jac)

    hg = hg_results.get(subject_id, {})

    n_w = len(jac.window_centers)
    pair_tuples = [tuple(ep.gap_pair_indices[w]) for w in range(n_w)]
    pair_counts = {}
    for pt in pair_tuples:
        pair_counts[pt] = pair_counts.get(pt, 0) + 1
    stable_pair = max(pair_counts, key=pair_counts.get)
    pair_i, pair_j = stable_pair
    pair_stability = pair_counts[stable_pair] / n_w

    delta_lam = np.array([
        jac.eigenvalues[w, pair_i] - jac.eigenvalues[w, pair_j]
        for w in range(n_w)
    ])
    frac_complex = float(np.mean(np.abs(delta_lam.imag) > 1e-10))
    frac_sign_changes = float(np.mean(np.diff(np.sign(delta_lam.real)) != 0))
    mean_splitting_magnitude = float(np.mean(np.abs(delta_lam)))

    pv = chiral.phase_velocities
    frac_pi_jumps = float(np.mean(np.abs(np.abs(pv) - np.pi) < 0.1))

    summary = {
        "subject": subject_id,
        "n_channels_original": int(n_ch),
        "n_channels_used": result["n_channels_used"],
        "sfreq": sfreq,
        "n_samples": int(n_samples),
        "n_windows": n_w,
        "spectral_radius_mean": float(np.mean(jac.spectral_radius)),
        "ep_score_mean": float(np.mean(ep.ep_scores)),
        "eigenvector_overlap_max": float(np.max(ep.eigenvector_overlaps)),
        "chirality_index": float(chiral.chirality_index),
        "mean_phase_velocity": float(chiral.mean_phase_velocity),
        "phase_velocity_std": float(chiral.phase_velocity_std),
        "winding_number": float(chiral.winding_number),
        "berry_phase": float(chiral.berry_phase),
        "circular_variance": float(chiral.circular_variance),
        "circular_mean_direction": float(chiral.circular_mean_direction),
        "mean_tracking_quality": float(np.mean(chiral.tracking_quality)),
        "stable_pair": list(stable_pair),
        "pair_stability": pair_stability,
        "frac_complex_splitting": frac_complex,
        "frac_sign_changes": frac_sign_changes,
        "frac_pi_jumps": frac_pi_jumps,
        "mean_splitting_magnitude": mean_splitting_magnitude,
        "asymmetry_ratio_mean": float(nh.mean_asymmetry_ratio),
        "asymmetry_ratio_std": float(nh.std_asymmetry_ratio),
        "asymmetry_ratio_kurtosis": float(nh.kurtosis_asymmetry_ratio),
        "asymmetry_ratio_max": float(nh.max_asymmetry_ratio),
        "asymmetry_ratio_p95": float(nh.p95_asymmetry_ratio),
        "asymmetry_ratio_p99": float(nh.p99_asymmetry_ratio),
        "asymmetry_dynamic_range": float(nh.dynamic_range),
        "symmetric_power_mean": float(np.mean(nh.symmetric_power)),
        "antisymmetric_power_mean": float(np.mean(nh.antisymmetric_power)),
        "max_rotation_freq_mean": float(np.mean(nh.max_rotation_frequency)),
        "max_rotation_freq_std": float(np.std(nh.max_rotation_frequency)),
        "hg_sigma": hg.get("branching_sigma"),
        "hg_tau": hg.get("tau"),
        "hg_lzc": hg.get("lzc_normalized"),
        "elapsed_s": time.time() - t0,
    }

    log(f"    chirality_idx={summary['chirality_index']:.4f} "
        f"winding={summary['winding_number']:.1f} "
        f"asym_ratio={summary['asymmetry_ratio_mean']:.4f} "
        f"max_rot={summary['max_rotation_freq_mean']:.4f} "
        f"pair={stable_pair}({pair_stability:.2f}) "
        f"[{summary['elapsed_s']:.0f}s]")

    del data, result, jac, ep, chiral, nh
    gc.collect()

    return summary


def plot_chirality_summary(subjects_data, output_path):
    valid = [s for s in subjects_data if s is not None
             and s.get("hg_sigma") is not None]
    if len(valid) < 3:
        log("  Too few subjects for summary plot")
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Non-Hermitian Decomposition & Chirality - CMCC Cohort", fontsize=14, fontweight="bold")

    sigmas = [s["hg_sigma"] for s in valid]
    taus = [s["hg_tau"] for s in valid]
    lzcs = [s["hg_lzc"] for s in valid]
    chi_idx = [s["chirality_index"] for s in valid]
    windings = [s["winding_number"] for s in valid]
    circ_var = [s["circular_variance"] for s in valid]
    ep_scores = [s["ep_score_mean"] for s in valid]
    asym_ratios = [s["asymmetry_ratio_mean"] for s in valid]
    max_rot_freqs = [s["max_rotation_freq_mean"] for s in valid]

    ax = axes[0, 0]
    ax.scatter(sigmas, asym_ratios, c="darkred", s=50, edgecolor="k", linewidth=0.5)
    r, p = sp_stats.pearsonr(sigmas, asym_ratios)
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    ax.set_xlabel("Branching ratio (sigma)")
    ax.set_ylabel("Asymmetry ratio (||A||/||S||)")
    ax.set_title(f"Sigma vs asymmetry ratio\nr={r:.3f}, p={p:.4f}{sig}")

    ax = axes[0, 1]
    ax.scatter(taus, asym_ratios, c="darkblue", s=50, edgecolor="k", linewidth=0.5)
    r, p = sp_stats.pearsonr(taus, asym_ratios)
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    ax.set_xlabel("Power-law exponent (tau)")
    ax.set_ylabel("Asymmetry ratio (||A||/||S||)")
    ax.set_title(f"Tau vs asymmetry ratio\nr={r:.3f}, p={p:.4f}{sig}")

    ax = axes[0, 2]
    ax.scatter(lzcs, asym_ratios, c="darkgreen", s=50, edgecolor="k", linewidth=0.5)
    r, p = sp_stats.pearsonr(lzcs, asym_ratios)
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    ax.set_xlabel("LZc (normalized)")
    ax.set_ylabel("Asymmetry ratio (||A||/||S||)")
    ax.set_title(f"LZc vs asymmetry ratio\nr={r:.3f}, p={p:.4f}{sig}")

    ax = axes[0, 3]
    ax.scatter(ep_scores, asym_ratios, c="purple", s=50, edgecolor="k", linewidth=0.5)
    r, p = sp_stats.pearsonr(ep_scores, asym_ratios)
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    ax.set_xlabel("Mean EP score")
    ax.set_ylabel("Asymmetry ratio (||A||/||S||)")
    ax.set_title(f"EP score vs asymmetry ratio\nr={r:.3f}, p={p:.4f}{sig}")

    ax = axes[1, 0]
    ax.scatter(sigmas, max_rot_freqs, c="darkred", s=50, edgecolor="k", linewidth=0.5, marker="^")
    r, p = sp_stats.pearsonr(sigmas, max_rot_freqs)
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    ax.set_xlabel("Branching ratio (sigma)")
    ax.set_ylabel("Max rotation frequency")
    ax.set_title(f"Sigma vs max rot freq\nr={r:.3f}, p={p:.4f}{sig}")

    ax = axes[1, 1]
    ax.scatter(asym_ratios, chi_idx, c="teal", s=50, edgecolor="k", linewidth=0.5)
    r, p = sp_stats.pearsonr(asym_ratios, chi_idx)
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    ax.set_xlabel("Asymmetry ratio (||A||/||S||)")
    ax.set_ylabel("Chirality index")
    ax.set_title(f"Asymmetry vs chirality\nr={r:.3f}, p={p:.4f}{sig}")

    ax = axes[1, 2]
    ax.bar(range(len(valid)), asym_ratios, color="darkred", edgecolor="k", linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Subject index")
    ax.set_ylabel("Asymmetry ratio")
    ax.set_title("Asymmetry ratio per subject")

    ax = axes[1, 3]
    ax.scatter(asym_ratios, windings, c="navy", s=50, edgecolor="k", linewidth=0.5)
    r, p = sp_stats.pearsonr(asym_ratios, windings)
    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
    ax.set_xlabel("Asymmetry ratio (||A||/||S||)")
    ax.set_ylabel("Winding number")
    ax.set_title(f"Asymmetry vs winding\nr={r:.3f}, p={p:.4f}{sig}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure: {output_path}")


def channel_scaling_sweep(subject_id, config, channel_counts):
    from cmcc.analysis.dynamical_systems import estimate_jacobian as est_jac
    t0 = time.time()
    log(f"\n  Channel scaling: {subject_id}")

    data, sfreq = load_and_preprocess(subject_id, config)
    if data is None:
        return None

    n_ch, n_samples = data.shape
    log(f"    {n_ch} channels available")

    rng = np.random.default_rng(42)
    ch_mean = data.mean(axis=1, keepdims=True)
    ch_std = data.std(axis=1, keepdims=True)
    ch_std[ch_std == 0] = 1.0
    data_z = (data - ch_mean) / ch_std

    window_samples = max(int(WINDOW_SEC * sfreq), 40)
    step_samples = max(1, int(STEP_SEC * sfreq))

    results_per_nch = []
    for n_use in channel_counts:
        if n_use > n_ch:
            continue
        window_adj = max(window_samples, n_use + 10)
        if n_samples < window_adj + 1:
            continue

        if n_use < n_ch:
            ch_idx = np.sort(rng.choice(n_ch, n_use, replace=False))
            d = data_z[ch_idx]
        else:
            d = data_z

        try:
            jac = est_jac(d, window_size=window_adj, step_size=step_samples)
            nh = decompose_jacobian_hermiticity(jac)
            entry = {
                "n_channels": int(n_use),
                "asymmetry_ratio_mean": float(nh.mean_asymmetry_ratio),
                "asymmetry_ratio_std": float(nh.std_asymmetry_ratio),
                "asymmetry_kurtosis": float(nh.kurtosis_asymmetry_ratio),
                "asymmetry_max": float(nh.max_asymmetry_ratio),
                "asymmetry_p99": float(nh.p99_asymmetry_ratio),
                "asymmetry_dynamic_range": float(nh.dynamic_range),
                "max_rotation_freq_mean": float(np.mean(nh.max_rotation_frequency)),
                "spectral_radius_mean": float(np.mean(jac.spectral_radius)),
                "n_windows": len(jac.window_centers),
            }
            results_per_nch.append(entry)
            log(f"    n_ch={n_use:3d}: asym_mean={entry['asymmetry_ratio_mean']:.4f} "
                f"kurtosis={entry['asymmetry_kurtosis']:.2f} "
                f"max={entry['asymmetry_max']:.4f} "
                f"dyn_range={entry['asymmetry_dynamic_range']:.2f}")
        except Exception as e:
            log(f"    n_ch={n_use}: FAILED ({e})")

    del data, data_z
    gc.collect()

    return {
        "subject": subject_id,
        "n_channels_total": int(n_ch),
        "sweep": results_per_nch,
        "elapsed_s": time.time() - t0,
    }


def main():
    log("=" * 70)
    log("CHIRAL PHASE FLUX ANALYSIS - CMCC iEEG Cohort")
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

    valid = [s for s in all_results if s.get("hg_sigma") is not None]
    if len(valid) >= 3:
        sigmas = [s["hg_sigma"] for s in valid]
        taus = [s["hg_tau"] for s in valid]
        lzcs = [s["hg_lzc"] for s in valid]
        chi_idx = [s["chirality_index"] for s in valid]
        windings = [s["winding_number"] for s in valid]
        berry = [s["berry_phase"] for s in valid]
        circ_var = [s["circular_variance"] for s in valid]
        asym_ratios = [s["asymmetry_ratio_mean"] for s in valid]
        asym_kurtosis = [s["asymmetry_ratio_kurtosis"] for s in valid]
        asym_max = [s["asymmetry_ratio_max"] for s in valid]
        asym_p95 = [s["asymmetry_ratio_p95"] for s in valid]
        asym_p99 = [s["asymmetry_ratio_p99"] for s in valid]
        asym_dynrange = [s["asymmetry_dynamic_range"] for s in valid]
        max_rot_freqs = [s["max_rotation_freq_mean"] for s in valid]

        log(f"\n  Group statistics (n={len(valid)}):")
        log(f"    Chirality index:     {np.mean(chi_idx):.4f} +/- {np.std(chi_idx):.4f}")
        log(f"    Winding number:      {np.mean(windings):.4f} +/- {np.std(windings):.4f}")
        log(f"    Berry phase:         {np.mean(berry):.4f} +/- {np.std(berry):.4f}")
        log(f"    Circular variance:   {np.mean(circ_var):.4f} +/- {np.std(circ_var):.4f}")
        log(f"    Asymmetry ratio:     {np.mean(asym_ratios):.4f} +/- {np.std(asym_ratios):.4f}")
        log(f"    Asym kurtosis:       {np.mean(asym_kurtosis):.4f} +/- {np.std(asym_kurtosis):.4f}")
        log(f"    Asym max:            {np.mean(asym_max):.4f} +/- {np.std(asym_max):.4f}")
        log(f"    Asym p99:            {np.mean(asym_p99):.4f} +/- {np.std(asym_p99):.4f}")
        log(f"    Asym dynamic range:  {np.mean(asym_dynrange):.4f} +/- {np.std(asym_dynrange):.4f}")
        log(f"    Max rotation freq:   {np.mean(max_rot_freqs):.4f} +/- {np.std(max_rot_freqs):.4f}")

        n_chiral = sum(1 for c in chi_idx if c > 1.0)
        log(f"    Subjects with chirality_index > 1: {n_chiral}/{len(valid)}")

        log(f"\n  Correlations with criticality:")
        corr_table = {}
        for name, vals in [("sigma", sigmas), ("tau", taus), ("lzc", lzcs)]:
            for metric_name, metric_vals in [("chirality_index", chi_idx),
                                              ("winding_number", windings),
                                              ("berry_phase", berry),
                                              ("circular_variance", circ_var),
                                              ("asymmetry_ratio", asym_ratios),
                                              ("asymmetry_kurtosis", asym_kurtosis),
                                              ("asymmetry_max", asym_max),
                                              ("asymmetry_p99", asym_p99),
                                              ("asymmetry_dynamic_range", asym_dynrange),
                                              ("max_rotation_freq", max_rot_freqs)]:
                r, p = sp_stats.pearsonr(vals, metric_vals)
                key = f"{name}_vs_{metric_name}"
                corr_table[key] = {"r": float(r), "p": float(p), "n": len(valid)}
                if metric_name in ("chirality_index", "asymmetry_ratio",
                                   "asymmetry_kurtosis", "asymmetry_max",
                                   "asymmetry_p99", "asymmetry_dynamic_range"):
                    sig = " ***" if p < 0.001 else " **" if p < 0.01 else " *" if p < 0.05 else ""
                    log(f"    {name} vs {metric_name}:  r={r:.3f}, p={p:.3f}{sig}")

    plot_chirality_summary(all_results, FIG_DIR / "chirality_summary.png")

    out = {
        "n_subjects": len(all_results),
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "max_channels": MAX_CHANNELS,
        "subjects": all_results,
    }

    if len(valid) >= 3:
        out["group_stats"] = {
            "chirality_index_mean": float(np.mean(chi_idx)),
            "chirality_index_std": float(np.std(chi_idx)),
            "winding_number_mean": float(np.mean(windings)),
            "winding_number_std": float(np.std(windings)),
            "berry_phase_mean": float(np.mean(berry)),
            "berry_phase_std": float(np.std(berry)),
            "circular_variance_mean": float(np.mean(circ_var)),
            "circular_variance_std": float(np.std(circ_var)),
            "asymmetry_ratio_mean": float(np.mean(asym_ratios)),
            "asymmetry_ratio_std": float(np.std(asym_ratios)),
            "asymmetry_kurtosis_mean": float(np.mean(asym_kurtosis)),
            "asymmetry_kurtosis_std": float(np.std(asym_kurtosis)),
            "asymmetry_max_mean": float(np.mean(asym_max)),
            "asymmetry_p99_mean": float(np.mean(asym_p99)),
            "asymmetry_dynamic_range_mean": float(np.mean(asym_dynrange)),
            "asymmetry_dynamic_range_std": float(np.std(asym_dynrange)),
            "max_rotation_freq_mean": float(np.mean(max_rot_freqs)),
            "max_rotation_freq_std": float(np.std(max_rot_freqs)),
            "n_chiral_subjects": n_chiral,
        }
        out["correlations"] = corr_table

    log(f"\n{'='*70}")
    log("CHANNEL SCALING SWEEP")
    log("=" * 70)
    sweep_subjects = ["CE110", "CF105", "CF103", "CF124", "CG103"]
    channel_counts = [10, 20, 30, 50, 70, 100, 150, 200]
    sweep_results = []
    for subj in sweep_subjects:
        if subj not in hg_results:
            continue
        try:
            sr = channel_scaling_sweep(subj, config, channel_counts)
            if sr is not None:
                sweep_results.append(sr)
        except Exception as e:
            log(f"    SWEEP ERROR {subj}: {e}")

    out["channel_scaling_sweep"] = sweep_results

    out_path = RESULTS_HG / "analysis" / "chirality.json"
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
