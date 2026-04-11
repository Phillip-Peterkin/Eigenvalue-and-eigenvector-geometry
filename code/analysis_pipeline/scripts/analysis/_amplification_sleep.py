"""
Sleep convergence analysis for transient amplification.

Tests whether the propofol Kreiss finding replicates in natural sleep:
  - Wake vs N3: expect Kreiss increase (convergent with propofol sedation)
  - Wake vs REM: test whether dreaming (active cortex) differs from N3
  - N3 vs REM: dissociation within unconsciousness

If Kreiss increases in both propofol sedation AND natural N3 sleep,
the amplification-state relationship generalizes across two independent
datasets and two different mechanisms of consciousness loss.

What this analysis CAN establish
---------------------------------
- Whether operator non-normality is higher in slow-wave sleep
- Whether the direction matches propofol (convergent validity)
- Whether REM dissociates from N3 (mechanistic specificity)

What this analysis CANNOT establish
------------------------------------
- That non-normal amplification causes unconsciousness
- That the effect is in the same brain regions across modalities
  (scalp EEG vs polysomnography differ in coverage)

Pipeline stage: feature construction + statistical inference
Dataset: ANPHY-Sleep — 10 subjects, polysomnography, Wake/N3/REM
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

import mne
from cmcc.preprocess.scalp_eeg import apply_csd, pca_reduce
from cmcc.preprocess.qc import detect_bad_channels
from cmcc.analysis.dynamical_systems import estimate_jacobian
from cmcc.features.transient_amplification import (
    analyze_jacobian_amplification,
    compute_hump_magnitude,
    compute_residual_kreiss,
    compute_model_free_energy_growth,
)

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("ANPHY_SLEEP_ROOT", r"c:\openneuro\ANPHY-Sleep"))
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"
FIG_DIR = CMCC_ROOT / "results" / "figures" / "amplification_sleep"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 0.5
STEP_SEC = 0.1
N_COMPONENTS = 15
DOWNSAMPLE_TO = 500.0
LINE_FREQ = 50.0
MAX_HORIZON = 50
SEED = 42

NON_EEG_CHANNELS = [
    "SO1", "SO2", "ZY1", "ZY2",
    "ChEMG1", "ChEMG2",
    "RLEG-", "RLEG+", "LLEG-", "LLEG+",
    "EOG2", "EOG1", "ECG2", "ECG1",
]

OLD_TO_NEW_NAMES = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}

SUBJECTS = {
    "EPCTL01": ("EPCTL01 - fixed.edf", "test1.txt"),
    "EPCTL03": ("EPCTL03.edf", "EPCTL03.txt"),
    "EPCTL05": ("EPCTL05.edf", "EPCTL05.txt"),
    "EPCTL07": ("EPCTL07.edf", "EPCTL07.txt"),
    "EPCTL10": ("EPCTL10.edf", "EPCTL10.txt"),
    "EPCTL14": ("EPCTL14.edf", "EPCTL14.txt"),
    "EPCTL17": ("EPCTL17.edf", "EPCTL17.txt"),
    "EPCTL20": ("EPCTL20.edf", "EPCTL20.txt"),
    "EPCTL24": ("EPCTL24.edf", "EPCTL24.txt"),
    "EPCTL28": ("EPCTL28.edf", "EPCTL28.txt"),
}

MIN_CONTIGUOUS_EPOCHS = 4
TARGET_STATES = ["W", "N3", "R"]


def log(msg):
    print(msg, flush=True)


def parse_sleep_staging(staging_path):
    epochs = []
    with open(staging_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                stage = parts[0]
                onset_sec = float(parts[1])
                duration_sec = float(parts[2])
                epochs.append((stage, onset_sec, duration_sec))
    return epochs


def find_longest_contiguous_run(epochs, target_stage, min_epochs=4):
    runs = []
    current_start = None
    current_count = 0

    for i, (stage, onset, dur) in enumerate(epochs):
        if stage == target_stage:
            if current_start is None:
                current_start = i
                current_count = 1
            else:
                current_count += 1
        else:
            if current_start is not None and current_count >= min_epochs:
                start_sec = epochs[current_start][1]
                end_sec = epochs[current_start + current_count - 1][1] + epochs[current_start + current_count - 1][2]
                runs.append((start_sec, end_sec, current_count))
            current_start = None
            current_count = 0

    if current_start is not None and current_count >= min_epochs:
        start_sec = epochs[current_start][1]
        end_sec = epochs[current_start + current_count - 1][1] + epochs[current_start + current_count - 1][2]
        runs.append((start_sec, end_sec, current_count))

    if not runs:
        return None

    return max(runs, key=lambda r: r[2])


def load_and_preprocess_segment(edf_path, start_sec, end_sec, sfreq_target=500.0):
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)

    drop_chs = [ch for ch in NON_EEG_CHANNELS if ch in raw.ch_names]
    if drop_chs:
        raw.drop_channels(drop_chs)

    rename_map = {old: new for old, new in OLD_TO_NEW_NAMES.items() if old in raw.ch_names}
    if rename_map:
        raw.rename_channels(rename_map)

    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    montage = mne.channels.make_standard_montage("standard_1005")
    montage_names = set(montage.ch_names)
    missing = [ch for ch in raw.ch_names if ch not in montage_names]
    if missing:
        raw.drop_channels(missing)

    raw.set_montage(montage, on_missing="warn")

    raw.crop(tmin=start_sec, tmax=end_sec)
    raw.load_data()

    bad = detect_bad_channels(raw)
    raw.info["bads"] = bad
    if bad:
        raw.interpolate_bads(reset_bads=True)

    freqs = [LINE_FREQ * i for i in range(1, 4)]
    raw.notch_filter(freqs, verbose=False)

    raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)

    if raw.info["sfreq"] > sfreq_target:
        raw.resample(sfreq_target, verbose=False)

    raw = apply_csd(raw)

    data = raw.get_data()
    data_pca, pca_obj = pca_reduce(data, n_components=N_COMPONENTS, return_pca=True)

    info = {
        "n_channels_original": data.shape[0],
        "bad_channels": bad,
        "sfreq_final": raw.info["sfreq"],
        "duration_sec": float(data_pca.shape[1] / raw.info["sfreq"]),
        "n_components": data_pca.shape[0],
        "cumulative_variance": float(np.sum(pca_obj.explained_variance_ratio_)),
        "channels_dropped_no_montage": missing,
    }

    return data_pca, raw.info["sfreq"], info


def analyze_state_amplification(data_pca, sfreq, state_label):
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
        stable_envelopes = amp["envelopes"][stable_mask]
        stable_mean_env = np.nanmean(stable_envelopes, axis=0)
    else:
        stable_kreiss = np.array([np.nan])
        stable_amp_ratio = np.array([np.nan])
        stable_hump_mag = np.array([np.nan])
        stable_mean_env = amp["mean_envelope"]

    mfeg = compute_model_free_energy_growth(
        data_pca, sfreq=sfreq,
        percentile_threshold=95.0,
        horizon_sec=0.1,
    )

    return {
        "state": state_label,
        "n_windows": n_windows,
        "n_stable_windows": n_stable,
        "fraction_stable": frac_stable,
        "kreiss_mean": float(np.mean(amp["kreiss_constants"])),
        "kreiss_median": float(np.median(amp["kreiss_constants"])),
        "stable_kreiss_median": float(np.nanmedian(stable_kreiss)),
        "amplification_ratio_median": float(np.median(amp["amplification_ratios"])),
        "stable_amplification_ratio_median": float(np.nanmedian(stable_amp_ratio)),
        "hump_magnitude_median": float(np.median(hump_mags)),
        "stable_hump_magnitude_median": float(np.nanmedian(stable_hump_mag)),
        "hump_fraction": float(np.mean(amp["has_hump"])),
        "spectral_radius_mean": float(np.mean(jac_result.spectral_radius)),
        "spectral_radius_median": float(np.median(jac_result.spectral_radius)),
        "condition_number_median": float(np.median(jac_result.condition_numbers)),
        "mean_envelope": amp["mean_envelope"].tolist(),
        "stable_mean_envelope": stable_mean_env.tolist(),
        "kreiss_array": amp["kreiss_constants"],
        "spectral_radius_array": jac_result.spectral_radius,
        "model_free_energy": {
            "n_events": mfeg["n_events"],
            "fraction_growing": mfeg["fraction_growing_at_peak"],
            "peak_growth_ratio": mfeg["peak_growth_ratio"],
            "peak_lag_sec": mfeg["peak_lag_sec"],
        },
    }


def analyze_subject(subject_id, edf_file, staging_file):
    t0 = time.time()
    log(f"  {subject_id}...")

    edf_path = DATA_ROOT / subject_id / edf_file
    staging_path = DATA_ROOT / subject_id / staging_file

    if not edf_path.exists() or not staging_path.exists():
        log(f"    SKIP: missing files")
        return None

    epochs = parse_sleep_staging(staging_path)

    state_results = {}
    for state in TARGET_STATES:
        run = find_longest_contiguous_run(epochs, state, MIN_CONTIGUOUS_EPOCHS)
        if run is None:
            log(f"    {state}: no contiguous run >= {MIN_CONTIGUOUS_EPOCHS} epochs")
            state_results[state] = None
            continue

        start_sec, end_sec, n_epochs = run
        log(f"    {state}: {n_epochs} epochs ({end_sec - start_sec:.0f}s)")

        try:
            data_pca, sfreq, preprocess_info = load_and_preprocess_segment(
                edf_path, start_sec, end_sec, sfreq_target=DOWNSAMPLE_TO,
            )
        except Exception as e:
            log(f"    {state} ERROR: {e}")
            state_results[state] = None
            continue

        result = analyze_state_amplification(data_pca, sfreq, state)
        result["n_staging_epochs"] = n_epochs
        result["segment_start_sec"] = start_sec
        result["segment_end_sec"] = end_sec
        result["preprocess"] = preprocess_info

        log(f"    {state}: kreiss_med={result['kreiss_median']:.3f}, "
            f"stable_kr={result['stable_kreiss_median']:.3f}, "
            f"rho={result['spectral_radius_median']:.4f}")

        state_results[state] = result

        del data_pca
        gc.collect()

    elapsed = time.time() - t0

    if state_results.get("W") is None:
        log(f"    SKIP: no wake data")
        return None

    all_kreiss = []
    all_rho = []
    all_labels = []
    for st in TARGET_STATES:
        if state_results.get(st) is not None and "kreiss_array" in state_results[st]:
            kr = state_results[st]["kreiss_array"]
            rh = state_results[st]["spectral_radius_array"]
            all_kreiss.append(kr)
            all_rho.append(rh)
            all_labels.append(np.full(len(kr), TARGET_STATES.index(st)))

    if len(all_kreiss) >= 2:
        pooled_kr = np.concatenate(all_kreiss)
        pooled_rho = np.concatenate(all_rho)
        pooled_labels = np.concatenate(all_labels)

        pooled_resid = compute_residual_kreiss(pooled_kr, pooled_rho)

        if len(pooled_resid["residuals"]) > 0:
            stable_mask = pooled_rho < 1.0
            residuals = pooled_resid["residuals"]
            labels_stable = pooled_labels[stable_mask]

            for st in TARGET_STATES:
                st_idx = TARGET_STATES.index(st)
                if state_results.get(st) is not None:
                    st_mask = labels_stable == st_idx
                    if np.sum(st_mask) > 0:
                        state_results[st]["pooled_residual_kreiss_mean"] = float(np.mean(residuals[st_mask]))
                        state_results[st]["pooled_residual_kreiss_median"] = float(np.median(residuals[st_mask]))
                    else:
                        state_results[st]["pooled_residual_kreiss_mean"] = float("nan")
                        state_results[st]["pooled_residual_kreiss_median"] = float("nan")

            log(f"    pooled residual-kreiss slope={pooled_resid['slope']:.3f}, R2={pooled_resid['r_squared']:.3f}")

    for st in TARGET_STATES:
        if state_results.get(st) is not None:
            state_results[st].pop("kreiss_array", None)
            state_results[st].pop("spectral_radius_array", None)

    return {
        "subject": subject_id,
        "states": state_results,
        "elapsed_s": elapsed,
    }


def compute_group_statistics(subjects_data):
    complete = [
        s for s in subjects_data
        if all(s["states"].get(st) is not None for st in TARGET_STATES)
    ]
    log(f"\n  Complete subjects (all 3 states): {len(complete)}")

    if len(complete) < 3:
        return {"n_complete": len(complete), "insufficient_data": True}

    contrasts = [("W", "N3"), ("W", "R"), ("N3", "R")]
    metrics = [
        "stable_kreiss_median", "stable_amplification_ratio_median",
        "stable_hump_magnitude_median", "spectral_radius_median",
        "condition_number_median",
    ]

    group = {"n_complete": len(complete)}

    for st_a, st_b in contrasts:
        contrast_key = f"{st_a}_vs_{st_b}"
        group[contrast_key] = {}

        for metric in metrics:
            vals_a = [s["states"][st_a][metric] for s in complete]
            vals_b = [s["states"][st_b][metric] for s in complete]
            delta = np.array(vals_b) - np.array(vals_a)

            if np.std(delta) > 1e-30 and len(delta) >= 3:
                t, p = sp_stats.ttest_1samp(delta, 0.0)
                w, pw = sp_stats.wilcoxon(delta)
                d = float(np.mean(delta) / np.std(delta, ddof=1))
            else:
                t, p, w, pw, d = 0, 1.0, 0, 1.0, 0.0

            group[contrast_key][metric] = {
                "mean_a": float(np.mean(vals_a)),
                "mean_b": float(np.mean(vals_b)),
                "delta_mean": float(np.mean(delta)),
                "delta_median": float(np.median(delta)),
                "cohens_d": d,
                "t_stat": float(t),
                "p_ttest": float(p),
                "p_wilcoxon": float(pw),
                "n_positive": int(np.sum(delta > 0)),
                "n": len(complete),
            }

        resid_a = [s["states"][st_a].get("pooled_residual_kreiss_mean", float("nan")) for s in complete]
        resid_b = [s["states"][st_b].get("pooled_residual_kreiss_mean", float("nan")) for s in complete]
        delta_resid = np.array(resid_b) - np.array(resid_a)

        valid_resid = np.isfinite(delta_resid)
        if np.sum(valid_resid) >= 3:
            dr = delta_resid[valid_resid]
            t, p = sp_stats.ttest_1samp(dr, 0.0)
            if len(dr) >= 6:
                w, pw = sp_stats.wilcoxon(dr)
            else:
                w, pw = float("nan"), float("nan")
            d = float(np.mean(dr) / max(np.std(dr, ddof=1), 1e-30))
        else:
            t, p, w, pw, d = 0, 1.0, 0, 1.0, 0.0

        group[contrast_key]["residual_kreiss"] = {
            "delta_mean": float(np.nanmean(delta_resid)),
            "delta_median": float(np.nanmedian(delta_resid)),
            "cohens_d": d,
            "p_ttest": float(p),
            "p_wilcoxon": float(pw) if np.isfinite(pw) else None,
            "n_valid": int(np.sum(valid_resid)),
        }

        mf_a = [s["states"][st_a]["model_free_energy"]["peak_growth_ratio"] for s in complete]
        mf_b = [s["states"][st_b]["model_free_energy"]["peak_growth_ratio"] for s in complete]
        delta_mf = np.array(mf_b) - np.array(mf_a)
        if np.std(delta_mf) > 1e-30:
            t, p = sp_stats.ttest_1samp(delta_mf, 0.0)
            w, pw = sp_stats.wilcoxon(delta_mf)
            d = float(np.mean(delta_mf) / np.std(delta_mf, ddof=1))
        else:
            t, p, w, pw, d = 0, 1.0, 0, 1.0, 0.0

        group[contrast_key]["model_free_growth"] = {
            "delta_mean": float(np.mean(delta_mf)),
            "cohens_d": d,
            "p_wilcoxon": float(pw),
            "n_positive": int(np.sum(delta_mf > 0)),
        }

    per_state = {}
    for st in TARGET_STATES:
        vals = [s["states"][st] for s in complete]
        per_state[st] = {
            "kreiss_median": float(np.median([v["kreiss_median"] for v in vals])),
            "stable_kreiss_median": float(np.median([v["stable_kreiss_median"] for v in vals])),
            "spectral_radius_median": float(np.median([v["spectral_radius_median"] for v in vals])),
            "fraction_stable_mean": float(np.mean([v["fraction_stable"] for v in vals])),
            "hump_fraction_mean": float(np.mean([v["hump_fraction"] for v in vals])),
        }
    group["per_state_summary"] = per_state

    return group


def make_sleep_figure(subjects_data, group_stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    complete = [
        s for s in subjects_data
        if all(s["states"].get(st) is not None for st in TARGET_STATES)
    ]

    if len(complete) < 3:
        log("  Insufficient data for figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    for s in complete:
        vals = [s["states"][st]["stable_kreiss_median"] for st in TARGET_STATES]
        ax.plot(range(3), vals, "o-", alpha=0.3, color="gray")
    medians = [
        np.median([s["states"][st]["stable_kreiss_median"] for s in complete])
        for st in TARGET_STATES
    ]
    ax.plot(range(3), medians, "s-", color="red", linewidth=2, markersize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Wake", "N3", "REM"])
    ax.set_ylabel("Stable Kreiss median")
    wn3 = group_stats.get("W_vs_N3", {}).get("stable_kreiss_median", {})
    ax.set_title(f"Kreiss: Wake vs N3 vs REM\nW→N3: d={wn3.get('cohens_d', 0):.2f}, "
                 f"p={wn3.get('p_wilcoxon', 1):.4f}")

    ax = axes[0, 1]
    for s in complete:
        vals = [s["states"][st]["spectral_radius_median"] for st in TARGET_STATES]
        ax.plot(range(3), vals, "o-", alpha=0.3, color="gray")
    medians = [
        np.median([s["states"][st]["spectral_radius_median"] for s in complete])
        for st in TARGET_STATES
    ]
    ax.plot(range(3), medians, "s-", color="blue", linewidth=2, markersize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Wake", "N3", "REM"])
    ax.set_ylabel("Spectral radius median")
    ax.set_title("Spectral radius across states")

    ax = axes[1, 0]
    for s in complete:
        env_w = np.array(s["states"]["W"]["stable_mean_envelope"])
        env_n3 = np.array(s["states"]["N3"]["stable_mean_envelope"])
        ax.plot(env_w, alpha=0.2, color="blue")
        ax.plot(env_n3, alpha=0.2, color="red")
    mean_w = np.mean([s["states"]["W"]["stable_mean_envelope"] for s in complete], axis=0)
    mean_n3 = np.mean([s["states"]["N3"]["stable_mean_envelope"] for s in complete], axis=0)
    mean_r = np.mean([s["states"]["R"]["stable_mean_envelope"] for s in complete], axis=0)
    ax.plot(mean_w, "b-", linewidth=2, label="Wake")
    ax.plot(mean_n3, "r-", linewidth=2, label="N3")
    ax.plot(mean_r, "g-", linewidth=2, label="REM")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Time step k")
    ax.set_ylabel("||A^k||_2")
    ax.set_title("Energy envelopes (stable windows)")
    ax.legend()

    ax = axes[1, 1]
    for s in complete:
        vals = [s["states"][st].get("pooled_residual_kreiss_mean", float("nan")) for st in TARGET_STATES]
        ax.plot(range(3), vals, "o-", alpha=0.3, color="gray")
    medians = [
        np.nanmedian([s["states"][st].get("pooled_residual_kreiss_mean", float("nan")) for s in complete])
        for st in TARGET_STATES
    ]
    ax.plot(range(3), medians, "s-", color="green", linewidth=2, markersize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Wake", "N3", "REM"])
    ax.set_ylabel("Residual log-Kreiss")
    rk = group_stats.get("W_vs_N3", {}).get("residual_kreiss", {})
    ax.set_title(f"Rho-controlled Kreiss\nW→N3: d={rk.get('cohens_d', 0):.2f}, "
                 f"p={rk.get('p_ttest', 1):.4f}")

    fig.suptitle(
        "Sleep Convergence: Transient Amplification\n"
        "(Complementary dataset — natural sleep)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = FIG_DIR / "amplification_sleep_convergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Figure saved: {out_path}")


def main():
    log("=" * 60)
    log("SLEEP CONVERGENCE: TRANSIENT AMPLIFICATION")
    log("Complementary dataset test of propofol Kreiss finding")
    log("=" * 60)

    subjects_data = []
    for subj_id, (edf_file, staging_file) in SUBJECTS.items():
        result = analyze_subject(subj_id, edf_file, staging_file)
        if result is not None:
            subjects_data.append(result)
        gc.collect()

    log(f"\n{len(subjects_data)} subjects completed")

    group = compute_group_statistics(subjects_data)

    log("\n--- Group statistics ---")
    for contrast in ["W_vs_N3", "W_vs_R", "N3_vs_R"]:
        c = group.get(contrast, {})
        kr = c.get("stable_kreiss_median", {})
        rk = c.get("residual_kreiss", {})
        mf = c.get("model_free_growth", {})
        log(f"\n  {contrast}:")
        log(f"    Stable Kreiss: d={kr.get('cohens_d', 0):.3f}, "
            f"p(w)={kr.get('p_wilcoxon', 1):.4f}, "
            f"{kr.get('n_positive', 0)}/{kr.get('n', 0)} positive")
        log(f"    Residual Kreiss: d={rk.get('cohens_d', 0):.3f}, "
            f"p(t)={rk.get('p_ttest', 1):.4f}")
        log(f"    Model-free growth: d={mf.get('cohens_d', 0):.3f}, "
            f"p(w)={mf.get('p_wilcoxon', 1):.4f}")

    ps = group.get("per_state_summary", {})
    for st in TARGET_STATES:
        s = ps.get(st, {})
        log(f"\n  {st}: kreiss_med={s.get('stable_kreiss_median', 0):.3f}, "
            f"rho_med={s.get('spectral_radius_median', 0):.4f}, "
            f"frac_stable={s.get('fraction_stable_mean', 0):.3f}")

    output = {
        "analysis": "amplification_sleep_convergence",
        "description": (
            "Sleep convergence analysis for transient amplification. "
            "Tests whether the propofol Kreiss finding replicates in "
            "natural sleep (Wake vs N3 vs REM) using an independent "
            "dataset and mechanism of consciousness loss. Includes "
            "spectral-radius-controlled Kreiss and model-free energy "
            "growth as in the propofol analysis."
        ),
        "parameters": {
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "n_components": N_COMPONENTS,
            "max_horizon": MAX_HORIZON,
            "downsample_to": DOWNSAMPLE_TO,
            "min_contiguous_epochs": MIN_CONTIGUOUS_EPOCHS,
        },
        "n_subjects": len(subjects_data),
        "group_statistics": group,
        "subjects": subjects_data,
        "convergence_prediction": (
            "If propofol finding is genuine: N3 Kreiss > Wake Kreiss "
            "(same direction as sedation > awake). REM may differ from "
            "N3 if dreaming cortex has different non-normality structure."
        ),
        "guardrail": (
            "Sleep and propofol produce unconsciousness via different "
            "mechanisms. Convergence strengthens the state-dependence "
            "claim but does not prove a common mechanism. EEG coverage "
            "differs between datasets."
        ),
    }

    out_path = RESULTS_DIR / "amplification_sleep_convergence.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nResults saved: {out_path}")

    try:
        make_sleep_figure(subjects_data, group)
    except Exception as e:
        log(f"  Figure generation failed: {e}")

    log("\nDone.")


if __name__ == "__main__":
    main()
