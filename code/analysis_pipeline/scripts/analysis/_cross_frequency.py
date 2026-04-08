import os
"""Cross-frequency (PAC) analysis: broadband phase x high-gamma amplitude.

Tests whether phase-amplitude coupling between low-frequency phase and
high-gamma amplitude correlates with criticality metrics (tau, sigma)
and differs between task-relevant and task-irrelevant epochs.

Uses DurR1 only per subject to keep computation tractable.
Computes PAC on a channel subset (up to 30) for efficiency.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

from cmcc.config import load_config
from cmcc.preprocess.filter import SITE_LINE_FREQ
from cmcc.features.pac import compute_pac, compute_pac_per_channel

CMCC_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "default.yaml"
RESULTS_HG = CMCC_ROOT / "results"
RESULTS_BB = CMCC_ROOT / "results_broadband"
FIG_DIR = RESULTS_HG / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path(os.environ.get("IEEG_DATA_ROOT", "./data/Cogitate_IEEG_EXP1"))

MAX_CHANNELS = 30
N_SURROGATES = 50
PHASE_BAND = (4, 30)
AMP_BAND = (70, 150)


def log(msg):
    print(msg, flush=True)


def load_group_results():
    with open(RESULTS_HG / "group_all_subjects.json") as f:
        hg_data = json.load(f)
    hg_ok = {s["subject"]: s for s in hg_data if s.get("status") == "OK" and s.get("qc_pass", False)}

    bb_ok = {}
    bb_path = RESULTS_BB / "group_all_subjects.json"
    if bb_path.exists():
        with open(bb_path) as f:
            bb_data = json.load(f)
        bb_ok = {s["subject"]: s for s in bb_data if s.get("status") == "OK"}

    return hg_ok, bb_ok


def load_raw_durr1(subject_id, config):
    from cmcc.io.loader import load_subject
    from cmcc.preprocess.qc import detect_bad_channels, mark_bad_channels
    from cmcc.preprocess.filter import remove_line_noise
    from cmcc.preprocess.reference import apply_laplace

    site = subject_id[:2].upper()
    line_freq = SITE_LINE_FREQ.get(site, 60.0)

    subject_data = load_subject(subject_id, config["data"]["root"], runs=["DurR1"], preload_edf=True)
    if "DurR1" not in subject_data.raw:
        return None, None

    raw = subject_data.raw["DurR1"]

    non_ecog = [ch for ch in raw.ch_names
                if ch.startswith("DC") or ch.startswith("C12")
                or ch.startswith("EKG") or ch.startswith("EMG") or ch == "C128"]
    if non_ecog:
        raw.drop_channels(non_ecog)

    bad_channels = detect_bad_channels(raw)
    mark_bad_channels(raw, bad_channels)
    good_chs = [ch for ch in raw.ch_names if ch not in bad_channels]
    if len(good_chs) < 5:
        return None, None

    raw.pick(good_chs)
    raw = remove_line_noise(raw, line_freq=line_freq)
    if subject_data.laplace_map:
        raw = apply_laplace(raw, subject_data.laplace_map)

    return raw, subject_data


def load_events_and_epoch(raw, subject_id, config):
    meta_dir = DATA_ROOT / f"{subject_id}_ECOG_1" / "METADATA" / "electrode_coordinates"
    events_path = meta_dir / f"sub-{subject_id}_ses-1_task-Dur_events.tsv"
    if not events_path.exists():
        return None, None

    events_df = pd.read_csv(events_path, sep="\t", encoding="utf-8-sig")
    stim = events_df[events_df["trial_type"].str.contains("stimulus onset", na=False)].copy()
    stim = stim[stim["sample"] < raw.n_times].copy()

    if len(stim) == 0:
        return None, None

    parts = stim["trial_type"].str.split("/", expand=True)
    stim = stim.copy()
    stim["category"] = parts[3]
    stim["relevance"] = parts[7]
    stim["task_relevant"] = stim["relevance"].str.contains("Relevant").astype(int)

    return stim, raw


def compute_subject_pac(subject_id, config, hg_result):
    t0 = time.time()
    raw, subject_data = load_raw_durr1(subject_id, config)
    if raw is None:
        return None

    sfreq = raw.info["sfreq"]
    if AMP_BAND[1] >= sfreq / 2:
        log(f"  {subject_id}: sfreq={sfreq}, skip (Nyquist)")
        del raw, subject_data
        gc.collect()
        return None

    ch_names = raw.ch_names
    n_ch = len(ch_names)

    if n_ch > MAX_CHANNELS:
        rng = np.random.default_rng(42)
        ch_idx = sorted(rng.choice(n_ch, MAX_CHANNELS, replace=False))
        ch_subset = [ch_names[i] for i in ch_idx]
    else:
        ch_subset = list(ch_names)

    raw_subset = raw.copy().pick(ch_subset)
    data = raw_subset.get_data()

    log(f"  {subject_id}: {len(ch_subset)} ch, sfreq={sfreq}, computing PAC...")
    pac_results = compute_pac_per_channel(
        data, sfreq, ch_subset,
        phase_band=PHASE_BAND, amp_band=AMP_BAND,
        n_surrogates=N_SURROGATES, seed=42,
    )

    mi_values = {ch: r.modulation_index for ch, r in pac_results.items()}
    z_values = {ch: r.z_score for ch, r in pac_results.items()}
    p_values = {ch: r.p_value for ch, r in pac_results.items()}

    mean_mi = float(np.mean(list(mi_values.values())))
    mean_z = float(np.mean(list(z_values.values())))
    n_significant = sum(1 for p in p_values.values() if p < 0.05)

    stim_df, _ = load_events_and_epoch(raw, subject_id, config)
    task_pac = None
    if stim_df is not None and len(stim_df) >= 10:
        tmin_samp = int(0.0 * sfreq)
        tmax_samp = int(2.0 * sfreq)
        rel_trials = stim_df[stim_df["task_relevant"] == 1]
        irr_trials = stim_df[stim_df["task_relevant"] == 0]

        if len(rel_trials) >= 5 and len(irr_trials) >= 5:
            n_trial_ch = min(5, len(ch_subset))
            trial_data_ch = data[:n_trial_ch]

            rel_mis = []
            for _, row in rel_trials.head(20).iterrows():
                s0 = int(row["sample"]) + tmin_samp
                s1 = int(row["sample"]) + tmax_samp
                if s0 >= 0 and s1 < data.shape[1]:
                    for ci in range(n_trial_ch):
                        seg = data[ci, s0:s1]
                        if len(seg) > int(sfreq):
                            r = compute_pac(seg, sfreq, phase_band=PHASE_BAND,
                                          amp_band=AMP_BAND, n_surrogates=10, seed=42)
                            rel_mis.append(r.modulation_index)

            irr_mis = []
            for _, row in irr_trials.head(20).iterrows():
                s0 = int(row["sample"]) + tmin_samp
                s1 = int(row["sample"]) + tmax_samp
                if s0 >= 0 and s1 < data.shape[1]:
                    for ci in range(n_trial_ch):
                        seg = data[ci, s0:s1]
                        if len(seg) > int(sfreq):
                            r = compute_pac(seg, sfreq, phase_band=PHASE_BAND,
                                          amp_band=AMP_BAND, n_surrogates=10, seed=42)
                            irr_mis.append(r.modulation_index)

            if len(rel_mis) >= 5 and len(irr_mis) >= 5:
                t_stat, p_val = sp_stats.mannwhitneyu(rel_mis, irr_mis, alternative="two-sided")
                task_pac = {
                    "rel_mean_mi": float(np.mean(rel_mis)),
                    "irr_mean_mi": float(np.mean(irr_mis)),
                    "n_rel": len(rel_mis),
                    "n_irr": len(irr_mis),
                    "U": float(t_stat),
                    "p": float(p_val),
                }

    result = {
        "subject": subject_id,
        "n_channels": len(ch_subset),
        "sfreq": sfreq,
        "mean_mi": mean_mi,
        "mean_z": mean_z,
        "n_significant": n_significant,
        "frac_significant": n_significant / len(ch_subset),
        "per_channel_mi": {ch: float(v) for ch, v in mi_values.items()},
        "per_channel_z": {ch: float(v) for ch, v in z_values.items()},
        "hg_sigma": hg_result.get("branching_sigma"),
        "hg_tau": hg_result.get("tau"),
        "hg_lzc": hg_result.get("lzc_normalized"),
        "task_pac_contrast": task_pac,
        "elapsed_s": time.time() - t0,
    }

    del raw, subject_data, data, raw_subset, pac_results
    gc.collect()

    return result


def correlate_pac_with_criticality(all_results):
    subjects = [r for r in all_results if r is not None]
    if len(subjects) < 5:
        log("  Too few subjects for correlation")
        return {}

    mi = np.array([s["mean_mi"] for s in subjects])
    sigma = np.array([s["hg_sigma"] for s in subjects if s["hg_sigma"] is not None])
    tau = np.array([s["hg_tau"] for s in subjects if s["hg_tau"] is not None])
    lzc = np.array([s["hg_lzc"] for s in subjects if s["hg_lzc"] is not None])

    mi_sigma = mi[:len(sigma)]
    mi_tau = mi[:len(tau)]
    mi_lzc = mi[:len(lzc)]

    corrs = {}
    if len(sigma) >= 5:
        r, p = sp_stats.pearsonr(mi_sigma, sigma)
        corrs["mi_vs_sigma"] = {"r": float(r), "p": float(p), "n": len(sigma)}
    if len(tau) >= 5:
        r, p = sp_stats.pearsonr(mi_tau, tau)
        corrs["mi_vs_tau"] = {"r": float(r), "p": float(p), "n": len(tau)}
    if len(lzc) >= 5:
        r, p = sp_stats.pearsonr(mi_lzc, lzc)
        corrs["mi_vs_lzc"] = {"r": float(r), "p": float(p), "n": len(lzc)}

    return corrs


def plot_pac_summary(all_results, correlations, output_dir):
    subjects = [r for r in all_results if r is not None]
    if not subjects:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    mis = [s["mean_mi"] for s in subjects]
    names = [s["subject"] for s in subjects]
    ax = axes[0, 0]
    ax.barh(range(len(mis)), mis, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Mean Modulation Index")
    ax.set_title("PAC (broadband phase x HG amplitude)")
    ax.axvline(np.mean(mis), color="red", linestyle="--", label=f"mean={np.mean(mis):.4f}")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    sigmas = [s["hg_sigma"] for s in subjects if s["hg_sigma"] is not None]
    mi_for_sigma = [s["mean_mi"] for s in subjects if s["hg_sigma"] is not None]
    if len(sigmas) >= 3:
        ax.scatter(sigmas, mi_for_sigma, c="steelblue", edgecolors="black", s=50)
        ax.set_xlabel("HG sigma (branching ratio)")
        ax.set_ylabel("Mean PAC MI")
        ax.set_title("PAC vs Criticality (sigma)")
        if "mi_vs_sigma" in correlations:
            c = correlations["mi_vs_sigma"]
            ax.set_title(f"PAC vs sigma (r={c['r']:.3f}, p={c['p']:.3f})")

    ax = axes[1, 0]
    taus = [s["hg_tau"] for s in subjects if s["hg_tau"] is not None]
    mi_for_tau = [s["mean_mi"] for s in subjects if s["hg_tau"] is not None]
    if len(taus) >= 3:
        ax.scatter(taus, mi_for_tau, c="darkorange", edgecolors="black", s=50)
        ax.set_xlabel("HG tau (power-law exponent)")
        ax.set_ylabel("Mean PAC MI")
        if "mi_vs_tau" in correlations:
            c = correlations["mi_vs_tau"]
            ax.set_title(f"PAC vs tau (r={c['r']:.3f}, p={c['p']:.3f})")

    ax = axes[1, 1]
    task_subs = [s for s in subjects if s["task_pac_contrast"] is not None]
    if task_subs:
        rel_means = [s["task_pac_contrast"]["rel_mean_mi"] for s in task_subs]
        irr_means = [s["task_pac_contrast"]["irr_mean_mi"] for s in task_subs]
        x = np.arange(len(task_subs))
        w = 0.35
        ax.bar(x - w/2, rel_means, w, label="Relevant", color="coral", edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, irr_means, w, label="Irrelevant", color="skyblue", edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([s["subject"] for s in task_subs], rotation=45, fontsize=7)
        ax.set_ylabel("PAC MI")
        ax.set_title("Task Contrast: PAC MI")
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "pac_summary.png", dpi=150)
    plt.close(fig)
    log(f"  Saved: {output_dir / 'pac_summary.png'}")


def main():
    t0 = time.time()
    log("=" * 70)
    log("CROSS-FREQUENCY ANALYSIS: Phase-Amplitude Coupling")
    log("=" * 70)

    config = load_config(str(CONFIG_PATH))
    hg_ok, bb_ok = load_group_results()

    common = sorted(set(hg_ok) & set(bb_ok)) if bb_ok else sorted(hg_ok)
    log(f"  QC-passed HG subjects: {len(hg_ok)}")
    log(f"  Common HG+BB subjects: {len(common)}")

    subjects_to_process = common if common else sorted(hg_ok)
    log(f"  Processing {len(subjects_to_process)} subjects")

    all_results = []
    for i, subj in enumerate(subjects_to_process):
        log(f"\n[{i+1}/{len(subjects_to_process)}] {subj} [{time.time()-t0:.0f}s]")
        try:
            result = compute_subject_pac(subj, config, hg_ok[subj])
            if result:
                all_results.append(result)
                log(f"  MI={result['mean_mi']:.4f}, Z={result['mean_z']:.2f}, "
                    f"sig={result['n_significant']}/{result['n_channels']} [{result['elapsed_s']:.0f}s]")
            else:
                log(f"  SKIP")
        except Exception as e:
            log(f"  ERROR: {e}")

    log(f"\n{'='*70}")
    log(f"PAC RESULTS: {len(all_results)}/{len(subjects_to_process)} succeeded")

    correlations = correlate_pac_with_criticality(all_results)
    for key, val in correlations.items():
        log(f"  {key}: r={val['r']:.3f}, p={val['p']:.3f}, n={val['n']}")

    plot_pac_summary(all_results, correlations, FIG_DIR)

    output = {
        "n_subjects": len(all_results),
        "phase_band": list(PHASE_BAND),
        "amp_band": list(AMP_BAND),
        "n_surrogates": N_SURROGATES,
        "max_channels": MAX_CHANNELS,
        "correlations": correlations,
        "subjects": [],
    }
    for r in all_results:
        subj_out = {k: v for k, v in r.items() if k not in ("per_channel_mi", "per_channel_z")}
        subj_out["top_5_mi_channels"] = sorted(r["per_channel_mi"].items(), key=lambda x: -x[1])[:5]
        output["subjects"].append(subj_out)

    task_contrasts = [r["task_pac_contrast"] for r in all_results if r["task_pac_contrast"] is not None]
    if task_contrasts:
        rel_all = [t["rel_mean_mi"] for t in task_contrasts]
        irr_all = [t["irr_mean_mi"] for t in task_contrasts]
        if len(rel_all) >= 3:
            t_stat, p_val = sp_stats.ttest_rel(rel_all, irr_all)
            output["group_task_pac"] = {
                "mean_rel": float(np.mean(rel_all)),
                "mean_irr": float(np.mean(irr_all)),
                "paired_t": float(t_stat),
                "paired_p": float(p_val),
                "n": len(rel_all),
            }
            log(f"  Group task PAC: rel={np.mean(rel_all):.4f}, irr={np.mean(irr_all):.4f}, "
                f"t={t_stat:.3f}, p={p_val:.4f}")

    out_path = RESULTS_HG / "cross_frequency.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nSaved: {out_path}")
    log(f"\nDONE in {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
