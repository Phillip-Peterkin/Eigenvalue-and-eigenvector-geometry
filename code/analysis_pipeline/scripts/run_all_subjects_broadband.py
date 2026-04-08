"""CMCC Batch Pipeline Runner - All 38 Cogitate iEEG subjects.

BROADBAND REPLICATION: Identical pipeline using 1-200 Hz broadband instead of 70-150 Hz high-gamma.
Caps h_freq at Nyquist-1 for low sampling rate subjects.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from cmcc.config import load_config
from cmcc.preprocess.filter import SITE_LINE_FREQ

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "broadband.yaml"
RUNS = ["DurR1", "DurR2", "DurR3", "DurR4", "DurR5"]


def log(msg):
    print(msg, flush=True)


def load_events_tsv(data_root, subject_id):
    meta_dir = Path(data_root) / f"{subject_id}_ECOG_1" / "METADATA" / "electrode_coordinates"
    return pd.read_csv(meta_dir / f"sub-{subject_id}_ses-1_task-Dur_events.tsv", sep="\t", encoding="utf-8-sig")


def parse_stimulus_events(events_df, sample_offset, n_samples):
    stim = events_df[events_df["trial_type"].str.contains("stimulus onset", na=False)].copy()
    mask = (stim["sample"] >= sample_offset) & (stim["sample"] < sample_offset + n_samples)
    stim = stim[mask].copy()
    stim["sample"] = stim["sample"] - sample_offset
    parts = stim["trial_type"].str.split("/", expand=True)
    stim["category"] = parts[3]
    stim["duration_ms"] = parts[6].str.replace("ms", "").astype(int)
    stim["relevance"] = parts[7]
    stim["block"] = parts[1].str.extract(r"block_(\d+)").astype(int)
    stim["miniblock"] = parts[2].str.extract(r"miniblock_(\d+)").astype(int)
    stim["task_relevant"] = stim["relevance"].str.contains("Relevant").astype(int)
    return stim.reset_index(drop=True)


def run_single_subject(subject_id, config, results_dir):
    """Run the full CMCC pipeline for a single subject, pooling all runs."""
    t0 = time.time()
    site = subject_id[:2].upper()
    line_freq = SITE_LINE_FREQ.get(site, 60.0)
    seed = config["random_seed"]

    from cmcc.io.loader import load_subject
    from cmcc.preprocess.qc import detect_bad_channels, mark_bad_channels
    from cmcc.preprocess.filter import remove_line_noise, extract_high_gamma
    from cmcc.preprocess.reference import apply_laplace

    subject_data = load_subject(subject_id, config["data"]["root"], runs=RUNS, preload_edf=True)
    events_all = load_events_tsv(config["data"]["root"], subject_id)

    available_runs = [r for r in RUNS if r in subject_data.raw]
    if not available_runs:
        return {"subject": subject_id, "status": "SKIP", "error": "no runs found"}
    log(f"  Runs: {available_runs}")

    per_run_good_channels = {}
    per_run_raw = {}
    per_run_n_times = {}

    for run_id in available_runs:
        raw = subject_data.raw[run_id]
        per_run_n_times[run_id] = raw.n_times
        non_ecog = [ch for ch in raw.ch_names if ch.startswith("DC") or ch.startswith("C12") or ch.startswith("EKG") or ch.startswith("EMG") or ch == "C128"]
        if non_ecog:
            raw.drop_channels(non_ecog)
        bad_channels = detect_bad_channels(raw)
        mark_bad_channels(raw, bad_channels)
        good_chs = [ch for ch in raw.ch_names if ch not in bad_channels]
        if len(good_chs) < 5:
            log(f"    {run_id}: only {len(good_chs)} good channels, skipping run")
            continue
        raw.pick(good_chs)
        raw = remove_line_noise(raw, line_freq=line_freq)
        if subject_data.laplace_map:
            raw = apply_laplace(raw, subject_data.laplace_map)
        per_run_good_channels[run_id] = set(raw.ch_names)
        per_run_raw[run_id] = raw

    valid_runs = list(per_run_raw.keys())
    if len(valid_runs) < 1:
        return {"subject": subject_id, "status": "SKIP", "error": "no valid runs after QC"}

    common_channels = sorted(set.intersection(*[per_run_good_channels[r] for r in valid_runs]))
    if len(common_channels) < 5:
        return {"subject": subject_id, "status": "SKIP", "error": f"only {len(common_channels)} common channels"}
    log(f"  Common channels: {len(common_channels)} [{time.time()-t0:.0f}s]")

    for run_id in valid_runs:
        raw = per_run_raw[run_id]
        ch_to_drop = [ch for ch in raw.ch_names if ch not in common_channels]
        if ch_to_drop:
            raw.drop_channels(ch_to_drop)
        ch_order = [ch for ch in common_channels if ch in raw.ch_names]
        raw.reorder_channels(ch_order)
        per_run_raw[run_id] = raw

    all_epoch_data = []
    all_metadata = []
    all_gamma_continuous = []
    block_offset = 0
    cumulative_sample_offset = 0

    for run_idx, run_id in enumerate(valid_runs):
        raw = per_run_raw[run_id]
        passband = list(config["preprocessing"]["high_gamma_passband"])
        nyquist = raw.info["sfreq"] / 2.0
        if passband[1] >= nyquist:
            passband[1] = int(nyquist - 1)
        if passband[0] >= passband[1]:
            passband[0] = max(1, passband[1] - 10)
        gamma_raw = extract_high_gamma(raw, passband=tuple(passband))
        sfreq = gamma_raw.info["sfreq"]
        n_channels = gamma_raw.info["nchan"]

        gamma_cont = gamma_raw.get_data()
        all_gamma_continuous.append(gamma_cont)

        raw_n_times = per_run_n_times[run_id]
        stim_events = parse_stimulus_events(events_all, cumulative_sample_offset, raw_n_times)
        cumulative_sample_offset += raw_n_times

        if len(stim_events) == 0:
            log(f"    {run_id}: 0 stimulus events, skipping")
            all_gamma_continuous.pop()
            continue

        sample_indices = stim_events["sample"].values.astype(int)
        events_array = np.column_stack([
            sample_indices,
            np.zeros(len(sample_indices), dtype=int),
            np.ones(len(sample_indices), dtype=int),
        ])
        metadata = pd.DataFrame({
            "trial": np.arange(len(stim_events)),
            "block": stim_events["block"].values,
            "miniblock": stim_events["miniblock"].values,
            "category": stim_events["category"].values,
            "duration_ms": stim_events["duration_ms"].values,
            "task_relevant": stim_events["task_relevant"].values,
            "run": run_id,
            "run_idx": run_idx,
            "global_block": stim_events["block"].values + block_offset,
        })
        baseline = tuple(config["preprocessing"]["baseline"]) if config["preprocessing"]["baseline"] else None
        epochs = mne.Epochs(gamma_raw, events=events_array, event_id={"stimulus": 1},
                            tmin=config["preprocessing"]["epoch_tmin"],
                            tmax=config["preprocessing"]["epoch_tmax"],
                            baseline=baseline, metadata=metadata, preload=True, verbose=False)
        epoch_data = epochs.get_data(copy=True)
        n_trials = epoch_data.shape[0]

        max_block_this_run = stim_events["block"].max() if len(stim_events) > 0 else 0
        block_offset += max_block_this_run + 1

        all_epoch_data.append(epoch_data)
        all_metadata.append(metadata.iloc[:n_trials].reset_index(drop=True))
        log(f"    {run_id}: {n_trials} trials [{time.time()-t0:.0f}s]")
        del gamma_raw, epochs, epoch_data
        gc.collect()

    if not all_epoch_data:
        return {"subject": subject_id, "status": "SKIP", "error": "no valid epochs"}

    pooled_epochs = np.concatenate(all_epoch_data, axis=0)
    pooled_metadata = pd.concat(all_metadata, ignore_index=True)
    gamma_concatenated = np.concatenate(all_gamma_continuous, axis=1)
    ch_names = list(common_channels)
    n_trials_total, n_channels, n_timepoints = pooled_epochs.shape

    del all_epoch_data, all_gamma_continuous, per_run_raw, subject_data
    gc.collect()

    log(f"  Pooled: {n_trials_total} trials, {n_channels} ch [{time.time()-t0:.0f}s]")

    from cmcc.features.avalanche import detect_avalanches
    from cmcc.features.powerlaw_fit import fit_avalanche_distributions, _fit_single_distribution
    from cmcc.features.complexity import compute_lzc
    from cmcc.features.entropy import compute_mse
    from cmcc.features.dfa import compute_dfa
    from cmcc.features.branching import compute_branching_ratio
    from cmcc.analysis.contrasts import condition_contrast
    from cmcc.analysis.decoding import criticality_decoding_analysis

    avalanches = detect_avalanches(gamma_concatenated, sfreq=sfreq,
                                   threshold_sd=config["avalanche"]["threshold_sd"],
                                   bin_width_factor=config["avalanche"]["bin_width_factor"])
    sizes = np.array([a.size for a in avalanches])
    durations = np.array([a.duration_bins for a in avalanches])

    if len(avalanches) < 10:
        return {"subject": subject_id, "status": "SKIP", "error": f"only {len(avalanches)} avalanches"}

    pl_result = fit_avalanche_distributions(sizes, durations, discrete=True, n_bootstrap=100,
                                            compare_distributions=config["powerlaw"]["compare_distributions"])
    log(f"  tau={pl_result.tau:.3f} alpha={pl_result.alpha:.3f} [{time.time()-t0:.0f}s]")

    n_lzc_ch = min(10, n_channels)
    n_lzc_t = min(2000, gamma_concatenated.shape[1])
    lzc_result = compute_lzc(gamma_concatenated[:n_lzc_ch, :n_lzc_t], n_surrogates=3, seed=seed)

    mean_gamma = gamma_concatenated.mean(axis=0)
    mse_result = compute_mse(mean_gamma[:3000], scales=list(range(1, 21)), m=2, r_factor=0.15)
    dfa_result = compute_dfa(mean_gamma[:50000])

    ch_m = gamma_concatenated.mean(axis=1, keepdims=True)
    ch_s = gamma_concatenated.std(axis=1, keepdims=True)
    ch_s[ch_s == 0] = 1.0
    binary = (np.abs((gamma_concatenated - ch_m) / ch_s) > config["avalanche"]["threshold_sd"]).astype(np.int8)
    br_result = compute_branching_ratio(binary)
    log(f"  sigma={br_result.sigma:.4f} LZc={lzc_result.lzc_normalized:.4f} DFA={dfa_result.alpha:.4f} [{time.time()-t0:.0f}s]")

    qc_pass = True
    qc_reasons = []
    if len(avalanches) < 50:
        qc_pass = False
        qc_reasons.append(f"too few avalanches ({len(avalanches)}<50)")
    if not (0.8 <= br_result.sigma <= 1.1):
        qc_pass = False
        qc_reasons.append(f"sigma out of range ({br_result.sigma:.3f} not in [0.8,1.1])")
    if n_trials_total < 100:
        qc_pass = False
        qc_reasons.append(f"too few trials ({n_trials_total}<100)")
    if n_channels < 10:
        qc_pass = False
        qc_reasons.append(f"too few channels ({n_channels}<10)")

    power_contrast = None
    lzc_contrast = None
    cat_contrasts = {}
    decoding_results = None
    tau_per_channel = {}
    valid_tau = 0
    n_epoch_lzc_ch = 0

    if not qc_pass:
        log(f"  QC GATE FAILED — skipping contrasts+decoding: {'; '.join(qc_reasons)} [{time.time()-t0:.0f}s]")
    else:
        log(f"  QC GATE PASSED — running contrasts+decoding [{time.time()-t0:.0f}s]")

        stim_onset_idx = int(abs(config["preprocessing"]["epoch_tmin"]) * sfreq)
        stim_epochs = pooled_epochs[:, :, stim_onset_idx:]
        log(f"  Stimulus-only epochs: {stim_epochs.shape} (dropped {stim_onset_idx} baseline samples)")

        rel_mask = pooled_metadata["task_relevant"].values == 1
        irr_mask = ~rel_mask

        per_epoch_power = stim_epochs.mean(axis=2).mean(axis=1)
        power_contrast = condition_contrast(per_epoch_power[rel_mask], per_epoch_power[irr_mask],
                                            "Task-Relevant", "Task-Irrelevant", n_perm=500, seed=seed)
        log(f"  HG power contrast: g={power_contrast.effect_size:.3f} p={power_contrast.p_value:.4f} [{time.time()-t0:.0f}s]")

        n_epoch_lzc_ch = min(30, n_channels)
        per_epoch_lzc = np.zeros(n_trials_total)
        for i in range(n_trials_total):
            per_epoch_lzc[i] = compute_lzc(stim_epochs[i, :n_epoch_lzc_ch, :], normalize=False, n_surrogates=0).lzc_raw
            if (i + 1) % 200 == 0:
                log(f"    LZc {i+1}/{n_trials_total} [{time.time()-t0:.0f}s]")

        lzc_contrast = condition_contrast(per_epoch_lzc[rel_mask], per_epoch_lzc[irr_mask],
                                          "Task-Relevant", "Task-Irrelevant", n_perm=500, seed=seed)
        log(f"  LZc contrast (stim-only, {n_epoch_lzc_ch}ch): g={lzc_contrast.effect_size:.3f} p={lzc_contrast.p_value:.4f} [{time.time()-t0:.0f}s]")

        categories = pooled_metadata["category"].unique()
        for cat in sorted(categories):
            cat_mask = pooled_metadata["category"].values == cat
            other_mask = ~cat_mask
            c = condition_contrast(per_epoch_lzc[cat_mask], per_epoch_lzc[other_mask],
                                   cat, "other", n_perm=200, seed=seed)
            cat_contrasts[cat] = {"g": c.effect_size, "p": c.p_value, "n": int(c.n_a), "mean": c.mean_a}
        cat_str = ', '.join(f'{k}:p={v["p"]:.3f}' for k,v in cat_contrasts.items())
        log(f"  Category contrasts: {cat_str} [{time.time()-t0:.0f}s]")

        del stim_epochs

    categories = pooled_metadata["category"].unique()
    cat_to_int = {c: i for i, c in enumerate(sorted(categories))}
    labels = np.array([cat_to_int[c] for c in pooled_metadata["category"].values])

    block_labels = pooled_metadata["global_block"].values.astype(int)
    unique_blocks = np.unique(block_labels)
    if len(unique_blocks) < 2:
        rng = np.random.default_rng(seed)
        block_labels = np.zeros(n_trials_total, dtype=int)
        idx = rng.permutation(n_trials_total)
        for b in range(5):
            start = b * (n_trials_total // 5)
            end = start + (n_trials_total // 5) if b < 4 else n_trials_total
            block_labels[idx[start:end]] = b + 1

    if qc_pass:
        tau_per_channel = {}
        for ci in range(n_channels):
            ch_avals = detect_avalanches(gamma_concatenated[ci:ci+1, :], sfreq=sfreq,
                                          threshold_sd=config["avalanche"]["threshold_sd"],
                                          bin_width_factor=config["avalanche"]["bin_width_factor"])
            ch_sizes = np.array([a.size for a in ch_avals])
            if len(ch_sizes) >= 10:
                tau_per_channel[ch_names[ci]] = _fit_single_distribution(ch_sizes, discrete=True)["exponent"]
            else:
                tau_per_channel[ch_names[ci]] = float("nan")

        valid_tau = sum(1 for v in tau_per_channel.values() if not np.isnan(v))

        n_ch_list = [n for n in [5, 10, 20] if n <= valid_tau]
        if n_ch_list and n_trials_total >= 20:
            decoding_results = criticality_decoding_analysis(
                pooled_epochs, labels, block_labels, ch_names, tau_per_channel,
                n_channels_list=n_ch_list, classifier="lda", n_random=5, seed=seed,
                target_tau=1.5, n_perm_comparison=200)

    del gamma_concatenated, binary, pooled_epochs
    gc.collect()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cmcc.viz.distributions import plot_avalanche_distributions
    from cmcc.viz.summary import plot_dfa
    from cmcc.provenance import save_summary_json

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{subject_id}_pooled"

    fig = plot_avalanche_distributions(sizes, durations, fit_result=pl_result,
                                       title=f"{subject_id} Pooled - Avalanches")
    fig.savefig(fig_dir / f"{tag}_avalanche.png", dpi=100)
    plt.close(fig)

    fig = plot_dfa(dfa_result, title=f"{subject_id} Pooled - DFA")
    fig.savefig(fig_dir / f"{tag}_dfa.png", dpi=100)
    plt.close(fig)

    summary = {
        "subject": subject_id,
        "site": site,
        "status": "OK",
        "n_runs": len(valid_runs),
        "runs": valid_runs,
        "n_channels": n_channels,
        "n_trials_total": n_trials_total,
        "line_freq": line_freq,
        "n_avalanches": len(avalanches),
        "tau": pl_result.tau,
        "tau_ks": pl_result.tau_ks_distance,
        "alpha": pl_result.alpha,
        "alpha_ks": pl_result.alpha_ks_distance,
        "gamma": pl_result.gamma,
        "gamma_predicted": pl_result.gamma_predicted,
        "lzc_normalized": lzc_result.lzc_normalized,
        "mse_ci": mse_result.complexity_index,
        "dfa_alpha": dfa_result.alpha,
        "branching_sigma": br_result.sigma,
        "qc_pass": qc_pass,
        "qc_reasons": qc_reasons if not qc_pass else [],
        "hg_power_contrast_g": power_contrast.effect_size if power_contrast else None,
        "hg_power_contrast_p": power_contrast.p_value if power_contrast else None,
        "task_contrast_g": lzc_contrast.effect_size if lzc_contrast else None,
        "task_contrast_p": lzc_contrast.p_value if lzc_contrast else None,
        "task_contrast_n_relevant": int(lzc_contrast.n_a) if lzc_contrast else None,
        "task_contrast_n_irrelevant": int(lzc_contrast.n_b) if lzc_contrast else None,
        "task_contrast_lzc_ch": n_epoch_lzc_ch if qc_pass else None,
        "task_contrast_stim_only": True,
        "cat_contrasts": cat_contrasts if cat_contrasts else None,
        "categories": pooled_metadata["category"].value_counts().to_dict(),
        "comparison_results": pl_result.comparison_results,
        "elapsed_s": time.time() - t0,
    }
    if decoding_results:
        summary["decoding"] = [
            {
                "n": c.n_channels,
                "most": c.accuracy_most_critical,
                "least": c.accuracy_least_critical,
                "rand": c.accuracy_random,
                "diff_most_least": c.diff_most_vs_least,
                "p_most_vs_least": c.p_value_most_vs_least,
            }
            for c in decoding_results["comparisons"]
        ]

    save_summary_json(summary, results_dir, f"summary_{subject_id}_pooled.json")
    return summary


def main():
    t_global = time.time()
    config = load_config(str(CONFIG_PATH))
    data_root = Path(config["data"]["root"])
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    import os
    all_subjects = sorted(set(
        d.split("_")[0]
        for d in os.listdir(str(data_root))
        if "ECOG" in d and os.path.isdir(str(data_root / d))
    ))

    log(f"{'='*70}")
    log(f"CMCC BATCH PIPELINE [BROADBAND 1-200Hz] - {len(all_subjects)} subjects")
    log(f"{'='*70}")

    all_summaries = []
    for idx, subject_id in enumerate(all_subjects):
        log(f"\n[{idx+1}/{len(all_subjects)}] {subject_id} ...")
        try:
            summary = run_single_subject(subject_id, config, results_dir)
            all_summaries.append(summary)
            status = summary.get("status", "OK")
            if status == "OK":
                log(f"  DONE: tau={summary['tau']:.3f} sigma={summary['branching_sigma']:.4f} "
                    f"LZc={summary['lzc_normalized']:.4f} ({summary['elapsed_s']:.0f}s)")
            else:
                log(f"  {status}: {summary.get('error', '')}")
        except Exception as e:
            tb = traceback.format_exc()
            log(f"  FAILED: {e}")
            log(f"  {tb[-200:]}")
            all_summaries.append({"subject": subject_id, "status": "FAILED", "error": str(e)})

        gc.collect()

    from cmcc.provenance import save_summary_json
    save_summary_json(all_summaries, results_dir, "group_all_subjects.json")

    ok = [s for s in all_summaries if s.get("status") == "OK"]
    skipped = [s for s in all_summaries if s.get("status") == "SKIP"]
    failed = [s for s in all_summaries if s.get("status") == "FAILED"]

    elapsed = time.time() - t_global
    log(f"\n{'='*70}")
    log(f"BATCH COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"{'='*70}")
    log(f"  OK: {len(ok)}, SKIP: {len(skipped)}, FAILED: {len(failed)}")

    if ok:
        taus = [s["tau"] for s in ok]
        sigmas = [s["branching_sigma"] for s in ok]
        lzcs = [s["lzc_normalized"] for s in ok]
        dfas = [s["dfa_alpha"] for s in ok]
        log(f"\n  GROUP SUMMARY ({len(ok)} subjects):")
        log(f"    tau:   mean={np.mean(taus):.3f} std={np.std(taus):.3f} range=[{np.min(taus):.3f}, {np.max(taus):.3f}]")
        log(f"    sigma: mean={np.mean(sigmas):.4f} std={np.std(sigmas):.4f}")
        log(f"    LZc:   mean={np.mean(lzcs):.4f} std={np.std(lzcs):.4f}")
        log(f"    DFA:   mean={np.mean(dfas):.4f} std={np.std(dfas):.4f}")

        n10_most = [s["decoding"][1]["most"] for s in ok if "decoding" in s and len(s["decoding"]) > 1]
        n10_least = [s["decoding"][1]["least"] for s in ok if "decoding" in s and len(s["decoding"]) > 1]
        if n10_most:
            log(f"    Decoding n=10: most={np.mean(n10_most):.3f} least={np.mean(n10_least):.3f} "
                f"diff={np.mean(np.array(n10_most)-np.array(n10_least)):+.3f}")

    if skipped:
        log(f"\n  SKIPPED:")
        for s in skipped:
            log(f"    {s['subject']}: {s.get('error', '')}")
    if failed:
        log(f"\n  FAILED:")
        for s in failed:
            log(f"    {s['subject']}: {s.get('error', '')}")


if __name__ == "__main__":
    main()
