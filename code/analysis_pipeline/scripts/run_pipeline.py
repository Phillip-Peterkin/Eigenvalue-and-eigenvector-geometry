"""CMCC Multi-Run Pipeline Runner for CE103 iEEG data.

Processes all available runs (DurR1-DurR5) for a subject, pools epochs
across runs for increased statistical power, and computes all CMCC
criticality metrics on the aggregated data.

Scientific rationale: A single run (~142 trials) provides insufficient
trials for stable LDA covariance estimation with 4 categories. Pooling
~710 trials across 5 runs yields robust decoding estimates and adequate
permutation test power for condition contrasts.
"""
from __future__ import annotations

import sys
import time
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

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
RUNS = ["DurR1", "DurR2", "DurR3", "DurR4", "DurR5"]


def log(msg):
    print(msg, flush=True)


def load_events_tsv(data_root, subject_id):
    meta_dir = Path(data_root) / f"{subject_id}_ECOG_1" / "METADATA" / "electrode_coordinates"
    return pd.read_csv(meta_dir / f"sub-{subject_id}_ses-1_task-Dur_events.tsv", sep="\t", encoding="utf-8-sig")


def parse_stimulus_events(events_df, sample_offset, n_samples):
    """Extract stimulus events for a specific run using global sample offsets.

    The events TSV uses global sample indices across all concatenated runs.
    Each run's events fall within [sample_offset, sample_offset + n_samples).
    Local sample indices are computed by subtracting the offset.
    """
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


def main():
    t0 = time.time()
    log("=" * 70)
    log("CMCC Multi-Run Pipeline - CE103 (DurR1-DurR5)")
    log("=" * 70)

    log("\n[1/10] Config...")
    config = load_config(str(CONFIG_PATH))
    subject_id = config["data"]["subjects"][0]
    seed = config["random_seed"]

    from cmcc.io.loader import load_subject
    from cmcc.preprocess.qc import detect_bad_channels, mark_bad_channels
    from cmcc.preprocess.filter import remove_line_noise, extract_high_gamma
    from cmcc.preprocess.reference import apply_laplace

    log(f"\n[2/10] Loading {subject_id} runs: {RUNS}...")
    subject_data = load_subject(subject_id, config["data"]["root"], runs=RUNS, preload_edf=True)
    events_all = load_events_tsv(config["data"]["root"], subject_id)

    available_runs = [r for r in RUNS if r in subject_data.raw]
    log(f"  Available runs: {available_runs}")

    log(f"\n[3/10] Per-run QC + preprocessing... [{time.time()-t0:.0f}s]")
    per_run_good_channels = {}
    per_run_raw = {}
    per_run_n_times = {}

    for run_id in available_runs:
        raw = subject_data.raw[run_id]
        per_run_n_times[run_id] = raw.n_times
        non_ecog = [ch for ch in raw.ch_names if ch.startswith("DC") or ch == "C128"]
        if non_ecog:
            raw.drop_channels(non_ecog)
        bad_channels = detect_bad_channels(raw)
        mark_bad_channels(raw, bad_channels)
        good_chs = [ch for ch in raw.ch_names if ch not in bad_channels]
        raw.pick(good_chs)
        raw = remove_line_noise(raw, line_freq=config["preprocessing"]["line_freq"])
        if subject_data.laplace_map:
            raw = apply_laplace(raw, subject_data.laplace_map)
        per_run_good_channels[run_id] = set(raw.ch_names)
        per_run_raw[run_id] = raw
        log(f"  {run_id}: {len(good_chs)} good ch, {len(bad_channels)} bad [{time.time()-t0:.0f}s]")

    common_channels = sorted(set.intersection(*per_run_good_channels.values()))
    log(f"  Common channels across runs: {len(common_channels)}")

    for run_id in available_runs:
        raw = per_run_raw[run_id]
        ch_to_drop = [ch for ch in raw.ch_names if ch not in common_channels]
        if ch_to_drop:
            raw.drop_channels(ch_to_drop)
        ch_order = [ch for ch in common_channels if ch in raw.ch_names]
        raw.reorder_channels(ch_order)
        per_run_raw[run_id] = raw

    log(f"\n[4/10] High-gamma extraction + epoching... [{time.time()-t0:.0f}s]")
    all_epoch_data = []
    all_metadata = []
    all_gamma_continuous = []
    block_offset = 0
    cumulative_sample_offset = 0

    for run_idx, run_id in enumerate(available_runs):
        raw = per_run_raw[run_id]
        gamma_raw = extract_high_gamma(raw, passband=tuple(config["preprocessing"]["high_gamma_passband"]))
        sfreq = gamma_raw.info["sfreq"]
        n_channels = gamma_raw.info["nchan"]

        gamma_cont = gamma_raw.get_data()
        all_gamma_continuous.append(gamma_cont)

        raw_n_times = per_run_n_times[run_id]
        stim_events = parse_stimulus_events(events_all, cumulative_sample_offset, raw_n_times)
        cumulative_sample_offset += raw_n_times

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
        log(f"  {run_id}: {n_trials} trials, {n_channels} ch, gamma {gamma_cont.shape[1]} samples [{time.time()-t0:.0f}s]")

    pooled_epochs = np.concatenate(all_epoch_data, axis=0)
    pooled_metadata = pd.concat(all_metadata, ignore_index=True)
    gamma_concatenated = np.concatenate(all_gamma_continuous, axis=1)
    ch_names = list(common_channels)
    n_trials_total, n_channels, n_timepoints = pooled_epochs.shape

    log(f"  POOLED: {n_trials_total} trials, {n_channels} ch, {gamma_concatenated.shape[1]} continuous samples")

    log(f"\n[5/10] Avalanche detection (pooled)... [{time.time()-t0:.0f}s]")
    from cmcc.features.avalanche import detect_avalanches
    from cmcc.features.powerlaw_fit import fit_avalanche_distributions

    avalanches = detect_avalanches(gamma_concatenated, sfreq=sfreq,
                                   threshold_sd=config["avalanche"]["threshold_sd"],
                                   bin_width_factor=config["avalanche"]["bin_width_factor"])
    sizes = np.array([a.size for a in avalanches])
    durations = np.array([a.duration_bins for a in avalanches])
    log(f"  {len(avalanches)} avalanches, size=[{sizes.min()},{sizes.max()}] [{time.time()-t0:.0f}s]")

    log("  Fitting power-law...")
    pl_result = fit_avalanche_distributions(sizes, durations, discrete=True, n_bootstrap=100,
                                            compare_distributions=config["powerlaw"]["compare_distributions"])
    log(f"  tau={pl_result.tau:.3f}  alpha={pl_result.alpha:.3f}  gamma={pl_result.gamma:.3f} [{time.time()-t0:.0f}s]")

    log(f"\n[6/10] Complexity (pooled)... [{time.time()-t0:.0f}s]")
    from cmcc.features.complexity import compute_lzc
    from cmcc.features.entropy import compute_mse
    from cmcc.features.dfa import compute_dfa

    n_lzc_ch = min(10, n_channels)
    n_lzc_t = min(2000, gamma_concatenated.shape[1])
    lzc_result = compute_lzc(gamma_concatenated[:n_lzc_ch, :n_lzc_t], n_surrogates=3, seed=seed)
    log(f"  LZc={lzc_result.lzc_normalized:.4f} (on {n_lzc_ch}ch x {n_lzc_t}t) [{time.time()-t0:.0f}s]")

    mean_gamma = gamma_concatenated.mean(axis=0)
    mse_scales = list(range(1, 21))
    mse_result = compute_mse(mean_gamma[:3000], scales=mse_scales, m=2, r_factor=0.15)
    log(f"  MSE CI={mse_result.complexity_index:.4f} [{time.time()-t0:.0f}s]")

    dfa_result = compute_dfa(mean_gamma[:50000])
    log(f"  DFA alpha={dfa_result.alpha:.4f} [{time.time()-t0:.0f}s]")

    log(f"\n[7/10] Branching (pooled)... [{time.time()-t0:.0f}s]")
    from cmcc.features.branching import compute_branching_ratio
    ch_m = gamma_concatenated.mean(axis=1, keepdims=True)
    ch_s = gamma_concatenated.std(axis=1, keepdims=True)
    ch_s[ch_s == 0] = 1.0
    binary = (np.abs((gamma_concatenated - ch_m) / ch_s) > config["avalanche"]["threshold_sd"]).astype(np.int8)
    br_result = compute_branching_ratio(binary)
    log(f"  sigma={br_result.sigma:.4f} [{time.time()-t0:.0f}s]")

    log(f"\n[8/10] Contrasts (pooled, {n_trials_total} trials)... [{time.time()-t0:.0f}s]")
    from cmcc.analysis.contrasts import condition_contrast

    stim_onset_idx = int(abs(config["preprocessing"]["epoch_tmin"]) * sfreq)
    stim_epochs = pooled_epochs[:, :, stim_onset_idx:]
    log(f"  Stimulus-only epochs: {stim_epochs.shape} (dropped {stim_onset_idx} baseline samples)")

    rel_mask = pooled_metadata["task_relevant"].values == 1
    irr_mask = ~rel_mask

    per_epoch_power = stim_epochs.mean(axis=2).mean(axis=1)
    power_contrast = condition_contrast(per_epoch_power[rel_mask], per_epoch_power[irr_mask],
                                        "Task-Relevant", "Task-Irrelevant", n_perm=500, seed=seed)
    log(f"  HG power contrast: g={power_contrast.effect_size:.3f}, p={power_contrast.p_value:.4f}")
    log(f"    Relevant power: n={power_contrast.n_a}, mean={power_contrast.mean_a:.6f}")
    log(f"    Irrelevant power: n={power_contrast.n_b}, mean={power_contrast.mean_b:.6f} [{time.time()-t0:.0f}s]")

    n_epoch_lzc_ch = min(30, n_channels)
    per_epoch_lzc = np.zeros(n_trials_total)
    for i in range(n_trials_total):
        per_epoch_lzc[i] = compute_lzc(stim_epochs[i, :n_epoch_lzc_ch, :], normalize=False, n_surrogates=0).lzc_raw
        if (i + 1) % 100 == 0:
            log(f"    LZc {i+1}/{n_trials_total} [{time.time()-t0:.0f}s]")
    log(f"  Per-epoch LZc done ({n_epoch_lzc_ch} ch, stim-only) [{time.time()-t0:.0f}s]")

    lzc_contrast = condition_contrast(per_epoch_lzc[rel_mask], per_epoch_lzc[irr_mask],
                                      "Task-Relevant", "Task-Irrelevant", n_perm=500, seed=seed)
    log(f"  LZc task contrast: g={lzc_contrast.effect_size:.3f}, p={lzc_contrast.p_value:.4f}")
    log(f"    Relevant: n={lzc_contrast.n_a}, mean={lzc_contrast.mean_a:.4f}")
    log(f"    Irrelevant: n={lzc_contrast.n_b}, mean={lzc_contrast.mean_b:.4f} [{time.time()-t0:.0f}s]")

    categories = pooled_metadata["category"].unique()
    cat_contrasts = {}
    for cat in sorted(categories):
        cat_mask = pooled_metadata["category"].values == cat
        other_mask = ~cat_mask
        c = condition_contrast(per_epoch_lzc[cat_mask], per_epoch_lzc[other_mask],
                               cat, "other", n_perm=200, seed=seed)
        cat_contrasts[cat] = {"g": c.effect_size, "p": c.p_value, "n": c.n_a, "mean": c.mean_a}
        log(f"    {cat} vs rest: g={c.effect_size:.3f}, p={c.p_value:.4f} (n={c.n_a})")

    dur_contrasts = {}
    for dur_val in sorted(pooled_metadata["duration_ms"].unique()):
        dur_mask = pooled_metadata["duration_ms"].values == dur_val
        if dur_mask.sum() >= 5:
            c = condition_contrast(per_epoch_lzc[dur_mask], per_epoch_lzc[~dur_mask],
                                   f"{dur_val}ms", "other", n_perm=200, seed=seed)
            dur_contrasts[int(dur_val)] = {"g": c.effect_size, "p": c.p_value, "n": c.n_a, "mean": c.mean_a}
            log(f"    {dur_val}ms vs rest: g={c.effect_size:.3f}, p={c.p_value:.4f} (n={c.n_a})")

    log(f"\n[9/10] Decoding (pooled, {n_trials_total} trials)... [{time.time()-t0:.0f}s]")
    from cmcc.analysis.decoding import criticality_decoding_analysis
    from cmcc.features.powerlaw_fit import _fit_single_distribution

    cat_to_int = {c: i for i, c in enumerate(sorted(categories))}
    labels = np.array([cat_to_int[c] for c in pooled_metadata["category"].values])

    block_labels = pooled_metadata["global_block"].values.astype(int)
    unique_blocks = np.unique(block_labels)
    log(f"  Using {len(unique_blocks)} unique blocks for leave-one-block-out CV")

    if len(unique_blocks) < 2:
        rng = np.random.default_rng(seed)
        block_labels = np.zeros(n_trials_total, dtype=int)
        idx = rng.permutation(n_trials_total)
        n_pseudo = 10
        for b in range(n_pseudo):
            start = b * (n_trials_total // n_pseudo)
            end = start + (n_trials_total // n_pseudo) if b < n_pseudo - 1 else n_trials_total
            block_labels[idx[start:end]] = b + 1
        log(f"  Created {n_pseudo} pseudo-blocks")

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
        if (ci + 1) % 20 == 0:
            log(f"    tau {ci+1}/{n_channels} [{time.time()-t0:.0f}s]")

    valid_tau = sum(1 for v in tau_per_channel.values() if not np.isnan(v))
    log(f"  Valid tau: {valid_tau}/{n_channels} [{time.time()-t0:.0f}s]")

    decoding_results = None
    n_ch_list = [n for n in [5, 10, 20] if n <= valid_tau]
    if n_ch_list:
        log(f"  Decoding with channel subsets {n_ch_list}...")
        decoding_results = criticality_decoding_analysis(
            pooled_epochs, labels, block_labels, ch_names, tau_per_channel,
            n_channels_list=n_ch_list, classifier="lda", n_random=5, seed=seed,
            target_tau=1.5, n_perm_comparison=200)
        for comp in decoding_results["comparisons"]:
            log(f"  n={comp.n_channels}: most={comp.accuracy_most_critical:.3f} "
                f"least={comp.accuracy_least_critical:.3f} rand={comp.accuracy_random:.3f} "
                f"(most-least={comp.diff_most_vs_least:.3f}, p={comp.p_value_most_vs_least:.4f})")

    log(f"\n[10/10] Output... [{time.time()-t0:.0f}s]")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cmcc.viz.distributions import plot_avalanche_distributions
    from cmcc.viz.comparisons import plot_condition_comparison
    from cmcc.viz.spatial_maps import plot_electrode_metric_map
    from cmcc.viz.summary import plot_dfa
    from cmcc.provenance import log_run, save_config_snapshot, save_summary_json

    results_dir = Path(config["output"]["results_dir"])
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{subject_id}_pooled"

    fig = plot_avalanche_distributions(sizes, durations, fit_result=pl_result,
                                       title=f"{subject_id} Pooled ({len(available_runs)} runs) - Avalanche Distributions")
    fig.savefig(fig_dir / f"{tag}_avalanche_distributions.png", dpi=150)
    plt.close(fig)

    fig = plot_dfa(dfa_result, title=f"{subject_id} Pooled - DFA")
    fig.savefig(fig_dir / f"{tag}_dfa.png", dpi=150)
    plt.close(fig)

    fig = plot_condition_comparison(
        {"Task-Relevant": per_epoch_lzc[rel_mask], "Task-Irrelevant": per_epoch_lzc[irr_mask]},
        metric_name="LZc", title=f"{subject_id} Pooled - LZc by Task Relevance")
    fig.savefig(fig_dir / f"{tag}_lzc_task_contrast.png", dpi=150)
    plt.close(fig)

    cat_data = {cat: per_epoch_lzc[pooled_metadata["category"].values == cat] for cat in sorted(categories)}
    fig = plot_condition_comparison(cat_data, metric_name="LZc",
                                    title=f"{subject_id} Pooled - LZc by Category")
    fig.savefig(fig_dir / f"{tag}_lzc_category_contrast.png", dpi=150)
    plt.close(fig)

    fig = plot_electrode_metric_map(subject_data.electrodes, tau_per_channel,
                                     metric_name="tau", title=f"{subject_id} Pooled - tau Spatial Map")
    fig.savefig(fig_dir / f"{tag}_tau_spatial.png", dpi=150)
    plt.close(fig)
    log(f"  Figures saved [{time.time()-t0:.0f}s]")

    prov = log_run(config, results_dir)
    save_config_snapshot(config, results_dir)

    summary = {
        "subject": subject_id,
        "runs": available_runs,
        "n_runs": len(available_runs),
        "n_channels": n_channels,
        "n_trials_total": n_trials_total,
        "trials_per_run": {r: len(m) for r, m in zip(available_runs, all_metadata)},
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
        "hg_power_contrast_g": power_contrast.effect_size,
        "hg_power_contrast_p": power_contrast.p_value,
        "task_contrast_g": lzc_contrast.effect_size,
        "task_contrast_p": lzc_contrast.p_value,
        "task_contrast_n_relevant": int(lzc_contrast.n_a),
        "task_contrast_n_irrelevant": int(lzc_contrast.n_b),
        "task_contrast_lzc_ch": n_epoch_lzc_ch,
        "task_contrast_stim_only": True,
        "categories": pooled_metadata["category"].value_counts().to_dict(),
        "category_contrasts": cat_contrasts,
        "duration_contrasts": dur_contrasts,
        "comparison_results": pl_result.comparison_results,
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
                "diff_most_rand": c.diff_most_vs_random,
                "p_most_vs_rand": c.p_value_most_vs_random,
            }
            for c in decoding_results["comparisons"]
        ]

    json_path = save_summary_json(summary, results_dir, f"summary_{subject_id}_pooled.json")
    log(f"  Summary: {json_path}")

    elapsed = time.time() - t0
    log(f"\n{'='*70}")
    log(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"{'='*70}")
    log(f"\nKEY RESULTS ({len(available_runs)} runs, {n_trials_total} trials):")
    log(f"  Avalanches:    {len(avalanches)}")
    log(f"  tau:           {pl_result.tau:.3f} (critical=1.5, |dev|={abs(pl_result.tau-1.5):.3f})")
    log(f"  alpha:         {pl_result.alpha:.3f}")
    log(f"  gamma:         {pl_result.gamma:.3f} (pred={pl_result.gamma_predicted:.3f})")
    log(f"  sigma:         {br_result.sigma:.4f} (critical=1.0)")
    log(f"  LZc:           {lzc_result.lzc_normalized:.4f}")
    log(f"  DFA:           {dfa_result.alpha:.4f}")
    log(f"  MSE CI:        {mse_result.complexity_index:.4f}")
    log(f"  HG power:      g={power_contrast.effect_size:.3f}, p={power_contrast.p_value:.4f}")
    log(f"  LZc contrast:  g={lzc_contrast.effect_size:.3f}, p={lzc_contrast.p_value:.4f} (stim-only, {n_epoch_lzc_ch}ch)")
    if decoding_results:
        log(f"\n  DECODING (most-critical vs least-critical):")
        for comp in decoding_results["comparisons"]:
            log(f"    n={comp.n_channels}: most={comp.accuracy_most_critical:.3f} "
                f"least={comp.accuracy_least_critical:.3f} diff={comp.diff_most_vs_least:+.3f} "
                f"p={comp.p_value_most_vs_least:.4f}")


if __name__ == "__main__":
    main()
