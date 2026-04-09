"""CMCC Cross-Cohort Generalization Pipeline — ds004752 (Zurich SEEG).

Cross-dataset generalization test of criticality and exceptional-point
dynamics on an independent iEEG dataset. ds004752 contains SEEG depth
electrode recordings from 15 epilepsy patients performing a verbal
working memory (Sternberg) task at Schweizerische Epilepsie-Klinik,
Zurich (Dimakopoulos et al., eLife 2022).

Scientific rationale
--------------------
The CMCC theory was developed and tested on the Cogitate iEEG dataset
(visual consciousness paradigm, ECoG grids/strips, CE/CF/CG sites).
This script tests whether the same criticality signatures — power-law
avalanches, branching ratio near unity, high LZc, DFA alpha near 1,
and EP/Jacobian dynamics — are present in a completely independent
dataset recorded with:
  - Different electrodes (SEEG depth vs ECoG surface)
  - Different paradigm (verbal working memory vs visual consciousness)
  - Different lab (Zurich vs Cogitate consortium)
  - Different patient population (hippocampal/temporal vs broader)

This is a strict out-of-sample generalization test. If the theory's
predictions hold here, it strengthens the claim that these dynamical
signatures are general properties of cortical networks, not artifacts
of a specific recording setup or paradigm.

Key differences from Cogitate pipeline
---------------------------------------
- BIDS layout (sub-XX/ses-XX/ieeg/) vs Cogitate custom layout
- No Laplace re-referencing (SEEG geometry differs from ECoG grids)
- 50 Hz line noise (European site)
- Condition contrasts: SetSize (4/6/8) and Match (IN/OUT) instead
  of visual category and task relevance
- Sampling rates: 2000 or 4096 Hz (vs 1024-2048 Hz in Cogitate)
- Multiple sessions (2-8 per subject) instead of DurR1-5 runs
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from cmcc.config import load_config

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "ds004752.yaml"


def log(msg):
    print(msg, flush=True)


def run_single_subject(subject_id, config, results_dir):
    """Run the full CMCC pipeline for a single ds004752 subject."""
    t0 = time.time()
    seed = config["random_seed"]
    line_freq = config["preprocessing"]["line_freq"]

    from cmcc.io.loader_bids_ieeg import (
        load_bids_ieeg_subject,
        get_neural_channels,
    )
    from cmcc.preprocess.qc import detect_bad_channels, mark_bad_channels
    from cmcc.preprocess.filter import remove_line_noise, extract_high_gamma

    subject_data = load_bids_ieeg_subject(
        subject_id, config["data"]["root"], preload=True,
    )
    available_sessions = subject_data.sessions
    if not available_sessions:
        return {"subject": subject_id, "status": "SKIP", "error": "no sessions found"}
    log(f"  Sessions: {available_sessions}")

    per_ses_good_channels = {}
    per_ses_raw = {}
    per_ses_events = {}

    for ses in available_sessions:
        raw = subject_data.raw[ses]

        ch_info = subject_data.channels_info.get(ses)
        if ch_info is not None:
            neural_chs = get_neural_channels(ch_info)
            non_neural = [ch for ch in raw.ch_names if ch not in neural_chs]
        else:
            non_neural = [
                ch for ch in raw.ch_names
                if ch.startswith("DC") or ch.startswith("EKG")
                or ch.startswith("EMG") or ch.startswith("ECG")
            ]
        if non_neural:
            raw.drop_channels(non_neural)

        if len(raw.ch_names) < 5:
            log(f"    {ses}: only {len(raw.ch_names)} neural channels, skipping")
            continue

        bad_channels = detect_bad_channels(raw)
        mark_bad_channels(raw, bad_channels)
        good_chs = [ch for ch in raw.ch_names if ch not in bad_channels]
        if len(good_chs) < 5:
            log(f"    {ses}: only {len(good_chs)} good channels, skipping")
            continue
        raw.pick(good_chs)

        raw = remove_line_noise(raw, line_freq=line_freq)

        per_ses_good_channels[ses] = set(raw.ch_names)
        per_ses_raw[ses] = raw
        per_ses_events[ses] = subject_data.events.get(ses, pd.DataFrame())

    valid_sessions = list(per_ses_raw.keys())
    if len(valid_sessions) < 1:
        return {"subject": subject_id, "status": "SKIP", "error": "no valid sessions after QC"}

    common_channels = sorted(set.intersection(*[per_ses_good_channels[s] for s in valid_sessions]))
    if len(common_channels) < 5:
        return {"subject": subject_id, "status": "SKIP", "error": f"only {len(common_channels)} common channels"}
    log(f"  Common channels: {len(common_channels)} [{time.time()-t0:.0f}s]")

    for ses in valid_sessions:
        raw = per_ses_raw[ses]
        ch_to_drop = [ch for ch in raw.ch_names if ch not in common_channels]
        if ch_to_drop:
            raw.drop_channels(ch_to_drop)
        ch_order = [ch for ch in common_channels if ch in raw.ch_names]
        raw.reorder_channels(ch_order)
        per_ses_raw[ses] = raw

    all_epoch_data = []
    all_metadata = []
    all_gamma_continuous = []
    trial_offset = 0
    sfreq = None

    for ses_idx, ses in enumerate(valid_sessions):
        raw = per_ses_raw[ses]
        gamma_raw = extract_high_gamma(
            raw, passband=tuple(config["preprocessing"]["high_gamma_passband"]),
        )
        sfreq = gamma_raw.info["sfreq"]
        n_channels = gamma_raw.info["nchan"]

        gamma_cont = gamma_raw.get_data()
        all_gamma_continuous.append(gamma_cont)

        events_df = per_ses_events[ses]
        if events_df.empty or "begSample" not in events_df.columns:
            log(f"    {ses}: no valid events, using continuous data only")
            del gamma_raw
            gc.collect()
            continue

        artifact_col = "Artifact" if "Artifact" in events_df.columns else None
        if artifact_col:
            clean_events = events_df[events_df[artifact_col] == 0].copy()
        else:
            clean_events = events_df.copy()

        if len(clean_events) == 0:
            log(f"    {ses}: no clean trials")
            del gamma_raw
            gc.collect()
            continue

        sample_indices = clean_events["begSample"].values.astype(int) - 1
        events_array = np.column_stack([
            sample_indices,
            np.zeros(len(sample_indices), dtype=int),
            np.ones(len(sample_indices), dtype=int),
        ])

        metadata = pd.DataFrame({
            "trial": np.arange(len(clean_events)) + trial_offset,
            "set_size": clean_events["SetSize"].values,
            "match": clean_events["Match"].values,
            "correct": clean_events["Correct"].values,
            "session": ses,
            "session_idx": ses_idx,
            "global_trial": np.arange(len(clean_events)) + trial_offset,
        })

        baseline = tuple(config["preprocessing"]["baseline"]) if config["preprocessing"]["baseline"] else None
        epochs = mne.Epochs(
            gamma_raw, events=events_array, event_id={"trial": 1},
            tmin=config["preprocessing"]["epoch_tmin"],
            tmax=config["preprocessing"]["epoch_tmax"],
            baseline=baseline, metadata=metadata, preload=True, verbose=False,
        )
        epoch_data = epochs.get_data(copy=True)
        n_trials = epoch_data.shape[0]

        trial_offset += n_trials

        all_epoch_data.append(epoch_data)
        all_metadata.append(metadata.iloc[:n_trials].reset_index(drop=True))
        log(f"    {ses}: {n_trials} trials, {n_channels} ch [{time.time()-t0:.0f}s]")
        del gamma_raw, epochs, epoch_data
        gc.collect()

    if not all_gamma_continuous:
        return {"subject": subject_id, "status": "SKIP", "error": "no continuous data"}

    gamma_concatenated = np.concatenate(all_gamma_continuous, axis=1)
    ch_names = list(common_channels)
    n_channels = len(common_channels)

    has_epochs = len(all_epoch_data) > 0
    pooled_epochs = np.concatenate(all_epoch_data, axis=0) if has_epochs else None
    pooled_metadata = pd.concat(all_metadata, ignore_index=True) if has_epochs else None
    n_trials_total = pooled_epochs.shape[0] if has_epochs else 0

    del all_epoch_data, all_gamma_continuous, per_ses_raw, subject_data
    gc.collect()

    log(f"  Pooled: {n_trials_total} trials, {n_channels} ch, {gamma_concatenated.shape[1]} continuous samples [{time.time()-t0:.0f}s]")

    from cmcc.features.avalanche import detect_avalanches
    from cmcc.features.powerlaw_fit import fit_avalanche_distributions, _fit_single_distribution
    from cmcc.features.complexity import compute_lzc
    from cmcc.features.entropy import compute_mse
    from cmcc.features.dfa import compute_dfa
    from cmcc.features.branching import compute_branching_ratio
    from cmcc.analysis.contrasts import condition_contrast
    from cmcc.analysis.decoding import criticality_decoding_analysis

    avalanches = detect_avalanches(
        gamma_concatenated, sfreq=sfreq,
        threshold_sd=config["avalanche"]["threshold_sd"],
        bin_width_factor=config["avalanche"]["bin_width_factor"],
    )
    sizes = np.array([a.size for a in avalanches])
    durations = np.array([a.duration_bins for a in avalanches])

    if len(avalanches) < 10:
        return {"subject": subject_id, "status": "SKIP", "error": f"only {len(avalanches)} avalanches"}

    pl_result = fit_avalanche_distributions(
        sizes, durations, discrete=True, n_bootstrap=100,
        compare_distributions=config["powerlaw"]["compare_distributions"],
    )
    log(f"  tau={pl_result.tau:.3f} alpha={pl_result.alpha:.3f} gamma={pl_result.gamma:.3f} [{time.time()-t0:.0f}s]")

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
    if n_trials_total < 20:
        qc_pass = False
        qc_reasons.append(f"too few trials ({n_trials_total}<20)")
    if n_channels < 10:
        qc_pass = False
        qc_reasons.append(f"too few channels ({n_channels}<10)")

    setsize_contrast = None
    match_contrast = None
    lzc_contrast_setsize = None
    decoding_results = None
    tau_per_channel = {}
    valid_tau = 0
    n_epoch_lzc_ch = 0

    if not qc_pass:
        log(f"  QC GATE FAILED — skipping contrasts+decoding: {'; '.join(qc_reasons)} [{time.time()-t0:.0f}s]")
    elif not has_epochs:
        log(f"  No epochs available — skipping contrasts+decoding [{time.time()-t0:.0f}s]")
        qc_pass = False
        qc_reasons.append("no epochs")
    else:
        log(f"  QC GATE PASSED — running contrasts+decoding [{time.time()-t0:.0f}s]")

        stim_onset_idx = int(abs(config["preprocessing"]["epoch_tmin"]) * sfreq)
        stim_epochs = pooled_epochs[:, :, stim_onset_idx:]

        per_epoch_power = stim_epochs.mean(axis=2).mean(axis=1)

        high_load_mask = pooled_metadata["set_size"].values == 8
        low_load_mask = pooled_metadata["set_size"].values == 4
        if high_load_mask.sum() >= 5 and low_load_mask.sum() >= 5:
            setsize_contrast = condition_contrast(
                per_epoch_power[high_load_mask], per_epoch_power[low_load_mask],
                "SetSize-8", "SetSize-4", n_perm=500, seed=seed,
            )
            log(f"  HG power SetSize 8v4: g={setsize_contrast.effect_size:.3f} p={setsize_contrast.p_value:.4f} [{time.time()-t0:.0f}s]")

        match_in = pooled_metadata["match"].values == "IN"
        match_out = pooled_metadata["match"].values == "OUT"
        if match_in.sum() >= 5 and match_out.sum() >= 5:
            match_contrast = condition_contrast(
                per_epoch_power[match_in], per_epoch_power[match_out],
                "Match-IN", "Match-OUT", n_perm=500, seed=seed,
            )
            log(f"  HG power Match INvOUT: g={match_contrast.effect_size:.3f} p={match_contrast.p_value:.4f} [{time.time()-t0:.0f}s]")

        n_epoch_lzc_ch = min(30, n_channels)
        per_epoch_lzc = np.zeros(n_trials_total)
        for i in range(n_trials_total):
            per_epoch_lzc[i] = compute_lzc(stim_epochs[i, :n_epoch_lzc_ch, :], normalize=False, n_surrogates=0).lzc_raw
            if (i + 1) % 200 == 0:
                log(f"    LZc {i+1}/{n_trials_total} [{time.time()-t0:.0f}s]")

        if high_load_mask.sum() >= 5 and low_load_mask.sum() >= 5:
            lzc_contrast_setsize = condition_contrast(
                per_epoch_lzc[high_load_mask], per_epoch_lzc[low_load_mask],
                "SetSize-8", "SetSize-4", n_perm=500, seed=seed,
            )
            log(f"  LZc SetSize 8v4: g={lzc_contrast_setsize.effect_size:.3f} p={lzc_contrast_setsize.p_value:.4f} [{time.time()-t0:.0f}s]")

        del stim_epochs

    if qc_pass and has_epochs:
        tau_per_channel = {}
        for ci in range(n_channels):
            ch_avals = detect_avalanches(
                gamma_concatenated[ci:ci+1, :], sfreq=sfreq,
                threshold_sd=config["avalanche"]["threshold_sd"],
                bin_width_factor=config["avalanche"]["bin_width_factor"],
            )
            ch_sizes = np.array([a.size for a in ch_avals])
            if len(ch_sizes) >= 10:
                tau_per_channel[ch_names[ci]] = _fit_single_distribution(ch_sizes, discrete=True)["exponent"]
            else:
                tau_per_channel[ch_names[ci]] = float("nan")

        valid_tau = sum(1 for v in tau_per_channel.values() if not np.isnan(v))

        set_sizes = pooled_metadata["set_size"].unique()
        ss_to_int = {s: i for i, s in enumerate(sorted(set_sizes))}
        labels = np.array([ss_to_int[s] for s in pooled_metadata["set_size"].values])

        session_labels = pooled_metadata["session_idx"].values.astype(int)
        unique_sessions = np.unique(session_labels)
        if len(unique_sessions) < 2:
            rng = np.random.default_rng(seed)
            session_labels = np.zeros(n_trials_total, dtype=int)
            idx = rng.permutation(n_trials_total)
            for b in range(5):
                start = b * (n_trials_total // 5)
                end = start + (n_trials_total // 5) if b < 4 else n_trials_total
                session_labels[idx[start:end]] = b + 1

        n_ch_list = [n for n in [5, 10, 20] if n <= valid_tau]
        if n_ch_list and n_trials_total >= 20:
            decoding_results = criticality_decoding_analysis(
                pooled_epochs, labels, session_labels, ch_names, tau_per_channel,
                n_channels_list=n_ch_list, classifier="lda", n_random=5, seed=seed,
                target_tau=1.5, n_perm_comparison=200,
            )

    del gamma_concatenated, binary
    if pooled_epochs is not None:
        del pooled_epochs
    gc.collect()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cmcc.viz.distributions import plot_avalanche_distributions
    from cmcc.viz.summary import plot_dfa
    from cmcc.provenance import save_summary_json

    sub_results_dir = results_dir / "per_subject"
    sub_results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{subject_id}_pooled"

    fig = plot_avalanche_distributions(
        sizes, durations, fit_result=pl_result,
        title=f"{subject_id} Pooled - Avalanches",
    )
    fig.savefig(fig_dir / f"{tag}_avalanche.png", dpi=100)
    plt.close(fig)

    fig = plot_dfa(dfa_result, title=f"{subject_id} Pooled - DFA")
    fig.savefig(fig_dir / f"{tag}_dfa.png", dpi=100)
    plt.close(fig)

    summary = {
        "subject": subject_id,
        "dataset": "ds004752",
        "status": "OK",
        "n_sessions": len(valid_sessions),
        "sessions": valid_sessions,
        "n_channels": n_channels,
        "n_trials_total": n_trials_total,
        "line_freq": line_freq,
        "sfreq": sfreq,
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
        "setsize_contrast_g": setsize_contrast.effect_size if setsize_contrast else None,
        "setsize_contrast_p": setsize_contrast.p_value if setsize_contrast else None,
        "match_contrast_g": match_contrast.effect_size if match_contrast else None,
        "match_contrast_p": match_contrast.p_value if match_contrast else None,
        "lzc_setsize_contrast_g": lzc_contrast_setsize.effect_size if lzc_contrast_setsize else None,
        "lzc_setsize_contrast_p": lzc_contrast_setsize.p_value if lzc_contrast_setsize else None,
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

    save_summary_json(summary, sub_results_dir, f"summary_{subject_id}_pooled.json")
    return summary


def main():
    t_global = time.time()
    config = load_config(str(CONFIG_PATH))
    results_dir = Path(config["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    subjects = config["data"]["subjects"]

    log(f"{'='*70}")
    log(f"CMCC INDEPENDENT REPLICATION — ds004752 (Zurich SEEG)")
    log(f"  {len(subjects)} subjects, line_freq={config['preprocessing']['line_freq']} Hz")
    log(f"  HG passband: {config['preprocessing']['high_gamma_passband']}")
    log(f"  Results: {results_dir}")
    log(f"{'='*70}")

    all_summaries = []
    for idx, subject_id in enumerate(subjects):
        log(f"\n[{idx+1}/{len(subjects)}] {subject_id} ...")
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

    from cmcc.provenance import save_summary_json, log_run
    save_summary_json(all_summaries, results_dir, "group_all_subjects.json")
    log_run(config, results_dir / "provenance")

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
        log(f"    (Cogitate reference: tau~1.5, sigma~1.0)")

        if any("decoding" in s and s["decoding"] for s in ok):
            n10_most = [s["decoding"][1]["most"] for s in ok if "decoding" in s and s.get("decoding") and len(s["decoding"]) > 1]
            n10_least = [s["decoding"][1]["least"] for s in ok if "decoding" in s and s.get("decoding") and len(s["decoding"]) > 1]
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
