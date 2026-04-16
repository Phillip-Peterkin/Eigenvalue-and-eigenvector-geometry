"""Temporal Precedence Analysis — Test 7 of Geometry Brain-State Battery.

Tests whether operator geometry (eigenvalue gap, condition number, ND score,
spectral radius) changes *before* sleep state transitions, not just after.

Dataset: ANPHY-Sleep — 10 subjects, continuous full-night EEG, 30s epoch staging.
Transition types: N2→N3 (descent into deep sleep), N2→REM (REM onset).

Analysis window: −120s to +60s around each transition.
Pipeline: notch → bandpass → CSD → PCA → VAR(1) sliding windows → geometry metrics.
PCA fitted per-segment to avoid leakage across transition boundary.

Unit of observation: per-subject mean timecourse (averaged across transitions).
Group statistics: one-sample t-test on subject-level slopes, paired t-test on
early vs late pre-transition means.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

import mne
from cmcc.preprocess.scalp_eeg import apply_csd, pca_reduce
from cmcc.preprocess.qc import detect_bad_channels
from cmcc.analysis.temporal_precedence import (
    TransitionEvent,
    TransitionTimecourse,
    TemporalPrecedenceResult,
    parse_sleep_staging,
    find_transitions,
    compute_transition_geometry,
    analyze_temporal_precedence,
)

PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
RESULTS_JSON = REPO_ROOT / "results" / "json_results"
FIG_DIR = REPO_ROOT / "results" / "figures" / "temporal_precedence"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(os.environ.get("SLEEP_DATA_ROOT", str(REPO_ROOT / "ANPHY-Sleep")))

SEED = 42
N_BOOTSTRAP = 5000
DPI = 300

WINDOW_SEC = 0.5
STEP_SEC = 0.1
N_COMPONENTS = 15
DOWNSAMPLE_TO = 500.0
LINE_FREQ = 50.0
PRE_SEC = 120.0
POST_SEC = 60.0
MIN_PRE_EPOCHS = 2
MIN_POST_EPOCHS = 2

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

TRANSITION_TYPES = [
    ("N2", "N3"),
    ("N2", "R"),
]

GEOMETRY_METRICS = ["eigenvalue_gap", "condition_number", "nd_score", "spectral_radius"]

STATE_PALETTE = {
    "N2_to_N3": "#BB5566",
    "N2_to_R": "#4477AA",
}


def log(msg):
    print(msg, flush=True)


def default_ser(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def load_raw_eeg(edf_path):
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
    raw.load_data()

    bad = detect_bad_channels(raw)
    raw.info["bads"] = bad
    if bad:
        raw.interpolate_bads(reset_bads=True)

    freqs = [LINE_FREQ * i for i in range(1, 4)]
    raw.notch_filter(freqs, verbose=False)
    raw.filter(l_freq=0.5, h_freq=45.0, verbose=False)

    if raw.info["sfreq"] > DOWNSAMPLE_TO:
        raw.resample(DOWNSAMPLE_TO, verbose=False)

    raw = apply_csd(raw)
    return raw


def preprocess_transition_segment(raw, onset_sec, pre_sec, post_sec):
    start_sec = max(0, onset_sec - pre_sec)
    end_sec = min(raw.times[-1], onset_sec + post_sec)

    raw_seg = raw.copy().crop(tmin=start_sec, tmax=end_sec)
    data = raw_seg.get_data()
    data_pca, _ = pca_reduce(data, n_components=N_COMPONENTS, return_pca=True)
    sfreq = raw_seg.info["sfreq"]
    return data_pca, sfreq, start_sec


def process_subject(subject_id, edf_path, staging_path):
    log(f"\n  {subject_id}")
    epochs = parse_sleep_staging(str(staging_path))
    if not epochs:
        log(f"    No staging data")
        return {}

    all_timecourses = {}

    subject_transitions = {}
    for pre_state, post_state in TRANSITION_TYPES:
        ttype = f"{pre_state}_to_{post_state}"
        trans = find_transitions(epochs, pre_state, post_state,
                                 min_pre_epochs=MIN_PRE_EPOCHS,
                                 min_post_epochs=MIN_POST_EPOCHS)
        if trans:
            subject_transitions[ttype] = trans
            log(f"    {ttype}: {len(trans)} transitions found")
        else:
            log(f"    {ttype}: no qualifying transitions")

    if not subject_transitions:
        return {}

    log(f"    Loading EDF...")
    t0 = time.time()
    try:
        raw = load_raw_eeg(edf_path)
    except Exception as e:
        log(f"    ERROR loading EDF: {e}")
        return {}
    log(f"    EDF loaded in {time.time()-t0:.0f}s ({raw.n_times/raw.info['sfreq']:.0f}s recording)")

    for ttype, trans_list in subject_transitions.items():
        tcs = []
        for idx, tr in enumerate(trans_list):
            onset = tr["onset_sec"]
            try:
                data_pca, sfreq, actual_start = preprocess_transition_segment(
                    raw, onset, PRE_SEC, POST_SEC,
                )

                tc = compute_transition_geometry(
                    data_pca, sfreq,
                    transition_onset_sec=onset - actual_start,
                    pre_sec=PRE_SEC, post_sec=POST_SEC,
                    window_sec=WINDOW_SEC, step_sec=STEP_SEC,
                    seed=SEED,
                )

                if tc is not None:
                    tc.subject = subject_id
                    tc.transition_type = ttype
                    tc.transition_index = idx
                    tcs.append(tc)
                    log(f"    {ttype}[{idx}]: {len(tc.time_sec)} windows, "
                        f"t=[{tc.time_sec[0]:.1f}, {tc.time_sec[-1]:.1f}]s")
                else:
                    log(f"    {ttype}[{idx}]: segment too short, skipped")
            except Exception as e:
                log(f"    {ttype}[{idx}]: ERROR — {e}")

        if tcs:
            all_timecourses[ttype] = tcs

    del raw
    gc.collect()
    return all_timecourses


def plot_group_trajectory(result, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    t = result.group_time_axis
    m = result.group_mean_trajectory
    lo = result.group_ci_lower
    hi = result.group_ci_upper

    valid = np.isfinite(m)
    if valid.sum() < 5:
        plt.close()
        return

    ax.plot(t[valid], m[valid], color="#4477AA", linewidth=2)
    ax.fill_between(t[valid], lo[valid], hi[valid], color="#4477AA", alpha=0.2)
    ax.axvline(0, color="red", linestyle="--", alpha=0.7, label="Transition")
    ax.axvspan(-30, 0, color="orange", alpha=0.1, label="Late pre-transition")
    ax.axvspan(-120, -60, color="green", alpha=0.05, label="Early baseline")

    ax.set_xlabel("Time relative to transition (s)")
    ax.set_ylabel(result.metric_name.replace("_", " ").title())
    ax.set_title(
        f"{result.transition_type}: {result.metric_name}\n"
        f"slope={result.mean_slope:.2e} (p={result.slope_p_value:.4f}), "
        f"consistency={result.subject_consistency:.0%}, "
        f"d(early-late)={result.early_vs_late_d:.2f}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def plot_slope_forest(result, output_path):
    slopes = result.per_subject_slopes
    if not slopes:
        return

    valid_slopes = [(i, s) for i, s in enumerate(slopes) if np.isfinite(s)]
    if not valid_slopes:
        return

    fig, ax = plt.subplots(figsize=(6, max(4, len(valid_slopes) * 0.4)))
    indices, vals = zip(*valid_slopes)
    y_pos = range(len(vals))

    colors = ["#228833" if np.sign(v) == np.sign(result.mean_slope) else "#BB5566"
              for v in vals]
    ax.barh(y_pos, vals, color=colors, alpha=0.7, edgecolor="white")
    ax.axvline(0, color="gray", linestyle="-", alpha=0.5)
    ax.axvline(result.mean_slope, color="black", linestyle="--", linewidth=2,
               label=f"Mean={result.mean_slope:.2e}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"S{i}" for i in indices], fontsize=8)
    ax.set_xlabel("Pre-transition slope")
    ax.set_title(
        f"{result.transition_type}: {result.metric_name} slopes\n"
        f"Consistency: {result.subject_consistency:.0%}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()


def main():
    log("=" * 70)
    log("TEMPORAL PRECEDENCE ANALYSIS — Test 7")
    log(f"Data: {DATA_ROOT}")
    log(f"Transitions: {[f'{p}->{q}' for p, q in TRANSITION_TYPES]}")
    log(f"Window: -{PRE_SEC}s to +{POST_SEC}s around transition")
    log(f"Parameters: VAR window={WINDOW_SEC}s, step={STEP_SEC}s, PCA={N_COMPONENTS}")
    log(f"Seed: {SEED}")
    log("=" * 70)

    if not DATA_ROOT.exists():
        log(f"\nERROR: Data directory not found: {DATA_ROOT}")
        log("Set SLEEP_DATA_ROOT environment variable or place data in expected location.")
        log("Writing empty results file.")
        out = {
            "analysis": "temporal_precedence",
            "status": "DATA_NOT_FOUND",
            "data_root": str(DATA_ROOT),
        }
        out_path = RESULTS_JSON / "temporal_precedence.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        return

    t_start = time.time()

    all_subject_tcs = {}
    for subject_id, (edf_name, staging_name) in SUBJECTS.items():
        edf_path = DATA_ROOT / subject_id / edf_name
        staging_path = DATA_ROOT / subject_id / staging_name

        if not edf_path.exists():
            log(f"\n  {subject_id}: EDF not found — SKIP")
            continue
        if not staging_path.exists():
            log(f"\n  {subject_id}: staging not found — SKIP")
            continue

        tcs = process_subject(subject_id, edf_path, staging_path)
        if tcs:
            all_subject_tcs[subject_id] = tcs

    log(f"\n{'='*70}")
    log(f"PROCESSING COMPLETE: {len(all_subject_tcs)} subjects processed "
        f"in {time.time()-t_start:.0f}s")
    log(f"{'='*70}")

    results = {}
    for pre_state, post_state in TRANSITION_TYPES:
        ttype = f"{pre_state}_to_{post_state}"
        log(f"\n--- Analyzing {ttype} ---")

        by_subject = {}
        for subj, subj_tcs in all_subject_tcs.items():
            if ttype in subj_tcs:
                by_subject[subj] = subj_tcs[ttype]

        if len(by_subject) < 3:
            log(f"  Only {len(by_subject)} subjects — skipping")
            continue

        n_trans = sum(len(tcs) for tcs in by_subject.values())
        log(f"  {len(by_subject)} subjects, {n_trans} transitions total")

        transition_results = {}
        for metric in GEOMETRY_METRICS:
            log(f"\n  Metric: {metric}")
            result = analyze_temporal_precedence(
                by_subject, metric, ttype,
                pre_sec=PRE_SEC, post_sec=POST_SEC,
                seed=SEED, n_bootstrap=N_BOOTSTRAP,
            )
            transition_results[metric] = result

            status = "PASS" if result.passes_threshold else "FAIL"
            strict = "PASS" if result.passes_strict else "FAIL"
            log(f"    Slope: {result.mean_slope:.4e} (p={result.slope_p_value:.4f})")
            log(f"    Slope CI: [{result.slope_ci[0]:.4e}, {result.slope_ci[1]:.4e}]")
            log(f"    Early vs Late d: {result.early_vs_late_d:.3f} (p={result.early_vs_late_p:.4f})")
            log(f"    Subject consistency: {result.subject_consistency:.0%}")
            log(f"    Non-overlapping slope: {result.nonoverlap_slope:.4e} (p={result.nonoverlap_slope_p:.4f})")
            log(f"    Non-overlap survives: {result.nonoverlap_survives}")
            log(f"    Verdict (lenient): {status}")
            log(f"    Verdict (strict):  {strict}")

            fig_name = f"trajectory_{ttype}_{metric}.png"
            plot_group_trajectory(result, FIG_DIR / fig_name)

            fig_name = f"slopes_{ttype}_{metric}.png"
            plot_slope_forest(result, FIG_DIR / fig_name)

        results[ttype] = transition_results

    any_passes = False
    any_passes_strict = False
    for ttype, metrics in results.items():
        for metric, result in metrics.items():
            if result.passes_threshold:
                any_passes = True
            if result.passes_strict:
                any_passes_strict = True

    log(f"\n{'='*70}")
    log(f"TEST 7 TEMPORAL PRECEDENCE (lenient): {'PASS' if any_passes else 'FAIL'}")
    log(f"TEST 7 TEMPORAL PRECEDENCE (strict):  {'PASS' if any_passes_strict else 'FAIL'}")
    if any_passes and not any_passes_strict:
        log("  WARNING: Lenient pass but strict fail — some results may be")
        log("  inflated by temporal autocorrelation from 80% window overlap.")
        log("  Only metrics surviving non-overlapping sensitivity are reliable.")
    log(f"{'='*70}")

    output = {
        "analysis": "temporal_precedence",
        "dataset": "ANPHY-Sleep",
        "parameters": {
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "n_components": N_COMPONENTS,
            "pre_sec": PRE_SEC,
            "post_sec": POST_SEC,
            "min_pre_epochs": MIN_PRE_EPOCHS,
            "min_post_epochs": MIN_POST_EPOCHS,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
        },
        "transitions": {},
        "overall_passes": any_passes,
        "overall_passes_strict": any_passes_strict,
        "elapsed_sec": time.time() - t_start,
    }

    window_timecourses = {}
    for ttype in results:
        by_subject = {}
        for subj, subj_tcs in all_subject_tcs.items():
            if ttype in subj_tcs:
                by_subject[subj] = subj_tcs[ttype]
        if by_subject:
            ttype_tc = {}
            for metric in GEOMETRY_METRICS:
                per_subj = {}
                time_step = 0.5
                common_time = np.arange(-PRE_SEC, POST_SEC + time_step, time_step)
                for subj, tcs in sorted(by_subject.items()):
                    subj_vals = []
                    for tc in tcs:
                        raw_time = tc.time_sec
                        raw_vals = getattr(tc, metric, None)
                        if raw_vals is None:
                            continue
                        interp = np.interp(common_time, raw_time, raw_vals,
                                           left=np.nan, right=np.nan)
                        subj_vals.append(interp)
                    if subj_vals:
                        per_subj[subj] = np.nanmean(subj_vals, axis=0).tolist()
                ttype_tc[metric] = {
                    "common_time_sec": common_time.tolist(),
                    "per_subject_mean": per_subj,
                }
            window_timecourses[ttype] = ttype_tc

    for ttype, metrics in results.items():
        output["transitions"][ttype] = {}
        for metric, result in metrics.items():
            r = {
                "metric_name": result.metric_name,
                "transition_type": result.transition_type,
                "n_subjects": result.n_subjects,
                "n_transitions_total": result.n_transitions_total,
                "per_subject_slopes": result.per_subject_slopes,
                "mean_slope": result.mean_slope,
                "slope_p_value": result.slope_p_value,
                "slope_ci": list(result.slope_ci),
                "early_vs_late_d": result.early_vs_late_d,
                "early_vs_late_p": result.early_vs_late_p,
                "subject_consistency": result.subject_consistency,
                "nonoverlap_slope": result.nonoverlap_slope,
                "nonoverlap_slope_p": result.nonoverlap_slope_p,
                "nonoverlap_survives": result.nonoverlap_survives,
                "passes_threshold": result.passes_threshold,
                "passes_strict": result.passes_strict,
            }
            output["transitions"][ttype][metric] = r

    output["window_timecourses"] = window_timecourses

    out_path = RESULTS_JSON / "temporal_precedence.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=default_ser)
    log(f"\nResults: {out_path}")
    log(f"Figures: {FIG_DIR}")
    log("DONE")


if __name__ == "__main__":
    main()
