"""Multi-block sleep robustness analysis.

Tests whether sleep eigenvalue-gap results generalize across multiple
contiguous blocks (not just the single longest run per stage). For each
subject and sleep state, analyzes the top 3 longest contiguous blocks
(where available), computes per-block eigenvalue gap means, and tests
whether the N3-vs-REM and Awake-vs-REM spacing contrasts hold when
averaging across blocks rather than using only the longest block.

Scientific rationale
--------------------
The primary sleep analysis uses only the single longest contiguous run
of each sleep stage to avoid concatenation artifacts. A reviewer could
argue this introduces selection bias. This analysis demonstrates that
the headline gap contrasts are not an artifact of block selection by
showing consistency across multiple independent blocks.

Dataset: ANPHY-Sleep — 10 subjects, 93-channel EEG (10-20/10-05),
1000 Hz, ~7 hrs per recording. Sleep-staged in 30s epochs.

Methodological guardrail
------------------------
Multi-block analysis addresses segment-selection robustness only. It
demonstrates that gap contrasts are not artifacts of choosing the single
longest contiguous block, but does not address potential confounds from
sleep-stage scoring, slow oscillation power, or time-of-night effects.
"""
from __future__ import annotations

import gc
import os
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

import mne
from cmcc.preprocess.scalp_eeg import apply_csd, pca_reduce
from cmcc.preprocess.qc import detect_bad_channels
from cmcc.analysis.dynamical_systems import compute_ep_proximity_timecourse

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("SLEEP_DATA_ROOT", "./data/ANPHY-Sleep"))
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"
FIG_DIR = CMCC_ROOT / "results" / "figures" / "ep_sleep"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 0.5
STEP_SEC = 0.1
N_COMPONENTS = 15
DOWNSAMPLE_TO = 500.0
LINE_FREQ = 50.0
SEED = 42
MAX_BLOCKS = 3
MIN_CONTIGUOUS_EPOCHS = 4

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


def find_top_n_contiguous_runs(epochs, target_stage, min_epochs=4, max_runs=3):
    """Find the top N longest contiguous runs of a given sleep stage."""
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

    runs.sort(key=lambda r: r[2], reverse=True)
    return runs[:max_runs]


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
    raw.load_data()
    raw.crop(tmin=start_sec, tmax=end_sec)

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

    return data_pca, raw.info["sfreq"]


def analyze_block(data_pca, sfreq):
    """Run EP proximity analysis on a single block."""
    ep_tc = compute_ep_proximity_timecourse(
        data_pca, sfreq=sfreq,
        window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        max_channels=N_COMPONENTS, seed=SEED,
    )
    jac = ep_tc["jac_result"]
    ep = ep_tc["ep_result"]

    return {
        "mean_gap": float(np.mean(ep.min_eigenvalue_gaps)),
        "std_gap": float(np.std(ep.min_eigenvalue_gaps)),
        "mean_spectral_radius": float(np.mean(jac.spectral_radius)),
        "mean_ep_score": float(np.mean(ep.ep_scores)),
        "n_windows": len(jac.window_centers),
    }


def main():
    log("=" * 70)
    log("MULTI-BLOCK SLEEP ROBUSTNESS ANALYSIS")
    log(f"Max blocks per state: {MAX_BLOCKS}")
    log("=" * 70)

    t_start = time.time()
    all_subjects = []

    for subject_id, (edf_name, staging_name) in SUBJECTS.items():
        log(f"\n  {subject_id}...")
        edf_path = DATA_ROOT / subject_id / edf_name
        staging_path = DATA_ROOT / subject_id / staging_name

        if not edf_path.exists() or not staging_path.exists():
            log(f"    SKIP: files not found")
            continue

        epochs = parse_sleep_staging(staging_path)
        subject_result = {"subject": subject_id, "states": {}}

        for state in TARGET_STATES:
            runs = find_top_n_contiguous_runs(epochs, state, MIN_CONTIGUOUS_EPOCHS, MAX_BLOCKS)
            if not runs:
                log(f"    {state}: no valid blocks")
                subject_result["states"][state] = {"n_blocks": 0, "blocks": []}
                continue

            log(f"    {state}: {len(runs)} blocks")
            blocks = []
            for bi, (start_sec, end_sec, n_ep) in enumerate(runs):
                duration = end_sec - start_sec
                log(f"      block {bi}: {n_ep} epochs ({duration:.0f}s)")
                try:
                    data_pca, sfreq = load_and_preprocess_segment(
                        edf_path, start_sec, end_sec, sfreq_target=DOWNSAMPLE_TO,
                    )
                    block_result = analyze_block(data_pca, sfreq)
                    block_result["start_sec"] = start_sec
                    block_result["end_sec"] = end_sec
                    block_result["n_epochs"] = n_ep
                    block_result["duration_sec"] = duration
                    blocks.append(block_result)
                    log(f"        gap={block_result['mean_gap']:.6f}")

                    del data_pca
                    gc.collect()
                except Exception as e:
                    log(f"        ERROR: {e}")

            if blocks:
                gaps = [b["mean_gap"] for b in blocks]
                subject_result["states"][state] = {
                    "n_blocks": len(blocks),
                    "blocks": blocks,
                    "mean_gap_across_blocks": float(np.mean(gaps)),
                    "std_gap_across_blocks": float(np.std(gaps)),
                    "longest_block_gap": blocks[0]["mean_gap"],
                }
            else:
                subject_result["states"][state] = {"n_blocks": 0, "blocks": []}

        all_subjects.append(subject_result)

    log("\n" + "=" * 70)
    log("GROUP CONTRASTS (Multi-Block Average)")
    log("=" * 70)

    def get_multiblock_gaps(subjects, state):
        gaps = []
        sids = []
        for s in subjects:
            st = s["states"].get(state, {})
            if st.get("n_blocks", 0) > 0:
                gaps.append(st["mean_gap_across_blocks"])
                sids.append(s["subject"])
        return np.array(gaps), sids

    def get_longest_gaps(subjects, state):
        gaps = []
        sids = []
        for s in subjects:
            st = s["states"].get(state, {})
            if st.get("n_blocks", 0) > 0:
                gaps.append(st["longest_block_gap"])
                sids.append(s["subject"])
        return np.array(gaps), sids

    contrasts = {}

    for method_name, get_fn in [("multi_block_average", get_multiblock_gaps),
                                 ("longest_block_only", get_longest_gaps)]:
        w_gaps, w_sids = get_fn(all_subjects, "W")
        n3_gaps, n3_sids = get_fn(all_subjects, "N3")
        rem_gaps, rem_sids = get_fn(all_subjects, "R")

        method_results = {}

        common_n3_rem = sorted(set(n3_sids) & set(rem_sids))
        if len(common_n3_rem) >= 3:
            n3_v = np.array([n3_gaps[n3_sids.index(s)] for s in common_n3_rem])
            rem_v = np.array([rem_gaps[rem_sids.index(s)] for s in common_n3_rem])
            t, p = sp_stats.ttest_rel(n3_v, rem_v)
            diff = n3_v - rem_v
            d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else float("nan")
            method_results["n3_vs_rem"] = {
                "mean_n3": float(np.mean(n3_v)),
                "mean_rem": float(np.mean(rem_v)),
                "t": float(t),
                "p": float(p),
                "cohens_d": d,
                "n": len(common_n3_rem),
            }
            log(f"\n  {method_name} N3 vs REM: t={t:.3f}, p={p:.6f}, d={d:.3f}, n={len(common_n3_rem)}")

        common_w_rem = sorted(set(w_sids) & set(rem_sids))
        if len(common_w_rem) >= 3:
            w_v = np.array([w_gaps[w_sids.index(s)] for s in common_w_rem])
            rem_v = np.array([rem_gaps[rem_sids.index(s)] for s in common_w_rem])
            t, p = sp_stats.ttest_rel(w_v, rem_v)
            diff = w_v - rem_v
            d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else float("nan")
            method_results["awake_vs_rem"] = {
                "mean_awake": float(np.mean(w_v)),
                "mean_rem": float(np.mean(rem_v)),
                "t": float(t),
                "p": float(p),
                "cohens_d": d,
                "n": len(common_w_rem),
            }
            log(f"  {method_name} Awake vs REM: t={t:.3f}, p={p:.6f}, d={d:.3f}, n={len(common_w_rem)}")

        common_w_n3 = sorted(set(w_sids) & set(n3_sids))
        if len(common_w_n3) >= 3:
            w_v = np.array([w_gaps[w_sids.index(s)] for s in common_w_n3])
            n3_v = np.array([n3_gaps[n3_sids.index(s)] for s in common_w_n3])
            t, p = sp_stats.ttest_rel(w_v, n3_v)
            diff = w_v - n3_v
            d = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else float("nan")
            method_results["awake_vs_n3"] = {
                "mean_awake": float(np.mean(w_v)),
                "mean_n3": float(np.mean(n3_v)),
                "t": float(t),
                "p": float(p),
                "cohens_d": d,
                "n": len(common_w_n3),
            }
            log(f"  {method_name} Awake vs N3: t={t:.3f}, p={p:.6f}, d={d:.3f}, n={len(common_w_n3)}")

        contrasts[method_name] = method_results

    elapsed = time.time() - t_start
    log(f"\nTotal time: {elapsed:.0f}s")

    output = {
        "analysis": "sleep_multiblock_robustness",
        "max_blocks_per_state": MAX_BLOCKS,
        "min_contiguous_epochs": MIN_CONTIGUOUS_EPOCHS,
        "n_subjects_attempted": len(SUBJECTS),
        "n_subjects_completed": len(all_subjects),
        "per_subject": all_subjects,
        "group_contrasts": contrasts,
        "elapsed_seconds": elapsed,
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    out_path = RESULTS_DIR / "sleep_multiblock_robustness.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)
    log(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
