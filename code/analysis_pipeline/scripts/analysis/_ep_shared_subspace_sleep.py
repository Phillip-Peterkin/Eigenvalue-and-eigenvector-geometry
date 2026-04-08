import os
"""Shared-subspace robustness test for sleep EP geometry.

Pools W + N3 + REM data within each subject before PCA, fits once,
then projects all three states into the common subspace. Tests whether
the key finding (REM tightest gap, N3 vs REM d=-2.51) survives when
all states share the same coordinate system.

Also computes alternative spacing metrics (median NN, P10 NN).
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

import mne
from cmcc.preprocess.scalp_eeg import apply_csd, pca_reduce_shared
from cmcc.preprocess.qc import detect_bad_channels
from cmcc.analysis.dynamical_systems import compute_ep_proximity_timecourse
from cmcc.analysis.ep_advanced import (
    compute_spectral_radius_sensitivity,
    compute_alternative_spacing,
)
from cmcc.analysis.contrasts import fdr_correction

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = Path(os.environ.get("SLEEP_DATA_ROOT", "./data/ANPHY-Sleep"))
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"
FIG_DIR = CMCC_ROOT / "results" / "figures" / "ep_shared_subspace"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SEC = 0.5
STEP_SEC = 0.1
N_COMPONENTS = 15
DOWNSAMPLE_TO = 500.0
LINE_FREQ = 50.0
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


def load_csd_segment(edf_path, start_sec, end_sec, sfreq_target=500.0):
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

    data_csd = raw.get_data()
    ch_names = raw.ch_names
    sfreq = raw.info["sfreq"]

    info = {
        "n_channels": data_csd.shape[0],
        "bad_channels": bad,
        "sfreq": sfreq,
        "duration_sec": float(data_csd.shape[1] / sfreq),
        "channels_dropped_no_montage": missing,
    }

    return data_csd, sfreq, ch_names, info


def analyze_state_shared(data_pca, sfreq, subject_id, state):
    ep_tc = compute_ep_proximity_timecourse(
        data_pca, sfreq=sfreq,
        window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        max_channels=N_COMPONENTS, seed=SEED,
    )

    jac = ep_tc["jac_result"]
    ep = ep_tc["ep_result"]

    srs = compute_spectral_radius_sensitivity(jac, ep)
    alt_spacing = compute_alternative_spacing(jac.eigenvalues)

    return {
        "subject": subject_id,
        "state": state,
        "n_windows": len(jac.window_centers),
        "mean_spectral_radius": float(np.mean(jac.spectral_radius)),
        "mean_eigenvalue_gap": float(np.mean(ep.min_eigenvalue_gaps)),
        "mean_median_nn_gap": float(np.mean(alt_spacing["median_nn_gap"])),
        "mean_p10_nn_gap": float(np.mean(alt_spacing["p10_nn_gap"])),
        "mean_ep_score": float(np.mean(ep.ep_scores)),
        "spectral_sensitivity": srs,
    }


def analyze_single_subject(subject_id, edf_path, staging_path):
    t0 = time.time()
    log(f"\n  {subject_id}...")

    epochs = parse_sleep_staging(staging_path)

    state_csd = {}
    state_sfreq = {}
    state_info = {}
    state_segments = {}

    for state in TARGET_STATES:
        run = find_longest_contiguous_run(epochs, state, MIN_CONTIGUOUS_EPOCHS)
        if run is None:
            log(f"    {state}: no contiguous run >= {MIN_CONTIGUOUS_EPOCHS} epochs, SKIP")
            return None

        start_sec, end_sec, n_epochs = run
        log(f"    {state}: {n_epochs} epochs ({end_sec-start_sec:.0f}s)")

        try:
            data_csd, sfreq, ch_names, info = load_csd_segment(
                edf_path, start_sec, end_sec, sfreq_target=DOWNSAMPLE_TO,
            )
        except Exception as e:
            log(f"    {state} ERROR preprocessing: {e}")
            return None

        state_csd[state] = data_csd
        state_sfreq[state] = sfreq
        state_info[state] = info
        state_segments[state] = {"start_sec": start_sec, "end_sec": end_sec, "n_epochs": n_epochs}

    ch_counts = [d.shape[0] for d in state_csd.values()]
    if len(set(ch_counts)) > 1:
        min_ch = min(ch_counts)
        log(f"    WARNING: channel mismatch {ch_counts}, truncating to {min_ch}")
        state_csd = {k: v[:min_ch] for k, v in state_csd.items()}

    projected, pca_obj, pca_info = pca_reduce_shared(state_csd, n_components=N_COMPONENTS)

    log(f"    Shared PCA: {pca_info['n_components']} comp, "
        f"cumvar={pca_info['cumulative_variance']:.3f}, "
        f"pooled={pca_info['total_samples_pooled']} samples")

    del state_csd
    gc.collect()

    results = []
    for state in TARGET_STATES:
        r = analyze_state_shared(projected[state], state_sfreq[state], subject_id, state)
        r["segment"] = state_segments[state]
        r["preprocess"] = state_info[state]
        results.append(r)
        log(f"    {state}: rho={r['mean_spectral_radius']:.4f} "
            f"gap={r['mean_eigenvalue_gap']:.6f} "
            f"spec_sens_r={r['spectral_sensitivity']['r']:.4f}")

    del projected
    gc.collect()

    return {
        "subject": subject_id,
        "pca_mode": "shared_subspace",
        "shared_pca_info": pca_info,
        "states": {r["state"]: r for r in results},
        "elapsed_s": time.time() - t0,
    }


def _cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def compute_group_statistics(all_results, original_json_path=None):
    valid = [s for s in all_results if s is not None]
    if len(valid) < 3:
        return {}

    group = {"n_subjects": len(valid)}
    all_p = []
    all_labels = []

    sids = [s["subject"] for s in valid]

    gap_metrics = ["mean_eigenvalue_gap", "mean_median_nn_gap", "mean_p10_nn_gap"]
    gap_labels = ["min_gap", "median_nn_gap", "p10_nn_gap"]

    for metric, mlabel in zip(gap_metrics, gap_labels):
        awake_vals = [s["states"]["W"][metric] for s in valid]
        n3_vals = [s["states"]["N3"][metric] for s in valid]
        rem_vals = [s["states"]["R"][metric] for s in valid]

        for pair_label, a, b in [
            (f"{mlabel}_awake_vs_n3", n3_vals, awake_vals),
            (f"{mlabel}_n3_vs_rem", rem_vals, n3_vals),
            (f"{mlabel}_awake_vs_rem", rem_vals, awake_vals),
        ]:
            t_val, p_val = sp_stats.ttest_rel(a, b)
            d = _cohens_d_paired(a, b)
            group[pair_label] = {
                "mean_a": float(np.mean(a)),
                "mean_b": float(np.mean(b)),
                "mean_diff": float(np.mean(np.array(a) - np.array(b))),
                "t": float(t_val),
                "p": float(p_val),
                "cohens_d": d,
                "n": len(valid),
            }
            all_p.append(float(p_val))
            all_labels.append(pair_label)

    awake_rho = [s["states"]["W"]["mean_spectral_radius"] for s in valid]
    n3_rho = [s["states"]["N3"]["mean_spectral_radius"] for s in valid]
    rem_rho = [s["states"]["R"]["mean_spectral_radius"] for s in valid]
    for pair_label, a, b in [
        ("spectral_radius_awake_vs_n3", n3_rho, awake_rho),
        ("spectral_radius_n3_vs_rem", rem_rho, n3_rho),
    ]:
        t_val, p_val = sp_stats.ttest_rel(a, b)
        d = _cohens_d_paired(a, b)
        group[pair_label] = {
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(np.array(a) - np.array(b))),
            "t": float(t_val),
            "p": float(p_val),
            "cohens_d": d,
            "n": len(valid),
        }
        all_p.append(float(p_val))
        all_labels.append(pair_label)

    if len(all_p) >= 2:
        fdr_sig = fdr_correction(all_p, alpha=0.05)
        group["fdr_correction"] = {
            label: {"p": float(p), "fdr_significant": bool(sig)}
            for label, p, sig in zip(all_labels, all_p, fdr_sig)
        }

    if original_json_path and Path(original_json_path).exists():
        try:
            with open(original_json_path) as f:
                orig = json.load(f)
            orig_gs = orig.get("group_statistics", {})
            comparison = {}
            orig_map = {
                "min_gap_n3_vs_rem": "test_a_gap_n3_vs_rem",
                "min_gap_awake_vs_rem": "test_a_gap_awake_vs_rem",
                "min_gap_awake_vs_n3": "test_a_gap_awake_vs_n3",
            }
            for shared_key, orig_key in orig_map.items():
                if shared_key in group and orig_key in orig_gs:
                    comparison[shared_key] = {
                        "original_d": orig_gs[orig_key].get("cohens_d"),
                        "original_p": orig_gs[orig_key].get("p"),
                        "shared_d": group[shared_key]["cohens_d"],
                        "shared_p": group[shared_key]["p"],
                    }
            group["comparison_to_original"] = comparison
        except Exception as e:
            group["comparison_to_original"] = {"error": str(e)}

    return group


def plot_comparison(all_results, group_stats, output_dir):
    valid = [s for s in all_results if s is not None]
    if len(valid) < 3:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Shared-Subspace PCA — Sleep State Comparisons",
                 fontsize=14, fontweight="bold")

    sids = [s["subject"] for s in valid]
    x = np.arange(len(sids))
    w = 0.25

    metric_panels = [
        ("mean_eigenvalue_gap", "Min Gap", "min_gap"),
        ("mean_median_nn_gap", "Median NN Gap", "median_nn_gap"),
        ("mean_p10_nn_gap", "P10 NN Gap", "p10_nn_gap"),
    ]

    for idx, (key, ylabel, gs_prefix) in enumerate(metric_panels):
        ax = axes[0, idx]
        awake_vals = [s["states"]["W"][key] for s in valid]
        n3_vals = [s["states"]["N3"][key] for s in valid]
        rem_vals = [s["states"]["R"][key] for s in valid]

        ax.bar(x - w, awake_vals, w, label="Awake", color="steelblue", alpha=0.8)
        ax.bar(x, n3_vals, w, label="N3", color="coral", alpha=0.8)
        ax.bar(x + w, rem_vals, w, label="REM", color="seagreen", alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([s[-2:] for s in sids], fontsize=7)
        ax.legend(fontsize=7)

        gs = group_stats.get(f"{gs_prefix}_n3_vs_rem", {})
        ax.set_title(f"{ylabel}\nN3 vs REM: t={gs.get('t', 0):.2f}, p={gs.get('p', 1):.4f}, d={gs.get('cohens_d', 0):.2f}")

    ax = axes[1, 0]
    awake_rho = [s["states"]["W"]["mean_spectral_radius"] for s in valid]
    n3_rho = [s["states"]["N3"]["mean_spectral_radius"] for s in valid]
    rem_rho = [s["states"]["R"]["mean_spectral_radius"] for s in valid]
    ax.bar(x - w, awake_rho, w, label="Awake", color="steelblue", alpha=0.8)
    ax.bar(x, n3_rho, w, label="N3", color="coral", alpha=0.8)
    ax.bar(x + w, rem_rho, w, label="REM", color="seagreen", alpha=0.8)
    ax.set_ylabel("Spectral Radius")
    ax.set_xticks(x)
    ax.set_xticklabels([s[-2:] for s in sids], fontsize=7)
    ax.legend(fontsize=7)
    ax.set_title("Spectral Radius")

    axes[1, 1].axis("off")
    comp = group_stats.get("comparison_to_original", {})
    if comp and "error" not in comp:
        text = "Original vs Shared-Subspace:\n\n"
        for metric, vals in comp.items():
            text += f"{metric}:\n"
            text += f"  orig d={vals['original_d']:.2f}, p={vals['original_p']:.4f}\n"
            text += f"  shared d={vals['shared_d']:.2f}, p={vals['shared_p']:.4f}\n\n"
        axes[1, 1].text(0.1, 0.5, text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment="center", fontfamily="monospace")

    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "shared_subspace_sleep.png", dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Figure: {output_dir / 'shared_subspace_sleep.png'}")


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
    log("SHARED-SUBSPACE PCA — Sleep EP Robustness Test")
    log(f"Dataset: ANPHY-Sleep ({DATA_ROOT})")
    log(f"Parameters: window={WINDOW_SEC}s, step={STEP_SEC}s, "
        f"PCA={N_COMPONENTS} (SHARED), downsample={DOWNSAMPLE_TO} Hz")
    log("=" * 70)

    all_results = []
    for subject_id, (edf_name, staging_name) in SUBJECTS.items():
        edf_path = DATA_ROOT / subject_id / edf_name
        staging_path = DATA_ROOT / subject_id / staging_name

        if not edf_path.exists():
            log(f"\n  {subject_id}: EDF not found at {edf_path}, SKIP")
            continue
        if not staging_path.exists():
            log(f"\n  {subject_id}: staging file not found at {staging_path}, SKIP")
            continue

        try:
            result = analyze_single_subject(subject_id, edf_path, staging_path)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            log(f"\n  {subject_id}: ERROR: {e}")
            import traceback
            traceback.print_exc()

    log(f"\n{'='*70}")
    log(f"RESULTS: {len(all_results)} subjects analyzed")

    orig_json = RESULTS_DIR / "ep_sleep_dynamics.json"
    group_stats = compute_group_statistics(all_results, original_json_path=orig_json)

    if group_stats:
        log(f"\n  Group statistics (shared subspace):")
        for key, val in group_stats.items():
            if key not in ("fdr_correction", "comparison_to_original"):
                log(f"    {key}: {val}")
        if "fdr_correction" in group_stats:
            log(f"\n    FDR correction:")
            for label, info in group_stats["fdr_correction"].items():
                log(f"      {label}: p={info['p']:.6f} sig={info['fdr_significant']}")
        if "comparison_to_original" in group_stats:
            comp = group_stats["comparison_to_original"]
            if "error" not in comp:
                log(f"\n  Comparison to per-state PCA:")
                for metric, vals in comp.items():
                    log(f"    {metric}: orig d={vals['original_d']:.2f} -> shared d={vals['shared_d']:.2f}")

    plot_comparison(all_results, group_stats, FIG_DIR)

    out = {
        "analysis": "shared_subspace_sleep",
        "n_subjects": len(all_results),
        "parameters": {
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "n_components": N_COMPONENTS,
            "pca_mode": "shared_subspace",
            "downsample_to": DOWNSAMPLE_TO,
            "line_freq": LINE_FREQ,
            "seed": SEED,
        },
        "subjects": all_results,
        "group_statistics": group_stats,
    }

    out_path = RESULTS_DIR / "ep_shared_subspace_sleep.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=default_ser)
    log(f"\n  Results: {out_path}")
    log(f"\n{'='*70}")
    log("DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()
