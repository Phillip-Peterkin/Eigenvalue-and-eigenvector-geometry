import os
"""
Channel Scaling Sweep — Standalone

Tests whether the 30-channel subsampling cap suppresses directional (anti-symmetric)
signal in the Jacobian decomposition. Runs the NH decomposition at multiple channel
counts for selected subjects with many electrodes.

Scientific rationale: Chirality is a loop property (A→B→C→A). If only 30 of 254
channels are sampled, the anti-symmetric component cancels more readily than the
symmetric component, producing a biased-low asymmetry ratio.
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

from cmcc.config import load_config
from cmcc.preprocess.filter import SITE_LINE_FREQ
from cmcc.analysis.dynamical_systems import (
    estimate_jacobian,
    decompose_jacobian_hermiticity,
)

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "default.yaml"
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"
DATA_ROOT = Path(os.environ.get("IEEG_DATA_ROOT", "./data/Cogitate_IEEG_EXP1"))

WINDOW_SEC = 0.5
STEP_SEC = 0.1

SWEEP_SUBJECTS = ["CE110", "CF105", "CF103", "CF124", "CG103"]
CHANNEL_COUNTS = [10, 20, 30, 50, 70, 100, 150, 200]


def log(msg):
    print(msg, flush=True)


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


def run_sweep_single(subject_id, config):
    t0 = time.time()
    log(f"\n  Channel scaling: {subject_id}")

    data, sfreq = load_and_preprocess(subject_id, config)
    if data is None:
        log(f"    SKIP: no data")
        return None

    n_ch, n_samples = data.shape
    log(f"    {n_ch} channels available, {sfreq} Hz, {n_samples/sfreq:.1f}s")

    rng = np.random.default_rng(42)
    ch_mean = data.mean(axis=1, keepdims=True)
    ch_std = data.std(axis=1, keepdims=True)
    ch_std[ch_std == 0] = 1.0
    data_z = (data - ch_mean) / ch_std

    window_samples = max(int(WINDOW_SEC * sfreq), 40)
    step_samples = max(1, int(STEP_SEC * sfreq))

    results_per_nch = []
    for n_use in CHANNEL_COUNTS:
        if n_use > n_ch:
            continue

        window_adj = max(window_samples, n_use + 10)
        if n_samples < window_adj + 1:
            log(f"    n_ch={n_use}: SKIP (not enough samples for window)")
            continue

        if n_use < n_ch:
            ch_idx = np.sort(rng.choice(n_ch, n_use, replace=False))
            d = data_z[ch_idx]
        else:
            d = data_z

        t1 = time.time()
        try:
            jac = estimate_jacobian(d, window_size=window_adj, step_size=step_samples)
            nh = decompose_jacobian_hermiticity(jac)
            entry = {
                "n_channels": int(n_use),
                "asymmetry_ratio_mean": float(nh.mean_asymmetry_ratio),
                "asymmetry_ratio_std": float(nh.std_asymmetry_ratio),
                "asymmetry_kurtosis": float(nh.kurtosis_asymmetry_ratio),
                "asymmetry_max": float(nh.max_asymmetry_ratio),
                "asymmetry_p95": float(nh.p95_asymmetry_ratio),
                "asymmetry_p99": float(nh.p99_asymmetry_ratio),
                "asymmetry_dynamic_range": float(nh.dynamic_range),
                "max_rotation_freq_mean": float(np.mean(nh.max_rotation_frequency)),
                "spectral_radius_mean": float(np.mean(jac.spectral_radius)),
                "n_windows": len(jac.window_centers),
                "elapsed_s": time.time() - t1,
            }
            results_per_nch.append(entry)
            log(f"    n_ch={n_use:3d}: asym_mean={entry['asymmetry_ratio_mean']:.4f} "
                f"kurtosis={entry['asymmetry_kurtosis']:.2f} "
                f"max={entry['asymmetry_max']:.4f} "
                f"spec_rad={entry['spectral_radius_mean']:.4f} "
                f"[{entry['elapsed_s']:.0f}s]")
            del jac, nh
        except Exception as e:
            log(f"    n_ch={n_use}: FAILED ({e})")

    del data, data_z
    gc.collect()

    return {
        "subject": subject_id,
        "n_channels_total": int(n_ch),
        "sfreq": float(sfreq),
        "n_samples": int(n_samples),
        "sweep": results_per_nch,
        "elapsed_s": time.time() - t0,
    }


def main():
    log("=" * 70)
    log("CHANNEL SCALING SWEEP - Non-Hermitian Decomposition")
    log("=" * 70)

    config = load_config(str(CONFIG_PATH))

    with open(CMCC_ROOT / "results" / "group_all_subjects.json") as f:
        group_data = json.load(f)
    hg_subjects = {s["subject"] for s in group_data if s.get("status") == "OK"}

    all_sweep = []
    for subj in SWEEP_SUBJECTS:
        if subj not in hg_subjects:
            log(f"  {subj}: not in group results, skipping")
            continue
        try:
            sr = run_sweep_single(subj, config)
            if sr is not None:
                all_sweep.append(sr)
                out_path = RESULTS_DIR / "channel_sweep.json"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump({"sweep_results": all_sweep, "channel_counts": CHANNEL_COUNTS}, f, indent=2)
                log(f"  Saved intermediate: {out_path}")
        except Exception as e:
            log(f"  SWEEP ERROR {subj}: {e}")
            import traceback
            traceback.print_exc()

    log(f"\n{'='*70}")
    log(f"SWEEP COMPLETE: {len(all_sweep)} subjects")
    for sr in all_sweep:
        log(f"  {sr['subject']} ({sr['n_channels_total']} ch): {len(sr['sweep'])} points, {sr['elapsed_s']:.0f}s")
    log("=" * 70)


if __name__ == "__main__":
    main()
