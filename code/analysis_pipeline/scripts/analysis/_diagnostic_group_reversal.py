"""Diagnostic: check per-subject pre-ictal spacing direction with new params.

Loads all subjects, computes single-seizure trajectories with NEW params only,
and reports whether each seizure shows narrowing or widening.
"""
from __future__ import annotations

import gc
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

import yaml
from cmcc.io.loader_chbmit import build_seizure_catalog, load_raw_edf
from cmcc.preprocess.seizure_eeg import (
    fit_baseline_pca, preprocess_chbmit_raw, project_to_pca,
)
from cmcc.analysis.seizure_dynamics import compute_seizure_trajectory

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "chbmit.yaml"


def log(msg):
    print(msg, flush=True)


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    catalogs = build_seizure_catalog(
        cfg["data"]["root"],
        min_preictal_sec=cfg["seizure"]["min_preictal_sec"],
        min_inter_seizure_sec=cfg["seizure"]["min_inter_seizure_sec"],
    )

    bl_window = cfg["seizure"]["baseline_window"]
    pp = cfg["preprocessing"]
    var = cfg["var"]

    all_results = []

    for sid, cat in sorted(catalogs.items()):
        if cat.n_eligible == 0:
            continue

        sz = cat.eligible_seizures[0]
        log(f"\n{sid}: onset={sz.onset_sec:.0f}s, dur={sz.duration_sec:.0f}s")

        try:
            raw = load_raw_edf(cfg["data"]["root"], sz.subject_id, sz.session, sz.run, preload=True)
            data, sfreq, _ = preprocess_chbmit_raw(raw, line_freq=pp["line_freq"], bandpass=tuple(pp["bandpass"]))
            del raw

            baseline_start = max(0.0, sz.onset_sec + bl_window[0])
            baseline_end = sz.onset_sec + bl_window[1]

            pca, _ = fit_baseline_pca(data, sfreq, baseline_start, baseline_end, n_components=pp["n_components"])
            data_pca = project_to_pca(data, pca)
            del data

            traj = compute_seizure_trajectory(
                data_pca, sfreq,
                seizure_onset_sec=sz.onset_sec,
                seizure_offset_sec=sz.offset_sec,
                baseline_start_sec=baseline_start,
                baseline_end_sec=baseline_end,
                window_sec=var["window_sec"],
                step_sec=var["step_sec"],
                regularization=var["regularization"],
                smoothing_sec=cfg["smoothing"]["moving_average_sec"],
                subject_id=sid,
            )
            del data_pca

            bl_mask = (traj.time_sec >= -1800) & (traj.time_sec < -600)
            pre_mask = (traj.time_sec >= -600) & (traj.time_sec < 0)

            raw_bl = np.nanmean(traj.min_spacing_raw[bl_mask])
            raw_pre = np.nanmean(traj.min_spacing_raw[pre_mask])
            z_pre = np.nanmean(traj.min_spacing_z[pre_mask])
            z_bl = np.nanmean(traj.min_spacing_z[bl_mask])

            direction = "WIDER" if raw_pre > raw_bl else "NARROWER"
            z_direction = "+" if z_pre > 0 else "-"

            log(f"  raw_bl={raw_bl:.6f}, raw_pre={raw_pre:.6f} => {direction}")
            log(f"  z_bl={z_bl:.4f}, z_pre={z_pre:.4f} (z_direction={z_direction})")
            log(f"  slope={traj.preictal_slope:.6f}")

            all_results.append({
                "sid": sid,
                "raw_bl": raw_bl,
                "raw_pre": raw_pre,
                "raw_direction": direction,
                "z_pre": z_pre,
                "z_bl": z_bl,
                "slope": traj.preictal_slope,
            })

        except Exception as e:
            log(f"  SKIP: {e}")

        gc.collect()

    log("\n\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)

    n_narrower = sum(1 for r in all_results if r["raw_direction"] == "NARROWER")
    n_wider = sum(1 for r in all_results if r["raw_direction"] == "WIDER")
    n_neg_z = sum(1 for r in all_results if r["z_pre"] < 0)
    n_pos_z = sum(1 for r in all_results if r["z_pre"] > 0)
    n_neg_slope = sum(1 for r in all_results if r["slope"] < 0)
    n_pos_slope = sum(1 for r in all_results if r["slope"] > 0)

    log(f"Total subjects: {len(all_results)}")
    log(f"Raw pre-ictal NARROWER than baseline: {n_narrower}/{len(all_results)}")
    log(f"Raw pre-ictal WIDER than baseline:    {n_wider}/{len(all_results)}")
    log(f"Z-scored pre-ictal < 0 (narrower):    {n_neg_z}/{len(all_results)}")
    log(f"Z-scored pre-ictal > 0 (wider):       {n_pos_z}/{len(all_results)}")
    log(f"Negative slope (narrowing):           {n_neg_slope}/{len(all_results)}")
    log(f"Positive slope (widening):            {n_pos_slope}/{len(all_results)}")

    z_pre_all = np.array([r["z_pre"] for r in all_results])
    log(f"\nMean z-scored pre-ictal: {np.mean(z_pre_all):.4f}")
    log(f"Median z-scored pre-ictal: {np.median(z_pre_all):.4f}")

    slopes_all = np.array([r["slope"] for r in all_results])
    log(f"Mean slope: {np.mean(slopes_all):.6f}")
    log(f"Median slope: {np.median(slopes_all):.6f}")

    log("\nPer-subject (first seizure only):")
    for r in all_results:
        log(f"  {r['sid']}: raw_dir={r['raw_direction']}, z_pre={r['z_pre']:+.4f}, slope={r['slope']:+.6f}")


if __name__ == "__main__":
    main()
