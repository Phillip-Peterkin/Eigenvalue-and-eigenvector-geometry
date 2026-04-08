"""Diagnostic: why does spacing direction reverse between 0.5s/15comp and 2.0s/10comp?

Loads one seizure and computes trajectories with both parameter sets side-by-side.
Checks for sign errors, z-score direction, eigenvalue structure differences,
and condition number.
"""
from __future__ import annotations

import gc
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

import yaml
from cmcc.io.loader_chbmit import build_seizure_catalog, load_raw_edf
from cmcc.preprocess.seizure_eeg import (
    fit_baseline_pca, preprocess_chbmit_raw, project_to_pca,
)
from cmcc.analysis.seizure_dynamics import compute_seizure_trajectory
from cmcc.analysis.dynamical_systems import estimate_jacobian

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "chbmit.yaml"
FIG_DIR = CMCC_ROOT / "results_chbmit" / "figures" / "diagnostics"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    log("Building catalog...")
    catalogs = build_seizure_catalog(
        cfg["data"]["root"],
        min_preictal_sec=cfg["seizure"]["min_preictal_sec"],
        min_inter_seizure_sec=cfg["seizure"]["min_inter_seizure_sec"],
    )

    sid = "sub-03"
    cat = catalogs[sid]
    sz = cat.eligible_seizures[0]
    log(f"Using {sid}, seizure onset={sz.onset_sec:.0f}s, dur={sz.duration_sec:.0f}s")

    log("Loading and preprocessing...")
    raw = load_raw_edf(cfg["data"]["root"], sz.subject_id, sz.session, sz.run, preload=True)
    data, sfreq, _ = preprocess_chbmit_raw(raw, line_freq=60.0, bandpass=(0.5, 45.0))
    del raw
    gc.collect()

    bl_window = cfg["seizure"]["baseline_window"]
    baseline_start = max(0.0, sz.onset_sec + bl_window[0])
    baseline_end = sz.onset_sec + bl_window[1]

    configs = [
        {"label": "OLD: 0.5s/15comp/1e-4", "window": 0.5, "step": 0.1,
         "n_comp": 15, "reg": 1e-4, "color": "blue"},
        {"label": "NEW: 2.0s/10comp/1e-3", "window": 2.0, "step": 0.5,
         "n_comp": 10, "reg": 1e-3, "color": "red"},
        {"label": "CTRL: 2.0s/15comp/1e-3", "window": 2.0, "step": 0.5,
         "n_comp": 15, "reg": 1e-3, "color": "green"},
        {"label": "CTRL: 0.5s/10comp/1e-4", "window": 0.5, "step": 0.1,
         "n_comp": 10, "reg": 1e-4, "color": "orange"},
    ]

    results = {}
    for c in configs:
        log(f"\n--- {c['label']} ---")

        pca, _ = fit_baseline_pca(
            data, sfreq, baseline_start, baseline_end,
            n_components=c["n_comp"],
        )
        data_pca = project_to_pca(data, pca)
        log(f"  PCA shape: {data_pca.shape}")

        traj = compute_seizure_trajectory(
            data_pca, sfreq,
            seizure_onset_sec=sz.onset_sec,
            seizure_offset_sec=sz.offset_sec,
            baseline_start_sec=baseline_start,
            baseline_end_sec=baseline_end,
            window_sec=c["window"],
            step_sec=c["step"],
            regularization=c["reg"],
            smoothing_sec=30.0,
            subject_id=sid,
        )

        window_samples = int(c["window"] * sfreq)
        step_samples = max(1, int(c["step"] * sfreq))
        ch_mean = data_pca.mean(axis=1, keepdims=True)
        ch_std = data_pca.std(axis=1, keepdims=True)
        ch_std[ch_std == 0] = 1.0
        data_z = (data_pca - ch_mean) / ch_std

        jac = estimate_jacobian(
            data_z, window_size=max(window_samples, c["n_comp"] + 10),
            step_size=step_samples, regularization=c["reg"],
        )

        preictal_mask = (traj.time_sec >= -600) & (traj.time_sec < 0)
        baseline_mask = (traj.time_sec >= -1800) & (traj.time_sec < -600)

        raw_pre = traj.min_spacing_raw[preictal_mask]
        raw_bl = traj.min_spacing_raw[baseline_mask]
        z_pre = traj.min_spacing_z[preictal_mask]

        log(f"  Raw spacing — baseline mean: {np.nanmean(raw_bl):.6f}, "
            f"pre-ictal mean: {np.nanmean(raw_pre):.6f}")
        log(f"  Raw spacing — pre-ictal {'WIDER' if np.nanmean(raw_pre) > np.nanmean(raw_bl) else 'NARROWER'} "
            f"than baseline")
        log(f"  Z-scored pre-ictal mean: {np.nanmean(z_pre):.4f}")
        log(f"  Spectral radius — baseline: {np.nanmean(jac.spectral_radius[baseline_mask[:len(jac.spectral_radius)]]):.4f}, "
            f"pre-ictal: {np.nanmean(jac.spectral_radius[preictal_mask[:len(jac.spectral_radius)]]):.4f}")
        log(f"  Condition number — baseline: {np.nanmedian(jac.condition_numbers[baseline_mask[:len(jac.condition_numbers)]]):.1f}, "
            f"pre-ictal: {np.nanmedian(jac.condition_numbers[preictal_mask[:len(jac.condition_numbers)]]):.1f}")
        log(f"  Residual variance — baseline: {np.nanmean(jac.residual_variance[baseline_mask[:len(jac.residual_variance)]]):.4f}, "
            f"pre-ictal: {np.nanmean(jac.residual_variance[preictal_mask[:len(jac.residual_variance)]]):.4f}")
        log(f"  Pre-ictal slope: {traj.preictal_slope:.6f}")
        log(f"  Windows: {len(traj.time_sec)}, samples/window: {window_samples}, "
            f"params: {c['n_comp']**2}, ratio: {(window_samples-1)/c['n_comp']**2:.2f}")

        results[c["label"]] = {
            "traj": traj,
            "jac": jac,
            "config": c,
            "data_pca": data_pca,
        }

        del data_pca
        gc.collect()

    log("\n\n=== PLOTTING DIAGNOSTICS ===")

    fig, axes = plt.subplots(4, 1, figsize=(16, 20), sharex=True)

    for label, r in results.items():
        t = r["traj"]
        c = r["config"]
        t_min = t.time_sec / 60.0

        axes[0].plot(t_min, t.min_spacing_raw, alpha=0.7, label=label, color=c["color"], lw=0.5)
        axes[1].plot(t_min, t.min_spacing_z, alpha=0.7, label=label, color=c["color"], lw=0.5)
        axes[2].plot(t_min, t.spectral_radius_z, alpha=0.7, label=label, color=c["color"], lw=0.5)

        rv = r["jac"].residual_variance
        t_rv = np.arange(len(rv)) * c["step"] / 60.0
        t_rv = t_rv - (t.time_sec[0] / -60.0)
        jac_time = (r["jac"].window_centers / sfreq - sz.onset_sec) / 60.0
        axes[3].plot(jac_time, rv, alpha=0.7, label=label, color=c["color"], lw=0.5)

    for ax in axes:
        ax.axvline(0, color="black", ls="--", lw=1, alpha=0.5)
        ax.axvspan(-30, -10, color="green", alpha=0.05)
        ax.axvspan(-10, 0, color="yellow", alpha=0.05)
        ax.legend(fontsize=7, loc="upper left")

    axes[0].set_ylabel("Raw min spacing")
    axes[0].set_title("Raw minimum eigenvalue spacing (no z-score)")
    axes[1].set_ylabel("Z-scored spacing")
    axes[1].set_title("Z-scored minimum spacing (z to baseline)")
    axes[2].set_ylabel("Z-scored spectral radius")
    axes[2].set_title("Z-scored spectral radius")
    axes[3].set_ylabel("Residual variance")
    axes[3].set_title("VAR(1) residual variance (fit quality)")
    axes[3].set_xlabel("Time relative to onset (min)")

    for ax in axes:
        ax.set_xlim(-32, 12)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "parameter_reversal_diagnostic.png", dpi=200)
    plt.close(fig)
    log(f"Saved: {FIG_DIR / 'parameter_reversal_diagnostic.png'}")

    log("\n\n=== EIGENVALUE MAGNITUDE COMPARISON ===")
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
    for idx, (label, r) in enumerate(results.items()):
        ax = axes2.flat[idx]
        t = r["traj"]
        jac = r["jac"]
        n_comp = r["config"]["n_comp"]
        jac_time = (jac.window_centers / sfreq - sz.onset_sec) / 60.0

        ev_mags = np.abs(jac.eigenvalues)

        step = max(1, len(jac_time) // 500)
        for ci in range(min(n_comp, 5)):
            ax.plot(jac_time[::step], ev_mags[::step, ci], alpha=0.5, lw=0.5, label=f"|λ_{ci}|")

        ax.axvline(0, color="black", ls="--", lw=1)
        ax.axhline(1.0, color="red", ls=":", lw=1, alpha=0.5)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("|eigenvalue|")
        ax.set_xlim(-12, 5)
        ax.legend(fontsize=6)

    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "eigenvalue_magnitudes_comparison.png", dpi=200)
    plt.close(fig2)
    log(f"Saved: {FIG_DIR / 'eigenvalue_magnitudes_comparison.png'}")

    log("\n\n=== DIRECT SPACING COMPARISON: BASELINE vs PRE-ICTAL ===")
    for label, r in results.items():
        t = r["traj"]
        bl_mask = (t.time_sec >= -1800) & (t.time_sec < -600)
        pre_mask = (t.time_sec >= -600) & (t.time_sec < 0)
        peri_mask = (t.time_sec >= -60) & (t.time_sec < 0)

        raw_bl = t.min_spacing_raw[bl_mask]
        raw_pre = t.min_spacing_raw[pre_mask]
        raw_peri = t.min_spacing_raw[peri_mask]

        log(f"\n{label}:")
        log(f"  Raw baseline mean:    {np.nanmean(raw_bl):.8f}")
        log(f"  Raw pre-ictal mean:   {np.nanmean(raw_pre):.8f}")
        log(f"  Raw peri-onset mean:  {np.nanmean(raw_peri):.8f}")
        log(f"  Direction (pre vs bl): {'WIDER' if np.nanmean(raw_pre) > np.nanmean(raw_bl) else 'NARROWER'}")
        log(f"  Z-scored pre-ictal:   {np.nanmean(t.min_spacing_z[pre_mask]):.4f}")
        log(f"  Z-scored peri-onset:  {np.nanmean(t.min_spacing_z[peri_mask]):.4f}")


if __name__ == "__main__":
    main()
