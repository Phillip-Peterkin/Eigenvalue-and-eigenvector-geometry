"""Post-hoc Tests A and C from existing ep_propofol_eeg.json results.

Computes gap widening (Test A) and delta-delta correlation (Test C) from
the per-subject summaries already stored in the JSON, without re-processing
raw EEG data. Test B (alpha sensitivity) requires raw PCA data and must
use the full pipeline.

Usage:
    cd CMCC && python scripts/analysis/_ep_propofol_posthoc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from cmcc.analysis.contrasts import fdr_correction

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"
FIG_DIR = CMCC_ROOT / "results" / "figures" / "ep_propofol"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def _cohens_d_paired(a, b):
    diff = np.array(a) - np.array(b)
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(diff) / sd)


def main():
    json_path = RESULTS_DIR / "ep_propofol_eeg.json"
    if not json_path.exists():
        log(f"ERROR: {json_path} not found. Run _ep_propofol_eeg.py first.")
        return

    with open(json_path) as f:
        data = json.load(f)

    subjects = [s for s in data["subjects"] if s is not None]
    log(f"Loaded {len(subjects)} subjects from {json_path}")

    awake_gap = [s["awake"]["mean_eigenvalue_gap"] for s in subjects]
    sed_gap_r1 = [s["sedation_run1"]["mean_eigenvalue_gap"] for s in subjects]

    all_p = []
    all_labels = []
    posthoc = {}

    # ── Test A: Gap Widening ──
    t_gw, p_gw = sp_stats.ttest_rel(sed_gap_r1, awake_gap, alternative="greater")
    d_gw = _cohens_d_paired(sed_gap_r1, awake_gap)
    posthoc["gap_widening"] = {
        "mean_awake": float(np.mean(awake_gap)),
        "mean_sed_run1": float(np.mean(sed_gap_r1)),
        "mean_diff_sed_minus_awake": float(np.mean(np.array(sed_gap_r1) - np.array(awake_gap))),
        "t": float(t_gw),
        "p_onesided": float(p_gw),
        "cohens_d": d_gw,
        "n": len(awake_gap),
        "hypothesis": "sed_run1 gap > awake gap (modes decouple under propofol)",
    }
    all_p.append(float(p_gw))
    all_labels.append("gap_widening")

    log(f"\n=== Test A: Gap Widening ===")
    log(f"  Awake gap:  {np.mean(awake_gap):.6f} +/- {np.std(awake_gap):.6f}")
    log(f"  Sed gap:    {np.mean(sed_gap_r1):.6f} +/- {np.std(sed_gap_r1):.6f}")
    log(f"  Diff (sed-awake): {np.mean(np.array(sed_gap_r1) - np.array(awake_gap)):.6f}")
    log(f"  t={t_gw:.4f}, p(one-sided)={p_gw:.6f}, d={d_gw:.3f}")

    # ── Test C: Delta-Delta Correlation ──
    delta_r = [s["delta_spectral_sensitivity_r_run1"] for s in subjects]
    delta_gap = [s["delta_eigenvalue_gap_run1"] for s in subjects]

    if len(delta_r) >= 5:
        r_dd, p_dd = sp_stats.pearsonr(delta_r, delta_gap)
        rho_dd, p_sp_dd = sp_stats.spearmanr(delta_r, delta_gap)
        posthoc["delta_delta_correlation"] = {
            "r": float(r_dd),
            "p": float(p_dd),
            "rho": float(rho_dd),
            "p_spearman": float(p_sp_dd),
            "n_subjects": len(delta_r),
            "description": (
                "cross-subject correlation of delta_r (awake-sed spec_sens) vs "
                "delta_gap (awake-sed eigenvalue gap). Deltas are awake minus sed: "
                "positive delta_r = sensitivity loss, negative delta_gap = gap widening. "
                "A NEGATIVE r supports the hypothesis."
            ),
        }
        all_p.append(float(p_dd))
        all_labels.append("delta_delta_correlation")

        log(f"\n=== Test C: Delta-Delta Correlation ===")
        log(f"  Pearson r={r_dd:.4f}, p={p_dd:.6f}")
        log(f"  Spearman rho={rho_dd:.4f}, p={p_sp_dd:.6f}")
        log(f"  N subjects={len(delta_r)}")

    # ── FDR on just these two tests ──
    if len(all_p) >= 2:
        fdr_sig = fdr_correction(all_p, alpha=0.05)
        posthoc["fdr_correction"] = {
            label: {"p": float(p), "fdr_significant": bool(sig)}
            for label, p, sig in zip(all_labels, all_p, fdr_sig)
        }
        log(f"\n=== FDR Correction ===")
        for label, p, sig in zip(all_labels, all_p, fdr_sig):
            log(f"  {label}: p={p:.6f} sig={sig}")

    # ── Figures ──
    # Test A histogram
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.linspace(
        min(min(awake_gap), min(sed_gap_r1)) * 0.9,
        max(max(awake_gap), max(sed_gap_r1)) * 1.1,
        15,
    )
    ax.hist(awake_gap, bins=bins, alpha=0.6, color="steelblue", label="Awake", edgecolor="white")
    ax.hist(sed_gap_r1, bins=bins, alpha=0.6, color="coral", label="Sedation (run-1)", edgecolor="white")
    ax.axvline(np.mean(awake_gap), color="steelblue", linestyle="--", linewidth=2)
    ax.axvline(np.mean(sed_gap_r1), color="coral", linestyle="--", linewidth=2)
    ax.set_xlabel("Mean Min Eigenvalue Gap")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Test A: Gap Widening Under Propofol\n"
        f"one-sided p={p_gw:.4f}, d={d_gw:.2f}"
    )
    ax.legend()
    plt.tight_layout()
    fig_path = FIG_DIR / "gap_widening_histogram.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"\n  Figure: {fig_path}")

    # Test C scatter
    if len(delta_r) >= 5:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.scatter(delta_r, delta_gap, s=50, c="steelblue", edgecolors="white", zorder=5)
        z = np.polyfit(delta_r, delta_gap, 1)
        x_line = np.linspace(min(delta_r), max(delta_r), 100)
        ax.plot(x_line, np.polyval(z, x_line), "--", color="coral", linewidth=2)
        ax.set_xlabel("Delta Spectral Sensitivity r (awake - sed)")
        ax.set_ylabel("Delta Eigenvalue Gap (awake - sed)")
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(
            f"Test C: Delta-Delta Correlation\n"
            f"r={r_dd:.3f}, p={p_dd:.4f}, N={len(delta_r)}"
        )
        plt.tight_layout()
        fig_path = FIG_DIR / "delta_delta_scatter.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        log(f"  Figure: {fig_path}")

    # ── Save ──
    out_path = RESULTS_DIR / "ep_propofol_posthoc_AC.json"
    with open(out_path, "w") as f:
        json.dump(posthoc, f, indent=2)
    log(f"\n  Results saved: {out_path}")
    log(f"\n  NOTE: Test B (alpha sensitivity) requires raw PCA data.")
    log(f"  For Test B, re-run: python scripts/analysis/_ep_propofol_eeg.py")


if __name__ == "__main__":
    main()
