import os
"""Electrode-level spatial mapping of band-specific effects.

Uses available per-channel tau from deep_dive (5 subjects with HG tau)
and computes per-channel metrics comparison where possible. For group
analysis, uses subject-level metrics with electrode coverage metadata.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

from cmcc.analysis.spatial import define_posterior_roi

CMCC_ROOT = Path(__file__).resolve().parent.parent
RESULTS_HG = CMCC_ROOT / "results"
RESULTS_BB = CMCC_ROOT / "results_broadband"
DEEP_DIVE = RESULTS_HG / "deep_dive" / "deep_dive_summary.json"
DATA_ROOT = Path(os.environ.get("IEEG_DATA_ROOT", "./data/Cogitate_IEEG_EXP1"))
FIG_DIR = RESULTS_HG / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def load_electrodes(subject_id):
    meta_dir = DATA_ROOT / f"{subject_id}_ECOG_1" / "METADATA" / "electrode_coordinates"
    tsv = meta_dir / f"sub-{subject_id}_ses-1_space-fsaverage_electrodes.tsv"
    if tsv.exists():
        return pd.read_csv(tsv, sep="\t")
    return None


def load_all_data():
    with open(RESULTS_HG / "group_all_subjects.json") as f:
        hg_data = json.load(f)
    with open(RESULTS_BB / "group_all_subjects.json") as f:
        bb_data = json.load(f)
    hg_ok = {s["subject"]: s for s in hg_data if s.get("status") == "OK"}
    bb_ok = {s["subject"]: s for s in bb_data if s.get("status") == "OK"}

    deep_dive = {}
    if DEEP_DIVE.exists():
        with open(DEEP_DIVE) as f:
            deep_dive = json.load(f)

    return hg_ok, bb_ok, deep_dive


def analyze_deep_dive_subjects(deep_dive, hg_ok, bb_ok):
    results = {}
    for subj, dd in deep_dive.items():
        if subj not in hg_ok or subj not in bb_ok:
            continue

        hg_tau_pc = dd.get("tau_per_channel", {})
        if not hg_tau_pc:
            continue

        electrodes = load_electrodes(subj)
        if electrodes is None:
            continue

        roi = define_posterior_roi(electrodes)
        post_ch = set(roi.posterior_channels) & set(hg_tau_pc)
        ant_ch = set(roi.anterior_channels) & set(hg_tau_pc)

        post_tau = [hg_tau_pc[ch] for ch in post_ch if not np.isnan(hg_tau_pc[ch])]
        ant_tau = [hg_tau_pc[ch] for ch in ant_ch if not np.isnan(hg_tau_pc[ch])]

        hg_sigma = hg_ok[subj].get("branching_sigma")
        bb_sigma = bb_ok[subj].get("branching_sigma")
        hg_lzc_g = hg_ok[subj].get("task_contrast_g")
        bb_lzc_g = bb_ok[subj].get("task_contrast_g")

        result = {
            "n_channels": len(hg_tau_pc),
            "n_posterior": len(post_tau),
            "n_anterior": len(ant_tau),
            "hg_sigma": hg_sigma,
            "bb_sigma": bb_sigma,
            "hg_lzc_g": hg_lzc_g,
            "bb_lzc_g": bb_lzc_g,
            "lzc_sign_flip": (hg_lzc_g is not None and bb_lzc_g is not None
                              and np.sign(hg_lzc_g) != np.sign(bb_lzc_g)),
        }

        if post_tau:
            result["posterior_tau_mean"] = float(np.mean(post_tau))
            result["posterior_tau_std"] = float(np.std(post_tau))
        if ant_tau:
            result["anterior_tau_mean"] = float(np.mean(ant_tau))
            result["anterior_tau_std"] = float(np.std(ant_tau))
        if post_tau and ant_tau and len(post_tau) >= 3 and len(ant_tau) >= 3:
            t, p = sp_stats.ttest_ind(post_tau, ant_tau)
            result["post_vs_ant_tau_t"] = float(t)
            result["post_vs_ant_tau_p"] = float(p)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x_coords, y_coords, tau_vals, colors = [], [], [], []
        for _, row in electrodes.iterrows():
            name = str(row["name"])
            if name in hg_tau_pc and not np.isnan(hg_tau_pc[name]):
                x_coords.append(float(row["x"]))
                y_coords.append(float(row["y"]))
                tau_vals.append(hg_tau_pc[name])
                if name in post_ch:
                    colors.append("posterior")
                elif name in ant_ch:
                    colors.append("anterior")
                else:
                    colors.append("unclassified")

        ax = axes[0]
        sc = ax.scatter(x_coords, y_coords, c=tau_vals, cmap="YlOrRd", s=40,
                       edgecolors="black", linewidths=0.3)
        fig.colorbar(sc, ax=ax, label="HG tau")
        ax.set_xlabel("x (fsaverage)")
        ax.set_ylabel("y (fsaverage)")
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        ax.set_title(f"{subj} HG Per-Channel Tau")

        ax = axes[1]
        color_map = {"posterior": "blue", "anterior": "red", "unclassified": "gray"}
        for c_label in ["posterior", "anterior", "unclassified"]:
            mask = [i for i, c in enumerate(colors) if c == c_label]
            if mask:
                ax.scatter([x_coords[i] for i in mask], [y_coords[i] for i in mask],
                          c=color_map[c_label], s=40, label=c_label, edgecolors="black", linewidths=0.3)
        ax.legend()
        ax.set_xlabel("x (fsaverage)")
        ax.set_ylabel("y (fsaverage)")
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
        lzc_str = f"HG g={hg_lzc_g:+.3f}" if hg_lzc_g else "HG: N/A"
        bb_str = f"BB g={bb_lzc_g:+.3f}" if bb_lzc_g else "BB: N/A"
        ax.set_title(f"{subj} ROI ({lzc_str}, {bb_str})")

        fig.suptitle(f"{subj}: Spatial Coverage & Band Contrast", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"spatial_band_{subj}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        results[subj] = result
        log(f"  {subj}: {len(hg_tau_pc)} ch, post={len(post_tau)}, ant={len(ant_tau)}, "
            f"LZc flip={result['lzc_sign_flip']}")

    return results


def group_coverage_analysis(hg_ok, bb_ok):
    common = sorted(set(hg_ok) & set(bb_ok))
    rows = []

    for subj in common:
        electrodes = load_electrodes(subj)
        if electrodes is None:
            continue

        roi = define_posterior_roi(electrodes)
        n_post = len(roi.posterior_channels)
        n_ant = len(roi.anterior_channels)
        n_total = n_post + n_ant + len(roi.unclassified_channels)
        frac_post = n_post / n_total if n_total > 0 else 0

        hg_g = hg_ok[subj].get("task_contrast_g")
        bb_g = bb_ok[subj].get("task_contrast_g")
        hg_sigma = hg_ok[subj].get("branching_sigma")
        bb_sigma = bb_ok[subj].get("branching_sigma")

        rows.append({
            "subject": subj,
            "n_posterior": n_post,
            "n_anterior": n_ant,
            "frac_posterior": frac_post,
            "hg_lzc_g": hg_g,
            "bb_lzc_g": bb_g,
            "delta_lzc_g": (bb_g - hg_g) if (hg_g is not None and bb_g is not None) else None,
            "hg_sigma": hg_sigma,
            "bb_sigma": bb_sigma,
            "delta_sigma": bb_sigma - hg_sigma if (hg_sigma and bb_sigma) else None,
        })

    df = pd.DataFrame(rows)

    correlations = {}
    valid = df.dropna(subset=["delta_lzc_g", "frac_posterior"])
    if len(valid) >= 5:
        r, p = sp_stats.pearsonr(valid["frac_posterior"], valid["delta_lzc_g"])
        correlations["frac_posterior_vs_delta_lzc"] = {"r": float(r), "p": float(p), "n": len(valid)}
        log(f"\n  Posterior fraction vs delta LZc: r={r:.3f}, p={p:.4f} (n={len(valid)})")

    valid2 = df.dropna(subset=["delta_sigma", "frac_posterior"])
    if len(valid2) >= 5:
        r, p = sp_stats.pearsonr(valid2["frac_posterior"], valid2["delta_sigma"])
        correlations["frac_posterior_vs_delta_sigma"] = {"r": float(r), "p": float(p), "n": len(valid2)}
        log(f"  Posterior fraction vs delta sigma: r={r:.3f}, p={p:.4f} (n={len(valid2)})")

    if len(valid) >= 5:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.scatter(valid["frac_posterior"], valid["delta_lzc_g"], s=60, edgecolors="black", linewidths=0.5)
        r_val = correlations["frac_posterior_vs_delta_lzc"]["r"]
        p_val = correlations["frac_posterior_vs_delta_lzc"]["p"]
        z = np.polyfit(valid["frac_posterior"], valid["delta_lzc_g"], 1)
        x_line = np.linspace(valid["frac_posterior"].min(), valid["frac_posterior"].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Fraction posterior electrodes")
        ax.set_ylabel("Delta LZc contrast (BB - HG)")
        ax.set_title(f"Coverage vs LZc reversal (r={r_val:.3f}, p={p_val:.4f})")

        ax = axes[1]
        valid_s = df.dropna(subset=["delta_sigma", "frac_posterior"])
        ax.scatter(valid_s["frac_posterior"], valid_s["delta_sigma"], s=60, edgecolors="black", linewidths=0.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Fraction posterior electrodes")
        ax.set_ylabel("Delta sigma (BB - HG)")
        ax.set_title("Coverage vs sigma shift")

        fig.suptitle("Spatial Coverage Predicts Band-Specific Effects?", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "spatial_band_coverage_correlation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log(f"  Saved: spatial_band_coverage_correlation.png")

    return {
        "coverage_table": rows,
        "correlations": correlations,
    }


def main():
    log("=" * 70)
    log("SPATIAL BAND-SPECIFIC ANALYSIS")
    log("=" * 70)

    hg_ok, bb_ok, deep_dive = load_all_data()
    common = sorted(set(hg_ok) & set(bb_ok))
    log(f"\n  Common subjects: {len(common)}")
    log(f"  Deep dive subjects: {list(deep_dive.keys())}")

    log("\n[1] Deep dive subjects — per-channel spatial maps...")
    dd_results = analyze_deep_dive_subjects(deep_dive, hg_ok, bb_ok)

    log("\n[2] Group coverage analysis...")
    coverage_results = group_coverage_analysis(hg_ok, bb_ok)

    out = {
        "deep_dive_spatial": dd_results,
        "group_coverage": coverage_results,
    }
    out_path = RESULTS_HG / "spatial_band_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"\n  Results: {out_path}")

    log("\n" + "=" * 70)
    log("SPATIAL BAND ANALYSIS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
