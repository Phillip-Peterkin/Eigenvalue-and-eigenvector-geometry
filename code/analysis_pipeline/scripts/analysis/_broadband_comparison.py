"""Compare high-gamma vs broadband pipeline results.

Loads both group result JSONs and performs paired comparisons
across subjects that succeeded in both pipelines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

RESULTS_HG = Path(__file__).resolve().parent.parent / "results" / "group_all_subjects.json"
RESULTS_BB = Path(__file__).resolve().parent.parent / "results_broadband" / "group_all_subjects.json"


def log(msg):
    print(msg, flush=True)


def main():
    if not RESULTS_HG.exists():
        log(f"ERROR: High-gamma results not found: {RESULTS_HG}")
        return
    if not RESULTS_BB.exists():
        log(f"ERROR: Broadband results not found: {RESULTS_BB}")
        return

    with open(RESULTS_HG) as f:
        hg_data = json.load(f)
    with open(RESULTS_BB) as f:
        bb_data = json.load(f)

    hg_ok = {s["subject"]: s for s in hg_data if s.get("status") == "OK"}
    bb_ok = {s["subject"]: s for s in bb_data if s.get("status") == "OK"}

    common_subjects = sorted(set(hg_ok.keys()) & set(bb_ok.keys()))

    log("=" * 78)
    log("HIGH-GAMMA vs BROADBAND COMPARISON")
    log("=" * 78)
    log(f"\n  High-gamma OK: {len(hg_ok)}")
    log(f"  Broadband OK:  {len(bb_ok)}")
    log(f"  Common:        {len(common_subjects)}")

    hg_only = sorted(set(hg_ok.keys()) - set(bb_ok.keys()))
    bb_only = sorted(set(bb_ok.keys()) - set(hg_ok.keys()))
    if hg_only:
        log(f"  HG-only: {hg_only}")
    if bb_only:
        log(f"  BB-only: {bb_only}")

    if len(common_subjects) < 3:
        log("\nNot enough common subjects for comparison.")
        return

    metrics = [
        ("tau", "tau"),
        ("sigma", "branching_sigma"),
        ("LZc", "lzc_normalized"),
        ("DFA", "dfa_alpha"),
    ]

    log(f"\n{'Metric':<12} {'HG mean':>10} {'BB mean':>10} {'Diff':>10} {'t':>8} {'p':>10} {'Direction':>15}")
    log("-" * 80)

    comparison_results = {}

    for name, key in metrics:
        hg_vals = np.array([hg_ok[s][key] for s in common_subjects])
        bb_vals = np.array([bb_ok[s][key] for s in common_subjects])
        diff = hg_vals - bb_vals
        t, p = sp_stats.ttest_rel(hg_vals, bb_vals)
        direction = "HG > BB" if np.mean(diff) > 0 else "BB > HG"
        sig = " *" if p < 0.05 else ""

        log(f"  {name:<10} {np.mean(hg_vals):>10.4f} {np.mean(bb_vals):>10.4f} "
            f"{np.mean(diff):>+10.4f} {t:>8.3f} {p:>10.4f}{sig} {direction:>15}")

        comparison_results[name] = {
            "hg_mean": float(np.mean(hg_vals)),
            "bb_mean": float(np.mean(bb_vals)),
            "diff_mean": float(np.mean(diff)),
            "t": float(t),
            "p": float(p),
            "direction": direction,
        }

    log(f"\n{'='*78}")
    log("DECODING COMPARISON")
    log(f"{'='*78}")

    for n_ch in [5, 10, 20]:
        hg_diffs = []
        bb_diffs = []
        paired_subjects = []
        for s in common_subjects:
            hg_dec = hg_ok[s].get("decoding", [])
            bb_dec = bb_ok[s].get("decoding", [])
            hg_d = [d["diff_most_least"] for d in hg_dec if d["n"] == n_ch]
            bb_d = [d["diff_most_least"] for d in bb_dec if d["n"] == n_ch]
            if hg_d and bb_d:
                hg_diffs.append(hg_d[0])
                bb_diffs.append(bb_d[0])
                paired_subjects.append(s)

        if len(hg_diffs) >= 3:
            hg_arr = np.array(hg_diffs)
            bb_arr = np.array(bb_diffs)

            hg_reversal = sum(hg_arr < 0)
            bb_reversal = sum(bb_arr < 0)

            t_hg, p_hg = sp_stats.ttest_1samp(hg_arr, 0)
            t_bb, p_bb = sp_stats.ttest_1samp(bb_arr, 0)
            t_pair, p_pair = sp_stats.ttest_rel(hg_arr, bb_arr)

            log(f"\n  n={n_ch} channels ({len(hg_diffs)} subjects):")
            log(f"    HG:  mean diff={np.mean(hg_arr):+.4f}, reversal {hg_reversal}/{len(hg_arr)}, t={t_hg:.3f}, p={p_hg:.4f}")
            log(f"    BB:  mean diff={np.mean(bb_arr):+.4f}, reversal {bb_reversal}/{len(bb_arr)}, t={t_bb:.3f}, p={p_bb:.4f}")
            log(f"    Paired: t={t_pair:.3f}, p={p_pair:.4f}")

            comparison_results[f"decoding_n={n_ch}"] = {
                "hg_mean_diff": float(np.mean(hg_arr)),
                "bb_mean_diff": float(np.mean(bb_arr)),
                "hg_reversal_count": int(hg_reversal),
                "bb_reversal_count": int(bb_reversal),
                "n_subjects": len(hg_diffs),
                "hg_ttest_p": float(p_hg),
                "bb_ttest_p": float(p_bb),
                "paired_t": float(t_pair),
                "paired_p": float(p_pair),
            }

    log(f"\n{'='*78}")
    log("TASK CONTRAST COMPARISON")
    log(f"{'='*78}")

    hg_lzc_g = []
    bb_lzc_g = []
    for s in common_subjects:
        hg_g = hg_ok[s].get("task_contrast_g")
        bb_g = bb_ok[s].get("task_contrast_g")
        if hg_g is not None and bb_g is not None:
            hg_lzc_g.append(hg_g)
            bb_lzc_g.append(bb_g)

    if len(hg_lzc_g) >= 3:
        hg_arr = np.array(hg_lzc_g)
        bb_arr = np.array(bb_lzc_g)
        t, p = sp_stats.ttest_rel(hg_arr, bb_arr)
        log(f"  LZc contrast effect (Hedges g):")
        log(f"    HG:  mean={np.mean(hg_arr):+.4f}")
        log(f"    BB:  mean={np.mean(bb_arr):+.4f}")
        log(f"    Paired: t={t:.3f}, p={p:.4f}")

    out_path = Path(__file__).resolve().parent.parent / "results" / "broadband_comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    log(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
