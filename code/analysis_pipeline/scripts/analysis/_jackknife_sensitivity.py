"""Jackknife (leave-one-subject-out) sensitivity analysis for cross-subject correlations.

Tests whether the headline cross-subject correlations (sigma vs EP score,
LZc vs EP score, etc.) are robust to individual subject influence.
Iteratively drops each subject, recomputes Pearson r and p, and reports
the range, mean, and stability of the jackknife distribution.

This is a purely post-hoc robustness check using existing per-subject
summary statistics from exceptional_points.json. No raw data is needed.

Scientific rationale
--------------------
With n=18 subjects, a single outlier could inflate or drive a correlation.
The jackknife reveals whether any single subject is disproportionately
influential. If all leave-one-out r values remain significant (p < 0.05),
the correlation is robust to individual subject influence.

Methodological guardrail
------------------------
Jackknife assesses sensitivity to single-subject influence on the
correlation coefficient. It does not establish causal validity,
generalizability to other datasets, or independence from confounds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = CMCC_ROOT / "results" / "analysis"


def log(msg):
    print(msg, flush=True)


def jackknife_correlation(x, y, labels):
    """Leave-one-out jackknife for Pearson correlation.

    Parameters
    ----------
    x, y : np.ndarray, shape (n,)
        Paired observations.
    labels : list[str]
        Subject labels for reporting.

    Returns
    -------
    dict
        Full-sample r/p, per-drop r/p, range, stability metrics.
    """
    n = len(x)
    r_full, p_full = sp_stats.pearsonr(x, y)

    loo_results = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        r_loo, p_loo = sp_stats.pearsonr(x[mask], y[mask])
        loo_results.append({
            "dropped_subject": labels[i],
            "r": float(r_loo),
            "p": float(p_loo),
            "n": int(n - 1),
        })

    r_values = np.array([d["r"] for d in loo_results])
    p_values = np.array([d["p"] for d in loo_results])

    all_significant = bool(np.all(p_values < 0.05))
    n_significant = int(np.sum(p_values < 0.05))

    most_influential_idx = int(np.argmin(np.abs(r_values)))
    most_influential = labels[most_influential_idx]
    r_without_most_influential = float(r_values[most_influential_idx])

    return {
        "full_sample": {
            "r": float(r_full),
            "p": float(p_full),
            "n": int(n),
        },
        "jackknife": {
            "r_mean": float(np.mean(r_values)),
            "r_std": float(np.std(r_values)),
            "r_min": float(np.min(r_values)),
            "r_max": float(np.max(r_values)),
            "r_range": float(np.max(r_values) - np.min(r_values)),
            "all_significant_at_0.05": all_significant,
            "n_significant_at_0.05": n_significant,
            "n_total_drops": int(n),
            "most_influential_subject": most_influential,
            "r_without_most_influential": r_without_most_influential,
        },
        "per_drop": loo_results,
    }


def main():
    ep_path = RESULTS_DIR / "exceptional_points.json"
    if not ep_path.exists():
        log(f"ERROR: {ep_path} not found")
        sys.exit(1)

    with open(ep_path) as f:
        ep_data = json.load(f)

    subjects = ep_data["subjects"]
    labels = [s["subject"] for s in subjects]
    n = len(subjects)
    log(f"Loaded {n} subjects from exceptional_points.json")

    sigma = np.array([s["hg_sigma"] for s in subjects])
    tau = np.array([s["hg_tau"] for s in subjects])
    lzc = np.array([s["hg_lzc"] for s in subjects])
    ep_score = np.array([s["ep_score_mean"] for s in subjects])
    min_gap = np.array([s["min_eigenvalue_gap_mean"] for s in subjects])

    correlations_to_test = [
        ("sigma_vs_ep_score", sigma, ep_score),
        ("sigma_vs_min_gap", sigma, min_gap),
        ("tau_vs_ep_score", tau, ep_score),
        ("lzc_vs_ep_score", lzc, ep_score),
        ("lzc_vs_min_gap", lzc, min_gap),
    ]

    results = {
        "analysis": "jackknife_sensitivity",
        "description": "Leave-one-subject-out jackknife for cross-subject correlations",
        "n_subjects": n,
        "correlations": {},
    }

    log("\n" + "=" * 70)
    log("JACKKNIFE SENSITIVITY ANALYSIS")
    log("=" * 70)

    for name, x, y in correlations_to_test:
        log(f"\n  {name}:")
        jk = jackknife_correlation(x, y, labels)
        results["correlations"][name] = jk

        full = jk["full_sample"]
        jkr = jk["jackknife"]
        log(f"    Full sample: r={full['r']:.4f}, p={full['p']:.2e}")
        log(f"    Jackknife:   r_mean={jkr['r_mean']:.4f} +/- {jkr['r_std']:.4f}")
        log(f"                 r_range=[{jkr['r_min']:.4f}, {jkr['r_max']:.4f}]")
        log(f"                 all p<0.05: {jkr['all_significant_at_0.05']}")
        log(f"                 n significant: {jkr['n_significant_at_0.05']}/{jkr['n_total_drops']}")
        log(f"                 most influential: {jkr['most_influential_subject']} "
            f"(r without={jkr['r_without_most_influential']:.4f})")

    out_path = RESULTS_DIR / "jackknife_sensitivity.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {out_path}")

    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    headline = results["correlations"]["sigma_vs_ep_score"]
    jk = headline["jackknife"]
    if jk["all_significant_at_0.05"]:
        log("  PASS: sigma vs EP score remains significant (p<0.05) after")
        log(f"        dropping ANY single subject. Range: [{jk['r_min']:.4f}, {jk['r_max']:.4f}]")
    else:
        log(f"  WARNING: {jk['n_total_drops'] - jk['n_significant_at_0.05']} drops lose significance")


if __name__ == "__main__":
    main()
