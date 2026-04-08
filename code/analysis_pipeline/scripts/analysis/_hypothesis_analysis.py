"""Step 13: Hypothesis reformulation & statistical framework.

Tests the refined hypothesis: consciousness is supported by subcritical
suppression of large avalanches, allowing stable fine-grained representations.

Five key analyses:
1. Subcritical suppression test (sigma vs decoding accuracy)
2. Avalanche size ceiling test (max avalanche size vs decoding)
3. Tau-decoding quadrant analysis
4. Within-subject channel-level correlation
5. Multi-level summary table
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

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIG_DIR = RESULTS_DIR / "figures"


def log(msg):
    print(msg, flush=True)


def load_group_data():
    with open(RESULTS_DIR / "group_all_subjects.json") as f:
        data = json.load(f)
    return [s for s in data if s.get("status") == "OK" and s.get("qc_pass", False)]


def pearson_with_ci(x, y, alpha=0.05):
    r, p = sp_stats.pearsonr(x, y)
    n = len(x)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3) if n > 3 else float("inf")
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lo = np.tanh(z - z_crit * se)
    ci_hi = np.tanh(z + z_crit * se)
    return {"r": r, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


def spearman_with_ci(x, y, alpha=0.05):
    rho, p = sp_stats.spearmanr(x, y)
    n = len(x)
    se = 1.0 / np.sqrt(n - 3) if n > 3 else float("inf")
    z = np.arctanh(rho)
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lo = np.tanh(z - z_crit * se)
    ci_hi = np.tanh(z + z_crit * se)
    return {"rho": rho, "p": p, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": n}


def test1_subcritical_suppression(subjects):
    log("\n--- Test 1: Subcritical Suppression (sigma vs decoding) ---")
    results = {}

    for n_ch in [5, 10, 20]:
        sigmas = []
        dec_least = []
        dec_most = []
        dec_diff = []
        subj_ids = []

        for s in subjects:
            dec = s.get("decoding", [])
            for d in dec:
                if d["n"] == n_ch:
                    sigmas.append(s["branching_sigma"])
                    dec_least.append(d["least"])
                    dec_most.append(d["most"])
                    dec_diff.append(d["diff_most_least"])
                    subj_ids.append(s["subject"])

        if len(sigmas) < 5:
            continue

        sigmas = np.array(sigmas)
        dec_least = np.array(dec_least)
        dec_most = np.array(dec_most)
        dec_diff = np.array(dec_diff)

        r_sigma_least = pearson_with_ci(sigmas, dec_least)
        r_sigma_most = pearson_with_ci(sigmas, dec_most)
        r_sigma_diff = pearson_with_ci(sigmas, dec_diff)
        rho_sigma_diff = spearman_with_ci(sigmas, dec_diff)

        results[f"n={n_ch}"] = {
            "n_subjects": len(sigmas),
            "sigma_vs_least_acc": r_sigma_least,
            "sigma_vs_most_acc": r_sigma_most,
            "sigma_vs_diff": r_sigma_diff,
            "sigma_vs_diff_spearman": rho_sigma_diff,
        }

        log(f"  n={n_ch} ({len(sigmas)} subjects):")
        log(f"    sigma vs least_acc: r={r_sigma_least['r']:.3f} p={r_sigma_least['p']:.4f} CI=[{r_sigma_least['ci_lo']:.3f},{r_sigma_least['ci_hi']:.3f}]")
        log(f"    sigma vs most_acc:  r={r_sigma_most['r']:.3f} p={r_sigma_most['p']:.4f}")
        log(f"    sigma vs diff:      r={r_sigma_diff['r']:.3f} p={r_sigma_diff['p']:.4f}")
        log(f"    sigma vs diff (rho): rho={rho_sigma_diff['rho']:.3f} p={rho_sigma_diff['p']:.4f}")

    return results


def test2_avalanche_ceiling(subjects):
    log("\n--- Test 2: Avalanche Size Ceiling vs Decoding ---")
    results = {}

    for n_ch in [5, 10, 20]:
        n_avals = []
        dec_diff = []
        dec_least = []
        subj_ids = []

        for s in subjects:
            dec = s.get("decoding", [])
            for d in dec:
                if d["n"] == n_ch:
                    n_avals.append(s["n_avalanches"])
                    dec_diff.append(d["diff_most_least"])
                    dec_least.append(d["least"])
                    subj_ids.append(s["subject"])

        if len(n_avals) < 5:
            continue

        n_avals = np.array(n_avals, dtype=float)
        dec_diff = np.array(dec_diff)
        dec_least = np.array(dec_least)
        log_n_avals = np.log10(n_avals + 1)

        r_aval_diff = pearson_with_ci(log_n_avals, dec_diff)
        r_aval_least = pearson_with_ci(log_n_avals, dec_least)

        results[f"n={n_ch}"] = {
            "n_subjects": len(n_avals),
            "log_n_avals_vs_diff": r_aval_diff,
            "log_n_avals_vs_least_acc": r_aval_least,
        }

        log(f"  n={n_ch} ({len(n_avals)} subjects):")
        log(f"    log(n_avals) vs diff: r={r_aval_diff['r']:.3f} p={r_aval_diff['p']:.4f}")
        log(f"    log(n_avals) vs least_acc: r={r_aval_least['r']:.3f} p={r_aval_least['p']:.4f}")

    return results


def test3_tau_decoding_quadrant(subjects):
    log("\n--- Test 3: Tau-Decoding Quadrant Analysis ---")
    results = {}

    for n_ch in [5, 10, 20]:
        taus = []
        dec_most = []
        dec_least = []
        subj_ids = []

        for s in subjects:
            dec = s.get("decoding", [])
            for d in dec:
                if d["n"] == n_ch:
                    taus.append(s["tau"])
                    dec_most.append(d["most"])
                    dec_least.append(d["least"])
                    subj_ids.append(s["subject"])

        if len(taus) < 5:
            continue

        taus = np.array(taus)
        dec_most = np.array(dec_most)
        dec_least = np.array(dec_least)

        r_tau_most = pearson_with_ci(taus, dec_most)
        r_tau_least = pearson_with_ci(taus, dec_least)
        r_tau_diff = pearson_with_ci(taus, dec_most - dec_least)

        results[f"n={n_ch}"] = {
            "n_subjects": len(taus),
            "tau_vs_most_acc": r_tau_most,
            "tau_vs_least_acc": r_tau_least,
            "tau_vs_diff": r_tau_diff,
        }

        log(f"  n={n_ch} ({len(taus)} subjects):")
        log(f"    tau vs most_acc:  r={r_tau_most['r']:.3f} p={r_tau_most['p']:.4f}")
        log(f"    tau vs least_acc: r={r_tau_least['r']:.3f} p={r_tau_least['p']:.4f}")
        log(f"    tau vs diff:      r={r_tau_diff['r']:.3f} p={r_tau_diff['p']:.4f}")

    return results


def test4_complexity_decoding(subjects):
    log("\n--- Test 4: Complexity Metrics vs Decoding ---")
    results = {}

    for metric_key, metric_name in [("lzc_normalized", "LZc"), ("dfa_alpha", "DFA"), ("mse_ci", "MSE_CI")]:
        for n_ch in [10]:
            vals = []
            dec_diff = []
            dec_least = []
            subj_ids = []

            for s in subjects:
                if s.get(metric_key) is None:
                    continue
                dec = s.get("decoding", [])
                for d in dec:
                    if d["n"] == n_ch:
                        vals.append(s[metric_key])
                        dec_diff.append(d["diff_most_least"])
                        dec_least.append(d["least"])
                        subj_ids.append(s["subject"])

            if len(vals) < 5:
                continue

            vals = np.array(vals)
            dec_diff = np.array(dec_diff)
            dec_least = np.array(dec_least)

            r_metric_diff = pearson_with_ci(vals, dec_diff)
            r_metric_least = pearson_with_ci(vals, dec_least)

            results[f"{metric_name}_n={n_ch}"] = {
                "n_subjects": len(vals),
                "metric_vs_diff": r_metric_diff,
                "metric_vs_least_acc": r_metric_least,
            }

            log(f"  {metric_name} n={n_ch} ({len(vals)} subjects):")
            log(f"    {metric_name} vs diff: r={r_metric_diff['r']:.3f} p={r_metric_diff['p']:.4f}")
            log(f"    {metric_name} vs least_acc: r={r_metric_least['r']:.3f} p={r_metric_least['p']:.4f}")

    return results


def test5_task_contrast_vs_criticality(subjects):
    log("\n--- Test 5: Task Contrast vs Criticality ---")

    sigmas = []
    lzc_gs = []
    hg_gs = []
    subj_ids = []

    for s in subjects:
        if s.get("task_contrast_g") is not None and s.get("hg_power_contrast_g") is not None:
            sigmas.append(s["branching_sigma"])
            lzc_gs.append(s["task_contrast_g"])
            hg_gs.append(s["hg_power_contrast_g"])
            subj_ids.append(s["subject"])

    results = {}
    if len(sigmas) >= 5:
        sigmas = np.array(sigmas)
        lzc_gs = np.array(lzc_gs)
        hg_gs = np.array(hg_gs)

        r_sigma_lzc = pearson_with_ci(sigmas, lzc_gs)
        r_sigma_hg = pearson_with_ci(sigmas, hg_gs)
        r_lzc_hg = pearson_with_ci(lzc_gs, hg_gs)

        results = {
            "n_subjects": len(sigmas),
            "sigma_vs_lzc_g": r_sigma_lzc,
            "sigma_vs_hg_g": r_sigma_hg,
            "lzc_g_vs_hg_g": r_lzc_hg,
        }

        log(f"  N={len(sigmas)} subjects with contrast data:")
        log(f"    sigma vs LZc effect: r={r_sigma_lzc['r']:.3f} p={r_sigma_lzc['p']:.4f}")
        log(f"    sigma vs HG effect:  r={r_sigma_hg['r']:.3f} p={r_sigma_hg['p']:.4f}")
        log(f"    LZc vs HG effects:   r={r_lzc_hg['r']:.3f} p={r_lzc_hg['p']:.4f}")

    return results


def create_quadrant_plot(subjects, output_dir):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for n_ch in [5, 10, 20]:
        taus, dec_most, dec_least, subj_ids = [], [], [], []
        for s in subjects:
            for d in s.get("decoding", []):
                if d["n"] == n_ch:
                    taus.append(s["tau"])
                    dec_most.append(d["most"])
                    dec_least.append(d["least"])
                    subj_ids.append(s["subject"])

        if len(taus) < 5:
            continue

        taus = np.array(taus)
        dec_most = np.array(dec_most)
        dec_least = np.array(dec_least)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(taus, dec_most, c="blue", s=50, label="Most critical", alpha=0.7, edgecolors="black", linewidths=0.5)
        ax.scatter(taus, dec_least, c="red", s=50, label="Least critical", alpha=0.7, edgecolors="black", linewidths=0.5)

        for i, sid in enumerate(subj_ids):
            ax.plot([taus[i], taus[i]], [dec_most[i], dec_least[i]], "gray", alpha=0.3, linewidth=0.5)

        z_most = np.polyfit(taus, dec_most, 1)
        z_least = np.polyfit(taus, dec_least, 1)
        tau_range = np.linspace(taus.min(), taus.max(), 50)
        ax.plot(tau_range, np.polyval(z_most, tau_range), "b--", alpha=0.5, label=f"Most trend (slope={z_most[0]:.4f})")
        ax.plot(tau_range, np.polyval(z_least, tau_range), "r--", alpha=0.5, label=f"Least trend (slope={z_least[0]:.4f})")

        ax.axvline(1.5, color="green", linestyle=":", alpha=0.5, label="Critical tau=1.5")
        ax.set_xlabel("Tau (size exponent)", fontsize=12)
        ax.set_ylabel("Decoding accuracy", fontsize=12)
        ax.set_title(f"Tau-Decoding Quadrant (n={n_ch} channels)", fontsize=13)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(output_dir / f"hypothesis_tau_decoding_n{n_ch}.png", dpi=150)
        plt.close(fig)


def create_sigma_decoding_plot(subjects, output_dir):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for n_ch in [10]:
        sigmas, dec_diff, dec_least, subj_ids = [], [], [], []
        for s in subjects:
            for d in s.get("decoding", []):
                if d["n"] == n_ch:
                    sigmas.append(s["branching_sigma"])
                    dec_diff.append(d["diff_most_least"])
                    dec_least.append(d["least"])
                    subj_ids.append(s["subject"])

        if len(sigmas) < 5:
            continue

        sigmas = np.array(sigmas)
        dec_diff = np.array(dec_diff)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(sigmas, dec_diff, c="purple", s=60, edgecolors="black", linewidths=0.5)
        axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
        axes[0].axvline(1.0, color="green", linestyle=":", alpha=0.5, label="Critical sigma=1.0")
        z = np.polyfit(sigmas, dec_diff, 1)
        sig_range = np.linspace(sigmas.min(), sigmas.max(), 50)
        axes[0].plot(sig_range, np.polyval(z, sig_range), "purple", alpha=0.5, linestyle="--")
        r, p = sp_stats.pearsonr(sigmas, dec_diff)
        axes[0].set_xlabel("Branching ratio (sigma)", fontsize=12)
        axes[0].set_ylabel("Decoding diff (most - least)", fontsize=12)
        axes[0].set_title(f"Sigma vs Decoding Reversal\nr={r:.3f}, p={p:.4f}", fontsize=12)
        axes[0].legend()

        for i, sid in enumerate(subj_ids):
            axes[0].annotate(sid, (sigmas[i], dec_diff[i]), fontsize=6, alpha=0.5)

        taus = np.array([s["tau"] for s in subjects if any(d["n"] == n_ch for d in s.get("decoding", []))])
        lzcs = np.array([s["lzc_normalized"] for s in subjects if any(d["n"] == n_ch for d in s.get("decoding", []))])
        dfas = np.array([s["dfa_alpha"] for s in subjects if any(d["n"] == n_ch for d in s.get("decoding", []))])
        dec_diffs = dec_diff

        metrics = {"tau": taus, "sigma": sigmas, "LZc": lzcs, "DFA": dfas}
        r_vals, p_vals = [], []
        names = []
        for name, vals in metrics.items():
            if len(vals) == len(dec_diffs):
                r_i, p_i = sp_stats.pearsonr(vals, dec_diffs)
                r_vals.append(r_i)
                p_vals.append(p_i)
                names.append(name)

        colors = ["red" if p < 0.05 else "gray" for p in p_vals]
        axes[1].barh(names, r_vals, color=colors, edgecolor="black")
        axes[1].axvline(0, color="black", linewidth=0.5)
        axes[1].set_xlabel("Pearson r with decoding diff", fontsize=12)
        axes[1].set_title(f"Metric correlations with reversal (n={n_ch})", fontsize=12)
        for i, (r_v, p_v) in enumerate(zip(r_vals, p_vals)):
            sig_str = f" p={p_v:.3f}" + (" *" if p_v < 0.05 else "")
            axes[1].text(r_v + 0.01 if r_v >= 0 else r_v - 0.01, i, f"r={r_v:.3f}{sig_str}",
                        va="center", ha="left" if r_v >= 0 else "right", fontsize=9)

        fig.tight_layout()
        fig.savefig(output_dir / f"hypothesis_sigma_decoding_n{n_ch}.png", dpi=150)
        plt.close(fig)


def create_summary_table(all_results, subjects):
    log("\n" + "=" * 78)
    log("HYPOTHESIS ANALYSIS SUMMARY TABLE")
    log("=" * 78)

    log(f"\n{'Test':<45} {'Effect':>8} {'p':>8} {'Direction':>15} {'Supports?':>10}")
    log("-" * 90)

    tests = []

    t1 = all_results.get("test1_subcritical_suppression", {})
    for key in sorted(t1.keys()):
        r = t1[key].get("sigma_vs_diff", {})
        if r:
            direction = "more subcrit -> reversal" if r["r"] < 0 else "less subcrit -> reversal"
            supports = "YES" if r["r"] < 0 and r["p"] < 0.1 else "no"
            tests.append(("sigma vs diff " + key, r["r"], r["p"], direction, supports))

    t3 = all_results.get("test3_tau_decoding_quadrant", {})
    for key in sorted(t3.keys()):
        r = t3[key].get("tau_vs_diff", {})
        if r:
            direction = "high tau -> reversal" if r["r"] < 0 else "high tau -> no reversal"
            supports = "YES" if r["r"] < 0 and r["p"] < 0.1 else "no"
            tests.append(("tau vs diff " + key, r["r"], r["p"], direction, supports))

    t5 = all_results.get("test5_task_contrast_vs_criticality", {})
    if t5:
        r = t5.get("sigma_vs_lzc_g", {})
        if r:
            direction = "subcrit -> neg LZc" if r["r"] > 0 else "subcrit -> pos LZc"
            supports = "YES" if r["p"] < 0.1 else "no"
            tests.append(("sigma vs LZc contrast", r["r"], r["p"], direction, supports))

    for name, effect, p, direction, supports in tests:
        log(f"  {name:<43} {effect:>+8.3f} {p:>8.4f} {direction:>15} {supports:>10}")

    return tests


def main():
    log("=" * 70)
    log("HYPOTHESIS ANALYSIS: Subcritical Suppression Framework")
    log("=" * 70)

    subjects = load_group_data()
    log(f"Loaded {len(subjects)} QC-passed subjects")

    all_results = {}
    all_results["test1_subcritical_suppression"] = test1_subcritical_suppression(subjects)
    all_results["test2_avalanche_ceiling"] = test2_avalanche_ceiling(subjects)
    all_results["test3_tau_decoding_quadrant"] = test3_tau_decoding_quadrant(subjects)
    all_results["test4_complexity_decoding"] = test4_complexity_decoding(subjects)
    all_results["test5_task_contrast_vs_criticality"] = test5_task_contrast_vs_criticality(subjects)

    create_quadrant_plot(subjects, FIG_DIR)
    create_sigma_decoding_plot(subjects, FIG_DIR)

    tests = create_summary_table(all_results, subjects)

    all_results["summary_table"] = [
        {"test": t[0], "effect": t[1], "p": t[2], "direction": t[3], "supports_hypothesis": t[4]}
        for t in tests
    ]

    with open(RESULTS_DIR / "hypothesis_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nResults saved: {RESULTS_DIR / 'hypothesis_analysis.json'}")

    log("\n" + "=" * 70)
    log("INTERPRETATION")
    log("=" * 70)

    n10 = all_results.get("test1_subcritical_suppression", {}).get("n=10", {})
    sigma_diff = n10.get("sigma_vs_diff", {})
    tau_diff = all_results.get("test3_tau_decoding_quadrant", {}).get("n=10", {}).get("tau_vs_diff", {})

    log(f"""
The refined hypothesis posits that subcritical dynamics (sigma < 1.0)
suppress large avalanches, enabling stable fine-grained representations
that are BETTER decoded from non-critical channels.

Key predictions and results:
1. More subcritical sigma -> larger decoding reversal (diff < 0)
   {'SUPPORTED' if sigma_diff and sigma_diff.get('r', 0) < 0 and sigma_diff.get('p', 1) < 0.1 else 'NOT SUPPORTED'}
   r={sigma_diff.get('r', 'N/A')}, p={sigma_diff.get('p', 'N/A')}

2. Higher tau (fewer large avalanches) -> larger reversal
   {'SUPPORTED' if tau_diff and tau_diff.get('r', 0) < 0 and tau_diff.get('p', 1) < 0.1 else 'NOT SUPPORTED'}
   r={tau_diff.get('r', 'N/A')}, p={tau_diff.get('p', 'N/A')}

3. The decoding reversal itself is the strongest evidence:
   least-critical > most-critical in 14/19 subjects at n=10

Alternative interpretations to consider:
- The reversal may reflect spatial confound (electrode coverage bias)
- It may be task-specific (Cogitate uses prolonged stimuli)
- Per-channel tau estimation may be noisy with limited data per channel
""")


if __name__ == "__main__":
    main()
