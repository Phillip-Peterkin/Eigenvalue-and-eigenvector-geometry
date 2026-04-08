import os
"""Compute slope statistics from the 95 seizure trajectories console output."""
import numpy as np
from scipy import stats as sp_stats
import json
from pathlib import Path

slopes = [
    -0.000075, -0.000514, 0.000124, 0.000950, 0.003474, -0.000807, 0.000311, 0.000072,
    -0.000700, 0.003118,
    -0.004176, 0.001506, -0.001793, 0.000271,
    -0.026992, -0.002178, -0.005796,
    0.003645, 0.002243, -0.001299,
    -0.000622, 0.000081, -0.001830, -0.000453, -0.000853, 0.000993, -0.000418, 0.000326,
    0.001196, 0.001600, -0.002381,
    0.001637, -0.001394, -0.003704, 0.001667, -0.001569,
    0.000047, -0.001753, 0.000849, -0.000054,
    -0.000144, -0.000592, -0.000268, -0.001328, 0.002498, -0.000553, -0.002471,
    0.004266, -0.000851,
    -0.000663, -0.000596, 0.000944, -0.000323, -0.001655,
    0.002681, 0.000860, 0.000446,
    -0.000392, 0.001161, -0.000322, 0.004260, 0.001680,
    0.001115, -0.003021, 0.002274, 0.003199,
    -0.000380, -0.000705, -0.001925,
    0.000863, -0.005109, -0.000502,
    0.001378, -0.006934, 0.000167, -0.001628,
    0.000226, 0.000588,
    0.001032, 0.001714, -0.000608,
    -0.000440, -0.002860, -0.000347,
    -0.001691, 0.001384, 0.002821, 0.000043,
    -0.001715, 0.001442, -0.001419, 0.002919, 0.001466, 0.000780, 0.000649,
]

subject_slopes = {
    "sub-01": [-0.000075, -0.000514, 0.000124, 0.000950, 0.003474, -0.000807, 0.000311, 0.000072],
    "sub-02": [-0.000700, 0.003118],
    "sub-03": [-0.004176, 0.001506, -0.001793, 0.000271],
    "sub-04": [-0.026992, -0.002178, -0.005796],
    "sub-05": [0.003645, 0.002243, -0.001299],
    "sub-06": [-0.000622, 0.000081, -0.001830, -0.000453, -0.000853, 0.000993, -0.000418, 0.000326],
    "sub-07": [0.001196, 0.001600, -0.002381],
    "sub-08": [0.001637, -0.001394, -0.003704, 0.001667, -0.001569],
    "sub-09": [0.000047, -0.001753, 0.000849, -0.000054],
    "sub-10": [-0.000144, -0.000592, -0.000268, -0.001328, 0.002498, -0.000553, -0.002471],
    "sub-11": [0.004266, -0.000851],
    "sub-12": [-0.000663, -0.000596, 0.000944, -0.000323, -0.001655],
    "sub-13": [0.002681, 0.000860, 0.000446],
    "sub-14": [-0.000392, 0.001161, -0.000322, 0.004260, 0.001680],
    "sub-15": [0.001115, -0.003021, 0.002274, 0.003199],
    "sub-16": [-0.000380, -0.000705, -0.001925],
    "sub-17": [0.000863, -0.005109, -0.000502],
    "sub-18": [0.001378, -0.006934, 0.000167, -0.001628],
    "sub-19": [0.000226, 0.000588],
    "sub-20": [0.001032, 0.001714, -0.000608],
    "sub-22": [-0.000440, -0.002860, -0.000347],
    "sub-23": [-0.001691, 0.001384, 0.002821, 0.000043],
    "sub-24": [-0.001715, 0.001442, -0.001419, 0.002919, 0.001466, 0.000780, 0.000649],
}

slopes = np.array(slopes)
n = len(slopes)
n_neg = int(np.sum(slopes < 0))
pct_neg = n_neg / n * 100

subj_means = np.array([np.mean(v) for v in subject_slopes.values()])
n_subj = len(subj_means)
n_subj_neg = int(np.sum(subj_means < 0))
pct_subj_neg = n_subj_neg / n_subj * 100

t_stat, p_val = sp_stats.ttest_1samp(subj_means, 0.0)
p_onesided = p_val / 2 if t_stat < 0 else 1 - p_val / 2
w_stat, w_p = sp_stats.wilcoxon(subj_means, alternative="less")
d = float(np.mean(subj_means) / np.std(subj_means, ddof=1))

q25, q50, q75 = np.percentile(slopes, [25, 50, 75])

print("=" * 60)
print("SEIZURE-LEVEL PRE-ICTAL SLOPE ANALYSIS")
print("=" * 60)
print(f"Total seizures: {n}")
print(f"Negative slopes: {n_neg} / {n} = {pct_neg:.1f}%")
print(f"Positive slopes: {n - n_neg} / {n} = {100-pct_neg:.1f}%")
print(f"Mean slope: {np.mean(slopes):.6f}")
print(f"Median slope: {np.median(slopes):.6f}")
print(f"Std slope: {np.std(slopes, ddof=1):.6f}")
print(f"Range: [{np.min(slopes):.6f}, {np.max(slopes):.6f}]")
print(f"Quartiles: Q1={q25:.6f}, Q2={q50:.6f}, Q3={q75:.6f}")
print(f"IQR: {q75-q25:.6f}")
print()
print("=" * 60)
print("SUBJECT-LEVEL (averaged within subject first)")
print("=" * 60)
print(f"Subjects: {n_subj}")
print(f"Subjects with negative mean slope: {n_subj_neg}/{n_subj} = {pct_subj_neg:.1f}%")
print(f"Mean of subject means: {np.mean(subj_means):.6f}")
print(f"Median of subject means: {np.median(subj_means):.6f}")
print(f"t-test vs 0 (two-sided): t={t_stat:.3f}, p={p_val:.4f}")
print(f"t-test vs 0 (one-sided, <0): p={p_onesided:.4f}")
print(f"Wilcoxon signed-rank (one-sided, <0): W={w_stat:.1f}, p={w_p:.4f}")
print(f"Cohen d: {d:.3f}")
print()
print("=" * 60)
print("PER-SUBJECT BREAKDOWN")
print("=" * 60)
for sid, vals in sorted(subject_slopes.items()):
    m = np.mean(vals)
    neg = sum(1 for v in vals if v < 0)
    print(f"  {sid}: mean={m:+.6f}, {neg}/{len(vals)} negative")

print()
print("=" * 60)
print("GROUP STATISTICS (from console output)")
print("=" * 60)
print("t-5min: mean_z=-0.1970, p=0.5088, d=-0.143")
print("t-2min: mean_z=-0.2812, p=0.4427, d=-0.167")
print("t-1min: mean_z=-0.1057, p=0.6516, d=-0.098")
print("Cluster permutation: 10 clusters found, 0 significant")
print()
print("=" * 60)
print("SCIENTIFIC INTERPRETATION")
print("=" * 60)
print("1. Pre-ictal slope: ~50% of seizures show negative slopes")
print("   => NO consistent pre-ictal narrowing across seizures")
print("2. Group z-scores at -5, -2, -1 min are all non-significant (p>0.4)")
print("   => Spacing is NOT reliably below baseline before seizure onset")
print("3. Cluster permutation found 0 significant clusters")
print("   => No sustained period of significant spacing change pre-ictally")
print("4. The effect sizes are tiny (d < 0.2)")
print("5. sub-04 has an extreme outlier slope (-0.027) - 5x larger than")
print("   any other seizure. This could be artifact or genuine.")
print()
print("CAVEAT: Ictal minimum timing and post-ictal slope require the full")
print("trajectory arrays which only exist in the running process memory.")
print("Re-run with vectorized fix to get those statistics.")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig_dir = Path(os.environ.get("RESULTS_ROOT", "./results"))
fig_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(slopes, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
ax.axvline(0, color="red", ls="--", lw=2, label="zero")
ax.axvline(np.mean(slopes), color="orange", ls="-", lw=2, label=f"mean={np.mean(slopes):.5f}")
ax.axvline(np.median(slopes), color="green", ls=":", lw=2, label=f"median={np.median(slopes):.5f}")
ax.set_xlabel("Pre-ictal slope (z-score / sec)")
ax.set_ylabel("Count")
ax.set_title(f"Per-Seizure Pre-ictal Slopes (n={n})\n{pct_neg:.0f}% negative")
ax.legend(fontsize=8)

ax = axes[1]
ax.hist(subj_means, bins=15, color="coral", edgecolor="black", alpha=0.8)
ax.axvline(0, color="red", ls="--", lw=2, label="zero")
ax.axvline(np.mean(subj_means), color="orange", ls="-", lw=2, label=f"mean={np.mean(subj_means):.5f}")
ax.set_xlabel("Mean pre-ictal slope (z-score / sec)")
ax.set_ylabel("Count")
ax.set_title(f"Per-Subject Mean Slopes (n={n_subj})\n{pct_subj_neg:.0f}% negative, p(one-sided)={p_onesided:.3f}")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(fig_dir / "slope_analysis_from_console.png", dpi=200)
plt.close(fig)
print(f"\nFigure saved: {fig_dir / 'slope_analysis_from_console.png'}")

results = {
    "seizure_level": {
        "n_seizures": n,
        "n_negative": n_neg,
        "pct_negative": round(pct_neg, 1),
        "mean_slope": round(float(np.mean(slopes)), 7),
        "median_slope": round(float(np.median(slopes)), 7),
        "std_slope": round(float(np.std(slopes, ddof=1)), 7),
        "q25": round(float(q25), 7),
        "q75": round(float(q75), 7),
    },
    "subject_level": {
        "n_subjects": n_subj,
        "n_negative": n_subj_neg,
        "pct_negative": round(pct_subj_neg, 1),
        "mean": round(float(np.mean(subj_means)), 7),
        "median": round(float(np.median(subj_means)), 7),
        "ttest_t": round(float(t_stat), 4),
        "ttest_p_twosided": round(float(p_val), 4),
        "ttest_p_onesided": round(float(p_onesided), 4),
        "wilcoxon_W": round(float(w_stat), 1),
        "wilcoxon_p_onesided": round(float(w_p), 4),
        "cohens_d": round(d, 3),
    },
    "group_zscores_from_console": {
        "t_minus_5min": {"mean_z": -0.1970, "p": 0.5088, "d": -0.143},
        "t_minus_2min": {"mean_z": -0.2812, "p": 0.4427, "d": -0.167},
        "t_minus_1min": {"mean_z": -0.1057, "p": 0.6516, "d": -0.098},
        "cluster_permutation": {"n_clusters": 10, "n_significant": 0},
    },
}

results_dir = Path(os.environ.get("RESULTS_ROOT", "./results"))
results_dir.mkdir(parents=True, exist_ok=True)
with open(results_dir / "slope_analysis_from_console.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved: {results_dir / 'slope_analysis_from_console.json'}")
