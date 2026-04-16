"""Operator-geometry brain-state test battery.

Runs eight quantitative tests evaluating whether operator geometry
(eigenvalue gap, condition number, ND score, spectral radius)
constitutes a sufficient, independent, and structured coordinate
system for brain state.

All analyses operate on pre-computed JSON summary statistics.
No raw EEG processing is performed.

Unit of observation: per-subject condition means.
Split strategy: leave-one-subject-out (LOSO).
Random seed: 42 throughout.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from cmcc.analysis.geometry_embedding import (
    extract_propofol_features,
    extract_sleep_features,
    classify_states_loso,
    analyze_geometric_structure,
    compare_geometry_vs_power,
    check_orthogonality,
    collate_existing_results,
    assemble_test_battery,
    load_temporal_precedence_summary,
    GeometryFeatureTable,
    TemporalPrecedenceSummary,
)

PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
RESULTS_JSON = REPO_ROOT / "results" / "json_results"
FIG_DIR = REPO_ROOT / "results" / "figures" / "geometry_embedding"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOTSTRAP = 1000
N_NULL_PERMUTATIONS = 100
DPI = 300

STATE_PALETTE = {
    "awake": "#4477AA",
    "propofol": "#EE6677",
    "N3": "#228833",
    "REM": "#CCBB44",
}


def log(msg: str) -> None:
    print(msg, flush=True)


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


def load_json(name: str) -> dict:
    path = RESULTS_JSON / name
    with open(path) as f:
        return json.load(f)


def _select_binary(table: GeometryFeatureTable, cond_a: str, cond_b: str):
    mask = np.array([(c == cond_a or c == cond_b) for c in table.conditions])
    features = table.features[mask]
    labels = np.array([0 if c == cond_a else 1 for c in np.array(table.conditions)[mask]])
    subjects = np.array(table.subjects)[mask]
    alpha = table.alpha_power[mask] if table.alpha_power is not None else None
    return features, labels, subjects, alpha


def run_sufficiency(table: GeometryFeatureTable, cond_a: str, cond_b: str, name: str):
    features, labels, subjects, _ = _select_binary(table, cond_a, cond_b)
    return classify_states_loso(
        features, labels, subjects,
        contrast_name=name,
        dataset=table.dataset,
        feature_names=table.feature_names,
        seed=SEED,
        n_bootstrap=N_BOOTSTRAP,
        n_null_permutations=N_NULL_PERMUTATIONS,
    )


def run_incremental(table: GeometryFeatureTable, cond_a: str, cond_b: str):
    features, labels, subjects, alpha = _select_binary(table, cond_a, cond_b)
    if alpha is None or not np.all(np.isfinite(alpha)):
        return None
    power = alpha.reshape(-1, 1)
    return compare_geometry_vs_power(
        features, power, labels, subjects,
        seed=SEED, n_bootstrap=N_BOOTSTRAP,
    )


def run_orthogonality(table: GeometryFeatureTable, cond_a: str, cond_b: str):
    features, labels, subjects, alpha = _select_binary(table, cond_a, cond_b)
    if alpha is None or not np.all(np.isfinite(alpha)):
        return None
    power = alpha.reshape(-1, 1)
    return check_orthogonality(
        features, power, labels,
        feature_names=table.feature_names,
        seed=SEED,
    )


def run_structure(table: GeometryFeatureTable):
    return analyze_geometric_structure(
        table.features,
        np.array(table.conditions),
        np.array(table.subjects),
        feature_names=table.feature_names,
        seed=SEED,
        n_bootstrap=N_BOOTSTRAP,
    )


def _load_amplification_r() -> float | None:
    """Extract condition_number vs Kreiss r from transient_amplification.json.

    Source: CMCC pipeline cross-subject correlations (n=18 propofol subjects).
    Falls back to operator-geometry-brain-states results if CMCC path unavailable.
    """
    candidates = [
        REPO_ROOT.parent / "CMCC" / "results" / "analysis" / "transient_amplification.json",
        RESULTS_JSON / "transient_amplification.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            try:
                r = data["cross_subject_correlations"]["condition_number_vs_kreiss"]["r"]
                log(f"  Amplification r={r:.4f} loaded from {path.name}")
                return float(r)
            except (KeyError, TypeError):
                continue
    return None


def run_collation():
    results_map = {}
    for name in ["exceptional_points", "jackknife_sensitivity",
                 "ep_shared_subspace_propofol", "ep_shared_subspace_sleep"]:
        try:
            results_map[name] = load_json(f"{name}.json")
        except FileNotFoundError:
            warnings.warn(f"Missing {name}.json for collation.", stacklevel=2)

    amp_r = _load_amplification_r()
    if amp_r is not None:
        results_map["condition_number_vs_kreiss_r"] = amp_r
    else:
        warnings.warn(
            "Could not load condition_number_vs_kreiss_r from transient_amplification.json. "
            "Test 8 (amplification link) will be marked as missing.",
            stacklevel=2,
        )

    return collate_existing_results(results_map)


def run_outlier_sensitivity(table: GeometryFeatureTable, cond_a: str, cond_b: str, name: str):
    features, labels, subjects, _ = _select_binary(table, cond_a, cond_b)
    cond_num_idx = table.feature_names.index("condition_number")
    cond_nums = features[:, cond_num_idx]
    threshold = np.percentile(np.abs(cond_nums), 99)
    keep = np.abs(cond_nums) <= threshold
    if keep.sum() < 6:
        return None
    return classify_states_loso(
        features[keep], labels[keep], subjects[keep],
        contrast_name=f"{name}_outlier_trimmed",
        dataset=table.dataset,
        feature_names=table.feature_names,
        seed=SEED,
        n_bootstrap=100,
        n_null_permutations=10,
    )


def _confidence_ellipse(data, ax, n_std=1.96, **kwargs):
    if data.shape[0] < 3:
        return
    cov = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.abs(eigenvalues))
    ellipse = Ellipse(
        xy=data.mean(axis=0),
        width=width, height=height, angle=angle,
        **kwargs,
    )
    ax.add_patch(ellipse)


def plot_pca_scatter(table: GeometryFeatureTable, output_path: Path):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=SEED)
    X = table.features.copy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-15)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for cond in sorted(set(table.conditions)):
        mask = np.array(table.conditions) == cond
        color = STATE_PALETTE.get(cond, "#999999")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=cond, s=60, alpha=0.8, edgecolors="white", linewidth=0.5)
        _confidence_ellipse(coords[mask], ax, facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.5)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"Geometry-only PCA — {table.dataset}")
    ax.legend(framealpha=0.9)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_auc_bars(sufficiency_results, output_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r.contrast_name for r in sufficiency_results]
    aucs = [r.auc_loso for r in sufficiency_results]
    ci_low = [r.auc_ci_lower for r in sufficiency_results]
    ci_high = [r.auc_ci_upper for r in sufficiency_results]

    x = np.arange(len(names))
    colors = ["#4477AA" if r.passes_threshold else "#BB5566" for r in sufficiency_results]
    yerr_low = [a - l for a, l in zip(aucs, ci_low)]
    yerr_high = [h - a for a, h in zip(aucs, ci_high)]

    ax.bar(x, aucs, color=colors, alpha=0.8, edgecolor="white")
    ax.errorbar(x, aucs, yerr=[yerr_low, yerr_high], fmt="none", ecolor="black", capsize=4)
    ax.axhline(0.80, color="red", linestyle="--", linewidth=1, label="Threshold (0.80)")
    ax.axhline(0.50, color="gray", linestyle=":", linewidth=0.8, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("LOSO AUC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test 1: Sufficiency — Geometry-only classification")
    ax.legend(loc="lower right")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_geometry_vs_power(table: GeometryFeatureTable, output_path: Path):
    if table.alpha_power is None:
        return
    alpha = table.alpha_power
    fig, axes = plt.subplots(1, len(table.feature_names), figsize=(4 * len(table.feature_names), 4))
    if len(table.feature_names) == 1:
        axes = [axes]
    for i, fname in enumerate(table.feature_names):
        ax = axes[i]
        for cond in sorted(set(table.conditions)):
            mask = np.array(table.conditions) == cond
            color = STATE_PALETTE.get(cond, "#999999")
            ax.scatter(alpha[mask], table.features[mask, i],
                       c=color, label=cond, s=40, alpha=0.7)
        ax.set_xlabel("Alpha power")
        ax.set_ylabel(fname)
        ax.legend(fontsize=7)
        sns.despine(ax=ax)
    fig.suptitle("Test 3: Geometry vs power", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_state_vectors(structure, output_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for state, vec in structure.state_change_vectors.items():
        if len(vec) < 2:
            continue
        color = STATE_PALETTE.get(state, "#999999")
        ax.annotate("", xy=(vec[0], vec[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
        ax.text(vec[0] * 1.05, vec[1] * 1.05, state, color=color, fontsize=10, fontweight="bold")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Δ eigenvalue gap")
    ax.set_ylabel("Δ condition number")
    ax.set_title("Test 4: State-change vectors from awake baseline")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_mahalanobis_heatmap(structure, output_path: Path):
    pairs = structure.pairwise_mahalanobis
    if not pairs:
        return
    states = sorted(set(s for pair in pairs for s in pair.split("_vs_")))
    n = len(states)
    mat = np.zeros((n, n))
    for pair, val in pairs.items():
        s1, s2 = pair.split("_vs_")
        i, j = states.index(s1), states.index(s2)
        mat[i, j] = val
        mat[j, i] = val

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mat, annot=True, fmt=".2f", xticklabels=states, yticklabels=states,
                cmap="YlOrRd", ax=ax, vmin=0, square=True, linewidths=0.5)
    ax.set_title("Test 4: Pairwise Mahalanobis distances")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_scorecard(battery, output_path: Path):
    tests = [
        ("1. Sufficiency", any(s.passes_threshold for s in battery.sufficiency),
         f"Best AUC: {max(s.auc_loso for s in battery.sufficiency):.3f}"),
        ("2. Incremental", battery.incremental.passes_threshold if battery.incremental else None,
         f"dAUC: {battery.incremental.delta_auc_vs_power:.3f}" if battery.incremental else "N/A (no power)"),
        ("3. Orthogonality", battery.orthogonality.passes_threshold if battery.orthogonality else None,
         f"|r|={battery.orthogonality.median_abs_correlation:.3f}, "
         f"{battery.orthogonality.n_primary_features_passing_d}/3 d>=0.5, "
         f"sens={'PASS' if battery.orthogonality.passes_sensitivity_threshold else 'FAIL'}"
         if battery.orthogonality else "N/A"),
        ("4. Structure", battery.structure.passes_threshold,
         f"min Mah: {min(battery.structure.pairwise_mahalanobis.values()):.2f}" if battery.structure.pairwise_mahalanobis else "N/A"),
        ("5. Stability", all(s.passes for s in battery.stability) if battery.stability else None,
         f"{sum(s.passes for s in battery.stability)}/{len(battery.stability)} pass"),
        ("6. Criticality", all(c.passes for c in battery.criticality_coupling) if battery.criticality_coupling else None,
         f"{sum(c.passes for c in battery.criticality_coupling)}/{len(battery.criticality_coupling)} pass"),
        ("7. Temporal",
         battery.temporal_precedence.any_metric_passes_strict
         if isinstance(battery.temporal_precedence, TemporalPrecedenceSummary)
         else None,
         f"strict p={battery.temporal_precedence.best_slope_p:.4f}, "
         f"d={battery.temporal_precedence.best_early_late_d:.2f}, "
         f"lenient={'PASS' if battery.temporal_precedence.any_metric_passes else 'FAIL'}"
         if isinstance(battery.temporal_precedence, TemporalPrecedenceSummary)
         else "DEFERRED"),
        ("8. Amplification", all(a.passes for a in battery.amplification_link) if battery.amplification_link else None,
         f"{sum(a.passes for a in battery.amplification_link)}/{len(battery.amplification_link)} pass"),
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    y_start = 0.92
    for i, (name, passed, detail) in enumerate(tests):
        y = y_start - i * 0.10
        if passed is True:
            symbol, color = "✓ PASS", "#228833"
        elif passed is False:
            symbol, color = "✗ FAIL", "#BB5566"
        else:
            symbol, color = "— N/A", "#888888"
        ax.text(0.02, y, name, fontsize=12, fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(0.35, y, symbol, fontsize=12, color=color, fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(0.55, y, detail, fontsize=10, color="#333333", transform=ax.transAxes, va="center")

    verdict_color = {"strong": "#228833", "complementary": "#CCBB44", "insufficient": "#BB5566"}
    ax.text(0.02, y_start - len(tests) * 0.10 - 0.03, f"VERDICT: {battery.overall_verdict.upper()}",
            fontsize=14, fontweight="bold",
            color=verdict_color.get(battery.overall_verdict, "#333333"),
            transform=ax.transAxes)
    ax.text(0.55, y_start - len(tests) * 0.10 - 0.03,
            f"({battery.n_tests_passed}/{battery.n_tests_total} tests passed)",
            fontsize=11, transform=ax.transAxes)

    ax.set_title("Operator-Geometry Brain-State Test Battery", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.time()
    log("=" * 70)
    log("OPERATOR-GEOMETRY BRAIN-STATE TEST BATTERY")
    log("=" * 70)

    log("\n[1/7] Loading JSON results...")
    ep_prop = load_json("ep_propofol_eeg.json")
    amp_prop = load_json("amplification_propofol.json")
    ep_sleep = load_json("ep_sleep_dynamics.json")
    amp_sleep = load_json("amplification_sleep_convergence.json")

    log("[2/7] Extracting features...")
    prop_table = extract_propofol_features(ep_prop, amp_prop)
    sleep_table = extract_sleep_features(ep_sleep, amp_sleep)
    log(f"  Propofol: {prop_table.features.shape[0]} obs, {len(set(prop_table.subjects))} subjects")
    log(f"  Sleep: {sleep_table.features.shape[0]} obs, {len(set(sleep_table.subjects))} subjects")
    if prop_table.excluded_subjects:
        log(f"  Propofol excluded: {prop_table.excluded_subjects}")
    if sleep_table.excluded_subjects:
        log(f"  Sleep excluded: {sleep_table.excluded_subjects}")

    log("\n[3/7] Test 1: Sufficiency (LOSO classification)...")
    contrasts = [
        (prop_table, "awake", "propofol", "awake_vs_propofol"),
        (sleep_table, "awake", "N3", "awake_vs_N3"),
        (sleep_table, "N3", "REM", "N3_vs_REM"),
        (sleep_table, "awake", "REM", "awake_vs_REM"),
    ]
    sufficiency_results = []
    for tbl, ca, cb, name in contrasts:
        log(f"  {name}...")
        result = run_sufficiency(tbl, ca, cb, name)
        log(f"    AUC={result.auc_loso:.3f} [{result.auc_ci_lower:.3f}, {result.auc_ci_upper:.3f}] "
            f"{'PASS' if result.passes_threshold else 'FAIL'} "
            f"(null={result.null_auc_mean:.3f}+/-{result.null_auc_std:.3f}, p={result.null_auc_p:.3f})")
        sufficiency_results.append(result)

    log("\n[4/7] Tests 2-3: Incremental value + Orthogonality (propofol)...")
    incremental = run_incremental(prop_table, "awake", "propofol")
    if incremental:
        log(f"  Geom AUC={incremental.auc_geometry_only:.3f}, "
            f"Power AUC={incremental.auc_power_only:.3f}, "
            f"Combined={incremental.auc_combined:.3f}, "
            f"dAUC={incremental.delta_auc_vs_power:.3f} "
            f"{'PASS' if incremental.passes_threshold else 'FAIL'}")
    else:
        log("  Skipped (no power data)")

    orthogonality = run_orthogonality(prop_table, "awake", "propofol")
    if orthogonality:
        strict_str = "PASS" if orthogonality.passes_threshold else "FAIL"
        sens_str = "PASS" if orthogonality.passes_sensitivity_threshold else "FAIL"
        log(f"  Median |r|={orthogonality.median_abs_correlation:.3f}")
        log(f"    Primary (threshold=0.20): {strict_str}")
        log(f"    Sensitivity (threshold=0.30): {sens_str}")
        log(f"    Primary features with |d|>=0.5: {orthogonality.n_primary_features_passing_d}/3")
        for feat, d in orthogonality.residualized_effect_sizes.items():
            log(f"    Residualized d({feat})={d:.3f}")
        log(f"    Interpretation: {orthogonality.interpretation}")
    else:
        log("  Orthogonality skipped (no power data)")

    log("\n[5/7] Test 4: Structure (multi-dimensional geometric state space)...")
    structure_prop = run_structure(prop_table)
    structure_sleep = run_structure(sleep_table)
    for label, struct in [("Propofol", structure_prop), ("Sleep", structure_sleep)]:
        log(f"  {label}:")
        for pair, d in struct.pairwise_mahalanobis.items():
            log(f"    Mahalanobis({pair})={d:.3f}")
        for pair, angle in struct.angular_separations.items():
            ci = struct.angular_separation_cis.get(pair, (float("nan"), float("nan")))
            log(f"    Angle({pair})={angle:.1f}° [{ci[0]:.1f}, {ci[1]:.1f}]")
        for state, cons in struct.subject_consistency.items():
            log(f"    Subject consistency({state})={cons:.2f}")

    log("\n[6/8] Tests 5,6,8: Collating existing results...")
    stability, criticality, amplification = run_collation()
    for label, results_list in [("Stability", stability), ("Criticality", criticality), ("Amplification", amplification)]:
        for r in results_list:
            log(f"  {label}: {r.metric_name}={r.value:.3f} (threshold={r.threshold:.3f}) {'PASS' if r.passes else 'FAIL'}")

    log("\n[7/8] Test 7: Temporal precedence...")
    temporal_json_path = RESULTS_JSON / "temporal_precedence.json"
    temporal_summary = "DEFERRED"
    if temporal_json_path.exists():
        with open(temporal_json_path) as f:
            temporal_data = json.load(f)
        temporal_summary = load_temporal_precedence_summary(temporal_data)
        if isinstance(temporal_summary, TemporalPrecedenceSummary):
            status = "PASS" if temporal_summary.any_metric_passes else "FAIL"
            log(f"  Loaded from {temporal_json_path}")
            log(f"  Best metric: {temporal_summary.best_metric}")
            log(f"  Best slope p: {temporal_summary.best_slope_p:.4f}")
            log(f"  Best consistency: {temporal_summary.best_consistency:.0%}")
            log(f"  Best early-late d: {temporal_summary.best_early_late_d:.3f}")
            log(f"  Status: {status}")
        else:
            log(f"  Temporal precedence data incomplete — DEFERRED")
    else:
        log(f"  No temporal_precedence.json found — DEFERRED")
        log(f"  Run _temporal_precedence.py first to generate results.")

    log("\n[8/8] Validation: outlier sensitivity + cross-dataset...")
    validation = {}
    for tbl, ca, cb, name in contrasts[:1]:
        sens = run_outlier_sensitivity(tbl, ca, cb, name)
        if sens:
            original_auc = sufficiency_results[0].auc_loso
            delta = abs(sens.auc_loso - original_auc)
            log(f"  Outlier sensitivity ({name}): original={original_auc:.3f}, trimmed={sens.auc_loso:.3f}, delta={delta:.3f}")
            validation["outlier_sensitivity"] = {
                "original_auc": original_auc,
                "trimmed_auc": sens.auc_loso,
                "delta": delta,
                "stable": delta < 0.05,
            }

    cross_dataset = {}
    for r in criticality:
        cross_dataset[r.metric_name] = r.value
    validation["cross_dataset_criticality"] = cross_dataset

    for suf in sufficiency_results:
        validation[f"subject_consistency_{suf.contrast_name}"] = suf.subject_consistency

    battery = assemble_test_battery(
        sufficiency_results,
        incremental,
        orthogonality,
        structure_prop,
        stability, criticality, amplification,
        temporal=temporal_summary,
    )

    log(f"\n{'='*70}")
    log(f"VERDICT: {battery.overall_verdict.upper()} ({battery.n_tests_passed}/{battery.n_tests_total} tests passed)")
    log(f"{'='*70}")

    log("\nGenerating figures...")
    plot_pca_scatter(prop_table, FIG_DIR / "pca_scatter_propofol.png")
    plot_pca_scatter(sleep_table, FIG_DIR / "pca_scatter_sleep.png")
    plot_auc_bars(sufficiency_results, FIG_DIR / "auc_bars.png")
    plot_geometry_vs_power(prop_table, FIG_DIR / "geometry_vs_power.png")
    plot_state_vectors(structure_prop, FIG_DIR / "state_vectors_propofol.png")
    plot_state_vectors(structure_sleep, FIG_DIR / "state_vectors_sleep.png")
    plot_mahalanobis_heatmap(structure_prop, FIG_DIR / "mahalanobis_propofol.png")
    plot_mahalanobis_heatmap(structure_sleep, FIG_DIR / "mahalanobis_sleep.png")
    plot_scorecard(battery, FIG_DIR / "scorecard.png")
    log(f"  Figures saved to {FIG_DIR}")

    out = {
        "analysis": "geometry_brain_state_tests",
        "parameters": {
            "seed": SEED,
            "n_bootstrap": N_BOOTSTRAP,
            "n_null_permutations": N_NULL_PERMUTATIONS,
            "loso_classifier": "LDA",
            "geometry_features": prop_table.feature_names,
            "thresholds": {
                "sufficiency_auc": 0.80,
                "incremental_delta_auc": 0.05,
                "incremental_combined_delta": 0.03,
                "orthogonality_median_r": 0.20,
                "orthogonality_residualized_d": 0.50,
                "structure_mahalanobis": 1.5,
                "stability_shrinkage": 0.25,
                "criticality_r": 0.80,
                "amplification_r": 0.80,
            },
        },
        "test_1_sufficiency": {
            "contrasts": [asdict(r) for r in sufficiency_results],
        },
        "test_2_incremental": asdict(incremental) if incremental else None,
        "test_3_orthogonality": asdict(orthogonality) if orthogonality else None,
        "test_4_structure": {
            "propofol": asdict(structure_prop),
            "sleep": asdict(structure_sleep),
        },
        "test_5_stability": [asdict(s) for s in stability],
        "test_6_criticality": [asdict(c) for c in criticality],
        "test_7_temporal": asdict(battery.temporal_precedence)
            if isinstance(battery.temporal_precedence, TemporalPrecedenceSummary)
            else "DEFERRED",
        "test_8_amplification": [asdict(a) for a in amplification],
        "validation": validation,
        "verdict": {
            "n_passed": battery.n_tests_passed,
            "n_total": battery.n_tests_total,
            "conclusion": battery.overall_verdict,
        },
    }

    out_path = RESULTS_JSON / "geometry_brain_states.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=default_ser)
    log(f"\nResults written to {out_path}")

    elapsed = time.time() - t0
    log(f"\nTotal time: {elapsed:.1f}s")
    log("DONE")


if __name__ == "__main__":
    main()
