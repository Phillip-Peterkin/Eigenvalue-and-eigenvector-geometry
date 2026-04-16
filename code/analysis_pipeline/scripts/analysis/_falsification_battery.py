"""Falsification battery orchestration script.

Runs seven categories of falsification attacks against the operator-geometry
brain-state claim, using pre-computed JSON summary statistics.

Categories:
1. Label destruction (state-label shuffle)
2. Subject jackknife and leave-k-out
3. Feature ablation
4. Stronger spectral confound checks (delta for sleep)
5. Window-parameter attacks (temporal decimation — limited, see below)
6. Simple-model competition
7. Temporal precedence jackknife (LOO and L2O on slopes)

NOT feasible from pre-computed JSONs (requires raw data re-processing):
- Circular shift null (needs per-subject timecourses, not saved in JSON)
- Pseudo-transition control (needs sleep staging epoch lists)
- Temporal decimation of raw windowed timeseries

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
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from cmcc.analysis.geometry_embedding import (
    extract_propofol_features,
    extract_sleep_features,
    GeometryFeatureTable,
)
from cmcc.analysis.falsification import (
    _loso_auc,
    run_label_shuffle,
    run_classification_jackknife,
    run_temporal_jackknife,
    run_leave_two_out,
    run_feature_ablation,
    run_spectral_confound_check,
    run_temporal_decimation,
    run_model_competition,
)

PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
RESULTS_JSON = REPO_ROOT / "results" / "json_results"
FIG_DIR = REPO_ROOT / "results" / "figures" / "falsification"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_PERMUTATIONS = 1000
DPI = 300


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


def _extract_sleep_delta(ep_sleep: dict) -> dict[str, dict[str, float]]:
    """Extract per-subject per-state mean delta power from sleep JSON.

    Returns dict[subject_id][state_label] = delta_power.
    State labels mapped: W -> awake, N3 -> N3, R -> REM.
    """
    state_map = {"W": "awake", "N3": "N3", "R": "REM"}
    result: dict[str, dict[str, float]] = {}
    for r in ep_sleep["per_state_results"]:
        sid = r["subject"]
        state = r["state"]
        label = state_map.get(state)
        if label is None:
            continue
        delta = r.get("gap_vs_delta", {}).get("mean_delta_power")
        if delta is not None:
            if sid not in result:
                result[sid] = {}
            result[sid][label] = delta
    return result


def _build_delta_array(table: GeometryFeatureTable, delta_map: dict) -> np.ndarray | None:
    """Build delta_power array aligned to feature table rows."""
    vals = []
    for subj, cond in zip(table.subjects, table.conditions):
        d = delta_map.get(subj, {}).get(cond)
        vals.append(d if d is not None else float("nan"))
    arr = np.array(vals, dtype=float)
    if np.all(np.isnan(arr)):
        return None
    return arr


def plot_null_histogram(real_val, null_vals, title, xlabel, output_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_vals, bins=40, alpha=0.7, color="#4477AA", edgecolor="white", density=True)
    ax.axvline(real_val, color="#EE6677", linewidth=2.5, linestyle="--", label=f"Real = {real_val:.3f}")
    p = float(np.mean([n >= real_val for n in null_vals]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"{title}\n(empirical p = {p:.4f})")
    ax.legend()
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_jackknife_stability(full_val, loo_vals, loo_subjects, title, ylabel, threshold, output_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(loo_vals))
    colors = ["#228833" if v >= threshold else "#BB5566" for v in loo_vals]
    ax.bar(x, loo_vals, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(full_val, color="#4477AA", linewidth=2, linestyle="-", label=f"Full = {full_val:.3f}")
    ax.axhline(threshold, color="#EE6677", linewidth=1.5, linestyle="--", label=f"Threshold = {threshold:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(loo_subjects, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bars(ablation_result, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    single = ablation_result.single_feature_aucs
    names = list(single.keys())
    vals = list(single.values())
    x = np.arange(len(names))
    axes[0].bar(x, vals, color="#4477AA", alpha=0.8, edgecolor="white")
    axes[0].axhline(ablation_result.full_auc, color="#EE6677", linewidth=2, linestyle="--",
                     label=f"Full model = {ablation_result.full_auc:.3f}")
    axes[0].axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    axes[0].set_ylabel("LOSO AUC")
    axes[0].set_title("Single-feature classification")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=8)
    sns.despine(ax=axes[0])

    loo = ablation_result.leave_one_out_aucs
    loo_names = list(loo.keys())
    loo_vals = list(loo.values())
    x2 = np.arange(len(loo_names))
    axes[1].bar(x2, loo_vals, color="#CCBB44", alpha=0.8, edgecolor="white")
    axes[1].axhline(ablation_result.full_auc, color="#EE6677", linewidth=2, linestyle="--",
                     label=f"Full model = {ablation_result.full_auc:.3f}")
    axes[1].axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels([n.replace("without_", "-") for n in loo_names], rotation=30, ha="right", fontsize=9)
    axes[1].set_ylabel("LOSO AUC")
    axes[1].set_title("Leave-one-feature-out classification")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    sns.despine(ax=axes[1])

    fig.suptitle(f"Feature Ablation — {ablation_result.contrast_name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confound_heatmap(confound_results, output_path):
    datasets = []
    spectral_names = []
    geom_names = set()
    for cr in confound_results:
        datasets.append(f"{cr.dataset}_{cr.spectral_feature_name}")
        spectral_names.append(cr.spectral_feature_name)
        geom_names.update(cr.per_geometry_correlations.keys())
    geom_names = sorted(geom_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    corr_mat = np.full((len(confound_results), len(geom_names)), np.nan)
    d_mat = np.full((len(confound_results), len(geom_names)), np.nan)
    for i, cr in enumerate(confound_results):
        for j, gn in enumerate(geom_names):
            corr_mat[i, j] = cr.per_geometry_correlations.get(gn, np.nan)
            d_mat[i, j] = cr.per_geometry_residualized_d.get(gn, np.nan)

    sns.heatmap(corr_mat, annot=True, fmt=".2f", xticklabels=geom_names,
                yticklabels=datasets, cmap="RdBu_r", center=0, ax=axes[0],
                vmin=-1, vmax=1, linewidths=0.5)
    axes[0].set_title("Correlation (geometry x spectral)")

    sns.heatmap(np.abs(d_mat), annot=True, fmt=".2f", xticklabels=geom_names,
                yticklabels=datasets, cmap="YlOrRd", ax=axes[1],
                vmin=0, linewidths=0.5)
    axes[1].set_title("|Cohen's d| after residualization")

    fig.suptitle("Spectral Confound Map", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_model_competition(comp_results, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x_offset = 0
    group_positions = []
    group_labels = []

    for comp in comp_results:
        names = ["Geometry"] + list(comp.baseline_aucs.keys())
        aucs = [comp.geometry_auc] + list(comp.baseline_aucs.values())
        colors = ["#4477AA"] + ["#999999"] * len(comp.baseline_aucs)

        x = np.arange(len(names)) + x_offset
        ax.bar(x, aucs, color=colors, alpha=0.8, edgecolor="white", width=0.7)
        group_positions.append(x_offset + len(names) / 2 - 0.5)
        group_labels.append(comp.contrast_name)
        x_offset += len(names) + 1

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylabel("LOSO AUC")
    ax.set_title("Model Competition: Geometry vs Simple Baselines")
    ax.set_ylim(0, 1.05)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_scorecard(results_summary, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    tests = [
        ("1. Label Destruction", results_summary.get("label_destruction", {})),
        ("2. Subject Jackknife", results_summary.get("subject_jackknife", {})),
        ("3. Feature Ablation", results_summary.get("feature_ablation", {})),
        ("4. Spectral Confounds", results_summary.get("spectral_confounds", {})),
        ("5. Window Attacks", results_summary.get("window_attacks", {})),
        ("6. Model Competition", results_summary.get("model_competition", {})),
        ("7. Temporal Jackknife", results_summary.get("temporal_jackknife", {})),
    ]

    y_start = 0.92
    for i, (name, info) in enumerate(tests):
        y = y_start - i * 0.11
        verdict = info.get("verdict", "N/A")
        detail = info.get("detail", "")

        if verdict == "SURVIVES":
            symbol, color = "SURVIVES", "#228833"
        elif verdict == "PARTIAL":
            symbol, color = "PARTIAL", "#CCBB44"
        elif verdict == "FAILS":
            symbol, color = "FAILS", "#BB5566"
        else:
            symbol, color = "N/A", "#888888"

        ax.text(0.02, y, name, fontsize=11, fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(0.30, y, symbol, fontsize=11, color=color, fontweight="bold", transform=ax.transAxes, va="center")
        ax.text(0.45, y, detail, fontsize=9, color="#333333", transform=ax.transAxes, va="center")

    n_survive = sum(1 for _, info in tests if info.get("verdict") == "SURVIVES")
    n_partial = sum(1 for _, info in tests if info.get("verdict") == "PARTIAL")
    n_total = sum(1 for _, info in tests if info.get("verdict") in ("SURVIVES", "PARTIAL", "FAILS"))
    summary = f"{n_survive} SURVIVE, {n_partial} PARTIAL, {n_total - n_survive - n_partial} FAIL out of {n_total}"
    ax.text(0.02, y_start - len(tests) * 0.11 - 0.04, f"SUMMARY: {summary}",
            fontsize=12, fontweight="bold", transform=ax.transAxes)

    ax.set_title("Falsification Battery Scorecard", fontsize=14, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.time()
    log("=" * 70)
    log("FALSIFICATION BATTERY")
    log("=" * 70)

    log("\n[1/8] Loading JSON results and extracting features...")
    ep_prop = load_json("ep_propofol_eeg.json")
    amp_prop = load_json("amplification_propofol.json")
    ep_sleep = load_json("ep_sleep_dynamics.json")
    amp_sleep = load_json("amplification_sleep_convergence.json")
    temporal_data = load_json("temporal_precedence.json")

    prop_table = extract_propofol_features(ep_prop, amp_prop)
    sleep_table = extract_sleep_features(ep_sleep, amp_sleep)
    log(f"  Propofol: {prop_table.features.shape[0]} obs, {len(set(prop_table.subjects))} subjects")
    log(f"  Sleep: {sleep_table.features.shape[0]} obs, {len(set(sleep_table.subjects))} subjects")

    delta_map = _extract_sleep_delta(ep_sleep)
    delta_arr = _build_delta_array(sleep_table, delta_map)
    if delta_arr is not None:
        n_valid = int(np.sum(np.isfinite(delta_arr)))
        log(f"  Sleep delta power: {n_valid}/{len(delta_arr)} valid values")
    else:
        log("  Sleep delta power: NOT AVAILABLE")

    contrasts = [
        (prop_table, "awake", "propofol", "awake_vs_propofol"),
        (sleep_table, "N3", "REM", "N3_vs_REM"),
    ]

    output = {
        "analysis": "falsification_battery",
        "seed": SEED,
        "n_permutations": N_PERMUTATIONS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    summary = {}

    log("\n[2/8] Label-destruction tests (state-label shuffle)...")
    label_results = {}
    for tbl, ca, cb, name in contrasts:
        features, labels, subjects, _ = _select_binary(tbl, ca, cb)
        log(f"  {name}: {len(labels)} obs, {len(np.unique(subjects))} subjects, "
            f"{N_PERMUTATIONS} permutations...")
        result = run_label_shuffle(features, labels, subjects,
                                   contrast_name=name,
                                   n_permutations=N_PERMUTATIONS,
                                   seed=SEED)
        log(f"    Real AUC={result.real_auc:.3f}, p={result.empirical_p:.4f}, "
            f"{'SURVIVES' if result.survives else 'FAILS'}")
        label_results[name] = asdict(result)

        plot_null_histogram(
            result.real_auc, result.null_aucs,
            f"Label Shuffle — {name}",
            "AUC",
            FIG_DIR / f"label_shuffle_{name}.png",
        )

    all_survive_labels = all(r["survives"] for r in label_results.values())
    output["label_destruction"] = label_results
    summary["label_destruction"] = {
        "verdict": "SURVIVES" if all_survive_labels else "FAILS",
        "detail": "; ".join(f"{k}: p={v['empirical_p']:.4f}" for k, v in label_results.items()),
    }

    log("\n[3/8] Subject jackknife (classification LOO)...")
    jk_results = {}
    for tbl, ca, cb, name in contrasts:
        features, labels, subjects, _ = _select_binary(tbl, ca, cb)
        result = run_classification_jackknife(features, labels, subjects,
                                              result_name=name, seed=SEED)
        log(f"  {name}: full={result.full_value:.3f}, "
            f"range=[{result.loo_min:.3f}, {result.loo_max:.3f}], "
            f"most influential={result.most_influential_subject} "
            f"(delta={result.max_influence:.3f})")
        log(f"    Sign preserved: {result.sign_preserved_all}, "
            f"Significance preserved (>=0.80): {result.significance_preserved_all}")
        jk_results[name] = asdict(result)

        if result.loo_values:
            plot_jackknife_stability(
                result.full_value, result.loo_values, result.loo_subjects,
                f"Jackknife — {name}", "LOSO AUC (LOO)", 0.80,
                FIG_DIR / f"jackknife_classification_{name}.png",
            )

    all_stable_jk = all(r["sign_preserved_all"] for r in jk_results.values())
    output["subject_jackknife_classification"] = jk_results
    summary["subject_jackknife"] = {
        "verdict": "SURVIVES" if all_stable_jk else "PARTIAL",
        "detail": "; ".join(
            f"{k}: range=[{v['loo_min']:.3f},{v['loo_max']:.3f}], influential={v['most_influential_subject']}"
            for k, v in jk_results.items()
        ),
    }

    log("\n[4/8] Temporal precedence jackknife (LOO + L2O on slopes)...")
    temporal_jk_results = {}
    n2_n3 = temporal_data.get("transitions", {}).get("N2_to_N3", {})
    sr_data = n2_n3.get("spectral_radius", {})

    TEMPORAL_SUBJECTS = [
        "EPCTL01", "EPCTL03", "EPCTL05", "EPCTL07", "EPCTL10",
        "EPCTL14", "EPCTL17", "EPCTL20", "EPCTL24", "EPCTL28",
    ]

    if sr_data and sr_data.get("passes_strict"):
        slopes = sr_data["per_subject_slopes"]
        if len(slopes) == len(TEMPORAL_SUBJECTS):
            subjects_list = TEMPORAL_SUBJECTS
        else:
            warnings.warn(
                f"Expected {len(TEMPORAL_SUBJECTS)} slopes, got {len(slopes)}. "
                f"Using generic IDs — most_influential_subject will be approximate.",
                stacklevel=2,
            )
            subjects_list = [f"subj_{i:02d}" for i in range(len(slopes))]
        slope_dict = dict(zip(subjects_list, slopes))

        jk = run_temporal_jackknife(slope_dict, result_name="spectral_radius_N2_to_N3")
        log(f"  spectral_radius N2->N3 LOO: full={jk.full_value:.4e}, "
            f"range=[{jk.loo_min:.4e}, {jk.loo_max:.4e}]")
        log(f"    Sign preserved: {jk.sign_preserved_all}, "
            f"Significance preserved: {jk.significance_preserved_all}")
        log(f"    Most influential: {jk.most_influential_subject} (delta={jk.max_influence:.4e})")
        temporal_jk_results["loo"] = asdict(jk)

        if jk.loo_values:
            plot_jackknife_stability(
                jk.full_value, jk.loo_values, jk.loo_subjects,
                "Temporal Jackknife - spectral_radius N2->N3",
                "Mean slope (LOO)", 0.0,
                FIG_DIR / "jackknife_temporal_sr_n2n3.png",
            )

        l2o = run_leave_two_out(slope_dict, result_name="spectral_radius_N2_to_N3")
        log(f"  L2O: {l2o.n_pairs} pairs, sign preserved {l2o.fraction_sign_preserved:.0%}, "
            f"significant {l2o.fraction_significant:.0%}")
        log(f"    Worst pair: {l2o.worst_pair}, worst value: {l2o.worst_value:.4e}")
        temporal_jk_results["l2o"] = asdict(l2o)

        temporal_jk_verdict = "SURVIVES" if jk.sign_preserved_all and l2o.fraction_sign_preserved >= 0.90 else "PARTIAL"
        if not jk.sign_preserved_all:
            temporal_jk_verdict = "FAILS"
    else:
        log("  spectral_radius N2->N3 did not pass strict -- skipping temporal jackknife")
        temporal_jk_verdict = "N/A"

    output["temporal_jackknife"] = temporal_jk_results
    summary["temporal_jackknife"] = {
        "verdict": temporal_jk_verdict,
        "detail": (
            f"LOO sign={temporal_jk_results.get('loo', {}).get('sign_preserved_all', 'N/A')}, "
            f"L2O sign={temporal_jk_results.get('l2o', {}).get('fraction_sign_preserved', 'N/A')}"
            if temporal_jk_results else "Not run"
        ),
    }

    log("\n[5/8] Feature ablation...")
    ablation_results = {}
    for tbl, ca, cb, name in contrasts:
        features, labels, subjects, _ = _select_binary(tbl, ca, cb)
        result = run_feature_ablation(features, labels, subjects,
                                      feature_names=tbl.feature_names,
                                      contrast_name=name, seed=SEED)
        log(f"  {name}:")
        log(f"    Full AUC: {result.full_auc:.3f}")
        log(f"    Single-feature AUCs: {', '.join(f'{k}={v:.3f}' for k, v in result.single_feature_aucs.items())}")
        log(f"    Leave-one-out AUCs: {', '.join(f'{k}={v:.3f}' for k, v in result.leave_one_out_aucs.items())}")
        log(f"    Best pair: {', '.join(f'{k}={v:.3f}' for k, v in result.pairwise_aucs.items())}")
        log(f"    Most important: {result.most_important_feature}")
        log(f"    Distributed: {result.is_distributed}")
        if result.forward_selection_order:
            log(f"    Forward selection order: {' -> '.join(result.forward_selection_order)}")
            for fname, fauc, fdelta in zip(result.forward_selection_order,
                                            result.forward_selection_aucs,
                                            result.forward_selection_marginal_deltas):
                log(f"      +{fname}: AUC={fauc:.3f} (delta={fdelta:+.3f})")
        ablation_results[name] = asdict(result)

        plot_ablation_bars(result, FIG_DIR / f"ablation_{name}.png")

    all_distributed = all(r["is_distributed"] for r in ablation_results.values())
    output["feature_ablation"] = ablation_results
    summary["feature_ablation"] = {
        "verdict": "SURVIVES" if all_distributed else "PARTIAL",
        "detail": "; ".join(
            f"{k}: most_important={v['most_important_feature']}, distributed={v['is_distributed']}"
            for k, v in ablation_results.items()
        ),
    }

    log("\n[6/8] Spectral confound checks...")
    confound_results_list = []

    features_prop, labels_prop, subjects_prop, alpha_prop = _select_binary(prop_table, "awake", "propofol")
    if alpha_prop is not None and np.any(np.isfinite(alpha_prop)):
        result_alpha = run_spectral_confound_check(
            features_prop, labels_prop, subjects_prop, alpha_prop,
            feature_names=prop_table.feature_names,
            spectral_feature_name="alpha_power",
            dataset="propofol",
            seed=SEED,
        )
        log(f"  Propofol vs alpha_power:")
        for feat, r in result_alpha.per_geometry_correlations.items():
            d = result_alpha.per_geometry_residualized_d.get(feat, float("nan"))
            log(f"    {feat}: r={r:.3f}, residualized d={d:.3f}")
        log(f"    Residualized classification AUC: {result_alpha.residualized_classification_auc}")
        log(f"    Features surviving (|d|>=0.5): {result_alpha.n_features_surviving}/{result_alpha.n_features_total}")
        confound_results_list.append(result_alpha)
    else:
        log("  Propofol alpha: NOT AVAILABLE")

    features_sleep, labels_sleep, subjects_sleep, _ = _select_binary(sleep_table, "N3", "REM")
    if delta_arr is not None:
        sleep_mask = np.array([(c == "N3" or c == "REM") for c in sleep_table.conditions])
        delta_binary = delta_arr[sleep_mask]
        result_delta = run_spectral_confound_check(
            features_sleep, labels_sleep, subjects_sleep, delta_binary,
            feature_names=sleep_table.feature_names,
            spectral_feature_name="delta_power",
            dataset="sleep",
            seed=SEED,
        )
        log(f"  Sleep N3 vs REM, delta_power:")
        if result_delta.regressor_untestable:
            log(f"    UNTESTABLE: {result_delta.regressor_untestable_reason}")
            log(f"    Regressor variance: {result_delta.regressor_variance:.2e}")
        else:
            for feat, r in result_delta.per_geometry_correlations.items():
                d = result_delta.per_geometry_residualized_d.get(feat, float("nan"))
                log(f"    {feat}: r={r:.3f}, residualized d={d:.3f}")
            log(f"    Residualized classification AUC: {result_delta.residualized_classification_auc}")
            log(f"    Features surviving (|d|>=0.5): {result_delta.n_features_surviving}/{result_delta.n_features_total}")
        confound_results_list.append(result_delta)
    else:
        log("  Sleep delta: NOT AVAILABLE")

    confound_output = [asdict(cr) for cr in confound_results_list]
    output["spectral_confounds"] = confound_output

    if confound_results_list:
        plot_confound_heatmap(confound_results_list, FIG_DIR / "spectral_confound_map.png")

    n_surviving_total = 0
    n_testable_features = 0
    untestable_notes = []
    for cr in confound_results_list:
        if cr.regressor_untestable:
            untestable_notes.append(f"{cr.dataset}/{cr.spectral_feature_name} (regressor untestable)")
            continue
        n_computable = sum(1 for d in cr.per_geometry_residualized_d.values() if np.isfinite(d))
        n_nan = cr.n_features_total - n_computable
        if n_computable == 0:
            untestable_notes.append(
                f"{cr.dataset}/{cr.spectral_feature_name} (no both-class coverage for d)"
            )
            continue
        n_surviving_total += cr.n_features_surviving
        n_testable_features += n_computable
    if n_testable_features > 0:
        frac = n_surviving_total / n_testable_features
        confound_verdict = "SURVIVES" if frac >= 0.5 else ("PARTIAL" if frac >= 0.25 else "FAILS")
    else:
        confound_verdict = "N/A"
    untestable_note = ""
    if untestable_notes:
        untestable_note = f"; untestable: {', '.join(untestable_notes)}"
    summary["spectral_confounds"] = {
        "verdict": confound_verdict,
        "detail": f"{n_surviving_total}/{n_testable_features} testable features survive residualization (|d|>=0.5){untestable_note}",
    }

    log("\n[7/8] Window attacks...")
    window_tc_data = temporal_data.get("window_timecourses", {})
    decimation_results = {}

    if window_tc_data:
        log("  Window timecourses found in temporal_precedence.json — running decimation sweep.")
        for ttype, metrics_tc in window_tc_data.items():
            for metric, tc_info in metrics_tc.items():
                common_time = np.array(tc_info.get("common_time_sec", []))
                per_subj = tc_info.get("per_subject_mean", {})
                if not per_subj or len(common_time) == 0:
                    continue
                per_subj_ts = {s: np.array(v) for s, v in per_subj.items()}
                per_subj_time = {s: common_time for s in per_subj_ts}
                result = run_temporal_decimation(
                    per_subj_ts, per_subj_time,
                    metric_name=metric,
                    transition_type=ttype,
                )
                key = f"{ttype}_{metric}"
                decimation_results[key] = {
                    "metric_name": metric,
                    "transition_type": ttype,
                    "settings": result.settings,
                }
                log(f"  {key}:")
                for s in result.settings:
                    log(f"    dec={s['decimation_factor']}: slope={s['mean_slope']:.4e}, p={s['slope_p']:.4f}")

        output["window_attacks"] = {
            "decimation_sweep": decimation_results,
            "nonoverlap_spectral_radius_p": sr_data.get("nonoverlap_slope_p"),
            "nonoverlap_survives": sr_data.get("nonoverlap_survives"),
        }

        window_survive = True
        window_detail_parts = []
        for key, dr in decimation_results.items():
            if "spectral_radius" in key and "N2_to_N3" in key:
                full_slope = None
                for s in dr["settings"]:
                    if s["decimation_factor"] == 1:
                        full_slope = s["mean_slope"]
                        break
                if full_slope is None or full_slope == 0:
                    window_survive = False
                    continue
                for s in dr["settings"]:
                    dec = s["decimation_factor"]
                    if dec <= 1:
                        continue
                    same_sign = (s["mean_slope"] > 0) == (full_slope > 0)
                    magnitude_ratio = abs(s["mean_slope"]) / abs(full_slope) if full_slope != 0 else 0
                    if dec <= 3:
                        if not same_sign or s["slope_p"] >= 0.10:
                            window_survive = False
                            window_detail_parts.append(
                                f"dec={dec}: sign_ok={same_sign}, p={s['slope_p']:.4f}"
                            )
                    else:
                        if not same_sign or magnitude_ratio < 0.25:
                            window_survive = False
                            window_detail_parts.append(
                                f"dec={dec}: sign_ok={same_sign}, mag_ratio={magnitude_ratio:.2f}"
                            )
        window_verdict = "SURVIVES" if (decimation_results and window_survive) else "PARTIAL"
        detail_str = (
            f"Decimation sweep on {len(decimation_results)} metric/transition combos; "
            f"nonoverlap SR p={sr_data.get('nonoverlap_slope_p', 'N/A')}"
        )
        if window_detail_parts:
            detail_str += "; issues: " + ", ".join(window_detail_parts)
        summary["window_attacks"] = {
            "verdict": window_verdict,
            "detail": detail_str,
        }
    else:
        log("  NOTE: No window_timecourses in temporal_precedence.json.")
        log("  Cannot run full decimation sweep. Re-run _temporal_precedence.py to generate.")
        log("  Using inline non-overlapping check only.")
        output["window_attacks"] = {
            "note": "Window timecourses not yet saved in temporal_precedence.json. "
                    "Re-run _temporal_precedence.py with updated code to generate them. "
                    "Non-overlapping window check already performed inline.",
            "nonoverlap_spectral_radius_p": sr_data.get("nonoverlap_slope_p"),
            "nonoverlap_survives": sr_data.get("nonoverlap_survives"),
        }
        summary["window_attacks"] = {
            "verdict": "PARTIAL",
            "detail": "Non-overlapping check done inline. Re-run temporal precedence to enable full sweep.",
        }

    log("\n[8/8] Model competition...")
    competition_results = {}
    for tbl, ca, cb, name in contrasts:
        features, labels, subjects, alpha = _select_binary(tbl, ca, cb)

        baselines: dict[str, np.ndarray] = {}

        if alpha is not None:
            alpha_finite = np.isfinite(alpha)
            has_both = np.any(alpha_finite & (labels == 0)) and np.any(alpha_finite & (labels == 1))
            if has_both:
                baselines["alpha_power"] = alpha

        if name == "N3_vs_REM" and delta_arr is not None:
            sleep_mask = np.array([(c == "N3" or c == "REM") for c in tbl.conditions])
            delta_binary = delta_arr[sleep_mask]
            delta_finite = np.isfinite(delta_binary)
            has_both_delta = np.any(delta_finite & (labels == 0)) and np.any(delta_finite & (labels == 1))
            if has_both_delta:
                baselines["delta_power"] = delta_binary

        if baselines:
            result = run_model_competition(
                features, labels, subjects,
                baseline_features=baselines,
                contrast_name=name, seed=SEED,
            )
            log(f"  {name}:")
            log(f"    Geometry AUC: {result.geometry_auc:.3f}")
            for bn, ba in result.baseline_aucs.items():
                log(f"    {bn}: {ba:.3f}")
            log(f"    Geometry beats all: {result.geometry_beats_all}")
            competition_results[name] = asdict(result)
        else:
            log(f"  {name}: No valid baselines (spectral features not available for both classes)")
            competition_results[name] = {
                "contrast_name": name,
                "geometry_auc": float(_loso_auc(features, labels, subjects, seed=SEED)),
                "baseline_aucs": {},
                "geometry_beats_all": None,
                "note": "No non-geometric baselines available with both-class coverage",
            }

    output["model_competition"] = competition_results

    if competition_results:
        comp_objects = []
        for tbl, ca, cb, name in contrasts:
            if name in competition_results:
                features, labels, subjects, alpha = _select_binary(tbl, ca, cb)
                baselines_for_plot: dict[str, np.ndarray] = {}
                if alpha is not None:
                    af = np.isfinite(alpha)
                    if np.any(af & (labels == 0)) and np.any(af & (labels == 1)):
                        baselines_for_plot["alpha_power"] = alpha
                if name == "N3_vs_REM" and delta_arr is not None:
                    sm = np.array([(c == "N3" or c == "REM") for c in tbl.conditions])
                    db = delta_arr[sm]
                    df = np.isfinite(db)
                    if np.any(df & (labels == 0)) and np.any(df & (labels == 1)):
                        baselines_for_plot["delta_power"] = db
                if baselines_for_plot:
                    comp_objects.append(run_model_competition(
                        features, labels, subjects,
                        baseline_features=baselines_for_plot,
                        contrast_name=name, seed=SEED,
                    ))
        if comp_objects:
            plot_model_competition(comp_objects, FIG_DIR / "model_competition.png")

    tested = {k: v for k, v in competition_results.items() if v.get("geometry_beats_all") is not None}
    if tested:
        all_beat = all(r["geometry_beats_all"] for r in tested.values())
        comp_verdict = "SURVIVES" if all_beat else "FAILS"
    else:
        comp_verdict = "N/A"
    summary["model_competition"] = {
        "verdict": comp_verdict,
        "detail": "; ".join(
            f"{k}: beats_all={v.get('geometry_beats_all', 'N/A')}" for k, v in competition_results.items()
        ),
    }

    output["summary"] = summary

    log(f"\n{'='*70}")
    log("FALSIFICATION BATTERY SCORECARD")
    log(f"{'='*70}")
    for cat, info in summary.items():
        verdict = info.get("verdict", "N/A")
        detail = info.get("detail", "")
        log(f"  {cat:30s} {verdict:10s} {detail}")

    n_survive = sum(1 for info in summary.values() if info.get("verdict") == "SURVIVES")
    n_partial = sum(1 for info in summary.values() if info.get("verdict") == "PARTIAL")
    n_fail = sum(1 for info in summary.values() if info.get("verdict") == "FAILS")
    log(f"\n  TOTAL: {n_survive} SURVIVE, {n_partial} PARTIAL, {n_fail} FAIL")
    log(f"{'='*70}")

    plot_scorecard(summary, FIG_DIR / "falsification_scorecard.png")

    output["elapsed_sec"] = time.time() - t0

    out_path = RESULTS_JSON / "falsification_battery.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=default_ser)
    log(f"\nResults: {out_path}")
    log(f"Figures: {FIG_DIR}")
    log(f"Total time: {output['elapsed_sec']:.1f}s")
    log("DONE")


if __name__ == "__main__":
    main()
