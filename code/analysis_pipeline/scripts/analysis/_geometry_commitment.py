"""
Geometric Transition Commitment Analysis — Orchestration Script.

Extends temporal precedence analysis with five sub-questions (Q1–Q5) testing
whether operator geometry shows an identifiable commitment moment before
N2→N3 sleep transitions.

Q1/Q2 require raw EEG (SLEEP_DATA_ROOT); Q3/Q4/Q5 use existing temporal_precedence.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from cmcc.analysis.temporal_precedence import TransitionTimecourse
from cmcc.analysis.geometry_commitment import (
    ExtendedWindowResult,
    ChangepointResult,
    BistabilityResult,
    TrajectoryPredictionResult,
    CommitmentWindowResult,
    analyze_extended_window,
    detect_commitment_changepoints,
    analyze_n2_bistability,
    predict_transition_type_from_trajectory,
    characterize_commitment_window,
)

PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = PIPELINE_ROOT.parent.parent
RESULTS_JSON = REPO_ROOT / "results" / "json_results"
FIG_DIR = REPO_ROOT / "results" / "figures" / "geometry_commitment"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(os.environ.get("SLEEP_DATA_ROOT", str(REPO_ROOT / "ANPHY-Sleep")))
RAW_DATA_AVAILABLE = DATA_ROOT.exists() and DATA_ROOT.is_dir()
if RAW_DATA_AVAILABLE:
    try:
        RAW_DATA_AVAILABLE = any(DATA_ROOT.iterdir())
    except Exception:
        RAW_DATA_AVAILABLE = False

PRE_SEC_EXTENDED = 600.0
POST_SEC = 60.0
MIN_QUALIFY_PRE_SEC = 600.0
MIN_QUALIFY_CHANGEPOINT_SEC = 240.0
MIN_SEGMENT_SEC = 30.0
SLOPE_HALFWINDOW_SEC = 30.0
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 1000
SEED = 42
DPI = 300

GEOMETRY_METRICS = ["spectral_radius", "eigenvalue_gap", "condition_number", "nd_score"]
Q4_METRICS = ["spectral_radius", "eigenvalue_gap", "condition_number", "nd_score"]
TIME_BINS_SEC = [-120.0, -100.0, -80.0, -60.0, -40.0, -20.0]

STATE_PALETTE = {
    "N2_to_N3": "#BB5566",
    "N2_to_R": "#4477AA",
}


def log(msg):
    print(msg, flush=True)


def default_ser(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if math.isnan(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return [default_ser(x) for x in obj.flat] if obj.ndim <= 1 else obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    raise TypeError(f"Not serializable: {type(obj)}")


def load_json_timecourses(json_path: Path):
    """Parse temporal_precedence.json window timecourses into TransitionTimecourse dicts.

    Returns
    -------
    n3_by_subject : dict[str, list[TransitionTimecourse]]
    rem_by_subject : dict[str, list[TransitionTimecourse]]
    """
    with open(json_path) as f:
        data = json.load(f)

    wt = data.get("window_timecourses", {})
    n3_by_subject: dict[str, list[TransitionTimecourse]] = {}
    rem_by_subject: dict[str, list[TransitionTimecourse]] = {}

    for ttype, metrics_dict in wt.items():
        if not metrics_dict:
            continue

        first_metric = list(metrics_dict.values())[0]
        common_time = np.array(first_metric.get("common_time_sec", []))
        per_subject_data = first_metric.get("per_subject_mean", {})
        subjects = sorted(per_subject_data.keys())

        for subj in subjects:
            metric_arrays = {}
            for metric in GEOMETRY_METRICS:
                m_data = metrics_dict.get(metric, {})
                ps = m_data.get("per_subject_mean", {})
                if subj in ps:
                    metric_arrays[metric] = np.array(ps[subj])
                else:
                    metric_arrays[metric] = np.full(len(common_time), np.nan)

            tc = TransitionTimecourse(
                subject=subj,
                transition_type=ttype,
                transition_index=0,
                time_sec=common_time.copy(),
                eigenvalue_gap=metric_arrays.get("eigenvalue_gap", np.full(len(common_time), np.nan)),
                condition_number=metric_arrays.get("condition_number", np.full(len(common_time), np.nan)),
                nd_score=metric_arrays.get("nd_score", np.full(len(common_time), np.nan)),
                spectral_radius=metric_arrays.get("spectral_radius", np.full(len(common_time), np.nan)),
            )

            if "N3" in ttype or "n3" in ttype.lower():
                n3_by_subject.setdefault(subj, []).append(tc)
            elif "R" in ttype.split("_")[-1] or "rem" in ttype.lower():
                rem_by_subject.setdefault(subj, []).append(tc)

    return n3_by_subject, rem_by_subject


def extract_bistability_windows(
    timecourses_by_subject: dict[str, list[TransitionTimecourse]],
    early_sec_range: tuple[float, float] = (-120.0, -60.0),
    late_sec_range: tuple[float, float] = (-60.0, 0.0),
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract early and late N2 window arrays for Q3 bistability analysis."""
    early_windows: dict[str, np.ndarray] = {}
    late_windows: dict[str, np.ndarray] = {}

    for subj, tcs in timecourses_by_subject.items():
        early_rows = []
        late_rows = []
        for tc in tcs:
            t = tc.time_sec
            for idx in range(len(t)):
                row = [
                    tc.spectral_radius[idx] if hasattr(tc, "spectral_radius") else np.nan,
                    tc.eigenvalue_gap[idx] if hasattr(tc, "eigenvalue_gap") else np.nan,
                    tc.condition_number[idx] if hasattr(tc, "condition_number") else np.nan,
                    tc.nd_score[idx] if hasattr(tc, "nd_score") else np.nan,
                ]
                if early_sec_range[0] <= t[idx] < early_sec_range[1]:
                    early_rows.append(row)
                elif late_sec_range[0] <= t[idx] < late_sec_range[1]:
                    late_rows.append(row)

        if early_rows:
            early_windows[subj] = np.array(early_rows)
        if late_rows:
            late_windows[subj] = np.array(late_rows)

    return early_windows, late_windows


def run_q3(n3_by_subject):
    log("\n--- Q3: Geometric Bistability ---")
    early, late = extract_bistability_windows(n3_by_subject)
    log(f"  Subjects with early windows: {len(early)}")
    log(f"  Subjects with late windows: {len(late)}")
    result = analyze_n2_bistability(early, late, GEOMETRY_METRICS, seed=SEED, n_bootstrap_dip=2000)
    log(f"  Early windows: {result.n_early_windows}, Late windows: {result.n_late_windows}")
    log(f"  BC statistic: {result.dip_test_statistic:.4f}, p={result.dip_test_p:.4f}")
    log(f"  GMM BIC delta: {result.gmm_bic_delta:.1f}")
    log(f"  Cohen's d (spectral_radius): {result.early_vs_late_d_spectral_radius:.3f}")
    log(f"  Cohen's d (eigenvalue_gap): {result.early_vs_late_d_eigenvalue_gap:.3f}")
    if result.hotelling_p is not None:
        log(f"  Hotelling's T² p: {result.hotelling_p:.4f}")
    return result


def run_q4(n3_by_subject, rem_by_subject):
    log("\n--- Q4: Trajectory Prediction ---")
    log(f"  N3-bound subjects: {len(n3_by_subject)}")
    log(f"  REM-bound subjects: {len(rem_by_subject)}")
    result = predict_transition_type_from_trajectory(
        n3_by_subject, rem_by_subject, Q4_METRICS,
        time_bins_sec=TIME_BINS_SEC,
        slope_halfwindow_sec=SLOPE_HALFWINDOW_SEC,
        n_permutations=N_PERMUTATIONS,
        seed=SEED,
    )
    log(f"  LOSO AUC: {result.loso_auc:.3f}")
    log(f"  LOSO balanced accuracy: {result.loso_balanced_accuracy:.3f}")
    log(f"  Majority baseline AUC: {result.majority_baseline_auc:.3f}")
    log(f"  Feature scale ratio: {result.feature_scale_ratio:.1f}")
    log(f"  Convergence failures: {result.n_convergence_failures}")
    log(f"  Permutation p: {result.permutation_p:.4f}")
    for metric in Q4_METRICS:
        d_vals = result.per_bin_cohens_d.get(metric, [])
        d_str = ", ".join(f"{d:.2f}" for d in d_vals)
        log(f"  {metric} d by bin: [{d_str}]")
    log(f"  Per-bin LOSO AUC: {[round(a, 3) for a in result.per_bin_loso_auc]}")
    return result


def run_q5(trajectory_result, changepoint_result=None, extended_window_result=None):
    log("\n--- Q5: Commitment Window Synthesis ---")
    result = characterize_commitment_window(
        trajectory_result,
        changepoint_result=changepoint_result,
        extended_window_result=extended_window_result,
        n_bootstrap=N_BOOTSTRAP,
        n_permutations=N_PERMUTATIONS,
        seed=SEED,
    )
    log(f"  Primary onset (d>=0.5): {result.discrimination_onset_sec}")
    log(f"  Commitment window: {result.commitment_window_sec}")
    if result.onset_ci_sec is not None:
        log(f"  Onset 95% CI: [{result.onset_ci_sec[0]:.1f}, {result.onset_ci_sec[1]:.1f}]")
    log(f"  Onset permutation p: {result.onset_permutation_p:.4f}")
    log(f"  Monotonicity fraction: {result.monotonicity_fraction:.3f}")
    for thresh, onset in result.onset_by_threshold.items():
        log(f"    d>={thresh}: onset={onset}")
    if result.changepoint_overlap is not None:
        log(f"  Changepoint overlap: {result.changepoint_overlap}")
    return result


def plot_q3(result: BistabilityResult):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if result.pca_coords_early.shape[0] > 0:
        ax.scatter(result.pca_coords_early[:, 0], result.pca_coords_early[:, 1],
                   alpha=0.3, s=8, color="#4477AA", label="Early N2")
    if result.pca_coords_late.shape[0] > 0:
        ax.scatter(result.pca_coords_late[:, 0], result.pca_coords_late[:, 1],
                   alpha=0.3, s=8, color="#BB5566", label="Late N2")

    if np.all(np.isfinite(result.gmm_means)):
        for k in range(2):
            mean = result.gmm_means[k, :2] if result.gmm_means.shape[1] >= 2 else result.gmm_means[k]
            cov = result.gmm_covariances[k, :2, :2] if result.gmm_covariances.shape[1] >= 2 else np.eye(2)
            if cov.shape == (2, 2):
                eigvals, eigvecs = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                w, h = 2 * 2 * np.sqrt(eigvals)
                ell = Ellipse(xy=mean[:2], width=w, height=h, angle=angle,
                              edgecolor="k", facecolor="none", linewidth=1.5, linestyle="--")
                ax.add_patch(ell)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"N2 Geometry PCA\nBC p(Gauss)={result.bc_gaussian_null_p:.3f}, GMM ΔBIC={result.gmm_bic_delta:.1f}")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    all_early_pc1 = result.pca_coords_early[:, 0] if result.pca_coords_early.shape[0] > 0 else np.array([])
    all_late_pc1 = result.pca_coords_late[:, 0] if result.pca_coords_late.shape[0] > 0 else np.array([])
    if len(all_early_pc1) > 0:
        ax2.hist(all_early_pc1, bins=30, alpha=0.5, color="#4477AA", label="Early N2", density=True)
    if len(all_late_pc1) > 0:
        ax2.hist(all_late_pc1, bins=30, alpha=0.5, color="#BB5566", label="Late N2", density=True)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("Density")
    ax2.set_title("PC1 distribution")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "geometry_bistability.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR / 'geometry_bistability.png'}")


def plot_q4(result: TrajectoryPredictionResult):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(result.metrics)))
    for i, metric in enumerate(result.metrics):
        d_vals = result.per_bin_cohens_d.get(metric, [])
        if d_vals:
            ax.plot(result.time_bins_sec[:len(d_vals)], d_vals, "o-",
                    color=colors[i], label=metric, markersize=4)

    for thresh in [0.3, 0.5, 0.8]:
        ax.axhline(thresh, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.text(result.time_bins_sec[0] - 5, thresh, f"d={thresh}", fontsize=7,
                va="center", ha="right", color="gray")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Time before transition (s)")
    ax.set_ylabel("Cohen's d (N3-bound vs REM-bound)")
    ax.set_title("Trajectory discrimination by time bin")
    ax.legend(fontsize=7, loc="upper left")

    ax2 = axes[1]
    if result.permutation_null_auc:
        ax2.hist(result.permutation_null_auc, bins=30, alpha=0.6, color="gray", label="Null")
    if np.isfinite(result.loso_auc):
        ax2.axvline(result.loso_auc, color="#BB5566", linewidth=2,
                    label=f"Observed AUC={result.loso_auc:.3f}")
    ax2.set_xlabel("AUC")
    ax2.set_ylabel("Count")
    ax2.set_title(f"LOSO AUC vs permutation null\np={result.permutation_p:.4f}")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "trajectory_discrimination.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR / 'trajectory_discrimination.png'}")


def plot_q5(result: CommitmentWindowResult, q4_result: TrajectoryPredictionResult = None):
    fig, ax = plt.subplots(figsize=(10, 3))

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="Staging boundary")

    ax.axvspan(-600, -120, alpha=0.08, color="#4477AA", label="Stable N2 (Q3 early)")

    if result.discrimination_onset_sec is not None:
        ax.axvline(result.discrimination_onset_sec, color="#BB5566", linewidth=2,
                   label=f"Discrimination onset ({result.discrimination_onset_sec:.0f} s)")
        ax.axvspan(result.discrimination_onset_sec, 0, alpha=0.15, color="#BB5566",
                   label=f"Commitment window ({result.commitment_window_sec:.0f} s)")

    for thresh, onset in result.onset_by_threshold.items():
        if onset is not None and thresh != 0.5:
            ax.axvline(onset, color="gray", linewidth=0.8, linestyle=":",
                       alpha=0.6)
            ax.text(onset, 0.8, f"d≥{thresh}", fontsize=7, ha="center",
                    transform=ax.get_xaxis_transform(), color="gray")

    ax.set_xlim(-650, 50)
    ax.set_xlabel("Time before transition (s)")
    ax.set_yticks([])
    ax.set_title("Geometric Commitment Window Summary")
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "commitment_window_summary.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {FIG_DIR / 'commitment_window_summary.png'}")


def append_csv_rows(q3_result, q4_result, q5_result):
    """Append geometry commitment rows to summary_statistics.csv."""
    csv_path = REPO_ROOT / "results" / "summary_statistics.csv"
    src = "geometry_commitment.json"

    def _fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return ""
        if isinstance(v, float):
            if abs(v) < 0.001 and v != 0:
                return f"{v:.2e}"
            return f"{v:.4f}"
        return str(v)

    rows = [
        f"geometry_commitment,q3_bistability,all_metrics,gmm_bic_delta,{_fmt(q3_result.gmm_bic_delta)},dip_p={_fmt(q3_result.dip_test_p)},,,{src}",
        f"geometry_commitment,q4_trajectory_prediction,all_metrics,loso_auc,{_fmt(q4_result.loso_auc)},{_fmt(q4_result.permutation_p)},,,{src}",
        f"geometry_commitment,q5_commitment,all_metrics,discrimination_onset_sec,{_fmt(q5_result.discrimination_onset_sec)},,,,{src}",
    ]

    existing = ""
    if csv_path.exists():
        existing = csv_path.read_text()

    marker = "geometry_commitment,"
    if marker in existing:
        lines = existing.splitlines()
        lines = [l for l in lines if not l.startswith(marker)]
        existing = "\n".join(lines)
        if existing and not existing.endswith("\n"):
            existing += "\n"

    with open(csv_path, "w") as f:
        f.write(existing)
        for r in rows:
            f.write(r + "\n")

    log(f"CSV rows appended to {csv_path}")


def main():
    t_start = time.time()
    log("=" * 70)
    log("GEOMETRIC TRANSITION COMMITMENT ANALYSIS")
    log("=" * 70)

    tp_json = RESULTS_JSON / "temporal_precedence.json"
    if not tp_json.exists():
        log(f"ERROR: {tp_json} not found. Run _temporal_precedence.py first.")
        sys.exit(1)

    log(f"Loading timecourses from {tp_json}")
    n3_by_subject, rem_by_subject = load_json_timecourses(tp_json)
    log(f"  N3-bound subjects: {len(n3_by_subject)}")
    log(f"  REM-bound subjects: {len(rem_by_subject)}")

    log(f"\nRaw data available: {RAW_DATA_AVAILABLE}")
    if not RAW_DATA_AVAILABLE:
        log("  Q1 and Q2 will be skipped (SLEEP_DATA_ROOT not available).")

    q1_results_sr = None
    q1_results_eg = None
    q2_results_sr = None
    q2_results_eg = None

    json_pre_sec = 120.0
    log(f"\nRunning Q1 and Q2 with JSON timecourses (window: {json_pre_sec} s)")
    log("  NOTE: Best-effort — limited to 120 s window (raw EEG not available for 600 s)")

    log("\n--- Q1: Extended Drift Window (120-s JSON approximation) ---")
    for metric in ["spectral_radius", "eigenvalue_gap"]:
        log(f"  Metric: {metric}")
        result = analyze_extended_window(
            n3_by_subject, metric,
            pre_sec=json_pre_sec, post_sec=POST_SEC,
            min_qualifying_pre_sec=json_pre_sec,
            seed=SEED, n_bootstrap=N_BOOTSTRAP,
        )
        log(f"    n_subjects={result.n_subjects}, n_transitions={result.n_qualifying_transitions}")
        log(f"    slope={result.mean_slope:.2e}, p={result.slope_p_value:.4f}")
        log(f"    d={result.cohens_d:.3f}, consistency={result.subject_consistency:.2f}")
        log(f"    AIC delta (quad-lin): {result.aic_delta:.2f}")
        log(f"    passes_threshold: {result.passes_threshold}")
        if metric == "spectral_radius":
            q1_results_sr = result
        else:
            q1_results_eg = result

    log("\n--- Q2: Changepoint Detection (120-s JSON approximation) ---")
    for metric in ["spectral_radius", "eigenvalue_gap"]:
        log(f"  Metric: {metric}")
        result = detect_commitment_changepoints(
            n3_by_subject, metric,
            pre_sec=json_pre_sec,
            min_segment_sec=MIN_SEGMENT_SEC,
            min_qualifying_pre_sec=json_pre_sec,
            penalty_values=[20.0, 30.0],
            seed=SEED,
        )
        log(f"    n_qualifying={result.n_qualifying_transitions}, n_subjects={result.n_subjects}")
        log(f"    group_mean_latency={result.group_mean_latency_sec:.1f} s")
        log(f"    group_p={result.group_latency_p_value:.4f}")
        log(f"    CI: [{result.group_latency_ci[0]:.1f}, {result.group_latency_ci[1]:.1f}]")
        log(f"    fraction_within_300s: {result.fraction_within_300s:.2f}")
        for pk, pv in result.penalty_sensitivity.items():
            log(f"    {pk}: mean={pv['group_mean_latency_sec']:.1f}, p={pv['group_p']:.4f}")
        if metric == "spectral_radius":
            q2_results_sr = result
        else:
            q2_results_eg = result

    q3_result = run_q3(n3_by_subject)
    plot_q3(q3_result)

    q4_result = run_q4(n3_by_subject, rem_by_subject)
    plot_q4(q4_result)

    ext_result = q1_results_sr
    cp_result = q2_results_sr
    q5_result = run_q5(q4_result, changepoint_result=cp_result, extended_window_result=ext_result)
    plot_q5(q5_result, q4_result)

    output = {
        "analysis": "geometry_commitment",
        "dataset": "ANPHY-Sleep",
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "pre_sec_extended": PRE_SEC_EXTENDED,
            "min_qualify_pre_sec": MIN_QUALIFY_PRE_SEC,
            "pre_sec_changepoint": PRE_SEC_EXTENDED,
            "min_qualify_changepoint_sec": MIN_QUALIFY_CHANGEPOINT_SEC,
            "min_segment_sec": MIN_SEGMENT_SEC,
            "slope_halfwindow_sec": SLOPE_HALFWINDOW_SEC,
            "n_bootstrap": N_BOOTSTRAP,
            "n_permutations": N_PERMUTATIONS,
            "seed": SEED,
            "raw_data_available": RAW_DATA_AVAILABLE,
        },
        "q1_extended_window": {
            "note": "Best-effort: 120-s JSON timecourses (raw EEG unavailable for 600-s window)",
            "spectral_radius": {
                "n_qualifying_transitions": q1_results_sr.n_qualifying_transitions,
                "n_subjects": q1_results_sr.n_subjects,
                "mean_slope": q1_results_sr.mean_slope,
                "slope_p_value": q1_results_sr.slope_p_value,
                "slope_ci": list(q1_results_sr.slope_ci),
                "cohens_d": q1_results_sr.cohens_d,
                "subject_consistency": q1_results_sr.subject_consistency,
                "linear_aic": q1_results_sr.linear_aic,
                "quadratic_aic": q1_results_sr.quadratic_aic,
                "aic_delta": q1_results_sr.aic_delta,
                "passes_threshold": q1_results_sr.passes_threshold,
                "data_constraint": q1_results_sr.data_constraint,
            } if q1_results_sr else {"skipped": True},
            "eigenvalue_gap": {
                "n_qualifying_transitions": q1_results_eg.n_qualifying_transitions,
                "n_subjects": q1_results_eg.n_subjects,
                "mean_slope": q1_results_eg.mean_slope,
                "slope_p_value": q1_results_eg.slope_p_value,
                "slope_ci": list(q1_results_eg.slope_ci),
                "cohens_d": q1_results_eg.cohens_d,
                "subject_consistency": q1_results_eg.subject_consistency,
                "linear_aic": q1_results_eg.linear_aic,
                "quadratic_aic": q1_results_eg.quadratic_aic,
                "aic_delta": q1_results_eg.aic_delta,
                "passes_threshold": q1_results_eg.passes_threshold,
                "data_constraint": q1_results_eg.data_constraint,
            } if q1_results_eg else {"skipped": True},
        },
        "q2_changepoint": {
            "note": "Best-effort: 120-s JSON timecourses (raw EEG unavailable for 600-s window)",
            "spectral_radius": {
                "n_qualifying_transitions": q2_results_sr.n_qualifying_transitions,
                "n_subjects": q2_results_sr.n_subjects,
                "group_mean_latency_sec": q2_results_sr.group_mean_latency_sec,
                "group_latency_p_value": q2_results_sr.group_latency_p_value,
                "group_latency_ci": list(q2_results_sr.group_latency_ci),
                "latency_sd_sec": q2_results_sr.latency_sd_sec,
                "fraction_within_300s": q2_results_sr.fraction_within_300s,
                "penalty_sensitivity": q2_results_sr.penalty_sensitivity,
                "edge_proximity_ratio": q2_results_sr.edge_proximity_ratio,
                "data_constraint": q2_results_sr.data_constraint,
                "window_truncation_sensitivity": q2_results_sr.window_truncation_sensitivity,
            } if q2_results_sr else {"skipped": True},
            "eigenvalue_gap": {
                "n_qualifying_transitions": q2_results_eg.n_qualifying_transitions,
                "n_subjects": q2_results_eg.n_subjects,
                "group_mean_latency_sec": q2_results_eg.group_mean_latency_sec,
                "group_latency_p_value": q2_results_eg.group_latency_p_value,
                "group_latency_ci": list(q2_results_eg.group_latency_ci),
                "latency_sd_sec": q2_results_eg.latency_sd_sec,
                "fraction_within_300s": q2_results_eg.fraction_within_300s,
                "penalty_sensitivity": q2_results_eg.penalty_sensitivity,
                "edge_proximity_ratio": q2_results_eg.edge_proximity_ratio,
                "data_constraint": q2_results_eg.data_constraint,
                "window_truncation_sensitivity": q2_results_eg.window_truncation_sensitivity,
            } if q2_results_eg else {"skipped": True},
        },
        "q3_bistability": {
            "n_early_windows": q3_result.n_early_windows,
            "n_late_windows": q3_result.n_late_windows,
            "dip_test_statistic": q3_result.dip_test_statistic,
            "dip_test_p_uniform_null": q3_result.dip_test_p,
            "bc_gaussian_null_p": q3_result.bc_gaussian_null_p,
            "silverman_p": q3_result.silverman_p,
            "bc_note": q3_result.bc_note,
            "gmm_bic_1": q3_result.gmm_bic_1,
            "gmm_bic_2": q3_result.gmm_bic_2,
            "gmm_bic_delta": q3_result.gmm_bic_delta,
            "gmm_cv_ll_delta": q3_result.gmm_cv_ll_delta,
            "early_vs_late_d_spectral_radius": q3_result.early_vs_late_d_spectral_radius,
            "early_vs_late_d_eigenvalue_gap": q3_result.early_vs_late_d_eigenvalue_gap,
            "hotelling_p": q3_result.hotelling_p,
            "gmm_means": q3_result.gmm_means,
            "gmm_covariances": q3_result.gmm_covariances,
        },
        "q4_trajectory_prediction": {
            "metrics": q4_result.metrics,
            "time_bins_sec": q4_result.time_bins_sec,
            "per_bin_cohens_d": q4_result.per_bin_cohens_d,
            "loso_auc": q4_result.loso_auc,
            "loso_balanced_accuracy": q4_result.loso_balanced_accuracy,
            "permutation_p": q4_result.permutation_p,
            "n_n3_bound": q4_result.n_n3_bound,
            "n_rem_bound": q4_result.n_rem_bound,
            "feature_scale_ratio": q4_result.feature_scale_ratio,
            "n_convergence_failures": q4_result.n_convergence_failures,
            "majority_baseline_auc": q4_result.majority_baseline_auc,
            "per_bin_loso_auc": q4_result.per_bin_loso_auc,
            "per_fold_predictions": q4_result.per_fold_predictions,
        },
        "q5_commitment": {
            "d_thresholds_tested": q5_result.thresholds_tested,
            "discrimination_onset_by_threshold": {
                str(k): v for k, v in q5_result.onset_by_threshold.items()
            },
            "primary_onset_sec": q5_result.discrimination_onset_sec,
            "primary_threshold": 0.5,
            "commitment_window_sec": q5_result.commitment_window_sec,
            "changepoint_overlap": q5_result.changepoint_overlap,
            "geometry_at_onset": q5_result.geometry_at_onset,
            "onset_ci_sec": list(q5_result.onset_ci_sec) if q5_result.onset_ci_sec else None,
            "onset_permutation_p": q5_result.onset_permutation_p,
            "monotonicity_fraction": q5_result.monotonicity_fraction,
            "n_bootstrap": q5_result.n_bootstrap,
            "n_permutations": q5_result.n_permutations,
        },
        "elapsed_sec": time.time() - t_start,
    }

    out_path = RESULTS_JSON / "geometry_commitment.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=default_ser)
    log(f"\nResults saved: {out_path}")

    summary_path = RESULTS_JSON / "commitment_summary.txt"
    with open(summary_path, "w") as f:
        f.write("GEOMETRIC TRANSITION COMMITMENT ANALYSIS — KEY STATISTICS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: ANPHY-Sleep, n=10 subjects\n")
        f.write(f"Raw data available: {RAW_DATA_AVAILABLE}\n\n")

        f.write("Q1: Extended Drift Window (120-s best-effort)\n")
        if q1_results_sr:
            f.write(f"  spectral_radius: slope={q1_results_sr.mean_slope:.2e}, p={q1_results_sr.slope_p_value:.4f}, d={q1_results_sr.cohens_d:.3f}\n")
        if q1_results_eg:
            f.write(f"  eigenvalue_gap: slope={q1_results_eg.mean_slope:.2e}, p={q1_results_eg.slope_p_value:.4f}, d={q1_results_eg.cohens_d:.3f}\n")
        f.write("\n")

        f.write("Q2: Changepoint Detection (120-s best-effort)\n")
        if q2_results_sr:
            f.write(f"  spectral_radius: latency={q2_results_sr.group_mean_latency_sec:.1f} s, p={q2_results_sr.group_latency_p_value:.4f}\n")
        if q2_results_eg:
            f.write(f"  eigenvalue_gap: latency={q2_results_eg.group_mean_latency_sec:.1f} s, p={q2_results_eg.group_latency_p_value:.4f}\n")
        f.write("\n")

        f.write("Q3: N2 Geometry Bistability\n")
        f.write(f"  BC statistic: {q3_result.dip_test_statistic:.4f}\n")
        f.write(f"  BC p (uniform null — uninformative): {q3_result.dip_test_p:.4f}\n")
        f.write(f"  BC p (Gaussian null): {q3_result.bc_gaussian_null_p:.4f}\n")
        f.write(f"  Silverman unimodality p: {q3_result.silverman_p:.4f}\n")
        f.write(f"  GMM BIC delta: {q3_result.gmm_bic_delta:.1f}\n")
        f.write(f"  GMM CV log-lik delta (k=2 minus k=1): {q3_result.gmm_cv_ll_delta:.4f}\n")
        f.write(f"  Cohen's d (spectral_radius, early vs late): {q3_result.early_vs_late_d_spectral_radius:.3f}\n")
        f.write(f"  Cohen's d (eigenvalue_gap, early vs late): {q3_result.early_vs_late_d_eigenvalue_gap:.3f}\n")
        if q3_result.hotelling_p is not None:
            f.write(f"  Hotelling's T^2 p: {q3_result.hotelling_p:.4f}\n")
        f.write("\n")

        f.write("Q4: Trajectory Prediction (N3-bound vs REM-bound)\n")
        f.write(f"  LOSO AUC: {q4_result.loso_auc:.3f}\n")
        f.write(f"  LOSO balanced accuracy: {q4_result.loso_balanced_accuracy:.3f}\n")
        f.write(f"  Majority baseline AUC: {q4_result.majority_baseline_auc:.3f}\n")
        f.write(f"  Feature scale ratio: {q4_result.feature_scale_ratio:.1f}\n")
        f.write(f"  Convergence failures: {q4_result.n_convergence_failures}\n")
        f.write(f"  Permutation p: {q4_result.permutation_p:.4f}\n")
        f.write(f"  N3-bound subjects: {q4_result.n_n3_bound}, REM-bound: {q4_result.n_rem_bound}\n")
        for metric in Q4_METRICS:
            d_vals = q4_result.per_bin_cohens_d.get(metric, [])
            f.write(f"  {metric} d by bin: {[round(d, 3) for d in d_vals]}\n")
        f.write(f"  Per-bin LOSO AUC: {[round(a, 3) for a in q4_result.per_bin_loso_auc]}\n")
        f.write("\n")

        f.write("Q5: Commitment Window\n")
        f.write(f"  Primary onset (d>=0.5): {q5_result.discrimination_onset_sec}\n")
        f.write(f"  Commitment window: {q5_result.commitment_window_sec}\n")
        if q5_result.onset_ci_sec is not None:
            f.write(f"  Onset 95% CI: [{q5_result.onset_ci_sec[0]:.1f}, {q5_result.onset_ci_sec[1]:.1f}]\n")
        f.write(f"  Onset permutation p: {q5_result.onset_permutation_p:.4f}\n")
        f.write(f"  Monotonicity fraction: {q5_result.monotonicity_fraction:.3f}\n")
        for thresh, onset in q5_result.onset_by_threshold.items():
            f.write(f"    d>={thresh}: onset={onset}\n")
        f.write("\n")

    log(f"Summary saved: {summary_path}")

    append_csv_rows(q3_result, q4_result, q5_result)

    log(f"Figures saved: {FIG_DIR}")
    log(f"\nTotal elapsed: {time.time() - t_start:.1f} s")
    log("DONE")


if __name__ == "__main__":
    main()
