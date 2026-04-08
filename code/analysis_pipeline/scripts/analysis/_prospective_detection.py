"""Prospective seizure detection — fixed-horizon pre-ictal vs sham.

Converts the two-regime discovery into a prospective detection problem.
Uses ONLY data available before onset. No self-alignment, no minimum
timing, no post-onset labels in features.

Loads trajectory_cache.npz and per_seizure_features.csv. For each
prediction horizon, extracts windowed features and classifies
pre-ictal vs matched interictal sham using LOSO logistic regression.

Scientific constraints:
- No feature may use data from onset or later
- No self-alignment
- No raw_spacing_change (spans full pre-ictal, not prospective)
- Sham segments come from each seizure's own interictal baseline
- Z-scored features are only meaningful for horizons outside baseline

Horizons (minutes before onset):
  H1: [-30, -20]  (within baseline normalization — expect null)
  H2: [-20, -10]  (within baseline normalization — expect null)
  H3: [-10,  -5]  (pre-ictal, outside baseline)
  H4: [ -5,  -1]  (pre-ictal, closest to onset)

H1 and H2 fall within the z-score normalization window [-30, -10] min.
Z-scored features there are near zero by construction. These horizons
serve as a methodological control: AUC should be near 0.5.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

import yaml

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "chbmit.yaml"

HORIZONS = {
    "H1_-30_-20": (-1800, -1200),
    "H2_-20_-10": (-1200, -600),
    "H3_-10_-5": (-600, -300),
    "H4_-5_-1": (-300, -60),
}

HORIZON_LABELS = {
    "H1_-30_-20": "[-30, -20] min",
    "H2_-20_-10": "[-20, -10] min",
    "H3_-10_-5": "[-10, -5] min",
    "H4_-5_-1": "[-5, -1] min",
}

FEATURE_SETS = {
    "spectral_radius_only": ["sr_mean"],
    "spacing_only": ["sp_mean", "sp_std", "sp_slope"],
    "geometry_only": ["sr_mean", "ep_mean", "sp_mean", "sp_std", "sp_slope"],
    "spectral_power_only": ["alpha_mean", "delta_mean"],
    "all_features": [
        "sp_mean", "sp_std", "sp_slope", "sp_raw_mean",
        "sr_mean", "ep_mean",
        "alpha_mean", "delta_mean",
        "med_nns_mean", "p10_nns_mean",
    ],
}


def log(msg):
    print(msg, flush=True)


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_trajectory_cache(cache_path):
    data = np.load(cache_path, allow_pickle=True)
    trajectories = {}
    for key in data.files:
        traj = json.loads(str(data[key]))
        for field in ["time_sec", "min_spacing_z", "min_spacing_raw",
                      "spectral_radius_z", "ep_score_z", "alpha_power_z",
                      "delta_power_z", "median_nns_z", "p10_nns_z"]:
            if field in traj:
                traj[field] = np.array(traj[field])
        trajectories[key] = traj
    return trajectories


def extract_window_features(traj, t_start, t_end):
    t = traj["time_sec"]
    mask = (t >= t_start) & (t < t_end)
    n_valid = mask.sum()

    if n_valid < 5:
        return None

    def safe_mean(arr):
        v = arr[mask]
        finite = np.isfinite(v)
        return float(np.mean(v[finite])) if finite.sum() > 0 else float("nan")

    def safe_std(arr):
        v = arr[mask]
        finite = np.isfinite(v)
        return float(np.std(v[finite])) if finite.sum() > 2 else float("nan")

    def safe_slope(arr):
        v = arr[mask]
        tt = t[mask]
        finite = np.isfinite(v)
        if finite.sum() < 5:
            return float("nan")
        tt_f = tt[finite] - tt[finite].mean()
        v_f = v[finite]
        denom = np.sum(tt_f ** 2)
        if denom == 0:
            return float("nan")
        return float(np.sum(tt_f * v_f) / denom)

    features = {
        "sp_mean": safe_mean(traj["min_spacing_z"]),
        "sp_std": safe_std(traj["min_spacing_z"]),
        "sp_slope": safe_slope(traj["min_spacing_z"]),
        "sp_raw_mean": safe_mean(traj["min_spacing_raw"]),
        "sr_mean": safe_mean(traj["spectral_radius_z"]),
        "ep_mean": safe_mean(traj["ep_score_z"]),
        "alpha_mean": safe_mean(traj["alpha_power_z"]),
        "delta_mean": safe_mean(traj["delta_power_z"]),
        "med_nns_mean": safe_mean(traj["median_nns_z"]),
        "p10_nns_mean": safe_mean(traj["p10_nns_z"]),
        "n_windows": int(n_valid),
    }

    return features


def build_dataset_for_horizon(trajectories, features_df, horizon_key, rng):
    t_start, t_end = HORIZONS[horizon_key]
    duration = t_end - t_start

    bl_start, bl_end = -1800, -600

    rows = []

    for _, row in features_df.iterrows():
        key = f"{row['subject_id']}_sz{int(row['seizure_idx_global'])}"
        if key not in trajectories:
            continue
        traj = trajectories[key]

        pre_feat = extract_window_features(traj, t_start, t_end)
        if pre_feat is None:
            continue

        sham_pool_start = bl_start
        sham_pool_end = min(bl_end, t_start - 30)

        if sham_pool_end - sham_pool_start < duration:
            sham_pool_start = bl_start
            sham_pool_end = bl_end
            if t_start >= bl_start and t_end <= bl_end:
                offset = rng.uniform(bl_start, bl_end - duration)
                while abs(offset - t_start) < 60 and bl_end - bl_start > duration + 120:
                    offset = rng.uniform(bl_start, bl_end - duration)
                sham_t_start = offset
                sham_t_end = offset + duration
            else:
                sham_t_start = rng.uniform(sham_pool_start, sham_pool_end - duration)
                sham_t_end = sham_t_start + duration
        else:
            sham_t_start = rng.uniform(sham_pool_start, sham_pool_end - duration)
            sham_t_end = sham_t_start + duration

        sham_feat = extract_window_features(traj, sham_t_start, sham_t_end)
        if sham_feat is None:
            continue

        pre_feat["label"] = 1
        pre_feat["subject_id"] = row["subject_id"]
        pre_feat["seizure_key"] = key
        pre_feat["group"] = row["group"]
        pre_feat["segment_type"] = "pre_ictal"

        sham_feat["label"] = 0
        sham_feat["subject_id"] = row["subject_id"]
        sham_feat["seizure_key"] = key
        sham_feat["group"] = row["group"]
        sham_feat["segment_type"] = "sham"

        rows.append(pre_feat)
        rows.append(sham_feat)

    return pd.DataFrame(rows)


def loso_classify(df, feature_cols, n_perm=200, seed=42):
    rng = np.random.default_rng(seed)

    X = df[feature_cols].values.copy()
    y = df["label"].values.copy()
    sids = df["subject_id"].values

    finite = np.all(np.isfinite(X), axis=1)
    X, y, sids = X[finite], y[finite], sids[finite]

    subjects = np.unique(sids)
    if len(subjects) < 3 or len(np.unique(y)) < 2:
        return {"error": "insufficient data"}

    def _loso(X_all, y_all, sid_all, subs):
        y_prob = np.full(len(y_all), np.nan)
        y_pred = np.full(len(y_all), -1)
        for sub in subs:
            tr = sid_all != sub
            te = sid_all == sub
            if tr.sum() < 5 or te.sum() == 0 or len(np.unique(y_all[tr])) < 2:
                continue
            sc = StandardScaler()
            clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                     max_iter=1000, random_state=42)
            clf.fit(sc.fit_transform(X_all[tr]), y_all[tr])
            y_prob[te] = clf.predict_proba(sc.transform(X_all[te]))[:, 1]
            y_pred[te] = clf.predict(sc.transform(X_all[te]))
        valid = y_pred >= 0
        return y_prob, y_pred, valid

    y_prob, y_pred, valid = _loso(X, y, sids, subjects)
    if valid.sum() < 10:
        return {"error": "too few valid predictions"}

    acc = float(accuracy_score(y[valid], y_pred[valid]))
    try:
        auc = float(roc_auc_score(y[valid], y_prob[valid]))
    except ValueError:
        auc = float("nan")

    fpr_curve, tpr_curve, _ = roc_curve(y[valid], y_prob[valid])
    sens_at_5pct_far = float(np.interp(0.05, fpr_curve, tpr_curve))
    sens_at_10pct_far = float(np.interp(0.10, fpr_curve, tpr_curve))

    null_aucs = np.zeros(n_perm)
    for p in range(n_perm):
        y_shuf = y.copy()
        for sub in subjects:
            sub_mask = sids == sub
            y_shuf[sub_mask] = rng.permutation(y_shuf[sub_mask])
        yp_null, _, v_null = _loso(X, y_shuf, sids, subjects)
        if v_null.sum() >= 10:
            try:
                null_aucs[p] = roc_auc_score(y_shuf[v_null], yp_null[v_null])
            except ValueError:
                null_aucs[p] = 0.5

    p_auc = float(np.mean(null_aucs >= auc))

    return {
        "accuracy": acc,
        "auc": auc,
        "sensitivity_at_5pct_far": sens_at_5pct_far,
        "sensitivity_at_10pct_far": sens_at_10pct_far,
        "n_valid": int(valid.sum()),
        "n_preictal": int((y[valid] == 1).sum()),
        "n_sham": int((y[valid] == 0).sum()),
        "n_subjects": len(subjects),
        "permutation_p_auc": p_auc,
        "null_auc_mean": float(np.mean(null_aucs)),
        "null_auc_std": float(np.std(null_aucs)),
        "n_permutations": n_perm,
        "features_used": feature_cols,
    }


def estimate_clinical_metrics(results_by_horizon, total_interictal_hours):
    metrics = {}
    earliest_significant = None

    for h_key in HORIZONS:
        h_results = results_by_horizon.get(h_key, {})
        best_auc = 0
        best_set = None
        best_r = None

        for fs_name, r in h_results.items():
            if isinstance(r, dict) and "auc" in r and not np.isnan(r["auc"]):
                if r["auc"] > best_auc:
                    best_auc = r["auc"]
                    best_set = fs_name
                    best_r = r

        if best_r is None:
            metrics[h_key] = {"best_feature_set": None, "auc": float("nan")}
            continue

        is_significant = best_r.get("permutation_p_auc", 1.0) < 0.05

        fa_per_hour = float("nan")
        if total_interictal_hours > 0 and "sensitivity_at_5pct_far" in best_r:
            fa_per_hour = 0.05 * 60 / max(1, total_interictal_hours)

        metrics[h_key] = {
            "best_feature_set": best_set,
            "auc": best_r["auc"],
            "accuracy": best_r["accuracy"],
            "sensitivity_at_5pct_far": best_r.get("sensitivity_at_5pct_far", float("nan")),
            "sensitivity_at_10pct_far": best_r.get("sensitivity_at_10pct_far", float("nan")),
            "permutation_p": best_r.get("permutation_p_auc", float("nan")),
            "significant": is_significant,
            "estimated_fa_per_hour_at_5pct_far": fa_per_hour,
        }

        if is_significant and earliest_significant is None:
            t_start, t_end = HORIZONS[h_key]
            earliest_significant = h_key

    if earliest_significant:
        t_start, _ = HORIZONS[earliest_significant]
        metrics["median_warning_time_min"] = float(abs(t_start) / 60)
    else:
        metrics["median_warning_time_min"] = float("nan")

    return metrics


def plot_auc_by_horizon(results_by_horizon, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    h_keys = list(HORIZONS.keys())
    x = np.arange(len(h_keys))
    x_labels = [HORIZON_LABELS[h] for h in h_keys]

    colors = {
        "spectral_radius_only": "#1b9e77",
        "spacing_only": "#d95f02",
        "geometry_only": "#7570b3",
        "spectral_power_only": "#e7298a",
        "all_features": "#333333",
    }

    width = 0.15
    offsets = np.linspace(-0.3, 0.3, len(FEATURE_SETS))

    for i, (fs_name, _) in enumerate(FEATURE_SETS.items()):
        aucs = []
        for h_key in h_keys:
            h_r = results_by_horizon.get(h_key, {})
            r = h_r.get(fs_name, {})
            auc = r.get("auc", float("nan")) if isinstance(r, dict) else float("nan")
            aucs.append(auc)

        color = colors.get(fs_name, "#999999")
        label = fs_name.replace("_", " ").title()
        ax.bar(x + offsets[i], aucs, width, label=label, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)

    ax.axhline(0.5, color="red", ls="--", lw=1.5, alpha=0.6, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("LOSO AUC", fontsize=12)
    ax.set_xlabel("Prediction Horizon (minutes before onset)", fontsize=12)
    ax.set_title("Prospective Detection: AUC by Horizon and Feature Set", fontsize=13, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(fig_dir / "auc_by_horizon.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_regime_stratified(regime_results, fig_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    h_keys = list(HORIZONS.keys())
    x = np.arange(len(h_keys))
    x_labels = [HORIZON_LABELS[h] for h in h_keys]

    best_fs = "all_features"

    for ax, (regime, label, color) in zip(axes, [
        ("narrowing", "Narrowing Regime", "#2166AC"),
        ("widening", "Widening Regime", "#B2182B"),
    ]):
        aucs_regime = []
        aucs_all = []
        for h_key in h_keys:
            r_regime = regime_results.get(regime, {}).get(h_key, {}).get(best_fs, {})
            auc_r = r_regime.get("auc", float("nan")) if isinstance(r_regime, dict) else float("nan")
            aucs_regime.append(auc_r)

        ax.bar(x, aucs_regime, 0.5, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(0.5, color="red", ls="--", lw=1.5, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel("LOSO AUC", fontsize=11)
        ax.set_title(f"{label} (all features)", fontsize=12, fontweight="bold")
        ax.set_ylim(0.3, 1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(fig_dir / "regime_stratified_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_clinical_summary(clinical_metrics, fig_dir):
    fig, ax = plt.subplots(figsize=(10, 4))

    h_keys = [h for h in HORIZONS if h in clinical_metrics and clinical_metrics[h].get("best_feature_set")]
    if not h_keys:
        plt.close(fig)
        return

    cols = ["Horizon", "Best Features", "AUC", "Sens@5%FA", "Sens@10%FA", "Sig?"]
    cell_text = []

    for h_key in h_keys:
        m = clinical_metrics[h_key]
        cell_text.append([
            HORIZON_LABELS[h_key],
            (m.get("best_feature_set", "?") or "?").replace("_", " "),
            f"{m.get('auc', float('nan')):.3f}",
            f"{m.get('sensitivity_at_5pct_far', float('nan')):.3f}",
            f"{m.get('sensitivity_at_10pct_far', float('nan')):.3f}",
            "Yes" if m.get("significant", False) else "No",
        ])

    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#4472C4")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#D9E2F3")

    wt = clinical_metrics.get("median_warning_time_min", float("nan"))
    title = "Prospective Detection: Clinical Summary"
    if not np.isnan(wt):
        title += f" (earliest warning: {wt:.0f} min)"
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(fig_dir / "clinical_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    log("=" * 70)
    log("PROSPECTIVE SEIZURE DETECTION")
    log("=" * 70)

    cfg = load_config()
    seed = cfg["random_seed"]
    rng = np.random.default_rng(seed)

    results_dir = Path(cfg["output"]["results_dir"]) / "analysis"
    fig_dir = Path(cfg["output"]["results_dir"]) / "figures" / "prospective"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log("\nLoading trajectory cache and features...")
    trajectories = load_trajectory_cache(results_dir / "trajectory_cache.npz")
    features_df = pd.read_csv(results_dir / "per_seizure_features.csv")
    log(f"  {len(trajectories)} trajectories, {len(features_df)} feature rows")

    total_interictal_hours = len(features_df) * 20 / 60

    log("\nNote: Horizons H1 and H2 overlap with the baseline normalization")
    log("window [-30, -10] min. Z-scored features are ~0 there by construction.")
    log("These horizons serve as methodological controls (expect AUC ~ 0.5).\n")

    all_results = {}
    regime_results = {"narrowing": {}, "widening": {}}

    for h_key, (t_start, t_end) in HORIZONS.items():
        log(f"\n{'='*50}")
        log(f"HORIZON: {HORIZON_LABELS[h_key]} ({t_start}s to {t_end}s)")
        log(f"{'='*50}")

        dataset = build_dataset_for_horizon(trajectories, features_df, h_key, rng)
        n_pre = (dataset["label"] == 1).sum()
        n_sham = (dataset["label"] == 0).sum()
        log(f"  Dataset: {n_pre} pre-ictal + {n_sham} sham = {len(dataset)} segments")

        if len(dataset) < 20:
            log(f"  SKIP: insufficient data")
            all_results[h_key] = {"error": "insufficient data"}
            continue

        horizon_results = {}
        for fs_name, feat_cols in FEATURE_SETS.items():
            available = [c for c in feat_cols if c in dataset.columns]
            if len(available) < len(feat_cols):
                missing = set(feat_cols) - set(available)
                log(f"  {fs_name}: missing features {missing}, using {available}")
            if not available:
                horizon_results[fs_name] = {"error": "no features available"}
                continue

            r = loso_classify(dataset, available, n_perm=200, seed=seed)
            if "error" not in r:
                sig = "*" if r["permutation_p_auc"] < 0.05 else ""
                log(f"  {fs_name}: AUC={r['auc']:.3f} (p={r['permutation_p_auc']:.3f}){sig}, "
                    f"acc={r['accuracy']:.1%}, sens@5%FA={r['sensitivity_at_5pct_far']:.3f}")
            else:
                log(f"  {fs_name}: {r['error']}")
            horizon_results[fs_name] = r

        all_results[h_key] = horizon_results

        for regime in ["narrowing", "widening"]:
            regime_df = dataset[dataset["group"] == regime]
            if len(regime_df) < 10:
                regime_results[regime][h_key] = {"error": "insufficient data"}
                continue

            regime_horizon = {}
            for fs_name, feat_cols in FEATURE_SETS.items():
                available = [c for c in feat_cols if c in regime_df.columns]
                if not available:
                    regime_horizon[fs_name] = {"error": "no features"}
                    continue
                r = loso_classify(regime_df, available, n_perm=200, seed=seed)
                if "error" not in r:
                    sig = "*" if r["permutation_p_auc"] < 0.05 else ""
                    log(f"    {regime}/{fs_name}: AUC={r['auc']:.3f}{sig}")
                regime_horizon[fs_name] = r
            regime_results[regime][h_key] = regime_horizon

    log(f"\n{'='*70}")
    log("CLINICAL METRICS")
    log(f"{'='*70}")

    clinical = estimate_clinical_metrics(all_results, total_interictal_hours)
    for h_key in HORIZONS:
        m = clinical.get(h_key, {})
        if m.get("best_feature_set"):
            log(f"  {HORIZON_LABELS[h_key]}: AUC={m['auc']:.3f} ({m['best_feature_set']}), "
                f"sens@5%FA={m['sensitivity_at_5pct_far']:.3f}, "
                f"sig={m['significant']}")
    wt = clinical.get("median_warning_time_min", float("nan"))
    if not np.isnan(wt):
        log(f"\n  Earliest significant detection: {wt:.0f} min before onset")
    else:
        log(f"\n  No horizon reached significance")

    log("\nGenerating figures...")
    plot_auc_by_horizon(all_results, fig_dir)
    plot_regime_stratified(regime_results, fig_dir)
    plot_clinical_summary(clinical, fig_dir)

    log("\nSaving results...")

    def sanitize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    final = sanitize({
        "horizons": {k: {"sec": v, "label": HORIZON_LABELS[k]} for k, v in HORIZONS.items()},
        "feature_sets": FEATURE_SETS,
        "results_by_horizon": all_results,
        "regime_stratified": regime_results,
        "clinical_metrics": clinical,
        "notes": {
            "H1_H2_control": "Horizons H1 and H2 overlap with baseline normalization "
                             "window. Z-scored features are ~0 by construction. "
                             "AUC near 0.5 expected and validates methodology.",
            "features_available": "spectral_slope and dfa_alpha are NOT available in "
                                  "trajectory cache (would require full reprocessing). "
                                  "Available: spacing, spectral_radius, ep_score, "
                                  "alpha/delta power, median/p10 NNS.",
            "sham_design": "Each pre-ictal segment matched with a same-seizure "
                           "interictal baseline segment of equal duration.",
        },
        "total_interictal_hours_approx": total_interictal_hours,
    })

    with open(results_dir / "prospective_detection.json", "w") as f:
        json.dump(final, f, indent=2, default=str)

    log(f"\nResults: {results_dir / 'prospective_detection.json'}")
    log(f"Figures: {fig_dir}")
    log(f"{'='*70}")
    log("PROSPECTIVE DETECTION COMPLETE")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
