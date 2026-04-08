"""Two-regime precision analysis.

Loads per_seizure_features.csv and tests:
14a. Locked group definition (raw_spacing_change sign)
14b. Cross-validated classifier separability (LOSO logistic regression)
14c. Timing/clinical differences between regimes
14d. Incremental value of geometry over spectral features

All analyses use the locked definition: narrowing = raw_spacing_change < 0.
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


def log(msg):
    print(msg, flush=True)


def load_data():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    results_dir = Path(cfg["output"]["results_dir"]) / "analysis"
    df = pd.read_csv(results_dir / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)
    return df, cfg, results_dir


def run_loso_classifier(df, feature_cols, label_col="group_label", n_perm=1000, seed=42):
    rng = np.random.default_rng(seed)
    subjects = df["subject_id"].unique()

    X = df[feature_cols].values.copy()
    y = df[label_col].values.copy()
    subject_ids = df["subject_id"].values

    finite_mask = np.all(np.isfinite(X), axis=1)
    X = X[finite_mask]
    y = y[finite_mask]
    subject_ids = subject_ids[finite_mask]

    subjects = np.unique(subject_ids)
    if len(subjects) < 3:
        return {"error": "insufficient subjects"}

    def _loso_predict(X_all, y_all, sid_all, subs):
        y_pred_prob = np.full(len(y_all), np.nan)
        y_pred_class = np.full(len(y_all), -1)

        for test_sub in subs:
            train_mask = sid_all != test_sub
            test_mask = sid_all == test_sub

            if train_mask.sum() < 5 or test_mask.sum() == 0:
                continue
            if len(np.unique(y_all[train_mask])) < 2:
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_all[train_mask])
            X_test = scaler.transform(X_all[test_mask])

            clf = LogisticRegression(
                penalty="l2", C=1.0, solver="lbfgs",
                max_iter=1000, random_state=42,
            )
            clf.fit(X_train, y_all[train_mask])

            y_pred_prob[test_mask] = clf.predict_proba(X_test)[:, 1]
            y_pred_class[test_mask] = clf.predict(X_test)

        valid = y_pred_class >= 0
        return y_pred_prob, y_pred_class, valid

    y_prob, y_class, valid = _loso_predict(X, y, subject_ids, subjects)

    if valid.sum() < 10:
        return {"error": "too few valid predictions"}

    acc = accuracy_score(y[valid], y_class[valid])
    try:
        auc = roc_auc_score(y[valid], y_prob[valid])
    except ValueError:
        auc = float("nan")

    null_accs = np.zeros(n_perm)
    null_aucs = np.zeros(n_perm)
    for p in range(n_perm):
        y_shuf = y.copy()
        for sub in subjects:
            sub_mask = subject_ids == sub
            y_shuf[sub_mask] = rng.permutation(y_shuf[sub_mask])

        yp_null, yc_null, v_null = _loso_predict(X, y_shuf, subject_ids, subjects)
        if v_null.sum() >= 10:
            null_accs[p] = accuracy_score(y_shuf[v_null], yc_null[v_null])
            try:
                null_aucs[p] = roc_auc_score(y_shuf[v_null], yp_null[v_null])
            except ValueError:
                null_aucs[p] = 0.5

    p_acc = float(np.mean(null_accs >= acc))
    p_auc = float(np.mean(null_aucs >= auc))

    return {
        "accuracy": float(acc),
        "auc": float(auc),
        "n_valid": int(valid.sum()),
        "n_total": len(y),
        "n_subjects": len(subjects),
        "permutation_p_accuracy": p_acc,
        "permutation_p_auc": p_auc,
        "n_permutations": n_perm,
        "null_accuracy_mean": float(np.mean(null_accs)),
        "null_accuracy_std": float(np.std(null_accs)),
        "null_auc_mean": float(np.mean(null_aucs)),
        "null_auc_std": float(np.std(null_aucs)),
        "features_used": feature_cols,
        "y_true": y[valid].tolist(),
        "y_prob": y_prob[valid].tolist(),
    }


def run_timing_comparison(df):
    narrowing = df[df["group_label"] == 1]
    widening = df[df["group_label"] == 0]

    comparisons = {}
    for col, label in [
        ("min_spacing_time_sec", "Time of minimum spacing (sec)"),
        ("seizure_duration_sec", "Seizure duration (sec)"),
        ("preictal_slope", "Pre-ictal slope"),
        ("baseline_std", "Baseline std"),
    ]:
        vals_n = narrowing[col].dropna().values
        vals_w = widening[col].dropna().values

        if len(vals_n) < 3 or len(vals_w) < 3:
            comparisons[col] = {"error": "insufficient data"}
            continue

        u_stat, p_val = sp_stats.mannwhitneyu(vals_n, vals_w, alternative="two-sided")
        r_rb = 1 - (2 * u_stat) / (len(vals_n) * len(vals_w))

        comparisons[col] = {
            "label": label,
            "median_narrowing": float(np.median(vals_n)),
            "median_widening": float(np.median(vals_w)),
            "mean_narrowing": float(np.mean(vals_n)),
            "mean_widening": float(np.mean(vals_w)),
            "u_stat": float(u_stat),
            "p_value": float(p_val),
            "rank_biserial_r": float(r_rb),
            "n_narrowing": len(vals_n),
            "n_widening": len(vals_w),
        }

    return comparisons


def run_incremental_value(df, seed=42):
    spectral_features = ["spectral_slope_change", "dfa_change"]
    geometry_features = ["spectral_radius_change", "residual_var_change", "pre_ep_score_z"]
    all_features = spectral_features + geometry_features

    y = df["group_label"].values
    subjects = df["subject_id"].values

    results = {}

    for model_name, feat_cols in [
        ("spectral_only", spectral_features),
        ("spectral_plus_geometry", all_features),
    ]:
        X = df[feat_cols].values.copy()
        finite_mask = np.all(np.isfinite(X), axis=1)
        X_f = X[finite_mask]
        y_f = y[finite_mask]
        sid_f = subjects[finite_mask]

        unique_subs = np.unique(sid_f)
        y_prob = np.full(len(y_f), np.nan)

        for test_sub in unique_subs:
            train = sid_f != test_sub
            test = sid_f == test_sub
            if train.sum() < 5 or test.sum() == 0:
                continue
            if len(np.unique(y_f[train])) < 2:
                continue

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_f[train])
            X_te = scaler.transform(X_f[test])

            clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
            clf.fit(X_tr, y_f[train])
            y_prob[test] = clf.predict_proba(X_te)[:, 1]

        valid = np.isfinite(y_prob)
        if valid.sum() < 10:
            results[model_name] = {"error": "insufficient predictions"}
            continue

        acc = accuracy_score(y_f[valid], (y_prob[valid] >= 0.5).astype(int))
        try:
            auc = roc_auc_score(y_f[valid], y_prob[valid])
        except ValueError:
            auc = float("nan")

        scaler_full = StandardScaler()
        X_full = scaler_full.fit_transform(X_f)
        clf_full = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
        clf_full.fit(X_full, y_f)

        log_likelihood = float(np.sum(
            y_f * np.log(clf_full.predict_proba(X_full)[:, 1].clip(1e-10, 1 - 1e-10)) +
            (1 - y_f) * np.log(clf_full.predict_proba(X_full)[:, 0].clip(1e-10, 1 - 1e-10))
        ))
        n_params = len(feat_cols) + 1
        bic = -2 * log_likelihood + n_params * np.log(len(y_f))

        results[model_name] = {
            "features": feat_cols,
            "n_features": len(feat_cols),
            "loso_accuracy": float(acc),
            "loso_auc": float(auc),
            "full_data_log_likelihood": log_likelihood,
            "bic": float(bic),
            "n_samples": int(valid.sum()),
            "y_true": y_f[valid].tolist(),
            "y_prob": y_prob[valid].tolist(),
        }

    if "spectral_only" in results and "spectral_plus_geometry" in results:
        if "error" not in results["spectral_only"] and "error" not in results["spectral_plus_geometry"]:
            delta_bic = results["spectral_only"]["bic"] - results["spectral_plus_geometry"]["bic"]
            delta_auc = results["spectral_plus_geometry"]["loso_auc"] - results["spectral_only"]["loso_auc"]
            lr_stat = 2 * (results["spectral_plus_geometry"]["full_data_log_likelihood"] -
                           results["spectral_only"]["full_data_log_likelihood"])
            df_diff = results["spectral_plus_geometry"]["n_features"] - results["spectral_only"]["n_features"]
            lr_p = float(sp_stats.chi2.sf(max(0, lr_stat), df_diff))

            results["comparison"] = {
                "delta_bic": float(delta_bic),
                "bic_favors": "spectral_plus_geometry" if delta_bic > 0 else "spectral_only",
                "delta_auc": float(delta_auc),
                "likelihood_ratio_stat": float(lr_stat),
                "likelihood_ratio_p": lr_p,
                "likelihood_ratio_df": df_diff,
                "geometry_adds_value": delta_bic > 2 and lr_p < 0.05,
            }

    return results


def plot_classifier_results(clf_results, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    if "error" in clf_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    y_true = np.array(clf_results["y_true"])
    y_prob = np.array(clf_results["y_prob"])
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, "b-", lw=2, label=f"AUC = {clf_results['auc']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("LOSO ROC Curve (5 Change Metrics)")
        ax.legend()

    ax = axes[1]
    ax.bar(["Real", "Null (mean)"],
           [clf_results["accuracy"], clf_results["null_accuracy_mean"]],
           yerr=[0, clf_results["null_accuracy_std"]],
           color=["steelblue", "gray"], alpha=0.7, edgecolor="black")
    ax.axhline(0.5, color="red", ls="--", lw=1, label="Chance")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Classifier: {clf_results['accuracy']:.1%} (p={clf_results['permutation_p_accuracy']:.4f})")
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(fig_dir / "classifier_roc.png", dpi=200)
    plt.close(fig)


def plot_timing_comparison(timing_results, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    valid_cols = [k for k, v in timing_results.items() if "error" not in v]
    if not valid_cols:
        return

    n = len(valid_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, valid_cols):
        r = timing_results[col]
        ax.bar(["Narrowing", "Widening"],
               [r["median_narrowing"], r["median_widening"]],
               color=["blue", "red"], alpha=0.6, edgecolor="black")
        ax.set_title(f"{r['label']}\np={r['p_value']:.4f}, r={r['rank_biserial_r']:.3f}")
        ax.set_ylabel("Median value")

    fig.tight_layout()
    fig.savefig(fig_dir / "timing_comparison.png", dpi=200)
    plt.close(fig)


def plot_incremental_value(incr_results, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)

    models = []
    aucs = []
    labels = []
    for name in ["spectral_only", "spectral_plus_geometry"]:
        if name in incr_results and "error" not in incr_results[name]:
            models.append(name)
            aucs.append(incr_results[name]["loso_auc"])
            labels.append(f"{name}\n({incr_results[name]['n_features']} feat)")

    if len(models) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    colors = ["orange", "steelblue"]
    ax.bar(labels, aucs, color=colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("LOSO AUC")
    ax.set_title("Incremental Value: Spectral vs Spectral + Geometry")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="red", ls="--", lw=1)

    if "comparison" in incr_results:
        c = incr_results["comparison"]
        ax.text(0.5, 0.95,
                f"ΔAUC = {c['delta_auc']:.3f}\nΔBIC = {c['delta_bic']:.1f}\nLR p = {c['likelihood_ratio_p']:.4f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax = axes[1]
    for name, color, label in zip(models, colors, labels):
        r = incr_results[name]
        if "y_true" in r and "y_prob" in r:
            yt = np.array(r["y_true"])
            yp = np.array(r["y_prob"])
            if len(np.unique(yt)) == 2:
                fpr, tpr, _ = roc_curve(yt, yp)
                ax.plot(fpr, tpr, color=color, lw=2,
                        label=f"{name} (AUC={r['loso_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Comparison")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_dir / "incremental_value.png", dpi=200)
    plt.close(fig)


def main():
    log("=" * 70)
    log("TWO-REGIME PRECISION ANALYSIS")
    log("=" * 70)

    df, cfg, results_dir = load_data()
    fig_dir = Path(cfg["output"]["results_dir"]) / "figures" / "regime_precision"
    fig_dir.mkdir(parents=True, exist_ok=True)

    log(f"\n14a. Locked group definition: raw_spacing_change sign")
    n_narrow = int((df["group_label"] == 1).sum())
    n_wide = int((df["group_label"] == 0).sum())
    log(f"   Narrowing (raw_spacing_change < 0): {n_narrow}")
    log(f"   Widening  (raw_spacing_change >= 0): {n_wide}")

    log(f"\n14b. Cross-validated classifier (LOSO, 5 change metrics)...")
    sig_features = [
        "spectral_slope_change", "dfa_change",
        "spectral_radius_change", "residual_var_change", "pre_ep_score_z",
    ]
    clf_results = run_loso_classifier(df, sig_features, n_perm=1000, seed=42)
    if "error" not in clf_results:
        log(f"   Accuracy: {clf_results['accuracy']:.1%} (null: {clf_results['null_accuracy_mean']:.1%} +/- {clf_results['null_accuracy_std']:.1%})")
        log(f"   AUC: {clf_results['auc']:.3f} (null: {clf_results['null_auc_mean']:.3f} +/- {clf_results['null_auc_std']:.3f})")
        log(f"   Permutation p (accuracy): {clf_results['permutation_p_accuracy']:.4f}")
        log(f"   Permutation p (AUC): {clf_results['permutation_p_auc']:.4f}")
    else:
        log(f"   ERROR: {clf_results['error']}")

    plot_classifier_results(clf_results, fig_dir)

    log(f"\n14c. Timing and clinical differences...")
    timing_results = run_timing_comparison(df)
    for col, r in timing_results.items():
        if "error" not in r:
            log(f"   {r['label']}: narrow={r['median_narrowing']:.1f}, wide={r['median_widening']:.1f}, "
                f"p={r['p_value']:.4f}, r={r['rank_biserial_r']:.3f}")
        else:
            log(f"   {col}: {r['error']}")

    plot_timing_comparison(timing_results, fig_dir)

    log(f"\n14d. Incremental value of geometry over spectral features...")
    incr_results = run_incremental_value(df, seed=42)
    for name in ["spectral_only", "spectral_plus_geometry"]:
        if name in incr_results and "error" not in incr_results[name]:
            r = incr_results[name]
            log(f"   {name}: AUC={r['loso_auc']:.3f}, BIC={r['bic']:.1f}")

    if "comparison" in incr_results:
        c = incr_results["comparison"]
        log(f"   delta_AUC = {c['delta_auc']:.3f}")
        log(f"   delta_BIC = {c['delta_bic']:.1f} (favors: {c['bic_favors']})")
        log(f"   LR test: chi2={c['likelihood_ratio_stat']:.2f}, p={c['likelihood_ratio_p']:.4f}")
        log(f"   Geometry adds value: {c['geometry_adds_value']}")

    plot_incremental_value(incr_results, fig_dir)

    log(f"\nSaving results...")

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
        "group_definition": "raw_spacing_change < 0 => narrowing (label=1), >= 0 => widening (label=0)",
        "n_narrowing": n_narrow,
        "n_widening": n_wide,
        "classifier": {k: v for k, v in clf_results.items() if k not in ("y_true", "y_prob")},
        "timing_comparison": timing_results,
        "incremental_value": {k: {kk: vv for kk, vv in v.items() if kk not in ("y_true", "y_prob")}
                              if isinstance(v, dict) else v
                              for k, v in incr_results.items()},
    })

    with open(results_dir / "regime_precision.json", "w") as f:
        json.dump(final, f, indent=2, default=str)

    log(f"\n{'=' * 70}")
    log("REGIME PRECISION ANALYSIS COMPLETE")
    log(f"Results: {results_dir / 'regime_precision.json'}")
    log(f"Figures: {fig_dir}")
    log(f"{'=' * 70}")


if __name__ == "__main__":
    main()
