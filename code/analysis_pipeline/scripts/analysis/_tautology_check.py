"""Tautology check for the two-regime classifier.

Groups are defined by raw_spacing_change sign. This script verifies:
1. raw_spacing_change is NOT in the classifier feature set
2. The spectral-only model (no geometry features) classifies above chance
3. Leave-one-feature-out ablation: no single feature drives the result

Uses the same LOSO logistic regression as _regime_precision.py.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

import yaml

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "chbmit.yaml"

ALL_FEATURES = [
    "spectral_slope_change", "dfa_change",
    "spectral_radius_change", "residual_var_change", "pre_ep_score_z",
]

SPECTRAL_FEATURES = ["spectral_slope_change", "dfa_change"]
GEOMETRY_FEATURES = ["spectral_radius_change", "residual_var_change", "pre_ep_score_z"]


def log(msg):
    print(msg, flush=True)


def loso_classify(df, feature_cols, label_col="group_label"):
    X = df[feature_cols].values.copy()
    y = df[label_col].values.copy()
    sids = df["subject_id"].values

    finite = np.all(np.isfinite(X), axis=1)
    X, y, sids = X[finite], y[finite], sids[finite]
    subjects = np.unique(sids)

    y_prob = np.full(len(y), np.nan)
    y_pred = np.full(len(y), -1)

    for sub in subjects:
        tr = sids != sub
        te = sids == sub
        if tr.sum() < 5 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler()
        clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                 max_iter=1000, random_state=42)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        y_prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        y_pred[te] = clf.predict(sc.transform(X[te]))

    valid = y_pred >= 0
    if valid.sum() < 10:
        return {"error": "too few valid predictions"}

    acc = float(accuracy_score(y[valid], y_pred[valid]))
    try:
        auc = float(roc_auc_score(y[valid], y_prob[valid]))
    except ValueError:
        auc = float("nan")

    return {
        "features": feature_cols,
        "n_features": len(feature_cols),
        "accuracy": acc,
        "auc": auc,
        "n_valid": int(valid.sum()),
    }


def main():
    log("=" * 70)
    log("TAUTOLOGY CHECK")
    log("=" * 70)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    results_dir = Path(cfg["output"]["results_dir"]) / "analysis"
    fig_dir = Path(cfg["output"]["results_dir"]) / "figures" / "publication"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_dir / "per_seizure_features.csv")
    df["group_label"] = (df["raw_spacing_change"] < 0).astype(int)

    results = {}

    log("\n1. Confirm raw_spacing_change is NOT a classifier feature")
    defining_in_features = "raw_spacing_change" in ALL_FEATURES
    log(f"   raw_spacing_change in feature set: {defining_in_features}")
    results["defining_variable_excluded"] = not defining_in_features

    log("\n2. Full model (all 5 features)")
    full = loso_classify(df, ALL_FEATURES)
    log(f"   Accuracy: {full['accuracy']:.1%}, AUC: {full['auc']:.3f}")
    results["full_model"] = full

    log("\n3. Spectral-only model (no geometry)")
    spectral = loso_classify(df, SPECTRAL_FEATURES)
    log(f"   Accuracy: {spectral['accuracy']:.1%}, AUC: {spectral['auc']:.3f}")
    results["spectral_only"] = spectral
    results["spectral_only_above_chance"] = spectral["auc"] > 0.5

    log("\n4. Geometry-only model (no spectral)")
    geom = loso_classify(df, GEOMETRY_FEATURES)
    log(f"   Accuracy: {geom['accuracy']:.1%}, AUC: {geom['auc']:.3f}")
    results["geometry_only"] = geom

    log("\n5. Leave-one-feature-out ablation")
    ablation = {}
    for drop_feat in ALL_FEATURES:
        remaining = [f for f in ALL_FEATURES if f != drop_feat]
        r = loso_classify(df, remaining)
        delta_auc = full["auc"] - r["auc"]
        ablation[drop_feat] = {
            **r,
            "dropped_feature": drop_feat,
            "delta_auc_from_full": float(delta_auc),
        }
        log(f"   Drop {drop_feat}: AUC={r['auc']:.3f} (delta={delta_auc:+.3f})")

    results["leave_one_out_ablation"] = ablation

    no_single_driver = all(
        v["auc"] > 0.7 for v in ablation.values()
    )
    results["no_single_feature_drives_result"] = no_single_driver
    log(f"\n   No single feature drives result (all ablated AUC > 0.7): {no_single_driver}")

    log("\n6. Correlation between defining variable and features")
    corr_with_defining = {}
    for feat in ALL_FEATURES:
        mask = np.isfinite(df[feat].values) & np.isfinite(df["raw_spacing_change"].values)
        if mask.sum() < 10:
            corr_with_defining[feat] = {"error": "insufficient data"}
            continue
        from scipy import stats as sp_stats
        r, p = sp_stats.spearmanr(df.loc[mask, "raw_spacing_change"], df.loc[mask, feat])
        corr_with_defining[feat] = {"spearman_r": float(r), "p_value": float(p)}
        log(f"   {feat} vs raw_spacing_change: rho={r:.3f}, p={p:.2e}")

    results["correlation_with_defining_variable"] = corr_with_defining

    log("\n--- TAUTOLOGY VERDICT ---")
    if not defining_in_features and spectral["auc"] > 0.6 and no_single_driver:
        verdict = "NOT TAUTOLOGICAL"
        log(f"   {verdict}")
        log("   The defining variable is excluded, spectral-only features classify")
        log("   above chance, and no single feature drives the full model.")
    else:
        verdict = "POTENTIAL TAUTOLOGY CONCERN"
        log(f"   {verdict}")

    results["verdict"] = verdict

    log("\nPlotting ablation summary...")

    fig, ax = plt.subplots(figsize=(10, 5))

    model_names = ["Full (5)"]
    model_aucs = [full["auc"]]
    model_colors = ["#5D3A9B"]

    for feat in ALL_FEATURES:
        ab = ablation[feat]
        short = feat.replace("_change", "").replace("pre_", "")
        model_names.append(f"Drop {short}")
        model_aucs.append(ab["auc"])
        if feat in GEOMETRY_FEATURES:
            model_colors.append("#B2182B")
        else:
            model_colors.append("#E66100")

    model_names.extend(["Spectral\nonly (2)", "Geometry\nonly (3)"])
    model_aucs.extend([spectral["auc"], geom["auc"]])
    model_colors.extend(["#E66100", "#B2182B"])

    bars = ax.barh(range(len(model_names)), model_aucs, color=model_colors,
                   edgecolor="black", linewidth=0.6, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_xlabel("LOSO AUC", fontsize=11)
    ax.set_title("Tautology Check: Feature Ablation", fontsize=12, fontweight="bold")
    ax.axvline(0.5, color="red", ls="--", lw=1, alpha=0.5, label="Chance")
    ax.set_xlim(0.4, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    for i, v in enumerate(model_aucs):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_tautology_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved: {fig_dir / 'fig_tautology_ablation.png'}")

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
        return obj

    with open(results_dir / "tautology_check.json", "w") as f:
        json.dump(sanitize(results), f, indent=2, default=str)

    log(f"\nResults: {results_dir / 'tautology_check.json'}")
    log("=" * 70)


if __name__ == "__main__":
    main()
