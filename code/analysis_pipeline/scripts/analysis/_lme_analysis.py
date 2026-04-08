"""Linear mixed-effects models for cross-band criticality analysis.

Fits LME models to test band effects on criticality metrics while
properly accounting for subject-level nesting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

CMCC_ROOT = Path(__file__).resolve().parent.parent
RESULTS_HG = CMCC_ROOT / "results" / "group_all_subjects.json"
RESULTS_BB = CMCC_ROOT / "results_broadband" / "group_all_subjects.json"
FIG_DIR = CMCC_ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def load_paired_data():
    with open(RESULTS_HG) as f:
        hg_data = json.load(f)
    with open(RESULTS_BB) as f:
        bb_data = json.load(f)

    hg_ok = {s["subject"]: s for s in hg_data if s.get("status") == "OK"}
    bb_ok = {s["subject"]: s for s in bb_data if s.get("status") == "OK"}
    common = sorted(set(hg_ok) & set(bb_ok))
    return hg_ok, bb_ok, common


def build_long_df(hg_ok, bb_ok, common):
    rows = []
    for subj in common:
        site = subj[:2]
        for band_label, data in [("HG", hg_ok[subj]), ("BB", bb_ok[subj])]:
            row = {
                "subject": subj,
                "site": site,
                "band": band_label,
                "tau": data.get("tau"),
                "sigma": data.get("branching_sigma"),
                "lzc": data.get("lzc_normalized"),
                "dfa": data.get("dfa_alpha"),
                "n_avalanches": data.get("n_avalanches"),
                "n_channels": data.get("n_channels"),
            }
            lzc_g = data.get("task_contrast_g")
            if lzc_g is not None:
                row["lzc_contrast_g"] = lzc_g
            dec = data.get("decoding", [])
            for d in dec:
                if d["n"] == 10:
                    row["dec_diff_10"] = d["diff_most_least"]
                if d["n"] == 5:
                    row["dec_diff_5"] = d["diff_most_least"]
            rows.append(row)
    return pd.DataFrame(rows)


def fit_lme_models(df):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        log("ERROR: statsmodels not installed")
        return {}

    results = {}
    models_spec = [
        ("sigma_band", "sigma ~ band", "subject"),
        ("lzc_band", "lzc ~ band", "subject"),
        ("dfa_band", "dfa ~ band", "subject"),
        ("tau_band", "tau ~ band", "subject"),
    ]

    for name, formula, groups in models_spec:
        try:
            model = smf.mixedlm(formula, df, groups=df[groups])
            fit = model.fit(reml=True)
            fe = fit.fe_params.to_dict()
            se = fit.bse_fe.to_dict()
            pvals = fit.pvalues.to_dict()
            ci = fit.conf_int()

            results[name] = {
                "formula": formula,
                "n_obs": int(fit.nobs),
                "n_groups": int(fit.k_fe + fit.k_re),
                "fixed_effects": {
                    k: {
                        "coef": float(fe[k]),
                        "se": float(se[k]),
                        "p": float(pvals[k]),
                        "ci_lo": float(ci.loc[k, 0]),
                        "ci_hi": float(ci.loc[k, 1]),
                    }
                    for k in fe
                },
                "random_effects_var": float(fit.cov_re.iloc[0, 0]) if hasattr(fit, "cov_re") else None,
                "aic": float(fit.aic) if hasattr(fit, "aic") else None,
                "bic": float(fit.bic) if hasattr(fit, "bic") else None,
                "converged": fit.converged,
            }
            log(f"  {name}: band coef={fe.get('band[T.HG]', fe.get('band[T.BB]', 'N/A')):.4f}, p={pvals.get('band[T.HG]', pvals.get('band[T.BB]', 'N/A')):.4f}, converged={fit.converged}")
        except Exception as e:
            log(f"  {name}: FAILED — {e}")
            results[name] = {"error": str(e)}

    lzc_df = df.dropna(subset=["lzc_contrast_g"])
    if len(lzc_df) >= 6:
        try:
            model = smf.mixedlm("lzc_contrast_g ~ band", lzc_df, groups=lzc_df["subject"])
            fit = model.fit(reml=True)
            fe = fit.fe_params.to_dict()
            pvals = fit.pvalues.to_dict()
            ci = fit.conf_int()
            results["lzc_contrast_band"] = {
                "formula": "lzc_contrast_g ~ band",
                "n_obs": int(fit.nobs),
                "fixed_effects": {
                    k: {
                        "coef": float(fe[k]),
                        "se": float(fit.bse_fe[k]),
                        "p": float(pvals[k]),
                        "ci_lo": float(ci.loc[k, 0]),
                        "ci_hi": float(ci.loc[k, 1]),
                    }
                    for k in fe
                },
                "random_effects_var": float(fit.cov_re.iloc[0, 0]) if hasattr(fit, "cov_re") else None,
                "converged": fit.converged,
            }
            log(f"  lzc_contrast_band: band coef={fe.get('band[T.HG]', 'N/A'):.4f}, p={pvals.get('band[T.HG]', 'N/A'):.4f}")
        except Exception as e:
            log(f"  lzc_contrast_band: FAILED — {e}")
            results["lzc_contrast_band"] = {"error": str(e)}

    dec_df = df.dropna(subset=["dec_diff_10"])
    if len(dec_df) >= 6:
        try:
            model = smf.mixedlm("dec_diff_10 ~ band + sigma + tau", dec_df, groups=dec_df["subject"])
            fit = model.fit(reml=True)
            fe = fit.fe_params.to_dict()
            pvals = fit.pvalues.to_dict()
            ci = fit.conf_int()
            results["decoding_predictors"] = {
                "formula": "dec_diff_10 ~ band + sigma + tau",
                "n_obs": int(fit.nobs),
                "fixed_effects": {
                    k: {
                        "coef": float(fe[k]),
                        "se": float(fit.bse_fe[k]),
                        "p": float(pvals[k]),
                        "ci_lo": float(ci.loc[k, 0]),
                        "ci_hi": float(ci.loc[k, 1]),
                    }
                    for k in fe
                },
                "random_effects_var": float(fit.cov_re.iloc[0, 0]) if hasattr(fit, "cov_re") else None,
                "converged": fit.converged,
            }
            log(f"  decoding_predictors: converged={fit.converged}")
        except Exception as e:
            log(f"  decoding_predictors: FAILED — {e}")
            results["decoding_predictors"] = {"error": str(e)}

    hg_rows = df[df["band"] == "HG"].set_index("subject")
    bb_rows = df[df["band"] == "BB"].set_index("subject")
    cross_subjects = sorted(set(hg_rows.index) & set(bb_rows.index))
    if len(cross_subjects) >= 5:
        cross_df = pd.DataFrame({
            "subject": cross_subjects,
            "site": [s[:2] for s in cross_subjects],
            "bb_sigma": [bb_rows.loc[s, "sigma"] for s in cross_subjects],
            "hg_sigma": [hg_rows.loc[s, "sigma"] for s in cross_subjects],
            "hg_tau": [hg_rows.loc[s, "tau"] for s in cross_subjects],
        })
        try:
            model = smf.mixedlm("bb_sigma ~ hg_sigma + hg_tau", cross_df, groups=cross_df["site"])
            fit = model.fit(reml=True)
            fe = fit.fe_params.to_dict()
            pvals = fit.pvalues.to_dict()
            ci = fit.conf_int()
            results["cross_band_prediction"] = {
                "formula": "bb_sigma ~ hg_sigma + hg_tau",
                "n_obs": int(fit.nobs),
                "fixed_effects": {
                    k: {
                        "coef": float(fe[k]),
                        "se": float(fit.bse_fe[k]),
                        "p": float(pvals[k]),
                        "ci_lo": float(ci.loc[k, 0]),
                        "ci_hi": float(ci.loc[k, 1]),
                    }
                    for k in fe
                },
                "converged": fit.converged,
            }
            log(f"  cross_band_prediction: hg_sigma coef={fe.get('hg_sigma', 'N/A'):.4f}")
        except Exception as e:
            log(f"  cross_band_prediction: FAILED — {e}")
            results["cross_band_prediction"] = {"error": str(e)}

    return results


def plot_diagnostics(df):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    paired_subjs = sorted(set(df[df["band"] == "HG"]["subject"]) & set(df[df["band"] == "BB"]["subject"]))
    hg = df[df["band"] == "HG"].set_index("subject")
    bb = df[df["band"] == "BB"].set_index("subject")

    ax = axes[0, 0]
    for s in paired_subjs:
        ax.plot([0, 1], [hg.loc[s, "sigma"], bb.loc[s, "sigma"]], "o-", color="gray", alpha=0.4, markersize=4)
    ax.plot([0, 1], [hg.loc[paired_subjs, "sigma"].mean(), bb.loc[paired_subjs, "sigma"].mean()],
            "s-", color="red", markersize=10, linewidth=2, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HG (70-150 Hz)", "BB (1-200 Hz)"])
    ax.set_ylabel("Branching ratio (sigma)")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("A: Sigma by band")

    ax = axes[0, 1]
    lzc_subjs = [s for s in paired_subjs if "lzc_contrast_g" in hg.columns and not pd.isna(hg.loc[s].get("lzc_contrast_g")) and not pd.isna(bb.loc[s].get("lzc_contrast_g"))]
    for s in lzc_subjs:
        ax.plot([0, 1], [hg.loc[s, "lzc_contrast_g"], bb.loc[s, "lzc_contrast_g"]], "o-", color="gray", alpha=0.4, markersize=4)
    if lzc_subjs:
        ax.plot([0, 1], [hg.loc[lzc_subjs, "lzc_contrast_g"].mean(), bb.loc[lzc_subjs, "lzc_contrast_g"].mean()],
                "s-", color="blue", markersize=10, linewidth=2, zorder=5)
    ax.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HG", "BB"])
    ax.set_ylabel("LZc contrast (Hedges g)")
    ax.set_title("B: LZc contrast sign reversal")

    ax = axes[1, 0]
    for s in paired_subjs:
        ax.plot([0, 1], [hg.loc[s, "lzc"], bb.loc[s, "lzc"]], "o-", color="gray", alpha=0.4, markersize=4)
    ax.plot([0, 1], [hg.loc[paired_subjs, "lzc"].mean(), bb.loc[paired_subjs, "lzc"].mean()],
            "s-", color="green", markersize=10, linewidth=2, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HG", "BB"])
    ax.set_ylabel("LZc (normalized)")
    ax.set_title("C: LZc by band")

    ax = axes[1, 1]
    for s in paired_subjs:
        ax.plot([0, 1], [hg.loc[s, "dfa"], bb.loc[s, "dfa"]], "o-", color="gray", alpha=0.4, markersize=4)
    ax.plot([0, 1], [hg.loc[paired_subjs, "dfa"].mean(), bb.loc[paired_subjs, "dfa"].mean()],
            "s-", color="purple", markersize=10, linewidth=2, zorder=5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["HG", "BB"])
    ax.set_ylabel("DFA alpha")
    ax.set_title("D: DFA by band")

    fig.suptitle("LME Diagnostic: Paired Band Comparisons", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "lme_paired_bands.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: lme_paired_bands.png")


def main():
    log("=" * 70)
    log("LINEAR MIXED-EFFECTS MODELING")
    log("=" * 70)

    hg_ok, bb_ok, common = load_paired_data()
    log(f"\n  Common subjects: {len(common)}")

    df = build_long_df(hg_ok, bb_ok, common)
    log(f"  Long-format rows: {len(df)}")
    log(f"  Columns: {list(df.columns)}")

    log("\n[1] Fitting LME models...")
    lme_results = fit_lme_models(df)

    log("\n[2] Generating diagnostic plots...")
    plot_diagnostics(df)

    log("\n[3] Residual QQ plots...")
    try:
        import statsmodels.formula.api as smf
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, (metric, title) in enumerate([("sigma", "Sigma"), ("lzc", "LZc"), ("dfa", "DFA")]):
            model = smf.mixedlm(f"{metric} ~ band", df, groups=df["subject"])
            fit = model.fit(reml=True)
            resid = fit.resid
            sp_stats.probplot(resid, plot=axes[i])
            axes[i].set_title(f"{title} residuals")
        fig.suptitle("LME Residual QQ Plots", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "lme_qq_residuals.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log("  Saved: lme_qq_residuals.png")
    except Exception as e:
        log(f"  QQ plots failed: {e}")

    out_path = CMCC_ROOT / "results" / "lme_results.json"
    with open(out_path, "w") as f:
        json.dump(lme_results, f, indent=2, default=str)
    log(f"\n  Results: {out_path}")

    log("\n" + "=" * 70)
    log("LME ANALYSIS COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
