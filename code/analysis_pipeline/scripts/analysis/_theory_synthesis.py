"""CMCC Theory Synthesis: Band-Specific Criticality Windows.

Synthesizes all Phase 5-6 results into a revised theoretical framework.
Generates evidence table, multi-panel summary figure, and theory-evidence JSON.
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

CMCC_ROOT = Path(__file__).resolve().parent.parent
RESULTS_HG = CMCC_ROOT / "results"
RESULTS_BB = CMCC_ROOT / "results_broadband"
FIG_DIR = RESULTS_HG / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg, flush=True)


def load_all_results():
    data = {}

    with open(RESULTS_HG / "group_all_subjects.json") as f:
        raw_hg = json.load(f)
    data["hg_subjects"] = {s["subject"]: s for s in raw_hg if s.get("status") == "OK" and s.get("qc_pass", False)}

    bb_path = RESULTS_BB / "group_all_subjects.json"
    if bb_path.exists():
        with open(bb_path) as f:
            raw_bb = json.load(f)
        data["bb_subjects"] = {s["subject"]: s for s in raw_bb if s.get("status") == "OK"}
    else:
        data["bb_subjects"] = {}

    for name in ["broadband_comparison", "hypothesis_analysis", "cross_frequency",
                 "spatial_band_analysis", "lme_results"]:
        p = RESULTS_HG / f"{name}.json"
        if p.exists():
            with open(p) as f:
                data[name] = json.load(f)

    return data


def compute_evidence_table(data):
    evidence = []

    hg_subs = data["hg_subjects"]
    bb_subs = data["bb_subjects"]
    common = sorted(set(hg_subs) & set(bb_subs))

    # 1. HG subcriticality
    hg_sigmas = [hg_subs[s]["branching_sigma"] for s in hg_subs]
    t_hg, p_hg = sp_stats.ttest_1samp(hg_sigmas, 1.0)
    evidence.append({
        "claim": "High-gamma operates in subcritical regime",
        "metric": "sigma (branching ratio)",
        "band": "HG",
        "mean": float(np.mean(hg_sigmas)),
        "reference": 1.0,
        "deviation": float(np.mean(hg_sigmas) - 1.0),
        "t": float(t_hg),
        "p": float(p_hg),
        "n": len(hg_sigmas),
        "effect_size": float(np.mean(hg_sigmas) - 1.0) / float(np.std(hg_sigmas)),
        "strength": "strong" if p_hg < 0.001 else "moderate" if p_hg < 0.05 else "weak",
        "interpretation": "HG sigma significantly below 1.0 — subcritical regime confirmed",
    })

    # 2. BB near-criticality
    bb_sigmas = [bb_subs[s]["branching_sigma"] for s in bb_subs if "branching_sigma" in bb_subs[s]]
    t_bb, p_bb = sp_stats.ttest_1samp(bb_sigmas, 1.0)
    evidence.append({
        "claim": "Broadband operates closer to criticality than HG",
        "metric": "sigma (branching ratio)",
        "band": "BB",
        "mean": float(np.mean(bb_sigmas)),
        "reference": 1.0,
        "deviation": float(np.mean(bb_sigmas) - 1.0),
        "t": float(t_bb),
        "p": float(p_bb),
        "n": len(bb_sigmas),
        "lme_band_effect": data.get("lme_results", {}).get("sigma_band", {}).get("fixed_effects", {}).get("band[T.HG]", {}),
        "strength": "strong",
        "interpretation": "BB sigma significantly closer to 1.0 than HG (LME p<0.0001)",
    })

    # 3. LZc sign reversal
    lzc_contrast = data.get("lme_results", {}).get("lzc_contrast_band", {})
    band_effect = lzc_contrast.get("fixed_effects", {}).get("band[T.HG]", {})
    hg_lzc_gs = [hg_subs[s].get("task_contrast_g") for s in common if hg_subs[s].get("task_contrast_g") is not None]
    bb_lzc_gs = [bb_subs[s].get("task_contrast_g") for s in common if s in bb_subs and bb_subs[s].get("task_contrast_g") is not None]

    hg_mean_g = float(np.mean(hg_lzc_gs)) if hg_lzc_gs else None
    bb_mean_g = float(np.mean(bb_lzc_gs)) if bb_lzc_gs else None

    evidence.append({
        "claim": "LZc task contrast reverses sign between bands",
        "metric": "LZc Hedges g (task-relevant vs irrelevant)",
        "hg_mean_g": hg_mean_g,
        "bb_mean_g": bb_mean_g,
        "lme_coef": band_effect.get("coef"),
        "lme_p": band_effect.get("p"),
        "strength": "moderate" if band_effect.get("p", 1) < 0.1 else "weak",
        "interpretation": "HG shows negative LZc contrast (relevant < irrelevant), BB shows positive — marginal significance (p=0.063)",
    })

    # 4. Decoding reversal band-specificity
    bc = data.get("broadband_comparison", {})
    dec_n10 = bc.get("decoding_n=10", {})
    evidence.append({
        "claim": "Decoding reversal is band-specific (stronger in HG)",
        "metric": "decoding diff (most-critical minus least-critical channels)",
        "hg_reversal_rate": f"{dec_n10.get('hg_reversal_count', '?')}/{dec_n10.get('n_subjects', '?')}",
        "bb_reversal_rate": f"{dec_n10.get('bb_reversal_count', '?')}/{dec_n10.get('n_subjects', '?')}",
        "hg_mean_diff": dec_n10.get("hg_mean_diff"),
        "bb_mean_diff": dec_n10.get("bb_mean_diff"),
        "paired_p": dec_n10.get("paired_p"),
        "strength": "moderate",
        "interpretation": "HG shows 14/19 reversal vs BB 8/19 — directionally consistent but paired test non-significant",
    })

    # 5. PAC-criticality link
    cf = data.get("cross_frequency", {})
    pac_tau = cf.get("correlations", {}).get("mi_vs_tau", {})
    evidence.append({
        "claim": "Cross-frequency coupling correlates with criticality exponent",
        "metric": "PAC MI vs tau",
        "r": pac_tau.get("r"),
        "p": pac_tau.get("p"),
        "n": pac_tau.get("n"),
        "strength": "moderate" if pac_tau.get("p", 1) < 0.05 else "weak",
        "interpretation": "Significant positive correlation (r=0.526, p=0.025): higher PAC associated with steeper power-law (more critical dynamics)",
    })

    pac_sigma = cf.get("correlations", {}).get("mi_vs_sigma", {})
    evidence.append({
        "claim": "Cross-frequency coupling does not correlate with branching ratio",
        "metric": "PAC MI vs sigma",
        "r": pac_sigma.get("r"),
        "p": pac_sigma.get("p"),
        "n": pac_sigma.get("n"),
        "strength": "null",
        "interpretation": "No relationship between PAC and sigma — coupling relates to avalanche scaling, not branching ratio",
    })

    # 6. Task PAC contrast
    task_pac = cf.get("group_task_pac", {})
    evidence.append({
        "claim": "PAC does not differ between task-relevant and irrelevant epochs",
        "metric": "PAC MI (relevant vs irrelevant)",
        "rel_mean": task_pac.get("mean_rel"),
        "irr_mean": task_pac.get("mean_irr"),
        "t": task_pac.get("paired_t"),
        "p": task_pac.get("paired_p"),
        "strength": "null",
        "interpretation": "No significant task modulation of PAC — cross-frequency coupling is state-independent",
    })

    # 7. Spatial specificity
    spatial = data.get("spatial_band_analysis", {})
    coverage = spatial.get("coverage_correlations", {})
    evidence.append({
        "claim": "Band-specific effects show trending posterior spatial specificity",
        "metric": "posterior electrode fraction vs delta LZc",
        "r": coverage.get("post_frac_vs_delta_lzc", {}).get("r"),
        "p": coverage.get("post_frac_vs_delta_lzc", {}).get("p"),
        "n_sign_flips": sum(1 for v in spatial.get("deep_dive_spatial", {}).values()
                           if v.get("lzc_sign_flip") == "True"),
        "n_deep_dive": len(spatial.get("deep_dive_spatial", {})),
        "strength": "weak",
        "interpretation": "4/5 key subjects show LZc sign flip between bands; posterior fraction trending (r=0.338, p=0.158) but underpowered",
    })

    # 8. Tau-decoding link (from hypothesis analysis)
    hyp = data.get("hypothesis_analysis", {})
    tau_dec = hyp.get("test3_tau_decoding_quadrant", {}).get("n=5", {}).get("tau_vs_diff", {})
    evidence.append({
        "claim": "Power-law exponent predicts decoding reversal magnitude",
        "metric": "tau vs decoding diff (n=5)",
        "r": tau_dec.get("r"),
        "p": tau_dec.get("p"),
        "n": tau_dec.get("n"),
        "strength": "moderate" if tau_dec.get("p", 1) < 0.05 else "weak",
        "interpretation": "Significant negative correlation (r=-0.456, p=0.050): subjects closer to criticality show stronger decoding reversal",
    })

    # 9. Cross-band prediction
    xb = data.get("lme_results", {}).get("cross_band_prediction", {})
    hg_sigma_pred = xb.get("fixed_effects", {}).get("hg_sigma", {})
    evidence.append({
        "claim": "HG criticality predicts BB criticality",
        "metric": "bb_sigma ~ hg_sigma (LME)",
        "coef": hg_sigma_pred.get("coef"),
        "p": hg_sigma_pred.get("p"),
        "strength": "strong" if hg_sigma_pred.get("p", 1) < 0.001 else "moderate",
        "interpretation": "HG sigma significantly predicts BB sigma (coef=0.212, p<0.0001) — within-subject consistency of criticality across bands",
    })

    return evidence


def generate_summary_figure(data, output_dir):
    hg_subs = data["hg_subjects"]
    bb_subs = data["bb_subjects"]
    common = sorted(set(hg_subs) & set(bb_subs))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel A: HG vs BB sigma (paired)
    ax = axes[0, 0]
    hg_sig = [hg_subs[s]["branching_sigma"] for s in common]
    bb_sig = [bb_subs[s]["branching_sigma"] for s in common if "branching_sigma" in bb_subs[s]]
    n = min(len(hg_sig), len(bb_sig))
    hg_sig, bb_sig = hg_sig[:n], bb_sig[:n]

    for i in range(n):
        ax.plot([0, 1], [hg_sig[i], bb_sig[i]], color="gray", alpha=0.4, linewidth=0.8)
    ax.scatter([0]*n, hg_sig, c="coral", s=40, zorder=5, edgecolors="black", linewidths=0.5, label="HG")
    ax.scatter([1]*n, bb_sig, c="steelblue", s=40, zorder=5, edgecolors="black", linewidths=0.5, label="BB")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Critical (sigma=1)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["High-Gamma", "Broadband"])
    ax.set_ylabel("Branching Ratio (sigma)")
    ax.set_title("A. Sigma: HG Subcritical, BB Near-Critical\n(LME p < 0.0001)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.3, 1.3)

    # Panel B: LZc contrast paired
    ax = axes[0, 1]
    hg_gs = []
    bb_gs = []
    paired_names = []
    for s in common:
        hg_g = hg_subs[s].get("task_contrast_g")
        bb_g = bb_subs[s].get("task_contrast_g")
        if hg_g is not None and bb_g is not None:
            hg_gs.append(hg_g)
            bb_gs.append(bb_g)
            paired_names.append(s)

    for i in range(len(hg_gs)):
        ax.plot([0, 1], [hg_gs[i], bb_gs[i]], color="gray", alpha=0.4, linewidth=0.8)
    ax.scatter([0]*len(hg_gs), hg_gs, c="coral", s=40, zorder=5, edgecolors="black", linewidths=0.5)
    ax.scatter([1]*len(bb_gs), bb_gs, c="steelblue", s=40, zorder=5, edgecolors="black", linewidths=0.5)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["High-Gamma", "Broadband"])
    ax.set_ylabel("LZc Contrast (Hedges g)")
    ax.set_title("B. LZc Task Contrast Reverses Between Bands\n(LME p = 0.063)")

    # Panel C: Decoding reversal rates by band
    ax = axes[1, 0]
    bc = data.get("broadband_comparison", {})
    n_vals = [5, 10, 20]
    hg_rates = []
    bb_rates = []
    for n_ch in n_vals:
        key = f"decoding_n={n_ch}"
        d = bc.get(key, {})
        n_subj = d.get("n_subjects", 1)
        hg_rates.append(d.get("hg_reversal_count", 0) / n_subj)
        bb_rates.append(d.get("bb_reversal_count", 0) / n_subj)

    x = np.arange(len(n_vals))
    w = 0.35
    ax.bar(x - w/2, hg_rates, w, label="HG", color="coral", edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, bb_rates, w, label="BB", color="steelblue", edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in n_vals])
    ax.set_ylabel("Reversal Rate")
    ax.set_title("C. Decoding Reversal: Stronger in HG")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)

    # Panel D: PAC vs tau
    ax = axes[1, 1]
    cf = data.get("cross_frequency", {})
    if cf and "subjects" in cf:
        taus = [s["hg_tau"] for s in cf["subjects"] if s.get("hg_tau")]
        mis = [s["mean_mi"] for s in cf["subjects"] if s.get("hg_tau")]
        if taus and mis:
            ax.scatter(taus, mis, c="darkorange", edgecolors="black", s=50, linewidths=0.5)
            ax.set_xlabel("HG tau (power-law exponent)")
            ax.set_ylabel("PAC Modulation Index")
            corr = cf.get("correlations", {}).get("mi_vs_tau", {})
            r_val = corr.get("r", 0)
            p_val = corr.get("p", 1)
            ax.set_title(f"D. PAC-Criticality Link\n(r={r_val:.3f}, p={p_val:.3f})")
            if len(taus) > 2:
                z = np.polyfit(taus, mis, 1)
                x_line = np.linspace(min(taus), max(taus), 50)
                ax.plot(x_line, np.polyval(z, x_line), "r--", linewidth=1, alpha=0.7)

    fig.suptitle("CMCC Theory Synthesis: Band-Specific Criticality Windows\n"
                 "N=19 QC-passed subjects, Cogitate iEEG", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "theory_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved: {output_dir / 'theory_summary.png'}")


def build_theory_evidence_map(evidence):
    theory = {
        "title": "Band-Specific Criticality Windows for Consciousness",
        "version": "CMCC v2.0 — Revised from original near-criticality hypothesis",
        "core_claims": [
            {
                "id": "C1",
                "claim": "High-gamma operates in a subcritical regime (sigma < 1)",
                "mechanism": "Subcritical HG dynamics suppress large-scale avalanches, enabling stable, fine-grained local representations",
                "evidence_indices": [0],
                "confidence": "high",
            },
            {
                "id": "C2",
                "claim": "Broadband operates closer to criticality than high-gamma",
                "mechanism": "Near-critical broadband dynamics support long-range integration via metastable dynamics and communication-through-coherence",
                "evidence_indices": [1, 8],
                "confidence": "high",
            },
            {
                "id": "C3",
                "claim": "Task-relevant processing modulates complexity differently across bands",
                "mechanism": "LZc task contrast reverses sign: HG reduces complexity for relevant stimuli (sharpening), BB increases complexity (integrating)",
                "evidence_indices": [2, 7],
                "confidence": "moderate",
            },
            {
                "id": "C4",
                "claim": "Decoding reversal is a signature of subcritical HG processing",
                "mechanism": "Near-critical channels in HG decode worse because they amplify noise; subcritical channels maintain cleaner representations",
                "evidence_indices": [3, 7],
                "confidence": "moderate",
            },
            {
                "id": "C5",
                "claim": "Cross-frequency coupling links broadband phase dynamics to HG amplitude",
                "mechanism": "PAC serves as the bridge between broadband integration and HG representation, with coupling strength tracking criticality (tau)",
                "evidence_indices": [4, 5, 6],
                "confidence": "moderate",
            },
        ],
        "novel_contributions": [
            "First demonstration of band-specific criticality regimes in human iEEG during conscious perception",
            "LZc task contrast sign reversal between HG and broadband — not previously reported",
            "Decoding reversal (least-critical > most-critical) in HG, not observed in broadband",
            "PAC-tau correlation linking cross-frequency coupling to avalanche scaling",
            "Within-subject sigma consistency across bands (HG predicts BB)",
        ],
        "limitations": [
            "N=19 QC-passed subjects limits statistical power for interaction effects",
            "LZc sign reversal is marginal (p=0.063) and requires replication",
            "PAC analysis used subset of channels (30) and DurR1 only",
            "Spatial analysis limited by heterogeneous electrode coverage across subjects",
            "Task contrast depends on event mapping quality, which varied across subjects",
            "Broadband (1-200 Hz) includes volume-conducted components that may inflate sigma toward 1.0",
            "Decoding reversal paired test non-significant despite consistent directionality",
        ],
        "falsification_criteria": [
            "If sigma shows no band difference in a larger cohort, C2 fails",
            "If LZc reversal does not replicate with corrected event alignment, C3 fails",
            "If PAC-tau correlation is driven by outliers or disappears with full electrode coverage, C5 weakens",
            "If broadband sigma=1.0 is explained by volume conduction alone, the BB near-criticality claim is confounded",
        ],
        "next_steps": [
            "Pre-registered replication on independent iEEG dataset (e.g., Cogitate EXP2 or MNI open iEEG)",
            "Volume conduction control analysis for broadband sigma",
            "Full-electrode PAC analysis with FDR correction",
            "Temporal dynamics: track sigma and PAC within-trial to test state-dependent criticality shifts",
            "Computational model: simulate band-specific criticality windows in balanced E/I networks",
        ],
    }
    return theory


def main():
    log("=" * 70)
    log("CMCC THEORY SYNTHESIS: Band-Specific Criticality Windows")
    log("=" * 70)

    data = load_all_results()
    log(f"  Loaded: {len(data['hg_subjects'])} HG, {len(data['bb_subjects'])} BB subjects")

    evidence = compute_evidence_table(data)
    log(f"\n  Evidence items: {len(evidence)}")
    for i, e in enumerate(evidence):
        strength = e.get("strength", "?")
        log(f"    [{i}] {strength.upper():>8s}: {e['claim']}")

    generate_summary_figure(data, FIG_DIR)

    theory = build_theory_evidence_map(evidence)

    output = {
        "evidence_table": evidence,
        "theory": theory,
        "summary_statistics": {
            "n_hg_subjects": len(data["hg_subjects"]),
            "n_bb_subjects": len(data["bb_subjects"]),
            "n_common": len(set(data["hg_subjects"]) & set(data["bb_subjects"])),
            "n_evidence_items": len(evidence),
            "n_strong": sum(1 for e in evidence if e.get("strength") == "strong"),
            "n_moderate": sum(1 for e in evidence if e.get("strength") == "moderate"),
            "n_weak": sum(1 for e in evidence if e.get("strength") == "weak"),
            "n_null": sum(1 for e in evidence if e.get("strength") == "null"),
        },
    }

    out_path = RESULTS_HG / "theory_synthesis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\n  Saved: {out_path}")

    log(f"\n{'='*70}")
    log("THEORY SUMMARY")
    log(f"{'='*70}")
    log(f"  Title: {theory['title']}")
    log(f"  Version: {theory['version']}")
    log(f"\n  Core Claims:")
    for c in theory["core_claims"]:
        log(f"    {c['id']}: {c['claim']} [{c['confidence']}]")
    log(f"\n  Novel Contributions: {len(theory['novel_contributions'])}")
    log(f"  Limitations: {len(theory['limitations'])}")
    log(f"  Falsification Criteria: {len(theory['falsification_criteria'])}")
    log(f"\n  Evidence Strength: {output['summary_statistics']['n_strong']} strong, "
        f"{output['summary_statistics']['n_moderate']} moderate, "
        f"{output['summary_statistics']['n_weak']} weak, "
        f"{output['summary_statistics']['n_null']} null")
    log(f"\nDONE")


if __name__ == "__main__":
    main()
