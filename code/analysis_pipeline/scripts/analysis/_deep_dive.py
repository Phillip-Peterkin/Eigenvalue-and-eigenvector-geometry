"""Step 12: Individual-difference deep dive for 5 key subjects.

Target subjects:
- CF124: Significant LZc contrast (g=-0.145, p=0.048)
- CF126: Significant LZc contrast (g=-0.154, p=0.030)
- CG103: Double-significant (LZc g=-0.247, p=0.002 AND HG g=+0.306, p=0.002)
- CF104: Dissociation (HG g=+0.279, p=0.002 but LZc null)
- CE110: Near-critical (sigma=0.998, closest to criticality)
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

from cmcc.config import load_config
from cmcc.preprocess.filter import SITE_LINE_FREQ

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEEP_DIVE_DIR = RESULTS_DIR / "deep_dive"
RUNS = ["DurR1", "DurR2", "DurR3", "DurR4", "DurR5"]

TARGETS = {
    "CF124": "sig_lzc_contrast",
    "CF126": "sig_lzc_contrast",
    "CG103": "double_significant",
    "CF104": "dissociation_hg_vs_lzc",
    "CE110": "near_critical_sigma",
}


def log(msg):
    print(msg, flush=True)


def load_electrodes(data_root, subject_id):
    meta_dir = Path(data_root) / f"{subject_id}_ECOG_1" / "METADATA" / "electrode_coordinates"
    tsv = meta_dir / f"sub-{subject_id}_ses-1_space-fsaverage_electrodes.tsv"
    if tsv.exists():
        return pd.read_csv(tsv, sep="\t")
    return None


def load_atlas_labels(data_root, subject_id, atlas="desikan"):
    meta_dir = Path(data_root) / f"{subject_id}_ECOG_1" / "METADATA" / "electrode_coordinates"
    tsv = meta_dir / f"sub-{subject_id}_ses-1_atlas-{atlas}_labels.tsv"
    if tsv.exists():
        return pd.read_csv(tsv, sep="\t")
    return None


def compute_per_channel_tau(subject_id, config):
    from cmcc.io.loader import load_subject
    from cmcc.preprocess.qc import detect_bad_channels, mark_bad_channels
    from cmcc.preprocess.filter import remove_line_noise, extract_high_gamma
    from cmcc.preprocess.reference import apply_laplace
    from cmcc.features.avalanche import detect_avalanches
    from cmcc.features.powerlaw_fit import _fit_single_distribution

    t0 = time.time()
    site = subject_id[:2].upper()
    line_freq = SITE_LINE_FREQ.get(site, 60.0)

    subject_data = load_subject(subject_id, config["data"]["root"], runs=RUNS, preload_edf=True)

    per_run_raw = {}
    per_run_good_channels = {}
    for run_id in RUNS:
        if run_id not in subject_data.raw:
            continue
        raw = subject_data.raw[run_id]
        non_ecog = [ch for ch in raw.ch_names if ch.startswith("DC") or ch.startswith("C12") or ch.startswith("EKG") or ch.startswith("EMG") or ch == "C128"]
        if non_ecog:
            raw.drop_channels(non_ecog)
        bad_channels = detect_bad_channels(raw)
        mark_bad_channels(raw, bad_channels)
        good_chs = [ch for ch in raw.ch_names if ch not in bad_channels]
        if len(good_chs) < 5:
            continue
        raw.pick(good_chs)
        raw = remove_line_noise(raw, line_freq=line_freq)
        if subject_data.laplace_map:
            raw = apply_laplace(raw, subject_data.laplace_map)
        per_run_good_channels[run_id] = set(raw.ch_names)
        per_run_raw[run_id] = raw

    valid_runs = list(per_run_raw.keys())
    common_channels = sorted(set.intersection(*[per_run_good_channels[r] for r in valid_runs]))

    for run_id in valid_runs:
        raw = per_run_raw[run_id]
        ch_to_drop = [ch for ch in raw.ch_names if ch not in common_channels]
        if ch_to_drop:
            raw.drop_channels(ch_to_drop)
        ch_order = [ch for ch in common_channels if ch in raw.ch_names]
        raw.reorder_channels(ch_order)
        per_run_raw[run_id] = raw

    all_gamma_continuous = []
    for run_id in valid_runs:
        raw = per_run_raw[run_id]
        gamma_raw = extract_high_gamma(raw, passband=tuple(config["preprocessing"]["high_gamma_passband"]))
        all_gamma_continuous.append(gamma_raw.get_data())
        del gamma_raw

    gamma_concatenated = np.concatenate(all_gamma_continuous, axis=1)
    sfreq = per_run_raw[valid_runs[0]].info["sfreq"]

    del per_run_raw, subject_data, all_gamma_continuous
    gc.collect()

    tau_per_channel = {}
    n_avals_per_channel = {}
    for ci, ch_name in enumerate(common_channels):
        ch_avals = detect_avalanches(
            gamma_concatenated[ci:ci+1, :], sfreq=sfreq,
            threshold_sd=config["avalanche"]["threshold_sd"],
            bin_width_factor=config["avalanche"]["bin_width_factor"]
        )
        ch_sizes = np.array([a.size for a in ch_avals])
        n_avals_per_channel[ch_name] = len(ch_avals)
        if len(ch_sizes) >= 10:
            tau_per_channel[ch_name] = _fit_single_distribution(ch_sizes, discrete=True)["exponent"]
        else:
            tau_per_channel[ch_name] = float("nan")
        if (ci + 1) % 20 == 0:
            log(f"    tau {ci+1}/{len(common_channels)} [{time.time()-t0:.0f}s]")

    del gamma_concatenated
    gc.collect()

    log(f"    Per-channel tau done: {sum(1 for v in tau_per_channel.values() if not np.isnan(v))}/{len(common_channels)} valid [{time.time()-t0:.0f}s]")
    return tau_per_channel, n_avals_per_channel, common_channels


def plot_electrode_tau_map(electrodes, tau_values, subject_id, reason, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    valid_taus = {k: v for k, v in tau_values.items() if not np.isnan(v)}
    if not valid_taus:
        log(f"    No valid per-channel tau for {subject_id}")
        plt.close(fig)
        return

    x_coords, y_coords, tau_vals, crit_vals = [], [], [], []
    for _, row in electrodes.iterrows():
        name = str(row["name"])
        if name in valid_taus:
            x_coords.append(float(row["x"]))
            y_coords.append(float(row["y"]))
            tau_vals.append(valid_taus[name])
            crit_vals.append(abs(valid_taus[name] - 1.5))

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    tau_vals = np.array(tau_vals)
    crit_vals = np.array(crit_vals)

    sc1 = axes[0].scatter(x_coords, y_coords, c=tau_vals, cmap="RdYlBu_r",
                          s=60, edgecolors="black", linewidths=0.5, vmin=1.0, vmax=3.5)
    fig.colorbar(sc1, ax=axes[0], label="tau")
    axes[0].set_title(f"{subject_id} - Per-channel tau")
    axes[0].set_xlabel("x (fsaverage)")
    axes[0].set_ylabel("y (fsaverage)")
    axes[0].set_aspect("equal")

    sc2 = axes[1].scatter(x_coords, y_coords, c=crit_vals, cmap="viridis_r",
                          s=60, edgecolors="black", linewidths=0.5)
    fig.colorbar(sc2, ax=axes[1], label="|tau - 1.5|")
    axes[1].set_title(f"{subject_id} - Criticality distance")
    axes[1].set_xlabel("x (fsaverage)")
    axes[1].set_ylabel("y (fsaverage)")
    axes[1].set_aspect("equal")

    ranked = sorted(valid_taus.keys(), key=lambda ch: abs(valid_taus[ch] - 1.5))
    n_select = min(10, len(ranked))
    most_critical = set(ranked[:n_select])
    least_critical = set(ranked[-n_select:])

    colors = []
    for _, row in electrodes.iterrows():
        name = str(row["name"])
        if name in most_critical:
            colors.append("blue")
        elif name in least_critical:
            colors.append("red")
        elif name in valid_taus:
            colors.append("lightgray")
        else:
            continue

    x_all, y_all = [], []
    c_all = []
    for _, row in electrodes.iterrows():
        name = str(row["name"])
        if name in valid_taus:
            x_all.append(float(row["x"]))
            y_all.append(float(row["y"]))
            if name in most_critical:
                c_all.append("blue")
            elif name in least_critical:
                c_all.append("red")
            else:
                c_all.append("lightgray")

    axes[2].scatter(x_all, y_all, c=c_all, s=60, edgecolors="black", linewidths=0.5)
    axes[2].scatter([], [], c="blue", s=60, edgecolors="black", linewidths=0.5, label=f"Most critical (n={n_select})")
    axes[2].scatter([], [], c="red", s=60, edgecolors="black", linewidths=0.5, label=f"Least critical (n={n_select})")
    axes[2].legend(loc="lower left", fontsize=8)
    axes[2].set_title(f"{subject_id} - Decoding channel selection")
    axes[2].set_xlabel("x (fsaverage)")
    axes[2].set_ylabel("y (fsaverage)")
    axes[2].set_aspect("equal")

    fig.suptitle(f"{subject_id} [{reason}]", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{subject_id}_electrode_maps.png", dpi=150)
    plt.close(fig)
    log(f"    Electrode maps saved")


def plot_behavioral_profile(summary, subject_id, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cats = summary.get("categories", {})
    if cats:
        axes[0].bar(cats.keys(), cats.values(), color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        axes[0].set_title("Category distribution")
        axes[0].set_ylabel("N trials")

    n_rel = summary.get("task_contrast_n_relevant", 0)
    n_irr = summary.get("task_contrast_n_irrelevant", 0)
    axes[1].bar(["Relevant", "Irrelevant"], [n_rel, n_irr], color=["#1f77b4", "#ff7f0e"])
    axes[1].set_title("Task relevance")
    axes[1].set_ylabel("N trials")

    metrics = {
        "tau": summary.get("tau"),
        "sigma": summary.get("branching_sigma"),
        "LZc": summary.get("lzc_normalized"),
        "DFA": summary.get("dfa_alpha"),
    }
    refs = {"tau": 1.5, "sigma": 1.0, "LZc": None, "DFA": 1.0}
    x_pos = np.arange(len(metrics))
    vals = list(metrics.values())
    axes[2].bar(x_pos, vals, color="#1f77b4")
    for i, (k, ref) in enumerate(refs.items()):
        if ref is not None:
            axes[2].plot([i - 0.4, i + 0.4], [ref, ref], "r--", linewidth=1.5)
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(list(metrics.keys()))
    axes[2].set_title("Criticality metrics")

    fig.suptitle(f"{subject_id} - Behavioral & metric profile", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{subject_id}_profile.png", dpi=150)
    plt.close(fig)


def main():
    config = load_config(str(CONFIG_PATH))
    DEEP_DIVE_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("DEEP DIVE: 5 Key Subjects")
    log("=" * 70)

    all_results = {}

    for subject_id, reason in TARGETS.items():
        log(f"\n--- {subject_id} ({reason}) ---")

        summary_path = RESULTS_DIR / f"summary_{subject_id}_pooled.json"
        if not summary_path.exists():
            log(f"  No summary found, skipping")
            continue
        with open(summary_path) as f:
            summary = json.load(f)

        electrodes = load_electrodes(config["data"]["root"], subject_id)
        atlas = load_atlas_labels(config["data"]["root"], subject_id, "desikan")

        if electrodes is None:
            log(f"  No electrode coordinates found, skipping spatial maps")
            continue
        log(f"  Electrodes: {len(electrodes)} contacts")
        if atlas is not None:
            log(f"  Atlas labels: {len(atlas)} entries")

        log(f"  Computing per-channel tau...")
        tau_per_ch, n_avals_per_ch, ch_names = compute_per_channel_tau(subject_id, config)

        plot_electrode_tau_map(electrodes, tau_per_ch, subject_id, reason, DEEP_DIVE_DIR)
        plot_behavioral_profile(summary, subject_id, DEEP_DIVE_DIR)

        valid_taus = {k: v for k, v in tau_per_ch.items() if not np.isnan(v)}
        ranked = sorted(valid_taus.keys(), key=lambda ch: abs(valid_taus[ch] - 1.5))

        posterior_chs = []
        anterior_chs = []
        for _, row in electrodes.iterrows():
            name = str(row["name"])
            if name in valid_taus:
                try:
                    y = float(row["y"])
                    if y < 0:
                        posterior_chs.append(name)
                    else:
                        anterior_chs.append(name)
                except (ValueError, TypeError):
                    pass

        post_taus = [valid_taus[ch] for ch in posterior_chs if ch in valid_taus]
        ant_taus = [valid_taus[ch] for ch in anterior_chs if ch in valid_taus]

        subject_result = {
            "subject": subject_id,
            "reason": reason,
            "n_channels": len(ch_names),
            "n_valid_tau": len(valid_taus),
            "tau_global": summary.get("tau"),
            "sigma": summary.get("branching_sigma"),
            "lzc": summary.get("lzc_normalized"),
            "dfa": summary.get("dfa_alpha"),
            "hg_power_g": summary.get("hg_power_contrast_g"),
            "hg_power_p": summary.get("hg_power_contrast_p"),
            "lzc_contrast_g": summary.get("task_contrast_g"),
            "lzc_contrast_p": summary.get("task_contrast_p"),
            "n_posterior": len(posterior_chs),
            "n_anterior": len(anterior_chs),
            "tau_posterior_mean": float(np.mean(post_taus)) if post_taus else None,
            "tau_anterior_mean": float(np.mean(ant_taus)) if ant_taus else None,
            "most_critical_10": ranked[:10] if len(ranked) >= 10 else ranked,
            "least_critical_10": ranked[-10:] if len(ranked) >= 10 else [],
            "tau_per_channel": {k: round(v, 4) for k, v in valid_taus.items()},
            "n_avalanches_per_channel": n_avals_per_ch,
            "decoding": summary.get("decoding", []),
            "cat_contrasts": summary.get("cat_contrasts", {}),
        }

        if atlas is not None:
            region_taus = {}
            for _, row in atlas.iterrows():
                ch_name = str(row.iloc[0])
                region = str(row.iloc[1]) if len(row) > 1 else "unknown"
                if ch_name in valid_taus:
                    region_taus.setdefault(region, []).append(valid_taus[ch_name])
            subject_result["region_taus"] = {r: {"mean": round(np.mean(v), 4), "n": len(v)} for r, v in region_taus.items() if len(v) >= 2}

        all_results[subject_id] = subject_result
        log(f"  Posterior: {len(posterior_chs)} ch, mean tau={np.mean(post_taus):.3f}" if post_taus else "  Posterior: 0 ch")
        log(f"  Anterior: {len(anterior_chs)} ch, mean tau={np.mean(ant_taus):.3f}" if ant_taus else "  Anterior: 0 ch")

        gc.collect()

    with open(DEEP_DIVE_DIR / "deep_dive_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSummary saved: {DEEP_DIVE_DIR / 'deep_dive_summary.json'}")

    log("\n" + "=" * 70)
    log("DEEP DIVE COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()
