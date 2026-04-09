"""EP Advanced Analyses — ds004752 Cross-Cohort Generalization.

Runs the same four mechanistic tests of non-Hermitian operator geometry
on the Zurich SEEG dataset that were run on the Cogitate iEEG cohort:

1. State-Switch Contrast — EP metrics in high-load vs low-load windows
2. Spectral Radius Sensitivity — rho vs eigenvalue gap correlation
3. SVD Dimension — effective rank collapse near EPs
4. Petermann Noise — K factor predicting high-gamma power bursts

Uses session-01 high-gamma data per subject (first session only,
analogous to DurR1-only analysis in the Cogitate pipeline).

Scientific rationale
--------------------
If the EP dynamics observed in Cogitate iEEG (visual consciousness,
ECoG grids, CE/CF/CG sites) are a general property of cortical
networks, they should also appear in SEEG depth recordings from a
different lab, paradigm (verbal working memory), and patient
population (primarily hippocampal/temporal epilepsy).
"""
from __future__ import annotations

import gc
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
warnings.filterwarnings("ignore")

from cmcc.config import load_config
from cmcc.analysis.dynamical_systems import compute_ep_proximity_timecourse
from cmcc.analysis.contrasts import fdr_correction
from cmcc.analysis.ep_advanced import (
    compute_spectral_radius_sensitivity,
    compute_svd_dimension,
    compute_petermann_noise,
)

CMCC_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = CMCC_ROOT / "configs" / "ds004752.yaml"
RESULTS_DIR = CMCC_ROOT / "results_ds004752"
FIG_DIR = RESULTS_DIR / "figures" / "ep_advanced"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHANNELS = 30
WINDOW_SEC = 0.5
STEP_SEC = 0.1


def log(msg):
    print(msg, flush=True)


def load_and_preprocess(subject_id, config):
    from cmcc.io.loader_bids_ieeg import (
        load_bids_ieeg_subject,
        get_neural_channels,
    )
    from cmcc.preprocess.qc import detect_bad_channels, mark_bad_channels
    from cmcc.preprocess.filter import remove_line_noise, extract_high_gamma

    line_freq = config["preprocessing"]["line_freq"]

    subject_data = load_bids_ieeg_subject(
        subject_id, config["data"]["root"],
        sessions=["ses-01"], preload=True,
    )
    if "ses-01" not in subject_data.raw:
        return None, None, None, None

    raw = subject_data.raw["ses-01"]
    n_samples_raw = raw.n_times

    ch_info = subject_data.channels_info.get("ses-01")
    if ch_info is not None:
        neural_chs = get_neural_channels(ch_info)
        non_neural = [ch for ch in raw.ch_names if ch not in neural_chs]
    else:
        non_neural = []
    if non_neural:
        raw.drop_channels(non_neural)

    bad = detect_bad_channels(raw)
    mark_bad_channels(raw, bad)
    good = [ch for ch in raw.ch_names if ch not in bad]
    if len(good) < 5:
        return None, None, None, None
    raw.pick(good)

    raw = remove_line_noise(raw, line_freq=line_freq)

    passband = tuple(config["preprocessing"]["high_gamma_passband"])
    gamma_raw = extract_high_gamma(raw, passband=passband)

    events_df = subject_data.events.get("ses-01", None)

    return gamma_raw.get_data(), gamma_raw.info["sfreq"], n_samples_raw, events_df


def parse_load_intervals(events_df, sfreq):
    """Extract high-load (SetSize=8) and low-load (SetSize=4) intervals."""
    if events_df is None or events_df.empty:
        return None, None
    if "SetSize" not in events_df.columns or "begSample" not in events_df.columns:
        return None, None

    high_intervals = []
    low_intervals = []
    encoding_duration_sec = 2.0
    dur_samples = int(encoding_duration_sec * sfreq)

    for _, row in events_df.iterrows():
        onset = int(row["begSample"]) - 1
        interval = (onset, onset + dur_samples)
        if row["SetSize"] == 8:
            high_intervals.append(interval)
        elif row["SetSize"] == 4:
            low_intervals.append(interval)

    return high_intervals, low_intervals


def compute_hg_power_per_window(data, window_centers, window_samples):
    n_ch, n_total = data.shape
    half_w = window_samples // 2
    power = np.zeros(len(window_centers))
    for i, c in enumerate(window_centers):
        c = int(c)
        start = max(0, c - half_w)
        end = min(n_total, c + half_w)
        if end > start:
            power[i] = np.mean(data[:, start:end] ** 2)
    return power


def analyze_single_subject(subject_id, config):
    t0 = time.time()
    log(f"\n  {subject_id}...")

    data, sfreq, n_samples_raw, events_df = load_and_preprocess(subject_id, config)
    if data is None:
        log(f"    SKIP: no data")
        return None

    n_ch, n_samples = data.shape
    log(f"    {n_ch} ch, {sfreq} Hz, {n_samples/sfreq:.1f}s")

    ep_tc = compute_ep_proximity_timecourse(
        data, sfreq=sfreq,
        window_sec=WINDOW_SEC, step_sec=STEP_SEC,
        max_channels=MAX_CHANNELS, seed=42,
    )

    jac = ep_tc["jac_result"]
    ep = ep_tc["ep_result"]
    n_windows = len(jac.window_centers)
    window_samples = max(int(WINDOW_SEC * sfreq), ep_tc["n_channels_used"] + 10)

    summary = {
        "subject": subject_id,
        "dataset": "ds004752",
        "n_channels_used": ep_tc["n_channels_used"],
        "sfreq": sfreq,
        "n_windows": n_windows,
    }

    high_intervals, low_intervals = parse_load_intervals(events_df, sfreq)
    if high_intervals and low_intervals:
        from cmcc.analysis.ep_advanced import compute_state_contrast
        sc = compute_state_contrast(
            jac, ep, high_intervals, low_intervals, n_perm=500, seed=42,
        )
        summary["state_contrast"] = {
            "n_high_load_windows": sc.n_relevant_windows,
            "n_low_load_windows": sc.n_irrelevant_windows,
            "n_eff_high": sc.n_eff_relevant,
            "n_eff_low": sc.n_eff_irrelevant,
            "fdr_significant": sc.fdr_significant,
            "circular_shift_p": sc.circular_shift_p,
            "gap": {
                "mean_high": sc.gap_contrast.mean_a,
                "mean_low": sc.gap_contrast.mean_b,
                "g": sc.gap_contrast.effect_size,
                "p_nominal": sc.gap_contrast.p_value,
                "p_circular_shift": sc.circular_shift_p.get("gap", float("nan")),
                "fdr_significant": sc.fdr_significant.get("gap", False),
            },
            "condition_number": {
                "mean_high": sc.condition_number_contrast.mean_a,
                "mean_low": sc.condition_number_contrast.mean_b,
                "g": sc.condition_number_contrast.effect_size,
                "p_nominal": sc.condition_number_contrast.p_value,
                "p_circular_shift": sc.circular_shift_p.get("condition_number", float("nan")),
                "fdr_significant": sc.fdr_significant.get("condition_number", False),
            },
            "ep_score": {
                "mean_high": sc.ep_score_contrast.mean_a,
                "mean_low": sc.ep_score_contrast.mean_b,
                "g": sc.ep_score_contrast.effect_size,
                "p_nominal": sc.ep_score_contrast.p_value,
                "p_circular_shift": sc.circular_shift_p.get("ep_score", float("nan")),
                "fdr_significant": sc.fdr_significant.get("ep_score", False),
            },
            "spectral_radius": {
                "mean_high": sc.spectral_radius_contrast.mean_a,
                "mean_low": sc.spectral_radius_contrast.mean_b,
                "g": sc.spectral_radius_contrast.effect_size,
                "p_nominal": sc.spectral_radius_contrast.p_value,
                "p_circular_shift": sc.circular_shift_p.get("spectral_radius", float("nan")),
                "fdr_significant": sc.fdr_significant.get("spectral_radius", False),
            },
        }
        log(f"    state_contrast: high={sc.n_relevant_windows} low={sc.n_irrelevant_windows} "
            f"gap_g={sc.gap_contrast.effect_size:.3f} ep_g={sc.ep_score_contrast.effect_size:.3f}")
    else:
        summary["state_contrast"] = None
        log(f"    state_contrast: SKIP (no events)")

    srs = compute_spectral_radius_sensitivity(jac, ep)
    summary["spectral_sensitivity"] = srs
    log(f"    spectral_sensitivity: r={srs['r']:.3f} p_adj={srs['p_adjusted']:.4f} n_eff={srs['n_eff']}")

    svd = compute_svd_dimension(jac, ep, run_null=True)
    summary["svd_dimension"] = {
        "mean_pr": svd.mean_pr,
        "mean_erank": svd.mean_erank,
        "pr_vs_ep": svd.pr_vs_ep_score,
        "erank_vs_ep": svd.erank_vs_ep_score,
        "null_r_mean": svd.null_r_mean,
        "null_r_std": svd.null_r_std,
        "null_p": svd.null_p,
    }
    log(f"    svd_dimension: mean_erank={svd.mean_erank:.2f} "
        f"erank_vs_ep r={svd.erank_vs_ep_score['r']:.3f} null_p={svd.null_p:.3f}")

    hg_power = compute_hg_power_per_window(data, jac.window_centers, window_samples)
    pet = compute_petermann_noise(
        ep, hg_power, step_sec=STEP_SEC, window_sec=WINDOW_SEC,
    )
    summary["petermann_noise"] = {
        "r": pet.correlation_r,
        "p_nominal": pet.correlation_p,
        "p_adjusted": pet.p_adjusted,
        "n_eff": pet.n_eff,
        "peak_lag": pet.peak_lag,
        "peak_lag_sec": pet.peak_lag_seconds,
        "granger_f": pet.granger_f,
        "granger_p": pet.granger_p,
        "granger_stride": pet.granger_stride,
        "n_valid_windows": pet.n_valid_windows,
    }
    log(f"    petermann_noise: r={pet.correlation_r:.3f} p_adj={pet.p_adjusted:.4f} "
        f"peak_lag={pet.peak_lag}")

    summary["elapsed_s"] = time.time() - t0

    del data, ep_tc
    gc.collect()

    return summary


def compute_group_statistics(all_results):
    valid = [s for s in all_results if s is not None]
    if len(valid) < 3:
        return {}

    group = {}
    all_group_p = []
    all_group_p_labels = []

    sc_valid = [s for s in valid if s.get("state_contrast") is not None]
    if len(sc_valid) >= 3:
        sc_stats = {}
        for metric in ["gap", "condition_number", "ep_score", "spectral_radius"]:
            gs = [s["state_contrast"][metric]["g"] for s in sc_valid
                  if s["state_contrast"][metric]["g"] is not None
                  and np.isfinite(s["state_contrast"][metric]["g"])]
            if len(gs) >= 3:
                t_val, p_val = sp_stats.ttest_1samp(gs, 0.0)
                sc_stats[metric] = {
                    "mean_g": float(np.mean(gs)),
                    "std_g": float(np.std(gs)),
                    "t": float(t_val),
                    "p": float(p_val),
                    "n": len(gs),
                }
                all_group_p.append(p_val)
                all_group_p_labels.append(f"state_contrast_{metric}")
        group["state_contrast"] = sc_stats

    rs = [s["spectral_sensitivity"]["r"] for s in valid
          if np.isfinite(s["spectral_sensitivity"]["r"])]
    if len(rs) >= 3:
        t_val, p_val = sp_stats.ttest_1samp(rs, 0.0)
        group["spectral_sensitivity"] = {
            "mean_r": float(np.mean(rs)),
            "std_r": float(np.std(rs)),
            "t": float(t_val),
            "p": float(p_val),
            "n": len(rs),
        }
        all_group_p.append(p_val)
        all_group_p_labels.append("spectral_sensitivity")

    erank_rs = [s["svd_dimension"]["erank_vs_ep"]["r"] for s in valid
                if np.isfinite(s["svd_dimension"]["erank_vs_ep"]["r"])]
    if len(erank_rs) >= 3:
        t_val, p_val = sp_stats.ttest_1samp(erank_rs, 0.0)
        group["svd_dimension"] = {
            "mean_erank_vs_ep_r": float(np.mean(erank_rs)),
            "t": float(t_val),
            "p": float(p_val),
            "n": len(erank_rs),
        }
        all_group_p.append(p_val)
        all_group_p_labels.append("svd_dimension")

    pet_valid = [s for s in valid if s.get("petermann_noise") is not None
                 and np.isfinite(s["petermann_noise"]["peak_lag"])]
    if len(pet_valid) >= 3:
        lags = [s["petermann_noise"]["peak_lag"] for s in pet_valid]
        rs_pet = [s["petermann_noise"]["r"] for s in pet_valid
                  if np.isfinite(s["petermann_noise"]["r"])]
        t_lag, p_lag = sp_stats.ttest_1samp(lags, 0.0)
        group["petermann_noise"] = {
            "mean_peak_lag": float(np.mean(lags)),
            "std_peak_lag": float(np.std(lags)),
            "t_lag": float(t_lag),
            "p_lag": float(p_lag),
            "mean_r": float(np.mean(rs_pet)) if rs_pet else float("nan"),
            "n": len(lags),
        }
        all_group_p.append(p_lag)
        all_group_p_labels.append("petermann_noise_lag")

    if len(all_group_p) >= 2:
        fdr_sig = fdr_correction(all_group_p, alpha=0.05)
        group["fdr_correction"] = {
            label: {"p": float(p), "fdr_significant": bool(sig)}
            for label, p, sig in zip(all_group_p_labels, all_group_p, fdr_sig)
        }

    return group


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


def main():
    log("=" * 70)
    log("ADVANCED EP ANALYSES — ds004752 (Zurich SEEG) Cross-Cohort Generalization")
    log("=" * 70)

    config = load_config(str(CONFIG_PATH))
    subjects = config["data"]["subjects"]
    log(f"\nSubjects: {len(subjects)}")

    all_results = []
    for subj in subjects:
        try:
            result = analyze_single_subject(subj, config)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            log(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
        gc.collect()

    log(f"\n{'='*70}")
    log(f"RESULTS: {len(all_results)} subjects analyzed")

    group_stats = compute_group_statistics(all_results)

    if group_stats:
        log(f"\n  Group statistics:")
        for key, val in group_stats.items():
            log(f"    {key}: {val}")

    out = {
        "dataset": "ds004752",
        "n_subjects": len(all_results),
        "parameters": {
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "max_channels": MAX_CHANNELS,
        },
        "subjects": all_results,
        "group_statistics": group_stats,
    }

    out_path = RESULTS_DIR / "analysis" / "ep_advanced_ds004752.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=default_ser)
    log(f"\n  Results: {out_path}")
    log(f"\n{'='*70}")
    log("DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()
