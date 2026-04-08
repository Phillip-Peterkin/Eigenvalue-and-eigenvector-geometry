"""Phase-amplitude coupling (PAC) using Tort et al. (2010) modulation index.

Computes coupling between low-frequency phase and high-gamma amplitude
to test cross-frequency interactions in criticality dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


@dataclass
class PACResult:
    modulation_index: float
    mean_amplitude_per_bin: np.ndarray
    phase_bins: np.ndarray
    surrogate_mi_mean: float
    surrogate_mi_std: float
    z_score: float
    p_value: float


def _bandpass(data: np.ndarray, lo: float, hi: float, sfreq: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    lo_n = max(lo / nyq, 0.001)
    hi_n = min(hi / nyq, 0.999)
    if lo_n >= hi_n:
        return data
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, data, axis=-1)


def _modulation_index(phase: np.ndarray, amplitude: np.ndarray, n_bins: int = 18) -> tuple[float, np.ndarray]:
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mean_amp = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if mask.any():
            mean_amp[i] = np.mean(amplitude[mask])

    total = mean_amp.sum()
    if total == 0:
        return 0.0, mean_amp

    p = mean_amp / total
    p = p[p > 0]
    h = -np.sum(p * np.log(p))
    h_max = np.log(n_bins)
    mi = (h_max - h) / h_max

    return float(mi), mean_amp


def compute_pac(
    signal: np.ndarray,
    sfreq: float,
    phase_band: tuple[float, float] = (4, 30),
    amp_band: tuple[float, float] = (70, 150),
    n_bins: int = 18,
    n_surrogates: int = 200,
    seed: int = 42,
) -> PACResult:
    """Compute phase-amplitude coupling for a single channel.

    Parameters
    ----------
    signal : np.ndarray, shape (n_samples,)
    sfreq : float
    phase_band : tuple
        Frequency range for phase extraction.
    amp_band : tuple
        Frequency range for amplitude extraction.
    n_bins : int
        Number of phase bins.
    n_surrogates : int
        Number of time-shifted surrogates for significance.
    seed : int

    Returns
    -------
    PACResult
    """
    nyq = sfreq / 2.0
    if amp_band[1] >= nyq:
        amp_band = (amp_band[0], nyq - 1)
    if phase_band[1] >= nyq:
        phase_band = (phase_band[0], min(phase_band[1], nyq - 1))

    phase_sig = _bandpass(signal, phase_band[0], phase_band[1], sfreq)
    amp_sig = _bandpass(signal, amp_band[0], amp_band[1], sfreq)

    phase = np.angle(hilbert(phase_sig))
    amplitude = np.abs(hilbert(amp_sig))

    mi, mean_amp = _modulation_index(phase, amplitude, n_bins)

    rng = np.random.default_rng(seed)
    surrogate_mis = np.zeros(n_surrogates)
    n_samples = len(signal)
    min_shift = int(sfreq)
    max_shift = n_samples - min_shift

    if max_shift <= min_shift:
        max_shift = n_samples

    for i in range(n_surrogates):
        shift = rng.integers(min_shift, max_shift)
        amp_shifted = np.roll(amplitude, shift)
        surrogate_mis[i], _ = _modulation_index(phase, amp_shifted, n_bins)

    surr_mean = float(np.mean(surrogate_mis))
    surr_std = float(np.std(surrogate_mis))
    z = (mi - surr_mean) / surr_std if surr_std > 0 else 0.0
    p_val = float(np.mean(surrogate_mis >= mi))

    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    return PACResult(
        modulation_index=mi,
        mean_amplitude_per_bin=mean_amp,
        phase_bins=bin_centers,
        surrogate_mi_mean=surr_mean,
        surrogate_mi_std=surr_std,
        z_score=z,
        p_value=p_val,
    )


def compute_pac_per_channel(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    phase_band: tuple[float, float] = (4, 30),
    amp_band: tuple[float, float] = (70, 150),
    n_surrogates: int = 50,
    seed: int = 42,
) -> dict[str, PACResult]:
    """Compute PAC for each channel.

    Parameters
    ----------
    data : np.ndarray, shape (n_channels, n_samples)
    sfreq : float
    ch_names : list[str]
    phase_band, amp_band : tuple
    n_surrogates : int
    seed : int

    Returns
    -------
    dict[str, PACResult]
    """
    results = {}
    for i, ch in enumerate(ch_names):
        results[ch] = compute_pac(
            data[i], sfreq,
            phase_band=phase_band,
            amp_band=amp_band,
            n_surrogates=n_surrogates,
            seed=seed + i,
        )
    return results
