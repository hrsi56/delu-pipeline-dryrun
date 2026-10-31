"""Bounded descriptive spectral utilities for the three CP-1 figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

TARGET_PERIODS_HOURS = (24, 168, 12)


def welch_periodogram(values: np.ndarray | pd.Series, nperseg: int = 512) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    if nperseg < 336 or nperseg & (nperseg - 1):
        raise ValueError("nperseg must be a power of two >= 336")
    if len(array) < nperseg:
        raise ValueError("series is shorter than nperseg")
    detrended = signal.detrend(array, type="linear")
    return signal.welch(
        detrended,
        fs=1.0,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        scaling="density",
    )


def annotate_peaks(frequencies: np.ndarray, power: np.ndarray) -> pd.DataFrame:
    frequencies = np.asarray(frequencies, dtype="float64")
    power = np.asarray(power, dtype="float64")
    if len(frequencies) != len(power) or len(frequencies) < 2:
        raise ValueError("frequency and power arrays must have equal non-trivial length")
    fft_bin = float(np.median(np.diff(frequencies)))
    rows = []
    for period in TARGET_PERIODS_HOURS:
        target = 1.0 / period
        position = int(np.argmin(np.abs(frequencies - target)))
        rows.append(
            {
                "label": f"{period}h",
                "period_hours": period,
                "target_frequency_hz_per_hour": target,
                "bin_frequency_hz_per_hour": float(frequencies[position]),
                "bin_error": float(abs(frequencies[position] - target)),
                "fft_bin_width": fft_bin,
                "power": float(power[position]),
            }
        )
    result = pd.DataFrame(rows)
    if (result["bin_error"] > result["fft_bin_width"] + 1e-15).any():
        raise AssertionError("a requested spectral label is more than one FFT bin away")
    return result


def autocorrelation(values: np.ndarray | pd.Series, max_lag: int = 336) -> np.ndarray:
    array = np.asarray(values, dtype="float64")
    array = array[np.isfinite(array)]
    array = signal.detrend(array, type="linear")
    array -= array.mean()
    correlations = signal.correlate(array, array, mode="full", method="fft")[len(array) - 1 : len(array) + max_lag]
    correlations /= np.arange(len(array), len(array) - max_lag - 1, -1)
    return correlations / correlations[0]
