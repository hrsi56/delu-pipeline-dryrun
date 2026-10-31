"""CQR threshold estimation (§6.2) with an exact one-based order statistic.

The rank is `k = ceil((n_cal+1)(1-alpha))`, a ONE-BASED rank, so a zero-based
implementation reads `scores[k-1]`. Alphas are carried as `Fraction`, not float:
`21 * 0.95` evaluates to 19.949999999999999 in binary floating point, so a float
`ceil` at alpha=0.05 sits directly on the boundary the plan spends a paragraph
forbidding. Exact rational arithmetic removes that failure mode rather than
testing around it.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .postprocess import QUANTILE_LABELS, SYMMETRIC_PAIRS, apply_cqr_thresholds, cqr_then_isotonic, isotonic_last

Pair = tuple[str, str]

#: Symmetric interval pairs and their miscoverage levels, in plan order (§6.2).
PAIR_ALPHAS: tuple[tuple[Pair, Fraction], ...] = (
    (SYMMETRIC_PAIRS[0], Fraction(1, 20)),
    (SYMMETRIC_PAIRS[1], Fraction(1, 10)),
    (SYMMETRIC_PAIRS[2], Fraction(1, 5)),
    (SYMMETRIC_PAIRS[3], Fraction(1, 2)),
)


class UndersizedCalibrationSet(ValueError):
    """Raised when `k = ceil((n_cal+1)(1-alpha))` falls outside `1..n_cal`.

    Clipping to the largest observed score would silently substitute a finite
    threshold for the augmented `+inf` order statistic and overstate the
    coverage guarantee, so this fails loudly instead.
    """


def conformity_scores(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """E_i = max{ q_lo(x_i) - y_i , y_i - q_hi(x_i) } (§6.2 step 2)."""
    lo = np.asarray(lower, dtype="float64")
    hi = np.asarray(upper, dtype="float64")
    truth = np.asarray(y_true, dtype="float64")
    if not (lo.shape == hi.shape == truth.shape):
        raise ValueError("lower, upper and y must share a shape")
    if np.isnan(lo).any() or np.isnan(hi).any() or np.isnan(truth).any():
        raise ValueError("conformity scores require complete calibration rows")
    return np.maximum(lo - truth, truth - hi)


def order_statistic_rank(n_cal: int, alpha: Fraction | float) -> int:
    """The one-based rank k = ceil((n_cal+1)(1-alpha)); validated against 1..n_cal."""
    if n_cal < 1:
        raise UndersizedCalibrationSet("calibration set is empty")
    exact = Fraction(alpha) if isinstance(alpha, Fraction) else Fraction(alpha).limit_denominator(10**6)
    k = math.ceil((n_cal + 1) * (1 - exact))
    if not 1 <= k <= n_cal:
        raise UndersizedCalibrationSet(
            f"n_cal={n_cal} is too small for alpha={float(exact)}: "
            f"k={k} lies outside 1..{n_cal}; the conformal construction would need "
            "the augmented +inf order statistic, and clipping would overstate coverage"
        )
    return k


def cqr_threshold(scores: np.ndarray, alpha: Fraction | float) -> float:
    """The k-th smallest conformity score, read one-based (`sorted[k-1]`)."""
    ascending = np.sort(np.asarray(scores, dtype="float64"))
    k = order_statistic_rank(ascending.size, alpha)
    return float(ascending[k - 1])


def fit_cqr_thresholds(calibration_quantiles: np.ndarray, y_calibration: np.ndarray) -> dict[Pair, float]:
    """Estimate the four thresholds from raw calibration-set head predictions."""
    matrix = np.asarray(calibration_quantiles, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[1] != len(QUANTILE_LABELS):
        raise ValueError(f"expected an n x {len(QUANTILE_LABELS)} raw quantile matrix")
    position = {label: index for index, label in enumerate(QUANTILE_LABELS)}
    thresholds: dict[Pair, float] = {}
    for pair, alpha in PAIR_ALPHAS:
        scores = conformity_scores(matrix[:, position[pair[0]]], matrix[:, position[pair[1]]], y_calibration)
        thresholds[pair] = cqr_threshold(scores, alpha)
    return thresholds


def thresholds_to_json(thresholds: dict[Pair, float]) -> dict[str, float]:
    return {f"{pair[0]}_{pair[1]}": value for pair, value in thresholds.items()}


def thresholds_from_json(payload: dict[str, float]) -> dict[Pair, float]:
    return {pair: float(payload[f"{pair[0]}_{pair[1]}"]) for pair, _ in PAIR_ALPHAS}


__all__ = [
    "PAIR_ALPHAS",
    "UndersizedCalibrationSet",
    "apply_cqr_thresholds",
    "conformity_scores",
    "cqr_then_isotonic",
    "cqr_threshold",
    "fit_cqr_thresholds",
    "isotonic_last",
    "order_statistic_rank",
    "thresholds_from_json",
    "thresholds_to_json",
]
