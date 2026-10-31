"""Minimal CQR-shift and isotonic-last primitives needed by CP-1 invariants."""

from __future__ import annotations

import numpy as np

QUANTILE_LABELS = ("p025", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p975")
SYMMETRIC_PAIRS = (("p025", "p975"), ("p05", "p95"), ("p10", "p90"), ("p25", "p75"))


def apply_cqr_thresholds(raw: np.ndarray, thresholds: dict[tuple[str, str], float]) -> np.ndarray:
    values = np.asarray(raw, dtype="float64").copy()
    if values.ndim != 2 or values.shape[1] != len(QUANTILE_LABELS):
        raise ValueError("expected n x 9 quantile matrix")
    positions = {label: index for index, label in enumerate(QUANTILE_LABELS)}
    for pair in SYMMETRIC_PAIRS:
        threshold = thresholds[pair]
        values[:, positions[pair[0]]] -= threshold
        values[:, positions[pair[1]]] += threshold
    return values


def _pava(values: np.ndarray) -> np.ndarray:
    levels: list[float] = []
    weights: list[int] = []
    for value in values:
        levels.append(float(value))
        weights.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / weight
            levels[-2:] = [level]
            weights[-2:] = [weight]
    return np.repeat(np.asarray(levels), np.asarray(weights))


def isotonic_last(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype="float64")
    if matrix.ndim != 2:
        raise ValueError("expected a two-dimensional matrix")
    return np.vstack([_pava(row) for row in matrix])


def cqr_then_isotonic(raw: np.ndarray, thresholds: dict[tuple[str, str], float]) -> np.ndarray:
    return isotonic_last(apply_cqr_thresholds(raw, thresholds))
