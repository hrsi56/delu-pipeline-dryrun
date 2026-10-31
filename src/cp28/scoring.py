"""Daily losses on CP-25's frozen eligible keys (protocol §8).

Each actual hour's p50 MAE is |y − q50|. For α = .50, .20, .05 the interval score is
`IS_α = (u−l) + (2/α)(l−y)·1[y<l] + (2/α)(y−u)·1[y>u]`, and the seven-quantile WIS is
`[0.5·|y−q50| + Σ_α (α/2)·IS_α] / 3.5`. Hours average within a date, dates equally.
"""
from __future__ import annotations

import numpy as np

ALPHAS = ((0.50, 2, 4), (0.20, 1, 5), (0.05, 0, 6))   # α, lower and upper quantile column


def hourly(q, y):
    q, y = np.asarray(q, float), np.asarray(y, float)
    if q.ndim != 2 or q.shape[1] != 7 or q.shape[0] != y.shape[0]:
        raise ValueError('scoring needs (hours, 7) quantiles and one outcome per hour')
    mae = np.abs(y - q[:, 3])
    total = 0.5 * mae
    for alpha, lo, hi in ALPHAS:
        lower, upper = q[:, lo], q[:, hi]
        score = (upper - lower) + (2 / alpha) * (lower - y) * (y < lower) + (2 / alpha) * (y - upper) * (y > upper)
        total = total + (alpha / 2) * score
    return mae, total / 3.5


def daily(q, y):
    """(MAE, WIS) for one date: means over its scored hours."""
    mae, wis = hourly(q, y)
    if not len(mae) or not np.isfinite(mae).all() or not np.isfinite(wis).all():
        return np.nan, np.nan
    return float(mae.mean()), float(wis.mean())


def coverage(q, y):
    """Descriptive calibration for one date: 50/80/95% coverage and widths."""
    q, y = np.asarray(q, float), np.asarray(y, float)
    out = {}
    for name, lo, hi in (('50', 2, 4), ('80', 1, 5), ('95', 0, 6)):
        out[f'coverage{name}'] = float(np.mean((y >= q[:, lo]) & (y <= q[:, hi])))
        out[f'below{name}'] = float(np.mean(y < q[:, lo]))
        out[f'above{name}'] = float(np.mean(y > q[:, hi]))
        out[f'width{name}'] = float(np.mean(q[:, hi] - q[:, lo]))
    return out
