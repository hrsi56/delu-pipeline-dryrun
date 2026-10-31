"""Pinball / MAE / coverage metrics, Diebold-Mariano tests, and day-block bootstrap.

Every metric here is location-equivariant: the DE-LU price is routinely negative
(573 hours in 2025) and the plan forbids a log/Box-Cox target transform (§5.2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .postprocess import QUANTILE_LABELS

QUANTILES: tuple[float, ...] = (0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975)
MEDIAN_POSITION: int = QUANTILES.index(0.50)
COVERAGE_LEVELS: tuple[tuple[str, int, int], ...] = (("50", 3, 5), ("80", 2, 6), ("95", 0, 8))


def _as_matrix(quantiles: np.ndarray) -> np.ndarray:
    matrix = np.asarray(quantiles, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[1] != len(QUANTILES):
        raise ValueError(f"expected an n x {len(QUANTILES)} quantile matrix")
    return matrix


def pinball_matrix(y_true: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """Row/quantile matrix of rho_tau(y - q) = max(tau*u, (tau-1)*u)."""
    matrix = _as_matrix(quantiles)
    residual = np.asarray(y_true, dtype="float64").reshape(-1, 1) - matrix
    taus = np.asarray(QUANTILES, dtype="float64").reshape(1, -1)
    return np.maximum(taus * residual, (taus - 1.0) * residual)


def pooled_mean_pinball(y_true: np.ndarray, quantiles: np.ndarray) -> float:
    """Observation-weighted pooled mean pinball loss: sum(rho) / (9 * n_rows) (§4.1)."""
    losses = pinball_matrix(y_true, quantiles)
    return float(losses.sum() / losses.size)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64"))))


def point_forecast_quantiles(y_pred: np.ndarray) -> np.ndarray:
    """A point forecast as a degenerate predictive distribution: all nine heads equal.

    Its mean pinball loss is therefore exactly 0.5 * MAE, because the nine tau
    values are symmetric about 0.5. Stated rather than left implicit, so the
    baseline pinball numbers are readable.
    """
    column = np.asarray(y_pred, dtype="float64").reshape(-1, 1)
    return np.repeat(column, len(QUANTILES), axis=1)


def interval_coverage(y_true: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    """Empirical coverage of the 50/80/95 nominal intervals."""
    matrix = _as_matrix(quantiles)
    truth = np.asarray(y_true, dtype="float64")
    result: dict[str, float] = {}
    for name, low, high in COVERAGE_LEVELS:
        inside = (truth >= matrix[:, low]) & (truth <= matrix[:, high])
        result[name] = float(np.mean(inside))
    return result


def crossing_violations(quantiles: np.ndarray) -> int:
    """Count (row, position) pairs where the nine quantiles decrease.

    The one retained hard gate (§12) is that this returns 0 after the full
    CQR-then-isotonic pipeline. No tolerance: isotonic emits exact block means,
    so a violation is a real ordering failure rather than float noise.
    """
    matrix = _as_matrix(quantiles)
    return int(np.count_nonzero(np.diff(matrix, axis=1) < 0))


def daily_pinball_series(y_true: np.ndarray, quantiles: np.ndarray, delivery_dates: pd.Index) -> pd.Series:
    """L(t) = (1/H_t) * sum_h (1/9) * sum_tau rho_tau(t,h), per §7.1(b).

    H_t is the actual local delivery-hour count (23, 24 or 25); grouping by
    delivery date gets that mechanically from the UTC index.
    """
    per_row = pinball_matrix(y_true, quantiles).mean(axis=1)
    return pd.Series(per_row, index=pd.Index(delivery_dates, name="delivery_date")).groupby(level=0).mean()


def daily_absolute_error_series(y_true: np.ndarray, y_pred: np.ndarray, delivery_dates: pd.Index) -> pd.Series:
    per_row = np.abs(np.asarray(y_true, dtype="float64") - np.asarray(y_pred, dtype="float64"))
    return pd.Series(per_row, index=pd.Index(delivery_dates, name="delivery_date")).groupby(level=0).mean()


def newey_west_lrv(values: np.ndarray, lag: int) -> float:
    """Bartlett-kernel long-run variance with truncation `lag`."""
    series = np.asarray(values, dtype="float64")
    n = series.size
    if lag < 0 or lag >= n:
        raise ValueError("lag truncation must satisfy 0 <= lag < n")
    centered = series - series.mean()
    lrv = float(np.dot(centered, centered) / n)
    for k in range(1, lag + 1):
        gamma = float(np.dot(centered[k:], centered[:-k]) / n)
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * gamma
    return lrv


@dataclass(frozen=True)
class DMResult:
    comparator: str
    analysis: str
    evidence_class: str
    n_days: int
    lag_truncation: int
    mean_loss_differential: float
    long_run_variance: float
    statistic: float
    p_value: float
    relative_improvement_pct: float
    standardized_effect_size: float
    model_better_at_5pct: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def diebold_mariano(
    model_daily: pd.Series,
    naive_daily: pd.Series,
    *,
    comparator: str,
    analysis: str,
    evidence_class: str,
) -> DMResult:
    """One-sided DM on the daily loss differential; reject (model better) if DM < -1.645.

    LRV via Newey-West with lag truncation floor(N^(1/3)), exactly as §7.1 pins it.
    """
    aligned = pd.concat({"model": model_daily, "naive": naive_daily}, axis=1).dropna()
    differential = (aligned["model"] - aligned["naive"]).to_numpy(dtype="float64")
    n = differential.size
    if n < 3:
        raise ValueError("DM needs at least three days")
    lag = int(np.floor(n ** (1.0 / 3.0)))
    lrv = newey_west_lrv(differential, lag)
    if lrv <= 0:
        raise ValueError(
            "non-positive long-run variance: the daily loss differential is degenerate, "
            "so no DM statistic is defined"
        )
    statistic = float(differential.mean() / np.sqrt(lrv / n))
    naive_mean = float(aligned["naive"].mean())
    spread = float(differential.std(ddof=1))
    return DMResult(
        comparator=comparator,
        analysis=analysis,
        evidence_class=evidence_class,
        n_days=n,
        lag_truncation=lag,
        mean_loss_differential=float(differential.mean()),
        long_run_variance=lrv,
        statistic=statistic,
        p_value=float(stats.norm.cdf(statistic)),
        relative_improvement_pct=float(-100.0 * differential.mean() / naive_mean) if naive_mean else float("nan"),
        standardized_effect_size=float(differential.mean() / spread) if spread else float("nan"),
        model_better_at_5pct=bool(statistic < -1.645),
    )


def bootstrap_day_ci(
    values: np.ndarray,
    delivery_dates: pd.Index,
    *,
    reps: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for a mean, resampling whole delivery days with replacement.

    Days, not hours: intra-day errors are strongly dependent, and an hour-level
    bootstrap would report an interval several times too narrow.
    """
    frame = pd.DataFrame({"value": np.asarray(values, dtype="float64"), "day": pd.Index(delivery_dates)})
    groups = [group.to_numpy() for _, group in frame.groupby("day", sort=True)["value"]]
    if len(groups) < 2:
        return (float("nan"), float("nan"))
    sums = np.array([group.sum() for group in groups], dtype="float64")
    counts = np.array([group.size for group in groups], dtype="float64")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(reps, len(groups)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def summarize(y_true: np.ndarray, quantiles: np.ndarray) -> dict[str, float]:
    matrix = _as_matrix(quantiles)
    result: dict[str, float] = {
        "mae": mae(y_true, matrix[:, MEDIAN_POSITION]),
        "mean_pinball": pooled_mean_pinball(y_true, matrix),
        "n_obs": float(matrix.shape[0]),
    }
    for name, value in interval_coverage(y_true, matrix).items():
        result[f"coverage_{name}"] = value
    return result


__all__ = [
    "COVERAGE_LEVELS",
    "DMResult",
    "MEDIAN_POSITION",
    "QUANTILES",
    "QUANTILE_LABELS",
    "bootstrap_day_ci",
    "crossing_violations",
    "daily_absolute_error_series",
    "daily_pinball_series",
    "diebold_mariano",
    "interval_coverage",
    "mae",
    "newey_west_lrv",
    "pinball_matrix",
    "point_forecast_quantiles",
    "pooled_mean_pinball",
    "summarize",
]
