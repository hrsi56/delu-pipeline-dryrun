"""CP-10 calibration only; v1 model, scores and release remain untouched.

ACI state is COVERAGE, complementing the paper's miscoverage convention. Its
update is exactly v20 §4.3. No clipping or interpolation of the state/rank.
"""
from __future__ import annotations

from datetime import date, timedelta
from fractions import Fraction

import numpy as np
import pandas as pd

from .conformal import PAIR_ALPHAS, conformity_scores, cqr_threshold, order_statistic_rank
from .ingest import BERLIN
from .postprocess import QUANTILE_LABELS, apply_cqr_thresholds, isotonic_last

PAIR_POSITIONS = ((0, 8), (1, 7), (2, 6), (3, 5))
SELECTION_FOLDS = ('fold_1', 'fold_2', 'fold_4', 'fold_5')
GAMMA_GRID = ('0.000001', '0.000005', '0.00001', '0.00002')
CANDIDATES = ('c1_head_spread', 'c1_price_volatility') + tuple(
    f'c2_aci_gamma_{gamma}' for gamma in GAMMA_GRID
)


def _matrix(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(QUANTILE_LABELS) or not np.isfinite(values).all():
        raise ValueError('raw heads must be a finite n x 9 matrix')
    return values


def _scale(values: np.ndarray, n: int) -> np.ndarray:
    scale = np.asarray(values, dtype=float)
    if scale.shape != (n,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('scale must have one finite positive value per row')
    return scale


def head_spread(raw: np.ndarray) -> np.ndarray:
    values = _matrix(raw)
    return np.maximum(values[:, 7] - values[:, 1], 1.0)


def price_volatility(price: pd.Series, days: pd.Index) -> np.ndarray:
    """168 exact UTC observations ending before D local midnight; sample std.

    Uses the inherited D-1 day-ahead PRICE boundary. C-2 outcome feedback has
    the separately ratified D-2 boundary. Missing hours fail, never extend the
    lookback or silently fall back. Only requested days are calculated.
    """
    index = pd.DatetimeIndex(price.index)
    if index.tz is None or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError('prices need unique increasing timezone-aware timestamps')
    result = {}
    for day in pd.Index(days).unique():
        boundary = pd.Timestamp(day, tz=BERLIN).tz_convert('UTC')
        hours = pd.date_range(end=boundary - pd.Timedelta(hours=1), periods=168, freq='h')
        window = price.reindex(hours).to_numpy(dtype=float)
        if not np.isfinite(window).all():
            raise ValueError(f'incomplete trailing price window for {day}')
        result[day] = max(float(np.std(window, ddof=1)), 1.0)
    return np.array([result[day] for day in days])


def fully_observed_days(price: pd.Series) -> set[date]:
    """All canonical prices exist for a local day, including 23/25-hour DST days."""
    result = set()
    for day in pd.Index(price.index.tz_convert(BERLIN).date).unique():
        hours = pd.date_range(pd.Timestamp(day, tz=BERLIN),
                              pd.Timestamp(day + timedelta(days=1), tz=BERLIN),
                              freq='h', inclusive='left').tz_convert('UTC')
        if np.isfinite(price.reindex(hours).to_numpy(dtype=float)).all():
            result.add(day)
    return result


def fit_scaled(raw: np.ndarray, truth: np.ndarray, scale: np.ndarray) -> dict:
    raw = _matrix(raw)
    scale = _scale(scale, len(raw))
    truth = np.asarray(truth, dtype=float)
    if truth.shape != (len(raw),) or not np.isfinite(truth).all():
        raise ValueError('calibration truth must be complete')
    return {
        pair: cqr_threshold(conformity_scores(raw[:, lo], raw[:, hi], truth) / scale, alpha)
        for ((pair, alpha), (lo, hi)) in zip(PAIR_ALPHAS, PAIR_POSITIONS, strict=True)
    }


def predict_scaled(raw: np.ndarray, thresholds: dict, scale: np.ndarray) -> np.ndarray:
    raw = _matrix(raw)
    scale = _scale(scale, len(raw))
    return isotonic_last(apply_cqr_thresholds(raw, {pair: value * scale for pair, value in thresholds.items()}))


def aci_predict(cal_raw: np.ndarray, cal_y: np.ndarray, raw: np.ndarray,
                days: pd.Index, feedback_y: np.ndarray, *, gamma: str,
                observed_days: set[date]) -> tuple[np.ndarray, pd.DataFrame]:
    """Emit all hours of each D with one state; release each D-2 label once.

    A fixed calibration score reservoir isolates the effect of the state
    adaptation. The forecast loop only indexes feedback_y after the day gate.
    feedback_y may be masked from D-1 onward when reproducing the origin for D.
    Feedback refers to the final, isotonic intervals actually emitted.
    """
    cal_raw, raw = _matrix(cal_raw), _matrix(raw)
    cal_y = np.asarray(cal_y, dtype=float)
    y = np.asarray(feedback_y, dtype=float)
    days = pd.Index(days)
    step = Fraction(gamma)
    if step <= 0 or y.shape != (len(raw),) or len(days) != len(raw):
        raise ValueError('positive gamma and aligned rows required')
    if not days.is_monotonic_increasing or days.hasnans:
        raise ValueError('delivery days must be complete and chronological')
    if cal_y.shape != (len(cal_raw),) or not np.isfinite(cal_y).all():
        raise ValueError('calibration truth must be complete')
    scores = [np.sort(conformity_scores(cal_raw[:, lo], cal_raw[:, hi], cal_y))
              for lo, hi in PAIR_POSITIONS]
    nominal = [1 - alpha for _, alpha in PAIR_ALPHAS]
    state = nominal.copy()
    result = np.empty_like(raw)
    groups = [(day, np.flatnonzero(days == day)) for day in days.unique()]
    released = 0
    n_feedback = 0
    trace = []
    for day, rows in groups:
        cutoff = day - timedelta(days=2)
        while released < len(groups) and groups[released][0] <= cutoff:
            feedback_day, indices = groups[released]
            if feedback_day in observed_days:
                if not np.isfinite(y[indices]).all():
                    raise ValueError(f'missing released feedback on {feedback_day}')
                for i in indices:
                    for j, (lo, hi) in enumerate(PAIR_POSITIONS):
                        hit = int(result[i, lo] <= y[i] <= result[i, hi])
                        state[j] += step * (nominal[j] - hit)
                n_feedback += len(indices)
            released += 1
        ranks = [order_statistic_rank(len(cal_raw), 1 - level) for level in state]
        thresholds = {pair: float(score[rank - 1]) for (pair, _), score, rank
                      in zip(PAIR_ALPHAS, scores, ranks, strict=True)}
        result[rows] = isotonic_last(apply_cqr_thresholds(raw[rows], thresholds))
        trace.append({'delivery_date': day, 'feedback_cutoff': cutoff,
                      'n_feedback': n_feedback,
                      **{f'coverage_state_{name}': float(level) for name, level in zip(('95','90','80','50'), state)},
                      **{f'rank_{name}': rank for name, rank in zip(('95','90','80','50'), ranks)}})
    return result, pd.DataFrame(trace)


def selection_scores(metrics: pd.DataFrame) -> dict[str, float]:
    """Exclude diagnostic rows before any arithmetic; require every frozen arm/fold."""
    selected = metrics.loc[metrics['fold'].isin(SELECTION_FOLDS)]
    scores = {}
    for candidate in CANDIDATES:
        rows = selected.loc[selected['candidate'].eq(candidate)]
        if len(rows) != 4 or set(rows['fold']) != set(SELECTION_FOLDS):
            raise ValueError(f'incomplete or duplicated selection folds: {candidate}')
        n = rows['n_obs'].to_numpy(dtype=float)
        losses = rows['mean_pinball'].to_numpy(dtype=float)
        if np.any(n <= 0) or not np.isfinite(n).all() or not np.isfinite(losses).all():
            raise ValueError('selection requires positive counts and finite losses')
        scores[candidate] = float(np.dot(n, losses) / n.sum())
    return scores


def select_candidate(metrics: pd.DataFrame) -> tuple[str, dict[str, float]]:
    scores = selection_scores(metrics)
    return min(CANDIDATES, key=scores.__getitem__), scores


def falsification(coverage: float) -> dict:
    if not np.isfinite(coverage) or not 0 <= coverage <= 1:
        raise ValueError('coverage must be a probability')
    worked = coverage >= 0.394
    return {'coverage_95': coverage, 'baseline_literal': 0.194,
            'required_absolute_improvement': 0.20, 'threshold': 0.394,
            'absolute_improvement_over_literal': coverage - 0.194,
            'fix_worked': worked,
            'result': 'fix cleared the pre-registered falsification threshold' if worked else 'the fix did not work',
            'recommended_frozen_artifact': 'v1' if not worked else 'no new artifact frozen by CP-10'}
