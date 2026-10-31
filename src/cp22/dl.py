"""The dynamic interval layer DL (capstone v21-r9 §20.3; Owner decision D7), frozen before scoring.

DL keeps everything of HG's H layer (§14.2) that §20.3 keeps: the residuals `r_t = (y_t - c_t)/s_t`
with the same issuance scale, the 28-complete-released-day buffer and its warm-up, the hour
shrinkage form `w_h = n_h/(n_h + 56)` with `w_h = 0` below 14 distinct days, the seven quantiles,
and the release, consume-once and failure rules. It is a subclass of CP-16's frozen
`SharedResidualState`: `issue`, the release mechanics and the buffer are inherited unchanged, and
only the quantile step and an adaptive-coverage update at each release are added.

**Recency weights.** Each complete released day in the buffer has weight `2^(-a/7)`, `a` its age
in calendar days from the newest buffer day (age 0). Weights are normalised to sum to one; every
canonical observation carries its day's weight. Weighted empirical quantiles replace `Q_P` and
`Q_h`; `n_h` becomes hour h's effective count `(Σw)²/Σw²`; the p50 bias correction is the
weighted median. W+DLF's weights are `(2/3)·k7/Σk7 + (1/3)·k1/Σk1` with `k1(a) = 2^(-a)`.

**Weighted quantile (frozen definition).** Sort the values; with normalised weights `w` and
cumulative sums `S`, place each sorted value at `p_k = (S_k - w_k/2 - w_1/2) / (1 - w_1/2 - w_n/2)`
(weighted mid-points, rescaled so the smallest value sits at 0 and the largest at 1), and
interpolate linearly in `p`. With equal weights `p_k = (k-1)/(n-1)`: exactly the
`numpy.quantile(method='linear')` positions. It is monotone in the level and reflection-symmetric.
With equal weights (W+ACI) the layer uses `numpy.quantile` itself, so that W+ACI differs from W
only by ACI; with ACI off as well, DL reproduces H bit for bit (a control).

**Adaptive coverage (ACI; Gibbs-Candès).** For each central interval with nominal miscoverage
α ∈ {0.05, 0.20, 0.50}, a working `α_t` starts at α at the fold's genuine warm-up start. The
interval at `α_t` uses the residual levels `α_t/2` and `1 - α_t/2`. After each released complete
day that carried an emitted interval, `α_{t+1} = clip(α_t + γ·(α - m_t), α/5, min(2α, 0.9))` with
γ = 0.10 and `m_t` that day's fraction of canonical hours outside the interval emitted at `α_t`
(the final, rearranged quantiles: below the lower or above the upper bound). Releases of several
days apply in delivery-day order. A day with more misses than nominal lowers `α_t` (wider); fewer
raises it. If adjusted quantiles cross, each row's seven quantiles are sorted (fixed monotone
rearrangement). DL emits an interval on every day its buffer holds 28 complete days, warm-up
included, so the warm-up updates `α_t`.
"""
from __future__ import annotations

from datetime import date
import json

import numpy as np
import pandas as pd

from cp15.data import LEVELS, array_hash, day_hours
from cp16.residuals import SharedResidualState, _date, _index, _vector

ALPHAS = (0.05, 0.20, 0.50)
#: Positions of each nominal interval's (lower, upper) in the seven emitted quantiles.
PAIRS = {0.05: (0, 6), 0.20: (1, 5), 0.50: (2, 4)}
GAMMA = 0.10
HALF_LIFE_DAYS = 7.0
FAST_HALF_LIFE_DAYS = 1.0
FAST_SHARE = 1.0 / 3.0
SHRINK = 56
MIN_HOUR_DAYS = 14
BUFFER_DAYS = 28

#: The frozen variants (§20.2-§20.3). W+ACI: unweighted buffer, ACI. DL: 7-day weights, ACI.
#: DLF: DL plus the fast kernel. H-PARITY: no weights, no ACI -- must reproduce H bit for bit.
VARIANTS = {
    'ACI': {'half_life': None, 'fast_share': 0.0, 'aci': True},
    'DL': {'half_life': HALF_LIFE_DAYS, 'fast_share': 0.0, 'aci': True},
    'DLF': {'half_life': HALF_LIFE_DAYS, 'fast_share': FAST_SHARE, 'aci': True},
    'H-PARITY': {'half_life': None, 'fast_share': 0.0, 'aci': False},
}


def clip_bounds(alpha: float) -> tuple[float, float]:
    return alpha / 5, min(2 * alpha, 0.9)


def aci_update(alpha_t: float, alpha: float, miss_fraction: float, gamma: float = GAMMA) -> float:
    lo, hi = clip_bounds(alpha)
    return float(min(max(alpha_t + gamma * (alpha - miss_fraction), lo), hi))


def day_weights(ages, half_life: float | None, fast_share: float = 0.0) -> np.ndarray:
    """Normalised day weights from ages in days (newest = 0). `half_life=None`: equal weights."""
    ages = np.asarray(ages, float)
    if ages.ndim != 1 or not len(ages) or (ages < 0).any() or not np.isfinite(ages).all():
        raise ValueError('invalid buffer ages')
    if half_life is None:
        if fast_share:
            raise ValueError('a fast component needs the 7-day memory')
        return np.full(len(ages), 1.0 / len(ages))
    slow = 2.0 ** (-ages / half_life)
    weights = slow / slow.sum()
    if fast_share:
        if not 0 < fast_share < 1:
            raise ValueError('fast share must lie in (0, 1)')
        fast = 2.0 ** (-ages / FAST_HALF_LIFE_DAYS)
        weights = (1 - fast_share) * weights + fast_share * (fast / fast.sum())
    return weights


def weighted_quantile(values, weights, levels) -> np.ndarray:
    """The frozen weighted linear quantile (module docstring)."""
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    levels = np.asarray(levels, float)
    if values.ndim != 1 or values.shape != weights.shape or not len(values):
        raise ValueError('values and weights must be one nonempty vector each')
    if not (np.isfinite(values).all() and np.isfinite(weights).all() and (weights > 0).all()):
        raise ValueError('weights must be finite and positive, values finite')
    if ((levels < 0) | (levels > 1)).any():
        raise ValueError('levels must lie in [0, 1]')
    if len(values) == 1:
        return np.full(levels.shape, values[0])
    order = np.argsort(values, kind='stable')
    x, w = values[order], weights[order] / weights.sum()
    mid = np.cumsum(w) - w / 2
    p = (mid - w[0] / 2) / (1 - w[0] / 2 - w[-1] / 2)
    p[0], p[-1] = 0.0, 1.0
    return np.interp(levels, p, x)


def effective_count(weights) -> float:
    weights = np.asarray(weights, float)
    return float(weights.sum() ** 2 / np.square(weights).sum())


class DynamicResidualState(SharedResidualState):
    """CP-16's shared residual state with DL's quantile step and ACI (one state per policy)."""

    def __init__(self, variant: str = 'DL'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'unknown DL variant {variant}')
        self.variant = variant
        self.params = dict(VARIANTS[variant])
        self.alpha = {a: a for a in ALPHAS}
        self._emitted = {}
        self.aci_trace = []

    # -------------------------------------------------------------- the quantile step
    def levels(self) -> np.ndarray:
        a95, a80, a50 = (self.alpha[a] for a in ALPHAS)
        return np.array([a95 / 2, a80 / 2, a50 / 2, .5, 1 - a50 / 2, 1 - a80 / 2, 1 - a95 / 2])

    def ready(self) -> bool:
        return len(self._buffer) == BUFFER_DAYS

    def predict(self, origin_day, index, a1_central, b2_central, current_scale):
        origin_day = _date(origin_day)
        if self.last_origin != origin_day:
            raise ValueError('release must be called for the prediction origin')
        index = _index(origin_day, index)
        n = len(index)
        central = _vector(_vector(a1_central, n, 'A1 central') / 2 + _vector(b2_central, n, 'B2 central') / 2, n, 'blend central')
        scale = _vector(current_scale, n, 'current scale', scale=True)
        if not self.ready():
            raise ValueError(f'need {BUFFER_DAYS} complete released days, got {len(self._buffer)}')
        if origin_day in self._emitted:
            raise ValueError('an interval was already emitted for this day')
        days = [d for d, _ in self._buffer]
        newest = days[-1]
        ages = np.array([(newest - d).days for d in days], float)
        dw = day_weights(ages, self.params['half_life'], self.params['fast_share'])
        errors = np.concatenate([e for _, e in self._buffer])
        obs_w = np.concatenate([np.full(len(e), w) for (_, e), w in zip(self._buffer, dw)])
        hours = np.concatenate([day_hours(d).tz_convert('Europe/Berlin').hour for d in days])
        obs_days = [d for d, e in self._buffer for _ in e]
        levels = self.levels() if self.params['aci'] else LEVELS.copy()
        weighted = self.params['half_life'] is not None
        pooled = weighted_quantile(errors, obs_w, levels) if weighted else np.quantile(errors, levels, method='linear')
        hourly, support = {}, {}
        for hour in range(24):
            mask = hours == hour
            hour_days = len({d for d, m in zip(obs_days, mask) if m})
            n_h = effective_count(obs_w[mask]) if mask.any() else 0.0
            if weighted:
                weight = n_h / (n_h + SHRINK) if hour_days >= MIN_HOUR_DAYS else 0.
                hourly[hour] = (weight * weighted_quantile(errors[mask], obs_w[mask], levels) + (1 - weight) * pooled
                                if weight else pooled.copy())
            else:  # cp16.residuals.hour_quantiles' arithmetic, at the current levels
                count = int(mask.sum())
                weight = count / (count + SHRINK) if hour_days >= MIN_HOUR_DAYS else 0.
                hourly[hour] = (weight * np.quantile(errors[mask], levels, method='linear') + (1 - weight) * pooled
                                if weight else pooled.copy())
            support[str(hour)] = {'n': int(mask.sum()), 'n_effective': n_h, 'distinct_days': hour_days, 'weight': weight}
        h = np.asarray([hourly[hour] for hour in index.tz_convert('Europe/Berlin').hour])
        raw = central[:, None] + scale[:, None] * h
        crossed = int((np.diff(raw, axis=1) < 0).any(axis=1).sum())
        out = np.sort(raw, axis=1)  # fixed monotone rearrangement; a no-op when already ordered
        if not np.isfinite(out).all():
            raise ValueError('invalid emitted quantiles')
        self._emitted[origin_day] = {'index': index, 'alpha_t': dict(self.alpha),
                                     'bounds': {a: (out[:, lo].copy(), out[:, hi].copy()) for a, (lo, hi) in PAIRS.items()}}
        first, last = days[0], days[-1]
        metadata = {'buffer_start': str(first), 'buffer_end': str(last), 'buffer_days': len(self._buffer),
                    'buffer_hours': len(errors), 'buffer_sha256': array_hash(errors), 'central_sha256': array_hash(central),
                    'scale_sha256': array_hash(scale), 'buffer_age_days': (origin_day - last).days,
                    'buffer_oldest_age_days': (origin_day - first).days, 'buffer_calendar_span_days': (last - first).days + 1,
                    'hour_support': support, 'variant': self.variant, 'alpha_t': {str(a): v for a, v in self.alpha.items()},
                    'levels': levels.tolist(), 'day_weights_sum': float(dw.sum()), 'newest_day_weight': float(dw[-1]),
                    'effective_days': effective_count(dw), 'crossed_rows_rearranged': crossed}
        return {'DL': out}, metadata

    # -------------------------------------------------------------- ACI at release
    def release(self, origin_day, truth):
        before = set(self._consumed)
        super().release(origin_day, truth)
        buffered = {d for d, _ in self._buffer}
        for day in sorted(set(self._consumed) - before):
            emitted = self._emitted.pop(day, None)
            if emitted is None or not self.params['aci']:
                continue
            if day not in buffered:  # an incomplete issued day never updates anything
                self.aci_trace.append({'origin': str(origin_day), 'day': str(day), 'status': 'not_a_complete_released_day'})
                continue
            actual = np.asarray(truth(emitted['index'].copy(deep=True)), float)
            if actual.shape != (len(emitted['index']),) or not np.isfinite(actual).all():
                raise ValueError('released day without complete truth')
            record = {'origin': str(origin_day), 'day': str(day), 'status': 'updated'}
            for a in ALPHAS:
                lower, upper = emitted['bounds'][a]
                miss = float(np.mean((actual < lower) | (actual > upper)))
                new = aci_update(self.alpha[a], a, miss)
                record[str(a)] = {'alpha_emitted': emitted['alpha_t'][a], 'alpha_before': self.alpha[a],
                                  'miss_fraction': miss, 'alpha_after': new}
                self.alpha[a] = new
            self.aci_trace.append(record)

    # -------------------------------------------------------------- persistence
    def to_dict(self):
        return {'schema': 'cp22-dynamic-residuals-v1', 'variant': self.variant, 'params': self.params,
                'base': super().to_dict(), 'alpha': {str(a): v for a, v in self.alpha.items()},
                'emitted': [{'day': str(d), 'index': [t.isoformat() for t in e['index']],
                             'alpha_t': {str(a): v for a, v in e['alpha_t'].items()},
                             'bounds': {str(a): [lo.tolist(), hi.tolist()] for a, (lo, hi) in e['bounds'].items()}}
                            for d, e in sorted(self._emitted.items())],
                'aci_trace': json.loads(json.dumps(self.aci_trace, allow_nan=False))}

    @classmethod
    def from_dict(cls, state):
        if set(state) != {'schema', 'variant', 'params', 'base', 'alpha', 'emitted', 'aci_trace'} \
                or state['schema'] != 'cp22-dynamic-residuals-v1':
            raise ValueError('invalid dynamic residual state schema')
        result = cls(state['variant'])
        if state['params'] != result.params:
            raise ValueError('persisted DL parameters differ from the frozen variant')
        base = SharedResidualState.from_dict(state['base'])
        result._pending, result._consumed, result._buffer = base._pending, base._consumed, base._buffer
        result.trace, result.last_origin = base.trace, base.last_origin
        result.alpha = {a: float(state['alpha'][str(a)]) for a in ALPHAS}
        for a in ALPHAS:
            lo, hi = clip_bounds(a)
            if not lo <= result.alpha[a] <= hi:
                raise ValueError('persisted alpha_t outside its clipping bounds')
        for item in state['emitted']:
            day = date.fromisoformat(item['day'])
            index = _index(day, pd.to_datetime(item['index'], utc=True))
            result._emitted[day] = {'index': index, 'alpha_t': {a: float(item['alpha_t'][str(a)]) for a in ALPHAS},
                                    'bounds': {a: (np.asarray(item['bounds'][str(a)][0], float),
                                                   np.asarray(item['bounds'][str(a)][1], float)) for a in ALPHAS}}
        result.aci_trace = json.loads(json.dumps(state['aci_trace'], allow_nan=False))
        return result
