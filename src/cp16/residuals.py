"""CP-16's fixed shared, causal blend-error state; no model fitting occurs here."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cp15.data import LEVELS, array_hash, day_hours


def _date(value):
    if type(value) is not date:
        raise ValueError('delivery/origin day must be a datetime.date')
    return value


def _index(day, index):
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        raise ValueError('canonical timestamps must be timezone aware')
    index = index.tz_convert('UTC').as_unit('ns').copy(deep=True)
    if (not len(index) or index.has_duplicates or not index.is_monotonic_increasing
            or not index.isin(day_hours(_date(day))).all()):
        raise ValueError('invalid canonical delivery-day timestamps')
    return index


def _vector(value, n, name, *, scale=False):
    result = np.array(value, dtype=float, copy=True)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError(f'invalid {name}')
    if scale and (result < 1).any():
        raise ValueError('A1 scale must be floored at 1')
    result.setflags(write=False)
    return result


def _blend(a1, b2, n):
    a1 = _vector(a1, n, 'A1 central')
    b2 = _vector(b2, n, 'B2 central')
    return _vector(a1 / 2 + b2 / 2, n, 'blend central')


def hour_quantiles(pooled, errors, delivery_days):
    """Return the fixed hourly shrinkage and support, including sparse fallback.

    ``delivery_days`` has one entry per canonical observation, so a repeated
    autumn hour raises n but does not count as two distinct delivery days.
    """
    pooled = _vector(pooled, len(LEVELS), 'pooled quantiles')
    if (np.diff(pooled) < 0).any():
        raise ValueError('pooled quantiles must be ordered')
    days = list(delivery_days)
    for day in days:
        _date(day)
    errors = _vector(errors, len(days), 'hour residuals')
    distinct_days = len(set(days))
    weight = len(errors) / (len(errors) + 56) if distinct_days >= 14 else 0.
    # Do not even evaluate an empty hourly empirical distribution.
    result = (weight * np.quantile(errors, LEVELS, method='linear')
              + (1 - weight) * pooled) if weight else pooled.copy()
    return result, {'n': len(errors), 'distinct_days': distinct_days, 'weight': weight}


@dataclass(frozen=True)
class _Issued:
    index: pd.DatetimeIndex
    central: np.ndarray
    scale: np.ndarray


class SharedResidualState:
    """One buffer for both arms, retaining the latest 28 complete released days.

    Call release(D, truth) before predict(D, ...), then register the genuine
    component issuance with issue. Warm-up component issuances may be registered
    before the common history supports prediction. Truth is a callable accepting
    an ordered UTC DatetimeIndex and returning a matching float vector; nonfinite
    or incomplete truth leaves that day pending for a later release call.
    """
    def __init__(self):
        self._pending = {}
        self._consumed = set()
        self._buffer = []
        self.trace = []
        self.last_origin = None

    def issue(self, day, index, a1_central, b2_central, issued_scale):
        day = _date(day)
        if day in self._pending or day in self._consumed:
            raise ValueError('issued day already registered')
        index = _index(day, index)
        central = _blend(a1_central, b2_central, len(index))
        scale = _vector(issued_scale, len(index), 'issued scale', scale=True)
        self._pending[day] = _Issued(index, central, scale)

    def release(self, origin_day, truth):
        origin_day = _date(origin_day)
        if self.last_origin is not None and origin_day < self.last_origin:
            raise ValueError('feedback origin cannot move backwards')
        self.last_origin = origin_day
        cutoff = origin_day - timedelta(days=2)
        for day in sorted(self._pending):
            if day > cutoff:
                break
            issued = self._pending[day]
            if not issued.index.equals(day_hours(day).as_unit('ns')):
                self.trace.append({'origin': str(origin_day), 'feedback_day': str(day),
                                   'status': 'incomplete_issued_day', 'n': len(issued.index)})
                self._consumed.add(day)
                continue
            actual = np.asarray(truth(issued.index.copy(deep=True)), dtype=float)
            if actual.shape != (len(issued.index),) or not np.isfinite(actual).all():
                self.trace.append({'origin': str(origin_day), 'feedback_day': str(day),
                                   'status': 'unavailable_complete_truth', 'n': 0})
                continue
            errors = _vector((actual - issued.central) / issued.scale,
                             len(issued.index), 'standardized residuals')
            # Late truth must not evict a newer delivery day.
            self._buffer = sorted([*self._buffer, (day, errors)], key=lambda x: x[0])[-28:]
            self._consumed.add(day)
            self.trace.append({'origin': str(origin_day), 'feedback_day': str(day),
                               'status': 'consumed', 'n': len(errors),
                               'issued_center_sha256': array_hash(issued.central),
                               'issued_scale_sha256': array_hash(issued.scale),
                               'error_sha256': array_hash(errors)})
        for day in self._consumed:
            self._pending.pop(day, None)

    def predict(self, origin_day, index, a1_central, b2_central, current_scale):
        origin_day = _date(origin_day)
        if self.last_origin != origin_day:
            raise ValueError('release must be called for the prediction origin')
        index = _index(origin_day, index)
        central = _blend(a1_central, b2_central, len(index))
        scale = _vector(current_scale, len(index), 'current scale', scale=True)
        if len(self._buffer) != 28:
            raise ValueError(f'need 28 complete released days, got {len(self._buffer)}')
        errors = np.concatenate([e for _, e in self._buffer])
        hours = np.concatenate([day_hours(d).tz_convert('Europe/Berlin').hour for d, _ in self._buffer])
        days = [d for d, e in self._buffer for _ in e]
        pooled = np.quantile(errors, LEVELS, method='linear')
        hourly, support = {}, {}
        for hour in range(24):
            mask = hours == hour
            hourly[hour], support[str(hour)] = hour_quantiles(
                pooled, errors[mask], [d for d, selected in zip(days, mask) if selected])
        h = np.asarray([hourly[hour] for hour in index.tz_convert('Europe/Berlin').hour])
        predictions = {'V2-H': central[:, None] + scale[:, None] * h,
                       'V2-P': central[:, None] + scale[:, None] * pooled}
        for output in predictions.values():
            if not np.isfinite(output).all() or (np.diff(output, axis=1) < 0).any():
                raise ValueError('invalid emitted quantiles')
        first, last = self._buffer[0][0], self._buffer[-1][0]
        metadata = {'buffer_start': str(first), 'buffer_end': str(last),
                    'buffer_days': len(self._buffer), 'buffer_hours': len(errors),
                    'buffer_sha256': array_hash(errors), 'central_sha256': array_hash(central),
                    'scale_sha256': array_hash(scale), 'buffer_age_days': (origin_day - last).days,
                    'buffer_oldest_age_days': (origin_day - first).days,
                    'buffer_calendar_span_days': (last - first).days + 1,
                    'hour_support': support}
        return predictions, metadata

    def to_dict(self):
        return {'schema': 'cp16-shared-residuals-v1',
                'last_origin': str(self.last_origin) if self.last_origin else None,
                'pending': [{'day': str(d), 'index': [t.isoformat() for t in item.index],
                             'central': item.central.tolist(), 'scale': item.scale.tolist()}
                            for d, item in sorted(self._pending.items())],
                'consumed': [str(d) for d in sorted(self._consumed)],
                'buffer': [{'day': str(d), 'errors': e.tolist()} for d, e in self._buffer],
                'trace': json.loads(json.dumps(self.trace, allow_nan=False))}

    @classmethod
    def from_dict(cls, state):
        if set(state) != {'schema', 'last_origin', 'pending', 'consumed', 'buffer', 'trace'} or state['schema'] != 'cp16-shared-residuals-v1':
            raise ValueError('invalid residual state schema')
        result = cls()
        result.last_origin = date.fromisoformat(state['last_origin']) if state['last_origin'] else None
        consumed = [date.fromisoformat(d) for d in state['consumed']]
        if len(consumed) != len(set(consumed)):
            raise ValueError('duplicate consumed day')
        result._consumed = set(consumed)
        for item in state['pending']:
            day = date.fromisoformat(item['day'])
            if day in result._pending or day in result._consumed:
                raise ValueError('issued day already registered')
            index = _index(day, pd.to_datetime(item['index'], utc=True))
            # Preserve the saved blend exactly; do not perform a second blend.
            central = _vector(item['central'], len(index), 'persisted central')
            scale = _vector(item['scale'], len(index), 'persisted scale', scale=True)
            result._pending[day] = _Issued(index, central, scale)
        for item in state['buffer']:
            day = date.fromisoformat(item['day'])
            errors = _vector(item['errors'], len(day_hours(day)), 'persisted residuals')
            result._buffer.append((day, errors))
        buffer_days = [d for d, _ in result._buffer]
        if (len(buffer_days) > 28 or buffer_days != sorted(set(buffer_days))
                or not set(buffer_days).issubset(result._consumed)
                or (result._consumed and result.last_origin is None)
                or any(d > result.last_origin - timedelta(days=2) for d in result._consumed)):
            raise ValueError('invalid causal buffer state')
        if not isinstance(state['trace'], list):
            raise ValueError('invalid residual trace')
        result.trace = json.loads(json.dumps(state['trace'], allow_nan=False))
        return result

    def dumps(self):
        return json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)

    @classmethod
    def loads(cls, value):
        return cls.from_dict(json.loads(value))

    def save(self, path):
        Path(path).write_text(self.dumps() + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls.loads(Path(path).read_text(encoding='utf-8'))


ResidualBuffer = SharedResidualState
