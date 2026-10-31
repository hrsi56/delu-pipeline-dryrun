"""HG component fits: the unchanged CP-15 A1/B2 LEAR recipe on a design with only the three
frozen weather columns appended, fitted per origin with every attempt charged first.

H0 and HG call the *same* ``cp15.models.fit_day``; the only difference is the input design
(`augment`): three same-target-local-hour weather columns appended to the raw and
normalised LEAR matrices. The inherited ``prepared_linear`` then adds their missing
indicators and fits medians and scalers on each inner/final training split only.
"""
from __future__ import annotations

import contextlib
from dataclasses import replace
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np

from cp15.data import array_hash, history_start, origin_utc
from .budget import Budget, atomic
from .weather import WeatherDesign

POLICIES = ('A1', 'B2')


@contextlib.contextmanager
def counted_fits(budget: Budget, main: bool):
    """Reserve every component-day attempt and every Lasso solver call (continuations
    included, failures included) before it runs; exhaustion raises before the call."""
    import cp15.models as m
    original_day, original_lasso, original_fit = m.fit_day, m._lasso, m.Lasso.fit
    state = {'ordinal': 0, 'kind': None}

    def day(*args, **kwargs):
        if args[2] not in POLICIES:
            raise ValueError('only the two inherited A1/B2 components are authorised')
        budget.reserve(component_attempts=1, **({'main_component_attempts': 1} if main else {}))
        state['ordinal'], state['kind'] = 0, None
        return original_day(*args, **kwargs)

    def lasso(*args, **kwargs):
        state['kind'] = 'final_fits' if state['ordinal'] % 5 == 4 else 'inner_fits'
        state['ordinal'] += 1
        return original_lasso(*args, **kwargs)

    def fit(self, *args, **kwargs):
        if state['kind'] is None:
            raise RuntimeError('uncategorised estimator attempt')
        budget.reserve(primitive_fits=1, **{state['kind']: 1})
        return original_fit(self, *args, **kwargs)

    m.fit_day, m._lasso, m.Lasso.fit = day, lasso, fit
    try:
        yield m.fit_day
    finally:
        m.fit_day, m._lasso, m.Lasso.fit = original_day, original_lasso, original_fit


def augment(data, design: WeatherDesign):
    """Append the three weather columns; nothing else about the inputs changes."""
    wx = design.matrix(data.dates, data.hours, required=np.zeros(len(data.dates), bool))
    present = design.present(data.dates, data.hours)
    out = replace(data, lear_raw=np.column_stack((data.lear_raw, wx)),
                  lear_normalized=np.column_stack((data.lear_normalized, wx)))
    return out, present


def training_rows(data, day: date, policy: str) -> np.ndarray:
    lower = history_start(day, policy)
    return np.flatnonzero((data.dates >= np.datetime64(lower)) & (data.dates < np.datetime64(day)) & data.eligible)


def fingerprint(identities: dict, design_sha: str, protocol_sha: str) -> str:
    return hashlib.sha256(json.dumps({'inputs': identities, 'weather_design': design_sha, 'protocol': protocol_sha},
                                     sort_keys=True).encode()).hexdigest()


def digest(item: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in item.items() if k != 'content_sha256'}, sort_keys=True,
                                     allow_nan=False, default=str).encode()).hexdigest()


def fit_hg(data_aug, present, day: date, fold: str, budget: Budget, *, main: bool, identity: dict) -> dict:
    """Fresh HG A1/B2 fits for one origin (the component cache entry)."""
    rows = data_aug.rows(day)
    item = {'day': str(day), 'fold': fold, 'origin_utc': str(origin_utc(day).tz_convert('UTC')),
            'timestamp_utc': list(map(str, data_aug.index[rows])), 'scale_sha256': array_hash(data_aug.scale[rows]),
            **identity, 'central': {}, 'fits': {}, 'weather_rows': {}}
    if not len(rows):
        item['source'] = 'original_no_eligible_hours'
        item['content_sha256'] = digest(item)
        return item
    with counted_fits(budget, main) as fit:
        for policy in POLICIES:
            train = training_rows(data_aug, day, policy)
            if not present[train].all() or not present[rows].all():
                raise ValueError(f'{fold} {day}: a training/forecast row lacks a frozen weather record')
            pred, logs = fit(data_aug, day, policy, rows=rows)
            item['central'][policy] = pred.tolist()
            item['fits'][policy] = logs
            xw = data_aug.lear_raw[:, -3:]
            item['weather_rows'][policy] = {
                'train_rows': int(len(train)), 'train_rows_weather_missing': int((~np.isfinite(xw[train])).any(axis=1).sum()),
                'forecast_rows_weather_missing': int((~np.isfinite(xw[rows])).any(axis=1).sum()),
                'forecast_weather_sha256': array_hash(xw[rows])}
    item['source'] = 'fresh_hg_components'
    item['content_sha256'] = digest(item)
    return item


class HGComponents:
    """Verified HG component cache; the replay never fits."""

    def __init__(self, cache_dir: Path, fold: str, data, identity: dict):
        self.dir, self.fold, self.data, self.identity = Path(cache_dir) / fold, fold, data, identity

    def path(self, day):
        return self.dir / f'{day}.json'

    def load(self, day: date):
        path = self.path(day)
        if not path.exists():
            return None
        item = json.loads(path.read_text())
        rows = self.data.rows(day)
        if item.get('content_sha256') != digest(item) or item['day'] != str(day) or item['fold'] != self.fold \
                or any(item.get(k) != v for k, v in self.identity.items()) \
                or item['origin_utc'] != str(origin_utc(day).tz_convert('UTC')) \
                or item['timestamp_utc'] != list(map(str, self.data.index[rows])) \
                or item['scale_sha256'] != array_hash(self.data.scale[rows]):
            raise ValueError(f'stale or wrong HG component cache {self.fold} {day}')
        return item

    def get(self, day: date):
        item = self.load(day)
        if item is None:
            raise ValueError(f'HG component cache miss {self.fold} {day}: fit in the components job first')
        rows = self.data.rows(day)
        if not len(rows):
            return rows, {'A1': np.array([]), 'B2': np.array([])}, item['source']
        centers = {p: np.asarray(item['central'][p], float) for p in POLICIES}
        if any(v.shape != (len(rows),) or not np.isfinite(v).all() for v in centers.values()):
            raise ValueError('invalid HG central vector')
        return rows, centers, 'verified_cp20_hg_cache'

    def save(self, item: dict):
        atomic(self.path(date.fromisoformat(item['day'])), item)
