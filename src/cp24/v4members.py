"""v4's members at a pre-fold origin, through their unchanged code (capstone v21-r11 §23.6, §23.10).

* A1_w and B2_w through `cp21.components.refit` (CP-15's `fit_day` on CP-20's weather-augmented LEAR
  design, every component-day attempt and Lasso call charged first);
* L-N and L-R through `cp21.lgbm.fit_arm` (CP-21's training-only capacity selection and refit).

**The weather-coverage wrapper** (§23.6, scoped and logged). Delivery days 2022-09-29..2023-03-24 have
no frozen weather record, and §15.3 makes unattempted weather BLOCKED, never imputed, so v4's
unchanged code refuses any origin whose training window reaches into them (the fold-4 gate days).
The wrapper marks every delivery day without a frozen weather record ineligible *before* the call --
`data.eligible & present` on a copy of the inputs -- and records the days it left out. Nothing else
changes: the code, the features, the forecast rows and every rule are the members' own. At a covered
origin the wrapper leaves out nothing, so its call is the unwrapped call; parity with the committed
vectors there is proven by `cp24.gate.job_v4_parity`.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import time

import numpy as np

from cp15.data import array_hash, history_start, origin_utc
from cp20.components import augment
from cp21 import components as C21
from cp21.lgbm import fit_arm, hg_central, hgl_central
from cp21.inputs import weather_matrix
from .budget import LedgerAdapter

ARMS = ('L-N', 'L-R')


def excluded_days(data, present: np.ndarray, day: date) -> list[str]:
    """The eligible training-window days of `day` that the wrapper leaves out (no frozen weather)."""
    lower = history_start(day, 'B3')
    window = (data.dates >= np.datetime64(lower)) & (data.dates < np.datetime64(day)) & data.eligible
    out = np.unique(data.dates[window & ~present])
    return [str(d) for d in out]


def wrap(data, present: np.ndarray):
    """The scoped wrapper: a copy of the inputs whose days without a frozen weather record are ineligible."""
    return replace(data, eligible=data.eligible & present)


def fit(data, design, day: date, budget, purpose: str, *, wrapped: bool = True) -> dict:
    """A1_w, B2_w, L-N and L-R central forecasts for every feature-valid row of `day` (EUR/MWh)."""
    started = time.perf_counter()
    rows = data.rows(day)
    if not len(rows):
        raise ValueError(f'{day}: no forecast rows')
    data_aug, present = augment(data, design)
    wx, present_w = weather_matrix(design, data)
    if not np.array_equal(present, present_w):
        raise ValueError('weather presence differs between the LEAR and LightGBM designs')
    excluded = excluded_days(data, present, day) if wrapped else []
    lear_data = wrap(data_aug, present) if wrapped else data_aug
    lgbm_data = wrap(data, present) if wrapped else data
    comps = C21.refit(LedgerAdapter(budget), purpose, lear_data, present, day, rows)

    def charge(role):
        budget.reserve(lgbm_fits=1, **{f'lgbm_fits_{purpose}': 1})
    centrals, selection = {'A1': comps['A1'], 'B2': comps['B2']}, {}
    for arm in ARMS:
        central, records, sel = fit_arm(lgbm_data, wx, present, day, arm, charge=charge, n_jobs=1)
        centrals[arm] = central
        selection[arm] = {'selection': sel, 'fits': len(records),
                          'trees_sha256': [r['model_sha256'] for r in records if r['role'] == 'final']}
    for name, v in centrals.items():
        if v.shape != (len(rows),) or not np.isfinite(v).all():
            raise ValueError(f'{day}: invalid {name} central')
    hg = hg_central(centrals['A1'], centrals['B2'])
    v4 = hgl_central(centrals['A1'], centrals['B2'], centrals['L-N'], centrals['L-R'])
    return {'day': str(day), 'origin_utc': str(origin_utc(day).tz_convert('UTC')),
            'timestamp_utc': list(map(str, data.index[rows])), 'rows_sha256': array_hash(rows),
            'central': {k: v.tolist() for k, v in centrals.items()},
            'central_sha256': {k: array_hash(v) for k, v in centrals.items()},
            'HG_sha256': array_hash(hg), 'v4_sha256': array_hash(v4), 'selection': selection,
            'wrapper': {'applied': bool(wrapped), 'excluded_training_days': excluded,
                        'n_excluded_training_days': len(excluded)},
            'seconds': time.perf_counter() - started}


def refuses_unwrapped(data, design, day: date, budget) -> dict:
    """The positive control: v4's unwrapped member code refuses an origin whose window reaches into
    the uncovered days (it raises before any fit)."""
    out = {}
    data_aug, present = augment(data, design)
    wx, _ = weather_matrix(design, data)
    rows = data.rows(day)
    try:
        C21.refit(LedgerAdapter(budget), 'control', data_aug, present, day, rows)
        out['A1_w/B2_w'] = 'ran (NOT refused)'
    except ValueError as exc:
        out['A1_w/B2_w'] = f'refused: {exc}'
    for arm in ARMS:
        try:
            fit_arm(data, wx, present, day, arm, charge=lambda role: budget.reserve(lgbm_fits=1, lgbm_fits_control=1))
            out[arm] = 'ran (NOT refused)'
        except ValueError as exc:
            out[arm] = f'refused: {exc}'
    out['all_refused'] = all(v.startswith('refused') for k, v in out.items())
    out['excluded_by_wrapper'] = excluded_days(data, present, day)
    return out


def window_reaches_gap(day: date) -> bool:
    lower = max(date(2019, 1, 1), day - timedelta(days=728))
    return lower <= date(2023, 3, 24) and day > date(2022, 9, 29)
