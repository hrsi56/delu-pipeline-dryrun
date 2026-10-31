"""Real-data causal, weather-origin and arm-difference controls for CP-20 (fits are charged).

For each representative origin D (component-level; no outer scoring, no extra recipe):
* reproduction: a fresh HG refit equals the cached HG component (vector and fingerprints);
* delivery-day/future mask: prices from D on removed and later loads/weather scrambled
  -> HG forecast changes by exactly 0.0; positive control: an available D-1 price
  mutation moves the forecast;
* weather origin: weather for delivery days after D scrambled -> exactly 0.0; positive
  control: available weather (training days and the D-1 run's target-day values) scaled
  -> forecast moves;
* arm difference: HG with every weather value missing (the three columns neutralised,
  indicators constant) reproduces the no-weather H0 component vectors, so the weather
  columns are the only difference between the arms.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from cp15.data import prepare
from .budget import atomic
from .components import HGComponents, augment, counted_fits
from .execution import OUT, cache_dir, check_protocol, hg_identity, ledger, weather_design
from .inputs import H0Components, load

DAYS = (('fold_1', date(2020, 7, 1)), ('fold_4', date(2025, 5, 1)))


def _fit(aug, day, rows, budget):
    out = {}
    with counted_fits(budget, main=False) as fit:
        for policy in ('A1', 'B2'):
            out[policy], _ = fit(aug, day, policy, rows=rows)
    return out


def _delta(a, b):
    return {p: float(np.max(np.abs(a[p] - b[p]))) for p in ('A1', 'B2')}


def run(root: Path):
    p = check_protocol(root)
    design = weather_design(root, p)
    identity = hg_identity(root, p)
    budget = ledger()
    results = []
    for fold, day in DAYS:
        data, _ = load(root)
        aug, present = augment(data, design)
        rows = data.rows(day)
        # Seven real-data component forecast variants per day, conservatively charged as policy-days.
        budget.reserve(policy_days=7)
        cached = HGComponents(cache_dir(), fold, data, identity).get(day)[1]
        record = {'fold': fold, 'day': str(day), 'n_hours': int(len(rows))}
        fresh = _fit(aug, day, rows, budget)
        record['hg_reproduction_max_abs'] = _delta(fresh, cached)
        if max(record['hg_reproduction_max_abs'].values()) > 1e-8:
            raise ValueError('HG component reproduction failed')
        # Delivery-day / future mask (negative) and available D-1 mutation (positive).
        masked = data.frame.copy()
        masked.loc[masked.delivery_date >= day, 'price_eur_mwh'] = np.nan
        masked.loc[masked.delivery_date > day, 'load_forecast_mw'] = 1e8
        mdata = prepare(masked, data.p, data.spec)
        mfuture = design.table.copy()
        future = mfuture.index.get_level_values(0) > day
        mfuture.loc[future, ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean']] = 1e6
        mdesign = replace(design, table=mfuture)
        m_aug, _ = augment(mdata, mdesign)
        record['delivery_day_and_future_mask'] = _delta(_fit(m_aug, day, rows, budget), fresh)
        if any(v != 0.0 for v in record['delivery_day_and_future_mask'].values()):
            raise ValueError('delivery-day or future leakage into HG')
        changed = data.frame.copy()
        changed.loc[changed.delivery_date.eq(day - timedelta(days=1)), 'price_eur_mwh'] += 500
        c_aug, _ = augment(prepare(changed, data.p, data.spec), design)
        record['available_d1_price_mutation'] = _delta(_fit(c_aug, day, rows, budget), fresh)
        if not all(v > 0 for v in record['available_d1_price_mutation'].values()):
            raise ValueError('positive causal control insensitive')
        # Weather origin: future weather only (negative) vs available weather (positive).
        f_aug, _ = augment(data, mdesign)
        record['future_weather_mutation'] = _delta(_fit(f_aug, day, rows, budget), fresh)
        if any(v != 0.0 for v in record['future_weather_mutation'].values()):
            raise ValueError('weather after the origin leaked into HG')
        avail = design.table.copy()
        keep = avail.index.get_level_values(0) <= day
        avail.loc[keep, ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean']] *= 3.0
        a_aug, _ = augment(data, replace(design, table=avail))
        record['available_weather_mutation'] = _delta(_fit(a_aug, day, rows, budget), fresh)
        if not any(v > 0 for v in record['available_weather_mutation'].values()):
            raise ValueError('available weather positive control insensitive')
        # Arm difference: neutralised weather reproduces H0's cached no-weather components.
        blank = design.table.copy()
        blank[['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean']] = np.nan
        n_aug, _ = augment(data, replace(design, table=blank))
        h0 = H0Components(root, fold, data).get(day)[1]
        record['neutralised_weather_vs_h0'] = _delta(_fit(n_aug, day, rows, budget), h0)
        if max(record['neutralised_weather_vs_h0'].values()) > 1e-8:
            raise ValueError('HG with neutralised weather does not reproduce H0')
        record['weather_changes_hg_vs_h0'] = _delta(fresh, h0)
        results.append(record)
        print(json.dumps(record), flush=True)
    atomic(root / OUT / 'causal-controls.json', {'controls': results,
                                                  'fits': 'each control fits A1 and B2 once; charged to component/primitive counters, not main'})
