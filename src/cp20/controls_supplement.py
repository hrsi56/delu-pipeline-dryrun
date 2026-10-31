"""Supplementary available-weather positive controls (r13; added after the pre-fit freeze).

The frozen ``controls.py`` positive control multiplies all available weather by 3.0. The
inherited LEAR recipe standardises every column on the training split, so a uniform
positive rescaling of a column (training and forecast rows alike) cancels exactly: the
observed movement (<= 4.4e-13) is floating-point round-off, not sensitivity. That result is
kept as a units/scaling-invariance check; it is not a valid positive control.

These two mutations cannot be absorbed by standardisation (same days, same recipe, charged):
* target-day shift: only delivery day D's D-1-run weather is shifted (+5 m/s at 10 m and
  100 m, +200 W m-2 DSWRF). Training rows, medians and scalers are unchanged, so the
  forecast moves iff a fitted weather coefficient is nonzero;
* training permutation: finite training-history weather rows before D are permuted
  (seed 15042) and the components refitted; the forecast moves iff training weather
  enters the fit.
A mutation passes only above 1e-6 EUR/MWh, far above round-off. The frozen implementation,
its outputs and the comparison are unchanged; this module computes no score.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from .budget import atomic
from .components import HGComponents, augment
from .controls import DAYS, _delta, _fit
from .execution import OUT, cache_dir, check_protocol, hg_identity, ledger, weather_design
from .inputs import load
from .weather import COLUMNS

SHIFT = {'wx_wind10_mean': 5.0, 'wx_wind100_mean': 5.0, 'wx_dswrf_mean': 200.0}
SEED = 15042
THRESHOLD = 1e-6


def run(root: Path):
    p = check_protocol(root)
    design = weather_design(root, p)
    identity = hg_identity(root, p)
    budget = ledger()
    frozen = json.loads((root / OUT / 'causal-controls.json').read_text())['controls']
    results = []
    for (fold, day), original in zip(DAYS, frozen):
        data, _ = load(root)
        rows = data.rows(day)
        budget.reserve(policy_days=3)  # base refit + two variants, conservatively charged
        base = _fit(augment(data, design)[0], day, rows, budget)
        dates = design.table.index.get_level_values(0)
        shifted = design.table.copy()
        target = dates == day
        for column, delta in SHIFT.items():
            shifted.loc[target, column] += delta
        record = {'fold': fold, 'day': str(day), 'target_rows_shifted': int(target.sum())}
        record['target_day_weather_shift'] = _delta(_fit(augment(data, replace(design, table=shifted))[0], day, rows, budget), base)
        permuted = design.table.copy()
        pool = np.flatnonzero((dates < day) & np.isfinite(permuted[list(COLUMNS)].to_numpy(float)).all(axis=1))
        values = permuted[list(COLUMNS)].to_numpy(float, copy=True)
        values[pool] = values[np.random.default_rng(SEED).permutation(pool)]
        permuted[list(COLUMNS)] = values
        record['training_rows_permuted'] = int(len(pool))
        record['training_weather_permutation'] = _delta(_fit(augment(data, replace(design, table=permuted))[0], day, rows, budget), base)
        record['base_refit_vs_cached_hg'] = _delta(base, HGComponents(cache_dir(), fold, data, identity).get(day)[1])
        if max(record['base_refit_vs_cached_hg'].values()) > 1e-8:
            raise ValueError('HG component reproduction failed')
        record['uniform_scaling_x3_invariance'] = original['available_weather_mutation']
        for key in ('target_day_weather_shift', 'training_weather_permutation'):
            record[f'{key}_passed'] = max(record[key].values()) > THRESHOLD
        results.append(record)
        print(json.dumps(record), flush=True)
    atomic(root / OUT / 'causal-controls-supplement.json', {
        'repair': 'r13', 'reason': __doc__.split('\n\n')[1].replace('\n', ' '),
        'threshold_eur_mwh': THRESHOLD, 'seed': SEED, 'shift': SHIFT, 'controls': results,
        'all_passed': all(r['target_day_weather_shift_passed'] and r['training_weather_permutation_passed'] for r in results)})
    if not all(r['target_day_weather_shift_passed'] and r['training_weather_permutation_passed'] for r in results):
        raise ValueError('available-weather positive control insensitive')
