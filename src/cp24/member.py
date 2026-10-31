"""One DDNN-2 member fit, and the per-row emission of one member or an ensemble (capstone v21-r11 §23.3).

Glue between the day table (`cp24.design`) and the model code (`cp24.ddnn2`, NumPy only): it selects
the window's days, draws the member's held-out weeks, fits the preprocessor on the training rows,
trains with early stopping, and emits the member's capped quantiles in EUR/MWh for the forecast days.
Every member fit is charged to the ledger before it trains (`charge`). Nothing here fits on held-out,
forecast or later outcomes.

**Keys.** A forecast day's keys are its feature-valid canonical hours (`data.rows(d)`), as A1_w and
B2_w issue them; each key takes the forecast of its local-hour slot, so on a 25-hour day both keys of
the repeated hour receive that slot's forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import resource
import time

import numpy as np
import pandas as pd

from cp15.data import array_hash, origin_utc
from . import ddnn2 as M
from . import design as G


def maxrss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # bytes on macOS


@dataclass
class Keys:
    """The hourly keys of the day table's days: forecast keys (feature-valid rows) and scored keys
    (eligible rows), each with its day index, local-hour slot, timestamp and truth."""
    day: np.ndarray
    slot: np.ndarray
    timestamp: pd.DatetimeIndex
    y: np.ndarray
    eligible: np.ndarray

    @classmethod
    def from_data(cls, data, dd: G.DayData):
        valid = data.feature_valid
        rows = np.flatnonzero(valid)
        day = (data.dates[rows] - dd.days[0]).astype(int)
        return cls(day, data.hours[rows].astype(int), data.index[rows], data.y[rows], data.eligible[rows])

    def of_days(self, day_ix: np.ndarray) -> np.ndarray:
        return np.flatnonzero(np.isin(self.day, day_ix))


def config_hash(config: dict) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def fit_member(dd: G.DayData, origin: date, config: dict, seed: int, forecast_days: np.ndarray, *,
               exclude_uncovered: bool, charge=None, settings: dict | None = None) -> dict:
    """Train one member on [max(2019-01-01, D-728), D) (minus its held-out weeks) and emit its capped
    EUR/MWh quantiles for `forecast_days` (day-table indices): days x 24 x 7, with every guard count."""
    started = time.perf_counter()
    lower, ix, excluded = G.window(dd, origin, exclude_uncovered=exclude_uncovered)
    weeks, pool = G.holdout_weeks(dd, lower, origin, seed)
    train_ix, hold_ix = G.split(dd, ix, weeks)
    if len(train_ix) < G.MIN_TRAIN_DAYS or len(hold_ix) < G.MIN_HOLD_DAYS:
        raise G.InsufficientRows(f'{origin}: {len(train_ix)} training days, {len(hold_ix)} held-out days '
                                 f'(minimum {G.MIN_TRAIN_DAYS}, {G.MIN_HOLD_DAYS})')
    form = config['transform']
    stat = M.statistics_of(form)
    groups = tuple(config['groups'])
    C_tr, B_tr, cnames, bnames = G.raw_inputs(dd, train_ix, groups, form)
    pre = G.Preprocessor(C_tr, B_tr)
    C_ho, B_ho, _, _ = G.raw_inputs(dd, hold_ix, groups, form)
    C_fc, B_fc, _, _ = G.raw_inputs(dd, forecast_days, groups, form)
    if not (np.isfinite(dd.centre[stat][forecast_days]).all() and np.isfinite(dd.scale[stat][forecast_days]).all()):
        raise M.TrainingFailure('a forecast day lacks its origin statistics')
    x_tr, x_ho, x_fc = pre.transform(C_tr, B_tr), pre.transform(C_ho, B_ho), pre.transform(C_fc, B_fc)
    z_tr = G.target_z(dd, train_ix, stat)
    mask_tr = dd.mask[train_ix]
    cap = M.CAP_MULTIPLE * float(np.max(np.abs(z_tr[mask_tr])))
    t_tr = np.where(mask_tr, M.forward_transform(form, np.where(mask_tr, z_tr, 0.0)), 0.0)
    weight = G.recency_weights(dd, train_ix, origin, config.get('half_life'))
    if charge is not None:
        charge()
    s = settings or {}
    member = M.train(x_tr, t_tr, mask_tr, weight, x_ho, np.where(dd.mask[hold_ix], dd.y[hold_ix], 0.0), dd.mask[hold_ix],
                     dd.centre[stat][hold_ix], dd.scale[stat][hold_ix], config, seed, cap, **s)
    zq, active = M.member_quantiles_z(member.params, x_fc, config['activation'], form, cap)
    jsu = np.stack(M.head(M.forward(member.params, x_fc, config['activation'])[0]), axis=-1)   # days x 24 x 4
    eur = dd.centre[stat][forecast_days, None, None] + dd.scale[stat][forecast_days, None, None] * zq
    if not np.isfinite(eur).all():
        raise M.TrainingFailure('nonfinite emitted quantile')
    guards = {'winsor_train': pre.activations(C_tr), 'winsor_hold': pre.activations(C_ho),
              'winsor_forecast': pre.activations(C_fc), 'cap_forecast_slot_levels': int(active.sum()),
              'cap_forecast_by_day': active.sum(axis=(1, 2)).tolist(), 'events': member.events}
    return {'origin': str(origin), 'origin_utc': str(origin_utc(origin).tz_convert('UTC')), 'config': config,
            'config_sha256': config_hash(config), 'seed': int(seed), 'member': member.record(), 'params': member.params,
            'window': {'start': str(lower), 'end_exclusive': str(origin), 'n_days': int(len(ix)),
                       'n_train_days': int(len(train_ix)), 'n_held_out_days': int(len(hold_ix)),
                       'held_out_weeks': [str(w) for w in weeks], 'week_pool': pool,
                       'excluded_uncovered_days': excluded, 'train_days_sha256': array_hash(train_ix)},
            'n_inputs': int(x_tr.shape[1]), 'columns': {'continuous': len(cnames), 'binary': len(bnames)},
            'preprocessor': pre.record(), 'cap_z': cap, 'active': active, 'eur': eur, 'forecast_days': forecast_days,
            'jsu': jsu, 'centre': dd.centre[stat][forecast_days], 'scale': dd.scale[stat][forecast_days],
            'guards': guards, 'seconds': time.perf_counter() - started, 'maxrss_bytes': maxrss()}


def emit_rows(keys: Keys, rows: np.ndarray, forecast_days: np.ndarray, day_eur: np.ndarray) -> np.ndarray:
    """Map day-slot quantiles (days x 24 x L) to the keys `rows` (indices into `keys`): n x L."""
    pos = {int(d): i for i, d in enumerate(forecast_days)}
    out = np.empty((len(rows), day_eur.shape[-1]))
    for j, r in enumerate(rows):
        out[j] = day_eur[pos[int(keys.day[r])], keys.slot[r]]
    return out


def pinball_rows(q: np.ndarray, y: np.ndarray, tau=M.TAU_LEVELS) -> np.ndarray:
    diff = y[:, None] - q
    return np.maximum(tau * diff, (tau - 1.0) * diff)
