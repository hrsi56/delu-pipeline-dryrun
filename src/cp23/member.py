"""One DDNN fit at one origin, and one fold's configuration choice (capstone v21-r10 §21.2).

Glue between the design matrix (`cp23.features`) and the model code (`cp23.ddnn`, NumPy only): it
selects rows, fits the §15.3 preprocessor on the training rows, trains the seeded ensemble with early
stopping, and emits the EUR/MWh quantiles and the central forecast D (the ensemble median). Every
member fit is charged to the ledger before it trains (`charge`). Nothing here fits on validation,
holdout or forecast outcomes.
"""
from __future__ import annotations

from datetime import date
import resource
import time

import numpy as np

from cp15.data import array_hash, origin_utc
from . import ddnn as D
from .features import origin_rows, rows_record, selection_rows

CONFIG_BY_ID = {c['id']: c for c in D.CONFIGS}


def maxrss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # bytes on macOS


def fit_origin(data, x: np.ndarray, wx: np.ndarray, z: np.ndarray, present: np.ndarray, day: date, config_id: str, *,
               seeds=D.SEEDS, charge=None, keep_members: bool = False) -> dict:
    """DDNN's forecast for every eligible hour of `day`: the ensemble of `seeds` in configuration
    `config_id`, each member trained on `[max(2019-01-01, D-728), D-28)` and early-stopped on `[D-28, D)`."""
    config = CONFIG_BY_ID[config_id]
    started = time.perf_counter()
    rows = origin_rows(data, day, present)
    train, stop, forecast = rows['train'], rows['stop'], rows['forecast']
    pre = D.Preprocessor(x[train], wx[train])
    x_train, x_stop, x_fc = (pre.transform(x[r], wx[r]) for r in (train, stop, forecast))
    members = D.fit_ensemble(x_train, z[train], x_stop, z[stop], config, seeds, charge=charge)
    t_predict = time.perf_counter()
    zq, each = D.ensemble_quantiles(members, x_fc)
    eur, central, crossed = D.emit(zq, data.level[forecast], data.scale[forecast])
    predict_seconds = time.perf_counter() - t_predict
    jsu = np.stack([np.column_stack(D.head(D.forward(m.params, x_fc)[0])) for m in members])  # members x n x 4
    out = {'day': str(day), 'origin_utc': str(origin_utc(day).tz_convert('UTC')), 'config': config_id,
           'seeds': list(map(int, seeds)), 'n_inputs': int(x_train.shape[1]),
           'timestamp_utc': list(map(str, data.index[forecast])), 'rows': rows_record(data, rows),
           'scale_sha256': array_hash(data.scale[forecast]), 'level_sha256': array_hash(data.level[forecast]),
           'central': central, 'quantiles': eur, 'z_quantiles': zq, 'member_z_quantiles': each, 'member_jsu_params': jsu,
           'central_sha256': array_hash(central), 'quantiles_sha256': array_hash(eur), 'crossed_rows': crossed,
           'members': [m.record() for m in members], 'preprocessor': pre.record(),
           'predict_seconds': predict_seconds, 'seconds': time.perf_counter() - started, 'maxrss_bytes': maxrss()}
    if keep_members:
        out['member_objects'] = members
    return out


def fit_selection(data, x: np.ndarray, wx: np.ndarray, z: np.ndarray, present: np.ndarray, d0: date, *, seeds=D.SEEDS,
                  charge=None) -> dict:
    """The fold's one configuration choice before its first origin `d0` (training data only):
    each configuration's seeded ensemble trained on `[max(2019-01-01, D0-728), D0-56)`, early-stopped
    on `[D0-56, D0-28)`, and judged by the MAE (EUR/MWh) of its ensemble median on `[D0-28, D0)`;
    an exact tie goes to the smaller configuration."""
    started = time.perf_counter()
    rows = selection_rows(data, d0, present)
    train, stop, holdout = rows['train'], rows['stop'], rows['holdout']
    pre = D.Preprocessor(x[train], wx[train])
    x_train, x_stop, x_hold = (pre.transform(x[r], wx[r]) for r in (train, stop, holdout))
    candidates, records = {}, {}
    for config in D.CONFIGS:
        members = D.fit_ensemble(x_train, z[train], x_stop, z[stop], config, seeds, charge=charge)
        zq, _ = D.ensemble_quantiles(members, x_hold)
        _, median, crossed = D.emit(zq, data.level[holdout], data.scale[holdout])
        candidates[config['id']] = (median, None)
        records[config['id']] = {'members': [m.record() for m in members], 'crossed_rows': crossed,
                                 'holdout_median_sha256': array_hash(median),
                                 'n_parameters': D.n_parameters(x_train.shape[1], config['hidden'])}
    chosen, info = D.select_configuration(candidates, data.y[holdout])
    return {'first_origin': str(d0), 'rows': rows_record(data, rows), 'preprocessor': pre.record(), **info,
            'configurations': records, 'holdout_truth_sha256': array_hash(data.y[holdout]),
            'seconds': time.perf_counter() - started, 'maxrss_bytes': maxrss()}
