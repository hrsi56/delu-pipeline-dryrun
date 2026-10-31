"""PN: one pooled LightGBM for all 24 local hours on §4's normalised target (capstone v21-r9 §20.2).

PN is L-N's representation fitted as L-P's single pooled model: CP-15's 23-feature normalised
LightGBM design (price-valued features centred and scaled by each row's own origin statistics),
the three frozen same-target-local-hour GFS columns and their indicators under the §15.3 rule
(fitted on each fit's own training partition), the target `z = (y - level) / scale`, the inherited
CP-15 parameters with only capacity from the frozen G1-G4 grid, seed 42, deterministic row-wise
training. Window `[max(2019-01-01, D-728), D)`, at least 8,760 rows, fresh fits at every origin.

At every origin PN makes eight fits, every one charged before it runs:

* four **inner** fits (G1-G4) on the window minus its last 28 calendar delivery days, each scored
  by validation MAE in EUR/MWh after inversion with each row's own origin level and scale;
* four **full-window** fits (G1-G4).

**PN-avg** is the equal mean of the four full-window forecasts, each inverted to EUR/MWh with the
origin's common level and scale. **PN-sel** applies §17.3's rule (lowest inner validation MAE, an
exact tie to the smaller configuration) and issues the selected configuration's full-window fit --
the same fitted model, so §17.3's "refit on the whole window" is that fit, bit for bit (a control
refits it independently and compares the trees).
"""
from __future__ import annotations

import time

import numpy as np

from cp15.data import array_hash, origin_utc
from cp21.lgbm import (GRID, INNER_DAYS, MIN_TRAIN_ROWS, MIN_VALIDATION_ROWS, POOLED, Design, WeatherImputer,  # noqa: F401
                       _maxrss, _model_sha, model_rows, parameters)
from lightgbm import LGBMRegressor

FITS_PER_ORIGIN = 2 * len(GRID)
#: PN-avg's two routes (invert-then-average, average-then-invert) must agree within this (EUR/MWh).
AVERAGING_TOLERANCE = 1e-9


def pn_design(data, wx: np.ndarray) -> Design:
    """L-N's representation: the normalised CP-15 LightGBM features and the §4 normalised target."""
    return Design(data.lgbm_normalized, wx, (data.y - data.level) / data.scale, True)


def _fit(design: Design, rows: np.ndarray, config: dict, p: dict, n_jobs: int):
    imputer = WeatherImputer(design.wx[rows])
    x = np.column_stack((design.x[rows], imputer.transform(design.wx[rows])))
    model = LGBMRegressor(alpha=.5, random_state=p['seed'], **parameters(p, config, n_jobs))
    wall, cpu = time.perf_counter(), time.process_time()
    model.fit(x, design.target[rows])
    return model, imputer, time.perf_counter() - wall, time.process_time() - cpu


def _z(model, imputer, design: Design, rows: np.ndarray) -> np.ndarray:
    return model.predict(np.column_stack((design.x[rows], imputer.transform(design.wx[rows]))))


def invert(z: np.ndarray, data, rows: np.ndarray) -> np.ndarray:
    return z * data.scale[rows] + data.level[rows]


def pn_avg(full_eur: list[np.ndarray]) -> np.ndarray:
    """The equal mean of the four full-window EUR/MWh forecasts, float64, G1..G4 left to right."""
    if len(full_eur) != len(GRID):
        raise ValueError('PN-avg needs exactly the four full-window forecasts')
    a, b, c, d = (np.asarray(v, float) for v in full_eur)
    return (a + b + c + d) / 4


def select(losses: dict) -> int:
    """§17.3: the lowest validation MAE; an exact tie goes to the smaller configuration (grid order)."""
    return min(range(len(GRID)), key=lambda i: (losses[GRID[i]['id']], i))


def fit_pn(data, wx: np.ndarray, present: np.ndarray, day, *, charge, n_jobs: int = 1) -> tuple[dict, list[dict]]:
    """PN at one origin: returns the issued member vectors and one record per fit attempt."""
    p = data.p
    design = pn_design(data, wx)
    lower, window, inner, validation, forecast = model_rows(data, day, POOLED, present)
    if not np.array_equal(forecast, data.rows(day)):
        raise ValueError(f'PN {day}: pooled forecast rows differ from the eligible rows')
    base = {'arm': 'PN', 'model': POOLED, 'delivery_date': str(day), 'origin_utc': str(origin_utc(day).tz_convert('UTC')),
            'history_start': str(lower), 'history_end_exclusive': str(day), 'n_window': int(len(window)),
            'n_inner': int(len(inner)), 'n_validation': int(len(validation)), 'n_forecast': int(len(forecast)),
            'window_rows_sha256': array_hash(window), 'window_target_sha256': array_hash(design.target[window]),
            'inner_rows_sha256': array_hash(inner), 'validation_rows_sha256': array_hash(validation),
            'forecast_rows_sha256': array_hash(forecast)}
    if not len(forecast):
        raise ValueError(f'PN {day}: no forecast rows')
    if len(window) < MIN_TRAIN_ROWS[POOLED] or len(inner) < MIN_TRAIN_ROWS[POOLED]:
        raise ValueError(f'PN {day}: LightGBM training insufficiency (window {len(window)}, inner {len(inner)})')
    if len(validation) < MIN_VALIDATION_ROWS[POOLED]:
        raise ValueError(f'PN {day}: validation insufficiency {len(validation)}')
    if not present[window].all() or not present[forecast].all():
        raise ValueError(f'PN {day}: a training/forecast row lacks a frozen weather record')
    records, losses = [], {}
    for index, config in enumerate(GRID):
        charge('inner')
        model, imputer, wall, cpu = _fit(design, inner, config, p, n_jobs)
        start = time.perf_counter()
        vp = invert(_z(model, imputer, design, validation), data, validation)
        mae = float(np.mean(np.abs(vp - data.y[validation])))
        if not np.isfinite(mae):
            raise ValueError(f'PN {day}: nonfinite validation MAE')
        losses[config['id']] = mae
        records.append({**base, 'role': 'inner', 'config': config['id'], 'config_index': index,
                        'n_estimators': config['n_estimators'], 'num_leaves': config['num_leaves'], 'n_train': int(len(inner)),
                        'validation_mae': mae, 'fit_wall_seconds': wall, 'fit_cpu_seconds': cpu,
                        'predict_wall_seconds': time.perf_counter() - start, 'maxrss_bytes': _maxrss(),
                        'imputer_fill': imputer.fill.tolist(), 'model_sha256': _model_sha(model)})
    selected = select(losses)
    full_z, full_eur = [], []
    for index, config in enumerate(GRID):
        charge('final')
        model, imputer, wall, cpu = _fit(design, window, config, p, n_jobs)
        start = time.perf_counter()
        z = _z(model, imputer, design, forecast)
        eur = invert(z, data, forecast)
        if not np.isfinite(eur).all():
            raise ValueError(f'PN {day} {config["id"]}: nonfinite forecast')
        full_z.append(z)
        full_eur.append(eur)
        records.append({**base, 'role': 'final', 'config': config['id'], 'config_index': index,
                        'n_estimators': config['n_estimators'], 'num_leaves': config['num_leaves'], 'n_train': int(len(window)),
                        'validation_mae': None, 'fit_wall_seconds': wall, 'fit_cpu_seconds': cpu,
                        'predict_wall_seconds': time.perf_counter() - start, 'maxrss_bytes': _maxrss(),
                        'imputer_fill': imputer.fill.tolist(), 'model_sha256': _model_sha(model),
                        'selected': index == selected, 'forecast_sha256': array_hash(eur)})
    avg = pn_avg(full_eur)
    avg_z_route = invert(pn_avg(full_z), data, forecast)
    gap = float(np.max(np.abs(avg - avg_z_route)))
    if gap > AVERAGING_TOLERANCE:
        raise ValueError(f'PN {day}: averaging before and after inversion disagree by {gap}')
    sel = full_eur[selected]
    ordered = sorted(losses.values())
    member = {'pn_avg': avg, 'pn_sel': sel, 'full_eur': full_eur, 'full_z': full_z, 'selected': GRID[selected]['id'],
              'selected_index': selected, 'validation_mae': losses,
              'tie': sum(1 for v in losses.values() if v == losses[GRID[selected]['id']]) > 1,
              'winner_margin_relative': (ordered[1] - ordered[0]) / ordered[0] if ordered[0] > 0 else None,
              'averaging_route_max_abs_gap': gap}
    return member, records


def composite(a1: np.ndarray, b2: np.ndarray, terms: tuple[tuple[np.ndarray, int], ...]) -> np.ndarray:
    """(2/3)c_HG + (1/3)member in float64, evaluated left to right as A1/3 + B2/3 + sum(m/k):
    a single member enters as m/3, a two-member mean as m1/6 + m2/6 (cp21.lgbm.hgl_central's form)."""
    arrays = [np.asarray(a1, float), np.asarray(b2, float), *(np.asarray(m, float) for m, _ in terms)]
    if len({a.shape for a in arrays}) != 1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('composite needs finite vectors of one shape')
    if sorted(k for _, k in terms) not in ([3], [6, 6]):
        raise ValueError('the member carries exactly one third of the weight')
    out = arrays[0] / 3 + arrays[1] / 3
    for m, k in terms:
        out = out + np.asarray(m, float) / k
    return out
