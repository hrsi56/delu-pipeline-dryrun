"""CP-21 LightGBM central forecasts (capstone v21-r6 §17.2-§17.3).

One implementation serves all three LightGBM arms, so their differences are exactly the ones the
plan names:

* **L-P**: one pooled model for all 24 local hours, raw target;
* **L-R**: one model per block (night 22-05, solar 10-16, shoulder/peak 06-09 and 17-21), raw
  target -- the same rows as L-P, split by block;
* **L-N**: L-R with the §4 normalised target and CP-15's price-valued centre/scale features.

Every model sees exactly HG's information: CP-15's 23-feature B3/A2 LightGBM design (raw or
normalised), plus the three frozen same-target-local-hour GFS columns under the §15.3 rule --
training-partition median imputation, an all-null column filled with 0, and one missing
indicator per column, fitted separately on each inner-training split and on each final window.
(§15.3's StandardScaler is a strictly increasing per-column map; tree splits are invariant to it,
so it is not applied to tree inputs.) LightGBM's native missing-value routing is never used for
weather: the imputed columns are always finite.

Capacity is chosen on training data only, at every origin and for every model: each of at most
four configurations is fitted on the window minus its last 28 calendar delivery days and scored by
validation MAE in EUR/MWh (L-N inverted first); the lowest wins, an exact tie goes to the smaller
configuration; the winner is refitted on the whole window. Every fit attempt is charged to the
ledger before it runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import resource
import time
import warnings

import numpy as np
from lightgbm import LGBMRegressor

from cp15.data import array_hash, history_start, origin_utc

#: The exhaustive Europe/Berlin local-hour blocks (§17.3). A target hour belongs to exactly one.
BLOCKS: dict[str, tuple[int, ...]] = {
    'night': (22, 23, 0, 1, 2, 3, 4, 5),
    'solar': (10, 11, 12, 13, 14, 15, 16),
    'shoulder': (6, 7, 8, 9, 17, 18, 19, 20, 21),
}
POOLED = 'pooled'
ALL_HOURS = tuple(range(24))
MODEL_HOURS: dict[str, tuple[int, ...]] = {POOLED: ALL_HOURS, **BLOCKS}
#: Inherited LightGBM minimum (8,760 = 365 x 24) and its block equivalents, 365 x block hours.
MIN_TRAIN_ROWS = {m: 365 * len(h) for m, h in MODEL_HOURS.items()}
#: Validation sufficiency, the LightGBM analogue of LEAR's 14 validation rows per hour.
MIN_VALIDATION_ROWS = {m: 14 * len(h) for m, h in MODEL_HOURS.items()}
INNER_DAYS = 28

#: The frozen capacity grid, identical for L-P, L-R and L-N, smallest to largest. The largest is
#: the inherited CP-15 setting; the others have fewer leaves, trees or both (§17.3). "Smaller"
#: for the tie rule is this order (strictly increasing trees x leaves).
GRID: tuple[dict, ...] = (
    {'id': 'G1', 'n_estimators': 150, 'num_leaves': 15},
    {'id': 'G2', 'n_estimators': 300, 'num_leaves': 31},
    {'id': 'G3', 'n_estimators': 600, 'num_leaves': 31},
    {'id': 'G4', 'n_estimators': 600, 'num_leaves': 63},
)
FITS_PER_MODEL = len(GRID) + 1

ARMS: dict[str, dict] = {
    'L-P': {'models': (POOLED,), 'target': 'raw'},
    'L-R': {'models': tuple(BLOCKS), 'target': 'raw'},
    'L-N': {'models': tuple(BLOCKS), 'target': 'normalized'},
}
LGBM_ARMS = tuple(ARMS)


def block_of(hour: int) -> str:
    found = [name for name, hours in BLOCKS.items() if hour in hours]
    if len(found) != 1:
        raise ValueError(f'local hour {hour} is not in exactly one block')
    return found[0]


def parameters(p: dict, config: dict, n_jobs: int) -> dict:
    """The inherited CP-15 parameters with only capacity replaced (§17.3). `n_jobs` is the
    execution thread count; deterministic, row-wise training makes the model independent of it."""
    params = dict(p['lgbm']['parameters'])
    if params['n_estimators'] != 600 or params['num_leaves'] != 63 or not params['deterministic'] \
            or not params['force_row_wise'] or params['objective'] != 'quantile':
        raise ValueError('inherited CP-15 LightGBM recipe changed')
    params.update(n_estimators=config['n_estimators'], num_leaves=config['num_leaves'], n_jobs=n_jobs)
    return params


class WeatherImputer:
    """§15.3 on one training partition: medians (all-null -> 0) and one indicator per column."""

    def __init__(self, train: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            fill = np.nanmedian(train, axis=0)
        self.fill = np.where(np.isfinite(fill), fill, 0.)

    def transform(self, wx: np.ndarray) -> np.ndarray:
        missing = ~np.isfinite(wx)
        return np.column_stack((np.where(missing, self.fill, wx), missing.astype(float)))


@dataclass
class Design:
    """The per-origin inputs one model needs, all rows selected by date and block."""
    x: np.ndarray            # CP-15 LightGBM features, raw or normalised (N x 23)
    wx: np.ndarray           # the three weather columns (N x 3), NaN when missing
    target: np.ndarray       # raw price or (y - level) / scale
    normalized: bool


def arm_design(data, wx: np.ndarray, arm: str) -> Design:
    normalized = ARMS[arm]['target'] == 'normalized'
    target = (data.y - data.level) / data.scale if normalized else data.y
    return Design(data.lgbm_normalized if normalized else data.lgbm_raw, wx, target, normalized)


def model_rows(data, day: date, model: str, present: np.ndarray):
    """Training window, inner split and forecast rows of one model at one origin."""
    hours = MODEL_HOURS[model]
    lower = history_start(day, 'B3')  # [max(2019-01-01, D-728), D), the inherited long window
    in_block = np.isin(data.hours, hours)
    window = np.flatnonzero((data.dates >= np.datetime64(lower)) & (data.dates < np.datetime64(day))
                            & data.eligible & in_block)
    forecast = data.rows(day)
    forecast = forecast[np.isin(data.hours[forecast], hours)]
    split = np.datetime64(day - timedelta(days=INNER_DAYS))
    inner, validation = window[data.dates[window] < split], window[data.dates[window] >= split]
    return lower, window, inner, validation, forecast


def _fit(design: Design, rows: np.ndarray, config: dict, p: dict, n_jobs: int):
    imputer = WeatherImputer(design.wx[rows])
    x = np.column_stack((design.x[rows], imputer.transform(design.wx[rows])))
    model = LGBMRegressor(alpha=.5, random_state=p['seed'], **parameters(p, config, n_jobs))
    wall, cpu = time.perf_counter(), time.process_time()
    model.fit(x, design.target[rows])
    return model, imputer, time.perf_counter() - wall, time.process_time() - cpu


def _predict(model, imputer, design: Design, rows: np.ndarray, data) -> np.ndarray:
    x = np.column_stack((design.x[rows], imputer.transform(design.wx[rows])))
    pred = model.predict(x)
    return pred * data.scale[rows] + data.level[rows] if design.normalized else pred


def _model_sha(model) -> str:
    """SHA-256 of the fitted trees: the model string through its 'end of trees' marker. The
    parameter block that follows records the execution thread count, which never changes the
    trees under deterministic row-wise training (reports/block-challenger/thread-determinism.json)."""
    text = model.booster_.model_to_string()
    end = text.index('end of trees') + len('end of trees')
    return hashlib.sha256(text[:end].encode()).hexdigest()


def _maxrss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # bytes on macOS


def fit_select(data, design: Design, present: np.ndarray, day: date, arm: str, model: str, *,
               charge, n_jobs: int = 1) -> tuple[np.ndarray, list[dict], dict]:
    """Training-only capacity selection and the final refit for one model at one origin.

    `charge(role)` must reserve one fit attempt in the ledger before each fit (role = 'inner' or
    'final'); a refusal raises before the fit. Returns the forecast for the model's rows of `day`
    (EUR/MWh), one record per fit attempt, and the selection summary."""
    p = data.p
    lower, window, inner, validation, forecast = model_rows(data, day, model, present)
    base = {'arm': arm, 'model': model, 'delivery_date': str(day), 'origin_utc': str(origin_utc(day).tz_convert('UTC')),
            'history_start': str(lower), 'history_end_exclusive': str(day), 'n_window': int(len(window)),
            'n_inner': int(len(inner)), 'n_validation': int(len(validation)), 'n_forecast': int(len(forecast)),
            'window_rows_sha256': array_hash(window), 'window_target_sha256': array_hash(design.target[window]),
            'inner_rows_sha256': array_hash(inner), 'validation_rows_sha256': array_hash(validation),
            'forecast_rows_sha256': array_hash(forecast)}
    if not len(forecast):
        raise ValueError(f'{arm} {model} {day}: no forecast rows')
    if len(window) < MIN_TRAIN_ROWS[model] or len(inner) < MIN_TRAIN_ROWS[model]:
        raise ValueError(f'{arm} {model} {day}: LightGBM training insufficiency '
                         f'(window {len(window)}, inner {len(inner)} < {MIN_TRAIN_ROWS[model]})')
    if len(validation) < MIN_VALIDATION_ROWS[model]:
        raise ValueError(f'{arm} {model} {day}: validation insufficiency {len(validation)} < {MIN_VALIDATION_ROWS[model]}')
    if not present[window].all() or not present[forecast].all():
        raise ValueError(f'{arm} {model} {day}: a training/forecast row lacks a frozen weather record')
    records, losses = [], {}
    for index, config in enumerate(GRID):
        charge('inner')
        fitted, imputer, wall, cpu = _fit(design, inner, config, p, n_jobs)
        start = time.perf_counter()
        vp = _predict(fitted, imputer, design, validation, data)
        mae = float(np.mean(np.abs(vp - data.y[validation])))
        if not np.isfinite(mae):
            raise ValueError(f'{arm} {model} {day}: nonfinite validation MAE')
        losses[config['id']] = mae
        records.append({**base, 'role': 'inner', 'config': config['id'], 'config_index': index,
                        'n_estimators': config['n_estimators'], 'num_leaves': config['num_leaves'],
                        'n_train': int(len(inner)), 'validation_mae': mae, 'fit_wall_seconds': wall,
                        'fit_cpu_seconds': cpu, 'predict_wall_seconds': time.perf_counter() - start,
                        'maxrss_bytes': _maxrss(), 'imputer_fill': imputer.fill.tolist(), 'model_sha256': _model_sha(fitted)})
    # Lowest validation MAE; an exact tie goes to the smaller configuration (grid order).
    selected_index = min(range(len(GRID)), key=lambda i: (losses[GRID[i]['id']], i))
    selected = GRID[selected_index]
    charge('final')
    fitted, imputer, wall, cpu = _fit(design, window, selected, p, n_jobs)
    start = time.perf_counter()
    pred = _predict(fitted, imputer, design, forecast, data)
    if not np.isfinite(pred).all():
        raise ValueError(f'{arm} {model} {day}: nonfinite forecast')
    records.append({**base, 'role': 'final', 'config': selected['id'], 'config_index': selected_index,
                    'n_estimators': selected['n_estimators'], 'num_leaves': selected['num_leaves'],
                    'n_train': int(len(window)), 'validation_mae': None, 'fit_wall_seconds': wall, 'fit_cpu_seconds': cpu,
                    'predict_wall_seconds': time.perf_counter() - start, 'maxrss_bytes': _maxrss(),
                    'imputer_fill': imputer.fill.tolist(), 'model_sha256': _model_sha(fitted),
                    'forecast_sha256': array_hash(pred)})
    summary = {'selected': selected['id'], 'validation_mae': losses,
               'tie': sum(1 for v in losses.values() if v == losses[selected['id']]) > 1}
    return pred, records, summary


def fit_arm(data, wx: np.ndarray, present: np.ndarray, day: date, arm: str, *, charge, n_jobs: int = 1):
    """One arm's central forecast for every eligible row of `day`: L-P's pooled model, or the
    three block models of L-R / L-N assembled back into row order."""
    design = arm_design(data, wx, arm)
    rows = data.rows(day)
    central = np.full(len(rows), np.nan)
    records, selection = [], {}
    for model in ARMS[arm]['models']:
        pred, recs, summary = fit_select(data, design, present, day, arm, model, charge=charge, n_jobs=n_jobs)
        mask = np.isin(data.hours[rows], MODEL_HOURS[model])
        if mask.sum() != len(pred):
            raise ValueError('model forecast rows disagree with block membership')
        central[mask] = pred
        records += recs
        selection[model] = summary
    if not np.isfinite(central).all():
        raise ValueError(f'{arm} {day}: an eligible hour has no block forecast')
    return central, records, selection


def hgl_central(a1: np.ndarray, b2: np.ndarray, ln: np.ndarray, lr: np.ndarray) -> np.ndarray:
    """HGL's fixed blend (§17.2), evaluated left to right in float64:
    c = A1_w/3 + B2_w/3 + L-N/6 + L-R/6, i.e. (2/3)c_HG + (1/3)mean(L-N, L-R)."""
    arrays = [np.asarray(v, float) for v in (a1, b2, ln, lr)]
    if len({a.shape for a in arrays}) != 1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('HGL blend needs four finite vectors of one shape')
    a1, b2, ln, lr = arrays
    return a1 / 3 + b2 / 3 + ln / 6 + lr / 6


def hg_central(a1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """HG's central forecast, exactly as `cp16.residuals._blend` forms it."""
    return np.asarray(a1, float) / 2 + np.asarray(b2, float) / 2
