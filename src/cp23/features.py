"""DDNN's representation: exactly v4's information on §4's normalised target (capstone v21-r10 §21.2).

**One row per delivery hour**, pooled over the 24 local hours, as PN and L-P are: the rows, history
window, inner split and sufficiency rules are `cp21.lgbm.model_rows`'s pooled ones.

**Inputs.**

* CP-15's 23-feature normalised LightGBM design (`data.lgbm_normalized`, L-N's and PN's
  representation: price-valued features centred and scaled by each row's own origin statistics).
  Its four categorical columns -- `local_hour`, `day_of_week`, `month` and `day_type` -- are one-hot
  encoded, the same information as a network-readable code.
* The three frozen same-target-local-hour GFS columns and their three missing indicators under §15.3
  (`cp23.ddnn.Preprocessor`), fitted on each fit's own training rows, with the training-only
  StandardScaler over every column.

Nothing else enters: no other feature, lag, cross-hour expansion or source.

**Target.** `z = (y - level) / scale`, §4's normalised target, and every quantile is inverted with
the row's own origin level and scale.

**Rows of one fit at origin D:**

* training: `[max(2019-01-01, D-728), D-28)`;
* inner validation for early stopping: the window's last 28 calendar delivery days, `[D-28, D)`;
* forecast: the eligible hours of D.

**The fold's configuration choice**, made once, before the fold's first origin D0, from data before
D0 only:

* training: `[max(2019-01-01, D0-728), D0-56)`;
* early stopping: `[D0-56, D0-28)`;
* selection holdout: `[D0-28, D0)`.

The holdout is kept apart from the early-stopping days, so the configuration is judged on days its
training never saw.
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from cp15.data import array_hash, history_start
from cp21.lgbm import INNER_DAYS, MIN_TRAIN_ROWS, MIN_VALIDATION_ROWS, POOLED, model_rows

CATEGORICAL = {'local_hour': tuple(range(24)), 'day_of_week': tuple(range(7)), 'month': tuple(range(1, 13)),
               'day_type': (0, 1, 2)}
MIN_ROWS = MIN_TRAIN_ROWS[POOLED]            # 8,760 = 365 x 24, the inherited pooled minimum
MIN_HOLDOUT_ROWS = MIN_VALIDATION_ROWS[POOLED]  # 336 = 14 x 24
SELECTION_DAYS = 28


class InsufficientRows(ValueError):
    pass


def encode(data) -> tuple[np.ndarray, list[str]]:
    """All rows' inputs before §15.3: the normalised CP-15 features with the categoricals one-hot."""
    names = data.p['lgbm']['features']
    cols, labels = [], []
    eligible = data.eligible
    for j, name in enumerate(names):
        v = data.lgbm_normalized[:, j]
        if name in CATEGORICAL:
            levels = CATEGORICAL[name]
            if not np.isin(v[eligible], levels).all():
                raise ValueError(f'{name} has a value outside {levels}')
            for level in levels:
                cols.append((v == level).astype(float))
                labels.append(f'{name}={level}')
        else:
            cols.append(v.astype(float))
            labels.append(name)
    return np.column_stack(cols), labels


def target(data) -> np.ndarray:
    return (data.y - data.level) / data.scale


def origin_rows(data, day: date, present: np.ndarray) -> dict:
    """The training, early-stopping and forecast rows of one fit at origin `day`."""
    lower, window, inner, validation, forecast = model_rows(data, day, POOLED, present)
    if not len(forecast):
        raise InsufficientRows(f'{day}: no eligible forecast hours')
    if len(window) < MIN_ROWS or len(inner) < MIN_ROWS:
        raise InsufficientRows(f'{day}: training insufficiency (window {len(window)}, inner {len(inner)} < {MIN_ROWS})')
    if len(validation) < MIN_HOLDOUT_ROWS:
        raise InsufficientRows(f'{day}: early-stopping insufficiency {len(validation)} < {MIN_HOLDOUT_ROWS}')
    if not present[window].all() or not present[forecast].all():
        raise InsufficientRows(f'{day}: a training or forecast row lacks a frozen weather record')
    return {'history_start': lower, 'train': inner, 'stop': validation, 'forecast': forecast}


def selection_rows(data, d0: date, present: np.ndarray) -> dict:
    """The rows of the fold's one configuration choice, before its first origin `d0`."""
    lower = history_start(d0, 'B3')
    window = np.flatnonzero((data.dates >= np.datetime64(lower)) & (data.dates < np.datetime64(d0)) & data.eligible)
    stop_start = np.datetime64(d0 - timedelta(days=2 * SELECTION_DAYS))
    holdout_start = np.datetime64(d0 - timedelta(days=SELECTION_DAYS))
    train = window[data.dates[window] < stop_start]
    stop = window[(data.dates[window] >= stop_start) & (data.dates[window] < holdout_start)]
    holdout = window[data.dates[window] >= holdout_start]
    if len(train) < MIN_ROWS or len(stop) < MIN_HOLDOUT_ROWS or len(holdout) < MIN_HOLDOUT_ROWS:
        raise InsufficientRows(f'selection before {d0}: train {len(train)}, stop {len(stop)}, holdout {len(holdout)}')
    if not present[window].all():
        raise InsufficientRows(f'selection before {d0}: a row lacks a frozen weather record')
    return {'history_start': lower, 'train': train, 'stop': stop, 'holdout': holdout}


def rows_record(data, rows: dict) -> dict:
    out = {}
    for name, ix in rows.items():
        if isinstance(ix, np.ndarray):
            out[f'n_{name}'] = int(len(ix))
            out[f'{name}_rows_sha256'] = array_hash(ix)
            if len(ix):
                out[f'{name}_first'] = str(data.dates[ix].min())
                out[f'{name}_last'] = str(data.dates[ix].max())
        else:
            out[name] = str(ix)
    return out


assert INNER_DAYS == 28
