"""One delivery day's daily fit, from its shipped payload, in every runtime (§19.2, §19.3 item 1).

The same file runs under native CPython and under Pyodide (the issuing host in Node.js, and the browser).
It holds no model code of its own. Each policy is the landed code CP-26 ran, laid out as CP-26 laid it out:

| Policy | Landed calls | CP-26's layout restated here |
|---|---|---|
| A1 (DDNN-2 alone) | `cp24.member.fit_member` x 8, `emit_rows`, `cp24.ddnn2.ensemble_median` | `cp24.execution.fit_origin`, `cp26.members.fit_d2` |
| v2 | `cp15.models.fit_day` (LEAR A1, B2), `cp21.lgbm.hg_central`, `cp16.residuals` H layer | `cp26.members.fit_lear`, `cp26.state.PolicyState.issue` |
| v4 | `fit_day` on the weather-augmented design (A1_w, B2_w), `cp21.lgbm.fit_arm` (L-N, L-R), `hgl_central`, H | `cp26.members.fit_lear`/`fit_lgbm`, `cp26.replay.central_for` |
| v5 | v4's members, A1's D2 central, `cp24.execution.v5_central`, H | `cp26.replay.central_for` |
| B0 | `delu_forecast.baselines.similar_day_naive`, `cp26.state.RawResidualState` | `cp26.members.fit_b0`, `PolicyState.issue` |

Only that glue is restated, because the wrappers' own import chains (CP-24's job ledger, CP-26's identity
pins, guards and resource monitor) cannot load in a browser; the native parity check proves the
restatement equal to CP-26's stored forecasts. Records compared across runtimes hash row and day indices
as int64, since NumPy's default integer is 32-bit on wasm32.

A payload is a dict of NumPy arrays plus a JSON-able `meta`; `meta['payload_sha256']` binds both. The
target day's own outcome is never in it (the embargo): D's price, targets and masks are blanked.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import time

import numpy as np

SCHEMAS = {'A1': 'cp17-a1-payload-v1', 'v2': 'cp17-v2-payload-v1', 'v4': 'cp17-v4-payload-v1',
           'v5': 'cp17-v5-payload-v1', 'B0': 'cp17-b0-payload-v1'}
A1_ARRAYS = ('days', 'price', 'load', 'weather', 'covered', 'centre_s4', 'centre_mad', 'scale_s4', 'scale_mad',
             'stats', 'weekday', 'calendar', 'y', 'mask', 'key_slot', 'key_ts')
ROWS_ARRAYS = ('index_ns', 'dates', 'hours', 'y', 'eligible', 'feature_valid', 'level', 'scale', 'lear_raw',
               'lear_normalized')
V4_ARRAYS = ROWS_ARRAYS + ('lgbm_raw', 'lgbm_normalized', 'wx', 'present')
B0_ARRAYS = ('price_ns', 'price', 'key_ns')
LGBM_ARMS = ('L-N', 'L-R')


class FitFailure(Exception):
    """A policy's daily fit did not succeed (the success definition's items 2-3); the cause is recorded."""


def array_hash(a) -> str:
    """`cp15.data.array_hash`, restated so that payload checks need no landed import."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def index_hash(ix, offset=0) -> str:
    """The hash of absolute row/day indices as int64, equal in every runtime and to CP-26's native hash."""
    return array_hash(np.asarray(ix, dtype=np.int64) + int(offset))


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def payload_digest(arrays: dict, meta: dict) -> str:
    """One identity for a payload: every array's hash, in name order, and the canonical meta."""
    return hashlib.sha256(canonical({'arrays': {k: array_hash(arrays[k]) for k in sorted(arrays)}, 'meta': meta})).hexdigest()


def check_payload(arrays: dict, meta: dict, policy: str, names=()) -> None:
    if meta.get('schema') != SCHEMAS[policy]:
        raise ValueError(f'not a {policy} payload')
    missing = [n for n in names if n not in arrays]
    if missing:
        raise ValueError(f'{policy} payload lacks {missing}')
    body = {k: v for k, v in meta.items() if k != 'payload_sha256'}
    if meta.get('payload_sha256') != payload_digest(arrays, body):
        raise ValueError('payload differs from its recorded identity')


def _quantiles(q, n: int) -> np.ndarray:
    q = np.asarray(q, float)
    if q.shape != (n, 7) or not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
        raise FitFailure('the seven quantiles are not finite and ordered on every key')
    return q


# ------------------------------------------------------------------ A1: DDNN-2 alone
def a1_daydata(arrays: dict, meta: dict):
    """The landed `cp24.design.DayData` over the payload's days."""
    from cp24.design import DayData
    return DayData(arrays['days'].astype('datetime64[D]'), arrays['price'], arrays['load'], arrays['weather'],
                   arrays['covered'].astype(bool), {'s4': arrays['centre_s4'], 'mad': arrays['centre_mad']},
                   {'s4': arrays['scale_s4'], 'mad': arrays['scale_mad']}, arrays['stats'], arrays['weekday'],
                   arrays['calendar'], arrays['y'], arrays['mask'].astype(bool), date.fromisoformat(meta['first']))


def a1_keys(arrays: dict, dd, day: date):
    """The forecast day's keys as the landed `cp24.member.Keys`; outcomes are not shipped (NaN, False)."""
    import pandas as pd
    from cp24.member import Keys
    n = len(arrays['key_slot'])
    return Keys(np.full(n, dd.ix(day), dtype=int), arrays['key_slot'].astype(int),
                pd.DatetimeIndex(pd.to_datetime(arrays['key_ts'], utc=True)), np.full(n, np.nan), np.zeros(n, bool))


def fit_a1(arrays: dict, meta: dict, charge=None, attribution: bool = False, export: bool = False) -> dict:
    """A1's daily fit: the eight frozen members fitted fresh at the day, and their per-level median.

    The member loop, the uncovered-window rule and the ensemble median are `cp24.execution.fit_origin`'s
    (`uncovered_fails=True`, `exclude_uncovered=False`), on the payload's day table. One failed member fails
    the fit: there is no median of a subset (§24.7)."""
    import pandas as pd
    from cp24 import design as G
    from cp24 import ddnn2 as M
    from cp24.member import emit_rows, fit_member
    check_payload(arrays, meta, 'A1', A1_ARRAYS)
    started = time.perf_counter()
    day = date.fromisoformat(meta['day'])
    if len(meta['members']) != 8:
        raise FitFailure('A1 requires exactly eight frozen members')
    dd = a1_daydata(arrays, meta)
    keys = a1_keys(arrays, dd, day)
    di = np.array([dd.ix(day)])
    rows_k = keys.of_days(di)
    eurs, members, fits = [], [], []
    for j, m in enumerate(meta['members']):
        try:
            fit = fit_member(dd, day, m['config'], m['seed'], di, exclude_uncovered=False, charge=charge)
        except M.TrainingFailure as exc:
            raise FitFailure(f'member {j} (seed {m["seed"]}): {type(exc).__name__}: {exc}') from exc
        start = date.fromisoformat(fit['window']['start'])
        if any(not dd.covered[i] for i in range(dd.ix(start), dd.ix(day)) if dd.mask[i].any()):
            raise FitFailure(f'{day}: a warm-up or evaluation window day lacks a frozen weather record')
        q = emit_rows(keys, rows_k, di, fit['eur'])
        eurs.append(q)
        fits.append(fit)
        weeks = [date.fromisoformat(w) for w in fit['window']['held_out_weeks']]
        _, ix, _ = G.window(dd, day, exclude_uncovered=False)
        train, hold = G.split(dd, ix, weeks)
        record = fit['member']
        members.append({
            'member': j, 'rank': m['rank'], 'config': m['config']['id'], 'seed': int(m['seed']),
            'held_out_weeks': fit['window']['held_out_weeks'], 'week_pool': fit['window']['week_pool'],
            'n_days': fit['window']['n_days'], 'n_train_days': fit['window']['n_train_days'],
            'n_held_out_days': fit['window']['n_held_out_days'],
            'train_days_sha256': index_hash(train, meta['offset']),
            'holdout_days_sha256': index_hash(hold, meta['offset']),
            'latest_training_day': str(dd.date_of(int(train[-1]))) if len(train) else None,
            'n_inputs': fit['n_inputs'], 'best_epoch': record['best_epoch'], 'epochs_run': record['epochs_run'],
            'stop_reason': record['stop_reason'], 'params_sha256': record['params_sha256'],
            'best_stopping_metric_eur': record['best_stopping_metric_eur'], 'cap_z': fit['cap_z'],
            'winsor_train': fit['guards']['winsor_train'], 'winsor_hold': fit['guards']['winsor_hold'],
            'winsor_forecast': fit['guards']['winsor_forecast'],
            'cap_forecast_slot_levels': fit['guards']['cap_forecast_slot_levels'], 'events': record['events'],
            'preprocessor': fit['preprocessor'], 'quantiles': q.tolist(), 'seconds': fit['seconds']})
    stack = np.stack(eurs)
    med, crossed = M.ensemble_median(stack[:, :, None, :])
    q = med[:, 0, :]
    if not np.isfinite(q).all():
        raise FitFailure('nonfinite ensemble quantile')
    index = pd.DatetimeIndex(pd.to_datetime(arrays['key_ts'], utc=True))
    out = {'policy': 'A1', 'day': str(day), 'quantiles': q.tolist(), 'central': q[:, M.MEDIAN].tolist(),
           'quantiles_sha256': array_hash(q), 'crossed_rows': int(crossed), 'members': members,
           'index': [t.isoformat() for t in index], 'search': meta['search'],
           'ensemble_content_sha256': meta['ensemble_content_sha256'], 'fit_seconds': time.perf_counter() - started}
    if attribution:
        from .attribution import group_shapley
        out['attribution'] = group_shapley(dd, day, di, fits)
        slot_p50 = M.ensemble_median(np.stack([f['eur'] for f in fits]))[0][0, :, M.MEDIAN]
        out['attribution']['issued_gap_max_abs'] = float(np.abs(np.asarray(out['attribution']['full_p50']) - slot_p50).max())
    if export:   # the fitted artifact's arrays: every member's parameters and its target centre and scale
        out['_arrays'] = {f'member{j}_{name}': np.asarray(value, float)
                          for j, fit in enumerate(fits)
                          for name, value in [*((f'param{k}', p) for k, p in enumerate(fit['params'])),
                                              ('centre', fit['centre']), ('scale', fit['scale'])]}
    return out


def save_npz(path, arrays: dict) -> None:
    """`np.savez_compressed`'s format with fixed entry times and sorted names, so equal arrays give equal bytes
    in every runtime."""
    import io
    import zipfile
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(f'{name}.npy', (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, buffer.getvalue())


# ------------------------------------------------------------------ the CP-15 row designs (v2, v4)
def rows_inputs(arrays: dict, meta: dict):
    """The landed `cp15.data.Inputs` over the payload's rows. v2's LightGBM blocks are empty (LEAR only)."""
    import pandas as pd
    from cp15.data import Inputs
    n = len(arrays['dates'])
    empty = np.zeros((n, 0))
    return Inputs(None, None, pd.DatetimeIndex(pd.to_datetime(arrays['index_ns'], utc=True)),
                  arrays['dates'].astype(meta['dates_dtype']), arrays['hours'], arrays['y'],
                  arrays['eligible'].astype(bool), arrays['feature_valid'].astype(bool), arrays['level'],
                  arrays['scale'], arrays.get('lgbm_raw', empty), arrays.get('lgbm_normalized', empty),
                  arrays['lear_raw'], arrays['lear_normalized'], np.full(n, np.nan), meta['protocol'])


LEAR_SELECTED = ('local_hour', 'n_hour_train', 'selected_relative_alpha', 'alpha', 'final_n_iter',
                 'final_solver_passes', 'n_iter_by_alpha', 'solver_passes_by_alpha', 'validation_mae_by_alpha',
                 'model_sha256', 'scaler_sha256', 'imputer_sha256')


def training_rows(data, day: date, policy: str) -> np.ndarray:
    """`cp15.models.fit_day`'s (and `cp20.components.training_rows`') training rows."""
    from cp15.models import history_start
    lower = history_start(day, policy)
    return np.flatnonzero((data.dates >= np.datetime64(lower)) & (data.dates < np.datetime64(day)) & data.eligible)


def fit_lear(data, day: date, policy: str, rows, offset: int, present=None):
    """One LEAR member at the day (`cp26.members.fit_lear`): the weather check first when the design carries it."""
    import cp15.models as m
    train = training_rows(data, day, policy)
    if present is not None and (not present[train].all() or not present[rows].all()):
        raise FitFailure(f'{day}: a training/forecast row lacks a weather record')
    try:
        pred, log = m.fit_day(data, day, policy, rows=rows)
    except ValueError as exc:
        raise FitFailure(f'LEAR {policy}: {exc}') from exc
    pred = np.asarray(pred, float)
    if not np.isfinite(pred).all():
        raise FitFailure(f'LEAR {policy}: nonfinite central forecast')
    selections = [{k: r[k] for k in LEAR_SELECTED if k in r} for r in log]
    latest = str(data.dates[train].max()) if len(train) else None
    return pred, selections, {'rows_sha256': index_hash(train, offset), 'rows': int(len(train)), 'latest_day': latest}


def fit_lgbm(data, wx, present, day: date, arm: str, offset: int):
    """One three-block LightGBM member at the day (`cp26.members.fit_lgbm`, one thread)."""
    from cp21.lgbm import ARMS, fit_arm, model_rows
    try:
        central, records, selection = fit_arm(data, wx, present, day, arm, charge=lambda role: None, n_jobs=1)
    except ValueError as exc:
        raise FitFailure(f'{arm}: {exc}') from exc
    central = np.asarray(central, float)
    rows = {}
    for model in ARMS[arm]['models']:
        _, window, inner, validation, forecast = model_rows(data, day, model, present)
        rows[model] = {'window_sha256': index_hash(window, offset), 'inner_sha256': index_hash(inner, offset),
                       'validation_sha256': index_hash(validation, offset), 'forecast_sha256': index_hash(forecast, offset),
                       'latest_window_day': str(data.dates[window].max()) if len(window) else None}
    fits = [{k: r[k] for k in ('model', 'role', 'config', 'n_train', 'validation_mae', 'model_sha256') if k in r}
            for r in records]
    return central, {'selected': {k: v['selected'] for k, v in selection.items()},
                     'validation_mae': {k: v['validation_mae'] for k, v in selection.items()},
                     'ties': {k: v['tie'] for k, v in selection.items()}, 'rows': rows, 'fits': fits}


def h_layer(day: date, index, central, scale, buffer: list[dict]):
    """The H layer's seven quantiles from the 28 released own-error days, as `cp26.state.PolicyState.issue`
    issues an H policy: the single-central identity check, then `SharedResidualState.predict`."""
    from cp16.residuals import SharedResidualState
    central = np.asarray(central, float)
    if not np.array_equal(central / 2 + central / 2, central):
        raise FitFailure('single-central H identity underflowed')
    if np.asarray(scale, float).min() < 1:
        raise FitFailure('issued scale below its floor of 1')
    state = SharedResidualState.from_dict({'schema': 'cp16-shared-residuals-v1', 'last_origin': str(day),
                                           'pending': [], 'consumed': [b['day'] for b in buffer],
                                           'buffer': buffer, 'trace': []})
    try:
        predictions, meta = state.predict(day, index, central, central, scale)
    except ValueError as exc:
        raise FitFailure(f'H layer: {exc}') from exc
    q = _quantiles(predictions['V2-H'], len(index))
    keep = ('buffer_start', 'buffer_end', 'buffer_days', 'buffer_hours', 'buffer_sha256', 'central_sha256', 'scale_sha256')
    return q, {k: meta[k] for k in keep}


def _released(day: date, buffer: list[dict], what: str) -> None:
    if any(date.fromisoformat(b['day']) > day - timedelta(days=2) for b in buffer):
        raise ValueError(f'{what}: a buffer day is not released under the complete D-2 rule')


def fit_v2(arrays: dict, meta: dict) -> dict:
    """v2's daily fit: LEAR A1 and B2 refitted at the day, HG's central, and its own H layer."""
    import pandas as pd
    from cp21.lgbm import hg_central
    check_payload(arrays, meta, 'v2', ROWS_ARRAYS)
    started = time.perf_counter()
    day = date.fromisoformat(meta['day'])
    _released(day, meta['h_buffer'], 'v2')
    data = rows_inputs(arrays, meta)
    rows = data.rows(day)
    out, selections, train = {}, {}, {}
    for policy in ('A1', 'B2'):
        out[policy], selections[policy], train[policy] = fit_lear(data, day, policy, rows, meta['offset'])
    central = hg_central(out['A1'], out['B2'])
    index = data.index[rows]
    q, h = h_layer(day, index, central, data.scale[rows], meta['h_buffer'])
    return {'policy': 'v2', 'day': str(day), 'quantiles': q.tolist(), 'central': central.tolist(),
            'quantiles_sha256': array_hash(q), 'members_central': {k: v.tolist() for k, v in out.items()},
            'selections': selections, 'train_rows': train, 'train_rows_sha256': {k: v['rows_sha256'] for k, v in train.items()},
            'h_layer': h, 'index': [pd.Timestamp(t).isoformat() for t in index], 'scale': data.scale[rows].tolist(),
            'fit_seconds': time.perf_counter() - started}


def fit_v4_members(arrays: dict, meta: dict) -> dict:
    """v4's four members at the day: LEAR A1_w and B2_w on the weather-augmented design, and L-N and L-R."""
    import pandas as pd
    check_payload(arrays, meta, 'v4', V4_ARRAYS)
    started = time.perf_counter()
    day = date.fromisoformat(meta['day'])
    data = rows_inputs(arrays, meta)
    present = arrays['present'].astype(bool)
    rows = data.rows(day)
    central, selections, train = {}, {}, {}
    for name, policy in (('A1_w', 'A1'), ('B2_w', 'B2')):
        central[name], selections[name], train[name] = fit_lear(data, day, policy, rows, meta['offset'], present)
    for arm in LGBM_ARMS:
        central[arm], selections[arm] = fit_lgbm(data, arrays['wx'], present, day, arm, meta['offset'])
    return {'day': day, 'index': data.index[rows], 'scale': np.asarray(data.scale[rows], float), 'central': central,
            'selections': selections, 'train_rows': train,
            'index_iso': [pd.Timestamp(t).isoformat() for t in data.index[rows]], 'seconds': time.perf_counter() - started}


def issue_h_policy(policy: str, members: dict, central, buffer: list[dict], started: float) -> dict:
    day = members['day']
    _released(day, buffer, policy)
    q, h = h_layer(day, members['index'], central, members['scale'], buffer)
    return {'policy': policy, 'day': str(day), 'quantiles': q.tolist(), 'central': np.asarray(central).tolist(),
            'quantiles_sha256': array_hash(q),
            'members_central': {k: np.asarray(v).tolist() for k, v in members['central'].items()},
            'selections': members['selections'], 'train_rows': members['train_rows'],
            'train_rows_sha256': {k: v['rows_sha256'] for k, v in members['train_rows'].items()},
            'h_layer': h, 'index': members['index_iso'], 'scale': members['scale'].tolist(),
            'fit_seconds': time.perf_counter() - started}


def fit_v4(arrays: dict, meta: dict, members: dict | None = None) -> dict:
    """v4 = HGL: (2/3)·c_HG + (1/3)·mean(L-N, L-R), evaluated as CP-21 lands it, with v4's own H layer."""
    from cp21.lgbm import hgl_central
    started = time.perf_counter()
    members = members or fit_v4_members(arrays, meta)
    c = members['central']
    return issue_h_policy('v4', members, hgl_central(c['A1_w'], c['B2_w'], c['L-N'], c['L-R']), meta['h_buffer'],
                          started - members['seconds'])


def fit_v5(meta: dict, members: dict, a1: dict) -> dict:
    """v5 (CP-26's control, run natively for 4.7T): v4's members and A1's D2 central, with v5's own H layer."""
    from cp24.execution import v5_central
    check_payload({}, meta, 'v5')
    if a1.get('payload_sha256') not in (None, meta['a1_payload_sha256']):
        raise ValueError('v5: the A1 fit is not the one this payload names')
    if a1['day'] != str(members['day']) or a1['index'] != members['index_iso']:
        raise FitFailure('v5: A1 and v4 members differ in day or keys')
    started = time.perf_counter()
    c = members['central']
    d2 = np.asarray(a1['central'], float)
    out = issue_h_policy('v5', members, v5_central(c['A1_w'], c['B2_w'], c['L-N'], c['L-R'], d2), meta['h_buffer'], started)
    out['members_central']['D2'] = d2.tolist()
    return out


def fit_b0(arrays: dict, meta: dict) -> dict:
    """B0: the similar-day naive (D-1 Tue-Fri, D-7 Mon/Sat/Sun) on the forecast keys, and its raw-error
    quantiles from the 28 released raw errors (`cp26.state.RawResidualState`)."""
    import pandas as pd
    from delu_forecast.baselines import similar_day_naive
    from cp26.state import RawResidualState
    check_payload(arrays, meta, 'B0', B0_ARRAYS)
    started = time.perf_counter()
    day = date.fromisoformat(meta['day'])
    _released(day, meta['b0_buffer'], 'B0')
    frame = pd.DataFrame({'timestamp_utc': pd.to_datetime(arrays['price_ns'], utc=True), 'price_eur_mwh': arrays['price']})
    naive = similar_day_naive(frame)
    index = pd.DatetimeIndex(pd.to_datetime(arrays['key_ns'], utc=True))
    central = naive.reindex(index).to_numpy(float)
    if not np.isfinite(central).all():
        raise FitFailure('B0: the similar day lacks a published price on a forecast key')
    state = RawResidualState.from_dict({'last_origin': str(day), 'pending': [], 'trace': [],
                                        'consumed': [b['day'] for b in meta['b0_buffer']], 'buffer': meta['b0_buffer']})
    try:
        q = _quantiles(state.predict(central), len(index))
    except ValueError as exc:
        raise FitFailure(f'B0: {exc}') from exc
    errors = np.concatenate([np.asarray(b['errors'], float) for b in meta['b0_buffer']])
    return {'policy': 'B0', 'day': str(day), 'quantiles': q.tolist(), 'central': central.tolist(),
            'quantiles_sha256': array_hash(q), 'source_day': meta['source_day'],
            'b0_layer': {'buffer_start': meta['b0_buffer'][0]['day'], 'buffer_end': meta['b0_buffer'][-1]['day'],
                         'buffer_days': len(meta['b0_buffer']), 'buffer_hours': int(len(errors)),
                         'buffer_sha256': array_hash(errors)},
            'index': [t.isoformat() for t in index], 'fit_seconds': time.perf_counter() - started}


# ------------------------------------------------------------------ entry point for every runtime
def load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load(npz_path, meta_path) -> tuple[dict, dict]:
    arrays = load_npz(npz_path) if npz_path else {}
    with open(meta_path, encoding='utf-8') as handle:
        return arrays, json.load(handle)


def run(policy: str, npz_path, meta_path, attribution: bool = False, extra: dict | None = None,
        export: bool = False) -> dict:
    """Load one payload from files and fit it. `extra` carries v5's inputs: the v4 members and A1's result."""
    arrays, meta = load(npz_path, meta_path)
    if policy == 'A1':
        result = fit_a1(arrays, meta, attribution=attribution, export=export)
    elif policy == 'v2':
        result = fit_v2(arrays, meta)
        if attribution:
            from .attribution import v2_linear
            result['attribution'] = v2_linear(rows_inputs(arrays, meta), date.fromisoformat(meta['day']), meta['offset'], result)
    elif policy == 'v4':
        members = fit_v4_members(arrays, meta)
        result = fit_v4(arrays, meta, members)
        if attribution:
            from .attribution import v4_attribution
            result['attribution'] = v4_attribution(rows_inputs(arrays, meta), arrays, date.fromisoformat(meta['day']),
                                                   meta['offset'], result)
        if extra is not None:
            extra['v4_members'] = members
    elif policy == 'v5':
        result = fit_v5(meta, extra['v4_members'], extra['a1'])
    elif policy == 'B0':
        result = fit_b0(arrays, meta)
    else:
        raise ValueError(f'unknown policy {policy}')
    result['payload_sha256'] = meta['payload_sha256'] if policy != 'v5' else meta.get('payload_sha256')
    return result
