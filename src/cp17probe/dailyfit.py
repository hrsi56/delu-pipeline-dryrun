"""One issued delivery day's daily fit, from its shipped training window, in either runtime (§19.3 item 1).

The same file runs under native CPython and under Pyodide. It holds no model code of its own: A1 is
`cp24.member.fit_member` for each of the eight frozen DDNN-2 members, `cp24.member.emit_rows` and
`cp24.ddnn2.ensemble_median`, laid out exactly as `cp24.execution.fit_origin` lays them out; v2 is
`cp15.models.fit_day` for LEAR's A1 and B2, `cp21.lgbm.hg_central`, and the H layer of
`cp16.residuals.SharedResidualState`, as `cp26.state.PolicyState.issue` calls it. Only the glue that
`cp24.execution` and `cp26` wrap around those calls is restated here, because their own import chains
(the CP-24 job ledger, CP-26's identity pins and resource monitor) cannot load in a browser; the
native tests prove the restatement equal to the landed wrappers, bit for bit.

A payload is a dict of NumPy arrays plus a JSON-able `meta`. The target day's own outcome is never in
it: its price row, slot targets and slot mask (A1) and its hourly targets (v2) are blanked before
shipping, which the native reference checks by reproducing CP-26's stored forecast from the blanked
payload.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import time

import numpy as np

A1_SCHEMA = 'cp17-probe-a1-payload-v1'
V2_SCHEMA = 'cp17-probe-v2-payload-v1'
A1_ARRAYS = ('days', 'price', 'load', 'weather', 'covered', 'centre_s4', 'centre_mad', 'scale_s4', 'scale_mad',
             'stats', 'weekday', 'calendar', 'y', 'mask', 'key_slot', 'key_ts')
V2_ARRAYS = ('index_ns', 'dates', 'hours', 'y', 'eligible', 'feature_valid', 'level', 'scale', 'lear_raw',
             'lear_normalized')


def array_hash(a) -> str:
    """`cp15.data.array_hash`, restated so that payload checks need no landed import."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def index_hash(ix, offset) -> str:
    """The hash of absolute row/day indices as int64. NumPy's default integer is 32-bit on wasm32 and
    64-bit natively, so hashing an index array as computed would differ by runtime with equal values;
    int64 is the native width, so the native hash equals CP-26's recorded one."""
    return array_hash(np.asarray(ix, dtype=np.int64) + int(offset))


def payload_digest(arrays: dict, meta: dict) -> str:
    """One identity for a payload: every array's hash, in name order, and the canonical meta."""
    item = {'arrays': {k: array_hash(arrays[k]) for k in sorted(arrays)}, 'meta': meta}
    return hashlib.sha256(json.dumps(item, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


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


def fit_a1(arrays: dict, meta: dict, charge=None, attribution: bool = False) -> dict:
    """A1's daily fit: the eight frozen members fitted fresh at the day, and their per-level median.

    The member loop, the uncovered-window rule and the ensemble median are `cp24.execution.fit_origin`'s
    (`uncovered_fails=True`, `exclude_uncovered=False`), on the payload's day table."""
    from cp24 import design as G
    from cp24 import ddnn2 as M
    from cp24.member import emit_rows, fit_member
    if meta.get('schema') != A1_SCHEMA:
        raise ValueError('not an A1 payload')
    started = time.perf_counter()
    day = date.fromisoformat(meta['day'])
    dd = a1_daydata(arrays, meta)
    keys = a1_keys(arrays, dd, day)
    di = np.array([dd.ix(day)])
    rows_k = keys.of_days(di)
    eurs, members, fits = [], [], []
    for j, m in enumerate(meta['members']):
        fit = fit_member(dd, day, m['config'], m['seed'], di, exclude_uncovered=False, charge=charge)
        start = date.fromisoformat(fit['window']['start'])
        if any(not dd.covered[i] for i in range(dd.ix(start), dd.ix(day)) if dd.mask[i].any()):
            raise ValueError(f'{day}: a warm-up or evaluation window day lacks a frozen weather record')
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
        raise M.TrainingFailure('nonfinite ensemble quantile')
    out = {'policy': 'A1', 'day': str(day), 'quantiles': q.tolist(), 'central': q[:, M.MEDIAN].tolist(),
           'quantiles_sha256': array_hash(q), 'crossed_rows': int(crossed), 'members': members,
           'fit_seconds': time.perf_counter() - started}
    if attribution:
        from .attribution import group_shapley
        out['attribution'] = group_shapley(dd, day, di, fits)
        slot_p50 = M.ensemble_median(np.stack([f['eur'] for f in fits]))[0][0, :, M.MEDIAN]
        out['attribution']['issued_gap_max_abs'] = float(np.abs(np.asarray(out['attribution']['full_p50']) - slot_p50).max())
    return out


# ------------------------------------------------------------------ v2: HG's LEAR pair with the H layer
def v2_inputs(arrays: dict, meta: dict):
    """The landed `cp15.data.Inputs` over the payload's rows; LightGBM blocks are absent (LEAR only)."""
    import pandas as pd
    from cp15.data import Inputs
    n = len(arrays['dates'])
    empty = np.zeros((n, 0))
    return Inputs(None, None, pd.DatetimeIndex(pd.to_datetime(arrays['index_ns'], utc=True)),
                  arrays['dates'].astype(meta['dates_dtype']), arrays['hours'], arrays['y'],
                  arrays['eligible'].astype(bool), arrays['feature_valid'].astype(bool), arrays['level'],
                  arrays['scale'], empty, empty, arrays['lear_raw'], arrays['lear_normalized'], np.full(n, np.nan),
                  meta['protocol'])


def _lear_selections(log: list[dict]) -> list[dict]:
    keep = ('local_hour', 'n_hour_train', 'selected_relative_alpha', 'alpha', 'final_n_iter', 'final_solver_passes',
            'n_iter_by_alpha', 'solver_passes_by_alpha', 'validation_mae_by_alpha', 'model_sha256')
    return [{k: r[k] for k in keep if k in r} for r in log]


def fit_v2(arrays: dict, meta: dict) -> dict:
    """v2's daily fit: LEAR A1 and B2 refitted at the day, HG's central, and the H layer's seven quantiles
    from the 28 released own-error days (as `cp26.state.PolicyState.issue` issues an H policy)."""
    import pandas as pd
    import cp15.models as m
    from cp21.lgbm import hg_central
    from cp16.residuals import SharedResidualState
    if meta.get('schema') != V2_SCHEMA:
        raise ValueError('not a v2 payload')
    started = time.perf_counter()
    day = date.fromisoformat(meta['day'])
    data = v2_inputs(arrays, meta)
    rows = data.rows(day)
    out, selections, train_hashes = {}, {}, {}
    for policy in ('A1', 'B2'):
        pred, log = m.fit_day(data, day, policy, rows=rows)
        if not np.isfinite(pred).all():
            raise ValueError(f'{policy}: nonfinite central forecast')
        out[policy] = np.asarray(pred, float)
        selections[policy] = _lear_selections(log)
        lower = m.history_start(day, policy)
        train = np.flatnonzero((data.dates >= np.datetime64(lower)) & (data.dates < np.datetime64(day)) & data.eligible)
        train_hashes[policy] = index_hash(train, meta['offset'])
    central = hg_central(out['A1'], out['B2'])
    if not np.array_equal(central / 2 + central / 2, central):
        raise FloatingPointError('single-central H identity underflowed')
    state = SharedResidualState.from_dict({'schema': 'cp16-shared-residuals-v1', 'last_origin': str(day),
                                           'pending': [], 'consumed': [b['day'] for b in meta['h_buffer']],
                                           'buffer': meta['h_buffer'], 'trace': []})
    index = data.index[rows]
    scale = data.scale[rows]
    predictions, h_meta = state.predict(day, index, central, central, scale)
    q = np.asarray(predictions['V2-H'], float)
    return {'policy': 'v2', 'day': str(day), 'quantiles': q.tolist(), 'central': central.tolist(),
            'quantiles_sha256': array_hash(q), 'members_central': {k: v.tolist() for k, v in out.items()},
            'selections': selections, 'train_rows_sha256': train_hashes,
            'h_layer': {k: h_meta[k] for k in ('buffer_start', 'buffer_end', 'buffer_days', 'buffer_hours',
                                                'buffer_sha256', 'central_sha256', 'scale_sha256')},
            'index': [str(pd.Timestamp(t).isoformat()) for t in index],
            'fit_seconds': time.perf_counter() - started}


FITS = {'A1': fit_a1, 'v2': fit_v2}


def load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def run(policy: str, npz_path, meta_path, attribution: bool = False) -> dict:
    """Entry point for both runtimes: load the payload from files and fit it (and, for A1, attribute it)."""
    arrays = load_npz(npz_path)
    with open(meta_path, encoding='utf-8') as handle:
        meta = json.load(handle)
    if meta.get('payload_sha256') != payload_digest(arrays, {k: v for k, v in meta.items() if k != 'payload_sha256'}):
        raise ValueError('payload differs from its recorded identity')
    result = fit_a1(arrays, meta, attribution=attribution) if policy == 'A1' else FITS[policy](arrays, meta)
    result['payload_sha256'] = meta['payload_sha256']
    return result
