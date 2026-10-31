"""Native construction of the shipped training payloads (§19.3 item 1: "that fit's shipped training window").

The development inputs are read through CP-26's pure loaders (`cp26.members.Context.build` under
`cp26.harness.NEW`: CP-25's partition specification, the 2026-09-06 boundary through `cp25.guard`,
CP-20's and CP-25's weather, ruling 4). CP-26's identity pins, authority checks and resource monitor
are not called. Nothing after 2026-09-06 exists in the inputs; `cp25.guard` refuses it.

**A1** ships the DDNN-2 day table from D-735 (the 728-day window plus the seven days its D-7 lags
reach) through D, the forecast day's keys, and the eight frozen members of the serving search.
**v2** ships CP-15's prepared rows from LEAR's history start through D, CP-15's protocol (the LEAR
recipe), and the 28 released own-error days of v2's H layer at the origin.

**The target-day embargo.** D's own price row, slot targets and slot mask (A1), and D's hourly
targets (v2), are blanked; the keys carry no outcome. The native reference reproduces CP-26's stored
forecast from the blanked payload.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from .dailyfit import A1_SCHEMA, V2_SCHEMA, array_hash, payload_digest

BOUNDARY = date(2026, 9, 6)


def _admit(day: date) -> None:
    from cp25 import guard
    guard.admit_delivery_day(day)
    if day > BOUNDARY:
        raise ValueError(f'{day} is after the development boundary')


def context(root: Path):
    """CP-26's nominal route inputs, through its pure loaders only."""
    from cp26 import harness as H
    from cp26.members import Context
    H.check_blas()
    return Context.build(Path(root), H.NEW, 'nominal')


def a1_payload(ctx, day: date, ensemble: dict) -> tuple[dict, dict]:
    from cp24 import design as G
    from cp26.search import serving_search, validate_ensemble
    _admit(day)
    k = serving_search(day)
    validate_ensemble(ensemble, k, 'nominal')
    dd, keys, data = ctx.dd, ctx.keys, ctx.data
    lower = max(G.FLOOR, day - timedelta(days=G.WINDOW_DAYS))
    reach = max(max(G.PRICE_LAGS.values()), max(G.LOAD_LAGS.values()))
    first = max(dd.first, lower - timedelta(days=reach))
    a, b = dd.ix(first), dd.ix(day) + 1
    sl = slice(a, b)
    rows_k = keys.of_days(np.array([dd.ix(day)]))
    if not np.array_equal(keys.timestamp[rows_k], data.index[data.rows(day)]):
        raise ValueError(f'{day}: forecast keys differ between the day table and the inputs')
    arrays = {'days': dd.days[sl].astype('int64'), 'price': dd.price[sl].copy(), 'load': dd.load[sl].copy(),
              'weather': dd.weather[sl].copy(), 'covered': dd.covered[sl].astype(np.int8),
              'centre_s4': dd.centre['s4'][sl].copy(), 'centre_mad': dd.centre['mad'][sl].copy(),
              'scale_s4': dd.scale['s4'][sl].copy(), 'scale_mad': dd.scale['mad'][sl].copy(),
              'stats': dd.stats[sl].copy(), 'weekday': dd.weekday[sl].copy(), 'calendar': dd.calendar[sl].copy(),
              'y': dd.y[sl].copy(), 'mask': dd.mask[sl].astype(np.int8),
              'key_slot': keys.slot[rows_k].astype(np.int64),
              'key_ts': keys.timestamp[rows_k].tz_convert('UTC').as_unit('ns').asi8.astype(np.int64)}
    arrays['price'][-1] = np.nan
    arrays['y'][-1] = np.nan
    arrays['mask'][-1] = 0
    members = [{'config': m['config'], 'seed': int(m['seed']), 'rank': int(m['rank']), 'repeat': int(m['repeat'])}
               for m in ensemble['members']]
    meta = {'schema': A1_SCHEMA, 'policy': 'A1', 'day': str(day), 'first': str(first), 'offset': int(a),
            'search': int(k), 'ensemble_content_sha256': ensemble['content_sha256'], 'members': members,
            'embargo': 'price[D], y[D] and mask[D] blanked; keys carry no outcome',
            'array_sha256': {name: array_hash(arrays[name]) for name in sorted(arrays)}}
    meta['payload_sha256'] = payload_digest(arrays, meta)
    return arrays, meta


def v2_payload(ctx, day: date, h_state: dict, protocol: dict) -> tuple[dict, dict]:
    import cp15.models as m
    _admit(day)
    data = ctx.data
    lower = min(m.history_start(day, 'A1'), m.history_start(day, 'B2'))
    sel = np.flatnonzero((data.dates >= np.datetime64(lower)) & (data.dates <= np.datetime64(day)))
    if not len(sel) or not np.array_equal(sel, np.arange(sel[0], sel[-1] + 1)):
        raise ValueError('v2 payload rows are not one contiguous block')
    offset = int(sel[0])
    on_day = data.dates[sel] == np.datetime64(day)
    arrays = {'index_ns': data.index[sel].tz_convert('UTC').as_unit('ns').asi8.astype(np.int64),
              'dates': data.dates[sel].astype('int64'), 'hours': data.hours[sel].copy(), 'y': data.y[sel].copy(),
              'eligible': data.eligible[sel].astype(np.int8), 'feature_valid': data.feature_valid[sel].astype(np.int8),
              'level': data.level[sel].copy(), 'scale': data.scale[sel].copy(),
              'lear_raw': data.lear_raw[sel].copy(), 'lear_normalized': data.lear_normalized[sel].copy()}
    arrays['y'][on_day] = np.nan
    buffer = h_state['buffer']
    if h_state['last_origin'] != str(day) or len(buffer) != 28:
        raise ValueError('the H state is not the one released at this origin')
    if any(date.fromisoformat(b['day']) > day - timedelta(days=2) for b in buffer):
        raise ValueError('an H buffer day is not released under the complete D-2 rule')
    meta = {'schema': V2_SCHEMA, 'policy': 'v2', 'day': str(day), 'offset': offset,
            'dates_dtype': str(data.dates.dtype), 'protocol': protocol,
            'h_buffer': [{'day': b['day'], 'errors': b['errors']} for b in buffer],
            'embargo': "y on D's rows blanked; the H buffer holds released days <= D-2 only",
            'array_sha256': {name: array_hash(arrays[name]) for name in sorted(arrays)}}
    meta['payload_sha256'] = payload_digest(arrays, meta)
    return arrays, meta


def write(arrays: dict, meta: dict, folder: Path, stem: str) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    npz, js = folder / f'{stem}.npz', folder / f'{stem}.json'
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps(meta, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return {'npz': str(npz), 'meta': str(js), 'payload_sha256': meta['payload_sha256'],
            'npz_bytes': npz.stat().st_size, 'meta_bytes': js.stat().st_size}
