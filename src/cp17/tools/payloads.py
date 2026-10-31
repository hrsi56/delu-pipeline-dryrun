"""Native construction of the shipped training-window payloads (item 7's "training window" kind).

The development inputs are read through CP-26's pure loaders (`cp26.members.Context.build` under
`cp26.harness.NEW`: CP-25's partition specification, the 2026-09-06 boundary through `cp25.guard`, CP-20's
and CP-25's weather, ruling 4), as CP-17-probe read them. CP-26's identity pins, authority checks and
resource monitor are not called. Nothing after 2026-09-06 exists in the inputs; `cp25.guard` refuses it.

| Payload | Arrays | Meta |
|---|---|---|
| A1 | DDNN-2's day table D-735..D, the forecast keys | the eight frozen members of the serving search |
| v2 | CP-15's prepared rows from LEAR's history start through D | CP-15's protocol, v2's H buffer |
| v4 | the same rows with the three weather columns appended to LEAR's design (`cp20.components.augment`), CP-15's LightGBM design, the weather matrix and its presence | the protocol, v4's H buffer |
| v5 | none (v4's members and A1's D2 are fitted from their payloads) | v5's H buffer, the two payload identities |
| B0 | hourly prices D-7..D and the forecast keys | B0's raw-error buffer, the similar day |

**The target-day embargo.** D's own price row, slot targets and slot mask (A1), D's hourly targets (v2, v4),
and D's prices (B0) are blanked. Every buffer holds released days <= D-2 only. The native parity check
reproduces CP-26's stored forecasts from the blanked payloads.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from ..dailyfit import SCHEMAS, array_hash, payload_digest

BOUNDARY = date(2026, 9, 6)
RUN = 'reports/rebaseline/cold-rerun-2026-10-15'


def admit(day: date) -> None:
    from cp25 import guard
    guard.admit_delivery_day(day)
    if day > BOUNDARY:
        raise ValueError(f'{day} is after the development boundary')


def context(root: Path):
    """CP-26's nominal route inputs, through its pure loaders only (BLAS must be at one thread)."""
    from cp26 import harness as H
    from cp26.members import Context
    H.check_blas()
    return Context.build(Path(root), H.NEW, 'nominal')


def ensemble(inputs: Path, day: date) -> dict:
    """The nominal search that serves `day` (§24.7: S_k serves S_k .. S_(k+1) - 1), from the verified extract."""
    from cp26.search import serving_search, validate_ensemble
    k = serving_search(day)
    value = json.loads((Path(inputs) / RUN / 'search-results' / 'nominal' / f'k{k:02d}.json').read_text())
    validate_ensemble(value, k, 'nominal')
    return value


def _finish(arrays: dict, meta: dict) -> tuple[dict, dict]:
    meta['array_sha256'] = {name: array_hash(arrays[name]) for name in sorted(arrays)}
    meta['payload_sha256'] = payload_digest(arrays, meta)
    return arrays, meta


def a1_payload(ctx, day: date, ens: dict) -> tuple[dict, dict]:
    from cp24 import design as G
    from cp26.search import serving_search, validate_ensemble
    admit(day)
    k = serving_search(day)
    validate_ensemble(ens, k, 'nominal')
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
               for m in ens['members']]
    meta = {'schema': SCHEMAS['A1'], 'policy': 'A1', 'day': str(day), 'first': str(first), 'offset': int(a),
            'search': int(k), 'search_day': ens['search_day'], 'ensemble_content_sha256': ens['content_sha256'],
            'members': members, 'embargo': 'price[D], y[D] and mask[D] blanked; keys carry no outcome'}
    return _finish(arrays, meta)


def _rows(data, day: date, policies) -> np.ndarray:
    import cp15.models as m
    lower = min(m.history_start(day, p) for p in policies)
    sel = np.flatnonzero((data.dates >= np.datetime64(lower)) & (data.dates <= np.datetime64(day)))
    if not len(sel) or not np.array_equal(sel, np.arange(sel[0], sel[-1] + 1)):
        raise ValueError('payload rows are not one contiguous block')
    return sel


def _row_arrays(data, sel, day: date) -> dict:
    arrays = {'index_ns': data.index[sel].tz_convert('UTC').as_unit('ns').asi8.astype(np.int64),
              'dates': data.dates[sel].astype('int64'), 'hours': data.hours[sel].copy(), 'y': data.y[sel].copy(),
              'eligible': data.eligible[sel].astype(np.int8), 'feature_valid': data.feature_valid[sel].astype(np.int8),
              'level': data.level[sel].copy(), 'scale': data.scale[sel].copy(),
              'lear_raw': data.lear_raw[sel].copy(), 'lear_normalized': data.lear_normalized[sel].copy()}
    arrays['y'][data.dates[sel] == np.datetime64(day)] = np.nan
    return arrays


def _released(buffer: list[dict], day: date, what: str) -> None:
    if len(buffer) != 28 or any(date.fromisoformat(b['day']) > day - timedelta(days=2) for b in buffer):
        raise ValueError(f'{what}: the buffer is not 28 days released under the complete D-2 rule')


def v2_payload(ctx, day: date, buffer: list[dict]) -> tuple[dict, dict]:
    admit(day)
    _released(buffer, day, 'v2')
    data = ctx.data
    sel = _rows(data, day, ('A1', 'B2'))
    meta = {'schema': SCHEMAS['v2'], 'policy': 'v2', 'day': str(day), 'offset': int(sel[0]),
            'dates_dtype': str(data.dates.dtype), 'protocol': data.p, 'h_buffer': buffer,
            'embargo': "y on D's rows blanked; the H buffer holds released days <= D-2 only"}
    return _finish(_row_arrays(data, sel, day), meta)


def v4_payload(ctx, day: date, buffer: list[dict]) -> tuple[dict, dict]:
    admit(day)
    _released(buffer, day, 'v4')
    data = ctx.data_aug
    if data.lgbm_raw is not ctx.data.lgbm_raw or data.y is not ctx.data.y or data.scale is not ctx.data.scale:
        raise ValueError('the weather-augmented inputs differ from the nominal ones outside the LEAR design')
    sel = _rows(data, day, ('A1', 'B2', 'B3'))
    arrays = _row_arrays(data, sel, day)
    arrays.update({'lgbm_raw': data.lgbm_raw[sel].copy(), 'lgbm_normalized': data.lgbm_normalized[sel].copy(),
                   'wx': ctx.wx[sel].copy(), 'present': ctx.present[sel].astype(np.int8)})
    meta = {'schema': SCHEMAS['v4'], 'policy': 'v4', 'day': str(day), 'offset': int(sel[0]),
            'dates_dtype': str(data.dates.dtype), 'protocol': data.p, 'h_buffer': buffer,
            'design': 'LEAR rows carry the three weather columns appended (cp20.components.augment); LightGBM rows '
                      'carry CP-15\'s design, and the weather matrix with its presence',
            'embargo': "y on D's rows blanked; the H buffer holds released days <= D-2 only"}
    return _finish(arrays, meta)


def v5_payload(day: date, buffer: list[dict], v4_sha: str, a1_sha: str) -> tuple[dict, dict]:
    admit(day)
    _released(buffer, day, 'v5')
    meta = {'schema': SCHEMAS['v5'], 'policy': 'v5', 'day': str(day), 'h_buffer': buffer,
            'v4_payload_sha256': v4_sha, 'a1_payload_sha256': a1_sha,
            'composition': "v4's four members and A1's D2 central, blended by cp24.execution.v5_central"}
    return _finish({}, meta)


def b0_payload(ctx, day: date, buffer: list[dict]) -> tuple[dict, dict]:
    from delu_forecast.baselines import SIMILAR_DAY_LAG_BY_WEEKDAY
    admit(day)
    _released(buffer, day, 'B0')
    data = ctx.data
    first = day - timedelta(days=7)
    sel = np.flatnonzero((data.dates >= np.datetime64(first)) & (data.dates <= np.datetime64(day)))
    rows = data.rows(day)
    price = data.y[sel].copy()
    price[data.dates[sel] == np.datetime64(day)] = np.nan
    arrays = {'price_ns': data.index[sel].tz_convert('UTC').as_unit('ns').asi8.astype(np.int64), 'price': price,
              'key_ns': data.index[rows].tz_convert('UTC').as_unit('ns').asi8.astype(np.int64)}
    source = day - timedelta(days=SIMILAR_DAY_LAG_BY_WEEKDAY[day.weekday()])
    meta = {'schema': SCHEMAS['B0'], 'policy': 'B0', 'day': str(day), 'source_day': str(source), 'b0_buffer': buffer,
            'rule': 'similar-day naive: D-1 Tue-Fri, D-7 Mon/Sat/Sun',
            'embargo': "D's prices blanked; the raw-error buffer holds released days <= D-2 only"}
    return _finish(arrays, meta)


from ..dailyfit import save_npz  # noqa: E402  (one writer for every runtime)


def write(arrays: dict, meta: dict, folder: Path, stem: str) -> dict:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    js = folder / f'{stem}.json'
    out = {'meta': str(js), 'payload_sha256': meta['payload_sha256']}
    if arrays:
        npz = folder / f'{stem}.npz'
        save_npz(npz, arrays)
        out.update(npz=str(npz), npz_bytes=npz.stat().st_size)
    js.write_text(json.dumps(meta, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    out['meta_bytes'] = js.stat().st_size
    return out
