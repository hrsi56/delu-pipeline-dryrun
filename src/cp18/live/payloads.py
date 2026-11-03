"""The frozen payload kinds `cp17.daily` takes (A1, v4, v2, B0), built from a live history ending at D.

The construction is `cp17.tools.payloads`' (same arrays, same meta, `cp17.dailyfit.save_npz`, the same JSON
serialization), carried here so that its own `BOUNDARY` constant is not a second, hidden boundary: the only red
line on this path is `cp25.guard.admit_delivery_day`. The context is assembled from the landed building blocks
that `cp26.members.Context.build` composes, without `cp26.harness.load_frame`/`spec_for`/`NEW`, which bind the
development boundary:

| Step | Landed block |
|---|---|
| CP-15's prepared inputs | `cp15.data.prepare(frame, protocol, spec)`, the spec's cutoff being D itself |
| the weather design | `cp20.weather.WeatherDesign.from_features` |
| LEAR's weather columns, LightGBM's weather matrix | `cp20.components.augment`, `cp26.harness.weather_matrix` |
| DDNN-2's day table and keys | `cp24.design.build`, `cp24.member.Keys.from_data` |

`cp26.harness.install_issue_label`, which `Context.build` calls first, is not: it only rebinds the issue-time label
in the members' fit records, and it imports the native LightGBM, which a GitHub runner cannot load (no libomp).
Nothing here fits or writes a fit record.

**The one construction difference.** The development builder's frame holds D's price, so CP-15's `eligible`
on D's own rows is `feature_valid & isfinite(price)`. The live frame never holds D's price (the embargo), so
`prepare` would mark D's rows ineligible. The live builder sets `eligible` on D's rows to their `feature_valid`:
the development value whenever D's price is complete (every development day), and a value no frozen fit reads -
every fit takes D's rows by feature validity (`Inputs.rows`) and trains on delivery days before D only
(`cp15.models.fit_day`, `cp21.lgbm.model_rows`, `cp20.components.training_rows`, `cp17.dailyfit`). D's price,
targets and masks stay blanked in every payload, as before.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import json
from pathlib import Path
import time

import numpy as np

from cp17.dailyfit import SCHEMAS, array_hash, payload_digest, save_npz
from cp25 import guard

PROTOCOL = 'reports/cp15/protocol.json'


@dataclass(frozen=True)
class LiveSpec:
    """What `cp15.data.prepare` reads from a partition spec: the last admissible delivery day, here D."""
    eda_cutoff: date
    source: str = 'cp18.live: the frame ends at the day being built'


@dataclass
class LiveContext:
    day: date
    data: object
    data_aug: object
    design: object
    wx: np.ndarray
    present: np.ndarray
    dd: object
    keys: object
    timings: dict = field(default_factory=dict)


def protocol(root: Path) -> dict:
    """CP-15's recipe sections, verified at the pin CP-26 recorded (`cp26.harness.protocol_recipe`)."""
    from cp26.harness import protocol_recipe
    return protocol_recipe(Path(root))


def context(root: Path, frame, features, day: date) -> LiveContext:
    from cp15.data import prepare
    from cp20.components import augment
    from cp20.weather import WeatherDesign
    from cp24 import design as G
    from cp24.member import Keys
    from cp26.harness import HarnessError, check_blas, weather_matrix
    check_blas()
    guard.admit_delivery_day(day)
    t0 = time.perf_counter()
    # CP-26's `install_issue_label` is not called: it rebinds the issue-time label in the landed members' fit records
    # (cp15.models, cp21.lgbm, cp24.member), and importing them loads the native LightGBM, which a GitHub runner
    # cannot load (no libomp). Building a payload fits nothing and writes no fit record, so the bytes are the same
    # (tests/cp18a/test_live_runner_env.py builds with LightGBM blocked).
    data = prepare(frame, protocol(root), LiveSpec(day))
    rows = np.flatnonzero(data.dates == np.datetime64(day))
    if not len(rows) or data.dates.max() != np.datetime64(day):
        raise ValueError(f'{day}: the frame does not end at D')
    if np.isfinite(data.y[rows]).any():
        raise ValueError(f"{day}: the frame holds D's price, which the embargo forbids")
    data.eligible[rows] = data.feature_valid[rows]
    design = WeatherDesign.from_features(features)
    data_aug, present_l = augment(data, design)
    wx, present = weather_matrix(design, data)
    if not np.array_equal(present_l, present):
        raise HarnessError('weather presence differs between the LEAR and LightGBM designs')
    dd = G.build(data, design)
    keys = Keys.from_data(data, dd)
    return LiveContext(day, data, data_aug, design, wx, present, dd, keys,
                       {'context_seconds': time.perf_counter() - t0})


def load_ensemble(search_dir: Path, day: date) -> tuple[int, Path, dict | None]:
    """The serving search's state `k{NN}.json` (§24.7: S_k serves S_k .. S_(k+1) - 1); None if absent."""
    from cp26.search import serving_search, validate_ensemble
    k = serving_search(day)
    if k is None:
        raise ValueError(f'{day} precedes the first search')
    path = Path(search_dir) / f'k{k:02d}.json'
    if not path.exists():
        return k, path, None
    value = json.loads(path.read_text())
    validate_ensemble(value, k, 'nominal')
    return k, path, value


# ------------------------------------------------------------------ the payload kinds (cp17.tools.payloads)
def _finish(arrays: dict, meta: dict) -> tuple[dict, dict]:
    meta['array_sha256'] = {name: array_hash(arrays[name]) for name in sorted(arrays)}
    meta['payload_sha256'] = payload_digest(arrays, meta)
    return arrays, meta


def a1_payload(ctx: LiveContext, day: date, ens: dict) -> tuple[dict, dict]:
    from cp24 import design as G
    from cp26.search import serving_search, validate_ensemble
    guard.admit_delivery_day(day)
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
    from cp15.data import history_start     # the function cp15.models re-exports, without its LightGBM import
    lower = min(history_start(day, p) for p in policies)
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


def v2_payload(ctx: LiveContext, day: date, buffer: list[dict]) -> tuple[dict, dict]:
    guard.admit_delivery_day(day)
    _released(buffer, day, 'v2')
    data = ctx.data
    sel = _rows(data, day, ('A1', 'B2'))
    meta = {'schema': SCHEMAS['v2'], 'policy': 'v2', 'day': str(day), 'offset': int(sel[0]),
            'dates_dtype': str(data.dates.dtype), 'protocol': data.p, 'h_buffer': buffer,
            'embargo': "y on D's rows blanked; the H buffer holds released days <= D-2 only"}
    return _finish(_row_arrays(data, sel, day), meta)


def v4_payload(ctx: LiveContext, day: date, buffer: list[dict]) -> tuple[dict, dict]:
    guard.admit_delivery_day(day)
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


def b0_payload(ctx: LiveContext, day: date, buffer: list[dict]) -> tuple[dict, dict]:
    from delu_forecast.baselines import SIMILAR_DAY_LAG_BY_WEEKDAY
    guard.admit_delivery_day(day)
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


def write(arrays: dict, meta: dict, folder: Path, stem: str) -> dict:
    """`cp17.tools.payloads.write`: the npz through `cp17.dailyfit.save_npz`, the meta as sorted JSON."""
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
