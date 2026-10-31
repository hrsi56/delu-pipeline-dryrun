"""The day's entry point for CP-18's pipeline: `build_day`, and the published prices the release needs.

See the package docstring (`cp18.live`) for the contract. In order:

1. `cp25.guard.admit_delivery_day(D)` and the base end (`base_end < D`);
2. the archive's integrity for every archive day in (base_end, D] (`Archive.verify`);
3. the history: the committed base through `base_end`, the archive's increments to D (`history.assemble`),
   which judges every input against D's cutoff and sets each day's GFS flag;
4. the serving search's state `k{NN}.json` from `search_dir` (`cp26.search.serving_search(D)`), validated by
   `cp26.search.validate_ensemble`; absent, the result is `search_state_missing` and nothing is built;
5. if every required input is shown (or admitted under a recorded exception valid for that day), the context
   and the four payloads `A1`, `v4`, `v2`, `B0` as `<policy>-<D>.{npz,json}` in `out`, for `cp17.daily.issue_day`.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import time
import traceback

from cp25 import guard

from . import history as Hs
from . import payloads as P
from .archive import Archive

SCHEMA = 'cp18-live-build-v1'
POLICIES = ('A1', 'v4', 'v2', 'B0')
BUFFER_POLICIES = ('v4', 'v2', 'B0')


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _availability(history: Hs.History, day: date, integrity: list) -> dict:
    records = []
    for j in history.judgements:
        dd = date.fromisoformat(j['data_day'])
        required = (j['input'] == 'price' and dd <= day - timedelta(days=1)) or \
                   (j['input'] == 'load_forecast' and dd == day)
        records.append({**j, 'required': required})
    for d, g in sorted(history.gfs.items()):
        j = g.get('judgement') or {}
        records.append({'input': 'gfs', 'tier': 2, 'data_day': d, 'run': g['run'], 'for_day': day.isoformat(),
                        'cutoff_utc': j.get('cutoff_utc'), 'record': j.get('record'), 'sha256': j.get('sha256'),
                        'provenance': j.get('provenance'), 'publication_utc': j.get('publication_utc'),
                        'basis_id': j.get('basis_id'), 'basis': j.get('basis'), 'admitted': g['available'],
                        'by_exception': False, 'vector_valid': g.get('vector_valid'), 'status': g['status'],
                        'reason': g['reason'], 'required': False})
    reasons = list(history.missing_required) + [f'archive integrity: {p}' for p in integrity]
    return {'shown': not reasons, 'reasons': reasons,
            'by_exception': [{'input': r['input'], 'data_day': r['data_day'], 'record': r.get('record'),
                              'basis': r.get('basis')} for r in records if r.get('admitted') and r.get('by_exception')],
            'records': records,
            'base': {'through': history.base_end.isoformat(), 'snapshot': Hs.SNAPSHOT,
                     'weather': [Hs.CP20_FEATURES, Hs.GAP_FEATURES],
                     'load_forecast_basis': ("the committed development history; its tier-3 load forecast is admitted "
                                             "under ruling 1's recorded exception (development data only)"),
                     'ruling_4': [d.isoformat() for d in Hs.late_gfs_delivery() if d <= history.base_end]}}


def build_day(day: date, *, archive, base_end: date, buffers: dict, search_dir, out, root) -> dict:
    """Build D's four payloads from the archive. Never raises for a day's own failure: `status` says what
    happened (`built`, `search_state_missing`, `not_shown`, `failed`), with the cause."""
    t0 = time.perf_counter()
    guard.admit_delivery_day(day)
    guard.admit_delivery_day(base_end)
    archive = archive if isinstance(archive, Archive) else Archive(Path(archive))
    root, out = Path(root), Path(out)
    result = {'schema': SCHEMA, 'day': day.isoformat(), 'status': None, 'cause': None, 'payloads': {},
              'gfs_available': None, 'gfs': None, 'availability': None, 'search': None,
              'history': {'base_end': base_end.isoformat()}, 'problems': [], 'timings': {}}
    try:
        if not base_end < day:
            raise ValueError('base_end must precede the day being built')
        integrity = []
        for x in archive.days():
            if base_end < x <= day:
                integrity += archive.verify(x)
        history = Hs.assemble(root, archive, day, base_end)
        result['timings']['history_seconds'] = round(time.perf_counter() - t0, 3)
        result['history'].update(increment_days=[d.isoformat() for d in history.increment_days],
                                 frame_rows=int(len(history.frame)), weather_rows=int(len(history.features)))
        result['problems'] += history.problems
        gfs = history.gfs[day.isoformat()]
        result['gfs'] = gfs
        result['gfs_available'] = bool(gfs['available'])
        result['availability'] = _availability(history, day, integrity)
        k, path, ens = P.load_ensemble(Path(search_dir), day)
        result['search'] = {'k': k, 'path': str(path), 'present': ens is not None,
                            'search_day': ens['search_day'] if ens else None,
                            'content_sha256': ens['content_sha256'] if ens else None,
                            'file_sha256': _sha_file(path) if ens else None}
        if ens is None:
            result.update(status='search_state_missing',
                          cause=f"the serving search's state k{k:02d}.json is not in place ({path}); no payload built")
            return result
        if not result['availability']['shown']:
            result.update(status='not_shown',
                          cause="a live day's inputs that cannot be shown to precede its cutoff: "
                                + '; '.join(result['availability']['reasons']))
            return result
        missing = [p for p in BUFFER_POLICIES if p not in buffers]
        if missing:
            raise ValueError(f'no released buffer for {missing}')
        t1 = time.perf_counter()
        ctx = P.context(root, history.frame, history.features, day)
        result['timings']['context_seconds'] = round(time.perf_counter() - t1, 3)
        built = {'A1': P.a1_payload(ctx, day, ens), 'v4': P.v4_payload(ctx, day, buffers['v4']),
                 'v2': P.v2_payload(ctx, day, buffers['v2']), 'B0': P.b0_payload(ctx, day, buffers['B0'])}
        for policy in POLICIES:
            arrays, meta = built[policy]
            result['payloads'][policy] = P.write(arrays, meta, out, f'{policy}-{day}')
        result['status'] = 'built'
    except Exception as exc:  # noqa: BLE001 - a day's failure is a result, with its cause
        result.update(status='failed', cause=guard.redact(f'{type(exc).__name__}: {exc}'),
                      traceback=guard.redact(traceback.format_exc(limit=6)))
    finally:
        result['timings']['total_seconds'] = round(time.perf_counter() - t0, 3)
    return result


def published_prices(day: date, *, archive, base_end: date, root):
    """`truth(keys) -> array` for `cp17.shadow.ShadowStates.release(D, truth)`: published prices of delivery
    days <= D-1 only (the base and the archive's admitted increments), NaN unless every key is present."""
    archive = archive if isinstance(archive, Archive) else Archive(Path(archive))
    history = Hs.assemble(Path(root), archive, day, base_end)
    return history.truth()


def write_record(result: dict, path: Path) -> None:
    """The build result as JSON (the pipeline's record of the day's inputs and their availability)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, sort_keys=True, indent=1, default=str) + '\n')
