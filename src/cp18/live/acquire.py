"""Live acquisition (CP-18c): every input a live archive day needs, retrieved through the live transport and archived.

An archive day D (`live/inputs/days/<D>/`) holds, as CP-18a's rehearsal laid it out:
- the day-ahead **prices** of D−3..D−1 (SMARD filter 4169; tier 2, the documented schedule);
- the day-ahead **load forecast** of D (SMARD filter 411; tier 3: only a retrieval, or the document's own creation time,
  strictly before D's cutoff shows it). **This is the one time-critical input**: it must be retrieved and archived
  before D's cutoff, D−1 11:00 Europe/Berlin, or D cannot be issued (§24.13, "The publication time");
- the **GFS run** that serves D (D−1 00 UTC, f021–f048 from the AWS archive; its `LastModified` listing is the
  vintage-preserving evidence of ruling 4, so a later retrieval still shows when it was published).

Each acquisition writes records whose ids carry a tag (`@<retrieval time>`), so the archive stays append-only when a
day is acquired in parts (the load forecast before the cutoff, the rest afterwards) or retried. Which input a day
still lacks is read from its manifest (`missing`), never from whether the day's folder exists.

Everything here runs inside `cp18.boundary.live()`: the development guard is retired for the live pipeline only
(r20 §24.14 item 2). Credentials: SMARD and the AWS archive are anonymous; nothing here reads a credential.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import traceback

from cp25 import guard

from .. import boundary
from . import archive as AR
from . import availability as A
from . import entsoe, smard

UTC = timezone.utc
INPUTS = ('price', 'load_forecast', 'gfs')
PRICE_LAGS = (3, 2, 1)
# The live sources. The load forecast is ENTSO-E's A65 (process A01), the plan's own series (`data/README.md`): TSOs
# publish it by D-1 10:00 local (Reg. 543/2013 Art. 6), while SMARD republishes it only in the afternoon of D-1
# (its bucket's own creation time), after D's cutoff, so SMARD can never show it in time. Prices are SMARD's,
# the snapshot's route (tier 2: the documented schedule admits a later retrieval).
SOURCES = {'price': 'smard', 'load_forecast': 'entsoe'}
CLIENTS = {'smard': smard, 'entsoe': entsoe}


def tag_now() -> str:
    return datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')


def price_days(day: date) -> list[date]:
    return [day - timedelta(days=k) for k in PRICE_LAGS]


def _complete(archive: AR.Archive, day: date, record: dict, data_day: str) -> bool:
    """Whether an archived market record holds every hour of `data_day` (a "no data yet" answer does not)."""
    if record.get('status') != 200 or not record.get('body') or data_day not in record.get('covers', []):
        return False
    try:
        d = date.fromisoformat(data_day)
        body = archive.body(day, record)
        values = (smard.values_from_body(record['input'], record['resolution'], body, [d]) if record['source'] == 'smard'
                  else entsoe.values_from_body(record['input'], body, [d]))[d]
    except Exception:  # noqa: BLE001 - an unreadable record is not complete
        return False
    return bool(len(values)) and not bool((values != values).any())


def missing(archive: AR.Archive, day: date) -> list[str]:
    """Which of the day's inputs have no complete record yet."""
    if not archive.has(day):
        return list(INPUTS)
    recs = archive.records(day)
    out = []
    prices = [r for r in recs if r['input'] == 'price']
    if not all(any(_complete(archive, day, r, d.isoformat()) for r in prices) for d in price_days(day)):
        out.append('price')
    if not any(_complete(archive, day, r, day.isoformat()) for r in recs if r['input'] == 'load_forecast'):
        out.append('load_forecast')
    if not any(r['input'] == 'gfs_run' for r in recs):
        out.append('gfs')
    return out


def load_forecast_shown(archive: AR.Archive, day: date) -> dict:
    """Whether D's own load forecast is archived with a publication strictly before D's cutoff, and complete."""
    best = None
    for r in archive.records(day, 'load_forecast') if archive.has(day) else []:
        if r.get('status') != 200 or not r.get('body') or day.isoformat() not in r.get('covers', []):
            continue
        j = A.judge('load_forecast', day, day, r['publication'].get(day.isoformat(), {}), r, archive_day=day)
        body = archive.body(day, r)
        values = (smard.values_from_body('load_forecast', r['resolution'], body, [day]) if r['source'] == 'smard'
                  else entsoe.values_from_body('load_forecast', body, [day]))[day]
        complete = bool(len(values)) and not bool((values != values).any())
        item = {'record': r['id'], 'source': r['source'], 'admitted': j['admitted'], 'complete': complete, 'reason': j['reason'],
                'publication_utc': j['publication_utc'], 'basis_id': j['basis_id']}
        if j['admitted'] and complete:
            return {**item, 'shown': True}
        best = best or item
    return {**(best or {'reason': 'no archived load forecast of the day'}), 'shown': False}


def acquire_market(archive: AR.Archive, day: date, transport, input_name: str, tag: str | None = None,
                   source: str | None = None) -> dict:
    """One market input of archive day D from its live source, archived with its retrieval and publication evidence."""
    # The tag is the retrieval time and the day's record count, so every acquisition's ids are new (append-only).
    tag = tag or f"{tag_now()}n{len(archive.records(day)) if archive.has(day) else 0:04d}"
    source = source or SOURCES[input_name]
    days = price_days(day) if input_name == 'price' else [day]
    fetched = CLIENTS[source].fetch(transport, input_name, days)
    stored = AR.store_market(archive, day, fetched, tag=tag)
    complete = all(d in fetched.values and not bool((fetched.values[d] != fetched.values[d]).any()) for d in days)
    return {'input': input_name, 'source': source, 'day': day.isoformat(), 'data_days': [d.isoformat() for d in days],
            'records': [r['id'] for r in stored], 'complete': complete, 'problems': fetched.problems,
            'bytes': sum((r.get('bytes') or 0) for r in stored)}


def acquire_gfs(archive: AR.Archive, day: date, transport, raw_dir: Path | None = None) -> dict:
    """The D−1 00 UTC run serving D: listing, index, byte ranges, decoded by CP-20's decoder (ecCodes)."""
    from . import gfs
    run = day - timedelta(days=1)
    with tempfile.TemporaryDirectory(prefix=f'cp18c-gfs-{run}-') as tmp:
        raw = Path(raw_dir) if raw_dir else Path(tmp) / 'raw'
        raw.mkdir(parents=True, exist_ok=True)
        retrieval = gfs.retrieve_run(transport, run, raw)
        if retrieval.problems and not retrieval.messages:
            return {'input': 'gfs', 'day': day.isoformat(), 'run': run.isoformat(), 'stored': False,
                    'problems': retrieval.problems}
        decoded = gfs.decode_run(run, day, retrieval.messages, raw, Path(tmp) / 'decoded')
        record = AR.store_gfs_retrieval(archive, day, retrieval, decoded)
    return {'input': 'gfs', 'day': day.isoformat(), 'run': run.isoformat(), 'stored': True,
            'record': record['id'], 'publication': record['publication'], 'problems': retrieval.problems,
            'checks': record.get('checks')}


def acquire_day(archive: AR.Archive, day: date, transport, inputs=INPUTS, *, raw_dir: Path | None = None) -> dict:
    """Whatever of `inputs` the day still lacks. A failure is recorded as the item's result, never raised."""
    result = {'day': day.isoformat(), 'items': [], 'started_utc': guard_now()}
    boundary.admit_delivery_day(day)
    need = missing(archive, day)
    for name in inputs:
        if name not in need:
            result['items'].append({'input': name, 'skipped': 'already archived'})
            continue
        try:
            if name == 'gfs':
                result['items'].append(acquire_gfs(archive, day, transport, raw_dir))
            else:
                result['items'].append(acquire_market(archive, day, transport, name))
        except Exception as exc:  # noqa: BLE001 - N5: an acquisition failure is a recorded result
            result['items'].append({'input': name, 'error': guard.redact(f'{type(exc).__name__}: {exc}'),
                                    'traceback': guard.redact(traceback.format_exc(limit=4))})
    result['missing_after'] = missing(archive, day)
    result['verify'] = archive.verify(day) if archive.has(day) else []
    result['finished_utc'] = guard_now()
    return result


def guard_now() -> str:
    return datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')


def main(argv=None) -> int:
    import argparse
    from .transport import HttpTransport
    ap = argparse.ArgumentParser(prog='python -m cp18.live.acquire')
    ap.add_argument('--archive', type=Path, required=True, help='the live input archive (live/inputs)')
    ap.add_argument('--from', dest='first', required=True)
    ap.add_argument('--to', dest='last', required=True)
    ap.add_argument('--inputs', default=','.join(INPUTS))
    ap.add_argument('--out', type=Path, required=True, help='the acquisition record (JSON lines, appended)')
    args = ap.parse_args(argv)
    first, last = date.fromisoformat(args.first), date.fromisoformat(args.last)
    inputs = tuple(x for x in args.inputs.split(',') if x)
    worst = 0
    with boundary.live():
        archive = AR.Archive(args.archive)
        transport = HttpTransport()
        d = first
        while d <= last:
            res = acquire_day(archive, d, transport, inputs)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, 'a') as f:
                f.write(json.dumps(res, sort_keys=True, default=str) + '\n')
            errors = [i for i in res['items'] if i.get('error')]
            print(d, 'missing after:', res['missing_after'] or 'none', '| errors:', len(errors),
                  '| verify problems:', len(res['verify']), flush=True)
            worst = max(worst, 1 if (errors or res['verify'] or res['missing_after']) else 0)
            d += timedelta(days=1)
    return worst


if __name__ == '__main__':
    raise SystemExit(main())
