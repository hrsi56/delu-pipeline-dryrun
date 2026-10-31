"""Guarded CP-25 GFS extraction (extraction environment: CP-20's pinned ecCodes venv).

Usage: python -m cp25.extract --manifest M --out DIR --admission reports/weather-admission
                              --ledger L [--workers N] [--stop-transfer-gib G] [--select all|R1,R2]

CP-20's extraction code is reused by import and runs unchanged (capstone v21-r13 §24.6 item 3:
§15.2's source, recipe, fields and leads): `cp20.extract.Extractor` with its per-message attempt
ledger, AWS `.idx` byte ranges, NCAR adaptive locator for the one alternate attempt, GRIB2 framing,
local ecCodes decoding and field-by-field metadata validation. CP-25 adds only:

* **the boundary guard** (§24.2): every request goes through `GuardedFetcher`, which admits a URL
  only after `cp25.guard.admit_url` has parsed it into an admitted (run, lead) on one of the two
  anonymous archives, and redacts any error text before it is raised;
* **targeted checks** (§15.2) for these runs outside the inventoried scope: one exact AWS listing
  per (run, lead), recording each GRIB object's and index's size and `LastModified` (the per-run
  availability evidence); the decoded field metadata is validated by CP-20's `decode`;
* **CP-25's ceilings** (`cp25.ledger`): transfer reserved before each request, machine time charged
  as workers x wall time every 30 s, at most 4 workers.

Exit codes follow CP-20: 0 complete; 2 stopped/partial; 4 contradiction with admission;
5 runs failed after the bounded attempts; 6 no data for 60 minutes; 7 a cap would be crossed.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
from types import SimpleNamespace

from cp20.budget import CapExceeded, GIB, atomic
from cp20.extract import NO_DATA_SECONDS, PER_RUN_CONCURRENCY, Extractor
from cp20.gfs import LEADS, NcarLocator
from cp20.net import Fetcher, TransferError

from . import guard
from .ledger import Ledger

LISTING_BOUND = 20_000
CHARGE_EVERY_S = 30.0


class _RedactingFile:
    def __init__(self, fh):
        self.fh = fh

    def write(self, text):
        return self.fh.write(guard.redact(text))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fh.close()
        return False


class RedactingPath:
    """The request log's path, as CP-20's `Fetcher` uses it (`.parent`, `.open('a')`): every line is
    passed through `guard.redact` before it reaches the file, so no credential value can be logged."""

    def __init__(self, path):
        self.path = Path(path)
        self.parent = self.path.parent

    def open(self, mode='r', *args, **kwargs):
        fh = self.path.open(mode, *args, **kwargs)
        return _RedactingFile(fh) if any(m in mode for m in 'aw') else fh

    def __fspath__(self):
        return str(self.path)


class GuardedFetcher(Fetcher):
    """CP-20's metered fetcher behind the §24.2 boundary guard and the credential rule."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_path = RedactingPath(self.log_path)

    def get(self, url, purpose, **kwargs):
        guard.admit_url(url)
        try:
            return super().get(url, guard.redact(purpose), **kwargs)
        except TransferError as exc:
            raise type(exc)(guard.redact(str(exc))) from None


def listing_url(run: str, lead: int) -> str:
    ymd = run.replace('-', '')
    prefix = f'gfs.{ymd}/00/atmos/gfs.t00z.pgrb2.0p25.f{lead:03d}'
    return f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/?list-type=2&prefix={prefix}'


def parse_listing(body: bytes) -> dict:
    text = body.decode('utf8', 'replace')
    out = {}
    for block in re.findall(r'<Contents>(.*?)</Contents>', text, re.S):
        key = re.search(r'<Key>([^<]+)</Key>', block).group(1)
        out[key] = {'size': int(re.search(r'<Size>(\d+)</Size>', block).group(1)),
                    'last_modified': re.search(r'<LastModified>([^<]+)</LastModified>', block).group(1),
                    'etag': (re.search(r'<ETag>([^<]+)</ETag>', block) or [None, None])[1]}
    truncated = '<IsTruncated>true</IsTruncated>' in text
    return {'objects': out, 'truncated': truncated}


def targeted_listing(ex: Extractor, rec: dict) -> dict:
    """One exact AWS listing per (run, lead): presence, sizes and LastModified of object and index."""
    path = ex.out / 'listing.jsonl'
    run = rec['run_00z']
    entry = {'run': run, 'delivery_day': rec['delivery_day'], 'leads': {}}
    ymd = run.replace('-', '')
    for lead in LEADS:
        _, _, body = ex.fetcher.get(listing_url(run, lead), f'aws listing {run} f{lead:03d}', max_body=LISTING_BOUND)
        parsed = parse_listing(body)
        stem = f'gfs.{ymd}/00/atmos/gfs.t00z.pgrb2.0p25.f{lead:03d}'
        obj, idx = parsed['objects'].get(stem), parsed['objects'].get(stem + '.idx')
        entry['leads'][f'f{lead:03d}'] = {'object': obj, 'index': idx, 'listed_keys': sorted(parsed['objects']),
                                          'truncated': parsed['truncated']}
        if obj:
            rec['object_bytes'][f'f{lead:03d}']['aws'] = obj['size']
    entry['aws_complete'] = all(v['object'] and v['index'] for v in entry['leads'].values())
    entry['listed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with ex.lock:
        with path.open('a') as fh:
            fh.write(json.dumps(entry, sort_keys=True) + '\n')
    return entry


def listed(out: Path) -> dict:
    path = out / 'listing.jsonl'
    if not path.exists():
        return {}
    return {json.loads(l)['run']: json.loads(l) for l in path.read_text().splitlines()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--admission', required=True)
    ap.add_argument('--ledger', required=True)
    ap.add_argument('--select', default='all')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--stop-transfer-gib', type=float, default=38.0)
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= 4:
        raise SystemExit('workers must be 1..4')
    if args.stop_transfer_gib > 40:
        raise SystemExit('the stop line cannot exceed the 40 GiB ceiling')
    ledger = Ledger(args.ledger)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'extract.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another extraction instance holds this output directory')
    manifest = json.loads(Path(args.manifest).read_text())
    runs = manifest['runs']
    if args.select != 'all':
        wanted = set(args.select.split(','))
        runs = [r for r in runs if r['run_00z'] in wanted]
    for r in runs:  # the guard again, at job start, on every run and lead the job may touch
        for lead in LEADS:
            guard.admit_gfs(guard.run_init(guard.date.fromisoformat(r['run_00z'])), lead)
    runs = sorted(runs, key=lambda r: r['run_00z'])
    job = ledger.begin_job('gfs-extract', args.workers, ' '.join(sys.argv))
    ns = SimpleNamespace(out=str(out), stop_transfer_gib=args.stop_transfer_gib, admission=args.admission)
    ex = Extractor(ns, ledger, None)
    ex.fetcher = GuardedFetcher(ledger, out / 'requests.jsonl', int(args.stop_transfer_gib * GIB), stop_event=ex.stop)
    ex.locator = NcarLocator(ex.fetcher, ex.model)
    with (out / 'job-code-versions.jsonl').open('a') as fh:
        fh.write(json.dumps({'job_index': job, 'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                             'workers': args.workers, 'select': args.select if len(args.select) < 200 else f'{len(runs)} runs',
                             'guard': 'cp25.guard.admit_url via GuardedFetcher'}, sort_keys=True) + '\n')

    def on_signal(signum, _frame):
        ex.reason = ex.reason or f'signal_{signum}'
        if ex.drain.is_set():
            ex.stop.set()
        ex.drain.set()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    finished = threading.Event()

    def meter():
        while not finished.wait(CHARGE_EVERY_S):
            try:
                used = ledger.charge_job(job)
            except Exception as exc:  # noqa: BLE001
                ex._halt('ledger_error', {'error': guard.redact(repr(exc))})
                return
            if used >= ledger.read()['caps']['machine_seconds'] * 0.98:
                ex._halt('cap_or_stop_line', {'error': 'machine-hour stop line (98% of 24 h)'})
                return
            idle = time.time() - ex.fetcher.last_data
            if idle > NO_DATA_SECONDS and not ex.stop.is_set():
                ex._halt('no_data_60_minutes', {'seconds_without_data': round(idle)})
                return
    threading.Thread(target=meter, daemon=True).start()

    done_listing = listed(out)
    queue = deque(runs)
    qlock = threading.Lock()

    def lane(_):
        pool = ThreadPoolExecutor(max_workers=PER_RUN_CONCURRENCY['aws'])
        while not ex.stop.is_set() and not ex.drain.is_set():
            with qlock:
                if not queue:
                    break
                rec = queue.popleft()
            try:
                if ex.completed(rec['run_00z']):
                    with ex.lock:
                        ex.progress['reused'] += 1
                    continue
                entry = done_listing.get(rec['run_00z']) or targeted_listing(ex, rec)
                for lead, v in entry['leads'].items():
                    if v['object']:
                        rec['object_bytes'][lead]['aws'] = v['object']['size']
                ex.run(rec, pool)
            except CapExceeded as exc:
                ex._halt('cap_or_stop_line', {'run': rec['run_00z'], 'error': guard.redact(str(exc))})
            except Exception as exc:  # noqa: BLE001 - an unexpected error (including a guard refusal) halts
                ex._halt('worker_error', {'run': rec['run_00z'], 'error': guard.redact(repr(exc))})
        pool.shutdown(wait=True)

    print(f'cp25 extraction start: {len(runs)} runs, workers={args.workers}', flush=True)
    threads = [threading.Thread(target=lane, args=(k,), daemon=True) for k in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    finished.set()
    ex._save_progress()
    done = [r['run_00z'] for r in runs if ex.completed(r['run_00z'])]
    summary = {'selected': len(runs), 'complete': len(done), 'failed': sorted(set(ex.progress['failed'])),
               'halt_reason': ex.reason, 'requests_this_invocation': ex.fetcher.stats,
               'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'drained': ex.drain.is_set() and not ex.stop.is_set()}
    code = (4 if ex.reason == 'contradiction' else 6 if ex.reason == 'no_data_60_minutes'
            else 7 if ex.reason == 'cap_or_stop_line' else 2 if ex.reason is not None
            else 5 if summary['failed'] else 0 if len(done) == len(runs) else 2)
    ledger.charge_job(job, final=True, exit_code=code)
    atomic(out / f'summary-{job}.json', summary)
    print(json.dumps(summary), flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
