"""The live builder shown equal to the development builder, byte for byte (CP-18a item 2).

    PYTHONPATH=src python -m cp18.live.equality --archive ARCH --search-dir S --out OUT \\
        --report reports/cp18a/live-equality.json [--base-lag 8] [--workers 2] [--sets parity,shadow]

For each reference stem - A1, v4, v2 and B0 at CP-17b's nine parity origins
(`.local/artifacts/cp-17b/runs/native-parity/payloads/`) and on the 14 shadow days 2026-08-24..2026-09-06
(`.../shadow-host/payloads/`), all built by the development builder `cp17.tools.payloads` - `build_day` runs
with a base history cut `base_lag` days before D and the archive's increments for the days after it up to D
(rendered market responses, recorded GFS runs), with the reference meta's own buffers. The `.json` and the
`.npz` must be equal byte for byte, and so must `payload_sha256`. The report records both files' SHA-256 for
every stem, the base cut, the increment days and the timings.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

PROJECT = Path(os.environ.get('CP26_PROJECT_ROOT', '/Users/djourno/Downloads/PJM'))
REFERENCE = {'parity': PROJECT / '.local/artifacts/cp-17b/runs/native-parity/payloads',
             'shadow': PROJECT / '.local/artifacts/cp-17b/runs/shadow-host/payloads'}
SEARCH_DIR = PROJECT / '.local/artifacts/cp-17/inputs/reports/rebaseline/cold-rerun-2026-10-15/search-results/nominal'
POLICIES = ('A1', 'v4', 'v2', 'B0')
BUFFER_KEY = {'v4': 'h_buffer', 'v2': 'h_buffer', 'B0': 'b0_buffer'}


def parity_origins() -> list[date]:
    from cp17.tools.parity import ORIGINS
    return sorted(ORIGINS)


def shadow_days() -> list[date]:
    first = date(2026, 8, 24)
    return [first + timedelta(days=i) for i in range(14)]


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_buffers(folder: Path, day: date) -> dict:
    return {p: json.loads((folder / f'{p}-{day}.json').read_text())[k] for p, k in BUFFER_KEY.items()}


def compare(ref: Path, out: Path, policy: str, day: date) -> dict:
    stem = f'{policy}-{day}'
    item = {'stem': stem, 'policy': policy, 'day': day.isoformat()}
    for ext in ('json', 'npz'):
        a, b = ref / f'{stem}.{ext}', out / f'{stem}.{ext}'
        item[ext] = {'reference_sha256': sha_file(a) if a.exists() else None,
                     'live_sha256': sha_file(b) if b.exists() else None}
        item[ext]['equal'] = bool(a.exists() and b.exists() and a.read_bytes() == b.read_bytes())
    pr = json.loads((ref / f'{stem}.json').read_text())['payload_sha256']
    pl = json.loads((out / f'{stem}.json').read_text())['payload_sha256'] if (out / f'{stem}.json').exists() else None
    item['payload_sha256'] = {'reference': pr, 'live': pl, 'equal': pr == pl}
    item['equal'] = item['json']['equal'] and item['npz']['equal'] and item['payload_sha256']['equal']
    return item


def one(task: tuple) -> dict:
    set_name, day_text, archive, out_root, base_lag, search_dir, root = task
    from .build import build_day, write_record
    day = date.fromisoformat(day_text)
    ref = REFERENCE[set_name]
    out = Path(out_root) / set_name / day_text
    base_end = day - timedelta(days=base_lag)
    t0 = time.perf_counter()
    result = build_day(day, archive=archive, base_end=base_end, buffers=reference_buffers(ref, day),
                       search_dir=search_dir, out=out, root=root)
    write_record(result, out / f'live-build-{day}.json')
    stems = [compare(ref, out, p, day) for p in POLICIES] if result['status'] == 'built' else []
    a = result.get('availability') or {}
    return {'set': set_name, 'day': day_text, 'status': result['status'], 'cause': result['cause'],
            'base_end': base_end.isoformat(), 'increment_days': result['history'].get('increment_days'),
            'gfs_available': result['gfs_available'], 'shown': a.get('shown'),
            'by_exception': len(a.get('by_exception', [])), 'problems': result['problems'],
            'search': {k: (result['search'] or {}).get(k) for k in ('k', 'search_day', 'content_sha256')},
            'timings': result['timings'], 'wall_seconds': round(time.perf_counter() - t0, 2), 'stems': stems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m cp18.live.equality')
    ap.add_argument('--archive', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', required=True)
    ap.add_argument('--search-dir', default=str(SEARCH_DIR))
    ap.add_argument('--root', default='.')
    ap.add_argument('--base-lag', type=int, default=8)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--sets', default='parity,shadow')
    ap.add_argument('--days', default=None, help='comma-separated days to keep (default: every day of the sets)')
    args = ap.parse_args(argv)
    if not 1 <= args.workers <= 2:
        raise SystemExit('at most 2 workers')
    t0 = time.perf_counter()
    tasks = []
    for s in args.sets.split(','):
        days = parity_origins() if s == 'parity' else shadow_days()
        if args.days:
            days = [d for d in days if d.isoformat() in args.days.split(',')]
        tasks += [(s, d.isoformat(), str(Path(args.archive).resolve()), str(Path(args.out).resolve()), args.base_lag,
                   str(Path(args.search_dir)), str(Path(args.root).resolve())) for d in days]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, tasks))
    stems = [s for r in results for s in r['stems']]
    report = {'schema': 'cp18a-live-equality-v1',
              'what': ("cp18.live.build_day's payloads against the development builder's (cp17.tools.payloads) "
                       "reference files: .json and .npz byte for byte, and payload_sha256"),
              'reference': {k: str(v) for k, v in REFERENCE.items()}, 'search_dir': args.search_dir,
              'base_lag_days': args.base_lag, 'archive': args.archive,
              'inputs': ('base: data/snapshot.parquet and the committed weather features through D - base_lag; '
                         'increments D - base_lag + 1 .. D from the archive: SMARD responses rendered from the '
                         'snapshot (not recorded raw responses), GFS from CP-20/CP-25 recorded decoded runs'),
              'stems_expected': 4 * len(tasks), 'stems_compared': len(stems),
              'stems_equal': sum(s['equal'] for s in stems),
              'all_equal': len(stems) == 4 * len(tasks) and all(s['equal'] for s in stems),
              'days': results, 'workers': args.workers,
              'machine': {'platform': platform.platform(), 'python': sys.version.split()[0]},
              'wall_seconds': round(time.perf_counter() - t0, 1)}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, sort_keys=True, indent=1) + '\n')
    print(json.dumps({'all_equal': report['all_equal'], 'stems_equal': report['stems_equal'],
                      'stems_expected': report['stems_expected'], 'wall_seconds': report['wall_seconds']}))
    return 0 if report['all_equal'] else 1


if __name__ == '__main__':
    sys.exit(main())
