"""The pipeline's commands, the same on the runner and on the Owner's Mac (`cp18.local` wraps them in one command).

    PYTHONPATH=src python -m cp18 prepare --root . --project . --pinned <folder> --retrieve-pyodide --retrieve-node \
        --out <file>
    PYTHONPATH=src python -m cp18 seed-store --root . --genesis <state.json>:<first day>[:<origin>] | --carried <folder>
    PYTHONPATH=src python -m cp18 run --root . --project . --out <dir> --host runner --now <ISO> \
        --inputs prebuilt:<payload folder> | live:<recorded responses> [--remote origin]
    PYTHONPATH=src python -m cp18 carry --root . --last-window-day <day> --first-live-day <day> --reason <text>
    PYTHONPATH=src python -m cp18 search --root . --k 14 --file <k14.json> --provenance <text>
    PYTHONPATH=src python -m cp18 status --root .

**Two modes.** `--mode dry-run` (the default): development days through 2026-09-06, with `--now` required, so the plan
is computed from a simulated wall clock; the development guard refuses any plan past the boundary. `--mode live`
(CP-18c, r20 §24.14): the real clock, `--inputs live-http`, inside `cp18.boundary.live()`, where the guard is retired
for the live pipeline only and each day is judged against its own cutoff. `acquire` runs before the cutoff and
archives the next delivery day's inputs, its tier-3 load forecast above all:

    PYTHONPATH=src python -m cp18 acquire --root . --out <file> [--day <D>] [--remote origin]

The run's exit code: 0 done (or nothing to do, or a lost claim, which is a correct outcome); 1 a day was not issued;
2 refused before issuing, or the publication was refused; 3 the red line.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

BLAS = ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=str) + '\n')


def _check_blas() -> None:
    import os
    if any(os.environ.get(n) != '1' for n in BLAS):
        raise SystemExit('export the five *_NUM_THREADS=1 before NumPy loads, as the freeze ran')


def make_inputs(spec: str, root: Path, project: Path):
    kind, _, where = spec.partition(':')
    if kind == 'prebuilt':
        from .inputs import PrebuiltInputs
        return PrebuiltInputs(root, project, Path(where))
    if kind == 'live':
        from .live_inputs import LiveInputs
        return LiveInputs(root, project, Path(where))
    if kind == 'live-http':
        from .live.transport import HttpTransport
        from .live_inputs import LiveInputs
        return LiveInputs(root, project, None, transport=HttpTransport())
    raise SystemExit(f'--inputs {spec!r}: expected prebuilt:<folder>, live:<recorded responses> or live-http')


def pin_project(root: Path, project: Path) -> None:
    """The H2 pins check (`refreeze verify`) finds the issuing host's local files in the project's `.local/`, which it
    resolves from `$CP17_PROJECT_ROOT` or else from the checkout's own git directory. A pipeline checkout that is not the
    project (the local fallback's dedicated clone) names the project explicitly; the runner's workflow sets it already.
    Without it the first live run on the Owner's Mac failed its pins check (CP-18c's disclosure)."""
    import os
    if root != project:
        os.environ.setdefault('CP17_PROJECT_ROOT', str(project))


def _mode(args):
    """The run's mode and wall clock: a dry run's simulated clock on development days, or a live run's real one
    inside the live boundary (`cp18.boundary`, r20 §24.14 item 2)."""
    from contextlib import nullcontext
    from datetime import datetime, timezone
    from . import boundary, plan
    if args.mode == 'live':
        if args.now:
            raise SystemExit('a live run reads the real clock; --now is for dry runs only')
        if not args.inputs.startswith('live-http'):
            raise SystemExit('a live run takes --inputs live-http')
        return datetime.now(timezone.utc), boundary.live()
    if args.mode != 'dry-run':
        raise SystemExit(f'--mode {args.mode!r}: expected dry-run or live')
    if not args.now:
        raise SystemExit('a dry run needs --now, its simulated wall clock')
    if args.inputs.startswith('live-http'):
        raise SystemExit('a dry run never retrieves: --inputs live-http is for --mode live')
    return plan.parse_now(args.now), nullcontext()


def cmd_prepare(args) -> int:
    from . import host
    root, project = args.root.resolve(), args.project.resolve()
    record = {'materialized': host.materialize(root, project, args.pinned) if args.pinned else None,
              'pyodide': host.pyodide(root, project, args.retrieve_pyodide),
              'node': host.node(project, args.retrieve_node), 'package': host.package(root, project)}
    _dump(args.out, record)
    print('prepare:', 'pinned states', len(record['materialized'] or {}), '| pyodide', len(record['pyodide']), '| node',
          (record['node'].get('binary') or {}).get('sha256', 'in place')[:16], '| package',
          'equal' if record['package']['equal_to_manifest'] else 'NOT EQUAL', flush=True)
    return 0 if record['package']['equal_to_manifest'] else 1


def cmd_run(args) -> int:
    _check_blas()
    from . import pipeline
    from .claim import Claim
    root, project, out = args.root.resolve(), args.project.resolve(), args.out.resolve()
    pin_project(root, project)
    now, scope = _mode(args)
    settings = pipeline.Settings(root=root, project=project, out=out, host_kind=args.host, mode=args.mode,
                                 run_label=args.run_label or '')
    with scope:
        record = pipeline.run(settings, now, claim=Claim(root, remote=args.remote, branch=args.branch),
                              inputs=make_inputs(args.inputs, root, project))
    _dump(out / 'run.json', record)
    print('run', record['run_id'], '|', record['outcome'], '|', 'planned', record['planned'], flush=True)
    for day, item in record['days'].items():
        print(' ', day, item.get('status'), item.get('policy') or item.get('cause_text') or '',
              f"{item.get('seconds', 0):.0f}s", flush=True)
    for item in record['closed']:
        print('  closed', item['day'], 'not issued:', item['cause_text'], flush=True)
    return record['exit']


def cmd_acquire(args) -> int:
    """Before the cutoff: the next delivery day's inputs, its load forecast above all (tier 3, shown only by a
    retrieval before D's cutoff), retrieved, archived and published through the claim (`cp18.acquire`)."""
    from . import acquire
    from .claim import Claim
    root = args.root.resolve()
    record = acquire.run(root, claim=Claim(root, remote=args.remote, branch=args.branch),
                         day=args.day, poll_seconds=args.poll_seconds, margin_seconds=args.margin_seconds,
                         window_hours=args.window_hours if args.window_hours > 0 else None)
    _dump(args.out, record)
    print('acquire', record.get('day'), '|', record['outcome'], flush=True)
    return record['exit']


def cmd_seed_store(args) -> int:
    """A dry run's store in this checkout: a new one from a development state, or one an earlier run carried out."""
    import shutil
    from cp17.shadow import ShadowStates
    from .store import STORE_PATH, LiveStore
    folder = args.root / STORE_PATH
    if args.genesis:
        state_file, *days = args.genesis.split(':')
        state = ShadowStates.from_dict(json.loads(Path(state_file).read_text()))
        first_day = date.fromisoformat(days[0])
        origin = date.fromisoformat(days[1]) if len(days) > 1 else first_day
        store = LiveStore.init(folder, state, origin, args.source or f'state {Path(state_file).name}', first_day)
    else:
        if folder.exists():
            raise SystemExit(f'{folder}: a store exists there; a carried store is never copied over one')
        shutil.copytree(args.carried, folder)
        store = LiveStore(folder)
    head = store.verify()
    print('store', STORE_PATH, 'head', head['entry_sha256'][:16], 'last day', head['last_day'])
    return 0


def cmd_carry(args) -> int:
    """The carry from 4.7T's last window day to the first live day: the days between are recorded as outside both."""
    from .store import STORE_PATH, LiveStore
    store = LiveStore(args.root / STORE_PATH)
    store.verify()
    entry = store.record_carry(date.fromisoformat(args.last_window_day), date.fromisoformat(args.first_live_day),
                               args.reason)
    print('carry', entry['last_window_day'], '->', entry['first_live_day'], 'head', store.verify()['entry_sha256'][:16])
    return 0


def cmd_search(args) -> int:
    """S_14 (H4): a search state written natively on the Owner's M3 enters the store, bound by hash."""
    from .store import STORE_PATH, LiveStore
    store = LiveStore(args.root / STORE_PATH)
    store.verify()
    entry = store.record_search(args.k, args.file, args.provenance)
    print('search', entry['k'], entry['search_file'], entry['search_file_sha256'][:16], 'head', store.verify()['entry_sha256'][:16])
    return 0


def cmd_status(args) -> int:
    from .store import STORE_PATH, LiveStore
    store = LiveStore(args.root / STORE_PATH)
    head = store.verify()
    print(json.dumps({'head': head, 'days': store.days(), 'runs': len(store.runs())}, indent=1, sort_keys=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m cp18')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--root', default=Path('.'), type=Path)
    p.add_argument('--project', default=Path('.'), type=Path)
    p.add_argument('--pinned', help='the committed folder of pinned states to copy into .local/ by hash '
                                    '(live/pinned in the product; reports/cp18a/dryrun/pinned in a dry run)')
    p.add_argument('--retrieve-pyodide', action='store_true', help="retrieve Pyodide's 23 files by hash (the runner)")
    p.add_argument('--retrieve-node', action='store_true', help='retrieve the official binary by hash (the runner)')
    p.add_argument('--out', required=True, type=Path)
    r = sub.add_parser('run')
    r.add_argument('--root', default=Path('.'), type=Path)
    r.add_argument('--project', default=Path('.'), type=Path)
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--host', choices=('runner', 'local'), required=True)
    r.add_argument('--mode', default='dry-run', choices=('dry-run', 'live'))
    r.add_argument('--now', help='the simulated wall clock of a dry run, with its offset (never in a live run)')
    r.add_argument('--inputs', required=True, help='prebuilt:<folder> | live:<recorded responses> | live-http')
    r.add_argument('--remote', default='origin')
    r.add_argument('--branch', default='main')
    r.add_argument('--run-label')
    a = sub.add_parser('acquire')
    a.add_argument('--root', default=Path('.'), type=Path)
    a.add_argument('--out', required=True, type=Path)
    a.add_argument('--day', help='the delivery day (default: the next one whose cutoff is still ahead)')
    a.add_argument('--remote', default='origin')
    a.add_argument('--branch', default='main')
    a.add_argument('--poll-seconds', type=int, default=300, help='while the load forecast is not yet published')
    a.add_argument('--margin-seconds', type=int, default=120, help='stop polling this long before the cutoff')
    a.add_argument('--window-hours', type=float, default=3.0, help='do nothing when the next cutoff is further away '
                                                                   '(0: no window); a given --day ignores it')
    i = sub.add_parser('seed-store')
    i.add_argument('--root', default=Path('.'), type=Path)
    g = i.add_mutually_exclusive_group(required=True)
    g.add_argument('--genesis', help='<state.json>:<first day>[:<origin>]: a new store whose states are those of '
                                    '<origin> (default: the first day); for 4.7T and the live product, the '
                                    'freeze-origin state at 2026-09-06 and the first day 2026-09-07')
    g.add_argument('--carried', type=Path, help='a store folder an earlier run left, copied and verified')
    i.add_argument('--source')
    c = sub.add_parser('carry')
    c.add_argument('--root', default=Path('.'), type=Path)
    c.add_argument('--last-window-day', required=True)
    c.add_argument('--first-live-day', required=True)
    c.add_argument('--reason', required=True)
    k = sub.add_parser('search')
    k.add_argument('--root', default=Path('.'), type=Path)
    k.add_argument('--k', required=True, type=int)
    k.add_argument('--file', required=True, type=Path)
    k.add_argument('--provenance', required=True)
    s = sub.add_parser('status')
    s.add_argument('--root', default=Path('.'), type=Path)
    args = ap.parse_args(argv)
    return {'prepare': cmd_prepare, 'run': cmd_run, 'acquire': cmd_acquire, 'seed-store': cmd_seed_store,
            'carry': cmd_carry, 'search': cmd_search, 'status': cmd_status}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
