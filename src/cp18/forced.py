"""One forced cause of each kind of day not issued, through the real pipeline, on development days (run locally).

Everything is real except the one thing each case forces: the checkout, its `.githooks`, the claim against a bare
origin, the store (CP-17b's host state before 2026-08-31's release), the H2 checks (`cp17.hostcheck`, the canary fit
included), the live-input builder replaying the recorded responses, and `cp17.daily.issue_day`. No issuance fit runs:
the two chain failures fail at their process start.

- **Run A**, wall clock 2026-08-30 12:00 Berlin, plan 2026-08-31: the machine check sees a runner whose image is not
  `macos26` (a forced environment), so the gate stops there, the canary never runs, nothing is issued, and 2026-08-31
  stays open with "a failed H2 check".
- **Run B**, wall clock 2026-09-04 12:00 Berlin, target 2026-09-05, lookback 2026-09-02..04, with the real gate:
  - 2026-08-31 has left the lookback and is closed with its last attempt's cause, the failed H2 check;
  - 2026-09-01 was never attempted: closed as "no run within the lookback";
  - 2026-09-02 is served, by a forced serving rule, by a search whose state is not in the store: "S_14's state not
    in place";
  - 2026-09-03's load forecast is archived as a live retrieval made two hours after the day's cutoff, with no
    document time, so its publication cannot be shown to precede the cutoff: "a live day's inputs that cannot be
    shown to precede its cutoff";
  - 2026-09-04 and 2026-09-05 run the frozen chain under a host whose binary is absent, so every attempt fails and
    the chain records a failed issuance: "a failed issuance by the chain". 2026-09-02 and 2026-09-03 are closed
    before 2026-09-04's record, since days go forward.

    PYTHONPATH=src python -m cp18.forced --project /Users/djourno/Downloads/PJM --work <dir> --out <record>
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

from . import causes, pipeline, plan
from .claim import Claim
from .live_inputs import GFS_SOURCE, REPLAY, LiveInputs
from .store import STORE_PATH, LiveStore

STATE = 'reports/cp18a/dryrun/states/start-2026-08-31.json'
RECORDED = 'reports/cp18a/dryrun/recorded'
FIRST = date(2026, 8, 31)
EXPECTED = {'2026-08-31': causes.H2, '2026-09-01': causes.NO_RUN, '2026-09-02': causes.S14,
            '2026-09-03': causes.INPUTS, '2026-09-04': causes.CHAIN, '2026-09-05': causes.CHAIN}


class LateLoadInputs(LiveInputs):
    """The live builder, with the named days' load forecast archived as a live retrieval after the cutoff."""

    def __init__(self, *args, late=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.late = {str(d) for d in late}

    def acquire(self, day: date) -> None:
        from .live import archive as AR, recorded as R, smard
        from .live.transport import ReplayTransport
        archive = AR.Archive(self.archive_root)
        for name in sorted(self.late):
            x = date.fromisoformat(name)
            if self.base_end < x <= day and not archive.has(x):
                tr = ReplayTransport(self.recorded / REPLAY / 'smard')
                prices = smard.fetch(tr, 'price', [x - timedelta(days=k) for k in (3, 2, 1)])
                AR.store_market(archive, x, prices, development_snapshot=True)
                load = smard.fetch(tr, 'load_forecast', [x])
                after = (plan.cutoff(x) + timedelta(hours=2)).isoformat()
                for ex in load.exchanges:
                    if ex.response is not None:
                        ex.response = replace(ex.response, provenance='live', retrieved_utc=after,
                                              note='forced: retrieved two hours after the cutoff, no document time')
                AR.store_market(archive, x, load, development_snapshot=False)
                R.archive_gfs_recorded(archive, x, self.root, [self.recorded / GFS_SOURCE])
        super().acquire(day)


def _git(*args, cwd=None):
    done = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f'git {args[0]}: {done.stderr.strip()[-400:]}')
    return done.stdout.strip()


def setup(candidate: Path, work: Path) -> Path:
    """A bare origin whose main holds the candidate's tree and a store at CP-17b's state before 2026-08-31, and the
    pipeline's checkout of it, its hooks active."""
    from cp17.shadow import ShadowStates
    bare, seed, checkout = work / 'origin.git', work / 'seed', work / 'checkout'
    _git('init', '--quiet', '--bare', '-b', 'main', str(bare))
    _git('clone', '--quiet', '--no-local', str(candidate), str(seed))
    _git('-C', str(seed), 'checkout', '--quiet', '--detach', _git('-C', str(candidate), 'rev-parse', 'HEAD'))
    _git('-C', str(seed), 'config', 'core.hooksPath', '.githooks')
    state = ShadowStates.from_dict(json.loads((seed / STATE).read_text()))
    LiveStore.init(seed / STORE_PATH, state, FIRST, f'forced causes: {STATE}', FIRST)
    _git('-C', str(seed), 'add', STORE_PATH)
    _git('-C', str(seed), '-c', 'user.name=delu-daily', '-c', 'user.email=delu-daily@users.noreply.github.com',
         'commit', '--quiet', '-m', 'forced causes: the store at its start')
    _git('-C', str(seed), 'push', '--quiet', str(bare), 'HEAD:refs/heads/main')
    _git('clone', '--quiet', str(bare), str(checkout))
    _git('-C', str(checkout), 'config', 'core.hooksPath', '.githooks')
    return checkout


def run(candidate: Path, project: Path, work: Path) -> dict:
    from cp17 import daily, hostcheck, runtime
    from cp26.search import serving_search
    os.environ.setdefault('CP17_PROJECT_ROOT', str(project))
    checkout = setup(candidate, work)
    inputs = LateLoadInputs(checkout, project, checkout / RECORDED, late=[date(2026, 9, 3)])
    a = pipeline.Settings(root=checkout, project=project, out=work / 'out-a', host_kind='local', run_label='forced-h2')
    checks = pipeline.h2_checks(a)
    checks['machine'] = lambda: hostcheck.machine({'GITHUB_ACTIONS': 'true', 'ImageOS': 'macos15',
                                                   'ImageVersion': 'forced-failure'})
    run_a = pipeline.run(a, plan.parse_now('2026-08-30T12:00:00+02:00'), claim=Claim(checkout), inputs=inputs,
                         checks=checks)

    def serving(day):
        return 14 if day == date(2026, 9, 2) else serving_search(day)

    def broken(day, gfs_available, payloads, *, settings, parent_sha256, data_refresh_utc, states):
        zip_path = project / pipeline.PACKAGE_ZIP
        host = daily.Host(project / runtime.PYODIDE_DIR, zip_path, node=str(work / 'absent' / 'node'))
        return daily.issue_day(day, gfs_available, payloads, host=host, out=settings.out / 'runs',
                               identity=pipeline.frozen_identity(settings.root), parent_sha256=parent_sha256,
                               data_refresh_utc=data_refresh_utc, states=states)
    b = pipeline.Settings(root=checkout, project=project, out=work / 'out-b', host_kind='local', run_label='forced-b')
    run_b = pipeline.run(b, plan.parse_now('2026-09-04T12:00:00+02:00'), claim=Claim(checkout), inputs=inputs,
                         issue=broken, serving=serving)
    _git('-C', str(checkout), 'fetch', '--quiet', 'origin')
    _git('-C', str(checkout), 'checkout', '--quiet', '--detach', 'origin/main')
    store = LiveStore(checkout / STORE_PATH)
    store.verify()
    days = store.days()
    found = {d: v['cause'] for d, v in days.items()}
    record = {'schema': 'cp18a-forced-causes-v1', 'candidate': _git('-C', str(candidate), 'rev-parse', 'HEAD'),
              'expected': EXPECTED, 'found': found, 'as_expected': found == EXPECTED,
              'every_cause_forced': sorted(set(found.values())) == sorted(causes.ORDER),
              'issued': [d for d, v in days.items() if v['status'] == 'issued'],
              'runs': {'a': {k: run_a[k] for k in ('run_id', 'planned', 'outcome', 'days', 'closed')},
                       'b': {k: run_b[k] for k in ('run_id', 'planned', 'outcome', 'days', 'closed')}},
              'gate_a': {'failed': run_a['gate']['failed'], 'not_run': run_a['gate']['not_run']},
              'gate_b': {'passed': run_b['gate']['passed'],
                         'canary_quantiles_sha256': next(c.get('quantiles_sha256') for c in run_b['gate']['checks']
                                                         if c['check'] == 'canary')},
              'inputs_2026_09_03': (run_b['days']['2026-09-03'].get('inputs') or {}).get('availability'),
              'chain_2026_09_04': json.loads((checkout / STORE_PATH / 'issued/2026-09-04/record.json').read_text())['path'],
              'store_head': store.verify()['entry_sha256'],
              'origin_log': _git('-C', str(work / 'origin.git'), 'log', '--format=%s', 'main').splitlines()[:5]}
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m cp18.forced')
    ap.add_argument('--candidate', default=Path('.'), type=Path)
    ap.add_argument('--project', required=True, type=Path)
    ap.add_argument('--work', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    if args.work.exists() and any(args.work.iterdir()):
        raise SystemExit(f'{args.work} is not empty')
    args.work.mkdir(parents=True, exist_ok=True)
    started = datetime.now()
    record = run(args.candidate.resolve(), args.project.resolve(), args.work.resolve())
    record['seconds'] = (datetime.now() - started).total_seconds()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True, default=str) + '\n')
    for day, cause in record['found'].items():
        print(day, causes.TEXT.get(cause, cause), 'as expected' if EXPECTED.get(day) == cause else 'NOT AS EXPECTED')
    print('every cause forced:', record['every_cause_forced'], '| as expected:', record['as_expected'])
    return 0 if record['as_expected'] and record['every_cause_forced'] else 1


if __name__ == '__main__':
    sys.exit(main())
