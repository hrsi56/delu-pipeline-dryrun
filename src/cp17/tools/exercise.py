"""CP-17b's exercises on real development days, through the frozen driver `cp17.daily` (items 1, 2 and 5; N1).

- **`shadow-run`** (item 2): consecutive delivery days from an earlier origin through 2026-09-06. The fallbacks'
  states start from CP-26's at the first origin, rebuilt from its ledger (`cp17.tools.states`): the released
  buffers, and the issuance of the day before, registered but not yet released. Each day releases D-2, builds
  the day's payloads from the states' own buffers, issues the day (A1 and the shadows, `cp17.daily.issue_day`
  under the issuing host; or the shadows natively), and registers the shadows. Each day's buffers are compared
  with CP-26's rebuilt state at that origin, bitwise natively and by their largest difference under the host; the
  last day's with the freeze-origin state CP-17 recorded. Natively, each shadow's quantiles are also compared
  bitwise with CP-26's stored forecast. Under the issuing host, two days run at once (a day needs only D-2's
  registrations), each day's own attempts one after another.
- **`partial`** (item 1): one development day, under the issuing host and natively, from CP-26's states at it.
- **`control`** (N1): the freeze-origin day with A1's issuing-host process killed by a short timeout and a stale
  A1 record planted where a reused path would find it. A1 must fail and the chain move on.
- **`issue`** (item 5): one day's issuance under the issuing host with a given identity, from given payloads.

Every run reads only development data through 2026-09-06 (`cp17.tools.payloads.admit`). An exercise's artifacts
cite no policy identity, which is fixed only after the code (`cp17.tools.refreeze`); the freeze-origin artifact
cites it.

    OPENBLAS_NUM_THREADS=1 ... PYTHONPATH=src python -m cp17.tools.exercise shadow-run --mode host \\
        --first 2026-08-24 --bundle <src.zip> --out <dir>
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .. import daily, dailyfit, issue, retrain
from ..shadow import POLICIES as SHADOWS, ShadowStates
from . import ledger, payloads, states
from .parity import BLAS, bits_equal, check_d2, check_policy

FREEZE_ORIGIN = date(2026, 9, 6)
PYODIDE = Path('/Users/djourno/Downloads/PJM/.local/tools/pyodide-0.28.0')
NO_IDENTITY = {'policy_identity_sha256': None, 'package_sha256': None}


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=str) + '\n')


def truth(data):
    """Published prices on the requested keys (`release` asks only for complete days <= D-2)."""
    def at(index):
        pos = data.index.get_indexer(ledger.keys(index))
        if (pos < 0).any():
            return np.full(len(index), np.nan)
        return np.asarray(data.y[pos], float)
    return at


def start(root, ctx, origin: date) -> tuple[ShadowStates, dict]:
    """CP-26's states at `origin`, after its release: the rebuilt buffers, and D-1's issuance pending."""
    rebuilt = states.rebuild(root, ctx, [origin])
    buffers = {p: rebuilt['buffers'][p][origin] for p in SHADOWS}
    before = ledger.day(root, origin - timedelta(days=1))
    pending = {}
    for p in SHADOWS:
        record = before['policies'][f'nominal/{p}']
        if record['error_eligible']:
            item = {'day': str(origin - timedelta(days=1)), 'index': list(record['index']),
                    'central': list(record['native_central'])}
            if p != 'B0':
                item['scale'] = list(record['issued_scale'])
            pending[p] = [item]
    return ShadowStates.from_buffers(origin, buffers, pending), {'buffers_sha256': {
        p: states.buffer_sha256(b) for p, b in buffers.items()}, 'pending': {p: [i['day'] for i in v] for p, v in pending.items()}}


def build(ctx, root, day: date, folder: Path, buffers: dict) -> dict:
    """The day's payloads (A1, v4, v2, B0), written to `folder`; the buffers are the states' own."""
    built = {'A1': payloads.a1_payload(ctx, day, payloads.ensemble(ledger.inputs_dir(root), day)),
             'v4': payloads.v4_payload(ctx, day, buffers['v4']), 'v2': payloads.v2_payload(ctx, day, buffers['v2']),
             'B0': payloads.b0_payload(ctx, day, buffers['B0'])}
    return {p: payloads.write(a, m, folder, f'{p}-{day}') for p, (a, m) in built.items()}


def compare_buffers(mine: list[dict], theirs: list[dict]) -> dict:
    days_equal = [b['day'] for b in mine] == [b['day'] for b in theirs]
    out = {'days_equal': days_equal, 'sha256': states.buffer_sha256(mine), 'cp26_sha256': states.buffer_sha256(theirs)}
    out['bitwise'] = out['sha256'] == out['cp26_sha256']
    if days_equal:
        a = np.concatenate([np.asarray(b['errors'], float) for b in mine])
        b = np.concatenate([np.asarray(b['errors'], float) for b in theirs])
        out['max_abs_standardized'] = float(np.abs(a - b).max()) if a.shape == b.shape else None
    return out


def _host(bundle: Path, timeout: float = daily.TIMEOUT_SECONDS) -> daily.Host:
    return daily.Host(PYODIDE, bundle, timeout=timeout)


def shadow_run(root: Path, mode: str, first: date, last: date, out: Path, bundle: Path | None) -> dict:
    if any(os.environ.get(n) != '1' for n in BLAS):
        raise SystemExit('export the five *_NUM_THREADS=1 before NumPy loads, as CP-26 ran')
    t0 = time.perf_counter()
    ctx = payloads.context(root)
    origins = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    reference = states.rebuild(root, ctx, origins)['buffers']
    state, start_record = start(root, ctx, first)
    at = truth(ctx.data)
    record = {'schema': 'cp17b-shadow-run-v1', 'mode': mode, 'first': str(first), 'last': str(last),
              'start': {'origin': str(first), **start_record,
                        'source': "CP-26's nominal ledger, rebuilt and checked (cp17.tools.states)"},
              'bundle_sha256': file_sha256(bundle) if bundle else None, 'days': {}}
    host = _host(bundle) if mode == 'host' else None
    pairs = [origins[i:i + 2] for i in range(0, len(origins), 2)] if mode == 'host' else [[d] for d in origins]
    for group in pairs:
        prepared = {}
        for day in group:
            state.release(day, at)
            buffers = {p: state.buffer(p) for p in SHADOWS}
            folder = out / 'payloads'
            written = build(ctx, root, day, folder, buffers)
            prepared[day] = {'buffers': {p: compare_buffers(buffers[p], reference[p][day]) for p in SHADOWS},
                             'payload_sha256': {p: w['payload_sha256'] for p, w in written.items()}}
        if mode == 'host':
            with ThreadPoolExecutor(max_workers=len(group)) as pool:
                done = dict(zip(group, pool.map(lambda d: daily.issue_day(
                    d, True, out / 'payloads', host=host, out=out / 'runs', identity=NO_IDENTITY), group)))
        else:
            done = {day: _native_shadows(root, day, out / 'payloads') for day in group}
        for day in group:
            item = done[day]
            intake = {}
            for p in SHADOWS:
                result = _result(item, p, mode)
                intake[p] = state.issue(p, day, result)
            prepared[day]['intake'] = intake
            prepared[day].update(_summary(item, mode))
            record['days'][str(day)] = prepared[day]
            print(day, mode, prepared[day].get('path'), intake, {p: v['bitwise'] for p, v in prepared[day]['buffers'].items()},
                  f"{prepared[day].get('total_seconds', 0):.0f}s", flush=True)
        _dump(out / 'shadow-run.json', record)
    # The released buffers at the last origin (registrations add pending issuances, never released errors). At
    # the freeze origin they are compared with the state CP-17 recorded there.
    freeze = json.loads((Path(root) / 'docs/track-b/evidence/cp-17/freeze-manifest.json').read_text())['initial_state']
    digest = state.digest()
    record['final'] = {'origin': str(last), 'buffers_sha256': digest, 'vs_cp26': record['days'][str(last)]['buffers']}
    if last == FREEZE_ORIGIN:
        record['final'].update(cp17_recorded_sha256={p: freeze['buffers_sha256'][p] for p in SHADOWS},
                               equal_to_cp17_bitwise={p: digest[p] == freeze['buffers_sha256'][p] for p in SHADOWS})
    record['state'] = state.to_dict()
    record['seconds'] = time.perf_counter() - t0
    _dump(out / 'shadow-run.json', record)
    return record


def file_sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _native_shadows(root, day: date, folder: Path) -> dict:
    stored = ledger.day(root, day)['policies']
    out = {'shadows': {}, 'results': {}, 'parity': {}}
    started = time.perf_counter()
    for p in SHADOWS:
        npz, meta = daily._paths(folder, p, day)
        arrays, value = dailyfit.load(npz, meta)
        problems = issue.check_inputs(p, arrays, value)
        if problems:
            out['shadows'][p] = {'status': 'failed', 'cause': 'inputs: ' + '; '.join(problems)}
            continue
        t = time.perf_counter()
        result = dailyfit.run(p, npz, meta)
        out['results'][p] = result
        out['shadows'][p] = {'status': 'ok', 'seconds': time.perf_counter() - t, 'keys': len(result['index'])}
        out['parity'][p] = check_policy(result, stored[f'nominal/{p}'], 'b0_layer' if p == 'B0' else 'h_layer')
    out['total_seconds'] = time.perf_counter() - started
    return out


def _result(item: dict, policy: str, mode: str):
    if mode == 'native':
        return item['results'].get(policy)
    run = item['runs'].get(policy)
    if run is None or run.get('errors'):
        return None
    return run['python']['result']


def _summary(item: dict, mode: str) -> dict:
    if mode == 'native':
        return {'shadows': item['shadows'], 'parity_cp26': {p: c['pass'] for p, c in item['parity'].items()},
                'parity_quantiles_bitwise': {p: c['quantiles_bitwise'] for p, c in item['parity'].items()},
                'total_seconds': item['total_seconds']}
    art = item['artifact']
    return {'path': [(a['policy'], a['status']) for a in item['decision']['attempts']], 'display': item['decision']['display'],
            'issued_policy': art['issued_policy'], 'artifact_verifies': issue.verify_artifact(art) == [],
            'shadows': item['shadows'], 'process_seconds': item['process_seconds'],
            'process_peak_bytes': {p: r.get('process_rss_peak_bytes') for p, r in item['runs'].items()},
            'total_seconds': item['process_seconds_sum'], 'driver_seconds': item['total_seconds'],
            'within_ceiling': item['process_seconds_sum'] <= issue.WALL_LIMIT_SECONDS,
            'blocked_non_local_requests': item['blocked_non_local_requests'],
            'executed_files': {p: (r.get('python') or {}).get('source_sha256') for p, r in item['runs'].items()}}


def partial(root: Path, day: date, out: Path, bundle: Path) -> dict:
    """Item 1: one day under the issuing host and natively, from CP-26's states at it."""
    if any(os.environ.get(n) != '1' for n in BLAS):
        raise SystemExit('export the five *_NUM_THREADS=1 before NumPy loads, as CP-26 ran')
    ctx = payloads.context(root)
    host_state, start_record = start(root, ctx, day)
    native_state, _ = start(root, ctx, day)
    buffers = {p: host_state.buffer(p) for p in SHADOWS}
    written = build(ctx, root, day, out / 'payloads', buffers)
    keys = {p: issue.forecastable_keys(p, *dailyfit.load(*daily._paths(out / 'payloads', p, day))) for p in written}
    hosted = daily.issue_day(day, True, out / 'payloads', host=_host(bundle), out=out / 'runs', identity=NO_IDENTITY,
                             states=host_state)
    native = daily.comparison_day(day, True, out / 'payloads', states=native_state)
    nd = native['decision']
    art = hosted['artifact']
    record = {'schema': 'cp17b-partial-day-v1', 'day': str(day), 'canonical_hours': len(issue.canonical_hours(day)),
              'forecastable_hours': {p: len(k) for p, k in keys.items()},
              'missing_hours': issue.partial(day, keys['A1']), 'start': start_record,
              'payload_sha256': {p: w['payload_sha256'] for p, w in written.items()},
              'host': {'path': [(a['policy'], a['status']) for a in hosted['decision']['attempts']],
                       'causes': {a['policy']: a.get('cause') for a in hosted['decision']['attempts'] if a['status'] != 'ok'},
                       'display': hosted['decision']['display'], 'artifact': art,
                       'artifact_verifies': issue.verify_artifact(art) == [], 'shadows': hosted['shadows'],
                       'process_seconds': hosted['process_seconds'],
                       'blocked_non_local_requests': hosted['blocked_non_local_requests']},
              'native': {'path': [(a['policy'], a['status']) for a in nd['attempts']],
                         'causes': {a['policy']: a.get('cause') for a in nd['attempts'] if a['status'] != 'ok'},
                         'display': nd['display'], 'label': native['comparison'], 'shadows': native['shadows'],
                         'resources': {a['policy']: a.get('resources') for a in nd['attempts']}}}
    record['same_path'] = record['host']['path'] == record['native']['path']
    record['same_display'] = record['host']['display'] == record['native']['display']
    if art['issued_policy']:
        n = next(a for a in nd['attempts'] if a['policy'] == art['issued_policy'])
        h = hosted['runs'][art['issued_policy']]['python']['result']
        c = retrain.compare(art['issued_policy'], n['result'], h)
        record['issued_vs_native'] = {'status': c['status'], 'max_abs_eur': c['deviation'].get('max_abs_eur'),
                                      'selections_match': c['selections_match']}
        stored = ledger.day(root, day)
        if art['issued_policy'] == 'A1':
            d2 = check_d2(n['result'], stored['members']['nominal']['D2'])
            record['native_vs_cp26'] = {'A1': {k: d2[k] for k in ('pass', 'quantiles_bitwise', 'max_abs_eur')}}
    stored = ledger.day(root, day)['policies']
    record['cp26_keys'] = {p: len(stored[f'nominal/{p}']['index']) for p in SHADOWS}
    record['cp26_error_eligible'] = {p: stored[f'nominal/{p}']['error_eligible'] for p in SHADOWS}
    _dump(out / f'partial-{day}.json', record)
    print(day, record['host']['path'], record['host']['display'], record['same_path'], flush=True)
    return record


class _KillingHost(daily.Host):
    """The issuing host with one policy's process killed by a short timeout (the control only)."""

    def __init__(self, *args, kill: str, kill_after: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.kill, self.kill_after = kill, kill_after

    def run(self, payloads, stem, out):
        if stem.split('-', 1)[0] != self.kill:
            return super().run(payloads, stem, out)
        normal = self.timeout
        self.timeout = self.kill_after
        try:
            return super().run(payloads, stem, out)
        finally:
            self.timeout = normal


def control(root: Path, day: date, payload_dir: Path, out: Path, bundle: Path, stale: Path, kill_after: float) -> dict:
    """N1: A1's process is killed; a stale A1 record of a successful fit lies where a reused path would find it."""
    runs = out / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    planted = []
    for name in (f'node-attempt-A1-{day}.json', f'node-A1-{day}.json', 'run.json'):
        shutil.copyfile(stale, runs / name)
        planted.append(str(runs / name))
    host = _KillingHost(PYODIDE, bundle, kill='A1', kill_after=kill_after)
    hosted = daily.issue_day(day, True, payload_dir, host=host, out=out / 'runs', identity=NO_IDENTITY)
    art = hosted['artifact']
    record = {'schema': 'cp17b-killed-process-control-v1', 'day': str(day), 'kill_after_seconds': kill_after,
              'stale_record': {'source': str(stale), 'planted': planted,
                               'its_quantiles_sha256': json.loads(stale.read_text())['python']['result']['quantiles_sha256']},
              'path': [(a['policy'], a['status']) for a in hosted['decision']['attempts']],
              'causes': {a['policy']: a.get('cause') for a in hosted['decision']['attempts'] if a['status'] != 'ok'},
              'display': hosted['decision']['display'], 'issued_policy': art['issued_policy'],
              'artifact_verifies': issue.verify_artifact(art) == [], 'artifact': art, 'shadows': hosted['shadows'],
              'process_seconds': hosted['process_seconds'],
              'a1_folder': hosted['runs']['A1'].get('folder'), 'a1_errors': hosted['runs']['A1'].get('errors')}
    record['pass'] = (record['path'][:2] == [('A1', 'failed'), ('v4', 'ok')] and art['issued_policy'] == 'v4'
                      and record['artifact_verifies'] and 'killed' in (record['causes'].get('A1') or ''))
    _dump(out / f'control-{day}.json', record)
    print(record['path'], record['display'], 'PASS' if record['pass'] else 'FAIL', flush=True)
    return record


def issue_one(day: date, gfs: bool, payload_dir: Path, out: Path, bundle: Path, identity: dict) -> dict:
    """One day's issuance under the issuing host, citing `identity` (the freeze-origin reissue, item 5)."""
    hosted = daily.issue_day(day, gfs, payload_dir, host=_host(bundle), out=out / 'runs', identity=identity,
                             data_refresh_utc=issue.issue_utc(day))
    art = hosted['artifact']
    art_path = out / f'artifact-{day}.json'
    art_path.write_text(json.dumps(art, sort_keys=True, allow_nan=False) + '\n')
    run = hosted['runs'].get(art['issued_policy'] or '', {})
    arrays = Path(run['folder']) / 'run-arrays.npz' if run.get('folder') else None
    record = {'schema': 'cp17b-issuance-v1', 'day': str(day), 'gfs_available': gfs, 'artifact': art,
              'artifact_path': str(art_path), 'artifact_file_sha256': file_sha256(art_path),
              'artifact_arrays_path': str(arrays) if arrays and arrays.exists() else None,
              'artifact_arrays_sha256': file_sha256(arrays) if arrays and arrays.exists() else None,
              'artifact_verifies': issue.verify_artifact(art) == [],
              'path': [(a['policy'], a['status']) for a in hosted['decision']['attempts']],
              'display': hosted['decision']['display'], 'shadows': hosted['shadows'],
              'process_seconds': hosted['process_seconds'], 'total_seconds': hosted['process_seconds_sum'],
              'process_peak_bytes': {p: r.get('process_rss_peak_bytes') for p, r in hosted['runs'].items()},
              'blocked_non_local_requests': hosted['blocked_non_local_requests'],
              'executed_files': {p: (r.get('python') or {}).get('source_sha256') for p, r in hosted['runs'].items()},
              'host': art['lineage']['host']}
    _dump(out / f'issuance-{day}.json', record)
    print(day, record['path'], record['display'], record['artifact_verifies'], flush=True)
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('shadow-run')
    a.add_argument('--mode', choices=('host', 'native'), required=True)
    a.add_argument('--first', required=True)
    a.add_argument('--last', default=str(FREEZE_ORIGIN))
    a.add_argument('--bundle', type=Path)
    a.add_argument('--out', required=True, type=Path)
    b = sub.add_parser('partial')
    b.add_argument('--day', required=True)
    b.add_argument('--bundle', required=True, type=Path)
    b.add_argument('--out', required=True, type=Path)
    c = sub.add_parser('control')
    c.add_argument('--day', default=str(FREEZE_ORIGIN))
    c.add_argument('--payloads', required=True, type=Path)
    c.add_argument('--bundle', required=True, type=Path)
    c.add_argument('--stale', required=True, type=Path, help="a successful A1 record of the day (CP-17's)")
    c.add_argument('--kill-after', type=float, default=15.0)
    c.add_argument('--out', required=True, type=Path)
    d = sub.add_parser('issue')
    d.add_argument('--day', default=str(FREEZE_ORIGIN))
    d.add_argument('--gfs', choices=('0', '1'), default='1')
    d.add_argument('--payloads', required=True, type=Path)
    d.add_argument('--bundle', required=True, type=Path)
    d.add_argument('--identity', required=True, type=Path, help='the policy identity file the artifact cites')
    d.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'shadow-run':
        if args.mode == 'host' and not args.bundle:
            ap.error('--bundle is required under the issuing host')
        r = shadow_run(args.root, args.mode, date.fromisoformat(args.first), date.fromisoformat(args.last), args.out,
                       args.bundle)
        print('final', r['final'].get('equal_to_cp17_bitwise'), f"{r['seconds']:.0f}s")
    elif args.cmd == 'partial':
        partial(args.root, date.fromisoformat(args.day), args.out, args.bundle)
    elif args.cmd == 'control':
        r = control(args.root, date.fromisoformat(args.day), args.payloads, args.out, args.bundle, args.stale,
                    args.kill_after)
        return 0 if r['pass'] else 1
    else:
        from .freeze import digest
        identity = json.loads(args.identity.read_text())
        short = {'policy_identity_sha256': digest(identity), 'package_sha256': identity['code']['package_sha256']}
        r = issue_one(date.fromisoformat(args.day), args.gfs == '1', args.payloads, args.out, args.bundle, short)
        return 0 if r['artifact_verifies'] else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
