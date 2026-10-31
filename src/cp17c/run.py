"""CP-17c's runs, the same on a GitHub-hosted `macos-26` runner and on the Owner's Mac.

Nothing frozen changes here. Every fit is `cp17.daily.Host.run` under the official binary and the pinned Pyodide
folder, at their project-relative paths; every shadow day is `cp17.daily.issue_day`, citing the frozen identity; the
checks are `cp17.hostcheck`'s; the states are `cp17.shadow.ShadowStates`, carried by the state store
(`cp17c.store`). The comparison with CP-17b's records is offline (`cp17c.compare`).

- **node:** retrieve the tarball and `SHASUMS256.txt` and check all three by hash (`cp17.runtime.retrieve`).
- **package:** build the package zip from `src/` and check it against the manifest's `zip_sha256`.
- **checks:** the H2 gate, for real; then one forced failure of each check, each in a gate of its own: the gate must
  stop at that check, run none after it, and issue nothing. Then a store is initialized at the freeze origin from
  the freeze-origin states and loaded back, as 4.7T's first window day would read it. With `--stems`, the run's own
  gate then admits A1's parity fits at those stems (as `parity` runs them), so one run pays for one gate.
- **parity:** the H2 gate, then A1 at the given stems, each in a fresh issuing-host process, judged as an attempt.
- **shadow:** the H2 gate, then a block of consecutive delivery days, one process per day (`day`), carried by a
  store initialized with the block's start state. Days run one after another, so the day's measurements are its
  sequential cost.
- **day:** one delivery day in its own process: load and check the store, release D-2, check the buffers against the
  ones the day's payloads carry, issue the day, register the shadows, append to the store.
- **resume:** the H2 gate, then each given block's remaining days, one block after the other, each from the store an
  earlier run left (carried here by the repository, checked by its chain before any day runs): the states' route
  between runs, exercised.

    OPENBLAS_NUM_THREADS=1 ... PYTHONPATH=src python -m cp17c.run checks --out <dir>
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BLAS = ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
MANIFEST = 'docs/track-b/evidence/cp-17c/freeze-manifest.json'
PACKAGE_ZIP = '.local/artifacts/cp-17c/bundle/src.zip'


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=str) + '\n')


def sha256_file(path) -> str:
    from cp17 import runtime
    return runtime.sha256_file(path)


def buffer_sha256(buffer: list[dict]) -> str:
    return hashlib.sha256(json.dumps(buffer, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_blas() -> None:
    if any(os.environ.get(n) != '1' for n in BLAS):
        raise SystemExit('export the five *_NUM_THREADS=1 before NumPy loads, as CP-17b ran')


def project() -> Path:
    from cp17 import runtime
    return runtime.project_root(Path.cwd())


def manifest(root: Path) -> dict:
    return json.loads((root / MANIFEST).read_text())


def identity(root: Path) -> dict:
    value = manifest(root)
    return {'policy_identity_sha256': value['policy_identity_sha256'],
            'package_sha256': value['policy_identity']['code']['package_sha256']}


def host(root: Path, timeout: float | None = None):
    from cp17 import daily, runtime
    proj = project()
    zip_path = proj / PACKAGE_ZIP
    expected = manifest(root)['policy_identity']['code']['zip_sha256']
    if not zip_path.exists() or sha256_file(zip_path) != expected:
        raise SystemExit(f'{zip_path}: not the frozen package zip; run `package` first')
    kwargs = {} if timeout is None else {'timeout': timeout}
    return daily.Host(proj / runtime.PYODIDE_DIR, zip_path, **kwargs)


# --- node, package ------------------------------------------------------------------------------------------------

def node(args) -> int:
    from cp17 import runtime
    record = runtime.retrieve(project())
    dump(args.out, record)
    print(record['binary']['path'], record['binary']['sha256'], record['tarball']['seconds'], flush=True)
    return 0


def package(args) -> int:
    from cp17.tools import bundle
    root, proj = Path(args.root).resolve(), project()
    built = bundle.build(root / 'src', proj / PACKAGE_ZIP)
    expected = manifest(root)['policy_identity']['code']
    ok = built['zip_sha256'] == expected['zip_sha256'] and built['package_sha256'] == expected['package_sha256']
    dump(args.out, {'zip_sha256': built['zip_sha256'], 'package_sha256': built['package_sha256'],
                    'zip_bytes': built['zip_bytes'], 'equal_to_manifest': ok})
    print('package', built['zip_sha256'][:16], 'equal' if ok else 'NOT EQUAL', flush=True)
    return 0 if ok else 1


# --- checks -------------------------------------------------------------------------------------------------------

def _gate(root: Path, out: Path, overrides: dict | None = None) -> dict:
    """The H2 gate with `overrides` replacing named checks, and an issuance that only records being called."""
    from cp17 import hostcheck, runtime
    proj = project()
    called = []
    checks = {'pins': lambda: hostcheck.pins(root),
              'binary': lambda: hostcheck.binary(proj),
              'machine': lambda: hostcheck.machine(),
              'canary': lambda: hostcheck.canary(host(root), proj / runtime.CANARY_DIR, out / 'canary')}
    checks.update(overrides or {})
    record = hostcheck.gate(checks)
    if record['passed']:
        called.append('issued')
    record['issuance_called'] = bool(called)
    return record


def gate_or_stop(root: Path, out: Path) -> dict:
    record = _gate(root, out)
    dump(out / 'checks.json', record)
    print('H2', 'passed' if record['passed'] else f"FAILED at {record['failed']}", flush=True)
    return record


def forced(root: Path, out: Path, canary_expected: bool) -> dict:
    """One forced failure of each check, each in its own gate."""
    from cp17 import hostcheck, runtime
    proj = project()
    work = out / 'forced'
    work.mkdir(parents=True, exist_ok=True)
    # pins: a manifest with one byte changed, so `refreeze verify` refuses it and its output is not `verified`
    bad_manifest = work / 'manifest-one-byte.json'
    data = bytearray((root / MANIFEST).read_bytes())
    i = data.index(b'"zip_sha256": "') + len(b'"zip_sha256": "')
    data[i] = ord('0') if data[i] != ord('0') else ord('1')
    bad_manifest.write_bytes(bytes(data))
    # binary: a project whose bin/node has one byte appended, beside the real tarball
    fake = work / 'project'
    (fake / runtime.NODE_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    (fake / runtime.TARBALL_RELATIVE).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(proj / runtime.NODE_RELATIVE, fake / runtime.NODE_RELATIVE)
    with open(fake / runtime.NODE_RELATIVE, 'ab') as handle:
        handle.write(b'\0')
    shutil.copyfile(proj / runtime.TARBALL_RELATIVE, fake / runtime.TARBALL_RELATIVE)
    # machine: a runner whose image is not macos26
    env = {**os.environ, 'GITHUB_ACTIONS': 'true', 'ImageOS': 'macos15', 'ImageVersion': 'forced-failure'}
    # canary: its payload with one byte changed, so the attempt fails at its inputs before any process starts
    payloads = work / 'canary-payload'
    payloads.mkdir(exist_ok=True)
    for suffix in ('.json', '.npz'):
        shutil.copyfile(proj / runtime.CANARY_DIR / f'{hostcheck.CANARY_STEM}{suffix}', payloads / f'{hostcheck.CANARY_STEM}{suffix}')
    blob = bytearray((payloads / f'{hostcheck.CANARY_STEM}.npz').read_bytes())
    blob[len(blob) // 2] ^= 0x01
    (payloads / f'{hostcheck.CANARY_STEM}.npz').write_bytes(bytes(blob))
    cases = {
        'pins': {'pins': lambda: hostcheck.pins(root, str(bad_manifest))},
        'binary': {'binary': lambda: hostcheck.binary(fake)},
        'machine': {'machine': lambda: hostcheck.machine(env)},
        'canary': {'canary': lambda: hostcheck.canary(host(root), payloads, work / 'canary-runs')}}
    if canary_expected:
        cases['canary_quantiles'] = {'canary': lambda: hostcheck.canary(host(root), proj / runtime.CANARY_DIR,
                                                                        work / 'canary-runs', expected='0' * 64)}
    results = {}
    for name, override in cases.items():
        record = _gate(root, work, override)
        target = name.split('_')[0]
        record['forced'] = target
        record['as_expected'] = (record['failed'] == target and not record['issuance_called']
                                 and record['not_run'] == list(hostcheck.ORDER[hostcheck.ORDER.index(target) + 1:]))
        results[name] = record
        print('forced', name, 'failed at', record['failed'], 'issued' if record['issuance_called'] else 'nothing issued',
              'as expected' if record['as_expected'] else 'NOT AS EXPECTED', flush=True)
    return results


def checks(args) -> int:
    check_blas()
    from cp17 import runtime
    from cp17.shadow import ShadowStates
    from .store import Store
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    record = {'schema': 'cp17c-checks-run-v1', 'gate': gate_or_stop(root, out)}
    record['forced'] = forced(root, out, args.canary_expected)
    proj = project()
    state = ShadowStates.from_dict(json.loads((proj / runtime.FREEZE_STATE).read_text()))
    store = Store.init(out / 'store-freeze-origin', state, date(2026, 9, 6),
                       f'the freeze-origin states of v4, v2 and B0 ({runtime.FREEZE_STATE})')
    loaded, head = store.state()
    record['freeze_origin_store'] = {'head': head, 'buffers_sha256': loaded.digest(),
                                     'state_file_sha256': sha256_file(proj / runtime.FREEZE_STATE)}
    dump(out / 'checks-run.json', record)
    ok = record['gate']['passed'] and all(r['as_expected'] for r in record['forced'].values())
    if args.stems:
        ok = parity_fits(root, out, Path(args.payloads), args.stems, record['gate']['passed']) == 0 and ok
    return 0 if ok else 1


# --- parity -------------------------------------------------------------------------------------------------------

def _record(run: dict, out: Path) -> dict:
    folder = Path(run['folder']) if run.get('folder') else None
    path = folder / 'run.json' if folder else None
    python = run.get('python') or {}
    result = python.get('result') or {}
    return {'record': str(path.relative_to(out)) if path and path.exists() else None,
            'record_sha256': sha256_file(path) if path and path.exists() else None,
            'errors': run.get('errors'), 'returncode': run.get('returncode'),
            'quantiles_sha256': result.get('quantiles_sha256'), 'result_payload_sha256': result.get('payload_sha256'),
            'total_ms': run.get('total_ms'), 'process_rss_peak_bytes': run.get('process_rss_peak_bytes'),
            'wasm_memory_bytes': run.get('wasm_memory_bytes'), 'driver_wall_ms': run.get('driver_wall_ms'),
            'node': run.get('node'), 'node_execpath': run.get('node_execpath'), 'pyodide': run.get('pyodide_version'),
            'blocked_non_local_requests': run.get('blocked_non_local_requests'),
            'executed_files': python.get('source_sha256'), 'stubs': python.get('stubs'),
            'runtime_python': python.get('python'), 'runtime_numpy': python.get('numpy')}


def parity(args) -> int:
    check_blas()
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    gate = gate_or_stop(root, out)
    return parity_fits(root, out, Path(args.payloads), args.stems, gate['passed'])


def parity_fits(root: Path, out: Path, payloads: Path, stems: str, gate_passed: bool) -> int:
    """A1's parity fits, each in a fresh issuing-host process, one at a time, only after a passed gate."""
    from cp17 import daily, dailyfit, issue
    record = {'schema': 'cp17c-parity-v1', 'gate_passed': gate_passed, 'stems': [], 'concurrency': 1}
    if not gate_passed:
        dump(out / 'parity.json', record)
        return 1
    hosted = host(root)
    for stem in stems.split(','):
        policy, day = stem.split('-', 1)[0], date.fromisoformat(stem.split('-', 1)[1])
        npz, meta_path = daily._paths(payloads, policy, day)
        arrays, meta = dailyfit.load(npz, meta_path)
        runs = {}

        def fit():
            run = runs['run'] = hosted.run(payloads, stem, out / 'runs')
            if run.get('errors'):
                raise dailyfit.FitFailure(run['errors'][0][-2000:])
            return run['python']['result']

        def measured():
            return {'wall_seconds': runs['run']['total_ms'] / 1000, 'peak_bytes': int(runs['run']['process_rss_peak_bytes'])}
        item = issue.attempt(policy, fit, day, arrays, meta, resources=measured)
        item.pop('result', None)
        item.pop('traceback', None)
        row = {'stem': stem, 'policy': policy, 'day': str(day), 'payload_sha256': meta.get('payload_sha256'),
               'attempt': item, **_record(runs.get('run', {}), out)}
        record['stems'].append(row)
        dump(out / 'parity.json', record)
        print(stem, item['status'], f"{(row['total_ms'] or 0) / 1000:.1f}s", f"{(row['process_rss_peak_bytes'] or 0) >> 20} MiB",
              (row['quantiles_sha256'] or '')[:16], flush=True)
    return 0 if all(r['attempt']['status'] == 'ok' for r in record['stems']) else 1


# --- shadow -------------------------------------------------------------------------------------------------------

def truth_from(path: Path):
    """The published prices the release reads: NaN unless every requested key is present."""
    import numpy as np
    import pandas as pd
    value = json.loads(Path(path).read_text())
    index = pd.DatetimeIndex(pd.to_datetime(value['keys'], utc=True)).as_unit('ns')
    y = np.asarray(value['y'], float)

    def at(keys):
        pos = index.get_indexer(pd.DatetimeIndex(pd.to_datetime(keys, utc=True)).as_unit('ns'))
        if (pos < 0).any():
            return np.full(len(keys), np.nan)
        return np.asarray(y[pos], float)
    return at


def day(args) -> int:
    """One delivery day in its own process, from the store and back to it."""
    check_blas()
    from cp17 import daily, issue
    from cp17.shadow import POLICIES
    from .store import Store
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    the_day = date.fromisoformat(args.day)
    store = Store(args.store)
    state, head = store.state()
    store.check_day(the_day)
    parent_state = state.digest()
    state.release(the_day, truth_from(args.truth))
    mine = {p: buffer_sha256(state.buffer(p)) for p in POLICIES}
    theirs = {}
    for p in POLICIES:
        meta = json.loads((Path(args.payloads) / f'{p}-{the_day}.json').read_text())
        theirs[p] = buffer_sha256(meta['b0_buffer' if p == 'B0' else 'h_buffer'])
    started = time.perf_counter()
    item = daily.issue_day(the_day, True, Path(args.payloads), host=host(root), out=out / 'runs', identity=identity(root),
                           parent_sha256=store.last_artifact_sha256(), data_refresh_utc=issue.issue_utc(the_day),
                           states=state)
    art = item['artifact']
    record = {'day': str(the_day), 'buffers_sha256': mine, 'payload_buffers_sha256': theirs,
              'buffers_equal_payload': {p: mine[p] == theirs[p] for p in POLICIES},
              'path': [(a['policy'], a['status']) for a in item['decision']['attempts']],
              'causes': {a['policy']: a.get('cause') for a in item['decision']['attempts'] if a['status'] != 'ok'},
              'resources': {a['policy']: a.get('resources') for a in item['decision']['attempts']},
              'display': item['decision']['display'], 'issued_policy': art['issued_policy'],
              'issued_quantiles_sha256': art.get('quantiles_sha256'), 'artifact_sha256': art['artifact_sha256'],
              'artifact_verifies': issue.verify_artifact(art) == [], 'shadows': item['shadows'],
              'intake': {p: s.get('intake') for p, s in item['shadows'].items()},
              'process_seconds': item['process_seconds'], 'process_seconds_sum': item['process_seconds_sum'],
              'driver_seconds': item['total_seconds'], 'issue_day_seconds': time.perf_counter() - started,
              'within_ceiling': all(s <= issue.WALL_LIMIT_SECONDS for s in item['process_seconds'].values()),
              'blocked_non_local_requests': item['blocked_non_local_requests'],
              'runs': {p: _record(r, out) for p, r in item['runs'].items()}}
    entry = store.record_day(the_day, state, art, record, parent_state)
    print(the_day, record['path'], record['intake'], all(record['buffers_equal_payload'].values()),
          f"{record['process_seconds_sum']:.0f}s", entry['entry_sha256'][:12], flush=True)
    return 0 if record['artifact_verifies'] and art['issued_policy'] else 1


def shadow(args) -> int:
    check_blas()
    from cp17.shadow import ShadowStates
    from .store import Store
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    first, last = date.fromisoformat(args.first), date.fromisoformat(args.last)
    gate = gate_or_stop(root, out)
    record = {'schema': 'cp17c-shadow-v1', 'first': str(first), 'last': str(last), 'gate_passed': gate['passed'],
              'start_state_sha256': sha256_file(args.state), 'truth_sha256': sha256_file(args.truth), 'days': {}}
    if not gate['passed']:
        dump(out / 'shadow.json', record)
        return 1
    state = ShadowStates.from_dict(json.loads(Path(args.state).read_text()))
    store_dir = out / 'store'
    Store.init(store_dir, state, first, f'{args.state} ({record["start_state_sha256"]})')
    status = 0
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    for d in days:
        started = time.perf_counter()
        done = subprocess.run([sys.executable, '-m', 'cp17c.run', '--root', str(root), 'day', '--store', str(store_dir),
                               '--day', str(d), '--payloads', str(args.payloads), '--truth', str(args.truth),
                               '--out', str(out)], text=True, capture_output=True)
        sys.stdout.write(done.stdout)
        sys.stdout.flush()
        record['days'][str(d)] = {'exit': done.returncode, 'process_seconds': time.perf_counter() - started,
                                  'stderr_tail': done.stderr[-1500:] if done.returncode else ''}
        dump(out / 'shadow.json', record)
        if done.returncode:
            status = 1
            break
    _, head = Store(store_dir).state()
    record['final'] = {'head': head, 'ledger_sha256': sha256_file(store_dir / 'ledger.jsonl')}
    dump(out / 'shadow.json', record)
    return status


def resume(args) -> int:
    """One gate, then each block's days from its carried store, one after another."""
    check_blas()
    from .store import Store
    root, out = Path(args.root).resolve(), Path(args.out).resolve()
    gate = gate_or_stop(root, out)
    status = 0 if gate['passed'] else 1
    for spec in args.block:
        name, carried, first, last = spec.split(':')
        first, last = date.fromisoformat(first), date.fromisoformat(last)
        folder = out / f'shadow-{name}'
        record = {'schema': 'cp17c-shadow-resume-v1', 'first': str(first), 'last': str(last), 'gate_passed': gate['passed'],
                  'truth_sha256': sha256_file(args.truth), 'days': {}}
        store_dir = folder / 'store'
        shutil.copytree(carried, store_dir)
        head = Store(store_dir).verify()                  # a carried store that does not verify issues nothing
        record['resumed_from'] = {'store': str(carried), 'head': head}
        if head['last_day'] != str(first - timedelta(days=1)):
            raise SystemExit(f'{carried}: its last day is {head["last_day"]}, not the day before {first}')
        if gate['passed']:
            for d in [first + timedelta(days=i) for i in range((last - first).days + 1)]:
                started = time.perf_counter()
                done = subprocess.run([sys.executable, '-m', 'cp17c.run', '--root', str(root), 'day', '--store', str(store_dir),
                                       '--day', str(d), '--payloads', str(args.payloads), '--truth', str(args.truth),
                                       '--out', str(folder)], text=True, capture_output=True)
                sys.stdout.write(done.stdout)
                sys.stdout.flush()
                record['days'][str(d)] = {'exit': done.returncode, 'process_seconds': time.perf_counter() - started,
                                          'stderr_tail': done.stderr[-1500:] if done.returncode else ''}
                if done.returncode:
                    status = 1
                    break
        _, final = Store(store_dir).state()
        record['final'] = {'head': final, 'ledger_sha256': sha256_file(store_dir / 'ledger.jsonl')}
        dump(folder / 'shadow.json', record)
        shutil.copyfile(out / 'checks.json', folder / 'checks.json')
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('node').add_argument('--out', required=True, type=Path)
    sub.add_parser('package').add_argument('--out', required=True, type=Path)
    c = sub.add_parser('checks')
    c.add_argument('--out', required=True, type=Path)
    c.add_argument('--canary-expected', action='store_true',
                   help='also force the canary to fail on its quantile hash (one more canary fit)')
    c.add_argument('--stems', help="A1's parity stems the run's gate admits")
    c.add_argument('--payloads', type=Path)
    p = sub.add_parser('parity')
    p.add_argument('--payloads', required=True, type=Path)
    p.add_argument('--stems', required=True)
    p.add_argument('--out', required=True, type=Path)
    for name in ('shadow', 'day'):
        s = sub.add_parser(name)
        s.add_argument('--payloads', required=True, type=Path)
        s.add_argument('--truth', required=True, type=Path)
        s.add_argument('--out', required=True, type=Path)
        if name == 'shadow':
            s.add_argument('--first', required=True)
            s.add_argument('--last', required=True)
            s.add_argument('--state', required=True, type=Path)
        else:
            s.add_argument('--day', required=True)
            s.add_argument('--store', required=True, type=Path)
    r = sub.add_parser('resume')
    r.add_argument('--payloads', required=True, type=Path)
    r.add_argument('--truth', required=True, type=Path)
    r.add_argument('--out', required=True, type=Path)
    r.add_argument('--block', action='append', required=True, help='name:carried store:first day:last day')
    args = ap.parse_args(argv)
    if hasattr(args, 'out'):
        args.out = Path(args.out).resolve()
        if args.cmd not in ('node', 'package'):
            args.out.mkdir(parents=True, exist_ok=True)
    return {'node': node, 'package': package, 'checks': checks, 'parity': parity, 'shadow': shadow, 'day': day,
            'resume': resume}[args.cmd](args)


if __name__ == '__main__':
    raise SystemExit(main())
