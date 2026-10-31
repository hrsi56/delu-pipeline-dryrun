"""What a runner needs, taken from the frozen host's own reference runs and checked by hash (run locally).

The reference is CP-17b's, under the frozen issuing host: `reports/cp17b/host-parity.json`,
`reports/cp17b/partial-days.json` (F1's day, 2026-03-30) and `reports/cp17b/shadow-run.json`. Each committed report
cites its source record by SHA-256; the source records and the runs under them are local material in
`.local/artifacts/cp-17b/runs/`. A runner receives exactly what those runs consumed, so a difference in its result
is the host's, not the inputs':

- **the frozen package's files** the CPython side imports (the package's 34, the driver `cp17/daily.py`, the
  Node.js runner), each against the freeze manifest's hash, and the package zip the runs loaded;
- **Pyodide 0.28.0's 23 files,** against the freeze manifest's hashes;
- **the payloads** of every run, each against its own identity and the identity the reference recorded;
- **the shadow states** at 2026-08-24 (CP-26's, rebuilt as `cp17.tools.exercise.start` rebuilds it) and at
  2026-08-30 (the reference's own host state there), and the published prices the release reads. The states are
  checked by a dry replay: the reference's own registered fits, released day by day, give the reference's
  recorded buffers at every origin and its final state.

Nothing after 2026-09-06 is read: every payload and price is a development day's.

    OPENBLAS_NUM_THREADS=1 ... PYTHONPATH=src python -m cp17cprobe.inputs --out <scratch tree> --record <inputs.json>
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import glob
import json
from pathlib import Path
import shutil

from .run import buffer_sha256, check_blas, dump, sha256_file, truth_from

FIRST, SPLIT, LAST = date(2026, 8, 24), date(2026, 8, 30), date(2026, 9, 6)
F1_DAY = date(2026, 3, 30)
BOUNDARY = date(2026, 9, 6)
RUNS = '.local/artifacts/cp-17b/runs'
REPORTS = {'host_parity': 'reports/cp17b/host-parity.json', 'partial_days': 'reports/cp17b/partial-days.json',
           'shadow_run': 'reports/cp17b/shadow-run.json', 'f1': 'reports/cp17b/f1-2026-03-30.json'}
#: The source records the committed reports cite, and where they are.
SOURCES = {'host_parity': f'{RUNS}/host-parity/host-parity.json', 'shadow_run': f'{RUNS}/shadow-host/shadow-run.json',
           'f1': f'{RUNS}/partial/f1-2026-03-30.json'}
FREEZE = 'docs/track-b/evidence/cp-17b/freeze-manifest.json'
ZIP = '.local/artifacts/cp-17b/bundle/src.zip'
PYODIDE = '.local/tools/pyodide-0.28.0'
PAYLOADS = {'parity': f'{RUNS}/native-parity/payloads', 'f1': f'{RUNS}/partial/payloads',
            'shadow': f'{RUNS}/shadow-host/payloads'}
RUN_FOLDERS = {'parity': f'{RUNS}/host-parity/nodes', 'f1': f'{RUNS}/partial/runs', 'shadow': f'{RUNS}/shadow-host/runs'}
SHIPPED = ('src/cp17cprobe/__init__.py', 'src/cp17cprobe/run.py')
PROJECT_FILES = ('pyproject.toml', 'uv.lock')
WORKFLOW = 'src/cp17cprobe/probe.yml'
README = """# delu-host-probe

Private scratch repository for CP-17c-probe of the DE-LU day-ahead forecasting project: whether a GitHub-hosted
runner reproduces the frozen issuing host (Pyodide 0.28.0 in Node.js v26.7.0) bit for bit.

It holds only what the runs need: the frozen package's files, the probe's harness (`src/cp17cprobe/`), the workflow,
the pinned Pyodide files, and payloads and shadow states for development delivery days (through 2026-09-06).
`inputs-files.json` lists every file's SHA-256, and each job checks them before it runs.

It creates no secret and is never made public. Deleting it is the Owner's decision.
"""


def _json(path: Path):
    return json.loads(Path(path).read_text())


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def _one(pattern: str) -> Path:
    found = glob.glob(pattern)
    if len(found) != 1:
        raise ValueError(f'{pattern}: {len(found)} matches, not one')
    return Path(found[0])


def references(project: Path, root: Path) -> dict:
    """Every reference run the comparison uses, bound to a committed report: {kind: {stem: row}}."""
    reports = {k: _json(root / v) for k, v in REPORTS.items()}
    cited = {'host_parity': reports['host_parity']['source'], 'shadow_run': reports['shadow_run']['host']['source'],
             'f1': reports['f1']['source']}
    for kind, rel in SOURCES.items():
        if cited[kind]['path'] != rel or sha256_file(project / rel) != cited[kind]['sha256']:
            raise ValueError(f'{rel}: not the source record {REPORTS[kind]} cites')
    out = {'parity': {}, 'f1': {}, 'shadow': {}}
    for row in reports['host_parity']['issuing_host_parity']:
        path = project / RUN_FOLDERS['parity'] / f"node-{row['stem']}.json"
        run = _json(path)
        result = run['python']['result']
        if result['quantiles_sha256'] != row['issued_quantiles_sha256']:
            raise ValueError(f"{path}: not the run host-parity.json records")
        out['parity'][row['stem']] = _row(project, path, run, 'reports/cp17b/host-parity.json')
    f1 = reports['partial_days'][str(F1_DAY)]['host']
    issued = {f1['artifact']['issued_policy']: f1['artifact']['quantiles_sha256'],
              **{p: s['quantiles_sha256'] for p, s in f1['shadows'].items()}}
    for policy, expected in issued.items():
        path = _one(str(project / RUN_FOLDERS['f1'] / f'{policy}-{F1_DAY}-*' / 'run.json'))
        run = _json(path)
        if run['python']['result']['quantiles_sha256'] != expected:
            raise ValueError(f'{path}: not the run partial-days.json records')
        out['f1'][f'{policy}-{F1_DAY}'] = _row(project, path, run, 'reports/cp17b/partial-days.json')
    source = _json(project / SOURCES['shadow_run'])
    for day in _days(FIRST, LAST):
        record = source['days'][str(day)]
        for policy in ('A1', 'v4', 'v2', 'B0'):
            path = _one(str(project / RUN_FOLDERS['shadow'] / f'{policy}-{day}-*' / 'run.json'))
            run = _json(path)
            result = run['python']['result']
            expected = record['shadows'][policy]['quantiles_sha256'] if policy != 'A1' else None
            if result['payload_sha256'] != record['payload_sha256'][policy] or expected not in (None, result['quantiles_sha256']) \
                    or run['python']['source_sha256'] != record['executed_files'][policy]:
                raise ValueError(f'{path}: not the run shadow-run.json records')
            out['shadow'][f'{policy}-{day}'] = _row(project, path, run, SOURCES['shadow_run'])
    return {'reports_sha256': {k: sha256_file(root / v) for k, v in REPORTS.items()},
            'sources_sha256': {k: sha256_file(project / v) for k, v in SOURCES.items()}, 'runs': out,
            'shadow_buffers_sha256': {str(d): {p: v['sha256'] for p, v in source['days'][str(d)]['buffers'].items()}
                                      for d in _days(FIRST, LAST)},
            'shadow_final_sha256': source['final']['buffers_sha256'],
            'shadow_start_sha256': source['start']['buffers_sha256']}


def _row(project: Path, path: Path, run: dict, bound_by: str) -> dict:
    result = run['python']['result']
    return {'record': str(path.relative_to(project)), 'record_sha256': sha256_file(path), 'bound_by': bound_by,
            'quantiles_sha256': result['quantiles_sha256'], 'payload_sha256': result.get('payload_sha256'),
            'total_ms': run['total_ms'], 'process_rss_peak_bytes': run['process_rss_peak_bytes'],
            'wasm_memory_bytes': run.get('wasm_memory_bytes'), 'node': run['node'], 'node_execpath': run.get('node_execpath')}


def _copy(origin: Path, target: Path, manifest: dict, tree: Path, **provenance) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, target)
    manifest[str(target.relative_to(tree))] = {'sha256': sha256_file(target), 'bytes': target.stat().st_size, **provenance}


def payload_folder(project: Path, kind: str, days: list[date], tree: Path, manifest: dict, expected: dict) -> None:
    """Every payload file of the days, each verified against its own identity and the reference's."""
    from cp17 import dailyfit
    source = project / PAYLOADS[kind]
    names = sorted(p.name for p in source.iterdir() if any(p.name.endswith(f'-{d}.json') or p.name.endswith(f'-{d}.npz')
                                                            for d in days))
    for name in names:
        if name.endswith('.json'):
            policy, day = name.removesuffix('.json').split('-', 1)
            if date.fromisoformat(day) > BOUNDARY:
                raise ValueError(f'{name}: after the development boundary')
            npz = source / f'{policy}-{day}.npz'
            arrays, meta = dailyfit.load(npz if npz.exists() else None, source / name)
            body = {k: v for k, v in meta.items() if k != 'payload_sha256'}
            if meta.get('payload_sha256') != dailyfit.payload_digest(arrays, body):
                raise ValueError(f'{name}: the payload differs from its recorded identity')
            stem = f'{policy}-{day}'
            if stem in expected and meta['payload_sha256'] != expected[stem]:
                raise ValueError(f'{name}: not the payload the reference used')
        _copy(source / name, tree / 'payloads' / kind / name, manifest, tree, source=f'{PAYLOADS[kind]}/{name}')


def start_state(root: Path):
    """CP-26's states at 2026-08-24, after its release, as `cp17.tools.exercise.start` rebuilds them; the
    published prices of every delivery day the release reads (2026-08-22 to 2026-09-04)."""
    import numpy as np
    from cp17.tools import exercise, payloads
    ctx = payloads.context(root)
    state, record = exercise.start(root, ctx, FIRST)
    index = ctx.data.index
    berlin = index.tz_convert('Europe/Berlin').date
    keep = np.flatnonzero((berlin >= FIRST - timedelta(days=2)) & (berlin <= LAST - timedelta(days=2)))
    truth = {'source': 'published day-ahead prices, through CP-26\'s loaders (cp17.tools.payloads.context)',
             'first_day': str(FIRST - timedelta(days=2)), 'last_day': str(LAST - timedelta(days=2)),
             'keys': [t.isoformat() for t in index[keep]], 'y': [float(v) for v in np.asarray(ctx.data.y, float)[keep]]}
    return state, record, truth


def replay(state, truth, project: Path, refs: dict, split: date):
    """The reference's own host registrations, released day by day from the start state: the buffers at every
    origin and the final state must be the reference's. Returns the state just before `split`'s release, as a
    runner reads it from its file."""
    from cp17.shadow import POLICIES, ShadowStates
    checks, before_split = {}, None
    for day in _days(FIRST, LAST):
        if day == split:
            # The late days replay from the state as a runner reads it, so the file's round trip is checked too.
            before_split = json.loads(json.dumps(state.to_dict(), allow_nan=False))
            state = ShadowStates.from_dict(before_split)
        state.release(day, truth)
        mine = {p: buffer_sha256(state.buffer(p)) for p in POLICIES}
        checks[str(day)] = mine == refs['shadow_buffers_sha256'][str(day)]
        for p in POLICIES:
            run = _json(project / refs['runs']['shadow'][f'{p}-{day}']['record'])
            if state.issue(p, day, run['python']['result']) != 'registered':
                raise ValueError(f'{p} {day}: the reference registration does not replay')
    final = state.digest() == refs['shadow_final_sha256']
    if not all(checks.values()) or not final:
        raise ValueError(f'the dry replay does not give the reference buffers: {checks}, final {final}')
    return before_split, checks


def build(root: Path, project: Path, tree: Path) -> dict:
    check_blas()
    from cp17.shadow import ShadowStates
    if tree.exists():
        raise SystemExit(f'{tree} exists; the scratch tree is built once, from nothing')
    refs = references(project, root)
    freeze = _json(root / FREEZE)
    code, host = freeze['policy_identity']['code'], freeze['policy_identity']['issuing_host']
    manifest = {}
    frozen = {**code['files_sha256'], **{k: v for k, v in code['runner_files_sha256'].items() if k.endswith('.mjs')},
              **code['pipeline_files_sha256']}
    for rel, expected in sorted(frozen.items()):
        if sha256_file(root / 'src' / rel) != expected:
            raise ValueError(f'src/{rel}: not the frozen file')
        _copy(root / 'src' / rel, tree / 'src' / rel, manifest, tree, frozen_sha256=expected)
    for rel in SHIPPED + PROJECT_FILES:
        _copy(root / rel, tree / rel, manifest, tree, source=rel)
    if sha256_file(project / ZIP) != code['zip_sha256']:
        raise ValueError(f'{ZIP}: not the frozen package zip')
    _copy(project / ZIP, tree / 'bundle' / 'src.zip', manifest, tree, frozen_sha256=code['zip_sha256'])
    for name, expected in sorted(host['pyodide']['files_sha256'].items()):
        _copy(project / PYODIDE / name, tree / 'pyodide' / name, manifest, tree, frozen_sha256=expected)
        if manifest[f'pyodide/{name}']['sha256'] != expected:
            raise ValueError(f'{PYODIDE}/{name}: not the frozen file')
    dump(tree / 'pyodide-sha256.json', host['pyodide']['files_sha256'])
    expected = {kind: {stem: row['payload_sha256'] for stem, row in rows.items()} for kind, rows in refs['runs'].items()}
    parity_days = sorted({date.fromisoformat(s.split('-', 1)[1]) for s in refs['runs']['parity']})
    payload_folder(project, 'parity', parity_days, tree, manifest, expected['parity'])
    payload_folder(project, 'f1', [F1_DAY], tree, manifest, expected['f1'])
    payload_folder(project, 'shadow', _days(FIRST, LAST), tree, manifest, expected['shadow'])
    state, start_record, truth = start_state(root)
    if start_record['buffers_sha256'] != refs['shadow_start_sha256']:
        raise ValueError('the rebuilt start state is not the reference\'s')
    dump(tree / 'states' / 'truth.json', truth)
    dump(tree / 'states' / f'start-{FIRST}.json', state.to_dict())
    at = truth_from(tree / 'states' / 'truth.json')
    split_value, checks = replay(ShadowStates.from_dict(json.loads(json.dumps(state.to_dict()))), at, project, refs, SPLIT)
    split_state = ShadowStates.from_dict(split_value)
    dump(tree / 'states' / f'start-{SPLIT}.json', split_value)
    for name in ('truth.json', f'start-{FIRST}.json', f'start-{SPLIT}.json'):
        manifest[f'states/{name}'] = {'sha256': sha256_file(tree / 'states' / name),
                                      'bytes': (tree / 'states' / name).stat().st_size}
    manifest['pyodide-sha256.json'] = {'sha256': sha256_file(tree / 'pyodide-sha256.json'),
                                       'bytes': (tree / 'pyodide-sha256.json').stat().st_size}
    _copy(root / WORKFLOW, tree / '.github' / 'workflows' / 'probe.yml', manifest, tree, source=WORKFLOW)
    (tree / 'README.md').write_text(README)
    manifest['README.md'] = {'sha256': sha256_file(tree / 'README.md'), 'bytes': (tree / 'README.md').stat().st_size}
    # The runner checks every file but this list itself against it.
    dump(tree / 'inputs-files.json', {rel: row['sha256'] for rel, row in sorted(manifest.items())})
    return {'schema': 'cp17c-probe-inputs-v1', 'files': manifest, 'references': refs,
            'start': {str(FIRST): {**start_record, 'state_file_sha256': manifest[f'states/start-{FIRST}.json']['sha256']},
                      str(SPLIT): {'state_file_sha256': manifest[f'states/start-{SPLIT}.json']['sha256'],
                                   'digest': split_state.digest(),
                                   'source': "the reference's host state before 2026-08-30's release, by the dry replay"}},
            'dry_replay_buffers_equal_reference': checks,
            'inputs_files_sha256': sha256_file(tree / 'inputs-files.json'),
            'bytes': sum(v['bytes'] for v in manifest.values()), 'largest_file_bytes': max(v['bytes'] for v in manifest.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--project', type=Path, help='the primary checkout holding .local/ (default: from git)')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--record', required=True, type=Path)
    args = ap.parse_args(argv)
    from cp17.tools import ledger
    project = args.project or ledger.project_root(args.root)
    value = build(args.root.resolve(), Path(project), args.out)
    dump(args.record, value)
    print(len(value['files']), 'files', value['bytes'], 'bytes; largest', value['largest_file_bytes'],
          '; dry replay', all(value['dry_replay_buffers_equal_reference'].values()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
