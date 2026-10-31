"""The private scratch repository's tree: exactly what the runs need, each file checked by hash (run locally).

The runner's checkout mirrors the project: every path the freeze manifest names is at the same project-relative path,
so `refreeze verify` and the H2 checks read the same files on the runner and on the Owner's Mac. The tree holds:

- **the candidate's tracked files the runs and `refreeze verify` read:** `src/`, `pyproject.toml`, `uv.lock`, the
  computational inputs, CP-15's protocol, the governance copies (CP-17b's and CP-17c's), the payload schemas, CP-17b's
  superseded records, and CP-17c's manifest and freeze-origin artifact;
- **the issuing host's local files:** Pyodide 0.28.0's 23 pinned files at `cp17.runtime.PYODIDE_DIR`. The Node.js
  binary is not in the tree: each run retrieves it (`cp17.runtime.retrieve`);
- **the states' route:** A1's `k13.json` at `cp17.runtime.K13`, the freeze-origin states and the canary's payload under
  `cp17.runtime.STATE_DIR`;
- **development payloads:** A1 at 2025-10-26 and 2026-03-29 (CP-17b's parity payloads), and every policy's payload of
  the 14 shadow days (CP-17b's shadow run's), each checked by its identity against CP-17b's records;
- **the shadow blocks' start states** at 2026-08-24 (CP-26's, rebuilt as CP-17b's shadow run started) and before
  2026-08-31's release (by a dry replay of CP-17b's host registrations, which must give CP-17b's buffers at every
  origin), and the published prices the releases read (2026-08-22 to 2026-09-04);
- **with `--stores`,** the state stores an earlier run's shadow blocks left, from its downloaded artifact, each file
  checked against that job's `results-sha256.txt`, at `states/store-<block>/`: the states' route between runs;
- the workflow at `.github/workflows/refreeze.yml`, a README, and `inputs-files.json` (every file's SHA-256).

Nothing dated after 2026-09-06 is shipped: every payload's day, every price, and the rows of the snapshot and the
weather features pass the boundary guard (`cp25.guard`), which fails closed.

    OPENBLAS_NUM_THREADS=1 ... PYTHONPATH=src python -m cp17c.inputs --out <scratch tree> --record <inputs.json>
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import shutil
import subprocess

from .run import buffer_sha256, check_blas, dump, sha256_file

FIRST, SPLIT, LAST = date(2026, 8, 24), date(2026, 8, 31), date(2026, 9, 6)
PARITY_STEMS = ('A1-2025-10-26', 'A1-2026-03-29')
RUNS = '.local/artifacts/cp-17b/runs'
PAYLOADS = {'parity': f'{RUNS}/native-parity/payloads', 'shadow': f'{RUNS}/shadow-host/payloads'}
WORKFLOW = 'src/cp17c/refreeze.yml'
TRACKED_PREFIXES = ('src/', 'docs/track-b/evidence/cp-17b/pinned/', 'docs/track-b/evidence/cp-17b/schemas/',
                    'docs/track-b/evidence/cp-17c/pinned/')
README = """# delu-host-refreeze

Private scratch repository for CP-17c of the DE-LU day-ahead forecasting project: A1 re-frozen under the issuing
host that r19 §24.13 pins (Pyodide 0.28.0 in the official Node.js v26.7.0 darwin-arm64 binary), shown on GitHub-hosted
`macos-26` runners.

It holds only what the runs need: the candidate's code and the files its freeze manifest verifies, the harness
(`src/cp17c/`), the workflow, the pinned Pyodide files, A1's search state, the freeze-origin states, the canary's
payload, and development payloads and states for delivery days through 2026-09-06. `inputs-files.json` lists every
file's SHA-256, and each job checks them before it runs.

It creates no secret and no variable, and is never made public. Deleting it is the Owner's decision.
"""


def _json(path: Path):
    return json.loads(Path(path).read_text())


def _days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def tracked(root: Path) -> list[str]:
    """The candidate's tracked files the runs and `refreeze verify` read."""
    from cp17.tools import refreeze, refreeze17c
    listed = subprocess.run(['git', '-C', str(root), 'ls-files'], capture_output=True, text=True, check=True).stdout.split('\n')
    out = {p for p in listed if p.startswith(TRACKED_PREFIXES)}
    out |= {'pyproject.toml', 'uv.lock', refreeze.PROTOCOL, *refreeze.COMPUTATIONAL,
            refreeze17c.MANIFEST_SCHEMA and 'docs/track-b/evidence/cp-17c/freeze-manifest.json', refreeze17c.ARTIFACT_PATH,
            *(p for p, _ in refreeze17c.CP17B.values())}
    return sorted(p for p in out if p)


def boundary(root: Path, truth: dict) -> dict:
    """The red line on what is shipped: the snapshot's rows, the prices and every payload day pass the guard."""
    import pandas as pd
    from cp25 import guard
    try:
        rows = {rel: guard.admit_rows(pd.DatetimeIndex(pd.read_parquet(root / rel, columns=['timestamp_utc'])['timestamp_utc']))
                for rel in ('data/snapshot.parquet', 'reports/weather-ablation/weather-features.parquet')}
        prices = guard.admit_rows(pd.to_datetime(truth['keys'], utc=True))
    except guard.BoundaryRefused:
        raise SystemExit('the boundary guard refused a shipped row; nothing is built') from None
    return {'rows_admitted': rows, 'price_rows_admitted': prices, 'boundary': str(guard.BOUNDARY)}


def _copy(origin: Path, tree: Path, rel: str, files: dict, **provenance) -> None:
    target = tree / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(origin, target)
    files[rel] = {'sha256': sha256_file(target), 'bytes': target.stat().st_size, **provenance}


def payloads(project: Path, kind: str, stems: list[str], tree: Path, files: dict, expected: dict) -> None:
    """Each payload, verified against its own identity and the identity CP-17b's reference run used."""
    from cp17 import dailyfit
    from cp25 import guard
    source = project / PAYLOADS[kind]
    for stem in stems:
        policy, day = stem.split('-', 1)
        guard.admit_delivery_day(date.fromisoformat(day))
        npz = source / f'{stem}.npz'
        arrays, meta = dailyfit.load(npz if npz.exists() else None, source / f'{stem}.json')
        body = {k: v for k, v in meta.items() if k != 'payload_sha256'}
        if meta.get('payload_sha256') != dailyfit.payload_digest(arrays, body) or meta['payload_sha256'] != expected[stem]:
            raise ValueError(f'{stem}: not the payload CP-17b\'s reference run used')
        for path in (source / f'{stem}.json', npz):
            if path.exists():
                _copy(path, tree, f'payloads/{kind}/{path.name}', files, source=f'{PAYLOADS[kind]}/{path.name}')


def carried_stores(artifact: Path, tree: Path, files: dict) -> dict:
    """Each block's store from a downloaded `shadows` artifact, every file against the job's own hashes."""
    listed = {}
    for line in (artifact / 'results-sha256.txt').read_text().splitlines():
        digest, rel = line.split(None, 1)
        listed[rel.strip().removeprefix('./')] = digest
    out = {}
    for block in ('late', 'early'):
        prefix = f'shadow-{block}/store/'
        names = sorted(rel for rel in listed if rel.startswith(prefix))
        if not names:
            raise ValueError(f'{artifact}: no store for the {block} block')
        for rel in names:
            if sha256_file(artifact / rel) != listed[rel]:
                raise ValueError(f'{artifact / rel}: not the file its job hashed')
            _copy(artifact / rel, tree, f'states/store-{block}/{rel.removeprefix(prefix)}', files,
                  source=f'{artifact.parent.name}/{artifact.name}/{rel}')
        from .store import Store
        out[block] = Store(tree / f'states/store-{block}').verify()
    return out


def build(root: Path, project: Path, tree: Path, stores: Path | None = None) -> dict:
    check_blas()
    from cp17 import runtime
    from cp17.shadow import ShadowStates
    from cp17cprobe import inputs as probe
    if tree.exists():
        raise SystemExit(f'{tree} exists; the scratch tree is built once, from nothing')
    if subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True).stdout.strip():
        raise SystemExit('the candidate has uncommitted changes; the scratch tree is built from a commit')
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    files = {}
    for rel in tracked(root):
        _copy(root / rel, tree, rel, files, source=f'candidate {head[:12]}')
    manifest = _json(root / 'docs/track-b/evidence/cp-17c/freeze-manifest.json')
    identity = manifest['policy_identity']
    for name, expected in sorted(identity['issuing_host']['pyodide']['files_sha256'].items()):
        _copy(project / runtime.PYODIDE_DIR / name, tree, f'{runtime.PYODIDE_DIR}/{name}', files, frozen_sha256=expected)
        if files[f'{runtime.PYODIDE_DIR}/{name}']['sha256'] != expected:
            raise ValueError(f'{runtime.PYODIDE_DIR}/{name}: not the pinned file')
    local = {identity['state_route']['k13']['path']: identity['state_route']['k13']['file_sha256'],
             identity['state_route']['freeze_origin_state']['path']: identity['state_route']['freeze_origin_state']['file_sha256'],
             **identity['state_route']['canary_payload']['files_sha256']}
    for rel, expected in sorted(local.items()):
        _copy(project / rel, tree, rel, files, frozen_sha256=expected)
        if files[rel]['sha256'] != expected:
            raise ValueError(f'{rel}: not the frozen state file')
    refs = probe.references(project, root)
    expected = {kind: {stem: row['payload_sha256'] for stem, row in rows.items()} for kind, rows in refs['runs'].items()}
    payloads(project, 'parity', list(PARITY_STEMS), tree, files, expected['parity'])
    payloads(project, 'shadow', [f'{p}-{d}' for d in _days(FIRST, LAST) for p in ('A1', 'v4', 'v2', 'B0')], tree, files,
             expected['shadow'])
    state, start_record, truth = probe.start_state(root)
    if start_record['buffers_sha256'] != refs['shadow_start_sha256']:
        raise ValueError("the rebuilt start state is not CP-17b's")
    record_boundary = boundary(root, truth)
    dump(tree / 'states' / 'truth.json', truth)
    dump(tree / 'states' / f'start-{FIRST}.json', state.to_dict())
    at = probe.truth_from(tree / 'states' / 'truth.json')
    split_value, checks = probe.replay(ShadowStates.from_dict(json.loads(json.dumps(state.to_dict()))), at, project,
                                       refs, SPLIT)
    dump(tree / 'states' / f'start-{SPLIT}.json', split_value)
    for name in ('truth.json', f'start-{FIRST}.json', f'start-{SPLIT}.json'):
        files[f'states/{name}'] = {'sha256': sha256_file(tree / 'states' / name), 'bytes': (tree / 'states' / name).stat().st_size}
    carried = carried_stores(stores, tree, files) if stores is not None else None
    _copy(root / WORKFLOW, tree, '.github/workflows/refreeze.yml', files, source=WORKFLOW)
    (tree / 'README.md').write_text(README)
    files['README.md'] = {'sha256': sha256_file(tree / 'README.md'), 'bytes': (tree / 'README.md').stat().st_size}
    dump(tree / 'inputs-files.json', {rel: row['sha256'] for rel, row in sorted(files.items())})
    return {'schema': 'cp17c-inputs-v1', 'candidate': head, 'files': files, 'references': refs,
            'start': {str(FIRST): {**start_record, 'state_file_sha256': files[f'states/start-{FIRST}.json']['sha256']},
                      str(SPLIT): {'state_file_sha256': files[f'states/start-{SPLIT}.json']['sha256'],
                                   'digest': ShadowStates.from_dict(split_value).digest(),
                                   'source': "CP-17b's host state before 2026-08-31's release, by the dry replay"}},
            'dry_replay_buffers_equal_reference': checks, 'boundary': record_boundary, 'carried_stores': carried,
            'inputs_files_sha256': sha256_file(tree / 'inputs-files.json'),
            'bytes': sum(v['bytes'] for v in files.values()), 'largest_file_bytes': max(v['bytes'] for v in files.values())}


def verify(tree: Path) -> dict:
    """On the runner: every shipped file against `inputs-files.json`."""
    listed = _json(tree / 'inputs-files.json')
    problems = [rel for rel, h in sorted(listed.items()) if not (tree / rel).is_file() or sha256_file(tree / rel) != h]
    return {'files': len(listed), 'problems': problems, 'verified': not problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--out', required=True, type=Path)
    b.add_argument('--record', required=True, type=Path)
    b.add_argument('--stores', type=Path, help="a downloaded `shadows` artifact whose stores the next run resumes")
    v = sub.add_parser('verify')
    v.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'verify':
        value = verify(args.root)
        dump(args.out, value)
        print(value['files'], 'inputs', 'verified' if value['verified'] else f"NOT verified: {value['problems'][:5]}")
        return 0 if value['verified'] else 1
    from cp17.tools import ledger
    value = build(args.root.resolve(), ledger.project_root(args.root), args.out, args.stores)
    dump(args.record, value)
    print(len(value['files']), 'files', value['bytes'], 'bytes; largest', value['largest_file_bytes'],
          '; dry replay', all(value['dry_replay_buffers_equal_reference'].values()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
