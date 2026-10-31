"""The public scratch repository's tree (`hrsi56/delu-pipeline-dryrun`): only what the dry runs need (run locally).

Built from a commit of the candidate, file by file, each recorded with its SHA-256:
- the files the pipeline, the frozen package and `refreeze verify` read: CP-17c's own list (`cp17c.inputs.tracked`:
  `src/`, `pyproject.toml`, `uv.lock`, CP-15's protocol, the computational inputs, the governance copies of CP-17b and
  CP-17c, CP-17c's manifest and freeze-origin artifact), plus the committed weather features the live-input builder
  reads;
- the guards the pipeline's commits and pushes run (`.githooks/`, `scripts/secret_guard.py`, `scripts/leak_scan.py`,
  `scripts/publication_guard.py` with the final page and build record it checks), and `.gitignore`;
- the workflow at `.github/workflows/daily.yml`;
- the dry run's material under `reports/cp18a/dryrun/`: A1's `k13.json`, the freeze-origin states and the canary
  payload (by the freeze manifest's hashes), the development state the store starts from, the recorded responses
  the live-input builder replays, and any store an earlier run carried out (`carried/<name>/`);
- a README and `scratch-files.json` (every file's SHA-256).

Pyodide's 23 files and the Node.js binary are not in the tree: each run retrieves them from their official URLs and
checks them by hash. Nothing dated after 2026-09-06 is shipped: the snapshot's and the weather features' rows pass the
boundary guard (`cp25.guard`), and every shipped file passes the round's red-line scan.

    PYTHONPATH=src python -m cp18.scratch --out <empty folder> --record <file> [--carried <name>=<store folder>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

EXTRA = ('.gitignore', '.githooks/pre-commit', '.githooks/commit-msg', '.githooks/pre-push', 'scripts/secret_guard.py',
         'scripts/leak_scan.py', 'scripts/publication_guard.py', 'docs/index.html', 'reports/cp3/pages_build.json',
         'reports/weather-ablation/weather-features.parquet', 'data/partitions.json', '.github/workflows/daily.yml')
DRY_RUN = 'reports/cp18a/dryrun/'
README = """# delu-pipeline-dryrun

Public scratch repository for CP-18a of the DE-LU day-ahead forecasting project
([hrsi56/delu-day-ahead-forecast](https://github.com/hrsi56/delu-day-ahead-forecast)): dry runs of the daily pipeline
around the frozen A1 package on GitHub-hosted `macos-26` runners, on development days through 2026-09-06 only.

It holds only what the runs need: the candidate's code and the files its freeze manifest verifies, the guards, the
workflow, A1's search state, the freeze-origin states, the canary's payload, a development state for the store and the
recorded responses the pipeline replays. `scratch-files.json` lists every file's SHA-256. Pyodide's files and the
Node.js binary are retrieved by each run from their official URLs and checked by hash.

It is not a publication: it has no Pages site, no secret and no variable, and its Git history is not the product.
Data: ENTSO-E Transparency Platform; Bundesnetzagentur | SMARD.de - CC BY 4.0. NOAA GFS (public domain).
"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, check=True).stdout


def tracked(root: Path) -> list[str]:
    from cp17c import inputs
    listed = set(_git(root, 'ls-files').splitlines())
    out = set(inputs.tracked(root)) | {p for p in EXTRA if p in listed} | {p for p in listed if p.startswith(DRY_RUN)}
    return sorted(out)


def boundary(root: Path) -> dict:
    import pandas as pd
    from cp25 import guard
    rows = {}
    for rel in ('data/snapshot.parquet', 'reports/weather-ablation/weather-features.parquet',
                'reports/development-inputs/weather-features-gaps.parquet'):
        frame = pd.read_parquet(root / rel, columns=['timestamp_utc'])
        rows[rel] = guard.admit_rows(pd.DatetimeIndex(pd.to_datetime(frame['timestamp_utc'], utc=True)))
    return {'rows_admitted': rows, 'boundary': str(guard.BOUNDARY)}


def payloads(source: Path, days: list[str], out: Path, files: dict) -> None:
    """Development payloads the development builder made (`cp17.tools.payloads`), each checked by its own identity and
    by the boundary guard, for a prebuilt-inputs dry run."""
    from datetime import date
    from cp17 import dailyfit, runtime
    from cp25 import guard
    for day in days:
        guard.admit_delivery_day(date.fromisoformat(day))
        for policy in ('A1', 'v4', 'v2', 'B0'):
            npz, meta = source / f'{policy}-{day}.npz', source / f'{policy}-{day}.json'
            arrays, value = dailyfit.load(npz if npz.exists() else None, meta)
            if value['payload_sha256'] != dailyfit.payload_digest(arrays, {k: v for k, v in value.items() if k != 'payload_sha256'}):
                raise ValueError(f'{meta}: the payload identity does not verify')
            for path in (meta, npz):
                if path.exists():
                    rel = f'{DRY_RUN}payloads/{path.name}'
                    (out / rel).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, out / rel)
                    files[rel] = {'sha256': runtime.sha256_file(out / rel), 'bytes': path.stat().st_size,
                                  'payload_sha256': value['payload_sha256']}


def build(root: Path, out: Path, carried: dict[str, Path] | None = None, prebuilt: tuple | None = None) -> dict:
    from cp17 import runtime
    root, out = Path(root).resolve(), Path(out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'{out} is not empty; the scratch tree is built once, from nothing')
    if _git(root, 'status', '--porcelain').strip():
        raise SystemExit('the candidate has uncommitted changes; the scratch tree is built from a commit')
    head = _git(root, 'rev-parse', 'HEAD').strip()
    files = {}
    for rel in tracked(root):
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / rel, target)
        if rel.startswith('.githooks/'):
            target.chmod(0o755)
        files[rel] = {'sha256': runtime.sha256_file(target), 'bytes': target.stat().st_size}
    for name, folder in (carried or {}).items():
        from .store import LiveStore
        head_ = LiveStore(Path(folder)).verify()
        for path in sorted(p for p in Path(folder).rglob('*') if p.is_file()):
            rel = f'{DRY_RUN}carried/{name}/{path.relative_to(folder)}'
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out / rel)
            files[rel] = {'sha256': runtime.sha256_file(out / rel), 'bytes': (out / rel).stat().st_size,
                          'carried_head': head_['entry_sha256']}
    if prebuilt:
        payloads(Path(prebuilt[0]), list(prebuilt[1]), out, files)
    (out / 'README.md').write_text(README)
    files['README.md'] = {'sha256': runtime.sha256_file(out / 'README.md'), 'bytes': (out / 'README.md').stat().st_size}
    (out / 'scratch-files.json').write_text(json.dumps({k: v['sha256'] for k, v in sorted(files.items())}, indent=1) + '\n')
    big = max(files.items(), key=lambda kv: kv[1]['bytes'])
    return {'schema': 'cp18a-scratch-tree-v1', 'candidate': head, 'files': len(files),
            'bytes': sum(v['bytes'] for v in files.values()), 'largest': {'path': big[0], 'bytes': big[1]['bytes']},
            'boundary': boundary(out), 'scratch_files_sha256': runtime.sha256_file(out / 'scratch-files.json'),
            'carried': {k: str(v) for k, v in (carried or {}).items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m cp18.scratch')
    ap.add_argument('--root', default=Path('.'), type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--record', required=True, type=Path)
    ap.add_argument('--carried', action='append', default=[], help='<name>=<store folder>')
    ap.add_argument('--payloads', help='<folder>:<day>,<day>,...: development payloads for a prebuilt dry run')
    args = ap.parse_args(argv)
    carried = dict((spec.split('=', 1)[0], Path(spec.split('=', 1)[1])) for spec in args.carried)
    prebuilt = None
    if args.payloads:
        folder, _, days = args.payloads.rpartition(':')
        prebuilt = (folder, days.split(','))
    record = build(args.root, args.out, carried, prebuilt)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print(record['files'], 'files', record['bytes'], 'bytes; largest', record['largest'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
