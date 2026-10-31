"""Collect one reproduction's outputs (`tools/reproduce.sh`) into the committed records.

- `reports/cp17/native-parity.json`: every origin's checks, hashes, payload identities and times (the full
  record, with every fitted result, stays local; its SHA-256 is recorded);
- `reports/cp17/host-parity.json`: the issuing host and the browsers against the native fits;
- `reports/cp17/issuance.json`: the three issuances, with the freeze-origin artifact;
- `reports/cp17/controls.json`, `scoring-check.json`, `reference-bundle.json`, `bundle-manifest.json`;
- `docs/track-b/evidence/cp-17/freeze-manifest.json` and `artifact-2026-09-06.json`.

    PYTHONPATH=src python -m cp17.tools.records --run <output folder>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPORTS = Path('reports/cp17')
EVIDENCE = Path('docs/track-b/evidence/cp-17')


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def native(run: Path) -> dict:
    full = run / 'native' / 'native-parity.json'
    value = json.loads(full.read_text())
    origins = {}
    for day, o in value['origins'].items():
        results = o['results']
        origins[day] = {'pass': o['pass'], 'checks': o['checks'], 'seconds': o['seconds'],
                        'ledger_file_sha256': o['ledger_file_sha256'],
                        'payload_sha256': {p: w['payload_sha256'] for p, w in o['payloads'].items()},
                        'payload_bytes': {p: w.get('npz_bytes', 0) + w['meta_bytes'] for p, w in o['payloads'].items()},
                        'quantiles_sha256': {p: r['quantiles_sha256'] for p, r in results.items()},
                        'fit_seconds': {p: r['fit_seconds'] for p, r in results.items()}}
    keep = ('schema', 'python', 'numpy', 'machine', 'platform', 'blas_threads', 'context_build_seconds', 'states',
            'standard', 'maxrss_bytes', 'total_seconds', 'status')
    return {**{k: value[k] for k in keep}, 'origins': origins,
            'full_record': {'path': '.local/artifacts/cp-17/final/native/native-parity.json', 'sha256': _sha(full)}}


def issuance(run: Path) -> dict:
    out = {}
    for path in sorted((run / 'issuance').glob('issuance-*.json')):
        item = json.loads(path.read_text())
        art = item['artifact']
        out[path.stem.replace('issuance-', '')] = {
            'day': item['day'], 'gfs_available': item['gfs_available'], 'comparison': item['comparison'],
            'artifact_sha256': art['artifact_sha256'], 'artifact_file_sha256': item['artifact_file_sha256'],
            'artifact_arrays_sha256': item['artifact_arrays_sha256'], 'artifact_verifies': item['artifact_verifies'],
            'issued_policy': art['issued_policy'], 'display': art['display'], 'attempts': art['attempts'],
            'resources': art['resources'], 'host_errors': item['host_errors'],
            'host': {'node': item['host']['node'], 'pyodide': item['host']['pyodide'],
                     'blocked_non_local_requests': item['host']['blocked_non_local_requests'],
                     'attempts': {p: {k: v for k, v in a.items() if k != 'executed_files'}
                                  for p, a in item['host']['attempts'].items()},
                     'executed_files_sha256': {p: hashlib.sha256(json.dumps(a['executed_files'], sort_keys=True).encode()).hexdigest()
                                               for p, a in item['host']['attempts'].items()}},
            'native_seconds': item['native_seconds']}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, type=Path)
    args = ap.parse_args(argv)
    run = args.run
    REPORTS.mkdir(parents=True, exist_ok=True)
    _write(REPORTS / 'native-parity.json', native(run))
    _write(REPORTS / 'issuance.json', issuance(run))
    for name in ('host-parity.json', 'controls.json', 'scoring-check.json', 'reference-bundle.json'):
        shutil.copyfile(run / name, REPORTS / name)
    shutil.copyfile(run / 'bundle' / 'src.json', REPORTS / 'bundle-manifest.json')
    shutil.copyfile(run / 'freeze-manifest.json', EVIDENCE / 'freeze-manifest.json')
    shutil.copyfile(run / 'issuance' / 'artifact-2026-09-06-gfs1.json', EVIDENCE / 'artifact-2026-09-06.json')
    print('collected', sorted(p.name for p in REPORTS.iterdir()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
