"""The product's daily issuance at one origin, under the issuing host and natively (items 3 and 4).

- **The freeze-origin artifact:** delivery 2026-09-06, GFS available. A1 issues under the issuing host.
- **The failure policy, exercised for real:** 2026-09-06 with GFS declared unavailable (A1 skipped, v2
  issues), and 2022-12-04, one of the 18 development dates on which a DDNN-2 member stopped with a nonfinite
  loss (A1 fails, v4 issues).

**Under the issuing host,** each attempt the chain reaches runs in its own fresh Node.js process (Pyodide, the
package's `host/driver.py` in fit mode), so that its wall time (the whole process, runtime load included) and
its peak resident memory are its own. The package's `cp17.issue` then judges every attempt on the success
definition's five criteria, criterion 5 on those measurements, and walks the frozen chain (`decide`); the
issued artifact is `cp17.issue.artifact`. **Natively,** the same chain is taken from the same payloads with the
fits run in-process and criterion 5 not judged. The two paths are compared: the chain walked, each attempt's
status, the issued policy, and the issued quantiles by `cp17.retrain.compare`.

    OPENBLAS_NUM_THREADS=1 ... PYTHONPATH=src python -m cp17.tools.issuance --day 2026-09-06 --gfs 1 \
        --payloads <dir> --bundle <src.zip> --identity <policy-identity.json> --out <dir>
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .. import dailyfit, issue, retrain
from . import ledger, payloads, states
from .bundle import PACKAGES
from .hosts import NODE, RUNNER, node_identity


def build_payloads(root: Path, day: date, folder: Path) -> dict:
    ctx = payloads.context(root)
    rebuilt = states.rebuild(root, ctx, [day])['buffers']
    built = {'A1': payloads.a1_payload(ctx, day, payloads.ensemble(ledger.inputs_dir(root), day)),
             'v4': payloads.v4_payload(ctx, day, rebuilt['v4'][day]), 'v2': payloads.v2_payload(ctx, day, rebuilt['v2'][day]),
             'B0': payloads.b0_payload(ctx, day, rebuilt['B0'][day])}
    return {p: payloads.write(a, m, folder, f'{p}-{day}') for p, (a, m) in built.items()}


def native_issuance(folder: Path, day: date, gfs: bool) -> dict:
    def attempts(policy):
        npz, meta = folder / f'{policy}-{day}.npz', folder / f'{policy}-{day}.json'
        arrays, value = dailyfit.load(npz, meta)
        return issue.attempt(policy, lambda: dailyfit.run(policy, npz, meta, attribution=True), day, arrays, value)
    started = time.perf_counter()
    decision = issue.decide(day, gfs, attempts)
    decision['seconds'] = time.perf_counter() - started
    return decision


def host_fit(pyodide: Path, bundle: Path, folder: Path, stem: str, out: Path) -> dict:
    """One attempt's fit in a fresh issuing-host process, with attribution and the fitted arrays exported."""
    import os
    target = out / f'node-attempt-{stem}.json'
    env = {**os.environ, 'CP17_EXPORT': '1'}
    subprocess.run([NODE, str(RUNNER), 'fit', str(pyodide), str(bundle), str(folder), stem, '1', ','.join(PACKAGES),
                    str(target)], capture_output=True, text=True, env=env)
    return json.loads(target.read_text())


def host_issuance(pyodide: Path, bundle: Path, folder: Path, day: date, gfs: bool, out: Path) -> tuple[dict, dict]:
    runs = {}

    def attempts(policy):
        stem = f'{policy}-{day}'
        npz, meta = folder / f'{stem}.npz', folder / f'{stem}.json'
        arrays, value = dailyfit.load(npz, meta)
        run = runs[policy] = host_fit(pyodide, bundle, folder, stem, out)

        def fit():
            if run.get('errors'):
                raise dailyfit.FitFailure(run['errors'][0][-2000:])
            return run['python']['result']
        return issue.attempt(policy, fit, day, arrays, value,
                             resources=lambda: {'wall_seconds': run['total_ms'] / 1000,
                                                'peak_bytes': int(run['process_rss_peak_bytes'])})
    return issue.decide(day, gfs, attempts), runs


def compare(native: dict, hosted: dict) -> dict:
    path_n = [(a['policy'], a['status']) for a in native['attempts']]
    path_h = [(a['policy'], a['status']) for a in hosted['attempts']]
    out = {'native_path': path_n, 'host_path': path_h, 'same_path': path_n == path_h,
           'issued_policy': hosted['issued_policy'], 'same_issued_policy': native['issued_policy'] == hosted['issued_policy'],
           'native_display': native['display'], 'host_display': hosted['display'],
           'causes': {a['policy']: a.get('cause') for a in hosted['attempts'] if a['status'] != 'ok'},
           'native_causes': {a['policy']: a.get('cause') for a in native['attempts'] if a['status'] != 'ok'}}
    n_issued = next((a for a in native['attempts'] if a['policy'] == native['issued_policy']), None)
    h_issued = next((a for a in hosted['attempts'] if a['policy'] == hosted['issued_policy']), None)
    if n_issued and h_issued:
        c = retrain.compare(hosted['issued_policy'], n_issued['result'], h_issued['result'])
        out.update(status_vs_native=c['status'], max_abs_eur_vs_native=c['deviation'].get('max_abs_eur'),
                   selections_match_native=c['selections_match'])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--day', required=True)
    ap.add_argument('--gfs', required=True, choices=('0', '1'))
    ap.add_argument('--payloads', required=True, type=Path)
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--identity', required=True, type=Path)
    ap.add_argument('--pyodide', type=Path, default=Path('/Users/djourno/Downloads/PJM/.local/tools/pyodide-0.28.0'))
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    day, gfs = date.fromisoformat(args.day), args.gfs == '1'
    args.out.mkdir(parents=True, exist_ok=True)
    built = None
    if not (args.payloads / f'A1-{day}.json').exists():
        built = build_payloads(args.root, day, args.payloads)
    identity = json.loads(args.identity.read_text())
    from .freeze import digest
    short = args.out / 'host-identity.json'
    short.write_text(json.dumps({'policy_identity_sha256': digest(identity),
                                 'package_sha256': identity['code']['package_sha256']}, sort_keys=True))
    native = native_issuance(args.payloads, day, gfs)
    hosted, runs = host_issuance(args.pyodide, args.bundle, args.payloads, day, gfs, args.out)
    failed_runs = {p: r['errors'] for p, r in runs.items() if r.get('errors')}
    payloads_used = {p: {'payload_sha256': json.loads((args.payloads / f'{p}-{day}.json').read_text())['payload_sha256']}
                     for p in runs}
    artifact = issue.artifact(hosted, identity=json.loads(short.read_text()), payloads=payloads_used, parent_sha256=None,
                              data_refresh_utc=issue.issue_utc(day),
                              host={'runtime': 'pyodide', 'pyodide': next(iter(runs.values())).get('pyodide_version'),
                                    'node': next(iter(runs.values()))['node'], 'process': 'one fresh process per attempt'})
    art_path = args.out / f'artifact-{day}-gfs{int(gfs)}.json'
    art_path.write_text(json.dumps(artifact, sort_keys=True, allow_nan=False) + '\n')
    arrays = args.out / f"node-attempt-{hosted['issued_policy']}-{day}-arrays.npz" if hosted['issued_policy'] else None
    record = {'day': str(day), 'gfs_available': gfs, 'built_payloads': built is not None,
              'artifact': artifact, 'artifact_file_sha256': hashlib.sha256(art_path.read_bytes()).hexdigest(),
              'artifact_arrays_sha256': hashlib.sha256(arrays.read_bytes()).hexdigest() if arrays and arrays.exists() else None,
              'artifact_verifies': issue.verify_artifact(artifact) == [], 'comparison': compare(native, hosted),
              'host_errors': failed_runs,
              'host': {'node': node_identity(), 'pyodide': next(iter(runs.values())).get('pyodide_version'),
                       'attempts': {p: {'total_seconds': r['total_ms'] / 1000, 'wasm_memory_bytes': r.get('wasm_memory_bytes'),
                                        'process_rss_peak_bytes': r['process_rss_peak_bytes'],
                                        'blocked_non_local_requests': r['blocked_non_local_requests'],
                                        'executed_files': (r.get('python') or {}).get('source_sha256'),
                                        'stubs': (r.get('python') or {}).get('stubs')} for p, r in runs.items()},
                       'blocked_non_local_requests': sorted({u for r in runs.values() for u in r['blocked_non_local_requests']})},
              'native_seconds': native['seconds']}
    path = args.out / f'issuance-{day}-gfs{int(gfs)}.json'
    path.write_text(json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + '\n')
    c = record.get('comparison', {})
    print(day, 'gfs' if gfs else 'no-gfs', (c.get('host_path'), c.get('host_display'), c.get('status_vs_native')))
    return 0 if record['artifact_verifies'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
