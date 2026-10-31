"""Item 3's issuing-host parity and browser identity, and v4's browser measurement.

- **Issuing host:** Pyodide 0.28.0 under Node.js v26.7.0 (`src/cp17/host/node_run.mjs`), the frozen package.
  Every run is compared with the native fit of the same payload by `cp17.retrain.compare`: the status, every
  data-driven selection, and the largest deviation per emitted level. A miss is reported with its numbers,
  never tuned.
- **Browser identity:** A1 retrained in Chromium and WebKit (`cp17.tools.measure`) must be bit-identical to
  the issuing host's result of the same payload.
- **v4 in the browser:** time, memory, payload sizes, deviation from the native fit and from the issuing host,
  selection agreement.

    PYTHONPATH=src python -m cp17.tools.hosts node --payloads <dir> --bundle <zip> --out <dir> --runs A1-2026-09-06,...
    PYTHONPATH=src python -m cp17.tools.hosts compare --native <native-parity.json> --nodes <dir> \\
        [--browser <browser-results.json>] --bundle-manifest <src.json> --out <host-parity.json>
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .. import retrain, runtime
from .bundle import PACKAGES

#: The official Node.js v26.7.0 darwin-arm64 binary (CP-17c, r19 §24.13), in place of Homebrew's
#: `/opt/homebrew/Cellar/node/26.7.0/bin/node` (`1ef99ea2…d040`).
NODE = str(runtime.node_path())
NODE_SHA256 = runtime.BINARY['sha256']
RUNNER = Path(__file__).resolve().parents[1] / 'host' / 'node_run.mjs'


def node_identity() -> dict:
    data = Path(NODE).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != NODE_SHA256:
        raise ValueError(f'the Node.js binary is not the pinned one: {actual}')
    version = subprocess.run([NODE, '--version'], capture_output=True, text=True, check=True).stdout.strip()
    return {'path': NODE, 'sha256': actual, 'version': version}


def run_node(pyodide: Path, bundle: Path, payloads: Path, stem: str, out: Path, attribution: bool) -> dict:
    started = time.perf_counter()
    target = out / f'node-{stem}.json'
    proc = subprocess.run([NODE, str(RUNNER), 'fit', str(pyodide), str(bundle), str(payloads), stem,
                           '1' if attribution else '0', ','.join(PACKAGES), str(target)],
                          capture_output=True, text=True)
    return {'stem': stem, 'returncode': proc.returncode, 'wall_seconds': time.perf_counter() - started,
            'stdout_tail': proc.stdout[-500:], 'stderr_lines': len(proc.stderr.splitlines())}


def node_main(args) -> int:
    node_identity()
    args.out.mkdir(parents=True, exist_ok=True)
    stems = args.runs.split(',')
    attribute = set(args.attribution.split(',')) if args.attribution else set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_node, args.pyodide, args.bundle, args.payloads, s, args.out, s in attribute) for s in stems]
        for job in jobs:
            r = job.result()
            print(r['stem'], 'ok' if r['returncode'] == 0 else 'FAILED', f"{r['wall_seconds']:.0f}s", flush=True)
            with (args.out / 'node-ledger.jsonl').open('a') as handle:
                handle.write(json.dumps(r) + '\n')
    return 0


def _source_ok(run: dict, manifest: dict) -> bool:
    executed = run['source_sha256']
    return all(manifest['files_sha256'].get(k) == v for k, v in executed.items()) and bool(executed)


def host_row(native: dict, node: dict, manifest: dict) -> dict:
    stem = node['target']
    policy = stem.split('-', 1)[0]
    if node.get('errors'):
        return {'stem': stem, 'policy': policy, 'status': 'failed', 'errors': node['errors']}
    b = node['python']['result']
    cmp = retrain.compare(policy, native, b)
    return {'stem': stem, 'policy': policy, 'day': b['day'], 'status': cmp['status'],
            'max_abs_eur': cmp['deviation'].get('max_abs_eur'),
            'max_abs_eur_by_level': cmp['deviation'].get('max_abs_eur_by_level'),
            'bitwise_equal': cmp['deviation'].get('bitwise_equal'), 'selections_match': cmp['selections_match'],
            'selections': cmp['selections'], 'payload_sha256_equal': cmp['payload_sha256_equal'],
            'issued_quantiles_sha256': b['quantiles_sha256'], 'native_quantiles_sha256': native['quantiles_sha256'],
            'fit_seconds': b['fit_seconds'], 'native_fit_seconds': native['fit_seconds'],
            'host_total_seconds': node['total_ms'] / 1000, 'wasm_memory_bytes': node['wasm_memory_bytes'],
            'process_rss_peak_bytes': node['process_rss_peak_bytes'],
            'blocked_non_local_requests': node['blocked_non_local_requests'],
            'executed_files_match_package': _source_ok(node['python'], manifest), 'stubs': node['python']['stubs'],
            'runtime': {'pyodide': node.get('pyodide_version'), 'node': node['node'], 'python': node['python']['python'],
                        'numpy': node['python']['numpy']},
            'attribution': _attribution_row(native.get('attribution'), b.get('attribution')),
            'members_max_abs_eur': _members(native, b)}


def _members(native: dict, hosted: dict) -> dict | None:
    """Each member's largest deviation: a blend's member centrals (v4: A1_w, B2_w, L-N, L-R; v2: A1, B2), or
    A1's eight DDNN-2 members' quantiles, keyed by member, configuration, seed and epochs run."""
    import numpy as np
    if native.get('policy') == 'A1':
        return {f"member {a['member']} ({a['config']}, seed {a['seed']}, {a['epochs_run']} epochs, {a['stop_reason']})":
                float(np.abs(np.asarray(a['quantiles'], float) - np.asarray(b['quantiles'], float)).max())
                for a, b in zip(native['members'], hosted['members'])}
    if 'members_central' not in native:
        return None
    return {k: float(np.abs(np.asarray(v, float) - np.asarray(hosted['members_central'][k], float)).max())
            for k, v in native['members_central'].items()}


def _attribution_row(native, hosted):
    if native is None or hosted is None:
        return None
    import numpy as np
    if 'phi_eur' in hosted:
        a, b = np.asarray(native['phi_eur'], float), np.asarray(hosted['phi_eur'], float)
        return {'method': hosted['method'], 'players_equal': native['players'] == hosted['players'],
                'phi_max_abs_eur': float(np.abs(a - b).max()), 'efficiency_max_abs': hosted['efficiency_max_abs'],
                'seconds': hosted['seconds']}
    return {'method': hosted['method'], 'blend_gap_max_abs': hosted['blend_gap_max_abs'],
            'component_efficiency_max_abs': hosted['component_efficiency_max_abs'], 'seconds': hosted['seconds']}


def gzip_size(path: Path) -> int:
    return len(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))


def browser_row(run: dict, node: dict | None, native: dict, manifest: dict, roots: dict) -> dict:
    from .measure import category
    stem = run['stem']
    policy = stem.split('-', 1)[0]
    row = {'browser': run['browser'], 'browser_version': run.get('browser_version'), 'stem': stem, 'policy': policy,
           'errors': run.get('errors', [])}
    if row['errors'] or 'python' not in run:
        row['status'] = 'failed'
        return row
    b = run['python']['result']
    raw, gz = {}, {}
    for item in run['served']:
        if item['status'] != 200:
            continue
        cat = category(item['path'])
        raw[cat] = raw.get(cat, 0) + item['bytes']
        prefix = next(p for p in roots if item['path'].startswith(p))
        gz[cat] = gz.get(cat, 0) + gzip_size(roots[prefix] / item['path'][len(prefix):])
    vs_native = retrain.compare(policy, native, b)
    vs_host = retrain.compare(policy, node['python']['result'], b) if node and not node.get('errors') else None
    row.update({
        'status_vs_issuing_host': vs_host['status'] if vs_host else None,
        'bitwise_equal_to_issuing_host': bool(vs_host and vs_host['deviation'].get('bitwise_equal')),
        'status_vs_native': vs_native['status'], 'max_abs_eur_vs_native': vs_native['deviation'].get('max_abs_eur'),
        'max_abs_eur_by_level_vs_native': vs_native['deviation'].get('max_abs_eur_by_level'),
        'selections_match_native': vs_native['selections_match'],
        'selections_match_issuing_host': vs_host['selections_match'] if vs_host else None,
        'executed_files_match_package': _source_ok(run['python'], manifest), 'stubs': run['python']['stubs'],
        'time_seconds': {'runtime_ready': run['marks']['runtime_ms'] / 1000, 'packages_ready': run['marks']['packages_ms'] / 1000,
                         'payload_ready': run['marks']['payload_ms'] / 1000, 'fit': b['fit_seconds'],
                         'attribution': (b.get('attribution') or {}).get('seconds'), 'total_in_worker': run['total_ms'] / 1000,
                         'native_fit': native['fit_seconds']},
        'memory_bytes': {'wasm_linear_memory_peak': run['wasm_memory_bytes'],
                         'browser_processes_rss_peak': run['process_rss_peak_bytes']},
        'payload_bytes': {'raw': raw, 'gzip': gz, 'first_visit': raw.get('page', 0),
                          'training_payload': sum(raw.get(c, 0) for c in ('runtime', 'packages', 'code', 'data')),
                          'training_payload_gzip': sum(gz.get(c, 0) for c in ('runtime', 'packages', 'code', 'data'))},
        'blocked_non_local_requests': run.get('blocked_non_local_requests', []),
        'runtime': {'pyodide': run.get('pyodide_version'), 'python': run['python']['python'], 'numpy': run['python']['numpy']}})
    return row


def compare_main(args) -> int:
    native = json.loads(args.native.read_text())
    manifest = json.loads(args.bundle_manifest.read_text())
    nodes = {}
    for path in sorted(args.nodes.glob('node-*.json')):
        if path.name == 'node-ledger.jsonl':
            continue
        run = json.loads(path.read_text())
        nodes[run['target']] = run

    def native_result(stem):
        policy, day = stem.split('-', 1)
        return native['origins'][day]['results'][policy]
    hosts = [host_row(native_result(s), run, manifest) for s, run in sorted(nodes.items())]
    out = {'schema': 'cp17-host-parity-v1', 'issuing_host': node_identity(), 'tau_eq_eur': retrain.TAU_EQ,
           'package_sha256': manifest['package_sha256'], 'issuing_host_parity': hosts}
    if args.browser:
        browser = json.loads(args.browser.read_text())
        roots = {'/app/': Path(__file__).resolve().parents[1] / 'browser', '/pyodide/': args.pyodide,
                 '/bundle/': args.bundle_manifest.parent, '/payloads/': args.payloads}
        out['browser_host'] = {k: browser[k] for k in ('playwright', 'machine', 'platform', 'cpu_count')}
        out['browser_runs'] = [browser_row(r, nodes.get(r['stem']), native_result(r['stem']), manifest, roots)
                               for r in browser['runs']]
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + '\n')
    for r in hosts:
        print('node', r['stem'], r['status'], r.get('max_abs_eur'), 'selections', r.get('selections_match'))
    for r in out.get('browser_runs', []):
        print(r['browser'], r['stem'], 'vs host:', r.get('status_vs_issuing_host'), 'vs native:', r.get('status_vs_native'))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    n = sub.add_parser('node')
    n.add_argument('--pyodide', type=Path, default=Path('/Users/djourno/Downloads/PJM/.local/tools/pyodide-0.28.0'))
    n.add_argument('--bundle', required=True, type=Path)
    n.add_argument('--payloads', required=True, type=Path)
    n.add_argument('--out', required=True, type=Path)
    n.add_argument('--runs', required=True)
    n.add_argument('--attribution', default='')
    n.add_argument('--workers', default=2, type=int)
    c = sub.add_parser('compare')
    c.add_argument('--native', required=True, type=Path)
    c.add_argument('--nodes', required=True, type=Path)
    c.add_argument('--browser', type=Path)
    c.add_argument('--bundle-manifest', required=True, type=Path)
    c.add_argument('--pyodide', type=Path, default=Path('/Users/djourno/Downloads/PJM/.local/tools/pyodide-0.28.0'))
    c.add_argument('--payloads', type=Path)
    c.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    return node_main(args) if args.cmd == 'node' else compare_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
