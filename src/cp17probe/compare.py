"""The probe's measurement table: each browser fit against the native fit of the same payload (§19.3 items 2, 4).

For every browser and day it reports what §19.3 item 2 says the action reports -- whether every
data-driven selection matched, the largest absolute difference per emitted quantile in EUR/MWh, and a
status stating only what was measured -- together with item 4's time, memory and payload sizes. No
τ_eq exists yet, so the status is "identical bit for bit" or "different"; the Owner sets τ_eq from
these numbers.

**A1's data-driven selections**, per member: the seeded held-out weeks and the week pool; the training
and held-out day sets (their counts and index hashes); the input count; the winsorisation activations
on the training, held-out and forecast rows; early stopping's best epoch, epochs run and stop reason;
the cap's activations on the forecast; the guard events. Across members: the crossing restorations of
the ensemble median. **v2's**: LEAR's selected relative alpha for every local hour of A1 and B2, and the
H layer's buffer identity (its released days, their errors' hash and the issuing scale). Weight hashes and solver iteration counts are reported beside them as
bitwise diagnostics, not as selections.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .dailyfit import array_hash

LEVELS = ('p025', 'p10', 'p25', 'p50', 'p75', 'p90', 'p975')
A1_SELECTIONS = ('held_out_weeks', 'week_pool', 'n_days', 'n_train_days', 'n_held_out_days', 'train_days_sha256',
                 'holdout_days_sha256', 'n_inputs', 'winsor_train', 'winsor_hold', 'winsor_forecast', 'best_epoch',
                 'epochs_run', 'stop_reason', 'cap_forecast_slot_levels', 'events')
A1_DIAGNOSTICS = ('params_sha256', 'best_stopping_metric_eur', 'cap_z', 'preprocessor')
H_SELECTION = ('buffer_start', 'buffer_end', 'buffer_days', 'buffer_hours', 'buffer_sha256', 'scale_sha256')
V2_DIAGNOSTICS = ('alpha', 'final_n_iter', 'final_solver_passes', 'n_iter_by_alpha', 'solver_passes_by_alpha',
                  'model_sha256', 'validation_mae_by_alpha')


def deviation(native, browser) -> dict:
    a, b = np.asarray(native, float), np.asarray(browser, float)
    if a.shape != b.shape:
        return {'shape_equal': False}
    diff = np.abs(a - b)
    return {'shape_equal': True, 'bitwise_equal': bool(np.array_equal(a.view(np.uint64), b.view(np.uint64))),
            'max_abs_eur': float(diff.max()), 'max_abs_eur_by_level': dict(zip(LEVELS, map(float, diff.max(axis=0)))),
            'mean_abs_eur': float(diff.mean()), 'native_sha256': array_hash(a), 'browser_sha256': array_hash(b)}


def a1_selections(native: dict, browser: dict) -> dict:
    rows, diagnostics = [], []
    for n, b in zip(native['members'], browser['members']):
        rows.append({'member': n['member'], 'seed': n['seed'],
                     'differences': [f for f in A1_SELECTIONS if n[f] != b[f]]})
        diagnostics.append({'member': n['member'], 'differences': [f for f in A1_DIAGNOSTICS if n[f] != b[f]]})
    crossed = native['crossed_rows'] == browser['crossed_rows']
    ok = len(rows) == 8 == len(browser['members']) and crossed and all(not r['differences'] for r in rows)
    return {'all_match': ok, 'crossed_rows_match': crossed, 'members': rows, 'diagnostics': diagnostics}


def v2_selections(native: dict, browser: dict) -> dict:
    rows, diagnostics = {}, {}
    for member in ('A1', 'B2'):
        n_sel, b_sel = native['selections'][member], browser['selections'][member]
        rows[member] = [n['local_hour'] for n, b in zip(n_sel, b_sel)
                        if n['selected_relative_alpha'] != b['selected_relative_alpha'] or n['local_hour'] != b['local_hour']]
        if len(n_sel) != len(b_sel):
            rows[member].append('hour count')
        diagnostics[member] = sorted({f for n, b in zip(n_sel, b_sel) for f in V2_DIAGNOSTICS if n.get(f) != b.get(f)})
    # The H layer's selection is its buffer: which released days, their errors and the issuing scale. The hash of
    # the central it is applied to is a value, reported with the deviation, not a selection.
    h = all(native['h_layer'][k] == browser['h_layer'][k] for k in H_SELECTION)
    rows_ok = native['train_rows_sha256'] == browser['train_rows_sha256']
    ok = h and rows_ok and not any(rows.values())
    return {'all_match': ok, 'alpha_differences_by_hour': rows, 'h_layer_match': h, 'train_rows_match': rows_ok,
            'diagnostics': diagnostics}


def attribution_deviation(native: dict | None, browser: dict | None) -> dict | None:
    if native is None or browser is None:
        return None
    a, b = np.asarray(native['phi_eur'], float), np.asarray(browser['phi_eur'], float)
    return {'players_equal': native['players'] == browser['players'], 'coalitions': browser['coalitions'],
            'phi_max_abs_eur': float(np.abs(a - b).max()) if a.shape == b.shape else None,
            'browser_efficiency_max_abs': browser['efficiency_max_abs'],
            'browser_issued_gap_max_abs': browser['issued_gap_max_abs']}


def gzip_size(path: Path) -> int:
    return len(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))


def sizes(run: dict, files: dict[str, Path]) -> dict:
    """Raw bytes served by category, and the same files' gzip -9 sizes (what a compressing host sends)."""
    raw, gz = {}, {}
    for item in run['served']:
        if item['status'] != 200:
            continue
        cat = item['category']
        raw[cat] = raw.get(cat, 0) + item['bytes']
        gz[cat] = gz.get(cat, 0) + files[item['path']]
    first = raw.get('page', 0)
    training = sum(raw.get(c, 0) for c in ('runtime', 'packages', 'code', 'data'))
    return {'raw_bytes': raw, 'gzip_bytes': gz, 'first_visit_bytes': first, 'training_payload_bytes': training,
            'training_payload_gzip_bytes': sum(gz.get(c, 0) for c in ('runtime', 'packages', 'code', 'data')),
            'note': 'first visit = the page alone, with the runtime loaded only when the visitor asks to retrain; '
                    'training payload = runtime + packages + code + the day\'s data'}


def build(native_path: Path, browser_path: Path, roots: dict[str, Path], bundle_manifest: Path,
          node_paths=None) -> dict:
    from .measure import category
    native = json.loads(native_path.read_text())
    browser = json.loads(browser_path.read_text())
    bundle = json.loads(bundle_manifest.read_text())
    gz_cache: dict[str, int] = {}
    table = []
    for run in browser['runs']:
        stem = run['stem']
        row = {'browser': run['browser'], 'browser_version': run.get('browser_version'), 'stem': stem,
               'policy': stem.split('-', 1)[0], 'day': stem.split('-', 1)[1], 'errors': run.get('errors', [])}
        if row['errors'] or 'python' not in run:
            row['status'] = 'failed'
            table.append(row)
            continue
        for item in run['served']:
            item['category'] = category(item['path'])
            if item['status'] == 200 and item['path'] not in gz_cache:
                prefix = next(p for p in roots if item['path'].startswith(p))
                gz_cache[item['path']] = gzip_size(roots[prefix] / item['path'][len(prefix):])
        b = run['python']['result']
        n = native['results'][stem]
        dev = deviation(n['quantiles'], b['quantiles'])
        sel = a1_selections(n, b) if row['policy'] == 'A1' else v2_selections(n, b)
        row.update({
            'payload_sha256_equal': b['payload_sha256'] == n['payload_sha256'],
            'source_identical_to_checkout': run['python']['source_sha256'] == {k: v for k, v in bundle['files_sha256'].items()
                                                                               if k in run['python']['source_sha256']},
            'source_files': len(run['python']['source_sha256']), 'stubs_engaged': run['python']['stubs'],
            'runtime': {'pyodide': run.get('pyodide_version'), 'python': run['python']['python'],
                        'numpy': run['python']['numpy'], 'pandas': run['python']['pandas'],
                        'platform': run['python']['platform']},
            'time_seconds': {'runtime_ready': run['marks']['runtime_ms'] / 1000,
                             'packages_ready': run['marks']['packages_ms'] / 1000,
                             'payload_ready': run['marks']['payload_ms'] / 1000,
                             'fit': b['fit_seconds'], 'attribution': (b.get('attribution') or {}).get('seconds'),
                             'total_in_worker': run['total_ms'] / 1000, 'native_fit': n['fit_seconds'],
                             'native_attribution': (n.get('attribution') or {}).get('seconds'),
                             'browser_over_native': b['fit_seconds'] / n['fit_seconds']},
            'memory_bytes': {'wasm_linear_memory_peak': run['wasm_memory_bytes'],
                             'browser_processes_rss_peak': run['process_rss_peak_bytes'],
                             'page_js_heap_used': (run.get('page_js_heap') or {}).get('used')},
            'payload': sizes(run, gz_cache), 'deviation': dev, 'selections': sel,
            'attribution': attribution_deviation(n.get('attribution'), b.get('attribution')),
            'blocked_non_local_requests': run.get('blocked_non_local_requests', []),
            'status': 'identical bit for bit' if dev.get('bitwise_equal') else 'different'})
        table.append(row)
    hosts = {}
    for path in node_paths or ():
        run = json.loads(Path(path).read_text())
        if run.get('errors'):
            hosts[run['stem']] = {'errors': run['errors']}
            continue
        b, n = run['python']['result'], native['results'][run['stem']]
        same = sorted(r['browser'] for r in browser['runs'] if r['stem'] == run['stem'] and not r.get('errors')
                      and r['python']['result']['quantiles_sha256'] == b['quantiles_sha256'])
        hosts[run['stem']] = {'host': f"node {run['node']}", 'pyodide': run.get('pyodide_version'),
                              'deviation': deviation(n['quantiles'], b['quantiles']),
                              'selections': (a1_selections if run['stem'].startswith('A1') else v2_selections)(n, b),
                              'browsers_bitwise_equal_to_node': same, 'fit_seconds': b['fit_seconds'],
                              'wasm_linear_memory_peak': run['wasm_memory_bytes']}
    return {'schema': 'cp17-probe-measurements-v1', 'native_status': native['status'], 'node_pyodide': hosts,
            'native_runtime': {k: native[k] for k in ('python', 'numpy', 'machine', 'platform', 'blas_threads')},
            'browser_host': {k: browser[k] for k in ('playwright', 'machine', 'platform', 'cpu_count')},
            'tau_eq': None, 'runs': table}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', required=True, type=Path)
    ap.add_argument('--browser', required=True, type=Path)
    ap.add_argument('--bundle-manifest', required=True, type=Path)
    ap.add_argument('--pyodide', required=True, type=Path)
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--payloads', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--node', nargs='*', default=[], type=Path)
    args = ap.parse_args(argv)
    roots = {'/app/': Path(__file__).resolve().parent / 'browser', '/pyodide/': args.pyodide, '/bundle/': args.bundle,
             '/payloads/': args.payloads}
    out = build(args.native, args.browser, roots, args.bundle_manifest, args.node)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True, allow_nan=False) + '\n')
    for r in out['runs']:
        print(r['browser'], r['stem'], r['status'], (r.get('deviation') or {}).get('max_abs_eur'),
              'selections', (r.get('selections') or {}).get('all_match'), r.get('time_seconds', {}).get('fit'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
