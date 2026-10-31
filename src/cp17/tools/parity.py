"""Native parity, the code-identity check (brief item 3): the package, run natively with BLAS at one thread,
against CP-26's stored nominal forecasts for A1, v5, v4, v2 and B0 at the nine origins.

CP-26's recorded parity standard is bitwise (its `parity.json` compares IEEE-754 bits), so every check here is
bitwise: each policy's emitted seven quantiles, each member's central, and every selection CP-26 recorded
(DDNN-2's per-member stopping, windows and cap; LEAR's selected alpha, iteration counts and coefficient hash
per hour; LightGBM's selected capacity and tree hash per block model; the interval layers' buffers).

    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
        PYTHONPATH=src python -m cp17.tools.parity --out <folder> [--origins 2026-09-06,...]

It writes the payloads it built to `<folder>/payloads/` and its record to `<folder>/native-parity.json`.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time

import numpy as np

from .. import dailyfit
from . import ledger, payloads, states

ORIGINS = tuple(date(2026, 8, d) for d in (31,)) + tuple(date(2026, 9, d) for d in range(1, 7)) + \
    (date(2025, 10, 26), date(2026, 3, 29))
HOST_ORIGINS = (date(2026, 9, 6), date(2026, 3, 29))
POLICIES = ('A1', 'v2', 'v4', 'v5', 'B0')
BLAS = ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def bits_equal(a, b) -> bool:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))


def max_abs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max()) if a.shape == b.shape else None


def check_d2(result: dict, stored: dict) -> dict:
    fields = ('best_epoch', 'epochs_run', 'stop_reason', 'params_sha256', 'n_inputs')
    window = ('held_out_weeks', 'week_pool', 'n_days', 'n_train_days', 'n_held_out_days', 'train_days_sha256')
    rows = []
    for mine, theirs in zip(result['members'], stored['records']):
        diff = [f for f in fields if mine[f] != theirs[f]] + [f for f in window if mine[f] != theirs['window'][f]]
        if mine['seed'] != theirs['seed']:
            diff.append('seed')
        if mine['cap_forecast_slot_levels'] != theirs['cap_slot_levels']:
            diff.append('cap_forecast_slot_levels')
        rows.append({'member': mine['member'], 'seed': mine['seed'], 'differences': diff})
    q = bits_equal(result['quantiles'], stored['quantiles'])
    ok = q and len(rows) == 8 == len(stored['records']) and all(not r['differences'] for r in rows) \
        and result['crossed_rows'] == stored['crossed_rows'] and bits_equal(result['central'], stored['central'])
    return {'pass': bool(ok), 'quantiles_bitwise': q, 'max_abs_eur': max_abs(result['quantiles'], stored['quantiles']),
            'members': rows, 'search': stored['search']}


def check_lear(central, selections: list[dict], train: dict, stored: dict) -> dict:
    fields = ('local_hour', 'selected_relative_alpha', 'alpha', 'final_n_iter', 'final_solver_passes', 'model_sha256',
              'scaler_sha256', 'imputer_sha256', 'n_hour_train')
    diff = [f"hour {a['local_hour']}: {f}" for a, b in zip(selections, stored['records']) for f in fields
            if f in b and a.get(f) != b[f]]
    if len(selections) != len(stored['records']):
        diff.append('hour count')
    rows_ok = all(train['rows_sha256'] == r['train_rows_sha256'] for r in stored['records'])
    c = bits_equal(central, stored['central'])
    return {'pass': bool(c and not diff and rows_ok), 'central_bitwise': c, 'selection_differences': diff,
            'train_rows_match': rows_ok}


def check_lgbm(central, selection: dict, stored: dict) -> dict:
    sel = {k: v['selected'] for k, v in stored['selection'].items()}
    trees = {(r['model'], r['role'], r['config']): r['model_sha256'] for r in stored['records']}
    mine = {(f['model'], f['role'], f['config']): f['model_sha256'] for f in selection['fits']}
    c = bits_equal(central, stored['central'])
    return {'pass': bool(c and sel == selection['selected'] and trees == mine), 'central_bitwise': c,
            'selection_match': sel == selection['selected'], 'tree_hashes_match': trees == mine}


def check_policy(result: dict, record: dict, layer_key: str | None) -> dict:
    q = bits_equal(result['quantiles'], record['native_quantiles'])
    c = bits_equal(result['central'], record['native_central'])
    keys = result['index'] == [ledger.keys([t])[0].isoformat() for t in record['index']]
    out = {'pass': bool(q and c and keys), 'quantiles_bitwise': q, 'central_bitwise': c, 'keys_equal': keys,
           'max_abs_eur': max_abs(result['quantiles'], record['native_quantiles']),
           'stored_output_source': record['output_source']}
    if layer_key:
        buffer = record['buffer']
        layer = result[layer_key]
        same = (layer['buffer_days'] == buffer['days'] and layer['buffer_hours'] == buffer['hours']
                and layer['buffer_start'] == buffer['oldest_day'] and layer['buffer_end'] == buffer['latest_day'])
        out['buffer_matches_ledger'] = same
        out['pass'] = out['pass'] and same
    return out


def run_origin(root, ctx, day: date, buffers: dict, folder: Path, attribution: bool) -> dict:
    inputs = ledger.inputs_dir(root)
    value = ledger.day(root, day)
    members, policies = value['members']['nominal'], value['policies']
    written, results, checks, seconds = {}, {}, {}, {}
    a1_arrays, a1_meta = payloads.a1_payload(ctx, day, payloads.ensemble(inputs, day))
    v4_arrays, v4_meta = payloads.v4_payload(ctx, day, buffers['v4'][day])
    built = {'A1': (a1_arrays, a1_meta), 'v2': payloads.v2_payload(ctx, day, buffers['v2'][day]), 'v4': (v4_arrays, v4_meta),
             'v5': payloads.v5_payload(day, buffers['v5'][day], v4_meta['payload_sha256'], a1_meta['payload_sha256']),
             'B0': payloads.b0_payload(ctx, day, buffers['B0'][day])}
    for policy, (arrays, meta) in built.items():
        written[policy] = payloads.write(arrays, meta, folder, f'{policy}-{day}')
    extra = {}
    for policy in POLICIES:
        started = time.perf_counter()
        w = written[policy]
        if policy == 'v5':
            extra['a1'] = results['A1']
        results[policy] = dailyfit.run(policy, w.get('npz'), w['meta'], attribution=attribution and policy != 'v5',
                                       extra=extra if policy in ('v4', 'v5') else None)
        seconds[policy] = time.perf_counter() - started
    r = results
    checks['A1'] = check_d2(r['A1'], members['D2'])
    checks['v2'] = {'policy': check_policy(r['v2'], policies['nominal/v2'], 'h_layer'),
                    'A1': check_lear(r['v2']['members_central']['A1'], r['v2']['selections']['A1'], r['v2']['train_rows']['A1'], members['A1']),
                    'B2': check_lear(r['v2']['members_central']['B2'], r['v2']['selections']['B2'], r['v2']['train_rows']['B2'], members['B2'])}
    checks['v4'] = {'policy': check_policy(r['v4'], policies['nominal/v4'], 'h_layer'),
                    **{n: check_lear(r['v4']['members_central'][n], r['v4']['selections'][n], r['v4']['train_rows'][n], members[n])
                       for n in ('A1_w', 'B2_w')},
                    **{a: check_lgbm(r['v4']['members_central'][a], r['v4']['selections'][a], members[a]) for a in ('L-N', 'L-R')}}
    checks['v5'] = {'policy': check_policy(r['v5'], policies['nominal/v5'], 'h_layer')}
    checks['B0'] = {'policy': check_policy(r['B0'], policies['nominal/B0'], 'b0_layer'),
                    'central_bitwise': bits_equal(r['B0']['central'], members['B0']['central']),
                    'source_day_match': r['B0']['source_day'] == members['B0']['records']['source_delivery_day']}
    for policy in ('v2', 'v4', 'v5', 'B0'):
        c = checks[policy]
        c['pass'] = all([v['pass'] for v in c.values() if isinstance(v, dict)] + [v for v in c.values() if isinstance(v, bool)])
    return {'day': str(day), 'ledger_file_sha256': value['_file_sha256'], 'payloads': written, 'results': results,
            'checks': checks, 'seconds': seconds, 'pass': all(c['pass'] for c in checks.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--origins', default=','.join(map(str, ORIGINS)))
    ap.add_argument('--attribution', default=','.join(map(str, HOST_ORIGINS)))
    args = ap.parse_args(argv)
    if any(os.environ.get(n) != '1' for n in BLAS):
        raise SystemExit('export the five *_NUM_THREADS=1 before NumPy loads, as CP-26 ran')
    origins = sorted(date.fromisoformat(d) for d in args.origins.split(','))
    attribute = {date.fromisoformat(d) for d in args.attribution.split(',') if d}
    t0 = time.perf_counter()
    ctx = payloads.context(args.root)
    build = time.perf_counter() - t0
    rebuilt = states.rebuild(args.root, ctx, origins)
    report = {'schema': 'cp17-native-parity-v1', 'python': sys.version.split()[0], 'numpy': np.__version__,
              'machine': platform.machine(), 'platform': platform.platform(),
              'blas_threads': {n: os.environ.get(n) for n in BLAS}, 'context_build_seconds': build,
              'states': {k: v for k, v in rebuilt.items() if k != 'buffers'},
              'standard': "CP-26's recorded parity: IEEE-754 bits (reports/rebaseline/cold-rerun-2026-10-15/parity.json)",
              'origins': {}}
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for day in origins:
        item = run_origin(args.root, ctx, day, rebuilt['buffers'], out / 'payloads', day in attribute)
        report['origins'][str(day)] = item
        print(day, 'PASS' if item['pass'] else 'FAIL', {p: c['pass'] for p, c in item['checks'].items()},
              {p: round(s, 1) for p, s in item['seconds'].items()}, flush=True)
        (out / 'native-parity.json').write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + '\n')
    report['maxrss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report['total_seconds'] = time.perf_counter() - t0
    report['status'] = 'PASS' if all(o['pass'] for o in report['origins'].values()) else 'FAIL'
    (out / 'native-parity.json').write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + '\n')
    print(report['status'], f"{report['total_seconds']:.0f}s")
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
