"""The native reference (brief, Observable outcome 2): build the payloads, fit them natively, and check the
fits against CP-26's stored nominal forecasts to CP-26's recorded parity standard, which is bitwise.

Run from a checkout with BLAS limited to one thread:

    OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
        python -m cp17probe.native --inputs <extracted evidence/cp-26 folder> --fits <CP-26 nominal fit cache> \\
        --state <CP-26 replay-state.json> --out <folder>

The inputs are CP-26's local-only stored state; nothing here trains a search or reads a day after
2026-09-06.
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

from . import dailyfit, payload

DAYS_A1 = (date(2026, 9, 6), date(2026, 3, 29))
DAY_V2 = date(2026, 9, 6)
RUN = 'reports/rebaseline/cold-rerun-2026-10-15'


def bits(a) -> list[str]:
    return [format(int(v), '016x') for v in np.asarray(a, dtype=np.float64).ravel().view(np.uint64)]


def stored(inputs: Path, fits: Path, day: date, policy: str) -> dict:
    """CP-26's stored nominal issuance for `policy` ('D2' or 'v2') and its member fit records."""
    ledger = json.loads((inputs / RUN / 'forecast-ledger' / f'{day}.json').read_text())
    row = ledger['policies'][f'nominal/{policy}']
    out = {'native_quantiles': row['native_quantiles'], 'index': row['index'], 'output_source': row['output_source'],
           'ledger_content_sha256': ledger['content_sha256']}
    for member in (('D2',) if policy == 'D2' else ('A1', 'B2')):
        item = json.loads((fits / str(day) / f'{member}.json').read_text())
        out[f'fit_{member}'] = {'records': item['result']['records'], 'content_sha256': item['content_sha256'],
                                'quantiles_sha256': item.get('quantiles_sha256'),
                                'central_sha256': item.get('central_sha256')}
    return out


def a1_selections_match(result: dict, record: list[dict]) -> dict:
    """Every data-driven selection CP-26 recorded for D2's eight members, against the refit."""
    fields = ('best_epoch', 'epochs_run', 'stop_reason', 'params_sha256', 'n_inputs')
    window = ('held_out_weeks', 'week_pool', 'n_days', 'n_train_days', 'n_held_out_days', 'train_days_sha256')
    rows = []
    for mine, theirs in zip(result['members'], record):
        diff = [f for f in fields if mine[f] != theirs[f]] + [f for f in window if mine[f] != theirs['window'][f]]
        if mine['cap_forecast_slot_levels'] != theirs['cap_slot_levels']:
            diff.append('cap_forecast_slot_levels')
        rows.append({'member': mine['member'], 'seed': mine['seed'], 'differences': diff})
    return {'all_match': all(not r['differences'] for r in rows) and len(rows) == 8, 'members': rows}


def v2_selections_match(result: dict, rec: dict) -> dict:
    out = {}
    for member in ('A1', 'B2'):
        theirs = rec[f'fit_{member}']['records']
        mine = result['selections'][member]
        diff = []
        for a, b in zip(mine, theirs):
            for f in ('local_hour', 'selected_relative_alpha', 'alpha', 'final_n_iter', 'final_solver_passes', 'model_sha256'):
                if f in b and a.get(f) != b[f]:
                    diff.append(f"hour {a['local_hour']}: {f}")
        if len(mine) != len(theirs):
            diff.append('hour count')
        out[member] = diff
    return {'all_match': not any(out.values()), 'differences': out}


def check(result: dict, ref: dict, policy: str) -> dict:
    expected = np.asarray(ref['native_quantiles'], float)
    actual = np.asarray(result['quantiles'], float)
    same_shape = expected.shape == actual.shape
    bitwise = same_shape and bits(expected) == bits(actual)
    max_abs = float(np.max(np.abs(expected - actual))) if same_shape else None
    keys = result.get('index', ref['index']) == ref['index'] if policy == 'v2' else True
    sel = a1_selections_match(result, ref['fit_D2']['records']) if policy == 'A1' \
        else v2_selections_match(result, ref)
    return {'bitwise_equal': bool(bitwise), 'max_abs_difference': max_abs, 'keys_equal': bool(keys),
            'selections': sel, 'expected_quantiles_sha256': dailyfit.array_hash(expected),
            'actual_quantiles_sha256': result['quantiles_sha256'], 'pass': bool(bitwise and keys and sel['all_match'])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--inputs', required=True, type=Path)
    ap.add_argument('--fits', required=True, type=Path)
    ap.add_argument('--state', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    ctx = payload.context(Path(args.root))
    build_seconds = time.perf_counter() - t0
    protocol = ctx.data.p
    state = json.loads(args.state.read_text())
    v2_state = _find_v2_state(state)
    del state
    folder = args.out / 'payloads'
    report = {'schema': 'cp17-probe-native-reference-v1', 'python': sys.version.split()[0], 'numpy': np.__version__,
              'machine': platform.machine(), 'platform': platform.platform(),
              'blas_threads': {n: os.environ.get(n) for n in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                                                              'OMP_NUM_THREADS')},
              'context_build_seconds': build_seconds, 'payloads': {}, 'results': {}, 'checks': {}}
    jobs = [('A1', d) for d in DAYS_A1] + [('v2', DAY_V2)]
    for policy, day in jobs:
        stem = f'{policy}-{day}'
        if policy == 'A1':
            from cp26.search import serving_search
            k = serving_search(day)
            ensemble = json.loads((args.inputs / RUN / 'search-results' / 'nominal' / f'k{k:02d}.json').read_text())
            arrays, meta = payload.a1_payload(ctx, day, ensemble)
            ref = stored(args.inputs, args.fits, day, 'D2')
        else:
            arrays, meta = payload.v2_payload(ctx, day, v2_state, protocol)
            ref = stored(args.inputs, args.fits, day, 'v2')
        report['payloads'][stem] = payload.write(arrays, meta, folder, stem)
        started = time.perf_counter()
        result = dailyfit.run(policy, report['payloads'][stem]['npz'], report['payloads'][stem]['meta'],
                             attribution=policy == 'A1')
        result['wall_seconds'] = time.perf_counter() - started
        report['results'][stem] = result
        report['checks'][stem] = check(result, ref, policy)
        print(stem, 'bitwise' if report['checks'][stem]['bitwise_equal'] else 'DIFFERENT',
              'selections', report['checks'][stem]['selections']['all_match'], f"{result['wall_seconds']:.1f}s", flush=True)
    report['maxrss_bytes'] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    report['total_seconds'] = time.perf_counter() - t0
    report['status'] = 'PASS' if all(c['pass'] for c in report['checks'].values()) else 'FAIL'
    (args.out / 'native-reference.json').write_text(json.dumps(report, indent=1, sort_keys=True, allow_nan=False) + '\n')
    print(report['status'], f"{report['total_seconds']:.0f}s")
    return 0 if report['status'] == 'PASS' else 1


def _find_v2_state(state: dict) -> dict:
    """The nominal v2 policy's H state, wherever the replay state file keeps its policies."""
    stack = [state]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if item.get('policy') == 'v2' and item.get('route') == 'nominal' and item.get('layer') == 'H':
                return item['errors']
            stack.extend(v for v in item.values() if isinstance(v, (dict, list)))
        elif isinstance(item, list):
            stack.extend(v for v in item if isinstance(v, (dict, list)))
    raise ValueError('no nominal v2 H state in the replay state')


if __name__ == '__main__':
    raise SystemExit(main())
