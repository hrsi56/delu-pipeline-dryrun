"""Item 5: §19.5's definitions computed natively and under the issuing host, on real development windows.

The windows are CP-26's stored nominal forecasts (A1's D2 quantiles, and B0's as the Space's context) with the
development snapshot's prices, on two windows that hold a 25-hour and a 23-hour day, zero and negative
prices, and partial days (CP-25's frozen eligible keys):

- 2025-10-01 to 2025-11-30 (autumn change, 2025-10-26);
- 2026-03-01 to 2026-09-06 (spring change, 2026-03-29, and the spring-summer negative prices).

The record is computed natively by `cp17.scoring.evaluate`, then recomputed under Pyodide in Node.js by the
same function (`host/driver.py`, MODE 'score'), and the two are compared by `cp17.scoring.recompute_check`:
counts exactly, reals within 1e-9 * max(1, |b|). Development data only; evidence class
`development_post_selection`; descriptive.

    PYTHONPATH=src python -m cp17.tools.scorecheck --bundle <src.zip> --out <dir> --record reports/cp17/scoring-check.json
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import subprocess

import numpy as np

from .. import scoring
from ..issue import canonical_hours
from . import ledger
from .bundle import PACKAGES
from .hosts import NODE, RUNNER, node_identity

WINDOWS = {'autumn-2025': (date(2025, 10, 1), date(2025, 11, 30)), 'spring-summer-2026': (date(2026, 3, 1), date(2026, 9, 6))}


def windows(root) -> tuple[dict, dict]:
    from cp28 import reader
    prices = reader.prices(root)
    out, outcomes = {}, {}
    for name, (first, last) in WINDOWS.items():
        for policy, take in (('A1', lambda v: v['members']['nominal']['D2']), ('B0', lambda v: v['policies']['nominal/B0'])):
            issued = []
            for d, value in ledger.iter_days(root, first=first, last=last):
                scoring_keys = value['policies']['nominal/v5']['scoring_index']
                if not scoring_keys:
                    continue
                item = take(value)
                q = item.get('quantiles') if policy == 'A1' else item.get('native_quantiles')
                if q is None:
                    continue
                q = ledger.on_keys(q, value['policies']['nominal/v5']['index'] if policy == 'A1' else item['index'],
                                   scoring_keys)
                if q is None:
                    continue
                keys = [ledger.keys([k])[0].isoformat() for k in scoring_keys]
                issued.append({'day': str(d), 'keys': keys, 'quantiles': q.tolist(), 'canonical_hours': len(canonical_hours(d))})
                for k in keys:
                    v = prices.get(k)
                    outcomes[k] = None if v is None or not np.isfinite(v) else float(v)
            out[f'{name}/{policy}'] = issued
    return out, outcomes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--pyodide', type=Path, default=Path('/Users/djourno/Downloads/PJM/.local/tools/pyodide-0.28.0'))
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--record', required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    issued, outcomes = windows(args.root)
    (args.out / 'score-input.json').write_text(json.dumps({'windows': issued, 'outcomes': outcomes}, allow_nan=False))
    native = {name: scoring.evaluate(items, {k: (np.nan if v is None else v) for k, v in outcomes.items()})
              for name, items in issued.items()}
    target = args.out / 'node-score.json'
    subprocess.run([NODE, str(RUNNER), 'score', str(args.pyodide), str(args.bundle), str(args.out), 'score-input.json', '0',
                    ','.join(PACKAGES), str(target)], capture_output=True, text=True)
    run = json.loads(target.read_text())
    if run.get('errors'):
        raise SystemExit(run['errors'][0][-2000:])
    hosted = run['python']['records']
    checks = {name: scoring.recompute_check(hosted[name], native[name]) for name in native}
    summary = {name: {'days': r['pooled']['scored_days'], 'hours': r['pooled']['scored_hours'],
                      'partial_days': r['pooled']['partial_days'], 'mae_eur': r['pooled']['mae_eur'],
                      'wis_eur': r['pooled']['wis_eur'], 'percentage': r['pooled']['percentage']['share'],
                      'coverage': {b: c['share'] for b, c in r['pooled']['coverage'].items()},
                      'negative_price_hours': sum(1 for d in issued[name] for k in d['keys']
                                                  if outcomes.get(k) is not None and outcomes[k] < 0),
                      'zero_price_hours': sum(1 for d in issued[name] for k in d['keys'] if outcomes.get(k) == 0.0),
                      'canonical_hours_seen': sorted({d['canonical_hours'] for d in issued[name]})}
               for name, r in native.items()}
    record = {'schema': 'cp17-scoring-check-v1', 'evidence_class': 'development_post_selection',
              'what': '§19.5 records computed natively and recomputed under the issuing host; descriptive',
              'windows': {k: [str(a), str(b)] for k, (a, b) in WINDOWS.items()}, 'summary': summary,
              'recompute_checks': {k: {'compared': v['compared'], 'match': v['match'], 'mismatches': v['mismatches'][:5]}
                                   for k, v in checks.items()},
              'all_match': all(v['match'] for v in checks.values()),
              'bitwise_equal': {name: scoring.flatten(hosted[name]) == scoring.flatten(native[name]) for name in native},
              'host': {'node': node_identity(), 'pyodide': run.get('pyodide_version'), 'seconds': run['total_ms'] / 1000,
                       'executed_files': run['python']['source_sha256'],
                       'blocked_non_local_requests': run['blocked_non_local_requests']}}
    args.record.write_text(json.dumps(record, indent=1, sort_keys=True, allow_nan=False) + '\n')
    print('all match' if record['all_match'] else 'MISMATCH', record['bitwise_equal'])
    return 0 if record['all_match'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
