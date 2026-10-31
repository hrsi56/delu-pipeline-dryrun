"""CP-17b's report tables, rendered from its committed records, so that every figure in them is the record's.

`reports/cp17b/report.md` holds the generated block between the markers below; `render` writes it, and
`tests/cp17b/test_records.py` requires the committed block to equal a fresh rendering.

    PYTHONPATH=src python -m cp17.tools.report17b
"""
from __future__ import annotations

import json
from pathlib import Path

from .report import BEGIN, END, ORDER

REPORTS = Path('reports/cp17b')


def _load(name: str, root: Path):
    return json.loads((root / REPORTS / name).read_text())


def _mib(n) -> str:
    return f'{n / 2 ** 20:,.0f}'


def _e(x) -> str:
    return '0' if not x else f'{x:.1e}'


def native_table(root: Path) -> str:
    r = _load('native-parity.json', root)
    lines = ['| Origin (delivery) | A1 | v5 | v4 | v2 | B0 | equal to CP-17\'s record |', '|---|---|---|---|---|---|---|']
    for day in ORDER:
        o = r['origins'][day]
        cell = {p: 'bitwise' if o['checks'][p]['pass'] else 'FAIL' for p in ('A1', 'v5', 'v4', 'v2', 'B0')}
        same = 'yes' if all(r['equal_to_cp17'][day].values()) else 'NO'
        lines.append(f"| {day} | {cell['A1']} | {cell['v5']} | {cell['v4']} | {cell['v2']} | {cell['B0']} | {same} |")
    return '\n'.join(lines)


def host_table(root: Path) -> str:
    r = _load('host-parity.json', root)
    lines = ['| Run | Status vs native | Largest deviation (EUR/MWh) | Selections | Equal to CP-17\'s record |',
             '|---|---|---:|---|---|']
    for row in sorted(r['issuing_host_parity'], key=lambda x: x['stem']):
        same = 'yes' if all(r['equal_to_cp17'][row['stem']].values()) else 'NO'
        lines.append(f"| {row['stem']} | {row['status']} | {_e(row['max_abs_eur'])} | "
                     f"{'all matched' if row['selections_match'] else 'DIFFER'} | {same} |")
    for b in r.get('browser_runs', []):
        lines.append(f"| {b['browser']} {b['stem']} | {b.get('status_vs_issuing_host')} (vs the issuing host) | "
                     f"{_e(b.get('max_abs_eur_vs_issuing_host', 0))} | {'all matched' if b.get('selections_match_issuing_host', True) else 'DIFFER'} | "
                     f"{'yes' if r['browser_equal_to_cp17'].get(b['browser'] + ' ' + b['stem']) else 'NO'} |")
    return '\n'.join(lines)


def partial_table(root: Path) -> str:
    r = _load('partial-days.json', root)
    lines = ['| Day | Hours | Forecastable | Missing | Issuing host | Native | Same path | Shadows\' intake |',
             '|---|---:|---:|---|---|---|---|---|']
    for day, d in sorted(r.items()):
        missing = ', '.join(m.split(' (')[0] for m in (d['missing_hours'] or {}).get('missing_hours', [])) or 'none'
        if not d['forecastable_hours']['A1']:
            missing = 'all'
        intake = '; '.join(sorted({s.get('intake', s['status']) for s in d['host']['shadows'].values()}))
        lines.append(f"| {day} | {d['canonical_hours']} | {d['forecastable_hours']['A1']} | {missing} | {d['host']['display']} | "
                     f"{d['native']['display']} | {'yes' if d['same_path'] else 'NO'} | {intake} |")
    return '\n'.join(lines)


def shadow_table(root: Path) -> str:
    r = _load('shadow-run.json', root)
    h, n = r['host'], r['native']
    lines = ['| Delivery day | Issued | A1 (s) | v4 (s) | v2 (s) | B0 (s) | Total (s) | Buffers vs CP-26, host (largest) | Natively |',
             '|---|---|---:|---:|---:|---:|---:|---|---|']
    for day in sorted(h['days']):
        d, m = h['days'][day], n['days'][day]
        s = d['process_seconds']
        largest = max(b['max_abs_standardized'] or 0.0 for b in d['buffers'].values())
        host = 'bitwise' if all(b['bitwise'] for b in d['buffers'].values()) else f'within {_e(largest)}'
        native = 'bitwise' if all(b['bitwise'] for b in m['buffers'].values()) else 'DIFFER'
        lines.append(f"| {day} | {d['issued_policy']} | {s.get('A1', 0):.0f} | {s.get('v4', 0):.0f} | {s.get('v2', 0):.0f} | "
                     f"{s.get('B0', 0):.0f} | {d['total_seconds']:.0f} | {host} | {native} |")
    return '\n'.join(lines)


def shadow_final(root: Path) -> str:
    r = _load('shadow-run.json', root)
    lines = ['| Mode | Days | Final buffers equal to CP-17\'s recorded state | Largest standardized difference | '
             'Every day\'s buffers | Every shadow registered | Other |', '|---|---:|---|---:|---|---|---|']
    for mode in ('native', 'host'):
        v = r[mode]
        eq = v['final'].get('equal_to_cp17_bitwise', {})
        final = ', '.join(f"{p} {'bitwise' if ok else 'not bitwise'}" for p, ok in sorted(eq.items()))
        s = v['summary']
        other = (f"CP-26's stored forecasts reproduced bitwise: {'yes' if s.get('all_parity_cp26') else 'NO'}" if mode == 'native'
                 else f"total per day max {s['total_seconds_max']:.0f} s, median {s['total_seconds_median']:.0f} s; "
                      f"within the 900 s ceiling: {'yes' if s['all_within_ceiling'] else 'NO'}; peak {_mib(s['peak_bytes_max'])} MiB")
        lines.append(f"| {mode} | {s['days']} | {final} | {_e(s['max_abs_standardized'])} | "
                     f"{'bitwise' if s['all_buffers_bitwise'] else 'within rounding'} | {'yes' if s['all_registered'] else 'NO'} | {other} |")
    return '\n'.join(lines)


def b0_table(root: Path) -> str:
    r = _load('b0-rebuild.json', root)
    reasons = '; '.join(f'{k}: {v}' for k, v in sorted(r['reasons_counted'].items()))
    return '\n'.join(['| Question | Answer |', '|---|---|',
                      f"| Days CP-26 consumed into B0's buffer | {r['cp26_consumed_days']:,} |",
                      f"| Of them, rebuilt from prices bit for bit | {r['common_days_bitwise_equal']:,} of {r['common_days']:,} |",
                      f"| Days the price rebuild takes and CP-26 did not | {len(r['days_only_in_price_rebuild'])} ({reasons}) |",
                      f"| Days CP-26 took and the price rebuild cannot | {len(r['days_only_in_cp26'])} |",
                      f"| Origins whose 28-day buffer differs | {r['origins_whose_28_day_buffer_differs']:,} of {r['origins_compared']:,} |",
                      f"| Conclusion | {r['conclusion']} |"])


def control_table(root: Path) -> str:
    c = _load('control.json', root)
    i = _load('issuance.json', root)
    path = lambda p: ' → '.join(f'{a} {s}' for a, s in p)       # noqa: E731
    return '\n'.join(['| Case | Path | Display | Artifact verifies | A1\'s cause |', '|---|---|---|---|---|',
                      f"| N1 control: A1's process killed after {c['kill_after_seconds']:.0f} s, a stale A1 record planted | "
                      f"{path(c['path'])} | {c['display']} | {'yes' if c['artifact_verifies'] else 'NO'} | {c['causes'].get('A1')} |",
                      f"| The freeze-origin issuance, {i['day']} | {path(i['path'])} | {i['display']} | "
                      f"{'yes' if i['artifact_verifies'] else 'NO'} | — |"])


def render(root: Path = Path('.')) -> str:
    parts = [('Item 6: native parity, against CP-17\'s record', native_table),
             ('Item 6: issuing-host parity and browser identity, against CP-17\'s record', host_table),
             ('Item 1: partial days, under the issuing host and natively', partial_table),
             ('Item 2: the shadow run, day by day', shadow_table),
             ('Item 2: the shadow run, at the freeze origin', shadow_final),
             ('Item 2: B0\'s buffer rebuilt from published prices', b0_table),
             ('N1 and item 5: the killed-process control and the freeze-origin issuance', control_table)]
    body = '\n\n'.join(f'### {title}\n\n{fn(Path(root))}' for title, fn in parts)
    return f'{BEGIN}\n\n{body}\n\n{END}'


def main() -> int:
    path = REPORTS / 'report.md'
    text = path.read_text()
    a, b = text.index(BEGIN), text.index(END) + len(END)
    path.write_text(text[:a] + render() + text[b:])
    print('rendered')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
