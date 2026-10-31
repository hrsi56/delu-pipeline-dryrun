"""The report's tables, rendered from the committed records, so that every figure in them is the record's.

`reports/cp17/report.md` holds the generated block between the markers below; `render` writes it, and
`tests/cp17/test_records.py` requires the committed block to equal a fresh rendering.

    PYTHONPATH=src python -m cp17.tools.report
"""
from __future__ import annotations

import json
from pathlib import Path

REPORTS = Path('reports/cp17')
BEGIN, END = '<!-- generated tables: begin -->', '<!-- generated tables: end -->'
ORDER = ('2025-10-26', '2026-03-29', '2026-08-31', '2026-09-01', '2026-09-02', '2026-09-03', '2026-09-04', '2026-09-05',
         '2026-09-06')


def _load(name: str, root: Path):
    return json.loads((root / REPORTS / name).read_text())


def _mib(n) -> str:
    return f'{n / 2 ** 20:,.0f}'


def _mb(n) -> str:
    return f'{n / 1e6:,.1f}'


def _e(x) -> str:
    return '0' if x == 0 else f'{x:.1e}'


def native_table(root: Path) -> str:
    r = _load('native-parity.json', root)
    lines = ['| Origin (delivery) | A1 | v5 | v4 | v2 | B0 | A1 fit (s) | v4 fit (s) |', '|---|---|---|---|---|---|---:|---:|']
    for day in ORDER:
        o = r['origins'][day]
        cell = {p: 'bitwise' if o['checks'][p]['pass'] else 'FAIL' for p in ('A1', 'v5', 'v4', 'v2', 'B0')}
        lines.append(f"| {day} | {cell['A1']} | {cell['v5']} | {cell['v4']} | {cell['v2']} | {cell['B0']} | "
                     f"{o['fit_seconds']['A1']:.1f} | {o['fit_seconds']['v4']:.1f} |")
    lines.append('')
    lines.append(f"Status {r['status']}; BLAS threads {sorted(set(r['blas_threads'].values()))}; interval-state days checked "
                 f"against CP-26's consumption trace: {r['states']['checked_days']}.")
    return '\n'.join(lines)


def host_table(root: Path) -> str:
    r = _load('host-parity.json', root)
    rows = {x['stem']: x for x in r['issuing_host_parity']}
    lines = ['| Run | Status vs native | Largest deviation (EUR/MWh) | Selections matched | Fit (s) | Native fit (s) | '
             'Wasm heap (MiB) | Peak RSS (MiB) |', '|---|---|---:|---|---:|---:|---:|---:|']
    stems = [f'A1-{d}' for d in ORDER] + ['v4-2026-03-29', 'v4-2026-09-06', 'v2-2026-03-29', 'v2-2026-09-06',
                                          'B0-2026-03-29', 'B0-2026-09-06']
    for stem in stems:
        x = rows[stem]
        if x['status'] == 'failed':
            lines.append(f"| {stem} | failed | | | | | | |")
            continue
        lines.append(f"| {stem} | {x['status']} | {_e(x['max_abs_eur'])} | {'yes' if x['selections_match'] else 'NO'} | "
                     f"{x['fit_seconds']:.1f} | {x['native_fit_seconds']:.1f} | {_mib(x['wasm_memory_bytes'])} | "
                     f"{_mib(x['process_rss_peak_bytes'])} |")
    blocked = sum(len(x['blocked_non_local_requests']) for x in rows.values())
    files = all(x['executed_files_match_package'] for x in rows.values())
    lines += ['', '| Run | Member deviations, issuing host vs native (EUR/MWh) |', '|---|---|']
    for stem in stems:
        members = rows[stem].get('members_max_abs_eur')
        if not members:
            continue
        if stem.startswith('A1-'):
            worst = max(members, key=members.get)
            rest = max(v for k, v in members.items() if k != worst)
            lines.append(f"| {stem} | largest: {worst} {_e(members[worst])}; the other seven at most {_e(rest)} |")
        else:
            lines.append(f"| {stem} | {', '.join(f'{k} {_e(v)}' for k, v in members.items())} |")
    lines.append('')
    lines.append(f"Issuing host: Node.js {r['issuing_host']['version']} (SHA-256 {r['issuing_host']['sha256'][:12]}…), "
                 f"Pyodide 0.28.0; τ_eq = {r['tau_eq_eur']} EUR/MWh. Non-local requests: {blocked}. Executed files equal "
                 f"the package: {'yes' if files else 'NO'}. Attribution was computed at 2026-09-06 and 2026-03-29.")
    return '\n'.join(lines)


def browser_table(root: Path) -> str:
    r = _load('host-parity.json', root)
    lines = ['| Browser | Run | vs the issuing host | vs native (largest, EUR/MWh) | Selections matched | Fit (s) | '
             'Wasm heap (MiB) | Browser RSS peak (MiB) | Training payload (MB, gzip) | First visit (KB) |',
             '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for x in r['browser_runs']:
        if x.get('errors'):
            lines.append(f"| {x['browser']} | {x['stem']} | failed: {x['errors'][0][:80]} | | | | | | | |")
            continue
        p = x['payload_bytes']
        lines.append(f"| {x['browser']} {x['browser_version']} | {x['stem']} | {x['status_vs_issuing_host']} | "
                     f"{x['status_vs_native']} ({_e(x['max_abs_eur_vs_native'])}) | "
                     f"{'yes' if x['selections_match_native'] else 'NO'} | {x['time_seconds']['fit']:.1f} | "
                     f"{_mib(x['memory_bytes']['wasm_linear_memory_peak'])} | "
                     f"{_mib(x['memory_bytes']['browser_processes_rss_peak'])} | "
                     f"{_mb(p['training_payload'])} ({_mb(p['training_payload_gzip'])}) | {p['first_visit'] / 1e3:,.1f} |")
    return '\n'.join(lines)


def browser_payload_table(root: Path) -> str:
    r = _load('host-parity.json', root)
    lines = ['| Browser | Run | Runtime (MB) | Packages (MB) | Code (KB) | Data (MB) | Ready to fit (s) | Attribution (s) |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for x in r['browser_runs']:
        if x.get('errors'):
            continue
        raw, t = x['payload_bytes']['raw'], x['time_seconds']
        lines.append(f"| {x['browser']} | {x['stem']} | {_mb(raw.get('runtime', 0))} | {_mb(raw.get('packages', 0))} | "
                     f"{raw.get('code', 0) / 1e3:,.1f} | {_mb(raw.get('data', 0))} | {t['payload_ready']:.1f} | "
                     f"{t['attribution'] or 0:.1f} |")
    return '\n'.join(lines)


def issuance_table(root: Path) -> str:
    r = _load('issuance.json', root)
    lines = ['| Case | Path (issuing host) | Same path natively | Display | Issued vs native | Wall (s) | Peak (MiB) |',
             '|---|---|---|---|---|---:|---:|']
    for key in ('2026-09-06-gfs1', '2026-09-06-gfs0', '2022-12-04-gfs1'):
        x = r[key]
        c = x['comparison']
        path = ' → '.join(f'{p} {s}' for p, s in c['host_path'])
        lines.append(f"| {x['day']}, GFS {'available' if x['gfs_available'] else 'declared unavailable'} | {path} | "
                     f"{'yes' if c['same_path'] else 'NO'} | {x['display']} | {c.get('status_vs_native')} "
                     f"({_e(c.get('max_abs_eur_vs_native') or 0)}) | {x['resources']['wall_seconds']:.1f} | "
                     f"{_mib(x['resources']['peak_bytes'])} |")
    return '\n'.join(lines)


def controls_table(root: Path) -> str:
    r = _load('controls.json', root)
    t = r['target_day_outcome']
    s, w = r['seed'], r['training_window']
    lines = ['| Control | Change | Status | Quantiles and selections alone | Largest deviation (EUR/MWh) | Expected |',
             '|---|---|---|---|---:|---|',
             f"| Changed seed | {s['change']} | {s['status']} | {s['measured_status']} | {_e(s['max_abs_eur'])} | different |",
             f"| Changed training-window value | {w['change']} | {w['status']} | {w['measured_status']} | "
             f"{_e(w['max_abs_eur'])} | different |",
             f"| Changed target-day outcome | {t['target_day_changed']} | payloads unchanged: "
             f"{', '.join(p for p in ('A1', 'v2', 'v4', 'B0') if t[p]['payload_sha256_equal'] and t[p]['file_bytes_equal'])} "
             f"| — | — | unchanged |"]
    return '\n'.join(lines)


def scoring_table(root: Path) -> str:
    r = _load('scoring-check.json', root)
    lines = ['| Window / stream | Days | Hours | Partial days | Negative-price hours | Items compared | Native = issuing host |',
             '|---|---:|---:|---:|---:|---:|---|']
    for name, s in r['summary'].items():
        c = r['recompute_checks'][name]
        lines.append(f"| {name} | {s['days']} | {s['hours']:,} | {s['partial_days']} | {s['negative_price_hours']} | "
                     f"{c['compared']:,} | {'bit for bit' if r['bitwise_equal'][name] else ('within tolerance' if c['match'] else 'MISMATCH')} |")
    return '\n'.join(lines)


def render(root: Path = Path('.')) -> str:
    parts = [('Native parity (the code-identity check)', native_table),
             ('Issuing-host parity (Pyodide 0.28.0 in Node.js)', host_table),
             ('Browser identity and v4 in the browser', browser_table),
             ('What the browser loaded', browser_payload_table),
             ('The failure policy, exercised under the issuing host', issuance_table),
             ('Retrain negative controls', controls_table),
             ('§19.5 records, native against the issuing host', scoring_table)]
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
