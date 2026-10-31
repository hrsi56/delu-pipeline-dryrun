"""CP-17c-probe's report: every table rendered from `reports/cp17c-probe/measurement.json`, so a figure in the
report is the measurement's (a test renders it again and compares).

    PYTHONPATH=src python -m cp17cprobe.report --measurement reports/cp17c-probe/measurement.json \\
        --proposals reports/cp17c-probe/proposals.md --out reports/cp17c-probe/report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MIB = 1024 ** 2
KINDS = (('parity', 'The issuing-host parity origins (`reports/cp17b/host-parity.json`)'),
         ('f1', "F1's day, 2026-03-30 (`reports/cp17b/partial-days.json`)"))


def _yes(value) -> str:
    return 'yes' if value else ('**no**' if value is not None else '—')


def _eur(value) -> str:
    if value is None:
        return '—'
    return '0' if value == 0 else f'{value:.3g}'


def _s(value) -> str:
    return '—' if value is None else f'{value:.1f}'


def _mib(value) -> str:
    return '—' if value is None else f'{value / MIB:.0f}'


def host_table(runners: dict) -> list[str]:
    """One row per job: each job is its own machine, so its image, CPU and CPython are its own (a runner label can
    land on more than one image version or CPU)."""
    out = ['| Runner | Job | Image | OS | CPU | Logical cores | Memory | Node.js binary (SHA-256) | Node.js / V8 | '
           'Pyodide\'s 23 files = the freeze manifest\'s | CPython (the chain\'s side) |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    for label, r in runners.items():
        for name, job in sorted(r['jobs'].items()):
            host = job.get('host')
            if not host:
                continue
            m, env = host['machine'], host['machine']['runner_env']
            image = f"{env['ImageOS']} {env['ImageVersion']}" if env.get('ImageOS') else '— (this Mac)'
            out.append(f"| {label} | {name} | {image} | {m.get('os_version')} | {m.get('cpu_model')} | {m.get('cpu_count')} | "
                       f"{m.get('memory_bytes', 0) / 1024 ** 3:.1f} GiB | `{host['node']['sha256'][:16]}…` | "
                       f"{host['node']['version']} / {host['node']['v8']} | "
                       f"{_yes(host['pyodide'].get('files_equal_freeze_manifest'))} | {host['python']} |")
    return out


def fit_table(rows: dict) -> list[str]:
    out = ['| Fit | Bit-identical | Status (frozen comparison) | Largest difference, EUR/MWh | Selections | Payload | '
           'Wall s (reference) | Peak MiB (reference) | Within 15 min and 4 GiB |',
           '|---|---|---|---|---|---|---|---|---|']
    for stem, f in rows.items():
        out.append(f"| {stem} | {_yes(f['bit_identical'])} | {f.get('status', '—')} | {_eur(f.get('max_abs_eur'))} | "
                   f"{_yes(f.get('selections_match'))} | {_yes(f['payload_identity_equal'])} | "
                   f"{_s(f.get('wall_seconds'))} ({_s(f['reference_wall_seconds'])}) | "
                   f"{_mib(f.get('peak_bytes'))} ({_mib(f['reference_peak_bytes'])}) | {_yes(f.get('within_ceilings'))} |")
    return out


def shadow_table(shadow: dict) -> list[str]:
    out = ['| Day | Path | A1 | v4 | v2 | B0 | Buffers = reference (v4, v2, B0) | Buffers = payload | Intake | '
           'Day\'s processes, s | Largest peak, MiB |',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    days = {d: v for s in shadow.values() for d, v in s['days'].items()}
    for day, d in sorted(days.items()):
        fits = d['fits']
        cell = lambda p: 'bit-identical' if fits.get(p, {}).get('bit_identical') else (   # noqa: E731
            f"{fits[p].get('status')} ({_eur(fits[p].get('max_abs_eur'))})" if p in fits else '—')
        path = ', '.join(f'{p} {s}' for p, s in d['path'])
        buffers = ', '.join(_yes(d['buffers_equal_reference'].get(p)) for p in ('v4', 'v2', 'B0'))
        payload = ', '.join(_yes(d['buffers_equal_payload'].get(p)) for p in ('v4', 'v2', 'B0'))
        intake = 'all registered' if set(d['intake'].values()) == {'registered'} else json.dumps(d['intake'])
        peak = max((f['peak_bytes'] or 0) for f in fits.values()) if fits else None
        out.append(f"| {day} | {path} | {cell('A1')} | {cell('v4')} | {cell('v2')} | {cell('B0')} | {buffers} | {payload} | "
                   f"{intake} | {_s(d['day_seconds_sum'])} | {_mib(peak)} |")
    return out


def summary_table(runners: dict) -> list[str]:
    out = ['| | ' + ' | '.join(runners) + ' |', '|---|' + '---|' * len(runners)]

    def row(label, get):
        values = []
        for r in runners.values():
            try:
                values.append(str(get(r['summary'])))
            except (KeyError, TypeError):
                values.append('—')
        out.append(f'| {label} | ' + ' | '.join(values) + ' |')
    row('Fits measured', lambda s: s['all']['fits'])
    row('Bit-identical to the frozen host', lambda s: f"{s['all']['bit_identical']} of {s['all']['fits']}")
    row('Every data-driven selection matched', lambda s: _yes(s['all']['all_selections_match']))
    row('Every payload identity equal', lambda s: _yes(s['all']['all_payload_identities_equal']))
    row('Executed files = the frozen package', lambda s: _yes(s['all']['all_executed_files_match_package']))
    row('Network requests refused', lambda s: len(s['all']['blocked_non_local_requests']))
    row('Every fit within 15 min and 4 GiB', lambda s: _yes(s['all']['all_within_ceilings']))
    row('Longest fit, s', lambda s: _s(s['all']['wall_seconds_max']))
    row('Largest peak, MiB', lambda s: _mib(s['all']['peak_bytes_max']))
    row('Wall time / the frozen host\'s, median (max)',
        lambda s: f"{s['all']['wall_ratio_to_reference_median']:.2f} ({s['all']['wall_ratio_to_reference_max']:.2f})")
    row('Shadow days', lambda s: s['shadow_days'])
    row('Buffers = the reference\'s at every origin', lambda s: _yes(s['shadow_buffers_all_equal_reference']))
    row('A1 issued every day; every shadow registered',
        lambda s: _yes(s['shadow_all_issued_a1'] and s['shadow_all_registered']))
    row('Artifacts verify', lambda s: _yes(s['shadow_all_artifacts_verify']))
    row('A day\'s four processes, s: median (max)',
        lambda s: f"{_s(s['shadow_day_seconds_sum_median'])} ({_s(s['shadow_day_seconds_sum_max'])})")
    row('The two shadow jobs stitch', lambda s: _yes(s.get('stitch_early_final_equals_late_start')))
    row('Final state = the reference\'s', lambda s: _yes(s.get('late_final_equals_reference_final')))
    return out


def issuance_table(runners: dict) -> list[str]:
    """One issuance's four processes (A1 and the three shadows), one at a time, from the parity job's days that
    hold all four (2026-03-29, 2026-09-06 and F1's 2026-03-30); and the shadow days, two at once."""
    out = ['| | ' + ' | '.join(runners) + ' |', '|---|' + '---|' * len(runners)]
    days = ('2026-03-29', '2026-09-06', '2026-03-30')
    for day in days:
        cells = []
        for r in runners.values():
            fits = {**r['parity'], **r['f1']}
            stems = [f'{p}-{day}' for p in ('A1', 'v4', 'v2', 'B0')]
            cells.append(_s(sum(fits[s]['wall_seconds'] for s in stems)) if all(s in fits for s in stems) else '—')
        out.append(f'| {day}: the four processes one after another, s | ' + ' | '.join(cells) + ' |')
    cells = []
    for r in runners.values():
        s = r['summary']
        cells.append(f"{_s(s['shadow_day_seconds_sum_median'])} ({_s(s['shadow_day_seconds_sum_max'])})"
                     if 'shadow_day_seconds_sum_median' in s else '—')
    out.append('| The shadow days, two at once: a day\'s process seconds, median (max) | ' + ' | '.join(cells) + ' |')
    return out


def retrieval_table(m: dict, local: dict | None) -> list[str]:
    """Every retrieval: the Node.js tarball and binary, `uv`, and the lock its packages came from, per job."""
    out = ['| Where | What | URL | Bytes | SHA-256 |', '|---|---|---|---|---|']
    if local:
        out.append(f"| this Mac | `SHASUMS256.txt` | {local['shasums']['url']} | {local['shasums']['bytes']:,} | `{local['shasums']['sha256']}` |")
        out.append(f"| this Mac | Node.js {local['version']} tarball | {local['tarball']['url']} | {local['tarball']['bytes']:,} | "
                   f"`{local['tarball']['sha256']}` |")
        out.append(f"| this Mac | its `bin/node` | — | {local['binary']['bytes']:,} | `{local['binary']['sha256']}` |")
    for label, r in m['runners'].items():
        for name, job in sorted(r['jobs'].items()):
            node, uv = job.get('node-retrieval'), job.get('uv-retrieval')
            if node:
                out.append(f"| {name} | Node.js {node['version']} tarball | {node['tarball']['url']} | {node['tarball']['bytes']:,} | "
                           f"`{node['tarball']['sha256']}` |")
                out.append(f"| {name} | its `bin/node` | — | {node['binary']['bytes']:,} | `{node['binary']['sha256']}` |")
            if uv:
                out.append(f"| {name} | `{uv['file']}` ({uv['version']}) | {uv['url']} | {uv['bytes']:,} | `{uv['sha256']}` |")
            if job.get('project_sha256'):
                out.append(f"| {name} | the Python packages `uv.lock` pins (`uv sync --locked`); the lock | — | — | "
                           f"`{job['project_sha256'][0]}` |")
    return out


def minutes_table(minutes: dict) -> list[str]:
    out = ['| Run | Job | OS | Seconds | Setup before the chain, s | Counted minutes |', '|---|---|---|---|---|---|']
    for run_id, run in sorted(minutes['runs'].items()):
        for name, job in sorted(run['jobs'].items()):
            out.append(f"| {run_id} | {name} | {job['os']} | {job['seconds']:.0f} | {job['setup_seconds']:.0f} | "
                       f"{job['counted']} |")
    out.append(f"| **Total** | | | | | **{minutes['counted_total']}** |")
    return out


def render(m: dict, proposals: str) -> str:
    runners = m['runners']
    parts = ['# CP-17c-probe — can a GitHub-hosted runner be the issuing host?', '',
             '*Generated by `cp17cprobe.report` from `measurement.json`; the proposals are `proposals.md`.*', '',
             proposals.strip(), '', '## The hosts', '', *host_table({**runners, 'local control': m['local_control']}), '',
             '## Summary', '', *summary_table({**runners, 'local control': m['local_control']}), '',
             '## One issuance\'s cost in time', '', *issuance_table({**runners, 'local control': m['local_control']}), '']
    for label, r in {**runners, 'local control': m['local_control']}.items():
        parts += [f'## {label}', '']
        for kind, title in KINDS:
            if r[kind]:
                parts += [f'### {title}', '', *fit_table(r[kind]), '']
        if r['shadow']:
            parts += ['### The 14 shadow days (`reports/cp17b/shadow-run.json`), A1 and the three shadows', '',
                      *shadow_table(r['shadow']), '']
    if m.get('supplementary'):
        parts += ['## Supplementary: the records of the superseded run 1', '',
                  'Run 1\'s harness failed after each fit and the run was cancelled; these records are the frozen Node.js '
                  'runner\'s own, written before the failure and checked against their jobs\' own hashes. They are not part '
                  'of the measurement above. They add machines to the evidence.', '',
                  '| Job | CPU | Fit | Bit-identical | Payload | Wall s | Peak MiB |', '|---|---|---|---|---|---|---|']
        for row in m['supplementary']:
            parts.append(f"| {row['artifact']} | {row['cpu_model']} | {row['stem']} | {_yes(row['bit_identical'])} | "
                         f"{_yes(row['payload_identity_equal'])} | {_s(row['wall_seconds'])} | {_mib(row['peak_bytes'])} |")
        parts.append('')
    if m['runners'] or m.get('local_retrieval'):
        parts += ['## Retrievals', '', 'Software only. Each Node.js tarball was checked against the release\'s '
                  '`SHASUMS256.txt` (itself pinned by its SHA-256) before use; Pyodide\'s files came from the scratch '
                  'repository, checked against the freeze manifest; GitHub\'s own actions are pinned by commit SHA in the '
                  'workflow.', '', *retrieval_table(m, m.get('local_retrieval')), '']
    if 'minutes' in m:
        zero = all(run['counted_from_billable'] == 0 for run in m['minutes']['runs'].values())
        note = (' GitHub\'s run-timing API reported 0 billable milliseconds for every job, so the count is taken from the '
                'jobs\' own start and end times.' if zero else '')
        parts += ['## GitHub minutes', '', 'As GitHub\'s usage report counts them for a private repository: each job '
                  'rounded up to the minute, macOS at ten times.' + note, '', *minutes_table(m['minutes']), '']
    return '\n'.join(parts).rstrip() + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--measurement', required=True, type=Path)
    ap.add_argument('--proposals', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    args.out.write_text(render(json.loads(args.measurement.read_text()), args.proposals.read_text()))
    print(args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
