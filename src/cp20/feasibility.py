"""E1 pre-run feasibility: measured subset costs extrapolated against every section 15.5 cap.

Planning estimates, not guarantees. Transfer is extrapolated from bytes actually charged
per *completed* subset run (requests attributed by run date; partial redo bytes included,
which is conservative). Wall time is bandwidth-bound on this link, so it is projected from
measured aggregate throughput in a nominal and a pessimistic scenario; machine time is
charged as wall x declared workers, exactly as the monitor charges it.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from .budget import CAPS, GIB, HOUR, Budget


def per_run_charges(weather: Path) -> dict:
    done = {}
    for path in (weather / 'runs').glob('*.json'):
        rec = json.loads(path.read_text())
        if rec.get('status') == 'complete':
            done[rec['run_00z']] = rec
    out = {k: {'endpoint': v['messages'][0]['endpoint'], 'charged': 0,
               'payload': sum(m['bytes'] for m in v['messages']), 'elapsed_s': v['elapsed_s']} for k, v in done.items()}
    for line in (weather / 'requests.jsonl').read_text().splitlines():
        r = json.loads(line)
        m = re.search(r'(\d{4}-\d{2}-\d{2})', r['purpose'])
        if m and m.group(1) in out and not r['purpose'].startswith('added'):
            out[m.group(1)]['charged'] += r['charged_bytes']
    return out


def estimate(root: Path, weather: Path, ledger: Path, workers: int, rates: dict) -> dict:
    manifest = json.loads((root / 'reports/weather-ablation/run-manifest.json').read_text())
    runs = per_run_charges(weather)
    state = Budget(ledger).read()
    used = state['counts']
    measured = {}
    for ep in ('ncar', 'aws'):
        xs = [v['charged'] for v in runs.values() if v['endpoint'] == ep]
        measured[ep] = {'completed_runs': len(xs), 'max_charged_bytes_per_run': max(xs),
                        'median_charged_bytes_per_run': sorted(xs)[len(xs) // 2]}
    remaining = {ep: sum(r['primary_endpoint'] == ep and r['run_00z'] not in runs for r in manifest['runs'])
                 for ep in ('ncar', 'aws')}
    # Conservative per-run bytes from single-pass completed runs (no message retried: a re-fetch
    # after the restart test double-counts), NCAR at the wide pre-r3 windows; AWS at least the
    # largest v16 payload plus 2 MB of idx/request overhead.
    retried = set()
    for line in (weather / 'attempts.jsonl').read_text().splitlines():
        a = json.loads(line)
        if a['attempt'] > 1:
            retried.add(a['run'])
    single = {k: v for k, v in runs.items() if k not in retried}
    v16_payload = max(v['payload'] for k, v in runs.items() if v['endpoint'] == 'aws' and k >= '2021-03-23')
    per_run = {'ncar': max(v['charged'] for v in single.values() if v['endpoint'] == 'ncar'),
               'aws': max([v['charged'] for v in single.values() if v['endpoint'] == 'aws'] + [v16_payload + 2_000_000])}
    remaining_bytes = sum(remaining[ep] * per_run[ep] for ep in remaining) * 1.05  # 5% retries/failures
    fits = {'hg_main_component_days': 2 * 638, 'controls_component_days': 28, 'review_reserve_component_days': 60}
    fit_hours = sum(fits.values()) * 18.0 / HOUR  # CP-16 measured ~9 s per component-day, x2 allowance
    other = {'assembly_protocol_tests_scoring_report': 3.0, 'critic_review_reserve': 10.0}
    scenarios = {}
    for name, rate in rates.items():
        wall = remaining_bytes / rate / HOUR
        machine = used.get('machine_seconds', 0) / HOUR + workers * wall + fit_hours + sum(other.values())
        scenarios[name] = {'aggregate_MBps': rate / 1e6, 'extraction_wall_hours': wall,
                           'extraction_machine_hours': workers * wall, 'total_machine_hours': machine,
                           'within_machine_cap': machine <= CAPS['machine_seconds'] / HOUR}
    transfer = used.get('transfer_bytes', 0) + remaining_bytes + 2 * GIB
    return {
        'basis': 'bytes charged per single-pass completed subset run x remaining runs (+5%); bandwidth-bound wall time; machine = workers x wall',
        'single_pass_runs': sorted(single), 'retried_runs_excluded': sorted(retried),
        'measured_subset': measured, 'per_run_bytes_used': per_run, 'remaining_runs': remaining,
        'remaining_transfer_gib': remaining_bytes / GIB, 'workers_for_extraction': workers, 'scenarios': scenarios,
        'fits': {**fits, 'assumed_seconds_per_component_day': 18.0, 'machine_hours': fit_hours,
                 'primitive_fits_nominal': sum(fits.values()) * 120},
        'other_machine_hours': other,
        'policy_days_plan': {'paired_admission_plus_comparison': 2 * 638, 'controls': 14,
                             'independent_review_reserve': 2 * 638 + 100},
        'totals_vs_caps': {
            'transfer_gib': [transfer / GIB, CAPS['transfer_bytes'] / GIB],
            'component_attempts': [used.get('component_attempts', 0) + sum(fits.values()), CAPS['component_attempts']],
            'main_component_attempts': [fits['hg_main_component_days'], CAPS['main_component_attempts']],
            'primitive_fits_nominal': [sum(fits.values()) * 120, CAPS['primitive_fits']],
            'policy_days': [2 * 638 + 14 + 2 * 638 + 100, CAPS['policy_days']],
            'message_attempts': [used.get('message_attempts', 0) + 50 * sum(remaining.values()) * 1.05 + 50 * 35,
                                 CAPS['message_attempts']],
            'added_disk_gib_upper': [2476 * 0.6e6 / GIB + 1.2 + 1.0, CAPS['additional_disk_bytes'] / GIB],
            'aggregate_rss_gib_upper': [2 * 1.2 + 0.6, CAPS['rss_bytes'] / GIB]},
        'active_hours_so_far_upper_bound': Budget.active_seconds(state) / HOUR,
        'stop_lines': {'extraction_transfer_gib': 150,
                       'extraction_machine_hours_cumulative': 85,
                       'rationale': 'leave >=35 machine-hours and >=10 GiB for fitting, controls, scoring and independent review'}}
