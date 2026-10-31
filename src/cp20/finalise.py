"""CP-20 finalisation (formatting only; added after the pre-fit freeze, computes no new result).

Writes reports/weather-ablation/resources.json (cumulative ledger vs the section 15.5 caps and
extraction accounting), failures.csv (failed/stopped jobs and extraction failure classes, all
later resolved or superseded), report.md (``cp20.report.render`` over the saved tables) and
artifact-manifest.json (sha256 of every CP-20 report artifact plus the freeze commits).
Run under the monitor after ``score`` as ``python -m cp20.finalise`` (moved from
``scripts/cp20_finalise.py`` into the section 15.7 paths after the first Integration review).
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pandas as pd

from .budget import CAPS, GIB, HOUR, TIMEBOX_ACTIVE_SECONDS, Budget, atomic
from .report import render

ROOT = Path(__file__).resolve().parents[2]

OUT = ROOT / 'reports/weather-ablation'
PROJECT = Path(os.environ.get('CP20_PROJECT_ROOT', '/Users/djourno/Downloads/PJM'))
A = PROJECT / '.local/artifacts/cp-20'
WEATHER = A / 'weather'
# Failed jobs whose resolution is not a later successful job of the same name.
JOB_NOTES = {68: ('Integration attempt 1 ran the whole tests/cp16 directory beyond the listed guards: 3 failed, '
                  '3 errors (CP16_LEDGER unset; v21-r3 anchor identity). tests/cp16, src/cp15, src/cp16 and '
                  'capstone_v21.md are identical to main 88c68cf, so these are inherited, not CP-20 regressions; '
                  'the listed guards passed in job 69')}


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(name):
    path = WEATHER / name
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def git(*args) -> str:
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def extraction_accounting() -> dict:
    """Per target message (run, lead, field): the section 15.5 attempt unit."""
    outcomes = jsonl('attempt-outcomes.jsonl')
    restored_records = jsonl('o2-a1-restored-attempts.jsonl')
    restored = {(r['run'], r['lead'], r['field'], r['endpoint'], r['try_id']) for r in restored_records}
    effective, tries = Counter(), Counter()
    for r in outcomes:
        key = (r['run'], r['lead'], r['field'])
        tries[key] += 1
        if r.get('counted') and (r['run'], r['lead'], r['field'], r['endpoint'], r['try_id']) not in restored:
            effective[key] += 1
    stats = Counter()
    for line in (WEATHER / 'requests.jsonl').open():
        r = json.loads(line)
        stats['requests'] += 1
        stats[f"requests_{r['endpoint']}"] += 1
        if r.get('error'):
            stats[f"failed_{r.get('error_kind')}"] += 1
    return {
        'required_runs_complete': len(list((WEATHER / 'runs').glob('*.npz'))),
        'attempt_outcomes': dict(Counter(r['outcome'] for r in outcomes)),
        'o2_a1_restored_locator_defect_message_attempts': len(restored),
        'o2_a1_restored_lead_tries': len({(r['run'], r['lead'], r['endpoint'], r['try_id']) for r in restored_records}),
        'o1_prerun_replacement_attempts': len(jsonl('prerun_replacement_attempts.jsonl')),
        'effective_counted_attempts_after_o2_a1': sum(effective.values()),
        'target_messages_with_effective_counted_attempts': len(effective),
        'max_effective_counted_attempts_per_target_message': max(effective.values(), default=0),
        'max_tries_any_outcome_per_target_message': max(tries.values(), default=0),
        'target_messages_tried_more_than_once': sum(1 for v in tries.values() if v > 1),
        'target_messages_seen': len(tries),
        'request_log': dict(stats),
        'classified_failure_records': len(jsonl('failures.jsonl')),
        'failure_records_resolved_by_later_completion': True,
        'job_code_versions': [{k: r.get(k) for k in ('job_index', 'endpoint', 'workers', 'start_utc')}
                              | {'git_head': r['code_version']['git_head'][:12]} for r in jsonl('job-code-versions.jsonl')],
    }


def resources() -> dict:
    state = Budget(A / 'ledger/budget.json').read()
    counts, peaks = state['counts'], state['peaks']
    active = Budget.active_seconds(state)
    jobs = state['jobs']
    by_name = {}
    for j in jobs:
        n = by_name.setdefault(j['name'], {'jobs': 0, 'charged_machine_hours': 0.0, 'exit_codes': Counter()})
        n['jobs'] += 1
        n['charged_machine_hours'] += (j.get('charged_seconds') or 0) / HOUR
        n['exit_codes'][str(j.get('exit_code'))] += 1
    for n in by_name.values():
        n['charged_machine_hours'] = round(n['charged_machine_hours'], 3)
        n['exit_codes'] = dict(n['exit_codes'])
    usage = {k: {'used': counts.get(k, 0), 'cap': CAPS[k]} for k in CAPS if k not in ('rss_bytes', 'additional_disk_bytes', 'workers')}
    usage['active_seconds']['used'] = round(active, 1)  # derived from effort and pauses, not a counter
    return {
        'schema': 'cp20-resources-v1', 'ledger': '.local/artifacts/cp-20/ledger/budget.json (cumulative, never reset)',
        'caps_vs_use': usage,
        'machine_hours': round(counts.get('machine_seconds', 0) / HOUR, 3), 'machine_hours_cap': CAPS['machine_seconds'] / HOUR,
        'extraction_machine_hour_stop_line': 100,
        'active_hours': round(active / HOUR, 3), 'active_hours_timebox': TIMEBOX_ACTIVE_SECONDS / HOUR,
        'active_hours_hard_stop': CAPS['active_seconds'] / HOUR,
        'active_hours_basis': 'from the conservative session start 2026-09-23T15:50Z minus recorded pauses (usage-limit idle gap; Owner decision O3: unattended waits)',
        'pauses': state['effort']['pauses'],
        'transfer_gib': round(counts.get('transfer_bytes', 0) / GIB, 3), 'transfer_gib_cap': CAPS['transfer_bytes'] / GIB,
        'transfer_stop_line_gib': 150,
        'peak_aggregate_rss_gib': round(peaks.get('rss_bytes', 0) / GIB, 3), 'rss_cap_gib': CAPS['rss_bytes'] / GIB,
        'peak_added_disk_gib': round(peaks.get('additional_disk_bytes', 0) / GIB, 3), 'added_disk_cap_gib': CAPS['additional_disk_bytes'] / GIB,
        'workers_max': CAPS['workers'], 'blas_threads': 1,
        'message_attempts_note': ('ledger message_attempts is the cumulative charge: 1,885 pre-O2 charges, the 4,800 '
                                  'locator-defect failures later restored under O2-A1 (not refunded in the ledger) and one '
                                  'charge per retained raw message decoded by an Integration review; message_tries counts '
                                  'every extraction try of any outcome'),
        'tracked': {k: counts.get(k, 0) for k in ('requests', 'failed_requests', 'network_retries', 'message_tries',
                                                    'uncounted_message_tries', 'decoded_messages', 'runs_completed',
                                                    'prerun_replacement_attempts')},
        'jobs_total': len(jobs), 'jobs_by_name': by_name,
        'events': state['events'],
        'extraction': extraction_accounting(),
        'external_cost_usd': 0, 'gpu': 0, 'cloud': 0,
    }


def failures(res: dict) -> pd.DataFrame:
    rows = []
    jobs = Budget(A / 'ledger/budget.json').read()['jobs']
    for j in jobs:
        if j.get('exit_code') not in (0, None) or j.get('abort_reason'):
            later = [k['index'] for k in jobs if k['name'] == j['name'] and k['index'] > j['index'] and k.get('exit_code') == 0]
            resolution = (JOB_NOTES.get(j['index']) or (f"later successful job(s) of the same name: {', '.join(map(str, later))}" if later
                                                        else 'not superseded; see the job log'))
            rows.append({'kind': 'job', 'job_index': j['index'], 'name': j['name'], 'exit_code': j.get('exit_code'),
                         'abort_reason': j.get('abort_reason'), 'resolution': resolution})
    for cls, n in Counter(r['class'] for r in jsonl('failures.jsonl')).items():
        rows.append({'kind': 'extraction_run_failure_records', 'name': cls, 'count': n,
                     'resolution': 'every affected run later completed; none imputed'})
    for outcome, n in res['extraction']['attempt_outcomes'].items():
        if outcome != 'success':
            rows.append({'kind': 'message_attempt_outcome', 'name': outcome, 'count': n,
                         'resolution': 'see extraction-repairs.json (r1-r12, O1, O2, O2-A1)'})
    frame = pd.DataFrame(rows)
    for column in ('job_index', 'exit_code', 'count'):
        frame[column] = frame[column].astype('Int64')
    return frame


def post_freeze_section() -> str:
    sup = json.loads((OUT / 'causal-controls-supplement.json').read_text())
    lines = ['', '## Post-freeze repair r13: available-weather positive control', '',
             'The frozen "available weather" control above multiplies all available weather by 3.0. The inherited '
             'training-only standardisation cancels a uniform rescaling exactly, so its movement (at most 4.4e-13) is '
             'floating-point round-off: it is a units/scaling-invariance check, not a positive control. The frozen '
             'control and its output are unchanged. Two non-affine positive controls on the same days replace it as '
             f'evidence (`causal-controls-supplement.json`, pass threshold {sup["threshold_eur_mwh"]} EUR/MWh):', '']
    for c in sup['controls']:
        lines.append(f"- {c['fold']} {c['day']}: target-day D−1 weather shift {c['target_day_weather_shift']}; "
                     f"training-weather permutation (seed {sup['seed']}, {c['training_rows_permuted']} rows) "
                     f"{c['training_weather_permutation']}; base refit vs cached HG {c['base_refit_vs_cached_hg']}.")
    lines += ['', f"All passed: {sup['all_passed']}. Post-freeze changes are listed in `post-freeze-repairs.json`; "
              'rights and attribution for the weather-derived artifacts are in `rights-notice.md`.', '']
    return '\n'.join(lines)


def main():
    res = resources()
    atomic(OUT / 'resources.json', res)
    failures(res).to_csv(OUT / 'failures.csv', index=False)
    brief = {k: v for k, v in res.items() if k not in ('events', 'extraction')} | {
        'events': f"{len(res['events'])} ledger events: see resources.json",
        'extraction': {k: v for k, v in res['extraction'].items() if k != 'job_code_versions'}}
    (OUT / 'report.md').write_text(render(ROOT, brief) + post_freeze_section())
    names = sorted(p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file() and p.name != 'artifact-manifest.json')
    atomic(OUT / 'artifact-manifest.json', {
        'schema': 'cp20-artifact-manifest-v1', 'evidence_class': 'development_post_selection',
        'protocol_freeze_commit': git('log', '-1', '--format=%H', '--diff-filter=A', '--', 'reports/weather-ablation/protocol.json'),
        'admission_freeze_commit': git('log', '-1', '--format=%H', '--diff-filter=A', '--', 'reports/weather-ablation/lineage.json'),
        'weather_design_sha256': json.loads((OUT / 'protocol.json').read_text())['weather_design_sha256'],
        'artifact_sha256': {n: sha(ROOT / n) for n in names},
        'finaliser': 'src/cp20/finalise.py (formatting only; added after the pre-fit freeze)'})
    print(json.dumps({k: res[k] for k in ('machine_hours', 'active_hours', 'transfer_gib', 'jobs_total')}))


if __name__ == '__main__':
    main()
