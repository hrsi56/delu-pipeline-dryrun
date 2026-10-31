"""CP-21 durable evidence (§17.10 item 10): failures, resources and the artifact manifest.

`resources.json` is the ledger at the candidate: every §17.8 counter against its cap, peaks,
jobs, pauses and the active-hour upper bound. The final totals, which include the independent
review, are recorded after it in `docs/track-b/evidence/cp-21/resource-final.json`.
`artifact-manifest.json` binds every committed CP-21 file by SHA-256.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pandas as pd

from cp15.data import sha
from .budget import CAPS, GAUGES, TIMEBOX_ACTIVE_SECONDS, Budget, atomic, ledger
from .execution import OUT
from .jobs import art, stamp

MANIFESTED = ('reports/block-challenger', 'src/cp21', 'tests/cp21', 'scripts/cp21_blocks.py', 'scripts/mlflow_export.py',
              'docs/track-b/research-content/cp21-claims.md', 'docs/track-b/evidence/cp-21/issued-brief.md',
              'docs/track-b/evidence/cp-21/publication-packet.md', 'docs/track-b/evidence/cp-21/.gitattributes')


def failures_table(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((art() / 'failures').glob('*.json')) if (art() / 'failures').exists() else []:
        item = json.loads(path.read_text())
        rows.append({k: item[k] for k in ('fold', 'day', 'arm', 'stage', 'error', 'written_utc')})
    frame = pd.DataFrame(rows, columns=['fold', 'day', 'arm', 'stage', 'error', 'written_utc'])
    frame.to_csv(root / OUT / 'failures.csv', index=False)
    return frame


def resources(state: dict) -> dict:
    counts, peaks = state['counts'], state['peaks']
    jobs = {}
    for job in state['jobs']:
        j = jobs.setdefault(job['name'], {'jobs': 0, 'charged_machine_hours': 0.0, 'exit_codes': {}, 'cpu_seconds_observed': 0.0})
        j['jobs'] += 1
        j['charged_machine_hours'] += job.get('charged_seconds', 0) / 3600
        j['cpu_seconds_observed'] += job.get('cpu_seconds_observed', 0.0)
        code = str(job.get('exit_code'))
        j['exit_codes'][code] = j['exit_codes'].get(code, 0) + 1
    caps = {}
    for key, cap in CAPS.items():
        used = peaks.get(key, 0) if key in GAUGES else counts.get(key, 0)
        caps[key] = {'cap': cap, 'used': used}
    caps['workers']['used'] = max((j['workers'] for j in state['jobs']), default=0)
    active = Budget.active_seconds(state)
    caps['active_seconds']['used'] = active  # computed from the session start minus pauses, never a charged counter
    return {'schema': 'cp21-resources-v1', 'written_utc': stamp(),
            'caps_vs_use': caps, 'tracked': {k: v for k, v in counts.items() if k not in CAPS},
            'machine_hours': counts.get('machine_seconds', 0) / 3600, 'machine_hours_cap': CAPS['machine_seconds'] / 3600,
            'cpu_hours_observed': sum(j['cpu_seconds_observed'] for j in jobs.values()) / 3600,
            'active_hours_upper_bound': active / 3600, 'active_hours_timebox': TIMEBOX_ACTIVE_SECONDS / 3600,
            'active_hours_hard_stop': CAPS['active_seconds'] / 3600,
            'active_hours_basis': 'from the Lead\'s first command 2026-09-29T15:51:31Z minus recorded pauses (the usage-limit idle gap)',
            'pauses': state['effort']['pauses'], 'peak_aggregate_rss_gib': peaks.get('rss_bytes', 0) / 1024**3,
            'peak_added_disk_gib': peaks.get('additional_disk_bytes', 0) / 1024**3, 'jobs_by_name': jobs,
            'jobs_total': len(state['jobs']), 'download_bytes': counts.get('download_bytes', 0),
            'remote_writes': counts.get('remote_writes', 0), 'gpu': 0, 'cloud': 0, 'external_cost_usd': 0,
            'blas_threads': 1, 'events': state['events'],
            'ledger': '.local/artifacts/cp-21/ledger/budget.json (cumulative, never reset)'}


def manifest(root: Path) -> dict:
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', *MANIFESTED],
                                    cwd=root, text=True).split('\n')
    files = sorted({f for f in files if f and (root / f).is_file() and not f.endswith('artifact-manifest.json')})
    return {'schema': 'cp21-artifact-manifest-v1', 'evidence_class': 'development_post_selection',
            'artifact_sha256': {f: sha(root / f) for f in files}}


def job_finalise(root: Path, rest) -> int:
    """Failures and the resource snapshot, then the manifest; `--manifest-only` rebinds the
    manifest after the report (which quotes the snapshot) without moving the snapshot."""
    root = Path(root)
    if '--manifest-only' in rest:
        atomic(root / OUT / 'artifact-manifest.json', manifest(root))
        print(json.dumps({'manifest_files': len(manifest(root)['artifact_sha256'])}), flush=True)
        return 0
    failures = failures_table(root)
    atomic(root / OUT / 'resources.json', resources(ledger().read()))
    atomic(root / OUT / 'artifact-manifest.json', manifest(root))
    print(json.dumps({'failures': len(failures), 'manifest_files': len(manifest(root)['artifact_sha256'])}), flush=True)
    return 0
