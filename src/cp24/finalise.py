"""CP-24 durable evidence (§23.13 items 15-16): failures, resources and the artifact manifest.

`reports/ddnn2/resources.json` is the ledger at the candidate: every §23.11 counter against its cap (with
every committed §23.6 raise), peaks, jobs, pauses and the active-hour upper bound. The final totals, which
include the independent review, are recorded after it in `docs/track-b/evidence/cp-24/resource-final.json`.
`reports/ddnn2/artifact-manifest.json` binds every committed CP-24 file by SHA-256 (byte-exact storage).
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pandas as pd

from cp15.data import sha
from .budget import CAPS, GAUGES, HOUR, TIMEBOX_ACTIVE_SECONDS, Budget, atomic, effective_caps, ledger
from .jobs import art, stamp

OUT = Path('reports/ddnn2')
MANIFESTED = ('reports/ddnn2', 'src/cp24', 'tests/cp24', 'scripts/cp24_ddnn2.py',
              'docs/track-b/research-content/cp24-claims.md', 'docs/track-b/evidence/cp-24/issued-brief.md',
              'docs/track-b/evidence/cp-24/publication-packet.md', 'docs/track-b/evidence/cp-24/continuation-brief.md',
              'docs/track-b/evidence/cp-24/work-availability-2026-10-10.md',
              'docs/track-b/evidence/cp-24/steering',
              'docs/track-b/evidence/cp-24/.gitattributes')


def failures_table(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(art().rglob('failures/*.json')) + sorted((art() / 'failures').glob('*.json') if (art() / 'failures').exists() else []):
        item = json.loads(path.read_text())
        rows.append({'file': str(path.relative_to(art())), **{k: item.get(k) for k in ('fold', 'day', 'stage', 'error', 'written_utc')}})
    seen, unique = set(), []
    for r in rows:
        if r['file'] not in seen:
            seen.add(r['file'])
            unique.append(r)
    for rd in sorted((art() / 'rounds').glob('round-*')):
        for path in sorted(rd.rglob('*.json')):
            try:
                item = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get('status') == 'failed':
                unique.append({'file': str(path.relative_to(art())), 'fold': item.get('fold'),
                               'day': item.get('day') or item.get('batch_start'), 'stage': 'search' if '/search/' in str(path) else 'gate',
                               'error': item.get('error'), 'written_utc': item.get('written_utc')})
    frame = pd.DataFrame(unique, columns=['file', 'fold', 'day', 'stage', 'error', 'written_utc'])
    frame.to_csv(Path(root) / OUT / 'failures.csv', index=False)
    return frame


def resources(state: dict) -> dict:
    counts, peaks = state['counts'], state['peaks']
    caps = effective_caps(state)
    jobs = {}
    for job in state['jobs']:
        j = jobs.setdefault(job['name'], {'jobs': 0, 'charged_machine_hours': 0.0, 'exit_codes': {}, 'cpu_seconds_observed': 0.0})
        j['jobs'] += 1
        j['charged_machine_hours'] += job.get('charged_seconds', 0) / HOUR
        j['cpu_seconds_observed'] += job.get('cpu_seconds_observed', 0.0)
        code = str(job.get('exit_code'))
        j['exit_codes'][code] = j['exit_codes'].get(code, 0) + 1
    out_caps = {}
    for key, cap in caps.items():
        used = peaks.get(key, 0) if key in GAUGES else counts.get(key, 0)
        out_caps[key] = {'cap': cap, 'base_cap': CAPS[key], 'used': used}
    out_caps['workers']['used'] = max((j['workers'] for j in state['jobs']), default=0)
    active = Budget.active_seconds(state)
    out_caps['active_seconds']['used'] = active
    return {'schema': 'cp24-resources-v1', 'written_utc': stamp(), 'caps_vs_use': out_caps, 'raises': state.get('raises', []),
            'tracked': {k: v for k, v in counts.items() if k not in CAPS},
            'machine_hours': counts.get('machine_seconds', 0) / HOUR, 'machine_hours_cap': caps['machine_seconds'] / HOUR,
            'cpu_hours_observed': sum(j['cpu_seconds_observed'] for j in jobs.values()) / HOUR,
            'active_hours_upper_bound': active / HOUR, 'active_hours_timebox': TIMEBOX_ACTIVE_SECONDS / HOUR,
            'active_hours_hard_stop': caps['active_seconds'] / HOUR,
            'active_hours_basis': 'from 2026-10-05T03:14:00Z (before the Lead\'s first command) minus recorded pauses',
            'pauses': state['effort']['pauses'], 'pause_reasons': state['effort'].get('pause_reasons', []),
            'peak_aggregate_rss_gib': peaks.get('rss_bytes', 0) / 1024**3,
            'peak_added_disk_gib': peaks.get('additional_disk_bytes', 0) / 1024**3, 'jobs_by_name': jobs,
            'jobs_total': len(state['jobs']), 'data_download_bytes': counts.get('data_download_bytes', 0),
            'test_dependency_download_bytes': counts.get('test_dependency_download_bytes', 0),
            'remote_writes': counts.get('remote_writes', 0), 'gpu': 0, 'mps': 0, 'cloud': 0, 'external_cost_usd': 0,
            'blas_threads': 1, 'events': state['events'],
            'ledger': '.local/artifacts/cp-24/ledger/budget.json (cumulative, never reset)'}


def manifest(root: Path) -> dict:
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', *MANIFESTED],
                                    cwd=root, text=True).split('\n')
    files = sorted({f for f in files if f and (Path(root) / f).is_file() and not f.endswith('artifact-manifest.json')})
    return {'schema': 'cp24-artifact-manifest-v1', 'evidence_class': 'development_post_selection',
            'artifact_sha256': {f: sha(Path(root) / f) for f in files}}


def job_finalise(root: Path, rest) -> int:
    """Failures and the resource snapshot, then the manifest; `--manifest-only` rebinds the manifest."""
    root = Path(root)
    if '--manifest-only' not in rest:
        failures_table(root)
        atomic(root / OUT / 'resources.json', resources(ledger().read()))
    atomic(root / OUT / 'artifact-manifest.json', manifest(root))
    print(json.dumps({'manifest_files': len(manifest(root)['artifact_sha256'])}), flush=True)
    return 0
