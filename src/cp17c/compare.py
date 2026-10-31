"""CP-17c's comparison, offline: the runner's and the Owner's M3's results against CP-17b's records.

- **fetch** (needs `gh`, read-only): one run's artifacts, each downloaded as its zip and checked against the digest
  GitHub recorded, with the run's jobs, steps and billable time.
- **compare:** from the downloaded results (each file checked against its job's `results-sha256.txt`) and the
  local results, for each machine:
  - **the H2 checks:** the gate passed in every run, `refreeze verify` printed exactly `verified`, and each forced
    failure stopped at its own check with nothing issued; the machine as recorded (image, CPU, cores, memory);
  - **the freeze-origin store:** initialized from the freeze-origin states and loaded back, CP-17's buffers;
  - **A1's parity at the nine origins,** bit for bit against `reports/cp17b/host-parity.json`: 2025-10-26 and
    2026-03-29 from the checks run, 2026-08-31 to 2026-09-06 from the late shadow block's A1 issuances, whose payloads
    are the same (checked);
  - **the 14 shadow days:** every buffer against CP-17b's host shadow run at every origin, every fit's quantiles
    against its reference run, every intake, every artifact, the final state against CP-17b's, the store's chain, and
    the stitch (the early block's last state is the late block's start state);
  - **timing and memory** per fit against 15 minutes and 4 GiB, and the cost: one issuance, and decision 23's four
    issuances with the canary, from the sequential issuances;
  - **minutes** as a private repository counts them.

    PYTHONPATH=src python -m cp17c.compare fetch --repo hrsi56/delu-host-refreeze --run <id> --dest <dir> --record <json>
    PYTHONPATH=src python -m cp17c.compare compare --inputs <inputs.json> --artifacts <json> --downloads <dir> \\
        --local <dir> --out <measurement.json>
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
import statistics
import subprocess
import zipfile

from .run import dump, sha256_file

FIRST, SPLIT, LAST = date(2026, 8, 24), date(2026, 8, 31), date(2026, 9, 6)
NINE = ('A1-2025-10-26', 'A1-2026-03-29') + tuple(f'A1-{SPLIT + timedelta(days=i)}' for i in range(7))
POLICIES = ('A1', 'v4', 'v2', 'B0')
WALL, PEAK = 900, 4 * 1024 ** 3
MULTIPLIER = {'MACOS': 10, 'UBUNTU': 1}
#: The Owner's decision 26 (2026-10-30): the scratch repository was made public at this time. A run created from then
#: on ran on the public repository's runner class, whose standard runners GitHub does not count against the quota.
PUBLIC_FROM = '2026-10-30T14:21:00Z'
BLOCKS = {'late': (SPLIT, LAST), 'early': (FIRST, SPLIT - timedelta(days=1))}


# --- fetch --------------------------------------------------------------------------------------------------------

def _gh(*args: str) -> str:
    return subprocess.run(['gh', *args], capture_output=True, text=True, check=True).stdout


def _seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    parse = lambda t: datetime.fromisoformat(t.replace('Z', '+00:00'))   # noqa: E731
    return (parse(end) - parse(start)).total_seconds()


def fetch(repo: str, run_id: str, dest: Path, record: Path) -> dict:
    run = json.loads(_gh('api', f'repos/{repo}/actions/runs/{run_id}'))
    jobs = json.loads(_gh('api', f'repos/{repo}/actions/runs/{run_id}/jobs?per_page=100'))['jobs']
    timing = json.loads(_gh('api', f'repos/{repo}/actions/runs/{run_id}/timing'))
    listed = json.loads(_gh('api', f'repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100'))['artifacts']
    folder = dest / str(run_id)
    folder.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for item in listed:
        zpath = folder / f"{item['name']}.zip"
        zpath.write_bytes(subprocess.run(['gh', 'api', f"repos/{repo}/actions/artifacts/{item['id']}/zip"],
                                         capture_output=True, check=True).stdout)
        digest = sha256_file(zpath)
        if item.get('digest') and item['digest'] != f'sha256:{digest}':
            raise SystemExit(f"{item['name']}: the zip is not the artifact GitHub recorded")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(folder / item['name'])
        artifacts.append({'id': item['id'], 'name': item['name'], 'size_in_bytes': item['size_in_bytes'],
                          'digest': item.get('digest'), 'zip_sha256': digest, 'path': f"{run_id}/{item['name']}"})
    value = json.loads(record.read_text()) if record.exists() else {'schema': 'cp17c-artifacts-v1', 'repo': repo, 'runs': {}}
    value['runs'][str(run_id)] = {
        'run_attempt': run.get('run_attempt'), 'head_sha': run.get('head_sha'), 'status': run.get('status'),
        'conclusion': run.get('conclusion'), 'created_at': run.get('created_at'), 'updated_at': run.get('updated_at'),
        'jobs': [{'id': j['id'], 'name': j['name'], 'labels': j.get('labels'), 'runner_name': j.get('runner_name'),
                  'annotations': [a.get('message') for a in json.loads(_gh('api', f"repos/{repo}/check-runs/{j['id']}/annotations"))
                                  if a.get('annotation_level') in ('failure', 'warning')] if j['conclusion'] != 'success' else [],
                  'status': j['status'], 'conclusion': j['conclusion'], 'started_at': j.get('started_at'),
                  'completed_at': j.get('completed_at'), 'seconds': _seconds(j.get('started_at'), j.get('completed_at')),
                  'steps': [{'name': s['name'], 'conclusion': s['conclusion'],
                             'seconds': _seconds(s.get('started_at'), s.get('completed_at'))} for s in j.get('steps', [])]}
                 for j in jobs],
        'billable': timing.get('billable'), 'artifacts': artifacts}
    dump(record, value)
    return value


def counted_minutes(artifacts: dict) -> dict:
    """Each job rounded up to a whole minute, macOS at ten times; GitHub's billable figures kept beside."""
    out, total = {}, 0
    for run_id, run in sorted(artifacts['runs'].items()):
        own = {}
        for job in run['jobs']:
            if job['seconds'] is None:
                continue
            if not job.get('runner_name') or not job.get('steps'):
                # cancelled while queued: it never had a runner, so GitHub counts nothing for it
                own[job['name']] = {'os': 'MACOS', 'seconds': 0.0, 'queued_seconds': job['seconds'], 'setup_seconds': 0.0,
                                    'minutes': 0, 'counted': 0, 'never_ran': True,
                                    'counted_if_queued_time_were_billed': math.ceil(job['seconds'] / 60) * 10}
                continue
            setup = 0.0
            for step in job.get('steps', []):
                if step['name'].startswith('The run'):
                    break
                setup += step['seconds'] or 0.0
            os_name = 'MACOS' if any('macos' in label for label in job.get('labels') or []) else 'UBUNTU'
            own[job['name']] = {'os': os_name, 'seconds': job['seconds'], 'setup_seconds': setup,
                                'minutes': math.ceil(job['seconds'] / 60),
                                'counted': math.ceil(job['seconds'] / 60) * MULTIPLIER[os_name]}
        billable = sum(math.ceil(j['duration_ms'] / 60000) * MULTIPLIER[o]
                       for o, item in (run.get('billable') or {}).items() for j in item.get('job_runs', []))
        public = run['created_at'] >= PUBLIC_FROM
        if public:      # decision 26: a public repository's standard runners are not counted; the minutes are recorded
            for job in own.values():
                job['counted_if_private'], job['counted'] = job['counted'], 0
            billable = 0
        out[run_id] = {'jobs': own, 'repository': 'public (decision 26)' if public else 'private',
                       'counted_from_job_times': sum(v['counted'] for v in own.values()), 'counted_from_billable': billable}
        total += max(billable, out[run_id]['counted_from_job_times'])
    bound = total + sum(j.get('counted_if_queued_time_were_billed', 0) for r in out.values() for j in r['jobs'].values())
    public_minutes = sum(j['minutes'] for r in out.values() if r['repository'] != 'private' for j in r['jobs'].values())
    return {'runs': out, 'counted_total': total, 'upper_bound_if_queued_time_were_billed': bound,
            'public_job_minutes_uncounted': public_minutes,
            'rule': 'per job that ran, rounded up to the minute; macOS x10 (the larger of the two counts per run); '
                    'a job cancelled while queued never had a runner and counts nothing; a run on the public repository '
                    '(decision 26, from ' + PUBLIC_FROM + ') counts nothing, its job minutes recorded'}


def repository(repo: str) -> dict:
    """The scratch repository's state, read with `gh api` (read-only)."""
    meta = json.loads(_gh('api', f'repos/{repo}'))
    perms = json.loads(_gh('api', f'repos/{repo}/actions/permissions/workflow'))
    secrets = json.loads(_gh('api', f'repos/{repo}/actions/secrets'))
    variables = json.loads(_gh('api', f'repos/{repo}/actions/variables'))
    commits = json.loads(_gh('api', f'repos/{repo}/commits?per_page=100'))
    runs = json.loads(_gh('api', f'repos/{repo}/actions/runs?per_page=100'))['workflow_runs']
    branches = json.loads(_gh('api', f'repos/{repo}/branches'))
    return {'schema': 'cp17c-scratch-repository-v1', 'repository': repo, 'url': meta['html_url'], 'private': meta['private'],
            'visibility': meta['visibility'], 'default_branch': meta['default_branch'], 'created_at': meta['created_at'],
            'default_workflow_permissions': perms['default_workflow_permissions'],
            'actions_secrets': secrets['total_count'], 'actions_variables': variables['total_count'],
            'branches': [b['name'] for b in branches],
            'commits': [{'sha': c['sha'], 'message': c['commit']['message'].splitlines()[0]} for c in commits],
            'workflow_runs': [{'id': r['id'], 'head_sha': r['head_sha'], 'created_at': r['created_at'],
                               'conclusion': r['conclusion']} for r in runs],
            'note': 'read with gh api (read-only) by the Lead after the last run; the repository is left in place, and '
                    "deleting it is the Owner's decision"}


# --- compare ------------------------------------------------------------------------------------------------------

HIDDEN_EXCLUDED = 'forced/project/.local/'


def _hashes(folder: Path) -> dict:
    """The job's own results-sha256.txt, every file checked against it. `upload-artifact` leaves out hidden paths:
    the only ones are the forced binary check's scaffolding (`forced/project/.local/`, a copy of the tarball and an
    altered `bin/node`), which are inputs to a forced failure, not results; they are named, never silently passed."""
    listed = {}
    for line in (folder / 'results-sha256.txt').read_text().splitlines():
        digest, rel = line.split(None, 1)
        listed[rel.strip().removeprefix('./')] = digest
    hidden = [rel for rel in listed if any(part.startswith('.') for part in Path(rel).parts)]
    if any(HIDDEN_EXCLUDED not in rel for rel in hidden):
        raise SystemExit(f'{folder}: hidden results other than the forced scaffolding: {hidden[:5]}')
    listed = {rel: digest for rel, digest in listed.items() if rel not in hidden}
    bad = [rel for rel, digest in listed.items() if sha256_file(folder / rel) != digest]
    if bad:
        raise SystemExit(f'{folder}: files differ from results-sha256.txt: {bad[:5]}')
    return listed


def _load(folder: Path, rel: str, listed: dict | None):
    if listed is not None and rel not in listed:
        raise SystemExit(f'{folder / rel}: not among the job\'s hashed results')
    return json.loads((folder / rel).read_text())


def _fit(ref: dict, attempt_resources: dict | None, row: dict, executed_ok) -> dict:
    wall = (row.get('total_ms') or 0) / 1000
    peak = row.get('process_rss_peak_bytes') or 0
    return {'quantiles_sha256': row.get('quantiles_sha256'), 'reference_quantiles_sha256': ref['quantiles_sha256'],
            'bit_identical': row.get('quantiles_sha256') == ref['quantiles_sha256'] and not row.get('errors'),
            'payload_equal': row.get('result_payload_sha256') == ref['payload_sha256'],
            'executed_files_match_package': executed_ok(row.get('executed_files')),
            'wall_seconds': wall, 'peak_bytes': peak, 'within_ceilings': 0 < wall <= WALL and 0 < peak <= PEAK,
            'reference_wall_seconds': ref['total_ms'] / 1000, 'wall_ratio_to_reference': wall / (ref['total_ms'] / 1000),
            'node': row.get('node'), 'pyodide': row.get('pyodide'), 'blocked_non_local_requests': row.get('blocked_non_local_requests'),
            'resources': attempt_resources}


def _executed_check(files_sha256: dict):
    def ok(executed) -> bool:
        return bool(executed) and all(files_sha256.get(k) == v for k, v in executed.items())
    return ok


def checks_part(folder: Path, listed: dict | None) -> dict:
    run = _load(folder, 'checks-run.json', listed)
    gate = run['gate']
    machine = next(c for c in gate['checks'] if c['check'] == 'machine')
    canary = next(c for c in gate['checks'] if c['check'] == 'canary')
    pins = next(c for c in gate['checks'] if c['check'] == 'pins')
    return {'gate_passed': gate['passed'], 'pins_output': pins['output'], 'pins_seconds': pins['seconds'],
            'canary': {k: canary.get(k) for k in ('quantiles_sha256', 'expected_quantiles_sha256', 'seconds', 'payload_sha256',
                                                  'executed_files', 'node', 'pyodide')}
            | {'resources': canary['attempt'].get('resources')},
            'gate_seconds': sum(c['seconds'] for c in gate['checks']),
            'machine': {'kind': machine['kind'], 'image_version': machine['image_version'],
                        **{k: machine['described'].get(k) for k in ('system', 'machine', 'cpu_model', 'hardware_model',
                                                                    'cpu_count', 'physical_cores', 'memory_bytes', 'os_version')},
                        'image_os': machine['described']['runner_env'].get('ImageOS')},
            'forced': {name: {'failed_at': r['failed'], 'not_run': r['not_run'], 'issuance_called': r['issuance_called'],
                              'as_expected': r['as_expected'],
                              'problem': next(c['problems'] for c in r['checks'] if c['check'] == r['failed'])[:1]}
                       for name, r in run['forced'].items()},
            'all_forced_as_expected': all(r['as_expected'] for r in run['forced'].values()),
            'freeze_origin_store': run['freeze_origin_store']}


def _store_hashes(folder: Path, listed: dict | None) -> dict:
    """A store's files by SHA-256: from the job's hash list where there is one, else computed."""
    if listed is not None:
        return {rel.removeprefix('store/'): h for rel, h in listed.items() if rel.startswith('store/')}
    return {str(p.relative_to(folder / 'store')): sha256_file(p) for p in sorted((folder / 'store').rglob('*')) if p.is_file()}


def machine_part(checks_folder: Path, checks_listed: dict | None, blocks: dict, inputs: dict, executed_ok) -> dict:
    """One machine: its checks run and its two shadow blocks. `blocks[name]` is a list of (folder, hashes), one per run
    that carried the block: the first run's days, then any run that resumed it from the store the one before left."""
    from .store import Store
    refs = inputs['references']
    out = {'checks': checks_part(checks_folder, checks_listed), 'parity': {}, 'shadow': {}}
    parity = _load(checks_folder, 'parity.json', checks_listed)
    for row in parity['stems']:
        out['parity'][row['stem']] = _fit(refs['runs']['parity'][row['stem']], row['attempt'].get('resources'), row, executed_ok)
    gates = {'checks': out['checks']['gate_passed']}
    for block, (first, last) in BLOCKS.items():
        parts = blocks.get(block) or []
        if not parts:
            out['shadow'][block] = None
            continue
        merged, part_of, part_gates = {}, {}, []
        for i, (folder, listed) in enumerate(parts):
            record = _load(folder, 'shadow.json', listed)
            gate = _load(folder, 'checks.json', listed)
            machine = next((c for c in gate['checks'] if c['check'] == 'machine'), {})
            described = machine.get('described') or {}
            part_gates.append({'passed': gate['passed'], 'seconds': sum(c.get('seconds', 0) for c in gate['checks']),
                               'machine': {'kind': machine.get('kind'), 'image_version': machine.get('image_version'),
                                           'image_os': (described.get('runner_env') or {}).get('ImageOS'),
                                           **{k: described.get(k) for k in ('cpu_model', 'cpu_count', 'memory_bytes',
                                                                            'os_version', 'hardware_model')}}})
            for d, v in record['days'].items():
                merged[d], part_of[d] = v, i
        gates[block] = all(g['passed'] for g in part_gates)
        folder, listed = parts[-1]
        head = Store(folder / 'store').verify()
        carried = None
        if len(parts) > 1:
            before, after = _store_hashes(*parts[0]), _store_hashes(*parts[-1])
            grown = {'head.json', 'ledger.jsonl'}
            carried = all(after.get(rel) == h for rel, h in before.items() if rel not in grown)
            lines = (folder / 'store' / 'ledger.jsonl').read_text().splitlines()
            first_lines = (parts[0][0] / 'store' / 'ledger.jsonl').read_text().splitlines()
            carried = carried and lines[:len(first_lines)] == first_lines
        days, missing = {}, []
        for d in [first + timedelta(days=i) for i in range((last - first).days + 1)]:
            if str(d) not in head_days(folder):
                missing.append(str(d))                  # never run: recorded as missing, never passed
                continue
            day_rec = _load(folder, f'store/issued/{d}/record.json', listed)
            fits = {p: _fit(refs['runs']['shadow'][f'{p}-{d}'], day_rec['resources'].get(p) if p == 'A1' else None,
                            day_rec['runs'][p], executed_ok) for p in POLICIES}
            days[str(d)] = {
                'path': day_rec['path'], 'issued_policy': day_rec['issued_policy'], 'artifact_verifies': day_rec['artifact_verifies'],
                'buffers_equal_payload': all(day_rec['buffers_equal_payload'].values()),
                'buffers_equal_reference': day_rec['buffers_sha256'] == refs['shadow_buffers_sha256'][str(d)],
                'intake': day_rec['intake'], 'all_registered': all(v == 'registered' for v in day_rec['intake'].values()),
                'fits': fits, 'all_bit_identical': all(f['bit_identical'] for f in fits.values()),
                'day_seconds_sum': day_rec['process_seconds_sum'], 'issue_day_seconds': day_rec['issue_day_seconds'],
                'day_process_seconds': merged[str(d)]['process_seconds'], 'run_part': part_of[str(d)]}
            if d >= SPLIT:
                out['parity'][f'A1-{d}'] = fits['A1'] | {'from': f'the late block\'s issuance of {d}'}
        final_state = json.loads((folder / 'store' / head['state_file']).read_text())
        from cp17.shadow import ShadowStates
        out['shadow'][block] = {'days': days, 'missing_days': missing, 'gates': part_gates, 'gate_passed': gates[block],
                                'gate_seconds': part_gates[0]['seconds'], 'parts': len(parts),
                                'carried_store_equal_to_the_first_runs': carried, 'store_verifies': True,
                                'entries': head['entries'], 'final_digest': ShadowStates.from_dict(final_state).digest(),
                                'final_state': final_state}
    canary = out['checks']['canary']
    if f'A1-{LAST}' not in out['parity'] and canary['quantiles_sha256']:
        # the freeze origin's parity fit is the run's canary: A1 at 2026-09-06, the same payload, judged as an attempt
        ref = refs['runs']['parity'][f'A1-{LAST}']
        resources = canary['resources'] or {}
        wall, peak = resources.get('wall_seconds', 0), resources.get('peak_bytes', 0)
        out['parity'][f'A1-{LAST}'] = {
            'quantiles_sha256': canary['quantiles_sha256'], 'reference_quantiles_sha256': ref['quantiles_sha256'],
            'bit_identical': canary['quantiles_sha256'] == ref['quantiles_sha256'],
            'payload_equal': canary['payload_sha256'] == ref['payload_sha256'],
            'executed_files_match_package': executed_ok(canary.get('executed_files')),
            'wall_seconds': wall, 'peak_bytes': peak, 'within_ceilings': 0 < wall <= WALL and 0 < peak <= PEAK,
            'reference_wall_seconds': ref['total_ms'] / 1000, 'wall_ratio_to_reference': wall / (ref['total_ms'] / 1000),
            'node': canary.get('node'), 'pyodide': canary.get('pyodide'), 'blocked_non_local_requests': [],
            'resources': resources, 'from': "the checks run's canary (A1 at the freeze origin, the parity payload)"}
    late, early = out['shadow'].get('late'), out['shadow'].get('early')
    summary = {'gates_passed': gates}
    summary['shadow_missing_days'] = sorted(d for b in out['shadow'].values() if b for d in b['missing_days'])
    if late and not late['missing_days']:
        summary['late_final_equals_reference_final'] = late['final_digest'] == refs['shadow_final_sha256']
    if late and early and not early['missing_days']:
        from .state import _carried
        start = json.loads((Path(inputs['tree']) / 'states' / f'start-{SPLIT}.json').read_text()) if inputs.get('tree') else None
        summary['stitch_early_final_equals_late_start'] = (start is not None and _carried(early['final_state']) == _carried(start))
    for block in (late, early):
        if block:
            block.pop('final_state')
    fits = [f for b in out['shadow'].values() if b for d in b['days'].values() for f in d['fits'].values()] + \
           [f for s, f in out['parity'].items() if 'from' not in f]
    days = [d for b in out['shadow'].values() if b for d in b['days'].values()]
    summary.update({
        'parity_origins': sorted(out['parity']), 'parity_all_bit_identical': all(f['bit_identical'] for f in out['parity'].values())
        and len(out['parity']) == 9,
        'fits': len(fits), 'fits_bit_identical': sum(f['bit_identical'] for f in fits),
        'all_payloads_equal': all(f['payload_equal'] for f in fits),
        'all_executed_files_match_package': all(f['executed_files_match_package'] for f in fits),
        'all_within_ceilings': all(f['within_ceilings'] for f in fits),
        'wall_seconds_max': max(f['wall_seconds'] for f in fits), 'peak_bytes_max': max(f['peak_bytes'] for f in fits),
        'wall_ratio_to_reference_median': statistics.median(f['wall_ratio_to_reference'] for f in fits),
        'blocked_non_local_requests': sorted({u for f in fits for u in f['blocked_non_local_requests'] or []}),
        'shadow_days': len(days), 'shadow_all_buffers_equal_reference': all(d['buffers_equal_reference'] for d in days),
        'shadow_all_buffers_equal_payload': all(d['buffers_equal_payload'] for d in days),
        'shadow_all_registered': all(d['all_registered'] for d in days),
        'shadow_all_issued_a1': all(d['issued_policy'] == 'A1' for d in days),
        'shadow_all_artifacts_verify': all(d['artifact_verifies'] for d in days),
        'shadow_all_fits_bit_identical': all(d['all_bit_identical'] for d in days)})
    if days:
        summary['cost'] = cost(out)
    out['summary'] = summary
    return out


def head_days(folder: Path) -> set:
    """The days a store has issued, from its ledger (the store itself is verified separately)."""
    lines = (folder / 'store' / 'ledger.jsonl').read_text().splitlines()
    return {json.loads(line)['day'] for line in lines if line.strip() and json.loads(line).get('kind') == 'issuance'}


def _spread(values: list) -> dict | None:
    return {'n': len(values), 'min': min(values), 'median': statistics.median(values), 'max': max(values)} if values else None


def cost(part: dict) -> dict:
    """One issuance (A1 and the three shadows, one after another), and decision 23's four issuances with the canary
    (the run's gate, the four checks, plus four consecutive days of a block), from the sequential issuances. A block's
    first run ran beside the other block (two at once); a resumed day ran alone."""
    beside, alone, fours = [], [], []
    for b in part['shadow'].values():
        if not b:
            continue
        first = [d['day_process_seconds'] for d in b['days'].values() if d['run_part'] == 0]
        beside += first
        alone += [d['day_process_seconds'] for d in b['days'].values() if d['run_part'] > 0]
        fours += [b['gate_seconds'] + sum(first[i:i + 4]) for i in range(len(first) - 3)]
    return {'one_issuance_seconds_beside_the_other_block': _spread(beside), 'one_issuance_seconds_alone': _spread(alone),
            'four_issuances_with_canary_seconds_beside_the_other_block': _spread(fours),
            'basis': "each day's own process (load and check the store, release D-2, issue A1 and three shadows one after "
                     "another, append); the gate is the run's four checks, the canary included"}


def compare(inputs_path: Path, artifacts_path: Path | None, downloads: Path | None, local: Path | None, tree: Path | None) -> dict:
    root = Path('.')
    inputs = json.loads(inputs_path.read_text())
    inputs['tree'] = str(tree) if tree else None
    manifest = json.loads((root / 'docs/track-b/evidence/cp-17c/freeze-manifest.json').read_text())
    executed_ok = _executed_check(manifest['policy_identity']['code']['files_sha256'])
    out = {'schema': 'cp17c-measurement-v1', 'inputs_sha256': sha256_file(inputs_path),
           'policy_identity_sha256': manifest['policy_identity_sha256'], 'ceilings': {'wall_seconds': WALL, 'peak_bytes': PEAK},
           'reference': {'host_parity': 'reports/cp17b/host-parity.json', 'shadow_run': 'reports/cp17b/shadow-run.json',
                         'reports_sha256': inputs['references']['reports_sha256']}}
    if artifacts_path is not None:
        artifacts = json.loads(artifacts_path.read_text())
        out['artifacts_sha256'] = sha256_file(artifacts_path)
        out['minutes'] = counted_minutes(artifacts)
        chosen = _chosen(artifacts)
        listed = {name: _hashes(downloads / path) for name, (run_id, path) in chosen.items()}
        mapping = {name: downloads / path for name, (run_id, path) in chosen.items()}
        blocks = {}
        for block in ('late', 'early'):
            for name in ('shadows', 'resume'):
                if name in mapping and (mapping[name] / f'shadow-{block}').is_dir():
                    blocks.setdefault(block, []).append((mapping[name] / f'shadow-{block}', _under(listed[name], f'shadow-{block}/')))
        out['runner'] = machine_part(mapping['checks'] / 'checks', _under(listed['checks'], 'checks/'), blocks, inputs, executed_ok)
        out['runner']['artifacts'] = {name: {'run': run_id, 'path': path} for name, (run_id, path) in chosen.items()}
        out['runner']['host'] = {name: _host_files(mapping[name], listed[name]) for name in listed}
    if local is not None:
        out['local'] = machine_part(local / 'checks', None, {b: [(local / f'shadow-{b}', None)] for b in ('late', 'early')
                                                             if (local / f'shadow-{b}' / 'shadow.json').exists()},
                                    inputs, executed_ok)
    return out


def _chosen(artifacts: dict) -> dict:
    """The artifact each part of the comparison reads: `checks` from the last run whose checks job succeeded;
    `shadows` from the last run that uploaded it (its job may have reached its limit: the days it completed are read,
    each by its store's ledger, and a day it did not reach is recorded as missing); and `resume`, where a later run
    carried a block's store on. Earlier runs' artifacts stay recorded, unread."""
    out = {}
    for run_id, run in sorted(artifacts['runs'].items(), key=lambda kv: kv[1]['created_at']):
        ok = {j['name'].removeprefix('macos-26 ') for j in run['jobs'] if j['conclusion'] == 'success'}
        for item in run['artifacts']:
            if item['name'] in ok | {'shadows'} and item['name'] in ('checks', 'shadows', 'resume'):
                out[item['name']] = (run_id, item['path'])
    return out


def _host_files(folder: Path, listed: dict) -> dict:
    out = {}
    for rel in ('cpython.json', 'node-retrieval.json', 'package.json', 'inputs-verified.json', 'uv-retrieval.json'):
        if rel in listed:
            out[rel.removesuffix('.json')] = json.loads((folder / rel).read_text())
    return out


def _under(listed: dict, prefix: str) -> dict:
    return {rel.removeprefix(prefix): h for rel, h in listed.items() if rel.startswith(prefix)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch')
    f.add_argument('--repo', required=True)
    f.add_argument('--run', required=True)
    f.add_argument('--dest', required=True, type=Path)
    f.add_argument('--record', required=True, type=Path)
    g = sub.add_parser('repository')
    g.add_argument('--repo', required=True)
    g.add_argument('--out', required=True, type=Path)
    c = sub.add_parser('compare')
    c.add_argument('--inputs', required=True, type=Path)
    c.add_argument('--artifacts', type=Path)
    c.add_argument('--downloads', type=Path)
    c.add_argument('--local', type=Path)
    c.add_argument('--tree', type=Path, help='the scratch tree (for the late block\'s start state)')
    c.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'fetch':
        value = fetch(args.repo, args.run, args.dest, args.record)
        print(len(value['runs'][args.run]['artifacts']), 'artifacts', value['runs'][args.run]['conclusion'])
        return 0
    if args.cmd == 'repository':
        value = repository(args.repo)
        dump(args.out, value)
        print(value['repository'], value['visibility'], value['actions_secrets'], value['actions_variables'])
        return 0
    value = compare(args.inputs, args.artifacts, args.downloads, args.local, args.tree)
    dump(args.out, value)
    for machine in ('runner', 'local'):
        if machine in value:
            s = value[machine]['summary']
            print(machine, {k: s[k] for k in ('parity_all_bit_identical', 'fits', 'fits_bit_identical', 'all_within_ceilings',
                                              'shadow_days', 'shadow_all_buffers_equal_reference', 'gates_passed')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
