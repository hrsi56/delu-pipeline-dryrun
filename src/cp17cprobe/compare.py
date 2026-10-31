"""The runners' records against the frozen host's, offline and verified by hash (run locally; item 5's recomputation).

- **fetch** (needs `gh`): one run's artifacts, each downloaded as its zip and checked against the digest GitHub
  recorded for it, then unpacked; with each job's duration and the run's billable time, as GitHub reports them.
- **compare** (needs nothing online): every downloaded file is checked against the job's own `results-sha256.txt`,
  every reference record against the hash `cp17cprobe.inputs` bound to a committed CP-17b report. Then, per runner,
  day and policy, by the frozen comparison `cp17.retrain.compare`:
  - whether the issued quantiles are bit-identical to the reference's (their SHA-256);
  - if not, the largest absolute difference per emitted quantile in EUR/MWh, and whether every data-driven
    selection matched;
  - the buffers' SHA-256 against the reference's, at every shadow origin, and the stitch of the two shadow jobs;
  - each attempt's wall time and peak memory, as the success definition measures them (the process's own, judged
    by `cp17.issue.attempt` on the runner), against 15 minutes and 4 GiB;
  - that each payload's identity equals the reference's.

The output has no clock in it: the same downloads give the same bytes.

    PYTHONPATH=src python -m cp17cprobe.compare fetch --repo hrsi56/delu-host-probe --run <id> --dest <dir> --record <artifacts.json>
    PYTHONPATH=src python -m cp17cprobe.compare compare --inputs <inputs.json> --artifacts <artifacts.json> \\
        --downloads <dir> --local <local control dir> --retrieval <retrieval-local.json> --out <measurement.json>
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import statistics
import subprocess
import zipfile

from .run import dump, sha256_file

ROOT = Path(__file__).resolve().parents[2]
WALL_LIMIT_SECONDS, MEMORY_LIMIT_BYTES = 15 * 60, 4 * 1024 ** 3
#: GitHub's included-minute multipliers for a private repository's standard hosted runners.
MULTIPLIER = {'UBUNTU': 1, 'MACOS': 10, 'WINDOWS': 2}
POLICIES = ('A1', 'v4', 'v2', 'B0')


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
        with open(zpath, 'wb') as handle:
            handle.write(subprocess.run(['gh', 'api', f"repos/{repo}/actions/artifacts/{item['id']}/zip"],
                                        capture_output=True, check=True).stdout)
        digest = sha256_file(zpath)
        if item.get('digest') and item['digest'] != f'sha256:{digest}':
            raise SystemExit(f"{item['name']}: the zip is not the artifact GitHub recorded")
        target = folder / item['name']
        with zipfile.ZipFile(zpath) as z:
            z.extractall(target)
        artifacts.append({'id': item['id'], 'name': item['name'], 'size_in_bytes': item['size_in_bytes'],
                          'digest': item.get('digest'), 'zip_sha256': digest, 'path': str(target.relative_to(dest))})
    value = json.loads(record.read_text()) if record.exists() else {'schema': 'cp17c-probe-artifacts-v1', 'repo': repo,
                                                                     'runs': {}}
    value['runs'][str(run_id)] = {
        'run_attempt': run.get('run_attempt'), 'head_sha': run.get('head_sha'), 'status': run.get('status'),
        'conclusion': run.get('conclusion'), 'created_at': run.get('created_at'), 'updated_at': run.get('updated_at'),
        'jobs': [{'id': j['id'], 'name': j['name'], 'labels': j.get('labels'), 'runner_name': j.get('runner_name'),
                  'status': j['status'], 'conclusion': j['conclusion'], 'started_at': j.get('started_at'),
                  'completed_at': j.get('completed_at'),
                  'seconds': _seconds(j.get('started_at'), j.get('completed_at')),
                  'steps': [{'name': s['name'], 'conclusion': s['conclusion'],
                             'seconds': _seconds(s.get('started_at'), s.get('completed_at'))} for s in j.get('steps', [])]}
                 for j in jobs],
        'billable': timing.get('billable'), 'artifacts': artifacts}
    dump(record, value)
    return value


def counted_minutes(artifacts: dict) -> dict:
    """Minutes as GitHub's usage report counts them for a private repository: each job rounded up to a whole
    minute, macOS at its multiplier. GitHub's own billable figures per run are kept beside the computation."""
    out, total = {}, 0
    for run_id, run in sorted(artifacts['runs'].items()):
        rows = []
        for os_name, item in (run.get('billable') or {}).items():
            for job in item.get('job_runs', []):
                minutes = math.ceil(job['duration_ms'] / 60000)
                rows.append({'job_id': job['job_id'], 'os': os_name, 'duration_ms': job['duration_ms'], 'minutes': minutes,
                             'counted': minutes * MULTIPLIER[os_name]})
        own = {}
        for job in run['jobs']:
            if job['seconds'] is not None:
                os_name = 'MACOS' if any('macos' in label for label in job.get('labels') or []) else 'UBUNTU'
                setup = 0.0
                for step in job.get('steps', []):
                    if step['name'].startswith('The frozen chain'):
                        break
                    setup += step['seconds'] or 0.0
                own[job['name']] = {'os': os_name, 'seconds': job['seconds'], 'setup_seconds': setup,
                                    'counted': math.ceil(job['seconds'] / 60) * MULTIPLIER[os_name]}
        counted = sum(r['counted'] for r in rows)
        out[run_id] = {'billable_jobs': rows, 'counted_from_billable': counted,
                       'counted_from_job_times': sum(v['counted'] for v in own.values()), 'jobs': own}
        total += max(counted, out[run_id]['counted_from_job_times'])
    return {'runs': out, 'counted_total': total,
            'rule': 'per job, rounded up to the minute; Linux x1, macOS x10 (the larger of the two counts per run)'}


# --- compare ------------------------------------------------------------------------------------------------------

def _hashes(folder: Path) -> dict:
    """The job's own `results-sha256.txt`, checked file by file; {relative path: sha256}."""
    listed = {}
    for line in (folder / 'results-sha256.txt').read_text().splitlines():
        digest, rel = line.split(None, 1)
        listed[rel.strip().removeprefix('./')] = digest
    bad = [rel for rel, digest in listed.items() if sha256_file(folder / rel) != digest]
    if bad:
        raise SystemExit(f'{folder}: files differ from the job\'s own hashes: {bad[:5]}')
    return listed


def _load(path: Path, expected: str):
    if sha256_file(path) != expected:
        raise SystemExit(f'{path}: not the record its row cites')
    return json.loads(path.read_text())


def project() -> Path:
    """The primary checkout holding `.local/` (the reference runs), as the frozen tools find it."""
    from cp17.tools import ledger
    return ledger.project_root(ROOT)


def _reference(inputs: dict, kind: str, stem: str) -> tuple[dict, dict]:
    row = inputs['references']['runs'][kind][stem]
    return row, _load(project() / row['record'], row['record_sha256'])


def _fit_row(policy: str, ref_row: dict, ref_run: dict, row: dict, run: dict | None, executed_ok) -> dict:
    """One fit against its reference: identity, the frozen comparison, the success definition's measurement."""
    from cp17 import retrain
    out = {'policy': policy, 'reference_quantiles_sha256': ref_row['quantiles_sha256'],
           'quantiles_sha256': row.get('quantiles_sha256'),
           'bit_identical': row.get('quantiles_sha256') is not None and row.get('quantiles_sha256') == ref_row['quantiles_sha256'],
           'payload_identity_equal': row.get('result_payload_sha256') == ref_row['payload_sha256'],
           'wall_seconds': (row.get('total_ms') or 0) / 1000 or None, 'peak_bytes': row.get('process_rss_peak_bytes'),
           'reference_wall_seconds': ref_row['total_ms'] / 1000, 'reference_peak_bytes': ref_row['process_rss_peak_bytes'],
           'errors': row.get('errors'), 'blocked_non_local_requests': row.get('blocked_non_local_requests'),
           'executed_files_match_package': executed_ok(row.get('executed_files'))}
    if out['wall_seconds'] is not None and out['peak_bytes'] is not None:
        out['within_ceilings'] = out['wall_seconds'] <= WALL_LIMIT_SECONDS and out['peak_bytes'] <= MEMORY_LIMIT_BYTES
    if run is not None and not row.get('errors'):
        c = retrain.compare(policy, ref_run['python']['result'], run['python']['result'])
        out.update(status=c['status'], selections_match=c['selections_match'],
                   bitwise_equal=c['deviation'].get('bitwise_equal'), max_abs_eur=c['deviation'].get('max_abs_eur'),
                   max_abs_eur_by_level=c['deviation'].get('max_abs_eur_by_level'),
                   compare_payload_sha256_equal=c['payload_sha256_equal'])
    return out


def _executed_check(freeze_files: dict):
    def ok(executed):
        return bool(executed) and all(freeze_files.get(k) == v for k, v in executed.items())
    return ok


def parity_rows(inputs: dict, kind: str, folder: Path, listed: dict | None, executed_ok) -> dict:
    record_path = folder / 'parity.json'
    if not record_path.exists():
        return {}
    record = json.loads(record_path.read_text())
    rows = {}
    for row in record['stems']:
        ref_row, ref_run = _reference(inputs, kind, row['stem'])
        run = None
        if row.get('record'):
            path = folder / row['record']
            if listed is not None and listed.get(str(path.relative_to(folder.parent))) != row['record_sha256']:
                raise SystemExit(f'{path}: not in the job\'s own hashes')
            run = _load(path, row['record_sha256'])
        item = _fit_row(row['policy'], ref_row, ref_run, row, run, executed_ok)
        item.update(day=row['day'], payload_meta_equal=row['payload_sha256'] == ref_row['payload_sha256'],
                    attempt_status=row['attempt']['status'], attempt_cause=row['attempt'].get('cause'),
                    attempt_resources=row['attempt'].get('resources'))
        rows[row['stem']] = item
    return rows


def shadow_rows(inputs: dict, folder: Path, listed: dict | None, executed_ok) -> dict:
    record_path = folder / 'shadow.json'
    if not record_path.exists():
        return {}
    record = json.loads(record_path.read_text())
    days = {}
    for day, d in sorted(record['days'].items()):
        fits = {}
        for policy in POLICIES:
            row = d['runs'].get(policy)
            if row is None:
                continue
            ref_row, ref_run = _reference(inputs, 'shadow', f'{policy}-{day}')
            run = None
            if row.get('record'):
                path = folder / row['record']
                if listed is not None and listed.get(str(path.relative_to(folder.parent))) != row['record_sha256']:
                    raise SystemExit(f'{path}: not in the job\'s own hashes')
                run = _load(path, row['record_sha256'])
            fits[policy] = _fit_row(policy, ref_row, ref_run, row, run, executed_ok)
        reference_buffers = inputs['references']['shadow_buffers_sha256'][day]
        days[day] = {'path': d['path'], 'issued_policy': d['issued_policy'], 'display': d['display'],
                     'artifact_verifies': d['artifact_verifies'], 'intake': d['intake'],
                     'buffers_equal_reference': {p: d['buffers_sha256'][p] == reference_buffers[p] for p in reference_buffers},
                     'buffers_equal_payload': d['buffers_equal_payload'],
                     'attempt_resources': d['resources'], 'day_seconds_sum': d['total_seconds'],
                     'day_within_ceiling': d['within_ceiling'], 'fits': fits}
    final = record.get('final')
    return {'first': record['first'], 'last': record['last'], 'concurrency': record['concurrency'],
            'start_state_sha256': record['start']['state_file_sha256'], 'days': days,
            'final_digest': final and final['digest'], 'final_state_sha256': final and final['state_file_sha256'],
            'completed': final is not None}


def _summary(fits: list[dict]) -> dict:
    walls = [f['wall_seconds'] for f in fits if f.get('wall_seconds')]
    peaks = [f['peak_bytes'] for f in fits if f.get('peak_bytes')]
    ratios = [f['wall_seconds'] / f['reference_wall_seconds'] for f in fits if f.get('wall_seconds')]
    return {'fits': len(fits), 'bit_identical': sum(bool(f['bit_identical']) for f in fits),
            'all_bit_identical': bool(fits) and all(f['bit_identical'] for f in fits),
            'all_selections_match': all(f.get('selections_match', False) for f in fits),
            'all_payload_identities_equal': all(f['payload_identity_equal'] for f in fits),
            'all_within_ceilings': bool(fits) and all(f.get('within_ceilings', False) for f in fits),
            'all_executed_files_match_package': all(f['executed_files_match_package'] for f in fits),
            'blocked_non_local_requests': sorted({u for f in fits for u in f.get('blocked_non_local_requests') or []}),
            'max_abs_eur': max((f.get('max_abs_eur') or 0.0 for f in fits), default=None),
            'wall_seconds_max': max(walls, default=None), 'wall_seconds_median': statistics.median(walls) if walls else None,
            'peak_bytes_max': max(peaks, default=None),
            'wall_ratio_to_reference_median': statistics.median(ratios) if ratios else None,
            'wall_ratio_to_reference_max': max(ratios, default=None)}


def runner(inputs: dict, folder: Path, executed_ok, local: bool = False) -> dict:
    """One runner's records: {host, retrieval, parity, f1, shadow}; `folder` holds its jobs' artifact folders."""
    out = {'host': None, 'jobs': {}, 'parity': {}, 'f1': {}, 'shadow': {}}
    for job in [folder] if local else sorted(p for p in folder.iterdir() if p.is_dir()):
        listed = None if local else _hashes(job)
        kind = job.name.rsplit('-', 1)[-1] if not local else job.name
        entry = {}
        for name in ('host.json', 'node-retrieval.json', 'uv-retrieval.json', 'inputs-verified.json'):
            if (job / name).exists():
                entry[name.removesuffix('.json')] = json.loads((job / name).read_text())
        if (job / 'uv-freeze.txt').exists():
            entry['uv_freeze_sha256'] = sha256_file(job / 'uv-freeze.txt')
        if (job / 'project-sha256.txt').exists():
            entry['project_sha256'] = (job / 'project-sha256.txt').read_text().split()
        if listed is not None:
            entry['results_sha256_file'] = sha256_file(job / 'results-sha256.txt')
        out['jobs'][job.name] = entry
        out['host'] = out['host'] or entry.get('host')
        if (job / 'parity').exists():
            out['parity'].update(parity_rows(inputs, 'parity', job / 'parity', listed, executed_ok))
        if (job / 'f1').exists():
            out['f1'].update(parity_rows(inputs, 'f1', job / 'f1', listed, executed_ok))
        if (job / 'shadow').exists():
            out['shadow'][kind if kind in ('late', 'early') else job.name] = shadow_rows(inputs, job / 'shadow', listed, executed_ok)
    fits = list(out['parity'].values()) + list(out['f1'].values()) + [
        f for s in out['shadow'].values() for d in s['days'].values() for f in d['fits'].values()]
    days = {day: d for s in out['shadow'].values() for day, d in s['days'].items()}
    out['summary'] = {
        'parity': _summary(list(out['parity'].values())), 'f1': _summary(list(out['f1'].values())),
        'shadow_fits': _summary([f for d in days.values() for f in d['fits'].values()]),
        'all': _summary(fits),
        'shadow_days': len(days)}
    if days:     # a runner with no shadow day has these unmeasured, not false
        out['summary'].update({
            'shadow_buffers_all_equal_reference': all(all(d['buffers_equal_reference'].values()) for d in days.values()),
            'shadow_buffers_all_equal_payload': all(all(d['buffers_equal_payload'].values()) for d in days.values()),
            'shadow_all_issued_a1': all(d['issued_policy'] == 'A1' and d['path'] == [['A1', 'ok']] for d in days.values()),
            'shadow_all_registered': all(set(d['intake'].values()) == {'registered'} for d in days.values()),
            'shadow_all_artifacts_verify': all(d['artifact_verifies'] for d in days.values()),
            'shadow_day_seconds_sum_max': max(d['day_seconds_sum'] for d in days.values()),
            'shadow_day_seconds_sum_median': statistics.median([d['day_seconds_sum'] for d in days.values()])})
    late, early = out['shadow'].get('late'), out['shadow'].get('early')
    if late and early:
        out['summary']['stitch_early_final_equals_late_start'] = early['final_state_sha256'] == late['start_state_sha256']
    if late:
        out['summary']['late_final_equals_reference_final'] = late['final_digest'] == inputs['references']['shadow_final_sha256']
    return out


def supplementary(inputs: dict, artifacts: dict, downloads: Path) -> list[dict]:
    """The fit records of superseded jobs, each checked against its job's own hashes and compared with the reference.

    Run 1's harness failed after each fit, but every record it left is the frozen Node.js runner's own, written
    before the failure. They are not part of the measurement; they add machines (CPU models) to the evidence of
    bit-for-bit reproduction."""
    rows = []
    for run_id, run in sorted(artifacts['runs'].items()):
        for item in run['artifacts']:
            if not item.get('superseded'):
                continue
            folder = downloads / item['path']
            listed = _hashes(folder)
            cpu = json.loads((folder / 'host.json').read_text())['machine']['cpu_model'] if (folder / 'host.json').exists() else None
            for rel in sorted(r for r in listed if r.endswith('/run.json')):
                kind = rel.split('/', 1)[0]
                record = json.loads((folder / rel).read_text())
                ref_row = inputs['references']['runs'][kind][record['target']]
                result = (record.get('python') or {}).get('result') or {}
                rows.append({'run': run_id, 'artifact': item['name'], 'cpu_model': cpu, 'kind': kind, 'stem': record['target'],
                             'record_sha256': listed[rel], 'errors': record.get('errors'),
                             'quantiles_sha256': result.get('quantiles_sha256'),
                             'bit_identical': result.get('quantiles_sha256') == ref_row['quantiles_sha256'],
                             'payload_identity_equal': result.get('payload_sha256') == ref_row['payload_sha256'],
                             'wall_seconds': record['total_ms'] / 1000, 'peak_bytes': record.get('process_rss_peak_bytes')})
    return rows


def local_retrieval(path: Path) -> dict:
    """This Mac's retrieval record, its files rehashed where they are present (local material)."""
    value = json.loads(path.read_text())
    for key in ('shasums', 'tarball', 'binary'):
        target = project() / value[key]['path']
        if target.exists() and sha256_file(target) != value[key]['sha256']:
            raise SystemExit(f'{target}: not the file the retrieval record names')
    return value


def compare(inputs_path: Path, artifacts_path: Path | None, downloads: Path, local: Path | None,
            retrieval: Path | None = None) -> dict:
    inputs = json.loads(inputs_path.read_text())
    freeze = json.loads((ROOT / 'docs/track-b/evidence/cp-17b/freeze-manifest.json').read_text())
    executed_ok = _executed_check(freeze['policy_identity']['code']['files_sha256'])
    out = {'schema': 'cp17c-probe-measurement-v1', 'inputs_sha256': sha256_file(inputs_path),
           'reference': {'reports_sha256': inputs['references']['reports_sha256'],
                         'sources_sha256': inputs['references']['sources_sha256'],
                         'node': {'path': '/opt/homebrew/Cellar/node/26.7.0/bin/node',
                                  'sha256': freeze['policy_identity']['issuing_host']['node']['sha256']}},
           'ceilings': {'wall_seconds': WALL_LIMIT_SECONDS, 'peak_bytes': MEMORY_LIMIT_BYTES}, 'runners': {}}
    if artifacts_path is not None:
        artifacts = json.loads(artifacts_path.read_text())
        out['artifacts_sha256'] = sha256_file(artifacts_path)
        out['minutes'] = counted_minutes(artifacts)
        by_runner: dict[str, dict] = {}
        for run_id, run in sorted(artifacts['runs'].items()):
            for item in run['artifacts']:
                if sha256_file(downloads / f"{run_id}/{item['name']}.zip") != item['zip_sha256']:
                    raise SystemExit(f"{item['name']}: the downloaded zip is not the recorded one")
                label = item['name'].split('-parity')[0].split('-shadow')[0]
                if item.get('superseded'):
                    continue
                by_runner.setdefault(label, {})[item['name']] = downloads / item['path']
        for label, jobs in sorted(by_runner.items()):
            staging = downloads / '_by_runner' / label
            staging.mkdir(parents=True, exist_ok=True)
            for name, path in jobs.items():
                link = staging / name
                if link.is_symlink() or link.exists():
                    link.unlink()
                link.symlink_to(path.resolve())
            out['runners'][label] = runner(inputs, staging, executed_ok)
        out['supplementary'] = supplementary(inputs, artifacts, downloads)
    if local is not None:
        out['local_control'] = runner(inputs, local, executed_ok, local=True)
    if retrieval is not None:
        out['local_retrieval'] = local_retrieval(retrieval)
    pinned = freeze['policy_identity']['issuing_host']['pyodide']['files_sha256']
    for r in [*out['runners'].values(), out.get('local_control')]:
        for job in (r or {}).get('jobs', {}).values():
            if job.get('host'):     # recomputed here, not taken from the runner's own claim
                job['host']['pyodide']['files_equal_freeze_manifest'] = job['host']['pyodide']['files_sha256'] == pinned
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch')
    f.add_argument('--repo', required=True)
    f.add_argument('--run', required=True)
    f.add_argument('--dest', required=True, type=Path)
    f.add_argument('--record', required=True, type=Path)
    c = sub.add_parser('compare')
    c.add_argument('--inputs', required=True, type=Path)
    c.add_argument('--artifacts', type=Path)
    c.add_argument('--downloads', type=Path)
    c.add_argument('--local', type=Path)
    c.add_argument('--retrieval', type=Path, help="this Mac's retrieval record (retrieval-local.json)")
    c.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'fetch':
        value = fetch(args.repo, args.run, args.dest, args.record)
        print(args.run, [a['name'] for a in value['runs'][args.run]['artifacts']])
        return 0
    value = compare(args.inputs, args.artifacts, args.downloads, args.local, args.retrieval)
    dump(args.out, value)
    for label, r in {**value['runners'], **({'local-control': value['local_control']} if 'local_control' in value else {})}.items():
        s = r['summary']
        print(label, 'fits', s['all']['fits'], 'bit-identical', s['all']['bit_identical'], 'ceilings', s['all']['all_within_ceilings'],
              'shadow days', s['shadow_days'], 'buffers', s.get('shadow_buffers_all_equal_reference'),
              'max wall', s['all']['wall_seconds_max'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
