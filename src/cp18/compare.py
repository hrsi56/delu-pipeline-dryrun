"""The runner's results, fetched and checked by hash, and every comparison recomputed from them (run locally).

    PYTHONPATH=src python -m cp18.compare fetch --repo hrsi56/delu-pipeline-dryrun --run <id> --dest <dir> --record <file>
    PYTHONPATH=src python -m cp18.compare timing --downloads <dir> --record <fetch records...> --out reports/cp18a/runner-timing.json

`fetch` downloads a run's `pipeline` artifact and its job metadata (`gh`, read only), checks every file against the
run's own `results-sha256.txt`, and records the listing's SHA-256, the job's start and end, and its minutes.

`timing` reads only files that hashed as listed and recomputes, for each run:
- **each day's outcome** from the run's own store (`live/store/issued/<day>/artifact.json`): the issued policy and its
  quantiles' SHA-256, compared with CP-17c's record of the same day (`reports/cp17c/measurement.json`, which CP-17c's
  Integration Critic checked) wherever it holds one;
- **the public runner class's timing (§24.13)** against criterion 5's ceilings (900 s and 4 GiB per fit, one worker):
  the canary's fit, each issuance's process seconds (A1 and the three shadows, one after another), each fit's wall
  time and peak memory, the run's pipeline seconds and the job's wall time; and decision 23's bound for a
  four-issuance run (four issuances and the canary inside GitHub's six-hour job limit).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from cp17 import runtime

CEILINGS = {'wall_seconds': 900, 'peak_bytes': 4 * 1024 ** 3}
JOB_LIMIT_SECONDS = 6 * 3600


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=str) + '\n')


def listing(folder: Path) -> dict[str, str]:
    out = {}
    for line in (folder / 'results-sha256.txt').read_text().splitlines():
        digest, rel = line.split(None, 1)
        out[rel.strip().removeprefix('./')] = digest
    return out


def verified(folder: Path) -> dict:
    listed = listing(folder)
    problems = [rel for rel, h in sorted(listed.items()) if not (folder / rel).is_file() or runtime.sha256_file(folder / rel) != h]
    return {'files': len(listed), 'problems': problems, 'listing_sha256': runtime.sha256_file(folder / 'results-sha256.txt')}


def fetch(repo: str, run: str, dest: Path) -> dict:
    target = dest / run
    if not target.exists():
        subprocess.run(['gh', 'run', 'download', run, '-R', repo, '-n', 'pipeline', '-D', str(target)], check=True)
    meta = json.loads(subprocess.run(['gh', 'run', 'view', run, '-R', repo, '--json',
                                      'conclusion,createdAt,updatedAt,headSha,jobs,event,displayTitle'],
                                     capture_output=True, text=True, check=True).stdout)
    job = meta['jobs'][0]
    from datetime import datetime
    started = datetime.fromisoformat(job['startedAt'].replace('Z', '+00:00'))
    ended = datetime.fromisoformat(job['completedAt'].replace('Z', '+00:00'))
    return {'repo': repo, 'run': run, 'head_sha': meta['headSha'], 'conclusion': meta['conclusion'],
            'job': {'name': job['name'], 'conclusion': job['conclusion'], 'started': job['startedAt'],
                    'completed': job['completedAt'], 'seconds': (ended - started).total_seconds(),
                    'steps': [{'name': s['name'], 'conclusion': s['conclusion']} for s in job['steps']]},
            'class': 'public repository, standard macos-26 runner (minutes not counted)',
            'results': verified(target)}


def _reference(root: Path) -> dict:
    """CP-17c's fits by day and policy (A1 and the three shadows), from its measurement record, where the runner and
    the M3 agreed on all 14 shadow days: {day: {policy: quantiles SHA-256}}."""
    value = json.loads((root / 'reports/cp17c/measurement.json').read_text())
    out: dict[str, dict] = {}
    for block in (value['local'].get('shadow') or {}).values():
        for day, row in (block.get('days') or {}).items():
            out[day] = {p: f.get('quantiles_sha256') for p, f in (row.get('fits') or {}).items() if f.get('quantiles_sha256')}
    return out


def run_timing(folder: Path, reference: dict) -> dict:
    run = json.loads((folder / 'run/run.json').read_text())
    gate = run.get('gate') or {}
    canary = next((c for c in gate.get('checks', []) if c['check'] == 'canary'), {})
    days = {}
    for day, item in run['days'].items():
        art_path = folder / 'live/store/issued' / day / 'artifact.json'
        rec_path = folder / 'live/store/issued' / day / 'record.json'
        row = {'status': item.get('status'), 'cause': item.get('cause_text')}
        if art_path.is_file():
            art, rec = json.loads(art_path.read_text()), json.loads(rec_path.read_text())
            fits = {}
            for policy, shadow in rec['shadows'].items():
                fits[policy] = {'wall_seconds': shadow.get('wall_seconds'), 'peak_bytes': shadow.get('peak_bytes')}
            fits['A1'] = art.get('resources') if art['issued_policy'] == 'A1' else fits.get('A1')
            mine = {'A1' if art['issued_policy'] == 'A1' else art['issued_policy']: art['quantiles_sha256'],
                    **{p: sh.get('quantiles_sha256') for p, sh in rec['shadows'].items() if sh.get('quantiles_sha256')}}
            theirs = reference.get(day) or {}
            row.update(issued_policy=art['issued_policy'], quantiles_sha256=art['quantiles_sha256'], fits_sha256=mine,
                       equal_to_cp17c=({p: mine.get(p) == q for p, q in theirs.items()} if theirs else None),
                       process_seconds=rec['process_seconds'], process_seconds_sum=rec['process_seconds_sum'],
                       issue_day_seconds=rec['issue_day_seconds'], fits=fits,
                       within_ceilings=all(f and f.get('wall_seconds') is not None and f['wall_seconds'] <= CEILINGS['wall_seconds']
                                           and (f.get('peak_bytes') or 0) <= CEILINGS['peak_bytes'] for f in fits.values()))
        days[day] = row
    issued = [d for d in days.values() if d.get('issued_policy')]
    canary_attempt = canary.get('attempt') or {}
    return {'run_id': run['run_id'], 'wall_clock': run['wall_clock'], 'planned': run['planned'],
            'outcome': run['outcome'], 'gate_passed': gate.get('passed'),
            'machine': next((c.get('described') for c in gate.get('checks', []) if c['check'] == 'machine'), None),
            'image_version': next((c.get('image_version') for c in gate.get('checks', []) if c['check'] == 'machine'), None),
            'canary': {'quantiles_sha256': canary.get('quantiles_sha256'), 'passed': canary.get('passed'),
                       'seconds': canary.get('seconds'), 'resources': canary_attempt.get('resources')},
            'days': days, 'issuances': len(issued),
            'pipeline_seconds': run.get('seconds'),
            'issuance_seconds': [d['process_seconds_sum'] for d in issued],
            'all_fits_within_ceilings': all(d['within_ceilings'] for d in issued) if issued else None,
            'all_equal_to_cp17c_where_recorded': all(all(d['equal_to_cp17c'].values()) for d in issued
                                                     if d['equal_to_cp17c'] is not None)}


def carry(carried: Path, resumed: dict[str, Path]) -> dict:
    """The carry rehearsal: a store carried across a gap, resumed on two machines. The carried ledger is a byte prefix
    of each resumed ledger and every carried file is unchanged; the resumed days' fits and the state after them are
    the same on both machines; the days of the gap are recorded as outside, never issued."""
    from .store import LiveStore
    carried = Path(carried)
    base = (carried / 'ledger.jsonl').read_bytes()
    files = {p.relative_to(carried).as_posix(): runtime.sha256_file(p) for p in carried.rglob('*') if p.is_file()
             and p.name not in ('head.json', 'ledger.jsonl')}
    out = {'carried': {'head_sha256': runtime.sha256_file(carried / 'head.json'),
                       'ledger_sha256': runtime.sha256_file(carried / 'ledger.jsonl'),
                       'entries': len(base.splitlines())}, 'resumed': {}}
    fits = {}
    for name, folder in resumed.items():
        store = LiveStore(folder)
        head = store.verify()
        ledger = (Path(folder) / 'ledger.jsonl').read_bytes()
        unchanged = all((Path(folder) / rel).is_file() and runtime.sha256_file(Path(folder) / rel) == h for rel, h in files.items())
        days = store.days()
        new = [e for e in store.entries()[out['carried']['entries']:] if e['kind'] == 'issuance']
        fits[name] = {}
        for e in new:
            rec = json.loads((Path(folder) / e['record_file']).read_text())
            art = json.loads((Path(folder) / e['artifact_file']).read_text())
            fits[name][e['day']] = {'issued': (art['issued_policy'], art['quantiles_sha256']),
                                    **{p: sh.get('quantiles_sha256') for p, sh in rec['shadows'].items()},
                                    'state_sha256': e['state_sha256']}
        out['resumed'][name] = {'ledger_prefix': ledger.startswith(base), 'carried_files_unchanged': unchanged,
                                'head': head['entry_sha256'], 'days': {d: v['status'] for d, v in days.items()},
                                'resumed_days': list(fits[name])}
    names = list(fits)
    out['same_on_both'] = len(names) == 2 and fits[names[0]] == fits[names[1]] and bool(fits[names[0]])
    out['fits'] = fits
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m cp18.compare')
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fetch')
    f.add_argument('--repo', required=True)
    f.add_argument('--run', required=True)
    f.add_argument('--dest', required=True, type=Path)
    f.add_argument('--record', required=True, type=Path)
    t = sub.add_parser('timing')
    t.add_argument('--root', default=Path('.'), type=Path)
    t.add_argument('--downloads', required=True, type=Path)
    t.add_argument('--record', required=True, nargs='+', type=Path)
    t.add_argument('--out', required=True, type=Path)
    c = sub.add_parser('carry')
    c.add_argument('--carried', required=True, type=Path)
    c.add_argument('--resumed', required=True, action='append', help='<name>=<resumed store folder>')
    c.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    if args.cmd == 'carry':
        value = carry(args.carried, dict((s.split('=', 1)[0], Path(s.split('=', 1)[1])) for s in args.resumed))
        dump(args.out, value)
        print(json.dumps({k: value[k] for k in ('carried', 'same_on_both')}, indent=1),
              json.dumps(value['resumed'], indent=1))
        return 0 if value['same_on_both'] else 1
    if args.cmd == 'fetch':
        record = fetch(args.repo, args.run, args.dest)
        dump(args.record, record)
        print(args.run, record['conclusion'], record['results'])
        return 0 if not record['results']['problems'] else 1
    reference = _reference(args.root)
    out = {'schema': 'cp18a-runner-timing-v1', 'ceilings': CEILINGS, 'job_limit_seconds': JOB_LIMIT_SECONDS, 'runs': {}}
    for path in args.record:
        fetched = json.loads(path.read_text())
        folder = args.downloads / fetched['run']
        check = verified(folder)
        if check['problems'] or check['listing_sha256'] != fetched['results']['listing_sha256']:
            raise SystemExit(f'{folder}: the downloaded results do not match their listing; nothing is recomputed')
        out['runs'][fetched['run']] = {'fetch': fetched, **run_timing(folder, reference),
                                       'job_seconds': fetched['job']['seconds']}
    runs = list(out['runs'].values())
    four = [r for r in runs if r['issuances'] == 4]
    out['summary'] = {
        'one_issuance_seconds': sorted(s for r in runs for s in r['issuance_seconds']),
        'four_with_canary_pipeline_seconds': [r['pipeline_seconds'] for r in four],
        'four_with_canary_job_seconds': [r['job_seconds'] for r in four],
        'four_with_canary_inside_job_limit': all(r['job_seconds'] < JOB_LIMIT_SECONDS for r in four) if four else None,
        'all_fits_within_ceilings': all(r['all_fits_within_ceilings'] for r in runs if r['all_fits_within_ceilings'] is not None),
        'all_equal_to_cp17c_where_recorded': all(r['all_equal_to_cp17c_where_recorded'] for r in runs)}
    dump(args.out, out)
    print(json.dumps(out['summary'], indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
