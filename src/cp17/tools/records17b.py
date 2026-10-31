"""Collect CP-17b's run outputs into its committed records (`reports/cp17b/`).

- `native-parity.json`: item 6's native parity (five policies, nine origins), compacted as CP-17 compacted it, and
  compared with CP-17's committed record: every check, payload identity and quantile hash;
- `host-parity.json`: item 6's issuing-host parity and the browsers' identity, compared with CP-17's record;
- `shadow-run.json`: item 2's consecutive-day run, under the issuing host and natively;
- `b0-rebuild.json`: item 2's B0 check;
- `partial-days.json`: item 1's two development days;
- `control.json`: N1's killed-process and stale-record control;
- `issuance.json`: the freeze-origin issuance (item 5);
- `f1-2026-03-30.json`: item 1's spring day, A1 under the issuing host against native, member by member (F1's
  mechanism, observed again);
- `bundle-manifest.json`, `compute-ledger.jsonl`, `retrieval-manifest.json`.

The full outputs, with every fitted result, stay local; each compacted record names its source's SHA-256.

    PYTHONPATH=src python -m cp17.tools.records17b --runs .local/artifacts/cp-17b/runs
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

REPORTS = Path('reports/cp17b')
CP17 = Path('reports/cp17')
LOCAL = '.local/artifacts/cp-17b'


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def _source(path: Path, runs: Path) -> dict:
    return {'path': f'{LOCAL}/runs/{path.relative_to(runs)}', 'sha256': _sha(path)}


def native_parity(runs: Path, root: Path) -> dict:
    full = runs / 'native-parity' / 'native-parity.json'
    value = json.loads(full.read_text())
    committed = json.loads((root / CP17 / 'native-parity.json').read_text())['origins']
    origins, equal = {}, {}
    for day, o in value['origins'].items():
        results = o['results']
        origins[day] = {'pass': o['pass'], 'checks': o['checks'], 'seconds': o['seconds'],
                        'ledger_file_sha256': o['ledger_file_sha256'],
                        'payload_sha256': {p: w['payload_sha256'] for p, w in o['payloads'].items()},
                        'quantiles_sha256': {p: r['quantiles_sha256'] for p, r in results.items()}}
        c = committed[day]
        equal[day] = {k: origins[day][k] == c[k] for k in ('checks', 'payload_sha256', 'quantiles_sha256', 'ledger_file_sha256')}
    keep = ('schema', 'python', 'numpy', 'machine', 'platform', 'blas_threads', 'context_build_seconds', 'states',
            'standard', 'maxrss_bytes', 'total_seconds', 'status')
    return {**{k: value[k] for k in keep}, 'origins': origins, 'full_record': _source(full, runs),
            'equal_to_cp17': equal, 'all_equal_to_cp17': all(all(v.values()) for v in equal.values())}


def host_parity(runs: Path, root: Path) -> dict:
    value = json.loads((runs / 'host-parity' / 'host-parity.json').read_text())
    committed = json.loads((root / CP17 / 'host-parity.json').read_text())
    theirs = {r['stem']: r for r in committed['issuing_host_parity']}
    fields = ('issued_quantiles_sha256', 'native_quantiles_sha256', 'status', 'max_abs_eur', 'selections_match',
              'max_abs_eur_by_level')
    value['equal_to_cp17'] = {r['stem']: {f: r.get(f) == theirs[r['stem']].get(f) for f in fields}
                              for r in value['issuing_host_parity']}
    value['all_equal_to_cp17'] = all(all(v.values()) for v in value['equal_to_cp17'].values()) \
        and set(value['equal_to_cp17']) == set(theirs)
    browsers = {(b['stem'], b['browser']): b for b in committed.get('browser_runs', [])}
    value['browser_equal_to_cp17'] = {f"{b['browser']} {b['stem']}": b.get('issued_quantiles_sha256', b.get('quantiles_sha256'))
                                      == browsers.get((b['stem'], b['browser']), {}).get('issued_quantiles_sha256',
                                      browsers.get((b['stem'], b['browser']), {}).get('quantiles_sha256'))
                                      for b in value.get('browser_runs', [])}
    value['source'] = _source(runs / 'host-parity' / 'host-parity.json', runs)
    return value


def _match(executed: dict | None, files: dict) -> bool:
    return bool(executed) and all(files.get(k) == v for k, v in executed.items())


def shadow_run(runs: Path, files: dict) -> dict:
    out = {}
    for mode in ('host', 'native'):
        path = runs / f'shadow-{mode}' / 'shadow-run.json'
        value = json.loads(path.read_text())
        days = {}
        for day, d in value['days'].items():
            item = {'buffers': {p: {k: b[k] for k in ('bitwise', 'days_equal', 'max_abs_standardized')}
                                for p, b in d['buffers'].items()},
                    'intake': d['intake'], 'payload_sha256': d['payload_sha256'],
                    'shadows': {p: {k: s.get(k) for k in ('status', 'cause', 'keys', 'fitted_by', 'quantiles_sha256',
                                                         'central_sha256', 'scale_sha256', 'wall_seconds', 'seconds')}
                                for p, s in d['shadows'].items()},
                    'total_seconds': d['total_seconds']}
            if mode == 'host':
                item.update({k: d[k] for k in ('path', 'display', 'issued_policy', 'artifact_verifies', 'process_seconds',
                                               'process_peak_bytes', 'within_ceiling', 'blocked_non_local_requests')})
                item['executed_files_match_package'] = all(_match(e, files) for e in d['executed_files'].values())
            else:
                item.update(parity_cp26=d['parity_cp26'], parity_quantiles_bitwise=d['parity_quantiles_bitwise'])
            days[day] = item
        summary = {'days': len(days), 'all_buffers_bitwise': all(b['bitwise'] for d in days.values() for b in d['buffers'].values()),
                   'max_abs_standardized': max(b['max_abs_standardized'] or 0.0 for d in days.values()
                                               for b in d['buffers'].values()),
                   'all_registered': all(v == 'registered' for d in days.values() for v in d['intake'].values())}
        if mode == 'host':
            totals = [d['total_seconds'] for d in days.values()]
            summary.update(total_seconds_max=max(totals), total_seconds_median=sorted(totals)[len(totals) // 2],
                           all_within_ceiling=all(d['within_ceiling'] for d in days.values()),
                           all_artifacts_verify=all(d['artifact_verifies'] for d in days.values()),
                           all_executed_files_match_package=all(d['executed_files_match_package'] for d in days.values()),
                           blocked_non_local_requests=sorted({u for d in days.values() for u in d['blocked_non_local_requests']}),
                           peak_bytes_max=max(max(v for v in d['process_peak_bytes'].values() if v) for d in days.values()))
        else:
            summary['all_parity_cp26'] = all(all(d['parity_cp26'].values()) for d in days.values())
        out[mode] = {'first': value['first'], 'last': value['last'], 'start': value['start'], 'days': days,
                     'final': value['final'], 'summary': summary, 'bundle_sha256': value.get('bundle_sha256'),
                     'seconds': value.get('seconds'), 'source': _source(path, runs)}
    return out


def _compact_partial(value: dict) -> dict:
    art = value['host']['artifact']
    host = {k: v for k, v in value['host'].items() if k != 'artifact'}
    host['artifact'] = {k: art[k] for k in ('artifact_sha256', 'issued_policy', 'display', 'partial', 'keys', 'quantiles_sha256',
                                            'resources', 'attempts')}
    return {**{k: v for k, v in value.items() if k != 'host'}, 'host': host}


def partial_days(runs: Path) -> dict:
    return {p.stem.replace('partial-', ''): {**_compact_partial(json.loads(p.read_text())), 'source': _source(p, runs)}
            for p in sorted((runs / 'partial').glob('partial-*.json'))}


def control(runs: Path) -> dict:
    path = next((runs / 'control').glob('control-*.json'))
    value = json.loads(path.read_text())
    art = value.pop('artifact')
    value['artifact'] = {k: art[k] for k in ('artifact_sha256', 'issued_policy', 'display', 'resources', 'attempts')}
    value['stale_record']['planted'] = [Path(p).name for p in value['stale_record']['planted']]
    value['stale_record']['source'] = Path(value['stale_record']['source']).name
    value['a1_folder'] = Path(value['a1_folder']).name if value.get('a1_folder') else None
    return {**value, 'source': _source(path, runs)}


def issuance(runs: Path, files: dict) -> dict:
    path = next((runs / 'issuance').glob('issuance-*.json'))
    value = json.loads(path.read_text())
    value['artifact_path'] = Path(value['artifact_path']).name
    value['artifact_arrays_path'] = Path(value['artifact_arrays_path']).name if value.get('artifact_arrays_path') else None
    value['executed_files_match_package'] = {p: _match(e, files) for p, e in value.pop('executed_files').items()}
    return {**value, 'source': _source(path, runs)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--runs', required=True, type=Path)
    ap.add_argument('--only', default='')
    args = ap.parse_args(argv)
    root, runs, out = args.root, args.runs, args.root / REPORTS
    bundle = json.loads((runs.parent / 'bundle' / 'src.json').read_text())
    files = bundle['files_sha256']
    jobs = {'bundle-manifest.json': lambda: bundle,
            'native-parity.json': lambda: native_parity(runs, root), 'host-parity.json': lambda: host_parity(runs, root),
            'shadow-run.json': lambda: shadow_run(runs, files), 'b0-rebuild.json': lambda: json.loads((runs / 'b0-rebuild.json').read_text()),
            'partial-days.json': lambda: partial_days(runs), 'control.json': lambda: control(runs),
            'issuance.json': lambda: issuance(runs, files),
            'f1-2026-03-30.json': lambda: {**json.loads((runs / 'partial' / 'f1-2026-03-30.json').read_text()),
                                           'source': _source(runs / 'partial' / 'f1-2026-03-30.json', runs)},
            'retrieval-manifest.json': lambda: {'schema': 'cp17b-retrieval-manifest-v1', 'data_bytes_retrieved': 0,
                                                'software_bytes_retrieved': 0, 'remote_writes': 0,
                                                'note': 'every input was local: the evidence/cp-26 extract, the snapshot '
                                                        'and the pinned runtimes; every host and browser run blocked and '
                                                        'recorded non-local requests (none)'}}
    only = set(args.only.split(',')) if args.only else set(jobs)
    for name, job in jobs.items():
        if name in only:
            _write(out / name, job())
            print('wrote', out / name)
    if 'compute-ledger.jsonl' in only or not args.only:
        shutil.copyfile(runs.parent / 'compute-ledger.jsonl', out / 'compute-ledger.jsonl')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
