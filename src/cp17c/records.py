"""CP-17c's committed records, from its local runs (run locally; each cites its source file by SHA-256).

- `native-parity.json`: native parity with CP-26, five policies at nine origins, compacted as CP-17b compacted it, and
  compared with CP-17b's record (`reports/cp17b/native-parity.json`): the check of code identity, since `schema.py`
  changed.
- `browser.json`: Chromium and WebKit for A1 at the freeze origin on CP-17c's package, since its zip changed: their
  quantiles against the issuing host's (the canary hash) and CP-17b's browser record.
- `r7-controls.json`: H3. The reference bundle rebuilt for CP-17c's freeze origin, valid, with CP-17b's negative
  controls refused as before, and the malformed-day controls: listed by CP-17c's `verify_bundle`, raised by CP-17b's
  (its `schema.py`, checked against CP-17b's frozen hash), on the same bytes.
- `freeze-origin.json`: the reissue of 2026-09-06 under the official binary, field by field against CP-17b's.
- `state-route.json`: the freeze-origin states and the canary payload, as built (`cp17c.state`).
- `resume-rehearsal.json`: the M3 resuming the runner's early-block store and issuing 2026-08-30.

    PYTHONPATH=src python -m cp17c.records --local .local/artifacts/cp-17c --old-src <CP-17b's src> --out reports/cp17c
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .run import dump, sha256_file

LOCAL = '.local/artifacts/cp-17c'
CP17B = 'reports/cp17b'
CANARY = '4e6abb02fbb02aa8ef202207f21b76bc33b34c37d025a0a93a491135ea43cea5'
CP17B_SCHEMA_SHA256 = None      # read from CP-17b's frozen identity


def _source(path: Path, local: Path) -> dict:
    return {'path': f'{LOCAL}/{path.relative_to(local)}', 'sha256': sha256_file(path)}


def native_parity(local: Path, root: Path) -> dict:
    full = local / 'runs' / 'native-parity' / 'native-parity.json'
    value = json.loads(full.read_text())
    committed = json.loads((root / CP17B / 'native-parity.json').read_text())['origins']
    origins, equal = {}, {}
    for day, o in value['origins'].items():
        origins[day] = {'pass': o['pass'], 'checks': o['checks'], 'seconds': o['seconds'],
                        'ledger_file_sha256': o['ledger_file_sha256'],
                        'payload_sha256': {p: w['payload_sha256'] for p, w in o['payloads'].items()},
                        'quantiles_sha256': {p: r['quantiles_sha256'] for p, r in o['results'].items()}}
        equal[day] = {k: origins[day][k] == committed[day][k]
                      for k in ('checks', 'payload_sha256', 'quantiles_sha256', 'ledger_file_sha256')}
    keep = ('schema', 'python', 'numpy', 'machine', 'platform', 'blas_threads', 'context_build_seconds', 'states',
            'standard', 'maxrss_bytes', 'total_seconds', 'status')
    return {**{k: value[k] for k in keep}, 'origins': origins, 'full_record': _source(full, local),
            'equal_to_cp17b': equal, 'all_equal_to_cp17b': all(all(v.values()) for v in equal.values()) and len(equal) == 9}


def browser(local: Path, root: Path, manifest: dict) -> dict:
    path = local / 'runs' / 'browser-results.json'
    value = json.loads(path.read_text())
    committed = json.loads((root / CP17B / 'host-parity.json').read_text())
    # CP-17b's browser runs (A1 at 2026-09-06) record their bitwise equality with its issuing host's A1 there
    host = next(r['issued_quantiles_sha256'] for r in committed['issuing_host_parity'] if r['stem'] == 'A1-2026-09-06')
    theirs = {b['browser']: host if b['bitwise_equal_to_issuing_host'] and not b['errors'] else None
              for b in committed['browser_runs']}
    runs = [{'browser': r['browser'], 'browser_version': r.get('browser_version'), 'stem': r.get('stem'),
             'errors': r.get('errors'), 'quantiles_sha256': ((r.get('python') or {}).get('result') or {}).get('quantiles_sha256'),
             'seconds': r.get('total_ms', 0) / 1000 if r.get('total_ms') else None,
             'executed_files_match_package': all(manifest['policy_identity']['code']['files_sha256'].get(k) == v
                                                 for k, v in ((r.get('python') or {}).get('source_sha256') or {}).items())
             and bool((r.get('python') or {}).get('source_sha256'))} for r in value['runs']]
    return {'schema': 'cp17c-browser-v1', 'playwright': value['playwright'], 'platform': value['platform'],
            'machine': value['machine'], 'cpu_count': value['cpu_count'], 'runs': runs,
            'package_zip_sha256': manifest['policy_identity']['code']['zip_sha256'],
            'equal_to_issuing_host': {r['browser']: r['quantiles_sha256'] == CANARY for r in runs},
            'equal_to_cp17b_browser_record': {r['browser']: r['quantiles_sha256'] == theirs.get(r['browser']) for r in runs},
            'source': _source(path, local)}


R7_CASES = ('delivery_day_absent', 'keys_absent', 'day_not_an_object', 'impossible_date', 'keys_not_a_list', 'days_not_a_list')
R7_SCRIPT = """
import hashlib, json, sys
from pathlib import Path
from cp17 import schema
from cp17.tools import refbundle
files = refbundle.read(Path(sys.argv[1]))
def rehash(value):
    out = dict(files); out['issued-forecasts.json'] = json.dumps(value, sort_keys=True).encode()
    m = json.loads(files['manifest.json'])
    for f in m['files']:
        if f['path'] == 'issued-forecasts.json':
            f.update(sha256=hashlib.sha256(out['issued-forecasts.json']).hexdigest(), bytes=len(out['issued-forecasts.json']))
    out['manifest.json'] = json.dumps(m, sort_keys=True).encode()
    return out
base = json.loads(files['issued-forecasts.json'])
def case(name):
    v = json.loads(json.dumps(base)); d = v['days'][0]
    if name == 'delivery_day_absent': d.pop('delivery_day')
    elif name == 'keys_absent': d.pop('keys')
    elif name == 'day_not_an_object': v['days'][0] = 'not a day'
    elif name == 'impossible_date': d['delivery_day'] = '2026-13-45'
    elif name == 'keys_not_a_list': d['keys'] = 5
    elif name == 'days_not_a_list': v['days'] = {'not': 'a list'}
    return v
out = {'schema_py_sha256': hashlib.sha256(Path(schema.__file__).read_bytes()).hexdigest(),
       'valid': schema.verify_bundle(files), 'controls': refbundle.tamper_controls(files), 'r7': {}}
for name in sys.argv[2:]:
    try:
        errors = schema.verify_bundle(rehash(case(name)))
        out['r7'][name] = {'raised': None, 'refused': bool(errors), 'listed': [e for e in errors if 'R7' in e]}
    except Exception as exc:
        out['r7'][name] = {'raised': f'{type(exc).__name__}: {exc}', 'refused': None, 'listed': []}
print(json.dumps(out, sort_keys=True))
"""


def r7(local: Path, root: Path, old_src: Path) -> dict:
    cp17b_schema = json.loads((root / 'docs/track-b/evidence/cp-17b/freeze-manifest.json').read_text())[
        'policy_identity']['code']['files_sha256']['cp17/schema.py']
    folder = local / 'reference-bundle'
    runs = {}
    for label, src in (('cp17c', root / 'src'), ('cp17b', old_src)):
        done = subprocess.run([sys.executable, '-c', R7_SCRIPT, str(folder), *R7_CASES], capture_output=True, text=True,
                              env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(src)}, check=True)
        runs[label] = json.loads(done.stdout)
    if runs['cp17b']['schema_py_sha256'] != cp17b_schema:
        raise SystemExit("the old schema.py is not CP-17b's frozen file")
    new, old = runs['cp17c'], runs['cp17b']
    return {'schema': 'cp17c-r7-controls-v1', 'bundle': {'location': f'{LOCAL}/reference-bundle/',
                                                          'manifest_sha256': sha256_file(folder / 'manifest.json')},
            'schema_py_sha256': {'cp17c': new['schema_py_sha256'], 'cp17b': old['schema_py_sha256']},
            'valid': {'cp17c': new['valid'] == [], 'cp17b': old['valid'] == []},
            'negative_controls_equal': new['controls'] == old['controls'],
            'negative_controls_all_refused': all(c['refused'] for c in new['controls'].values()),
            'negative_controls': new['controls'],
            'malformed_day': {name: {'cp17c': new['r7'][name], 'cp17b': old['r7'][name]} for name in R7_CASES},
            'all_listed_by_cp17c': all(new['r7'][n]['refused'] and not new['r7'][n]['raised'] for n in R7_CASES),
            'all_raised_by_cp17b': all(old['r7'][n]['raised'] for n in R7_CASES)}


def freeze_origin(local: Path, root: Path) -> dict:
    path = local / 'freeze' / 'issuance' / 'issuance-2026-09-06.json'
    value = json.loads(path.read_text())
    art = value['artifact']
    old = json.loads((root / 'docs/track-b/evidence/cp-17b/artifact-2026-09-06.json').read_text())
    host = json.loads((root / CP17B / 'host-parity.json').read_text())
    shadows = {r['stem'].split('-', 1)[0]: r['issued_quantiles_sha256'] for r in host['issuing_host_parity']
               if r['stem'].endswith('2026-09-06')}

    def flat(d, p=''):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f'{p}{k}.') if isinstance(v, dict) else {f'{p}{k}': v})
        return out
    a, b = flat(art), flat(old)
    differ = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    return {'schema': 'cp17c-freeze-origin-v1', 'artifact_sha256': art['artifact_sha256'],
            'quantiles_sha256': art['quantiles_sha256'], 'equal_to_canary': art['quantiles_sha256'] == CANARY,
            'path': value['path'], 'display': value['display'], 'artifact_verifies': value['artifact_verifies'],
            'fields_differing_from_cp17b': differ,
            'expected_differences': ['artifact_sha256', 'attempts', 'lineage.code_sha256', 'lineage.policy_identity_sha256',
                                     'resources.peak_bytes', 'resources.wall_seconds'],
            'artifact_arrays_sha256': value['artifact_arrays_sha256'],
            'arrays_equal_to_cp17b': value['artifact_arrays_sha256'] == json.loads(
                (root / 'docs/track-b/evidence/cp-17b/freeze-manifest.json').read_text())['freeze_origin']['artifact_arrays_sha256'],
            'shadows': {p: s.get('quantiles_sha256') for p, s in value['shadows'].items()},
            'shadows_equal_to_cp17b_host_parity': {p: value['shadows'][p].get('quantiles_sha256') == shadows[p]
                                                   for p in ('v4', 'v2', 'B0')},
            'process_seconds': value['process_seconds'], 'total_seconds': value['total_seconds'],
            'process_peak_bytes': value['process_peak_bytes'], 'executed_files': value['executed_files'],
            'blocked_non_local_requests': value['blocked_non_local_requests'], 'source': _source(path, local)}


def resume_rehearsal(local: Path, root: Path) -> dict:
    """The M3 resumed the runner's early-block store (run 3's artifact) and issued 2026-08-30: its buffers and every
    fit against CP-17b's references, and the carried part of the store unchanged."""
    from .store import Store
    refs = json.loads((root / 'reports/cp17c/inputs.json').read_text())['references']
    folder = local / 'local-resume-rehearsal' / 'shadow-early'
    shadow = json.loads((folder / 'shadow.json').read_text())
    head = Store(folder / 'store').verify()
    day = json.loads((folder / 'store/issued/2026-08-30/record.json').read_text())
    carried = folder.parents[1] / 'downloads/41033064719/shadows/shadow-early/store'
    before = {str(p.relative_to(carried)): sha256_file(p) for p in sorted(carried.rglob('*')) if p.is_file()}
    after = {str(p.relative_to(folder / 'store')): sha256_file(p) for p in sorted((folder / 'store').rglob('*')) if p.is_file()}
    return {'schema': 'cp17c-resume-rehearsal-v1', 'resumed_from': shadow['resumed_from']['head'],
            'gate_passed': shadow['gate_passed'], 'day': '2026-08-30', 'store_verifies': True, 'entries': head['entries'],
            'carried_files_unchanged': all(after.get(rel) == h for rel, h in before.items()
                                           if rel not in ('head.json', 'ledger.jsonl')),
            'buffers_equal_reference': day['buffers_sha256'] == refs['shadow_buffers_sha256']['2026-08-30'],
            'buffers_equal_payload': all(day['buffers_equal_payload'].values()),
            'fits_bit_identical': {p: day['runs'][p]['quantiles_sha256'] == refs['runs']['shadow'][f'{p}-2026-08-30']['quantiles_sha256']
                                   for p in ('A1', 'v4', 'v2', 'B0')},
            'intake': day['intake'], 'issued_policy': day['issued_policy'],
            'source': _source(folder / 'shadow.json', local)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--local', required=True, type=Path)
    ap.add_argument('--old-src', required=True, type=Path, help="CP-17b's src/ (for R7's before), checked by hash")
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    root, local = args.root.resolve(), args.local.resolve()
    manifest = json.loads((root / 'docs/track-b/evidence/cp-17c/freeze-manifest.json').read_text())
    made = {'native-parity.json': native_parity(local, root), 'browser.json': browser(local, root, manifest),
            'r7-controls.json': r7(local, root, args.old_src.resolve()), 'freeze-origin.json': freeze_origin(local, root),
            'state-route.json': json.loads((local / 'records' / 'state-route.json').read_text()),
            'resume-rehearsal.json': resume_rehearsal(local, root)}
    for name, value in made.items():
        dump(args.out / name, value)
    print({k: v.get('all_equal_to_cp17b', v.get('all_listed_by_cp17c', v.get('equal_to_canary', True))) for k, v in made.items()})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
