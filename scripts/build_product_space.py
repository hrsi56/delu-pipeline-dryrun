#!/usr/bin/env python3
"""Assemble the final-product Space (A9; CP-18b) from `space-product/`, the pinned runtime and one dated bundle.

    PYTHONPATH=src python scripts/build_product_space.py --bundle <dated bundle> --out <dist> \\
        [--history <v1 build dir> [--history-note TEXT]] [--allow-refused]

The served layout is fixed by the frozen "Train it yourself" worker (`src/cp17/browser/worker.js`, served byte for
byte at `/app/worker.js`): `/index.html`, `/app/...`, `/pyodide/<the 23 pinned files>`, `/bundle/src.zip` (the frozen
package), `/payloads/<the dated bundle>` and, when given, `/history/` (v1's demo as it is).

**It refuses** (exit 1, with the reason) unless `cp18.r7.verify` accepts the bundle and `cp18.bundle.gate` finds it
publishable; `--allow-refused` builds anyway for the acceptance's negative controls and stamps the build record
`refused`. It also refuses any Pyodide file, the package zip or the frozen worker that differs from the CP-17c freeze
manifest's hashes.

**What it writes besides the copies:**

- `app/claims.json`: the live claims (`cp18.live_claims.build_live_claims`, with the bundle's first and last issued day),
  the shared record sentences and the limitations as label, key and text. The page reads every status, badge, wording
  and limitation sentence from here.
- `app/space-build.json`: every served file with its SHA-256 and bytes (all but this record itself); the glue's and the
  frozen worker's SHA-256; the bundle manifest's SHA-256 and the gate's verdict; the Pyodide and zip checks; the
  reference measurements (`reports/cp17/host-parity.json`'s A1 browser runs); and the size budget: what the page loads
  at start against what loads on request, by action.
- `README.md`: the Hugging Face Static Space card, its text from the live claims. Written, never published.
- `.gitattributes`: Git LFS rules for the binary files (a Space repository needs them for files over 10 MB).

The output is deterministic: no timestamps, sorted listings, the same inputs give the same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

SCHEMA = 'cp18b-space-build-v1'
SOURCE = ROOT / 'space-product'
FROZEN_WORKER = ROOT / 'src/cp17/browser/worker.js'
GLUE = ROOT / 'src/cp18/spaceglue.py'
FREEZE_MANIFEST = ROOT / 'docs/track-b/evidence/cp-17c/freeze-manifest.json'
HOST_PARITY = ROOT / 'reports/cp17/host-parity.json'
#: The files the page itself loads at start, beside the bundle's start records (`cp18.bundle.START_FILES`).
PAGE_START = ('index.html', 'app/style.css', 'app/charts.js', 'app/main.js', 'app/claims.json', 'app/space-build.json')
#: The runtime files Pyodide's loader fetches (pyodide.mjs is served with the release and never loaded by a worker).
RUNTIME_LOADED = ('pyodide.js', 'pyodide.asm.js', 'pyodide.asm.wasm', 'python_stdlib.zip', 'pyodide-lock.json')
GITATTRIBUTES = ''.join(f'*.{ext} filter=lfs diff=lfs merge=lfs -text\n'
                        for ext in ('wasm', 'whl', 'zip', 'npz', 'png', 'ico', 'woff2', 'woff', 'ttf'))
REPORT_URL = 'https://hrsi56.github.io/delu-day-ahead-forecast/'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(value) -> str:
    return json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n'


class Refused(Exception):
    pass


# ------------------------------------------------------------------ inputs
def project_root() -> Path:
    from cp17 import runtime
    return Path(runtime.project_root(ROOT))


def check_bundle(files: dict[str, bytes]) -> dict:
    """R7's verification, then the publication gate (which repeats R7 and recomputes natively)."""
    from cp18 import bundle, r7
    verified = r7.verify(files)
    verdict = bundle.gate(files)
    recompute = verdict.get('recompute') or {}
    return {'r7_accepted': verified['accepted'], 'r7_problems': verified['problems'],
            'publishable': bool(verified['accepted'] and verdict['publishable']), 'reason': verdict.get('reason'),
            'gate_problems': verdict['verify']['problems'],
            'recompute': None if not recompute else {
                'match': recompute.get('match'), 'blocks_publication': recompute.get('blocks_publication'),
                'recomputed_windows': recompute.get('recomputed_windows'), 'error': recompute.get('error')}}


def pyodide_files(folder: Path, freeze: dict) -> dict:
    expected = freeze['policy_identity']['issuing_host']['pyodide']['files_sha256']
    checks = {}
    for name in sorted(expected):
        path = folder / name
        if not path.is_file():
            raise Refused(f'pyodide: {name} is absent from {folder}')
        got = sha(path.read_bytes())
        if got != expected[name]:
            raise Refused(f'pyodide: {name} hashes {got[:16]}..., the freeze manifest records {expected[name][:16]}...')
        checks[name] = {'sha256': got, 'bytes': path.stat().st_size, 'matches_freeze_manifest': True}
    return checks


def loaded_packages(folder: Path, packages: list[str]) -> list[str]:
    """The package files `loadPackage(packages)` fetches: the named packages and their dependencies, by the lock."""
    lock = json.loads((folder / 'pyodide-lock.json').read_text())['packages']
    seen, order = set(), []

    def add(name):
        name = name.lower()
        if name in seen:
            return
        seen.add(name)
        for dep in lock[name].get('depends', []):
            add(dep)
        order.append(lock[name]['file_name'])
    for name in packages:
        add(name)
    return sorted(order)


def reference_measurements(claims: dict) -> dict:
    """The A1 browser runs of CP-17's host parity, read as recorded; the measurement date from the visitor wording."""
    parity = json.loads(HOST_PARITY.read_text())
    m = re.search(r'measured (\d{4}-\d{2}-\d{2})', claims['values']['live_retrain_wording'])
    runs = [{'browser': r['browser'], 'browser_version': r['browser_version'], 'stem': r['stem'],
             'fit_seconds': r['time_seconds']['fit'], 'total_in_worker_seconds': r['time_seconds']['total_in_worker'],
             'payload_bytes_raw': r['payload_bytes']['raw'], 'runtime': r['runtime'],
             'status_vs_issuing_host': r['status_vs_issuing_host']}
            for r in parity['browser_runs'] if r['policy'] == 'A1']
    return {'source': 'reports/cp17/host-parity.json (browser_runs, policy A1)', 'runs': runs,
            'measured_date': m.group(1) if m else None,
            'measured_date_source': 'live_retrain_wording', 'machine': parity['browser_host']}


def issued_days(files: dict[str, bytes]) -> list[str] | None:
    """The bundle's issued days; None when its issued forecasts do not parse (a refused bundle, built for a control)."""
    try:
        return sorted(d['delivery_day'] for d in json.loads(files['issued-forecasts.json'])['days'])
    except (ValueError, KeyError, TypeError):
        return None


def claims_for(files: dict[str, bytes]) -> dict:
    from cp18.live_claims import build_live_claims, text
    days = issued_days(files) or []
    claims = build_live_claims(ROOT, rehearsal_days=(days[0], days[-1]) if days else ())
    return {'schema': 'cp18b-space-claims-v1', 'source': 'cp18.live_claims.build_live_claims',
            'rehearsal_days': [days[0], days[-1]] if days else [],
            'values': claims['values'], 'shared': claims['shared'],
            'limitations': [{'label': label, 'key': key, 'text': text(claims, key)} for label, key in claims['limitations']]}


# ------------------------------------------------------------------ outputs
def card(claims: dict, manifest: dict, refused: bool) -> str:
    v, s = claims['values'], claims['shared']
    lines = ['---', f"title: {json.dumps(v['live_product_name'], ensure_ascii=False)}", 'sdk: static',
             'app_file: index.html', 'pinned: false', '---', '', f"# {v['live_product_name']}", '',
             f"**Status:** {v['live_product_status']}. {v['live_released_product']}.", '',
             f"**{v['live_prospective_label']}.** {v['live_hero_rule']}", '',
             f"**Evidence:** {manifest['product']['evidence_label']}", '',
             f"**Served identity:** policy identity `{manifest['policy']['policy_identity_sha256']}`, package "
             f"`{manifest['policy']['package_sha256']}`, issuing host {manifest['policy']['issuing_host']}, daily "
             f"artifact `{manifest['artifact']['artifact_sha256']}` ({manifest['artifact']['delivery_day']}).", '',
             f"The static report: [{REPORT_URL}]({REPORT_URL}) (external).", '']
    if refused:
        lines += ['**This build is refused:** its bundle did not pass the publication gate; it exists for an '
                  'acceptance check only.', '']
    lines += ['## Limitations', '']
    lines += [f"- **{item['label']}.** {item['text']}" for item in claims['limitations']]
    lines += ['', s['attribution'], '']
    return '\n'.join(lines)


def size_budget(out: Path, start_records: tuple, packages: list[str], artifact: dict, record_bytes: int) -> dict:
    def b(rel):
        return (out / rel).stat().st_size if rel != 'app/space-build.json' else record_bytes

    def group(names):
        items = {n: b(n) for n in sorted(names) if (out / n).exists() or n == 'app/space-build.json'}
        return {'bytes': sum(items.values()), 'files': items}
    payloads = sorted(p.name for p in (out / 'payloads').iterdir())
    train_stem = f"{artifact['issued_policy']}-{artifact['delivery_day']}" if artifact.get('issued_policy') else None
    first = group(list(PAGE_START) + [f'payloads/{n}' for n in start_records])
    runtime = group([f'pyodide/{n}' for n in RUNTIME_LOADED])
    pk = group([f'pyodide/{n}' for n in packages])
    code_train = group(['bundle/src.zip', 'app/worker.js'])
    code_product = group(['bundle/src.zip', 'app/glue/spaceglue.py', 'app/product-worker.js'])
    training = group([f'payloads/{n}' for n in payloads if train_stem and n.startswith(train_stem + '.')])
    arrays = group(['payloads/artifact-arrays.npz'] if 'artifact-arrays.npz' in payloads else [])
    b0 = group([f'payloads/{n}' for n in payloads if n.startswith('B0-')])
    series = group(['payloads/series.json'] if 'series.json' in payloads else [])
    product_runtime = runtime['bytes'] + pk['bytes'] + code_product['bytes']
    on_request = {'runtime': runtime, 'packages': pk, 'code_train': code_train, 'code_product': code_product,
                  'training_window': training, 'fitted_arrays': arrays, 'b0_windows': b0, 'series': series}
    seen = set()
    for g in on_request.values():
        seen |= set(g['files'])
    return {
        'unit': 'bytes, uncompressed, as served',
        'first_visit': first,
        'on_request': on_request,
        'on_request_bytes': sum(b(n) for n in seen),
        'actions': {
            'train_it_yourself': {'bytes': runtime['bytes'] + pk['bytes'] + code_train['bytes'] + training['bytes'],
                                  'parts': {'runtime': runtime['bytes'], 'packages': pk['bytes'],
                                            'code': code_train['bytes'], 'training_window': training['bytes']},
                                  'stem': train_stem},
            'product_runtime': {'bytes': product_runtime,
                                'parts': {'runtime': runtime['bytes'], 'packages': pk['bytes'],
                                          'code': code_product['bytes']},
                                'note': 'loaded once per visit, by the first in-browser computation'},
            'recompute_data': {'bytes': b0['bytes']},
            'explain_data': {'bytes': arrays['bytes'] + training['bytes']},
            'data_view': {'bytes': series['bytes']},
        },
        'served_not_loaded': sorted(f'pyodide/{n}' for n in
                                    set(p.name for p in (out / 'pyodide').iterdir()) - set(RUNTIME_LOADED) - set(packages)),
        'not_counted': ('history/ (v1\'s demo, outside the product Space\'s network rule) and its listing '
                        'app/history-files.json, README.md and .gitattributes'),
    }


def listing(out: Path, history: bool = False) -> list[dict]:
    """Every served file with its SHA-256 and bytes: the product's (all but the build record itself), or history/'s."""
    rows = []
    for path in sorted(out.rglob('*')):
        rel = path.relative_to(out).as_posix()
        if not path.is_file() or rel == 'app/space-build.json' or rel.startswith('history/') != history:
            continue
        data = path.read_bytes()
        rows.append({'path': rel, 'sha256': sha(data), 'bytes': len(data)})
    return rows


def build(bundle_dir: Path, out: Path, *, history: Path | None = None, history_note: str | None = None,
          allow_refused: bool = False, pyodide: Path | None = None, package_zip: Path | None = None) -> dict:
    from cp17 import runtime
    from cp18 import bundle as B
    from cp18.host import PACKAGE_ZIP
    project = project_root()
    pyodide = Path(pyodide or project / runtime.PYODIDE_DIR)
    package_zip = Path(package_zip or project / PACKAGE_ZIP)
    files = B.read(bundle_dir)
    if 'manifest.json' not in files or 'issued-forecasts.json' not in files:
        raise Refused(f'{bundle_dir}: not a dated bundle (no manifest.json or issued-forecasts.json)')
    gate = check_bundle(files)
    if not gate['publishable'] and not allow_refused:
        raise Refused('the bundle is refused: ' + (gate['reason'] or '; '.join(gate['r7_problems'][:3])) +
                      (' -- ' + '; '.join(gate['gate_problems'][:3]) if gate['gate_problems'] else ''))
    freeze = json.loads(FREEZE_MANIFEST.read_text())
    code = freeze['policy_identity']['code']
    checks_pyodide = pyodide_files(pyodide, freeze)
    zip_bytes = package_zip.read_bytes()
    if sha(zip_bytes) != code['zip_sha256']:
        raise Refused(f"the package zip hashes {sha(zip_bytes)[:16]}..., the freeze manifest records {code['zip_sha256'][:16]}...")
    worker = FROZEN_WORKER.read_bytes()
    frozen_worker_sha = code['runner_files_sha256']['cp17/browser/worker.js']
    if sha(worker) != frozen_worker_sha:
        raise Refused('src/cp17/browser/worker.js differs from the frozen runner file')

    if out.exists():
        shutil.rmtree(out)
    (out / 'app/glue').mkdir(parents=True)
    for path in sorted(SOURCE.rglob('*')):
        if path.is_file() and path.name != '.DS_Store':
            target = out / path.relative_to(SOURCE)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    (out / 'pyodide').mkdir()
    for name in checks_pyodide:
        (out / 'pyodide' / name).write_bytes((pyodide / name).read_bytes())
    (out / 'bundle').mkdir()
    (out / 'bundle/src.zip').write_bytes(zip_bytes)
    (out / 'app/worker.js').write_bytes(worker)
    if (out / 'app/worker.js').read_bytes() != FROZEN_WORKER.read_bytes():
        raise Refused('the served worker is not byte-identical to the frozen one')
    (out / 'app/glue/spaceglue.py').write_bytes(GLUE.read_bytes())
    (out / 'payloads').mkdir()
    for name, data in sorted(files.items()):
        (out / 'payloads' / name).write_bytes(data)
    history_info = {'present': False, 'note': history_note}
    if history is not None:
        skipped = []
        for path in sorted(Path(history).rglob('*')):
            rel = path.relative_to(history)
            if path.name == '.DS_Store':
                skipped.append(rel.as_posix())
                continue
            if path.is_file():
                target = out / 'history' / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
        hist = [p for p in (out / 'history').rglob('*') if p.is_file()]
        history_info = {'present': True, 'note': history_note, 'files': len(hist),
                        'bytes': sum(p.stat().st_size for p in hist),
                        'index_sha256': sha((out / 'history/index.html').read_bytes())
                        if (out / 'history/index.html').is_file() else None,
                        'skipped': skipped or None}

    claims = claims_for(files)
    (out / 'app/claims.json').write_text(dump(claims), encoding='utf-8')
    manifest = json.loads(files['manifest.json'])
    try:
        artifact = json.loads(files['artifact.json'])
    except (KeyError, ValueError):
        artifact = {}
    (out / 'README.md').write_text(card(claims, manifest, not gate['publishable']), encoding='utf-8')
    (out / '.gitattributes').write_text(GITATTRIBUTES, encoding='utf-8')

    packages = loaded_packages(pyodide, code['pyodide_packages'])
    record = {
        'schema': SCHEMA,
        'refused': not gate['publishable'],
        'gate': gate,
        'bundle': {'manifest_sha256': sha(files['manifest.json']), 'freshness': manifest['freshness'],
                   'artifact': manifest['artifact'], 'files': len(files),
                   'start_records': list(B.START_FILES[1:])},
        'identity': {'policy_identity_sha256': freeze['policy_identity_sha256'],
                     'package_sha256': code['package_sha256'], 'zip_sha256': code['zip_sha256']},
        'pyodide': {'version': freeze['policy_identity']['issuing_host']['pyodide']['version'],
                    'packages': code['pyodide_packages'], 'package_files': packages, 'files': checks_pyodide},
        'zip': {'path': 'bundle/src.zip', 'sha256': sha(zip_bytes), 'bytes': len(zip_bytes),
                'matches_freeze_manifest': True},
        'glue': {'path': 'app/glue/spaceglue.py', 'sha256': sha(GLUE.read_bytes()),
                 'source': 'src/cp18/spaceglue.py'},
        'frozen_worker': {'path': 'app/worker.js', 'sha256': sha(worker), 'frozen_sha256': frozen_worker_sha,
                          'byte_identical_to_source': True, 'source': 'src/cp17/browser/worker.js'},
        'claims_sha256': sha((out / 'app/claims.json').read_bytes()),
        'claims_rehearsal_days': claims['rehearsal_days'] or None,
        'reference': reference_measurements(claims),
        'history': history_info,
        'report_url': REPORT_URL,
        'files_note': ('every served file but this record itself; history/\'s files are listed in '
                       'app/history-files.json, which is listed here with its SHA-256'),
    }
    if history_info['present']:
        # v1's ~800 files are listed in their own record, bound here by its hash, so the first visit stays small
        (out / 'app/history-files.json').write_text(dump({'schema': 'cp18b-space-history-files-v1',
                                                          'files': listing(out, history=True)}), encoding='utf-8')
        history_info['listing'] = 'app/history-files.json'
    record['files'] = listing(out)
    # the record's own size is part of the first visit: settle it at a fixed point
    size = 0
    for _ in range(8):
        record['size'] = size_budget(out, B.START_FILES, packages, artifact, size)
        text = dump(record)
        if len(text.encode('utf-8')) == size:
            break
        size = len(text.encode('utf-8'))
    else:
        raise RuntimeError('the build record size did not settle')
    (out / 'app/space-build.json').write_text(text, encoding='utf-8')
    return record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='build_product_space.py')
    ap.add_argument('--bundle', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--history', type=Path, help="v1's build directory, copied as it is to <out>/history/")
    ap.add_argument('--history-note', help='a note the page and the build record show beside the History route')
    ap.add_argument('--allow-refused', action='store_true',
                    help='build a refused bundle anyway (negative controls only); the build record says refused')
    ap.add_argument('--pyodide', type=Path, help='the pinned Pyodide folder (default: the project\'s)')
    ap.add_argument('--zip', type=Path, help='the frozen package zip (default: the project\'s)')
    args = ap.parse_args(argv)
    try:
        record = build(args.bundle.resolve(), args.out.resolve(), history=args.history, history_note=args.history_note,
                       allow_refused=args.allow_refused, pyodide=args.pyodide, package_zip=args.zip)
    except Refused as exc:
        print('REFUSED:', exc)
        return 1
    s = record['size']
    print('built', args.out, '| bundle', record['bundle']['manifest_sha256'][:16],
          record['bundle']['freshness']['state'], '| REFUSED (allowed)' if record['refused'] else '| publishable',
          '| first visit', s['first_visit']['bytes'], 'bytes | on request', s['on_request_bytes'], 'bytes')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
