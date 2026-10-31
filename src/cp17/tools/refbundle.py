"""Item 7's reference bundle: one dated bundle for the freeze origin (delivery 2026-09-06), built from
development data by `src/cp17`, validated against the frozen schemas, and checked to refuse a tampered file.

Its files stay under `.local/` (the reproduction's `reference-bundle/`; the record names the folder); the committed
record is its manifest and the validation result (`reports/cp17/reference-bundle.json`). Building the daily pipeline that publishes bundles
is CP-18's.

- the fitted artifact: the issuing host's artifact for 2026-09-06 and its exported arrays (A1's members'
  parameters);
- the issued forecasts: that artifact's day;
- the outcomes: the published prices of that day, from the development snapshot;
- the evaluation record: `cp17.scoring.evaluate` over the shipped forecasts and outcomes;
- the training window: A1's payload for 2026-09-06;
- the series: the hourly prices of the last 31 development days, through 2026-09-06.

    PYTHONPATH=src python -m cp17.tools.refbundle --issuance <issuance-2026-09-06-gfs1.json> \\
        --artifact-dir <issuance dir> --payloads <dir> --identity <policy-identity.json> \\
        --out <bundle dir> --record reports/cp17/reference-bundle.json
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from .. import schema, scoring
from ..issue import canonical_hours


DAY = date(2026, 9, 6)
SERIES_DAYS = 31
SOURCE = 'SMARD chart_data filter 4169 (day-ahead price), data/snapshot.parquet'


def _dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n', encoding='utf-8')


def prices(root: Path, first: date, last: date) -> list[tuple[str, str, float | None]]:
    from cp28 import reader
    series = reader.prices(root)
    out = []
    for d in (first + timedelta(days=i) for i in range((last - first).days + 1)):
        for key in canonical_hours(d):
            value = series.get(key)
            out.append((key, str(d), None if value is None or not np.isfinite(value) else float(value)))
    return out


def build(root: Path, issuance: dict, artifact_dir: Path, payload_dir: Path, identity: dict, out: Path,
          arrays: Path | None = None) -> dict:
    """`arrays` is the issued fit's exported arrays; by default CP-17's driver's path in `artifact_dir`."""
    from .freeze import digest
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        old.unlink()
    artifact = issuance['artifact']
    day = artifact['delivery_day']
    _dump(out / 'artifact.json', artifact)
    if arrays is None:
        arrays = artifact_dir / f"node-attempt-{artifact['issued_policy']}-{day}-arrays.npz"
    shutil.copyfile(arrays, out / 'artifact-arrays.npz')
    for ext in ('json', 'npz'):
        shutil.copyfile(payload_dir / f"{artifact['issued_policy']}-{day}.{ext}", out / f"{artifact['issued_policy']}-{day}.{ext}")
    forecasts = {'schema': schema.ISSUED_FORECASTS_SCHEMA, 'levels': list(scoring.LEVELS),
                 'days': [{'delivery_day': day, 'issue_utc': artifact['issue_utc'],
                           'issuing_artifact_sha256': artifact['artifact_sha256'], 'issued_policy': artifact['issued_policy'],
                           'display': artifact['display'], 'keys': artifact['keys'], 'quantiles': artifact['quantiles'],
                           'partial': artifact['partial']}]}
    _dump(out / 'issued-forecasts.json', forecasts)
    rows = prices(root, DAY, DAY)
    outcomes = {'schema': 'cp17-outcomes-v1', 'unit': 'EUR/MWh',
                'prices': [{'timestamp_utc': k, 'delivery_day': d, 'price': v, 'source': SOURCE, 'published_utc': '',
                            'revised_utc': ''} for k, d, v in rows]}
    _dump(out / 'outcomes.json', outcomes)
    issued = [{'day': f['delivery_day'], 'keys': f['keys'], 'quantiles': f['quantiles'],
               'canonical_hours': len(canonical_hours(date.fromisoformat(f['delivery_day'])))} for f in forecasts['days']]
    record = scoring.evaluate(issued, {p['timestamp_utc']: (np.nan if p['price'] is None else p['price'])
                                       for p in outcomes['prices']})
    _dump(out / 'evaluation.json', {'schema': 'cp17-evaluation-bundle-v1', 'windows': [
        {'window': 'development', 'evidence_class': 'development_post_selection (reference bundle, not a result)',
         'policy': 'product', 'record': record}]})
    series = prices(root, DAY - timedelta(days=SERIES_DAYS - 1), DAY)
    _dump(out / 'series.json', {'schema': 'cp17-series-v1', 'unit': 'EUR/MWh', 'first_day': series[0][1],
                                'last_day': series[-1][1], 'timestamp_utc': [k for k, _, _ in series],
                                'price': [v for _, _, v in series]})
    files = []
    for path in sorted(out.iterdir()):
        kind = schema.FILE_KIND.get(path.name) or ('training_window' if schema.TRAINING_FILE.match(path.name) else None)
        files.append({'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'bytes': path.stat().st_size, 'kind': kind})
    manifest = {'schema': 'cp17-manifest-v1',
                'product': {'name': 'delu-day-ahead-policy', 'designated_policy': 'A1',
                            'evidence_label': 'reference bundle for the freeze origin; development data; not a result'},
                'policy': {'policy_identity_sha256': digest(identity),
                           'package_sha256': identity['code']['package_sha256'],
                           'issuing_host': 'Pyodide 0.28.0 in Node.js v26.7.0'},
                'artifact': {'artifact_sha256': artifact['artifact_sha256'], 'delivery_day': day,
                             'issued_policy': artifact['issued_policy'], 'display': artifact['display']},
                'times': artifact['times'], 'freshness': {'state': 'fresh', 'as_of_day': day, 'latest_issued_day': day},
                'files': files}
    _dump(out / 'manifest.json', manifest)
    return manifest


def read(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(folder.iterdir()) if p.is_file() and p.name != '.DS_Store'}


def tamper_controls(files: dict[str, bytes]) -> dict:
    """Negative controls: each must fail validation."""
    out = {}
    changed = dict(files)
    raw = bytearray(changed['issued-forecasts.json'])
    i = raw.index(b'"quantiles":[[') + 15
    raw[i] = ord('9') if raw[i] != ord('9') else ord('8')
    changed['issued-forecasts.json'] = bytes(raw)
    out['issued_forecasts_byte_changed'] = schema.verify_bundle(changed)
    extra = dict(files)
    extra['unlisted.json'] = b'{}'
    out['unlisted_file_served'] = schema.verify_bundle(extra)
    missing = {k: v for k, v in files.items() if k != 'outcomes.json'}
    out['listed_file_missing'] = schema.verify_bundle(missing)
    npz = [k for k in files if k.endswith('.npz') and k != 'artifact-arrays.npz'][0]
    flipped = dict(files)
    blob = bytearray(flipped[npz])
    blob[len(blob) // 2] ^= 0x01
    flipped[npz] = bytes(blob)
    out['training_window_byte_flipped'] = schema.verify_bundle(flipped)
    # decision 13: a partial record that does not name exactly the missing hours, with the manifest re-hashed so
    # that only the partial check can refuse it
    forecasts = json.loads(files['issued-forecasts.json'])
    forecasts['days'][0]['partial'] = {'issued_hours': 23, 'canonical_hours': 24, 'missing_hours': ['00:00 Europe/Berlin']}
    renamed = dict(files)
    renamed['issued-forecasts.json'] = json.dumps(forecasts, sort_keys=True).encode()
    manifest = json.loads(files['manifest.json'])
    for f in manifest['files']:
        if f['path'] == 'issued-forecasts.json':
            f.update(sha256=hashlib.sha256(renamed['issued-forecasts.json']).hexdigest(), bytes=len(renamed['issued-forecasts.json']))
    renamed['manifest.json'] = json.dumps(manifest, sort_keys=True).encode()
    out['partial_record_wrong'] = schema.verify_bundle(renamed)
    return {k: {'refused': bool(v), 'errors': v[:3]} for k, v in out.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--issuance', required=True, type=Path)
    ap.add_argument('--artifact-dir', required=True, type=Path)
    ap.add_argument('--payloads', required=True, type=Path)
    ap.add_argument('--identity', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--record', required=True, type=Path)
    ap.add_argument('--arrays', type=Path, help="the issued fit's exported arrays (default: CP-17's driver's path)")
    args = ap.parse_args(argv)
    manifest = build(args.root, json.loads(args.issuance.read_text()), args.artifact_dir, args.payloads,
                     json.loads(args.identity.read_text()), args.out, args.arrays)
    files = read(args.out)
    errors = schema.verify_bundle(files)
    controls = tamper_controls(files)
    parts = args.out.resolve().parts
    location = '/'.join(parts[parts.index('.local'):]) if '.local' in parts else str(args.out)
    record = {'schema': 'cp17-reference-bundle-record-v1', 'issued_forecasts_schema': schema.ISSUED_FORECASTS_SCHEMA, 'location': location + '/',
              'manifest': manifest, 'manifest_sha256': hashlib.sha256(files['manifest.json']).hexdigest(),
              'validation_errors': errors, 'valid': not errors, 'negative_controls': controls,
              'all_controls_refused': all(c['refused'] for c in controls.values())}
    args.record.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print('valid' if not errors else errors[:5], 'controls refused:', record['all_controls_refused'])
    return 0 if not errors and record['all_controls_refused'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
