"""Assemble durable extraction evidence and the frozen weather features (modelling env).

Reads only the local extraction outputs, the committed run manifest and read-only
admission evidence. Refuses unless every required run is complete or classified as a
confirmed section 15.3 missing class (unfinished extraction is never imputed).
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .budget import atomic
from .weather import COLUMNS, WeatherDesign, build_features

ADMISSION_NAMES = {'u10': 'u10', 'v10': 'v10', 'u100': 'u_hub', 'v100': 'v_hub', 'dswrf': 'ssrd'}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def messages_table(weather: Path, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, runs = [], []
    for r in manifest['runs']:
        path = weather / 'runs' / f'{r["run_00z"]}.json'
        if not path.exists():
            runs.append({'run_00z': r['run_00z'], 'status': 'not_extracted'})
            continue
        rec = json.loads(path.read_text())
        endpoints = Counter(m['endpoint'] for m in rec['messages'])
        runs.append({'run_00z': r['run_00z'], 'delivery_day': r['delivery_day'], 'version': rec['version'],
                     'status': rec['status'], 'primary_endpoint': r['primary_endpoint'],
                     'endpoints_used': json.dumps(dict(endpoints)), 'messages': len(rec['messages']),
                     'failed_attempts': len(rec['failed_attempts']), 'nonfinite_box_values': rec['nonfinite_box_values'],
                     'elapsed_s': rec['elapsed_s'], 'npz_sha256': rec['npz_sha256'], 'record_sha256': sha(path),
                     'raw_retained': bool(r['retain_raw'])})
        for m in rec['messages']:
            meta = m['meta']
            rows.append({'run_00z': r['run_00z'], 'lead': m['lead'], 'field': m['field'], 'endpoint': m['endpoint'],
                         'url': m['url'], 'byte_start': m['byte_range'][0], 'byte_end': m['byte_range'][1],
                         'bytes': m['bytes'], 'sha256': m['sha256'], 'retrieved_utc': m['retrieved_utc'],
                         'purpose': m['purpose'], 'idx_sha256': m.get('idx_sha256'), 'idx_line': m.get('idx_line'),
                         'window_bytes_reused': m.get('window_bytes_reused'), 'raw_retained': m.get('raw_retained', False),
                         'version': rec['version'], 'packing_quantum': m['packing_quantum'],
                         **{k: meta.get(k) for k in ('shortName', 'units', 'typeOfLevel', 'level', 'stepType', 'startStep',
                                                      'endStep', 'dataDate', 'dataTime', 'validityDate', 'validityTime',
                                                      'packingType', 'bitsPerValue', 'binaryScaleFactor', 'decimalScaleFactor',
                                                      'numberOfMissing', 'box_nonfinite', 'parameterNumber',
                                                      'typeOfFirstFixedSurface', 'typeOfFirstFixedSurface_native',
                                                      'productionStatusOfProcessedData', 'typeOfProcessedData',
                                                      'generatingProcessIdentifier', 'centre')}})
    return pd.DataFrame(rows), pd.DataFrame(runs)


def sample_hashes(messages: pd.DataFrame, admission_root: Path) -> pd.DataFrame:
    """Byte identity of CP-20 target messages with the admission's retained raw samples."""
    listed = {}
    for line in (admission_root / 'hashes/raw_sample_objects.sha256').read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        name = Path(path).name
        if name.endswith('.grib2') and name.count('_') >= 3:
            listed[name] = digest
    out = []
    for m in messages.itertuples():
        name = f'{m.run_00z}_f{m.lead:03d}_{ADMISSION_NAMES[m.field]}_{m.endpoint}.grib2'
        if name in listed:
            out.append({'run_00z': m.run_00z, 'lead': m.lead, 'field': m.field, 'endpoint': m.endpoint,
                        'admission_object': name, 'admission_sha256': listed[name], 'cp20_sha256': m.sha256,
                        'identical': listed[name] == m.sha256})
    return pd.DataFrame(out)


def added_checks(weather: Path, manifest: dict, messages: pd.DataFrame) -> pd.DataFrame:
    inv = {json.loads(l)['run']: json.loads(l) for l in (weather / 'added_inventory.jsonl').read_text().splitlines()}
    out = []
    for r in manifest['runs']:
        if r['admission']['inventoried']:
            continue
        i = inv[r['run_00z']]
        m = messages.loc[messages.run_00z.eq(r['run_00z'])]
        out.append({'run_00z': r['run_00z'], 'delivery_day': r['delivery_day'], 'version': r['version'],
                    'aws_leads_listed': len(i['aws_files']), 'aws_idx_listed': len(i['aws_idx']),
                    'aws_complete': i['aws_complete'], 'ncar_leads_in_catalog': len(i['ncar_files']),
                    'ncar_complete': i['ncar_complete'], 'decoded_messages': len(m),
                    'decoded_leads': m.lead.nunique(), 'decoded_fields': m.field.nunique(),
                    'endpoint_used': ','.join(sorted(m.endpoint.unique())) if len(m) else None,
                    'metadata_validated': len(m) == 50, 'availability_basis': r['availability']['basis'],
                    'capture_before': r['availability']['capture_before'], 'capture_after': r['availability']['capture_after'],
                    'field_lead_availability_check': bool(i['aws_complete'] and len(m) == 50 and r['availability']['bracketed'])})
    return pd.DataFrame(out)


def run(root: Path, weather: Path, classifications: dict | None = None) -> dict:
    root, weather = Path(root), Path(weather)
    out = root / 'reports/weather-ablation'
    manifest = json.loads((out / 'run-manifest.json').read_text())
    messages, runs = messages_table(weather, manifest)
    unfinished = runs.loc[runs.status.ne('complete') & ~runs.run_00z.isin(list(classifications or {})), 'run_00z'].tolist()
    if unfinished:
        raise RuntimeError(f'{len(unfinished)} required runs unfinished; BLOCKED, not imputable: {unfinished[:10]}')
    features = build_features(manifest, weather, classifications)
    design = WeatherDesign.from_features(features)
    messages.to_parquet(out / 'messages.parquet', index=False)
    runs.to_csv(out / 'runs.csv', index=False)
    features.to_parquet(out / 'weather-features.parquet', index=False)
    missing = features.loc[features.status.ne('ok')]
    missing.groupby(['delivery_date', 'run_00z', 'status'], dropna=False).size().rename('hours').reset_index() \
        .to_csv(out / 'missingness.csv', index=False)
    samples = sample_hashes(messages, root / 'reports/weather-admission')
    samples.to_csv(out / 'sample-hash-comparison.csv', index=False)
    added = added_checks(weather, manifest, messages)
    added.to_csv(out / 'extended-inventory.csv', index=False)
    clip = features.groupby('version').agg(hours=('status', 'size'), clipped_cells=('dswrf_clipped_cells', 'sum'),
                                           clipped_magnitude=('dswrf_clipped_magnitude_sum', 'sum'),
                                           min_block=('dswrf_min_block', 'min'), max_quantum=('dswrf_quantum', 'max'),
                                           min_quantum=('dswrf_quantum', 'min')).reset_index()
    clip.to_csv(out / 'radiation-clipping.csv', index=False)
    summary = {'required_runs': len(manifest['runs']), 'complete_runs': int(runs.status.eq('complete').sum()),
               'classified_missing_runs': sorted(classifications or {}), 'target_messages': int(len(messages)),
               'messages_by_endpoint': messages.endpoint.value_counts().to_dict(),
               'messages_by_purpose': messages.purpose.value_counts().to_dict(),
               'payload_bytes': int(messages['bytes'].sum()),
               'nonfinite_box_values': int(runs.nonfinite_box_values.fillna(0).sum()),
               'feature_hours': int(len(features)), 'feature_status': features.status.value_counts().to_dict(),
               'structural_missing_days': [str(d) for d in sorted(features.loc[features.status.eq('structural_missing_pre_2019_run'), 'delivery_date'].unique())],
               'weather_design_sha256': design.sha256, 'local_hour_cells': int(len(design.table)),
               'local_hour_cells_missing': int(design.table[list(COLUMNS)].isna().any(axis=1).sum()),
               'repeated_local_hours': int((design.table.n_canonical == 2).sum()),
               'sample_hash_comparisons': int(len(samples)), 'sample_hash_identical': int(samples.identical.sum()) if len(samples) else 0,
               'added_runs_checked': int(len(added)), 'added_runs_passed': int(added.field_lead_availability_check.sum()) if len(added) else 0,
               'feature_ranges': {c: [float(features[c].min()), float(features[c].max())] for c in COLUMNS},
               'files_sha256': {n: sha(out / n) for n in ('messages.parquet', 'runs.csv', 'weather-features.parquet',
                                                          'missingness.csv', 'sample-hash-comparison.csv',
                                                          'extended-inventory.csv', 'radiation-clipping.csv')}}
    atomic(out / 'extraction-summary.json', summary)
    return summary
