"""Weather features for the gaps and their provenance (capstone v21-r13 §24.6 item 3; §15.2–15.3).

The conversion is CP-20's frozen recipe, imported unchanged (`cp20.weather`): per-cell linear
interpolation of u/v to canonical hour starts h22..h46, speed before averaging, DSWRF three-hour
de-averaging with the quantum-bounded clipping rule, cos-latitude weights over the fixed 34 x 41
box, and §15.3's missing rule. The output has exactly the columns of CP-20's committed
`reports/weather-ablation/weather-features.parquet`, so the two files concatenate into one frozen
weather record for every development delivery day.

**Conversion identity control.** The same import path re-converts every one of CP-20's retained
grids (read in place, read-only) and must reproduce CP-20's committed features exactly.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cp20 import weather as W

from . import guard

CP20_FEATURES = 'reports/weather-ablation/weather-features.parquet'
CP20_GRIDS = Path('/Users/djourno/Downloads/PJM/.local/artifacts/cp-20/weather')


def _runs(manifest: dict, weather_dir: Path) -> list[dict]:
    out = []
    for r in manifest['runs']:
        guard.admit_delivery_day(date.fromisoformat(r['delivery_day']))
        out.append({'run_00z': r['run_00z'], 'delivery_day': r['delivery_day'], 'version': r['version']})
    return out


def gap_features(manifest: dict, weather_dir: Path, classifications: dict | None = None,
                 columns: list[str] | None = None, dtypes: dict | None = None) -> pd.DataFrame:
    """CP-20's `build_features` on CP-25's runs; unfinished extraction refuses (never imputed)."""
    frame = W.build_features({'structural_missing': [], 'runs': _runs(manifest, weather_dir)}, Path(weather_dir),
                             classifications or {})
    guard.admit_rows(pd.DatetimeIndex(frame.timestamp_utc).to_pydatetime())
    if 'reason' not in frame.columns:
        frame['reason'] = pd.Series([None] * len(frame), dtype='str')
    frame['timestamp_utc'] = frame.timestamp_utc.astype('datetime64[ms, UTC]')   # CP-20's stored unit
    if columns is not None:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f'gap features lack CP-20 columns {sorted(missing)}')
        frame = frame[columns]
    for col, dtype in (dtypes or {}).items():   # CP-20's stored dtypes, so the two files concatenate exactly
        if str(frame[col].dtype) != str(dtype):
            frame[col] = frame[col].astype(dtype)
    return frame


def conversion_identity(root: Path, grids: Path = CP20_GRIDS) -> dict:
    """Re-convert every CP-20 retained grid through the same import path; compare with CP-20's file."""
    committed = pd.read_parquet(Path(root) / CP20_FEATURES)
    manifest = json.loads((Path(root) / 'reports/weather-ablation/run-manifest.json').read_text())
    rebuilt = W.build_features(manifest, grids, {})
    cols = list(committed.columns)
    rebuilt = rebuilt[cols].sort_values('timestamp_utc').reset_index(drop=True)
    committed = committed.sort_values('timestamp_utc').reset_index(drop=True)
    numeric = ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean']
    # The instants are compared, not their storage unit (CP-20's file holds ms; a fresh build holds s).
    same_keys = bool(len(rebuilt) == len(committed)) and bool(
        (rebuilt.timestamp_utc.astype('datetime64[ns, UTC]').to_numpy() == committed.timestamp_utc.astype('datetime64[ns, UTC]').to_numpy()).all()
        and (rebuilt.delivery_date.astype(str) == committed.delivery_date.astype(str)).all()
        and (rebuilt.run_00z.fillna('').astype(str) == committed.run_00z.fillna('').astype(str)).all())
    a, b = rebuilt[numeric].to_numpy(float), committed[numeric].to_numpy(float)
    bitwise = bool(np.array_equal(a, b, equal_nan=True))
    status = bool((rebuilt.status.astype(str) == committed.status.astype(str)).all())
    return {'rows': int(len(committed)), 'runs': len(manifest['runs']), 'keys_identical': bool(same_keys),
            'values_bitwise_identical': bitwise, 'status_identical': status,
            'max_abs_difference': float(np.nanmax(np.abs(a - b))) if a.size else 0.0,
            'committed_sha256': hashlib.sha256((Path(root) / CP20_FEATURES).read_bytes()).hexdigest()}


def provenance(manifest: dict, weather_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-message provenance, per-run status and the targeted checks for the CP-25 runs."""
    weather_dir = Path(weather_dir)
    listing = {}
    path = weather_dir / 'listing.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            e = json.loads(line)
            listing[e['run']] = e
    messages, run_rows, checks = [], [], []
    for r in manifest['runs']:
        key = r['run_00z']
        rec_path = weather_dir / 'runs' / f'{key}.json'
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
        lst = listing.get(key)
        stamps = []
        if lst:
            for v in lst['leads'].values():
                for k in ('object', 'index'):
                    if v.get(k):
                        stamps.append(v[k]['last_modified'])
        issue = guard.issue_time(date.fromisoformat(r['delivery_day']))
        last_mod = max(stamps) if len(stamps) == 20 else None
        lm_dt = (pd.Timestamp(last_mod).to_pydatetime() if last_mod else None)
        if rec is not None:
            endpoints = {}
            for m in rec['messages']:
                endpoints[m['endpoint']] = endpoints.get(m['endpoint'], 0) + 1
                meta = m['meta']
                messages.append({'run_00z': key, 'delivery_day': r['delivery_day'], 'lead': m['lead'], 'field': m['field'],
                                 'endpoint': m['endpoint'], 'url': m['url'], 'byte_start': m['byte_range'][0],
                                 'byte_end': m['byte_range'][1], 'bytes': m['bytes'], 'sha256': m['sha256'],
                                 'packing_quantum': m['packing_quantum'], 'idx_line': m.get('idx_line'),
                                 'retrieved_utc': m['retrieved_utc'], 'purpose': m['purpose'],
                                 'raw_retained': bool(m.get('raw_retained', False)),
                                 'shortName': meta.get('shortName'), 'units': meta.get('units'),
                                 'typeOfLevel': meta.get('typeOfLevel'), 'level': meta.get('level'),
                                 'stepType': meta.get('stepType'), 'startStep': meta.get('startStep'),
                                 'endStep': meta.get('endStep'), 'dataDate': meta.get('dataDate'),
                                 'dataTime': meta.get('dataTime'), 'validityDate': meta.get('validityDate'),
                                 'validityTime': meta.get('validityTime'), 'productionStatusOfProcessedData':
                                 meta.get('productionStatusOfProcessedData'), 'typeOfProcessedData': meta.get('typeOfProcessedData'),
                                 'box_nonfinite': meta.get('box_nonfinite')})
            run_rows.append({'run_00z': key, 'delivery_day': r['delivery_day'], 'version': rec['version'], 'status': rec['status'],
                             'endpoints_used': json.dumps(endpoints, sort_keys=True), 'messages': len(rec['messages']),
                             'failed_attempts': len(rec['failed_attempts']), 'nonfinite_box_values': rec['nonfinite_box_values'],
                             'elapsed_s': rec['elapsed_s'], 'npz_sha256': rec['npz_sha256'],
                             'record_sha256': hashlib.sha256(rec_path.read_bytes()).hexdigest(),
                             'raw_retained': any(m.get('raw_retained') for m in rec['messages'])})
        else:
            run_rows.append({'run_00z': key, 'delivery_day': r['delivery_day'], 'version': r['version'], 'status': 'not_complete'})
        fields_ok = rec is not None and len(rec['messages']) == 50 and \
            sorted({(m['lead'], m['field']) for m in rec['messages']}) == sorted((l, f) for l in manifest['leads'] for f in manifest['fields'])
        a = r['availability']
        checks.append({'run_00z': key, 'delivery_day': r['delivery_day'], 'version': r['version'],
                       'inventoried_by_admission': False,
                       'aws_leads_listed': sum(1 for v in (lst or {}).get('leads', {}).values() if v.get('object')),
                       'aws_idx_listed': sum(1 for v in (lst or {}).get('leads', {}).values() if v.get('index')),
                       'aws_complete': bool(lst and lst['aws_complete']),
                       'decoded_messages': len(rec['messages']) if rec else 0,
                       'decoded_leads': len({m['lead'] for m in rec['messages']}) if rec else 0,
                       'decoded_fields': len({m['field'] for m in rec['messages']}) if rec else 0,
                       'endpoint_used': ','.join(sorted({m['endpoint'] for m in rec['messages']})) if rec else '',
                       'metadata_validated': bool(fields_ok),
                       'availability_basis': 'reconstructed: NCEP 48h-products completion, bracketing dated captures (dossier §6)',
                       'capture_before': a.get('capture_before'), 'capture_after': a.get('capture_after'),
                       'reconstructed_available_utc': a.get('available_utc'),
                       'issue_utc': issue.strftime('%Y-%m-%dT%H:%M:%SZ'),
                       'reconstructed_margin_hours': a.get('margin_hours'),
                       'aws_last_modified_max_utc': last_mod,
                       'per_run_margin_hours': round((issue - lm_dt).total_seconds() / 3600, 4) if lm_dt else None,
                       'per_run_strictly_before_issue': (lm_dt < issue) if lm_dt else None,
                       'field_lead_availability_check': bool(fields_ok and lst and lst['aws_complete'] and a.get('strictly_before_issue'))})
    return pd.DataFrame(messages), pd.DataFrame(run_rows), pd.DataFrame(checks)


def retrieval_log(weather_dir: Path) -> pd.DataFrame:
    """Every metered request, as logged before any byte was kept (URLs without query strings)."""
    rows = [json.loads(l) for l in (Path(weather_dir) / 'requests.jsonl').read_text().splitlines()]
    frame = pd.DataFrame(rows)
    frame['range'] = frame['range'].map(lambda x: f'{x[0]}-{x[1]}' if isinstance(x, list) else '')
    cols = ['t_utc', 'endpoint', 'url', 'range', 'status', 'body_bytes', 'wire_bytes_estimate', 'charged_bytes', 'elapsed_s',
            'purpose', 'request_try', 'error_kind', 'error']
    return frame[cols]
