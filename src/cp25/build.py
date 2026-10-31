"""Build every CP-25 output under `reports/development-inputs/` (capstone v21-r13 §24.6 "Outputs").

Deterministic given the preserved inputs and the CP-25 extraction directory: the manifest records
the SHA-256 of every output, every decision-bearing input and every implementation file.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import pandas as pd

from . import calendar as C
from . import features as F
from . import inventory as I
from . import tiers as T

OUT = 'reports/development-inputs'
IMPLEMENTATION = ('src/cp25/__init__.py', 'src/cp25/guard.py', 'src/cp25/ledger.py', 'src/cp25/runs.py', 'src/cp25/extract.py',
                  'src/cp25/features.py', 'src/cp25/inventory.py', 'src/cp25/tiers.py', 'src/cp25/calendar.py', 'src/cp25/build.py',
                  'scripts/cp25_data_gaps.py')
REUSED = ('src/cp20/extract.py', 'src/cp20/gfs.py', 'src/cp20/net.py', 'src/cp20/budget.py', 'src/cp20/weather.py',
          'src/cp15/data.py', 'src/delu_forecast/features.py')
INPUTS = ('data/snapshot.parquet', 'data/partitions.json', 'data/source_manifest.json', 'data/reconciliation_summary.json',
          'reports/weather-ablation/weather-features.parquet', 'reports/weather-ablation/run-manifest.json',
          'reports/weather-admission/dossier.md', 'reports/weather-admission/availability/gfs_ncep_production_status.csv',
          'reports/weather-admission/inventory/gfs_required_runs.csv', 'reports/cp15/protocol.json',
          'reports/v2-causal/input-manifest.json', 'reports/ddnn2/rounds/round-1/search-ledger.json',
          'reports/ddnn2/rounds/round-1/gate.json', 'reports/distribution-challenger/selection.json',
          'capstone_v21.md', 'docs/track-b/evidence/cp-25/issued-brief.md', 'DATA-LICENSE.md')


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _csv(frame: pd.DataFrame, path: Path):
    frame.to_csv(path, index=False, lineterminator='\n')


def _json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + '\n')


def weather(root: Path, weather_dir: Path, classifications: dict | None = None, out: Path | None = None) -> dict:
    out = Path(out) if out else root / OUT
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / OUT / 'gfs-run-manifest.json').read_text())
    cp20 = pd.read_parquet(root / F.CP20_FEATURES)
    feats = F.gap_features(manifest, weather_dir, classifications, list(cp20.columns),
                           {c: cp20[c].dtype for c in ('dswrf_clipped_cells', 'dswrf_quantum', 'dswrf_min_block',
                                                       'dswrf_clipped_magnitude_sum')})
    feats.to_parquet(out / 'weather-features-gaps.parquet', index=False)
    messages, run_rows, checks = F.provenance(manifest, weather_dir)
    messages.to_parquet(out / 'gfs-messages.parquet', index=False)
    _csv(run_rows, out / 'gfs-runs.csv')
    _csv(checks, out / 'gfs-targeted-checks.csv')
    log = F.retrieval_log(weather_dir)
    _csv(log, out / 'retrieval-log.csv')
    # Per-run availability evidence: the exact AWS listings (keys, sizes, LastModified, ETags).
    listing = [json.loads(l) for l in (weather_dir / 'listing.jsonl').read_text().splitlines()]
    (out / 'gfs-aws-listing.jsonl').write_text(''.join(json.dumps(e, sort_keys=True) + '\n'
                                                       for e in sorted(listing, key=lambda e: e['run'])))
    outcomes = pd.DataFrame([json.loads(l) for l in (weather_dir / 'attempt-outcomes.jsonl').read_text().splitlines()])
    attempts = {'outcome_counts': outcomes.outcome.value_counts().sort_index().to_dict(),
                'counted_message_attempts': int(outcomes.counted.sum()),
                'not_success': outcomes[outcomes.outcome != 'success'].drop(columns=['epoch', 'try_id']).to_dict('records')}
    _json(attempts, out / 'gfs-attempt-outcomes.json')
    return {'feature_rows': len(feats), 'delivery_days': int(feats.delivery_date.nunique()),
            'status_counts': feats.status.value_counts().to_dict(), 'runs_complete': int((run_rows.status == 'complete').sum()),
            'messages': len(messages), 'requests': len(log)}


def data_side(root: Path, weather_dir: Path, out: Path | None = None) -> dict:
    out = Path(out) if out else root / OUT
    out.mkdir(parents=True, exist_ok=True)
    cal = C.eligibility_calendar(root)
    _csv(cal, out / 'eligibility-calendar.csv')
    _csv(C.ineligible_hour_reasons(root, cal), out / 'ineligible-hours.csv')
    mask, mask_summary = C.masks(root, cal)
    _csv(mask, out / 'record-masks.csv')
    _json(mask_summary, out / 'record-masks.json')
    spec = C.partition_spec(root, sha_file(out / 'eligibility-calendar.csv'), sha_file(out / 'record-masks.csv'))
    spec['eligibility_summary'] = {'development_days': int(len(cal)), 'eligible_days': int(cal.eligible.sum()),
                                   'excluded_days': int((~cal.eligible).sum()),
                                   'exclusions': cal.loc[~cal.eligible, ['delivery_date', 'exclusion_reason']].astype(str).to_dict('records')}
    _json(spec, out / 'partitions-development.json')
    snap_rows, gaps = I.snapshot_inventory(root)
    _csv(snap_rows, out / 'inventory-by-year.csv')
    _csv(gaps, out / 'inventory-gaps.csv')
    wx = I.weather_inventory(pd.read_parquet(root / F.CP20_FEATURES), pd.read_parquet(out / 'weather-features-gaps.parquet'))
    _csv(wx, out / 'inventory-weather-by-year.csv')
    _json(I.describe(root, snap_rows, wx), out / 'inventory-series.json')
    evidence = T.per_run_evidence(weather_dir / 'listing.jsonl')
    avail = T.availability(root, {k: v for k, v in evidence.items() if v is not None})
    eligible = set(cal.delivery_date[cal.eligible])
    table, per_day = T.margin_table(avail, eligible)
    _csv(table, out / 'margin-table.csv')
    _csv(per_day, out / 'margin-days.csv')
    reg = T.register()
    reg['margin_table_summary'] = table[['group', 'subgroup', 'tier', 'shift_hours', 'affected_days', 'days_considered', 'class',
                                         'class_on_eligible_days']].to_dict('records')
    ev = {k: v for k, v in evidence.items() if v is not None}
    margins = {r: (T.guard.issue_time(date.fromisoformat(r) + timedelta(days=1)) - t).total_seconds() / 3600 for r, t in ev.items()}
    reg['gfs_per_run_evidence'] = {
        'runs': len(evidence), 'runs_with_all_20_listing_timestamps': len(ev),
        'all_strictly_before_issue': bool(ev) and all(m > 0 for m in margins.values()),
        'latest_last_modified_time_of_day_utc': max(t.strftime('%H:%M:%S') for t in ev.values()) if ev else None,
        'min_margin_hours': round(min(margins.values()), 4) if ev else None,
        'basis': "the latest AWS LastModified among each run's 10 target objects and 10 indexes (listing.jsonl)"}
    _json(reg, out / 'tier-register.json')
    return {'calendar': spec['eligibility_summary'], 'masks': mask_summary, 'margin': reg['margin_table_summary'],
            'per_run': reg['gfs_per_run_evidence']}


def manifest(root: Path, extra: dict) -> dict:
    out = root / OUT
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'manifest.json')
    m = {'schema': 'cp25-manifest-v1', 'checkpoint': 'CP-25',
         'anchor': {'file': 'capstone_v21.md', 'revision': 'v21-r13', 'section': '§24.6',
                    'sha256': '173799d818387b9d82db16b9ccd7195afaebbea8e11259809e4a8b704bd525a8',
                    'ratified_at': 'd58486da826b337bd58f469e122315c3142e360b'},
         'outputs_sha256': {f'{OUT}/{p.name}': sha_file(p) for p in files},
         'inputs_sha256': {p: sha_file(root / p) for p in INPUTS},
         'implementation_sha256': {p: sha_file(root / p) for p in IMPLEMENTATION},
         'reused_landed_code_sha256': {p: sha_file(root / p) for p in REUSED},
         **extra}
    _json(m, out / 'manifest.json')
    return m
