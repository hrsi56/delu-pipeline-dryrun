"""Frozen required-run manifest for CP-20 (modelling environment; metadata only).

Required weather = every delivery day in the union, over the 638 frozen CP-16 origins,
of ``[max(2019-01-01, D-728), D]`` (training history plus the target day). Each delivery
day X needs the X-1 00 UTC run; delivery 2019-01-01 would need a pre-2019 run and is
structurally missing. Endpoint choice follows the admission inventory; the eight
2023-03-24..31 runs that admission did not inventory use AWS first.
"""
from __future__ import annotations

import csv
from datetime import date, timedelta
import json
from pathlib import Path

from cp15.data import history_start

ADDED = [date(2023, 3, 24) + timedelta(days=i) for i in range(8)]
# Runs whose raw target messages are retained locally for independent reconstruction.
RETAIN = ['2019-01-01', '2019-03-30', '2019-06-13', '2019-10-26', '2020-10-24', '2021-02-02', '2021-03-22',
          '2021-03-23', '2022-03-26', '2022-08-25', '2024-03-30', '2024-09-15', '2024-10-26', '2026-03-28',
          '2026-04-06', *[d.isoformat() for d in ADDED]]
# Representative subset proven first (restart test), all reused by the full extraction.
SUBSET = ['2019-01-01', '2019-03-30', '2019-06-13', '2019-10-26', '2020-10-24', '2021-02-02', '2021-01-04',
          '2021-03-22', '2021-03-23', '2022-08-25', '2024-09-15', '2026-04-06', *[d.isoformat() for d in ADDED]]


def version(run):
    return 'v14' if run < date(2019, 6, 13) else ('v15.1' if run < date(2021, 3, 23) else 'v16')


def availability_basis(run, captures):
    """Admission's accepted basis (dossier section 6): dated NCEP captures bracketing the run,
    plus the captures dated inside the run's model version. Reconstructed, not per-day proof."""
    before = [c for c in captures if c['date'] <= run]
    after = [c for c in captures if c['date'] >= run]
    same = [c for c in captures if c['version'] == version(run)]
    return {'basis': 'reconstructed_availability_ncep_48h_products_30day_average',
            'capture_before': before[-1]['date'].isoformat() if before else None,
            'capture_after': after[0]['date'].isoformat() if after else None,
            'bracketed': bool(before and after),
            'captures_dated_in_version': [c['date'].isoformat() for c in same],
            'latest_48h_end_utc_in_version': max(c['end'] for c in same) if same else None,
            'origin_cutoff_utc': '11:00 on D-1 (run day)',
            'limits': 'NCEP completion averages; public dissemination lag assumed, not evidenced; no per-day delivery guarantee'}


def capture_table(root):
    rows = list(csv.DictReader(open(root / 'reports/weather-admission/availability/gfs_ncep_production_status.csv')))
    out = []
    for r in rows:
        if not r['gfs_00z_48h_products_avg_end_utc']:
            continue
        d = date.fromisoformat(r['capture_utc_date'])
        # 30-day running averages: straddling captures are attributed to the later version.
        out.append({'date': d, 'end': r['gfs_00z_48h_products_avg_end_utc'], 'version': version(d),
                    'source': r['source'], 'sha256': r['sha256']})
    return sorted(out, key=lambda c: c['date'])


def build(root: Path, admission_cache: Path) -> dict:
    manifest = json.loads((root / 'reports/v2-causal/input-manifest.json').read_text())
    origins = [date.fromisoformat(o['day']) for f in manifest['folds'] for o in f['origins']]
    if len(origins) != 638:
        raise ValueError('CP-16 origin manifest must contain 638 fold/date origins')
    needed = {}
    for f in manifest['folds']:
        for o in f['origins']:
            d = date.fromisoformat(o['day'])
            start = history_start(d, 'A1')
            if str(start) != o['history_start']:
                raise ValueError('history window disagrees with the frozen CP-16 manifest')
            x = start
            while x <= d:
                needed.setdefault(x, set()).add(f['fold'])
                x += timedelta(days=1)
    inventory = {r['run_00z']: r for r in csv.DictReader(open(root / 'reports/weather-admission/inventory/gfs_required_runs.csv'))}
    tail = {}
    for line in (admission_cache / 'ncar_tail.jsonl').read_text().splitlines():
        r = json.loads(line)
        tail[r['key']] = r['bytes']
    aws = {}
    for line in (admission_cache / 'aws_days.jsonl').read_text().splitlines():
        r = json.loads(line)
        aws[r['key']] = r['files']
    captures = capture_table(root)
    runs, structural = [], []
    for delivery in sorted(needed):
        run = delivery - timedelta(days=1)
        if run < date(2019, 1, 1):
            structural.append({'delivery_day': delivery.isoformat(), 'run_00z': run.isoformat(),
                               'class': 'structural_missing_pre_2019_run', 'folds': sorted(needed[delivery])})
            continue
        key, ymd = run.isoformat(), run.strftime('%Y%m%d')
        inv = inventory.get(key)
        if inv is None and run not in ADDED:
            raise ValueError(f'required run {key} neither inventoried nor one of the eight added runs')
        primary = inv['chosen_endpoint'].lower() if inv else 'aws'
        alternate = None
        if primary == 'aws':
            alternate = 'ncar'
        elif run >= date(2021, 1, 1):
            alternate = 'aws'
        sizes = {}
        for lead in (21, 24, 27, 30, 33, 36, 39, 42, 45, 48):
            sizes[f'f{lead:03d}'] = {'ncar': tail.get(f'{ymd}_f{lead:03d}'), 'aws': aws.get(ymd, {}).get(f'f{lead:03d}')}
        runs.append({'run_00z': key, 'delivery_day': delivery.isoformat(), 'version': version(run),
                     'folds': sorted(needed[delivery]), 'primary_endpoint': primary, 'alternate_endpoint': alternate,
                     'admission': ({'inventoried': True, 'field_basis': inv['field_basis'],
                                    'ncar_complete': inv['ncar_complete'], 'aws_complete': inv['aws_complete']}
                                   if inv else {'inventoried': False, 'added_run': True,
                                                'requires': 'targeted field/lead/availability check before fitting'}),
                     'availability': availability_basis(run, captures), 'object_bytes': sizes,
                     'retain_raw': key in RETAIN, 'subset': key in SUBSET})
    if len(runs) != 2476 or len(structural) != 1:
        raise ValueError(f'expected 2,476 runs and one structural day, got {len(runs)} / {len(structural)}')
    missing_basis = [r['run_00z'] for r in runs if not r['availability']['bracketed']]
    return {'schema': 'cp20-run-manifest-v1', 'required_runs': len(runs), 'target_messages': 50 * len(runs),
            'fields': ['u10', 'v10', 'u100', 'v100', 'dswrf'], 'leads': [21, 24, 27, 30, 33, 36, 39, 42, 45, 48],
            'rule': 'union over 638 CP-16 origins of [max(2019-01-01,D-728),D]; run = delivery-1 00 UTC',
            'endpoint_rule': 'admission-inventoried endpoint; added runs AWS first; at most one alternate-endpoint attempt per failed object',
            'structural_missing': structural, 'availability_not_bracketed': missing_basis,
            'added_runs': [d.isoformat() for d in ADDED], 'retain_raw': RETAIN, 'subset': SUBSET,
            'captures': [{**c, 'date': c['date'].isoformat()} for c in captures], 'runs': runs}
