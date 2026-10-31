"""The frozen CP-25 GFS run manifest (capstone v21-r13 §24.6 item 3). Standard library only.

GFS 0.25° D−1 00 UTC, with §15.2's source, recipe, fields and leads, for delivery days
2022-09-29..2023-03-24 (runs 2022-09-28..2023-03-23, 177) and 2026-04-08..2026-09-06 (runs
2026-04-07..2026-09-05, 152): 329 runs. None lies in the admission dossier's inventory
(`reports/weather-admission/inventory/gfs_required_runs.csv`, runs 2019-01-01..2026-04-06 outside
the first gap), so every run takes §15.2's targeted field/lead/availability checks and is not
labelled already admitted. Every run is v16 (from 2021-03-23), which the AWS bucket holds, so AWS
is the primary endpoint (§15.2: "for added runs use AWS first") and NCAR d084001 the one
same-product alternate. Each run and lead passes the boundary guard before it enters the manifest.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

from . import guard

GAPS = ((date(2022, 9, 29), date(2023, 3, 24)), (date(2026, 4, 8), date(2026, 9, 6)))
EXPECTED_RUNS = (177, 152)
LEADS = (21, 24, 27, 30, 33, 36, 39, 42, 45, 48)
FIELDS = ('u10', 'v10', 'u100', 'v100', 'dswrf')
V16_FIRST = date(2021, 3, 23)
# Raw target messages retained locally (never committed) for independent reconstruction: each gap's
# first and last run, the fall-back delivery day 2022-10-30, the year end, mid-summer 2026 and the
# boundary run whose f048 serves delivery day 2026-09-06.
RETAIN = ('2022-09-28', '2022-10-29', '2022-12-31', '2023-03-23', '2026-04-07', '2026-06-20', '2026-09-05')
CAPTURES = 'reports/weather-admission/availability/gfs_ncep_production_status.csv'
INVENTORY = 'reports/weather-admission/inventory/gfs_required_runs.csv'


def version(run: date) -> str:
    return 'v14' if run < date(2019, 6, 13) else ('v15.1' if run < V16_FIRST else 'v16')


def delivery_days():
    for first, last in GAPS:
        d = first
        while d <= last:
            yield d
            d += timedelta(days=1)


def capture_table(root: Path) -> list[dict]:
    """The dossier's dated NCEP production-status captures (30-day running averages)."""
    rows = list(csv.DictReader(open(Path(root) / CAPTURES)))
    out = []
    for r in rows:
        if not r['gfs_00z_48h_products_avg_end_utc']:
            continue
        out.append({'date': date.fromisoformat(r['capture_utc_date']), 'end_48h': r['gfs_00z_48h_products_avg_end_utc'],
                    'forecast_end': r['forecast_avg_end_utc'], 'source': r['source'], 'sha256': r['sha256']})
    return sorted(out, key=lambda c: c['date'])


def _hms(s: str) -> timedelta:
    h, m, sec = (int(x) for x in s.split(':'))
    return timedelta(hours=h, minutes=m, seconds=sec)


def reconstructed_availability(run: date, captures: list[dict]) -> dict:
    """Tier-2 availability from the bracketing dated captures: the later of the two 48-hour-products
    completion averages on the run date (conservative). Reconstructed, not a per-day delivery proof."""
    before = [c for c in captures if c['date'] <= run]
    after = [c for c in captures if c['date'] >= run]
    if not before or not after:
        return {'bracketed': False}
    b, a = before[-1], after[0]
    worst = max(_hms(b['end_48h']), _hms(a['end_48h']))
    at = datetime.combine(run, datetime.min.time(), tzinfo=timezone.utc) + worst
    return {'bracketed': True, 'capture_before': b['date'].isoformat(), 'capture_after': a['date'].isoformat(),
            'end_48h_before': b['end_48h'], 'end_48h_after': a['end_48h'],
            'available_utc': at.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'basis': 'reconstructed: later of the bracketing NCEP 00 UTC 48-hour-products 30-day average completions'}


def build(root: Path) -> dict:
    root = Path(root)
    inventoried = {r['run_00z'] for r in csv.DictReader(open(root / INVENTORY))}
    captures = capture_table(root)
    runs = []
    for delivery in delivery_days():
        guard.admit_delivery_day(delivery)
        run = delivery - timedelta(days=1)
        for lead in LEADS:
            guard.admit_gfs(guard.run_init(run), lead)
        key = run.isoformat()
        if key in inventoried:
            raise ValueError(f'{key} is in the admission inventory; CP-25 runs are outside it by construction')
        if version(run) != 'v16':
            raise ValueError(f'{key}: expected a v16 run')
        issue = guard.issue_time(delivery)
        avail = reconstructed_availability(run, captures)
        if avail['bracketed']:
            at = datetime.strptime(avail['available_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
            avail['issue_utc'] = issue.strftime('%Y-%m-%dT%H:%M:%SZ')
            avail['strictly_before_issue'] = guard.strictly_before(at, delivery)
            avail['margin_hours'] = round((issue - at).total_seconds() / 3600, 4)
        runs.append({'run_00z': key, 'delivery_day': delivery.isoformat(), 'version': 'v16',
                     'primary_endpoint': 'aws', 'alternate_endpoint': 'ncar',
                     'admission': {'inventoried': False, 'added_run': True,
                                   'requires': "§15.2's targeted field/lead/availability checks"},
                     'availability': avail,
                     'object_bytes': {f'f{l:03d}': {'aws': None, 'ncar': None} for l in LEADS},
                     'retain_raw': key in RETAIN, 'subset': False})
    counts = tuple(sum(1 for r in runs if first <= date.fromisoformat(r['delivery_day']) <= last) for first, last in GAPS)
    if counts != EXPECTED_RUNS:
        raise ValueError(f'expected {EXPECTED_RUNS} runs per gap, got {counts}')
    unbracketed = [r['run_00z'] for r in runs if not r['availability']['bracketed']]
    return {'schema': 'cp25-run-manifest-v1', 'anchor': 'capstone_v21.md v21-r13 §24.6 item 3',
            'required_runs': len(runs), 'runs_per_gap': list(counts), 'target_messages': 50 * len(runs),
            'fields': list(FIELDS), 'leads': list(LEADS),
            'gaps_delivery_days': [[a.isoformat(), b.isoformat()] for a, b in GAPS],
            'rule': 'run = delivery day - 1, 00 UTC; leads f021..f048 every 3 h',
            'endpoint_rule': ('AWS first (all runs are v16 and post-2021); per failed (run, lead) object at most '
                              'one same-product NCAR d084001 alternate attempt (CP-20 extract.py unchanged)'),
            'structural_missing': [], 'availability_not_bracketed': unbracketed, 'retain_raw': list(RETAIN),
            'captures': [{**c, 'date': c['date'].isoformat()} for c in captures],
            'inputs_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in (CAPTURES, INVENTORY)},
            'runs': runs}


def write(root: Path, out: Path) -> str:
    manifest = build(root)
    text = json.dumps(manifest, indent=1, sort_keys=True) + '\n'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()
