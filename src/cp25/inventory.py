"""The inventory of every input series (capstone v21-r13 §24.6 item 2).

For every input series of B0, v1-R and v2–v5, and for the snapshot's other columns (which no
policy uses): its source, product, licence, granularity and timezone, and its gaps by year, each
with a cause — the source, or our extraction. Prices and load are not re-pulled (item 1): they
come from the preserved bytes of `data/snapshot.parquet`, read behind the boundary guard.

Cause rules:

* **Snapshot series.** The snapshot keeps, for each hour, how many of the source's quarter-hour
  (or price-interval) values were non-null. A null hour with zero source values is a **source**
  gap: SMARD returned no value. A null hour with one to three is **our extraction**: the pull's
  fail-closed rule turned a partial hour into null (`data/source_manifest.json`,
  `source_bin_integrity`).
* **Weather.** A delivery day without a weather record is **our extraction** when its run was
  never retrieved (CP-20's scope; CP-25 now fills both gaps), **structural** when its run would
  precede the 2019 input floor (delivery 2019-01-01, §15.3), and **source** only for a confirmed
  archive absence after the bounded attempts.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import guard
from .calendar import DEV_FIRST, DEV_LAST, days, expected_hours

SMARD_LICENCE = 'CC BY 4.0; attribution "Data: ENTSO-E Transparency Platform; Bundesnetzagentur | SMARD.de" (DATA-LICENSE.md)'
GFS_LICENCE = ('NOAA open data (NODD) and NCAR GDEX d084001 "Freely Available" (doi:10.5065/D65D8PWK); derived aggregates carry '
               'the CP-20 weather attribution; no raw GRIB or decoded grid committed (DATA-LICENSE.md, rights-notice.md)')
SNAPSHOT_TZ = 'timestamp_utc in UTC (canonical hours); delivery calendar Europe/Berlin (23/24/25 hours); SMARD clock CET/CEST'

SERIES = [
    # column, quarter-count column, product, source, role, policies
    ('price_eur_mwh', 'price_source_interval_count', 'DE-LU day-ahead price, SDAC clearing price (ENTSO-E 12.1.D / A44)',
     'SMARD filter 4169 "Day-ahead price" (Bundesnetzagentur | SMARD.de; ENTSO-E Transparency Platform)',
     'target, and price lags / statistics / normalisation / released errors', 'B0, v1-R, v2, v3, v4, v5'),
    ('load_forecast_mw', 'load_forecast_mw_source_quarter_count',
     'DE-LU day-ahead total load forecast (ENTSO-E 6.1.B / A65 A01)',
     'SMARD filter 411 "Forecast consumption: total grid load"', 'input (load forecast of D; LEAR and DDNN-2 also D-1, D-7)',
     'v1-R, v2, v3, v4, v5'),
    ('wind_onshore_forecast_mw', 'wind_onshore_forecast_mw_source_quarter_count', 'A69 day-ahead wind onshore forecast',
     'SMARD filter 123', 'not an input: post-gate A69, refused by §2', 'none'),
    ('solar_forecast_mw', 'solar_forecast_mw_source_quarter_count', 'A69 day-ahead solar forecast', 'SMARD filter 125',
     'not an input: post-gate A69, refused by §2', 'none'),
    ('wind_offshore_forecast_mw', 'wind_offshore_forecast_mw_source_quarter_count', 'A69 day-ahead wind offshore forecast',
     'SMARD filter 3791', 'not an input: post-gate A69, refused by §2', 'none'),
    ('wind_onshore_actual_mw', 'wind_onshore_actual_mw_source_quarter_count', 'A75 actual wind onshore generation',
     'SMARD filter 4067', "not an input of B0, v1-R or v2-v5 (v1's unselected residual-load catalog only)", 'none'),
    ('solar_actual_mw', 'solar_actual_mw_source_quarter_count', 'A75 actual solar generation', 'SMARD filter 4068',
     "not an input of B0, v1-R or v2-v5 (v1's unselected residual-load catalog only)", 'none'),
    ('wind_offshore_actual_mw', 'wind_offshore_actual_mw_source_quarter_count', 'A75 actual wind offshore generation',
     'SMARD filter 1225', "not an input of B0, v1-R or v2-v5 (v1's unselected residual-load catalog only)", 'none'),
]


def snapshot_inventory(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-series, per-year counts of expected, present and null hours, with each null hour's cause."""
    cols = ['timestamp_utc', 'delivery_date'] + [c for s in SERIES for c in s[:2]]
    frame = pd.read_parquet(Path(root) / 'data/snapshot.parquet', columns=cols,
                            filters=[('delivery_date', '>=', DEV_FIRST), ('delivery_date', '<=', DEV_LAST)])
    guard.admit_rows(pd.DatetimeIndex(frame.timestamp_utc).to_pydatetime())
    frame['year'] = pd.to_datetime(frame.delivery_date).dt.year
    expected = pd.Series({y: sum(len(expected_hours(d)) for d in days() if d.year == y)
                          for y in range(DEV_FIRST.year, DEV_LAST.year + 1)})
    rows, gaps = [], []
    present = frame.groupby('year').size()
    for column, qcol, product, source, role, policies in SERIES:
        null = frame[column].isna()
        for y in expected.index:
            sel = frame.year == y
            n_null = int((null & sel).sum())
            source_gap = int((null & sel & (frame[qcol] == 0)).sum())
            ours = int((null & sel & (frame[qcol] > 0)).sum())
            missing_rows = int(expected[y] - present.get(y, 0))
            null_days = sorted({str(d) for d in frame.delivery_date[null & sel]})
            rows.append({'series': column, 'year': int(y), 'expected_hours': int(expected[y]),
                         'missing_rows': missing_rows, 'null_hours': n_null,
                         'null_hours_cause_source': source_gap, 'null_hours_cause_our_extraction': ours,
                         'days_with_null_hours': len(null_days)})
        for _, r in frame[null].iterrows():
            gaps.append({'series': column, 'delivery_date': str(r.delivery_date),
                         'timestamp_utc': pd.Timestamp(r.timestamp_utc).isoformat(),
                         'local_hour': int(pd.Timestamp(r.timestamp_utc).tz_convert('Europe/Berlin').hour),
                         'source_values_in_hour': int(r[qcol]),
                         'cause': 'source' if r[qcol] == 0 else 'our_extraction (partial hour fail-closed to null)'})
    return pd.DataFrame(rows), pd.DataFrame(gaps)


def price_granularity(root: Path) -> list[dict]:
    """The price's native interval: hourly SDAC products, then quarter-hour products averaged to the hour."""
    frame = pd.read_parquet(Path(root) / 'data/snapshot.parquet', columns=['delivery_date', 'price_source_interval_count'],
                            filters=[('delivery_date', '<=', DEV_LAST)])
    out = []
    for count, g in frame.groupby('price_source_interval_count'):
        out.append({'source_intervals_per_hour': int(count), 'first_delivery_day': str(min(g.delivery_date)),
                    'last_delivery_day': str(max(g.delivery_date)), 'hours': int(len(g))})
    return out


def weather_inventory(cp20_features: pd.DataFrame, cp25_features: pd.DataFrame | None) -> pd.DataFrame:
    """Per-year weather coverage of every development delivery day, before and after CP-25."""
    before = {date.fromisoformat(str(d)) if not isinstance(d, date) else d: s
              for d, s in cp20_features.groupby('delivery_date').status.first().items()}
    after = dict(before)
    if cp25_features is not None:
        for d, s in cp25_features.groupby('delivery_date').status.first().items():
            after[d if isinstance(d, date) else date.fromisoformat(str(d))] = s
    rows = []
    for y in range(DEV_FIRST.year, DEV_LAST.year + 1):
        ds = [d for d in days() if d.year == y]
        def tally(table):
            absent = [d for d in ds if d not in table]
            structural = [d for d in ds if table.get(d) == 'structural_missing_pre_2019_run']
            confirmed = [d for d in ds if table.get(d) in ('confirmed_archive_absence', 'documented_lateness', 'invalid_support')]
            return absent, structural, confirmed
        a0, s0, c0 = tally(before)
        a1, s1, c1 = tally(after)
        rows.append({'series': 'gfs_weather (u10, v10, u100, v100, dswrf -> 3 derived columns)', 'year': y,
                     'delivery_days': len(ds),
                     'before_cp25_days_without_record': len(a0), 'before_cp25_cause': 'our_extraction (outside CP-20 scope)' if a0 else '',
                     'before_cp25_first_last_absent': f'{min(a0)}..{max(a0)}' if a0 else '',
                     'after_cp25_days_without_record': len(a1),
                     'structural_missing_days': len(s1), 'structural_cause': 'run before the 2019 input floor (§15.3)' if s1 else '',
                     'confirmed_missing_or_invalid_days': len(c1)})
    return pd.DataFrame(rows)


def describe(root: Path, snap_rows: pd.DataFrame, weather_rows: pd.DataFrame) -> list[dict]:
    """One descriptive record per series: source, product, licence, granularity, timezone, role."""
    gran = price_granularity(root)
    out = []
    for column, _, product, source, role, policies in SERIES:
        if column == 'price_eur_mwh':
            granularity = ('hourly; SDAC 60-minute MTU through 2025-09-30, then four 15-minute MTU prices per hour averaged by '
                           'the snapshot pull (aggregation "mean"); intervals: ' + json.dumps(gran))
        else:
            granularity = 'hourly; four quarter-hour SMARD values summed to MWh per hour (= hour-average MW); partial hours null'
        out.append({'series': column, 'product': product, 'source': source, 'licence': SMARD_LICENCE,
                    'granularity': granularity, 'timezone': SNAPSHOT_TZ, 'role': role, 'policies': policies,
                    'retrieval': 'preserved snapshot bytes (pulled 2026-09-08, SMARD route); not re-pulled (§24.6 item 1)',
                    'gap_hours_total': int(snap_rows[snap_rows.series == column].null_hours.sum())})
    out.append({'series': 'gfs_weather', 'product': ('NCEP GFS 0.25° operational D-1 00 UTC run, leads f021-f048 (3-hourly); fields '
                                                     'u10, v10, u100, v100 (m/s), DSWRF (W/m², 6-hour-reset averages)'),
                'source': 'NOAA AWS noaa-gfs-bdp-pds (primary for 2021+), NCAR GDEX d084001 THREDDS (alternate; CP-20 2019-2020)',
                'licence': GFS_LICENCE,
                'granularity': ('0.25° grid, box 47-55.25°N 5.5-15.5°E (34 x 41 cells); converted by §15.2 to hourly mean 10 m and '
                                '100 m wind speed and mean DSWRF per canonical delivery hour'),
                'timezone': 'run initialization and leads in UTC; delivery hours Europe/Berlin (h22-h46 after the run)',
                'role': 'input (three weather columns + three missing indicators)', 'policies': 'v3, v4, v5',
                'retrieval': 'CP-20 retained grids (2,476 runs) + CP-25 (329 runs), anonymous archives',
                'gap_days_after_cp25': int(weather_rows.after_cp25_days_without_record.sum())})
    out.append({'series': 'calendar (deterministic)', 'product': ('local hour, day of week, month, German federal holidays, day after '
                                                                  'holiday, bridge day, day type, DST-transition day, summer/winter '
                                                                  'peak flags, weekday dummies, month one-hot'),
                'source': 'computed: Python `holidays` (DE national calendar; version pinned in uv.lock) and the IANA tz database',
                'licence': 'derived in code (MIT, LICENSE); no third-party data',
                'granularity': 'per canonical hour / delivery day', 'timezone': 'Europe/Berlin',
                'role': 'input', 'policies': 'B0 (weekday rule), v1-R, v2, v3, v4, v5', 'retrieval': 'none',
                'gap_hours_total': 0})
    return out
