"""SMARD `chart_data` client: the route the 2026-09-08 snapshot used (`delu_forecast.ingest`, `data/README.md`).

* **Series.** Day-ahead price, filter 4169 (plan series A44): the hourly product before 2025-10-01 (resolution
  `hour`), the mean of the four quarter-hour prices from 2025-10-01 (resolution `quarterhour`). Day-ahead load
  forecast, filter 411 (plan series A65): the sum of the four quarter-hour MWh values, numerically the
  hour-average MW. Both rules are the landed `SeriesSpec`s and `aggregate_quarterhours`, imported unchanged.
* **Fail closed.** A quarter-hour bin with one to three non-null quarters gives no value (NaN) and a recorded
  problem, never a partial sum or mean (the snapshot's rule; for prices the snapshot's pull refused outright).
* **Requests.** `index_<resolution>.json` lists the bucket start instants (ms); each bucket file
  `<filter>_DE-LU_<resolution>_<start>.json` holds `{"meta_data": {...}, "series": [[ms, value|null], ...]}`
  for one week. A bucket request's scope is the delivery days the bucket can contain, so `cp25.guard` refuses a
  bucket that reaches past the boundary before it is asked for.
* **Document time.** If a bucket carries `meta_data.created` (ms), it is returned as the document's own creation
  time. Rendered development responses carry none.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd

from cp15.data import day_hours
from delu_forecast.ingest import (BERLIN, SERIES_SPECS, TRANSITION_LOCAL_DATE, aggregate_quarterhours,
                                  quarterhour_source_counts)

from .transport import ReplayMiss, Request, Response, TransportError

ROOT = 'https://www.smard.de/app/chart_data'
REGION = 'DE-LU'
PRICE, LOAD = SERIES_SPECS[0], SERIES_SPECS[1]
SPECS = {'price': PRICE, 'load_forecast': LOAD}
WEEK = timedelta(days=7)
UTC = timezone.utc


def resolution(input_name: str, day: date) -> str:
    if input_name == 'price':
        return 'hour' if day < TRANSITION_LOCAL_DATE else 'quarterhour'
    if input_name == 'load_forecast':
        return 'quarterhour'
    raise ValueError(f'unknown SMARD input {input_name!r}')


def index_request(input_name: str, res: str) -> Request:
    spec = SPECS[input_name]
    return Request('GET', f'{ROOT}/{spec.filter_id}/{REGION}/index_{res}.json', scope=('metadata',))


def bucket_days(start_ms: int) -> list[date]:
    """Every Europe/Berlin delivery day a bucket starting at `start_ms` can hold (one week)."""
    start = pd.Timestamp(start_ms, unit='ms', tz='UTC').tz_convert(BERLIN)
    first = start.date()
    last = (start + pd.DateOffset(days=7) - pd.Timedelta(milliseconds=1)).date()   # seven local calendar days
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def bucket_request(input_name: str, res: str, start_ms: int) -> Request:
    spec = SPECS[input_name]
    url = f'{ROOT}/{spec.filter_id}/{REGION}/{spec.filter_id}_{REGION}_{res}_{int(start_ms)}.json'
    return Request('GET', url, scope=('market', *[d.isoformat() for d in bucket_days(start_ms)]))


def parse_index(body: bytes) -> list[int]:
    value = json.loads(body)
    stamps = value.get('timestamps')
    if not isinstance(stamps, list) or not all(isinstance(t, int) for t in stamps):
        raise ValueError('SMARD index without integer timestamps')
    return sorted(stamps)


def parse_bucket(body: bytes) -> tuple[pd.DataFrame, dict]:
    value = json.loads(body)
    rows = value.get('series')
    if not isinstance(rows, list):
        raise ValueError('SMARD bucket without a series')
    frame = pd.DataFrame({'timestamp_utc': pd.to_datetime([r[0] for r in rows], unit='ms', utc=True),
                          'value': pd.to_numeric([r[1] for r in rows], errors='coerce')})
    if frame.timestamp_utc.duplicated().any():
        raise ValueError('SMARD bucket repeats a timestamp')
    meta = value.get('meta_data') or {}
    created = meta.get('created')
    doc = (datetime.fromtimestamp(created / 1000, UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
           if isinstance(created, (int, float)) else None)
    return frame.sort_values('timestamp_utc').reset_index(drop=True), {'document_time_utc': doc,
                                                                      'version': meta.get('version')}


def hourly(input_name: str, res: str, raw: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Hourly values under the snapshot's rule, and the non-null source count per hour (1 for hourly data)."""
    spec = SPECS[input_name]
    if res == 'hour':
        s = raw.set_index('timestamp_utc')['value']
        if not s.index.equals(s.index.floor('h')):
            raise ValueError('hourly SMARD series off the hour')
        return s.rename(spec.column), s.notna().astype('int8')
    if res != 'quarterhour':
        raise ValueError(f'unknown resolution {res!r}')
    values = aggregate_quarterhours(raw, spec, fail_on_partial=False)
    return values, quarterhour_source_counts(raw)


def day_values(series: pd.Series, counts: pd.Series, res: str, day: date) -> tuple[np.ndarray, list[str]]:
    """The day's canonical hours (NaN where absent), with any partial quarter-hour bins named."""
    hours = day_hours(day)
    values = series.reindex(hours).to_numpy(float)
    problems = []
    if res == 'quarterhour':
        c = counts.reindex(hours).fillna(0).to_numpy(int)
        partial = [str(h) for h, k in zip(hours, c) if 0 < k < 4]
        if partial:
            problems.append(f'{day}: {len(partial)} partial quarter-hour bins fail closed to NaN ({partial[:3]})')
    missing = int(np.isnan(values).sum())
    if missing:
        problems.append(f'{day}: {missing} of {len(hours)} hours without a value')
    return values, problems


def values_from_body(input_name: str, res: str, body: bytes, days) -> dict:
    """Re-parse an archived bucket body into the covered days' canonical-hour values (the history's route)."""
    raw, _ = parse_bucket(body)
    series, counts = hourly(input_name, res, raw)
    return {d: day_values(series, counts, res, d)[0] for d in days}


@dataclass
class Exchange:
    request: Request
    response: Response | None
    role: str                       # 'index' or 'bucket'
    resolution: str
    covers: list = field(default_factory=list)
    document_time_utc: str | None = None
    error: str | None = None


@dataclass
class Fetched:
    input: str
    days: list
    source: str = 'smard'
    exchanges: list = field(default_factory=list)
    values: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)


def fetch(transport, input_name: str, days) -> Fetched:
    """Retrieve `days` (delivery days) of one input: one index per resolution, then each needed bucket once."""
    days = sorted(set(days))
    out = Fetched(input_name, days)
    by_res: dict[str, list[date]] = {}
    for d in days:
        by_res.setdefault(resolution(input_name, d), []).append(d)
    for res, group in by_res.items():
        req = index_request(input_name, res)
        try:
            resp = transport.send(req)
        except (ReplayMiss, TransportError) as exc:
            out.exchanges.append(Exchange(req, None, 'index', res, error=str(exc)))
            out.problems.append(f'{input_name} index_{res}: not retrieved ({exc})')
            continue
        out.exchanges.append(Exchange(req, resp, 'index', res))
        if resp.status != 200:
            out.problems.append(f'{input_name} index_{res}: HTTP {resp.status}')
            continue
        starts = parse_index(resp.body)
        needed: dict[int, list[date]] = {}
        for d in group:
            hours = day_hours(d)
            first_ms = int(hours[0].timestamp() * 1000)
            last_ms = int(hours[-1].timestamp() * 1000)
            hits = [s for s in starts if s <= last_ms and s + int(WEEK.total_seconds() * 1000) > first_ms]
            if not hits:
                out.problems.append(f'{input_name} {d}: no SMARD bucket covers the day')
            for s in hits:
                needed.setdefault(s, []).append(d)
        series_parts, count_parts = [], []
        for start, covered in sorted(needed.items()):
            req = bucket_request(input_name, res, start)
            try:
                resp = transport.send(req)
            except (ReplayMiss, TransportError) as exc:
                out.exchanges.append(Exchange(req, None, 'bucket', res, covered, error=str(exc)))
                out.problems.append(f'{input_name} bucket {start}: not retrieved ({exc})')
                continue
            if resp.status != 200:
                out.exchanges.append(Exchange(req, resp, 'bucket', res, covered))
                out.problems.append(f'{input_name} bucket {start}: HTTP {resp.status}')
                continue
            raw, meta = parse_bucket(resp.body)
            out.exchanges.append(Exchange(req, resp, 'bucket', res, covered, meta['document_time_utc']))
            s, c = hourly(input_name, res, raw)
            series_parts.append(s)
            count_parts.append(c)
        if not series_parts:
            continue
        series = pd.concat(series_parts)
        counts = pd.concat(count_parts)
        if series.index.has_duplicates:
            dup = series[series.index.duplicated(keep=False)]
            if dup.groupby(level=0).nunique(dropna=False).gt(1).any():
                out.problems.append(f'{input_name}: buckets disagree where they overlap')
            series = series[~series.index.duplicated()]
            counts = counts[~counts.index.duplicated()]
        for d in group:
            values, problems = day_values(series, counts, res, d)
            out.values[d] = values
            out.problems.extend(f'{input_name} {p}' for p in problems)
    return out
