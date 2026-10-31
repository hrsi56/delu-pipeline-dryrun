"""ENTSO-E Transparency Platform REST client: A44 day-ahead prices and A65 (process A01) day-ahead total load.

The plan series behind the snapshot (`data/README.md`); the 2026-09-08 pull used SMARD because the API was
unavailable then. One request asks for a contiguous run of Europe/Berlin delivery days, DE-LU
(`10Y1001A1001A82H`), `periodStart`/`periodEnd` in UTC `yyyyMMddHHmm`.

* **The token.** The request names `ENTSOE_API_TOKEN` as its credential *variable*; only the live transport reads
  it (`os.environ.get`, inside `send`), so the token is never in a request, a key, a record or a log.
* **Parsing** is namespace-agnostic (`Publication_MarketDocument`, `GL_MarketDocument`; an
  `Acknowledgement_MarketDocument` is "no data" with its reason). Each `Period` has a `timeInterval`, a
  `resolution` (`PT15M`, `PT30M`, `PT60M`) and `Point`s by position. Curve type `A01` leaves an absent position
  missing; `A03` carries the previous point's value forward (the documented variable-block curve).
* **Hourly values** follow the snapshot's rules: prices use the hourly product before 2025-10-01 and the mean of
  four quarter-hour prices from 2025-10-01; the load is the mean of the four quarter-hour MW values (A65 is
  hour-average MW). Aggregation is the landed `aggregate_quarterhours`; a partial bin fails closed to NaN.
* **Document time.** `createdDateTime` is returned as the document's own time. Rendered development responses
  set it to the snapshot's pull time, and the archive does not use a rendered document's time as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from cp15.data import day_hours
from delu_forecast.ingest import BERLIN, SERIES_SPECS, TRANSITION_LOCAL_DATE, SeriesSpec, aggregate_quarterhours

from .transport import ReplayMiss, Request, Response, TransportError

API = 'https://web-api.tp.entsoe.eu/api'
DELU = '10Y1001A1001A82H'
CREDENTIAL = 'ENTSOE_API_TOKEN'
PRICE_SPEC = SERIES_SPECS[0]                       # mean of quarter-hour prices
LOAD_SPEC = SeriesSpec(411, 'load_forecast_mw', 'A65/A01', 'ENTSO-E A65 day-ahead total load (MW)', 'mean',
                       'MW-hour-average', 'kft_assumption')
RESOLUTIONS = {'PT15M': 15, 'PT30M': 30, 'PT60M': 60}
DOCS = {'price': 'A44', 'load_forecast': 'A65'}


def _utc(stamp: pd.Timestamp) -> str:
    return stamp.tz_convert('UTC').strftime('%Y%m%d%H%M')


def period(days) -> tuple[str, str]:
    days = sorted(days)
    if [(b - a).days for a, b in zip(days, days[1:])] != [1] * (len(days) - 1):
        raise ValueError('an ENTSO-E request covers contiguous delivery days')
    start = pd.Timestamp(days[0], tz=BERLIN)
    end = pd.Timestamp(days[-1] + timedelta(days=1), tz=BERLIN)
    return _utc(start), _utc(end)


def request(input_name: str, days) -> Request:
    start, end = period(days)
    if input_name == 'price':
        params = {'documentType': 'A44', 'in_Domain': DELU, 'out_Domain': DELU, 'periodStart': start, 'periodEnd': end}
    elif input_name == 'load_forecast':
        params = {'documentType': 'A65', 'processType': 'A01', 'outBiddingZone_Domain': DELU,
                  'periodStart': start, 'periodEnd': end}
    else:
        raise ValueError(f'unknown ENTSO-E input {input_name!r}')
    return Request('GET', API, params=tuple(sorted(params.items())), credential=CREDENTIAL,
                   scope=('market', *[d.isoformat() for d in sorted(days)]))


def _local(tag: str) -> str:
    return tag.rsplit('}', 1)[-1]


def _find(node, name):
    for child in node:
        if _local(child.tag) == name:
            return child
    return None


def _findall(node, name):
    return [child for child in node if _local(child.tag) == name]


def _text(node, path: str):
    for part in path.split('/'):
        if node is None:
            return None
        node = _find(node, part)
    return None if node is None else (node.text or '').strip()


def parse(body: bytes) -> dict:
    """The document's kind, time, and every TimeSeries/Period as points by position."""
    root = ET.fromstring(body)
    kind = _local(root.tag)
    out = {'kind': kind, 'created_utc': _text(root, 'createdDateTime'), 'type': _text(root, 'type'),
           'revision': _text(root, 'revisionNumber'), 'series': [], 'reason': None}
    if kind == 'Acknowledgement_MarketDocument':
        out['reason'] = _text(root, 'Reason/text') or _text(root, 'Reason/code')
        return out
    for ts in _findall(root, 'TimeSeries'):
        curve = _text(ts, 'curveType') or 'A01'
        domain = _text(ts, 'in_Domain.mRID') or _text(ts, 'outBiddingZone_Domain.mRID')
        unit = _text(ts, 'price_Measure_Unit.name') or _text(ts, 'quantity_Measure_Unit.name')
        for p in _findall(ts, 'Period'):
            res = _text(p, 'resolution')
            start, end = _text(p, 'timeInterval/start'), _text(p, 'timeInterval/end')
            points = {}
            for pt in _findall(p, 'Point'):
                pos = int(_text(pt, 'position'))
                raw = _text(pt, 'price.amount')
                if raw is None:
                    raw = _text(pt, 'quantity')
                points[pos] = float(raw)
            out['series'].append({'curve': curve, 'domain': domain, 'unit': unit, 'resolution': res,
                                  'start': start, 'end': end, 'points': points})
    return out


def _instant(text: str) -> pd.Timestamp:
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?Z', text or ''):
        raise ValueError(f'unexpected ENTSO-E instant {text!r}')
    return pd.Timestamp(text)


def points_frame(series: list[dict], res: str) -> tuple[pd.DataFrame, list[str]]:
    """Every point of one resolution as (timestamp_utc, value); A03 positions carried forward."""
    step = pd.Timedelta(minutes=RESOLUTIONS[res])
    rows, problems = {}, []
    for s in series:
        if s['resolution'] != res:
            continue
        start, end = _instant(s['start']), _instant(s['end'])
        n = int((end - start) / step)
        if start + n * step != end:
            problems.append(f'period {s["start"]}..{s["end"]} is not a whole number of {res}')
            continue
        last = np.nan
        for pos in range(1, n + 1):
            if pos in s['points']:
                value = s['points'][pos]
                last = value
            elif s['curve'] == 'A03':
                value = last
            else:
                value = np.nan
            t = start + (pos - 1) * step
            if t in rows and not (rows[t] == value or (np.isnan(rows[t]) and np.isnan(value))):
                problems.append(f'time series disagree at {t}')
                rows[t] = np.nan
            else:
                rows[t] = value
    frame = pd.DataFrame({'timestamp_utc': pd.DatetimeIndex(sorted(rows)),
                          'value': [rows[t] for t in sorted(rows)]}) if rows else \
        pd.DataFrame({'timestamp_utc': pd.DatetimeIndex([], tz='UTC'), 'value': []})
    return frame, problems


def hourly(input_name: str, doc: dict, day: date) -> tuple[np.ndarray, list[str]]:
    """The day's canonical hours under the snapshot's aggregation rules."""
    problems = []
    if doc['kind'] == 'Acknowledgement_MarketDocument':
        return np.full(len(day_hours(day)), np.nan), [f'{day}: no data ({doc["reason"]})']
    for s in doc['series']:
        if s['domain'] not in (None, DELU):
            problems.append(f'{day}: a time series for domain {s["domain"]}, not DE-LU')
    available = {s['resolution'] for s in doc['series'] if s['domain'] in (None, DELU)}
    if input_name == 'price':
        res = 'PT60M' if day < TRANSITION_LOCAL_DATE else 'PT15M'
        spec = PRICE_SPEC
    else:
        res = 'PT15M' if 'PT15M' in available else 'PT60M'
        spec = LOAD_SPEC
    hours = day_hours(day)
    if res not in available:
        return np.full(len(hours), np.nan), problems + [f'{day}: no {res} series for {input_name}']
    frame, p = points_frame([s for s in doc['series'] if s['domain'] in (None, DELU)], res)
    problems += [f'{day}: {x}' for x in p]
    if res == 'PT60M':
        values = frame.set_index('timestamp_utc')['value'].reindex(hours).to_numpy(float)
    else:
        agg = aggregate_quarterhours(frame, spec, fail_on_partial=False)
        counts = frame.assign(h=frame.timestamp_utc.dt.floor('h')).groupby('h')['value'].count()
        partial = [str(h) for h in hours if 0 < counts.get(h, 0) < 4]
        if partial:
            problems.append(f'{day}: {len(partial)} partial quarter-hour bins fail closed to NaN')
        values = agg.reindex(hours).to_numpy(float)
    missing = int(np.isnan(values).sum())
    if missing:
        problems.append(f'{day}: {missing} of {len(hours)} hours without a value')
    return values, problems


def values_from_body(input_name: str, body: bytes, days) -> dict:
    doc = parse(body)
    return {d: hourly(input_name, doc, d)[0] for d in days}


@dataclass
class Exchange:
    request: Request
    response: Response | None
    role: str = 'document'
    resolution: str = ''
    covers: list = field(default_factory=list)
    document_time_utc: str | None = None
    error: str | None = None


@dataclass
class Fetched:
    input: str
    days: list
    source: str = 'entsoe'
    exchanges: list = field(default_factory=list)
    values: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)


def _runs(days) -> list[list[date]]:
    days = sorted(set(days))
    out = []
    for d in days:
        if out and (d - out[-1][-1]).days == 1:
            out[-1].append(d)
        else:
            out.append([d])
    return out


def fetch(transport, input_name: str, days, *, per_day: bool = True) -> Fetched:
    """One document per delivery day (default), or one per contiguous run of days."""
    out = Fetched(input_name, sorted(set(days)))
    for group in ([[d] for d in sorted(set(days))] if per_day else _runs(days)):
        req = request(input_name, group)
        try:
            resp = transport.send(req)
        except (ReplayMiss, TransportError) as exc:
            out.exchanges.append(Exchange(req, None, covers=group, error=str(exc)))
            out.problems.append(f'{input_name} {group[0]}..{group[-1]}: not retrieved ({exc})')
            for d in group:
                out.values[d] = np.full(len(day_hours(d)), np.nan)
            continue
        if resp.status != 200:
            out.exchanges.append(Exchange(req, resp, covers=group))
            out.problems.append(f'{input_name} {group[0]}..{group[-1]}: HTTP {resp.status}')
            for d in group:
                out.values[d] = np.full(len(day_hours(d)), np.nan)
            continue
        doc = parse(resp.body)
        out.exchanges.append(Exchange(req, resp, covers=group, document_time_utc=_iso(doc['created_utc'])))
        for d in group:
            values, problems = hourly(input_name, doc, d)
            out.values[d] = values
            out.problems.extend(f'{input_name} {p}' for p in problems)
    return out


def _iso(text):
    if not text:
        return None
    return pd.Timestamp(text).tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
