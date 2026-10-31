"""History for delivery day D: the committed base through `base_end`, extended by the archive's increments to D.

**The base** is the committed development history, read with filters before materialisation and then through
`cp25.guard` (every delivery day, every row):
- `data/snapshot.parquet`'s four raw columns (`cp15.data.RAW_COLUMNS`) for delivery days 2019-01-01..base_end.
  Its load forecast is the stored development history that ruling 1's recorded exception admits;
- the frozen weather record: CP-20's `reports/weather-ablation/weather-features.parquet` plus CP-25's
  `reports/development-inputs/weather-features-gaps.parquet`, through base_end, with ruling 4's two late runs
  (delivery days 2022-12-01, 2023-01-21) set missing under §15.3 exactly as `cp26.harness._features` does.

**The increments**, each delivery day P in (base_end, D], from archive days <= D only (never a later one):
- **prices of P** (P <= D-1) from the earliest archived record that covers P and is admitted against D's cutoff
  (tier 2: the documented schedule). Further admitted records of P must agree (SDAC results are firm). **D's own
  price is never read: D's rows carry NaN** (the target-day embargo);
- **the load forecast of P**, preferring P's own archive day, then later ones, admitted against D's cutoff
  (tier 3: retrieval or document time strictly before it, or ruling 1 for development data). A load forecast
  not admitted is left missing (NaN), and the judgement says why;
- **the weather of P** from the D-1 00 UTC run archived on day P, converted by CP-20's frozen recipe
  (`cp20.weather.convert_run`) if P's GFS flag is True; otherwise all of P's hours are missing under §15.3
  (`cp20.weather.missing_day`), with the reason.

Nothing after D is ever in the frame. `published_prices` gives the shadow states' release its truth, from
delivery days <= D-1 only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cp15.data import RAW_COLUMNS, day_hours
from cp20 import weather as W
from cp25 import guard

from . import availability as A
from . import entsoe, smard
from .archive import Archive

FLOOR = date(2019, 1, 1)
SNAPSHOT = 'data/snapshot.parquet'
CP20_FEATURES = 'reports/weather-ablation/weather-features.parquet'
GAP_FEATURES = 'reports/development-inputs/weather-features-gaps.parquet'
RULING_4 = "Owner's ruling 4 (2026-10-12): AWS objects written after the issue time"


def late_gfs_delivery() -> tuple:
    """Ruling 4's two runs, as CP-26's harness records them (a constant; nothing there binds a boundary)."""
    from cp26.harness import LATE_GFS_DELIVERY
    return LATE_GFS_DELIVERY


class HistoryError(ValueError):
    pass


# ------------------------------------------------------------------------------------------------ base
def base_frame(root: Path, base_end: date) -> pd.DataFrame:
    guard.admit_delivery_day(base_end)
    frame = pd.read_parquet(Path(root) / SNAPSHOT, columns=RAW_COLUMNS,
                            filters=[('delivery_date', '>=', FLOOR), ('delivery_date', '<=', base_end)])
    days = pd.Index(frame['delivery_date'])
    if not len(days) or max(days) != base_end or min(days) != FLOOR:
        raise HistoryError(f'the committed snapshot does not cover {FLOOR}..{base_end}')
    guard.admit_delivery_day(max(days))
    guard.admit_rows(pd.DatetimeIndex(frame['timestamp_utc']).tz_convert('UTC'))
    return frame


def _guarded_features(path: Path, base_end: date) -> pd.DataFrame:
    frame = pd.read_parquet(path, filters=[('delivery_date', '<=', base_end)])
    instants = pd.DatetimeIndex(pd.to_datetime(frame.timestamp_utc, utc=True))
    guard.admit_rows(instants)
    if not np.array_equal(instants.tz_convert('Europe/Berlin').date, frame.delivery_date.to_numpy()):
        raise HistoryError('weather delivery dates disagree with canonical timestamps')
    return frame


def base_weather(root: Path, base_end: date) -> pd.DataFrame:
    root = Path(root)
    guard.admit_delivery_day(base_end)
    cp20 = _guarded_features(root / CP20_FEATURES, base_end)
    gaps = _guarded_features(root / GAP_FEATURES, base_end)
    if set(cp20.delivery_date) & set(gaps.delivery_date):
        raise HistoryError('CP-20 and CP-25 weather features overlap')
    features = pd.concat([cp20, gaps], ignore_index=True)
    features['timestamp_utc'] = pd.to_datetime(features.timestamp_utc, utc=True)
    if features.timestamp_utc.duplicated().any():
        raise HistoryError('duplicate canonical weather hour')
    late = features.delivery_date.isin(late_gfs_delivery())
    features.loc[late, list(W.COLUMNS)] = np.nan
    features.loc[late, 'status'] = 'documented_lateness'
    features.loc[late, 'reason'] = RULING_4
    return features


# ------------------------------------------------------------------------------------------------ increments
VALUES = {'smard': lambda name, rec, body, days: smard.values_from_body(name, rec['resolution'], body, days),
          'entsoe': lambda name, rec, body, days: entsoe.values_from_body(name, body, days)}


@dataclass
class History:
    day: date
    base_end: date
    frame: pd.DataFrame
    features: pd.DataFrame
    increment_days: list
    judgements: list = field(default_factory=list)
    gfs: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)
    missing_required: list = field(default_factory=list)

    def truth(self):
        return published_prices(self.frame, self.day)


class _Reader:
    def __init__(self, archive: Archive, for_day: date):
        self.archive, self.for_day = archive, for_day
        self.days = [d for d in archive.days() if d <= for_day]
        self._cache = {}

    def candidates(self, input_name: str, data_day: date) -> list:
        lo = data_day + timedelta(days=1) if input_name == 'price' else data_day
        found = []
        for x in self.days:
            if x < lo:
                continue
            for r in self.archive.records(x, input_name):
                if data_day.isoformat() in r.get('covers', []) and r.get('status') == 200 and r.get('body'):
                    found.append((x, r))
        found.sort(key=lambda xr: (xr[0] != data_day if input_name == 'load_forecast' else False, xr[0]))
        return found

    def values(self, x: date, record: dict, input_name: str, data_day: date) -> np.ndarray:
        key = (x, record['id'])
        if key not in self._cache:
            body = self.archive.body(x, record)
            covered = [date.fromisoformat(c) for c in record['covers']]
            self._cache[key] = VALUES[record['source']](input_name, record, body, covered)
        return self._cache[key][data_day]


def market(reader: _Reader, input_name: str, data_day: date) -> tuple[np.ndarray | None, dict, list]:
    """The admitted values of one input on one data day for D, its judgement, and any problems."""
    problems, chosen, chosen_values, judgements = [], None, None, []
    for x, rec in reader.candidates(input_name, data_day):
        pub = rec['publication'].get(data_day.isoformat(), {})
        j = A.judge(input_name, reader.for_day, data_day, pub, rec, archive_day=x)
        judgements.append(j)
        if not j['admitted']:
            continue
        values = reader.values(x, rec, input_name, data_day)
        if chosen is None:
            chosen, chosen_values = j, values
        elif input_name == 'price' and not np.array_equal(values, chosen_values, equal_nan=True):
            problems.append(f'price {data_day}: archived records disagree ({chosen["record"]} on '
                            f'{chosen["archive_day"]} vs {rec["id"]} on {x}); the earliest is used')
    if chosen is None:
        reason = ('no archived record covers it' if not judgements else
                  '; '.join(f'{j["record"]}@{j["archive_day"]}: {j["reason"]}' for j in judgements))
        j = {'input': input_name, 'data_day': data_day.isoformat(), 'for_day': reader.for_day.isoformat(),
             'admitted': False, 'by_exception': False, 'reason': reason, 'candidates': judgements,
             'cutoff_utc': A.iso(A.cutoff(reader.for_day))}
        return None, j, problems
    chosen = dict(chosen, candidates=len(judgements))
    return chosen_values, chosen, problems


def gfs_for_day(archive: Archive, day: date) -> dict:
    """The GFS flag of `day` (its D-1 00 UTC run against its own cutoff) and the day's weather rows."""
    run = day - timedelta(days=1)
    out = {'day': day.isoformat(), 'run': run.isoformat(), 'available': False, 'judgement': None, 'reason': '',
           'status': 'documented_lateness'}
    recs = archive.records(day, 'gfs_run') if archive.has(day) else []
    frame, version = None, None
    if len(recs) != 1:
        out['reason'] = ('no archived evidence that the run preceded the cutoff' if not recs
                         else f'{len(recs)} archived runs for one day')
    else:
        rec = recs[0]
        pub = rec['publication'].get(day.isoformat(), {})
        j = A.judge('gfs', day, day, pub, rec, archive_day=day)
        out['judgement'] = j
        try:
            for rel in rec['files']:
                archive.file(day, rec, rel)
            record, data, bounds, quanta = W.load_run(archive.day_dir(day) / 'gfs', run.isoformat())
            version = record.get('version')
            frame = W.convert_run(day, data, bounds, quanta)
            valid = bool((frame.status == 'ok').all()) and bool(np.isfinite(frame[list(W.COLUMNS)].to_numpy()).all())
        except Exception as exc:  # noqa: BLE001 - a run that cannot be read or converted is invalid support
            valid, out['conversion_error'] = False, f'{type(exc).__name__}: {exc}'
        checks_ok = rec.get('checks', {}).get('messages') == 50 and not rec.get('problems')
        out['vector_valid'] = valid
        if not j['admitted']:
            out['reason'] = f'not shown published before the cutoff: {j["reason"]}'
        elif not checks_ok:
            out['reason'], out['status'] = f'the run is incomplete: {rec.get("problems")}', 'invalid_support'
        elif not valid:
            out['reason'], out['status'] = 'the converted vector is not valid on every hour (§15.3)', 'invalid_support'
        else:
            out.update(available=True, status='ok', reason=j['reason'])
    if out['available']:
        features = frame
    else:
        features = W.missing_day(day, out['status'], f'GFS flag False: {out["reason"]}')
    features['run_00z'], features['version'] = run.isoformat(), version
    features['timestamp_utc'] = pd.to_datetime(features.timestamp_utc, utc=True)
    features['local_hour'] = features.timestamp_utc.dt.tz_convert('Europe/Berlin').dt.hour
    out['features'] = features
    return out


def assemble(root: Path, archive: Archive, day: date, base_end: date) -> History:
    """The frame (RAW_COLUMNS) and the weather record through D: the base, then the archive's increments."""
    guard.admit_delivery_day(day)
    if not FLOOR < base_end < day:
        raise HistoryError('the base must end before the day being built')
    frame = base_frame(root, base_end)
    features = base_weather(root, base_end)
    reader = _Reader(archive, day)
    inc_days = [base_end + timedelta(days=i) for i in range(1, (day - base_end).days + 1)]
    history = History(day, base_end, frame, features, inc_days)
    rows, wx = [], []
    for p in inc_days:
        hours = day_hours(p)
        if p < day:
            price, j, probs = market(reader, 'price', p)
            history.judgements.append(j)
            history.problems += probs
            if price is None:
                history.missing_required.append(f'price of {p}: {j["reason"]}')
                price = np.full(len(hours), np.nan)
            elif np.isnan(price).any():
                history.problems.append(f'price of {p}: {int(np.isnan(price).sum())} hours without a value')
                history.missing_required.append(f'price of {p}: incomplete')
        else:
            price = np.full(len(hours), np.nan)      # the target-day embargo: D's price is never read
        load, j, probs = market(reader, 'load_forecast', p)
        history.judgements.append(j)
        history.problems += probs
        if load is None:
            load = np.full(len(hours), np.nan)
            if p == day:
                history.missing_required.append(f'load forecast of {p}: {j["reason"]}')
        elif p == day and np.isnan(load).any():
            history.problems.append(f'load forecast of {p}: {int(np.isnan(load).sum())} hours without a value')
        rows.append(pd.DataFrame({'timestamp_utc': hours, 'delivery_date': [p] * len(hours),
                                  'price_eur_mwh': price, 'load_forecast_mw': load}))
        g = gfs_for_day(archive, p)
        history.gfs[p.isoformat()] = {k: v for k, v in g.items() if k != 'features'}
        wx.append(g['features'])
    inc = pd.concat(rows, ignore_index=True)
    inc['timestamp_utc'] = inc['timestamp_utc'].astype(frame['timestamp_utc'].dtype)
    guard.admit_rows(pd.DatetimeIndex(inc['timestamp_utc']).tz_convert('UTC'))
    last = pd.Timestamp(frame['timestamp_utc'].iloc[-1])
    if pd.Timestamp(inc['timestamp_utc'].iloc[0]) != last + pd.Timedelta(hours=1):
        raise HistoryError('the increments do not continue the base hour by hour')
    history.frame = pd.concat([frame, inc[RAW_COLUMNS]], ignore_index=True)
    wxf = pd.concat(wx, ignore_index=True)
    for col in features.columns:
        if col not in wxf.columns:
            wxf[col] = np.nan if features[col].dtype.kind == 'f' else None
    wxf = wxf[list(features.columns)]
    for col, dtype in features.dtypes.items():
        if str(wxf[col].dtype) != str(dtype):
            wxf[col] = wxf[col].astype(dtype)
    guard.admit_rows(pd.DatetimeIndex(wxf.timestamp_utc))
    allf = pd.concat([features, wxf], ignore_index=True)
    if allf.timestamp_utc.duplicated().any():
        raise HistoryError('duplicate canonical weather hour')
    history.features = allf.sort_values('timestamp_utc').reset_index(drop=True)
    if max(history.frame.delivery_date) != day or max(history.features.delivery_date) != day:
        raise HistoryError('the history must end at D')
    return history


def published_prices(frame: pd.DataFrame, day: date):
    """`truth(keys) -> array`: published prices on the requested keys, NaN unless every key is present; only
    delivery days <= D-1 (the release asks for complete days <= D-2)."""
    keep = frame.loc[pd.Index(frame.delivery_date) < day]
    index = pd.DatetimeIndex(keep.timestamp_utc).tz_convert('UTC').as_unit('ns')
    y = keep.price_eur_mwh.to_numpy(float)

    def at(keys):
        pos = index.get_indexer(pd.DatetimeIndex(pd.to_datetime(keys, utc=True)).as_unit('ns'))
        if (pos < 0).any():
            return np.full(len(keys), np.nan)
        return np.asarray(y[pos], float)
    return at
