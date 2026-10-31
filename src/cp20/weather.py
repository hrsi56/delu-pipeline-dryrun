"""Frozen v21-r4 section 15.2 conversion and section 15.3 missing-input rule (modelling env).

One recipe only. From the decoded regional boxes of the D-1 00 UTC run:

* wind: each u/v component is linearly interpolated per grid cell between the bracketing
  three-hour endpoints to every canonical delivery-hour start h22..h46 (no extrapolation);
  speed = sqrt(u^2 + v^2) per cell, separately at 10 m and 100 m, *then* averaged;
* radiation: per cell, three-hour block means from the six-hour-reset averages,
  A(L-3,L) at L = 3 mod 6 and 2A(L-6,L) - A(L-6,L-3) at L = 0 mod 6, validated against
  the decoded averaging bounds; each hour [h, h+1) takes its block's mean; negative
  block means >= -3q (q = max packing quantum of the contributing messages) are clipped
  to zero and logged, anything below -3q or with unknown quantum/bounds is a conversion
  failure (invalid support), never a tolerance;
* aggregation: cosine(latitude) weights normalised over the fixed 34 x 41 grid, applied
  per hour; no renormalisation over missing cells; any nonfinite required cell makes the
  hour's three-value weather vector missing;
* design: three same-target-local-hour columns (mean 10 m speed, mean 100 m speed, mean
  DSWRF); a repeated autumn local hour averages its two canonical vectors and requires
  both; the spring gap stays absent. Missing vectors stay NaN: the inherited CP-15 LEAR
  imputer adds the indicators and fits medians/scalers on training rows only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cp15.data import day_hours

LEADS = (21, 24, 27, 30, 33, 36, 39, 42, 45, 48)
FIELDS = ('u10', 'v10', 'u100', 'v100', 'dswrf')
BOX_LATS = 90.0 - 0.25 * np.arange(139, 173)
BOX_LONS = 0.25 * np.arange(22, 63)
COLUMNS = ('wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean')
FLOOR = date(2019, 1, 1)
# Missingness classes of section 15.3, kept separate; only these become imputed NaN.
MISSING_CLASSES = ('structural_missing_pre_2019_run', 'confirmed_archive_absence', 'documented_lateness',
                   'invalid_support')


class ConversionRefused(RuntimeError):
    """Input that the frozen recipe cannot convert and that is not confirmed missing weather."""


def weights() -> np.ndarray:
    w = np.repeat(np.cos(np.deg2rad(BOX_LATS))[:, None], len(BOX_LONS), axis=1)
    return w / w.sum()


def hour_leads(delivery: date) -> np.ndarray:
    """Canonical delivery-hour starts as hours after the D-1 00 UTC run."""
    run = pd.Timestamp(delivery - timedelta(days=1), tz='UTC')
    leads = (day_hours(delivery) - run) / pd.Timedelta(hours=1)
    leads = np.asarray(leads, dtype=float)
    if not np.all(leads == np.round(leads)) or leads.min() < 22 or leads.max() > 46:
        raise ConversionRefused(f'{delivery}: canonical hours outside h22..h46')
    return leads.astype(int)


def interpolate_wind(u: np.ndarray, v: np.ndarray, h: int) -> np.ndarray:
    """Per-cell speed at hour start h from the component fields at LEADS (axis 0)."""
    lo = 3 * (h // 3)
    hi = lo if h % 3 == 0 else lo + 3
    if lo not in LEADS or hi not in LEADS:
        raise ConversionRefused(f'no bracketing endpoints for h{h}')
    i, j = LEADS.index(lo), LEADS.index(hi)
    w = 0.0 if hi == lo else (h - lo) / 3.0
    uu = (1 - w) * u[i] + w * u[j]
    vv = (1 - w) * v[i] + w * v[j]
    return np.sqrt(uu * uu + vv * vv)


def radiation_block(a: np.ndarray, bounds: list, quanta: list, end: int):
    """Three-hour block mean (end-3, end] per cell with the quantum-bounded clipping rule."""
    k = LEADS.index(end)
    if end % 6 == 3:
        if tuple(bounds[k]) != (end - 3, end):
            raise ConversionRefused(f'averaging bounds {bounds[k]} at f{end:03d}')
        block, contributing = a[k].copy(), [quanta[k]]
    else:
        if tuple(bounds[k]) != (end - 6, end) or tuple(bounds[k - 1]) != (end - 6, end - 3):
            raise ConversionRefused(f'averaging bounds {bounds[k - 1]}/{bounds[k]} at f{end:03d}')
        block, contributing = 2.0 * a[k] - a[k - 1], [quanta[k], quanta[k - 1]]
    if any(x is None or not np.isfinite(x) or x <= 0 for x in contributing):
        return None, {'failure': 'unknown_quantum'}
    q = max(contributing)
    finite = np.isfinite(block)
    negative = finite & (block < 0)
    below = finite & (block < -3.0 * q)
    log = {'quantum': float(q), 'clipped_cells': int((negative & ~below).sum()),
           'min_block': float(block[finite].min()) if finite.any() else None,
           'clipped_magnitude_sum': float(-block[negative & ~below].sum())}
    if below.any():
        return None, {**log, 'failure': 'negative_block_below_minus_3q', 'cells_below': int(below.sum())}
    block[negative] = 0.0
    return block, log


def convert_run(delivery: date, data: np.ndarray, bounds: list, quanta: list) -> pd.DataFrame:
    """Canonical-hour weather for one delivery day from its run's decoded boxes."""
    if data.shape != (len(FIELDS), len(LEADS), len(BOX_LATS), len(BOX_LONS)):
        raise ConversionRefused('unexpected decoded array shape')
    w = weights()
    rows = []
    for t, h in zip(day_hours(delivery), hour_leads(delivery)):
        out = {'delivery_date': delivery, 'timestamp_utc': t, 'lead_hour': int(h)}
        values, status, notes = [], 'ok', {}
        for a, b in (('u10', 'v10'), ('u100', 'v100')):
            speed = interpolate_wind(data[FIELDS.index(a)], data[FIELDS.index(b)], int(h))
            values.append(float((w * speed).sum()) if np.isfinite(speed).all() else np.nan)
        end = 3 * (int(h) // 3) + 3
        block, log = radiation_block(data[FIELDS.index('dswrf')], bounds, quanta, end)
        notes.update({f'dswrf_{k}': v for k, v in log.items()})
        if block is None:
            status = 'invalid_support'
            values.append(np.nan)
        else:
            values.append(float((w * block).sum()) if np.isfinite(block).all() else np.nan)
        if status == 'ok' and not np.isfinite(values).all():
            status = 'invalid_support'
            notes['reason'] = 'nonfinite required support'
        if status != 'ok':  # all three derived columns go missing together
            values = [np.nan, np.nan, np.nan]
        rows.append({**out, **dict(zip(COLUMNS, values)), 'status': status, **notes})
    return pd.DataFrame(rows)


def missing_day(delivery: date, status: str, reason: str) -> pd.DataFrame:
    if status not in MISSING_CLASSES:
        raise ConversionRefused(f'{status} is not an imputable missing-weather class')
    return pd.DataFrame([{'delivery_date': delivery, 'timestamp_utc': t, 'lead_hour': int(h),
                          **{c: np.nan for c in COLUMNS}, 'status': status, 'reason': reason}
                         for t, h in zip(day_hours(delivery), hour_leads(delivery))])


def load_run(weather_dir: Path, run_00z: str):
    record = json.loads((weather_dir / 'runs' / f'{run_00z}.json').read_text())
    arrays = (weather_dir / 'runs' / f'{run_00z}.npz').read_bytes()
    if record.get('status') != 'complete' or hashlib.sha256(arrays).hexdigest() != record['npz_sha256']:
        raise ConversionRefused(f'{run_00z}: extraction artifact incomplete or altered')
    with np.load(weather_dir / 'runs' / f'{run_00z}.npz') as z:
        data = z['data']
        if tuple(z['fields']) != FIELDS or tuple(z['leads']) != LEADS or not np.array_equal(z['lats'], BOX_LATS) \
                or not np.array_equal(z['lons'], BOX_LONS):
            raise ConversionRefused(f'{run_00z}: array layout mismatch')
    dsw = {m['lead']: m for m in record['messages'] if m['field'] == 'dswrf'}
    bounds = [(dsw[l]['meta']['startStep'], dsw[l]['meta']['endStep']) for l in LEADS]
    quanta = [dsw[l]['packing_quantum'] for l in LEADS]
    return record, data, bounds, quanta


def build_features(manifest: dict, weather_dir: Path, classifications: dict | None = None):
    """Canonical-hour features for every required delivery day. Unfinished extraction refuses."""
    classifications = classifications or {}
    frames = []
    for s in manifest['structural_missing']:
        frames.append(missing_day(date.fromisoformat(s['delivery_day']), 'structural_missing_pre_2019_run',
                                  'D-1 00 UTC run would be 2018-12-31, before the 2019 input floor'))
    for r in manifest['runs']:
        delivery = date.fromisoformat(r['delivery_day'])
        if r['run_00z'] in classifications:
            status, reason = classifications[r['run_00z']]
            frame = missing_day(delivery, status, reason)
        else:
            path = weather_dir / 'runs' / f'{r["run_00z"]}.json'
            if not path.exists():
                raise ConversionRefused(f'{r["run_00z"]}: extraction unfinished; not imputable missing weather')
            _, data, bounds, quanta = load_run(weather_dir, r['run_00z'])
            frame = convert_run(delivery, data, bounds, quanta)
        frame['run_00z'] = r['run_00z']
        frame['version'] = r['version']
        frames.append(frame)
    features = pd.concat(frames, ignore_index=True)
    features['timestamp_utc'] = pd.to_datetime(features.timestamp_utc, utc=True)
    local = features.timestamp_utc.dt.tz_convert('Europe/Berlin')
    features['local_hour'] = local.dt.hour
    if not (local.dt.date == features.delivery_date).all() or features.timestamp_utc.duplicated().any():
        raise ConversionRefused('canonical hour identity failed')
    return features.sort_values('timestamp_utc').reset_index(drop=True)


def local_hour_design(features: pd.DataFrame) -> pd.DataFrame:
    """Weather per (delivery_date, local_hour): repeated autumn hours average both canonical
    vectors and require both; any missing constituent leaves the local-hour vector missing."""
    g = features.groupby(['delivery_date', 'local_hour'])
    design = g[list(COLUMNS)].mean()
    complete = g[list(COLUMNS)].apply(lambda x: bool(np.isfinite(x.to_numpy()).all()))
    design.loc[~complete, list(COLUMNS)] = np.nan
    design['n_canonical'] = g.size()
    return design


@dataclass(frozen=True)
class WeatherDesign:
    table: pd.DataFrame
    sha256: str

    @classmethod
    def from_features(cls, features: pd.DataFrame):
        table = local_hour_design(features)
        digest = hashlib.sha256(np.ascontiguousarray(table[list(COLUMNS)].to_numpy(float)).tobytes()
                                + json.dumps([str(i) for i in table.index]).encode()).hexdigest()
        return cls(table, digest)

    def present(self, dates: np.ndarray, hours: np.ndarray) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([pd.Index(pd.to_datetime(dates).date), hours])
        return np.asarray(keys.isin(self.table.index))

    def matrix(self, dates: np.ndarray, hours: np.ndarray, required: np.ndarray) -> np.ndarray:
        """Rows aligned to the modelling inputs; ``required`` rows must have a weather record."""
        keys = pd.MultiIndex.from_arrays([pd.Index(pd.to_datetime(dates).date), hours])
        present = keys.isin(self.table.index)
        if (required & ~present).any():
            raise ConversionRefused('a training/forecast row has no frozen weather record')
        out = np.full((len(dates), len(COLUMNS)), np.nan)
        out[present] = self.table.loc[keys[present], list(COLUMNS)].to_numpy(float)
        return out
