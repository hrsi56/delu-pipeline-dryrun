"""DDNN-2's day-level representation (capstone v21-r11 §23.3, §23.4).

**One row per delivery day D**, with whole-day inputs by Europe/Berlin local hour under the LEAR
design's convention (`src/cp15/data.py`): a repeated local hour averages its two observations, and a
missing local hour stays missing (NaN) until training-only imputation. Exactly v4's sources (§17.3),
in DDNN-2's day-level form:

| Group | Columns | Always / optional |
|---|---|---|
| `price_d1` | the D-1 price curve (24) | always |
| `load_d0` | the TSO day-ahead load forecast for D (24) | always |
| `weekday` | LEAR's seven weekday dummies of D | always |
| `price_d2`, `price_d3`, `price_d7` | the D-2, D-3, D-7 price curves (24 each) | optional |
| `load_d1`, `load_d7` | the load forecasts for D-1 and D-7 (24 each) | optional |
| `gfs` | the three frozen GFS columns for D's 24 local hours (72) and their 72 missing indicators (§15.3) | optional |
| `stats` | CP-15's LightGBM price statistics at the origin: over 168 h the mean, SD, 5/50/95% quantiles and negative-price count; over 720 h the mean and SD (8) | optional |
| `calendar` | CP-15's LightGBM day-type and month set: federal holiday, day after a holiday, bridge day, day type (3 one-hot), DST-transition day, month (12 one-hot) (19) | optional |

The optional groups are §23.4's inclusion flags; the search chooses among them on training data only.

**Target and price inputs** (§23.3, §4). Each row's centre c_d and scale s_d are its own origin
statistics, from prices delivered up to the day before d:

* `s4`: §4's level and scale, the mean and the sample SD (floor 1 EUR/MWh) of the 168 canonical
  prices before d's local midnight (`data.level`, `data.scale`);
* `mad`: the median of the same 168 prices and 1.4826 x their median absolute deviation, floor
  1 EUR/MWh (Uniejewski, Weron and Ziel, 2018).

z = (price - c_d) / s_d. The target is t = z or asinh(z) (four forms with the two statistics). The
price curves and the price-location statistics enter as the same transform of (value - c_d) / s_d;
the two SD statistics as value / s_d; the negative-price count as is. Load and weather enter in
their units. Quantiles are inverted with the forecast origin's statistics.

**Targets.** Slot h of day d is the mean of d's eligible observations at local hour h (two on the
repeated hour of a 25-hour day); a slot without one is masked (the missing hour of a 23-hour day).

**Preprocessing, fitted on the fit's own training rows only** (never its held-out weeks or forecast
rows): every continuous column is winsorised at its training rows' 0.5% and 99.5% quantiles (binary
columns and missing indicators never are); then missing values take the training median (an
all-missing column 0), with the weather group's missing indicators; then a StandardScaler over every
column (a zero-variance column keeps scale 1). Every winsorisation activation is counted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import math
import warnings

import numpy as np
import pandas as pd

from cp15.data import array_hash
from delu_forecast.ingest import BERLIN

ALWAYS = ('price_d1', 'load_d0', 'weekday')
OPTIONAL = ('price_d2', 'price_d3', 'price_d7', 'load_d1', 'load_d7', 'gfs', 'stats', 'calendar')
PRICE_LAGS = {'price_d1': 1, 'price_d2': 2, 'price_d3': 3, 'price_d7': 7}
LOAD_LAGS = {'load_d0': 0, 'load_d1': 1, 'load_d7': 7}
STAT_LOCATION = ('price_roll_mean_168h', 'price_roll_q05_168h', 'price_roll_q50_168h', 'price_roll_q95_168h',
                 'price_roll_mean_720h')
STAT_SCALE = ('price_roll_std_168h', 'price_roll_std_720h')
STAT_COUNT = ('negative_price_count_168h',)
CAL_FLAGS = ('is_federal_holiday', 'is_day_after_holiday', 'is_bridge_day', 'dst_transition_day')
MAD_CONSISTENCY = 1.482602218505602   # 1 / Phi^-1(0.75)
SCALE_FLOOR = 1.0
WINSOR = (0.005, 0.995)
WINDOW_DAYS = 728
FLOOR = date(2019, 1, 1)
HOLDOUT_SHARE = 0.20
RECENT_EXCLUDED_DAYS = 7
MIN_TRAIN_DAYS = 250
MIN_HOLD_DAYS = 14


class InsufficientRows(ValueError):
    pass


@dataclass
class DayData:
    """Day-level arrays over every calendar day from the data's first to its last delivery date."""
    days: np.ndarray            # datetime64[D], consecutive
    price: np.ndarray           # n x 24, local-hour mean price (NaN where absent)
    load: np.ndarray            # n x 24, local-hour mean load forecast
    weather: np.ndarray         # n x 24 x 3, the frozen GFS columns (NaN where missing or absent)
    covered: np.ndarray         # n, every canonical hour of the day has a frozen weather record
    centre: dict                # {'s4': n, 'mad': n} origin centres (NaN where undefined)
    scale: dict                 # {'s4': n, 'mad': n} origin scales
    stats: np.ndarray           # n x 8, in STAT_LOCATION + STAT_SCALE + STAT_COUNT order
    weekday: np.ndarray         # n x 7
    calendar: np.ndarray        # n x 19
    y: np.ndarray               # n x 24 slot targets (NaN where masked)
    mask: np.ndarray            # n x 24, slot has an eligible target
    first: date

    def ix(self, d: date) -> int:
        i = (np.datetime64(d, 'D') - self.days[0]).astype(int)
        if not 0 <= i < len(self.days):
            raise KeyError(f'{d} outside the day table')
        return int(i)

    def date_of(self, i: int) -> date:
        return pd.Timestamp(self.days[i]).date()

    def sha256(self) -> str:
        h = hashlib.sha256()
        for a in (self.days.astype('int64'), self.price, self.load, self.weather, self.covered, self.centre['s4'],
                  self.scale['s4'], self.centre['mad'], self.scale['mad'], self.stats, self.weekday, self.calendar,
                  self.y, self.mask):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def _local_hour_table(dates, hours, values, days) -> np.ndarray:
    frame = pd.DataFrame({'day': pd.Index(pd.to_datetime(dates).date), 'hour': hours, 'v': values})
    table = frame.pivot_table(index='day', columns='hour', values='v', aggfunc='mean', dropna=False)
    return table.reindex(index=pd.Index(pd.to_datetime(days).date), columns=range(24)).to_numpy(float)


def _mad_statistics(data, days) -> tuple[np.ndarray, np.ndarray]:
    """Median and 1.4826 x MAD (floor 1) of the 168 canonical prices before each day's local midnight."""
    price = pd.Series(data.frame.price_eur_mwh.to_numpy(float), index=data.index)
    start = data.index.min() - pd.Timedelta(hours=168)
    full = price.reindex(pd.date_range(start, data.index.max(), freq='h', tz='UTC')).to_numpy(float)
    origin = pd.DatetimeIndex([pd.Timestamp(pd.Timestamp(d).date(), tz=BERLIN) for d in days]).tz_convert('UTC')
    pos = ((origin - start) / pd.Timedelta(hours=1)).to_numpy().astype(int)
    centre, scale = np.full(len(days), np.nan), np.full(len(days), np.nan)
    for i, p in enumerate(pos):
        if p - 168 < 0 or p > len(full):
            continue
        window = full[p - 168:p]
        if len(window) != 168 or not np.isfinite(window).all():
            continue
        m = float(np.median(window))
        centre[i] = m
        scale[i] = max(MAD_CONSISTENCY * float(np.median(np.abs(window - m))), SCALE_FLOOR)
    return centre, scale


def build(data, design) -> DayData:
    """The day table from CP-15's prepared inputs and CP-20's frozen weather design."""
    first, last = pd.Timestamp(data.dates.min()).date(), pd.Timestamp(data.dates.max()).date()
    days = np.arange(np.datetime64(first, 'D'), np.datetime64(last, 'D') + 1)
    n = len(days)
    price = _local_hour_table(data.dates, data.hours, data.frame.price_eur_mwh.to_numpy(float), days)
    load = _local_hour_table(data.dates, data.hours, data.frame.load_forecast_mw.to_numpy(float), days)
    table = design.table
    weather = np.full((n, 24, 3), np.nan)
    keys = pd.MultiIndex.from_arrays([pd.Index(pd.to_datetime(np.repeat(days, 24)).date), np.tile(np.arange(24), n)])
    present_key = keys.isin(table.index)
    cols = ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean']
    vals = np.full((n * 24, 3), np.nan)
    vals[present_key] = table.loc[keys[present_key], cols].to_numpy(float)
    weather[:] = vals.reshape(n, 24, 3)
    # Coverage: every canonical hour of the day has a frozen weather record.
    row_present = design.present(data.dates, data.hours)
    day_of_row = (data.dates - days[0]).astype(int)
    rows_per_day = np.bincount(day_of_row, minlength=n)
    present_per_day = np.bincount(day_of_row, weights=row_present.astype(float), minlength=n)
    covered = (rows_per_day > 0) & (present_per_day == rows_per_day)
    # Origin statistics (constant within a day; checked).
    centre4, scale4 = np.full(n, np.nan), np.full(n, np.nan)
    feats = data.p['lgbm']['features']
    col = {name: feats.index(name) for name in feats}
    stats = np.full((n, len(STAT_LOCATION) + len(STAT_SCALE) + len(STAT_COUNT)), np.nan)
    weekday = np.full((n, 7), np.nan)
    calendar = np.full((n, len(CAL_FLAGS) + 3 + 12), np.nan)
    first_row = np.full(n, -1)
    order = np.argsort(day_of_row, kind='stable')
    seen = np.zeros(n, bool)
    for r in order:
        i = day_of_row[r]
        if not seen[i]:
            seen[i], first_row[i] = True, r
    for i in np.flatnonzero(first_row >= 0):
        r = first_row[i]
        centre4[i], scale4[i] = data.level[r], data.scale[r]
        stats[i] = [data.lgbm_raw[r, col[c]] for c in STAT_LOCATION + STAT_SCALE + STAT_COUNT]
        weekday[i] = data.lear_raw[r, 168:175]
        flags = [data.lgbm_raw[r, col[c]] for c in CAL_FLAGS]
        dtype = np.eye(3)[int(data.lgbm_raw[r, col['day_type']])]
        month = np.eye(12)[int(data.lgbm_raw[r, col['month']]) - 1]
        calendar[i] = np.concatenate([flags, dtype, month])
    # The origin statistics must be identical across a day's rows (they are day-level by construction).
    for name, per_row, per_day in (('level', data.level, centre4), ('scale', data.scale, scale4)):
        same = np.isclose(per_row, per_day[day_of_row], rtol=0, atol=0) | (np.isnan(per_row) & np.isnan(per_day[day_of_row]))
        if not same.all():
            raise ValueError(f'origin {name} differs within a delivery day')
    mad_c, mad_s = _mad_statistics(data, days)
    y = _local_hour_table(data.dates[data.eligible], data.hours[data.eligible], data.y[data.eligible], days)
    mask = np.isfinite(y)
    return DayData(days, price, load, weather, covered, {'s4': centre4, 'mad': mad_c}, {'s4': scale4, 'mad': mad_s},
                   stats, weekday, calendar, y, mask, first)


def shifted(a: np.ndarray, lag: int) -> np.ndarray:
    """a[d - lag] for every day d (NaN before the first day)."""
    if lag == 0:
        return a
    out = np.full_like(a, np.nan)
    out[lag:] = a[:-lag]
    return out


def vst(form: str, x: np.ndarray) -> np.ndarray:
    return np.arcsinh(x) if form.startswith('asinh') else x


def raw_inputs(dd: DayData, ix: np.ndarray, groups: tuple[str, ...], form: str):
    """Continuous block, binary block, weather missing indicators and column names for days `ix`."""
    stat = form.split('-')[1]
    c, s = dd.centre[stat][ix], dd.scale[stat][ix]
    cont, cont_names, binary, bin_names = [], [], [], []
    wanted = set(ALWAYS) | set(groups)
    for g, lag in PRICE_LAGS.items():
        if g in wanted:
            v = shifted(dd.price, lag)[ix]
            cont.append(vst(form, (v - c[:, None]) / s[:, None]))
            cont_names += [f'{g}_h{h:02d}' for h in range(24)]
    for g, lag in LOAD_LAGS.items():
        if g in wanted:
            cont.append(shifted(dd.load, lag)[ix])
            cont_names += [f'{g}_h{h:02d}' for h in range(24)]
    missing, gfs_names = None, []
    if 'gfs' in wanted:
        w = dd.weather[ix].reshape(len(ix), 72)
        cont.append(w)
        gfs_names = [f'gfs_h{h:02d}_{k}' for h in range(24) for k in ('wind10', 'wind100', 'dswrf')]
        cont_names += gfs_names
        missing = ~np.isfinite(w)
    if 'stats' in wanted:
        st = dd.stats[ix]
        nl, ns = len(STAT_LOCATION), len(STAT_SCALE)
        cont.append(vst(form, (st[:, :nl] - c[:, None]) / s[:, None]))
        cont.append(st[:, nl:nl + ns] / s[:, None])
        cont.append(st[:, nl + ns:])
        cont_names += list(STAT_LOCATION + STAT_SCALE + STAT_COUNT)
    binary.append(dd.weekday[ix])
    bin_names += [f'weekday_{k}' for k in range(7)]
    if 'calendar' in wanted:
        binary.append(dd.calendar[ix])
        bin_names += list(CAL_FLAGS) + [f'day_type_{k}' for k in range(3)] + [f'month_{k}' for k in range(1, 13)]
    if missing is not None:
        binary.append(missing.astype(float))
        bin_names += [f'missing_{n}' for n in gfs_names]
    C = np.column_stack(cont) if cont else np.zeros((len(ix), 0))
    B = np.column_stack(binary)
    return C, B, cont_names, bin_names


class Preprocessor:
    """Winsorise (continuous only), impute (training medians; all-missing -> 0) and standardise, every
    statistic fitted on the training rows only."""

    def __init__(self, C: np.ndarray, B: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            q = np.nanquantile(C, WINSOR, axis=0, method='linear') if C.shape[1] else np.zeros((2, 0))
        self.lo, self.hi = q[0], q[1]
        clipped = self._clip(C)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            fill = np.nanmedian(clipped, axis=0) if C.shape[1] else np.zeros(0)
        self.fill = np.where(np.isfinite(fill), fill, 0.0)
        if not np.isfinite(B).all():
            raise ValueError('a binary input is missing')
        full = np.column_stack((self._impute(clipped), B))
        self.mean = full.mean(axis=0)
        sd = full.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)

    def _clip(self, C):
        lo = np.where(np.isfinite(self.lo), self.lo, -np.inf)
        hi = np.where(np.isfinite(self.hi), self.hi, np.inf)
        return np.clip(C, lo, hi)  # NaN stays NaN

    def _impute(self, C):
        return np.where(np.isfinite(C), C, self.fill)

    def activations(self, C) -> dict:
        lo = np.isfinite(C) & (C < self.lo)
        hi = np.isfinite(C) & (C > self.hi)
        return {'winsor_low': int(lo.sum()), 'winsor_high': int(hi.sum()), 'values': int(np.isfinite(C).sum())}

    def transform(self, C, B) -> np.ndarray:
        out = (np.column_stack((self._impute(self._clip(C)), B)) - self.mean) / self.sd
        if not np.isfinite(out).all():
            raise ValueError('nonfinite transformed input')
        return out

    def record(self) -> dict:
        return {'winsor_lo_sha256': array_hash(self.lo), 'winsor_hi_sha256': array_hash(self.hi),
                'fill_sha256': array_hash(self.fill), 'mean_sha256': array_hash(self.mean), 'sd_sha256': array_hash(self.sd)}


def target_z(dd: DayData, ix: np.ndarray, stat: str) -> np.ndarray:
    return (dd.y[ix] - dd.centre[stat][ix, None]) / dd.scale[stat][ix, None]


def window(dd: DayData, origin: date, *, exclude_uncovered: bool) -> tuple[date, np.ndarray, list]:
    """Training-window day indices [max(2019-01-01, D-728), D) with at least one eligible target slot
    and finite origin statistics; pre-fold fits also leave out every day without a frozen weather
    record (§23.6). Returns the window start, the day indices and the uncovered days left out."""
    lower = max(FLOOR, origin - timedelta(days=WINDOW_DAYS))
    a, b = dd.ix(lower) if lower >= dd.first else 0, dd.ix(origin - timedelta(days=1)) + 1
    ix = np.arange(a, b)
    ok = dd.mask[ix].any(axis=1) & np.isfinite(dd.centre['s4'][ix]) & np.isfinite(dd.centre['mad'][ix]) \
        & np.isfinite(dd.scale['s4'][ix]) & np.isfinite(dd.scale['mad'][ix])
    excluded = []
    if exclude_uncovered:
        unc = ok & ~dd.covered[ix]
        excluded = [str(dd.date_of(i)) for i in ix[unc]]
        ok &= dd.covered[ix]
    return lower, ix[ok], excluded


def holdout_weeks(dd: DayData, lower: date, origin: date, seed: int) -> tuple[list[date], int]:
    """The member's seeded random 20% of the window's whole calendar (Monday-Sunday) weeks that lie in
    [lower, D-7): the pool excludes the most recent 7 days."""
    end = origin - timedelta(days=RECENT_EXCLUDED_DAYS)
    monday = lower + timedelta(days=(7 - lower.weekday()) % 7)
    pool = []
    while monday + timedelta(days=6) < end:
        pool.append(monday)
        monday += timedelta(days=7)
    k = max(1, int(math.floor(HOLDOUT_SHARE * len(pool) + 0.5)))
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), 0x5EED])))
    chosen = sorted(pool[i] for i in rng.choice(len(pool), size=k, replace=False))
    return chosen, len(pool)


def split(dd: DayData, ix: np.ndarray, weeks: list[date]) -> tuple[np.ndarray, np.ndarray]:
    held = np.zeros(len(ix), bool)
    for monday in weeks:
        a = np.datetime64(monday, 'D')
        held |= (dd.days[ix] >= a) & (dd.days[ix] < a + 7)
    return ix[~held], ix[held]


def recency_weights(dd: DayData, ix: np.ndarray, origin: date, half_life) -> np.ndarray:
    if not half_life:
        return np.ones(len(ix))
    age = (np.datetime64(origin, 'D') - dd.days[ix]).astype(float)
    w = 0.5 ** (age / float(half_life))
    return w / w.mean()
