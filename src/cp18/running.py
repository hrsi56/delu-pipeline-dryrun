"""The live record's running means with their 95% intervals (the Orchestrator's ruling O33, CP-18c).

For the issued days of the live record, in delivery order, after each scored day n:
- the running mean of the product's daily MAE and of its daily interval score (WIS), EUR/MWh;
- the running mean of the paired difference against the similar-day naive (B0), MAE and WIS, on the days both scored;
- each with a 95% interval from n = 14 scored days (two 7-day blocks); before that the running mean stands alone,
  with "interval from 14 scored days".

**The interval method is 4.7T's registered dependence treatment** (the protocol's §6; `cp17.tools.fresh_test`): CP-26's
paired, non-circular, moving-block studentized bootstrap as CP-28 landed it (`cp28.inference`), with 7-day blocks,
10,000 replicates, Bartlett HAC at lag 14, calendar runs split at any missing date and stratified by calendar year, and
a fresh `default_rng(fresh_test.SEED)` per series and n. The interval is the studentized (percentile-t) one that the
frozen replicate statistics give: `[mean − q₀.₉₇₅·√V, mean − q₀.₀₂₅·√V]`, where `V` is the HAC variance of the mean and
`q` the quantiles of the replicate statistics `(mean_b − mean)/√V_b`.

**Through the frozen code.** `cp28.inference.test_endpoint` admits every date through `cp25.guard`, which refuses any
day after 2026-09-06, and `cp28` is landed code this round never changes. So the runs are computed here by a
transcription of `cp28.inference.calendar_runs` without that admission, and everything else is `cp28`'s own:
`draw_plan`, `DrawPlan.indices`, `hac_variance`, `CHUNK`. `tests/cp18c/test_running.py` shows the replicate statistics,
the variance and the mean bit-identical to `test_endpoint`'s on development dates.

**Wording** (O33): "95% interval of the mean daily error over n scored days"; never "confidence", never a percentage
correct, never 4.7T's or a test result: the live record is "prospective evaluation in progress". It never pools with
development or with 4.7T's windows: only the prospective window's days enter.
"""
from __future__ import annotations

from datetime import date
import math

import numpy as np

from cp17.tools import fresh_test as F
from cp28 import inference as I

SCHEMA = 'cp18c-running-intervals-v1'
LEVEL = 0.95
MINIMUM_DAYS = 2 * F.BLOCK_DAYS
WORDING = '95% interval of the mean daily error over {n} scored days'
BEFORE_MINIMUM = f'interval from {MINIMUM_DAYS} scored days'
LABEL = 'prospective evaluation in progress'
SERIES = ('mae', 'wis', 'mae_vs_B0', 'wis_vs_B0')


def calendar_runs(dates, *, stratified=True):
    """`cp28.inference.calendar_runs`, transcribed without its admission of each date through `cp25.guard`."""
    days = list(dates)
    if days != sorted(set(days)):
        raise ValueError('paired dates must be unique and chronological')
    runs, start, prior = [], 0, None
    for i, day in enumerate(days):
        stratum = max(2021, day.year) if stratified else 0
        if i and ((day - days[i - 1]).days != 1 or stratum != prior):
            runs.append((start, i, prior))
            start = i
        prior = stratum
    if days:
        runs.append((start, len(days), prior))
    return tuple(runs)


def studentized(d, dates, *, replicates=F.REPLICATES, seed=F.SEED) -> dict:
    """`cp28.inference.test_endpoint`'s computation for one series (its 7-day blocks, stratified runs, HAC variance and
    replicate statistics), through `cp28`'s own functions; the runs from `calendar_runs` above."""
    d = np.asarray(d, float)
    out = {'n': len(d), 'mean': float(d.mean()) if len(d) else math.nan, 'variance': math.nan, 'stats': None,
           'status': None}
    if len(d) < 2 or not np.isfinite(d).all():
        out['status'] = 'fewer_than_two_or_nonfinite'
        return out
    runs = calendar_runs(dates, stratified=True)
    variance = float(I.hac_variance(d, runs))
    out['variance'] = variance
    if not np.isfinite(variance) or variance <= 0:
        out['status'] = 'nonpositive_or_nonfinite_variance'
        return out
    mean = d.mean()
    plan = I.draw_plan(runs, block_days=F.BLOCK_DAYS, rng=np.random.default_rng(seed), replicates=replicates)
    stats = np.full(replicates, np.nan)
    for first in range(0, replicates, I.CHUNK):
        last = min(first + I.CHUNK, replicates)
        sample = d[plan.indices(first, last)]
        v = I.hac_variance(sample, runs)
        good = np.isfinite(v) & (v > 0)
        stats[first:last][good] = (sample.mean(axis=1)[good] - mean) / np.sqrt(v[good])
    out['stats'] = stats
    out['status'] = 'computable' if np.isfinite(stats).all() else 'nonfinite_replicate_statistic'
    return out


def interval(d, dates, *, replicates=F.REPLICATES, seed=F.SEED) -> dict:
    """The running point after the last of `dates`: the mean, and its 95% interval from `MINIMUM_DAYS` on."""
    n = len(d)
    point = {'n': n, 'last_day': str(dates[-1]) if n else None, 'mean': float(np.mean(d)) if n else None,
             'lower': None, 'upper': None, 'interval': None}
    if n < MINIMUM_DAYS:
        point['interval'] = BEFORE_MINIMUM
        return point
    s = studentized(d, dates, replicates=replicates, seed=seed)
    if s['status'] != 'computable':
        point['interval'] = f"not computable ({s['status']})"
        return point
    lo_q, hi_q = np.quantile(s['stats'], [(1 + LEVEL) / 2, (1 - LEVEL) / 2])
    root = math.sqrt(s['variance'])
    point.update(lower=float(s['mean'] - lo_q * root), upper=float(s['mean'] - hi_q * root),
                 interval=WORDING.format(n=n))
    return point


def _daily(window: dict) -> dict[str, tuple[float, float]]:
    out = {}
    for item in window['record']['days']:
        mae, wis = item.get('mae_eur'), item.get('wis_eur')
        if isinstance(mae, (int, float)) and isinstance(wis, (int, float)) and np.isfinite(mae) and np.isfinite(wis):
            out[item['day']] = (float(mae), float(wis))
    return out


def running(windows: list[dict], *, replicates=F.REPLICATES) -> dict:
    """The four running series of the prospective window (the live record), from the bundle's evaluation windows."""
    chosen = {w['policy']: w for w in windows if w.get('window') == 'prospective'}
    if 'product' not in chosen:
        raise ValueError('no prospective window: the running intervals are the live record\'s only')
    product = _daily(chosen['product'])
    naive = _daily(chosen['B0']) if 'B0' in chosen else {}
    series = {name: [] for name in SERIES}
    days = sorted(product)
    for k in range(1, len(days) + 1):
        dates = [date.fromisoformat(x) for x in days[:k]]
        for j, name in enumerate(('mae', 'wis')):
            series[name].append(interval([product[x][j] for x in days[:k]], dates, replicates=replicates))
    paired = sorted(x for x in product if x in naive)
    for k in range(1, len(paired) + 1):
        dates = [date.fromisoformat(x) for x in paired[:k]]
        for j, name in enumerate(('mae_vs_B0', 'wis_vs_B0')):
            series[name].append(interval([product[x][j] - naive[x][j] for x in paired[:k]], dates,
                                         replicates=replicates))
    return {'schema': SCHEMA, 'label': LABEL, 'unit': 'EUR/MWh', 'level': LEVEL, 'minimum_days': MINIMUM_DAYS,
            'before_minimum': BEFORE_MINIMUM, 'wording': WORDING,
            'method': ("4.7T's registered dependence treatment (cp17.tools.fresh_test): cp28.inference's paired, "
                       "non-circular, moving-block studentized bootstrap, 7-day blocks, 10,000 replicates, Bartlett HAC "
                       "at lag 14, stratified calendar runs; the studentized (percentile-t) interval"),
            'block_days': F.BLOCK_DAYS, 'replicates': replicates, 'seed': F.SEED,
            'difference': 'the product minus the similar-day naive (B0), on the days both scored', 'series': series}
