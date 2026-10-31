"""The twenty endpoints, the compound test, Holm, sensitivity and selection (protocol §§8–9).

The bootstrap is CP-26's paired non-circular moving-block studentized bootstrap, reimplemented
here rather than imported (`src/cp26` is never called by the run); `tests/cp28` checks its draw
plan and HAC variance against CP-26's functions on synthetic data.

* Runs: contiguous calendar runs of an endpoint's paired dates, split at any gap and, when
  stratified, at calendar-year boundaries (2020-12-29..31 join 2021). Blocks never cross a run.
* Draws: for each run, uniformly from its full non-circular blocks (a run shorter than the block
  is taken whole), appended until the run's count is reached, the last block truncated on the
  right. One `rng.integers` call per endpoint, shape (replicates, blocks), so the RNG order is
  replicate, then original run, then appended block.
* Variance: Bartlett HAC at lag 14 with weights 1−l/15, global centering, divisor n², counting
  only pairs exactly l days apart inside one run (pseudo-days inside a resampled run).
* Statistic: T = mean(d)/√V; replicate T_b = (mean(d_b) − mean(d))/√V_b; one-sided
  p = (1 + #{T_b ≤ T})/(R + 1). Fewer than two dates, a nonfinite or nonpositive variance, or a
  nonfinite replicate statistic makes the endpoint uncomputable (p = 1). An uncomputable endpoint,
  and a required year/season cell below 90 paired dates, consume no draw.
* RNG: a fresh `default_rng(20261012)` per variant; arms A1..A7; endpoints overall MAE, overall
  WIS, years ascending (MAE then WIS), seasons DJF/MAM/JJA/SON (MAE then WIS).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import numpy as np

from cp25 import guard

from .states import SEASON_ORDER, season

SEED = 20261012
REPLICATES = 10_000
HAC_LAG = 14
ALPHA = 0.05
IMPROVEMENT, NONINFERIORITY = 0.995, 1.02
YEARS = (2021, 2022, 2023, 2024, 2025)
METRICS = ('MAE', 'WIS')
MIN_CELL = 90
ARMS = ('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7')
SIMPLICITY = ('A1', 'A2', 'A3', 'A6', 'A4', 'A5', 'A7')
VARIANTS = (('primary14', 14, True), ('sensitivity7', 7, True), ('sensitivity28', 28, True), ('unstratified14', 14, False))
CHUNK = 500


def endpoints() -> list[dict]:
    out = [{'kind': 'improvement', 'group': 'overall', 'value': 'overall', 'metric': m, 'factor': IMPROVEMENT, 'required_cell': False}
           for m in METRICS]
    out += [{'kind': 'noninferiority', 'group': 'year', 'value': str(y), 'metric': m, 'factor': NONINFERIORITY, 'required_cell': True}
            for y in YEARS for m in METRICS]
    out += [{'kind': 'noninferiority', 'group': 'season', 'value': s, 'metric': m, 'factor': NONINFERIORITY, 'required_cell': True}
            for s in SEASON_ORDER for m in METRICS]
    for i, e in enumerate(out):
        e['endpoint'] = i
        e['name'] = f"{e['group']}:{e['value']}:{e['metric']}"
    assert len(out) == 20
    return out


def in_cell(day: date, group: str, value: str) -> bool:
    if group == 'overall':
        return True
    if group == 'year':
        return day.year == int(value)
    return season(day) == value


def calendar_runs(dates, *, stratified=True):
    days = list(dates)
    if days != sorted(set(days)):
        raise ValueError('paired dates must be unique and chronological')
    for day in days:
        guard.admit_delivery_day(day)
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


@dataclass(frozen=True)
class DrawPlan:
    runs: tuple
    block_days: int
    starts: np.ndarray            # (replicates, blocks), zero-based within the original run
    columns: tuple                # each run's [first, last) column

    @property
    def n(self):
        return self.runs[-1][1] if self.runs else 0

    def indices(self, first=0, last=None):
        last = len(self.starts) if last is None else last
        out = np.empty((last - first, self.n), dtype=np.int64)
        for (begin, end, _), (c0, c1) in zip(self.runs, self.columns):
            length = min(self.block_days, end - begin)
            for j, col in enumerate(range(c0, c1)):
                count = min(length, end - begin - j * length)
                at = begin + j * length
                out[:, at:at + count] = begin + self.starts[first:last, col, None] + np.arange(count)[None, :]
        return out


def draw_plan(runs, *, block_days, rng, replicates=REPLICATES) -> DrawPlan:
    columns, highs, col = [], [], 0
    for begin, end, _ in runs:
        n = end - begin
        length = min(block_days, n)
        blocks = math.ceil(n / length)
        columns.append((col, col + blocks))
        highs.extend([n - length + 1] * blocks)
        col += blocks
    starts = (rng.integers(0, np.array(highs, dtype=np.int64), size=(replicates, col), dtype=np.int32)
              if col else np.empty((replicates, 0), np.int32))
    return DrawPlan(tuple(runs), block_days, starts, tuple(columns))


def hac_variance(values, runs):
    """Bartlett-14 variance of the mean, divisor n², global centering, within-run pairs only."""
    x = np.asarray(values, float)
    single = x.ndim == 1
    x = x[None, :] if single else x
    n = x.shape[1]
    if n < 2:
        out = np.full(x.shape[0], np.nan)
        return out[0] if single else out
    z = x - x.mean(axis=1, keepdims=True)
    total = np.einsum('bn,bn->b', z, z)
    for lag in range(1, HAC_LAG + 1):
        weight = 1 - lag / (HAC_LAG + 1)
        for begin, end, _ in runs:
            if end - begin > lag:
                total = total + 2 * weight * np.einsum('bn,bn->b', z[:, begin + lag:end], z[:, begin:end - lag])
    out = total / (n * n)
    return out[0] if single else out


def test_endpoint(d, dates, *, block_days, stratified, rng, required_cell, replicates=REPLICATES):
    d = np.asarray(d, float)
    dates = list(dates)
    result = {'n': len(d), 'p': 1.0, 'status': None, 'mean': float(d.mean()) if len(d) else np.nan,
              'variance': np.nan, 'statistic': np.nan, 'lower_tail_count': None, 'starts': None, 'replicate_statistics': None,
              'runs': None}
    if required_cell and len(d) < MIN_CELL:
        result['status'] = f'deficient_cell_below_{MIN_CELL}'
        return result
    if len(d) < 2 or not np.isfinite(d).all():
        result['status'] = 'fewer_than_two_or_nonfinite'
        return result
    runs = calendar_runs(dates, stratified=stratified)
    variance = float(hac_variance(d, runs))
    result.update(variance=variance, runs=[list(r) for r in runs])
    if not np.isfinite(variance) or variance <= 0:
        result['status'] = 'nonpositive_or_nonfinite_variance'
        return result
    mean = d.mean()
    observed = mean / math.sqrt(variance)
    plan = draw_plan(runs, block_days=block_days, rng=rng, replicates=replicates)
    stats = np.full(replicates, np.nan)
    for first in range(0, replicates, CHUNK):
        last = min(first + CHUNK, replicates)
        sample = d[plan.indices(first, last)]
        v = hac_variance(sample, runs)
        good = np.isfinite(v) & (v > 0)
        stats[first:last][good] = (sample.mean(axis=1)[good] - mean) / np.sqrt(v[good])
    result.update(statistic=float(observed), starts=plan.starts, replicate_statistics=stats)
    if not np.isfinite(stats).all():
        result['status'] = 'nonfinite_replicate_statistic'
        return result
    count = int((stats <= observed).sum())
    result.update(p=(1 + count) / (replicates + 1), lower_tail_count=count, status='computable')
    return result


def paired(losses: dict, arm: str):
    """Dates where both the arm and the control have finite daily losses, chronological."""
    a, c = losses[arm], losses['v5']
    days = sorted(d for d in a if d in c and all(np.isfinite(a[d])) and all(np.isfinite(c[d])))
    return days


def run_variant(losses: dict, variant: tuple, *, replicates=REPLICATES, executed=None):
    """All seven arms' twenty endpoints for one analysis variant. ``losses[name][day] = (MAE, WIS)``."""
    name, block_days, stratified = variant
    rng = np.random.default_rng(SEED)
    executed = set(ARMS) if executed is None else set(executed)
    out = {}
    for arm in ARMS:
        rows = []
        if arm not in executed:
            out[arm] = {'endpoints': [], 'compound_p': 1.0, 'status': 'unexecuted'}
            continue
        days = paired(losses, arm)
        for e in endpoints():
            m = METRICS.index(e['metric'])
            cell = [d for d in days if in_cell(d, e['group'], e['value'])]
            diff = np.array([losses[arm][d][m] - e['factor'] * losses['v5'][d][m] for d in cell])
            r = test_endpoint(diff, cell, block_days=block_days, stratified=stratified, rng=rng,
                              required_cell=e['required_cell'], replicates=replicates)
            rows.append({**e, **r, 'arm': arm, 'variant': name})
        compound = max(r['p'] for r in rows)
        out[arm] = {'endpoints': rows, 'compound_p': compound, 'status': 'executed',
                    'first_unmet': next((r['name'] for r in rows if r['p'] == compound), None)}
    return out


def holm(pvalues: dict, alpha=ALPHA):
    order = sorted(ARMS, key=lambda a: (pvalues[a], ARMS.index(a)))
    k = len(ARMS)
    rejected, stop, running = {}, False, 0.0
    rows = []
    for j, arm in enumerate(order, start=1):
        threshold = alpha / (k + 1 - j)
        reject = (not stop) and pvalues[arm] <= threshold
        stop = stop or not reject
        running = max(running, (k + 1 - j) * pvalues[arm])
        rejected[arm] = reject
        rows.append({'position': j, 'arm': arm, 'p': pvalues[arm], 'threshold': threshold, 'rejected': reject,
                     'adjusted_p': min(1.0, running)})
    return rows


def select(passing: list[str], wis: dict):
    """Lowest mean daily WIS among passing arms; within 0.5% the simpler arm; none → the control."""
    if not passing:
        return {'selected': 'control', 'tie_set': [], 'best_arm': None, 'best_wis': None, 'reason': 'no arm passes'}
    best_arm = min(passing, key=lambda a: (wis[a], SIMPLICITY.index(a)))
    best = wis[best_arm]
    tie = [a for a in passing if (wis[a] < 1.005 * best if best > 0 else wis[a] == 0)]
    winner = min(tie, key=SIMPLICITY.index)
    return {'selected': winner, 'tie_set': sorted(tie, key=SIMPLICITY.index), 'best_arm': best_arm, 'best_wis': best,
            'tie_bound': 1.005 * best if best > 0 else 0.0,
            'reason': 'lowest WIS' if winner == best_arm else 'simpler arm within 0.5% of the best WIS'}


def ratio_interval(a, b, starts_plan_indices, runs):
    """Descriptive 95% studentized ratio interval mean(a)/mean(b) on the endpoint's own resamples."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mb = b.mean()
    if not np.isfinite(mb) or mb == 0:
        return {'ratio': np.nan, 'ci95': (np.nan, np.nan), 'status': 'undefined_denominator', 'pivots': np.empty(0)}
    ratio = a.mean() / mb
    v = float(hac_variance((a - ratio * b) / mb, runs))
    if not np.isfinite(v) or v <= 0:
        return {'ratio': ratio, 'ci95': (np.nan, np.nan), 'status': 'nonpositive_variance', 'pivots': np.empty(0)}
    pivots = []
    for idx in starts_plan_indices:
        ar, br = a[idx], b[idx]
        bm = br.mean(axis=1)
        rr = ar.mean(axis=1) / bm
        vb = hac_variance((ar - rr[:, None] * br) / bm[:, None], runs)
        pivots.append((rr - ratio) / np.sqrt(vb))
    pivots = np.concatenate(pivots)
    if not np.isfinite(pivots).all():
        return {'ratio': ratio, 'ci95': (np.nan, np.nan), 'status': 'nonfinite_replicate', 'pivots': pivots}
    lo, hi = np.quantile(pivots, [.025, .975], method='linear')
    se = math.sqrt(v)
    return {'ratio': float(ratio), 'ci95': (float(ratio - hi * se), float(ratio - lo * se)), 'status': 'computable',
            'pivots': pivots}
