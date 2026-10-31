"""§19.5's definitions, frozen at CP-17 before the fresh-data test, and the recomputation check.

The same functions compute the committed evaluation records (natively, in the pipeline) and recompute them
in the Space (in the browser), so the comparison is between one implementation and itself.

**Inputs.** A list of issued days, each with its UTC keys and its seven emitted quantiles (levels .025, .10,
.25, .50, .75, .90, .975), and the published outcomes as a mapping from key to price. A key whose price is
missing (not published, or not yet published) is unscored; a day with no published price is pending and
unscored. Zero and negative prices count as they are: no measure here divides by a price.

**The measures** (`evaluate`):

- **MAE** (EUR/MWh): each day's mean |y - p50| over its scored hours; days weigh equally (§24.7's rule).
- **WIS** (EUR/MWh): §24.7's seven-quantile WIS (50/80/95% intervals, weights α/2, median weight 1/2, divided
  by 3.5), `cp28.scoring.hourly`, aggregated as MAE is.
- **The percentage measure:** the share of scored hours with |p50 - y| <= τ, τ = 10 EUR/MWh (the Owner,
  decision 8). The denominator is the scored hours with a published price, pooled over the window. Partial
  days score their published hours, and each day's completeness (scored hours over its canonical hours) is
  shown. It is labelled "share of scored hours within ±10 EUR/MWh of the published price", never accuracy
  and never relative to a benchmark.
- **Coverage and width** per nominal band (50%: p25-p75; 80%: p10-p90; 95%: p025-p975): the share of scored
  hours with l <= y <= u, and the mean of u - l over the same hours, pooled. The rolling series uses the
  last 28 scored delivery days ending at each scored day.
- **The reliability diagram:** for each emitted level q, the share of scored hours with y <= the q forecast,
  with its counts.
- **The quantile-band PIT:** the share of scored hours in each of the eight bands the seven levels define,
  (-inf, q.025], (q.025, q.10], ..., (q.975, inf), against each band's nominal share, with counts. A price
  equal to a quantile falls in the band that quantile closes, as in the reliability diagram.

**The recomputation check** (`recompute_check`): counts match exactly; a real value a matches its committed
value b when |a - b| <= 1e-9 * max(1, |b|). Any mismatch is shown with both values and blocks that day's
publication as a product failure.
"""
from __future__ import annotations

import numpy as np

TAU_EUR = 10.0
LEVELS = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
LEVEL_NAMES = ('p025', 'p10', 'p25', 'p50', 'p75', 'p90', 'p975')
BANDS = {'50': (2, 4), '80': (1, 5), '95': (0, 6)}
PIT_NOMINAL = (0.025, 0.075, 0.15, 0.25, 0.25, 0.15, 0.075, 0.025)
ROLLING_DAYS = 28
RELATIVE_TOLERANCE = 1e-9
PERCENTAGE_LABEL = 'share of scored hours within ±10 EUR/MWh of the published price'


def _q(value, n):
    q = np.asarray(value, float)
    if q.shape != (n, 7) or not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
        raise ValueError('issued quantiles must be finite and ordered, one row per key')
    return q


def hourly(q, y):
    """Per-hour |y - p50| and seven-quantile WIS, by CP-28's landed arithmetic (§24.7)."""
    from cp28.scoring import hourly as landed
    return landed(q, y)


def day_record(day: str, keys: list[str], quantiles, outcomes: dict, canonical_hours: int, tau: float = TAU_EUR) -> dict:
    """One issued day's scored-hour statistics; unscored when no price is published yet."""
    q = _q(quantiles, len(keys))
    y = np.array([outcomes.get(k, np.nan) for k in keys], float)
    scored = np.isfinite(y)
    n = int(scored.sum())
    item = {'day': day, 'keys': len(keys), 'canonical_hours': int(canonical_hours), 'scored_hours': n,
            'completeness': n / canonical_hours if canonical_hours else 0.0, 'scored': n > 0}
    if not n:
        return item
    q, y = q[scored], y[scored]
    mae, wis = hourly(q, y)
    item.update({'mae_eur': float(mae.mean()), 'wis_eur': float(wis.mean()),
                 'within_tau_hours': int((np.abs(q[:, 3] - y) <= tau).sum()),
                 'at_or_below': [int((y <= q[:, j]).sum()) for j in range(7)],
                 'pit_counts': np.bincount((q < y[:, None]).sum(axis=1), minlength=8).astype(int).tolist(),
                 'covered': {b: int(((y >= q[:, lo]) & (y <= q[:, hi])).sum()) for b, (lo, hi) in BANDS.items()},
                 'width_sum_eur': {b: float((q[:, hi] - q[:, lo]).sum()) for b, (lo, hi) in BANDS.items()}})
    return item


def _pool(days: list[dict], tau: float) -> dict:
    scored = [d for d in days if d['scored']]
    hours = sum(d['scored_hours'] for d in scored)
    out = {'scored_days': len(scored), 'scored_hours': hours, 'pending_or_unscored_days': len(days) - len(scored),
           'partial_days': sum(1 for d in scored if d['scored_hours'] < d['canonical_hours'])}
    if not scored:
        return out
    within = sum(d['within_tau_hours'] for d in scored)
    below = [sum(d['at_or_below'][j] for d in scored) for j in range(7)]
    pit = [sum(d['pit_counts'][k] for d in scored) for k in range(8)]
    covered = {b: sum(d['covered'][b] for d in scored) for b in BANDS}
    width = {b: sum(d['width_sum_eur'][b] for d in scored) for b in BANDS}
    out.update({
        'mae_eur': float(np.mean([d['mae_eur'] for d in scored])),
        'wis_eur': float(np.mean([d['wis_eur'] for d in scored])),
        'percentage': {'label': PERCENTAGE_LABEL, 'tau_eur': tau, 'within_hours': within, 'scored_hours': hours,
                       'share': within / hours},
        'coverage': {b: {'nominal': float(b) / 100, 'covered_hours': covered[b], 'scored_hours': hours,
                         'share': covered[b] / hours, 'mean_width_eur': width[b] / hours} for b in BANDS},
        'reliability': [{'level': LEVELS[j], 'name': LEVEL_NAMES[j], 'at_or_below': below[j], 'scored_hours': hours,
                         'share': below[j] / hours} for j in range(7)],
        'pit': [{'band': k, 'nominal_share': PIT_NOMINAL[k], 'count': pit[k], 'share': pit[k] / hours} for k in range(8)]})
    return out


def rolling(days: list[dict], length: int = ROLLING_DAYS) -> list[dict]:
    """Coverage and mean width per band over the last `length` scored delivery days ending at each scored day."""
    scored = [d for d in days if d['scored']]
    series = []
    for i, d in enumerate(scored):
        window = scored[max(0, i - length + 1):i + 1]
        hours = sum(w['scored_hours'] for w in window)
        series.append({'day': d['day'], 'window_days': len(window), 'window_first_day': window[0]['day'],
                       'scored_hours': hours,
                       **{f'coverage{b}': sum(w['covered'][b] for w in window) / hours for b in BANDS},
                       **{f'width{b}_eur': sum(w['width_sum_eur'][b] for w in window) / hours for b in BANDS}})
    return series


def evaluate(issued: list[dict], outcomes: dict, tau: float = TAU_EUR) -> dict:
    """The evaluation record of one window. `issued` items: {'day', 'keys', 'quantiles', 'canonical_hours'}."""
    days = sorted((day_record(i['day'], i['keys'], i['quantiles'], outcomes, i['canonical_hours'], tau) for i in issued),
                  key=lambda d: d['day'])
    if len({d['day'] for d in days}) != len(days):
        raise ValueError('a delivery day is issued twice in one window')
    return {'schema': 'cp17-evaluation-v1', 'tau_eur': tau, 'rolling_days': ROLLING_DAYS,
            'pooled': _pool(days, tau), 'days': days, 'rolling': rolling(days)}


def flatten(value, prefix: str = '') -> dict:
    """Every leaf of a record as path -> value, for the item-by-item comparison."""
    out = {}
    if isinstance(value, dict):
        for k in sorted(value):
            out.update(flatten(value[k], f'{prefix}/{k}' if prefix else str(k)))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out.update(flatten(v, f'{prefix}[{i}]'))
    else:
        out[prefix] = value
    return out


def _matches(a, b) -> bool:
    """The committed value b decides: a count (an integer) must be equal exactly, a real within the frozen
    relative tolerance. A JSON hop through JavaScript may turn 1.0 into 1, so a's own type does not decide."""
    if isinstance(b, bool) or isinstance(a, bool):
        return a is b if isinstance(a, bool) and isinstance(b, bool) else False
    if isinstance(a, (int, float)) and isinstance(b, int):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, float):
        return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= RELATIVE_TOLERANCE * max(1.0, abs(b)))
    return a == b


def recompute_check(recomputed: dict, committed: dict) -> dict:
    """Counts exactly, reals within 1e-9 * max(1, |b|); every mismatch is listed, and any one blocks publication."""
    a, b = flatten(recomputed), flatten(committed)
    rows = []
    for path in sorted(set(a) | set(b)):
        if path not in a or path not in b:
            rows.append({'path': path, 'recomputed': a.get(path), 'committed': b.get(path), 'match': False,
                         'reason': 'missing'})
        elif not _matches(a[path], b[path]):
            rows.append({'path': path, 'recomputed': a[path], 'committed': b[path], 'match': False})
    return {'compared': len(set(a) | set(b)), 'mismatches': rows, 'match': not rows, 'blocks_publication': bool(rows),
            'rule': 'counts exactly; reals |a - b| <= 1e-9 * max(1, |b|)'}
