"""4.7T's frozen analysis (item 8), executable and outcome-blind: written before any day after 2026-09-06
exists, and tested on synthetic losses only. It is run once, after the 4.7T gate, on the window's losses.

- **Window:** delivery days 2026-09-07 to 2026-12-05 (§24.3), 90 days, one look.
- **Primary scores:** each date's p50 MAE and seven-quantile WIS (`cp28.scoring.daily`), averaged over the
  date's paired scoring keys; dates weigh equally.
- **Endpoints** (decision 5, mirroring CP-28's gates):
  - against the control v5: `d = L_A1 - 0.995 * L_v5`, H0 `E[d] >= 0`, for MAE and for WIS;
  - against B0: `d = L_A1 - L_B0`, H0 `E[d] >= 0`, for MAE and for WIS.
- **The compound p-value** is the largest of the four; 4.7T passes when it is <= 0.05. One candidate, no Holm.
- **The dependence treatment:** CP-26's paired, non-circular, moving-block studentized bootstrap as CP-28
  lands it (`cp28.inference.test_endpoint`): 7-day blocks, 10,000 replicates, Bartlett HAC at lag 14, one
  calendar stratum (all of 2026), runs split at any missing date; a fresh `default_rng(SEED)` for the
  analysis, endpoints visited in the order v5-MAE, v5-WIS, B0-MAE, B0-WIS.
- **The minimum scorable support:** each comparison needs at least 60 of the 90 dates paired and scored;
  below it, 4.7T is inconclusive, the window is spent, and nothing is extended or rerun.

The product's losses are those of what it emitted: A1 on a successful fit, the frozen fallback otherwise
(counted by cause), and missing on a failed issuance (missing in every pairing that involves it).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

WINDOW = (date(2026, 9, 7), date(2026, 12, 5))
SEED = 20261204
BLOCK_DAYS = 7
REPLICATES = 10_000
ALPHA = 0.05
IMPROVEMENT = 0.995
MIN_SUPPORT_DAYS = 60
ENDPOINTS = (('v5', 'MAE', IMPROVEMENT), ('v5', 'WIS', IMPROVEMENT), ('B0', 'MAE', 1.0), ('B0', 'WIS', 1.0))


def window_days(window=WINDOW) -> list[date]:
    first, last = window
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def daily_losses(issued: dict, outcomes: dict, a: str, b: str, window=WINDOW) -> dict:
    """Paired daily (MAE, WIS) of two policies: on each date, the keys both emitted that have a published,
    eligible price. `outcomes` holds only CP-25-eligible keys (an excluded or unpublished key is absent or NaN);
    `issued[policy][day] = (keys, quantiles)`; a policy without that day is missing."""
    from cp28.scoring import daily
    out = {}
    for day in window_days(window):
        x, y = issued.get(a, {}).get(str(day)), issued.get(b, {}).get(str(day))
        if x is None or y is None:
            continue
        common = [k for k in x[0] if k in set(y[0]) and np.isfinite(outcomes.get(k, np.nan))]
        if not common:
            continue
        ix, iy = [x[0].index(k) for k in common], [y[0].index(k) for k in common]
        truth = np.array([outcomes[k] for k in common], float)
        out[str(day)] = {a: daily(np.asarray(x[1], float)[ix], truth), b: daily(np.asarray(y[1], float)[iy], truth)}
    return out


def analyse(issued: dict, outcomes: dict, product: str = 'A1', window=WINDOW) -> dict:
    """The four endpoints, the compound p-value and the decision, from the window's emitted forecasts.

    The landed bootstrap admits every date through `cp25.guard`, so on the real window this refuses until the
    4.7T gate moves the boundary to 2026-12-05; the synthetic tests pass a development-dated window."""
    from cp28.inference import test_endpoint
    rng = np.random.default_rng(SEED)
    results, support = [], {}
    for comparator, metric, factor in ENDPOINTS:
        losses = daily_losses(issued, outcomes, product, comparator, window)
        days = sorted(losses)
        support[comparator] = len(days)
        j = 0 if metric == 'MAE' else 1
        d = np.array([losses[k][product][j] - factor * losses[k][comparator][j] for k in days], float)
        if len(days) < MIN_SUPPORT_DAYS:
            results.append({'comparator': comparator, 'metric': metric, 'n': len(days), 'p': 1.0,
                            'status': f'below_minimum_support_{MIN_SUPPORT_DAYS}'})
            continue
        r = test_endpoint(d, [date.fromisoformat(k) for k in days], block_days=BLOCK_DAYS, stratified=True, rng=rng,
                          required_cell=False, replicates=REPLICATES)
        results.append({'comparator': comparator, 'metric': metric, 'factor': factor, 'n': r['n'], 'mean_d': r['mean'],
                        'statistic': r['statistic'], 'p': r['p'], 'status': r['status']})
    compound = max(r['p'] for r in results)
    sufficient = all(n >= MIN_SUPPORT_DAYS for n in support.values())
    return {'endpoints': results, 'compound_p': compound, 'paired_days': support, 'support_sufficient': sufficient,
            'decision': ('pass' if compound <= ALPHA else 'fail') if sufficient else 'inconclusive (support)',
            'seed': SEED, 'block_days': BLOCK_DAYS, 'replicates': REPLICATES}
