"""The proposed §19.7 method for A1, a neural component: exact baseline Shapley over DDNN-2's input groups.

**What it attributes.** The issued p50 of each local-hour slot: the per-level median over the eight
members, as `cp24.ddnn2.ensemble_median` forms it. The players are the input groups (§23.3's
`price_d1`, `price_d2`, `price_d3`, `price_d7`, `load_d0`, `load_d1`, `load_d7`, `gfs` with its missing
indicators, `stats`, `weekday`, `calendar`), the union over the eight members' configurations; a
member that does not use a group is unaffected by it.

**The baseline.** A group "absent" takes each member's training-row mean: zero after the member's own
preprocessor, which centres every column on its training rows. So the attribution reads "relative to
an average training day", within one fitted artifact.

**The value.** v(S) = the issued p50 with the groups in S at the day's values and the others at the
baseline, computed by the landed forward pass, cap and ensemble median. Every coalition is evaluated
(2^G, G <= 11), so the Shapley values are exact, not sampled, and add up: sum_g phi_g = v(all) - v(none)
in every slot. Deterministic: no seed and no sampling.

It is presented as attribution within one fitted artifact, never as causality or incremental value
(§19.7).
"""
from __future__ import annotations

from datetime import date
from math import factorial
import time

import numpy as np

GROUPS = ('price_d1', 'price_d2', 'price_d3', 'price_d7', 'load_d0', 'load_d1', 'load_d7', 'gfs', 'stats',
          'weekday', 'calendar')


def column_group(name: str) -> str:
    from cp24 import design as G
    if name.startswith(('gfs_', 'missing_gfs_')):
        return 'gfs'
    if name.startswith(('price_d', 'load_d')):
        return name.split('_h', 1)[0]
    if name.startswith('weekday_'):
        return 'weekday'
    if name in G.STAT_LOCATION + G.STAT_SCALE + G.STAT_COUNT:
        return 'stats'
    if name in G.CAL_FLAGS or name.startswith(('day_type_', 'month_')):
        return 'calendar'
    raise ValueError(f'unassigned input column {name}')


def _member_inputs(dd, day: date, di: np.ndarray, fit: dict):
    """The member's forecast row through its own training-only preprocessor, rebuilt as fit_member built it."""
    from cp24 import design as G
    config = fit['config']
    weeks = [date.fromisoformat(w) for w in fit['window']['held_out_weeks']]
    _, ix, _ = G.window(dd, day, exclude_uncovered=False)
    train, _ = G.split(dd, ix, weeks)
    form, groups = config['transform'], tuple(config['groups'])
    C_tr, B_tr, cnames, bnames = G.raw_inputs(dd, train, groups, form)
    pre = G.Preprocessor(C_tr, B_tr)
    C_fc, B_fc, _, _ = G.raw_inputs(dd, di, groups, form)
    return pre.transform(C_fc, B_fc), [column_group(n) for n in cnames + bnames]


def group_shapley(dd, day: date, di: np.ndarray, fits: list[dict]) -> dict:
    from cp24 import ddnn2 as M
    started = time.perf_counter()
    rows, union = [], set()
    for fit in fits:
        x, groups = _member_inputs(dd, day, di, fit)
        zq, _ = M.member_quantiles_z(fit['params'], x, fit['config']['activation'], fit['config']['transform'], fit['cap_z'])
        eur = fit['centre'][:, None, None] + fit['scale'][:, None, None] * zq
        if not np.array_equal(eur, fit['eur']):
            raise ValueError('the rebuilt forecast row does not reproduce the member\'s emitted quantiles')
        rows.append((x, groups))
        union |= set(groups)
    players = [g for g in GROUPS if g in union]
    n = len(players)
    masks = np.arange(2 ** n)
    stack = []
    for fit, (x, groups) in zip(fits, rows):
        bit = np.array([players.index(g) for g in groups])
        keep = ((masks[:, None] >> bit[None, :]) & 1).astype(float)
        zq, _ = M.member_quantiles_z(fit['params'], x * keep, fit['config']['activation'], fit['config']['transform'],
                                     fit['cap_z'])
        stack.append(fit['centre'][0] + fit['scale'][0] * zq)               # coalitions x 24 x 7
    med, _ = M.ensemble_median(np.stack(stack))
    v = med[:, :, M.MEDIAN]                                                 # coalitions x 24
    size = np.array([bin(int(m)).count('1') for m in masks])
    weight = np.array([factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)])
    phi = np.zeros((n, v.shape[1]))
    for g in range(n):
        without = masks[(masks >> g) & 1 == 0]
        phi[g] = (weight[size[without]][:, None] * (v[without | (1 << g)] - v[without])).sum(axis=0)
    full, none = v[-1], v[0]
    return {'method': 'exact baseline Shapley over input groups, issued p50, training-mean baseline',
            'players': players, 'coalitions': int(2 ** n), 'phi_eur': phi.tolist(), 'baseline_p50': none.tolist(),
            'full_p50': full.tolist(), 'efficiency_max_abs': float(np.abs(phi.sum(axis=0) - (full - none)).max()),
            'seconds': time.perf_counter() - started}
