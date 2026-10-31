"""§19.7's attribution methods, by component class, computed from one fitted artifact.

- **A1, a neural component** (the method CP-17 freezes): exact baseline Shapley over DDNN-2's input groups,
  on the issued p50, with each member's training-row mean as the baseline (`group_shapley`, CP-17-probe's
  method unchanged). At most 2^11 = 2,048 coalitions; every coalition is evaluated, so the values are exact
  and add up: sum_g phi_g = v(all) - v(none) in every slot. No seed, no sampling.
- **Linear components (LEAR):** coefficient times the centered, standardized input (`lear_linear`). The
  final Lasso of each local hour is refitted from the landed `cp15.models.prepared_linear` and `_lasso` at
  the recorded selected alpha, and must reproduce the recorded coefficient hash; the standardized row is the
  landed scaler's output, so contribution_j = coef_j * x_j and baseline = the intercept, exactly.
- **Tree components (LightGBM):** per-prediction TreeSHAP contributions (`pred_contrib`) and split gain,
  read from the fitted booster, refitted from the landed `cp21.lgbm._fit` at the recorded selected
  configuration and required to reproduce the recorded tree hash.
- **A blend:** each component's weight times its forecast, an exact decomposition (v2 = A1/2 + B2/2;
  v4 = A1_w/3 + B2_w/3 + L-N/6 + L-R/6).

The interval layers (H, and B0's raw errors) have no per-feature attribution; they are explained in words.
Everything is attribution within one fitted artifact, relative to the stated baseline, never causality or
incremental value (§19.7).
"""
from __future__ import annotations

from datetime import date
from math import factorial
import time

import numpy as np

from .dailyfit import FitFailure, array_hash, training_rows

GROUPS = ('price_d1', 'price_d2', 'price_d3', 'price_d7', 'load_d0', 'load_d1', 'load_d7', 'gfs', 'stats',
          'weekday', 'calendar')
MAX_COALITIONS = 2048
WORDING = ('attribution within one fitted artifact, relative to its stated baseline; '
           'never causality or incremental value')


# ------------------------------------------------------------------ A1: exact baseline Shapley over input groups
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
    if 2 ** n > MAX_COALITIONS:
        raise ValueError('more input groups than the frozen 2,048-coalition bound')
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
            'wording': WORDING, 'seconds': time.perf_counter() - started}


# ------------------------------------------------------------------ linear components: coefficient x standardized input
def lear_linear(data, day: date, policy: str, selections: list[dict], names: list[str]) -> dict:
    """One LEAR member's per-row contributions in EUR/MWh, hour by hour, from its refitted final Lasso."""
    import cp15.models as m
    p = data.p['lear']
    norm = policy.startswith('A')
    x = data.lear_normalized if norm else data.lear_raw
    target = (data.y - data.level) / data.scale if norm else data.y
    train = training_rows(data, day, policy)
    rows = data.rows(day)
    n_features = x.shape[1]
    columns = list(names) + [f'missing_{n}' for n in names]
    if len(columns) != 2 * n_features:
        raise ValueError('LEAR feature names do not match the design')
    contrib = np.zeros((len(rows), 2 * n_features))
    baseline = np.zeros(len(rows))
    for record in selections:
        hour = record['local_hour']
        tr = train[data.hours[train] == hour]
        mask = data.hours[rows] == hour
        selected_rows = rows[mask]
        xt, xp, _, _ = m.prepared_linear(x[tr], x[selected_rows])
        unit = max(float(np.std(target[tr])), 1e-8)
        final = m._lasso(xt, target[tr], record['selected_relative_alpha'] * unit, p)
        if array_hash(np.r_[final.coef_, final.intercept_]) != record['model_sha256']:
            raise FitFailure(f'LEAR {policy} hour {hour}: the refitted final Lasso differs from the issued one')
        part = xp * final.coef_[None, :]
        base = np.full(mask.sum(), float(final.intercept_))
        if norm:   # back to EUR/MWh: y = level + scale * (intercept + sum_j coef_j x_j)
            s = data.scale[selected_rows]
            part, base = part * s[:, None], base * s + data.level[selected_rows]
        contrib[mask], baseline[mask] = part, base
    return {'columns': columns, 'contributions_eur': contrib, 'baseline_eur': baseline,
            'baseline': 'the intercept at the standardized training mean' + (', scaled by the issued scale plus the '
                                                                               'level' if norm else '')}


def _lear_names(data, extra=()) -> list[str]:
    return list(data.p['lear']['features']) + list(extra)


def _top(columns, contrib_row, k=8):
    order = np.argsort(-np.abs(contrib_row))[:k]
    return [{'input': columns[i], 'eur': float(contrib_row[i])} for i in order if contrib_row[i] != 0]


def _summarize(parts: dict, weights: dict, central, index) -> dict:
    """Blend decomposition plus each component's own per-row contributions, top inputs per hour."""
    central = np.asarray(central, float)
    blend = {k: (weights[k] * np.asarray(v['forecast'], float)).tolist() for k, v in parts.items()}
    total = np.sum([np.asarray(b) for b in blend.values()], axis=0)
    rows = []
    for i, ts in enumerate(index):
        item = {'timestamp_utc': ts, 'central_eur': float(central[i]),
                'components': {k: {'weight': weights[k], 'forecast_eur': float(parts[k]['forecast'][i]),
                                   'weighted_eur': blend[k][i], 'baseline_eur': float(parts[k]['baseline_eur'][i]),
                                   'top_inputs': _top(parts[k]['columns'], parts[k]['contributions_eur'][i])}
                               for k in parts}}
        rows.append(item)
    checks = {k: float(np.abs(np.asarray(v['baseline_eur']) + np.asarray(v['contributions_eur']).sum(axis=1)
                              - np.asarray(v['forecast'])).max()) for k, v in parts.items()}
    return {'blend_weights': weights, 'blend_gap_max_abs': float(np.abs(total - central).max()),
            'component_efficiency_max_abs': checks, 'rows': rows}


def v2_linear(data, day: date, offset: int, result: dict) -> dict:
    """v2 = A1/2 + B2/2: the blend decomposition, then the linear rule for both LEAR members."""
    started = time.perf_counter()
    names = _lear_names(data)
    parts = {}
    for policy in ('A1', 'B2'):
        lin = lear_linear(data, day, policy, result['selections'][policy], names)
        parts[policy] = {**lin, 'forecast': result['members_central'][policy]}
    out = _summarize(parts, {'A1': 0.5, 'B2': 0.5}, result['central'], result['index'])
    out.update({'method': 'blend: weight x forecast; LEAR: coefficient x centered, standardized input',
                'interval_layer': 'the H layer: hour-aware quantiles of the last 28 released standardized errors, '
                                  'scaled by the issued scale; no per-input attribution',
                'wording': WORDING, 'seconds': time.perf_counter() - started})
    return out


# ------------------------------------------------------------------ tree components: TreeSHAP and gain
def lgbm_treeshap(data, wx, present, day: date, arm: str, selection: dict) -> dict:
    """One LightGBM member's per-row TreeSHAP contributions in EUR/MWh and each block model's split gain."""
    from cp21 import lgbm as L
    design = L.arm_design(data, wx, arm)
    rows = data.rows(day)
    names = list(data.p['lgbm']['features']) + ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean',
                                                 'missing_wx_wind10_mean', 'missing_wx_wind100_mean',
                                                 'missing_wx_dswrf_mean']
    contrib = np.zeros((len(rows), len(names)))
    baseline = np.zeros(len(rows))
    gain = {}
    finals = {f['model']: f for f in selection['fits'] if f['role'] == 'final'}
    for model in L.ARMS[arm]['models']:
        _, window, _, _, forecast = L.model_rows(data, day, model, present)
        config = next(c for c in L.GRID if c['id'] == selection['selected'][model])
        fitted, imputer, _, _ = L._fit(design, window, config, data.p, 1)
        if L._model_sha(fitted) != finals[model]['model_sha256']:
            raise FitFailure(f'{arm} {model}: the refitted final booster differs from the issued one')
        x = np.column_stack((design.x[forecast], imputer.transform(design.wx[forecast])))
        phi = np.asarray(fitted.predict(x, pred_contrib=True), float)
        part, base = phi[:, :-1], phi[:, -1]
        if design.normalized:   # back to EUR/MWh, as `cp21.lgbm._predict` maps the normalized target
            s, lvl = data.scale[forecast], data.level[forecast]
            part, base = part * s[:, None], base * s + lvl
        mask = np.isin(data.hours[rows], L.MODEL_HOURS[model])
        if not np.array_equal(rows[mask], forecast):
            raise ValueError('block forecast rows disagree with row order')
        contrib[mask], baseline[mask] = part, base
        gain[model] = dict(zip(names, map(float, fitted.booster_.feature_importance(importance_type='gain'))))
    return {'columns': names, 'contributions_eur': contrib, 'baseline_eur': baseline, 'gain': gain,
            'baseline': 'the booster\'s expected value over its training rows' + (', scaled by the issued scale plus '
                                                                                 'the level' if arm == 'L-N' else '')}


def v4_attribution(data, arrays: dict, day: date, offset: int, result: dict) -> dict:
    """v4 = A1_w/3 + B2_w/3 + L-N/6 + L-R/6: the blend decomposition, the linear rule for the LEAR members,
    and TreeSHAP with gain for the LightGBM members."""
    started = time.perf_counter()
    present = arrays['present'].astype(bool)
    names = _lear_names(data, ('wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean'))
    parts = {}
    for name, policy in (('A1_w', 'A1'), ('B2_w', 'B2')):
        lin = lear_linear(data, day, policy, result['selections'][name], names)
        parts[name] = {**lin, 'forecast': result['members_central'][name]}
    gains = {}
    for arm in ('L-N', 'L-R'):
        tree = lgbm_treeshap(data, arrays['wx'], present, day, arm, result['selections'][arm])
        gains[arm] = tree.pop('gain')
        parts[arm] = {**tree, 'forecast': result['members_central'][arm]}
    out = _summarize(parts, {'A1_w': 1 / 3, 'B2_w': 1 / 3, 'L-N': 1 / 6, 'L-R': 1 / 6}, result['central'], result['index'])
    out.update({'method': 'blend: weight x forecast; LEAR: coefficient x centered, standardized input; '
                          'LightGBM: TreeSHAP per prediction, and split gain from the fitted boosters',
                'gain': gains,
                'interval_layer': 'the H layer: hour-aware quantiles of the last 28 released standardized errors, '
                                  'scaled by the issued scale; no per-input attribution',
                'wording': WORDING, 'seconds': time.perf_counter() - started})
    return out
