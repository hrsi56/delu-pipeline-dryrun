"""CP-22 causal and integrity controls on real data (capstone v21-r9 §20.7, with §17.7; charged).

Every negative assertion has a positive control that can fail and survives PN's own transforms:
trees are invariant to monotone feature rescaling and §4's normalisation cancels a uniform price
scaling, so weather influence is shown with non-monotone perturbations and the D-1 price control
uses a non-uniform mutation (evening hours only).

* ``controls``: two representative origins (PN refits under each variant, charged as control fits;
  A1_w/B2_w refitted where a composite itself is controlled), the state, cache and DL controls on the
  committed admission-freeze states, and the population controls over every key and every PN fit.
* ``reproduce``: §20.10 item 2's independent representative HG and v4 slice -- A1_w/B2_w and L-N, L-R
  and L-P refitted at two origins and compared, bit for bit, with CP-20's and CP-21's committed vectors.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from cp15.data import LABELS, day_hours, prepare
from cp16.residuals import SharedResidualState
from cp20.components import HGComponents
from cp21.components import augment, refit
from cp21.lgbm import BLOCKS, GRID, POOLED, arm_design, fit_arm, hg_central, hgl_central, model_rows
from .budget import atomic, ledger
from .dl import ALPHAS, DynamicResidualState, clip_bounds
from .execution import (FIXED_NEW, LAYER, OUT, PNCache, Sources, _identities, _truth, _twice, charge_fits, check_protocol,
                        replay, state_from_dict)
from .inputs import BOUNDARY, BoundaryViolation, guard_boundary, load, weather_design, weather_matrix
from .jobs import art, cp20_art, stamp
from .pn import _fit, _model_sha, _z, composite, fit_pn, invert, pn_avg, pn_design, select

DAYS = (('fold_1', date(2020, 7, 1)), ('fold_4', date(2025, 5, 1)))
REPRODUCE = (('fold_2', date(2021, 4, 1)), ('fold_5', date(2026, 1, 8)))
THRESHOLD = 1e-6
SEED = 15042
WX = ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean']


def _pn(budget, data, design, day, n_jobs=1):
    wx, present = weather_matrix(design, data)
    member, records = fit_pn(data, wx, present, day, charge=charge_fits(budget, purpose='control', main=False), n_jobs=n_jobs)
    return {'avg': member['pn_avg'], 'sel': member['pn_sel'], 'selected': member['selected'],
            'validation_mae': member['validation_mae'], 'trees': [r['model_sha256'] for r in records]}


def _composites(budget, data, design, day, pn, lp):
    aug, present = augment(data, design)
    comps = refit(budget, 'control', aug, present, day, data.rows(day))
    a1, b2 = comps['A1'], comps['B2']
    return {'R': composite(a1, b2, ((pn['avg'], 3),)), 'M': composite(a1, b2, ((pn['sel'], 6), (lp, 6))),
            'A-PN-sel': composite(a1, b2, ((pn['sel'], 3),))}, comps


def _d(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def representative(root: Path, budget, fold: str, day: date, idents) -> dict:
    fit_ident, cp21_ident, hg_ident, hashes = idents
    data = load(root)
    design = weather_design(root)
    rows = data.rows(day)
    record = {'fold': fold, 'day': str(day), 'n_hours': int(len(rows))}
    sources = Sources(fold, data, fit_ident, cp21_ident, hg_ident, hashes, FIXED_NEW)
    m = sources.members(day)
    cached = PNCache(fold, data, fit_ident).load(day)
    # 1. reproduction of the main-run PN fits; PN-sel's final fit is the selected configuration's full-window fit
    base = _pn(budget, data, design, day)
    record['pn_reproduction'] = {'avg_bitwise': bool(np.array_equal(base['avg'], m['PN-avg'])),
                                 'sel_bitwise': bool(np.array_equal(base['sel'], m['PN-sel'])),
                                 'trees_equal': base['trees'] == [r['model_sha256'] for r in cached['fits']],
                                 'selected_equal': base['selected'] == cached['selected']}
    wx, present = weather_matrix(design, data)
    pdes = pn_design(data, wx)
    _, window, inner, validation, forecast = model_rows(data, day, POOLED, present)
    charge_fits(budget, purpose='control', main=False)('final')
    k = [g['id'] for g in GRID].index(cached['selected'])
    model, imputer, *_ = _fit(pdes, window, GRID[k], data.p, 1)
    refit_sel = invert(_z(model, imputer, pdes, forecast), data, forecast)
    full_tree = [r['model_sha256'] for r in cached['fits'] if r['role'] == 'final'][k]
    record['pn_sel_independent_refit'] = {'config': cached['selected'], 'tree_equal': _model_sha(model) == full_tree,
                                          'forecast_bitwise': bool(np.array_equal(refit_sel, m['PN-sel']))}
    base_c, comps = _composites(budget, data, design, day, base, m['L-P'])
    hg = HGComponents(cp20_art() / 'hg-components', fold, data, hg_ident).get(day)[1]
    record['hg_components_refit_bitwise_vs_cp20_cache'] = {p: bool(np.array_equal(comps[p], hg[p])) for p in ('A1', 'B2')}
    committed = pd.read_parquet(root / OUT / 'predictions.parquet')
    committed = committed.loc[pd.to_datetime(committed.delivery_date).dt.date.eq(day)]
    record['composites_bitwise_vs_committed'] = {
        p: bool(np.array_equal(base_c[p], committed.loc[committed.policy.eq(p)].sort_values('timestamp_utc').central.to_numpy(float)))
        for p in base_c}
    # 2. delivery-day prices masked, later loads and weather scrambled -> exactly 0.0, selection and trees unchanged
    masked = data.frame.copy()
    masked.loc[masked.delivery_date >= day, 'price_eur_mwh'] = np.nan
    masked.loc[masked.delivery_date > day, 'load_forecast_mw'] = 1e8
    mdata = prepare(masked, data.p, data.spec)
    future = design.table.copy()
    future.loc[future.index.get_level_values(0) > day, WX] = 1e6
    mdesign = replace(design, table=future)
    mpn = _pn(budget, mdata, mdesign, day)
    mc, _ = _composites(budget, mdata, mdesign, day, mpn, m['L-P'])
    record['delivery_day_and_future_mask'] = {'PN-avg': _d(mpn['avg'], base['avg']), 'PN-sel': _d(mpn['sel'], base['sel']),
                                              **{p: _d(mc[p], base_c[p]) for p in mc}}
    record['mask_selection_and_trees_unchanged'] = mpn['selected'] == base['selected'] and mpn['trees'] == base['trees']
    # 3. available D-1 prices mutated non-uniformly (evening hours) -> PN and the composites move
    changed = data.frame.copy()
    local = pd.to_datetime(changed.timestamp_utc, utc=True).dt.tz_convert('Europe/Berlin').dt.hour
    target = changed.delivery_date.eq(day - timedelta(days=1)) & local.between(17, 21)
    changed.loc[target, 'price_eur_mwh'] += 300.0
    cdata = prepare(changed, data.p, data.spec)
    cpn = _pn(budget, cdata, design, day)
    cc, _ = _composites(budget, cdata, design, day, cpn, m['L-P'])
    record['available_d1_nonuniform_price_mutation'] = {'PN-avg': _d(cpn['avg'], base['avg']), 'PN-sel': _d(cpn['sel'], base['sel']),
                                                        **{p: _d(cc[p], base_c[p]) for p in cc}}
    record['d1_mutation_rows'] = int(target.sum())
    # 4. weather after the origin only -> exactly 0.0
    fpn = _pn(budget, data, mdesign, day)
    record['future_weather_mutation'] = {'PN-avg': _d(fpn['avg'], base['avg']), 'PN-sel': _d(fpn['sel'], base['sel'])}
    # 5. training weather permuted across dates (non-monotone) -> PN moves
    permuted = design.table.copy()
    dates = permuted.index.get_level_values(0)
    pool = np.flatnonzero((dates < day) & np.isfinite(permuted[WX].to_numpy(float)).all(axis=1))
    values = permuted[WX].to_numpy(float, copy=True)
    values[pool] = values[np.random.default_rng(SEED).permutation(pool)]
    permuted[WX] = values
    ppn = _pn(budget, data, replace(design, table=permuted), day)
    record['training_weather_cross_date_permutation'] = {'PN-avg': _d(ppn['avg'], base['avg']), 'PN-sel': _d(ppn['sel'], base['sel'])}
    # 6. the target day's own weather rearranged across its hours -> PN moves
    swapped = design.table.copy()
    today = np.flatnonzero(swapped.index.get_level_values(0) == day)
    swapped.iloc[today, [swapped.columns.get_loc(c) for c in WX]] = swapped.iloc[today[::-1]][WX].to_numpy()
    spn = _pn(budget, data, replace(design, table=swapped), day)
    record['target_day_weather_rearranged'] = {'PN-avg': _d(spn['avg'], base['avg']), 'PN-sel': _d(spn['sel'], base['sel'])}
    # 7. uniform x3 weather scaling (monotone): invariance diagnostic, not a positive control
    scaled = design.table.copy()
    scaled[WX] = scaled[WX] * 3.0
    xpn = _pn(budget, data, replace(design, table=scaled), day)
    record['uniform_weather_scaling_x3_invariance'] = {'PN-avg': _d(xpn['avg'], base['avg']), 'PN-sel': _d(xpn['sel'], base['sel'])}
    # 8. inner-validation outcomes altered -> validation losses change (selection may)
    vframe = data.frame.copy()
    vlocal = pd.to_datetime(vframe.timestamp_utc, utc=True).dt.tz_convert('Europe/Berlin').dt.hour
    vwin = vframe.delivery_date.between(day - timedelta(days=28), day - timedelta(days=8))
    vframe.loc[vwin, 'price_eur_mwh'] += 150.0 * np.sin(vlocal[vwin].to_numpy() / 3.0)
    vpn = _pn(budget, prepare(vframe, data.p, data.spec), design, day)
    record['validation_outcome_mutation'] = {'validation_mae_changed': vpn['validation_mae'] != base['validation_mae'],
                                             'selection_before': base['selected'], 'selection_after': vpn['selected'],
                                             'PN-avg_change': _d(vpn['avg'], base['avg'])}
    # 9. n_jobs=4 gives the main run's trees and forecasts
    tpn = _pn(budget, data, design, day, n_jobs=4)
    record['n_jobs_4_vs_main_run'] = bool(np.array_equal(tpn['avg'], m['PN-avg']) and tpn['trees'] == base['trees'])
    # 10. ladder parity on real rows: PN's rows are L-P's pooled rows and the union of L-N's block rows;
    #     PN's target is L-N's normalised target (one change per step)
    fits21 = pd.read_parquet(root / 'reports/block-challenger/fits.parquet',
                             filters=[('delivery_date', '==', str(day)), ('arm', '==', 'L-P'), ('role', '==', 'final')])
    blocks = [model_rows(data, day, b, present) for b in BLOCKS]
    ln_design = arm_design(data, wx, 'L-N')
    record['ladder_rows'] = {
        'pn_window_rows_equal_cp21_lp': bool(len(fits21) == 1 and fits21.window_rows_sha256.iloc[0] == cached['fits'][-1]['window_rows_sha256']),
        **{f'pn_{name}_equals_union_of_ln_blocks': bool(np.array_equal(np.sort(np.concatenate([b[i] for b in blocks])),
                                                                       np.sort((window, inner, validation, forecast)[i - 1])))
           for i, name in ((1, 'window'), (2, 'inner'), (3, 'validation'), (4, 'forecast'))},
        # NaN marks ineligible rows in both designs: compare with equal_nan, against CP-21's own L-N design
        'pn_target_is_ln_normalised_target': bool(np.array_equal(pdes.target, ln_design.target, equal_nan=True)),
        'pn_features_are_ln_features': bool(np.array_equal(pdes.x, ln_design.x, equal_nan=True)
                                            and np.array_equal(pdes.wx, ln_design.wx, equal_nan=True)),
        'lp_target_differs_positive_control': not bool(np.array_equal(pdes.target, arm_design(data, wx, 'L-P').target,
                                                                      equal_nan=True))}
    return record


def admission_states(root: Path, name: str = 'lineage.json'):
    shas = subprocess.check_output(['git', 'log', '--format=%H', '--', f'{OUT}/{name}'], cwd=root, text=True).split()
    for commit in shas:
        doc = json.loads(subprocess.check_output(['git', 'show', f'{commit}:{OUT}/{name}'], cwd=root))
        if doc['execution_stage'] == 'training_only_admission_complete_frozen_before_outer_scoring':
            return commit, doc['states']
    raise ValueError('no committed training-only admission freeze')


def state_controls(root: Path, budget, idents) -> dict:
    """Release/consume/restart/cache and DL controls on real states (policy-days charged)."""
    fit_ident, cp21_ident, hg_ident, hashes = idents
    commit, admitted = admission_states(root)
    data = load(root)
    fold, first = 'fold_1', date(2020, 7, 1)
    dates = list(pd.date_range(first, periods=10).date)
    truth = _truth(data)
    out = {'admission_freeze_commit': commit}
    runs = {}
    for label in ('continuous', 'restart'):
        states = {p: state_from_dict(p, admitted[fold][p]) for p in FIXED_NEW}
        log, frames = {'origins': []}, []
        for i, d in enumerate(dates):
            if label == 'restart' and i == 5:
                states = {p: state_from_dict(p, json.loads(json.dumps(states[p].to_dict()))) for p in FIXED_NEW}
            replay(data, fold, [d], Sources(fold, data, fit_ident, cp21_ident, hg_ident, hashes, FIXED_NEW), states, truth,
                   None, log, 'control', frames, budget, 'policy_days_control')
        runs[label] = (pd.concat(frames, ignore_index=True), {p: json.dumps(states[p].to_dict(), sort_keys=True) for p in FIXED_NEW})
    a, b = runs['continuous'], runs['restart']
    out['restart_replay_identical'] = bool(a[0].drop(columns=['origin_utc']).equals(b[0].drop(columns=['origin_utc'])) and a[1] == b[1])
    committed = pd.read_parquet(root / OUT / 'predictions.parquet')
    committed = committed.loc[pd.to_datetime(committed.delivery_date).dt.date.isin(dates)]
    num = ['central', *LABELS]
    mine = a[0].sort_values(['policy', 'timestamp_utc'])[num].to_numpy(float)
    theirs = committed.sort_values(['policy', 'timestamp_utc'])[num].to_numpy(float)
    out['replay_equals_committed_vectors'] = bool(mine.shape == theirs.shape and np.array_equal(mine, theirs))
    # release rule on an H state (R) and a DL state (v4+DL)
    for policy in ('R', 'v4+DL'):
        state = state_from_dict(policy, admitted[fold][policy])
        budget.reserve(policy_days=4, policy_days_control=4)
        d0 = first
        rows = data.rows(d0)
        src = Sources(fold, data, fit_ident, cp21_ident, hg_ident, hashes, (policy,))
        c = _twice(src.get(d0)[1][policy])
        state.release(d0, truth)
        if LAYER[policy] != 'H':
            state.predict(d0, data.index[rows], c, c, data.scale[rows])
        state.issue(d0, data.index[rows], c, c, data.scale[rows])
        before = state.to_dict()
        state.release(d0 + timedelta(days=1), truth)
        after = state.to_dict()
        base_after = after if LAYER[policy] == 'H' else after['base']
        r = {'d_minus_1_error_refused': str(d0) in [p['day'] for p in base_after['pending']] and str(d0) not in base_after['consumed']}
        state.release(d0 + timedelta(days=2), truth)
        consumed = state.to_dict() if LAYER[policy] == 'H' else state.to_dict()['base']
        r['d_minus_2_error_accepted'] = str(d0) in consumed['consumed'] and consumed['buffer'][-1]['day'] == str(d0)
        if LAYER[policy] != 'H':
            last = state.aci_trace[-1]
            r['aci_updated_once_for_the_released_day'] = last['day'] == str(d0) and sum(t['day'] == str(d0) for t in state.aci_trace) == 1
        state.release(d0 + timedelta(days=2), truth)
        again = state.to_dict() if LAYER[policy] == 'H' else state.to_dict()['base']
        r['consume_once'] = again['buffer'] == consumed['buffer']
        try:
            state.issue(d0, data.index[rows], c, c, data.scale[rows])
            r['duplicate_issue_refused'] = False
        except ValueError:
            r['duplicate_issue_refused'] = True
        partial = state_from_dict(policy, before)
        d1 = d0 + timedelta(days=1)
        r1 = data.rows(d1)
        c1 = _twice(src.get(d1)[1][policy])
        partial.issue(d1, data.index[r1[:-1]], c1[:-1], c1[:-1], data.scale[r1[:-1]])
        partial.release(d1 + timedelta(days=2), truth)
        pbase = partial.to_dict() if LAYER[policy] == 'H' else partial.to_dict()['base']
        r['partial_day_not_buffered'] = str(d1) not in [x['day'] for x in pbase['buffer']] and any(
            t.get('status') == 'incomplete_issued_day' and t['feedback_day'] == str(d1) for t in partial.trace)
        nan_truth = state_from_dict(policy, before)
        nan_truth.release(d0 + timedelta(days=2), lambda ix: np.full(len(ix), np.nan))
        nbase = nan_truth.to_dict() if LAYER[policy] == 'H' else nan_truth.to_dict()['base']
        r['unavailable_truth_stays_pending'] = str(d0) in [p['day'] for p in nbase['pending']]
        out[f'release_rule_{policy}'] = r
    # DL with no weights and no ACI reproduces the committed H vectors of R, bit for bit, through the
    # whole fold-1 warm-up and the first ten evaluation days (positive control: DL itself differs)
    early = load(root, before=first)
    m = json.loads((root / 'reports/v2-causal/input-manifest.json').read_text())
    f1 = next(f for f in m['folds'] if f['fold'] == fold)
    frames = {}
    for label, variant in (('h_parity', 'H-PARITY'), ('dl', 'DL')):
        state = DynamicResidualState(variant)
        src_early = Sources(fold, early, fit_ident, cp21_ident, hg_ident, hashes, ('R',))
        src_full = Sources(fold, data, fit_ident, cp21_ident, hg_ident, hashes, ('R',))
        vectors = []
        for phase, source, frame_data, days in (('warmup', src_early, early, list(pd.date_range(f1['warmup_start'], first - timedelta(days=1)).date)),
                                                ('evaluation', src_full, data, dates)):
            t = _truth(frame_data)
            if phase == 'evaluation':
                state = DynamicResidualState.from_dict(state.to_dict())  # the same state, reloaded on the full data
            for d in days:
                rows, centers, _, _ = source.get(d)
                budget.reserve(policy_days=1, policy_days_control=1)
                state.release(d, t)
                if len(rows) and state.ready():
                    c = _twice(centers['R'])
                    q, _ = state.predict(d, frame_data.index[rows], c, c, frame_data.scale[rows])
                    if phase == 'evaluation':
                        vectors.append(q['DL'])
                if len(rows):
                    c = _twice(centers['R'])
                    state.issue(d, frame_data.index[rows], c, c, frame_data.scale[rows])
        frames[label] = np.concatenate(vectors)
    r_committed = committed.loc[committed.policy.eq('R')].sort_values('timestamp_utc')[LABELS].to_numpy(float)
    out['dl_without_weights_or_aci_equals_committed_R_bitwise'] = bool(np.array_equal(frames['h_parity'], r_committed))
    out['dl_differs_from_committed_R'] = bool(not np.array_equal(frames['dl'], r_committed))
    # stale or wrong PN cache entries are refused (positive control: the intact entry loads)
    cache = PNCache(fold, data, fit_ident)
    tmp = art() / 'tmp-controls'
    shutil.rmtree(tmp, ignore_errors=True)
    probe = PNCache(fold, data, fit_ident, base=tmp)
    item = cache.load(first)
    probe.save(item)
    out['intact_pn_cache_loads'] = probe.load(first) is not None
    tampered = dict(item)
    tampered['pn_avg'] = list(item['pn_avg'])
    tampered['pn_avg'][0] += 1.0
    probe.save(tampered)
    try:
        probe.load(first)
        out['tampered_pn_cache_refused'] = False
    except ValueError:
        out['tampered_pn_cache_refused'] = True
    try:
        PNCache(fold, data, {**fit_ident, 'cp22_protocol_sha256': '0' * 64}, base=tmp).load(first)
        out['wrong_identity_pn_cache_refused'] = False
    except ValueError:
        out['wrong_identity_pn_cache_refused'] = True
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        Sources(fold, data, fit_ident, {**cp21_ident, 'cp21_protocol_sha256': '0' * 64}, hg_ident, hashes, ('M',)).get(first)
        out['wrong_identity_cp21_cache_refused'] = False
    except ValueError:
        out['wrong_identity_cp21_cache_refused'] = True
    bad = dict(hashes)
    bad[('L-P', fold, str(first))] = '0' * 64
    try:
        Sources(fold, data, fit_ident, cp21_ident, hg_ident, bad, ('M',)).get(first)
        out['cp21_vector_differing_from_committed_lineage_refused'] = False
    except ValueError:
        out['cp21_vector_differing_from_committed_lineage_refused'] = True
    out['intact_sources_load'] = len(Sources(fold, data, fit_ident, cp21_ident, hg_ident, hashes, ('M',)).get(first)[1]['M']) > 0
    return out


def population_controls(root: Path, idents) -> dict:
    """Composite parity on every key, PN's averaging identity over every fit, DL bounds over every
    emission, 23/24/25-hour identity, finite ordered quantiles, the boundary guard."""
    fit_ident = idents[0]
    new = pd.read_parquet(root / OUT / 'predictions.parquet')
    members = pd.read_parquet(root / OUT / 'members.parquet')
    key = ['fold', 'timestamp_utc']
    mem = members.sort_values(key).reset_index(drop=True)
    piv = {p: g.sort_values(key).reset_index(drop=True) for p, g in new.groupby('policy')}
    hg = mem.HG.to_numpy()
    terms = {'R': mem['PN-avg'], 'M': (mem['PN-sel'] + mem['L-P']) / 2, 'A-PN-sel': mem['PN-sel'], 'A-LP': mem['L-P'], 'A-LN': mem['L-N']}
    parity = {p: float(np.max(np.abs(piv[p].central.to_numpy() - (2 / 3) * hg - terms[p].to_numpy() / 3))) for p in terms}
    out = {'composite_parity_max_abs_eur_mwh': parity, 'composite_parity_passed': all(v <= 1e-9 for v in parity.values()),
           'v4_dl_central_is_v4': bool(np.array_equal(piv['v4+DL'].central.to_numpy(), mem.HGL.to_numpy())),
           'v3_dl_central_is_v3': bool(np.array_equal(piv['v3+DL'].central.to_numpy(), hg)),
           'keys_aligned_with_members': all(np.array_equal(piv[p].timestamp_utc.to_numpy(), mem.timestamp_utc.to_numpy()) for p in piv)}
    lineage = json.loads((root / OUT / 'lineage.json').read_text())
    layers = {}
    for rec in lineage['origins']:
        layers.setdefault(rec['policy'], set()).add(rec['layer'])
    out['layer_by_policy'] = {p: sorted(v) for p, v in layers.items()}
    out['non_dl_composites_on_h_path'] = all(layers[p] == {'H'} for p in ('R', 'M', 'A-PN-sel', 'A-LP', 'A-LN'))
    emissions = [r for r in lineage['origins'] if r.get('emitted') and r['layer'] != 'H']
    bounds_ok = all(clip_bounds(a)[0] <= r['alpha_t'][str(a)] <= clip_bounds(a)[1] for r in emissions for a in ALPHAS)
    weights_ok = all(abs(r['day_weights_sum'] - 1) <= 1e-12 for r in emissions)
    out['dl_emissions'] = len(emissions)
    out['dl_alpha_within_clip_bounds_every_emission'] = bool(bounds_ok)
    out['dl_recency_weights_sum_to_one_every_emission'] = bool(weights_ok)
    out['dl_rows_rearranged'] = int(sum(r.get('crossed_rows_rearranged', 0) for r in emissions))
    # PN capacity-averaging identity over every cached fit
    data_full = load(root)
    m = json.loads((root / 'reports/v2-causal/input-manifest.json').read_text())
    checked, avg_ok, sel_ok, route, rule_ok = 0, True, True, 0.0, True
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        early = load(root, before=first)
        for o in f['origins']:
            d = date.fromisoformat(o['day'])
            data = data_full if d >= first else early
            if not len(data.rows(d)):
                continue
            item = PNCache(f['fold'], data, fit_ident).load(d)
            full = [np.asarray(v, float) for v in item['full_eur']]
            avg_ok &= bool(np.array_equal(np.asarray(item['pn_avg']), pn_avg(full)))
            sel_ok &= bool(np.array_equal(np.asarray(item['pn_sel']), full[item['selected_index']]))
            route = max(route, item['averaging_route_max_abs_gap'])
            rule_ok &= select(item['validation_mae']) == item['selected_index']
            checked += 1
    out['pn_fits_checked'] = checked
    out['pn_avg_equals_mean_of_four_full_window_fits'] = bool(avg_ok)
    out['pn_sel_equals_selected_full_window_fit'] = bool(sel_ok)
    out['pn_averaging_routes_max_abs_gap'] = route
    out['pn_selection_follows_rule'] = bool(rule_ok)
    q = new[LABELS].to_numpy(float)
    counts = new.groupby(['policy', 'delivery_date']).size()
    out['rows_per_policy'] = new.groupby('policy').size().to_dict()
    out['quantiles_finite'] = bool(np.isfinite(q).all())
    out['quantiles_ordered'] = bool((np.diff(q, axis=1) >= 0).all())
    out['p50_separate_from_central'] = bool((new.p50 != new.central).any())
    out['dst_rows'] = {str(d): {p: int(counts.loc[(p, d)]) for p in FIXED_NEW}
                       for d in (date(2026, 3, 29),) if ('R', d) in counts.index}
    out['max_delivery_date'] = str(pd.to_datetime(new.delivery_date).max().date())
    try:
        guard_boundary(pd.DataFrame({'delivery_date': [BOUNDARY + timedelta(days=1)]}))
        out['boundary_guard_refuses_2026_04_08'] = False
    except BoundaryViolation:
        out['boundary_guard_refuses_2026_04_08'] = True
    out['boundary_guard_accepts_2026_04_07'] = guard_boundary(pd.DataFrame({'delivery_date': [BOUNDARY]})) is not None
    out['loader_max_date'] = str(data_full.dates.max())
    out['spring_2026_03_29_canonical_hours'] = int(len(day_hours(date(2026, 3, 29))))
    out['autumn_2025_10_26_canonical_hours'] = int(len(day_hours(date(2025, 10, 26))))
    return out


def job_controls(root: Path, rest) -> int:
    check_protocol(root)
    idents = _identities(root)
    budget = ledger()
    result = {'schema': 'cp22-controls-v1', 'threshold_eur_mwh': THRESHOLD, 'seed': SEED, 'origins': []}
    for fold, day in DAYS:
        record = representative(root, budget, fold, day, idents)
        result['origins'].append(record)
        atomic(root / OUT / 'controls.json', result)
        print(json.dumps(record, default=str)[:3000], flush=True)
    result['state'] = state_controls(root, budget, idents)
    result['population'] = population_controls(root, idents)
    checks = []
    for r in result['origins']:
        checks += [all(r['pn_reproduction'].values()), all(r['pn_sel_independent_refit'][k] for k in ('tree_equal', 'forecast_bitwise')),
                   all(r['hg_components_refit_bitwise_vs_cp20_cache'].values()), all(r['composites_bitwise_vs_committed'].values()),
                   all(v == 0.0 for v in r['delivery_day_and_future_mask'].values()), r['mask_selection_and_trees_unchanged'],
                   all(v > THRESHOLD for v in r['available_d1_nonuniform_price_mutation'].values()),
                   all(v == 0.0 for v in r['future_weather_mutation'].values()),
                   all(v > THRESHOLD for v in r['training_weather_cross_date_permutation'].values()),
                   all(v > THRESHOLD for v in r['target_day_weather_rearranged'].values()),
                   r['validation_outcome_mutation']['validation_mae_changed'], r['n_jobs_4_vs_main_run'], all(r['ladder_rows'].values())]
    for k, v in result['state'].items():
        if isinstance(v, bool):
            checks.append(v)
        elif isinstance(v, dict):
            checks += [x for x in v.values() if isinstance(x, bool)]
    p = result['population']
    checks += [p['composite_parity_passed'], p['v4_dl_central_is_v4'], p['v3_dl_central_is_v3'], p['keys_aligned_with_members'],
               p['non_dl_composites_on_h_path'], p['dl_alpha_within_clip_bounds_every_emission'],
               p['dl_recency_weights_sum_to_one_every_emission'], p['pn_fits_checked'] == 636,
               p['pn_avg_equals_mean_of_four_full_window_fits'], p['pn_sel_equals_selected_full_window_fit'],
               p['pn_averaging_routes_max_abs_gap'] <= 1e-9, p['pn_selection_follows_rule'], p['quantiles_finite'],
               p['quantiles_ordered'], p['p50_separate_from_central'], p['boundary_guard_refuses_2026_04_08'],
               p['boundary_guard_accepts_2026_04_07'], p['loader_max_date'] == '2026-04-07', p['max_delivery_date'] <= '2026-04-07',
               all(v == 10747 for v in p['rows_per_policy'].values()), p['spring_2026_03_29_canonical_hours'] == 23,
               p['autumn_2025_10_26_canonical_hours'] == 25]
    result['all_passed'] = bool(all(checks))
    result['checks'] = len(checks)
    result['written_utc'] = stamp()
    atomic(root / OUT / 'controls.json', result)
    print(json.dumps({'all_passed': result['all_passed'], 'checks': len(checks), 'state': result['state'],
                      'population': {k: v for k, v in p.items() if k != 'layer_by_policy'}}, default=str), flush=True)
    return 0 if result['all_passed'] else 7


def job_reproduce(root: Path, rest) -> int:
    """§20.10 item 2: an independent representative HG and v4 slice, refitted and compared with the
    committed CP-20 and CP-21 vectors bit for bit (charged as reproduction)."""
    check_protocol(root)
    _, cp21_ident, hg_ident, hashes = _identities(root)
    budget = ledger()
    data = load(root)
    design = weather_design(root)
    wx, present = weather_matrix(design, data)
    aug, apresent = augment(data, design)
    hg_acc = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet', filters=[('policy', '==', 'HG')])
    v4_acc = pd.read_parquet(root / 'reports/block-challenger/predictions.parquet')
    out = {'schema': 'cp22-reproduction-v1', 'origins': []}
    for fold, day in REPRODUCE:
        rows = data.rows(day)
        comps = refit(budget, 'reproduction', aug, apresent, day, rows)
        charge = charge_fits(budget, purpose='reproduction', main=False)
        lgbm = {arm: fit_arm(data, wx, present, day, arm, charge=charge)[0] for arm in ('L-N', 'L-R', 'L-P')}
        hgc = hg_central(comps['A1'], comps['B2'])
        v4c = hgl_central(comps['A1'], comps['B2'], lgbm['L-N'], lgbm['L-R'])
        def acc(frame, policy):
            part = frame.loc[frame.policy.eq(policy) & pd.to_datetime(frame.delivery_date).dt.date.eq(day)]
            return part.sort_values('timestamp_utc').central.to_numpy(float)
        rec = {'fold': fold, 'day': str(day), 'n_hours': int(len(rows)),
               'hg_central_bitwise_vs_cp20_committed': bool(np.array_equal(hgc, acc(hg_acc, 'HG'))),
               'v4_central_bitwise_vs_cp21_committed': bool(np.array_equal(v4c, acc(v4_acc, 'HGL'))),
               **{f'{arm}_bitwise_vs_cp21_committed': bool(np.array_equal(lgbm[arm], acc(v4_acc, arm))) for arm in lgbm}}
        out['origins'].append(rec)
        print(json.dumps(rec), flush=True)
    out['all_bitwise'] = all(v for r in out['origins'] for k, v in r.items() if k.endswith('committed'))
    out['written_utc'] = stamp()
    atomic(root / OUT / 'reproduction.json', out)
    return 0 if out['all_bitwise'] else 8
