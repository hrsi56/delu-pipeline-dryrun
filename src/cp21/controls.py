"""CP-21 causal and integrity controls on real data (capstone v21-r6 §17.7; charged).

Every negative assertion has a positive control that can fail and that survives the model's own
transforms. Trees are invariant to monotone feature rescaling and the §4 normalisation cancels a
uniform price scaling, so weather influence is shown with non-monotone perturbations (a
cross-date permutation of training weather; a within-day rearrangement of the target day's
weather) and the D-1 price control uses a non-uniform mutation (evening hours only).

For each representative origin, every LightGBM arm is refitted under each variant (charged as
control fits); HGL's A1_w/B2_w are refitted where HGL itself is controlled (charged component-days).
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

from cp15.data import day_hours, prepare
from cp16.residuals import SharedResidualState
from cp20.components import HGComponents
from .budget import atomic, ledger
from .components import augment, refit
from .execution import (OUT, NEW_ARMS, FitCache, Sources, _truth, _twice, charge_fits, check_protocol, fit_identity,
                        hg_cache_dir, replay)
from .inputs import BOUNDARY, BoundaryViolation, guard_boundary, hg_identity, load, weather_design, weather_matrix
from .jobs import art, stamp
from .lgbm import BLOCKS, LGBM_ARMS, MODEL_HOURS, fit_arm, hgl_central, model_rows

DAYS = (('fold_1', date(2020, 7, 1)), ('fold_4', date(2025, 5, 1)))
THRESHOLD = 1e-6
SEED = 15042
WX = ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean']


def _fit_all(budget, data, design, day, n_jobs=1):
    wx, present = weather_matrix(design, data)
    charge = charge_fits(budget, purpose='control', main=False)
    out = {}
    for arm in LGBM_ARMS:
        central, records, selection = fit_arm(data, wx, present, day, arm, charge=charge, n_jobs=n_jobs)
        out[arm] = {'central': central, 'selection': {m: s['selected'] for m, s in selection.items()},
                    'validation_mae': {m: s['validation_mae'] for m, s in selection.items()},
                    'trees': [r['model_sha256'] for r in records]}
    return out


def _hgl(budget, data, design, day, fits):
    aug, present = augment(data, design)
    comps = refit(budget, 'control', aug, present, day, data.rows(day))
    return hgl_central(comps['A1'], comps['B2'], fits['L-N']['central'], fits['L-R']['central']), comps


def _delta(a, b, data, day):
    """Max |difference| per arm, and per block for the block arms."""
    hours = data.hours[data.rows(day)]
    out = {}
    for arm in a:
        d = np.abs(np.asarray(a[arm]['central'] if isinstance(a[arm], dict) else a[arm])
                   - np.asarray(b[arm]['central'] if isinstance(b[arm], dict) else b[arm]))
        out[arm] = float(d.max())
        if arm in ('L-R', 'L-N'):
            for block, hs in BLOCKS.items():
                out[f'{arm}/{block}'] = float(d[np.isin(hours, hs)].max())
    return out


def _with_weather(design, table):
    return replace(design, table=table)


def representative(root: Path, budget, fold: str, day: date, fit_ident, hg_ident) -> dict:
    data = load(root)
    design = weather_design(root)
    rows = data.rows(day)
    record = {'fold': fold, 'day': str(day), 'n_hours': int(len(rows))}
    # 1. reproduction of the main-run fits and of HG's cached components
    base = _fit_all(budget, data, design, day)
    cache = FitCache(fold, data, fit_ident)
    record['reproduction_vs_main_run'] = {arm: bool(np.array_equal(base[arm]['central'], cache.get(day, arm))) for arm in LGBM_ARMS}
    main_trees = {arm: [r['model_sha256'] for r in cache.load(day, arm)['fits']] for arm in LGBM_ARMS}
    record['reproduction_trees_equal'] = {arm: base[arm]['trees'] == main_trees[arm] for arm in LGBM_ARMS}
    hgl_base, comps = _hgl(budget, data, design, day, base)
    hg = HGComponents(hg_cache_dir(), fold, data, hg_ident).get(day)[1]
    record['hg_components_refit_bitwise_vs_cp20_cache'] = {p: bool(np.array_equal(comps[p], hg[p])) for p in ('A1', 'B2')}
    committed = pd.read_parquet(root / OUT / 'predictions.parquet', filters=[('policy', '==', 'HGL')])
    committed['delivery_date'] = pd.to_datetime(committed.delivery_date).dt.date
    hgl_committed = committed.loc[committed.delivery_date.eq(day)].sort_values('timestamp_utc').central.to_numpy(float)
    record['hgl_central_bitwise_vs_committed'] = bool(np.array_equal(hgl_base, hgl_committed))
    base_all = {**base, 'HGL': hgl_base}
    # 2. delivery-day prices masked, later loads and weather scrambled -> exactly 0.0 (negative)
    masked = data.frame.copy()
    masked.loc[masked.delivery_date >= day, 'price_eur_mwh'] = np.nan
    masked.loc[masked.delivery_date > day, 'load_forecast_mw'] = 1e8
    mdata = prepare(masked, data.p, data.spec)
    future = design.table.copy()
    later = future.index.get_level_values(0) > day
    future.loc[later, WX] = 1e6
    mdesign = _with_weather(design, future)
    mfits = _fit_all(budget, mdata, mdesign, day)
    mhgl, _ = _hgl(budget, mdata, mdesign, day, mfits)
    record['delivery_day_and_future_mask'] = _delta({**mfits, 'HGL': mhgl}, base_all, data, day)
    record['mask_selection_and_trees_unchanged'] = {arm: mfits[arm]['selection'] == base[arm]['selection']
                                                    and mfits[arm]['trees'] == base[arm]['trees'] for arm in LGBM_ARMS}
    # 3. available D-1 prices mutated non-uniformly (evening hours only) -> forecasts move (positive)
    changed = data.frame.copy()
    local = pd.to_datetime(changed.timestamp_utc, utc=True).dt.tz_convert('Europe/Berlin').dt.hour
    target = changed.delivery_date.eq(day - timedelta(days=1)) & local.between(17, 21)
    changed.loc[target, 'price_eur_mwh'] += 300.0
    cdata = prepare(changed, data.p, data.spec)
    cfits = _fit_all(budget, cdata, design, day)
    chgl, _ = _hgl(budget, cdata, design, day, cfits)
    record['available_d1_nonuniform_price_mutation'] = _delta({**cfits, 'HGL': chgl}, base_all, data, day)
    record['d1_mutation_rows'] = int(target.sum())
    # 4. weather after the origin only -> exactly 0.0 (negative)
    ffits = _fit_all(budget, data, mdesign, day)
    record['future_weather_mutation'] = _delta(ffits, base, data, day)
    # 5. training-history weather permuted across dates (non-monotone) -> forecasts move (positive)
    permuted = design.table.copy()
    dates = permuted.index.get_level_values(0)
    pool = np.flatnonzero((dates < day) & np.isfinite(permuted[WX].to_numpy(float)).all(axis=1))
    values = permuted[WX].to_numpy(float, copy=True)
    values[pool] = values[np.random.default_rng(SEED).permutation(pool)]
    permuted[WX] = values
    pfits = _fit_all(budget, data, _with_weather(design, permuted), day)
    record['training_weather_cross_date_permutation'] = _delta(pfits, base, data, day)
    record['training_weather_rows_permuted'] = int(len(pool))
    # 6. the target day's own weather rearranged across its hours (non-monotone) -> forecasts move
    swapped = design.table.copy()
    today = np.flatnonzero(swapped.index.get_level_values(0) == day)
    swapped.iloc[today, [swapped.columns.get_loc(c) for c in WX]] = swapped.iloc[today[::-1]][WX].to_numpy()
    sfits = _fit_all(budget, data, _with_weather(design, swapped), day)
    record['target_day_weather_rearranged'] = _delta(sfits, base, data, day)
    # 7. uniform x3 scaling of all weather (monotone): an invariance diagnostic, not a positive control
    scaled = design.table.copy()
    scaled[WX] = scaled[WX] * 3.0
    xfits = _fit_all(budget, data, _with_weather(design, scaled), day)
    record['uniform_weather_scaling_x3_invariance'] = _delta(xfits, base, data, day)
    # 8. inner-validation outcomes altered -> validation losses (and possibly selection) change (positive)
    vframe = data.frame.copy()
    vlocal = pd.to_datetime(vframe.timestamp_utc, utc=True).dt.tz_convert('Europe/Berlin').dt.hour
    window = vframe.delivery_date.between(day - timedelta(days=28), day - timedelta(days=8))
    vframe.loc[window, 'price_eur_mwh'] += 150.0 * np.sin(vlocal[window].to_numpy() / 3.0)
    vdata = prepare(vframe, data.p, data.spec)
    vfits = _fit_all(budget, vdata, design, day)
    record['validation_outcome_mutation'] = {
        'validation_mae_changed': {arm: vfits[arm]['validation_mae'] != base[arm]['validation_mae'] for arm in LGBM_ARMS},
        'selection_changed': {arm: vfits[arm]['selection'] != base[arm]['selection'] for arm in LGBM_ARMS},
        'selection_before': {arm: base[arm]['selection'] for arm in LGBM_ARMS},
        'selection_after': {arm: vfits[arm]['selection'] for arm in LGBM_ARMS},
        'forecast_max_abs_change': _delta(vfits, base, data, day)}
    # 9. the inherited thread count: n_jobs=4 gives the main run's trees and forecasts
    tfits = _fit_all(budget, data, design, day, n_jobs=4)
    record['n_jobs_4_vs_main_run'] = {arm: bool(np.array_equal(tfits[arm]['central'], cache.get(day, arm))
                                                and tfits[arm]['trees'] == main_trees[arm]) for arm in LGBM_ARMS}
    # pooled-block parity and block membership on the real rows of this origin
    wx, present = weather_matrix(design, data)
    pooled = model_rows(data, day, 'pooled', present)
    blocks = [model_rows(data, day, b, present) for b in BLOCKS]
    record['pooled_block_rows_equal'] = {name: bool(np.array_equal(np.sort(np.concatenate([b[i] for b in blocks])), np.sort(pooled[i])))
                                         for i, name in ((1, 'window'), (2, 'inner'), (3, 'validation'), (4, 'forecast'))}
    return record


def dst_membership(root: Path) -> dict:
    """25-hour and 23-hour days inside a real training window: both repeated local-hour-2 rows in
    the night block, the missing spring hour absent; every scored key in exactly one block."""
    data = load(root)
    design = weather_design(root)
    wx, present = weather_matrix(design, data)
    day = date(2022, 7, 1)
    _, window, *_ = model_rows(data, day, 'night', present)
    _, pooled, *_ = model_rows(data, day, 'pooled', present)
    autumn, spring = np.datetime64('2021-10-31'), np.datetime64('2022-03-27')
    out = {'origin': str(day),
           'autumn_2021_10_31_canonical_hours': int(len(day_hours(date(2021, 10, 31)))),
           'autumn_local_hour_2_rows_in_night_window': int(((data.dates[window] == autumn) & (data.hours[window] == 2)).sum()),
           'autumn_local_hour_2_rows_in_pooled_window': int(((data.dates[pooled] == autumn) & (data.hours[pooled] == 2)).sum()),
           'spring_2022_03_27_canonical_hours': int(len(day_hours(date(2022, 3, 27)))),
           'spring_local_hour_2_rows_anywhere': int(((data.dates == spring) & (data.hours == 2)).sum())}
    blocks = {b: 0 for b in MODEL_HOURS if b != 'pooled'}
    for h in data.hours:
        found = [b for b in blocks if h in MODEL_HOURS[b]]
        if len(found) != 1:
            raise ValueError('an hour outside exactly one block')
    out['every_row_in_exactly_one_block'] = True
    out['passed'] = (out['autumn_2021_10_31_canonical_hours'] == 25 and out['autumn_local_hour_2_rows_in_night_window'] == 2
                     and out['autumn_local_hour_2_rows_in_pooled_window'] == 2 and out['spring_2022_03_27_canonical_hours'] == 23
                     and out['spring_local_hour_2_rows_anywhere'] == 0)
    return out


def admission_states(root: Path):
    """The four arms' states at the end of the training-only admission, from the committed
    admission-freeze lineage (the comparison job advances the working states afterwards)."""
    shas = subprocess.check_output(['git', 'log', '--format=%H', '--', f'{OUT}/lineage.json'], cwd=root, text=True).split()
    for commit in shas:
        doc = json.loads(subprocess.check_output(['git', 'show', f'{commit}:{OUT}/lineage.json'], cwd=root))
        if doc['execution_stage'] == 'training_only_admission_complete_frozen_before_outer_scoring':
            return commit, doc['states']
    raise ValueError('no committed training-only admission freeze')


def state_controls(root: Path, budget, fit_ident, hg_ident) -> dict:
    """Release/consume/restart/cache controls on the new arms' real states (policy-days charged)."""
    commit, admitted = admission_states(root)
    lineage = {'states_after_admission': admitted}
    data = load(root)
    fold = 'fold_1'
    first = date(2020, 7, 1)
    dates = list(pd.date_range(first, periods=10).date)
    truth = _truth(data)
    out = {'admission_freeze_commit': commit}
    # restart replay: continuous vs saved-and-reloaded after day 5 -> identical vectors and states
    runs = {}
    for label in ('continuous', 'restart'):
        states = {arm: SharedResidualState.from_dict(lineage['states_after_admission'][fold][arm]) for arm in NEW_ARMS}
        log, frames = {'origins': []}, []
        for i, d in enumerate(dates):
            if label == 'restart' and i == 5:
                states = {arm: SharedResidualState.loads(states[arm].dumps()) for arm in NEW_ARMS}
            replay(data, fold, [d], Sources(fold, data, fit_ident, hg_ident, NEW_ARMS), states, truth, None, log,
                   'control', frames, budget, 'policy_days_control')
        runs[label] = (pd.concat(frames, ignore_index=True), {a: states[a].dumps() for a in NEW_ARMS})
    a, b = runs['continuous'], runs['restart']
    out['restart_replay_identical'] = bool(a[0].drop(columns=['origin_utc']).equals(b[0].drop(columns=['origin_utc'])) and a[1] == b[1])
    committed = pd.read_parquet(root / OUT / 'predictions.parquet')
    committed = committed.loc[pd.to_datetime(committed.delivery_date).dt.date.isin(dates)]
    num = ['central', 'p025', 'p10', 'p25', 'p50', 'p75', 'p90', 'p975']
    mine = a[0].sort_values(['policy', 'timestamp_utc'])[num].to_numpy(float)
    theirs = committed.sort_values(['policy', 'timestamp_utc'])[num].to_numpy(float)
    out['replay_equals_committed_vectors'] = bool(mine.shape == theirs.shape and np.array_equal(mine, theirs))
    # D-1 refused, D-2 accepted once released; consume once; incomplete day and missing truth never enter
    state = SharedResidualState.from_dict(lineage['states_after_admission'][fold]['HGL'])
    budget.reserve(policy_days=4, policy_days_control=4)
    d0 = first
    rows = data.rows(d0)
    c = _twice(Sources(fold, data, fit_ident, hg_ident, ('HGL',)).get(d0)[1]['HGL'])
    state.release(d0, truth)
    state.issue(d0, data.index[rows], c, c, data.scale[rows])
    before = state.to_dict()
    state.release(d0 + timedelta(days=1), truth)  # d0 is D-1 for this origin: must stay pending
    after = state.to_dict()
    out['d_minus_1_error_refused'] = str(d0) in [p['day'] for p in after['pending']] and str(d0) not in after['consumed']
    state.release(d0 + timedelta(days=2), truth)  # now D-2: consumed once
    consumed = state.to_dict()
    out['d_minus_2_error_accepted'] = str(d0) in consumed['consumed'] and consumed['buffer'][-1]['day'] == str(d0)
    state.release(d0 + timedelta(days=2), truth)  # the same origin again: no second consumption
    out['consume_once'] = state.to_dict()['buffer'] == consumed['buffer']
    try:
        state.issue(d0, data.index[rows], c, c, data.scale[rows])
        out['duplicate_issue_refused'] = False
    except ValueError:
        out['duplicate_issue_refused'] = True
    partial = SharedResidualState.from_dict(before)
    d1 = d0 + timedelta(days=1)
    r1 = data.rows(d1)
    c1 = _twice(Sources(fold, data, fit_ident, hg_ident, ('HGL',)).get(d1)[1]['HGL'])
    partial.issue(d1, data.index[r1[:-1]], c1[:-1], c1[:-1], data.scale[r1[:-1]])  # one hour missing
    partial.release(d1 + timedelta(days=2), truth)
    out['partial_day_not_buffered'] = str(d1) not in [x['day'] for x in partial.to_dict()['buffer']] and any(
        t.get('status') == 'incomplete_issued_day' and t['feedback_day'] == str(d1) for t in partial.trace)
    nan_truth = SharedResidualState.from_dict(before)
    nan_truth.release(d0 + timedelta(days=2), lambda ix: np.full(len(ix), np.nan))
    out['unavailable_truth_stays_pending'] = str(d0) in [p['day'] for p in nan_truth.to_dict()['pending']]
    # stale or wrong caches are refused (positive control: the untouched entry loads)
    cache = FitCache(fold, data, fit_ident)
    tmp = art() / 'tmp-controls'
    shutil.rmtree(tmp, ignore_errors=True)
    probe = FitCache(fold, data, fit_ident, base=tmp)
    item = cache.load(d0, 'L-R')
    probe.save(item)
    out['intact_fit_cache_loads'] = probe.load(d0, 'L-R') is not None
    tampered = dict(item)
    tampered['central'] = list(item['central'])
    tampered['central'][0] += 1.0
    probe.save(tampered)
    try:
        probe.load(d0, 'L-R')
        out['tampered_fit_cache_refused'] = False
    except ValueError:
        out['tampered_fit_cache_refused'] = True
    wrong = FitCache(fold, data, {**fit_ident, 'cp21_protocol_sha256': '0' * 64})
    try:
        wrong.load(d0, 'L-R')
        out['wrong_identity_fit_cache_refused'] = False
    except ValueError:
        out['wrong_identity_fit_cache_refused'] = True
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        HGComponents(hg_cache_dir(), fold, data, {**hg_ident, 'protocol_sha256': '0' * 64}).get(d0)
        out['wrong_identity_hg_cache_refused'] = False
    except ValueError:
        out['wrong_identity_hg_cache_refused'] = True
    out['intact_hg_cache_loads'] = HGComponents(hg_cache_dir(), fold, data, hg_ident).get(d0) is not None
    return out


def population_controls(root: Path) -> dict:
    """Blend parity on every key, 23/24/25-hour identity, finite ordered quantiles, boundary."""
    new = pd.read_parquet(root / OUT / 'predictions.parquet')
    saved = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet', filters=[('policy', '==', 'HG')])
    key = ['fold', 'timestamp_utc']
    piv = {p: g.sort_values(key).reset_index(drop=True) for p, g in new.groupby('policy')}
    hg = saved.sort_values(key).reset_index(drop=True)
    def instants(series):  # compare instants, not storage units (CP-20 stored milliseconds)
        return pd.DatetimeIndex(pd.to_datetime(series, utc=True)).as_unit('ns').asi8

    same_keys = all(np.array_equal(instants(piv[p].timestamp_utc), instants(hg.timestamp_utc))
                    and np.array_equal(piv[p].fold.to_numpy(), hg.fold.to_numpy()) for p in piv)
    gap = piv['HGL'].central - (2 / 3) * hg.central - (1 / 3) * (piv['L-N'].central + piv['L-R'].central) / 2
    q = new[['p025', 'p10', 'p25', 'p50', 'p75', 'p90', 'p975']].to_numpy(float)
    counts = new.groupby(['policy', 'delivery_date']).size()
    dst = {str(d): int(counts.loc[('HGL', d)]) for d in (date(2026, 3, 29),) if ('HGL', d) in counts.index}
    out = {'keys_aligned_with_hg': bool(same_keys),
           'hgl_blend_parity_max_abs_eur_mwh': float(np.abs(gap).max()),
           'hgl_blend_parity_passed': bool(np.abs(gap).max() <= 1e-9),
           'rows_per_arm': new.groupby('policy').size().to_dict(),
           'quantiles_finite': bool(np.isfinite(q).all()), 'quantiles_ordered': bool((np.diff(q, axis=1) >= 0).all()),
           'spring_day_2026_03_29_rows_per_arm': dst,
           'max_delivery_date': str(pd.to_datetime(new.delivery_date).max().date())}
    try:
        guard_boundary(pd.DataFrame({'delivery_date': [BOUNDARY + timedelta(days=1)]}))
        out['boundary_guard_refuses_2026_04_08'] = False
    except BoundaryViolation:
        out['boundary_guard_refuses_2026_04_08'] = True
    out['boundary_guard_accepts_2026_04_07'] = guard_boundary(pd.DataFrame({'delivery_date': [BOUNDARY]})) is not None
    out['loader_max_date'] = str(load(root).dates.max())
    return out


def job_controls(root: Path, rest) -> int:
    check_protocol(root)
    fit_ident, hg_ident = fit_identity(root), hg_identity(root)
    budget = ledger()
    result = {'schema': 'cp21-controls-v1', 'threshold_eur_mwh': THRESHOLD, 'seed': SEED, 'origins': []}
    for fold, day in DAYS:
        record = representative(root, budget, fold, day, fit_ident, hg_ident)
        result['origins'].append(record)
        atomic(root / OUT / 'controls.json', result)
        print(json.dumps({k: v for k, v in record.items() if k != 'validation_outcome_mutation'}), flush=True)
    result['dst_block_membership'] = dst_membership(root)
    result['state'] = state_controls(root, budget, fit_ident, hg_ident)
    result['population'] = population_controls(root)
    checks = []
    for r in result['origins']:
        checks += [all(r['reproduction_vs_main_run'].values()), all(r['reproduction_trees_equal'].values()),
                   all(r['hg_components_refit_bitwise_vs_cp20_cache'].values()), r['hgl_central_bitwise_vs_committed'],
                   all(v == 0.0 for v in r['delivery_day_and_future_mask'].values()), all(r['mask_selection_and_trees_unchanged'].values()),
                   all(r['available_d1_nonuniform_price_mutation'][a] > THRESHOLD for a in (*LGBM_ARMS, 'HGL')),
                   all(v == 0.0 for v in r['future_weather_mutation'].values()),
                   all(r['training_weather_cross_date_permutation'][a] > THRESHOLD for a in LGBM_ARMS),
                   all(r['target_day_weather_rearranged'][a] > THRESHOLD for a in LGBM_ARMS),
                   all(r['validation_outcome_mutation']['validation_mae_changed'].values()),
                   all(r['n_jobs_4_vs_main_run'].values()), all(r['pooled_block_rows_equal'].values())]
    checks.append(result['dst_block_membership']['passed'])
    checks += [v for k, v in result['state'].items() if isinstance(v, bool)]
    p = result['population']
    checks += [p['keys_aligned_with_hg'], p['hgl_blend_parity_passed'], p['quantiles_finite'], p['quantiles_ordered'],
               p['boundary_guard_refuses_2026_04_08'], p['boundary_guard_accepts_2026_04_07'], p['loader_max_date'] == '2026-04-07',
               p['max_delivery_date'] <= '2026-04-07', all(v == 10747 for v in p['rows_per_arm'].values())]
    result['all_passed'] = bool(all(checks))
    result['checks'] = len(checks)
    result['written_utc'] = stamp()
    atomic(root / OUT / 'controls.json', result)
    print(json.dumps({'all_passed': result['all_passed'], 'checks': len(checks), 'state': result['state'],
                      'population': result['population'], 'dst': result['dst_block_membership']}), flush=True)
    return 0 if result['all_passed'] else 7
