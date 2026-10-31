"""CP-23 causal and integrity controls on real data (capstone v21-r10 §21.7, with §17.7 and §20.7's
applicable controls; DDNN fits charged as control fits).

Every negative assertion is paired with a positive control that can fail and survives DDNN's own
transforms. §4's normalisation cancels a uniform price scaling, so the D-1 price control uses a
non-uniform mutation (evening hours only), and weather influence is shown with non-monotone
perturbations.

* ``controls``:
  - two representative origins (fold 1's and fold 4's first evaluation days), with the DDNN ensemble
    refitted under each variant;
  - one fold's configuration choice, repeated with mutated holdout outcomes and with mutated
    evaluation outcomes;
  - the state and cache controls on the committed admission-freeze states;
  - the population controls over every key and every DDNN cache entry.
* ``reproduce``: §21.10 item 6's independent representative HG and v4 slice. A1_w/B2_w and L-N/L-R
  are refitted at two origins and compared, bit for bit, with CP-20's and CP-21's committed vectors.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from cp15.data import LABELS, day_hours, prepare
from cp16.residuals import SharedResidualState
from cp21.components import augment, refit
from cp21.lgbm import fit_arm, hg_central, hgl_central
from . import ddnn as D
from .audit import audit
from .budget import atomic, charge_fits, ledger
from .execution import (COMPOSITE_TOLERANCE, H_POLICIES, LAYER, NEW_POLICIES, OUT, DDNNCache, Sources, _identities, _truth,
                        _twice, check_protocol, replay, selected_config, v3d_central, v5_central)
from .features import encode, target
from .inputs import BOUNDARY, BoundaryViolation, guard_boundary, load, origin_manifest, weather_design, weather_matrix
from .jobs import art, stamp
from .member import fit_origin, fit_selection
from .reference import require_passing_record

DAYS = (('fold_1', date(2020, 7, 1)), ('fold_4', date(2025, 5, 1)))
REPRODUCE = (('fold_2', date(2021, 4, 1)), ('fold_5', date(2026, 1, 8)))
SELECTION_FOLD = 'fold_1'
THRESHOLD = 1e-6
SEED = 15042
WX = ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean']


class ReproductionCharges:
    """CP-21's component refit and LightGBM fits, charged to CP-23's reproduction counters (§21.8 sets
    no cap for them; they are recorded and inside the machine-hour ceiling)."""

    def __init__(self, budget):
        self.budget = budget

    def reserve(self, **increments):
        if 'component_attempts' in increments:
            return self.budget.reserve(component_attempts_reproduction=increments['component_attempts'])
        if 'primitive_fits' in increments:
            return self.budget.reserve(lasso_fits_reproduction=increments['primitive_fits'])
        raise ValueError(f'unexpected reproduction counter {sorted(increments)}')

    def lgbm(self, role: str) -> None:
        self.budget.reserve(lgbm_fits_reproduction=1)


def _d(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _fit(budget, data, design, day, config_id):
    wx, present = weather_matrix(design, data)
    x, _ = encode(data)
    return fit_origin(data, x, wx, target(data), present, day, config_id, charge=charge_fits(budget, 'control', main=False))


def _composites(m: dict, d: np.ndarray) -> dict:
    return {'v5': v5_central(m['HGL'], d), 'v3+D': v3d_central(m['A1'], m['B2'], d)}


def representative(root: Path, budget, fold: str, day: date, idents) -> dict:
    fit_ident, cp21_ident, hg_ident, hashes = idents
    data = load(root)
    design = weather_design(root)
    config_id = selected_config(root, fold)
    record = {'fold': fold, 'day': str(day), 'n_hours': int(len(data.rows(day))), 'config': config_id}
    sources = Sources(fold, data, fit_ident, cp21_ident, hg_ident, hashes, NEW_POLICIES)
    m = sources.members(day)
    cached = DDNNCache(fold, data, fit_ident).load(day)
    # 1. determinism: a fresh process refits the main run's four members bit for bit
    base = _fit(budget, data, design, day, config_id)
    record['determinism_vs_main_run'] = {
        'params_bitwise': [r['params_sha256'] for r in base['members']] == [r['params_sha256'] for r in cached['members']],
        'central_bitwise': bool(np.array_equal(base['central'], m['D'])),
        'quantiles_bitwise': bool(np.array_equal(base['quantiles'], m['D_quantiles'])),
        'epochs_equal': [r['best_epoch'] for r in base['members']] == [r['best_epoch'] for r in cached['members']]}
    base_c = _composites(m, base['central'])
    committed = pd.read_parquet(root / OUT / 'predictions.parquet')
    committed = committed.loc[pd.to_datetime(committed.delivery_date).dt.date.eq(day)]
    record['composites_bitwise_vs_committed'] = {
        p: bool(np.array_equal(base_c[p], committed.loc[committed.policy.eq(p)].sort_values('timestamp_utc').central.to_numpy(float)))
        for p in base_c}
    # 2. delivery-day prices masked and later loads scrambled -> exactly 0.0
    masked = data.frame.copy()
    masked.loc[masked.delivery_date >= day, 'price_eur_mwh'] = np.nan
    masked.loc[masked.delivery_date > day, 'load_forecast_mw'] = 1e8
    mdata = prepare(masked, data.p, data.spec)
    mfit = _fit(budget, mdata, design, day, config_id)
    mc = _composites(m, mfit['central'])
    record['delivery_day_and_future_mask'] = {'D_central': _d(mfit['central'], base['central']),
                                              'D_quantiles': _d(mfit['quantiles'], base['quantiles']),
                                              **{p: _d(mc[p], base_c[p]) for p in mc}}
    record['mask_epochs_and_params_unchanged'] = [r['params_sha256'] for r in mfit['members']] == [r['params_sha256'] for r in base['members']]
    # 3. available D-1 prices mutated non-uniformly (evening hours) -> D and the composites move
    changed = data.frame.copy()
    local = pd.to_datetime(changed.timestamp_utc, utc=True).dt.tz_convert('Europe/Berlin').dt.hour
    target_rows = changed.delivery_date.eq(day - timedelta(days=1)) & local.between(17, 21)
    changed.loc[target_rows, 'price_eur_mwh'] += 300.0
    cfit = _fit(budget, prepare(changed, data.p, data.spec), design, day, config_id)
    cc = _composites(m, cfit['central'])
    record['available_d1_nonuniform_price_mutation'] = {'D_central': _d(cfit['central'], base['central']),
                                                        'D_quantiles': _d(cfit['quantiles'], base['quantiles']),
                                                        **{p: _d(cc[p], base_c[p]) for p in cc}}
    record['d1_mutation_rows'] = int(target_rows.sum())
    # 4. weather after the origin only -> exactly 0.0
    future = design.table.copy()
    future.loc[future.index.get_level_values(0) > day, WX] = 1e6
    ffit = _fit(budget, data, replace(design, table=future), day, config_id)
    record['future_weather_mutation'] = {'D_central': _d(ffit['central'], base['central']),
                                         'D_quantiles': _d(ffit['quantiles'], base['quantiles'])}
    # 5. training weather permuted across dates (non-monotone) -> D moves
    permuted = design.table.copy()
    dates = permuted.index.get_level_values(0)
    pool = np.flatnonzero((dates < day) & np.isfinite(permuted[WX].to_numpy(float)).all(axis=1))
    values = permuted[WX].to_numpy(float, copy=True)
    values[pool] = values[np.random.default_rng(SEED).permutation(pool)]
    permuted[WX] = values
    pfit = _fit(budget, data, replace(design, table=permuted), day, config_id)
    record['training_weather_cross_date_permutation'] = {'D_central': _d(pfit['central'], base['central'])}
    # 6. the target day's own weather rearranged across its hours -> D moves
    swapped = design.table.copy()
    today = np.flatnonzero(swapped.index.get_level_values(0) == day)
    swapped.iloc[today, [swapped.columns.get_loc(c) for c in WX]] = swapped.iloc[today[::-1]][WX].to_numpy()
    sfit = _fit(budget, data, replace(design, table=swapped), day, config_id)
    record['target_day_weather_rearranged'] = {'D_central': _d(sfit['central'], base['central'])}
    # 7. inner-validation (early-stopping) outcomes altered -> early stopping changes
    vframe = data.frame.copy()
    vlocal = pd.to_datetime(vframe.timestamp_utc, utc=True).dt.tz_convert('Europe/Berlin').dt.hour
    vwin = vframe.delivery_date.between(day - timedelta(days=28), day - timedelta(days=8))
    vframe.loc[vwin, 'price_eur_mwh'] += 150.0 * np.sin(vlocal[vwin].to_numpy() / 3.0)
    vfit = _fit(budget, prepare(vframe, data.p, data.spec), design, day, config_id)
    record['validation_outcome_mutation'] = {
        'validation_history_changed': [r['validation_nll_history'] for r in vfit['members']] != [r['validation_nll_history'] for r in base['members']],
        'best_epochs_before': [r['best_epoch'] for r in base['members']], 'best_epochs_after': [r['best_epoch'] for r in vfit['members']],
        'D_central_change': _d(vfit['central'], base['central'])}
    # 8. post-gate and target-actual inputs refused by the strict raw schema
    extra = data.frame.copy()
    extra['actual_load_mw'] = 1.0
    try:
        prepare(extra, data.p, data.spec)
        record['extra_input_column_refused'] = False
    except ValueError:
        record['extra_input_column_refused'] = True
    # 9. the preprocessor is fitted on the training rows only (non-training rows altered: identical)
    record['preprocessor_equals_cached'] = base['preprocessor'] == cached['preprocessor']
    return record


def selection_controls(root: Path, budget) -> dict:
    """The configuration choice is training-only: mutated evaluation outcomes (on or after D0) leave
    it and its holdout losses exactly unchanged; mutated holdout outcomes change the losses."""
    m = origin_manifest(root)
    f = next(x for x in m['folds'] if x['fold'] == SELECTION_FOLD)
    first, d0 = date.fromisoformat(f['evaluation_start']), date.fromisoformat(f['warmup_start'])
    committed = json.loads((root / OUT / 'selection.json').read_text())['folds'][SELECTION_FOLD]
    design = weather_design(root)

    def run(frame):
        data = prepare(frame, base.p, base.spec)
        wx, present = weather_matrix(design, data)
        x, _ = encode(data)
        return fit_selection(data, x, wx, target(data), present, d0, charge=charge_fits(budget, 'control', main=False))

    base = load(root)  # the full frame: outcomes after D0, including the evaluation window, are present
    later = base.frame.copy()
    later.loc[later.delivery_date >= d0, 'price_eur_mwh'] += 400.0
    after = run(later)
    hold = base.frame.copy()
    hlocal = pd.to_datetime(hold.timestamp_utc, utc=True).dt.tz_convert('Europe/Berlin').dt.hour
    hwin = hold.delivery_date.between(d0 - timedelta(days=28), d0 - timedelta(days=1))
    hold.loc[hwin, 'price_eur_mwh'] += 120.0 * np.sin(hlocal[hwin].to_numpy() / 2.0)
    held = run(hold)
    return {'fold': SELECTION_FOLD, 'first_origin': str(d0), 'evaluation_start': str(first),
            'committed_choice': committed['selected'], 'committed_holdout_mae': committed['holdout_mae'],
            'evaluation_outcome_mutation': {'choice': after['selected'], 'holdout_mae': after['holdout_mae'],
                                            'identical': after['selected'] == committed['selected']
                                            and after['holdout_mae'] == committed['holdout_mae']},
            'holdout_outcome_mutation': {'choice': held['selected'], 'holdout_mae': held['holdout_mae'],
                                         'losses_changed': held['holdout_mae'] != committed['holdout_mae'],
                                         'choice_changed': held['selected'] != committed['selected']}}


def admission_states(root: Path, name: str = 'lineage.json'):
    shas = subprocess.check_output(['git', 'log', '--format=%H', '--', f'{OUT}/{name}'], cwd=root, text=True).split()
    for commit in shas:
        doc = json.loads(subprocess.check_output(['git', 'show', f'{commit}:{OUT}/{name}'], cwd=root))
        if doc['execution_stage'] == 'training_only_admission_complete_frozen_before_outer_scoring':
            return commit, doc['states']
    raise ValueError('no committed training-only admission freeze')


def state_controls(root: Path, budget, idents) -> dict:
    """Release, consume-once, restart replay and cache refusal on real states (policy-days charged)."""
    fit_ident, cp21_ident, hg_ident, hashes = idents
    commit, admitted = admission_states(root)
    data = load(root)
    fold, first = 'fold_1', date(2020, 7, 1)
    dates = list(pd.date_range(first, periods=10).date)
    truth = _truth(data)
    out = {'admission_freeze_commit': commit}
    runs = {}
    for label in ('continuous', 'restart'):
        states = {p: SharedResidualState.from_dict(admitted[fold][p]) for p in H_POLICIES}
        log, frames = {'origins': []}, []
        for i, d in enumerate(dates):
            if label == 'restart' and i == 5:
                states = {p: SharedResidualState.from_dict(json.loads(json.dumps(states[p].to_dict()))) for p in H_POLICIES}
            replay(data, fold, [d], Sources(fold, data, fit_ident, cp21_ident, hg_ident, hashes, NEW_POLICIES), states, truth,
                   None, log, 'control', frames, budget, 'policy_days_control')
        runs[label] = (pd.concat(frames, ignore_index=True), {p: json.dumps(states[p].to_dict(), sort_keys=True) for p in H_POLICIES})
    a, b = runs['continuous'], runs['restart']
    out['restart_replay_identical'] = bool(a[0].drop(columns=['origin_utc']).equals(b[0].drop(columns=['origin_utc'])) and a[1] == b[1])
    committed = pd.read_parquet(root / OUT / 'predictions.parquet')
    committed = committed.loc[pd.to_datetime(committed.delivery_date).dt.date.isin(dates)]
    num = ['central', *LABELS]
    mine = a[0].sort_values(['policy', 'timestamp_utc'])[num].to_numpy(float)
    theirs = committed.sort_values(['policy', 'timestamp_utc'])[num].to_numpy(float)
    out['replay_equals_committed_vectors'] = bool(mine.shape == theirs.shape and np.array_equal(mine, theirs))
    # release rule on v5's H state
    state = SharedResidualState.from_dict(admitted[fold]['v5'])
    budget.reserve(policy_days=4, policy_days_control=4)
    d0 = first
    rows = data.rows(d0)
    src = Sources(fold, data, fit_ident, cp21_ident, hg_ident, hashes, ('v5',))
    c = _twice(src.get(d0)[1]['v5'])
    state.release(d0, truth)
    state.issue(d0, data.index[rows], c, c, data.scale[rows])
    before = state.to_dict()
    state.release(d0 + timedelta(days=1), truth)
    after = state.to_dict()
    r = {'d_minus_1_error_refused': str(d0) in [p['day'] for p in after['pending']] and str(d0) not in after['consumed']}
    state.release(d0 + timedelta(days=2), truth)
    consumed = state.to_dict()
    r['d_minus_2_error_accepted'] = str(d0) in consumed['consumed'] and consumed['buffer'][-1]['day'] == str(d0)
    state.release(d0 + timedelta(days=2), truth)
    r['consume_once'] = state.to_dict()['buffer'] == consumed['buffer']
    try:
        state.issue(d0, data.index[rows], c, c, data.scale[rows])
        r['duplicate_issue_refused'] = False
    except ValueError:
        r['duplicate_issue_refused'] = True
    partial = SharedResidualState.from_dict(before)
    d1 = d0 + timedelta(days=1)
    r1 = data.rows(d1)
    c1 = _twice(src.get(d1)[1]['v5'])
    partial.issue(d1, data.index[r1[:-1]], c1[:-1], c1[:-1], data.scale[r1[:-1]])
    partial.release(d1 + timedelta(days=2), truth)
    r['partial_day_not_buffered'] = str(d1) not in [x['day'] for x in partial.to_dict()['buffer']] and any(
        t.get('status') == 'incomplete_issued_day' and t['feedback_day'] == str(d1) for t in partial.trace)
    nan_truth = SharedResidualState.from_dict(before)
    nan_truth.release(d0 + timedelta(days=2), lambda ix: np.full(len(ix), np.nan))
    r['unavailable_truth_stays_pending'] = str(d0) in [p['day'] for p in nan_truth.to_dict()['pending']]
    out['release_rule_v5'] = r
    # stale or wrong DDNN cache entries are refused (positive control: the intact entry loads)
    cache = DDNNCache(fold, data, fit_ident)
    tmp = art() / 'tmp-controls'
    shutil.rmtree(tmp, ignore_errors=True)
    probe = DDNNCache(fold, data, fit_ident, base=tmp)
    item = cache.load(first)
    probe.save(item)
    out['intact_ddnn_cache_loads'] = probe.load(first) is not None
    tampered = dict(item)
    tampered['central'] = list(item['central'])
    tampered['central'][0] += 1.0
    probe.save(tampered)
    try:
        probe.load(first)
        out['tampered_ddnn_cache_refused'] = False
    except ValueError:
        out['tampered_ddnn_cache_refused'] = True
    try:
        DDNNCache(fold, data, {**fit_ident, 'cp23_protocol_sha256': '0' * 64}, base=tmp).load(first)
        out['wrong_identity_ddnn_cache_refused'] = False
    except ValueError:
        out['wrong_identity_ddnn_cache_refused'] = True
    shutil.rmtree(tmp, ignore_errors=True)
    bad = dict(hashes)
    bad[('L-N', fold, str(first))] = '0' * 64
    try:
        Sources(fold, data, fit_ident, cp21_ident, hg_ident, bad, ('v5',)).get(first)
        out['v4_member_differing_from_committed_lineage_refused'] = False
    except ValueError:
        out['v4_member_differing_from_committed_lineage_refused'] = True
    out['intact_sources_load'] = len(Sources(fold, data, fit_ident, cp21_ident, hg_ident, hashes, ('v5',)).get(first)[1]['v5']) > 0
    return out


def population_controls(root: Path, idents) -> dict:
    """Composite parity on every key, the H path, every DDNN cache entry (finite, ordered, p50 = central,
    rearrangements), 23/24/25-hour identity, the boundary guard."""
    fit_ident = idents[0]
    new = pd.read_parquet(root / OUT / 'predictions.parquet')
    members = pd.read_parquet(root / OUT / 'members.parquet')
    key = ['fold', 'timestamp_utc']
    mem = members.sort_values(key).reset_index(drop=True)
    piv = {p: g.sort_values(key).reset_index(drop=True) for p, g in new.groupby('policy')}
    parity = {'v5': float(np.max(np.abs(piv['v5'].central.to_numpy() - (2 / 3) * mem.HGL.to_numpy() - mem.D.to_numpy() / 3))),
              'v3+D': float(np.max(np.abs(piv['v3+D'].central.to_numpy() - (2 / 3) * mem.HG.to_numpy() - mem.D.to_numpy() / 3)))}
    out = {'composite_parity_max_abs_eur_mwh': parity, 'composite_parity_passed': all(v <= COMPOSITE_TOLERANCE for v in parity.values()),
           'v4_members_rebuild_committed_v4_central': bool(np.array_equal(
               mem['A1'] / 3 + mem['B2'] / 3 + mem['L-N'] / 6 + mem['L-R'] / 6, mem.HGL)),
           'D_central_is_members_D': bool(np.array_equal(piv['D'].central.to_numpy(), mem.D.to_numpy())),
           'D_p50_is_its_central': bool(np.array_equal(piv['D'].central.to_numpy(), piv['D'].p50.to_numpy())),
           'keys_aligned_with_members': all(np.array_equal(piv[p].timestamp_utc.to_numpy(), mem.timestamp_utc.to_numpy()) for p in piv)}
    v4 = pd.read_parquet(root / 'reports/block-challenger/predictions.parquet', filters=[('policy', '==', 'HGL')]).sort_values(key)
    hg = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet', filters=[('policy', '==', 'HG')]).sort_values(key)
    out['members_v4_equals_committed_v4'] = bool(np.array_equal(mem.HGL.to_numpy(), v4.central.to_numpy(float)))
    out['members_v3_equals_committed_v3'] = bool(np.array_equal(mem.HG.to_numpy(), hg.central.to_numpy(float)))
    lineage = json.loads((root / OUT / 'lineage.json').read_text())
    layers = {}
    for rec in lineage['origins']:
        layers.setdefault(rec['policy'], set()).add(rec['layer'])
    out['layer_by_policy'] = {p: sorted(v) for p, v in layers.items()}
    out['v5_and_v3d_on_h_path'] = all(layers[p] == {'H'} for p in H_POLICIES)
    # every DDNN cache entry: finite, ordered, the p50 the central forecast, rearrangements recorded
    data_full = load(root)
    m = origin_manifest(root)
    checked, crossed, ok, configs = 0, 0, True, {}
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        early = load(root, before=first)
        for o in f['origins']:
            d = date.fromisoformat(o['day'])
            data = data_full if d >= first else early
            if not len(data.rows(d)):
                continue
            item = DDNNCache(f['fold'], data, fit_ident).load(d)
            q = np.asarray(item['quantiles'], float)
            c = np.asarray(item['central'], float)
            ok &= bool(np.isfinite(q).all() and (np.diff(q, axis=1) > 0).all() and np.array_equal(c, q[:, D.MEDIAN])
                       and len(item['members']) == len(D.SEEDS) and [r['seed'] for r in item['members']] == list(D.SEEDS))
            crossed += int(item['crossed_rows'])
            configs.setdefault(f['fold'], set()).add(item['config'])
            checked += 1
    out['ddnn_entries_checked'] = checked
    out['ddnn_entries_finite_ordered_p50_central_four_seeds'] = bool(ok)
    out['ddnn_rows_rearranged'] = crossed
    out['one_configuration_per_fold'] = {f: sorted(v) for f, v in configs.items()}
    out['configuration_fixed_through_each_fold'] = all(len(v) == 1 for v in configs.values())
    q = new[LABELS].to_numpy(float)
    counts = new.groupby(['policy', 'delivery_date']).size()
    out['rows_per_policy'] = new.groupby('policy').size().to_dict()
    out['quantiles_finite'] = bool(np.isfinite(q).all())
    out['quantiles_ordered'] = bool((np.diff(q, axis=1) >= 0).all())
    out['p50_separate_from_central_for_v5_and_v3d'] = bool(all((piv[p].p50 != piv[p].central).any() for p in H_POLICIES))
    out['dst_rows'] = {str(d): {p: int(counts.loc[(p, d)]) for p in NEW_POLICIES}
                       for d in (date(2026, 3, 29),) if ('v5', d) in counts.index}
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


def ddnn_code_controls(root: Path) -> dict:
    """§21.3 at controls time: the import audit, the default-suite gradient checks and the reference record."""
    run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/cp23/test_numpy_only.py',
                          'tests/cp23/test_ddnn_gradients.py', 'tests/cp23/test_reference_record.py'], cwd=root,
                         capture_output=True, text=True)
    record = require_passing_record(root)
    return {'import_audit': audit(root), 'gradient_and_audit_tests': {'exit_code': run.returncode,
                                                                       'summary': run.stdout.strip().splitlines()[-1]},
            'reference': {'passed': record['passed'], 'counts': record['counts'], 'ddnn_sha256': record['ddnn_sha256'],
                          'versions': record['versions']}}


def job_controls(root: Path, rest) -> int:
    check_protocol(root)
    idents = _identities(root)
    budget = ledger()
    result = {'schema': 'cp23-controls-v1', 'threshold_eur_mwh': THRESHOLD, 'seed': SEED, 'origins': []}
    for fold, day in DAYS:
        record = representative(root, budget, fold, day, idents)
        result['origins'].append(record)
        atomic(root / OUT / 'controls.json', result)
        print(json.dumps(record, default=str)[:3000], flush=True)
    result['selection'] = selection_controls(root, budget)
    result['state'] = state_controls(root, budget, idents)
    result['population'] = population_controls(root, idents)
    result['ddnn_code'] = ddnn_code_controls(root)
    checks = []
    for r in result['origins']:
        checks += [all(r['determinism_vs_main_run'].values()), all(r['composites_bitwise_vs_committed'].values()),
                   all(v == 0.0 for v in r['delivery_day_and_future_mask'].values()), r['mask_epochs_and_params_unchanged'],
                   all(v > THRESHOLD for v in r['available_d1_nonuniform_price_mutation'].values()),
                   all(v == 0.0 for v in r['future_weather_mutation'].values()),
                   all(v > THRESHOLD for v in r['training_weather_cross_date_permutation'].values()),
                   all(v > THRESHOLD for v in r['target_day_weather_rearranged'].values()),
                   r['validation_outcome_mutation']['validation_history_changed'], r['extra_input_column_refused'],
                   r['preprocessor_equals_cached']]
    s = result['selection']
    checks += [s['evaluation_outcome_mutation']['identical'], s['holdout_outcome_mutation']['losses_changed']]
    for k, v in result['state'].items():
        if isinstance(v, bool):
            checks.append(v)
        elif isinstance(v, dict):
            checks += [x for x in v.values() if isinstance(x, bool)]
    p = result['population']
    checks += [p['composite_parity_passed'], p['v4_members_rebuild_committed_v4_central'], p['D_central_is_members_D'],
               p['D_p50_is_its_central'], p['keys_aligned_with_members'], p['members_v4_equals_committed_v4'],
               p['members_v3_equals_committed_v3'], p['v5_and_v3d_on_h_path'], p['ddnn_entries_checked'] == 636,
               p['ddnn_entries_finite_ordered_p50_central_four_seeds'], p['configuration_fixed_through_each_fold'],
               p['quantiles_finite'], p['quantiles_ordered'], p['p50_separate_from_central_for_v5_and_v3d'],
               p['boundary_guard_refuses_2026_04_08'], p['boundary_guard_accepts_2026_04_07'],
               p['loader_max_date'] == '2026-04-07', p['max_delivery_date'] <= '2026-04-07',
               all(v == 10747 for v in p['rows_per_policy'].values()), p['spring_2026_03_29_canonical_hours'] == 23,
               p['autumn_2025_10_26_canonical_hours'] == 25]
    c = result['ddnn_code']
    checks += [c['import_audit']['passed'], c['gradient_and_audit_tests']['exit_code'] == 0, c['reference']['passed']]
    result['all_passed'] = bool(all(checks))
    result['checks'] = len(checks)
    result['written_utc'] = stamp()
    atomic(root / OUT / 'controls.json', result)
    print(json.dumps({'all_passed': result['all_passed'], 'checks': len(checks), 'selection': s, 'state': result['state'],
                      'population': {k: v for k, v in p.items() if k != 'layer_by_policy'}}, default=str), flush=True)
    return 0 if result['all_passed'] else 7


def job_reproduce(root: Path, rest) -> int:
    """§21.10 item 6: an independent representative HG and v4 slice, refitted and compared with the
    committed CP-20 and CP-21 vectors bit for bit (charged as reproduction)."""
    check_protocol(root)
    budget = ledger()
    charges = ReproductionCharges(budget)
    data = load(root)
    design = weather_design(root)
    wx, present = weather_matrix(design, data)
    aug, apresent = augment(data, design)
    hg_acc = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet', filters=[('policy', '==', 'HG')])
    v4_acc = pd.read_parquet(root / 'reports/block-challenger/predictions.parquet')
    out = {'schema': 'cp23-reproduction-v1', 'origins': []}
    for fold, day in REPRODUCE:
        rows = data.rows(day)
        comps = refit(charges, 'reproduction', aug, apresent, day, rows)
        lgbm = {arm: fit_arm(data, wx, present, day, arm, charge=charges.lgbm)[0] for arm in ('L-N', 'L-R')}
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
    return 0 if out['all_bitwise'] else 6
