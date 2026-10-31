"""CP-24's causal and integrity controls on real data (capstone v21-r11 §23.10, with §21.7's controls and
§20.7's and §17.7's applicable ones). DDNN-2 refits are charged as control fits.

Every negative assertion is paired with a positive control that can fail and survives DDNN-2's own
transforms: §4's and the MAD normalisation cancel a uniform price scaling, so the D-1 control uses a
non-uniform mutation (evening hours only), and weather influence is shown with non-monotone
perturbations (a member that uses the GFS block; if no frozen member does, the rank-1 configuration
with the GFS block added, labelled as a control configuration).

* **Representative origins** (fold 1's and fold 4's first evaluation days): determinism against the
  attempt's fits; delivery-day and future masking; a non-uniform D-1 mutation; future weather; weather
  permutations; held-out-week outcomes; recency; training-only statistics; the origin statistics; exact
  inversion; the strict raw schema.
* **The search and the gate, fold by fold** (§23.10): outcomes on or after D0_f - 56 (search) and D0_f
  (gate) change nothing; a validation-batch outcome, an earlier gate-day outcome and an earlier fold's
  evaluation day inside a later fold's window can change the fits.
* **Weather coverage**: no pre-fold fit trains on an uncovered day; removing the exclusion changes a
  fold-4 fit's training rows (the v4 wrapper's controls are `reports/ddnn2/v4-parity.json`).
* **Pre-registration**: the protocol commit is an ancestor of the first commit holding the attempt's
  forecasts and precedes its first fit in the ledger; the entry points refuse a changed protocol.
* **States, caches and the population**: restart replay, the release rule, consume-once, partial days,
  cache refusal, composite parity on every key, the H path, every DDNN-2 entry, the boundary guard.
* **Code**: the import audit, the finite-difference tests and the reference record.
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
from . import ddnn2 as M
from . import design as G
from .audit import audit
from .budget import atomic, charge_fits, ledger
from .execution import (COMPOSITE_TOLERANCE, H_POLICIES, NEW_POLICIES, D2Cache, Sources, _identities, _truth, _twice,
                        fit_identity, replay)
from .inputs import BOUNDARY, BoundaryViolation, guard_boundary, load, origin_manifest, weather_design
from .jobs import art, stamp
from .member import Keys, emit_rows, fit_member
from .preflight import fold_table
from .protocol import attempt_dir, check_protocol
from .reference import require_passing_record
from .search import score_batch

DAYS = (('fold_1', date(2020, 7, 1)), ('fold_4', date(2025, 5, 1)))
THRESHOLD = 1e-6
SEED = 15042
WX = ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean']


def _d(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


class Ctx:
    """One representative origin's inputs and a refit helper (one member, charged as a control fit)."""

    def __init__(self, root, budget, data, design, day):
        self.root, self.budget, self.data, self.design, self.day = root, budget, data, design, day
        self.dd = G.build(data, design)
        self.keys = Keys.from_data(data, self.dd)

    def variant(self, frame=None, design=None):
        data = self.data if frame is None else prepare(frame, self.data.p, self.data.spec)
        dd = G.build(data, design or self.design)
        return data, dd, Keys.from_data(data, dd)

    def fit(self, member, data=None, dd=None, keys=None, *, exclude=False):
        data, dd, keys = data or self.data, dd or self.dd, keys or self.keys
        di = np.array([dd.ix(self.day)])
        out = fit_member(dd, self.day, member['config'], member['seed'], di, exclude_uncovered=exclude,
                         charge=charge_fits(self.budget, 'control'))
        rows = keys.of_days(di)
        out['q'] = emit_rows(keys, rows, di, out['eur'])
        return out


def _gfs_member(members: list[dict]) -> tuple[dict, str]:
    for m in members:
        if 'gfs' in m['config']['groups']:
            return m, f'frozen member rank {m["rank"]} (uses the GFS block)'
    m = dict(members[0])
    m['config'] = {**m['config'], 'id': m['config']['id'] + '+gfs-control', 'groups': [*m['config']['groups'], 'gfs']}
    return m, 'control configuration: the rank-1 configuration with the GFS block added (no frozen member uses it)'


def representative(root: Path, k: int, budget, fold: str, day: date) -> dict:
    p = json.loads((attempt_dir(root, k) / 'protocol.json').read_text())
    members = p['folds'][fold]['ensemble']
    data, design = load(root), weather_design(root)
    cx = Ctx(root, budget, data, design, day)
    cached = D2Cache(k, fold, data, fit_identity(root, k)).load(day)
    rec = {'fold': fold, 'day': str(day), 'n_hours': int(len(data.rows(day))), 'member_0': members[0]['config']['id']}
    m0 = members[0]
    base = cx.fit(m0)
    rec['determinism_member_0'] = {
        'params_bitwise': base['member']['params_sha256'] == cached['members'][0]['record']['params_sha256'],
        'quantiles_bitwise': bool(np.array_equal(base['q'], np.asarray(cached['members'][0]['quantiles']))),
        'best_epoch_equal': base['member']['best_epoch'] == cached['members'][0]['record']['best_epoch']}
    # exact inversion at every quantile, for every cached member
    exact = True
    for mbr in cached['members']:
        jsu = np.asarray(mbr['jsu'])                                     # 24 x 4
        qt = M.jsu_quantiles(jsu[:, 0], jsu[:, 1], jsu[:, 2], jsu[:, 3])
        z = np.clip(M.inverse_transform(mbr['transform'], qt), -mbr['cap_z'], mbr['cap_z'])
        eur = mbr['centre'] + mbr['scale'] * z
        exact &= bool(np.array_equal(eur[data.hours[data.rows(day)]], np.asarray(mbr['quantiles'])))
    rec['inversion_exact_every_member_every_quantile'] = exact
    # delivery-day prices masked, later loads scrambled -> exactly 0.0, same weights
    masked = data.frame.copy()
    masked.loc[masked.delivery_date >= day, 'price_eur_mwh'] = np.nan
    masked.loc[masked.delivery_date > day, 'load_forecast_mw'] = 1e8
    md, mdd, mk = cx.variant(masked)
    mfit = cx.fit(m0, md, mdd, mk)
    rec['delivery_day_and_future_mask'] = {'quantiles': _d(mfit['q'], base['q']),
                                           'params_unchanged': mfit['member']['params_sha256'] == base['member']['params_sha256']}
    i = cx.dd.ix(day)
    rec['origin_statistics_unchanged_by_delivery_day_prices'] = bool(
        all(mdd.centre[s][i] == cx.dd.centre[s][i] and mdd.scale[s][i] == cx.dd.scale[s][i] for s in ('s4', 'mad')))
    # available D-1 prices mutated non-uniformly (evening hours) -> the forecast and the origin statistics move
    changed = data.frame.copy()
    local = pd.to_datetime(changed.timestamp_utc, utc=True).dt.tz_convert('Europe/Berlin').dt.hour
    rows = changed.delivery_date.eq(day - timedelta(days=1)) & local.between(17, 21)
    changed.loc[rows, 'price_eur_mwh'] += 300.0
    cd, cdd, ck = cx.variant(changed)
    rec['available_d1_nonuniform_price_mutation'] = {'quantiles': _d(cx.fit(m0, cd, cdd, ck)['q'], base['q']),
                                                     'rows_mutated': int(rows.sum())}
    rec['origin_statistics_move_with_d1_prices'] = bool(any(cdd.centre[s][i] != cx.dd.centre[s][i] for s in ('s4', 'mad')))
    # weather: a member that uses the GFS block
    gm, label = _gfs_member(members)
    gbase = cx.fit(gm)
    future = design.table.copy()
    future.loc[future.index.get_level_values(0) > day, WX] = 1e6
    fd = cx.variant(design=replace(design, table=future))
    permuted = design.table.copy()
    dates = permuted.index.get_level_values(0)
    pool = np.flatnonzero((dates < day) & np.isfinite(permuted[WX].to_numpy(float)).all(axis=1))
    vals = permuted[WX].to_numpy(float, copy=True)
    vals[pool] = vals[np.random.default_rng(SEED).permutation(pool)]
    permuted[WX] = vals
    pd_ = cx.variant(design=replace(design, table=permuted))
    swapped = design.table.copy()
    today = np.flatnonzero(swapped.index.get_level_values(0) == day)
    swapped.iloc[today, [swapped.columns.get_loc(c) for c in WX]] = swapped.iloc[today[::-1]][WX].to_numpy()
    sd = cx.variant(design=replace(design, table=swapped))
    rec['weather'] = {'member': label, 'future_weather_mutation': _d(cx.fit(gm, *fd)['q'], gbase['q']),
                      'training_weather_cross_date_permutation': _d(cx.fit(gm, *pd_)['q'], gbase['q']),
                      'target_day_weather_rearranged': _d(cx.fit(gm, *sd)['q'], gbase['q'])}
    # held-out weeks' outcomes altered -> early stopping changes; recency: a recent training day moves the fit
    weeks = [date.fromisoformat(w) for w in base['window']['held_out_weeks']]
    hframe = data.frame.copy()
    hloc = pd.to_datetime(hframe.timestamp_utc, utc=True).dt.tz_convert('Europe/Berlin').dt.hour
    hmask = np.zeros(len(hframe), bool)
    for w in weeks:
        hmask |= hframe.delivery_date.between(w, w + timedelta(days=6)).to_numpy()
    hframe.loc[hmask, 'price_eur_mwh'] += 120.0 * np.sin(hloc[hmask].to_numpy() / 2.0)
    hfit = cx.fit(m0, *cx.variant(hframe))
    rec['held_out_week_outcome_mutation'] = {
        'stopping_history_changed': hfit['member']['stopping_metric_history'] != base['member']['stopping_metric_history'],
        'best_epoch_before': base['member']['best_epoch'], 'best_epoch_after': hfit['member']['best_epoch']}
    lower, ix, _ = G.window(cx.dd, day, exclude_uncovered=False)
    train_ix, hold_ix = G.split(cx.dd, ix, weeks)
    recent = [cx.dd.ix(day - timedelta(days=j)) for j in range(1, 8)]
    rec['recency'] = {'last_seven_days_trained_on': bool(all(r in set(train_ix) or not cx.dd.mask[r].any() for r in recent)),
                      'no_held_out_day_in_last_seven': bool(not set(recent) & set(hold_ix))}
    rframe = data.frame.copy()
    rframe.loc[rframe.delivery_date.eq(day - timedelta(days=5)), 'price_eur_mwh'] += 80.0
    rec['recency']['mutating_d_minus_5_moves_the_fit'] = _d(cx.fit(m0, *cx.variant(rframe))['q'], base['q'])
    # training-only statistics: non-training rows never move the preprocessor; a training row does
    form = m0['config']['transform']
    C_tr, B_tr, _, _ = G.raw_inputs(cx.dd, train_ix, tuple(m0['config']['groups']), form)
    pre = G.Preprocessor(C_tr, B_tr)
    pre_again = G.Preprocessor(C_tr.copy(), B_tr.copy())
    C_alt = C_tr.copy()
    C_alt[0] += 37.0
    pre_alt = G.Preprocessor(C_alt, B_tr)
    rec['preprocessor'] = {'equals_member_record': pre.record() == base['preprocessor'],
                           'non_training_rows_cannot_enter': pre_again.record() == pre.record(),
                           'a_training_row_moves_it': pre_alt.record() != pre.record()}
    # the strict raw schema refuses post-gate and target-actual columns
    extra = data.frame.copy()
    extra['actual_load_mw'] = 1.0
    try:
        prepare(extra, data.p, data.spec)
        rec['extra_input_column_refused'] = False
    except ValueError:
        rec['extra_input_column_refused'] = True
    return rec


def ensemble_determinism(root: Path, k: int, budget, fold: str, day: date) -> dict:
    from .execution import fit_origin
    p = json.loads((attempt_dir(root, k) / 'protocol.json').read_text())
    data = load(root)
    dd = G.build(data, weather_design(root))
    keys = Keys.from_data(data, dd)
    out = fit_origin(dd, keys, data, day, p['folds'][fold]['ensemble'], charge_fits(budget, 'control'))
    cached = D2Cache(k, fold, data, fit_identity(root, k)).get(day)
    return {'fold': fold, 'day': str(day), 'central_bitwise': bool(np.array_equal(out['central'], cached['central'])),
            'quantiles_bitwise': bool(np.array_equal(out['quantiles'], cached['quantiles']))}


def _search_record(root: Path, r: int, fold: str, trial: int, batch: int) -> dict:
    return json.loads((art() / 'rounds' / f'round-{r}' / 'search' / fold / f't{trial:03d}-b{batch:02d}.json').read_text())


def search_and_gate(root: Path, k: int, budget) -> dict:
    p = json.loads((attempt_dir(root, k) / 'protocol.json').read_text())
    r = p['round']
    folds = {f['fold']: f for f in fold_table(root)}
    design = weather_design(root)
    full = load(root)
    out = {}
    # fold 2's rank-1 trial on its most recent batch, from the committed search ledger
    f2 = folds['fold_2']
    ledger_doc = json.loads((root / 'reports/ddnn2/rounds' / f'round-{r}' / 'search-ledger.json').read_text())
    t_id = ledger_doc['folds']['fold_2']['ranking_all_batches'][0]
    trial = next(t for t in ledger_doc['folds']['fold_2']['trials'] if t['config']['id'] == t_id)
    rec0 = _search_record(root, r, 'fold_2', trial['trial'], 0)
    b0 = date.fromisoformat(rec0['batch_start'])

    def batch_score(frame):
        data = prepare(frame, full.p, full.spec)
        dd = G.build(data, design)
        keys = Keys.from_data(data, dd)
        days = np.array([dd.ix(b0 + timedelta(days=j)) for j in range(28)])
        fit = fit_member(dd, b0, trial['config'], rec0['seed'], days, exclude_uncovered=True, charge=charge_fits(budget, 'control'))
        return score_batch(keys, days, fit['eur'])['pinball_sum'], fit['member']['params_sha256']

    later = full.frame.copy()
    later.loc[later.delivery_date >= f2['search_cutoff'], 'price_eur_mwh'] += 400.0
    s_later, _ = batch_score(later)
    out['search_outcomes_on_or_after_d0_minus_56_change_nothing'] = {
        'fold': 'fold_2', 'trial': t_id, 'batch_start': str(b0), 'pinball_sum_ledger': rec0['pinball_sum'],
        'pinball_sum_with_later_outcomes_mutated': s_later, 'identical': s_later == rec0['pinball_sum']}
    vb = full.frame.copy()
    vb.loc[vb.delivery_date.eq(b0 + timedelta(days=10)), 'price_eur_mwh'] += 90.0
    s_vb, _ = batch_score(vb)
    out['validation_batch_outcome_changes_the_search'] = {'pinball_sum': s_vb, 'changed': s_vb != rec0['pinball_sum']}
    early = full.frame.copy()
    early.loc[early.delivery_date.eq(date(2020, 8, 3)), 'price_eur_mwh'] += 250.0   # a fold-1 evaluation day
    s_early, _ = batch_score(early)
    out['earlier_fold_evaluation_day_changes_a_later_fold_fit'] = {'mutated_day': '2020-08-03 (fold_1 evaluation)',
                                                                   'pinball_sum': s_early, 'changed': s_early != rec0['pinball_sum']}
    # the gate: fold 4, member 0, on its first gate day (data materialised in full; outcomes on/after D0 mutated)
    f4 = folds['fold_4']
    g0 = f4['gate'][0]
    member = json.loads((root / 'reports/ddnn2/rounds' / f'round-{r}' / 'ensembles.json').read_text())['folds']['fold_4']['members'][0]
    stored = json.loads((art() / 'rounds' / f'round-{r}' / 'gate' / 'fold_4' / str(g0) / 'm0.json').read_text())

    def gate_fit(frame, day):
        data = prepare(frame, full.p, full.spec)
        dd = G.build(data, design)
        fit = fit_member(dd, day, member['config'], member['seed'], np.array([dd.ix(day)]), exclude_uncovered=True,
                         charge=charge_fits(budget, 'control'))
        return fit

    glater = full.frame.copy()
    glater.loc[glater.delivery_date >= f4['d0'], 'price_eur_mwh'] -= 300.0
    gl = gate_fit(glater, g0)
    out['gate_outcomes_on_or_after_d0_change_nothing'] = {'fold': 'fold_4', 'day': str(g0),
                                                          'max_abs_difference': _d(gl['eur'][0], stored['eur']),
                                                          'identical': bool(np.array_equal(gl['eur'][0], np.asarray(stored['eur'])))}
    g2 = g0 + timedelta(days=20)
    stored2 = json.loads((art() / 'rounds' / f'round-{r}' / 'gate' / 'fold_4' / str(g2) / 'm0.json').read_text())
    gmut = full.frame.copy()
    gmut.loc[gmut.delivery_date.eq(g0 + timedelta(days=5)), 'price_eur_mwh'] += 200.0
    out['earlier_gate_day_outcome_changes_a_later_gate_fit'] = {'mutated_day': str(g0 + timedelta(days=5)), 'fit_day': str(g2),
                                                               'max_abs_difference': _d(gate_fit(gmut, g2)['eur'][0], stored2['eur'])}
    # weather coverage: no pre-fold fit trains on an uncovered day; removing the exclusion changes the rows
    gdata = load(root, before=f4['d0'])
    gdd = G.build(gdata, design)
    _, with_excl, excluded = G.window(gdd, g0, exclude_uncovered=True)
    _, without, _ = G.window(gdd, g0, exclude_uncovered=False)
    gap = (np.datetime64('2022-09-29'), np.datetime64('2023-03-24'))
    in_gap = (gdd.days[with_excl] >= gap[0]) & (gdd.days[with_excl] <= gap[1])
    out['weather_coverage'] = {'fold4_gate_fit_trains_on_no_uncovered_day': bool(not in_gap.any()),
                               'excluded_days': len(excluded),
                               'removing_the_exclusion_changes_the_training_rows': bool(len(without) > len(with_excl)),
                               'stored_gate_member_excluded_count': len(stored['window']['excluded_uncovered_days'])}
    ledger_ok = True
    for fold in ('fold_4', 'fold_5'):
        for t in ledger_doc['folds'][fold]['trials']:
            for b in t['batches']:
                start = date.fromisoformat(b['start'])
                reaches = max(date(2019, 1, 1), start - timedelta(days=728)) <= date(2023, 3, 24)
                if b['status'] == 'ok' and reaches and not b['excluded_uncovered_days']:
                    ledger_ok = False
    out['weather_coverage']['every_search_fit_reaching_the_gap_excluded_it'] = ledger_ok
    return out


def preregistration(root: Path, k: int) -> dict:
    ad = attempt_dir(root, k)
    rel = lambda name: str((ad / name).relative_to(root))  # noqa: E731
    protocol_commit = subprocess.check_output(['git', 'log', '--reverse', '--format=%H', '--', rel('protocol.json')], cwd=root,
                                              text=True).split()[0]
    first_lineage = subprocess.check_output(['git', 'log', '--reverse', '--format=%H', '--', rel('lineage.json')], cwd=root,
                                            text=True).split()[0]
    first_predictions = subprocess.check_output(['git', 'log', '--reverse', '--format=%H', '--', rel('predictions.parquet')],
                                                cwd=root, text=True).split()[0]
    anc = lambda a, b: subprocess.run(['git', 'merge-base', '--is-ancestor', a, b], cwd=root).returncode == 0  # noqa: E731
    events = ledger().read()['events']
    first_fit = next((i for i, e in enumerate(events) if e['event'] == f'attempt_{k}_first_fit'), None)
    fits_start = next((i for i, e in enumerate(events) if e['event'] == 'fits_start' and e.get('attempt') == k), None)
    out = {'protocol_commit': protocol_commit, 'first_lineage_commit': first_lineage, 'first_predictions_commit': first_predictions,
           'protocol_ancestor_of_first_lineage': anc(protocol_commit, first_lineage) and protocol_commit != first_lineage,
           'protocol_ancestor_of_first_predictions': anc(protocol_commit, first_predictions),
           'ledger_first_fit_event_records_the_protocol_commit': first_fit is not None
           and events[first_fit].get('protocol_commit') == protocol_commit,
           'ledger_order_protocol_before_fits': first_fit is not None and fits_start is not None and first_fit < fits_start}
    tmp = art() / 'tmp-controls' / 'protocol'
    shutil.rmtree(tmp, ignore_errors=True)
    out['entry_points_refuse_a_missing_protocol'] = False
    try:
        check_protocol(tmp, k)
    except (ValueError, FileNotFoundError):
        out['entry_points_refuse_a_missing_protocol'] = True
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def admission_states(root: Path, k: int):
    name = str((attempt_dir(root, k) / 'lineage.json').relative_to(root))
    shas = subprocess.check_output(['git', 'log', '--format=%H', '--', name], cwd=root, text=True).split()
    for commit in shas:
        doc = json.loads(subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=root))
        if doc['execution_stage'] == 'training_only_admission_complete_frozen_before_outer_scoring':
            return commit, doc['states']
    raise ValueError('no committed training-only admission freeze')


def state_controls(root: Path, k: int, budget) -> dict:
    idents = _identities(root, k)
    fit_ident, cp21_ident, hg_ident, hashes = idents
    commit, admitted = admission_states(root, k)
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
            replay(k, data, fold, [d], Sources(k, fold, data, fit_ident, cp21_ident, hg_ident, hashes, NEW_POLICIES), states,
                   truth, None, log, 'control', frames, budget, 'policy_days_control')
        runs[label] = (pd.concat(frames, ignore_index=True), {p: json.dumps(states[p].to_dict(), sort_keys=True) for p in H_POLICIES})
    a, b = runs['continuous'], runs['restart']
    out['restart_replay_identical'] = bool(a[0].drop(columns=['origin_utc']).equals(b[0].drop(columns=['origin_utc'])) and a[1] == b[1])
    committed = pd.read_parquet(attempt_dir(root, k) / 'predictions.parquet')
    committed = committed.loc[pd.to_datetime(committed.delivery_date).dt.date.isin(dates)]
    num = ['central', *LABELS]
    mine = a[0].sort_values(['policy', 'timestamp_utc'])[num].to_numpy(float)
    theirs = committed.sort_values(['policy', 'timestamp_utc'])[num].to_numpy(float)
    out['replay_equals_committed_vectors'] = bool(mine.shape == theirs.shape and np.array_equal(mine, theirs))
    state = SharedResidualState.from_dict(admitted[fold]['v5'])
    budget.reserve(policy_days=4, policy_days_control=4)
    d0 = first
    rows = data.rows(d0)
    src = Sources(k, fold, data, fit_ident, cp21_ident, hg_ident, hashes, ('v5',))
    c = _twice(src.get(d0)[1]['v5'])
    state.release(d0, truth)
    state.issue(d0, data.index[rows], c, c, data.scale[rows])
    before = state.to_dict()
    state.release(d0 + timedelta(days=1), truth)
    after = state.to_dict()
    rr = {'d_minus_1_error_refused': str(d0) in [x['day'] for x in after['pending']] and str(d0) not in after['consumed']}
    state.release(d0 + timedelta(days=2), truth)
    consumed = state.to_dict()
    rr['d_minus_2_error_accepted'] = str(d0) in consumed['consumed'] and consumed['buffer'][-1]['day'] == str(d0)
    state.release(d0 + timedelta(days=2), truth)
    rr['consume_once'] = state.to_dict()['buffer'] == consumed['buffer']
    try:
        state.issue(d0, data.index[rows], c, c, data.scale[rows])
        rr['duplicate_issue_refused'] = False
    except ValueError:
        rr['duplicate_issue_refused'] = True
    partial = SharedResidualState.from_dict(before)
    d1 = d0 + timedelta(days=1)
    r1 = data.rows(d1)
    c1 = _twice(src.get(d1)[1]['v5'])
    partial.issue(d1, data.index[r1[:-1]], c1[:-1], c1[:-1], data.scale[r1[:-1]])
    partial.release(d1 + timedelta(days=2), truth)
    rr['partial_day_not_buffered'] = str(d1) not in [x['day'] for x in partial.to_dict()['buffer']] and any(
        t.get('status') == 'incomplete_issued_day' and t['feedback_day'] == str(d1) for t in partial.trace)
    nan_truth = SharedResidualState.from_dict(before)
    nan_truth.release(d0 + timedelta(days=2), lambda ix: np.full(len(ix), np.nan))
    rr['unavailable_truth_stays_pending'] = str(d0) in [x['day'] for x in nan_truth.to_dict()['pending']]
    out['release_rule_v5'] = rr
    cache = D2Cache(k, fold, data, fit_ident)
    tmp = art() / 'tmp-controls'
    shutil.rmtree(tmp, ignore_errors=True)
    probe = D2Cache(k, fold, data, fit_ident, base=tmp)
    item = cache.load(first)
    probe.save(item)
    out['intact_d2_cache_loads'] = probe.load(first) is not None
    tampered = dict(item)
    tampered['central'] = list(item['central'])
    tampered['central'][0] += 1.0
    probe.save(tampered)
    try:
        probe.load(first)
        out['tampered_d2_cache_refused'] = False
    except ValueError:
        out['tampered_d2_cache_refused'] = True
    try:
        D2Cache(k, fold, data, {**fit_ident, 'protocol_sha256': '0' * 64}, base=tmp).load(first)
        out['wrong_identity_d2_cache_refused'] = False
    except ValueError:
        out['wrong_identity_d2_cache_refused'] = True
    shutil.rmtree(tmp, ignore_errors=True)
    bad = dict(hashes)
    bad[('L-N', fold, str(first))] = '0' * 64
    try:
        Sources(k, fold, data, fit_ident, cp21_ident, hg_ident, bad, ('v5',)).get(first)
        out['v4_member_differing_from_committed_lineage_refused'] = False
    except ValueError:
        out['v4_member_differing_from_committed_lineage_refused'] = True
    out['intact_sources_load'] = len(Sources(k, fold, data, fit_ident, cp21_ident, hg_ident, hashes, ('v5',)).get(first)[1]['v5']) > 0
    return out


def population_controls(root: Path, k: int) -> dict:
    ad = attempt_dir(root, k)
    new = pd.read_parquet(ad / 'predictions.parquet')
    members = pd.read_parquet(ad / 'members.parquet')
    key = ['fold', 'timestamp_utc']
    mem = members.sort_values(key).reset_index(drop=True)
    piv = {p: g.sort_values(key).reset_index(drop=True) for p, g in new.groupby('policy')}
    L = mem['L-N'] / 2 + mem['L-R'] / 2
    parity = {'v5': float(np.max(np.abs(piv['v5'].central.to_numpy() - mem.HGL.to_numpy() - (mem.D2.to_numpy() - L.to_numpy()) / 6))),
              'v3+D2': float(np.max(np.abs(piv['v3+D2'].central.to_numpy() - (2 / 3) * mem.HG.to_numpy() - mem.D2.to_numpy() / 3)))}
    out = {'composite_parity_max_abs_eur_mwh': parity, 'composite_parity_passed': all(v <= COMPOSITE_TOLERANCE for v in parity.values()),
           'v4_members_rebuild_committed_v4_central': bool(np.array_equal(
               mem['A1'] / 3 + mem['B2'] / 3 + mem['L-N'] / 6 + mem['L-R'] / 6, mem.HGL)),
           'D2_central_is_members_D2': bool(np.array_equal(piv['D2'].central.to_numpy(), mem.D2.to_numpy())),
           'D2_p50_is_its_central': bool(np.array_equal(piv['D2'].central.to_numpy(), piv['D2'].p50.to_numpy())),
           'keys_aligned_with_members': all(np.array_equal(piv[p].timestamp_utc.to_numpy(), mem.timestamp_utc.to_numpy()) for p in piv)}
    v4 = pd.read_parquet(root / 'reports/block-challenger/predictions.parquet', filters=[('policy', '==', 'HGL')]).sort_values(key)
    hg = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet', filters=[('policy', '==', 'HG')]).sort_values(key)
    out['members_v4_equals_committed_v4'] = bool(np.array_equal(mem.HGL.to_numpy(), v4.central.to_numpy(float)))
    out['members_v3_equals_committed_v3'] = bool(np.array_equal(mem.HG.to_numpy(), hg.central.to_numpy(float)))
    lineage = json.loads((ad / 'lineage.json').read_text())
    layers = {}
    for rec in lineage['origins']:
        layers.setdefault(rec['policy'], set()).add(rec['layer'])
    out['layer_by_policy'] = {p: sorted(v) for p, v in layers.items()}
    out['v5_and_v3d2_on_h_path'] = all(layers[p] == {'H'} for p in H_POLICIES)
    out['max_composite_gap_in_lineage'] = max((r.get('composite_gap') or 0.0) for r in lineage['origins'])
    p = json.loads((ad / 'protocol.json').read_text())
    fit_ident = fit_identity(root, k)
    data_full = load(root)
    checked, crossed, ok = 0, 0, True
    for f in origin_manifest(root)['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        early = load(root, before=first)
        frozen = [(m['config']['id'], m['seed']) for m in p['folds'][f['fold']]['ensemble']]
        for o in f['origins']:
            d = date.fromisoformat(o['day'])
            data = data_full if d >= first else early
            if not len(data.rows(d)):
                continue
            item = D2Cache(k, f['fold'], data, fit_ident).load(d)
            q = np.asarray(item['quantiles'], float)
            c = np.asarray(item['central'], float)
            ok &= bool(np.isfinite(q).all() and (np.diff(q, axis=1) >= 0).all() and np.array_equal(c, q[:, M.MEDIAN])
                       and [(x['config'], x['seed']) for x in item['members']] == frozen)
            crossed += int(item['crossed_rows'])
            checked += 1
    out['d2_entries_checked'] = checked
    out['d2_entries_finite_ordered_p50_central_frozen_members'] = bool(ok)
    out['d2_rows_rearranged'] = crossed
    q = new[LABELS].to_numpy(float)
    counts = new.groupby(['policy', 'delivery_date']).size()
    out['rows_per_policy'] = new.groupby('policy').size().to_dict()
    out['quantiles_finite'] = bool(np.isfinite(q).all())
    out['quantiles_ordered'] = bool((np.diff(q, axis=1) >= 0).all())
    out['p50_separate_from_central_for_v5_and_v3d2'] = bool(all((piv[x].p50 != piv[x].central).any() for x in H_POLICIES))
    out['dst_rows'] = {str(d): {x: int(counts.loc[(x, pd.Timestamp(d))]) if (x, pd.Timestamp(d)) in counts.index else
                                int(counts.get((x, d), 0)) for x in NEW_POLICIES} for d in (date(2026, 3, 29),)}
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


def code_controls(root: Path) -> dict:
    run = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests/cp24/test_numpy_only.py',
                          'tests/cp24/test_ddnn2_gradients.py', 'tests/cp24/test_reference_record.py',
                          'tests/cp24/test_search_procedure.py'], cwd=root, capture_output=True, text=True)
    record = require_passing_record(root)
    return {'import_audit': audit(root), 'gradient_audit_search_tests': {'exit_code': run.returncode,
                                                                          'summary': run.stdout.strip().splitlines()[-1]},
            'reference': {'passed': record['passed'], 'counts': record['counts'], 'model_sha256': record['model_sha256'],
                          'versions': record['versions']}}


def job_controls(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True, choices=(1, 2))
    args = ap.parse_args(rest)
    root, k = Path(root), args.attempt
    check_protocol(root, k)
    budget = ledger()
    out_path = attempt_dir(root, k) / 'controls.json'
    result = {'schema': 'cp24-controls-v1', 'attempt': k, 'threshold_eur_mwh': THRESHOLD, 'seed': SEED, 'origins': []}
    for fold, day in DAYS:
        rec = representative(root, k, budget, fold, day)
        result['origins'].append(rec)
        atomic(out_path, result)
        print(json.dumps(rec, default=str)[:2000], flush=True)
    result['ensemble_determinism'] = ensemble_determinism(root, k, budget, *DAYS[0])
    result['search_and_gate'] = search_and_gate(root, k, budget)
    result['preregistration'] = preregistration(root, k)
    result['state'] = state_controls(root, k, budget)
    result['population'] = population_controls(root, k)
    result['code'] = code_controls(root)
    checks = []
    for r in result['origins']:
        checks += [all(r['determinism_member_0'].values()), r['inversion_exact_every_member_every_quantile'],
                   r['delivery_day_and_future_mask']['quantiles'] == 0.0, r['delivery_day_and_future_mask']['params_unchanged'],
                   r['origin_statistics_unchanged_by_delivery_day_prices'],
                   r['available_d1_nonuniform_price_mutation']['quantiles'] > THRESHOLD, r['origin_statistics_move_with_d1_prices'],
                   r['weather']['future_weather_mutation'] == 0.0, r['weather']['training_weather_cross_date_permutation'] > THRESHOLD,
                   r['weather']['target_day_weather_rearranged'] > THRESHOLD,
                   r['held_out_week_outcome_mutation']['stopping_history_changed'], r['recency']['last_seven_days_trained_on'],
                   r['recency']['no_held_out_day_in_last_seven'], r['recency']['mutating_d_minus_5_moves_the_fit'] > THRESHOLD,
                   all(r['preprocessor'].values()), r['extra_input_column_refused']]
    checks += [all(v for x, v in result['ensemble_determinism'].items() if isinstance(v, bool))]
    sg = result['search_and_gate']
    checks += [sg['search_outcomes_on_or_after_d0_minus_56_change_nothing']['identical'],
               sg['validation_batch_outcome_changes_the_search']['changed'],
               sg['earlier_fold_evaluation_day_changes_a_later_fold_fit']['changed'],
               sg['gate_outcomes_on_or_after_d0_change_nothing']['identical'],
               sg['earlier_gate_day_outcome_changes_a_later_gate_fit']['max_abs_difference'] > THRESHOLD,
               all(v for v in sg['weather_coverage'].values() if isinstance(v, bool))]
    checks += [v for v in result['preregistration'].values() if isinstance(v, bool)]
    for v in result['state'].values():
        if isinstance(v, bool):
            checks.append(v)
        elif isinstance(v, dict):
            checks += [x for x in v.values() if isinstance(x, bool)]
    pp = result['population']
    checks += [pp['composite_parity_passed'], pp['v4_members_rebuild_committed_v4_central'], pp['D2_central_is_members_D2'],
               pp['D2_p50_is_its_central'], pp['keys_aligned_with_members'], pp['members_v4_equals_committed_v4'],
               pp['members_v3_equals_committed_v3'], pp['v5_and_v3d2_on_h_path'], pp['d2_entries_checked'] == 636,
               pp['d2_entries_finite_ordered_p50_central_frozen_members'], pp['quantiles_finite'], pp['quantiles_ordered'],
               pp['p50_separate_from_central_for_v5_and_v3d2'], pp['boundary_guard_refuses_2026_04_08'],
               pp['boundary_guard_accepts_2026_04_07'], pp['loader_max_date'] == '2026-04-07',
               pp['max_delivery_date'] <= '2026-04-07', all(v == 10747 for v in pp['rows_per_policy'].values()),
               pp['spring_2026_03_29_canonical_hours'] == 23, pp['autumn_2025_10_26_canonical_hours'] == 25]
    c = result['code']
    checks += [c['import_audit']['passed'], c['gradient_audit_search_tests']['exit_code'] == 0, c['reference']['passed']]
    result['all_passed'] = bool(all(checks))
    result['checks'] = len(checks)
    result['written_utc'] = stamp()
    atomic(out_path, result)
    print(json.dumps({'all_passed': result['all_passed'], 'checks': len(checks)}, default=str), flush=True)
    return 0 if result['all_passed'] else 7
