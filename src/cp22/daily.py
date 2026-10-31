"""CP-22 fit-cost and daily-cycle diagnostic (capstone v21-r9 §20.5; §17.5's D3: a diagnostic only,
never a criterion). Written to `reports/v4-revision/`.

* ``fit-cost``: every main-run PN fit -- rows, configuration, inner and full-window wall/CPU seconds,
  worker peak memory -- and PN's inner and final fits against CP-21 L-P's (committed fits.parquet).
* ``daily-cycle``: the replacement's complete daily cycle, cold on the M3 with four worker processes,
  at 25 evaluation origins (five per fold). With no replacement, R's and M's cycles are measured
  each, descriptively. Each cycle starts a fresh pool: every worker loads the snapshot through delivery
  day D, exactly as the main run materialised it (day D's own prices are present in the frame but never
  enter a fit or a feature: the delivery-day mask control in `controls.json` shows they change no forecast
  by even 0.0); A1_w and B2_w are refitted; the member fits the policy needs (R: PN's four
  full-window fits; M: PN-sel's selection and refit and L-P's) run in parallel; the parent forms the
  central forecast, loads the interval-layer state persisted that morning, releases, predicts and
  issues. Refitted components must equal CP-20's, member fits the main run's and the issued vector
  the committed one, bit for bit.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time

import numpy as np
import pandas as pd

from cp15.data import LABELS
from cp20.components import HGComponents, training_rows
from cp21.components import augment, counted_lear
from cp21.execution import FitCache as CP21FitCache
from cp21.lgbm import GRID, POOLED, fit_select, arm_design, model_rows
from .budget import atomic, ledger
from .execution import (LAYER, OUT, PNCache, _identities, _truth, _twice, charge_fits, check_protocol, cycle_days,
                        state_from_dict)
from .inputs import load, origin_manifest, weather_design, weather_matrix
from .jobs import art, cp20_art, cp21_art, stamp
from .pn import _fit, _model_sha, _z, composite, invert, pn_avg, pn_design, select

WORKERS = 4


def _task(args):
    kind, name, root, day_s, ledger_path = args
    os.environ['CP22_LEDGER'] = ledger_path
    began = time.perf_counter()
    root, day = Path(root), date.fromisoformat(day_s)
    data = load(root, before=day + timedelta(days=1))
    design = weather_design(root)
    loaded = time.perf_counter() - began
    budget = ledger()
    start = time.perf_counter()
    if kind == 'component':
        aug, present = augment(data, design)
        rows = data.rows(day)
        train = training_rows(aug, day, name)
        if not present[train].all() or not present[rows].all():
            raise ValueError('a training/forecast row lacks a frozen weather record')
        with counted_lear(budget, 'daily_cycle') as fit_day:
            pred, _ = fit_day(aug, day, name, rows=rows)
        result = {'central': pred.tolist()}
    elif kind == 'pn_full':  # one of PN's four full-window fits (R)
        wx, present = weather_matrix(design, data)
        des = pn_design(data, wx)
        _, window, _, _, forecast = model_rows(data, day, POOLED, present)
        charge_fits(budget, purpose='daily_cycle', main=False)('final')
        config = next(g for g in GRID if g['id'] == name)
        model, imputer, *_ = _fit(des, window, config, data.p, 1)
        result = {'central': invert(_z(model, imputer, des, forecast), data, forecast).tolist(), 'trees': [_model_sha(model)]}
    elif kind == 'pn_sel':  # PN-sel's selection and refit (M)
        wx, present = weather_matrix(design, data)
        des = pn_design(data, wx)
        _, window, inner, validation, forecast = model_rows(data, day, POOLED, present)
        charge = charge_fits(budget, purpose='daily_cycle', main=False)
        losses, trees = {}, []
        for config in GRID:
            charge('inner')
            model, imputer, *_ = _fit(des, inner, config, data.p, 1)
            losses[config['id']] = float(np.mean(np.abs(invert(_z(model, imputer, des, validation), data, validation) - data.y[validation])))
            trees.append(_model_sha(model))
        k = select(losses)
        charge('final')
        model, imputer, *_ = _fit(des, window, GRID[k], data.p, 1)
        trees.append(_model_sha(model))
        result = {'central': invert(_z(model, imputer, des, forecast), data, forecast).tolist(), 'selected': GRID[k]['id'], 'trees': trees}
    elif kind == 'lp':  # L-P's selection and refit (M), CP-21's own code path
        wx, present = weather_matrix(design, data)
        pred, records, summary = fit_select(data, arm_design(data, wx, 'L-P'), present, day, 'L-P', POOLED,
                                            charge=charge_fits(budget, purpose='daily_cycle', main=False))
        result = {'central': pred.tolist(), 'selected': summary['selected'], 'trees': [r['model_sha256'] for r in records]}
    else:
        raise ValueError(kind)
    return {'kind': kind, 'name': name, **result, 'load_seconds': loaded, 'fit_seconds': time.perf_counter() - start,
            'wall_seconds': time.perf_counter() - began}


def one_cycle(root: Path, fold: str, day: date, policy: str, member_policy: str, idents, budget) -> dict:
    fit_ident, cp21_ident, hg_ident, hashes = idents
    ledger_path = os.environ['CP22_LEDGER']
    tasks = [('component', p, str(root), str(day), ledger_path) for p in ('A1', 'B2')]
    if member_policy == 'R':
        tasks += [('pn_full', g['id'], str(root), str(day), ledger_path) for g in GRID]
    else:
        tasks += [('pn_sel', 'PN-sel', str(root), str(day), ledger_path), ('lp', 'L-P', str(root), str(day), ledger_path)]
    began = time.perf_counter()
    data = load(root, before=day + timedelta(days=1))
    parent_load = time.perf_counter() - began
    with mp.get_context('spawn').Pool(WORKERS) as pool:
        results = pool.map(_task, tasks, chunksize=1)
    fitted = time.perf_counter() - began
    by = {r['name']: r for r in results}
    a1, b2 = np.asarray(by['A1']['central']), np.asarray(by['B2']['central'])
    if member_policy == 'R':
        member = pn_avg([np.asarray(by[g['id']]['central']) for g in GRID])
        central = composite(a1, b2, ((member, 3),))
    else:
        central = composite(a1, b2, ((np.asarray(by['PN-sel']['central']), 6), (np.asarray(by['L-P']['central']), 6)))
    h_start = time.perf_counter()
    c = _twice(central)
    state = state_from_dict(policy, json.loads((art() / 'states' / fold / str(day) / f'{policy}.json').read_text()))
    rows = data.rows(day)
    budget.reserve(policy_days=1, policy_days_daily_cycle=1)
    state.release(day, _truth(data))
    q, meta = state.predict(day, data.index[rows], c, c, data.scale[rows])
    vector = q['V2-H'] if LAYER[policy] == 'H' else q['DL']
    state.issue(day, data.index[rows], c, c, data.scale[rows])
    h_seconds = time.perf_counter() - h_start
    total = time.perf_counter() - began
    hg = HGComponents(cp20_art() / 'hg-components', fold, data, hg_ident).get(day)[1]
    pn = PNCache(fold, data, fit_ident).load(day)
    name = 'predictions.parquet' if policy not in ('W+ACI', 'W+DL', 'W+DLF') else 'predictions-w.parquet'
    committed = pd.read_parquet(Path(root) / OUT / name, filters=[('policy', '==', policy)])
    committed = committed.loc[pd.to_datetime(committed.delivery_date).dt.date.eq(day)].sort_values('timestamp_utc')
    checks = {'components_bitwise_vs_cp20': all(np.array_equal(np.asarray(by[p]['central']), hg[p]) for p in ('A1', 'B2')),
              'central_bitwise_vs_committed': bool(np.array_equal(c, committed.central.to_numpy(float))),
              'vector_bitwise_vs_committed': bool(np.array_equal(vector, committed[LABELS].to_numpy(float)))}
    if member_policy == 'R':
        checks['pn_full_window_fits_bitwise_vs_main_run'] = all(
            np.array_equal(np.asarray(by[g['id']]['central']), np.asarray(pn['full_eur'][i])) for i, g in enumerate(GRID))
    else:
        lp = CP21FitCache(fold, data, cp21_ident, cp21_art() / 'fits').get(day, 'L-P')
        checks['pn_sel_bitwise_vs_main_run'] = bool(np.array_equal(np.asarray(by['PN-sel']['central']), np.asarray(pn['pn_sel'])))
        checks['lp_bitwise_vs_cp21'] = bool(np.array_equal(np.asarray(by['L-P']['central']), lp))
    return {'policy': policy, 'fold': fold, 'day': str(day), 'n_hours': int(len(rows)), 'checks': checks,
            'seconds': {'total_cold_cycle': total, 'pool_and_fits': fitted, 'layer_and_issuance': h_seconds,
                        'parent_data_load': parent_load, 'max_worker_data_load': max(r['load_seconds'] for r in results),
                        'tasks': {n: r['fit_seconds'] for n, r in by.items()}},
            'buffer_days': meta['buffer_days']}


def job_daily_cycle(root: Path, rest) -> int:
    check_protocol(root)
    if int(os.environ.get('CP22_WORKERS', '1')) < WORKERS:
        raise ValueError('declare four workers for the daily cycle')
    decisions = json.loads((root / OUT / 'decisions.json').read_text())
    w = decisions['replacement']['winner']
    if w:
        layer = 'W+DLF' if decisions['fast_component'].get('adopted') else ('W+DL' if decisions['dynamic_layer'].get('adopted') else w)
        plan = [(layer, w)]
    else:
        plan = [('R', 'R'), ('M', 'M')]
    idents = _identities(root)
    budget = ledger()
    records = []
    for policy, member_policy in plan:
        for fold, day_s in sorted(cycle_days(root)):
            record = one_cycle(root, fold, date.fromisoformat(day_s), policy, member_policy, idents, budget)
            records.append(record)
            print(json.dumps({k: record[k] for k in ('policy', 'fold', 'day', 'checks')}), round(record['seconds']['total_cold_cycle'], 1), flush=True)
    summary = {}
    for policy, _ in plan:
        totals = [r['seconds']['total_cold_cycle'] for r in records if r['policy'] == policy]
        summary[policy] = {'origins': len(totals), 'median_seconds': statistics.median(totals), 'max_seconds': max(totals),
                           'min_seconds': min(totals)}
    ok = all(all(r['checks'].values()) for r in records)
    out = {'schema': 'cp22-daily-cycle-v1', 'written_utc': stamp(), 'machine': 'Apple M3, 16 GB, CPU only', 'workers': WORKERS,
           'lightgbm_threads_per_worker': 1, 'blas_threads': 1, 'plan': [p for p, _ in plan],
           'which': 'the replacement W with its adopted layer' if w else 'no replacement: R and M each, descriptively',
           'origins': 'five per fold at offsets 0/22/44/66/88 days into the 90-day window (next day with eligible hours)',
           'summary': summary, 'all_bitwise_checks_passed': bool(ok), 'records': records,
           'role': 'diagnostic only (section 17.5 D3); not a selection criterion'}
    atomic(root / OUT / 'daily-cycle.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'records'}), flush=True)
    return 0 if ok else 8


def job_fit_cost(root: Path, rest) -> int:
    """PN's main-run fit records from the verified cache, against CP-21 L-P's committed records."""
    check_protocol(root)
    fit_ident = _identities(root)[0]
    table = check_protocol(root)['population']['origin_table']
    m = {f['fold']: f for f in origin_manifest(root)['folds']}
    full, early = load(root), {}
    rows, per_origin = [], []
    for o in table:
        if not o['n_forecast']:
            continue
        fold, day = o['fold'], date.fromisoformat(o['day'])
        first = date.fromisoformat(m[fold]['evaluation_start'])
        if day < first:
            if fold not in early:
                early[fold] = load(root, before=first)
            data = early[fold]
        else:
            data = full
        item = PNCache(fold, data, fit_ident).load(day)
        for r in item['fits']:
            rows.append({'fold': fold, 'phase': o['phase'], 'arm': 'PN', 'model': POOLED, 'delivery_date': r['delivery_date'],
                         'role': r['role'], 'config': r['config'], 'n_estimators': r['n_estimators'], 'num_leaves': r['num_leaves'],
                         'n_train': r['n_train'], 'n_window': r['n_window'], 'n_inner': r['n_inner'], 'n_validation': r['n_validation'],
                         'n_forecast': r['n_forecast'], 'validation_mae': r['validation_mae'], 'selected': r.get('selected'),
                         'fit_wall_seconds': r['fit_wall_seconds'], 'fit_cpu_seconds': r['fit_cpu_seconds'],
                         'predict_wall_seconds': r['predict_wall_seconds'], 'worker_maxrss_bytes': r['maxrss_bytes'],
                         'model_sha256': r['model_sha256'], 'window_rows_sha256': r['window_rows_sha256'],
                         'imputer_fill': json.dumps(r['imputer_fill'])})
        inner = [r for r in item['fits'] if r['role'] == 'inner']
        final = [r for r in item['fits'] if r['role'] == 'final']
        per_origin.append({'fold': fold, 'phase': o['phase'], 'delivery_date': str(day), 'n_window': final[0]['n_train'],
                           'n_inner': inner[0]['n_train'], 'selected': item['selected'], 'tie': item['tie'],
                           'winner_margin_relative': item['winner_margin_relative'],
                           'inner_wall_seconds': sum(r['fit_wall_seconds'] for r in inner),
                           'inner_cpu_seconds': sum(r['fit_cpu_seconds'] for r in inner),
                           'full_window_wall_seconds': sum(r['fit_wall_seconds'] for r in final),
                           'full_window_cpu_seconds': sum(r['fit_cpu_seconds'] for r in final),
                           'selected_full_window_wall_seconds': final[item['selected_index']]['fit_wall_seconds'],
                           'worker_maxrss_bytes': max(r['maxrss_bytes'] for r in item['fits'])})
    fits = pd.DataFrame(rows)
    fits.to_parquet(root / OUT / 'fits.parquet', index=False)
    origin = pd.DataFrame(per_origin)
    origin.to_csv(root / OUT / 'fit-cost-by-origin.csv', index=False)
    lp = pd.read_csv(root / 'reports/block-challenger/fit-cost-by-origin.csv')
    lp = lp.loc[lp.arm.eq('L-P')]
    out = {'schema': 'cp22-fit-cost-v1', 'written_utc': stamp(), 'fits': int(len(fits)), 'fits_by_role': fits.role.value_counts().to_dict(),
           'origins': int(len(origin)),
           'PN': {'inner_wall_total': float(origin.inner_wall_seconds.sum()), 'inner_cpu_total': float(origin.inner_cpu_seconds.sum()),
                  'full_window_wall_total': float(origin.full_window_wall_seconds.sum()),
                  'full_window_cpu_total': float(origin.full_window_cpu_seconds.sum()),
                  'median_per_origin_wall_all_eight': float((origin.inner_wall_seconds + origin.full_window_wall_seconds).median()),
                  'median_per_origin_wall_r_four_full_window': float(origin.full_window_wall_seconds.median()),
                  'median_per_origin_wall_m_selection_and_refit': float((origin.inner_wall_seconds + origin.selected_full_window_wall_seconds).median()),
                  'peak_worker_maxrss_bytes': int(fits.worker_maxrss_bytes.max())},
           'L-P (CP-21, committed)': {'inner_wall_total': float(lp.inner_wall_seconds.sum()), 'inner_cpu_total': float(lp.inner_cpu_seconds.sum()),
                                      'final_wall_total': float(lp.final_wall_seconds.sum()), 'final_cpu_total': float(lp.final_cpu_seconds.sum()),
                                      'median_per_origin_wall_selection_and_refit': float((lp.inner_wall_seconds + lp.final_wall_seconds).median()),
                                      'origins': int(len(lp))},
           'note': 'PN makes four full-window fits where L-P makes one refit; R needs only the four full-window fits, M needs PN-sel\'s '
                   'selection and refit plus L-P\'s; diagnostic only (D3)'}
    atomic(root / OUT / 'fit-cost.json', out)
    print(json.dumps(out, default=str), flush=True)
    return 0
