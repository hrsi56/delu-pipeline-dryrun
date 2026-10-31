"""CP-21 fit-cost and daily-retrain diagnostic (capstone v21-r6 §17.5; Owner decision D3: a
diagnostic only, never a selection criterion).

* ``fit-cost``: every main-run LightGBM fit by arm, model (block or pooled) and origin -- training
  rows, selected capacity, inner and final wall/CPU seconds, worker peak memory -- and the block
  versus pooled cost comparison.
* ``daily-cycle``: HGL's complete daily cycle, measured cold on the M3 with at most four worker
  processes, at 25 evaluation origins (five per fold, fixed offsets 0/22/44/66/88 days into each
  90-day window, the next day with eligible hours if one has none). Each cycle starts a fresh pool:
  data and features, the A1_w/B2_w refits, the six block selections and fits, then HGL's H layer
  from the state a live system would have persisted that morning, and issuance. The refitted
  components must equal CP-20's bit for bit, and the block fits and issued vector the main run's.
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

from cp15.data import array_hash
from cp16.residuals import SharedResidualState
from cp20.components import HGComponents
from .budget import atomic, ledger
from .components import augment, counted_lear
from .execution import OUT, FitCache, _truth, _twice, charge_fits, check_protocol, fit_identity, hg_cache_dir, state_dir
from .inputs import hg_identity, load, origin_manifest, weather_design, weather_matrix
from .jobs import art, stamp
from .lgbm import ARMS, BLOCKS, MODEL_HOURS, arm_design, fit_select, hgl_central

OFFSETS = (0, 22, 44, 66, 88)
WORKERS = 4


def cycle_origins(root: Path) -> list[tuple[str, date]]:
    table = check_protocol(root)['population']['origin_table']
    with_rows = {(o['fold'], o['day']) for o in table if o['n_forecast']}
    out = []
    for f in origin_manifest(root)['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        for offset in OFFSETS:
            d = first + timedelta(days=offset)
            while (f['fold'], str(d)) not in with_rows:
                d += timedelta(days=1)
            out.append((f['fold'], d))
    return out


# ------------------------------------------------------------------ worker tasks (fresh processes)
def _block_task(args):
    kind, name, root, day_s, fold, ledger_path = args
    os.environ['CP21_LEDGER'] = ledger_path
    began = time.perf_counter()
    root, day = Path(root), date.fromisoformat(day_s)
    data = load(root, before=day + timedelta(days=1))  # everything a live system has on the morning of D-1
    design = weather_design(root)
    loaded = time.perf_counter() - began
    budget = ledger()
    arm, model = name.split(':')
    wx, present = weather_matrix(design, data)
    design_arm = arm_design(data, wx, arm)
    start = time.perf_counter()
    pred, records, summary = fit_select(data, design_arm, present, day, arm, model,
                                        charge=charge_fits(budget, purpose='daily_cycle', main=False))
    return {'kind': kind, 'name': name, 'central': pred.tolist(), 'selected': summary['selected'],
            'trees': [r['model_sha256'] for r in records], 'n_window': records[-1]['n_train'],
            'load_seconds': loaded, 'fit_seconds': time.perf_counter() - start, 'wall_seconds': time.perf_counter() - began,
            'fit_wall': sum(r['fit_wall_seconds'] for r in records), 'fit_cpu': sum(r['fit_cpu_seconds'] for r in records),
            'maxrss_bytes': max(r['maxrss_bytes'] for r in records)}


def one_cycle(root: Path, fold: str, day: date, fit_ident, hg_ident, budget) -> dict:
    """One cold daily cycle: a fresh pool, each worker loading its own data; each component
    worker refits exactly one of A1_w/B2_w, each block worker selects and fits one block model."""
    ledger_path = os.environ['CP21_LEDGER']
    tasks = [('component', p, str(root), str(day), fold, ledger_path) for p in ('A1', 'B2')]
    tasks += [('block', f'{arm}:{block}', str(root), str(day), fold, ledger_path) for arm in ('L-R', 'L-N') for block in BLOCKS]
    began = time.perf_counter()
    # The parent loads its own copy first, so no more than four processes ever compute at once.
    data = load(root, before=day + timedelta(days=1))
    parent_load = time.perf_counter() - began
    ctx = mp.get_context('spawn')
    with ctx.Pool(WORKERS) as pool:
        results = pool.map(_component_or_block, tasks, chunksize=1)
    fitted = time.perf_counter() - began
    by = {r['name']: r for r in results}
    rows = data.rows(day)
    hours = data.hours[rows]
    central = {}
    for arm in ('L-R', 'L-N'):
        c = np.full(len(rows), np.nan)
        for block in BLOCKS:
            c[np.isin(hours, MODEL_HOURS[block])] = by[f'{arm}:{block}']['central']
        central[arm] = c
    a1, b2 = np.asarray(by['A1']['central']), np.asarray(by['B2']['central'])
    h_start = time.perf_counter()
    c_hgl = _twice(hgl_central(a1, b2, central['L-N'], central['L-R']))
    state = SharedResidualState.load(state_dir() / fold / str(day) / 'HGL.json')
    budget.reserve(policy_days=1, policy_days_daily_cycle=1)
    state.release(day, _truth(data))
    q, meta = state.predict(day, data.index[rows], c_hgl, c_hgl, data.scale[rows])
    state.issue(day, data.index[rows], c_hgl, c_hgl, data.scale[rows])
    h_seconds = time.perf_counter() - h_start
    total = time.perf_counter() - began
    # identities: CP-20 components, main-run fits, committed HGL vector
    hg = HGComponents(hg_cache_dir(), fold, data, hg_ident).get(day)[1]
    cache = FitCache(fold, data, fit_ident)
    committed = pd.read_parquet(Path(root) / OUT / 'predictions.parquet', filters=[('policy', '==', 'HGL')])
    committed = committed.loc[pd.to_datetime(committed.delivery_date).dt.date.eq(day)].sort_values('timestamp_utc')
    main_trees = {arm: {r['model']: [] for r in cache.load(day, arm)['fits']} for arm in ('L-R', 'L-N')}
    for arm in ('L-R', 'L-N'):
        for r in cache.load(day, arm)['fits']:
            main_trees[arm][r['model']].append(r['model_sha256'])
    return {
        'fold': fold, 'day': str(day), 'n_hours': int(len(rows)),
        'seconds': {'total_cold_cycle': total, 'pool_and_fits': fitted, 'h_layer_and_issuance': h_seconds,
                    'parent_data_load': parent_load, 'max_worker_data_load': max(r['load_seconds'] for r in results),
                    'components': {p: by[p]['fit_seconds'] for p in ('A1', 'B2')},
                    'blocks': {n: by[n]['fit_seconds'] for n in by if ':' in n}},
        'selected': {n: by[n]['selected'] for n in by if ':' in n},
        'n_window': {n: by[n]['n_window'] for n in by if ':' in n},
        'peak_worker_maxrss_bytes': max(r.get('maxrss_bytes', 0) for r in results),
        'components_bitwise_vs_cp20': {p: bool(np.array_equal(np.asarray(by[p]['central']), hg[p])) for p in ('A1', 'B2')},
        'blocks_bitwise_vs_main_run': {arm: bool(np.array_equal(central[arm], cache.get(day, arm))) for arm in ('L-R', 'L-N')},
        'block_trees_equal_main_run': {n: by[n]['trees'] == main_trees[n.split(':')[0]][n.split(':')[1]] for n in by if ':' in n},
        'hgl_central_bitwise_vs_committed': bool(np.array_equal(c_hgl, committed.central.to_numpy(float))),
        'hgl_vector_bitwise_vs_committed': bool(np.array_equal(q['V2-H'], committed[['p025', 'p10', 'p25', 'p50', 'p75', 'p90', 'p975']].to_numpy(float))),
        'buffer_days': meta['buffer_days'], 'vector_sha256': array_hash(q['V2-H'])}


def _component_or_block(args):
    kind, name, root, day_s, fold, ledger_path = args
    if kind == 'component':
        os.environ['CP21_LEDGER'] = ledger_path
        began = time.perf_counter()
        root_p, day = Path(root), date.fromisoformat(day_s)
        data = load(root_p, before=day + timedelta(days=1))
        design = weather_design(root_p)
        loaded = time.perf_counter() - began
        aug, present = augment(data, design)
        start = time.perf_counter()
        from cp20.components import training_rows
        rows = data.rows(day)
        train = training_rows(aug, day, name)
        if not present[train].all() or not present[rows].all():
            raise ValueError('a training/forecast row lacks a frozen weather record')
        with counted_lear(ledger(), 'daily_cycle') as fit_day:
            pred, _ = fit_day(aug, day, name, rows=rows)
        return {'kind': kind, 'name': name, 'central': pred.tolist(), 'load_seconds': loaded,
                'fit_seconds': time.perf_counter() - start, 'wall_seconds': time.perf_counter() - began}
    return _block_task(args)


def job_daily_cycle(root: Path, rest) -> int:
    check_protocol(root)
    lineage = json.loads((root / OUT / 'lineage.json').read_text())
    if lineage['execution_stage'] not in ('comparison_vectors_complete_not_scored', 'scored_development_post_selection'):
        raise ValueError('the daily cycle is compared with the committed comparison vectors')
    if int(os.environ.get('CP21_WORKERS', '1')) < WORKERS:
        raise ValueError('declare four workers for the daily cycle')
    fit_ident, hg_ident = fit_identity(root), hg_identity(root)
    budget = ledger()
    records = []
    for fold, day in cycle_origins(root):
        record = one_cycle(root, fold, day, fit_ident, hg_ident, budget)
        records.append(record)
        print(json.dumps({k: record[k] for k in ('fold', 'day', 'components_bitwise_vs_cp20', 'blocks_bitwise_vs_main_run',
                                                 'hgl_vector_bitwise_vs_committed')}), round(record['seconds']['total_cold_cycle'], 1), flush=True)
    totals = [r['seconds']['total_cold_cycle'] for r in records]
    checks = all(all(r['components_bitwise_vs_cp20'].values()) and all(r['blocks_bitwise_vs_main_run'].values())
                 and all(r['block_trees_equal_main_run'].values()) and r['hgl_central_bitwise_vs_committed']
                 and r['hgl_vector_bitwise_vs_committed'] for r in records)
    out = {'schema': 'cp21-daily-cycle-v1', 'written_utc': stamp(), 'machine': 'Apple M3, 16 GB, CPU only',
           'workers': WORKERS, 'lightgbm_threads_per_worker': 1, 'blas_threads': 1, 'origins': len(records),
           'selection_rule': 'five per fold at offsets 0/22/44/66/88 days into the 90-day window (next day with eligible hours)',
           'cycle': 'fresh 4-process pool: data and features in every worker; A1_w and B2_w refits; six block selections and '
                    'fits (L-R, L-N); parent: HGL blend, persisted H-layer state loaded, released, predicted, issued',
           'total_cold_cycle_seconds': {'median': statistics.median(totals), 'max': max(totals), 'min': min(totals)},
           'all_bitwise_checks_passed': bool(checks), 'records': records,
           'role': 'diagnostic only (Owner decision D3); not a selection criterion'}
    atomic(root / OUT / 'daily-cycle.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'records'}), flush=True)
    return 0 if checks else 8


# ------------------------------------------------------------------ fit cost
def job_fit_cost(root: Path, rest) -> int:
    """Main-run fit records, from the verified fit cache, as committed tables."""
    check_protocol(root)
    fit_ident = fit_identity(root)
    table = check_protocol(root)['population']['origin_table']
    m = {f['fold']: f for f in origin_manifest(root)['folds']}
    rows, per_origin = [], []
    full = load(root)
    early = {}
    for o in table:
        if not o['n_forecast']:
            continue
        fold, day = o['fold'], date.fromisoformat(o['day'])
        first = date.fromisoformat(m[fold]['evaluation_start'])
        if day < first:
            if fold not in early:  # one training-only load per fold (setdefault would load every time)
                early[fold] = load(root, before=first)
            data = early[fold]
        else:
            data = full
        cache = FitCache(fold, data, fit_ident)
        for arm in ARMS:
            item = cache.load(day, arm)
            if item is None:
                raise ValueError(f'fit cache miss {fold} {day} {arm}')
            for r in item['fits']:
                rows.append({'fold': fold, 'phase': o['phase'], 'arm': arm, 'model': r['model'], 'delivery_date': r['delivery_date'],
                             'role': r['role'], 'config': r['config'], 'n_estimators': r['n_estimators'], 'num_leaves': r['num_leaves'],
                             'n_train': r['n_train'], 'n_window': r['n_window'], 'n_inner': r['n_inner'],
                             'n_validation': r['n_validation'], 'n_forecast': r['n_forecast'],
                             'validation_mae': r['validation_mae'], 'fit_wall_seconds': r['fit_wall_seconds'],
                             'fit_cpu_seconds': r['fit_cpu_seconds'], 'predict_wall_seconds': r['predict_wall_seconds'],
                             'worker_maxrss_bytes': r['maxrss_bytes'], 'model_sha256': r['model_sha256'],
                             'window_rows_sha256': r['window_rows_sha256'], 'imputer_fill': json.dumps(r['imputer_fill'])})
            for model, summary in item['selection'].items():
                fits = [r for r in item['fits'] if r['model'] == model]
                inner = [r for r in fits if r['role'] == 'inner']
                final = [r for r in fits if r['role'] == 'final'][0]
                per_origin.append({'fold': fold, 'phase': o['phase'], 'delivery_date': str(day), 'arm': arm, 'model': model,
                                   'n_window': final['n_train'], 'n_inner': inner[0]['n_train'], 'selected': summary['selected'],
                                   'tie': summary['tie'], 'inner_wall_seconds': sum(r['fit_wall_seconds'] for r in inner),
                                   'inner_cpu_seconds': sum(r['fit_cpu_seconds'] for r in inner),
                                   'final_wall_seconds': final['fit_wall_seconds'], 'final_cpu_seconds': final['fit_cpu_seconds'],
                                   'worker_maxrss_bytes': max(r['maxrss_bytes'] for r in fits)})
    fits = pd.DataFrame(rows)
    fits.to_parquet(root / OUT / 'fits.parquet', index=False)
    origin = pd.DataFrame(per_origin)
    origin.to_csv(root / OUT / 'fit-cost-by-origin.csv', index=False)
    summary = origin.groupby(['arm', 'model']).agg(
        origins=('delivery_date', 'nunique'), mean_window_rows=('n_window', 'mean'), min_window_rows=('n_window', 'min'),
        max_window_rows=('n_window', 'max'), inner_wall_total=('inner_wall_seconds', 'sum'), inner_cpu_total=('inner_cpu_seconds', 'sum'),
        final_wall_total=('final_wall_seconds', 'sum'), final_cpu_total=('final_cpu_seconds', 'sum'),
        median_model_wall=('final_wall_seconds', 'median'), peak_worker_maxrss_bytes=('worker_maxrss_bytes', 'max'),
        ties=('tie', 'sum')).reset_index()
    capacity = origin.pivot_table(index=['arm', 'model'], columns='selected', values='delivery_date', aggfunc='count', fill_value=0)
    capacity.columns = [f'selected_{c}' for c in capacity.columns]
    summary = summary.merge(capacity.reset_index(), on=['arm', 'model'])
    summary.to_csv(root / OUT / 'fit-cost.csv', index=False)
    arm_totals = origin.assign(total=origin.inner_wall_seconds + origin.final_wall_seconds,
                               cpu=origin.inner_cpu_seconds + origin.final_cpu_seconds).groupby('arm')[['total', 'cpu']].sum()
    per_arm_origin = origin.assign(total=origin.inner_wall_seconds + origin.final_wall_seconds).groupby(['arm', 'delivery_date', 'fold']).total.sum().groupby('arm').describe()
    out = {'schema': 'cp21-fit-cost-v1', 'written_utc': stamp(), 'fits': int(len(fits)),
           'fits_by_role': fits.role.value_counts().to_dict(), 'origins': int(origin.delivery_date.nunique()),
           'wall_seconds_by_arm': arm_totals.total.to_dict(), 'cpu_seconds_by_arm': arm_totals.cpu.to_dict(),
           'per_origin_wall_seconds_by_arm': {a: {k: float(v) for k, v in per_arm_origin.loc[a].items()} for a in per_arm_origin.index},
           'block_vs_pooled': {'pooled_L-P_wall_total': float(arm_totals.total.get('L-P', np.nan)),
                               'blocks_L-R_wall_total': float(arm_totals.total.get('L-R', np.nan)),
                               'ratio_L-R_to_L-P': float(arm_totals.total.get('L-R', np.nan) / arm_totals.total.get('L-P', np.nan))},
           'peak_worker_maxrss_bytes': int(fits.worker_maxrss_bytes.max()),
           'memory_note': 'worker process high-water RSS after each fit (an upper bound for that fit); the monitor records process-tree and aggregate peaks'}
    atomic(root / OUT / 'fit-cost.json', out)
    print(json.dumps(out, default=str), flush=True)
    return 0
