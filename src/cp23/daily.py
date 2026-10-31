"""CP-23 fit-cost and daily-cycle diagnostic (capstone v21-r10 §21.5; §17.5's D3: a diagnostic only, never
a criterion). Written to `reports/distribution-challenger/`.

* ``fit-cost``: every main-run DDNN member fit (the five selections, warm-up and evaluation), with its
  rows, configuration, seed, epochs and wall and CPU seconds, per origin and per fold.
* ``daily-cycle``: v5's daily cycle, cold on the M3 with four worker processes, at 25 evaluation origins
  (five per fold).
  - Each cycle starts a fresh pool. Every worker loads the snapshot through delivery day D, exactly as
    the main run materialised it, and fits one of the four seeds in the fold's configuration.
  - The parent averages the members' quantiles into D, forms v5's central from v4's verified cached
    members, loads v5's H-layer state persisted that morning, releases, predicts and issues.
  - Every member must equal the main run's bit for bit, and the issued v5 vector the committed one.
  - No LightGBM or LEAR component is refitted: §21.8 caps only DDNN fits. v4's own cold cycle is
    CP-21's committed measurement, shown beside it.
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
from cp16.residuals import SharedResidualState
from . import ddnn as D
from .budget import atomic, charge_fits, ledger
from .execution import (OUT, DDNNCache, Sources, _identities, _truth, _twice, check_protocol, cycle_days, selected_config,
                        v5_central)
from .features import encode, target
from .inputs import load, origin_manifest, weather_design, weather_matrix
from .jobs import art, stamp
from .member import fit_origin

WORKERS = 4


def _member_task(args):
    root, fold, day_s, config_id, seed, ledger_path = args
    os.environ['CP23_LEDGER'] = ledger_path
    began = time.perf_counter()
    root, day = Path(root), date.fromisoformat(day_s)
    data = load(root, before=day + timedelta(days=1))
    wx, present = weather_matrix(weather_design(root), data)
    x, _ = encode(data)
    z = target(data)
    loaded = time.perf_counter() - began
    start = time.perf_counter()
    fit = fit_origin(data, x, wx, z, present, day, config_id, seeds=(seed,),
                     charge=charge_fits(ledger(), 'daily_cycle', main=False))
    return {'seed': seed, 'member_z_quantiles': fit['member_z_quantiles'][0].tolist(),
            'params_sha256': fit['members'][0]['params_sha256'], 'best_epoch': fit['members'][0]['best_epoch'],
            'epochs_run': fit['members'][0]['epochs_run'], 'load_seconds': loaded, 'fit_seconds': time.perf_counter() - start,
            'wall_seconds': time.perf_counter() - began}


def one_cycle(root: Path, fold: str, day: date, idents, budget) -> dict:
    fit_ident, cp21_ident, hg_ident, hashes = idents
    config_id = selected_config(root, fold)
    ledger_path = os.environ['CP23_LEDGER']
    tasks = [(str(root), fold, str(day), config_id, seed, ledger_path) for seed in D.SEEDS]
    began = time.perf_counter()
    data = load(root, before=day + timedelta(days=1))
    parent_load = time.perf_counter() - began
    with mp.get_context('spawn').Pool(WORKERS) as pool:
        results = pool.map(_member_task, tasks, chunksize=1)
    fitted = time.perf_counter() - began
    results = sorted(results, key=lambda r: D.SEEDS.index(r['seed']))
    rows = data.rows(day)
    zq = np.stack([np.asarray(r['member_z_quantiles'], float) for r in results]).mean(axis=0)
    quantiles, d_central, crossed = D.emit(zq, data.level[rows], data.scale[rows])
    h_start = time.perf_counter()
    m = Sources(fold, data, None, cp21_ident, hg_ident, hashes, ('HGL',)).members(day)
    c = _twice(v5_central(m['HGL'], d_central))
    state = SharedResidualState.from_dict(json.loads((art() / 'states' / fold / str(day) / 'v5.json').read_text()))
    budget.reserve(policy_days=1, policy_days_daily_cycle=1)
    state.release(day, _truth(data))
    q, meta = state.predict(day, data.index[rows], c, c, data.scale[rows])
    vector = q['V2-H']
    state.issue(day, data.index[rows], c, c, data.scale[rows])
    h_seconds = time.perf_counter() - h_start
    total = time.perf_counter() - began
    cached = DDNNCache(fold, data, fit_ident).load(day)
    committed = pd.read_parquet(Path(root) / OUT / 'predictions.parquet', filters=[('policy', '==', 'v5')])
    committed = committed.loc[pd.to_datetime(committed.delivery_date).dt.date.eq(day)].sort_values('timestamp_utc')
    checks = {'members_bitwise_vs_main_run': [r['params_sha256'] for r in results] == [r['params_sha256'] for r in cached['members']],
              'D_bitwise_vs_main_run': bool(np.array_equal(d_central, np.asarray(cached['central'], float))
                                            and np.array_equal(quantiles, np.asarray(cached['quantiles'], float))),
              'v5_central_bitwise_vs_committed': bool(np.array_equal(c, committed.central.to_numpy(float))),
              'v5_vector_bitwise_vs_committed': bool(np.array_equal(vector, committed[LABELS].to_numpy(float)))}
    return {'policy': 'v5', 'fold': fold, 'day': str(day), 'config': config_id, 'n_hours': int(len(rows)), 'checks': checks,
            'crossed_rows': crossed,
            'seconds': {'total_cold_cycle': total, 'pool_and_member_fits': fitted, 'layer_and_issuance': h_seconds,
                        'parent_data_load': parent_load, 'max_worker_data_load': max(r['load_seconds'] for r in results),
                        'member_fits': {str(r['seed']): r['fit_seconds'] for r in results}},
            'epochs_run': {str(r['seed']): r['epochs_run'] for r in results}, 'buffer_days': meta['buffer_days']}


def v4_cycle_beside(root: Path) -> dict:
    """CP-21's committed cold-cycle measurement of v4's own components (its schema: `total_cold_cycle_seconds`)."""
    v4_cycle = json.loads((root / 'reports/block-challenger/daily-cycle.json').read_text())
    return {'source': 'reports/block-challenger/daily-cycle.json (CP-21, committed)',
            'summary': v4_cycle.get('summary') or v4_cycle.get('total_cold_cycle_seconds'),
            'origins': v4_cycle.get('origins'), 'cycle': v4_cycle.get('cycle'),
            'note': 'v4\'s A1_w/B2_w refits and six block selections and fits; not refitted in CP-23 (§21.8 caps only DDNN '
                    'fits). v5\'s full daily cycle needs both.'}


def job_daily_cycle(root: Path, rest) -> int:
    check_protocol(root)
    if '--beside-only' in rest:
        # Repair (recorded): the first run read CP-21's cycle under the wrong key and wrote null; no fit is repeated.
        path = root / OUT / 'daily-cycle.json'
        out = json.loads(path.read_text())
        out['v4_component_cycle_beside'] = v4_cycle_beside(root)
        out['repairs'] = out.get('repairs', []) + [{'utc': stamp(), 'field': 'v4_component_cycle_beside',
                                                    'what': 'CP-21 summary read from total_cold_cycle_seconds; no fit repeated'}]
        ledger().event('daily_cycle_beside_repair', field='v4_component_cycle_beside')
        atomic(path, out)
        print(json.dumps(out['v4_component_cycle_beside']), flush=True)
        return 0
    if int(os.environ.get('CP23_WORKERS', '1')) < WORKERS:
        raise ValueError('declare four workers for the daily cycle')
    idents = _identities(root)
    budget = ledger()
    records = []
    for fold, day_s in sorted(cycle_days(root)):
        record = one_cycle(root, fold, date.fromisoformat(day_s), idents, budget)
        records.append(record)
        print(json.dumps({k: record[k] for k in ('fold', 'day', 'config', 'checks')}),
              round(record['seconds']['total_cold_cycle'], 1), flush=True)
    totals = [r['seconds']['total_cold_cycle'] for r in records]
    by_fold = {f: statistics.median([r['seconds']['total_cold_cycle'] for r in records if r['fold'] == f])
               for f in sorted({r['fold'] for r in records})}
    ok = all(all(r['checks'].values()) for r in records)
    out = {'schema': 'cp23-daily-cycle-v1', 'written_utc': stamp(), 'machine': 'Apple M3, 16 GB, CPU only', 'workers': WORKERS,
           'blas_threads': 1, 'policy': 'v5',
           'origins': 'five per fold at offsets 0/22/44/66/88 days into the 90-day window (next day with eligible hours)',
           'summary': {'origins': len(totals), 'median_seconds': statistics.median(totals), 'max_seconds': max(totals),
                       'min_seconds': min(totals), 'median_by_fold': by_fold},
           'what_the_cycle_covers': 'data and features through D; the four DDNN members in the fold\'s configuration (one per '
                                    'worker); the ensemble; v5\'s central from v4\'s verified cached members; the H layer from '
                                    'the state persisted that morning; issuance',
           'v4_component_cycle_beside': v4_cycle_beside(root),
           'all_bitwise_checks_passed': bool(ok), 'records': records,
           'role': 'diagnostic only (section 17.5 D3); not a selection criterion'}
    atomic(root / OUT / 'daily-cycle.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'records'}, default=str), flush=True)
    return 0 if ok else 8


def job_fit_cost(root: Path, rest) -> int:
    """Every main-run DDNN member fit from the verified cache and the committed selection records."""
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
        item = DDNNCache(fold, data, fit_ident).load(day)
        for r in item['members']:
            rows.append({'fold': fold, 'phase': o['phase'], 'stage': 'origin', 'delivery_date': str(day), 'config': item['config'],
                         'seed': r['seed'], 'n_train': item['rows']['n_train'], 'n_stop': item['rows']['n_stop'],
                         'n_forecast': item['rows']['n_forecast'], 'best_epoch': r['best_epoch'], 'epochs_run': r['epochs_run'],
                         'fit_wall_seconds': r['fit_wall_seconds'], 'fit_cpu_seconds': r['fit_cpu_seconds'],
                         'params_sha256': r['params_sha256']})
        per_origin.append({'fold': fold, 'phase': o['phase'], 'delivery_date': str(day), 'config': item['config'],
                           'n_train': item['rows']['n_train'], 'ensemble_fit_wall_seconds': sum(r['fit_wall_seconds'] for r in item['members']),
                           'ensemble_fit_cpu_seconds': sum(r['fit_cpu_seconds'] for r in item['members']),
                           'origin_seconds': item['seconds'], 'predict_seconds': item['predict_seconds'],
                           'worker_maxrss_bytes': item['maxrss_bytes'],
                           'mean_epochs_run': float(np.mean([r['epochs_run'] for r in item['members']]))})
    for fold in m:
        sel = json.loads((art() / 'selection' / f'{fold}.json').read_text())
        for cid, rec in sel['configurations'].items():
            for r in rec['members']:
                rows.append({'fold': fold, 'phase': 'selection', 'stage': 'selection', 'delivery_date': sel['first_origin'],
                             'config': cid, 'seed': r['seed'], 'n_train': sel['rows']['n_train'], 'n_stop': sel['rows']['n_stop'],
                             'n_forecast': sel['rows']['n_holdout'], 'best_epoch': r['best_epoch'], 'epochs_run': r['epochs_run'],
                             'fit_wall_seconds': r['fit_wall_seconds'], 'fit_cpu_seconds': r['fit_cpu_seconds'],
                             'params_sha256': r['params_sha256']})
    fits = pd.DataFrame(rows)
    fits.to_parquet(root / OUT / 'fits.parquet', index=False)
    origin = pd.DataFrame(per_origin)
    origin.to_csv(root / OUT / 'fit-cost-by-origin.csv', index=False)
    by_config = fits.groupby('config').agg(fits=('seed', 'size'), mean_wall=('fit_wall_seconds', 'mean'),
                                           median_wall=('fit_wall_seconds', 'median'), max_wall=('fit_wall_seconds', 'max'),
                                           mean_epochs=('epochs_run', 'mean'), median_best_epoch=('best_epoch', 'median'))
    out = {'schema': 'cp23-fit-cost-v1', 'written_utc': stamp(), 'fits': int(len(fits)),
           'fits_by_stage': fits.stage.value_counts().to_dict(), 'fits_by_phase': fits.phase.value_counts().to_dict(),
           'origins': int(len(origin)),
           'totals': {'wall_seconds': float(fits.fit_wall_seconds.sum()), 'cpu_seconds': float(fits.fit_cpu_seconds.sum())},
           'per_origin': {'median_ensemble_fit_wall_seconds': float(origin.ensemble_fit_wall_seconds.median()),
                          'max_ensemble_fit_wall_seconds': float(origin.ensemble_fit_wall_seconds.max()),
                          'median_by_fold': origin.groupby('fold').ensemble_fit_wall_seconds.median().to_dict()},
           'by_config': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in by_config.to_dict('index').items()},
           'peak_worker_maxrss_bytes': int(origin.worker_maxrss_bytes.max()),
           'cp22_pn_beside': json.loads((root / 'reports/v4-revision/fit-cost.json').read_text()).get('PN'),
           'note': 'one DDNN member fit = one network, one seed, trained with early stopping; the ensemble is four fits; '
                   'diagnostic only (D3)'}
    atomic(root / OUT / 'fit-cost.json', out)
    print(json.dumps(out, default=str), flush=True)
    return 0
