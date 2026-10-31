"""Fit cost and the cold daily cycle of scored attempt k (capstone v21-r11 §23.8; §17.5's D3: a diagnostic,
never a criterion). Written to `reports/ddnn2/attempt-<k>/`.

* ``fit-cost``: every DDNN-2 member fit of the attempt (warm-up and evaluation), and of the round that
  froze it (the search and the gate), with epochs and wall seconds, per origin, per fold and per rank.
* ``daily-cycle``: v5's daily cycle, cold on the M3 with four worker processes, at 25 evaluation origins
  (five per fold). Each cycle starts a fresh pool; every worker loads the snapshot through delivery day D,
  builds the day table and fits two of the fold's eight frozen members; the parent forms D2 (the per-level
  median), v5's central from v4's verified cached members, loads v5's H-layer state persisted that
  morning, releases, predicts and issues. Every member must equal the main run's bit for bit, and the
  issued v5 vector the committed one. No LightGBM or LEAR component is refitted (CP-21's committed v4
  cycle is shown beside it).
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
from . import ddnn2 as M
from . import design as G
from .budget import atomic, charge_fits, ledger
from .execution import D2Cache, Sources, _identities, _truth, _twice, cycle_days, fit_identity, v5_central
from .inputs import load, origin_manifest, weather_design
from .jobs import art, stamp
from .member import Keys, emit_rows, fit_member
from .protocol import attempt_dir, check_protocol

WORKERS = 4


def _members_task(args):
    root, fold, day_s, members, ledger_path = args
    os.environ['CP24_LEDGER'] = ledger_path
    began = time.perf_counter()
    root, day = Path(root), date.fromisoformat(day_s)
    data = load(root, before=day + timedelta(days=1))
    dd = G.build(data, weather_design(root))
    keys = Keys.from_data(data, dd)
    loaded = time.perf_counter() - began
    di = np.array([dd.ix(day)])
    rows = keys.of_days(di)
    out = []
    for j, m in members:
        start = time.perf_counter()
        fit = fit_member(dd, day, m['config'], m['seed'], di, exclude_uncovered=False, charge=charge_fits(ledger(), 'daily_cycle'))
        out.append({'member': j, 'q': emit_rows(keys, rows, di, fit['eur']).tolist(), 'params_sha256': fit['member']['params_sha256'],
                    'epochs_run': fit['member']['epochs_run'], 'fit_seconds': time.perf_counter() - start})
    return {'load_seconds': loaded, 'members': out}


def one_cycle(root: Path, k: int, fold: str, day: date, idents, budget, ensemble) -> dict:
    fit_ident, cp21_ident, hg_ident, hashes = idents
    ledger_path = os.environ['CP24_LEDGER']
    pairs = list(enumerate(ensemble))
    tasks = [(str(root), fold, str(day), pairs[w::WORKERS], ledger_path) for w in range(WORKERS)]
    began = time.perf_counter()
    data = load(root, before=day + timedelta(days=1))
    parent_load = time.perf_counter() - began
    with mp.get_context('spawn').Pool(WORKERS) as pool:
        results = pool.map(_members_task, tasks, chunksize=1)
    fitted = time.perf_counter() - began
    mem = sorted((m for r in results for m in r['members']), key=lambda m: m['member'])
    stack = np.stack([np.asarray(m['q'], float) for m in mem])
    med, crossed = M.ensemble_median(stack[:, :, None, :])
    q = med[:, 0, :]
    d2 = q[:, M.MEDIAN].copy()
    h_start = time.perf_counter()
    rows = data.rows(day)
    mm = Sources(k, fold, data, None, cp21_ident, hg_ident, hashes, ('HGL',)).members(day)
    c = _twice(v5_central(mm['A1'], mm['B2'], mm['L-N'], mm['L-R'], d2))
    state = SharedResidualState.from_dict(json.loads((art() / f'attempt-{k}' / 'states' / fold / str(day) / 'v5.json').read_text()))
    budget.reserve(policy_days=1, policy_days_daily_cycle=1)
    state.release(day, _truth(data))
    qv, meta = state.predict(day, data.index[rows], c, c, data.scale[rows])
    vector = qv['V2-H']
    state.issue(day, data.index[rows], c, c, data.scale[rows])
    h_seconds = time.perf_counter() - h_start
    total = time.perf_counter() - began
    cached = D2Cache(k, fold, data, fit_ident).load(day)
    committed = pd.read_parquet(attempt_dir(root, k) / 'predictions.parquet', filters=[('policy', '==', 'v5')])
    committed = committed.loc[pd.to_datetime(committed.delivery_date).dt.date.eq(day)].sort_values('timestamp_utc')
    checks = {'members_bitwise_vs_main_run': [m['params_sha256'] for m in mem] == [m['record']['params_sha256'] for m in cached['members']],
              'D2_bitwise_vs_main_run': bool(np.array_equal(d2, np.asarray(cached['central'], float))
                                             and np.array_equal(q, np.asarray(cached['quantiles'], float))),
              'v5_central_bitwise_vs_committed': bool(np.array_equal(c, committed.central.to_numpy(float))),
              'v5_vector_bitwise_vs_committed': bool(np.array_equal(vector, committed[LABELS].to_numpy(float)))}
    return {'policy': 'v5', 'fold': fold, 'day': str(day), 'n_hours': int(len(rows)), 'checks': checks, 'crossed_rows': crossed,
            'seconds': {'total_cold_cycle': total, 'pool_and_member_fits': fitted, 'layer_and_issuance': h_seconds,
                        'parent_data_load': parent_load, 'max_worker_data_load': max(r['load_seconds'] for r in results),
                        'member_fits': {str(m['member']): m['fit_seconds'] for m in mem}},
            'epochs_run': {str(m['member']): m['epochs_run'] for m in mem}, 'buffer_days': meta['buffer_days']}


def job_daily_cycle(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True, choices=(1, 2))
    args = ap.parse_args(rest)
    root, k = Path(root), args.attempt
    p = check_protocol(root, k)
    if int(os.environ.get('CP24_WORKERS', '1')) < WORKERS:
        raise ValueError('declare four workers for the daily cycle')
    idents = _identities(root, k)
    budget = ledger()
    records = []
    for fold, day_s in sorted(cycle_days(root)):
        rec = one_cycle(root, k, fold, date.fromisoformat(day_s), idents, budget, p['folds'][fold]['ensemble'])
        records.append(rec)
        print(json.dumps({x: rec[x] for x in ('fold', 'day', 'checks')}), round(rec['seconds']['total_cold_cycle'], 1), flush=True)
    totals = [r['seconds']['total_cold_cycle'] for r in records]
    by_fold = {f: statistics.median([r['seconds']['total_cold_cycle'] for r in records if r['fold'] == f])
               for f in sorted({r['fold'] for r in records})}
    ok = all(all(r['checks'].values()) for r in records)
    v4_cycle = json.loads((root / 'reports/block-challenger/daily-cycle.json').read_text())
    out = {'schema': 'cp24-daily-cycle-v1', 'attempt': k, 'written_utc': stamp(), 'machine': 'Apple M3, 16 GB, CPU only',
           'workers': WORKERS, 'blas_threads': 1, 'policy': 'v5',
           'origins': 'five per fold at offsets 0/22/44/66/88 days into the 90-day window (next day with eligible hours)',
           'summary': {'origins': len(totals), 'median_seconds': statistics.median(totals), 'max_seconds': max(totals),
                       'min_seconds': min(totals), 'median_by_fold': by_fold},
           'what_the_cycle_covers': 'data and the day table through D; the eight DDNN-2 members (two per worker); the per-level '
                                    'median; v5\'s central from v4\'s verified cached members; the H layer from the state '
                                    'persisted that morning; issuance',
           'v4_component_cycle_beside': {'source': 'reports/block-challenger/daily-cycle.json (CP-21, committed)',
                                         'summary': v4_cycle.get('summary') or v4_cycle.get('total_cold_cycle_seconds'),
                                         'note': 'v4\'s A1_w/B2_w refits and six block fits; not refitted here'},
           'all_bitwise_checks_passed': bool(ok), 'records': records, 'role': 'diagnostic only (§17.5 D3); not a criterion'}
    atomic(attempt_dir(root, k) / 'daily-cycle.json', out)
    print(json.dumps({x: v for x, v in out.items() if x != 'records'}, default=str), flush=True)
    return 0 if ok else 8


def job_fit_cost(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True, choices=(1, 2))
    args = ap.parse_args(rest)
    root, k = Path(root), args.attempt
    p = check_protocol(root, k)
    fit_ident = fit_identity(root, k)
    m = {f['fold']: f for f in origin_manifest(root)['folds']}
    full, early = load(root), {}
    rows, per_origin = [], []
    for fold, f in m.items():
        first = date.fromisoformat(f['evaluation_start'])
        for o in f['origins']:
            day = date.fromisoformat(o['day'])
            if day < first:
                early.setdefault(fold, load(root, before=first))
                data = early[fold]
            else:
                data = full
            if not len(data.rows(day)):
                continue
            item = D2Cache(k, fold, data, fit_ident).load(day)
            for r in item['members']:
                rows.append({'fold': fold, 'phase': 'evaluation' if day >= first else 'warmup', 'delivery_date': str(day),
                             'member': r['member'], 'rank': r['rank'], 'config': r['config'], 'seed': r['seed'],
                             'n_train_days': r['window']['n_train_days'], 'n_held_out_days': r['window']['n_held_out_days'],
                             'best_epoch': r['record']['best_epoch'], 'epochs_run': r['record']['epochs_run'],
                             'stop_reason': r['record']['stop_reason'], 'fit_wall_seconds': r['record']['fit_wall_seconds'],
                             'fit_cpu_seconds': r['record']['fit_cpu_seconds'], 'params_sha256': r['record']['params_sha256'],
                             'cap_slot_levels': r['guards']['cap_forecast_slot_levels']})
            per_origin.append({'fold': fold, 'delivery_date': str(day), 'origin_seconds': item['seconds'],
                               'ensemble_fit_wall_seconds': sum(r['record']['fit_wall_seconds'] for r in item['members']),
                               'worker_maxrss_bytes': item['maxrss_bytes']})
    fits = pd.DataFrame(rows)
    fits.to_parquet(attempt_dir(root, k) / 'fits.parquet', index=False)
    origin = pd.DataFrame(per_origin)
    origin.to_csv(attempt_dir(root, k) / 'fit-cost-by-origin.csv', index=False)
    rd = root / 'reports/ddnn2/rounds' / f'round-{p["round"]}'
    search = json.loads((rd / 'search-ledger.json').read_text())
    sfits = [b for f in search['folds'].values() for t in f['trials'] for b in t['batches']]
    gate = json.loads((rd / 'gate.json').read_text())
    by_rank = fits.groupby('rank').agg(fits=('seed', 'size'), mean_wall=('fit_wall_seconds', 'mean'),
                                       median_wall=('fit_wall_seconds', 'median'), max_wall=('fit_wall_seconds', 'max'),
                                       median_epochs=('epochs_run', 'median'), median_best_epoch=('best_epoch', 'median'))
    out = {'schema': 'cp24-fit-cost-v1', 'attempt': k, 'written_utc': stamp(), 'attempt_fits': int(len(fits)),
           'origins': int(len(origin)), 'totals': {'wall_seconds': float(fits.fit_wall_seconds.sum()),
                                                   'cpu_seconds': float(fits.fit_cpu_seconds.sum())},
           'per_origin': {'median_ensemble_fit_wall_seconds': float(origin.ensemble_fit_wall_seconds.median()),
                          'max_ensemble_fit_wall_seconds': float(origin.ensemble_fit_wall_seconds.max()),
                          'median_by_fold': origin.groupby('fold').ensemble_fit_wall_seconds.median().to_dict()},
           'by_rank': {str(x): {kk: float(vv) for kk, vv in v.items()} for x, v in by_rank.to_dict('index').items()},
           'stop_reasons': fits.stop_reason.value_counts().to_dict(),
           'peak_worker_maxrss_bytes': int(origin.worker_maxrss_bytes.max()),
           'round': {'round': p['round'], 'search_fits': len(sfits),
                     'search_fit_seconds_total': float(sum(b['seconds'] or 0 for b in sfits)),
                     'gate_ddnn2_fit_seconds_total': gate['costs']['ddnn2_fit_seconds'],
                     'gate_v4_member_seconds_total': gate['costs']['v4_fit_seconds']},
           'cp23_beside': json.loads((root / 'reports/distribution-challenger/fit-cost.json').read_text())['per_origin'],
           'note': 'one DDNN-2 member fit = one network, one seed, trained once with early stopping; the ensemble is eight fits; '
                   'diagnostic only (D3)'}
    atomic(attempt_dir(root, k) / 'fit-cost.json', out)
    print(json.dumps({x: v for x, v in out.items() if x != 'by_rank'}, default=str), flush=True)
    return 0
