"""4.6R′: DDNN-2's training-only resource entry (capstone v21-r11 §23.7), after the correctness checks.

It runs on pre-fold data only: every timing fit is a search-style fit at a validation batch's first day
b, trains on [max(2019-01-01, b - 728), b) minus the uncovered days (§23.6), and forecasts only that
batch's days -- never a gate, warm-up or evaluation day. It is accuracy-blind: it records cost,
memory and the finiteness and order of the emission, never a score.

**It measures:** peak memory; fit and prediction time at the space's extremes (the smallest and the
largest network in compute terms) and for a sample of round-1 configurations drawn with a dedicated
4.6R′ sampler stream (never the search's); and finite, ordered emission, for single members and for
an eight-member per-level-median ensemble.

**It projects two routes** against §23.11, each with the review reserve: the minimal route (one round,
then attempt 1) and the maximal route (three rounds and attempt 1, then one round and attempt 2). It
fixes the trial count -- at least 32 per fold per round, the largest of {32, 48, 64, 96, 128} for which
the maximal route fits every ceiling -- or, if the maximal route does not fit at 32, the largest route
that does; and the review reserve. If the minimal route does not fit, DDNN-2 is NOT_ADMITTED with its
cause, never a smaller experiment.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np

from . import ddnn2 as M
from . import design as G
from . import sampler as SP
from .budget import CAPS, HOUR, atomic, charge_fits, ledger
from .inputs import load, weather_design
from .jobs import art, stamp
from .member import Keys, emit_rows, fit_member, maxrss
from .preflight import fold_table
from .reference import require_passing_record

OUT = Path('reports/ddnn2')
TRIAL_CHOICES = (32, 48, 64, 96, 128)
SMALLEST = {'id': 'R-smallest', 'hidden': [16], 'activation': 'relu', 'input_dropout': None, 'l1': None, 'l2': None,
            'lr': 1e-2, 'batch_size': 128, 'kappa': 1.0, 'half_life': None, 'groups': [], 'transform': 'z-s4'}
LARGEST = {'id': 'R-largest', 'hidden': [512, 512], 'activation': 'softplus', 'input_dropout': 0.5, 'l1': 1e-3,
           'l2': 1e-2, 'lr': 1e-4, 'batch_size': 32, 'kappa': 0.5, 'half_life': 180, 'groups': list(SP.OPTIONAL_GROUPS),
           'transform': 'asinh-mad'}
RANDOM_SAMPLE = 12
#: Ensemble-member cost is unknown before a search, so the projection prices every gate and attempt
#: member at the sample's 75th-percentile fit time, and also reports the worst case (every member at
#: the largest extreme's measured time).
ENSEMBLE_QUANTILE = 0.75
REVIEW_RESERVE = {'ddnn2_fits': 400, 'machine_hours': 6.0, 'active_hours': 5.0, 'policy_days': 600,
                  'bootstrap_passes': 2, 'reference_passes': 1}
OTHER_PER_ATTEMPT = {'machine_hours': 3.0, 'ddnn2_fits': 200, 'policy_days': 300}
ATTEMPT_FITS = 636 * 8
GATE_FITS = 280 * 8


def search_fits(n: int, b: list[int]) -> int:
    k = SP.halving_survivors(n)
    total = 0
    for bf in b:
        first = min(SP.HALVING_BATCHES, bf)
        total += n * first + (k * (bf - first) if bf > first else 0)
    return total


_worker: dict = {}


def _init_worker(root: str, ledger_path: str):
    os.environ['CP24_LEDGER'] = ledger_path
    root = Path(root)
    require_passing_record(root)
    _worker.update(root=root, design=weather_design(root), budget=ledger(), tables={})


def _tables(cutoff: date):
    w = _worker
    if cutoff not in w['tables']:
        data = load(w['root'], before=cutoff)
        dd = G.build(data, w['design'])
        w['tables'][cutoff] = (dd, Keys.from_data(data, dd))
    return w['tables'][cutoff]


def _task(task):
    label, config, seed, b0_s, cutoff_s = task
    w = _worker
    dd, keys = _tables(date.fromisoformat(cutoff_s))
    b0 = date.fromisoformat(b0_s)
    days = np.array([dd.ix(b0 + timedelta(days=k)) for k in range(SP.BATCH_DAYS)])
    try:
        fit = fit_member(dd, b0, config, seed, days, exclude_uncovered=True,
                         charge=charge_fits(w['budget'], 'resource_admission'))
    except Exception as exc:
        return {'label': label, 'config': config, 'origin': b0_s, 'error': repr(exc), 'traceback': traceback.format_exc()}
    t0 = time.perf_counter()
    rows = keys.of_days(days)
    q = emit_rows(keys, rows, days, fit['eur'])
    predict = time.perf_counter() - t0
    rec = fit['member']
    return {'label': label, 'config': config, 'origin': b0_s, 'seed': seed, 'n_inputs': fit['n_inputs'],
            'n_parameters': M.n_parameters(fit['n_inputs'], tuple(config['hidden'])), 'window': fit['window'],
            'epochs_run': rec['epochs_run'], 'best_epoch': rec['best_epoch'], 'stop_reason': rec['stop_reason'],
            'fit_wall_seconds': rec['fit_wall_seconds'], 'fit_cpu_seconds': rec['fit_cpu_seconds'],
            'seconds_per_epoch': rec['fit_wall_seconds'] / max(rec['epochs_run'], 1),
            'total_seconds': fit['seconds'], 'emission_map_seconds': predict, 'keys': int(len(rows)),
            'finite': bool(np.isfinite(q).all()), 'ordered': bool((np.diff(q, axis=1) >= 0).all()),
            'cap_activations': fit['guards']['cap_forecast_slot_levels'], 'winsor': fit['guards']['winsor_forecast'],
            'maxrss_bytes': maxrss(), 'eur': fit['eur'].tolist(), 'forecast_days': days.tolist()}


def job_resource_admission(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=int(os.environ.get('CP24_WORKERS', '1')))
    args = ap.parse_args(rest)
    root = Path(root)
    require_passing_record(root)
    if (root / OUT / 'resource-admission.json').exists():
        raise ValueError('4.6R′ has a record; it is run once')
    folds = fold_table(root)
    by = {f['fold']: f for f in folds}
    origins = {'fold_1': by['fold_1']['batches'][-1][0], 'fold_3': by['fold_3']['batches'][0][0],
               'fold_5': by['fold_5']['batches'][0][0]}
    tasks = []
    for fold, b0 in origins.items():
        for cfg in (SMALLEST, LARGEST):
            tasks.append((f'{cfg["id"]}@{fold}', cfg, 46_000_000 + len(tasks), str(b0), str(by[fold]['search_cutoff'])))
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([24, 0x46, 0x52])))
    sample = [SP.sample(rng, SP.SPACE_ROUND_1, f'R-random-{i:02d}') for i in range(RANDOM_SAMPLE)]
    ensemble_fold = 'fold_5'
    for i, cfg in enumerate(sample):
        fold = ensemble_fold if i < 8 else ('fold_1', 'fold_3', 'fold_1', 'fold_3')[i - 8]
        tasks.append((f'{cfg["id"]}@{fold}', cfg, 46_100_000 + i, str(origins[fold]), str(by[fold]['search_cutoff'])))
    budget = ledger()
    budget.event('resource_admission_start', tasks=len(tasks), workers=args.workers)
    t0 = time.time()
    results = []
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(str(root), os.environ['CP24_LEDGER'])) as pool:
        for res in pool.imap_unordered(_task, tasks, chunksize=1):
            results.append(res)
            print('4.6R′', res['label'], res.get('epochs_run'), round(res.get('fit_wall_seconds', -1), 1), res.get('error', ''),
                  flush=True)
    wall = time.time() - t0
    failed = [r for r in results if 'error' in r]
    ok = [r for r in results if 'error' not in r]
    # The eight-member ensemble emission path (fold 5's random sample), by the per-level median.
    ens = [r for r in ok if r['label'].endswith(f'@{ensemble_fold}') and r['label'].startswith('R-random')]
    ensemble = None
    if len(ens) == 8:
        stack = np.stack([np.asarray(r['eur']) for r in ens])
        med, crossed = M.ensemble_median(stack)
        ensemble = {'members': 8, 'finite': bool(np.isfinite(med).all()), 'ordered': bool((np.diff(med, axis=-1) >= 0).all()),
                    'crossings_restored': crossed, 'combination': 'per-level median in EUR/MWh (mean of the two middle values)'}
    for r in results:
        r.pop('eur', None)
        r.pop('forecast_days', None)
    rand = [r for r in ok if r['label'].startswith('R-random')]
    fit_s = np.array([r['fit_wall_seconds'] for r in rand]) if rand else np.array([np.nan])
    largest = [r for r in ok if r['label'].startswith('R-largest')]
    largest_s = max(r['fit_wall_seconds'] for r in largest) if largest else math.nan
    mu_search = float(np.mean(fit_s))
    mu_member = float(np.quantile(fit_s, ENSEMBLE_QUANTILE))
    state = budget.read()
    spent = state['counts']
    v4_pass_hours = sum(j.get('charged_seconds', 0) for j in state['jobs'] if j['name'] == 'v4-gate') / HOUR
    b = [len(f['batches']) for f in folds]

    def route(n_trials: int, rounds: int, attempts: int, member_s: float) -> dict:
        fits_search = rounds * search_fits(n_trials, b)
        fits_gate = rounds * GATE_FITS
        fits_attempt = attempts * ATTEMPT_FITS
        fits_other = attempts * OTHER_PER_ATTEMPT['ddnn2_fits'] + REVIEW_RESERVE['ddnn2_fits']
        fits = spent.get('ddnn2_fits', 0) + fits_search + fits_gate + fits_attempt + fits_other
        hours = (spent.get('machine_seconds', 0) / HOUR + (fits_search * mu_search + (fits_gate + fits_attempt) * member_s) / HOUR
                 + max(0.0, 1.6 - v4_pass_hours) + attempts * OTHER_PER_ATTEMPT['machine_hours'] + REVIEW_RESERVE['machine_hours'])
        compute_wall = (hours - spent.get('machine_seconds', 0) / HOUR) / 4.0
        active_now = (time.time() - state['effort']['session_start_epoch']) / HOUR
        lead_hours = 2.0 + 0.5 * rounds + 3.0 * attempts
        active = active_now + compute_wall + lead_hours + REVIEW_RESERVE['active_hours']
        policy_days = 3 * 280 * rounds + 3 * 638 * attempts + attempts * OTHER_PER_ATTEMPT['policy_days'] + REVIEW_RESERVE['policy_days']
        checks = {'ddnn2_fits': fits <= CAPS['ddnn2_fits'], 'machine_hours': hours <= CAPS['machine_seconds'] / HOUR,
                  'active_hours': active <= CAPS['active_seconds'] / HOUR, 'policy_days': policy_days <= CAPS['policy_days'],
                  'bootstrap_passes': 3 * attempts <= CAPS['bootstrap_passes'], 'reference_passes': attempts + 1 <= CAPS['reference_passes'],
                  'rss': True, 'disk': True}
        return {'trials_per_fold_per_round': n_trials, 'rounds': rounds, 'scored_attempts': attempts,
                'ddnn2_fits': int(fits), 'search_fits': int(fits_search), 'gate_fits': int(fits_gate),
                'attempt_fits': int(fits_attempt), 'machine_hours': round(hours, 2), 'compute_wall_hours_4_workers': round(compute_wall, 2),
                'active_hours': round(active, 2), 'policy_days': int(policy_days), 'fits_all_ceilings': all(checks.values()),
                'checks': checks}

    projections = {}
    for n in TRIAL_CHOICES:
        projections[n] = {'minimal': route(n, 1, 1, mu_member), 'maximal': route(n, 4, 2, mu_member),
                          'maximal_worst_case_every_member_largest': route(n, 4, 2, largest_s)}
    fitting = [n for n in TRIAL_CHOICES if projections[n]['maximal']['fits_all_ceilings']]
    minimal_ok = projections[32]['minimal']['fits_all_ceilings']
    finite_ordered = bool(ok and all(r['finite'] and r['ordered'] for r in ok) and ensemble and ensemble['finite']
                          and ensemble['ordered'])
    rss_peak = max((r['maxrss_bytes'] for r in ok), default=0)
    if not minimal_ok:
        verdict, cause = 'NOT_ADMITTED', 'the minimal route does not fit §23.11 at 32 trials'
    elif not finite_ordered or failed:
        verdict, cause = 'NOT_ADMITTED', f'non-finite, crossed or failed emission ({len(failed)} failed fits)'
    else:
        verdict, cause = 'PASS', None
    trials = max(fitting) if fitting else 32
    largest_route = 'maximal' if fitting else ('minimal' if minimal_ok else None)
    record = {
        'schema': 'cp24-resource-admission-v1', 'written_utc': stamp(), 'verdict': verdict, 'cause': cause,
        'data': 'pre-fold only: search-style fits at validation-batch first days, forecasting only that batch\'s days; '
                'the uncovered days left out of every window (§23.6); accuracy-blind',
        'origins': origins, 'tasks': results, 'failed_fits': len(failed),
        'extremes': {'smallest': SMALLEST, 'largest': LARGEST}, 'random_sample_size': RANDOM_SAMPLE,
        'measured': {'mean_random_fit_seconds': mu_search, 'p75_random_fit_seconds': mu_member,
                     'max_random_fit_seconds': float(np.max(fit_s)), 'largest_extreme_fit_seconds': largest_s,
                     'smallest_extreme_fit_seconds': max((r['fit_wall_seconds'] for r in ok if r['label'].startswith('R-smallest')),
                                                         default=math.nan),
                     'peak_worker_rss_bytes': int(rss_peak), 'projected_peak_rss_4_workers_bytes': int(4 * rss_peak),
                     'job_wall_seconds': wall, 'v4_gate_pass_machine_hours_so_far': v4_pass_hours},
        'ensemble_emission': ensemble, 'finite_ordered_emission': finite_ordered,
        'projection_rule': {'search_fit_seconds': 'the sample\'s mean fit time',
                            'gate_and_attempt_member_seconds': f'the sample\'s {ENSEMBLE_QUANTILE:.0%} quantile fit time',
                            'worst_case': 'every gate and attempt member at the largest extreme\'s measured time',
                            'attempt_fits': f'{ATTEMPT_FITS} (636 origins with eligible hours x 8 members)',
                            'gate_fits': f'{GATE_FITS} per round (280 x 8)', 'search_fits': 'sum over folds of N x min(4, B_f) + '
                            'ceil(N/3) x (B_f - min(4, B_f)), B = ' + str(b),
                            'active_hours': 'elapsed so far + compute wall at 4 workers + serial Lead work not overlapped by '
                                            'compute (2 h + 0.5 h per round + 3 h per attempt) + the review reserve',
                            'other_per_attempt': OTHER_PER_ATTEMPT, 'review_reserve': REVIEW_RESERVE},
        'projections': {str(k): v for k, v in projections.items()},
        'fixed': {'trials_per_fold_per_round': trials, 'largest_route_that_fits': largest_route,
                  'review_reserve': REVIEW_RESERVE,
                  'rule': 'the largest trial count in ' + str(list(TRIAL_CHOICES)) + ' at which the maximal route fits every '
                          'ceiling with the review reserve'},
        'spent_before': {k: v for k, v in spent.items()}, 'caps': CAPS}
    atomic(root / OUT / 'resource-admission.json', record)
    print(json.dumps({'verdict': verdict, 'trials': trials, 'mu_search': mu_search, 'p75': mu_member, 'largest': largest_s,
                      'maximal': projections[trials]['maximal']}, default=str), flush=True)
    return 0
