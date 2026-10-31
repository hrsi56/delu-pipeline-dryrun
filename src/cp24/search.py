"""The per-fold training-only search of one pre-fold round (capstone v21-r11 §23.4, §23.6).

For each fold f, before its warm-up start D0_f and by a procedure identical for every fold:

* the data is materialised before D0_f - 56, so nothing dated on or after the gate window enters;
* every trial (`cp24.sampler.trials`, seeded by round and fold) is fitted once per validation batch,
  with a recorded fixed seed, on [max(2019-01-01, b - 728), b) minus its held-out weeks and the
  uncovered days (§23.6), and forecasts the batch's 28 days;
* successive halving and the ranking are `cp24.sampler`'s; the fold's ensemble is the four best
  distinct configurations among the trials that ran all B_f batches, two seeds each.

The round's design (`reports/ddnn2/rounds/round-<r>/design.json`: the trial count, the space and any
S1-named change) must be committed at HEAD before the search runs, and the search ledger and the
ensembles it writes are committed before the gate runs. Every fit is charged before it trains; every
fit's record is kept, failures included (a failed trial ranks last), and a restart reuses only
identity-verified records.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

from cp15.data import sha
from cp21.execution import digest
from . import ddnn2 as M
from . import design as G
from . import sampler as SP
from .budget import atomic, charge_fits, ledger
from .inputs import input_fingerprint, load, weather_design
from .jobs import art, stamp
from .member import Keys, config_hash, emit_rows, fit_member, pinball_rows
from .preflight import fold_table
from .reference import require_passing_record

OUT = Path('reports/ddnn2')
MODEL_CODE = ('src/cp24/ddnn2.py', 'src/cp24/sampler.py', 'src/cp24/design.py', 'src/cp24/member.py',
              'src/cp24/search.py', 'src/cp15/data.py', 'src/cp20/weather.py')


def round_dir(root: Path, r: int) -> Path:
    return Path(root) / OUT / 'rounds' / f'round-{r}'


def committed_at_head(root: Path, name: str) -> None:
    committed = subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=root)
    if committed != (Path(root) / name).read_bytes():
        raise ValueError(f'{name} must be committed at HEAD')


def load_design(root: Path, r: int) -> dict:
    path = round_dir(root, r) / 'design.json'
    committed_at_head(root, str(path.relative_to(root)))
    d = json.loads(path.read_text())
    if d['round'] != r or d['trials_per_fold'] < 32:
        raise ValueError('invalid round design')
    return d


def search_identity(root: Path, r: int) -> dict:
    return {'round': r, 'cp24_input_fingerprint': input_fingerprint(root), 'weather_design_sha256': weather_design(root).sha256,
            'design_sha256': sha(round_dir(root, r) / 'design.json'),
            'code_sha256': {name: sha(Path(root) / name) for name in MODEL_CODE}}


def task_path(r: int, fold: str, trial: int, batch: int) -> Path:
    return art() / 'rounds' / f'round-{r}' / 'search' / fold / f't{trial:03d}-b{batch:02d}.json'


def _load_task(path: Path, identity: dict):
    if not path.exists():
        return None
    item = json.loads(path.read_text())
    if item.get('content_sha256') != digest(item) or item['identity'] != identity:
        return None
    return item


# ------------------------------------------------------------------ workers
_worker: dict = {}


def _init_worker(root: str, ledger_path: str, r: int):
    os.environ['CP24_LEDGER'] = ledger_path
    root = Path(root)
    require_passing_record(root)
    _worker.update(root=root, design=weather_design(root), budget=ledger(), tables={}, identity=search_identity(root, r),
                   round=r)


def tables(cutoff: date):
    w = _worker
    if cutoff not in w['tables']:
        data = load(w['root'], before=cutoff)
        dd = G.build(data, w['design'])
        w['tables'][cutoff] = (dd, Keys.from_data(data, dd))
    return w['tables'][cutoff]


def score_batch(keys: Keys, days: np.ndarray, eur: np.ndarray) -> dict:
    """Pinball (7 levels) and absolute-error sums over the batch's scored keys (eligible hours)."""
    rows = keys.of_days(days)
    rows = rows[keys.eligible[rows]]
    q = emit_rows(keys, rows, days, eur)
    y = keys.y[rows]
    pin = pinball_rows(q, y)
    return {'pinball_sum': float(pin.sum()), 'pinball_count': int(pin.size), 'abs_error_sum': float(np.abs(q[:, M.MEDIAN] - y).sum()),
            'hours': int(len(rows)), 'finite': bool(np.isfinite(q).all()), 'ordered': bool((np.diff(q, axis=1) >= 0).all())}


def _task(task):
    fold, fi, trial, batch, config, seed, b0_s, cutoff_s = task
    w = _worker
    path = task_path(w['round'], fold, trial, batch)
    item = _load_task(path, w['identity'])
    if item is not None:
        return item, 'reused'
    dd, keys = tables(date.fromisoformat(cutoff_s))
    b0 = date.fromisoformat(b0_s)
    days = np.array([dd.ix(b0 + timedelta(days=k)) for k in range(SP.BATCH_DAYS)])
    began = time.time()
    item = {'fold': fold, 'fold_index': fi, 'trial': trial, 'config_id': config['id'], 'config_sha256': config_hash(config),
            'batch': batch, 'batch_start': b0_s, 'batch_end': str(b0 + timedelta(days=SP.BATCH_DAYS - 1)), 'seed': seed,
            'data_cutoff_exclusive': cutoff_s, 'identity': w['identity']}
    try:
        fit = fit_member(dd, b0, config, seed, days, exclude_uncovered=True, charge=charge_fits(w['budget'], 'search'))
        w['budget'].reserve(ddnn2_epochs=int(fit['member']['epochs_run']))
        item.update(status='ok', **score_batch(keys, days, fit['eur']), n_inputs=fit['n_inputs'],
                    n_parameters=M.n_parameters(fit['n_inputs'], tuple(config['hidden'])),
                    member={k: v for k, v in fit['member'].items() if k not in ('stopping_metric_history', 'train_loss_history')},
                    window={k: (v if k != 'excluded_uncovered_days' else len(v)) for k, v in fit['window'].items()},
                    guards={k: v for k, v in fit['guards'].items() if k != 'cap_forecast_by_day'},
                    seconds=fit['seconds'], maxrss_bytes=fit['maxrss_bytes'])
    except Exception as exc:  # a failed fit is recorded and ranks last, never substituted
        item.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), seconds=time.time() - began)
    item['written_utc'] = stamp()
    item['content_sha256'] = digest(item)
    atomic(path, item)
    return item, item['status']


def _run(pool, tasks, label):
    out, t0 = [], time.time()
    for i, (item, source) in enumerate(pool.imap_unordered(_task, tasks, chunksize=1), 1):
        out.append(item)
        if i % 25 == 0 or item['status'] == 'failed':
            print(label, f'{i}/{len(tasks)}', item['fold'], item['config_id'], item['batch'], source, item['status'],
                  round(item.get('seconds', 0), 1), f'{time.time() - t0:.0f}s', item.get('error', ''), flush=True)
    return out


def pooled_metric(items: list[dict], batches: list[int]) -> tuple[float, float]:
    by = {it['batch']: it for it in items}
    if any(b not in by or by[b]['status'] != 'ok' for b in batches):
        return math.inf, math.inf
    pin = sum(by[b]['pinball_sum'] for b in batches) / sum(by[b]['pinball_count'] for b in batches)
    mae = sum(by[b]['abs_error_sum'] for b in batches) / sum(by[b]['hours'] for b in batches)
    return pin, mae


def job_search(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('CP24_WORKERS', '1')))
    args = ap.parse_args(rest)
    root, r = Path(root), args.round
    require_passing_record(root)
    committed_at_head(root, str(OUT / 'resource-admission.json'))
    admission = json.loads((root / OUT / 'resource-admission.json').read_text())
    if admission['verdict'] != 'PASS':
        raise ValueError('4.6R′ did not pass: no round (§23.7)')
    design = load_design(root, r)
    n = design['trials_per_fold']
    if n != admission['fixed']['trials_per_fold_per_round'] and not design.get('trial_count_change'):
        raise ValueError('the trial count differs from 4.6R′\'s record without a recorded change')
    if (round_dir(root, r) / 'search-ledger.json').exists():
        raise ValueError(f'round {r} has a search ledger: a round\'s search runs once')
    budget = ledger()
    counter = 'rounds_before_attempt_1' if design['before_attempt'] == 1 else 'rounds_before_attempt_2'
    marker = art() / 'rounds' / f'round-{r}' / 'started.json'
    if not marker.exists():
        budget.reserve(**{counter: 1, 'rounds': 1})
        atomic(marker, {'round': r, 'counter': counter, 'written_utc': stamp()})
    space = design['space']
    folds = fold_table(root)
    plan = {}
    for f in folds:
        trials = SP.trials(r, f['index'], n, space)
        plan[f['fold']] = {'fold': f, 'trials': trials}
    stage_a = []
    for name, p in plan.items():
        f = p['fold']
        nb = len(f['batches'])
        for t, cfg in enumerate(p['trials']):
            for j in range(min(SP.HALVING_BATCHES, nb)):
                stage_a.append((name, f['index'], t, j, cfg, SP.fit_seed(r, f['index'], t, j), str(f['batches'][j][0]),
                                str(f['search_cutoff'])))
    budget.event('search_start', round=r, trials=n, stage_a=len(stage_a), workers=args.workers)
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(str(root), os.environ['CP24_LEDGER'], r)) as pool:
        items = _run(pool, stage_a, 'search-A')
        results = {name: {t: [] for t in range(n)} for name in plan}
        for it in items:
            results[it['fold']][it['trial']].append(it)
        stage_b, survivors = [], {}
        for name, p in plan.items():
            f = p['fold']
            nb = len(f['batches'])
            first = list(range(min(SP.HALVING_BATCHES, nb)))
            metric = {p['trials'][t]['id']: pooled_metric(results[name][t], first)[0] for t in range(n)}
            size = {p['trials'][t]['id']: SP.n_params(p['trials'][t], next((i['n_inputs'] for i in results[name][t]
                                                                             if i['status'] == 'ok'), 10**6)) for t in range(n)}
            order = [c['id'] for c in p['trials']]
            ranked = SP.rank(metric, size, order)
            keep = ranked[:SP.halving_survivors(n)] if nb > SP.HALVING_BATCHES else ranked
            survivors[name] = keep
            if nb > SP.HALVING_BATCHES:
                for cid in keep:
                    t = order.index(cid)
                    for j in range(SP.HALVING_BATCHES, nb):
                        stage_b.append((name, f['index'], t, j, p['trials'][t], SP.fit_seed(r, f['index'], t, j),
                                        str(f['batches'][j][0]), str(f['search_cutoff'])))
        budget.event('search_stage_b', round=r, stage_b=len(stage_b))
        for it in _run(pool, stage_b, 'search-B'):
            results[it['fold']][it['trial']].append(it)
    ledger_out, ensembles = {}, {}
    failed_total = 0
    for name, p in plan.items():
        f = p['fold']
        nb = len(f['batches'])
        allb = list(range(nb))
        first = list(range(min(SP.HALVING_BATCHES, nb)))
        order = [c['id'] for c in p['trials']]
        rows = []
        for t, cfg in enumerate(p['trials']):
            its = sorted(results[name][t], key=lambda i: i['batch'])
            failed_total += sum(i['status'] == 'failed' for i in its)
            pin_a, mae_a = pooled_metric(its, first)
            pin_all, mae_all = pooled_metric(its, allb) if cfg['id'] in survivors[name] else (math.inf, math.inf)
            n_inputs = next((i['n_inputs'] for i in its if i['status'] == 'ok'), None)
            rows.append({'trial': t, 'config': cfg, 'config_sha256': config_hash(cfg), 'n_inputs': n_inputs,
                         'n_parameters': SP.n_params(cfg, n_inputs) if n_inputs else None,
                         'ran_all_batches': cfg['id'] in survivors[name] and len(its) == nb,
                         'pinball_first_batches': pin_a, 'mae_first_batches': mae_a,
                         'pinball_all_batches': pin_all, 'mae_all_batches': mae_all,
                         'batches': [{'batch': i['batch'], 'start': i['batch_start'], 'seed': i['seed'], 'status': i['status'],
                                      'pinball': (i['pinball_sum'] / i['pinball_count']) if i['status'] == 'ok' else None,
                                      'mae': (i['abs_error_sum'] / i['hours']) if i['status'] == 'ok' else None,
                                      'epochs_run': i.get('member', {}).get('epochs_run'),
                                      'best_epoch': i.get('member', {}).get('best_epoch'),
                                      'stop_reason': i.get('member', {}).get('stop_reason'),
                                      'seconds': i.get('seconds'), 'cap_activations': i.get('guards', {}).get('cap_forecast_slot_levels'),
                                      'excluded_uncovered_days': i.get('window', {}).get('excluded_uncovered_days'),
                                      'error': i.get('error'), 'record_sha256': i['content_sha256']} for i in its]})
        eligible = {row['config']['id']: row['pinball_all_batches'] for row in rows if row['ran_all_batches']}
        size = {row['config']['id']: row['n_parameters'] or 10**9 for row in rows}
        ranked = [cid for cid in SP.rank(eligible, size, order) if math.isfinite(eligible[cid])]
        distinct, chosen = set(), []
        for cid in ranked:
            cfg = p['trials'][order.index(cid)]
            key = json.dumps({k: v for k, v in cfg.items() if k != 'id'}, sort_keys=True)
            if key not in distinct:
                distinct.add(key)
                chosen.append(cid)
            if len(chosen) == SP.ENSEMBLE_CONFIGS:
                break
        seeds = SP.member_seeds(r, f['index'])
        members = []
        for j, cid in enumerate(chosen):
            row = rows[order.index(cid)]
            for s in range(SP.ENSEMBLE_SEEDS):
                members.append({'rank': j + 1, 'config': row['config'], 'seed': seeds[j * SP.ENSEMBLE_SEEDS + s],
                                'validation_pinball_all_batches': row['pinball_all_batches'],
                                'validation_mae_all_batches': row['mae_all_batches']})
        ensembles[name] = {'fold': name, 'd0': str(f['d0']), 'B': nb, 'members': members, 'complete': len(chosen) == SP.ENSEMBLE_CONFIGS}
        ledger_out[name] = {'fold': name, 'd0': str(f['d0']), 'data_cutoff_exclusive': str(f['search_cutoff']),
                            'batches': [[str(a), str(b)] for a, b in f['batches']], 'B': nb,
                            'sampler_seed': [24, r, f['index'], 0x5A3], 'trials': rows,
                            'survivors': survivors[name], 'ranking_all_batches': ranked, 'ensemble': chosen}
    rd = round_dir(root, r)
    identity = search_identity(root, r)
    atomic(rd / 'search-ledger.json', {'schema': 'cp24-search-ledger-v1', 'round': r, 'written_utc': stamp(),
                                       'identity': identity, 'trials_per_fold': n, 'space': space, 'folds': ledger_out,
                                       'failed_fits': failed_total,
                                       'procedure': 'cp24.sampler: N trials on the min(4, B_f) most recent batches; the best '
                                                    'ceil(N/3) on all B_f; ensemble = four best distinct configurations '
                                                    'among trials that ran all B_f batches; ties to fewer parameters'})
    atomic(rd / 'ensembles.json', {'schema': 'cp24-ensembles-v1', 'round': r, 'written_utc': stamp(), 'identity': identity,
                                   'folds': ensembles, 'combination': 'per-level median of the eight members\' EUR/MWh quantiles'})
    budget.event('search_complete', round=r, failed=failed_total)
    print(json.dumps({'round': r, 'failed_fits': failed_total,
                      'ensembles': {k: [m['config']['id'] for m in v['members'][::2]] for k, v in ensembles.items()}}), flush=True)
    return 0 if all(e['complete'] for e in ensembles.values()) else 5
