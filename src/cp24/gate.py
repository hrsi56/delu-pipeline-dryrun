"""The pre-fold gate (capstone v21-r11 §23.6, §23.10): v4's members and DDNN-2 at the 280 gate origins.

Each fold f has a gate window G_f = [D0_f - 56, D0_f), none of whose days is a warm-up or evaluation
day of any fold. On each gate day D two sets of forecasts are issued, each from a fresh fit at D's
origin that trains on all history before D (earlier folds' warm-up and evaluation days included);
nothing of fold f on or after D0_f is materialised (the data is loaded before D0_f):

* **v4's members** with their unchanged code (`cp24.v4members`), through the scoped weather-coverage
  wrapper. One pass at the 280 origins, cached and reused across rounds (§23.11), with its parity
  proven at covered origins (`job_v4_parity`) and the unwrapped code's refusal of a fold-4 gate day
  recorded as the paired positive control.
* **DDNN-2** with the round's ensemble for the fold (`job_gate`).

The gate passes if, pooled over the 280 days as point estimates: G0 every forecast is finite and
ordered; G1 v5's central MAE <= v4's central MAE, both built from the gate-day members; G2 D2's MAE
<= 1.10 x L's MAE; G3 the quantile cap binds on fewer than 0.1% of the members' emitted hour-levels.
It is a screen, not a claim: it can only prevent a fold look.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from cp15.data import array_hash, sha
from cp20.components import HGComponents
from cp21.execution import digest
from cp21.lgbm import hg_central, hgl_central
from cp22.execution import cp21_lineage_hashes
from . import v4members as V4
from .budget import atomic, ledger
from .inputs import hg_identity, input_fingerprint, load, weather_design
from .jobs import art, cp20_art, stamp
from .preflight import fold_table

OUT = Path('reports/ddnn2')
V4_CODE = ('src/cp24/v4members.py', 'src/cp21/components.py', 'src/cp21/lgbm.py', 'src/cp20/components.py',
           'src/cp15/models.py', 'src/cp15/data.py')


def v4_identity(root: Path) -> dict:
    return {'cp24_input_fingerprint': input_fingerprint(root), 'weather_design_sha256': weather_design(root).sha256,
            'code_sha256': {name: sha(root / name) for name in V4_CODE}}


def gate_days(folds: list[dict]) -> list[tuple[str, date, date]]:
    out = []
    for f in folds:
        g0, g1 = f['gate']
        out += [(f['fold'], g0 + timedelta(days=k), f['d0']) for k in range((g1 - g0).days + 1)]
    return out


def v4_path(fold: str, day: date) -> Path:
    return art() / 'v4-gate' / fold / f'{day}.json'


def load_v4(fold: str, day: date, identity: dict) -> dict | None:
    path = v4_path(fold, day)
    if not path.exists():
        return None
    item = json.loads(path.read_text())
    if item.get('content_sha256') != digest(item) or item['fold'] != fold or item['day'] != str(day) \
            or item['identity'] != identity:
        raise ValueError(f'stale or wrong v4 gate entry {fold} {day}')
    return item


# ------------------------------------------------------------------ workers
_worker: dict = {}


def _init_worker(root: str, ledger_path: str):
    os.environ['CP24_LEDGER'] = ledger_path
    root = Path(root)
    _worker.update(root=root, design=weather_design(root), budget=ledger(), data={}, identity=v4_identity(root))


def _data_before(cutoff: date):
    w = _worker
    if cutoff not in w['data']:
        w['data'] = {cutoff: load(w['root'], before=cutoff)}
    return w['data'][cutoff]


def _v4_task(task):
    fold, day_s, cutoff_s = task
    w = _worker
    day = date.fromisoformat(day_s)
    try:
        if load_v4(fold, day, w['identity']) is not None:
            return fold, day_s, 'reused', 0.0, None
    except ValueError:
        pass  # a stale or wrong entry is refitted, never reused (and the pass cap still binds)
    began = time.time()
    try:
        data = _data_before(date.fromisoformat(cutoff_s))
        w['budget'].reserve(v4_gate_origins=1)
        result = V4.fit(data, w['design'], day, w['budget'], 'gate', wrapped=True)
    except Exception as exc:  # recorded, never substituted
        atomic(art() / 'failures' / f'v4-gate_{fold}_{day_s}.json', {'fold': fold, 'day': day_s, 'error': repr(exc),
                                                                     'traceback': traceback.format_exc(), 'written_utc': stamp()})
        return fold, day_s, 'failed', time.time() - began, repr(exc)
    item = {'fold': fold, 'data_cutoff_exclusive': cutoff_s, 'identity': w['identity'], **result, 'written_utc': stamp()}
    item['content_sha256'] = digest(item)
    atomic(v4_path(fold, day), item)
    return fold, day_s, 'fitted', time.time() - began, None


def _pool(root: Path, workers: int):
    if workers > int(os.environ.get('CP24_WORKERS', '1')):
        raise ValueError('more pool workers than the monitor declared')
    return mp.get_context('spawn').Pool(workers, initializer=_init_worker, initargs=(str(root), os.environ['CP24_LEDGER']))


def job_v4_gate(root: Path, rest) -> int:
    """v4's members at the 280 gate origins: one pass, cached, reused across rounds."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=int(os.environ.get('CP24_WORKERS', '1')))
    args = ap.parse_args(rest)
    folds = fold_table(root)
    tasks = [(fold, str(day), str(d0)) for fold, day, d0 in gate_days(folds)]
    if len(tasks) != 280:
        raise ValueError('the gate has 280 origins')
    budget = ledger()
    budget.event('v4_gate_start', tasks=len(tasks), workers=args.workers)
    failed, t0 = 0, time.time()
    with _pool(root, args.workers) as pool:
        for i, (fold, day, source, seconds, error) in enumerate(pool.imap_unordered(_v4_task, tasks, chunksize=1), 1):
            failed += source == 'failed'
            if i % 20 == 0 or source == 'failed':
                print('v4-gate', fold, day, source, round(seconds, 1), f'{i}/{len(tasks)}', f'{time.time() - t0:.0f}s',
                      error or '', flush=True)
    budget.event('v4_gate_complete', failed=failed)
    print(json.dumps({'tasks': len(tasks), 'failed': failed, 'seconds': time.time() - t0}), flush=True)
    return 0 if not failed else 5


# ------------------------------------------------------------------ the parity proof and its control
PARITY_OFFSETS = (0, 45)
#: Fold 4's first warm-up day: its window starts on 2023-03-25, the first covered day after the gap.
BOUNDARY_ORIGIN = ('fold_4', date(2025, 3, 22))
REFUSAL_ORIGIN = ('fold_4', date(2025, 1, 25))


def parity_origins(root: Path) -> list[tuple[str, date, str]]:
    """Two evaluation origins per fold (offsets 0 and 45 days, the next day with eligible hours), plus
    fold 4's first warm-up day, whose window starts on the first covered day after the gap."""
    out = []
    full = load(root)
    for f in fold_table(root):
        for off in PARITY_OFFSETS:
            d = f['evaluation_start'] + timedelta(days=off)
            while not len(full.rows(d)):
                d += timedelta(days=1)
            out.append((f['fold'], d, 'evaluation'))
    out.append((*BOUNDARY_ORIGIN, 'warmup'))
    return out


def _parity_task(task):
    fold, day_s, phase, cutoff_s = task
    w = _worker
    day = date.fromisoformat(day_s)
    data = _data_before(date.fromisoformat(cutoff_s) if cutoff_s else date(2026, 4, 8))
    w['budget'].reserve(v4_parity_origins=1)
    result = V4.fit(data, w['design'], day, w['budget'], 'parity', wrapped=True)
    return fold, day_s, phase, result


def job_v4_parity(root: Path, rest) -> int:
    """The gate's v4 code path, wrapper included, against the committed fold vectors (bit for bit), and
    the unwrapped code's refusal of a fold-4 gate day (the paired positive control). This is also
    item 2's independent representative HG and v4 slice. Reproducing committed vectors at warm-up or
    evaluation origins reads no outcome that is not already committed (§23.6)."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=int(os.environ.get('CP24_WORKERS', '1')))
    args = ap.parse_args(rest)
    folds = {f['fold']: f for f in fold_table(root)}
    origins = parity_origins(root)
    tasks = [(fold, str(d), phase, None if phase == 'evaluation' else str(folds[fold]['evaluation_start']))
             for fold, d, phase in origins]
    members = pd.read_parquet(root / 'reports/distribution-challenger/members.parquet')
    cp21_saved = pd.read_parquet(root / 'reports/block-challenger/predictions.parquet')
    for frame in (members, cp21_saved):
        frame['timestamp_utc'] = pd.to_datetime(frame.timestamp_utc, utc=True)
    hashes = cp21_lineage_hashes(root)
    hg_ident = hg_identity(root)
    results = []
    with _pool(root, args.workers) as pool:
        for fold, day_s, phase, res in pool.imap_unordered(_parity_task, tasks, chunksize=1):
            day = date.fromisoformat(day_s)
            ts = pd.DatetimeIndex(pd.to_datetime(res['timestamp_utc'], utc=True))
            got = {k: np.asarray(v, float) for k, v in res['central'].items()}
            rec = {'fold': fold, 'day': day_s, 'phase': phase, 'excluded_training_days': res['wrapper']['excluded_training_days']}
            if phase == 'evaluation':
                mem = members.loc[members.fold.eq(fold)].set_index('timestamp_utc').loc[ts]
                ln = cp21_saved.loc[cp21_saved.policy.eq('L-N') & cp21_saved.fold.eq(fold)].set_index('timestamp_utc').loc[ts]
                lr = cp21_saved.loc[cp21_saved.policy.eq('L-R') & cp21_saved.fold.eq(fold)].set_index('timestamp_utc').loc[ts]
                committed = {'A1': mem['A1'].to_numpy(float), 'B2': mem['B2'].to_numpy(float),
                             'L-N': ln.central.to_numpy(float), 'L-R': lr.central.to_numpy(float)}
                rec['committed_source'] = ('reports/distribution-challenger/members.parquet (A1, B2); '
                                           'reports/block-challenger/predictions.parquet (L-N, L-R)')
                rec['bitwise_equal'] = {k: bool(np.array_equal(got[k], committed[k])) for k in committed}
                rec['max_abs_difference'] = {k: float(np.max(np.abs(got[k] - committed[k]))) for k in committed}
                rec['HG_bitwise_equal'] = bool(np.array_equal(hg_central(got['A1'], got['B2']), mem['HG'].to_numpy(float)))
                rec['v4_bitwise_equal'] = bool(np.array_equal(
                    hgl_central(got['A1'], got['B2'], got['L-N'], got['L-R']), mem['HGL'].to_numpy(float)))
            else:
                data = load(root, before=folds[fold]['evaluation_start'])
                _, centers, _ = HGComponents(cp20_art() / 'hg-components', fold, data, hg_ident).get(day)
                rec['committed_source'] = ('CP-20 HG component cache (A1, B2), identity-verified; CP-21 committed lineage '
                                           'central_sha256 (L-N, L-R, v4)')
                rec['bitwise_equal'] = {'A1': bool(np.array_equal(got['A1'], centers['A1'])),
                                        'B2': bool(np.array_equal(got['B2'], centers['B2'])),
                                        'L-N': array_hash(got['L-N']) == hashes[('L-N', fold, day_s)],
                                        'L-R': array_hash(got['L-R']) == hashes[('L-R', fold, day_s)]}
                rec['v4_bitwise_equal'] = array_hash(hgl_central(got['A1'], got['B2'], got['L-N'], got['L-R'])) \
                    == hashes[('HGL', fold, day_s)]
                rec['HG_bitwise_equal'] = bool(rec['bitwise_equal']['A1'] and rec['bitwise_equal']['B2'])
            rec['wrapper_changed_nothing'] = not rec['excluded_training_days']
            results.append(rec)
            print('parity', fold, day_s, phase, rec['bitwise_equal'], rec['v4_bitwise_equal'], flush=True)
    # The paired positive control: the unwrapped code refuses a fold-4 gate day; the wrapper does not.
    _init_worker(str(root), os.environ['CP24_LEDGER'])
    fold, day = REFUSAL_ORIGIN
    data = load(root, before=folds[fold]['d0'])
    refusal = V4.refuses_unwrapped(data, _worker['design'], day, _worker['budget'])
    passed = all(all(r['bitwise_equal'].values()) and r['HG_bitwise_equal'] and r['v4_bitwise_equal']
                 and r['wrapper_changed_nothing'] for r in results) and refusal['all_refused'] \
        and len(refusal['excluded_by_wrapper']) > 0
    out = {'schema': 'cp24-v4-parity-v1', 'written_utc': stamp(), 'passed': passed,
           'code_path': 'cp24.v4members.fit (wrapper applied): cp21.components.refit for A1_w/B2_w, cp21.lgbm.fit_arm for '
                        'L-N/L-R -- the gate\'s exact path',
           'origins': sorted(results, key=lambda r: (r['fold'], r['day'])),
           'unwrapped_refuses_fold4_gate_day': {'fold': fold, 'day': str(day), **refusal},
           'reading': 'bitwise parity with the committed vectors at covered origins, where the wrapper leaves out no '
                      'training day; the unwrapped member code refuses a fold-4 gate day whose window reaches the gap'}
    atomic(root / OUT / 'v4-parity.json', out)
    print(json.dumps({'passed': passed, 'refusal': refusal['all_refused'],
                      'n_excluded_at_refusal_day': len(refusal['excluded_by_wrapper'])}), flush=True)
    return 0 if passed else 6


# ------------------------------------------------------------------ DDNN-2 at the gate origins
G2_RATIO = 1.10
G3_SHARE = 0.001


def member_path(r: int, fold: str, day: date, j: int) -> Path:
    return art() / 'rounds' / f'round-{r}' / 'gate' / fold / str(day) / f'm{j}.json'


def gate_identity(root: Path, r: int) -> dict:
    from .search import search_identity
    ident = search_identity(root, r)
    ident['ensembles_sha256'] = sha(Path(root) / OUT / 'rounds' / f'round-{r}' / 'ensembles.json')
    ident['gate_code_sha256'] = sha(Path(root) / 'src/cp24/gate.py')
    return ident


def _init_gate_worker(root: str, ledger_path: str, r: int):
    os.environ['CP24_LEDGER'] = ledger_path
    from .reference import require_passing_record
    root = Path(root)
    require_passing_record(root)
    _worker.update(root=root, design=weather_design(root), budget=ledger(), tables={}, round=r,
                   identity=gate_identity(root, r))


def _gate_task(task):
    from . import design as G
    from .budget import charge_fits
    from .member import Keys, fit_member
    fold, day_s, d0_s, j, config, seed = task
    w = _worker
    day = date.fromisoformat(day_s)
    path = member_path(w['round'], fold, day, j)
    if path.exists():
        item = json.loads(path.read_text())
        if item.get('content_sha256') == digest(item) and item['identity'] == w['identity']:
            return fold, day_s, j, 'reused', 0.0, None
    cutoff = date.fromisoformat(d0_s)
    if cutoff not in w['tables']:
        data = load(w['root'], before=cutoff)
        dd = G.build(data, w['design'])
        w['tables'] = {cutoff: (dd, Keys.from_data(data, dd))}
    dd, keys = w['tables'][cutoff]
    began = time.time()
    item = {'fold': fold, 'day': day_s, 'member': j, 'config_id': config['id'], 'seed': seed,
            'data_cutoff_exclusive': d0_s, 'identity': w['identity']}
    try:
        days = np.array([dd.ix(day)])
        fit = fit_member(dd, day, config, seed, days, exclude_uncovered=True, charge=charge_fits(w['budget'], 'gate'))
        w['budget'].reserve(ddnn2_epochs=int(fit['member']['epochs_run']))
        item.update(status='ok', eur=fit['eur'][0].tolist(), active=fit['active'][0].astype(int).tolist(),
                    cap_z=fit['cap_z'], n_inputs=fit['n_inputs'],
                    record={k: v for k, v in fit['member'].items() if k not in ('stopping_metric_history', 'train_loss_history')},
                    window=fit['window'], guards={k: v for k, v in fit['guards'].items() if k != 'cap_forecast_by_day'},
                    seconds=fit['seconds'], maxrss_bytes=fit['maxrss_bytes'])
    except Exception as exc:  # recorded, never substituted
        item.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
    item['written_utc'] = stamp()
    item['content_sha256'] = digest(item)
    atomic(path, item)
    return fold, day_s, j, item['status'], time.time() - began, item.get('error')


def job_gate(root: Path, rest) -> int:
    """DDNN-2 with the round's ensembles at the 280 gate origins, then G0-G3 against v4's gate members."""
    import argparse
    from .search import committed_at_head, round_dir
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('CP24_WORKERS', '1')))
    args = ap.parse_args(rest)
    root, r = Path(root), args.round
    rd = round_dir(root, r)
    for name in ('design.json', 'search-ledger.json', 'ensembles.json'):
        committed_at_head(root, str((rd / name).relative_to(root)))
    if (rd / 'gate.json').exists():
        raise ValueError(f'round {r}\'s gate has a result: it runs once')
    ens = json.loads((rd / 'ensembles.json').read_text())['folds']
    folds = fold_table(root)
    tasks = []
    for fold, day, d0 in gate_days(folds):
        members = ens[fold]['members']
        if len(members) != 8:
            raise ValueError(f'{fold}: the ensemble is incomplete')
        for j, m in enumerate(members):
            tasks.append((fold, str(day), str(d0), j, m['config'], m['seed']))
    budget = ledger()
    v4_ident = v4_identity(root)
    missing = [(f, d) for f, d, _ in gate_days(folds) if load_v4(f, d, v4_ident) is None]
    if missing:
        raise ValueError(f'v4\'s gate pass is incomplete: {len(missing)} origins missing')
    budget.event('gate_start', round=r, tasks=len(tasks), workers=args.workers)
    t0, failed = time.time(), 0
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, initializer=_init_gate_worker, initargs=(str(root), os.environ['CP24_LEDGER'], r)) as pool:
        for i, (fold, day, j, status, seconds, error) in enumerate(pool.imap(_gate_task, tasks, chunksize=1), 1):
            failed += status == 'failed'
            if i % 80 == 0 or status == 'failed':
                print('gate', f'{i}/{len(tasks)}', fold, day, j, status, round(seconds, 1), f'{time.time() - t0:.0f}s',
                      error or '', flush=True)
    result = evaluate_gate(root, r, folds, ens, v4_ident)
    result['fits_failed'] = failed
    result['wall_seconds'] = time.time() - t0
    atomic(rd / 'gate.json', result)
    budget.event('gate_complete', round=r, passed=result['passed'], failed=failed)
    print(json.dumps({'round': r, 'passed': result['passed'], 'conditions': {k: v['met'] for k, v in result['conditions'].items()},
                      'pooled': result['pooled']}, default=str), flush=True)
    return 0


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_gate(root: Path, r: int, folds: list[dict], ens: dict, v4_ident: dict) -> dict:
    """G0-G3, pooled over the 280 gate days as point estimates, with the by-fold table, D2's error
    correlations with HG and L, the excluded training days and the costs."""
    from . import ddnn2 as M
    budget = ledger()
    per_fold, pooled_err = {}, {k: [] for k in ('v4', 'v5', 'D2', 'L', 'HG')}
    cap_active = cap_total = 0
    finite_ordered = True
    excluded_v4, excluded_d2, costs = {}, {}, {'ddnn2_fit_seconds': 0.0, 'v4_fit_seconds': 0.0}
    epochs = []
    for f in folds:
        data = load(root, before=f['d0'])
        days = [d for fold, d, _ in gate_days([f])]
        errs = {k: [] for k in pooled_err}
        fold_cap = fold_total = 0
        for day in days:
            budget.reserve(policy_days=3, policy_days_gate=3)
            rows = data.rows(day)
            ts = list(map(str, data.index[rows]))
            v4 = load_v4(f['fold'], day, v4_ident)
            if v4['timestamp_utc'] != ts:
                raise ValueError(f'{f["fold"]} {day}: v4 gate keys differ')
            excluded_v4[str(day)] = v4['wrapper']['n_excluded_training_days']
            costs['v4_fit_seconds'] += v4['seconds']
            c = {k: np.asarray(v, float) for k, v in v4['central'].items()}
            members = []
            for j in range(8):
                item = json.loads(member_path(r, f['fold'], day, j).read_text())
                if item.get('status') != 'ok':
                    finite_ordered = False
                    raise ValueError(f'{f["fold"]} {day} member {j} failed: {item.get("error")}')
                members.append(np.asarray(item['eur'], float))           # 24 x 7
                act = np.asarray(item['active'], int)                     # 24 x 7, the day's slots
                hours = data.hours[rows]
                fold_cap += int(act[hours].sum())                         # the emitted hour-levels of this member
                fold_total += int(len(rows) * len(M.LEVELS))
                excluded_d2.setdefault(str(day), len(item['window']['excluded_uncovered_days']))
                costs['ddnn2_fit_seconds'] += item['seconds']
                epochs.append(item['record']['epochs_run'])
            med, crossed = M.ensemble_median(np.stack(members)[:, None])  # 1 x 24 x 7
            q = med[0][data.hours[rows]]                                  # rows x 7
            d2 = q[:, M.MEDIAN]
            finite_ordered &= bool(np.isfinite(q).all() and (np.diff(q, axis=1) >= 0).all()
                                   and all(np.isfinite(v).all() for v in c.values()))
            L = c['L-N'] / 2 + c['L-R'] / 2
            HG = c['A1'] / 2 + c['B2'] / 2
            v4c = c['A1'] / 3 + c['B2'] / 3 + c['L-N'] / 6 + c['L-R'] / 6
            v5c = c['A1'] / 3 + c['B2'] / 3 + c['L-N'] / 12 + c['L-R'] / 12 + d2 / 6
            y = data.y[rows]
            ok = data.eligible[rows]
            for k, v in (('v4', v4c), ('v5', v5c), ('D2', d2), ('L', L), ('HG', HG)):
                errs[k].append((v - y)[ok])
        cap_active += fold_cap
        cap_total += fold_total
        e = {k: np.concatenate(v) for k, v in errs.items()}
        for k in pooled_err:
            pooled_err[k].append(e[k])
        per_fold[f['fold']] = {'gate': [str(f['gate'][0]), str(f['gate'][1])], 'days': len(days), 'hours': int(len(e['v4'])),
                               **{f'MAE_{k}': float(np.mean(np.abs(e[k]))) for k in e},
                               'cap_share': fold_cap / fold_total if fold_total else float('nan'),
                               'corr_D2_HG': _corr(e['D2'], e['HG']), 'corr_D2_L': _corr(e['D2'], e['L']),
                               'excluded_training_days_v4_max': max(excluded_v4[str(d)] for d in days),
                               'excluded_training_days_d2_max': max(excluded_d2[str(d)] for d in days)}
    e = {k: np.concatenate(v) for k, v in pooled_err.items()}
    mae = {k: float(np.mean(np.abs(v))) for k, v in e.items()}
    share = cap_active / cap_total
    conditions = {
        'G0': {'met': bool(finite_ordered), 'text': 'every forecast is finite and ordered'},
        'G1': {'met': mae['v5'] <= mae['v4'], 'text': "v5's central MAE <= v4's central MAE, both from the gate-day members",
               'v5': mae['v5'], 'v4': mae['v4'], 'difference': mae['v5'] - mae['v4']},
        'G2': {'met': mae['D2'] <= G2_RATIO * mae['L'], 'text': "D2's MAE <= 1.10 x L's MAE", 'D2': mae['D2'], 'L': mae['L'],
               'ratio': mae['D2'] / mae['L']},
        'G3': {'met': share < G3_SHARE, 'text': 'the quantile cap binds on fewer than 0.1% of the members\' emitted hour-levels',
               'share': share, 'binding': int(cap_active), 'emitted_hour_levels': int(cap_total)},
    }
    return {'schema': 'cp24-gate-v1', 'round': r, 'written_utc': stamp(), 'passed': all(c['met'] for c in conditions.values()),
            'conditions': conditions, 'pooled': {'days': sum(p['days'] for p in per_fold.values()), 'hours': int(len(e['v4'])),
                                                 **{f'MAE_{k}': v for k, v in mae.items()},
                                                 'corr_D2_HG': _corr(e['D2'], e['HG']), 'corr_D2_L': _corr(e['D2'], e['L'])},
            'by_fold': per_fold, 'excluded_training_days': {'v4_members_by_day': excluded_v4, 'ddnn2_by_day': excluded_d2},
            'costs': {**costs, 'ddnn2_epochs': {'median': float(np.median(epochs)), 'max': int(np.max(epochs))}},
            'reading': 'a screen, not a claim: point estimates pooled over the 280 gate days; it can only prevent a fold look',
            'information': 'every gate fit trains on all history before its day; data materialised before each fold\'s D0'}
