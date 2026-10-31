"""Future-blind refits of scored attempt 1's DDNN-2: §23.10's leakage controls at full coverage (item 9).

**Why.** D2 alone improves on v4 by 10.67% in S_MAE and 15.53% in S_WIS, while CP-23's DDNN was worse than
v3. For a member this strong, leakage is the first explanation to rule out. The committed controls
(`attempt-1/controls.json`, 80 of 80) prove the masks at two representative origins (fold 1's and fold 4's
first evaluation days), with member 0 and one weather member, and the search and gate negatives for one fold
each. A leak confined to another input group of the 20 frozen configurations, another fold, or another origin
(a DST day, a window boundary, the warm-up materialisation) would pass them. This module closes that gap.

**What it checks.** It is verification only: no frozen element changes, no forecast is produced or replaced and
nothing is rescored. Every refit is charged as a control fit (§23.11).

1. **Every origin, every member (negative).** At each of the 636 warm-up and evaluation origins, the fold's
   frozen eight-member ensemble is refitted from the production inputs after destroying every outcome dated on
   or after the delivery day D: prices on and after D are removed, and load forecasts and frozen weather after
   D are overwritten. The production fits had that whole history in memory. The refit must equal the committed
   D2 vector bit for bit, and each member must reach its recorded parameter hash. On the 448 evaluation origins
   it must also equal the committed `predictions.parquet` D2 rows.
2. **Paired positives.** At stratified origins in every fold, two changes move the comparison:
   (a) a non-uniform mutation of D−1's evening prices, the latest outcome legitimately available;
   (b) a planted one-day leak, the same member trained on a window that includes D, which the same
   destroy-and-compare test detects.
3. **The search and the gate, fold by fold.** For every fold, not only fold 2 and fold 4: the rank-1 trial on
   the batch nearest the cutoff is unchanged when every outcome on or after D0_f − 56 is mutated, and changes
   when a validation-batch outcome is. Gate member 0 is unchanged when every outcome on or after D0_f is
   mutated, and an earlier gate day's outcome changes a later gate fit.
4. **Structure, exhaustively and without fits.** Every cached member window of the 636 origins is
   [max(2019-01-01, D − 728), D) with its held-out weeks ending before D − 7 and the frozen configuration and
   seed. Every batch of every search trial ends before D0_f − 56 and meets no warm-up or evaluation day. Every
   gate day lies in [D0_f − 56, D0_f) and is no warm-up or evaluation day.

**The frozen code.** `check_protocol` refuses at the continuation's candidate. The Owner's work-availability
maintenance (`9667fb4`) changed two files in attempt 1's `implementation_sha256`, `scripts/cp24_ddnn2.py` and
`src/cp24/budget.py`, and removed only the calendar gate. `frozen_guard` applies `check_protocol`'s conditions
with that one recorded exception: each of the two files must equal its blob at `9667fb4`, and its blob at the
protocol commit must equal the frozen hash. Every other implementation file and frozen input must be unchanged.
The bit-for-bit reproduction in check 1 is the direct proof that the forecast path is the frozen one.

Run under the monitor only:

    python scripts/cp24_ddnn2.py monitor --name leakage-a1 --workers 4 --log <log> -- \
        <python> -m cp24.leakage --attempt 1 --workers 4

It writes `reports/ddnn2/attempt-<k>/leakage-controls.json` and `leakage-by-origin.csv`. Per-task results
are cached atomically under `.local/artifacts/cp-24/leakage/`, so a restart resumes without refitting.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

from cp15.data import LABELS, prepare, sha
from . import design as G
from .budget import atomic, charge_fits, ledger
from .execution import D2Cache, fit_identity, fit_origin
from .inputs import load, origin_manifest, weather_design
from .jobs import art, stamp
from .member import Keys, fit_member
from .preflight import fold_table
from .protocol import attempt_dir
from .reference import require_passing_record
from .search import score_batch

THRESHOLD = 1e-6
SEED = 24010
WX = ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean']
#: The Owner's work-availability maintenance commit (2026-10-10) and the two frozen-implementation files it
#: changed (calendar gate removed; docs/track-b/evidence/cp-24/work-availability-2026-10-10.md).
MAINTENANCE_COMMIT = '9667fb4698076fe942aae2246e44366e91efd388'
MAINTAINED = ('scripts/cp24_ddnn2.py', 'src/cp24/budget.py')
POSITIVES_PER_FOLD = 2


def _git(root: Path, *args) -> str:
    return subprocess.check_output(['git', *args], cwd=root, text=True).strip()


def _blob_sha256(root: Path, rev: str, name: str) -> str:
    return hashlib.sha256(subprocess.check_output(['git', 'show', f'{rev}:{name}'], cwd=root)).hexdigest()


def frozen_guard(root: Path, k: int) -> dict:
    """`check_protocol`'s conditions, with the recorded maintenance exception for exactly `MAINTAINED`."""
    root = Path(root)
    path = attempt_dir(root, k) / 'protocol.json'
    p = json.loads(path.read_text())
    rel = str(path.relative_to(root))
    if subprocess.check_output(['git', 'show', f'HEAD:{rel}'], cwd=root) != path.read_bytes():
        raise ValueError(f'attempt {k}\'s protocol is not committed at HEAD')
    commit = _git(root, 'log', '--reverse', '--format=%H', '--', rel).split()[0]
    if subprocess.run(['git', 'merge-base', '--is-ancestor', commit, 'HEAD'], cwd=root).returncode != 0:
        raise ValueError('the protocol commit is not an ancestor of HEAD')
    if subprocess.run(['git', 'merge-base', '--is-ancestor', MAINTENANCE_COMMIT, 'HEAD'], cwd=root).returncode != 0:
        raise ValueError('the maintenance commit is not an ancestor of HEAD')
    exceptions = {}
    for name, value in p['implementation_sha256'].items():
        live = sha(root / name)
        if name in MAINTAINED:
            at_freeze, at_maintenance = _blob_sha256(root, commit, name), _blob_sha256(root, MAINTENANCE_COMMIT, name)
            if at_freeze != value or live != at_maintenance:
                raise ValueError(f'{name}: neither the frozen bytes nor the recorded maintenance bytes')
            exceptions[name] = {'frozen_sha256': value, 'maintenance_sha256': live, 'maintenance_commit': MAINTENANCE_COMMIT}
        elif live != value:
            raise ValueError(f'implementation changed since attempt {k}\'s freeze: {name}')
    for name, value in p['frozen_inputs_sha256'].items():
        if sha(root / name) != value:
            raise ValueError(f'frozen input changed since attempt {k}\'s freeze: {name}')
    require_passing_record(root)
    return {'protocol_commit': commit, 'protocol_sha256': sha(path), 'head': _git(root, 'rev-parse', 'HEAD'),
            'implementation_files': len(p['implementation_sha256']), 'frozen_inputs': len(p['frozen_inputs_sha256']),
            'maintenance_exceptions': exceptions}


def destroy_future(frame: pd.DataFrame, table: pd.DataFrame, day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every outcome dated on or after `day` destroyed: prices on and after D removed; load forecasts and frozen
    weather after D overwritten. D's own load forecast and weather are forecasts issued before the gate."""
    f = frame.copy()
    f.loc[f.delivery_date >= day, 'price_eur_mwh'] = np.nan
    f.loc[f.delivery_date > day, 'load_forecast_mw'] = 1e8
    t = table.copy()
    t.loc[t.index.get_level_values(0) > day, WX] = 1e6
    return f, t


def _variant(data, design, frame=None, table=None):
    d = data if frame is None else prepare(frame, data.p, data.spec)
    w = design if table is None else replace(design, table=table)
    dd = G.build(d, w)
    return d, dd, Keys.from_data(d, dd)


# ------------------------------------------------------------------ workers
_w: dict = {}


def _init(root: str, ledger_path: str, k: int):
    os.environ['CP24_LEDGER'] = ledger_path
    root = Path(root)
    frozen_guard(root, k)
    _w.update(root=root, k=k, identity=fit_identity(root, k), design=weather_design(root), budget=ledger(), data={},
              protocol=json.loads((attempt_dir(root, k) / 'protocol.json').read_text()))


def _production(first_s: str, stage: str):
    """The production inputs of the fits job: the full history for evaluation, data before the evaluation start
    for warm-up (`execution._materialise`)."""
    key = first_s if stage == 'warmup' else 'full'
    if key not in _w['data']:
        data = load(_w['root'], before=date.fromisoformat(first_s) if stage == 'warmup' else None)
        _w['data'] = {key: data}
    return _w['data'][key]


def _result_path(kind: str, fold: str, day_s: str) -> Path:
    return art() / 'leakage' / f'attempt-{_w["k"]}' / kind / fold / f'{day_s}.json'


def _blind(fold: str, day: date, data, members, cached: dict) -> dict:
    frame, table = destroy_future(data.frame, _w['design'].table, day)
    md, mdd, mk = _variant(data, _w['design'], frame, table)
    out = fit_origin(mdd, mk, md, day, members, charge_fits(_w['budget'], 'control'))
    q, c = np.asarray(cached['quantiles'], float), np.asarray(cached['central'], float)
    same_shape = out['quantiles'].shape == q.shape
    params = [m['record']['params_sha256'] for m in out['members']]
    return {'rows': int(q.shape[0]), 'quantiles_bitwise': bool(same_shape and np.array_equal(out['quantiles'], q)),
            'central_bitwise': bool(out['central'].shape == c.shape and np.array_equal(out['central'], c)),
            'max_abs_difference': float(np.max(np.abs(out['quantiles'] - q))) if same_shape else None,
            'members_params_bitwise': params == [m['record']['params_sha256'] for m in cached['members']],
            'members_best_epoch_equal': [m['record']['best_epoch'] for m in out['members']]
            == [m['record']['best_epoch'] for m in cached['members']],
            'timestamps_equal': out['timestamp_utc'] == cached['timestamp_utc'],
            'quantiles': out['quantiles'].tolist()}


def _d1(fold: str, day: date, data, members, cached: dict) -> dict:
    frame = data.frame.copy()
    local = pd.to_datetime(frame.timestamp_utc, utc=True).dt.tz_convert('Europe/Berlin').dt.hour
    rows = frame.delivery_date.eq(day - timedelta(days=1)) & local.between(17, 21)
    frame.loc[rows, 'price_eur_mwh'] += 300.0
    md, mdd, mk = _variant(data, _w['design'], frame)
    out = fit_origin(mdd, mk, md, day, members, charge_fits(_w['budget'], 'control'))
    q = np.asarray(cached['quantiles'], float)
    return {'rows_mutated': int(rows.sum()), 'max_abs_difference': float(np.max(np.abs(out['quantiles'] - q)))}


def _plant(fold: str, day: date, data, members, cached: dict) -> dict:
    """Member 0 trained on [.., D] -- a planted one-day leak -- with and without the future destroyed."""
    m0 = members[0]
    d0, dd0, _ = _variant(data, _w['design'])
    leaky = fit_member(dd0, day + timedelta(days=1), m0['config'], m0['seed'], np.array([dd0.ix(day)]),
                       exclude_uncovered=False, charge=charge_fits(_w['budget'], 'control'))
    frame, table = destroy_future(data.frame, _w['design'].table, day)
    md, mdd, _ = _variant(data, _w['design'], frame, table)
    blind = fit_member(mdd, day + timedelta(days=1), m0['config'], m0['seed'], np.array([mdd.ix(day)]),
                       exclude_uncovered=False, charge=charge_fits(_w['budget'], 'control'))
    trains_on_d = date.fromisoformat(leaky['window']['end_exclusive']) > day
    return {'member': m0['config']['id'], 'leaky_window_end_exclusive': leaky['window']['end_exclusive'],
            'leaky_window_contains_d': trains_on_d,
            'leak_detected_max_abs_difference': float(np.max(np.abs(leaky['eur'] - blind['eur'])))}


KINDS = {'blind': _blind, 'd1': _d1, 'plant': _plant}


def _task(task):
    kind, fold, day_s, first_s, stage = task
    path = _result_path(kind, fold, day_s)
    if path.exists():
        rec = json.loads(path.read_text())
        rec['resumed'] = True
        return rec
    data = _production(first_s, stage)
    day = date.fromisoformat(day_s)
    cached = D2Cache(_w['k'], fold, data, _w['identity']).load(day)
    if cached is None:
        raise ValueError(f'no committed-run cache entry for {fold} {day_s}')
    members = _w['protocol']['folds'][fold]['ensemble']
    began = time.time()
    try:
        rec = KINDS[kind](fold, day, data, members, cached)
        rec['error'] = None
    except Exception as exc:  # recorded and counted as a failure, never substituted
        rec = {'error': repr(exc), 'traceback': traceback.format_exc()}
    rec.update(kind=kind, fold=fold, day=day_s, stage=stage, seconds=time.time() - began, written_utc=stamp())
    atomic(path, rec)
    return rec


# ------------------------------------------------------------------ fold-by-fold search and gate
def search_and_gate_by_fold(root: Path, k: int, budget) -> dict:
    p = json.loads((attempt_dir(root, k) / 'protocol.json').read_text())
    r = p['round']
    design = weather_design(root)
    full = load(root)
    ledger_doc = json.loads((root / 'reports/ddnn2/rounds' / f'round-{r}' / 'search-ledger.json').read_text())
    ensembles = json.loads((root / 'reports/ddnn2/rounds' / f'round-{r}' / 'ensembles.json').read_text())['folds']
    out = {}
    for f in fold_table(root):
        fold = f['fold']
        fl = ledger_doc['folds'][fold]
        t_id = fl['ranking_all_batches'][0]
        trial = next(t for t in fl['trials'] if t['config']['id'] == t_id)
        rec0 = json.loads((art() / 'rounds' / f'round-{r}' / 'search' / fold / f't{trial["trial"]:03d}-b00.json').read_text())
        b0 = date.fromisoformat(rec0['batch_start'])

        def batch_fit(frame):
            data = prepare(frame, full.p, full.spec)
            dd = G.build(data, design)
            keys = Keys.from_data(data, dd)
            days = np.array([dd.ix(b0 + timedelta(days=j)) for j in range(28)])
            fit = fit_member(dd, b0, trial['config'], rec0['seed'], days, exclude_uncovered=True,
                             charge=charge_fits(budget, 'control'))
            return score_batch(keys, days, fit['eur'])['pinball_sum'], fit['member']['params_sha256']

        later = full.frame.copy()
        later.loc[later.delivery_date >= f['search_cutoff'], 'price_eur_mwh'] += 400.0
        later.loc[later.delivery_date >= f['search_cutoff'], 'load_forecast_mw'] *= 1.5
        s_later, params_later = batch_fit(later)
        vb = full.frame.copy()
        vb.loc[vb.delivery_date.eq(b0 + timedelta(days=10)), 'price_eur_mwh'] += 90.0
        s_vb, _ = batch_fit(vb)
        search = {'trial': t_id, 'batch_start': str(b0), 'batch_end': rec0['batch_end'], 'cutoff': str(f['search_cutoff']),
                  'pinball_sum_ledger': rec0['pinball_sum'], 'pinball_sum_with_outcomes_from_cutoff_mutated': s_later,
                  'negative_identical': s_later == rec0['pinball_sum'] and params_later == rec0['member']['params_sha256'],
                  'pinball_sum_with_a_validation_day_mutated': s_vb, 'positive_changed': s_vb != rec0['pinball_sum']}
        member = ensembles[fold]['members'][0]
        g0 = f['gate'][0]
        g2 = g0 + timedelta(days=20)
        stored = json.loads((art() / 'rounds' / f'round-{r}' / 'gate' / fold / str(g0) / 'm0.json').read_text())
        stored2 = json.loads((art() / 'rounds' / f'round-{r}' / 'gate' / fold / str(g2) / 'm0.json').read_text())

        def gate_fit(frame, day):
            data = prepare(frame, full.p, full.spec)
            dd = G.build(data, design)
            return fit_member(dd, day, member['config'], member['seed'], np.array([dd.ix(day)]), exclude_uncovered=True,
                              charge=charge_fits(budget, 'control'))

        glater = full.frame.copy()
        glater.loc[glater.delivery_date >= f['d0'], 'price_eur_mwh'] -= 300.0
        glater.loc[glater.delivery_date >= f['d0'], 'load_forecast_mw'] *= 0.5
        gl = gate_fit(glater, g0)
        gmut = full.frame.copy()
        gmut.loc[gmut.delivery_date.eq(g0 + timedelta(days=5)), 'price_eur_mwh'] += 200.0
        gm = gate_fit(gmut, g2)
        gate = {'day': str(g0), 'd0': str(f['d0']),
                'negative_identical': bool(np.array_equal(gl['eur'][0], np.asarray(stored['eur']))),
                'negative_max_abs_difference': float(np.max(np.abs(gl['eur'][0] - np.asarray(stored['eur'])))),
                'positive_mutated_day': str(g0 + timedelta(days=5)), 'positive_fit_day': str(g2),
                'positive_max_abs_difference': float(np.max(np.abs(gm['eur'][0] - np.asarray(stored2['eur']))))}
        out[fold] = {'search': search, 'gate': gate}
    return out


# ------------------------------------------------------------------ structure, without fits
def structure(root: Path, k: int) -> dict:
    p = json.loads((attempt_dir(root, k) / 'protocol.json').read_text())
    m = origin_manifest(root)
    folds = {f['fold']: f for f in fold_table(root)}
    scored = set()
    for f in m['folds']:
        for o in f['origins']:
            scored.add(date.fromisoformat(o['day']))
    out = {'members_checked': 0, 'entries_checked': 0, 'window_problems': []}
    for fold, f in folds.items():
        frozen = [(x['config']['id'], x['seed']) for x in p['folds'][fold]['ensemble']]
        for path in sorted((art() / f'attempt-{k}' / 'fits' / fold).glob('*.json')):
            item = json.loads(path.read_text())
            day = date.fromisoformat(item['day'])
            out['entries_checked'] += 1
            if [(x['config'], x['seed']) for x in item['members']] != frozen:
                out['window_problems'].append(f'{fold} {day}: members differ from the frozen ensemble')
            for x in item['members']:
                w = x['window']
                out['members_checked'] += 1
                weeks = [date.fromisoformat(s) for s in w['held_out_weeks']]
                ok = (w['end_exclusive'] == str(day) and w['start'] == str(max(G.FLOOR, day - timedelta(days=G.WINDOW_DAYS)))
                      and all(wk + timedelta(days=6) < day - timedelta(days=G.RECENT_EXCLUDED_DAYS) for wk in weeks)
                      and w['n_train_days'] + w['n_held_out_days'] == w['n_days'] and w['excluded_uncovered_days'] == 0)
                if not ok:
                    out['window_problems'].append(f'{fold} {day} member {x["member"]}')
    ledger_doc = json.loads((root / 'reports/ddnn2/rounds' / f'round-{p["round"]}' / 'search-ledger.json').read_text())
    batches, batch_problems = 0, []
    for fold, f in folds.items():
        for t in ledger_doc['folds'][fold]['trials']:
            for b in t['batches']:
                start = date.fromisoformat(b['start'])
                end = start + timedelta(days=27)
                batches += 1
                days = {start + timedelta(days=j) for j in range(28)}
                if not end < f['search_cutoff'] or days & scored:
                    batch_problems.append(f'{fold} trial {t["trial"]} batch {b["batch"]}')
    gate_problems = []
    for fold, f in folds.items():
        g0, g1 = f['gate']
        days = {g0 + timedelta(days=j) for j in range((g1 - g0).days + 1)}
        if len(days) != 56 or g0 != f['d0'] - timedelta(days=56) or g1 != f['d0'] - timedelta(days=1) or days & scored:
            gate_problems.append(fold)
    out.update(search_batches_checked=batches, search_batch_problems=batch_problems, gate_problems=gate_problems)
    return out


# ------------------------------------------------------------------ the job
def tasks(root: Path) -> tuple[list[tuple], list[tuple]]:
    m = origin_manifest(root)
    full = load(root)
    blind, positives = [], []
    rng = np.random.default_rng(SEED)
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        early = load(root, before=first)
        evaluation = []
        for o in f['origins']:
            d = date.fromisoformat(o['day'])
            stage = 'warmup' if d < first else 'evaluation'
            data = early if stage == 'warmup' else full
            if not len(data.rows(d)):
                continue
            blind.append(('blind', f['fold'], o['day'], str(first), stage))
            if stage == 'evaluation':
                evaluation.append(o['day'])
        picks = [evaluation[0], evaluation[-1]]
        dst = [d for d in evaluation if len(full.rows(date.fromisoformat(d))) in (23, 25, 46, 50, 92, 100)]
        picks += dst[:1]
        rest = [d for d in evaluation if d not in picks]
        picks += [rest[i] for i in sorted(rng.choice(len(rest), size=POSITIVES_PER_FOLD, replace=False))]
        for d in picks:
            positives.append(('d1', f['fold'], d, str(first), 'evaluation'))
        positives.append(('plant', f['fold'], picks[0], str(first), 'evaluation'))
    return blind, positives


def committed_d2_equal(root: Path, k: int, results: list[dict]) -> dict:
    """The refits against the committed `predictions.parquet` D2 rows, on the evaluation origins."""
    pred = pd.read_parquet(attempt_dir(root, k) / 'predictions.parquet', filters=[('policy', '==', 'D2')])
    pred['day'] = pd.to_datetime(pred.delivery_date).dt.date.astype(str)
    full = load(root)
    checked, equal, keys = 0, 0, 0
    by = {(r['fold'], r['day']): r for r in results if r['kind'] == 'blind' and r['stage'] == 'evaluation' and not r['error']}
    for (fold, day), part in pred.groupby(['fold', 'day']):
        r = by.get((fold, day))
        checked += 1
        if r is None:
            continue
        rows = full.rows(date.fromisoformat(day))
        stamp_ix = {str(t): i for i, t in enumerate(full.index[rows])}
        q = np.asarray(r['quantiles'], float)
        idx = [stamp_ix[str(pd.Timestamp(t))] for t in pd.to_datetime(part.timestamp_utc, utc=True)]
        same = np.array_equal(q[idx], part[list(LABELS)].to_numpy(float))
        equal += int(same)
        keys += len(part)
    return {'evaluation_days_in_predictions': checked, 'days_equal_bitwise': equal, 'keys_compared': keys}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True, choices=(1, 2))
    ap.add_argument('--workers', type=int, default=int(os.environ.get('CP24_WORKERS', '1')))
    ap.add_argument('--limit', type=int, default=None, help='smoke run: only the first N blind origins, no outputs')
    args = ap.parse_args(argv)
    if 'CP24_JOB_INDEX' not in os.environ:
        raise SystemExit('the leakage controls run only under the monitor (resource accounting, §23.11)')
    if args.workers > int(os.environ.get('CP24_WORKERS', '1')):
        raise SystemExit('more pool workers than the monitor declared')
    root, k = Path(__file__).resolve().parents[2], args.attempt
    guard = frozen_guard(root, k)
    budget = ledger()
    blind, positives = tasks(root)
    work = blind[:args.limit] if args.limit else blind + positives
    results = []
    with mp.get_context('spawn').Pool(args.workers, initializer=_init,
                                      initargs=(str(root), os.environ['CP24_LEDGER'], k)) as pool:
        for i, rec in enumerate(pool.imap_unordered(_task, work)):
            results.append(rec)
            if i % 25 == 0 or rec.get('error'):
                print(json.dumps({k_: v for k_, v in rec.items() if k_ not in ('quantiles', 'traceback')}, default=str)[:600],
                      flush=True)
    if args.limit:
        print(json.dumps({'smoke': len(results), 'bitwise': sum(bool(r.get('quantiles_bitwise')) for r in results)}))
        return 0
    sg = search_and_gate_by_fold(root, k, budget)
    st = structure(root, k)
    blinds = [r for r in results if r['kind'] == 'blind']
    rows = pd.DataFrame([{x: r.get(x) for x in ('fold', 'day', 'stage', 'rows', 'quantiles_bitwise', 'central_bitwise',
                                                 'members_params_bitwise', 'members_best_epoch_equal', 'timestamps_equal',
                                                 'max_abs_difference', 'error')} for r in blinds]).sort_values(['fold', 'day'])
    committed = committed_d2_equal(root, k, blinds)
    d1 = sorted((r for r in results if r['kind'] == 'd1'), key=lambda r: (r['fold'], r['day']))
    plant = sorted((r for r in results if r['kind'] == 'plant'), key=lambda r: (r['fold'], r['day']))
    ok_blind = [r for r in blinds if not r['error'] and r['quantiles_bitwise'] and r['central_bitwise']
                and r['members_params_bitwise'] and r['members_best_epoch_equal'] and r['timestamps_equal']]
    checks = {
        'every_origin_refitted': len(blinds) == 636 and not any(r['error'] for r in blinds),
        'every_origin_bitwise_with_the_future_destroyed': len(ok_blind) == len(blinds) == 636,
        'evaluation_origins_equal_committed_predictions': committed['days_equal_bitwise'] == committed['evaluation_days_in_predictions'] == 448
        and committed['keys_compared'] == 10747,
        'd1_positive_moves_every_pick': bool(d1) and all(not r['error'] and r['max_abs_difference'] > THRESHOLD for r in d1),
        'planted_leak_detected_in_every_fold': len(plant) == 5 and all(
            not r['error'] and r['leaky_window_contains_d'] and r['leak_detected_max_abs_difference'] > THRESHOLD for r in plant),
        'search_negative_every_fold': all(v['search']['negative_identical'] for v in sg.values()) and len(sg) == 5,
        'search_positive_every_fold': all(v['search']['positive_changed'] for v in sg.values()),
        'gate_negative_every_fold': all(v['gate']['negative_identical'] for v in sg.values()),
        'gate_positive_every_fold': all(v['gate']['positive_max_abs_difference'] > THRESHOLD for v in sg.values()),
        'member_windows_exhaustive': st['entries_checked'] == 636 and st['members_checked'] == 5088 and not st['window_problems'],
        'search_batches_exhaustive': st['search_batches_checked'] > 0 and not st['search_batch_problems'],
        'gate_days_exhaustive': not st['gate_problems'],
    }
    by_stage = rows.groupby('stage').agg(origins=('day', 'size'), bitwise=('quantiles_bitwise', 'sum')).to_dict('index')
    out = {'schema': 'cp24-leakage-controls-v1', 'attempt': k, 'threshold_eur_mwh': THRESHOLD, 'seed': SEED,
           'verification_only': 'no frozen element changed, no forecast produced or replaced, nothing rescored',
           'frozen_guard': guard,
           'destroyed_from_d': {'price_eur_mwh': 'removed on and after D', 'load_forecast_mw': 'set to 1e8 after D',
                                'weather': f'{WX} set to 1e6 after D'},
           'blind': {'origins': len(blinds), 'bitwise': len(ok_blind), 'by_stage': {s: {x: int(v) for x, v in d.items()}
                                                                                  for s, d in by_stage.items()},
                     'max_abs_difference': float(rows.max_abs_difference.max()), 'committed_predictions': committed},
           'positives': {'d1_evening_prices': [{x: r.get(x) for x in ('fold', 'day', 'rows_mutated', 'max_abs_difference', 'error')}
                                               for r in d1],
                         'planted_one_day_leak': [{x: r.get(x) for x in ('fold', 'day', 'member', 'leaky_window_end_exclusive',
                                                                         'leaky_window_contains_d',
                                                                         'leak_detected_max_abs_difference', 'error')}
                                                  for r in plant]},
           'search_and_gate_by_fold': sg, 'structure': st, 'checks': checks, 'all_passed': all(checks.values()),
           'written_utc': stamp()}
    ad = attempt_dir(root, k)
    atomic(ad / 'leakage-controls.json', out)
    (ad / 'leakage-by-origin.csv').write_text(rows.to_csv(index=False, float_format='%.17g', lineterminator='\n'))
    print(json.dumps({'all_passed': out['all_passed'], 'checks': checks}, default=str), flush=True)
    return 0 if out['all_passed'] else 7


if __name__ == '__main__':
    raise SystemExit(main())
