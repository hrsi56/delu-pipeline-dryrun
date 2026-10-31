"""Scored attempt k: the DDNN-2 fits at the warm-up and evaluation origins, the training-only admission
replay and the comparison replay (capstone v21-r11 §23.5, §23.6, §23.8).

**Policies.**

| ID | central forecast | intervals |
|---|---|---|
| v5 | `A1_w/3 + B2_w/3 + L-N/12 + L-R/12 + D2/6` = (2/3)·c_HG + (1/6)·L + (1/6)·D2, float64, left to right | HG's H layer on v5's own errors |
| v3+D2 | `A1_w/3 + B2_w/3 + D2/3` (CP-22's composite form) | HG's H layer on its own errors |
| D2 | DDNN-2's ensemble median | its own Johnson SU quantiles (per-level median); its p50 is the central forecast |

**One H layer.** v5 and v3+D2 run through CP-16's frozen `cp16.residuals.SharedResidualState` (V2-H),
exactly as CP-21's to CP-23's replays did: one state per policy, the central passed as both blend
inputs (`c/2 + c/2 == c`, asserted). Residuals are never shared across policies.

**Sources.** HG's A1_w/B2_w come only from CP-20's identity-verified cache; v4's L-N and L-R from
CP-21's retained fit cache under CP-21's recomputed identity, each equal to the `central_sha256` of
CP-21's committed lineage (`cp22.execution.Sources`, reused unchanged). D2 comes from this attempt's
own fits. The replay never fits.

**Pre-registration.** Every entry point calls `cp24.protocol.check_protocol(k)`: attempt k's frozen
protocol is committed at HEAD, unchanged, and its commit is an ancestor of HEAD; the first fit job
records the protocol commit in the ledger before any warm-up or evaluation forecast exists.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import pandas as pd

from cp15.data import LABELS, array_hash, origin_utc, sha
from cp16.residuals import SharedResidualState
from cp21.execution import digest
from cp22.execution import Sources as CP22Sources, cp21_lineage_hashes
from cp22.pn import composite
from . import ddnn2 as M
from . import design as G
from .budget import atomic, charge_fits, ledger
from .inputs import cp21_fit_identity, hg_identity, input_fingerprint, load, origin_manifest, weather_design
from .jobs import art, stamp
from .member import Keys, emit_rows, fit_member, maxrss
from .protocol import attempt_dir, check_protocol
from .reference import require_passing_record

OUT = Path('reports/ddnn2')
EVIDENCE_CLASS = 'development_post_selection'
NEW_POLICIES = ('v5', 'v3+D2', 'D2')
H_POLICIES = ('v5', 'v3+D2')
LAYER = {'v5': 'H', 'v3+D2': 'H', 'D2': 'JSU', 'HG': 'H', 'HGL': 'H'}
COMPOSITE_TOLERANCE = 1e-9
#: Five evaluation origins per fold at which every policy's pre-release state is persisted for the
#: cold daily-cycle diagnostic (offsets into the evaluation window; the next day with eligible hours).
CYCLE_OFFSETS = (0, 22, 44, 66, 88)


def committed_at_head(root: Path, name: str) -> None:
    committed = subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=root)
    if committed != (Path(root) / name).read_bytes():
        raise ValueError(f'{name} must be committed at HEAD')


# ------------------------------------------------------------------ composites
def v5_central(a1, b2, ln, lr, d2) -> np.ndarray:
    """v5 = (2/3)·c_HG + (1/6)·L + (1/6)·D2, evaluated left to right as A1/3 + B2/3 + L-N/12 + L-R/12 + D2/6."""
    arrays = [np.asarray(v, float) for v in (a1, b2, ln, lr, d2)]
    if len({a.shape for a in arrays}) != 1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('v5 needs five finite vectors of one shape')
    a1, b2, ln, lr, d2 = arrays
    return a1 / 3 + b2 / 3 + ln / 12 + lr / 12 + d2 / 6


def v3d2_central(a1, b2, d2) -> np.ndarray:
    """v3+D2 = (2/3)·c_HG + (1/3)·D2, CP-22's composite form A1/3 + B2/3 + D2/3."""
    return composite(a1, b2, ((d2, 3),))


def composite_gap(policy: str, central: np.ndarray, m: dict) -> float:
    """§23.10's composite parity: |c_v5 - c_v4 - (1/6)(D2 - L)| for v5, |c - (2/3)c_HG - (1/3)D2| for v3+D2."""
    if policy == 'v5':
        L = m['L-N'] / 2 + m['L-R'] / 2
        return float(np.max(np.abs(central - m['HGL'] - (m['D2'] - L) / 6)))
    return float(np.max(np.abs(central - (2 / 3) * m['HG'] - m['D2'] / 3)))


# ------------------------------------------------------------------ the attempt's DDNN-2 cache
def fit_identity(root: Path, k: int) -> dict:
    return {'cp24_input_fingerprint': input_fingerprint(root), 'attempt': k,
            'protocol_sha256': sha(attempt_dir(root, k) / 'protocol.json'), 'weather_design_sha256': weather_design(root).sha256}


class D2Cache:
    def __init__(self, k: int, fold: str, data, identity: dict, base: Path | None = None):
        self.k, self.fold, self.data, self.identity = k, fold, data, identity
        self.dir = (base or art() / f'attempt-{k}' / 'fits') / fold

    def path(self, day: date) -> Path:
        return self.dir / f'{day}.json'

    def load(self, day: date):
        path = self.path(day)
        if not path.exists():
            return None
        item = json.loads(path.read_text())
        rows = self.data.rows(day)
        if item.get('content_sha256') != digest(item) or item['day'] != str(day) or item['fold'] != self.fold \
                or item['identity'] != self.identity or item['origin_utc'] != str(origin_utc(day).tz_convert('UTC')) \
                or item['timestamp_utc'] != list(map(str, self.data.index[rows])):
            raise ValueError(f'stale or wrong CP-24 DDNN-2 cache {self.fold} {day}')
        return item

    def get(self, day: date) -> dict:
        item = self.load(day)
        if item is None:
            raise ValueError(f'CP-24 DDNN-2 cache miss {self.fold} {day}: fit in the fits job first')
        n = len(self.data.rows(day))
        central = np.asarray(item['central'], float)
        q = np.asarray(item['quantiles'], float)
        if central.shape != (n,) or q.shape != (n, 7) or not (np.isfinite(central).all() and np.isfinite(q).all()):
            raise ValueError('invalid cached DDNN-2 vector')
        if array_hash(central) != item['central_sha256'] or array_hash(q) != item['quantiles_sha256']:
            raise ValueError('DDNN-2 vector differs from its recorded hash')
        if (np.diff(q, axis=1) < 0).any() or not np.array_equal(central, q[:, M.MEDIAN]):
            raise ValueError('cached DDNN-2 quantiles crossed, or the p50 is not the central forecast')
        return {'central': central, 'quantiles': q}

    def save(self, item: dict):
        atomic(self.path(date.fromisoformat(item['day'])), item)


def fit_origin(dd: G.DayData, keys: Keys, data, day: date, members: list[dict], charge, *, uncovered_fails: bool = True) -> dict:
    """The eight frozen members fitted fresh at `day`, and their per-level median ensemble on the day's keys."""
    di = np.array([dd.ix(day)])
    rows_k = keys.of_days(di)
    rows = data.rows(day)
    if not np.array_equal(keys.timestamp[rows_k], data.index[rows]):
        raise ValueError(f'{day}: forecast keys differ between the day table and the inputs')
    eurs, recs = [], []
    for j, m in enumerate(members):
        fit = fit_member(dd, day, m['config'], m['seed'], di, exclude_uncovered=False, charge=charge)
        if uncovered_fails and any(not dd.covered[i] for i in range(dd.ix(date.fromisoformat(fit['window']['start'])),
                                                                     dd.ix(day))
                                   if dd.mask[i].any()):
            raise ValueError(f'{day}: a warm-up or evaluation window day lacks a frozen weather record')
        q = emit_rows(keys, rows_k, di, fit['eur'])
        eurs.append(q)
        recs.append({'member': j, 'rank': m['rank'], 'config': m['config']['id'], 'seed': m['seed'],
                     'record': {k: v for k, v in fit['member'].items() if k not in ('stopping_metric_history', 'train_loss_history')},
                     'window': {k: (v if k != 'excluded_uncovered_days' else len(v)) for k, v in fit['window'].items()},
                     'guards': {k: v for k, v in fit['guards'].items() if k != 'cap_forecast_by_day'},
                     'cap_z': fit['cap_z'], 'transform': m['config']['transform'],
                     'centre': float(fit['centre'][0]), 'scale': float(fit['scale'][0]),
                     'jsu': fit['jsu'][0].tolist(), 'active': fit['active'][0].astype(int).tolist(),
                     'quantiles': q.tolist(), 'seconds': fit['seconds'], 'n_inputs': fit['n_inputs'],
                     'preprocessor': fit['preprocessor']})
    stack = np.stack(eurs)                                   # members x rows x 7
    med, crossed = M.ensemble_median(stack[:, :, None, :])   # rows x 1 x 7
    q = med[:, 0, :]
    if not np.isfinite(q).all():
        raise M.TrainingFailure('nonfinite ensemble quantile')
    return {'quantiles': q, 'central': q[:, M.MEDIAN].copy(), 'crossed_rows': crossed, 'members': recs,
            'timestamp_utc': list(map(str, data.index[rows]))}


# ------------------------------------------------------------------ workers
_worker: dict = {}


def _init_worker(root: str, ledger_path: str, k: int):
    os.environ['CP24_LEDGER'] = ledger_path
    root = Path(root)
    check_protocol(root, k)
    require_passing_record(root)
    _worker.update(root=root, k=k, identity=fit_identity(root, k), design=weather_design(root), budget=ledger(), data={},
                   protocol=json.loads((attempt_dir(root, k) / 'protocol.json').read_text()))


def _materialise(first_s: str, stage: str):
    w = _worker
    key = first_s if stage == 'warmup' else 'full'
    if key not in w['data']:
        data = load(w['root'], before=date.fromisoformat(first_s) if stage == 'warmup' else None)
        dd = G.build(data, w['design'])
        w['data'] = {key: (data, dd, Keys.from_data(data, dd))}
    return w['data'][key]


def _fit_task(task):
    fold, day_s, first_s, stage = task
    w = _worker
    data, dd, keys = _materialise(first_s, stage)
    cache = D2Cache(w['k'], fold, data, w['identity'])
    day = date.fromisoformat(day_s)
    try:
        if cache.load(day) is not None:
            return fold, day_s, 'reused', 0.0, None
    except ValueError:
        pass  # a stale or wrong entry is refitted (and charged), never reused
    began = time.time()
    members = w['protocol']['folds'][fold]['ensemble']
    try:
        out = fit_origin(dd, keys, data, day, members, charge_fits(w['budget'], stage))
        w['budget'].reserve(ddnn2_epochs=sum(m['record']['epochs_run'] for m in out['members']))
    except Exception as exc:  # recorded, never substituted (§14.2 failure rule)
        atomic(art() / f'attempt-{w["k"]}' / 'failures' / f'{fold}_{day_s}.json',
               {'fold': fold, 'day': day_s, 'stage': stage, 'error': repr(exc), 'traceback': traceback.format_exc(),
                'written_utc': stamp()})
        return fold, day_s, 'failed', time.time() - began, repr(exc)
    item = {'day': day_s, 'fold': fold, 'stage': stage, 'origin_utc': str(origin_utc(day).tz_convert('UTC')),
            'timestamp_utc': out['timestamp_utc'], 'identity': w['identity'],
            'central': out['central'].tolist(), 'central_sha256': array_hash(out['central']),
            'quantiles': out['quantiles'].tolist(), 'quantiles_sha256': array_hash(out['quantiles']),
            'crossed_rows': out['crossed_rows'], 'members': out['members'], 'seconds': time.time() - began,
            'maxrss_bytes': maxrss(), 'written_utc': stamp(), 'source': 'fresh_cp24_attempt_fit'}
    item['content_sha256'] = digest(item)
    cache.save(item)
    return fold, day_s, 'fitted', time.time() - began, None


def _pool(root: Path, workers: int, k: int):
    if workers > int(os.environ.get('CP24_WORKERS', '1')):
        raise ValueError('more pool workers than the monitor declared')
    return mp.get_context('spawn').Pool(workers, initializer=_init_worker, initargs=(str(root), os.environ['CP24_LEDGER'], k))


def stage_tasks(root: Path, stage: str) -> list[tuple]:
    m = origin_manifest(root)
    full = load(root)
    tasks = []
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        early = load(root, before=first) if stage == 'warmup' else None
        for o in f['origins']:
            d = date.fromisoformat(o['day'])
            if (stage == 'warmup') != (d < first):
                continue
            data = early if stage == 'warmup' else full
            if not len(data.rows(d)):
                continue
            tasks.append((f['fold'], o['day'], str(first), stage))
    return tasks


def job_fits(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True, choices=(1, 2))
    ap.add_argument('--stage', choices=('warmup', 'evaluation'), required=True)
    ap.add_argument('--workers', type=int, default=int(os.environ.get('CP24_WORKERS', '1')))
    args = ap.parse_args(rest)
    root, k = Path(root), args.attempt
    p = check_protocol(root, k)
    require_passing_record(root)
    budget = ledger()
    marker = art() / f'attempt-{k}' / 'started.json'
    if not marker.exists():
        state = budget.read()
        if k == 2 and not state['counts'].get('scored_attempts'):
            raise ValueError('attempt 2 requires attempt 1')
        budget.reserve(scored_attempts=1)
        budget.event(f'attempt_{k}_first_fit', protocol_commit=p['_protocol_commit'],
                     protocol_sha256=sha(attempt_dir(root, k) / 'protocol.json'))
        atomic(marker, {'attempt': k, 'protocol_commit': p['_protocol_commit'], 'written_utc': stamp()})
    if args.stage == 'evaluation':
        lineage = json.loads((attempt_dir(root, k) / 'lineage.json').read_text())
        if lineage['execution_stage'] != 'training_only_admission_complete_frozen_before_outer_scoring':
            raise ValueError('evaluation fits require the committed training-only admission freeze')
        committed_at_head(root, str((attempt_dir(root, k) / 'lineage.json').relative_to(root)))
    tasks = stage_tasks(root, args.stage)
    budget.event('fits_start', attempt=k, stage=args.stage, tasks=len(tasks), workers=args.workers)
    done, failed, t0 = 0, 0, time.time()
    with _pool(root, args.workers, k) as pool:
        for fold, day, source, seconds, error in pool.imap_unordered(_fit_task, tasks, chunksize=1):
            done += 1
            failed += source == 'failed'
            if done % 25 == 0 or source == 'failed':
                print(args.stage, fold, day, source, round(seconds, 1), f'{done}/{len(tasks)}', f'{time.time() - t0:.0f}s',
                      error or '', flush=True)
    budget.event('fits_complete', attempt=k, stage=args.stage, tasks=len(tasks), failed=failed)
    print(json.dumps({'attempt': k, 'stage': args.stage, 'tasks': len(tasks), 'failed': failed, 'seconds': time.time() - t0}),
          flush=True)
    return 0 if not failed else 5


# ------------------------------------------------------------------ replay
class Sources(CP22Sources):
    """Central vectors per policy for one fold: CP-22's verified v3/v4 members plus this attempt's D2."""

    def __init__(self, k: int, fold: str, data, fit_ident: dict | None, cp21_ident: dict, hg_ident: dict, lineage_hashes: dict,
                 policies, *, base: Path | None = None):
        super().__init__(fold, data, None, cp21_ident, hg_ident, lineage_hashes, policies)
        self.d2 = D2Cache(k, fold, data, fit_ident, base) if fit_ident is not None else None

    def members(self, day: date) -> dict:
        out = super().members(day)
        if self.d2 is not None:
            item = self.d2.get(day)
            out['D2'], out['D2_quantiles'] = item['central'], item['quantiles']
        return out

    def central(self, policy: str, m: dict) -> np.ndarray:
        if policy == 'v5':
            return v5_central(m['A1'], m['B2'], m['L-N'], m['L-R'], m['D2'])
        if policy == 'v3+D2':
            return v3d2_central(m['A1'], m['B2'], m['D2'])
        if policy == 'D2':
            return m['D2']
        if policy in ('HG', 'HGL'):
            return super().central(policy, m)
        raise ValueError(f'unknown CP-24 policy {policy}')


def _twice(central: np.ndarray) -> np.ndarray:
    central = np.asarray(central, float)
    if not np.array_equal(central / 2 + central / 2, central):
        raise ValueError('single-central H-layer identity failed')
    return central


def output_frame(data, fold, day, rows, central, q, policy):
    item = pd.DataFrame({'fold': fold, 'policy': policy, 'timestamp_utc': data.index[rows], 'delivery_date': day,
                         'origin_utc': origin_utc(day).tz_convert('UTC'), 'y_true': data.y[rows],
                         'central': central, 'scale': data.scale[rows], 'level': data.level[rows],
                         'evidence_class': EVIDENCE_CLASS})
    for j, name in enumerate(LABELS):
        item[name] = q[:, j]
    return item


def members_frame(data, fold, day, rows, m, centers):
    item = pd.DataFrame({'fold': fold, 'timestamp_utc': data.index[rows], 'delivery_date': day})
    for name in ('D2', 'A1', 'B2', 'L-N', 'L-R', 'HG', 'HGL'):
        item[name] = m[name]
    item['L'] = m['L-N'] / 2 + m['L-R'] / 2
    item['c_v5'] = centers['v5']
    item['c_v3+D2'] = centers['v3+D2']
    return item


def save_state(state, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(state.to_dict(), sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def cycle_days(root: Path) -> set[tuple[str, str]]:
    full = load(root)
    out = set()
    for f in origin_manifest(root)['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        for offset in CYCLE_OFFSETS:
            d = first + timedelta(days=offset)
            while not len(full.rows(d)):
                d += timedelta(days=1)
            out.add((f['fold'], str(d)))
    return out


def replay(k, data, fold, dates, sources: Sources, states, truth, predict_days, lineage, phase, frames, budget, counter,
           *, snapshot: set | None = None, member_frames=None):
    """Advance every policy over the same dates with identical release, issue and support rules."""
    for d in dates:
        rows, centers, m, source = sources.get(d)
        if member_frames is not None and len(rows):
            member_frames.append(members_frame(data, fold, d, rows, m, centers))
        for policy in sources.policies:
            budget.reserve(policy_days=1, **{counter: 1})
            wanted = predict_days is None or d in predict_days
            meta, emitted = {}, False
            if LAYER[policy] == 'JSU':
                if len(rows) and wanted:
                    vector = m['D2_quantiles']
                    if not np.array_equal(centers['D2'], vector[:, M.MEDIAN]):
                        raise ValueError('D2: the emitted p50 is not the ensemble median')
                    emitted = True
                    meta = {'vector_sha256': array_hash(vector)}
                    if frames is not None:
                        frames.append(output_frame(data, fold, d, rows, centers[policy], vector, policy))
            else:
                state = states[policy]
                if snapshot is not None and (fold, str(d)) in snapshot:
                    save_state(state, art() / f'attempt-{k}' / 'states' / fold / str(d) / f'{policy}.json')
                state.release(d, truth)
                if len(rows) and wanted:
                    c = _twice(centers[policy])
                    q, meta = state.predict(d, data.index[rows], c, c, data.scale[rows])
                    vector = q['V2-H']
                    emitted = True
                    meta = dict(meta)
                    meta['vector_sha256'] = array_hash(vector)
                    meta['composite_gap'] = composite_gap(policy, centers[policy], m)
                    if meta['composite_gap'] > COMPOSITE_TOLERANCE:
                        raise ValueError(f'{policy} composite parity failed on {d}')
                    if frames is not None:
                        frames.append(output_frame(data, fold, d, rows, centers[policy], vector, policy))
                if len(rows):
                    c = _twice(centers[policy])
                    state.issue(d, data.index[rows], c, c, data.scale[rows])
            lineage['origins'].append({'policy': policy, 'fold': fold, 'day': str(d), 'phase': phase, 'source': source,
                                       'n_hours': int(len(rows)), 'predicted': bool(emitted), 'layer': LAYER[policy],
                                       'central_sha256': array_hash(centers[policy]) if len(rows) else None, **meta})


def _truth(data):
    series = pd.Series(data.y, index=data.index)
    return lambda ix: series.reindex(ix).to_numpy()


def _identities(root: Path, k: int):
    return fit_identity(root, k), cp21_fit_identity(root), hg_identity(root), cp21_lineage_hashes(root)


def new_states(policies):
    return {p: SharedResidualState() for p in policies if LAYER[p] == 'H'}


def job_admission(root: Path, rest) -> int:
    """Training-only admission (§14.2's slice): warm-up replay from each fold's genuine warm-up start,
    issuing on the seven admission dates, with data materialised before the fold's first evaluation day."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True, choices=(1, 2))
    args = ap.parse_args(rest)
    root, k = Path(root), args.attempt
    check_protocol(root, k)
    ad = attempt_dir(root, k)
    if (ad / 'lineage.json').exists():
        raise ValueError('existing admission evidence; no automatic retry or overwrite')
    m = origin_manifest(root)
    fit_ident, cp21_ident, hg_ident, hashes = _identities(root, k)
    budget = ledger()
    lineage = {'schema': 'cp24-lineage-v1', 'attempt': k, 'protocol_sha256': sha(ad / 'protocol.json'),
               'fit_identity': fit_ident, 'cp21_fit_identity': cp21_ident, 'hg_identity': hg_ident,
               'policies': list(NEW_POLICIES), 'origins': [], 'admission': [], 'states': {},
               'execution_stage': 'training_only_admission_in_progress'}
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        data = load(root, before=first)
        sources = Sources(k, f['fold'], data, fit_ident, cp21_ident, hg_ident, hashes, NEW_POLICIES)
        states = new_states(NEW_POLICIES)
        days = {date.fromisoformat(x['day']) for x in f['admission']}
        dates = list(pd.date_range(f['warmup_start'], first - timedelta(days=1)).date)
        before = len(lineage['origins'])
        replay(k, data, f['fold'], dates, sources, states, _truth(data), days, lineage, 'training_only', None, budget,
               'policy_days_admission')
        for rec in lineage['origins'][before:]:
            if rec['predicted']:
                lineage['admission'].append({x: rec.get(x) for x in ('policy', 'fold', 'day', 'n_hours', 'vector_sha256',
                                                                     'buffer_days', 'buffer_start', 'buffer_end', 'composite_gap')})
        lineage['states'][f['fold']] = {p: states[p].to_dict() for p in states}
        print(f['fold'], 'admission dates', len(days), flush=True)
    if len(lineage['admission']) != 35 * len(NEW_POLICIES):
        raise ValueError(f'admission must issue 35 dates for each of the {len(NEW_POLICIES)} policies')
    lineage['execution_stage'] = 'training_only_admission_complete_frozen_before_outer_scoring'
    atomic(ad / 'lineage.json', lineage)
    return 0


def job_comparison(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True, choices=(1, 2))
    args = ap.parse_args(rest)
    root, k = Path(root), args.attempt
    check_protocol(root, k)
    ad = attempt_dir(root, k)
    lineage = json.loads((ad / 'lineage.json').read_text())
    if lineage['execution_stage'] != 'training_only_admission_complete_frozen_before_outer_scoring':
        raise ValueError('admission not complete, or comparison already run')
    committed_at_head(root, str((ad / 'lineage.json').relative_to(root)))
    fit_ident, cp21_ident, hg_ident, hashes = _identities(root, k)
    if fit_ident != lineage['fit_identity'] or cp21_ident != lineage['cp21_fit_identity'] or hg_ident != lineage['hg_identity']:
        raise ValueError('identities changed after admission')
    budget = ledger()
    data = load(root)
    snapshot = cycle_days(root)
    frames, member_frames = [], []
    for f in data.spec.development_folds:
        sources = Sources(k, f.name, data, fit_ident, cp21_ident, hg_ident, hashes, NEW_POLICIES)
        states = {p: SharedResidualState.from_dict(lineage['states'][f.name][p]) for p in H_POLICIES}
        dates = list(pd.date_range(f.evaluation.start, f.evaluation.end).date)
        replay(k, data, f.name, dates, sources, states, _truth(data), None, lineage, 'evaluation', frames, budget,
               'policy_days_evaluation', snapshot=snapshot, member_frames=member_frames)
        lineage['states'][f.name] = {p: states[p].to_dict() for p in states}
        print(f.name, 'evaluation dates', len(dates), flush=True)
    new = pd.concat(frames, ignore_index=True)
    if len(new) != len(NEW_POLICIES) * 10747:
        raise ValueError(f'missing original eligible predictions: {len(new)} rows')
    new.to_parquet(ad / 'predictions.parquet', index=False)
    members = pd.concat(member_frames, ignore_index=True)
    if len(members) != 10747:
        raise ValueError('members must cover the 10,747 keys')
    members.to_parquet(ad / 'members.parquet', index=False)
    lineage['execution_stage'] = 'comparison_vectors_complete_not_scored'
    atomic(ad / 'lineage.json', lineage)
    return 0
