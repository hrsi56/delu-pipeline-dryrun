"""CP-21 ordered jobs (capstone v21-r6 §17.2-§17.5): LightGBM fits, training-only admission,
comparison replay and the HG parity replay.

**One H layer for every arm.** HG, HGL, L-P, L-R and L-N all run through CP-16's frozen
`cp16.residuals.SharedResidualState` (V2-H output only), one state per arm, identical dates and
release/support/update rules, each arm with its own issued errors. The state blends two central
vectors as `a/2 + b/2`; an arm with one central vector `c` passes it twice, and `c/2 + c/2 == c`
exactly in binary floating point (asserted on every call). HG passes its own central the same
way; the HG parity replay proves that this path reproduces every accepted CP-20 HG vector bit for bit.

**Sources.** HG's A1_w/B2_w come only from the identity-verified CP-20 cache (§17.2); L-P, L-R and
L-N come from this checkpoint's own fit cache, each entry bound to the frozen protocol and inputs.
HGL is formed from those exact vectors. The replay never fits.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
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
from cp20.components import HGComponents
from .budget import atomic, ledger
from .inputs import hg_identity, identities, load, origin_manifest, weather_design, weather_matrix
from .jobs import art, stamp
from .lgbm import LGBM_ARMS, fit_arm, hg_central, hgl_central

OUT = Path('reports/block-challenger')
NEW_ARMS = ('HGL', 'L-P', 'L-R', 'L-N')
EVIDENCE_CLASS = 'development_post_selection'


# ------------------------------------------------------------------ protocol and identity
def check_protocol(root: Path) -> dict:
    """The committed pre-run protocol must be at HEAD and the implementation unchanged."""
    path = root / OUT / 'protocol.json'
    p = json.loads(path.read_text())
    for name, digest in p['implementation_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError(f'implementation changed since the pre-run freeze: {name}')
    for name, digest in p['frozen_inputs_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError(f'frozen input changed since the pre-run freeze: {name}')
    committed = subprocess.check_output(['git', 'show', f'HEAD:{OUT}/protocol.json'], cwd=root)
    if committed != path.read_bytes():
        raise ValueError('the pre-run protocol is not committed at HEAD')
    return p


def fit_identity(root: Path) -> dict:
    ids = identities(root)
    return {'cp21_input_fingerprint': hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(),
            'cp21_protocol_sha256': sha(root / OUT / 'protocol.json'),
            'weather_design_sha256': weather_design(root).sha256}


def digest(item: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in item.items() if k != 'content_sha256'}, sort_keys=True,
                                     allow_nan=False, default=str).encode()).hexdigest()


def fit_cache_dir() -> Path:
    return art() / 'fits'


def hg_cache_dir() -> Path:
    return art().parents[0] / 'cp-20' / 'hg-components'


class FitCache:
    """This checkpoint's LightGBM central forecasts, one verified entry per (fold, origin, arm)."""

    def __init__(self, fold: str, data, identity: dict, base: Path | None = None):
        self.fold, self.data, self.identity = fold, data, identity
        self.dir = (base or fit_cache_dir()) / fold

    def path(self, day: date, arm: str) -> Path:
        return self.dir / str(day) / f'{arm}.json'

    def load(self, day: date, arm: str):
        path = self.path(day, arm)
        if not path.exists():
            return None
        item = json.loads(path.read_text())
        rows = self.data.rows(day)
        if item.get('content_sha256') != digest(item) or item['day'] != str(day) or item['fold'] != self.fold \
                or item['arm'] != arm or any(item.get(k) != v for k, v in self.identity.items()) \
                or item['origin_utc'] != str(origin_utc(day).tz_convert('UTC')) \
                or item['timestamp_utc'] != list(map(str, self.data.index[rows])) \
                or item['scale_sha256'] != array_hash(self.data.scale[rows]):
            raise ValueError(f'stale or wrong CP-21 fit cache {self.fold} {day} {arm}')
        return item

    def get(self, day: date, arm: str) -> np.ndarray:
        item = self.load(day, arm)
        if item is None:
            raise ValueError(f'CP-21 fit cache miss {self.fold} {day} {arm}: fit in the fits job first')
        central = np.asarray(item['central'], float)
        if central.shape != (len(self.data.rows(day)),) or not np.isfinite(central).all():
            raise ValueError('invalid cached CP-21 central vector')
        return central

    def save(self, item: dict):
        atomic(self.path(date.fromisoformat(item['day']), item['arm']), item)


def charge_fits(budget, *, purpose: str, main: bool):
    def charge(role: str):
        extra = {'main_lgbm_fits': 1, 'lgbm_fits_main': 1} if main else {f'lgbm_fits_{purpose}': 1}
        budget.reserve(lgbm_fits=1, **{f'lgbm_{role}_fits': 1}, **extra)
    return charge


def fit_entry(data, wx, present, day: date, fold: str, arm: str, identity: dict, charge, n_jobs: int = 1) -> dict:
    rows = data.rows(day)
    started = time.perf_counter()
    central, records, selection = fit_arm(data, wx, present, day, arm, charge=charge, n_jobs=n_jobs)
    item = {'day': str(day), 'fold': fold, 'arm': arm, 'origin_utc': str(origin_utc(day).tz_convert('UTC')),
            'timestamp_utc': list(map(str, data.index[rows])), 'scale_sha256': array_hash(data.scale[rows]),
            **identity, 'central': central.tolist(), 'central_sha256': array_hash(central), 'selection': selection,
            'fits': records, 'seconds': time.perf_counter() - started, 'n_jobs': n_jobs, 'source': 'fresh_cp21_fit'}
    item['content_sha256'] = digest(item)
    return item


# ------------------------------------------------------------------ the fits job
_worker: dict = {}


def _init_worker(root: str, ledger_path: str):
    os.environ['CP21_LEDGER'] = ledger_path
    root = Path(root)
    check_protocol(root)
    _worker.update(root=root, identity=fit_identity(root), design=weather_design(root), budget=ledger(), data={})


def _fit_task(task):
    fold, day_s, arm, first_s, stage = task
    w = _worker
    key = (fold, first_s) if stage == 'warmup' else 'full'
    if key not in w['data']:
        data = load(w['root'], before=date.fromisoformat(first_s) if stage == 'warmup' else None)
        wx, present = weather_matrix(w['design'], data)
        w['data'] = {key: (data, wx, present)}
    data, wx, present = w['data'][key]
    cache = FitCache(fold, data, w['identity'])
    day = date.fromisoformat(day_s)
    try:
        if cache.load(day, arm) is not None:
            return fold, day_s, arm, 'reused', 0.0, None
    except ValueError:
        pass  # a stale/wrong entry is refitted (and charged), never reused
    began = time.time()
    try:
        item = fit_entry(data, wx, present, day, fold, arm, w['identity'], charge_fits(w['budget'], purpose='main', main=True))
    except Exception as exc:  # recorded, never substituted (§17.4 failure rule)
        failure = {'fold': fold, 'day': day_s, 'arm': arm, 'stage': stage, 'error': repr(exc),
                   'traceback': traceback.format_exc(), 'written_utc': stamp()}
        atomic(art() / 'failures' / f'{fold}_{day_s}_{arm}.json', failure)
        return fold, day_s, arm, 'failed', time.time() - began, repr(exc)
    cache.save(item)
    return fold, day_s, arm, item['source'], time.time() - began, None


def _stage_tasks(root: Path, stage: str, folds: set[str] | None):
    """Every (fold, origin, arm) of one stage, from the committed protocol's E1 origin table."""
    m = origin_manifest(root)
    table = check_protocol(root)['population']['origin_table']
    if len(table) != sum(f['date_count'] for f in m['folds']):
        raise ValueError('protocol origin table disagrees with the frozen manifest')
    with_rows = {(o['fold'], o['day']) for o in table if o['n_forecast']}
    tasks = []
    for f in m['folds']:
        if folds and f['fold'] not in folds:
            continue
        first = date.fromisoformat(f['evaluation_start'])
        for o in f['origins']:
            d = date.fromisoformat(o['day'])
            if (stage == 'warmup') != (d < first) or (f['fold'], o['day']) not in with_rows:
                continue
            tasks += [(f['fold'], o['day'], arm, str(first), stage) for arm in LGBM_ARMS]
    return tasks


def job_fits(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('warmup', 'evaluation'), required=True)
    ap.add_argument('--fold', action='append')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('CP21_WORKERS', '1')))
    args = ap.parse_args(rest)
    check_protocol(root)
    if args.stage == 'evaluation':
        lineage = json.loads((root / OUT / 'lineage.json').read_text())
        committed = subprocess.check_output(['git', 'show', f'HEAD:{OUT}/lineage.json'], cwd=root)
        if lineage['execution_stage'] != 'training_only_admission_complete_frozen_before_outer_scoring' \
                or committed != (root / OUT / 'lineage.json').read_bytes():
            raise ValueError('evaluation fits require the committed training-only admission freeze')
    if args.workers > int(os.environ.get('CP21_WORKERS', '1')):
        raise ValueError('more pool workers than the monitor declared')
    tasks = _stage_tasks(root, args.stage, set(args.fold) if args.fold else None)
    budget = ledger()
    budget.event('fits_start', stage=args.stage, tasks=len(tasks), workers=args.workers, folds=args.fold)
    done, failed, t0 = 0, 0, time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(str(root), os.environ['CP21_LEDGER'])) as pool:
        for fold, day, arm, source, seconds, error in pool.imap_unordered(_fit_task, tasks, chunksize=1):
            done += 1
            failed += source == 'failed'
            if done % 25 == 0 or source == 'failed':
                print(args.stage, fold, day, arm, source, round(seconds, 1), f'{done}/{len(tasks)}',
                      f'{time.time() - t0:.0f}s', error or '', flush=True)
    budget.event('fits_complete', stage=args.stage, tasks=len(tasks), failed=failed)
    print(json.dumps({'stage': args.stage, 'tasks': len(tasks), 'failed': failed, 'seconds': time.time() - t0}), flush=True)
    return 0 if not failed else 5


# ------------------------------------------------------------------ replay
def _twice(central: np.ndarray) -> np.ndarray:
    central = np.asarray(central, float)
    if not np.array_equal(central / 2 + central / 2, central):
        raise ValueError('single-central H-layer identity failed')
    return central


class Sources:
    """Central vectors per arm for one fold, all identity-verified."""

    def __init__(self, fold: str, data, fit_ident: dict, hg_ident: dict, arms, base: Path | None = None):
        self.data, self.arms = data, tuple(arms)
        self.hg = HGComponents(hg_cache_dir(), fold, data, hg_ident)
        self.fits = FitCache(fold, data, fit_ident, base)

    def get(self, day: date):
        rows = self.data.rows(day)
        if not len(rows):
            return rows, {arm: np.array([]) for arm in self.arms}, {}, 'original_no_eligible_hours'
        _, hg, _ = self.hg.get(day)
        centers, components = {}, {'A1': hg['A1'], 'B2': hg['B2']}
        lgbm = {arm: self.fits.get(day, arm) for arm in LGBM_ARMS if arm in self.arms or 'HGL' in self.arms}
        components.update(lgbm)
        for arm in self.arms:
            if arm == 'HG':
                centers[arm] = hg_central(hg['A1'], hg['B2'])
            elif arm == 'HGL':
                centers[arm] = hgl_central(hg['A1'], hg['B2'], lgbm['L-N'], lgbm['L-R'])
            else:
                centers[arm] = lgbm[arm]
        return rows, centers, components, 'verified_caches'


def output_frame(data, fold, day, rows, central, q, arm):
    item = pd.DataFrame({'fold': fold, 'policy': arm, 'timestamp_utc': data.index[rows], 'delivery_date': day,
                         'origin_utc': origin_utc(day).tz_convert('UTC'), 'y_true': data.y[rows],
                         'central': central, 'scale': data.scale[rows], 'level': data.level[rows],
                         'evidence_class': EVIDENCE_CLASS})
    for j, name in enumerate(LABELS):
        item[name] = q[:, j]
    return item


def state_dir() -> Path:
    return art() / 'states'


def save_state(state: SharedResidualState, path: Path) -> None:
    """Atomic write of a state's canonical serialisation (`SharedResidualState.dumps`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(state.dumps() + '\n', encoding='utf-8')
    os.replace(temp, path)


def replay(data, fold, dates, sources: Sources, states, truth, predict_days, lineage, phase, frames, budget, counter,
           snapshot_arms=()):
    """Advance every arm over the same dates with identical release/issue/support rules. For
    `snapshot_arms`, the state as it stood before each date's release is persisted (the state a
    live system would load that morning; the daily-cycle diagnostic starts from it cold)."""
    for d in dates:
        rows, centers, components, source = sources.get(d)
        for arm in sources.arms:
            budget.reserve(policy_days=1, **{counter: 1})
            state = states[arm]
            if arm in snapshot_arms:
                save_state(state, state_dir() / fold / str(d) / f'{arm}.json')
            state.release(d, truth)
            meta = {}
            if len(rows) and (predict_days is None or d in predict_days):
                c = _twice(centers[arm])
                q, meta = state.predict(d, data.index[rows], c, c, data.scale[rows])
                h = q['V2-H']
                meta = dict(meta)
                meta['vector_sha256'] = array_hash(h)
                if frames is not None:
                    frames.append(output_frame(data, fold, d, rows, centers[arm], h, arm))
            if len(rows):
                c = _twice(centers[arm])
                state.issue(d, data.index[rows], c, c, data.scale[rows])
            lineage['origins'].append({'arm': arm, 'fold': fold, 'day': str(d), 'phase': phase, 'source': source,
                                       'n_hours': int(len(rows)), 'predicted': bool(meta),
                                       'central_sha256': array_hash(centers[arm]) if len(rows) else None, **meta})


def _truth(data):
    series = pd.Series(data.y, index=data.index)
    return lambda ix: series.reindex(ix).to_numpy()


def job_admission(root: Path, rest) -> int:
    p = check_protocol(root)
    out = root / OUT
    if (out / 'lineage.json').exists():
        raise ValueError('existing admission evidence; no automatic retry/overwrite')
    m = origin_manifest(root)
    fit_ident, hg_ident = fit_identity(root), hg_identity(root)
    budget = ledger()
    lineage = {'schema': 'cp21-lineage-v1', 'protocol_sha256': sha(out / 'protocol.json'), 'fit_identity': fit_ident,
               'hg_identity': hg_ident, 'arms': list(NEW_ARMS), 'origins': [], 'admission': [], 'states': {},
               'execution_stage': 'training_only_admission_in_progress'}
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        data = load(root, before=first)
        sources = Sources(f['fold'], data, fit_ident, hg_ident, NEW_ARMS)
        states = {arm: SharedResidualState() for arm in NEW_ARMS}
        days = {date.fromisoformat(x['day']) for x in f['admission']}
        dates = list(pd.date_range(f['warmup_start'], first - timedelta(days=1)).date)
        before = len(lineage['origins'])
        replay(data, f['fold'], dates, sources, states, _truth(data), days, lineage, 'training_only', None, budget,
               'policy_days_admission')
        for rec in lineage['origins'][before:]:
            if rec['predicted']:
                lineage['admission'].append({k: rec[k] for k in ('arm', 'fold', 'day', 'n_hours', 'vector_sha256',
                                                                 'buffer_days', 'buffer_start', 'buffer_end', 'hour_support')})
        lineage['states'][f['fold']] = {arm: states[arm].to_dict() for arm in NEW_ARMS}
        atomic(out / 'lineage.json', lineage)
        print(f['fold'], 'admission dates', len(days), flush=True)
    if len(lineage['admission']) != 35 * len(NEW_ARMS):
        raise ValueError('admission must issue 35 dates for each of the four new arms')
    lineage['execution_stage'] = 'training_only_admission_complete_frozen_before_outer_scoring'
    atomic(out / 'lineage.json', lineage)
    return 0


def job_comparison(root: Path, rest) -> int:
    check_protocol(root)
    out = root / OUT
    lineage = json.loads((out / 'lineage.json').read_text())
    if lineage['execution_stage'] != 'training_only_admission_complete_frozen_before_outer_scoring':
        raise ValueError('admission not complete, or comparison already run')
    committed = subprocess.check_output(['git', 'show', f'HEAD:{OUT}/lineage.json'], cwd=root)
    if committed != (out / 'lineage.json').read_bytes():
        raise ValueError('the admission freeze must be committed before the outer comparison')
    fit_ident, hg_ident = fit_identity(root), hg_identity(root)
    if fit_ident != lineage['fit_identity'] or hg_ident != lineage['hg_identity']:
        raise ValueError('identities changed after admission')
    budget = ledger()
    data = load(root)
    frames = []
    for f in data.spec.development_folds:
        sources = Sources(f.name, data, fit_ident, hg_ident, NEW_ARMS)
        states = {arm: SharedResidualState.from_dict(lineage['states'][f.name][arm]) for arm in NEW_ARMS}
        dates = list(pd.date_range(f.evaluation.start, f.evaluation.end).date)
        replay(data, f.name, dates, sources, states, _truth(data), None, lineage, 'evaluation', frames, budget,
               'policy_days_evaluation', snapshot_arms=('HGL',))
        lineage['states'][f.name] = {arm: states[arm].to_dict() for arm in NEW_ARMS}
        print(f.name, 'evaluation dates', len(dates), flush=True)
    new = pd.concat(frames, ignore_index=True)
    if len(new) != 4 * 10747:
        raise ValueError(f'missing original eligible predictions: {len(new)} rows, expected 42,988')
    new.to_parquet(out / 'predictions.parquet', index=False)
    lineage['execution_stage'] = 'comparison_vectors_complete_not_scored'
    atomic(out / 'lineage.json', lineage)
    return 0


def job_hg_parity(root: Path, rest) -> int:
    """HG through the CP-21 replay path (single central passed twice), admission then evaluation,
    against the accepted CP-20 HG vectors on all 10,747 keys. No fit."""
    check_protocol(root)
    m = origin_manifest(root)
    fit_ident, hg_ident = fit_identity(root), hg_identity(root)
    budget = ledger()
    lineage, frames = {'origins': []}, []
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        early = load(root, before=first)
        states = {'HG': SharedResidualState()}
        days = {date.fromisoformat(x['day']) for x in f['admission']}
        replay(early, f['fold'], list(pd.date_range(f['warmup_start'], first - timedelta(days=1)).date),
               Sources(f['fold'], early, fit_ident, hg_ident, ('HG',)), states, _truth(early), days, lineage,
               'training_only', None, budget, 'policy_days_hg_parity')
        full = load(root)
        states = {'HG': SharedResidualState.from_dict(states['HG'].to_dict())}
        replay(full, f['fold'], list(pd.date_range(first, f['evaluation_end']).date),
               Sources(f['fold'], full, fit_ident, hg_ident, ('HG',)), states, _truth(full), None, lineage,
               'evaluation', frames, budget, 'policy_days_hg_parity')
    ours = pd.concat(frames, ignore_index=True)
    accepted = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet', filters=[('policy', '==', 'HG')])
    cols = ['fold', 'timestamp_utc', 'y_true', 'central', 'scale', 'level', *LABELS]
    a = ours[cols].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    b = accepted[cols].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    num = ['y_true', 'central', 'scale', 'level', *LABELS]
    keys = bool(len(a) == len(b) and (a.fold.to_numpy() == b.fold.to_numpy()).all()
                and pd.DatetimeIndex(a.timestamp_utc).as_unit('ns').equals(pd.DatetimeIndex(b.timestamp_utc).as_unit('ns')))
    result = {'schema': 'cp21-hg-parity-v1', 'written_utc': stamp(), 'rows': len(a), 'keys_equal': keys,
              'bitwise_equal': bool(keys and np.array_equal(a[num].to_numpy(float), b[num].to_numpy(float))),
              'max_abs_difference': float(np.max(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float)))) if keys else None,
              'admission_vectors': sum(1 for r in lineage['origins'] if r['phase'] == 'training_only' and r['predicted']),
              'policy_days_charged': len(lineage['origins']),
              'path': 'cp16.residuals.SharedResidualState via cp21.execution.replay, central passed twice (c/2+c/2==c)'}
    atomic(root / OUT / 'hg-parity.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['bitwise_equal'] else 6
