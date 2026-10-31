"""CP-23 ordered jobs (capstone v21-r10 §21.2, §21.4-§21.6): the per-fold configuration choice, the
DDNN fits, the training-only admission, the comparison replay and the HG/v4 parity replay.

**Policies.**

| ID | central forecast | intervals |
|---|---|---|
| v5 | `(2/3)·c_v4 + (1/3)·D`, evaluated as `2*c_v4/3 + D/3` in float64 | HG's H layer on v5's own errors |
| v3+D | `(2/3)·c_HG + (1/3)·D`, evaluated as `A1/3 + B2/3 + D/3` (CP-22's composite form) | HG's H layer on its own errors |
| D | DDNN's ensemble median | DDNN's own Johnson SU quantiles; its p50 is the central forecast |

**One H layer.** v5 and v3+D run through CP-16's frozen `cp16.residuals.SharedResidualState` (V2-H),
exactly as CP-21's and CP-22's replays did: one state per policy, the central passed as both blend
inputs (`c/2 + c/2 == c`, asserted). Residuals are never shared across policies.

**Sources.** HG's A1_w/B2_w come only from CP-20's identity-verified cache. v4's L-N and L-R come from
CP-21's retained fit cache under CP-21's recomputed identity, and v4's composite must equal the
`central_sha256` that CP-21's committed lineage recorded for that fold and day, warm-up included
(`cp22.execution.Sources`, reused unchanged). D comes from this checkpoint's own DDNN cache. The replay
never fits.

**The configuration choice** is made once per fold, before the fold's first origin, from data before
that origin only (`cp23.member.fit_selection`). The fits then use that configuration at every origin of
the fold, with four seeds.
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
from cp21.execution import digest
from cp22.execution import Sources as CP22Sources, cp21_lineage_hashes
from cp22.pn import composite
from . import ddnn as D
from .budget import atomic, charge_fits, ledger
from .features import encode, target
from .inputs import cp21_fit_identity, hg_identity, identities, load, origin_manifest, weather_design, weather_matrix
from .jobs import art, stamp
from .member import fit_origin, fit_selection
from .reference import require_passing_record

OUT = Path('reports/distribution-challenger')
EVIDENCE_CLASS = 'development_post_selection'
NEW_POLICIES = ('v5', 'v3+D', 'D')
H_POLICIES = ('v5', 'v3+D')
LAYER = {'v5': 'H', 'v3+D': 'H', 'D': 'JSU', 'HG': 'H', 'HGL': 'H'}
#: Five evaluation origins per fold (offsets 0/22/44/66/88 days, next day with eligible hours), at
#: which every policy's pre-release state is persisted for the cold daily-cycle diagnostic.
CYCLE_OFFSETS = (0, 22, 44, 66, 88)
COMPOSITE_TOLERANCE = 1e-9


# ------------------------------------------------------------------ protocol and identity
def check_protocol(root: Path) -> dict:
    """The committed pre-run protocol must be at HEAD and the implementation unchanged."""
    path = root / OUT / 'protocol.json'
    p = json.loads(path.read_text())
    for name, value in p['implementation_sha256'].items():
        if sha(root / name) != value:
            raise ValueError(f'implementation changed since the pre-run freeze: {name}')
    for name, value in p['frozen_inputs_sha256'].items():
        if sha(root / name) != value:
            raise ValueError(f'frozen input changed since the pre-run freeze: {name}')
    committed = subprocess.check_output(['git', 'show', f'HEAD:{OUT}/protocol.json'], cwd=root)
    if committed != path.read_bytes():
        raise ValueError('the pre-run protocol is not committed at HEAD')
    return p


def committed_at_head(root: Path, name: str) -> None:
    committed = subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=root)
    if committed != (root / name).read_bytes():
        raise ValueError(f'{name} must be committed at HEAD')


def fit_identity(root: Path) -> dict:
    ids = identities(root)
    return {'cp23_input_fingerprint': hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(),
            'cp23_protocol_sha256': sha(root / OUT / 'protocol.json'),
            'weather_design_sha256': weather_design(root).sha256}


# ------------------------------------------------------------------ composites
def v5_central(c_v4: np.ndarray, d: np.ndarray) -> np.ndarray:
    """v5 = (2/3)·c_v4 + (1/3)·D in float64, evaluated as 2*c_v4/3 + D/3."""
    c_v4, d = np.asarray(c_v4, float), np.asarray(d, float)
    if c_v4.shape != d.shape or not (np.isfinite(c_v4).all() and np.isfinite(d).all()):
        raise ValueError('v5 needs two finite vectors of one shape')
    return 2 * c_v4 / 3 + d / 3


def v3d_central(a1: np.ndarray, b2: np.ndarray, d: np.ndarray) -> np.ndarray:
    """v3+D = (2/3)·c_HG + (1/3)·D, CP-22's composite form A1/3 + B2/3 + D/3."""
    return composite(a1, b2, ((d, 3),))


def composite_gap(policy: str, central: np.ndarray, m: dict) -> float:
    """|c - (2/3)·c_base - (1/3)·D|, the §21.7 composite parity."""
    base = m['HGL'] if policy == 'v5' else m['HG']
    return float(np.max(np.abs(central - (2 / 3) * base - m['D'] / 3)))


# ------------------------------------------------------------------ the DDNN cache
class DDNNCache:
    """This checkpoint's DDNN forecasts, one verified entry per (fold, origin)."""

    def __init__(self, fold: str, data, identity: dict, base: Path | None = None):
        self.fold, self.data, self.identity = fold, data, identity
        self.dir = (base or art() / 'fits') / fold

    def path(self, day: date) -> Path:
        return self.dir / str(day) / 'DDNN.json'

    def load(self, day: date):
        path = self.path(day)
        if not path.exists():
            return None
        item = json.loads(path.read_text())
        rows = self.data.rows(day)
        if item.get('content_sha256') != digest(item) or item['day'] != str(day) or item['fold'] != self.fold \
                or item['arm'] != 'DDNN' or any(item.get(k) != v for k, v in self.identity.items()) \
                or item['origin_utc'] != str(origin_utc(day).tz_convert('UTC')) \
                or item['timestamp_utc'] != list(map(str, self.data.index[rows])) \
                or item['scale_sha256'] != array_hash(self.data.scale[rows]) \
                or item['level_sha256'] != array_hash(self.data.level[rows]):
            raise ValueError(f'stale or wrong CP-23 DDNN cache {self.fold} {day}')
        return item

    def get(self, day: date) -> dict:
        item = self.load(day)
        if item is None:
            raise ValueError(f'CP-23 DDNN cache miss {self.fold} {day}: fit in the fits job first')
        n = len(self.data.rows(day))
        central = np.asarray(item['central'], float)
        quantiles = np.asarray(item['quantiles'], float)
        if central.shape != (n,) or quantiles.shape != (n, 7) or not (np.isfinite(central).all() and np.isfinite(quantiles).all()):
            raise ValueError('invalid cached DDNN vector')
        if array_hash(central) != item['central_sha256'] or array_hash(quantiles) != item['quantiles_sha256']:
            raise ValueError('DDNN vector differs from its recorded hash')
        if (np.diff(quantiles, axis=1) < 0).any() or not np.array_equal(central, quantiles[:, D.MEDIAN]):
            raise ValueError('cached DDNN quantiles crossed, or the p50 is not the central forecast')
        return {'central': central, 'quantiles': quantiles, 'config': item['config']}

    def save(self, item: dict):
        atomic(self.path(date.fromisoformat(item['day'])), item)


def ddnn_entry(fit: dict, fold: str, identity: dict, n_jobs: int = 1) -> dict:
    item = {'day': fit['day'], 'fold': fold, 'arm': 'DDNN', 'origin_utc': fit['origin_utc'],
            'timestamp_utc': fit['timestamp_utc'], 'scale_sha256': fit['scale_sha256'], 'level_sha256': fit['level_sha256'],
            **identity, 'config': fit['config'], 'seeds': fit['seeds'], 'n_inputs': fit['n_inputs'], 'rows': fit['rows'],
            'central': fit['central'].tolist(), 'central_sha256': fit['central_sha256'],
            'quantiles': fit['quantiles'].tolist(), 'quantiles_sha256': fit['quantiles_sha256'],
            'member_z_quantiles_sha256': array_hash(fit['member_z_quantiles']),
            'member_medians_z': fit['member_z_quantiles'][:, :, D.MEDIAN].tolist(),
            'member_quantiles_z': fit['member_z_quantiles'].tolist(),
            'member_jsu_params': fit['member_jsu_params'].tolist(),
            'crossed_rows': fit['crossed_rows'], 'members': fit['members'], 'preprocessor': fit['preprocessor'],
            'predict_seconds': fit['predict_seconds'], 'seconds': fit['seconds'], 'maxrss_bytes': fit['maxrss_bytes'],
            'n_jobs': n_jobs, 'source': 'fresh_cp23_fit'}
    item['content_sha256'] = digest(item)
    return item


def selection_path(fold: str) -> Path:
    return art() / 'selection' / f'{fold}.json'


def selected_config(root: Path, fold: str) -> str:
    """The fold's committed configuration choice (reports/distribution-challenger/selection.json)."""
    committed = json.loads((root / OUT / 'selection.json').read_text())
    entry = committed['folds'][fold]
    local = json.loads(selection_path(fold).read_text())
    if local['selected'] != entry['selected'] or local['content_sha256'] != entry['content_sha256'] \
            or digest(local) != local['content_sha256']:
        raise ValueError(f'{fold}: the configuration choice differs from its committed record')
    return entry['selected']


# ------------------------------------------------------------------ workers
_worker: dict = {}


def _init_worker(root: str, ledger_path: str):
    os.environ['CP23_LEDGER'] = ledger_path
    root = Path(root)
    check_protocol(root)
    require_passing_record(root)
    _worker.update(root=root, identity=fit_identity(root), design=weather_design(root), budget=ledger(), data={})


def _materialise(fold: str, first_s: str, stage: str):
    w = _worker
    key = (fold, first_s) if stage in ('warmup', 'selection') else 'full'
    if key not in w['data']:
        data = load(w['root'], before=date.fromisoformat(first_s) if stage in ('warmup', 'selection') else None)
        wx, present = weather_matrix(w['design'], data)
        x, _ = encode(data)
        w['data'] = {key: (data, x, wx, target(data), present)}
    return w['data'][key]


def _select_task(task):
    fold, d0_s, first_s = task
    w = _worker
    data, x, wx, z, present = _materialise(fold, first_s, 'selection')
    began = time.time()
    try:
        result = fit_selection(data, x, wx, z, present, date.fromisoformat(d0_s),
                               charge=charge_fits(w['budget'], 'selection', main=True))
    except Exception as exc:  # recorded, never substituted
        atomic(art() / 'failures' / f'{fold}_selection.json', {'fold': fold, 'stage': 'selection', 'error': repr(exc),
                                                               'traceback': traceback.format_exc(), 'written_utc': stamp()})
        return fold, None, time.time() - began, repr(exc)
    item = {'fold': fold, 'evaluation_start': first_s, **w['identity'], **result, 'written_utc': stamp()}
    item['content_sha256'] = digest(item)
    atomic(selection_path(fold), item)
    return fold, item['selected'], time.time() - began, None


def _fit_task(task):
    fold, day_s, first_s, stage, config_id = task
    w = _worker
    data, x, wx, z, present = _materialise(fold, first_s, stage)
    cache = DDNNCache(fold, data, w['identity'])
    day = date.fromisoformat(day_s)
    try:
        if cache.load(day) is not None:
            return fold, day_s, 'reused', 0.0, None
    except ValueError:
        pass  # a stale or wrong entry is refitted (and charged), never reused
    began = time.time()
    try:
        fit = fit_origin(data, x, wx, z, present, day, config_id, charge=charge_fits(w['budget'], stage, main=True))
        item = ddnn_entry(fit, fold, w['identity'])
    except Exception as exc:  # recorded, never substituted (§14.2 failure rule)
        atomic(art() / 'failures' / f'{fold}_{day_s}_DDNN.json',
               {'fold': fold, 'day': day_s, 'arm': 'DDNN', 'stage': stage, 'error': repr(exc),
                'traceback': traceback.format_exc(), 'written_utc': stamp()})
        return fold, day_s, 'failed', time.time() - began, repr(exc)
    cache.save(item)
    return fold, day_s, item['source'], time.time() - began, None


def _pool(root: Path, workers: int):
    if workers > int(os.environ.get('CP23_WORKERS', '1')):
        raise ValueError('more pool workers than the monitor declared')
    return mp.get_context('spawn').Pool(workers, initializer=_init_worker, initargs=(str(root), os.environ['CP23_LEDGER']))


def job_select(root: Path, rest) -> int:
    """The five folds' configuration choices (80 main fits), each before its fold's first origin."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=int(os.environ.get('CP23_WORKERS', '1')))
    args = ap.parse_args(rest)
    check_protocol(root)
    require_passing_record(root)
    m = origin_manifest(root)
    tasks = []
    for f in m['folds']:
        if selection_path(f['fold']).exists():
            raise ValueError(f'{f["fold"]}: a configuration choice exists; it is made once per fold, never repeated')
        d0 = min(date.fromisoformat(o['day']) for o in f['origins'])
        if str(d0) != f['warmup_start']:
            raise ValueError(f'{f["fold"]}: first origin {d0} is not the genuine warm-up start')
        tasks.append((f['fold'], str(d0), f['evaluation_start']))
    budget = ledger()
    budget.event('select_start', tasks=len(tasks), workers=args.workers)
    failed = 0
    with _pool(root, args.workers) as pool:
        for fold, chosen, seconds, error in pool.imap_unordered(_select_task, tasks, chunksize=1):
            failed += chosen is None
            print('selection', fold, chosen, round(seconds, 1), error or '', flush=True)
    budget.event('select_complete', failed=failed)
    if failed:
        return 5
    folds = {}
    for f in m['folds']:
        item = json.loads(selection_path(f['fold']).read_text())
        folds[f['fold']] = {'selected': item['selected'], 'first_origin': item['first_origin'],
                            'holdout_mae': item['holdout_mae'], 'tie': item['tie'],
                            'winner_margin_relative': item['winner_margin_relative'], 'rows': item['rows'],
                            'epochs': {c: [r['best_epoch'] for r in v['members']] for c, v in item['configurations'].items()},
                            'content_sha256': item['content_sha256']}
    atomic(root / OUT / 'selection.json', {'schema': 'cp23-selection-v1', 'written_utc': stamp(), 'folds': folds,
                                           'rule': 'lowest holdout MAE (EUR/MWh) of each configuration\'s four-seed ensemble '
                                                   'median on [D0-28, D0); exact tie to the smaller configuration; training '
                                                   'on [max(2019-01-01, D0-728), D0-56), early stopping on [D0-56, D0-28); '
                                                   'D0 = the fold\'s first origin (its genuine warm-up start)'})
    print(json.dumps({f: v['selected'] for f, v in folds.items()}), flush=True)
    return 0


def _stage_tasks(root: Path, stage: str, folds: set[str] | None):
    m = origin_manifest(root)
    table = check_protocol(root)['population']['origin_table']
    with_rows = {(o['fold'], o['day']) for o in table if o['n_forecast']}
    tasks = []
    for f in m['folds']:
        if folds and f['fold'] not in folds:
            continue
        config_id = selected_config(root, f['fold'])
        first = date.fromisoformat(f['evaluation_start'])
        for o in f['origins']:
            d = date.fromisoformat(o['day'])
            if (stage == 'warmup') != (d < first) or (f['fold'], o['day']) not in with_rows:
                continue
            tasks.append((f['fold'], o['day'], str(first), stage, config_id))
    return tasks


def job_fits(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('warmup', 'evaluation'), required=True)
    ap.add_argument('--fold', action='append')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('CP23_WORKERS', '1')))
    args = ap.parse_args(rest)
    check_protocol(root)
    require_passing_record(root)
    committed_at_head(root, str(OUT / 'selection.json'))
    if args.stage == 'evaluation':
        lineage = json.loads((root / OUT / 'lineage.json').read_text())
        if lineage['execution_stage'] != 'training_only_admission_complete_frozen_before_outer_scoring':
            raise ValueError('evaluation fits require the committed training-only admission freeze')
        committed_at_head(root, str(OUT / 'lineage.json'))
    tasks = _stage_tasks(root, args.stage, set(args.fold) if args.fold else None)
    budget = ledger()
    budget.event('fits_start', stage=args.stage, tasks=len(tasks), workers=args.workers, folds=args.fold)
    done, failed, t0 = 0, 0, time.time()
    with _pool(root, args.workers) as pool:
        for fold, day, source, seconds, error in pool.imap_unordered(_fit_task, tasks, chunksize=1):
            done += 1
            failed += source == 'failed'
            if done % 25 == 0 or source == 'failed':
                print(args.stage, fold, day, source, round(seconds, 1), f'{done}/{len(tasks)}', f'{time.time() - t0:.0f}s',
                      error or '', flush=True)
    budget.event('fits_complete', stage=args.stage, tasks=len(tasks), failed=failed)
    print(json.dumps({'stage': args.stage, 'tasks': len(tasks), 'failed': failed, 'seconds': time.time() - t0}), flush=True)
    return 0 if not failed else 5


# ------------------------------------------------------------------ replay
class Sources(CP22Sources):
    """Central vectors per policy for one fold: CP-22's verified v3/v4 members plus this fold's D."""

    def __init__(self, fold: str, data, fit_ident: dict | None, cp21_ident: dict, hg_ident: dict, lineage_hashes: dict,
                 policies, *, base: Path | None = None):
        super().__init__(fold, data, None, cp21_ident, hg_ident, lineage_hashes, policies)
        self.ddnn = DDNNCache(fold, data, fit_ident, base) if fit_ident is not None else None

    def members(self, day: date) -> dict:
        out = super().members(day)
        if self.ddnn is not None:
            item = self.ddnn.get(day)
            out['D'], out['D_quantiles'], out['D_config'] = item['central'], item['quantiles'], item['config']
        return out

    def central(self, policy: str, m: dict) -> np.ndarray:
        if policy == 'v5':
            return v5_central(m['HGL'], m['D'])
        if policy == 'v3+D':
            return v3d_central(m['A1'], m['B2'], m['D'])
        if policy == 'D':
            return m['D']
        if policy in ('HG', 'HGL'):
            return super().central(policy, m)
        raise ValueError(f'unknown CP-23 policy {policy}')


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
    item = pd.DataFrame({'fold': fold, 'timestamp_utc': data.index[rows], 'delivery_date': day, 'config': m['D_config']})
    for name in ('D', 'A1', 'B2', 'L-N', 'L-R', 'HG', 'HGL'):
        item[name] = m[name]
    item['c_v5'] = centers['v5'] if 'v5' in centers else np.nan
    item['c_v3+D'] = centers['v3+D'] if 'v3+D' in centers else np.nan
    return item


def save_state(state, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(state.to_dict(), sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def cycle_days(root: Path) -> set[tuple[str, str]]:
    table = check_protocol(root)['population']['origin_table']
    with_rows = {(o['fold'], o['day']) for o in table if o['n_forecast']}
    out = set()
    for f in origin_manifest(root)['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        for offset in CYCLE_OFFSETS:
            d = first + timedelta(days=offset)
            while (f['fold'], str(d)) not in with_rows:
                d += timedelta(days=1)
            out.add((f['fold'], str(d)))
    return out


def replay(data, fold, dates, sources: Sources, states, truth, predict_days, lineage, phase, frames, budget, counter,
           *, snapshot: set | None = None, member_frames=None):
    """Advance every policy over the same dates with identical release, issue and support rules.

    H policies release, predict on `predict_days` (None = every date) and issue every date. D has no
    state: it emits DDNN's own quantiles on `predict_days`."""
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
                    vector = m['D_quantiles']
                    if not np.array_equal(centers['D'], vector[:, D.MEDIAN]):
                        raise ValueError('D: the emitted p50 is not the ensemble median')
                    emitted = True
                    meta = {'vector_sha256': array_hash(vector), 'config': m['D_config']}
                    if frames is not None:
                        frames.append(output_frame(data, fold, d, rows, centers[policy], vector, policy))
            else:
                state = states[policy]
                if snapshot is not None and (fold, str(d)) in snapshot:
                    save_state(state, art() / 'states' / fold / str(d) / f'{policy}.json')
                state.release(d, truth)
                if len(rows) and wanted:
                    c = _twice(centers[policy])
                    q, meta = state.predict(d, data.index[rows], c, c, data.scale[rows])
                    vector = q['V2-H']
                    emitted = True
                    meta = dict(meta)
                    meta['vector_sha256'] = array_hash(vector)
                    if policy in ('v5', 'v3+D'):
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


def _identities(root: Path):
    return fit_identity(root), cp21_fit_identity(root), hg_identity(root), cp21_lineage_hashes(root)


def new_states(policies):
    return {p: SharedResidualState() for p in policies if LAYER[p] == 'H'}


def job_admission(root: Path, rest) -> int:
    """Training-only admission (§14.2's slice): warm-up replay from each fold's genuine warm-up start,
    issuing on the seven admission dates, with data materialised before the fold's first evaluation day."""
    check_protocol(root)
    if (root / OUT / 'lineage.json').exists():
        raise ValueError('existing admission evidence; no automatic retry/overwrite')
    committed_at_head(root, str(OUT / 'selection.json'))
    m = origin_manifest(root)
    fit_ident, cp21_ident, hg_ident, hashes = _identities(root)
    budget = ledger()
    lineage = {'schema': 'cp23-lineage-v1', 'protocol_sha256': sha(root / OUT / 'protocol.json'), 'fit_identity': fit_ident,
               'cp21_fit_identity': cp21_ident, 'hg_identity': hg_ident, 'policies': list(NEW_POLICIES),
               'selection_sha256': sha(root / OUT / 'selection.json'), 'origins': [], 'admission': [], 'states': {},
               'execution_stage': 'training_only_admission_in_progress'}
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        data = load(root, before=first)
        sources = Sources(f['fold'], data, fit_ident, cp21_ident, hg_ident, hashes, NEW_POLICIES)
        states = new_states(NEW_POLICIES)
        days = {date.fromisoformat(x['day']) for x in f['admission']}
        dates = list(pd.date_range(f['warmup_start'], first - timedelta(days=1)).date)
        before = len(lineage['origins'])
        replay(data, f['fold'], dates, sources, states, _truth(data), days, lineage, 'training_only', None, budget,
               'policy_days_admission')
        for rec in lineage['origins'][before:]:
            if rec['predicted']:
                lineage['admission'].append({k: rec.get(k) for k in ('policy', 'fold', 'day', 'n_hours', 'vector_sha256',
                                                                     'buffer_days', 'buffer_start', 'buffer_end',
                                                                     'composite_gap', 'config')})
        lineage['states'][f['fold']] = {p: states[p].to_dict() for p in states}
        print(f['fold'], 'admission dates', len(days), flush=True)
    if len(lineage['admission']) != 35 * len(NEW_POLICIES):
        raise ValueError(f'admission must issue 35 dates for each of the {len(NEW_POLICIES)} policies')
    lineage['execution_stage'] = 'training_only_admission_complete_frozen_before_outer_scoring'
    atomic(root / OUT / 'lineage.json', lineage)
    return 0


def job_comparison(root: Path, rest) -> int:
    check_protocol(root)
    out = root / OUT
    lineage = json.loads((out / 'lineage.json').read_text())
    if lineage['execution_stage'] != 'training_only_admission_complete_frozen_before_outer_scoring':
        raise ValueError('admission not complete, or comparison already run')
    committed_at_head(root, str(OUT / 'lineage.json'))
    fit_ident, cp21_ident, hg_ident, hashes = _identities(root)
    if fit_ident != lineage['fit_identity'] or cp21_ident != lineage['cp21_fit_identity'] or hg_ident != lineage['hg_identity']:
        raise ValueError('identities changed after admission')
    budget = ledger()
    data = load(root)
    snapshot = cycle_days(root)
    frames, member_frames = [], []
    for f in data.spec.development_folds:
        sources = Sources(f.name, data, fit_ident, cp21_ident, hg_ident, hashes, NEW_POLICIES)
        states = {p: SharedResidualState.from_dict(lineage['states'][f.name][p]) for p in H_POLICIES}
        dates = list(pd.date_range(f.evaluation.start, f.evaluation.end).date)
        replay(data, f.name, dates, sources, states, _truth(data), None, lineage, 'evaluation', frames, budget,
               'policy_days_evaluation', snapshot=snapshot, member_frames=member_frames)
        lineage['states'][f.name] = {p: states[p].to_dict() for p in states}
        print(f.name, 'evaluation dates', len(dates), flush=True)
    new = pd.concat(frames, ignore_index=True)
    if len(new) != len(NEW_POLICIES) * 10747:
        raise ValueError(f'missing original eligible predictions: {len(new)} rows')
    new.to_parquet(out / 'predictions.parquet', index=False)
    members = pd.concat(member_frames, ignore_index=True)
    if len(members) != 10747:
        raise ValueError('members must cover the 10,747 keys')
    members.to_parquet(out / 'members.parquet', index=False)
    lineage['execution_stage'] = 'comparison_vectors_complete_not_scored'
    atomic(out / 'lineage.json', lineage)
    return 0


def job_parity(root: Path, rest) -> int:
    """HG and v4 (HGL) through the CP-23 replay path, admission then evaluation, against the accepted
    CP-20 HG and committed CP-21 HGL vectors on all 10,747 keys. No fit."""
    check_protocol(root)
    m = origin_manifest(root)
    _, cp21_ident, hg_ident, hashes = _identities(root)
    budget = ledger()
    lineage, frames = {'origins': []}, []
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        early = load(root, before=first)
        states = new_states(('HG', 'HGL'))
        days = {date.fromisoformat(x['day']) for x in f['admission']}
        replay(early, f['fold'], list(pd.date_range(f['warmup_start'], first - timedelta(days=1)).date),
               Sources(f['fold'], early, None, cp21_ident, hg_ident, hashes, ('HG', 'HGL')), states, _truth(early), days,
               lineage, 'training_only', None, budget, 'policy_days_parity')
        full = load(root)
        states = {p: SharedResidualState.from_dict(s.to_dict()) for p, s in states.items()}
        replay(full, f['fold'], list(pd.date_range(first, f['evaluation_end']).date),
               Sources(f['fold'], full, None, cp21_ident, hg_ident, hashes, ('HG', 'HGL')), states, _truth(full), None,
               lineage, 'evaluation', frames, budget, 'policy_days_parity')
    ours = pd.concat(frames, ignore_index=True)
    cols = ['fold', 'timestamp_utc', 'y_true', 'central', 'scale', 'level', *LABELS]
    num = ['y_true', 'central', 'scale', 'level', *LABELS]
    result = {'schema': 'cp23-parity-v1', 'written_utc': stamp(),
              'path': 'cp16.residuals.SharedResidualState via cp23.execution.replay, central passed twice (c/2+c/2==c)'}
    for policy, path in (('HG', 'reports/weather-ablation/predictions.parquet'), ('HGL', 'reports/block-challenger/predictions.parquet')):
        accepted = pd.read_parquet(root / path, filters=[('policy', '==', policy)])
        a = ours.loc[ours.policy.eq(policy), cols].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
        b = accepted[cols].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
        keys = bool(len(a) == len(b) == 10747 and (a.fold.to_numpy() == b.fold.to_numpy()).all()
                    and pd.DatetimeIndex(a.timestamp_utc).as_unit('ns').equals(pd.DatetimeIndex(b.timestamp_utc).as_unit('ns')))
        result[policy] = {'rows': len(a), 'keys_equal': keys, 'committed': path,
                          'bitwise_equal': bool(keys and np.array_equal(a[num].to_numpy(float), b[num].to_numpy(float))),
                          'max_abs_difference': float(np.max(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float)))) if keys else None}
    result['policy_days_charged'] = len(lineage['origins'])
    atomic(root / OUT / 'parity.json', result)
    print(json.dumps(result), flush=True)
    return 0 if result['HG']['bitwise_equal'] and result['HGL']['bitwise_equal'] else 6
