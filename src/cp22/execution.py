"""CP-22 ordered jobs (capstone v21-r9 §20.2-§20.4, §20.6): PN fits, training-only admission,
comparison replay, the H/v4 parity replay, and the W layer arms after the replacement verdict.

**Policies.** Every composite is `(2/3)·c_HG + (1/3)·member` (`cp22.pn.composite`):

| ID | member | interval layer |
|---|---|---|
| R | PN-avg | H (§14.2) |
| M | mean(PN-sel, L-P) | H |
| A-PN-sel | PN-sel | H |
| A-LP | L-P | H |
| A-LN | L-N | H |
| v4+DL / v3+DL | v4's / v3's central | DL |
| W+ACI / W+DL / W+DLF | W's central (R or M) | ACI / DL / DLF |

**One H layer.** Every non-DL composite runs through CP-16's frozen
`cp16.residuals.SharedResidualState` (V2-H), exactly as CP-21's replay did: one state per policy,
one central vector passed as both blend inputs (`c/2 + c/2 == c`, asserted). The DL arms run
through `cp22.dl.DynamicResidualState`, a subclass of the same state.

**Sources.** HG's A1_w/B2_w come only from CP-20's identity-verified cache; L-P, L-N and L-R from
CP-21's retained fit cache, under CP-21's recomputed identity, and each central vector must equal
the `central_sha256` CP-21's committed lineage recorded for that arm, fold and day (warm-up
included); PN from this checkpoint's own fit cache. The replay never fits.
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
from cp21.execution import FitCache as CP21FitCache, digest
from cp21.lgbm import hg_central, hgl_central
from .budget import atomic, ledger
from .dl import DynamicResidualState
from .inputs import cp21_fit_identity, hg_identity, identities, load, origin_manifest, weather_design, weather_matrix
from .jobs import art, cp20_art, cp21_art, stamp
from .pn import composite, fit_pn

OUT = Path('reports/v4-revision')
EVIDENCE_CLASS = 'development_post_selection'
H_POLICIES = ('R', 'M', 'A-PN-sel', 'A-LP', 'A-LN')
FIXED_DL = ('v4+DL', 'v3+DL')
FIXED_NEW = H_POLICIES + FIXED_DL
W_POLICIES = ('W+ACI', 'W+DL', 'W+DLF')
LAYER = {**{p: 'H' for p in H_POLICIES}, 'v4+DL': 'DL', 'v3+DL': 'DL', 'W+ACI': 'ACI', 'W+DL': 'DL', 'W+DLF': 'DLF',
         'HG': 'H', 'HGL': 'H'}
#: Five evaluation origins per fold (offsets 0/22/44/66/88 days, next day with eligible hours),
#: at which every policy's pre-release state is persisted for the cold daily-cycle diagnostic.
CYCLE_OFFSETS = (0, 22, 44, 66, 88)


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
    return {'cp22_input_fingerprint': hashlib.sha256(json.dumps(ids, sort_keys=True).encode()).hexdigest(),
            'cp22_protocol_sha256': sha(root / OUT / 'protocol.json'),
            'weather_design_sha256': weather_design(root).sha256}


def charge_fits(budget, *, purpose: str, main: bool):
    def charge(role: str):
        extra = {'main_lgbm_fits': 1, 'lgbm_fits_main': 1} if main else {f'lgbm_fits_{purpose}': 1}
        budget.reserve(lgbm_fits=1, **{f'lgbm_{role}_fits': 1}, **extra)
    return charge


# ------------------------------------------------------------------ caches
class PNCache:
    """This checkpoint's PN member vectors, one verified entry per (fold, origin)."""

    def __init__(self, fold: str, data, identity: dict, base: Path | None = None):
        self.fold, self.data, self.identity = fold, data, identity
        self.dir = (base or art() / 'fits') / fold

    def path(self, day: date) -> Path:
        return self.dir / str(day) / 'PN.json'

    def load(self, day: date):
        path = self.path(day)
        if not path.exists():
            return None
        item = json.loads(path.read_text())
        rows = self.data.rows(day)
        if item.get('content_sha256') != digest(item) or item['day'] != str(day) or item['fold'] != self.fold \
                or item['arm'] != 'PN' or any(item.get(k) != v for k, v in self.identity.items()) \
                or item['origin_utc'] != str(origin_utc(day).tz_convert('UTC')) \
                or item['timestamp_utc'] != list(map(str, self.data.index[rows])) \
                or item['scale_sha256'] != array_hash(self.data.scale[rows]):
            raise ValueError(f'stale or wrong CP-22 PN cache {self.fold} {day}')
        return item

    def get(self, day: date) -> dict:
        item = self.load(day)
        if item is None:
            raise ValueError(f'CP-22 PN cache miss {self.fold} {day}: fit in the fits job first')
        n = len(self.data.rows(day))
        out = {k: np.asarray(item[k], float) for k in ('pn_avg', 'pn_sel')}
        if any(v.shape != (n,) or not np.isfinite(v).all() for v in out.values()):
            raise ValueError('invalid cached PN vector')
        if array_hash(out['pn_avg']) != item['pn_avg_sha256'] or array_hash(out['pn_sel']) != item['pn_sel_sha256']:
            raise ValueError('PN vector differs from its recorded hash')
        return out

    def save(self, item: dict):
        atomic(self.path(date.fromisoformat(item['day'])), item)


def cp21_lineage_hashes(root: Path) -> dict:
    """{(arm, fold, day): central_sha256} from CP-21's committed lineage (warm-up and evaluation)."""
    lineage = json.loads((root / 'reports/block-challenger/lineage.json').read_text())
    return {(r['arm'], r['fold'], r['day']): r['central_sha256'] for r in lineage['origins']}


class Sources:
    """Central vectors per policy for one fold, every one identity-verified."""

    def __init__(self, fold: str, data, fit_ident: dict, cp21_ident: dict, hg_ident: dict, lineage_hashes: dict,
                 policies, *, w_policy: str | None = None, base: Path | None = None):
        self.fold, self.data, self.policies, self.w = fold, data, tuple(policies), w_policy
        self.hg = HGComponents(cp20_art() / 'hg-components', fold, data, hg_ident)
        self.cp21 = CP21FitCache(fold, data, cp21_ident, cp21_art() / 'fits')
        self.pn = PNCache(fold, data, fit_ident, base) if fit_ident is not None else None
        self.hashes = lineage_hashes

    def _cp21(self, day: date, arm: str) -> np.ndarray:
        central = self.cp21.get(day, arm)
        if array_hash(central) != self.hashes[(arm, self.fold, str(day))]:
            raise ValueError(f'CP-21 {arm} {self.fold} {day} differs from its committed lineage hash')
        return central

    def members(self, day: date) -> dict:
        _, hg, _ = self.hg.get(day)
        out = {'A1': hg['A1'], 'B2': hg['B2'], 'L-P': self._cp21(day, 'L-P'), 'L-N': self._cp21(day, 'L-N'),
               'L-R': self._cp21(day, 'L-R')}
        if self.pn is not None:
            out.update({'PN-avg': (pn := self.pn.get(day))['pn_avg'], 'PN-sel': pn['pn_sel']})
        hgl = hgl_central(out['A1'], out['B2'], out['L-N'], out['L-R'])
        if array_hash(hgl) != self.hashes[('HGL', self.fold, str(day))]:
            raise ValueError(f'v4 {self.fold} {day} differs from its committed lineage hash')
        out['HGL'], out['HG'] = hgl, hg_central(out['A1'], out['B2'])
        return out

    def central(self, policy: str, m: dict) -> np.ndarray:
        a1, b2 = m['A1'], m['B2']
        if policy in ('HG', 'v3+DL'):
            return m['HG']
        if policy in ('HGL', 'v4+DL'):
            return m['HGL']
        if policy in W_POLICIES:
            return self.central(self.w, m)
        terms = {'R': ((m['PN-avg'], 3),), 'M': ((m['PN-sel'], 6), (m['L-P'], 6)), 'A-PN-sel': ((m['PN-sel'], 3),),
                 'A-LP': ((m['L-P'], 3),), 'A-LN': ((m['L-N'], 3),)}[policy]
        return composite(a1, b2, terms)

    def get(self, day: date):
        rows = self.data.rows(day)
        if not len(rows):
            return rows, {p: np.array([]) for p in self.policies}, {}, 'original_no_eligible_hours'
        m = self.members(day)
        return rows, {p: self.central(p, m) for p in self.policies}, m, 'verified_caches'


def new_state(policy: str):
    layer = LAYER[policy]
    return SharedResidualState() if layer == 'H' else DynamicResidualState(layer)


def state_from_dict(policy: str, state: dict):
    return SharedResidualState.from_dict(state) if LAYER[policy] == 'H' else DynamicResidualState.from_dict(state)


# ------------------------------------------------------------------ the fits job
def pn_entry(data, wx, present, day: date, fold: str, identity: dict, charge, n_jobs: int = 1) -> dict:
    rows = data.rows(day)
    started = time.perf_counter()
    member, records = fit_pn(data, wx, present, day, charge=charge, n_jobs=n_jobs)
    item = {'day': str(day), 'fold': fold, 'arm': 'PN', 'origin_utc': str(origin_utc(day).tz_convert('UTC')),
            'timestamp_utc': list(map(str, data.index[rows])), 'scale_sha256': array_hash(data.scale[rows]), **identity,
            'pn_avg': member['pn_avg'].tolist(), 'pn_avg_sha256': array_hash(member['pn_avg']),
            'pn_sel': member['pn_sel'].tolist(), 'pn_sel_sha256': array_hash(member['pn_sel']),
            'full_eur': [v.tolist() for v in member['full_eur']], 'full_z': [v.tolist() for v in member['full_z']],
            'selected': member['selected'], 'selected_index': member['selected_index'],
            'validation_mae': member['validation_mae'], 'tie': member['tie'],
            'winner_margin_relative': member['winner_margin_relative'],
            'averaging_route_max_abs_gap': member['averaging_route_max_abs_gap'],
            'fits': records, 'seconds': time.perf_counter() - started, 'n_jobs': n_jobs, 'source': 'fresh_cp22_fit'}
    item['content_sha256'] = digest(item)
    return item


_worker: dict = {}


def _init_worker(root: str, ledger_path: str):
    os.environ['CP22_LEDGER'] = ledger_path
    root = Path(root)
    check_protocol(root)
    _worker.update(root=root, identity=fit_identity(root), design=weather_design(root), budget=ledger(), data={})


def _fit_task(task):
    fold, day_s, first_s, stage = task
    w = _worker
    key = (fold, first_s) if stage == 'warmup' else 'full'
    if key not in w['data']:
        data = load(w['root'], before=date.fromisoformat(first_s) if stage == 'warmup' else None)
        wx, present = weather_matrix(w['design'], data)
        w['data'] = {key: (data, wx, present)}
    data, wx, present = w['data'][key]
    cache = PNCache(fold, data, w['identity'])
    day = date.fromisoformat(day_s)
    try:
        if cache.load(day) is not None:
            return fold, day_s, 'reused', 0.0, None
    except ValueError:
        pass  # a stale/wrong entry is refitted (and charged), never reused
    began = time.time()
    try:
        item = pn_entry(data, wx, present, day, fold, w['identity'], charge_fits(w['budget'], purpose='main', main=True))
    except Exception as exc:  # recorded, never substituted (§17.4 failure rule)
        atomic(art() / 'failures' / f'{fold}_{day_s}_PN.json',
               {'fold': fold, 'day': day_s, 'arm': 'PN', 'stage': stage, 'error': repr(exc),
                'traceback': traceback.format_exc(), 'written_utc': stamp()})
        return fold, day_s, 'failed', time.time() - began, repr(exc)
    cache.save(item)
    return fold, day_s, item['source'], time.time() - began, None


def _stage_tasks(root: Path, stage: str, folds: set[str] | None):
    m = origin_manifest(root)
    table = check_protocol(root)['population']['origin_table']
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
            tasks.append((f['fold'], o['day'], str(first), stage))
    return tasks


def job_fits(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('warmup', 'evaluation'), required=True)
    ap.add_argument('--fold', action='append')
    ap.add_argument('--workers', type=int, default=int(os.environ.get('CP22_WORKERS', '1')))
    args = ap.parse_args(rest)
    check_protocol(root)
    if args.stage == 'evaluation':
        lineage = json.loads((root / OUT / 'lineage.json').read_text())
        if lineage['execution_stage'] != 'training_only_admission_complete_frozen_before_outer_scoring':
            raise ValueError('evaluation fits require the committed training-only admission freeze')
        committed_at_head(root, str(OUT / 'lineage.json'))
    if args.workers > int(os.environ.get('CP22_WORKERS', '1')):
        raise ValueError('more pool workers than the monitor declared')
    tasks = _stage_tasks(root, args.stage, set(args.fold) if args.fold else None)
    budget = ledger()
    budget.event('fits_start', stage=args.stage, tasks=len(tasks), workers=args.workers, folds=args.fold)
    done, failed, t0 = 0, 0, time.time()
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, initializer=_init_worker, initargs=(str(root), os.environ['CP22_LEDGER'])) as pool:
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


def members_frame(data, fold, day, rows, m):
    item = pd.DataFrame({'fold': fold, 'timestamp_utc': data.index[rows], 'delivery_date': day})
    for name in ('PN-avg', 'PN-sel', 'L-P', 'L-N', 'L-R', 'A1', 'B2', 'HG', 'HGL'):
        item[name] = m[name]
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
    """Advance every policy over the same dates with identical release/issue/support rules.

    H policies emit on `predict_days` (None = every date). DL policies emit an interval on every
    date their buffer is full -- warm-up included, so ACI is updated -- and their vectors are kept
    on `predict_days` exactly as the H policies'."""
    for d in dates:
        rows, centers, m, source = sources.get(d)
        if member_frames is not None and len(rows):
            member_frames.append(members_frame(data, fold, d, rows, m))
        for policy in sources.policies:
            budget.reserve(policy_days=1, **{counter: 1})
            state = states[policy]
            if snapshot is not None and (fold, str(d)) in snapshot:
                save_state(state, art() / 'states' / fold / str(d) / f'{policy}.json')
            state.release(d, truth)
            wanted = predict_days is None or d in predict_days
            dl = LAYER[policy] != 'H'
            meta, emitted = {}, False
            if len(rows) and (wanted or (dl and state.ready())):
                c = _twice(centers[policy])
                q, meta = state.predict(d, data.index[rows], c, c, data.scale[rows])
                vector = q['DL'] if dl else q['V2-H']
                emitted = True
                meta = dict(meta)
                meta['vector_sha256'] = array_hash(vector)
                if wanted and frames is not None:
                    frames.append(output_frame(data, fold, d, rows, centers[policy], vector, policy))
            if len(rows):
                c = _twice(centers[policy])
                state.issue(d, data.index[rows], c, c, data.scale[rows])
            lineage['origins'].append({'policy': policy, 'fold': fold, 'day': str(d), 'phase': phase, 'source': source,
                                       'n_hours': int(len(rows)), 'predicted': bool(emitted and wanted),
                                       'emitted': emitted, 'layer': LAYER[policy],
                                       'central_sha256': array_hash(centers[policy]) if len(rows) else None, **meta})


def _truth(data):
    series = pd.Series(data.y, index=data.index)
    return lambda ix: series.reindex(ix).to_numpy()


def _identities(root: Path):
    return fit_identity(root), cp21_fit_identity(root), hg_identity(root), cp21_lineage_hashes(root)


def _admission(root: Path, policies, lineage_name: str, counter: str, w_policy: str | None, snapshot) -> dict:
    m = origin_manifest(root)
    fit_ident, cp21_ident, hg_ident, hashes = _identities(root)
    budget = ledger()
    lineage = {'schema': 'cp22-lineage-v1', 'protocol_sha256': sha(root / OUT / 'protocol.json'), 'fit_identity': fit_ident,
               'cp21_fit_identity': cp21_ident, 'hg_identity': hg_ident, 'policies': list(policies), 'w_policy': w_policy,
               'origins': [], 'admission': [], 'states': {}, 'execution_stage': 'training_only_admission_in_progress'}
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        data = load(root, before=first)
        sources = Sources(f['fold'], data, fit_ident, cp21_ident, hg_ident, hashes, policies, w_policy=w_policy)
        states = {p: new_state(p) for p in policies}
        days = {date.fromisoformat(x['day']) for x in f['admission']}
        dates = list(pd.date_range(f['warmup_start'], first - timedelta(days=1)).date)
        before = len(lineage['origins'])
        replay(data, f['fold'], dates, sources, states, _truth(data), days, lineage, 'training_only', None, budget, counter)
        for rec in lineage['origins'][before:]:
            if rec['predicted']:
                lineage['admission'].append({k: rec.get(k) for k in ('policy', 'fold', 'day', 'n_hours', 'vector_sha256',
                                                                     'buffer_days', 'buffer_start', 'buffer_end', 'alpha_t')})
        lineage['states'][f['fold']] = {p: states[p].to_dict() for p in policies}
        print(f['fold'], 'admission dates', len(days), flush=True)
    if len(lineage['admission']) != 35 * len(policies):
        raise ValueError(f'admission must issue 35 dates for each of the {len(policies)} policies')
    lineage['execution_stage'] = 'training_only_admission_complete_frozen_before_outer_scoring'
    atomic(root / OUT / lineage_name, lineage)
    return lineage


def _comparison(root: Path, policies, lineage_name: str, predictions_name: str, counter: str, w_policy: str | None,
                members_name: str | None) -> int:
    out = root / OUT
    lineage = json.loads((out / lineage_name).read_text())
    if lineage['execution_stage'] != 'training_only_admission_complete_frozen_before_outer_scoring':
        raise ValueError('admission not complete, or comparison already run')
    committed_at_head(root, str(OUT / lineage_name))
    fit_ident, cp21_ident, hg_ident, hashes = _identities(root)
    if fit_ident != lineage['fit_identity'] or cp21_ident != lineage['cp21_fit_identity'] or hg_ident != lineage['hg_identity']:
        raise ValueError('identities changed after admission')
    if lineage['w_policy'] != w_policy or tuple(lineage['policies']) != tuple(policies):
        raise ValueError('policy set changed after admission')
    budget = ledger()
    data = load(root)
    snapshot = cycle_days(root)
    frames, member_frames = [], ([] if members_name else None)
    for f in data.spec.development_folds:
        sources = Sources(f.name, data, fit_ident, cp21_ident, hg_ident, hashes, policies, w_policy=w_policy)
        states = {p: state_from_dict(p, lineage['states'][f.name][p]) for p in policies}
        dates = list(pd.date_range(f.evaluation.start, f.evaluation.end).date)
        replay(data, f.name, dates, sources, states, _truth(data), None, lineage, 'evaluation', frames, budget, counter,
               snapshot=snapshot, member_frames=member_frames)
        lineage['states'][f.name] = {p: states[p].to_dict() for p in policies}
        print(f.name, 'evaluation dates', len(dates), flush=True)
    new = pd.concat(frames, ignore_index=True)
    if len(new) != len(policies) * 10747:
        raise ValueError(f'missing original eligible predictions: {len(new)} rows')
    new.to_parquet(out / predictions_name, index=False)
    if members_name:
        members = pd.concat(member_frames, ignore_index=True)
        if len(members) != 10747:
            raise ValueError('members must cover the 10,747 keys')
        members.to_parquet(out / members_name, index=False)
    lineage['execution_stage'] = 'comparison_vectors_complete_not_scored'
    atomic(out / lineage_name, lineage)
    return 0


def job_admission(root: Path, rest) -> int:
    check_protocol(root)
    if (root / OUT / 'lineage.json').exists():
        raise ValueError('existing admission evidence; no automatic retry/overwrite')
    _admission(root, FIXED_NEW, 'lineage.json', 'policy_days_admission', None, None)
    return 0


def job_comparison(root: Path, rest) -> int:
    check_protocol(root)
    return _comparison(root, FIXED_NEW, 'lineage.json', 'predictions.parquet', 'policy_days_evaluation', None, 'members.parquet')


def _winner(root: Path) -> str:
    verdict = json.loads((root / OUT / 'replacement.json').read_text())
    committed_at_head(root, str(OUT / 'replacement.json'))
    if verdict['winner'] not in ('R', 'M'):
        raise ValueError('no replacement: the layer arms on W do not exist (§20.6)')
    return verdict['winner']


def job_w_admission(root: Path, rest) -> int:
    check_protocol(root)
    if (root / OUT / 'lineage-w.json').exists():
        raise ValueError('existing W admission evidence; no automatic retry/overwrite')
    _admission(root, W_POLICIES, 'lineage-w.json', 'policy_days_w_admission', _winner(root), None)
    return 0


def job_w_comparison(root: Path, rest) -> int:
    check_protocol(root)
    return _comparison(root, W_POLICIES, 'lineage-w.json', 'predictions-w.parquet', 'policy_days_w_evaluation',
                       _winner(root), None)


def job_parity(root: Path, rest) -> int:
    """HG and v4 (HGL) through the CP-22 replay path, admission then evaluation, against the
    accepted CP-20 HG and committed CP-21 HGL vectors on all 10,747 keys. No fit."""
    check_protocol(root)
    m = origin_manifest(root)
    _, cp21_ident, hg_ident, hashes = _identities(root)
    budget = ledger()
    lineage, frames = {'origins': []}, []
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        early = load(root, before=first)
        states = {'HG': SharedResidualState(), 'HGL': SharedResidualState()}
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
    result = {'schema': 'cp22-parity-v1', 'written_utc': stamp(),
              'path': 'cp16.residuals.SharedResidualState via cp22.execution.replay, central passed twice (c/2+c/2==c)'}
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
