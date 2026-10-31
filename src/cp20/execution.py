"""CP-20 ordered jobs: HG components, training-only admission, comparison, controls, scoring.

Both arms run through the identical replay: CP-16's frozen H layer
(``cp16.residuals.SharedResidualState``, V2-H output only), identical dates and
release/support/update rules, each arm with its own genuinely issued central errors.
The arms differ only in their component source: H0 = identity-verified no-weather
caches; HG = fresh fits on the weather-augmented design (``components.py``).
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

import numpy as np
import pandas as pd

from cp15.data import LABELS, array_hash, origin_utc, sha
from cp16.residuals import SharedResidualState
from .budget import Budget, atomic
from .components import HGComponents, augment, fingerprint, fit_hg
from .inputs import H0Components, identities, load, origin_manifest
from .weather import COLUMNS, WeatherDesign

ARMS = ('H0', 'HG')
OUT = Path('reports/weather-ablation')


def ledger():
    return Budget(os.environ['CP20_LEDGER'])


def cache_dir():
    return Path(os.environ.get('CP20_PROJECT_ROOT', '/Users/djourno/Downloads/PJM')) / '.local/artifacts/cp-20/hg-components'


def check_protocol(root: Path) -> dict:
    """The complete pre-fit protocol must be committed and the implementation unchanged."""
    path = root / OUT / 'protocol.json'
    p = json.loads(path.read_text())
    for name, digest in p['implementation_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError(f'implementation changed since pre-fit freeze: {name}')
    for name, digest in p['frozen_inputs_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError(f'frozen input changed since pre-fit freeze: {name}')
    committed = subprocess.check_output(['git', 'show', f'HEAD:{OUT}/protocol.json'], cwd=root)
    if committed != path.read_bytes():
        raise ValueError('pre-fit protocol is not committed at HEAD')
    return p


def weather_design(root: Path, p: dict) -> WeatherDesign:
    features = pd.read_parquet(root / OUT / 'weather-features.parquet')
    design = WeatherDesign.from_features(features)
    if design.sha256 != p['weather_design_sha256']:
        raise ValueError('weather design differs from the frozen protocol')
    return design


def hg_identity(root: Path, p: dict) -> dict:
    _, ids = identities(root)
    return {'input_fingerprint': fingerprint(ids, p['weather_design_sha256'], sha(root / OUT / 'protocol.json')),
            'weather_design_sha256': p['weather_design_sha256'], 'protocol_sha256': sha(root / OUT / 'protocol.json')}


# ------------------------------------------------------------------ HG component fitting
_worker = {}


def _init_worker(root, stage, ledger_path):
    os.environ['CP20_LEDGER'] = ledger_path
    root = Path(root)
    p = check_protocol(root)
    _worker.update(root=root, stage=stage, p=p, design=weather_design(root, p), identity=hg_identity(root, p),
                   budget=Budget(ledger_path), data={})


def _fit_task(task):
    fold, day_s, first_s = task
    w = _worker
    key = (fold, first_s) if w['stage'] == 'warmup' else 'full'
    if key not in w['data']:
        data, _ = load(w['root'], before=date.fromisoformat(first_s) if w['stage'] == 'warmup' else None)
        w['data'] = {key: (*augment(data, w['design']), data)}
    aug, present, data = w['data'][key]
    cache = HGComponents(cache_dir(), fold, data, w['identity'])
    day = date.fromisoformat(day_s)
    try:
        if cache.load(day) is not None:
            return fold, day_s, 'reused', 0.0
    except ValueError:
        pass  # a stale/wrong cache entry is refitted (and charged); never reused
    began = time.time()
    item = fit_hg(aug, present, day, fold, w['budget'], main=True, identity=w['identity'])
    cache.save(item)
    return fold, day_s, item['source'], time.time() - began


def components(root: Path, stage: str, workers: int):
    p = check_protocol(root)
    m = origin_manifest(root)
    tasks = []
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        for o in f['origins']:
            d = date.fromisoformat(o['day'])
            if (stage == 'warmup') == (d < first):
                tasks.append((f['fold'], o['day'], str(first)))
    if stage == 'evaluation':
        lineage = json.loads((root / OUT / 'lineage.json').read_text())
        if lineage['execution_stage'] != 'training_only_admission_complete_frozen_before_outer_scoring':
            raise ValueError('evaluation components require the committed admission freeze')
    budget = ledger()
    budget.event('hg_components_start', stage=stage, tasks=len(tasks), workers=workers)
    done = 0
    ctx = mp.get_context('spawn')
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(root), stage, os.environ['CP20_LEDGER'])) as pool:
        for fold, day, source, seconds in pool.imap_unordered(_fit_task, tasks, chunksize=1):
            done += 1
            print(stage, fold, day, source, round(seconds, 1), f'{done}/{len(tasks)}', flush=True)
    budget.event('hg_components_complete', stage=stage, tasks=len(tasks))


# ------------------------------------------------------------------ replay
def output_frame(data, fold, day, rows, centers, q, arm):
    item = pd.DataFrame({'fold': fold, 'policy': arm, 'timestamp_utc': data.index[rows], 'delivery_date': day,
                         'origin_utc': origin_utc(day).tz_convert('UTC'), 'y_true': data.y[rows],
                         'central': centers['A1'] / 2 + centers['B2'] / 2, 'scale': data.scale[rows],
                         'level': data.level[rows], 'evidence_class': 'development_post_selection'})
    for j, name in enumerate(LABELS):
        item[name] = q[:, j]
    return item


def replay(data, fold, dates, sources, states, truth, predict_days, lineage, phase, frames, budget, arms=ARMS):
    """Advance both arms over the same dates; identical release/issue/support rules."""
    for d in dates:
        for arm in arms:
            budget.reserve(policy_days=1)
            state = states[arm]
            state.release(d, truth)
            rows, centers, source = sources[arm].get(d)
            meta = {}
            if len(rows) and (predict_days is None or d in predict_days):
                q, meta = state.predict(d, data.index[rows], centers['A1'], centers['B2'], data.scale[rows])
                h = q['V2-H']  # frozen H recipe only; no pooled-P arm in CP-20
                meta = {k: v for k, v in meta.items()}
                meta['vector_sha256'] = array_hash(h)
                if frames is not None:
                    frames.append(output_frame(data, fold, d, rows, centers, h, arm))
            if len(rows):
                state.issue(d, data.index[rows], centers['A1'], centers['B2'], data.scale[rows])
            lineage['origins'].append({'arm': arm, 'fold': fold, 'day': str(d), 'phase': phase, 'source': source,
                                       'n_hours': int(len(rows)), 'predicted': bool(meta), **meta})


def admission(root: Path):
    p = check_protocol(root)
    out = root / OUT
    if (out / 'lineage.json').exists():
        raise ValueError('existing admission evidence; no automatic retry/overwrite')
    m = origin_manifest(root)
    design = weather_design(root, p)
    identity = hg_identity(root, p)
    budget = ledger()
    lineage = {'schema': 'cp20-lineage-v1', 'protocol_sha256': sha(out / 'protocol.json'), 'hg_identity': identity,
               'weather_design_sha256': design.sha256, 'origins': [], 'admission': [], 'states': {},
               'execution_stage': 'training_only_admission_in_progress'}
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        data, _ = load(root, before=first)
        truth = pd.Series(data.y, index=data.index)
        sources = {'H0': H0Components(root, f['fold'], data), 'HG': HGComponents(cache_dir(), f['fold'], data, identity)}
        states = {arm: SharedResidualState() for arm in ARMS}
        days = {date.fromisoformat(x['day']) for x in f['admission']}
        dates = list(pd.date_range(f['warmup_start'], first - timedelta(days=1)).date)
        before = len(lineage['origins'])
        replay(data, f['fold'], dates, sources, states, lambda ix: truth.reindex(ix).to_numpy(), days, lineage,
               'training_only', None, budget)
        for rec in lineage['origins'][before:]:
            if rec['predicted']:
                lineage['admission'].append({k: rec[k] for k in ('arm', 'fold', 'day', 'n_hours', 'vector_sha256',
                                                                 'buffer_days', 'buffer_start', 'buffer_end',
                                                                 'hour_support')})
        lineage['states'][f['fold']] = {arm: states[arm].to_dict() for arm in ARMS}
        atomic(out / 'lineage.json', lineage)
        print(f['fold'], 'admission dates', len(days), flush=True)
    if len(lineage['admission']) != 70:
        raise ValueError('admission must issue 35 dates for each of the two arms')
    lineage['execution_stage'] = 'training_only_admission_complete_frozen_before_outer_scoring'
    atomic(out / 'lineage.json', lineage)


def comparison(root: Path):
    p = check_protocol(root)
    out = root / OUT
    lineage = json.loads((out / 'lineage.json').read_text())
    if lineage['execution_stage'] != 'training_only_admission_complete_frozen_before_outer_scoring':
        raise ValueError('admission not complete, or comparison already run')
    committed = subprocess.check_output(['git', 'show', f'HEAD:{OUT}/lineage.json'], cwd=root)
    if committed != (out / 'lineage.json').read_bytes():
        raise ValueError('admission freeze must be committed before outer comparison')
    identity = hg_identity(root, p)
    if identity != lineage['hg_identity']:
        raise ValueError('HG identity changed after admission')
    budget = ledger()
    data, _ = load(root)
    truth = pd.Series(data.y, index=data.index)
    frames = []
    for f in data.spec.development_folds:
        sources = {'H0': H0Components(root, f.name, data), 'HG': HGComponents(cache_dir(), f.name, data, identity)}
        states = {arm: SharedResidualState.from_dict(lineage['states'][f.name][arm]) for arm in ARMS}
        dates = list(pd.date_range(f.evaluation.start, f.evaluation.end).date)
        replay(data, f.name, dates, sources, states, lambda ix: truth.reindex(ix).to_numpy(), None, lineage,
               'evaluation', frames, budget)
        lineage['states'][f.name] = {arm: states[arm].to_dict() for arm in ARMS}
        print(f.name, 'evaluation dates', len(dates), flush=True)
    new = pd.concat(frames, ignore_index=True)
    saved = pd.read_parquet(root / 'reports/cp15/predictions.parquet', filters=[('policy', 'in', ['B0', 'B1', 'B2', 'B3', 'A1'])])
    saved['evidence_class'] = 'development_post_selection'
    pred = pd.concat([saved, new], ignore_index=True)
    if len(pred) != 75229 or len(new) != 21494:
        raise ValueError('missing original eligible predictions')
    pred.to_parquet(out / 'predictions.parquet', index=False)
    lineage['execution_stage'] = 'comparison_vectors_complete_not_scored'
    atomic(out / 'lineage.json', lineage)


def verify_h0(root: Path) -> dict:
    """H0 must reproduce the accepted CP-16 V2-H vectors exactly (no fit involved)."""
    ours = pd.read_parquet(root / OUT / 'predictions.parquet', filters=[('policy', '==', 'H0')])
    accepted = pd.read_parquet(root / 'reports/v2-causal/predictions.parquet', filters=[('policy', '==', 'V2-H')])
    cols = ['fold', 'timestamp_utc', 'delivery_date', 'origin_utc', 'y_true', 'central', 'scale', 'level', *LABELS]
    a = ours[cols].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    b = accepted[cols].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    numeric = ['y_true', 'central', 'scale', 'level', *LABELS]
    diff = float(np.max(np.abs(a[numeric].to_numpy(float) - b[numeric].to_numpy(float))))
    keys_equal = a[['fold', 'timestamp_utc']].astype(str).equals(b[['fold', 'timestamp_utc']].astype(str))
    exact = bool(keys_equal and np.array_equal(a[numeric].to_numpy(float), b[numeric].to_numpy(float)))
    return {'rows': len(a), 'keys_equal': bool(keys_equal), 'max_abs_difference': diff, 'bitwise_equal': exact}


def score(root: Path):
    budget = ledger()
    budget.reserve(reference_passes=1, analysis_passes=1)
    from .scoring import evaluate
    out = root / OUT
    pred = pd.read_parquet(out / 'predictions.parquet')
    expected = pd.read_parquet(root / 'reports/cp15/predictions.parquet', columns=['fold', 'timestamp_utc', 'delivery_date'],
                               filters=[('policy', '==', 'B0')])
    lineage = json.loads((out / 'lineage.json').read_text())
    tables = evaluate(pred, expected, lineage)
    for name, value in tables.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out / f'{name}.csv', index=False)
    lineage['research_summary'] = tables['summary']
    lineage['h0_verification'] = verify_h0(root)
    lineage['execution_stage'] = 'scored_development_post_selection'
    atomic(out / 'lineage.json', lineage)
    print(json.dumps(tables['summary']['joint_conclusions']), json.dumps(lineage['h0_verification']), flush=True)


def h0_dryrun(root: Path, out_path: Path):
    """Preparation check (charged, local output only): H0 alone through the same replay,
    admission then evaluation, compared with the accepted CP-16 V2-H vectors. No fit, no weather."""
    identities(root)
    m = origin_manifest(root)
    budget = ledger()
    lineage = {'origins': []}
    frames = []
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        early, _ = load(root, before=first)
        truth = pd.Series(early.y, index=early.index)
        states = {'H0': SharedResidualState()}
        days = {date.fromisoformat(x['day']) for x in f['admission']}
        replay(early, f['fold'], list(pd.date_range(f['warmup_start'], first - timedelta(days=1)).date),
               {'H0': H0Components(root, f['fold'], early)}, states, lambda ix: truth.reindex(ix).to_numpy(),
               days, lineage, 'training_only', None, budget, arms=('H0',))
        full, _ = load(root)
        truth = pd.Series(full.y, index=full.index)
        states = {'H0': SharedResidualState.from_dict(states['H0'].to_dict())}
        replay(full, f['fold'], list(pd.date_range(first, f['evaluation_end']).date),
               {'H0': H0Components(root, f['fold'], full)}, states, lambda ix: truth.reindex(ix).to_numpy(),
               None, lineage, 'evaluation', frames, budget, arms=('H0',))
        print(f['fold'], 'H0 dry replay done', flush=True)
    ours = pd.concat(frames, ignore_index=True)
    accepted = pd.read_parquet(root / 'reports/v2-causal/predictions.parquet', filters=[('policy', '==', 'V2-H')])
    cols = ['fold', 'timestamp_utc', 'y_true', 'central', 'scale', 'level', *LABELS]
    a = ours[cols].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    b = accepted[cols].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    num = ['y_true', 'central', 'scale', 'level', *LABELS]
    result = {'rows': len(a), 'keys_equal': a[['fold', 'timestamp_utc']].astype(str).equals(b[['fold', 'timestamp_utc']].astype(str)),
              'bitwise_equal': bool(np.array_equal(a[num].to_numpy(float), b[num].to_numpy(float))),
              'max_abs_difference': float(np.max(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float)))),
              'admission_vectors': sum(1 for r in lineage['origins'] if r['phase'] == 'training_only' and r['predicted']),
              'sources': pd.Series([r['source'] for r in lineage['origins']]).value_counts().to_dict(),
              'policy_days_charged': len(lineage['origins'])}
    atomic(out_path, result)
    print(json.dumps(result), flush=True)
