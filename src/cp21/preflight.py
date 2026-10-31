"""CP-21 pre-run work (§14.6 E1-E4 for CP-21, §17.8, §17.10 items 1-2).

* ``verify-inputs``: issued/inherited identities, the recomputed CP-20 HG identity, the 638-origin
  manifest, the frozen weather design, every cached HG component against its CP-20 fingerprint and
  every evaluation-day HG central against the accepted CP-20 vectors, bit for bit.
* ``verify-weather``: the frozen weather features regenerated from the retained decoded grids
  through CP-20's frozen conversion, compared with the committed features.
* ``benchmark``: fit timing on training-only data (one fold-5 warm-up origin, the full 728-day
  window); accuracy-blind (no validation loss is computed), every fit charged.
* ``e1``: origin, row-sufficiency, fit and replay enumeration for every origin and model.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from cp15.data import array_hash
from cp20.components import HGComponents
from .budget import ledger
from .inputs import hg_identity, identities, load, origin_manifest, weather_design, weather_matrix
from .jobs import art, stamp, write_json
from .lgbm import ARMS, FITS_PER_MODEL, GRID, MIN_TRAIN_ROWS, MIN_VALIDATION_ROWS, MODEL_HOURS, _fit, _model_sha, arm_design, hg_central, model_rows

HG_CACHE = 'hg-components'


def cache_dir() -> Path:
    return art().parents[0] / 'cp-20' / HG_CACHE


def verify_inputs(root: Path, rest) -> int:
    started = time.time()
    ids = identities(root)
    ident = hg_identity(root)
    m = origin_manifest(root)
    design = weather_design(root)
    data = load(root)
    accepted = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet', filters=[('policy', '==', 'HG')])
    accepted['timestamp_utc'] = pd.to_datetime(accepted.timestamp_utc, utc=True)
    by_day = {d: part.sort_values('timestamp_utc') for d, part in accepted.groupby(pd.to_datetime(accepted.delivery_date).dt.date)}
    checked, sources, evaluation_equal, evaluation_days = 0, {}, 0, 0
    for f in m['folds']:
        cache = HGComponents(cache_dir(), f['fold'], data, ident)
        first = date.fromisoformat(f['evaluation_start'])
        for o in f['origins']:
            day = date.fromisoformat(o['day'])
            item = cache.load(day)  # raises on a stale/wrong entry
            if item is None:
                raise ValueError(f'HG component cache miss {f["fold"]} {day}')
            checked += 1
            sources[item['source']] = sources.get(item['source'], 0) + 1
            if day >= first and len(data.rows(day)):
                rows, centers, _ = cache.get(day)
                central = hg_central(centers['A1'], centers['B2'])
                acc = by_day[day]
                if not pd.DatetimeIndex(acc.timestamp_utc).equals(data.index[rows]):
                    raise ValueError(f'accepted HG keys differ {day}')
                evaluation_days += 1
                evaluation_equal += int(np.array_equal(central, acc.central.to_numpy(float)))
    if evaluation_equal != evaluation_days:
        raise ValueError(f'HG cache centrals differ from accepted CP-20 vectors on {evaluation_days - evaluation_equal} days')
    out = {'schema': 'cp21-input-verification-v1', 'written_utc': stamp(),
           'identities_sha256': ids, 'hg_identity': ident,
           'origin_manifest': {'origins': sum(f['date_count'] for f in m['folds']),
                               'folds': {f['fold']: {'origins': f['date_count'], 'warmup_start': f['warmup_start'],
                                                     'evaluation_start': f['evaluation_start'], 'evaluation_end': f['evaluation_end'],
                                                     'admission_dates': [a['day'] for a in f['admission']],
                                                     'original_target_keys': len(f['original_target_keys'])}
                                         for f in m['folds']}},
           'weather_design_sha256': design.sha256,
           'hg_cache': {'entries_verified': checked, 'sources': sources,
                        'evaluation_days_bitwise_equal_to_accepted_hg_central': evaluation_equal,
                        'evaluation_days_with_hours': evaluation_days,
                        'rule': 'content hash, CP-20 identity (input fingerprint, weather design, protocol), origin, '
                                'canonical timestamps and scale hash per entry; evaluation-day A1/2+B2/2 equals the '
                                'accepted HG central bit for bit'},
           'seconds': time.time() - started}
    write_json(art() / 'preflight' / 'input-verification.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'identities_sha256'}, default=str)[:2000], flush=True)
    return 0


def verify_weather(root: Path, rest) -> int:
    """Regenerate the frozen features from the retained grids through CP-20's frozen conversion."""
    from cp20.weather import COLUMNS, WeatherDesign, build_features
    started = time.time()
    manifest = json.loads((root / 'reports/weather-ablation/run-manifest.json').read_text())
    weather = art().parents[0] / 'cp-20' / 'weather'
    rebuilt = build_features(manifest, weather, {})
    committed = pd.read_parquet(root / 'reports/weather-ablation/weather-features.parquet')
    design = WeatherDesign.from_features(rebuilt)
    a = rebuilt.sort_values('timestamp_utc').reset_index(drop=True)
    b = committed.sort_values('timestamp_utc').reset_index(drop=True)
    # Compare instants, not storage units: a Parquet round trip may store microseconds.
    ta = pd.DatetimeIndex(pd.to_datetime(a.timestamp_utc, utc=True)).as_unit('ns')
    tb = pd.DatetimeIndex(pd.to_datetime(b.timestamp_utc, utc=True)).as_unit('ns')
    keys_equal = bool(len(a) == len(b) and ta.equals(tb)
                      and (a.delivery_date.astype(str).to_numpy() == b.delivery_date.astype(str).to_numpy()).all())
    values_equal = bool(np.array_equal(a[list(COLUMNS)].to_numpy(float), b[list(COLUMNS)].to_numpy(float), equal_nan=True))
    status_equal = bool((a.status == b.status).all())
    out = {'schema': 'cp21-weather-regeneration-v1', 'written_utc': stamp(), 'runs': len(manifest['runs']),
           'structural_missing_days': manifest.get('structural_missing', []), 'feature_hours': int(len(a)),
           'keys_equal': bool(keys_equal), 'values_bitwise_equal': values_equal, 'status_equal': status_equal,
           'rebuilt_design_sha256': design.sha256, 'committed_design_sha256': weather_design(root).sha256,
           'retained_grids': str(weather.relative_to(art().parents[2])), 'seconds': time.time() - started}
    write_json(art() / 'preflight' / 'weather-regeneration.json', out)
    print(json.dumps(out, default=str), flush=True)
    if not (keys_equal and values_equal and status_equal and out['rebuilt_design_sha256'] == out['committed_design_sha256']):
        raise ValueError('weather regenerated from the retained grids differs from the frozen features')
    return 0


def _charge(budget, purpose: str, main: bool):
    def charge(role: str):
        extra = {'main_lgbm_fits': 1, 'lgbm_fits_main': 1} if main else {f'lgbm_fits_{purpose}': 1}
        budget.reserve(lgbm_fits=1, **{f'lgbm_{role}_fits': 1}, **extra)
    return charge


def benchmark(root: Path, rest) -> int:
    """Accuracy-blind timing: fits only, no validation predictions or losses."""
    budget = ledger()
    charge = _charge(budget, 'benchmark', main=False)
    m = origin_manifest(root)
    fold = m['folds'][4]
    first = date.fromisoformat(fold['evaluation_start'])
    day = date.fromisoformat(fold['warmup_start'])
    t0 = time.perf_counter()
    data = load(root, before=first)
    design_w = weather_design(root)
    wx, present = weather_matrix(design_w, data)
    load_seconds = time.perf_counter() - t0
    records = []
    for arm in ('L-P', 'L-R', 'L-N'):
        design = arm_design(data, wx, arm)
        for model in ARMS[arm]['models']:
            lower, window, inner, validation, forecast = model_rows(data, day, model, present)
            for config in GRID:
                charge('inner')
                fitted, _, wall, cpu = _fit(design, inner, config, data.p, 1)
                records.append({'arm': arm, 'model': model, 'role': 'inner', 'config': config['id'], 'n_train': int(len(inner)),
                                'wall': wall, 'cpu': cpu, 'model_sha256': _model_sha(fitted)})
            charge('final')
            fitted, _, wall, cpu = _fit(design, window, GRID[-1], data.p, 1)
            records.append({'arm': arm, 'model': model, 'role': 'final_largest', 'config': GRID[-1]['id'],
                            'n_train': int(len(window)), 'wall': wall, 'cpu': cpu, 'model_sha256': _model_sha(fitted)})
    # Thread-count determinism and repeatability on the largest pooled fit.
    design = arm_design(data, wx, 'L-P')
    lower, window, *_ = model_rows(data, day, 'pooled', present)
    threads = {}
    for n_jobs in (4, 1):
        charge('final')
        fitted, _, wall, cpu = _fit(design, window, GRID[-1], data.p, n_jobs)
        threads[n_jobs] = {'wall': wall, 'cpu': cpu, 'model_sha256': _model_sha(fitted)}
    single = next(r for r in records if r['arm'] == 'L-P' and r['role'] == 'final_largest')
    frame = pd.DataFrame(records)
    per_origin = float(frame.wall.sum())
    out = {'schema': 'cp21-benchmark-v1', 'written_utc': stamp(), 'machine': platform.platform(),
           'origin': str(day), 'fold': fold['fold'], 'data': 'training-only, materialised before ' + str(first),
           'load_and_design_seconds': load_seconds, 'records': records,
           'thread_determinism': {'n_jobs_1_sha256': threads[1]['model_sha256'], 'n_jobs_4_sha256': threads[4]['model_sha256'],
                                  'repeat_n_jobs_1_sha256': single['model_sha256'],
                                  'identical': threads[1]['model_sha256'] == threads[4]['model_sha256'] == single['model_sha256'],
                                  'wall_n_jobs_1': threads[1]['wall'], 'wall_n_jobs_4': threads[4]['wall']},
           'per_origin_single_thread_wall_seconds_all_three_arms': per_origin,
           'by_arm_wall_seconds': frame.groupby('arm').wall.sum().to_dict(),
           'by_config_wall_seconds': {f'{m}/{c}': float(v) for (m, c), v in frame.groupby(['model', 'config']).wall.mean().items()}}
    write_json(art() / 'preflight' / 'benchmark.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'records'}, default=str), flush=True)
    return 0


def determinism(root: Path, rest) -> int:
    """Repeatability at the inherited n_jobs=4, and whether the thread count changes the fitted
    trees (charged; training-only data; each case's forecasts of its own origin day are compared
    between fits, never scored). Every capacity configuration, pooled and block, raw and normalised,
    at two origins with different history lengths."""
    from .lgbm import _predict
    budget = ledger()
    charge = _charge(budget, 'benchmark', main=False)
    m = origin_manifest(root)
    out = {'schema': 'cp21-thread-determinism-v1', 'written_utc': stamp(), 'cases': []}
    for fold in (m['folds'][0], m['folds'][4]):
        first = date.fromisoformat(fold['evaluation_start'])
        day = date.fromisoformat(fold['warmup_start'])
        data = load(root, before=first)
        wx, present = weather_matrix(weather_design(root), data)
        for arm, model in (('L-P', 'pooled'), ('L-R', 'solar'), ('L-N', 'night')):
            design = arm_design(data, wx, arm)
            lower, window, inner, validation, forecast = model_rows(data, day, model, present)
            for config in GRID:
                fits = {}
                for label, n_jobs in (('a4', 4), ('b4', 4), ('c1', 1)):
                    charge('final')
                    fitted, imputer, wall, cpu = _fit(design, window, config, data.p, n_jobs)
                    fits[label] = (_model_sha(fitted), _predict(fitted, imputer, design, forecast, data), wall, cpu)
                out['cases'].append({
                    'origin': str(day), 'fold': fold['fold'], 'arm': arm, 'model': model, 'config': config['id'],
                    'n_train': int(len(window)),
                    'n_jobs_4_repeat_trees_identical': fits['a4'][0] == fits['b4'][0],
                    'n_jobs_1_vs_4_trees_identical': fits['a4'][0] == fits['c1'][0],
                    'n_jobs_1_vs_4_forecasts_bitwise_identical': bool(np.array_equal(fits['a4'][1], fits['c1'][1])),
                    'tree_sha256': fits['a4'][0],
                    'wall_seconds': {k: v[2] for k, v in fits.items()}, 'cpu_seconds': {k: v[3] for k, v in fits.items()}})
    cases = out['cases']
    out['all_repeat_identical'] = all(c['n_jobs_4_repeat_trees_identical'] for c in cases)
    out['all_thread_counts_identical'] = all(c['n_jobs_1_vs_4_trees_identical'] and c['n_jobs_1_vs_4_forecasts_bitwise_identical'] for c in cases)
    out['model_string_difference'] = ('only the parameter line [num_threads: N] after the trees differs '
                                      '(tree hash = model string through its end-of-trees marker)')
    write_json(art() / 'preflight' / 'thread-determinism.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'cases'}, default=str), flush=True)
    return 0


def enumerate_e1(root: Path, rest) -> int:
    """E1: every origin's rows, block sufficiency and weather presence; fit and replay counts."""
    m = origin_manifest(root)
    design_w = weather_design(root)
    rows_out, problems = [], []
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        admission = {a['day'] for a in f['admission']}
        early = load(root, before=first)
        full = None
        for o in f['origins']:
            day = date.fromisoformat(o['day'])
            if day >= first and full is None:
                full = load(root)
            data = early if day < first else full
            wx, present = weather_matrix(design_w, data)
            forecast = data.rows(day)
            record = {'fold': f['fold'], 'day': str(day), 'phase': 'evaluation' if day >= first else
                      ('admission' if str(day) in admission else 'warmup'), 'n_forecast': int(len(forecast)),
                      'forecast_hours': sorted(set(int(h) for h in data.hours[forecast]))}
            for model in MODEL_HOURS:
                lower, window, inner, validation, fc = model_rows(data, day, model, present)
                ok = (len(window) >= MIN_TRAIN_ROWS[model] and len(inner) >= MIN_TRAIN_ROWS[model]
                      and len(validation) >= MIN_VALIDATION_ROWS[model] and present[window].all() and present[fc].all())
                record[model] = {'window': int(len(window)), 'inner': int(len(inner)), 'validation': int(len(validation)),
                                 'forecast': int(len(fc)), 'history_start': str(lower),
                                 'window_weather_missing_rows': int((~np.isfinite(wx[window])).any(axis=1).sum()),
                                 'weather_record_present': bool(present[window].all() and present[fc].all()),
                                 'sufficient': bool(ok) if len(fc) else None}
                if len(fc) and not ok:
                    problems.append(f'{f["fold"]} {day} {model}')
            rows_out.append(record)
    with_rows = [r for r in rows_out if r['n_forecast']]
    models_per_origin = sum(len(a['models']) for a in ARMS.values())
    out = {'schema': 'cp21-e1-v1', 'written_utc': stamp(), 'origins': len(rows_out),
           'origins_with_eligible_hours': len(with_rows),
           'origins_without_eligible_hours': [f"{r['fold']} {r['day']}" for r in rows_out if not r['n_forecast']],
           'phase_counts': pd.Series([r['phase'] for r in rows_out]).value_counts().to_dict(),
           'models_per_origin': models_per_origin, 'fits_per_model': FITS_PER_MODEL,
           'main_fits': len(with_rows) * models_per_origin * FITS_PER_MODEL,
           'policy_days_main_replay': 4 * len(rows_out), 'policy_days_hg_parity_replay': len(rows_out),
           'sufficiency_problems': problems, 'origins_detail': rows_out}
    write_json(art() / 'preflight' / 'e1.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'origins_detail'}, default=str), flush=True)
    return 0 if not problems else 4
