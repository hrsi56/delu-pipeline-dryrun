"""CP-22 pre-run work (§14.6 E1-E4 for CP-22, §20.8, §20.10 items 1-2).

* ``verify-inputs``: issued/inherited identities; CP-21's recomputed fit identity; every cached HG
  component against its CP-20 fingerprint and every evaluation-day HG central against the accepted
  CP-20 vectors; every retained CP-21 L-P/L-N/L-R entry (warm-up included) against CP-21's
  identity and its committed lineage hash, every evaluation-day vector against CP-21's committed
  predictions, and v4's composite against the committed v4 central -- bit for bit.
* ``verify-weather``: the frozen weather features regenerated from CP-20's retained decoded grids
  through the frozen conversion, compared with the committed features.
* ``benchmark``: PN fit timing and thread-count determinism on training-only data; accuracy-blind
  (no validation loss is computed), every fit charged.
* ``e1``: origin, row-sufficiency, fit and replay enumeration for every origin.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

from cp20.components import HGComponents
from cp21.execution import FitCache as CP21FitCache
from cp21.lgbm import GRID, MIN_TRAIN_ROWS, MIN_VALIDATION_ROWS, POOLED, hg_central, hgl_central, model_rows
from cp15.data import array_hash
from .budget import ledger
from .execution import FIXED_NEW, W_POLICIES, charge_fits, cp21_lineage_hashes
from .inputs import cp21_fit_identity, hg_identity, identities, load, origin_manifest, weather_design, weather_matrix
from .jobs import art, cp20_art, cp21_art, stamp, write_json
from .pn import FITS_PER_ORIGIN, _fit, _model_sha, _z, pn_design


def verify_inputs(root: Path, rest) -> int:
    started = time.time()
    ids = identities(root)
    hg_ident, cp21_ident = hg_identity(root), cp21_fit_identity(root)
    hashes = cp21_lineage_hashes(root)
    m = origin_manifest(root)
    design = weather_design(root)
    full = load(root)
    if max(full.dates) > np.datetime64('2026-04-07'):
        raise ValueError('materialised data past 2026-04-07')
    hg_acc = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet', filters=[('policy', '==', 'HG')])
    cp21_acc = pd.read_parquet(root / 'reports/block-challenger/predictions.parquet')
    counts = cp21_acc.groupby('policy').size().to_dict()
    if counts != {'HGL': 10747, 'L-N': 10747, 'L-P': 10747, 'L-R': 10747} or len(hg_acc) != 10747:
        raise ValueError(f'committed vector counts differ: {counts}')
    accepted = {}
    for policy, frame in (('HG', hg_acc), *((p, cp21_acc.loc[cp21_acc.policy.eq(p)]) for p in ('HGL', 'L-P', 'L-N', 'L-R'))):
        frame = frame.assign(timestamp_utc=pd.to_datetime(frame.timestamp_utc, utc=True))
        accepted[policy] = {d: part.sort_values('timestamp_utc')
                            for d, part in frame.groupby(pd.to_datetime(frame.delivery_date).dt.date)}
    tally = {'hg_entries': 0, 'cp21_entries': 0, 'evaluation_days': 0, 'warmup_days': 0,
             'bitwise_equal_evaluation_vectors': {p: 0 for p in accepted}, 'lineage_hash_equal': {p: 0 for p in ('HGL', 'L-P', 'L-N', 'L-R')}}
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        early = load(root, before=first)
        for o in f['origins']:
            day = date.fromisoformat(o['day'])
            data = full if day >= first else early
            hg = HGComponents(cp20_art() / 'hg-components', f['fold'], data, hg_ident)
            if hg.load(day) is None:
                raise ValueError(f'HG component cache miss {f["fold"]} {day}')
            tally['hg_entries'] += 1
            rows = data.rows(day)
            if not len(rows):
                continue
            _, comps, _ = hg.get(day)
            cache = CP21FitCache(f['fold'], data, cp21_ident, cp21_art() / 'fits')
            vec = {arm: cache.get(day, arm) for arm in ('L-P', 'L-N', 'L-R')}
            tally['cp21_entries'] += 3
            vec['HGL'] = hgl_central(comps['A1'], comps['B2'], vec['L-N'], vec['L-R'])
            vec['HG'] = hg_central(comps['A1'], comps['B2'])
            for arm in ('HGL', 'L-P', 'L-N', 'L-R'):
                if array_hash(vec[arm]) != hashes[(arm, f['fold'], str(day))]:
                    raise ValueError(f'{arm} {f["fold"]} {day} differs from CP-21 committed lineage')
                tally['lineage_hash_equal'][arm] += 1
            if day >= first:
                tally['evaluation_days'] += 1
                for policy, by_day in accepted.items():
                    acc = by_day[day]
                    if not pd.DatetimeIndex(acc.timestamp_utc).equals(data.index[rows]):
                        raise ValueError(f'accepted {policy} keys differ {day}')
                    tally['bitwise_equal_evaluation_vectors'][policy] += int(np.array_equal(vec[policy], acc.central.to_numpy(float)))
            else:
                tally['warmup_days'] += 1
    for policy, n in tally['bitwise_equal_evaluation_vectors'].items():
        if n != tally['evaluation_days']:
            raise ValueError(f'{policy}: {tally["evaluation_days"] - n} evaluation days differ from committed vectors')
    out = {'schema': 'cp22-input-verification-v1', 'written_utc': stamp(), 'identities_sha256': ids,
           'hg_identity': hg_ident, 'cp21_fit_identity': cp21_ident,
           'origin_manifest': {'origins': sum(f['date_count'] for f in m['folds']),
                               'folds': {f['fold']: {'origins': f['date_count'], 'warmup_start': f['warmup_start'],
                                                     'evaluation_start': f['evaluation_start'], 'evaluation_end': f['evaluation_end'],
                                                     'admission_dates': [a['day'] for a in f['admission']],
                                                     'original_target_keys': len(f['original_target_keys'])} for f in m['folds']}},
           'weather_design_sha256': design.sha256, 'max_materialised_date': str(max(full.dates)), 'tally': tally,
           'rule': 'HG: CP-20 content hash, identity, origin, timestamps and scale hash per entry, evaluation centrals bitwise '
                   'against reports/weather-ablation/predictions.parquet. CP-21: content hash and the recomputed CP-21 fit identity per '
                   'entry; every L-P/L-N/L-R central and the HGL composite equal to the central_sha256 in CP-21\'s committed lineage '
                   '(warm-up and evaluation); evaluation centrals bitwise against reports/block-challenger/predictions.parquet.',
           'seconds': time.time() - started}
    write_json(art() / 'preflight' / 'input-verification.json', out)
    print(json.dumps({k: v for k, v in out.items() if k not in ('identities_sha256', 'origin_manifest')}, default=str), flush=True)
    return 0


def verify_weather(root: Path, rest) -> int:
    """Regenerate the frozen features from the retained grids through CP-20's frozen conversion."""
    from cp20.weather import COLUMNS, WeatherDesign, build_features
    started = time.time()
    manifest = json.loads((root / 'reports/weather-ablation/run-manifest.json').read_text())
    rebuilt = build_features(manifest, cp20_art() / 'weather', {})
    committed = pd.read_parquet(root / 'reports/weather-ablation/weather-features.parquet')
    design = WeatherDesign.from_features(rebuilt)
    a = rebuilt.sort_values('timestamp_utc').reset_index(drop=True)
    b = committed.sort_values('timestamp_utc').reset_index(drop=True)
    ta = pd.DatetimeIndex(pd.to_datetime(a.timestamp_utc, utc=True)).as_unit('ns')
    tb = pd.DatetimeIndex(pd.to_datetime(b.timestamp_utc, utc=True)).as_unit('ns')
    keys_equal = bool(len(a) == len(b) and ta.equals(tb)
                      and (a.delivery_date.astype(str).to_numpy() == b.delivery_date.astype(str).to_numpy()).all())
    values_equal = bool(np.array_equal(a[list(COLUMNS)].to_numpy(float), b[list(COLUMNS)].to_numpy(float), equal_nan=True))
    status_equal = bool((a.status == b.status).all())
    out = {'schema': 'cp22-weather-regeneration-v1', 'written_utc': stamp(), 'runs': len(manifest['runs']),
           'feature_hours': int(len(a)), 'keys_equal': keys_equal, 'values_bitwise_equal': values_equal,
           'status_equal': status_equal, 'rebuilt_design_sha256': design.sha256,
           'committed_design_sha256': weather_design(root).sha256, 'retained_grids': '.local/artifacts/cp-20/weather',
           'max_feature_date': str(pd.to_datetime(a.delivery_date).max().date()), 'retrieval': 'none',
           'seconds': time.time() - started}
    write_json(art() / 'preflight' / 'weather-regeneration.json', out)
    print(json.dumps(out, default=str), flush=True)
    if not (keys_equal and values_equal and status_equal and out['rebuilt_design_sha256'] == out['committed_design_sha256']):
        raise ValueError('weather regenerated from the retained grids differs from the frozen features')
    return 0


def benchmark(root: Path, rest) -> int:
    """Accuracy-blind PN timing (eight fits) and thread-count determinism (n_jobs 4, 4, 1) for every
    configuration at two training-only origins; forecasts are compared between fits, never scored."""
    budget = ledger()
    charge = charge_fits(budget, purpose='benchmark', main=False)
    m = origin_manifest(root)
    out = {'schema': 'cp22-benchmark-v1', 'written_utc': stamp(), 'machine': platform.platform(), 'timing': [], 'cases': []}
    for fold in (m['folds'][0], m['folds'][4]):
        first = date.fromisoformat(fold['evaluation_start'])
        day = date.fromisoformat(fold['warmup_start'])
        t0 = time.perf_counter()
        data = load(root, before=first)
        wx, present = weather_matrix(weather_design(root), data)
        design = pn_design(data, wx)
        lower, window, inner, validation, forecast = model_rows(data, day, POOLED, present)
        out.setdefault('load_and_design_seconds', {})[fold['fold']] = time.perf_counter() - t0
        for config in GRID:
            charge('inner')
            fitted, _, wall, cpu = _fit(design, inner, config, data.p, 1)
            out['timing'].append({'fold': fold['fold'], 'origin': str(day), 'role': 'inner', 'config': config['id'],
                                  'n_train': int(len(inner)), 'wall': wall, 'cpu': cpu, 'tree_sha256': _model_sha(fitted)})
            fits = {}
            for label, n_jobs in (('a4', 4), ('b4', 4), ('c1', 1)):
                charge('final')
                fitted, imputer, wall, cpu = _fit(design, window, config, data.p, n_jobs)
                fits[label] = (_model_sha(fitted), _z(fitted, imputer, design, forecast), wall, cpu)
            out['timing'].append({'fold': fold['fold'], 'origin': str(day), 'role': 'final', 'config': config['id'],
                                  'n_train': int(len(window)), 'wall': fits['c1'][2], 'cpu': fits['c1'][3], 'tree_sha256': fits['c1'][0]})
            out['cases'].append({'fold': fold['fold'], 'origin': str(day), 'config': config['id'], 'n_train': int(len(window)),
                                 'n_jobs_4_repeat_trees_identical': fits['a4'][0] == fits['b4'][0],
                                 'n_jobs_1_vs_4_trees_identical': fits['a4'][0] == fits['c1'][0],
                                 'n_jobs_1_vs_4_forecasts_bitwise_identical': bool(np.array_equal(fits['a4'][1], fits['c1'][1])),
                                 'wall_seconds': {k: v[2] for k, v in fits.items()}})
    timing = pd.DataFrame(out['timing'])
    out['per_origin_single_thread_wall_seconds'] = float(timing.groupby('fold').wall.sum().mean())
    out['by_config_wall_seconds'] = {f'{r}/{c}': float(v) for (r, c), v in timing.groupby(['role', 'config']).wall.mean().items()}
    out['all_thread_counts_identical'] = all(c['n_jobs_4_repeat_trees_identical'] and c['n_jobs_1_vs_4_trees_identical']
                                             and c['n_jobs_1_vs_4_forecasts_bitwise_identical'] for c in out['cases'])
    write_json(art() / 'preflight' / 'benchmark.json', out)
    print(json.dumps({k: v for k, v in out.items() if k not in ('timing', 'cases')}, default=str), flush=True)
    return 0 if out['all_thread_counts_identical'] else 7


def enumerate_e1(root: Path, rest) -> int:
    """E1: every origin's rows, pooled sufficiency and weather presence; fit and replay counts."""
    m = origin_manifest(root)
    design_w = weather_design(root)
    rows_out, problems = [], []
    full = load(root)
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        admission = {a['day'] for a in f['admission']}
        early = load(root, before=first)
        for o in f['origins']:
            day = date.fromisoformat(o['day'])
            data = early if day < first else full
            wx, present = weather_matrix(design_w, data)
            forecast = data.rows(day)
            lower, window, inner, validation, fc = model_rows(data, day, POOLED, present)
            ok = (len(window) >= MIN_TRAIN_ROWS[POOLED] and len(inner) >= MIN_TRAIN_ROWS[POOLED]
                  and len(validation) >= MIN_VALIDATION_ROWS[POOLED] and present[window].all() and present[fc].all())
            record = {'fold': f['fold'], 'day': str(day), 'phase': 'evaluation' if day >= first else
                      ('admission' if str(day) in admission else 'warmup'), 'n_forecast': int(len(forecast)),
                      'pooled': {'window': int(len(window)), 'inner': int(len(inner)), 'validation': int(len(validation)),
                                 'forecast': int(len(fc)), 'history_start': str(lower),
                                 'window_weather_missing_rows': int((~np.isfinite(wx[window])).any(axis=1).sum()),
                                 'weather_record_present': bool(present[window].all() and present[fc].all()),
                                 'sufficient': bool(ok) if len(fc) else None}}
            if len(fc) and not ok:
                problems.append(f'{f["fold"]} {day}')
            rows_out.append(record)
    with_rows = [r for r in rows_out if r['n_forecast']]
    n = len(rows_out)
    out = {'schema': 'cp22-e1-v1', 'written_utc': stamp(), 'origins': n, 'origins_with_eligible_hours': len(with_rows),
           'origins_without_eligible_hours': [f"{r['fold']} {r['day']}" for r in rows_out if not r['n_forecast']],
           'phase_counts': pd.Series([r['phase'] for r in rows_out]).value_counts().to_dict(),
           'fits_per_origin': FITS_PER_ORIGIN, 'main_fits': len(with_rows) * FITS_PER_ORIGIN,
           'policy_days': {'fixed_new_admission_and_evaluation': len(FIXED_NEW) * n, 'w_arms_if_a_winner': len(W_POLICIES) * n,
                           'hg_and_v4_parity': 2 * n},
           'sufficiency_problems': problems, 'origins_detail': rows_out}
    write_json(art() / 'preflight' / 'e1.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'origins_detail'}, default=str), flush=True)
    return 0 if not problems else 4
