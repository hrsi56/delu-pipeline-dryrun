"""CP-23 pre-run work (§14.6 E1-E4 for CP-23; §21.8; §21.10 item 6).

* ``verify-inputs`` checks, bit for bit:
  - the issued and inherited identities, and CP-21's recomputed fit identity;
  - every cached HG component against its CP-20 fingerprint;
  - every retained CP-21 L-P/L-N/L-R entry (warm-up included) against CP-21's identity and its
    committed lineage hash, and v4's composite against the committed lineage hash at every origin;
  - every evaluation-day HG, v4, L-N and L-R central against the committed predictions.

  These are the v3 and v4 members v5 and v3+D are built from, at every origin, warm-up included.
* ``verify-weather``: the frozen weather features regenerated from CP-20's retained decoded grids
  through the frozen conversion, and compared with the committed features (no retrieval).
* ``e1``: every origin's DDNN rows (training, early stopping, forecast) and weather presence, every
  fold's selection rows, and the fit and replay counts.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cp15.data import array_hash
from cp20.components import HGComponents
from cp21.execution import FitCache as CP21FitCache
from cp21.lgbm import hg_central, hgl_central
from cp22.execution import cp21_lineage_hashes
from . import ddnn as D
from .features import InsufficientRows, origin_rows, rows_record, selection_rows
from .inputs import cp21_fit_identity, hg_identity, identities, load, origin_manifest, weather_design, weather_matrix
from .jobs import art, cp20_art, cp21_art, stamp, write_json

POLICIES_REPLAYED = ('v5', 'v3+D', 'D')


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
    weather_saved = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet')
    cp21_acc = pd.read_parquet(root / 'reports/block-challenger/predictions.parquet')
    counts = cp21_acc.groupby('policy').size().to_dict()
    saved_counts = weather_saved.groupby('policy').size().to_dict()
    if counts != {'HGL': 10747, 'L-N': 10747, 'L-P': 10747, 'L-R': 10747} or any(
            saved_counts.get(p) != 10747 for p in ('B0', 'B1', 'B2', 'B3', 'A1', 'HG')):
        raise ValueError(f'committed vector counts differ: {counts} {saved_counts}')
    accepted = {}
    for policy, frame in (('HG', weather_saved.loc[weather_saved.policy.eq('HG')]),
                          *((p, cp21_acc.loc[cp21_acc.policy.eq(p)]) for p in ('HGL', 'L-P', 'L-N', 'L-R'))):
        frame = frame.assign(timestamp_utc=pd.to_datetime(frame.timestamp_utc, utc=True))
        accepted[policy] = {d: part.sort_values('timestamp_utc')
                            for d, part in frame.groupby(pd.to_datetime(frame.delivery_date).dt.date)}
    tally = {'hg_entries': 0, 'cp21_entries': 0, 'evaluation_days': 0, 'warmup_days': 0,
             'bitwise_equal_evaluation_vectors': {p: 0 for p in accepted},
             'lineage_hash_equal': {p: 0 for p in ('HGL', 'L-P', 'L-N', 'L-R')}}
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
    out = {'schema': 'cp23-input-verification-v1', 'written_utc': stamp(), 'identities_sha256': ids,
           'hg_identity': hg_ident, 'cp21_fit_identity': cp21_ident,
           'origin_manifest': {'origins': sum(f['date_count'] for f in m['folds']),
                               'folds': {f['fold']: {'origins': f['date_count'], 'warmup_start': f['warmup_start'],
                                                     'evaluation_start': f['evaluation_start'], 'evaluation_end': f['evaluation_end'],
                                                     'admission_dates': [a['day'] for a in f['admission']],
                                                     'original_target_keys': len(f['original_target_keys'])} for f in m['folds']}},
           'saved_reference_counts': {**{p: saved_counts[p] for p in ('B0', 'B1', 'B2', 'B3', 'A1', 'HG')}, 'HGL': counts['HGL']},
           'weather_design_sha256': design.sha256, 'max_materialised_date': str(max(full.dates)), 'tally': tally,
           'rule': 'HG: CP-20 content hash, identity, origin, timestamps and scale hash per entry, evaluation centrals bitwise '
                   'against reports/weather-ablation/predictions.parquet. CP-21: content hash and the recomputed CP-21 fit identity '
                   'per entry; every L-P/L-N/L-R central and the HGL composite equal to the central_sha256 in CP-21\'s committed '
                   'lineage (warm-up and evaluation); evaluation centrals bitwise against reports/block-challenger/predictions.parquet. '
                   'Saved reference vectors (B0, B1, B2, B3, A1, HG, HGL): committed blobs equal to their manifests and evidence tags.',
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
    out = {'schema': 'cp23-weather-regeneration-v1', 'written_utc': stamp(), 'runs': len(manifest['runs']),
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


def enumerate_e1(root: Path, rest) -> int:
    """E1: every origin's DDNN rows and weather presence, every fold's selection rows, fit and replay counts."""
    m = origin_manifest(root)
    design_w = weather_design(root)
    rows_out, problems, selection = [], [], {}
    full = load(root)
    full_w = weather_matrix(design_w, full)
    for f in m['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        admission = {a['day'] for a in f['admission']}
        early = load(root, before=first)
        early_w = weather_matrix(design_w, early)
        d0 = date.fromisoformat(f['warmup_start'])
        try:
            selection[f['fold']] = {'first_origin': str(d0), **rows_record(early, selection_rows(early, d0, early_w[1]))}
        except InsufficientRows as exc:
            problems.append(f'{f["fold"]} selection: {exc}')
        for o in f['origins']:
            day = date.fromisoformat(o['day'])
            data, (wx, present) = (early, early_w) if day < first else (full, full_w)
            forecast = data.rows(day)
            record = {'fold': f['fold'], 'day': str(day), 'phase': 'evaluation' if day >= first else
                      ('admission' if str(day) in admission else 'warmup'), 'n_forecast': int(len(forecast))}
            if len(forecast):
                try:
                    rows = origin_rows(data, day, present)
                    record['ddnn'] = {**rows_record(data, rows), 'sufficient': True,
                                      'window_weather_missing_rows': int((~np.isfinite(wx[np.concatenate([rows['train'], rows['stop']])])).any(axis=1).sum())}
                except InsufficientRows as exc:
                    record['ddnn'] = {'sufficient': False, 'problem': str(exc)}
                    problems.append(f'{f["fold"]} {day}: {exc}')
            rows_out.append(record)
    with_rows = [r for r in rows_out if r['n_forecast']]
    n = len(rows_out)
    main_fits = len(with_rows) * len(D.SEEDS) + len(m['folds']) * len(D.CONFIGS) * len(D.SEEDS)
    out = {'schema': 'cp23-e1-v1', 'written_utc': stamp(), 'origins': n, 'origins_with_eligible_hours': len(with_rows),
           'origins_without_eligible_hours': [f"{r['fold']} {r['day']}" for r in rows_out if not r['n_forecast']],
           'phase_counts': pd.Series([r['phase'] for r in rows_out]).value_counts().to_dict(),
           'seeds': len(D.SEEDS), 'configurations': len(D.CONFIGS),
           'fits': {'selection': len(m['folds']) * len(D.CONFIGS) * len(D.SEEDS), 'origins': len(with_rows) * len(D.SEEDS),
                    'main_total': main_fits},
           'policy_days': {'new_policies_admission_and_evaluation': len(POLICIES_REPLAYED) * n, 'hg_and_v4_parity': 2 * n},
           'selection': selection, 'sufficiency_problems': problems, 'origins_detail': rows_out}
    write_json(art() / 'preflight' / 'e1.json', out)
    print(json.dumps({k: v for k, v in out.items() if k not in ('origins_detail', 'selection')}, default=str), flush=True)
    return 0 if not problems else 4
