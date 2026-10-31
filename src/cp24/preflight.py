"""CP-24's input verification (capstone v21-r11 §23.13 item 2; §14.6 E1-E2 for CP-24).

* ``verify-inputs`` checks:
  - every issued and inherited identity (`cp24.inputs.identities`, CP-23's reproduced from the objects
    at evidence/cp-23), CP-20's HG component identity and CP-21's recomputed fit identity;
  - the population: the frozen 638-origin manifest, the 10,747 keys and the fold counts;
  - the saved vectors CP-24 reads: CP-20's B0..B3, A1 and HG; CP-21's v4 (HGL), L-N and L-R; CP-23's
    D and its members table, whose `A1` and `B2` columns are HG's weather components A1_w and B2_w.
    Each file's bytes are bound to its manifest and evidence tag by the identities; here their
    contents are checked for completeness and for internal consistency (HG = (A1_w + B2_w)/2 and
    v4 = A1_w/3 + B2_w/3 + L-N/6 + L-R/6, bit for bit, on every key);
  - the frozen weather and its coverage: which delivery days have a frozen weather record, the gap
    2022-09-29..2023-03-24 (§23.6), and that every gate, warm-up and evaluation day is covered;
  - no retrieval, and nothing materialised after 2026-04-07.
* ``verify-weather``: the frozen features regenerated from CP-20's retained decoded grids through the
  frozen conversion, and compared with the committed features (no retrieval).
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from cp21.lgbm import hg_central, hgl_central
from . import design as G
from . import sampler as SP
from .inputs import cp21_fit_identity, hg_identity, identities, load, origin_manifest, weather_design
from .jobs import art, cp20_art, stamp, write_json

GAP = (date(2022, 9, 29), date(2023, 3, 24))
OUT = Path('reports/ddnn2')


def fold_table(root: Path) -> list[dict]:
    """The five folds' D0 (genuine warm-up start), evaluation window, gate window and batches."""
    m = origin_manifest(root)
    spans = [(date.fromisoformat(f['warmup_start']), date.fromisoformat(f['evaluation_end'])) for f in m['folds']]
    out = []
    for i, f in enumerate(m['folds'], start=1):
        d0 = date.fromisoformat(f['warmup_start'])
        g0, g1 = SP.gate_window(d0)
        out.append({'fold': f['fold'], 'index': i, 'd0': d0, 'evaluation_start': date.fromisoformat(f['evaluation_start']),
                    'evaluation_end': date.fromisoformat(f['evaluation_end']), 'gate': (g0, g1),
                    'batches': SP.batches(d0, spans), 'search_cutoff': d0 - timedelta(days=56)})
    return out


def _runs(days: list[date]) -> list[list[str]]:
    runs: list[list[date]] = []
    for d in sorted(days):
        if runs and (d - runs[-1][1]).days == 1:
            runs[-1][1] = d
        else:
            runs.append([d, d])
    return [[str(a), str(b), (b - a).days + 1] for a, b in runs]


def verify_inputs(root: Path, rest) -> int:
    started = time.time()
    ids = identities(root)
    hg_ident, cp21_ident = hg_identity(root), cp21_fit_identity(root)
    m = origin_manifest(root)
    folds = fold_table(root)
    design = weather_design(root)
    full = load(root)
    if max(full.dates) > np.datetime64('2026-04-07'):
        raise ValueError('materialised data past 2026-04-07')
    dd = G.build(full, design)
    # ---- the saved vectors
    keys = pd.read_parquet(root / 'reports/cp15/predictions.parquet', columns=['fold', 'timestamp_utc'],
                           filters=[('policy', '==', 'B0')])
    keys['timestamp_utc'] = pd.to_datetime(keys.timestamp_utc, utc=True)
    keyset = keys.set_index(['fold', 'timestamp_utc']).index
    if len(keyset) != 10747 or keyset.has_duplicates:
        raise ValueError('the 10,747 keys changed')
    weather_saved = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet')
    cp21_saved = pd.read_parquet(root / 'reports/block-challenger/predictions.parquet')
    cp23_saved = pd.read_parquet(root / 'reports/distribution-challenger/predictions.parquet')
    members = pd.read_parquet(root / 'reports/distribution-challenger/members.parquet')
    counts = {f'cp20:{p}': int(n) for p, n in weather_saved.groupby('policy').size().items()}
    counts.update({f'cp21:{p}': int(n) for p, n in cp21_saved.groupby('policy').size().items()})
    counts.update({f'cp23:{p}': int(n) for p, n in cp23_saved.groupby('policy').size().items()})
    counts['cp23:members'] = int(len(members))
    need = ['cp20:B0', 'cp20:B1', 'cp20:B2', 'cp20:B3', 'cp20:A1', 'cp20:HG', 'cp21:HGL', 'cp21:L-N', 'cp21:L-R', 'cp23:D',
            'cp23:members']
    if any(counts.get(k) != 10747 for k in need):
        raise ValueError(f'saved vector counts differ: {counts}')
    for frame in (weather_saved, cp21_saved, cp23_saved, members):
        frame['timestamp_utc'] = pd.to_datetime(frame.timestamp_utc, utc=True)

    def by_key(frame, policy=None):
        part = frame if policy is None else frame.loc[frame.policy.eq(policy)]
        part = part.set_index(['fold', 'timestamp_utc']).sort_index()
        if not part.index.equals(keyset.sort_values()):
            raise ValueError(f'saved vector keys differ ({policy})')
        return part

    mem = by_key(members)
    hg = by_key(weather_saved, 'HG')
    v4 = by_key(cp21_saved, 'HGL')
    ln, lr = by_key(cp21_saved, 'L-N'), by_key(cp21_saved, 'L-R')
    d = by_key(cp23_saved, 'D')
    checks = {
        'members_A1_B2_blend_equals_committed_HG': bool(np.array_equal(hg_central(mem['A1'], mem['B2']), hg.central.to_numpy())),
        'members_L-N_equals_committed_L-N': bool(np.array_equal(mem['L-N'].to_numpy(), ln.central.to_numpy())),
        'members_L-R_equals_committed_L-R': bool(np.array_equal(mem['L-R'].to_numpy(), lr.central.to_numpy())),
        'v4_composite_equals_committed_v4': bool(np.array_equal(
            hgl_central(mem['A1'], mem['B2'], mem['L-N'], mem['L-R']), v4.central.to_numpy())),
        'members_HG_and_HGL_columns_equal_committed': bool(np.array_equal(mem['HG'].to_numpy(), hg.central.to_numpy())
                                                          and np.array_equal(mem['HGL'].to_numpy(), v4.central.to_numpy())),
        'D_central_equals_members_D': bool(np.array_equal(d.central.to_numpy(), mem['D'].to_numpy())),
        'D_finite_ordered_p50_is_central': bool(np.isfinite(d[['p025', 'p10', 'p25', 'p50', 'p75', 'p90', 'p975']].to_numpy()).all()
                                                and (np.diff(d[['p025', 'p10', 'p25', 'p50', 'p75', 'p90', 'p975']].to_numpy(), axis=1) >= 0).all()
                                                and np.array_equal(d.p50.to_numpy(), d.central.to_numpy())),
        'truth_equal_across_saved_vectors': bool(np.array_equal(hg.y_true.to_numpy(), v4.y_true.to_numpy())
                                                 and np.array_equal(hg.y_true.to_numpy(), d.y_true.to_numpy())),
    }
    if not all(checks.values()):
        raise ValueError(f'saved-vector consistency failed: {checks}')
    # ---- weather coverage
    uncovered = [dd.date_of(i) for i in np.flatnonzero(~dd.covered)]
    runs = _runs(uncovered)
    if runs != [[str(GAP[0]), str(GAP[1]), (GAP[1] - GAP[0]).days + 1]]:
        raise ValueError(f'the uncovered delivery days differ from §23.6\'s gap: {runs}')
    must_cover = []
    for f in folds:
        g0, g1 = f['gate']
        must_cover += [g0 + timedelta(days=k) for k in range((g1 - g0).days + 1)]
        must_cover += [f['d0'] + timedelta(days=k) for k in range((f['evaluation_end'] - f['d0']).days + 1)]
        for b0, b1 in f['batches']:
            must_cover += [b0 + timedelta(days=k) for k in range(28)]
    not_covered = sorted({str(x) for x in must_cover if not dd.covered[dd.ix(x)] and dd.mask[dd.ix(x)].any()})
    if not_covered:
        raise ValueError(f'gate, warm-up, evaluation or batch days without a frozen weather record: {not_covered[:5]}')
    out = {'schema': 'cp24-input-verification-v1', 'written_utc': stamp(), 'identities_sha256': ids,
           'n_identities': len(ids), 'hg_identity': hg_ident, 'cp21_fit_identity': cp21_ident,
           'origin_manifest': {'origins': sum(f['date_count'] for f in m['folds']),
                               'folds': {f['fold']: {'origins': f['date_count'], 'warmup_start': f['warmup_start'],
                                                     'evaluation_start': f['evaluation_start'],
                                                     'evaluation_end': f['evaluation_end'],
                                                     'admission_dates': [a['day'] for a in f['admission']],
                                                     'original_target_keys': len(f['original_target_keys'])}
                                         for f in m['folds']}},
           'keys': int(len(keyset)), 'saved_vector_counts': counts, 'saved_vector_consistency': checks,
           'weather': {'design_sha256': design.sha256, 'uncovered_delivery_days': runs,
                       'gate_warmup_evaluation_and_batch_days_covered': True,
                       'rule': '§23.6: every pre-fold fit (4.6R\', the search, the gate) leaves out of its training window '
                               'every delivery day without a frozen weather record; no warm-up or evaluation fit changes'},
           'folds': [{**{k: str(v) for k, v in f.items() if k not in ('batches', 'gate', 'index')},
                      'gate': [str(f['gate'][0]), str(f['gate'][1])],
                      'batches': [[str(a), str(b)] for a, b in f['batches']], 'B': len(f['batches'])} for f in folds],
           'day_table_sha256': dd.sha256(), 'max_materialised_date': str(max(full.dates)),
           'retrieval': 'none', 'seconds': time.time() - started}
    write_json(art() / 'preflight' / 'input-verification.json', out)
    print(json.dumps({k: v for k, v in out.items() if k not in ('identities_sha256', 'origin_manifest', 'folds')},
                     default=str), flush=True)
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
    out = {'schema': 'cp24-weather-regeneration-v1', 'written_utc': stamp(), 'runs': len(manifest['runs']),
           'feature_hours': int(len(a)), 'keys_equal': keys_equal, 'values_bitwise_equal': values_equal,
           'status_equal': status_equal, 'rebuilt_design_sha256': design.sha256,
           'committed_design_sha256': weather_design(root).sha256, 'retained_grids': '.local/artifacts/cp-20/weather',
           'min_feature_date': str(pd.to_datetime(a.delivery_date).min().date()),
           'max_feature_date': str(pd.to_datetime(a.delivery_date).max().date()), 'retrieval': 'none',
           'seconds': time.time() - started}
    write_json(art() / 'preflight' / 'weather-regeneration.json', out)
    print(json.dumps(out, default=str), flush=True)
    if not (keys_equal and values_equal and status_equal and out['rebuilt_design_sha256'] == out['committed_design_sha256']):
        raise ValueError('weather regenerated from the retained grids differs from the frozen features')
    return 0
