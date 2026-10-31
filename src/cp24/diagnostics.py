"""§23.8's diagnostics for scored attempt k: descriptive, choosing nothing, delivered to `reports/ddnn2/`.

* D2's calibration: coverage by level and the PIT histogram;
* error correlations among D2, D, HG, A1_w, B2_w and L, pooled and by fold;
* MAE by local hour beside L and HG;
* extrapolation on extreme days, beside CP-23's record;
* the 2022 peak, fold 3 and fold 4;
* every guard activation and crossing (`guards.json`, written by the scoring job, summarised here), and
  per fold and stage (`guards-by-fold.csv`) and per fold and member (`guards-by-member.csv`): the cap's
  share of the members' emitted hour-levels (the gate's G3 measure, here over every warm-up and
  evaluation fit and never a condition), its concentration in member fits, beside the round's gate share;
* the search: each fold's ensemble, with its validation and gate scores beside its fold scores;
* member stability;
* a shape blend, descriptive only and never eligible: v4's p50 plus the average of the H layer's and
  D2's quantile offsets from their own medians;
* fit cost; the cold daily cycle is `cp24.daily`.

No forecast is produced or changed, and nothing here is fitted.
"""
from __future__ import annotations

from datetime import date
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from cp15.data import LABELS, history_start
from cp15.scoring import FOLDS, score_hourly
from . import ddnn2 as M
from .budget import atomic
from .evaluate import clean
from .execution import D2Cache, fit_identity
from .inputs import load, origin_manifest
from .jobs import art, stamp
from .protocol import attempt_dir, check_protocol

PIT_BINS = 20
CENTRAL = {'50': ('p25', 'p75'), '80': ('p10', 'p90'), '95': ('p025', 'p975')}
SHARE_TOP = 0.05
COLS = ['fold', 'policy', 'timestamp_utc', 'delivery_date', 'y_true', 'central', 'scale', 'level', *LABELS]


def _frame(root: Path, k: int) -> pd.DataFrame:
    parts = [pd.read_parquet(attempt_dir(root, k) / 'predictions.parquet')[COLS],
             pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet', filters=[('policy', '==', 'HG')])[COLS],
             pd.read_parquet(root / 'reports/block-challenger/predictions.parquet', filters=[('policy', '==', 'HGL')])[COLS],
             pd.read_parquet(root / 'reports/distribution-challenger/predictions.parquet', filters=[('policy', '==', 'D')])[COLS]]
    frame = pd.concat(parts, ignore_index=True)
    frame['timestamp_utc'] = pd.to_datetime(frame.timestamp_utc, utc=True).dt.as_unit('ns')
    frame['delivery_date'] = pd.to_datetime(frame.delivery_date)
    return frame


def _entries(root: Path, k: int) -> pd.DataFrame:
    """Per evaluation key: every member's Johnson SU parameters, transform, cap, centre and scale, and its
    emitted quantiles, from the verified attempt cache."""
    ident = fit_identity(root, k)
    data = load(root)
    rows = []
    for f in origin_manifest(root)['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        for o in f['origins']:
            d = date.fromisoformat(o['day'])
            if d < first or not len(data.rows(d)):
                continue
            item = D2Cache(k, f['fold'], data, ident).load(d)
            idx = data.rows(d)
            for j, i in enumerate(idx):
                h = int(data.hours[i])
                rows.append({'fold': f['fold'], 'timestamp_utc': data.index[i], 'delivery_date': pd.Timestamp(d), 'y': data.y[i],
                             'jsu': np.array([m['jsu'][h] for m in item['members']]),
                             'form': [m['transform'] for m in item['members']], 'cap': np.array([m['cap_z'] for m in item['members']]),
                             'centre': np.array([m['centre'] for m in item['members']]),
                             'scale': np.array([m['scale'] for m in item['members']]),
                             'member_q': np.array([m['quantiles'][j] for m in item['members']]),
                             'epochs': [m['record']['epochs_run'] for m in item['members']],
                             'best': [m['record']['best_epoch'] for m in item['members']],
                             'rank': [m['rank'] for m in item['members']], 'seed': [m['seed'] for m in item['members']]})
    return pd.DataFrame(rows)


def ensemble_pit(e: pd.DataFrame, iterations: int = 100) -> np.ndarray:
    """F_ens(y) for the per-level-median ensemble: p such that median_k Q_k(p) = y, by bisection in
    u = Phi^-1(p) (each member's capped quantile function is non-decreasing in u); returns Phi(u)."""
    jsu = np.stack(e.jsu.to_list())                      # n x members x 4
    cap = np.stack(e.cap.to_list())
    c, s = np.stack(e.centre.to_list()), np.stack(e.scale.to_list())
    asinh = np.array([[f.startswith('asinh') for f in forms] for forms in e.form])
    y = e.y.to_numpy(float)

    def q(u):
        with np.errstate(over='ignore'):
            t = jsu[..., 0] + jsu[..., 1] * np.sinh((u[:, None] - jsu[..., 2]) / jsu[..., 3])
            z = np.where(asinh, np.sinh(t), t)
        return np.median(c + s * np.clip(z, -cap, cap), axis=1)
    lo, hi = np.full(len(y), -40.0), np.full(len(y), 40.0)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        below = q(mid) < y
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return ndtr((lo + hi) / 2)


def calibration(frame, entries):
    d = frame.loc[frame.policy.eq('D2')].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    e = entries.sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    if not (np.array_equal(d.timestamp_utc.to_numpy(), pd.to_datetime(e.timestamp_utc, utc=True).dt.as_unit('ns').to_numpy())
            and np.array_equal(d.y_true.to_numpy(), e.y.to_numpy())):
        raise ValueError('cache entries do not align with the committed D2 vectors')
    rows = []
    for scope, part in [('pooled', d), *((f, d.loc[d.fold.eq(f)]) for f in FOLDS)]:
        rec = {'policy': 'D2', 'scope': scope, 'n_hours': int(len(part))}
        for level, label in zip(M.LEVELS, LABELS):
            rec[f'share_at_or_below_{label}'] = float((part.y_true <= part[label]).mean())
            rec[f'nominal_{label}'] = level
        for name, (lo, hi) in CENTRAL.items():
            width = part[hi] - part[lo]
            rec[f'coverage{name}'] = float(((part.y_true >= part[lo]) & (part.y_true <= part[hi])).mean())
            rec[f'mean_width{name}'] = float(width.mean())
            rec[f'median_width{name}'] = float(width.median())
            rec[f'p95_width{name}'] = float(width.quantile(0.95))
        rows.append(rec)
    pit = ensemble_pit(e)
    edges = np.linspace(0, 1, PIT_BINS + 1)
    hist = []
    for scope, mask in [('pooled', np.ones(len(e), bool)), *((f, (e.fold == f).to_numpy()) for f in FOLDS)]:
        counts, _ = np.histogram(pit[mask], bins=edges)
        for b in range(PIT_BINS):
            hist.append({'scope': scope, 'bin': b, 'lower': edges[b], 'upper': edges[b + 1], 'count': int(counts[b]),
                         'share': float(counts[b] / mask.sum()), 'uniform_share': 1 / PIT_BINS})
    agreement = {label: float(np.mean((pit <= level + 1e-9) == (d.y_true.to_numpy() <= d[label].to_numpy())))
                 for level, label in zip(M.LEVELS, LABELS)}
    return pd.DataFrame(rows), pd.DataFrame(hist), {'pit_mean': float(pit.mean()), 'pit_share_below_0.05': float((pit < 0.05).mean()),
                                                    'pit_share_above_0.95': float((pit > 0.95).mean()),
                                                    'pit_vs_emitted_quantile_agreement': agreement}


def correlations(frame, members) -> pd.DataFrame:
    m = members.copy()
    m['timestamp_utc'] = pd.to_datetime(m.timestamp_utc, utc=True).dt.as_unit('ns')
    y = frame.loc[frame.policy.eq('D2'), ['fold', 'timestamp_utc', 'y_true']]
    dd = frame.loc[frame.policy.eq('D'), ['fold', 'timestamp_utc', 'central']].rename(columns={'central': 'D'})
    t = m.merge(y, on=['fold', 'timestamp_utc']).merge(dd, on=['fold', 'timestamp_utc'])
    names = ['D2', 'D', 'HG', 'A1', 'B2', 'L']
    rows = []
    for scope, part in [('pooled', t), *((f, t.loc[t.fold.eq(f)]) for f in FOLDS)]:
        err = {n: (part[n] - part.y_true).to_numpy(float) for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                rows.append({'scope': scope, 'a': a, 'b': b, 'n_hours': int(len(part)), 'error_correlation': float(np.corrcoef(err[a], err[b])[0, 1])})
    return pd.DataFrame(rows)


def by_hour(frame, members) -> pd.DataFrame:
    m = members.copy()
    m['timestamp_utc'] = pd.to_datetime(m.timestamp_utc, utc=True).dt.as_unit('ns')
    y = frame.loc[frame.policy.eq('D2'), ['fold', 'timestamp_utc', 'y_true']]
    t = m.merge(y, on=['fold', 'timestamp_utc'])
    t['local_hour'] = t.timestamp_utc.dt.tz_convert('Europe/Berlin').dt.hour
    rows = []
    for scope, part in [('pooled', t), *((f, t.loc[t.fold.eq(f)]) for f in FOLDS)]:
        for h, g in part.groupby('local_hour'):
            rows.append({'scope': scope, 'local_hour': int(h), 'n_hours': int(len(g)),
                         **{f'MAE_{n}': float(np.mean(np.abs(g[n] - g.y_true))) for n in ('D2', 'L', 'HG', 'HGL')}})
    return pd.DataFrame(rows)


def extrapolation(root: Path, frame) -> pd.DataFrame:
    data = load(root)
    truth = frame.loc[frame.policy.eq('v5'), ['fold', 'timestamp_utc', 'delivery_date', 'y_true']]
    daily_max = truth.groupby(['fold', 'delivery_date']).y_true.max()
    cp23 = pd.read_csv(root / 'reports/distribution-challenger/diagnostics/extrapolation.csv')
    rows = []
    for fold in FOLDS:
        days = daily_max.xs(fold)
        top = set(days.sort_values(ascending=False).index[:math.ceil(SHARE_TOP * len(days))])
        for day, actual_max in days.items():
            d = day.date()
            lower = history_start(d, 'B3')
            window = (data.dates >= np.datetime64(lower)) & (data.dates < np.datetime64(d)) & data.eligible
            wmax = float(data.y[window].max())
            kinds = [k for k, hit in (('exceeds_window_max', actual_max > wmax), ('top_5pct_daily_max', day in top)) if hit]
            if not kinds:
                continue
            row = {'fold': fold, 'delivery_date': str(d), 'set': '+'.join(kinds), 'actual_max': float(actual_max),
                   'window_max': wmax, 'window_start': str(lower)}
            for p in ('D2', 'v5', 'v3+D2', 'HGL', 'HG', 'D'):
                part = frame.loc[frame.policy.eq(p) & frame.fold.eq(fold) & frame.delivery_date.eq(day)]
                row[f'max_central_{p}'] = float(part.central.max())
                row[f'hours_above_window_max_{p}'] = int((part.central > wmax).sum())
            dpart = frame.loc[frame.policy.eq('D2') & frame.fold.eq(fold) & frame.delivery_date.eq(day)]
            row['max_p975_D2'] = float(dpart.p975.max())
            row['hours_actual_above_D2_p975'] = int((dpart.y_true > dpart.p975).sum())
            rec = cp23.loc[cp23.fold.eq(fold) & cp23.delivery_date.eq(str(d))]
            row['cp23_max_central_D'] = float(rec['max_central_D'].iloc[0]) if len(rec) else np.nan
            row['cp23_max_p975_D'] = float(rec['max_p975_D'].iloc[0]) if len(rec) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def peak_fold3_fold4(root: Path, k: int):
    out = attempt_dir(root, k)
    metrics = pd.read_csv(out / 'metrics.csv')
    diag = pd.read_csv(out / 'diagnostics.csv', low_memory=False)
    cols = ['policy', 'n_hours', 'n_days', 'MAE', 'WIS', 'bias', 'coverage50', 'mean_width50', 'coverage80', 'mean_width80',
            'coverage95', 'mean_width95']
    keep = [c for c in cols if c in metrics.columns]
    parts = [diag.loc[diag.scope.eq('peak'), [c for c in keep if c in diag.columns]].assign(scope='peak 2022-08-15..31')]
    for fold, label in (('fold_3', 'fold_3 2022-07-01..09-28 (stress)'), ('fold_4', 'fold_4 2025-05-01..07-29')):
        parts.append(metrics.loc[metrics.scope.eq('per_fold') & metrics.fold.eq(fold), keep].assign(scope=label))
    unc = pd.read_csv(out / 'uncertainty.csv')
    intervals = unc.loc[unc.scope.isin(('fold_3', 'fold_4')), ['scope', 'candidate', 'baseline', 'metric', 'difference',
                                                                'ci_lower', 'ci_upper', 'role']]
    return pd.concat(parts, ignore_index=True), intervals


def search_beside_folds(root: Path, k: int) -> pd.DataFrame:
    p = json.loads((attempt_dir(root, k) / 'protocol.json').read_text())
    metrics = pd.read_csv(attempt_dir(root, k) / 'metrics.csv')
    pf = metrics.loc[metrics.scope.eq('per_fold') & metrics.policy.eq('D2')].set_index('fold')
    rows = []
    for fold, f in p['folds'].items():
        for m in f['ensemble'][::2]:
            rows.append({'fold': fold, 'rank': m['rank'], 'config': m['config']['id'], 'hidden': m['config']['hidden'],
                         'activation': m['config']['activation'], 'transform': m['config']['transform'],
                         'kappa': m['config']['kappa'], 'groups': ','.join(m['config']['groups']),
                         'validation_pinball': m['validation_pinball_all_batches'], 'validation_mae': m['validation_mae_all_batches'],
                         'gate_MAE_D2_fold': f['gate']['MAE_D2'], 'gate_MAE_L_fold': f['gate']['MAE_L'],
                         'fold_MAE_D2': float(pf.loc[fold, 'MAE']), 'fold_WIS_D2': float(pf.loc[fold, 'WIS'])})
    return pd.DataFrame(rows)


def member_stability(frame, entries) -> tuple[pd.DataFrame, pd.DataFrame]:
    mq = np.stack(entries.member_q.to_list())                  # n x members x 7
    med = mq[..., M.MEDIAN]
    y = entries.y.to_numpy(float)
    ens = np.median(med, axis=1)
    rows = []
    for scope, mask in [('pooled', np.ones(len(entries), bool)), *((f, (entries.fold == f).to_numpy()) for f in FOLDS)]:
        rec = {'scope': scope, 'n_hours': int(mask.sum()), 'ensemble_median_MAE': float(np.mean(np.abs(ens[mask] - y[mask]))),
               'member_median_spread_mean_abs': float(np.mean(np.abs(med[mask] - ens[mask, None]))),
               'share_hours_members_straddle_actual': float(np.mean((med[mask].min(axis=1) < y[mask]) & (med[mask].max(axis=1) > y[mask])))}
        for j in range(med.shape[1]):
            rec[f'member_{j}_median_MAE'] = float(np.mean(np.abs(med[mask, j] - y[mask])))
        rows.append(rec)
    ep = pd.DataFrame({'fold': np.repeat(entries.fold.to_numpy(), med.shape[1]),
                       'member': np.tile(np.arange(med.shape[1]), len(entries)),
                       'epochs_run': np.concatenate(entries.epochs.to_list()), 'best_epoch': np.concatenate(entries.best.to_list())})
    epochs = ep.groupby(['fold', 'member']).agg(best_epoch_median=('best_epoch', 'median'), best_epoch_p90=('best_epoch', lambda s: s.quantile(.9)),
                                                epochs_run_median=('epochs_run', 'median'), epochs_run_max=('epochs_run', 'max')).reset_index()
    return pd.DataFrame(rows), epochs


def shape_blend(frame) -> tuple[pd.DataFrame, dict]:
    """v4's p50 + the average of the H layer's (v4's) and D2's quantile offsets from their own medians."""
    v4 = frame.loc[frame.policy.eq('HGL')].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    d2 = frame.loc[frame.policy.eq('D2')].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    if not np.array_equal(v4.timestamp_utc.to_numpy(), d2.timestamp_utc.to_numpy()):
        raise ValueError('v4 and D2 keys differ')
    qv, qd = v4[LABELS].to_numpy(float), d2[LABELS].to_numpy(float)
    blend = qv[:, [3]] + 0.5 * (qv - qv[:, [3]]) + 0.5 * (qd - qd[:, [3]])
    crossed = int(np.any(np.diff(blend, axis=1) < 0, axis=1).sum())
    blend = np.sort(blend, axis=1)
    out = v4[['fold', 'timestamp_utc', 'delivery_date', 'y_true', 'scale', 'level']].copy()
    out['policy'], out['central'] = 'shape-blend', v4.central
    for j, lab in enumerate(LABELS):
        out[lab] = blend[:, j]
    rows = []
    for name, part in (('shape-blend', out), ('HGL', v4), ('D2', d2)):
        h = score_hourly(part.assign(timestamp_utc=part.timestamp_utc))
        for scope, g in [('pooled', h), *((f, h.loc[h.fold.eq(f)]) for f in FOLDS)]:
            rows.append({'policy': name, 'scope': scope, 'n_hours': int(len(g)), 'MAE': float(g.absolute_error.mean()),
                         'WIS': float(g.WIS.mean()), 'coverage95': float(g.hit95.mean()), 'mean_width95': float(g.width95.mean())})
    return pd.DataFrame(rows), {'crossings_restored': crossed, 'role': 'descriptive only, never eligible (§23.8)'}


GUARD_COUNTS = ('member_fits', 'cap_slot_levels_forecast', 'winsor_values_train', 'winsor_values_hold', 'winsor_values_forecast',
                'ensemble_crossings_restored', 'nonfinite_loss_stops')


def guards_by_fold(root: Path, k: int, guards: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every guard activation of the attempt's DDNN-2 fits, by fold and stage, and by fold and member.

    The cap's share is the gate's G3 measure (capped emitted hour-levels over the members' emitted
    hour-levels, each key taking its local-hour slot), here over every warm-up and evaluation fit; it is
    descriptive and never a condition. Concentration: the member fits with any capped hour-level, and the
    largest single member fit's share of the scope's capped hour-levels. The per-fold totals must equal
    the scoring job's `guards.json`, which reads the same member records."""
    protocol = json.loads((attempt_dir(root, k) / 'protocol.json').read_text())
    cells: dict = {}
    members: dict = {}
    for path in sorted((art() / f'attempt-{k}' / 'fits').glob('*/*.json')):
        item = json.loads(path.read_text())
        slots = pd.DatetimeIndex(pd.to_datetime(item['timestamp_utc'], utc=True)).tz_convert('Europe/Berlin').hour.to_numpy()
        for stage in (item['stage'], 'all'):
            c = cells.setdefault((item['fold'], stage), {'origins': 0, 'emitted_hour_levels': 0, 'cap_hour_levels': 0,
                                                         'member_fits_with_cap': 0, 'max_member_fit_cap_hour_levels': 0,
                                                         **{key: 0 for key in GUARD_COUNTS}})
            c['origins'] += 1
            c['ensemble_crossings_restored'] += item['crossed_rows']
            for mbr in item['members']:
                active = np.asarray(mbr['active'], int)                       # 24 slots x 7 levels
                g = mbr['guards']
                if int(active.sum()) != g['cap_forecast_slot_levels']:
                    raise ValueError(f'{path.name}: member {mbr["member"]} cap record disagrees with its guard count')
                capped = int(active[slots].sum())
                c['member_fits'] += 1
                c['emitted_hour_levels'] += len(slots) * active.shape[1]
                c['cap_hour_levels'] += capped
                c['member_fits_with_cap'] += capped > 0
                c['max_member_fit_cap_hour_levels'] = max(c['max_member_fit_cap_hour_levels'], capped)
                c['cap_slot_levels_forecast'] += g['cap_forecast_slot_levels']
                for side in ('train', 'hold', 'forecast'):
                    c[f'winsor_values_{side}'] += g[f'winsor_{side}']['winsor_low'] + g[f'winsor_{side}']['winsor_high']
                c['nonfinite_loss_stops'] += sum(e.get('event') == 'nonfinite_training_loss' for e in g['events'])
                if stage == 'all':
                    m = members.setdefault((item['fold'], mbr['member']), {'rank': mbr['rank'], 'config': mbr['config'],
                                                                           'seed': mbr['seed'], 'member_fits': 0,
                                                                           'emitted_hour_levels': 0, 'cap_hour_levels': 0,
                                                                           'member_fits_with_cap': 0})
                    m['member_fits'] += 1
                    m['emitted_hour_levels'] += len(slots) * active.shape[1]
                    m['cap_hour_levels'] += capped
                    m['member_fits_with_cap'] += capped > 0
    for fold, committed in guards['by_fold'].items():
        mine = cells[(fold, 'all')]
        if any(mine[key] != committed[key] for key in GUARD_COUNTS):
            raise ValueError(f'{fold}: guard counts differ from guards.json')
    rows = []
    for stage in ('warmup', 'evaluation', 'all'):
        pooled = {key: 0 for key in ('origins', 'emitted_hour_levels', 'cap_hour_levels', 'member_fits_with_cap', *GUARD_COUNTS)}
        pooled['max_member_fit_cap_hour_levels'] = 0
        for fold in FOLDS:
            c = cells.get((fold, stage))
            if c is None:
                continue
            for key in pooled:
                pooled[key] = max(pooled[key], c[key]) if key.startswith('max_') else pooled[key] + c[key]
            rows.append({'scope': fold, 'stage': stage, **c})
        rows.append({'scope': 'pooled', 'stage': stage, **pooled})
    table = pd.DataFrame(rows)
    table['cap_share'] = table.cap_hour_levels / table.emitted_hour_levels
    table['largest_member_fit_share_of_capped'] = np.where(table.cap_hour_levels > 0, table.max_member_fit_cap_hour_levels
                                                           / table.cap_hour_levels.where(table.cap_hour_levels > 0, 1), np.nan)
    gate = {fold: f['gate']['cap_share'] for fold, f in protocol['folds'].items()}
    gate['pooled'] = protocol['gate']['conditions']['G3']['share']
    table['gate_cap_share_round'] = table.scope.map(gate)
    by_member = pd.DataFrame([{'fold': fold, 'member': j, **m} for (fold, j), m in sorted(members.items())])
    by_member['cap_share'] = by_member.cap_hour_levels / by_member.emitted_hour_levels
    fold_cap = by_member.groupby('fold').cap_hour_levels.transform('sum')
    by_member['share_of_fold_capped'] = np.where(fold_cap > 0, by_member.cap_hour_levels / fold_cap.where(fold_cap > 0, 1), np.nan)
    return table, by_member


def job_diagnostics(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True, choices=(1, 2))
    args = ap.parse_args(rest)
    root, k = Path(root), args.attempt
    check_protocol(root, k)
    out = attempt_dir(root, k) / 'diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    frame = _frame(root, k)
    members = pd.read_parquet(attempt_dir(root, k) / 'members.parquet')
    entries = _entries(root, k)
    if len(entries) != 10747:
        raise ValueError(f'expected 10,747 evaluation keys in the DDNN-2 cache, found {len(entries)}')
    level, hist, pit = calibration(frame, entries)
    level.to_csv(out / 'calibration-by-level.csv', index=False)
    hist.to_csv(out / 'pit-histogram.csv', index=False)
    corr = correlations(frame, members)
    corr.to_csv(out / 'error-correlations.csv', index=False)
    hours = by_hour(frame, members)
    hours.to_csv(out / 'mae-by-local-hour.csv', index=False)
    extreme = extrapolation(root, frame)
    extreme.to_csv(out / 'extrapolation.csv', index=False)
    pf, intervals = peak_fold3_fold4(root, k)
    pf.to_csv(out / 'peak-fold3-fold4.csv', index=False)
    intervals.to_csv(out / 'fold3-fold4-intervals.csv', index=False)
    search = search_beside_folds(root, k)
    search.to_csv(out / 'search-validation-gate-fold.csv', index=False)
    stability, epochs = member_stability(frame, entries)
    stability.to_csv(out / 'member-stability.csv', index=False)
    epochs.to_csv(out / 'epochs.csv', index=False)
    blend, blend_meta = shape_blend(frame)
    blend.to_csv(out / 'shape-blend.csv', index=False)
    guards = json.loads((attempt_dir(root, k) / 'guards.json').read_text())
    guard_folds, guard_members = guards_by_fold(root, k, guards)
    guard_folds.to_csv(out / 'guards-by-fold.csv', index=False)
    guard_members.to_csv(out / 'guards-by-member.csv', index=False)
    pooled_corr = corr.loc[corr.scope.eq('pooled')].set_index(['a', 'b']).error_correlation
    summary = {'schema': 'cp24-diagnostics-v1', 'attempt': k, 'written_utc': stamp(),
               'role': 'descriptive; chooses nothing (§23.8)',
               'calibration': {'pooled': level.loc[level.scope.eq('pooled')].iloc[0].to_dict(), **pit},
               'error_correlations_pooled': {f'{a}~{b}': float(v) for (a, b), v in pooled_corr.items()},
               'extrapolation': {'days': int(len(extreme)), 'sets': extreme.set.value_counts().to_dict(),
                                 'hours_above_window_max': {p: int(extreme[f'hours_above_window_max_{p}'].sum())
                                                            for p in ('D2', 'v5', 'v3+D2', 'HGL', 'HG', 'D')}},
               'guards': guards['totals'],
               'guards_by_fold': guard_folds.loc[guard_folds.stage.eq('all')].set_index('scope')[
                   ['member_fits', 'cap_share', 'gate_cap_share_round', 'cap_hour_levels', 'member_fits_with_cap',
                    'largest_member_fit_share_of_capped', 'winsor_values_forecast', 'ensemble_crossings_restored',
                    'nonfinite_loss_stops']].to_dict('index'),
               'member_stability_pooled': stability.loc[stability.scope.eq('pooled')].iloc[0].to_dict(),
               'shape_blend': {**blend_meta, 'pooled': blend.loc[blend.scope.eq('pooled')].to_dict('records')},
               'files': sorted(p.name for p in out.iterdir())}
    atomic(attempt_dir(root, k) / 'diagnostics.json', clean(summary))
    print(json.dumps(clean(summary), default=str)[:3000], flush=True)
    return 0
