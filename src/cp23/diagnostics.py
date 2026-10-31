"""CP-23's §21.5 diagnostics, exactly as the frozen protocol defines them (`cp23.protocol.DIAGNOSTICS`).

Descriptive only: they choose nothing, and no forecast is produced or changed. Written to
`reports/distribution-challenger/diagnostics/` with a summary in `diagnostics.json`.

* DDNN's own calibration: coverage by level and the PIT histogram of D;
* extrapolation on extreme days against each origin's training-window maximum, beside CP-22's tree
  record;
* the 2022 peak and fold 4;
* ensemble and seed stability;
* the configuration chosen in each fold.

The fit cost and the cold daily cycle are `fit-cost.json` and `daily-cycle.json` (`cp23.daily`).
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
from cp15.scoring import FOLDS
from . import ddnn as D
from .budget import atomic
from .evaluate import clean
from .execution import OUT, DDNNCache, _identities, check_protocol
from .inputs import load, origin_manifest
from .jobs import stamp

DIR = OUT / 'diagnostics'
PIT_BINS = 20
CENTRAL = {'50': ('p25', 'p75'), '80': ('p10', 'p90'), '95': ('p025', 'p975')}
SHARE_TOP = 0.05


def _frame(root: Path) -> pd.DataFrame:
    cols = ['fold', 'policy', 'timestamp_utc', 'delivery_date', 'y_true', 'central', 'scale', 'level', *LABELS]
    parts = [pd.read_parquet(root / OUT / 'predictions.parquet')[cols],
             pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet', filters=[('policy', '==', 'HG')])[cols],
             pd.read_parquet(root / 'reports/block-challenger/predictions.parquet', filters=[('policy', '==', 'HGL')])[cols]]
    frame = pd.concat(parts, ignore_index=True)
    frame['timestamp_utc'] = pd.to_datetime(frame.timestamp_utc, utc=True).dt.as_unit('ns')
    frame['delivery_date'] = pd.to_datetime(frame.delivery_date)
    return frame


def _entries(root: Path) -> pd.DataFrame:
    """Per evaluation key: the members' Johnson SU parameters and medians from the verified DDNN cache."""
    fit_ident = _identities(root)[0]
    data = load(root)
    rows = []
    for f in origin_manifest(root)['folds']:
        first = date.fromisoformat(f['evaluation_start'])
        for o in f['origins']:
            d = date.fromisoformat(o['day'])
            if d < first or not len(data.rows(d)):
                continue
            item = DDNNCache(f['fold'], data, fit_ident).load(d)
            idx = data.rows(d)
            params = np.asarray(item['member_jsu_params'], float)          # members x n x 4
            medians = np.asarray(item['member_medians_z'], float)          # members x n
            for j, i in enumerate(idx):
                rows.append({'fold': f['fold'], 'timestamp_utc': data.index[i], 'delivery_date': pd.Timestamp(d),
                             'config': item['config'], 'y': data.y[i], 'level': data.level[i], 'scale': data.scale[i],
                             'params': params[:, j, :], 'member_medians_z': medians[:, j]})
    return pd.DataFrame(rows)


def ensemble_pit(params: np.ndarray, z: np.ndarray, iterations: int = 200) -> np.ndarray:
    """F_ens(z) for the quantile-averaged ensemble: the inverse of u -> mean_k Q_k(Phi(u)) at z, by bisection
    in u (the members' quantile functions are increasing in u = Phi^-1(p)); returns Phi(u)."""
    xi, lam, gamma, delta = (params[..., k] for k in range(4))           # n x members

    def q(u):
        return np.mean(xi + lam * np.sinh((u[:, None] - gamma) / delta), axis=1)
    lo = np.full(len(z), -40.0)
    hi = np.full(len(z), 40.0)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        below = q(mid) < z
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    u = (lo + hi) / 2
    return ndtr(u)


def calibration(frame: pd.DataFrame, entries: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = frame.loc[frame.policy.eq('D')].sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    e = entries.sort_values(['fold', 'timestamp_utc']).reset_index(drop=True)
    if not (np.array_equal(d.timestamp_utc.to_numpy(), e.timestamp_utc.to_numpy()) and np.array_equal(d.y_true.to_numpy(), e.y.to_numpy())):
        raise ValueError('cache entries do not align with the committed D vectors')
    rows = []
    for scope, part in [('pooled', d), *((f, d.loc[d.fold.eq(f)]) for f in FOLDS)]:
        rec = {'policy': 'D', 'scope': scope, 'n_hours': int(len(part))}
        for level, label in zip(D.LEVELS, LABELS):
            rec[f'share_at_or_below_{label}'] = float((part.y_true <= part[label]).mean())
            rec[f'nominal_{label}'] = level
        for name, (lo, hi) in CENTRAL.items():
            width = part[hi] - part[lo]
            rec[f'coverage{name}'] = float(((part.y_true >= part[lo]) & (part.y_true <= part[hi])).mean())
            rec[f'mean_width{name}'] = float(width.mean())
            rec[f'median_width{name}'] = float(width.median())
            rec[f'p95_width{name}'] = float(width.quantile(0.95))
        rows.append(rec)
    z = ((e.y - e.level) / e.scale).to_numpy(float)
    params = np.stack(e.params.to_list())                                  # n x members x 4
    pit = ensemble_pit(params, z)
    edges = np.linspace(0, 1, PIT_BINS + 1)
    hist = []
    for scope, mask in [('pooled', np.ones(len(e), bool)), *((f, (e.fold == f).to_numpy()) for f in FOLDS)]:
        counts, _ = np.histogram(pit[mask], bins=edges)
        for k in range(PIT_BINS):
            hist.append({'scope': scope, 'bin': k, 'lower': edges[k], 'upper': edges[k + 1], 'count': int(counts[k]),
                         'share': float(counts[k] / mask.sum()), 'uniform_share': 1 / PIT_BINS})
    # consistency: PIT <= q exactly when y <= the emitted q-quantile (up to the bisection tolerance)
    consistency = {label: float(np.mean((pit <= level + 1e-9) == (d.y_true.to_numpy() <= d[label].to_numpy())))
                   for level, label in zip(D.LEVELS, LABELS)}
    summary = {'pit_mean': float(pit.mean()), 'pit_share_below_0.05': float((pit < 0.05).mean()),
               'pit_share_above_0.95': float((pit > 0.95).mean()), 'pit_vs_emitted_quantile_agreement': consistency}
    return pd.DataFrame(rows), pd.DataFrame(hist), summary


def extrapolation(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    data = load(root)
    members = pd.read_parquet(root / OUT / 'members.parquet')
    members['delivery_date'] = pd.to_datetime(members.delivery_date)
    truth = frame.loc[frame.policy.eq('v5'), ['fold', 'timestamp_utc', 'delivery_date', 'y_true']]
    daily_max = truth.groupby(['fold', 'delivery_date']).y_true.max()
    cp22 = pd.read_csv(root / 'reports/v4-revision/investigation/extrapolation.csv')
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
            for p in ('D', 'v5', 'v3+D', 'HGL', 'HG'):
                part = frame.loc[frame.policy.eq(p) & frame.fold.eq(fold) & frame.delivery_date.eq(day)]
                row[f'max_central_{p}'] = float(part.central.max())
                row[f'hours_above_window_max_{p}'] = int((part.central > wmax).sum())
            dpart = frame.loc[frame.policy.eq('D') & frame.fold.eq(fold) & frame.delivery_date.eq(day)]
            row['max_p975_D'] = float(dpart.p975.max())
            row['hours_actual_above_D_p975'] = int((dpart.y_true > dpart.p975).sum())
            tree = cp22.loc[cp22.fold.eq(fold) & cp22.delivery_date.eq(str(d))]
            for name in ('L-P', 'L-N', 'L-R', 'PN-avg'):
                row[f'cp22_max_central_{name}'] = float(tree[f'max_central_{name}'].iloc[0]) if len(tree) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def peak_and_fold4(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(root / OUT / 'metrics.csv')
    diag = pd.read_csv(root / OUT / 'diagnostics.csv', low_memory=False)
    cols = ['policy', 'n_hours', 'n_days', 'MAE', 'WIS', 'bias', 'coverage50', 'mean_width50', 'coverage80', 'mean_width80',
            'coverage95', 'mean_width95']
    keep = [c for c in cols if c in metrics.columns]
    f4 = metrics.loc[metrics.scope.eq('per_fold') & metrics.fold.eq('fold_4'), keep].assign(scope='fold_4 2025-05-01..07-29')
    pk = diag.loc[diag.scope.eq('peak'), [c for c in keep if c in diag.columns]].assign(scope='peak 2022-08-15..31')
    unc = pd.read_csv(root / OUT / 'uncertainty.csv')
    intervals = unc.loc[unc.scope.eq('fold_4'), ['candidate', 'baseline', 'metric', 'difference', 'ci_lower', 'ci_upper', 'role']]
    return pd.concat([pk, f4], ignore_index=True), intervals


def seed_stability(root: Path, entries: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    med = np.stack(entries.member_medians_z.to_list())                     # n x members
    eur = entries.level.to_numpy()[:, None] + entries.scale.to_numpy()[:, None] * med
    ens = eur.mean(axis=1)
    y = entries.y.to_numpy()
    rows = []
    for scope, mask in [('pooled', np.ones(len(entries), bool)), *((f, (entries.fold == f).to_numpy()) for f in FOLDS)]:
        rec = {'scope': scope, 'n_hours': int(mask.sum()), 'ensemble_median_MAE': float(np.mean(np.abs(ens[mask] - y[mask]))),
               'member_median_spread_mean_abs': float(np.mean(np.abs(eur[mask] - ens[mask, None]))),
               'share_hours_members_straddle_actual': float(np.mean((eur[mask].min(axis=1) < y[mask]) & (eur[mask].max(axis=1) > y[mask])))}
        for k, seed in enumerate(D.SEEDS):
            rec[f'seed_{seed}_median_MAE'] = float(np.mean(np.abs(eur[mask, k] - y[mask])))
        rows.append(rec)
    fits = pd.read_parquet(root / OUT / 'fits.parquet')
    epochs = fits.loc[fits.stage.eq('origin')].groupby(['config', 'seed']).agg(
        fits=('best_epoch', 'size'), best_epoch_median=('best_epoch', 'median'), best_epoch_p90=('best_epoch', lambda s: s.quantile(.9)),
        epochs_run_median=('epochs_run', 'median'), epochs_run_max=('epochs_run', 'max')).reset_index()
    return pd.DataFrame(rows), {'epochs_by_config_and_seed': epochs.to_dict('records')}


def job_diagnostics(root: Path, rest) -> int:
    check_protocol(root)
    out = root / DIR
    out.mkdir(parents=True, exist_ok=True)
    frame = _frame(root)
    entries = _entries(root)
    if len(entries) != 10747:
        raise ValueError(f'expected 10,747 evaluation keys in the DDNN cache, found {len(entries)}')
    level, hist, pit_summary = calibration(frame, entries)
    level.to_csv(out / 'calibration-by-level.csv', index=False)
    hist.to_csv(out / 'pit-histogram.csv', index=False)
    extreme = extrapolation(root, frame)
    extreme.to_csv(out / 'extrapolation.csv', index=False)
    pf4, f4_intervals = peak_and_fold4(root)
    pf4.to_csv(out / 'peak-and-fold4.csv', index=False)
    f4_intervals.to_csv(out / 'fold4-intervals.csv', index=False)
    seeds, epochs = seed_stability(root, entries)
    seeds.to_csv(out / 'seed-stability.csv', index=False)
    pd.DataFrame(epochs['epochs_by_config_and_seed']).to_csv(out / 'epochs.csv', index=False)
    selection = json.loads((root / OUT / 'selection.json').read_text())
    summary = {'schema': 'cp23-diagnostics-v1', 'written_utc': stamp(), 'role': 'descriptive; chooses nothing (§21.5)',
               'definitions': 'reports/distribution-challenger/protocol.json -> diagnostics (frozen before scoring)',
               'calibration': {'pooled': level.loc[level.scope.eq('pooled')].iloc[0].to_dict(), **pit_summary},
               'extrapolation': {'days': int(len(extreme)), 'sets': extreme.set.value_counts().to_dict(),
                                 'hours_above_window_max': {p: int(extreme[f'hours_above_window_max_{p}'].sum())
                                                            for p in ('D', 'v5', 'v3+D', 'HGL', 'HG')}},
               'configuration_by_fold': {f: {'selected': v['selected'], 'holdout_mae': v['holdout_mae'], 'tie': v['tie'],
                                             'winner_margin_relative': v['winner_margin_relative']}
                                         for f, v in selection['folds'].items()},
               'seed_stability_pooled': seeds.loc[seeds.scope.eq('pooled')].iloc[0].to_dict(),
               'fit_cost': 'fit-cost.json, fit-cost-by-origin.csv, fits.parquet', 'daily_cycle': 'daily-cycle.json',
               'files': sorted(p.name for p in out.iterdir())}
    atomic(root / OUT / 'diagnostics.json', clean(summary))
    print(json.dumps(clean(summary), default=str)[:3000], flush=True)
    return 0
