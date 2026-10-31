"""The Owner's investigation (capstone v21-r9 §20.5), reported descriptively and choosing nothing.

Every definition is the one frozen in the pre-run protocol (`protocol.json` -> `investigation`):

* the ladder decomposition of v4 - v3 and the member-weight curve, labelled "oracle, not selectable";
* PN's capacity-selection stability (flip rate, winner margin, distribution) next to CP-21's L-P,
  L-R and L-N;
* extrapolation on extreme days against each origin's training-window maximum;
* coverage by hour, block and regime, each fold's alpha_t path and the days DL took to react after
  the peak began;
* the fixed shock-day set, with MAE, WIS and coverage on those days and the three days after;
* LEAR's penalty-selection stability from logged selections (no refit).

No forecast is produced or changed; nothing here is a criterion. Written to `reports/v4-revision/`.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cp15.data import LABELS
from cp15.scoring import FOLDS, FOLD_WINDOWS, score_hourly
from cp15.data import history_start
from cp21.lgbm import GRID
from .budget import atomic
from .evaluate import clean
from .execution import OUT, check_protocol
from .inputs import load
from .jobs import cp20_art, stamp
from . import scoring as S

PEAK_START = date(2022, 8, 15)


def _predictions(root: Path) -> pd.DataFrame:
    cols = ['fold', 'policy', 'timestamp_utc', 'delivery_date', 'y_true', 'central', 'scale', 'level', *LABELS]
    parts = [pd.read_parquet(root / OUT / 'predictions.parquet')[cols],
             pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet', filters=[('policy', '==', 'HG')])[cols],
             pd.read_parquet(root / 'reports/block-challenger/predictions.parquet', filters=[('policy', '==', 'HGL')])[cols]]
    if (root / OUT / 'predictions-w.parquet').exists():
        parts.append(pd.read_parquet(root / OUT / 'predictions-w.parquet')[cols])
    frame = pd.concat(parts, ignore_index=True)
    frame['timestamp_utc'] = pd.to_datetime(frame.timestamp_utc, utc=True).dt.as_unit('ns')
    frame['delivery_date'] = pd.to_datetime(frame.delivery_date)
    return frame


def ladder(root: Path) -> dict:
    p = json.loads((root / OUT / 'protocol.json').read_text())['investigation']['ladder_decomposition']
    metrics = pd.read_csv(root / OUT / 'metrics.csv')
    eq = metrics.loc[metrics.scope.eq('equal_fold')].set_index('policy')
    unc = pd.read_csv(root / OUT / 'uncertainty.csv')
    unc = unc.loc[unc.scope.eq('equal_fold')]
    steps = []
    for a, b, label in p['steps']:
        row = {'from': S.DISPLAY.get(a, a), 'to': S.DISPLAY.get(b, b), 'change': label}
        for m in ('MAE', 'WIS'):
            row[f'dS_{m}'] = float(eq.loc[b, f'S_{m}'] - eq.loc[a, f'S_{m}'])
            hit = unc.loc[unc.candidate.eq(b) & unc.baseline.eq(a) & unc.metric.eq(m)]
            neg = unc.loc[unc.candidate.eq(a) & unc.baseline.eq(b) & unc.metric.eq(m)]
            if len(hit):
                row[f'dS_{m}_ci'] = [float(hit.ci_lower.iloc[0]), float(hit.ci_upper.iloc[0])]
            elif len(neg):  # the contrast is stored the other way round: reverse its sign
                row[f'dS_{m}_ci'] = [-float(neg.ci_upper.iloc[0]), -float(neg.ci_lower.iloc[0])]
            else:
                row[f'dS_{m}_ci'] = None
        steps.append(row)
    total = {m: float(eq.loc['HGL', f'S_{m}'] - eq.loc['HG', f'S_{m}']) for m in ('MAE', 'WIS')}
    parts = {m: float(sum(eq.loc[b, f'S_{m}'] - eq.loc[a, f'S_{m}'] for a, b in
                          (('HG', 'R'), ('R', 'A-PN-sel'), ('A-PN-sel', 'M'), ('M', 'HGL')))) for m in ('MAE', 'WIS')}
    return {'steps': steps, 'v4_minus_v3': total, 'sum_of_ladder_brackets': parts,
            'identity_holds': all(abs(total[m] - parts[m]) < 1e-12 for m in total),
            'scores': {S.DISPLAY.get(k, k): {'S_MAE': float(eq.loc[k, 'S_MAE']), 'S_WIS': float(eq.loc[k, 'S_WIS'])}
                       for k in ('HG', 'R', 'A-PN-sel', 'M', 'HGL', 'A-LP', 'A-LN')}}


def weight_curve(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    spec = json.loads((root / OUT / 'protocol.json').read_text())['investigation']['ladder_decomposition']['member_weight_curve']
    members = pd.read_parquet(root / OUT / 'members.parquet')
    members['timestamp_utc'] = pd.to_datetime(members.timestamp_utc, utc=True).dt.as_unit('ns')
    truth = frame.loc[frame.policy.eq('R'), ['fold', 'timestamp_utc', 'y_true']]
    m = members.merge(truth, on=['fold', 'timestamp_utc'], validate='one_to_one')
    metrics = pd.read_csv(root / OUT / 'metrics.csv')
    b0 = metrics.loc[metrics.policy.eq('B0') & metrics.scope.eq('per_fold')].set_index('fold').MAE
    rows = []
    for name, cols in spec['members'].items():
        member = m[cols].mean(axis=1) if len(cols) > 1 else m[cols[0]]
        for w in spec['weights']:
            central = (1 - w) * m.HG + w * member
            err = (central - m.y_true).abs()
            per_fold = err.groupby(m.fold).mean()
            rows.append({'member': name, 'weight': w, 'S_MAE_central': float(np.mean([per_fold[f] / b0[f] for f in FOLDS])),
                         'pooled_central_MAE': float(err.mean()), 'label': spec['label']})
    return pd.DataFrame(rows)


def capacity_stability(root: Path) -> pd.DataFrame:
    rows = []
    pn = pd.read_parquet(root / OUT / 'fits.parquet')
    cp21 = pd.read_parquet(root / 'reports/block-challenger/fits.parquet')
    frames = [('PN', 'pooled', pn)] + [(arm, model, cp21.loc[cp21.arm.eq(arm)]) for arm, model in
                                         (('L-P', 'pooled'), ('L-R', 'night'), ('L-R', 'solar'), ('L-R', 'shoulder'),
                                          ('L-N', 'night'), ('L-N', 'solar'), ('L-N', 'shoulder'))]
    for arm, model, f in frames:
        f = f.loc[f.model.eq(model)]
        inner = f.loc[f.role.eq('inner')].pivot_table(index=['fold', 'phase', 'delivery_date'], columns='config',
                                                      values='validation_mae', aggfunc='first')
        for phase_group, phases in (('evaluation', ('evaluation',)), ('warm-up', ('warmup', 'admission'))):
            part = inner.loc[inner.index.get_level_values('phase').isin(phases)].sort_index()
            if part.empty:
                continue
            order = [c['id'] for c in GRID]
            part = part[order]
            losses = part.to_numpy(float)
            winner = np.array([min(range(4), key=lambda i: (row[i], i)) for row in losses])
            sorted_l = np.sort(losses, axis=1)
            margin = (sorted_l[:, 1] - sorted_l[:, 0]) / sorted_l[:, 0]
            folds = part.index.get_level_values('fold').to_numpy()
            flips = [int(winner[i] != winner[i - 1]) for i in range(1, len(winner)) if folds[i] == folds[i - 1]]
            rows.append({'arm': arm, 'model': model, 'phase': phase_group, 'origins': int(len(winner)),
                         'flip_rate': float(np.mean(flips)) if flips else None,
                         'winner_margin_median': float(np.median(margin)), 'winner_margin_q25': float(np.quantile(margin, .25)),
                         'winner_margin_q75': float(np.quantile(margin, .75)),
                         **{f'share_{g}': float(np.mean(winner == i)) for i, g in enumerate(order)}})
    return pd.DataFrame(rows)


def extrapolation(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    data = load(root)
    members = pd.read_parquet(root / OUT / 'members.parquet')
    members['timestamp_utc'] = pd.to_datetime(members.timestamp_utc, utc=True).dt.as_unit('ns')
    members['delivery_date'] = pd.to_datetime(members.delivery_date)
    truth = frame.loc[frame.policy.eq('R'), ['fold', 'timestamp_utc', 'delivery_date', 'y_true']]
    daily_max = truth.groupby(['fold', 'delivery_date']).y_true.max()
    rows = []
    policies = [p for p in ('HG', 'HGL', 'R', 'M', 'A-PN-sel') if p in set(frame.policy)]
    for fold in FOLDS:
        days = daily_max.xs(fold)
        top = set(days.sort_values(ascending=False).index[:math.ceil(S.SHOCK_SHARE * len(days))])
        for day, actual_max in days.items():
            d = day.date()
            lower = history_start(d, 'B3')
            window = (data.dates >= np.datetime64(lower)) & (data.dates < np.datetime64(d)) & data.eligible
            wmax = float(data.y[window].max())
            kinds = [k for k, hit in (('exceeds_window_max', actual_max > wmax), ('top_5pct_daily_max', day in top)) if hit]
            if not kinds:
                continue
            mem = members.loc[members.fold.eq(fold) & members.delivery_date.eq(day)]
            row = {'fold': fold, 'delivery_date': str(d), 'set': '+'.join(kinds), 'actual_max': float(actual_max),
                   'window_max': wmax, 'window_start': str(lower)}
            for name in ('PN-avg', 'PN-sel', 'L-P', 'L-N', 'L-R', 'HG', 'HGL'):
                row[f'max_central_{name}'] = float(mem[name].max())
                row[f'hours_above_window_max_{name}'] = int((mem[name] > wmax).sum())
            for p in policies:
                part = frame.loc[frame.policy.eq(p) & frame.fold.eq(fold) & frame.delivery_date.eq(day)]
                row[f'max_p50_{p}'] = float(part.p50.max())
            rows.append(row)
    return pd.DataFrame(rows)


def coverage_regime(frame: pd.DataFrame) -> pd.DataFrame:
    ref = frame.loc[frame.policy.eq('HG')]
    cuts = np.quantile(ref.scale.to_numpy(), [1 / 3, 2 / 3])
    hourly = score_hourly(frame.assign(delivery_date=frame.delivery_date))
    hourly['regime'] = np.select([frame.scale.to_numpy() <= cuts[0], frame.scale.to_numpy() <= cuts[1]], ['low', 'medium'], 'high')
    rows = []
    for (policy, regime), g in hourly.groupby(['policy', 'regime']):
        rows.append({'policy': policy, 'regime': regime, 'hours': int(len(g)), 'scale_cut_low': float(cuts[0]),
                     'scale_cut_high': float(cuts[1]), **{f'coverage{c}': float(g[f'hit{c}'].mean()) for c in (50, 80, 95)},
                     **{f'mean_width{c}': float(g[f'width{c}'].mean()) for c in (50, 80, 95)},
                     'MAE': float(g.absolute_error.mean()), 'WIS': float(g.WIS.mean())})
    return pd.DataFrame(rows)


def alpha_paths(root: Path) -> pd.DataFrame:
    rows = []
    for name in ('lineage.json', 'lineage-w.json'):
        path = root / OUT / name
        if not path.exists():
            continue
        for r in json.loads(path.read_text())['origins']:
            if r.get('emitted') and r['layer'] != 'H':
                rows.append({'policy': r['policy'], 'fold': r['fold'], 'day': r['day'], 'phase': r['phase'],
                             **{f'alpha_t_{a}': r['alpha_t'][a] for a in ('0.05', '0.2', '0.5')},
                             'newest_day_weight': r['newest_day_weight'], 'effective_days': r['effective_days'],
                             'crossed_rows_rearranged': r['crossed_rows_rearranged']})
    return pd.DataFrame(rows)


def reaction(frame: pd.DataFrame, paths: pd.DataFrame, w: str | None) -> list[dict]:
    names = [p for p in ((w, 'W+ACI', 'W+DL', 'W+DLF') if w else ()) + ('HGL', 'v4+DL', 'HG', 'v3+DL') if p in set(frame.policy)]
    out = []
    width = frame.assign(w95=frame.p975 - frame.p025).groupby(['policy', 'delivery_date']).w95.mean()
    for p in names:
        rec = {'policy': S.DISPLAY.get(p, p) if p != w else f'W ({p})'}
        series = width.xs(p)
        base = series.loc['2022-08-01':'2022-08-14'].mean()
        after = series.loc[str(PEAK_START):'2022-09-28']
        hit = after.loc[after > 1.25 * base]
        rec['width_reaction_days'] = int((hit.index[0].date() - PEAK_START).days) if len(hit) else None
        rec['pre_peak_mean_width95'] = float(base)
        if len(paths) and p in set(paths.policy):
            a = paths.loc[paths.policy.eq(p) & paths.fold.eq('fold_3')].set_index('day')['alpha_t_0.05']
            start = a.get(str(PEAK_START))
            later = a.loc[a.index > str(PEAK_START)]
            lower = later.loc[later < start] if start is not None else later.iloc[:0]
            rec['aci_reaction_days'] = int((date.fromisoformat(lower.index[0]) - PEAK_START).days) if len(lower) else None
            rec['alpha_t_0.05_at_peak_start'] = start
        else:
            rec['aci_reaction_days'] = 'not applicable (no ACI)'
        out.append(rec)
    return out


def shock_days(root: Path, frame: pd.DataFrame, w: str | None) -> tuple[pd.DataFrame, dict]:
    data = load(root)
    daily = pd.Series(data.y).groupby(pd.Index(data.dates)).mean()
    sets = {}
    for fold in FOLDS:
        start, end = (pd.Timestamp(x) for x in FOLD_WINDOWS[fold])
        rep = sorted(frame.loc[frame.fold.eq(fold) & frame.policy.eq('HG'), 'delivery_date'].unique())
        change = {d: abs(daily[np.datetime64(d.date())] - daily[np.datetime64((d - timedelta(days=1)).date())]) for d in rep}
        k = math.ceil(S.SHOCK_SHARE * len(rep))
        sets[fold] = sorted(sorted(change, key=lambda d: -change[d])[:k])
    peak = [pd.Timestamp(PEAK_START + timedelta(days=i)) for i in range(S.PEAK_FIRST_DAYS)]
    names = [p for p in ((w, 'W+ACI', 'W+DL', 'W+DLF') if w else ()) + ('HGL', 'v4+DL', 'HG', 'v3+DL') if p in set(frame.policy)]
    hourly = score_hourly(frame.loc[frame.policy.isin(names)])
    rows = []
    for label, days_by_fold in (('largest_daily_mean_change', sets), ('peak_first_ten_days', {'fold_3': peak})):
        for fold, days in days_by_fold.items():
            end = pd.Timestamp(FOLD_WINDOWS[fold][1])
            for offset in range(S.SHOCK_AFTER_DAYS + 1):
                window = {d + timedelta(days=offset) for d in days if d + timedelta(days=offset) <= end}
                part = hourly.loc[hourly.fold.eq(fold) & hourly.delivery_date.isin(window)]
                for p, g in part.groupby('policy'):
                    rows.append({'set': label, 'fold': fold, 'offset_days': offset, 'policy': p, 'hours': int(len(g)),
                                 'days': int(g.delivery_date.nunique()), 'MAE': float(g.absolute_error.mean()),
                                 'WIS': float(g.WIS.mean()), **{f'coverage{c}': float(g[f'hit{c}'].mean()) for c in (50, 80, 95)},
                                 'mean_width95': float(g.width95.mean())})
    table = pd.DataFrame(rows)
    pooled = (table.assign(mae_h=table.MAE * table.hours, wis_h=table.WIS * table.hours, cov_h=table.coverage95 * table.hours)
              .groupby(['set', 'policy'])[['hours', 'mae_h', 'wis_h', 'cov_h']].sum())
    summary = {f'{s}/{S.DISPLAY.get(p, p)}': {'hours': int(r.hours), 'MAE': r.mae_h / r.hours, 'WIS': r.wis_h / r.hours,
                                              'coverage95': r.cov_h / r.hours} for (s, p), r in pooled.iterrows()}
    return table, {'shock_days': {f: [str(d.date()) for d in v] for f, v in sets.items()},
                   'peak_first_ten_days': [str(d.date()) for d in peak], 'pooled_shock_and_three_after': summary}


def lear_penalty_stability(root: Path) -> pd.DataFrame:
    rows = []
    m = json.loads((root / 'reports/v2-causal/input-manifest.json').read_text())
    for f in m['folds']:
        first = f['evaluation_start']
        for o in f['origins']:
            item = json.loads((cp20_art() / 'hg-components' / f['fold'] / f'{o["day"]}.json').read_text())
            for policy in ('A1', 'B2'):
                for rec in item['fits'].get(policy, []):  # origins without eligible hours carry no fits
                    losses = rec['validation_mae_by_alpha']
                    ordered = sorted(losses.values())
                    rows.append({'fold': f['fold'], 'day': o['day'], 'phase': 'evaluation' if o['day'] >= first else 'warm-up',
                                 'component': policy, 'local_hour': rec['local_hour'], 'selected': rec['selected_relative_alpha'],
                                 'margin': (ordered[1] - ordered[0]) / ordered[0] if len(ordered) > 1 and ordered[0] > 0 else None})
    frame = pd.DataFrame(rows).sort_values(['component', 'local_hour', 'fold', 'day'])
    out = []
    for (comp, phase), g in frame.groupby(['component', 'phase']):
        flips = []
        for _, h in g.groupby(['local_hour', 'fold']):
            sel = h.selected.to_numpy()
            flips += list(sel[1:] != sel[:-1])
        out.append({'component': comp, 'phase': phase, 'hour_origins': int(len(g)), 'flip_rate': float(np.mean(flips)),
                    'margin_median': float(g.margin.dropna().median()),
                    **{f'share_alpha_{a}': float((g.selected == a).mean()) for a in sorted(g.selected.unique())}})
    return pd.DataFrame(out)


def job_investigation(root: Path, rest) -> int:
    check_protocol(root)
    decisions = json.loads((root / OUT / 'decisions.json').read_text())
    w = decisions['replacement']['winner']
    frame = _predictions(root)
    out = root / OUT / 'investigation'
    out.mkdir(parents=True, exist_ok=True)
    result = {'schema': 'cp22-investigation-v1', 'role': 'descriptive; chooses nothing (§20.5)', 'W': w}
    result['ladder'] = ladder(root)
    curve = weight_curve(root, frame)
    curve.to_csv(out / 'member-weight-curve.csv', index=False)
    result['member_weight_curve'] = {'label': 'oracle, not selectable', 'file': str(OUT / 'investigation/member-weight-curve.csv'),
                                     'argmin_weight_by_member': {k: float(g.loc[g.S_MAE_central.idxmin(), 'weight'])
                                                                 for k, g in curve.groupby('member')}}
    cap = capacity_stability(root)
    cap.to_csv(out / 'capacity-stability.csv', index=False)
    result['capacity_stability'] = cap.to_dict('records')
    ext = extrapolation(root, frame)
    ext.to_csv(out / 'extrapolation.csv', index=False)
    result['extrapolation'] = {'days': int(len(ext)), 'exceeds_window_max_days': int(ext.set.str.contains('exceeds').sum()) if len(ext) else 0}
    reg = coverage_regime(frame)
    reg.to_csv(out / 'coverage-regime.csv', index=False)
    paths = alpha_paths(root)
    paths.to_csv(out / 'alpha-paths.csv', index=False)
    result['reaction_after_peak_start'] = reaction(frame, paths, w)
    shocks, shock_summary = shock_days(root, frame, w)
    shocks.to_csv(out / 'shock-days.csv', index=False)
    result['shock_days'] = shock_summary
    lear = lear_penalty_stability(root)
    lear.to_csv(out / 'lear-penalty-stability.csv', index=False)
    result['lear_penalty_stability'] = {'source': 'logged selections in the verified CP-20 HG cache; no LEAR refit',
                                        'rows': lear.to_dict('records')}
    result['written_utc'] = stamp()
    atomic(root / OUT / 'investigation.json', clean(result))
    print(json.dumps(clean({k: v for k, v in result.items() if k in ('ladder', 'member_weight_curve', 'reaction_after_peak_start')}),
                     default=str)[:3000], flush=True)
    return 0
