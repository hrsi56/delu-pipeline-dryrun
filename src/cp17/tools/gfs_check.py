"""Item 1: the A1 GFS check — descriptive, from CP-26's stored forecasts, training nothing.

Three streams on CP-26's scored calendar (2020-12-29 to 2026-09-06), each scored on CP-25's frozen
eligible keys of the date (the `scoring_index` CP-26 recorded) with CP-28's daily MAE and WIS
(`cp28.scoring.daily`, §24.7's definitions):

- **A1 nominal:** D2's own seven Johnson SU quantiles from CP-26's nominal ledger (`members.nominal.D2`);
- **A1 GFS +6 h:** D2 in CP-26's `gfs6` route (`members.gfs6.D2`), where the weather of CP-25's +6 h
  affected dates is missing under §15.3, in training and at the forecast;
- **v2 nominal:** v2's native quantiles (`policies.nominal/v2`), which use no weather.

The primary population is the paired one: eligible scored dates on which all three streams have a valid
native forecast on every scoring key. Dates lacking a stream are counted, never filled. The masks are
CP-25's GFS +6 h classification (`reports/development-inputs/margin-days.csv`, group `gfs`): affected,
unaffected, and every eligible date.

**The brief's rule.** The fallback is v2. If v2's mean daily loss on the affected dates is more than 2%
above A1's GFS +6 h route on both MAE and WIS, the check returns BLOCKED for the Owner's decision.

    python -m cp17.tools.gfs_check --out reports/cp17/gfs-check.json
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cp25 import guard
from cp28 import reader, scoring
from . import ledger

SCORED_START = date(2020, 12, 29)
MARGIN_DAYS = 'reports/development-inputs/margin-days.csv'
STREAMS = ('A1_nominal', 'A1_gfs6', 'v2_nominal')
THRESHOLD = 0.02


def gfs_affected(root) -> dict:
    margins = pd.read_csv(reader.cp25_file(root, MARGIN_DAYS))
    g = margins[margins.group == 'gfs']
    out = {date.fromisoformat(d): str(v) == 'True' for d, v in zip(g.delivery_date, g.affected_plus_6h)}
    if sum(out.values()) != 1783:
        raise ValueError(f'CP-25 classifies {sum(out.values())} dates as GFS +6 h affected, not 1,783')
    return out


def stream_quantiles(value: dict) -> tuple[dict, pd.DatetimeIndex, bool]:
    nominal = value['policies']['nominal/v5']
    scoring_keys = ledger.keys(nominal['scoring_index'])
    out = {}
    for name, route in (('A1_nominal', 'nominal'), ('A1_gfs6', 'gfs6')):
        item = value['members'][route].get('D2')
        q = None
        if item is not None and item.get('status') == 'ok':
            q = ledger.on_keys(item.get('quantiles'), nominal['index'], scoring_keys)
            if q is not None and not np.array_equal(np.asarray(item['quantiles'], float)[:, 3], np.asarray(item['central'], float)):
                q = None
        out[name] = q
    v2 = value['policies']['nominal/v2']
    out['v2_nominal'] = ledger.on_keys(v2.get('native_quantiles'), v2['index'], scoring_keys) \
        if v2.get('native_central_valid') else None
    return out, scoring_keys, bool(nominal['scored'])


def daily_table(root) -> pd.DataFrame:
    affected = gfs_affected(root)
    prices = reader.prices(root)
    rows = []
    for d, value in ledger.iter_days(root, first=SCORED_START):
        guard.admit_delivery_day(d)
        q, scoring_keys, scored = stream_quantiles(value)
        if not scored or not len(scoring_keys):
            continue
        y = reader.truth_for(prices, scoring_keys)
        row = {'day': d, 'affected': affected[d], 'hours': len(scoring_keys), 'truth_finite': bool(np.isfinite(y).all())}
        for name in STREAMS:
            mae, wis = scoring.daily(q[name], y) if q[name] is not None else (np.nan, np.nan)
            row[f'{name}:MAE'], row[f'{name}:WIS'] = mae, wis
            row[f'{name}:native'] = q[name] is not None
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame) -> dict:
    complete = np.logical_and.reduce([table[f'{s}:native'] & np.isfinite(table[f'{s}:MAE']) for s in STREAMS])
    masks = {'affected': table.affected, 'unaffected': ~table.affected, 'all_eligible': pd.Series(True, index=table.index)}
    out = {}
    for name, mask in masks.items():
        mask = mask.to_numpy(bool)
        paired = mask & complete
        cell = {'dates_eligible': int(mask.sum()), 'dates_paired': int(paired.sum()),
                'dates_missing_by_stream': {s: int((mask & ~(table[f'{s}:native'] & np.isfinite(table[f'{s}:MAE']))).sum())
                                            for s in STREAMS},
                'hours_paired': int(table.hours[paired].sum())}
        for metric in ('MAE', 'WIS'):
            means = {s: float(table[f'{s}:{metric}'][paired].mean()) for s in STREAMS}
            cell[metric] = {'mean_daily_loss_eur': means,
                            'ratio_v2_to_A1_gfs6': means['v2_nominal'] / means['A1_gfs6'],
                            'ratio_A1_gfs6_to_A1_nominal': means['A1_gfs6'] / means['A1_nominal'],
                            'ratio_v2_to_A1_nominal': means['v2_nominal'] / means['A1_nominal']}
        out[name] = cell
    return out


def rule(summary: dict) -> dict:
    cell = summary['affected']
    # A ratio of exactly 1.02 is not "more than 2% worse"; comparing the ratio avoids 1.02 - 1 > 0.02 in float64.
    worse = {m: cell[m]['ratio_v2_to_A1_gfs6'] > 1 + THRESHOLD for m in ('MAE', 'WIS')}
    return {'statement': 'BLOCKED if v2 is more than 2% worse than A1\'s GFS +6 h route on the affected dates '
                         'on both MAE and WIS; otherwise the fallback is v2, as the Owner decided',
            'v2_excess_over_A1_gfs6': {m: cell[m]['ratio_v2_to_A1_gfs6'] - 1 for m in ('MAE', 'WIS')},
            'more_than_2pct_worse': worse, 'blocked': all(worse.values()),
            'outcome': 'BLOCKED' if all(worse.values()) else 'fallback v2 stands'}


def unpaired_sensitivity(table: pd.DataFrame) -> dict:
    """The rule's comparison on every affected date where v2 and A1's GFS +6 h route both have native
    quantiles, without requiring A1's nominal stream as well."""
    mask = table.affected & table['v2_nominal:native'] & table['A1_gfs6:native']
    return {'dates': int(mask.sum()),
            **{m: float(table[f'v2_nominal:{m}'][mask].mean() / table[f'A1_gfs6:{m}'][mask].mean() - 1)
               for m in ('MAE', 'WIS')}}


def crosscheck(root, table: pd.DataFrame) -> dict:
    """The daily losses against the stored ones: CP-28's committed A1 arm (D2 on its native days) and,
    where CP-26's local-only analysis tables exist, CP-26's nominal/v2 and nominal/D2 streams."""
    out = {}
    refs = {}
    a1 = pd.read_parquet(Path(root) / 'reports/cp28/results/daily-losses.parquet')
    refs['A1_nominal:cp28_A1'] = a1[a1.stream == 'A1'].assign(day=lambda f: [date.fromisoformat(d) for d in f.day]).set_index('day')
    local = ledger.project_root(root) / '.local'
    try:
        daily = reader.read_table(root, local, 'daily')
        refs['v2_nominal:cp26_nominal/v2'] = daily[daily.stream == 'nominal/v2'].set_index('delivery_date')
        refs['A1_nominal:cp26_nominal/D2'] = daily[daily.stream == 'nominal/D2'].set_index('delivery_date')
    except FileNotFoundError:
        out['cp26_tables'] = 'absent: CP-26 local-only analysis tables not on this machine'
    for name, ref in refs.items():
        stream = name.split(':')[0]
        for metric in ('MAE', 'WIS'):
            mine = table[f'{stream}:{metric}'].to_numpy(float)
            theirs = ref.reindex(list(table.day))[metric].to_numpy(float)
            both = np.isfinite(mine) & np.isfinite(theirs)
            out[f'{name}:{metric}'] = {'dates_compared': int(both.sum()), 'bitwise_equal': int((mine[both] == theirs[both]).sum()),
                                       'max_abs_difference': float(np.abs(mine[both] - theirs[both]).max())}
    return out


DAILY_COLUMNS = ['day', 'affected', 'hours', 'truth_finite'] + [f'{s}:{c}' for s in STREAMS for c in ('native', 'MAE', 'WIS')]


def daily_csv(table: pd.DataFrame) -> str:
    """The committed daily rows; floats as repr, so the CSV round-trips bit for bit."""
    lines = [','.join(DAILY_COLUMNS)]
    for row in table[DAILY_COLUMNS].itertuples(index=False):
        lines.append(','.join(repr(float(v)) if isinstance(v, float) else str(v) for v in row))
    return '\n'.join(lines) + '\n'


def read_daily_csv(path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision='round_trip')
    frame['day'] = [date.fromisoformat(d) for d in frame.day]
    for column in ['affected', 'truth_finite'] + [f'{s}:native' for s in STREAMS]:
        frame[column] = frame[column].astype(bool)
    return frame


def build(root, table: pd.DataFrame | None = None) -> dict:
    root = Path(root)
    table = daily_table(root) if table is None else table
    if table.day.max() > guard.BOUNDARY:
        raise ValueError('a scored date is after the development boundary')
    summary = summarize(table)
    canonical = daily_csv(table).encode()
    return {'schema': 'cp17-gfs-check-v1', 'evidence_class': 'development_post_selection',
            'what': 'descriptive; CP-26 stored forecasts only; nothing trained',
            'inputs': {'scores_manifest': reader.SCORES_MANIFEST, 'scores_manifest_sha256': reader.SCORES_MANIFEST_SHA256,
                       'margin_days': MARGIN_DAYS, 'margin_days_sha256': reader.sha256(reader.cp25_file(root, MARGIN_DAYS)),
                       'snapshot_sha256': reader.SNAPSHOT_SHA256, 'ledger_days_read': int(len(table)),
                       'first_day': str(table.day.min()), 'last_day': str(table.day.max())},
            'streams': {'A1_nominal': 'members.nominal.D2.quantiles', 'A1_gfs6': 'members.gfs6.D2.quantiles',
                        'v2_nominal': 'policies.nominal/v2.native_quantiles'},
            'scores': 'cp28.scoring.daily: p50 MAE and seven-quantile WIS on CP-25 eligible keys; hours within a '
                      'date, dates equally',
            'population': 'paired: eligible scored dates where all three streams have valid native quantiles',
            'summary': summary, 'rule': rule(summary), 'sensitivity_unpaired_affected': unpaired_sensitivity(table),
            'daily_rows': 'reports/cp17/gfs-check-daily.csv', 'daily_rows_sha256': hashlib.sha256(canonical).hexdigest()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    table = daily_table(args.root)
    result = build(args.root, table)
    result['crosscheck'] = crosscheck(args.root, table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / 'gfs-check-daily.csv').write_text(daily_csv(table), encoding='utf-8')
    args.out.write_text(json.dumps(result, indent=1, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    print(result['rule']['outcome'], json.dumps(result['rule']['v2_excess_over_A1_gfs6']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
