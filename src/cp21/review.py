"""Representative reproduction for the independent Integration review (§17.10 item 12), charged as
review. Run under the monitor as ``python -m cp21.review --out <file outside the checkout> ...``.

Each check recomputes from the frozen inputs and compares with *committed* rows, never with the
Lead's ignored caches:

* ``--component FOLD:DAY`` refits HG's A1_w and B2_w; A1/2 + B2/2 must equal the accepted CP-20 HG
  central bit for bit;
* ``--lgbm FOLD:DAY:ARM`` repeats one arm's training-only selection and fits; its central forecast
  must equal the committed prediction's `central` bit for bit, and its selected configurations,
  validation losses and tree hashes the committed `fits.parquet` rows;
* ``--replay FOLD:FIRST:LAST`` replays the four new arms from the committed admission-freeze states
  through the single H-layer path; every vector must equal the committed predictions;
* ``--mask FOLD:DAY:ARM`` repeats the delivery-day/future mask (exactly 0.0) and the non-uniform D-1
  price mutation (moves) for one arm.
Nothing is written inside the repository.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cp15.data import LABELS, prepare
from cp16.residuals import SharedResidualState
from cp20.components import HGComponents, training_rows
from .budget import atomic, ledger
from .components import augment, counted_lear
from .controls import admission_states
from .execution import OUT, NEW_ARMS, Sources, _truth, charge_fits, fit_identity, hg_cache_dir, replay
from .inputs import hg_identity, load, weather_design, weather_matrix
from .jobs import stamp
from .lgbm import fit_arm


def _committed(root: Path, policy: str, day: date) -> pd.DataFrame:
    path = root / (OUT / 'predictions.parquet' if policy in NEW_ARMS else 'reports/weather-ablation/predictions.parquet')
    frame = pd.read_parquet(path, filters=[('policy', '==', policy)])
    return frame.loc[pd.to_datetime(frame.delivery_date).dt.date.eq(day)].sort_values('timestamp_utc')


def component(root: Path, fold: str, day: date, budget) -> dict:
    data = load(root)
    aug, present = augment(data, weather_design(root))
    rows = data.rows(day)
    out = {}
    with counted_lear(budget, 'review') as fit_day:
        for policy in ('A1', 'B2'):
            train = training_rows(aug, day, policy)
            if not present[train].all() or not present[rows].all():
                raise ValueError('a training/forecast row lacks a frozen weather record')
            out[policy], _ = fit_day(aug, day, policy, rows=rows)
    central = out['A1'] / 2 + out['B2'] / 2
    accepted = _committed(root, 'HG', day).central.to_numpy(float)
    cached = HGComponents(hg_cache_dir(), fold, data, hg_identity(root)).get(day)[1]
    return {'check': 'component', 'fold': fold, 'day': str(day), 'n_hours': int(len(rows)),
            'hg_central_bitwise_vs_accepted_cp20': bool(np.array_equal(central, accepted)),
            'components_bitwise_vs_cp20_cache': {p: bool(np.array_equal(out[p], cached[p])) for p in ('A1', 'B2')}}


def lgbm(root: Path, fold: str, day: date, arm: str, budget) -> dict:
    data = load(root)
    wx, present = weather_matrix(weather_design(root), data)
    central, records, selection = fit_arm(data, wx, present, day, arm, charge=charge_fits(budget, purpose='review', main=False))
    committed = _committed(root, arm, day).central.to_numpy(float)
    fits = pd.read_parquet(root / OUT / 'fits.parquet', filters=[('arm', '==', arm), ('delivery_date', '==', str(day))])
    by_model = {}
    for model, part in fits.groupby('model'):
        ours = [r for r in records if r['model'] == model]
        final = part.loc[part.role.eq('final')]
        by_model[model] = {
            'selected_equal': selection[model]['selected'] == final.config.iloc[0],
            'validation_mae_equal': all(np.isclose(r['validation_mae'], part.loc[part.role.eq('inner') & part.config.eq(r['config']),
                                                                                 'validation_mae'].iloc[0], rtol=0, atol=0)
                                        for r in ours if r['role'] == 'inner'),
            'tree_hashes_equal': {(r['role'], r['config']): r['model_sha256'] for r in ours}
            == {(r.role, r.config): r.model_sha256 for r in part.itertuples()}}
    return {'check': 'lgbm', 'fold': fold, 'day': str(day), 'arm': arm, 'fits': len(records),
            'central_bitwise_vs_committed': bool(np.array_equal(central, committed)), 'models': by_model}


def replay_check(root: Path, fold: str, first: date, last: date, budget) -> dict:
    commit, admitted = admission_states(root)
    data = load(root)
    fold_first = {'fold_1': date(2020, 7, 1), 'fold_2': date(2021, 4, 1), 'fold_3': date(2022, 7, 1),
                  'fold_4': date(2025, 5, 1), 'fold_5': date(2026, 1, 8)}[fold]
    if first != fold_first:
        raise ValueError('a review replay starts at the fold\'s first evaluation day, from the admission-freeze state')
    states = {arm: SharedResidualState.from_dict(admitted[fold][arm]) for arm in NEW_ARMS}
    frames, log = [], {'origins': []}
    dates = list(pd.date_range(first, last).date)
    replay(data, fold, dates, Sources(fold, data, fit_identity(root), hg_identity(root), NEW_ARMS), states, _truth(data),
           None, log, 'review', frames, budget, 'policy_days_review')
    ours = pd.concat(frames, ignore_index=True).sort_values(['policy', 'timestamp_utc']).reset_index(drop=True)
    committed = pd.read_parquet(root / OUT / 'predictions.parquet')
    committed = committed.loc[committed.fold.eq(fold) & pd.to_datetime(committed.delivery_date).dt.date.between(first, last)]
    committed = committed.sort_values(['policy', 'timestamp_utc']).reset_index(drop=True)
    num = ['central', 'scale', 'level', *LABELS]
    return {'check': 'replay', 'fold': fold, 'first': str(first), 'last': str(last), 'rows': int(len(ours)),
            'admission_freeze_commit': commit,
            'bitwise_equal_to_committed': bool(len(ours) == len(committed)
                                               and np.array_equal(ours[num].to_numpy(float), committed[num].to_numpy(float)))}


def mask(root: Path, fold: str, day: date, arm: str, budget) -> dict:
    data = load(root)
    design = weather_design(root)
    charge = charge_fits(budget, purpose='review', main=False)

    def run(frame, table=None):
        d = prepare(frame, data.p, data.spec) if frame is not None else data
        dz = design if table is None else type(design)(table, design.sha256)
        wx, present = weather_matrix(dz, d)
        return fit_arm(d, wx, present, day, arm, charge=charge)[0]
    base = run(None)
    masked = data.frame.copy()
    masked.loc[masked.delivery_date >= day, 'price_eur_mwh'] = np.nan
    masked.loc[masked.delivery_date > day, 'load_forecast_mw'] = 1e8
    future = design.table.copy()
    future.loc[future.index.get_level_values(0) > day, ['wx_wind10_mean', 'wx_wind100_mean', 'wx_dswrf_mean']] = 1e6
    changed = data.frame.copy()
    hours = pd.to_datetime(changed.timestamp_utc, utc=True).dt.tz_convert('Europe/Berlin').dt.hour
    changed.loc[changed.delivery_date.eq(day - timedelta(days=1)) & hours.between(17, 21), 'price_eur_mwh'] += 300.0
    return {'check': 'mask', 'fold': fold, 'day': str(day), 'arm': arm,
            'delivery_day_and_future_mask_max_abs': float(np.max(np.abs(run(masked, future) - base))),
            'nonuniform_d1_mutation_max_abs': float(np.max(np.abs(run(changed) - base)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--component', action='append', default=[])
    ap.add_argument('--lgbm', action='append', default=[])
    ap.add_argument('--replay', action='append', default=[])
    ap.add_argument('--mask', action='append', default=[])
    args = ap.parse_args()
    root = Path.cwd()
    if args.out.resolve().is_relative_to(root.resolve()):
        raise SystemExit('write review output outside the checkout')
    if 'CP21_JOB_INDEX' not in os.environ:
        raise SystemExit('run under the monitor (resource accounting, §17.8)')
    budget = ledger()
    results = []
    for spec in args.component:
        fold, day = spec.split(':')
        results.append(component(root, fold, date.fromisoformat(day), budget))
    for spec in args.lgbm:
        fold, day, arm = spec.split(':')
        results.append(lgbm(root, fold, date.fromisoformat(day), arm, budget))
    for spec in args.replay:
        fold, first, last = spec.split(':')
        results.append(replay_check(root, fold, date.fromisoformat(first), date.fromisoformat(last), budget))
    for spec in args.mask:
        fold, day, arm = spec.split(':')
        results.append(mask(root, fold, date.fromisoformat(day), arm, budget))
    atomic(args.out, {'schema': 'cp21-review-reproduction-v1', 'written_utc': stamp(), 'results': results})
    print(json.dumps(results), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
