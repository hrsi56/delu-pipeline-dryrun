"""Representative reproduction for the independent Integration review (§21.10 item 15), charged as review.
Run under the monitor as ``python -m cp23.review --out <file outside the checkout> ...``.

Each check recomputes from the frozen inputs and compares with *committed* rows:

* ``--score`` recomputes every metric, interval, contrast reading and the `cp23-adoption` verdict from
  the committed predictions (one reference and one bootstrap pass) and compares them with the committed
  tables;
* ``--ddnn FOLD:DAY`` repeats one origin's DDNN fit -- the fold's committed configuration, the four seeds,
  early stopping -- and its ensemble and quantile emission. D's central and seven quantiles must equal the
  committed `predictions.parquet` and `members.parquet`, and every member's weights hash, best epoch and
  epochs run must equal the committed `fits.parquet`;
* ``--replay POLICY:FOLD:FIRST:LAST`` replays v5 or v3+D from the committed admission-freeze state through
  the H layer; every vector must equal the committed one.

Nothing is written inside the repository.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cp15.data import LABELS
from cp16.residuals import SharedResidualState
from .budget import atomic, charge_fits, ledger
from .controls import admission_states
from .evaluate import clean, load_predictions
from .execution import OUT, Sources, _identities, _truth, replay, selected_config
from .features import encode, target
from .inputs import load, weather_design, weather_matrix
from .jobs import stamp
from .member import fit_origin
from .reference import require_passing_record
from . import scoring as S


def score(root: Path, budget) -> dict:
    budget.reserve(reference_passes=1, analysis_passes=1)
    decisions = json.loads((root / OUT / 'decisions.json').read_text())
    predictions, expected = load_predictions(root)
    policies = list(S.SAVED + S.NEW)
    tables = S.evaluate(predictions, expected, policies)
    out = {'policies': len(policies)}
    for name, keys in (('metrics', ['policy', 'scope', 'fold']), ('uncertainty', ['scope', 'candidate', 'baseline', 'metric']),
                       ('criteria', ['policy', 'criterion', 'metric', 'scope'])):
        committed = pd.read_csv(root / OUT / f'{name}.csv')
        ours = tables[name]
        num = [c for c in committed.columns if c not in keys and pd.api.types.is_numeric_dtype(committed[c])
               and c in ours and pd.api.types.is_numeric_dtype(ours[c])]
        m = committed.merge(ours, on=keys, suffixes=('_c', '_r'), how='outer', indicator=True)
        diff, nan_mismatch = 0.0, 0
        for c in num:
            a, b = m[f'{c}_c'].to_numpy(float), m[f'{c}_r'].to_numpy(float)
            nan_mismatch += int((np.isnan(a) != np.isnan(b)).sum())
            both = ~np.isnan(a) & ~np.isnan(b)
            if both.any():
                diff = max(diff, float(np.max(np.abs(a[both] - b[both]))))
        out[name] = {'rows_committed': int(len(committed)), 'rows_recomputed': int(len(ours)),
                     'unmatched': int((m._merge != 'both').sum()), 'max_abs_numeric_difference': diff,
                     'nan_pattern_mismatches': nan_mismatch, 'numeric_columns_compared': len(num)}
    s = clean(tables['summary'])
    out['verdict_equal'] = s['adoption']['verdict'] == decisions['adoption']['verdict']
    out['first_unmet_equal'] = s['adoption']['first_unmet_condition'] == decisions['adoption']['first_unmet_condition']
    out['unmet_equal'] = s['adoption']['unmet_conditions'] == decisions['adoption']['unmet_conditions']
    out['readings_equal'] = all(s['contrasts'][k]['reading'] == v['reading'] for k, v in decisions['contrasts'].items())
    return out


def ddnn(root: Path, fold: str, day: date, budget) -> dict:
    require_passing_record(root)
    config = selected_config(root, fold) if (root / OUT / 'selection.json').exists() else None
    data = load(root)
    wx, present = weather_matrix(weather_design(root), data)
    x, _ = encode(data)
    fit = fit_origin(data, x, wx, target(data), present, day, config, charge=charge_fits(budget, 'review', main=False))
    pred = pd.read_parquet(root / OUT / 'predictions.parquet', filters=[('policy', '==', 'D')])
    pred = pred.loc[pred.fold.eq(fold) & pd.to_datetime(pred.delivery_date).dt.date.eq(day)].sort_values('timestamp_utc')
    members = pd.read_parquet(root / OUT / 'members.parquet')
    mm = members.loc[members.fold.eq(fold) & pd.to_datetime(members.delivery_date).dt.date.eq(day)].sort_values('timestamp_utc')
    fits = pd.read_parquet(root / OUT / 'fits.parquet', filters=[('delivery_date', '==', str(day))])
    fits = fits.loc[fits.fold.eq(fold) & fits.stage.eq('origin')].sort_values('seed')
    ours = [(r['seed'], r['params_sha256'], r['best_epoch'], r['epochs_run']) for r in fit['members']]
    theirs = list(zip(fits.seed, fits.params_sha256, fits.best_epoch, fits.epochs_run))
    return {'fold': fold, 'day': str(day), 'config': config, 'n_hours': len(fit['timestamp_utc']),
            'D_central_bitwise_vs_committed': bool(np.array_equal(fit['central'], pred.central.to_numpy(float))),
            'D_quantiles_bitwise_vs_committed': bool(np.array_equal(fit['quantiles'], pred[LABELS].to_numpy(float))),
            'D_equals_members_table': bool(np.array_equal(fit['central'], mm.D.to_numpy(float))),
            'members_equal_committed_fits': [tuple(map(str, o)) for o in ours] == [tuple(map(str, t)) for t in theirs],
            'quantiles_finite_ordered': bool(np.isfinite(fit['quantiles']).all() and (np.diff(fit['quantiles'], axis=1) > 0).all()),
            'p50_is_central': bool(np.array_equal(fit['central'], fit['quantiles'][:, 3])), 'crossed_rows': fit['crossed_rows'],
            'ensemble_is_member_quantile_mean': bool(np.array_equal(fit['z_quantiles'], fit['member_z_quantiles'].mean(axis=0)))}


def replay_check(root: Path, policy: str, fold: str, first: date, last: date, budget) -> dict:
    commit, states = admission_states(root)
    fit_ident, cp21_ident, hg_ident, hashes = _identities(root)
    data = load(root)
    state = {policy: SharedResidualState.from_dict(states[fold][policy])}
    evaluation_start = data.spec.development_folds[int(fold[-1]) - 1].evaluation.start
    dates = list(pd.date_range(evaluation_start, last).date)
    frames, log = [], {'origins': []}
    replay(data, fold, dates, Sources(fold, data, fit_ident, cp21_ident, hg_ident, hashes, (policy,)), state, _truth(data),
           None, log, 'review', frames, budget, 'policy_days_review')
    ours = pd.concat(frames, ignore_index=True)
    ours = ours.loc[pd.to_datetime(ours.delivery_date).dt.date >= first].sort_values('timestamp_utc')
    committed = pd.read_parquet(root / OUT / 'predictions.parquet', filters=[('policy', '==', policy)])
    committed = committed.loc[committed.fold.eq(fold) & pd.to_datetime(committed.delivery_date).dt.date.between(first, last)]
    committed = committed.sort_values('timestamp_utc')
    a, b = ours[['central', *LABELS]].to_numpy(float), committed[['central', *LABELS]].to_numpy(float)
    return {'policy': policy, 'fold': fold, 'first': str(first), 'last': str(last), 'admission_freeze_commit': commit,
            'rows': int(len(a)), 'bitwise_equal': bool(a.shape == b.shape and np.array_equal(a, b))}


def main() -> int:
    if 'CP23_JOB_INDEX' not in os.environ:
        raise SystemExit('review checks run only under the monitor (resource accounting)')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--ddnn', action='append', default=[])
    ap.add_argument('--replay', action='append', default=[])
    args = ap.parse_args()
    root = Path.cwd()
    budget = ledger()
    result = {'schema': 'cp23-review-v1', 'candidate_root': str(root)}
    if args.score:
        result['score'] = score(root, budget)
    result['ddnn'] = [ddnn(root, *(lambda f, d: (f, date.fromisoformat(d)))(*x.split(':')), budget) for x in args.ddnn]
    result['replay'] = []
    for x in args.replay:
        policy, fold, first, last = x.split(':')
        result['replay'].append(replay_check(root, policy, fold, date.fromisoformat(first), date.fromisoformat(last), budget))
    result['written_utc'] = stamp()
    atomic(args.out, clean(result))
    print(json.dumps(clean(result), default=str)[:4000], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
