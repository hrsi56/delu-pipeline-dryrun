"""Representative reproduction for the independent Integration review (§20.10 item 12), charged as
review. Run under the monitor as ``python -m cp22.review --out <file outside the checkout> ...``.

Each check recomputes from the frozen inputs and compares with *committed* rows, never with the
Lead's ignored caches:

* ``--score`` recomputes every metric, interval and the three §20.6 verdicts from the committed
  predictions (one reference and one bootstrap pass) and compares them with the committed tables;
* ``--pn FOLD:DAY`` repeats PN's eight fits at one origin; PN-avg and PN-sel must equal the
  committed `members.parquet`, and the selection, losses and trees the committed `fits.parquet`;
* ``--replay POLICY:FOLD:FIRST:LAST`` replays one policy from the committed admission-freeze state
  through the H or DL layer (ACI updates included); every vector must equal the committed one.
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
from .budget import atomic, ledger
from .controls import admission_states
from .evaluate import clean, load_predictions
from .execution import LAYER, OUT, W_POLICIES, Sources, _identities, _truth, charge_fits, replay, state_from_dict
from .inputs import load, weather_design, weather_matrix
from .jobs import stamp
from .pn import fit_pn
from . import scoring as S


def score(root: Path, budget) -> dict:
    budget.reserve(reference_passes=1, analysis_passes=1)
    decisions = json.loads((root / OUT / 'decisions.json').read_text())
    w = decisions['replacement']['winner']
    predictions, expected = load_predictions(root, w=bool(w))
    policies = list(S.SAVED + S.FIXED_NEW + (S.W_ARMS if w else ()))
    tables = S.evaluate(predictions, expected, policies, w)
    out = {'policies': len(policies), 'W': w}
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
    out['verdicts_equal'] = {k: s[k]['verdict'] == decisions[k]['verdict'] for k in ('replacement', 'dynamic_layer', 'fast_component')}
    out['first_unmet_equal'] = {k: s[k]['first_unmet_condition'] == decisions[k]['first_unmet_condition']
                                for k in ('replacement', 'dynamic_layer', 'fast_component')}
    out['readings_equal'] = all(s['contrasts'][k]['reading'] == v['reading'] for k, v in decisions['contrasts'].items())
    return out


def pn(root: Path, fold: str, day: date, budget) -> dict:
    data = load(root)
    wx, present = weather_matrix(weather_design(root), data)
    member, records = fit_pn(data, wx, present, day, charge=charge_fits(budget, purpose='review', main=False))
    members = pd.read_parquet(root / OUT / 'members.parquet')
    mm = members.loc[members.fold.eq(fold) & pd.to_datetime(members.delivery_date).dt.date.eq(day)].sort_values('timestamp_utc')
    fits = pd.read_parquet(root / OUT / 'fits.parquet', filters=[('delivery_date', '==', str(day))])
    fits = fits.loc[fits.fold.eq(fold)]
    ours = sorted((r['role'], r['config'], r['model_sha256']) for r in records)
    theirs = sorted(zip(fits.role, fits.config, fits.model_sha256))
    inner = fits.loc[fits.role.eq('inner')].set_index('config').validation_mae
    final = fits.loc[fits.role.eq('final')]
    return {'fold': fold, 'day': str(day), 'pn_avg_bitwise': bool(np.array_equal(member['pn_avg'], mm['PN-avg'].to_numpy(float))),
            'pn_sel_bitwise': bool(np.array_equal(member['pn_sel'], mm['PN-sel'].to_numpy(float))),
            'trees_equal': ours == theirs,
            'validation_losses_equal': all(member['validation_mae'][k] == inner[k] for k in inner.index),
            'selected_equal': member['selected'] == final.loc[final.selected.astype(bool), 'config'].iloc[0],
            'committed_rows': int(len(fits))}


def replay_check(root: Path, policy: str, fold: str, first: date, last: date, budget) -> dict:
    name = 'lineage-w.json' if policy in W_POLICIES else 'lineage.json'
    commit, states = admission_states(root, name)
    w = json.loads((root / OUT / 'replacement.json').read_text())['winner'] if policy in W_POLICIES else None
    fit_ident, cp21_ident, hg_ident, hashes = _identities(root)
    data = load(root)
    state = {policy: state_from_dict(policy, states[fold][policy])}
    evaluation_start = data.spec.development_folds[int(fold[-1]) - 1].evaluation.start
    dates = list(pd.date_range(evaluation_start, last).date)
    frames, log = [], {'origins': []}
    replay(data, fold, dates, Sources(fold, data, fit_ident, cp21_ident, hg_ident, hashes, (policy,), w_policy=w), state,
           _truth(data), None, log, 'review', frames, budget, 'policy_days_review')
    ours = pd.concat(frames, ignore_index=True)
    ours = ours.loc[pd.to_datetime(ours.delivery_date).dt.date >= first].sort_values('timestamp_utc')
    pred = 'predictions-w.parquet' if policy in W_POLICIES else 'predictions.parquet'
    committed = pd.read_parquet(root / OUT / pred, filters=[('policy', '==', policy)])
    committed = committed.loc[committed.fold.eq(fold) & pd.to_datetime(committed.delivery_date).dt.date.between(first, last)]
    committed = committed.sort_values('timestamp_utc')
    a, b = ours[['central', *LABELS]].to_numpy(float), committed[['central', *LABELS]].to_numpy(float)
    return {'policy': policy, 'layer': LAYER[policy], 'fold': fold, 'first': str(first), 'last': str(last),
            'admission_freeze_commit': commit, 'rows': int(len(a)), 'bitwise_equal': bool(a.shape == b.shape and np.array_equal(a, b)),
            'alpha_t_after': state[policy].alpha if LAYER[policy] != 'H' else None}


def main() -> int:
    if 'CP22_JOB_INDEX' not in os.environ:
        raise SystemExit('review checks run only under the monitor (resource accounting)')
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--pn', action='append', default=[])
    ap.add_argument('--replay', action='append', default=[])
    args = ap.parse_args()
    root = Path.cwd()
    budget = ledger()
    result = {'schema': 'cp22-review-v1', 'candidate_root': str(root)}
    if args.score:
        result['score'] = score(root, budget)
    result['pn'] = [pn(root, *(lambda f, d: (f, date.fromisoformat(d)))(*x.split(':')), budget) for x in args.pn]
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
