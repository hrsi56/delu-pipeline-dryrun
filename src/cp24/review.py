"""Representative reproduction for the independent Integration review (§23.13 item 17), charged as review.
Run under the monitor as ``python -m cp24.review --out <file outside the checkout> ...``.

Each check recomputes from frozen inputs and compares with *committed* rows:

* ``--score K`` recomputes every metric, interval (95% and 97.5%), contrast reading and the `cp24-adoption`
  verdict of scored attempt K from its committed predictions (one reference and one bootstrap pass) and
  compares them with the committed tables;
* ``--gate R`` recomputes round R's G0-G3, pooled and by fold, from the stored gate member records and v4's
  gate cache, and compares them with the committed `gate.json`;
* ``--trial R:FOLD:TRIAL:BATCH`` repeats one search trial's fit on one validation batch (its configuration
  and seed from the committed ledger) and compares its pinball and absolute-error sums with the ledger;
* ``--ensemble K:FOLD:DAY`` repeats one origin's eight-member DDNN-2 fit and its per-level-median emission;
  D2's central and quantiles must equal the committed `predictions.parquet`, and every member's weights hash
  the committed `fits.parquet`;
* ``--replay K:POLICY:FOLD:FIRST:LAST`` replays v5 or v3+D2 from the committed admission-freeze state through
  the H layer; every vector must equal the committed one.

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

from cp15.data import LABELS
from cp16.residuals import SharedResidualState
from . import design as G
from . import scoring as S
from .budget import atomic, charge_fits, ledger
from .controls import admission_states
from .evaluate import clean, guard_report, load_predictions
from .execution import Sources, _identities, _truth, fit_origin, replay
from .gate import evaluate_gate, v4_identity
from .inputs import load, weather_design
from .jobs import art, stamp
from .member import Keys, fit_member
from .preflight import fold_table
from .protocol import attempt_dir
from .reference import require_passing_record
from .search import score_batch


def compare_tables(root: Path, k: int, tables: dict) -> dict:
    out = {}
    ad = attempt_dir(root, k)
    for name, keys in (('metrics', ['policy', 'scope', 'fold']), ('uncertainty', ['scope', 'candidate', 'baseline', 'metric']),
                       ('criteria', ['policy', 'criterion', 'metric', 'scope'])):
        committed = pd.read_csv(ad / f'{name}.csv')
        ours = tables[name]
        num = [c for c in committed.columns if c not in keys and pd.api.types.is_numeric_dtype(committed[c])
               and c in ours and pd.api.types.is_numeric_dtype(ours[c])]
        m = committed.merge(ours, on=keys, suffixes=('_c', '_r'), how='outer', indicator=True)
        diff, nan_mismatch = 0.0, 0
        for c in num:
            a, b = m[f'{c}_c'].to_numpy(float), m[f'{c}_r'].to_numpy(float)
            nan_mismatch += int((np.isnan(a) != np.isnan(b)).sum())
            both = ~(np.isnan(a) | np.isnan(b))
            if both.any():
                diff = max(diff, float(np.max(np.abs(a[both] - b[both]))))
        out[name] = {'rows_committed': len(committed), 'rows_recomputed': len(ours), 'unmatched': int((m['_merge'] != 'both').sum()),
                     'max_abs_difference': diff, 'nan_mismatches': nan_mismatch}
    return out


def score(root: Path, k: int, budget) -> dict:
    budget.reserve(reference_passes=1, bootstrap_passes=1)
    decisions = json.loads((attempt_dir(root, k) / 'decisions.json').read_text())
    predictions, expected = load_predictions(root, k)
    guards = guard_report(root, k)
    tables = S.evaluate(predictions, expected, list(S.SAVED + S.POINT_ONLY + S.NEW), guards_reported=True)
    s = clean(tables['summary'])
    out = {'attempt': k, 'tables': compare_tables(root, k, tables), 'guards_equal_committed':
           guards == json.loads((attempt_dir(root, k) / 'guards.json').read_text()),
           'verdict_recomputed': s['adoption']['verdict'], 'verdict_committed': decisions['adoption']['verdict'],
           'first_unmet_recomputed': s['adoption']['first_unmet_condition'],
           'first_unmet_committed': decisions['adoption']['first_unmet_condition'],
           'readings_equal': {key: s['contrasts'][key]['reading'] == decisions['contrasts'][key]['reading'] for key in s['contrasts']}}
    out['all_equal'] = bool(all(v['unmatched'] == 0 and v['nan_mismatches'] == 0 and v['max_abs_difference'] <= 1e-9
                                for v in out['tables'].values()) and out['verdict_recomputed'] == out['verdict_committed']
                            and out['first_unmet_recomputed'] == out['first_unmet_committed'] and all(out['readings_equal'].values())
                            and out['guards_equal_committed'])
    return out


def gate(root: Path, r: int) -> dict:
    rd = root / 'reports/ddnn2/rounds' / f'round-{r}'
    committed = json.loads((rd / 'gate.json').read_text())
    ens = json.loads((rd / 'ensembles.json').read_text())['folds']
    again = evaluate_gate(root, r, fold_table(root), ens, v4_identity(root))
    same = clean(again['conditions']) == clean(committed['conditions']) and clean(again['by_fold']) == clean(committed['by_fold'])
    return {'round': r, 'passed_recomputed': again['passed'], 'passed_committed': committed['passed'],
            'conditions_and_folds_equal': bool(same), 'conditions': clean(again['conditions'])}


def trial(root: Path, spec: str, budget) -> dict:
    r, fold, t, b = spec.split(':')
    r, t, b = int(r), int(t), int(b)
    ledger_doc = json.loads((root / 'reports/ddnn2/rounds' / f'round-{r}' / 'search-ledger.json').read_text())
    row = next(x for x in ledger_doc['folds'][fold]['trials'] if x['trial'] == t)
    rec = next(x for x in row['batches'] if x['batch'] == b)
    f = next(x for x in fold_table(root) if x['fold'] == fold)
    data = load(root, before=f['search_cutoff'])
    dd = G.build(data, weather_design(root))
    keys = Keys.from_data(data, dd)
    b0 = date.fromisoformat(rec['start'])
    days = np.array([dd.ix(b0 + timedelta(days=j)) for j in range(28)])
    fit = fit_member(dd, b0, row['config'], rec['seed'], days, exclude_uncovered=True, charge=charge_fits(budget, 'review'))
    sc = score_batch(keys, days, fit['eur'])
    return {'trial': spec, 'pinball_recomputed': sc['pinball_sum'] / sc['pinball_count'], 'pinball_ledger': rec['pinball'],
            'mae_recomputed': sc['abs_error_sum'] / sc['hours'], 'mae_ledger': rec['mae'],
            'epochs_recomputed': fit['member']['epochs_run'], 'epochs_ledger': rec['epochs_run'],
            'equal': bool(sc['pinball_sum'] / sc['pinball_count'] == rec['pinball'] and fit['member']['epochs_run'] == rec['epochs_run'])}


def ensemble(root: Path, spec: str, budget) -> dict:
    k, fold, day_s = spec.split(':')
    k, day = int(k), date.fromisoformat(day_s)
    p = json.loads((attempt_dir(root, k) / 'protocol.json').read_text())
    f = next(x for x in fold_table(root) if x['fold'] == fold)
    data = load(root, before=f['evaluation_start']) if day < f['evaluation_start'] else load(root)
    dd = G.build(data, weather_design(root))
    keys = Keys.from_data(data, dd)
    out = fit_origin(dd, keys, data, day, p['folds'][fold]['ensemble'], charge_fits(budget, 'review'))
    preds = pd.read_parquet(attempt_dir(root, k) / 'predictions.parquet', filters=[('policy', '==', 'D2')])
    preds = preds.loc[pd.to_datetime(preds.delivery_date).dt.date.eq(day) & preds.fold.eq(fold)].sort_values('timestamp_utc')
    fits = pd.read_parquet(attempt_dir(root, k) / 'fits.parquet')
    fits = fits.loc[fits.fold.eq(fold) & fits.delivery_date.eq(day_s)].sort_values('member')
    return {'ensemble': spec, 'n_hours': len(preds),
            'D2_central_bitwise': bool(len(preds) and np.array_equal(out['central'], preds.central.to_numpy(float))),
            'D2_quantiles_bitwise': bool(len(preds) and np.array_equal(out['quantiles'], preds[LABELS].to_numpy(float))),
            'member_weights_bitwise': [m['record']['params_sha256'] for m in out['members']] == fits.params_sha256.tolist()}


def replay_check(root: Path, spec: str, budget) -> dict:
    k, policy, fold, first_s, last_s = spec.split(':')
    k = int(k)
    fit_ident, cp21_ident, hg_ident, hashes = _identities(root, k)
    _, admitted = admission_states(root, k)
    data = load(root)
    dates = list(pd.date_range(first_s, last_s).date)
    states = {policy: SharedResidualState.from_dict(admitted[fold][policy])}
    frames, log = [], {'origins': []}
    replay(k, data, fold, dates, Sources(k, fold, data, fit_ident, cp21_ident, hg_ident, hashes, (policy,)), states,
           _truth(data), None, log, 'review', frames, budget, 'policy_days_review')
    ours = pd.concat(frames, ignore_index=True).sort_values('timestamp_utc')
    committed = pd.read_parquet(attempt_dir(root, k) / 'predictions.parquet', filters=[('policy', '==', policy)])
    committed = committed.loc[pd.to_datetime(committed.delivery_date).dt.date.isin(dates) & committed.fold.eq(fold)].sort_values('timestamp_utc')
    num = ['central', *LABELS]
    return {'replay': spec, 'rows': len(ours), 'bitwise_equal': bool(np.array_equal(ours[num].to_numpy(float), committed[num].to_numpy(float)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--score', type=int, action='append', default=[])
    ap.add_argument('--gate', type=int, action='append', default=[])
    ap.add_argument('--trial', action='append', default=[])
    ap.add_argument('--ensemble', action='append', default=[])
    ap.add_argument('--replay', action='append', default=[])
    args = ap.parse_args()
    if 'CP24_JOB_INDEX' not in os.environ:
        raise SystemExit('run under the monitor (resource accounting)')
    root = Path.cwd()
    require_passing_record(root)
    budget = ledger()
    result = {'schema': 'cp24-review-v1', 'written_utc': stamp(), 'score': [], 'gate': [], 'trial': [], 'ensemble': [], 'replay': []}
    for k in args.score:
        result['score'].append(score(root, k, budget))
    for r in args.gate:
        result['gate'].append(gate(root, r))
    for spec in args.trial:
        result['trial'].append(trial(root, spec, budget))
    for spec in args.ensemble:
        result['ensemble'].append(ensemble(root, spec, budget))
    for spec in args.replay:
        result['replay'].append(replay_check(root, spec, budget))
    atomic(args.out, clean(result))
    print(json.dumps(clean(result), default=str)[:4000], flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
