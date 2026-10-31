"""The scoring job of scored attempt k (capstone v21-r11 §23.8-§23.9): one metric-only reference pass and
one bootstrap pass, then `cp24-adoption` applied mechanically.

It scores the saved references (CP-20's B0, B1, B2, B3, A1 and HG; CP-21's v4; CP-23's D and v3+D), the
point-only reference L, and the attempt's three new policies on the 10,747 keys. It writes the tables,
the stored replicates, the guard report, `adoption.json`, `decisions.json` and `summary.json` under
`reports/ddnn2/attempt-<k>/`.

**Consistency, in the same pass and on the same index set.** HG's and v4's recomputed §8 rows must equal
CP-20's and CP-21's committed criteria, and v4 − v3 must equal CP-21's committed interval rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cp15.data import sha
from . import scoring as S
from .budget import atomic, ledger
from .execution import NEW_POLICIES, committed_at_head
from .inputs import identities
from .jobs import art
from .protocol import attempt_dir, check_protocol

TABLES = ('metrics', 'diagnostics', 'uncertainty', 'criteria')


def clean(obj):
    """JSON-safe copy: NaN/inf -> None (an undefined value, never a number), numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


COLS = ['fold', 'policy', 'timestamp_utc', 'delivery_date', 'y_true', 'central', 'scale', 'level', *S.QUANTILES]


def point_only_L(root: Path, k: int, hg: pd.DataFrame) -> pd.DataFrame:
    """L = mean(L-N, L-R), v4's LightGBM member, as a point-only reference: its seven 'quantiles' are its
    central (no interval layer), on HG's keys, scale, level and truth."""
    mem = pd.read_parquet(attempt_dir(root, k) / 'members.parquet')
    mem['timestamp_utc'] = pd.to_datetime(mem.timestamp_utc, utc=True).dt.as_unit('ns')
    base = hg.set_index(['fold', 'timestamp_utc'])
    m = mem.set_index(['fold', 'timestamp_utc']).reindex(base.index)
    if m['L'].isna().any():
        raise ValueError('L does not cover the 10,747 keys')
    out = base[['delivery_date', 'y_true', 'scale', 'level']].copy()
    out['central'] = m['L'].to_numpy(float)
    for q in S.QUANTILES:
        out[q] = out['central']
    out['policy'] = 'L'
    return out.reset_index()[COLS]


def load_predictions(root: Path, k: int):
    saved = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet')
    saved = saved.loc[saved.policy.isin(('B0', 'B1', 'B2', 'B3', 'A1', 'HG'))]
    v4 = pd.read_parquet(root / 'reports/block-challenger/predictions.parquet', filters=[('policy', '==', 'HGL')])
    cp23 = pd.read_parquet(root / 'reports/distribution-challenger/predictions.parquet')
    cp23 = cp23.loc[cp23.policy.isin(('D', 'v3+D'))]
    new = pd.read_parquet(attempt_dir(root, k) / 'predictions.parquet')
    both = pd.concat([saved[COLS], v4[COLS], cp23[COLS], new[COLS]], ignore_index=True)
    both['timestamp_utc'] = pd.to_datetime(both.timestamp_utc, utc=True).dt.as_unit('ns')
    hg = both.loc[both.policy.eq('HG')]
    both = pd.concat([both, point_only_L(root, k, hg)], ignore_index=True)
    expected = pd.read_parquet(root / 'reports/cp15/predictions.parquet', columns=['fold', 'timestamp_utc', 'delivery_date'],
                               filters=[('policy', '==', 'B0')])
    return both, expected


def consistency(root: Path, uncertainty, criteria) -> dict:
    out = {}
    for policy, path in (('HG', 'reports/weather-ablation/criteria.csv'), ('HGL', 'reports/block-challenger/criteria.csv')):
        theirs = pd.read_csv(root / path)
        theirs = theirs.loc[theirs.policy.eq(policy)].reset_index(drop=True)
        ours = criteria.loc[criteria.policy.eq(policy)].reset_index(drop=True)
        merged = ours.merge(theirs, on=['criterion', 'metric', 'scope'], suffixes=('', '_committed'))
        out[f'{policy}_section8_rows_equal_committed'] = bool(
            len(merged) == len(theirs) == len(ours) and np.allclose(merged.actual, merged.actual_committed, rtol=0, atol=1e-12)
            and (merged.status == merged.status_committed).all())
    theirs = pd.read_csv(root / 'reports/block-challenger/uncertainty.csv')
    theirs = theirs.loc[theirs.candidate.eq('HGL') & theirs.baseline.eq('HG')]
    m = uncertainty.loc[uncertainty.candidate.eq('HGL') & uncertainty.baseline.eq('HG')].merge(
        theirs, on=['scope', 'candidate', 'baseline', 'metric'], suffixes=('', '_committed'))
    diff = max(float(np.max(np.abs(m[c] - m[f'{c}_committed']))) for c in ('difference', 'ci_lower', 'ci_upper'))
    out['HGL-HG_interval_rows'] = int(len(m))
    out['HGL-HG_max_abs_difference_vs_committed'] = diff
    out['HGL-HG_equal_committed'] = bool(len(m) == 12 and diff <= 1e-12)
    out['all_equal'] = all(v for k, v in out.items() if k.endswith('equal_committed'))
    return out


def guard_report(root: Path, k: int) -> dict:
    """Every guard activation of the attempt's DDNN-2 fits, per member fit and in total (§23.3)."""
    totals = {'member_fits': 0, 'cap_slot_levels_forecast': 0, 'winsor_values_train': 0, 'winsor_values_hold': 0,
              'winsor_values_forecast': 0, 'ensemble_crossings_restored': 0, 'nonfinite_loss_stops': 0}
    by_fold = {}
    for path in sorted((art() / f'attempt-{k}' / 'fits').glob('*/*.json')):
        item = json.loads(path.read_text())
        f = by_fold.setdefault(item['fold'], {key: 0 for key in totals})
        f['ensemble_crossings_restored'] += item['crossed_rows']
        for mbr in item['members']:
            g = mbr['guards']
            vals = {'member_fits': 1, 'cap_slot_levels_forecast': g['cap_forecast_slot_levels'],
                    'winsor_values_train': g['winsor_train']['winsor_low'] + g['winsor_train']['winsor_high'],
                    'winsor_values_hold': g['winsor_hold']['winsor_low'] + g['winsor_hold']['winsor_high'],
                    'winsor_values_forecast': g['winsor_forecast']['winsor_low'] + g['winsor_forecast']['winsor_high'],
                    'nonfinite_loss_stops': sum(e.get('event') == 'nonfinite_training_loss' for e in g['events'])}
            for key, v in vals.items():
                f[key] += v
    for f in by_fold.values():
        for key in totals:
            totals[key] += f[key]
    return {'schema': 'cp24-guards-v1', 'attempt': k, 'totals': totals, 'by_fold': by_fold,
            'guards': 'winsorisation (training-row 0.5/99.5% quantiles, continuous inputs only), the cap (+-1.25 x max|z| on '
                      'the member\'s training rows, every emitted slot-level), sorting of ensemble crossings, and the '
                      'nonfinite-loss stop', 'source': f'.local/artifacts/cp-24/attempt-{k}/fits (every member record)'}


def job_score(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True, choices=(1, 2))
    args = ap.parse_args(rest)
    root, k = Path(root), args.attempt
    check_protocol(root, k)
    out = attempt_dir(root, k)
    lineage = json.loads((out / 'lineage.json').read_text())
    if lineage['execution_stage'] != 'comparison_vectors_complete_not_scored':
        raise ValueError('scoring requires the complete comparison vectors, scored once')
    for name in ('predictions.parquet', 'members.parquet', 'lineage.json'):
        committed_at_head(root, str((out / name).relative_to(root)))
    identities(root)
    guards = guard_report(root, k)
    if guards['totals']['member_fits'] != 636 * 8:
        raise ValueError(f'the guard report covers {guards["totals"]["member_fits"]} member fits, not 5,088')
    atomic(out / 'guards.json', guards)
    ledger().reserve(reference_passes=1, bootstrap_passes=1)
    predictions, expected = load_predictions(root, k)
    policies = list(S.SAVED + S.POINT_ONLY + S.NEW)
    tables = S.evaluate(predictions, expected, policies, guards_reported=True)
    for name in TABLES:
        tables[name].to_csv(out / f'{name}.csv', index=False)
    tables['replicates'].to_parquet(out / 'replicates.parquet', index=False)
    tables['replicate_scores'].to_parquet(out / 'replicate-scores.parquet', index=False)
    check = consistency(root, tables['uncertainty'], tables['criteria'])
    summary = clean(tables['summary'])
    summary['consistency_with_committed'] = clean(check)
    summary['guards'] = guards['totals']
    inputs = {str((out / 'predictions.parquet').relative_to(root)): sha(out / 'predictions.parquet'),
              'reports/weather-ablation/predictions.parquet': sha(root / 'reports/weather-ablation/predictions.parquet'),
              'reports/block-challenger/predictions.parquet': sha(root / 'reports/block-challenger/predictions.parquet'),
              'reports/distribution-challenger/predictions.parquet': sha(root / 'reports/distribution-challenger/predictions.parquet')}
    atomic(out / 'adoption.json', {**summary['adoption'], 'attempt': k,
                                   'consistency_with_committed': summary['consistency_with_committed'], 'inputs_sha256': inputs})
    decisions = {'attempt': k, 'adoption': summary['adoption'], 'contrasts': summary['contrasts'], 'scores': summary['scores'],
                 'original_section8_status': summary['original_section8_status'], 'bootstrap': summary['bootstrap'],
                 'consistency_with_committed': summary['consistency_with_committed'], 'guards': guards['totals'],
                 'evidence_class': 'development_post_selection',
                 'statuses': {'engineering': 'bound to the fresh Integration verdict on the final candidate',
                              'research': 'development_post_selection finding under the pre-registered rule cp24-adoption',
                              'product': 'v1 remains the released product and demo; no designation, freeze or Live (§16)'}}
    atomic(out / 'decisions.json', decisions)
    atomic(out / 'summary.json', summary)
    lineage['execution_stage'] = 'scored_final'
    atomic(out / 'lineage.json', lineage)
    print(json.dumps({'attempt': k, 'verdict': summary['adoption']['verdict'],
                      'first_unmet': summary['adoption']['first_unmet_condition'], 'consistency': check,
                      'scores': {p: summary['scores'][p] for p in ('HG', 'HGL', *NEW_POLICIES)}}, default=str), flush=True)
    return 0 if check['all_equal'] else 9
