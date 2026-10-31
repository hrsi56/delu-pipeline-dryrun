"""The CP-23 scoring job (§21.5-§21.6): one metric-only reference pass and one bootstrap pass.

``score`` scores the seven saved references (CP-20's B0, B1, B2, B3, A1 and HG; CP-21's v4) and the
three new policies on the 10,747 keys, and applies `cp23-adoption` mechanically. It writes
`adoption.json`, `decisions.json` and the tables.

**Consistency, in the same pass and on the same index set.** HG's and v4's recomputed §8 rows must
equal CP-20's and CP-21's committed criteria, and v4 - HG must equal CP-21's committed interval rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cp15.data import sha
from .budget import atomic, ledger
from .execution import OUT, check_protocol, committed_at_head
from .inputs import identities
from . import scoring as S

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


def load_predictions(root: Path):
    cols = ['fold', 'policy', 'timestamp_utc', 'delivery_date', 'y_true', 'central', 'scale', 'level', *S.QUANTILES]
    saved = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet')
    saved = saved.loc[saved.policy.isin(S.SAVED)]
    v4 = pd.read_parquet(root / 'reports/block-challenger/predictions.parquet', filters=[('policy', '==', 'HGL')])
    both = pd.concat([saved[cols], v4[cols], pd.read_parquet(root / OUT / 'predictions.parquet')[cols]], ignore_index=True)
    both['timestamp_utc'] = pd.to_datetime(both.timestamp_utc, utc=True).dt.as_unit('ns')
    expected = pd.read_parquet(root / 'reports/cp15/predictions.parquet', columns=['fold', 'timestamp_utc', 'delivery_date'],
                               filters=[('policy', '==', 'B0')])
    return both, expected


def consistency(root: Path, uncertainty, criteria) -> dict:
    """HG's and v4's §8 rows, and the v4 - HG interval rows, against their committed rows."""
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


def job_score(root: Path, rest) -> int:
    check_protocol(root)
    out = root / OUT
    lineage = json.loads((out / 'lineage.json').read_text())
    if lineage['execution_stage'] != 'comparison_vectors_complete_not_scored':
        raise ValueError('scoring requires the complete comparison vectors, scored once')
    for name in ('predictions.parquet', 'members.parquet', 'lineage.json'):
        committed_at_head(root, str(OUT / name))
    identities(root)
    ledger().reserve(reference_passes=1, analysis_passes=1)
    predictions, expected = load_predictions(root)
    policies = list(S.SAVED + S.NEW)
    tables = S.evaluate(predictions, expected, policies)
    for name in TABLES:
        tables[name].to_csv(out / f'{name}.csv', index=False)
    tables['replicates'].to_parquet(out / 'replicates.parquet', index=False)
    tables['replicate_scores'].to_parquet(out / 'replicate-scores.parquet', index=False)
    check = consistency(root, tables['uncertainty'], tables['criteria'])
    summary = clean(tables['summary'])
    summary['consistency_with_committed'] = clean(check)
    inputs = {str(OUT / 'predictions.parquet'): sha(out / 'predictions.parquet'),
              'reports/weather-ablation/predictions.parquet': sha(root / 'reports/weather-ablation/predictions.parquet'),
              'reports/block-challenger/predictions.parquet': sha(root / 'reports/block-challenger/predictions.parquet')}
    atomic(out / 'adoption.json', {**summary['adoption'], 'consistency_with_committed': summary['consistency_with_committed'],
                                   'inputs_sha256': inputs})
    decisions = {'adoption': summary['adoption'], 'contrasts': summary['contrasts'], 'scores': summary['scores'],
                 'original_section8_status': summary['original_section8_status'], 'bootstrap': summary['bootstrap'],
                 'consistency_with_committed': summary['consistency_with_committed'],
                 'evidence_class': 'development_post_selection',
                 'statuses': {'engineering': 'bound to the fresh Integration verdict on the final candidate',
                              'research': 'development_post_selection finding under the pre-registered rule cp23-adoption',
                              'product': 'v1 remains the released product and demo; no designation, freeze or Live (§16)'}}
    atomic(out / 'decisions.json', decisions)
    atomic(out / 'summary.json', summary)
    lineage['execution_stage'] = 'scored_final'
    atomic(out / 'lineage.json', lineage)
    print(json.dumps({'verdict': summary['adoption']['verdict'], 'first_unmet': summary['adoption']['first_unmet_condition'],
                      'consistency': check, 'scores': {p: summary['scores'][p] for p in ('HG', 'HGL', *S.NEW)}},
                     default=str), flush=True)
    return 0 if check['all_equal'] else 9
