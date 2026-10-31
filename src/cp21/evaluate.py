"""The CP-21 scoring job (§17.5-§17.6): one metric-only reference pass and one bootstrap pass.

Reads the seven saved references from CP-20's accepted predictions (hash-verified), the four new
arms from this checkpoint's committed predictions, and the independent expected keys from CP-15.
Writes every table the report and the packet bind to. Two consistency checks ride in the same
pass, on the same index set: HG's recomputed §8 rows must equal CP-20's committed criteria, and
the HG - H0 intervals must equal CP-20's committed uncertainty rows.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from cp15.data import sha
from .budget import atomic, ledger
from .execution import OUT, check_protocol
from .inputs import identities
from . import scoring as S


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
    saved = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet')
    new = pd.read_parquet(root / OUT / 'predictions.parquet')
    if set(saved.policy) != set(S.SAVED) or set(new.policy) != set(S.NEW):
        raise ValueError('unexpected policy sets')
    cols = ['fold', 'policy', 'timestamp_utc', 'delivery_date', 'y_true', 'central', 'scale', 'level', *S.QUANTILES]
    both = pd.concat([saved[cols], new[cols]], ignore_index=True)
    both['timestamp_utc'] = pd.to_datetime(both.timestamp_utc, utc=True).dt.as_unit('ns')
    expected = pd.read_parquet(root / 'reports/cp15/predictions.parquet', columns=['fold', 'timestamp_utc', 'delivery_date'],
                               filters=[('policy', '==', 'B0')])
    return both, expected


def consistency(root: Path, daily, criteria) -> dict:
    """HG's §8 rows and the HG - H0 intervals, recomputed here, against CP-20's committed rows."""
    cp20c = pd.read_csv(root / 'reports/weather-ablation/criteria.csv')
    ours = criteria.loc[criteria.policy.eq('HG')].reset_index(drop=True)
    theirs = cp20c.loc[cp20c.policy.eq('HG')].reset_index(drop=True)
    keys = ['criterion', 'metric', 'scope']
    merged = ours.merge(theirs, on=keys, suffixes=('', '_cp20'))
    crit_equal = bool(len(merged) == len(theirs) == len(ours)
                      and np.allclose(merged.actual, merged.actual_cp20, rtol=0, atol=1e-12)
                      and (merged.status == merged.status_cp20).all())
    rows, *_ = S.bootstrap(daily, contrasts=(('HG', 'H0'),))
    cp20u = pd.read_csv(root / 'reports/weather-ablation/uncertainty.csv')
    m = rows.merge(cp20u, on=['scope', 'candidate', 'baseline', 'metric'], suffixes=('', '_cp20'))
    diffs = {c: float(np.max(np.abs(m[c] - m[f'{c}_cp20']))) for c in ('difference', 'ci_lower', 'ci_upper')}
    return {'hg_section8_rows_equal_cp20': crit_equal, 'hg_section8_rows': int(len(ours)),
            'hg_h0_interval_rows': int(len(m)), 'hg_h0_max_abs_difference_vs_cp20': diffs,
            'hg_h0_equal_cp20': bool(len(m) == 12 and max(diffs.values()) <= 1e-12)}


def job_score(root: Path, rest) -> int:
    check_protocol(root)
    out = root / OUT
    lineage = json.loads((out / 'lineage.json').read_text())
    if lineage['execution_stage'] != 'comparison_vectors_complete_not_scored':
        raise ValueError('scoring requires the complete comparison vectors, scored once')
    committed = subprocess.check_output(['git', 'show', f'HEAD:{OUT}/predictions.parquet'], cwd=root)
    if committed != (out / 'predictions.parquet').read_bytes():
        raise ValueError('the comparison vectors must be committed before scoring')
    identities(root)
    budget = ledger()
    budget.reserve(reference_passes=1, analysis_passes=1)
    predictions, expected = load_predictions(root)
    tables = S.evaluate(predictions, expected, lineage=lineage)
    for name in ('metrics', 'diagnostics', 'uncertainty', 'criteria', 'fallback'):
        tables[name].to_csv(out / f'{name}.csv', index=False)
    tables['replicates'].to_parquet(out / 'replicates.parquet', index=False)
    tables['replicate_scores'].to_parquet(out / 'replicate-scores.parquet', index=False)
    check = consistency(root, S._tables(_hourly(predictions, expected))[2], tables['criteria'])
    summary = clean(tables['summary'])
    summary['consistency_with_cp20'] = clean(check)
    atomic(out / 'adoption.json', {**summary['adoption'], 'block_split': summary['block_split'],
                                   'inputs_sha256': {'reports/block-challenger/predictions.parquet': sha(out / 'predictions.parquet'),
                                                     'reports/weather-ablation/predictions.parquet': sha(root / 'reports/weather-ablation/predictions.parquet')}})
    lineage['research_summary'] = summary
    lineage['execution_stage'] = 'scored_development_post_selection'
    atomic(out / 'lineage.json', lineage)
    print(json.dumps({'verdict': summary['adoption']['verdict'], 'first_unmet': summary['adoption']['first_unmet_condition'],
                      'block_split': summary['block_split']['reading'], 'consistency': check,
                      'scores': {p: summary['scores'][p] for p in ('HG', 'HGL', 'L-P', 'L-R', 'L-N', 'B3')}}, default=str), flush=True)
    return 0


def _hourly(predictions, expected):
    frame = S.validate_predictions(predictions, expected)
    hourly = S.score_hourly(frame)
    hourly['local_hour'] = hourly.timestamp_utc.dt.tz_convert('Europe/Berlin').dt.hour
    hourly['local_block'] = S.BLOCK_OF[hourly.local_hour.to_numpy()]
    return hourly
