"""The CP-22 scoring jobs (§20.5-§20.6), each one metric-only reference pass and one bootstrap pass.

* ``score-replacement`` (pass 1): the eight saved references (CP-20's seven, CP-21's v4) and the
  seven fixed new policies; applies `cp22-replacement` mechanically and writes
  `replacement.json`. Consistency rides in the same pass, on the same index set: HG's and v4's
  recomputed §8 rows must equal CP-20's and CP-21's committed criteria, and HG - H0 and v4 - HG
  must equal their committed interval rows.
* ``score`` (pass 2, only if a winner W exists): every policy including W's three layer arms;
  applies `cp22-dynamic-layer`, then `cp22-fast-component`; every pass-1 row must be reproduced.
  With no winner, no further pass is made: the pass-1 tables are the final tables.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

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


def load_predictions(root: Path, w: bool):
    cols = ['fold', 'policy', 'timestamp_utc', 'delivery_date', 'y_true', 'central', 'scale', 'level', *S.QUANTILES]
    saved = pd.read_parquet(root / 'reports/weather-ablation/predictions.parquet')
    v4 = pd.read_parquet(root / 'reports/block-challenger/predictions.parquet', filters=[('policy', '==', 'HGL')])
    parts = [saved[cols], v4[cols], pd.read_parquet(root / OUT / 'predictions.parquet')[cols]]
    if w:
        parts.append(pd.read_parquet(root / OUT / 'predictions-w.parquet')[cols])
    both = pd.concat(parts, ignore_index=True)
    both['timestamp_utc'] = pd.to_datetime(both.timestamp_utc, utc=True).dt.as_unit('ns')
    expected = pd.read_parquet(root / 'reports/cp15/predictions.parquet', columns=['fold', 'timestamp_utc', 'delivery_date'],
                               filters=[('policy', '==', 'B0')])
    return both, expected


def consistency(root: Path, daily, criteria, policies) -> dict:
    """HG's and v4's §8 rows and the HG - H0 / HGL - HG intervals against their committed rows."""
    out = {}
    for policy, path in (('HG', 'reports/weather-ablation/criteria.csv'), ('HGL', 'reports/block-challenger/criteria.csv')):
        theirs = pd.read_csv(root / path)
        theirs = theirs.loc[theirs.policy.eq(policy)].reset_index(drop=True)
        ours = criteria.loc[criteria.policy.eq(policy)].reset_index(drop=True)
        merged = ours.merge(theirs, on=['criterion', 'metric', 'scope'], suffixes=('', '_committed'))
        out[f'{policy}_section8_rows_equal_committed'] = bool(
            len(merged) == len(theirs) == len(ours) and np.allclose(merged.actual, merged.actual_committed, rtol=0, atol=1e-12)
            and (merged.status == merged.status_committed).all())
    rows, *_ = S.bootstrap(daily, policies, [('HG', 'H0', 'consistency'), ('HGL', 'HG', 'consistency')])
    for (cand, base), path in ((('HG', 'H0'), 'reports/weather-ablation/uncertainty.csv'),
                               (('HGL', 'HG'), 'reports/block-challenger/uncertainty.csv')):
        theirs = pd.read_csv(root / path)
        theirs = theirs.loc[theirs.candidate.eq(cand) & theirs.baseline.eq(base)]
        m = rows.loc[rows.candidate.eq(cand) & rows.baseline.eq(base)].merge(
            theirs, on=['scope', 'candidate', 'baseline', 'metric'], suffixes=('', '_committed'))
        diff = max(float(np.max(np.abs(m[c] - m[f'{c}_committed']))) for c in ('difference', 'ci_lower', 'ci_upper'))
        out[f'{cand}-{base}_interval_rows'] = int(len(m))
        out[f'{cand}-{base}_max_abs_difference_vs_committed'] = diff
        out[f'{cand}-{base}_equal_committed'] = bool(len(m) == 12 and diff <= 1e-12)
    out['all_equal'] = all(v for k, v in out.items() if k.endswith('equal_committed'))
    return out


def _write(out: Path, tables: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        tables[name].to_csv(out / f'{name}.csv', index=False)
    tables['replicates'].to_parquet(out / 'replicates.parquet', index=False)
    tables['replicate_scores'].to_parquet(out / 'replicate-scores.parquet', index=False)


def job_score_replacement(root: Path, rest) -> int:
    check_protocol(root)
    out = root / OUT
    lineage = json.loads((out / 'lineage.json').read_text())
    if lineage['execution_stage'] != 'comparison_vectors_complete_not_scored':
        raise ValueError('pass 1 requires the complete comparison vectors, scored once')
    for name in ('predictions.parquet', 'members.parquet', 'lineage.json'):
        committed_at_head(root, str(OUT / name))
    identities(root)
    budget = ledger()
    budget.reserve(reference_passes=1, analysis_passes=1)
    predictions, expected = load_predictions(root, w=False)
    policies = list(S.SAVED + S.FIXED_NEW)
    tables = S.evaluate(predictions, expected, policies, None)
    _write(out / 'pass1', tables)
    check = consistency(root, tables['daily'], tables['criteria'], policies)
    summary = clean(tables['summary'])
    summary['consistency_with_committed'] = clean(check)
    atomic(out / 'replacement.json', {**summary['replacement'], 'consistency_with_committed': summary['consistency_with_committed'],
                                      'inputs_sha256': {str(OUT / 'predictions.parquet'): sha(out / 'predictions.parquet'),
                                                        'reports/weather-ablation/predictions.parquet': sha(root / 'reports/weather-ablation/predictions.parquet'),
                                                        'reports/block-challenger/predictions.parquet': sha(root / 'reports/block-challenger/predictions.parquet')}})
    atomic(out / 'pass1' / 'summary.json', summary)
    lineage['execution_stage'] = 'scored_pass1_replacement_decided'
    atomic(out / 'lineage.json', lineage)
    print(json.dumps({'winner': summary['replacement']['winner'], 'first_unmet': summary['replacement']['first_unmet_condition'],
                      'consistency': check, 'scores': {p: summary['scores'][p] for p in ('HG', 'HGL', *S.FIXED_NEW)}},
                     default=str), flush=True)
    return 0 if check['all_equal'] else 9


def job_score(root: Path, rest) -> int:
    """Final tables: pass 2 when W exists; otherwise the pass-1 tables, with no further pass."""
    check_protocol(root)
    out = root / OUT
    committed_at_head(root, str(OUT / 'replacement.json'))
    rep = json.loads((out / 'replacement.json').read_text())
    w = rep['winner']
    pass1 = json.loads((out / 'pass1' / 'summary.json').read_text())
    if w is None:
        for name in TABLES:
            shutil.copyfile(out / 'pass1' / f'{name}.csv', out / f'{name}.csv')
        for name in ('replicates.parquet', 'replicate-scores.parquet'):
            shutil.copyfile(out / 'pass1' / name, out / name)
        summary = pass1
        summary['pass'] = 'pass 1 (no winner: no layer arms exist and no further pass is made)'
    else:
        lw = json.loads((out / 'lineage-w.json').read_text())
        if lw['execution_stage'] != 'comparison_vectors_complete_not_scored' or lw['w_policy'] != w:
            raise ValueError('pass 2 requires the complete W layer vectors for the winner')
        for name in ('predictions-w.parquet', 'lineage-w.json'):
            committed_at_head(root, str(OUT / name))
        identities(root)
        ledger().reserve(reference_passes=1, analysis_passes=1)
        predictions, expected = load_predictions(root, w=True)
        policies = list(S.SAVED + S.FIXED_NEW + S.W_ARMS)
        tables = S.evaluate(predictions, expected, policies, w)
        _write(out, tables)
        summary = clean(tables['summary'])
        # every pass-1 row is reproduced on the same index set
        p1 = pd.read_csv(out / 'pass1' / 'uncertainty.csv')
        p2 = tables['uncertainty']
        m = p1.merge(p2, on=['scope', 'candidate', 'baseline', 'metric'], suffixes=('_1', '_2'))
        gap = max(float(np.nanmax(np.abs(m[f'{c}_1'] - m[f'{c}_2']))) for c in ('difference', 'ci_lower', 'ci_upper'))
        summary['pass1_reproduced'] = {'rows': int(len(m)), 'pass1_rows': int(len(p1)), 'max_abs_difference': gap,
                                       'equal': bool(len(m) == len(p1) and gap <= 1e-12),
                                       'replacement_identical': summary['replacement']['winner'] == w}
        summary['pass'] = 'pass 2 (all policies, W layer arms included)'
        lw['execution_stage'] = 'scored_pass2'
        atomic(out / 'lineage-w.json', lw)
        if not summary['pass1_reproduced']['equal'] or not summary['pass1_reproduced']['replacement_identical']:
            raise ValueError('pass 2 does not reproduce pass 1')
    summary['consistency_with_committed'] = pass1.get('consistency_with_committed')
    decisions = {'replacement': summary['replacement'], 'dynamic_layer': summary['dynamic_layer'],
                 'fast_component': summary['fast_component'], 'contrasts': summary['contrasts'], 'scores': summary['scores'],
                 'original_section8_status': summary['original_section8_status'], 'bootstrap': summary['bootstrap'],
                 'pass': summary['pass'], 'pass1_reproduced': summary.get('pass1_reproduced'),
                 'consistency_with_committed': summary['consistency_with_committed'],
                 'evidence_class': 'development_post_selection',
                 'statuses': {'engineering': 'bound to the fresh Integration verdict on the final candidate',
                              'research': 'development_post_selection finding under the three pre-registered rules',
                              'product': 'v1 remains the released product and demo; no designation, freeze or Live'}}
    atomic(out / 'decisions.json', decisions)
    atomic(out / 'summary.json', summary)
    lineage = json.loads((out / 'lineage.json').read_text())
    lineage['execution_stage'] = 'scored_final'
    atomic(out / 'lineage.json', lineage)
    print(json.dumps({k: decisions[k]['verdict'] if isinstance(decisions[k], dict) and 'verdict' in decisions[k] else None
                      for k in ('replacement', 'dynamic_layer', 'fast_component')}), flush=True)
    return 0
