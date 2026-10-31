"""CP-23 research scores (capstone v21-r10 §21.5-§21.6): every policy on the 10,747 keys.

Built from CP-22's frozen scoring blocks, imported unchanged (`cp22.scoring`), which inherit CP-21's,
CP-20's and CP-16's:

* key and population validation;
* hourly losses (`cp15.scoring.score_hourly`) and the WIS weights;
* the per-fold, hour, block, peak and recovery diagnostics, with the 56-date support rule;
* equal-fold B0 normalisation and the original §8 criteria;
* CP-20's seed-15042, 2,000-replicate, 7-calendar-day paired block bootstrap. It is the same
  generator, hence the same shared index set, and its fingerprint is checked. With it come §17.5's
  ratio intervals, the stored replicates, the per-fold paired daily-loss intervals and the endpoint
  reading.

New here: the policy set, §21.5's contrast table and the mechanical `cp23-adoption` rule (§21.6).
There is one reference pass and one bootstrap pass, and no fitting, search or promotion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cp22.scoring import (BLOCK_DAYS, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CP20_INDEX_SHA256, FOLD_COUNTS, KEYS, PEAK,  # noqa: F401
                          QUANTILES, SUPPORT_DATES, _decide, _divide, _eq, _no_fold_worse, _section8_met, _values,
                          bootstrap, equal_fold_scores, hourly_scores, joint_reading, section8, tables, validate_predictions)

SAVED = ('B0', 'B1', 'B2', 'B3', 'A1', 'HG', 'HGL')
NEW = ('v5', 'v3+D', 'D')
ELIGIBLE = ('v5',)
DISPLAY = {'HG': 'v3', 'HGL': 'v4'}
#: The references §21.2 names; B0, B1 and B3 enter only as the B0 normaliser and §8's saved comparators.
REFERENCES = ('HGL', 'HG', 'A1', 'B2')
SECTION8_COMPARATORS = ('B0', 'B1', 'B2', 'B3')

RULES = {
    'cp23-adoption': {
        'set_on': '2026-10-04', 'source': 'capstone_v21.md v21-r10 §21.6 (Owner decisions D1, D2; D7 under the Owner\'s grant)',
        'candidate': 'v5', 'comparator': 'HGL (v4, three-block)',
        'conditions': {
            1: 'Joint improvement over v4: on the paired v5 - v4 differences, the upper 95% endpoint of dS_WIS is < 0 and the '
               'upper 95% endpoint of dS_MAE is <= 0.',
            2: 'No regression: v5 meets all six original section-8 diagnostics.',
            3: 'A complete, valid evaluation: Engineering PASS with a fresh binding Integration verdict, and all 10,747 keys '
               'issued with finite, ordered quantiles.',
            4: 'No resolved per-fold degradation: no fold has a 95% paired daily-loss interval (v5 - v4) lying entirely above '
               'zero, in MAE or in WIS.'},
        'otherwise': 'v5 is not adopted; CP-23 becomes the branch "DDNN member on v4", with the first unmet condition and its '
                     'values as the reason'},
}
APPLICATION = ('mechanical; never re-weighted, re-thresholded or overridden after results; D and v3+D are never eligible; a '
               'mixed result is no demonstrated joint preference, never equivalence; an INCOMPLETE or BLOCKED return yields '
               'no decision and nothing to publish')
ENGINEERING = ('bound to the fresh Integration verdict on the final candidate (docs/track-b/evidence/cp-23/integration.md); '
               'an INCOMPLETE or BLOCKED return yields no decision')


def contrasts() -> list[tuple[str, str, str]]:
    """§21.5's contrast table, (candidate, baseline, role), each read under §17.5."""
    return [('v5', 'HGL', 'adoption_decision'),
            ('v5', 'HG', 'reference_vs_v3'),
            ('D', 'HGL', 'ddnn_alone_vs_v4_descriptive'),
            ('D', 'HG', 'ddnn_alone_vs_v3_descriptive'),
            ('v3+D', 'HG', 'ddnn_as_v3_third_member'),
            ('HGL', 'HG', 'lightgbm_as_v3_third_member_beside'),
            ('v5', 'v3+D', 'lightgbm_still_adds_given_ddnn')]


def adoption(uncertainty, criteria, *, keys_complete: bool) -> dict:
    """`cp23-adoption`: v5 against v4 (HGL) on identical rows (§21.6), applied mechanically."""
    eq = _eq(uncertainty, 'v5', 'HGL')
    c1 = bool(eq.status.eq('resolved').all() and eq.loc['WIS', 'ci_upper'] < 0 and eq.loc['MAE', 'ci_upper'] <= 0)
    c2, v2 = _section8_met(criteria, 'v5')
    c4, v4 = _no_fold_worse(uncertainty, 'v5', 'HGL')
    d = _decide({1: {'met': c1, 'values': _values(eq)}, 2: {'met': c2, 'values': v2},
                 3: {'met': bool(keys_complete), 'values': {'keys_issued_finite_ordered': bool(keys_complete),
                                                            'engineering_pass': ENGINEERING}},
                 4: {'met': c4, 'values': v4}})
    adopted = d['met']
    return {'rule': RULES['cp23-adoption'], 'application': APPLICATION, 'candidate': 'v5', 'comparator': 'HGL', **d,
            'adopted': adopted,
            'verdict': ('v5 is adopted in research as v5 ("v5 · DDNN member added", predecessor v4)' if adopted else
                        'v5 is not adopted: CP-23 becomes the branch "DDNN member on v4"'),
            'reason': None if adopted else {'first_unmet_condition': d['first_unmet_condition'],
                                            'values': d['conditions'][d['first_unmet_condition']]['values']}}


def evaluate(predictions, expected_keys, policies, *, production=True, replicates=BOOTSTRAP_REPLICATES, indices=None):
    policies = list(policies)
    frame = validate_predictions(predictions, expected_keys, policies, production=production)
    hourly = hourly_scores(frame)
    metrics, diagnostics, daily = tables(hourly, policies)
    scores = equal_fold_scores(metrics, policies)
    candidates = [p for p in policies if p not in SAVED] + ['HG', 'HGL']
    criteria = section8(metrics, diagnostics, scores, candidates)
    pooled = metrics.loc[metrics.scope.eq('pooled')].set_index('policy')
    for metric in ('MAE', 'WIS'):
        metrics[f'pooled_ratio_{metric}'] = np.nan
        mask = metrics.scope.eq('pooled')
        metrics.loc[mask, f'pooled_ratio_{metric}'] = _divide(metrics.loc[mask, metric], pooled.loc['B0', metric])
    metrics = pd.concat([metrics, pd.DataFrame([dict(policy=p, fold='all', scope='equal_fold', **scores[p]) for p in policies])],
                        ignore_index=True)
    contrast_list = contrasts()
    uncertainty, replicate_draws, replicate_scores, meta = bootstrap(daily, policies, contrast_list, replicates=replicates,
                                                                     indices=indices)
    status = {}
    for policy in candidates:
        part = criteria.loc[criteria.policy.eq(policy)]
        status[policy] = 'unassessed' if part.status.eq('unassessed').any() else ('met' if part.passed.all() else 'not_met')
    decision = adoption(uncertainty, criteria, keys_complete=True)
    readings = {f'{c}-{b}': {**joint_reading(uncertainty, c, b), 'role': role} for c, b, role in contrast_list}
    summary = dict(
        policies=policies, scores=scores, original_section8_status=status,
        failed_criteria={p: sorted(criteria.loc[criteria.policy.eq(p) & ~criteria.passed, 'criterion'].unique().tolist())
                         for p in candidates},
        adoption=decision, contrasts=readings, bootstrap=meta,
        inference='exploratory post-selection; no equivalence, absence-of-benefit or absence-of-harm claim; mixed = no '
                  'demonstrated joint preference',
        primary_weighting='equal-fold B0-normalized eligible-hour losses', pooled_weighting='eligible hours',
        evidence_class='development_post_selection', product_status='v1 remains the released product and demo',
        engineering_status='not assessed by scoring; bound to the fresh Integration verdict',
        validation=dict(production=production, rows=len(frame), keys_per_policy=len(frame) // len(policies),
                        missing_predictions=0, nonfinite_quantiles=0, crossings=0),
        support_rule=dict(minimum_represented_dates=SUPPORT_DATES, role='reporting only; no product gate'),
        economics='zero new economic runs or thresholds')
    return dict(metrics=metrics, diagnostics=diagnostics, uncertainty=uncertainty, criteria=criteria, daily=daily,
                hourly=hourly, replicates=replicate_draws, replicate_scores=replicate_scores, summary=summary)
