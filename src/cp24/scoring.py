"""CP-24 research scores (capstone v21-r11 §23.8-§23.9): every policy of a scored attempt on the 10,747 keys.

Built from CP-22's frozen scoring blocks, imported unchanged (`cp22.scoring`), as CP-23's were: key and
population validation; hourly losses and WIS weights; per-fold, hour, block, peak and recovery
diagnostics with the 56-date support rule; equal-fold B0 normalisation and the original §8 criteria;
CP-20's seed-15042, 2,000-replicate, 7-calendar-day paired block bootstrap (the same generator, so the
same shared index set, its fingerprint checked), with ratio intervals, stored replicates and per-fold
paired daily-loss intervals.

New here: the policy set; §23.8's contrast table; the 97.5% intervals (the 1.25% and 98.75% percentiles
of the same stored replicates) beside the 95% ones; and the mechanical `cp24-adoption` rule (§23.9).

**D2 − L.** L = mean(L-N, L-R) is v4's LightGBM member, a central forecast without its own interval
layer. It enters the scoring only as a point-forecast reference, `L` (its seven "quantiles" set to its
central, so its WIS is not an interval score): the D2 − L contrast is read on MAE only, and its WIS row
is reported as not defined. L is never a scored policy, eligible or not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cp22.scoring import (BLOCK_DAYS, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CP20_INDEX_SHA256, FOLD_COUNTS, KEYS, PEAK,  # noqa: F401
                          QUANTILES, SUPPORT_DATES, _decide, _divide, _eq, _no_fold_worse, _section8_met, _values,
                          bootstrap, equal_fold_scores, hourly_scores, joint_reading, section8, tables, validate_predictions)

SAVED = ('B0', 'B1', 'B2', 'B3', 'A1', 'HG', 'HGL', 'D', 'v3+D')
NEW = ('v5', 'v3+D2', 'D2')
POINT_ONLY = ('L',)
ELIGIBLE = ('v5',)
DISPLAY = {'HG': 'v3', 'HGL': 'v4'}
REFERENCES = ('HGL', 'HG', 'A1', 'B2', 'D')
LEVEL_DECISION = 0.975
PRACTICAL = 0.005

RULES = {
    'cp24-adoption': {
        'set_on': '2026-10-05', 'source': 'capstone_v21.md v21-r11 §23.9 (D8, the Orchestrator under the Owner\'s delegation)',
        'candidate': 'v5', 'comparator': 'HGL (v4, three-block)',
        'conditions': {
            1: 'Joint improvement over v4, at the attempts-adjusted level: on the paired v5 - v4 differences, two-sided 97.5% '
               'intervals (the 1.25% and 98.75% percentiles of the shared replicates); the upper endpoint of dS_WIS is < 0 and '
               'the upper endpoint of dS_MAE is <= 0.',
            2: 'No regression: v5 meets all six original section-8 diagnostics.',
            3: 'A complete, valid evaluation: Engineering PASS with a fresh binding Integration verdict; all 10,747 keys issued '
               'with finite, ordered quantiles; every guard activation reported.',
            4: 'No resolved per-fold degradation: no fold has a 95% paired daily-loss interval (v5 - v4) lying entirely above '
               'zero, in MAE or in WIS.',
            5: 'A practical size: both point estimates improve v4\'s score by at least 0.5%: dS_MAE <= -0.005 x S_MAE(v4) and '
               'dS_WIS <= -0.005 x S_WIS(v4).'},
        'otherwise': 'attempt k\'s v5 is not adopted, with its first unmet condition and its values'},
}
APPLICATION = ('mechanical; never re-weighted, re-thresholded or overridden after results; D2, v3+D2 and D are never '
               'eligible; a mixed result is no demonstrated joint preference, never equivalence; an INCOMPLETE or BLOCKED '
               'return yields no decision and nothing to publish')
ENGINEERING = ('bound to the fresh Integration verdict on the final candidate (docs/track-b/evidence/cp-24/integration.md); '
               'an INCOMPLETE or BLOCKED return yields no decision')


def contrasts() -> list[tuple[str, str, str]]:
    """§23.8's contrast table, (candidate, baseline, role), each read under §17.5."""
    return [('v5', 'HGL', 'adoption_decision'),
            ('v5', 'HG', 'reference_vs_v3'),
            ('D2', 'L', 'ddnn2_alone_vs_same_information_twin_point_only'),
            ('D2', 'HG', 'ddnn2_alone_vs_lear'),
            ('D2', 'HGL', 'ddnn2_alone_vs_v4'),
            ('D2', 'D', 'ddnn2_vs_cp23_ddnn'),
            ('v3+D2', 'HGL', 'ddnn2_in_lightgbms_place'),
            ('v3+D', 'HGL', 'cp23_ddnn_in_lightgbms_place_beside'),
            ('v3+D2', 'HG', 'ddnn2_as_v3_third_member'),
            ('HGL', 'HG', 'lightgbm_as_v3_third_member_beside'),
            ('v5', 'v3+D2', 'lightgbm_still_adds_given_ddnn2')]


def with_level(uncertainty: pd.DataFrame, draws: pd.DataFrame, level: float = LEVEL_DECISION) -> pd.DataFrame:
    """Add two-sided `level` percentile intervals, from the same stored replicates, to every row."""
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = uncertainty.copy()
    tag = f'{level * 100:g}'
    lo_col, hi_col = f'ci{tag}_lower', f'ci{tag}_upper'
    out[lo_col], out[hi_col] = np.nan, np.nan
    keyed = draws.groupby(['scope', 'candidate', 'baseline', 'metric'])
    for i, row in out.iterrows():
        d = keyed.get_group((row['scope'], row['candidate'], row['baseline'], row['metric']))['difference'].to_numpy(float)
        if np.isfinite(d).all():
            out.at[i, lo_col], out.at[i, hi_col] = np.quantile(d, [lo_q, hi_q], method='linear')
    return out


def adoption(uncertainty, criteria, scores, *, keys_complete: bool, guards_reported: bool) -> dict:
    """`cp24-adoption`: v5 against v4 (HGL) on identical rows (§23.9), applied mechanically."""
    eq = _eq(uncertainty, 'v5', 'HGL')
    resolved = bool(eq.status.eq('resolved').all())
    c1 = bool(resolved and eq.loc['WIS', 'ci97.5_upper'] < 0 and eq.loc['MAE', 'ci97.5_upper'] <= 0)
    v1 = {m: {**_values(eq)[m], 'ci97.5_lower': float(eq.loc[m, 'ci97.5_lower']), 'ci97.5_upper': float(eq.loc[m, 'ci97.5_upper'])}
          for m in ('MAE', 'WIS')}
    c2, v2 = _section8_met(criteria, 'v5')
    c4, v4 = _no_fold_worse(uncertainty, 'v5', 'HGL')
    s4 = scores['HGL']
    d_mae, d_wis = float(eq.loc['MAE', 'difference']), float(eq.loc['WIS', 'difference'])
    c5 = bool(d_mae <= -PRACTICAL * s4['S_MAE'] and d_wis <= -PRACTICAL * s4['S_WIS'])
    d = _decide({1: {'met': c1, 'values': v1}, 2: {'met': c2, 'values': v2},
                 3: {'met': bool(keys_complete and guards_reported),
                     'values': {'keys_issued_finite_ordered': bool(keys_complete), 'guard_activations_reported': bool(guards_reported),
                                'engineering_pass': ENGINEERING}},
                 4: {'met': c4, 'values': v4},
                 5: {'met': c5, 'values': {'dS_MAE': d_mae, 'threshold_MAE': -PRACTICAL * s4['S_MAE'], 'dS_WIS': d_wis,
                                           'threshold_WIS': -PRACTICAL * s4['S_WIS'], 'S_MAE_v4': s4['S_MAE'],
                                           'S_WIS_v4': s4['S_WIS'], 'relative_MAE': d_mae / s4['S_MAE'],
                                           'relative_WIS': d_wis / s4['S_WIS']}}})
    adopted = d['met']
    not_adopted_for_s2 = any(not d['conditions'][c]['met'] for c in (1, 2, 4, 5)) or not (keys_complete and guards_reported)
    return {'rule': RULES['cp24-adoption'], 'application': APPLICATION, 'candidate': 'v5', 'comparator': 'HGL', **d,
            'adopted': adopted, 'counts_as_not_adopted_for_s2': bool(not_adopted_for_s2),
            'verdict': ('v5 is adopted in research as v5 ("v5 · DDNN-2 member added", predecessor v4)' if adopted else
                        'v5 is not adopted: the branch "DDNN-2 member on v4"'),
            'reason': None if adopted else {'first_unmet_condition': d['first_unmet_condition'],
                                            'values': d['conditions'][d['first_unmet_condition']]['values']}}


def evaluate(predictions, expected_keys, policies, *, guards_reported: bool, production=True,
             replicates=BOOTSTRAP_REPLICATES, indices=None):
    policies = list(policies)
    frame = validate_predictions(predictions, expected_keys, policies, production=production)
    hourly = hourly_scores(frame)
    metrics, diagnostics, daily = tables(hourly, policies)
    scores = equal_fold_scores(metrics, policies)
    candidates = [p for p in policies if p in NEW] + ['HG', 'HGL']
    criteria = section8(metrics, diagnostics, scores, candidates)
    pooled = metrics.loc[metrics.scope.eq('pooled')].set_index('policy')
    for metric in ('MAE', 'WIS'):
        metrics[f'pooled_ratio_{metric}'] = np.nan
        mask = metrics.scope.eq('pooled')
        metrics.loc[mask, f'pooled_ratio_{metric}'] = _divide(metrics.loc[mask, metric], pooled.loc['B0', metric])
    metrics = pd.concat([metrics, pd.DataFrame([dict(policy=p, fold='all', scope='equal_fold', **scores[p]) for p in policies])],
                        ignore_index=True)
    contrast_list = contrasts()
    uncertainty, draws, replicate_scores, meta = bootstrap(daily, policies, contrast_list, replicates=replicates, indices=indices)
    uncertainty = with_level(uncertainty, draws)
    point_only = uncertainty.candidate.isin(POINT_ONLY) | uncertainty.baseline.isin(POINT_ONLY)
    uncertainty['defined'] = ~(point_only & uncertainty.metric.eq('WIS'))
    status = {}
    for policy in candidates:
        part = criteria.loc[criteria.policy.eq(policy)]
        status[policy] = 'unassessed' if part.status.eq('unassessed').any() else ('met' if part.passed.all() else 'not_met')
    decision = adoption(uncertainty, criteria, scores, keys_complete=True, guards_reported=guards_reported)
    readings = {}
    for c, b, role in contrast_list:
        reading = {**joint_reading(uncertainty, c, b), 'role': role}
        part = _eq(uncertainty, c, b)
        for m in ('MAE', 'WIS'):
            reading[f'dS_{m}']['ci97.5'] = [float(part.loc[m, 'ci97.5_lower']), float(part.loc[m, 'ci97.5_upper'])]
        if c in POINT_ONLY or b in POINT_ONLY:
            reading['reading'] = ('MAE only (L has no interval forecast): ' +
                                  ('lower (better)' if part.loc['MAE', 'ci_upper'] < 0 else
                                   'higher (worse)' if part.loc['MAE', 'ci_lower'] > 0 else 'no resolved difference'))
            reading['dS_WIS'] = 'not defined: L is a central forecast without an interval layer'
        readings[f'{c}-{b}'] = reading
    summary = dict(
        policies=policies, scores=scores, original_section8_status=status,
        failed_criteria={p: sorted(criteria.loc[criteria.policy.eq(p) & ~criteria.passed, 'criterion'].unique().tolist())
                         for p in candidates},
        adoption=decision, contrasts=readings, bootstrap={**meta, 'levels': [0.95, LEVEL_DECISION],
                                                          'decision_level': LEVEL_DECISION},
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
                hourly=hourly, replicates=draws, replicate_scores=replicate_scores, summary=summary)
