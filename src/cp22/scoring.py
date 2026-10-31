"""CP-22 research scores (capstone v21-r9 §20.5-§20.6): every policy on the 10,747 keys.

Adapted from `src/cp21/scoring.py` (preserved unchanged), which adapted CP-20's and CP-16's
frozen scoring: identical hourly losses (`cp15.scoring.score_hourly`), WIS weights, equal-fold B0
normalisation, per-fold/hour/block/peak/recovery diagnostics, the 56-date support rule, the
original §8 criteria, and CP-20's seed-15042, 2,000-replicate, 7-calendar-day paired block
bootstrap -- the same generator, hence the same shared index set (its fingerprint is checked) --
with §17.5's ratio intervals and stored replicates. Changes are confined to the policy set, the
§20.5 contrast set and the three mechanical §20.6 rules. No fitting, search or promotion.

Two passes, in a fixed sequence (§20.6): pass 1 scores the saved references and the seven fixed
new policies and applies `cp22-replacement`; only if it yields a winner W are W's three layer arms
replayed, and pass 2 scores every policy and applies `cp22-dynamic-layer`, then
`cp22-fast-component`. Pass 2 reproduces every pass-1 row on the same index set.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from cp15.scoring import FOLDS, FOLD_WINDOWS, QUANTILES, _summary, score_hourly
from cp20.scoring import _indices

SAVED = ('B0', 'B1', 'B2', 'B3', 'A1', 'H0', 'HG', 'HGL')
FIXED_NEW = ('R', 'M', 'A-PN-sel', 'A-LP', 'A-LN', 'v4+DL', 'v3+DL')
W_ARMS = ('W+ACI', 'W+DL', 'W+DLF')
DISPLAY = {'HG': 'v3', 'HGL': 'v4'}
BASELINES = ('B0', 'B1', 'B2', 'B3')
KEYS = ['fold', 'timestamp_utc']
FOLD_COUNTS = dict(zip(FOLDS, (2160, 2159, 2112, 2160, 2156)))
BOOTSTRAP_SEED = 15042
BOOTSTRAP_REPLICATES = 2000
BLOCK_DAYS = 7
SUPPORT_DATES = 56
CP20_INDEX_SHA256 = 'e1df9a68dc6715aa2ecd9705ef61f504a3fe109ed917d1151ccbc46ea9e0f99b'
BLOCK_OF = np.array(['night'] * 6 + ['shoulder'] * 4 + ['solar'] * 7 + ['shoulder'] * 5 + ['night'] * 2)
SHOCK_SHARE = 0.05
SHOCK_AFTER_DAYS = 3
PEAK = ('2022-08-15', '2022-08-31')
PEAK_FIRST_DAYS = 10

RULES = {
    'cp22-replacement': {
        'set_on': '2026-10-01', 'source': 'capstone_v21.md v21-r9 §20.6 (Owner decisions D2, 2026-10-01)',
        'candidates_in_sequence': ['R', 'M'], 'comparator': 'HGL (v4)',
        'conditions': {
            1: 'Non-inferior on both scores: neither dS_MAE nor dS_WIS (candidate - v4) has a 95% interval lying entirely above zero.',
            2: 'No regression: the candidate meets all six original section-8 diagnostics.',
            3: 'A complete, valid evaluation: Engineering PASS with a binding Integration verdict, and all 10,747 keys issued.',
            4: 'No resolved per-fold degradation: no fold has a 95% paired daily-loss interval (candidate - v4) lying entirely '
               'above zero, in MAE or in WIS.'},
        'sequence': 'R replaces v4 if all four hold; otherwise M if all four hold; otherwise no replacement and CP-22 stops '
                    'for the Owner. The winner is W.'},
    'cp22-dynamic-layer': {
        'set_on': '2026-10-01', 'source': 'capstone_v21.md v21-r9 §20.6 (Owner decision D7)', 'applies': 'only if W exists',
        'candidate': 'W+DL', 'comparator': 'W',
        'conditions': {
            1: 'The upper 95% endpoint of dS_WIS is below zero.',
            2: 'dS_MAE has no 95% interval lying entirely above zero.',
            3: 'W+DL meets all six section-8 diagnostics.',
            4: "W+DL's pooled 95% coverage is closer to 0.95 than W's.",
            5: 'No fold has a 95% paired daily-loss interval (W+DL - W) lying entirely above zero, in MAE or in WIS.'},
        'otherwise': 'the replacement is W with HG\'s H layer'},
    'cp22-fast-component': {
        'set_on': '2026-10-01', 'source': 'capstone_v21.md v21-r9 §20.6 (Owner, 2026-10-01)', 'applies': 'only if W+DL was adopted',
        'candidate': 'W+DLF', 'comparator': 'W+DL',
        'conditions': {
            1: 'The upper 95% endpoint of dS_WIS is below zero.',
            2: 'dS_MAE has no 95% interval lying entirely above zero.',
            3: 'W+DLF meets all six section-8 diagnostics.',
            4: 'No fold has a 95% paired daily-loss interval (W+DLF - W+DL) lying entirely above zero, in MAE or in WIS.'},
        'otherwise': 'W+DL stands; if W+DL was not adopted, (W+DLF) - (W+DL) is descriptive only'},
}
APPLICATION = ('mechanical; never re-weighted, re-thresholded or overridden after results; attribution arms are never '
               'eligible; a mixed result is no demonstrated joint preference, never equivalence; an INCOMPLETE or BLOCKED '
               'return yields no decision and nothing to publish')


def contrasts(w: str | None) -> list[tuple[str, str, str]]:
    """§20.5's contrast table, (candidate, baseline, role), for the policies that exist."""
    out = [('R', 'HGL', 'replacement'), ('M', 'HGL', 'replacement')]
    if w:
        out.append(('W+DL', w, 'layer_decision'))
    out += [('R', 'M', 'full_vs_minimal_refinement'), ('A-PN-sel', 'M', 'dropping_the_raw_half'),
            ('R', 'A-PN-sel', 'averaging_vs_daily_selection'), ('A-PN-sel', 'A-LP', 'normalized_vs_raw_pooled'),
            ('A-LN', 'HGL', 'descriptive_blocks_under_normalization')]
    if w:
        out += [('W+ACI', w, 'dl_part_adaptive_coverage'), ('W+DL', 'W+ACI', 'dl_part_recency_weights'),
                ('W+DLF', 'W+DL', 'fast_component')]
    out += [('v4+DL', 'HGL', 'dl_on_v4_descriptive'), ('v3+DL', 'HG', 'dl_on_v3_descriptive')]
    for p in FIXED_NEW + ((W_ARMS if w else ()) + ('HGL',)):
        out.append((p, 'HG', 'reference_vs_v3'))
    seen, unique = set(), []
    for c in out:
        if c[:2] not in seen:
            seen.add(c[:2])
            unique.append(c)
    return unique


def validate_predictions(predictions: pd.DataFrame, expected_keys: pd.DataFrame, policies, *, production: bool = True):
    """Refuse incomplete, mismatched, nonfinite, crossed or wrongly-scaled populations."""
    required = {*KEYS, 'policy', 'delivery_date', 'y_true', 'central', 'scale', 'level', *QUANTILES}
    if not required.issubset(predictions):
        raise ValueError(f'missing prediction columns: {sorted(required - set(predictions))}')
    frame = predictions.copy(deep=True)
    if frame[list(required)].isna().any().any():
        raise ValueError('missing predictions or keys')
    if set(frame.policy) != set(policies) or set(frame.fold) != set(FOLDS):
        raise ValueError('exactly the declared policies and five folds required')
    for data in (frame, expected_keys):
        if not set(KEYS).issubset(data) or data[KEYS].isna().any().any():
            raise ValueError('independent expected_keys and prediction keys are required')
    frame['timestamp_utc'] = pd.to_datetime(frame.timestamp_utc, errors='raise')
    expected = expected_keys.copy(deep=True)
    expected['timestamp_utc'] = pd.to_datetime(expected.timestamp_utc, errors='raise')
    if frame.timestamp_utc.dt.tz is None or expected.timestamp_utc.dt.tz is None:
        raise ValueError('timestamp_utc must be timezone-aware')
    frame['timestamp_utc'] = frame.timestamp_utc.dt.tz_convert('UTC').dt.as_unit('ns')
    expected['timestamp_utc'] = expected.timestamp_utc.dt.tz_convert('UTC').dt.as_unit('ns')
    if not frame.timestamp_utc.eq(frame.timestamp_utc.dt.floor('h')).all():
        raise ValueError('timestamps must be whole hours')
    dates = pd.to_datetime(frame.delivery_date, errors='raise')
    local = frame.timestamp_utc.dt.tz_convert('Europe/Berlin').dt.tz_localize(None)
    if dates.dt.tz is not None or not dates.eq(local.dt.normalize()).all():
        raise ValueError('delivery_date disagrees with the local calendar date')
    frame['delivery_date'] = dates
    if dates.max() > pd.Timestamp('2026-04-07'):
        raise ValueError('a scored row is dated after 2026-04-07')
    for fold, (start, end) in FOLD_WINDOWS.items():
        if not frame.loc[frame.fold.eq(fold), 'delivery_date'].between(start, end).all():
            raise ValueError(f'date outside {fold} window')
    if frame.duplicated(['policy', 'timestamp_utc']).any() or expected.duplicated(KEYS).any():
        raise ValueError('duplicate target keys')
    numeric = ['y_true', 'central', 'scale', 'level', *QUANTILES]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='raise').astype(float)
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError('nonfinite predictions or truth')
    if frame.scale.le(0).any() or (np.diff(frame[list(QUANTILES)], axis=1) < 0).any():
        raise ValueError('nonpositive scale or crossed quantiles')
    expected = expected.set_index(KEYS).sort_index()
    reference = frame.loc[frame.policy.eq('B0')].set_index(KEYS).sort_index()
    if not expected.index.equals(reference.index):
        raise ValueError('predictions differ from independent expected keys')
    if 'delivery_date' in expected and not np.array_equal(pd.to_datetime(expected.delivery_date).to_numpy(),
                                                          reference.delivery_date.to_numpy()):
        raise ValueError('expected delivery_date disagrees')
    for policy in policies:
        part = frame.loc[frame.policy.eq(policy)].set_index(KEYS).sort_index()
        if not part.index.equals(reference.index):
            raise ValueError(f'unmatched keys: {policy}')
        if not part[['y_true', 'delivery_date']].equals(reference[['y_true', 'delivery_date']]):
            raise ValueError(f'unmatched truth or dates: {policy}')
    hg = frame.loc[frame.policy.eq('HG')].set_index(KEYS).sort_index()
    for policy in [p for p in policies if p not in SAVED or p == 'HGL']:
        part = frame.loc[frame.policy.eq(policy)].set_index(KEYS).sort_index()
        if not part[['scale', 'level', 'y_true']].equals(hg[['scale', 'level', 'y_true']]):
            raise ValueError(f'{policy}/HG price-only A1 scale, level and truth parity failed')
    if production:
        counts = reference.reset_index().groupby('fold').size().to_dict()
        peak = reference.loc[reference.delivery_date.between(*PEAK)]
        f3 = reference.xs('fold_3')
        if counts != FOLD_COUNTS or len(reference) != 10747:
            raise ValueError('production requires the original 10,747 keys and exact fold counts')
        if len(peak) != 408 or peak.delivery_date.nunique() != 17 or f3.delivery_date.nunique() != 88:
            raise ValueError('fold-3/peak dates and denominators disagree')
    return frame.sort_values(['policy', *KEYS]).reset_index(drop=True)


def _record(rows, **labels):
    result = _summary(rows)
    result['represented_dates'] = json.dumps(sorted(rows.delivery_date.dt.strftime('%Y-%m-%d').unique().tolist()))
    return labels | result


def hourly_scores(frame: pd.DataFrame) -> pd.DataFrame:
    hourly = score_hourly(frame)
    hourly['local_hour'] = hourly.timestamp_utc.dt.tz_convert('Europe/Berlin').dt.hour
    hourly['local_block'] = BLOCK_OF[hourly.local_hour.to_numpy()]
    return hourly


def tables(hourly, policies):
    metrics, diagnostics = [], []
    for policy in policies:
        selected = hourly.loc[hourly.policy.eq(policy)]
        metrics.append(_record(selected, policy=policy, scope='pooled', fold='all'))
        for fold in FOLDS:
            part = selected.loc[selected.fold.eq(fold)]
            start, end = FOLD_WINDOWS[fold]
            metrics.append(_record(part, policy=policy, scope='per_fold', fold=fold, window_start=start,
                                   window_end=end, calendar_days=90, stress_period=fold == 'fold_3'))
            for scope, groups in (('hour', [(str(h), part.loc[part.local_hour.eq(h)]) for h in range(24)]),
                                  ('block', [(b, part.loc[part.local_block.eq(b)]) for b in ('night', 'solar', 'shoulder')])):
                for label, group in groups:
                    enough = group.delivery_date.nunique() >= SUPPORT_DATES
                    diagnostics.append(_record(group, policy=policy, fold=fold, scope=scope, group=label,
                                               support_status='eligible' if enough else 'support_limited',
                                               uncertainty_claim='none_descriptive_diagnostics', minimum_dates=SUPPORT_DATES))
        for label in ('night', 'solar', 'shoulder'):
            diagnostics.append(_record(selected.loc[selected.local_block.eq(label)], policy=policy, fold='all',
                                       scope='block_pooled', group=label, uncertainty_claim='none_descriptive_diagnostics'))
        peak = selected.loc[selected.fold.eq('fold_3') & selected.delivery_date.between(*PEAK)]
        diagnostics.append(_record(peak, policy=policy, fold='fold_3', scope='peak', group='August15-31',
                                   window_start=PEAK[0], window_end=PEAK[1], calendar_days=17,
                                   support_status='support_limited', evidence_class='descriptive_only_small_effective_sample'))
        stress = selected.loc[selected.fold.eq('fold_3')]
        diagnostics.append(_record(stress, policy=policy, fold='fold_3', scope='stress', group='2022-07-01..09-28',
                                   window_start='2022-07-01', window_end='2022-09-28', calendar_days=90))
        for begin in (1, 8, 15, 22):
            start, end = f'2022-09-{begin:02}', f'2022-09-{begin + 6:02}'
            part = selected.loc[selected.fold.eq('fold_3') & selected.delivery_date.between(start, end)]
            diagnostics.append(_record(part, policy=policy, fold='fold_3', scope='recovery', group=start,
                                       window_start=start, window_end=end, calendar_days=7, evidence_class='descriptive_only'))
    daily = hourly.groupby(['policy', 'fold', 'delivery_date']).agg(
        n_hours=('WIS', 'size'), MAE=('absolute_error', 'mean'), WIS=('WIS', 'mean'),
        coverage95=('hit95', 'mean'), hit_count95=('hit95', 'sum'),
        lower_miss_count95=('lower_miss95', 'sum'), upper_miss_count95=('upper_miss95', 'sum'),
        mean_width95=('width95', 'mean'))
    parts = []
    for fold, (start, end) in FOLD_WINDOWS.items():
        index = pd.MultiIndex.from_product([list(policies), [fold], pd.date_range(start, end)],
                                          names=['policy', 'fold', 'delivery_date'])
        part = daily.reindex(index).reset_index()
        part['n_hours'] = part.n_hours.fillna(0).astype(int)
        parts.append(part)
    daily = pd.concat(parts, ignore_index=True)
    daily['scope'] = 'daily'
    return pd.DataFrame(metrics), pd.concat([pd.DataFrame(diagnostics), daily], ignore_index=True), daily


def _divide(numerator, denominator):
    """Undefined denominators remain NaN; never an epsilon or a zero outcome."""
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=np.isfinite(b) & (b > 0))


def equal_fold_scores(metrics, policies):
    per_fold = metrics.loc[metrics.scope.eq('per_fold')].set_index(['policy', 'fold'])
    return {policy: {f'S_{metric}': float(np.mean(_divide(per_fold.loc[policy, metric].reindex(FOLDS),
                                                          per_fold.loc['B0', metric].reindex(FOLDS))))
                     for metric in ('MAE', 'WIS')} for policy in policies}


def section8(metrics, diagnostics, scores, candidates, *, integrity_verified=True):
    """All six original §8 diagnostics, unchanged, with the saved B0-B3 comparators."""
    per_fold = metrics.loc[metrics.scope.eq('per_fold')].set_index(['policy', 'fold'])
    peak = diagnostics.loc[diagnostics.scope.eq('peak')].set_index('policy')
    criteria = []

    def add(policy, criterion, metric, scope, actual, lower=None, upper=None, comparator=None):
        limits = [x for x in (lower, upper) if x is not None]
        assessed = bool(np.isfinite(actual) and np.isfinite(limits).all())
        passed = assessed and (lower is None or actual >= lower) and (upper is None or actual <= upper)
        criteria.append(dict(policy=policy, criterion=criterion, metric=metric, scope=scope, actual=actual,
                             lower_limit=lower, upper_limit=upper, comparator=comparator,
                             status=('met' if passed else 'not_met') if assessed else 'unassessed', passed=bool(passed)))

    for policy in candidates:
        for number, metric in ((1, 'MAE'), (2, 'WIS')):
            add(policy, number, f'S_{metric}', 'equal_fold', scores[policy][f'S_{metric}'],
                upper=0.9 * np.min([scores[b][f'S_{metric}'] for b in BASELINES]), comparator='best B0-B3')
        for fold in FOLDS:
            add(policy, 3, 'coverage95', fold, per_fold.loc[(policy, fold), 'coverage95'], lower=.90, upper=.98)
        add(policy, 4, 'coverage95', 'peak', peak.loc[policy, 'coverage95'], lower=.90)
        for metric in ('MAE', 'WIS'):
            add(policy, 4, metric, 'peak', peak.loc[policy, metric],
                upper=np.min([peak.loc[b, metric] for b in BASELINES]), comparator='best B0-B3 on matched peak')
            for fold in FOLDS:
                add(policy, 5, metric, fold, per_fold.loc[(policy, fold), metric],
                    upper=1.05 * min(per_fold.loc[(b, fold), metric] for b in ('B2', 'B3')), comparator='best rolling B2/B3')
        add(policy, 6, 'complete_finite_ordered', 'all_eligible', int(integrity_verified), lower=1, upper=1)
    return pd.DataFrame(criteria)


def bootstrap(daily, policies, contrast_list, *, replicates=BOOTSTRAP_REPLICATES, indices=None):
    """One shared index set per analysis pass (CP-20's generator, seed 15042); see cp21.scoring."""
    indices = _indices(replicates) if indices is None else indices
    if set(indices) != set(FOLDS):
        raise ValueError('indices require exactly all five folds')
    policies = list(policies)
    samples, points, rows, stored = [], [], [], []

    def interval(draws, point):
        unresolved = int((~np.isfinite(draws)).sum())
        valid = unresolved == 0 and np.isfinite(point)
        lo, hi = np.quantile(draws, [.025, .975], method='linear') if valid else (np.nan, np.nan)
        return float(lo), float(hi), ('resolved' if valid else 'unresolved_uncertainty'), unresolved

    for fold in FOLDS:
        part = daily.loc[daily.fold.eq(fold)]
        dates = pd.date_range(*FOLD_WINDOWS[fold])
        expected = pd.MultiIndex.from_product([policies, dates], names=['policy', 'delivery_date'])
        if part.duplicated(['policy', 'delivery_date']).any() or set(part.set_index(['policy', 'delivery_date']).index) != set(expected):
            raise ValueError('bootstrap requires the full matched 90-calendar-date grid')
        values = np.stack([part.pivot(index='delivery_date', columns='policy', values=m)
                           .reindex(index=dates, columns=policies).to_numpy(float) for m in ('MAE', 'WIS')], axis=2)
        counts = part.pivot(index='delivery_date', columns='policy', values='n_hours').reindex(index=dates, columns=policies).to_numpy(float)
        if not np.isfinite(counts).all() or (counts < 0).any() or not (counts == counts[:, :1]).all():
            raise ValueError('daily observation counts must be paired and nonnegative')
        valid = counts[:, 0] > 0
        if not np.array_equal(np.isfinite(values), np.broadcast_to(valid[:, None, None], values.shape)):
            raise ValueError('daily missingness must be identical and agree with zero-hour dates')
        ix = np.asarray(indices[fold])
        if ix.shape != (replicates, 90) or not np.issubdtype(ix.dtype, np.integer) or (ix < 0).any() or (ix >= 90).any():
            raise ValueError('invalid bootstrap index fixture')
        sums = np.where(valid[:, None, None], values, 0) * counts[:, :, None]
        sampled_means = _divide(sums[ix].sum(axis=1), counts[ix].sum(axis=1)[:, :, None])
        point = _divide(sums.sum(axis=0), counts.sum(axis=0)[:, None])
        samples.append(_divide(sampled_means, sampled_means[:, :1, :]))
        points.append(_divide(point, point[:1, :]))
        daily_samples = _divide(np.where(valid[:, None, None], values, 0)[ix].sum(axis=1), valid[ix].sum(axis=1)[:, None, None])
        daily_point = _divide(np.where(valid[:, None, None], values, 0).sum(axis=0), valid.sum())
        for candidate, baseline, role in contrast_list:
            a, b = policies.index(candidate), policies.index(baseline)
            for mi, metric in enumerate(('MAE', 'WIS')):
                draws = daily_samples[:, a, mi] - daily_samples[:, b, mi]
                pt = daily_point[a, mi] - daily_point[b, mi]
                lo, hi, status, unresolved = interval(draws, pt)
                rows.append(dict(scope=fold, candidate=candidate, baseline=baseline, metric=metric, difference=float(pt),
                                 ci_lower=lo, ci_upper=hi, status=status, undefined_replicates=unresolved,
                                 replicates=replicates, estimand='paired mean daily loss difference', confidence=.95,
                                 evidence_class='descriptive_paired_daily', ratio=np.nan, ratio_ci_lower=np.nan,
                                 ratio_ci_upper=np.nan, role=role))
                stored.append(pd.DataFrame({'scope': fold, 'candidate': candidate, 'baseline': baseline, 'metric': metric,
                                            'replicate': np.arange(replicates), 'difference': draws, 'ratio': np.nan}))
    aggregate, point = np.mean(samples, axis=0), np.mean(points, axis=0)
    for candidate, baseline, role in contrast_list:
        a, b = policies.index(candidate), policies.index(baseline)
        for mi, metric in enumerate(('MAE', 'WIS')):
            draws = aggregate[:, a, mi] - aggregate[:, b, mi]
            pt = point[a, mi] - point[b, mi]
            lo, hi, status, unresolved = interval(draws, pt)
            ratios = _divide(aggregate[:, a, mi], aggregate[:, b, mi]) - 1
            rpt = float(_divide(point[a, mi], point[b, mi]) - 1)
            rlo, rhi, rstatus, runresolved = interval(ratios, rpt)
            rows.append(dict(scope='equal_fold', candidate=candidate, baseline=baseline, metric=metric, difference=float(pt),
                             ci_lower=lo, ci_upper=hi, status=status if rstatus == 'resolved' else rstatus,
                             undefined_replicates=max(unresolved, runresolved), replicates=replicates,
                             estimand='equal-fold B0-normalized eligible-hour mean difference', confidence=.95,
                             evidence_class='exploratory_post_selection', ratio=rpt, ratio_ci_lower=rlo, ratio_ci_upper=rhi,
                             role=role))
            stored.append(pd.DataFrame({'scope': 'equal_fold', 'candidate': candidate, 'baseline': baseline, 'metric': metric,
                                        'replicate': np.arange(replicates), 'difference': draws, 'ratio': ratios}))
    scores = pd.concat([pd.DataFrame({'policy': p, 'metric': metric, 'replicate': np.arange(replicates),
                                      'S': aggregate[:, policies.index(p), mi]})
                        for p in policies for mi, metric in enumerate(('MAE', 'WIS'))], ignore_index=True)
    fingerprint = hashlib.sha256(b''.join(np.asarray(indices[f], dtype='<i8').tobytes() for f in FOLDS)).hexdigest()
    meta = dict(seed=BOOTSTRAP_SEED, replicates=replicates, block_days=BLOCK_DAYS, calendar_days=90, index_sha256=fingerprint,
                index_equals_cp20=fingerprint == CP20_INDEX_SHA256, shared_index_sets=1,
                method='paired noncircular calendar moving-block percentile',
                ratio_method='R_b = S_policy,b / S_comparator,b - 1 from the same replicates; point = full-sample ratio - 1',
                undefined_handling='unresolved; no dropping or redrawing')
    return pd.DataFrame(rows), pd.concat(stored, ignore_index=True), scores, meta


def _eq(uncertainty, candidate, baseline):
    part = uncertainty.loc[uncertainty.scope.eq('equal_fold') & uncertainty.candidate.eq(candidate)
                           & uncertainty.baseline.eq(baseline)].set_index('metric')
    if set(part.index) != {'MAE', 'WIS'}:
        raise ValueError(f'missing contrast {candidate}-{baseline}')
    return part


def joint_reading(uncertainty, candidate, baseline):
    """§14.4/§17.5 endpoint reading of one contrast's equal-fold differences."""
    part = _eq(uncertainty, candidate, baseline)
    resolved = part.status.eq('resolved').all()
    directions = {m: ('lower (better)' if part.loc[m, 'difference'] < 0 else 'higher (worse)' if part.loc[m, 'difference'] > 0
                      else 'unchanged') for m in ('MAE', 'WIS')}
    if resolved and part.loc['WIS', 'ci_upper'] < 0 and part.loc['MAE', 'ci_upper'] <= 0:
        reading = 'observed joint improvement'
    elif resolved and part.loc['WIS', 'ci_lower'] > 0 and part.loc['MAE', 'ci_lower'] >= 0:
        reading = 'observed joint worsening'
    else:
        reading = 'no demonstrated joint preference'
    return {'contrast': f'{candidate}-{baseline}', 'reading': reading, 'directions': directions,
            **{f'dS_{m}': {'point': float(part.loc[m, 'difference']), 'ci': [float(part.loc[m, 'ci_lower']), float(part.loc[m, 'ci_upper'])],
                           'ratio': float(part.loc[m, 'ratio']), 'ratio_ci': [float(part.loc[m, 'ratio_ci_lower']), float(part.loc[m, 'ratio_ci_upper'])]}
               for m in ('MAE', 'WIS')}}


def _values(eq):
    return {m: {'difference': float(eq.loc[m, 'difference']), 'ci_lower': float(eq.loc[m, 'ci_lower']),
                'ci_upper': float(eq.loc[m, 'ci_upper']), 'ratio': float(eq.loc[m, 'ratio']),
                'ratio_ci': [float(eq.loc[m, 'ratio_ci_lower']), float(eq.loc[m, 'ratio_ci_upper'])]} for m in ('MAE', 'WIS')}


def _no_fold_worse(uncertainty, candidate, baseline):
    folds = uncertainty.loc[uncertainty.scope.isin(FOLDS) & uncertainty.candidate.eq(candidate) & uncertainty.baseline.eq(baseline)]
    if len(folds) != 10:
        raise ValueError(f'per-fold intervals missing for {candidate}-{baseline}')
    worse = folds.loc[folds.ci_lower > 0, ['scope', 'metric', 'difference', 'ci_lower', 'ci_upper']]
    met = bool(folds.status.eq('resolved').all() and worse.empty)
    return met, {'folds_decisively_worse': worse.to_dict('records'),
                 'per_fold': folds[['scope', 'metric', 'difference', 'ci_lower', 'ci_upper']].to_dict('records')}


def _section8_met(criteria, policy):
    part = criteria.loc[criteria.policy.eq(policy)]
    return bool(len(part) and part.status.eq('met').all()), {
        'not_met': part.loc[~part.passed, ['criterion', 'metric', 'scope', 'actual', 'lower_limit', 'upper_limit']].to_dict('records'),
        'criteria_rows': int(len(part))}


ENGINEERING = ('bound to the fresh Integration verdict on the final candidate (docs/track-b/evidence/cp-22/integration.md); '
               'an INCOMPLETE or BLOCKED return yields no decision')


def _decide(conditions: dict) -> dict:
    unmet = [k for k in sorted(conditions) if not conditions[k]['met']]
    return {'conditions': conditions, 'met': not unmet, 'first_unmet_condition': unmet[0] if unmet else None, 'unmet_conditions': unmet}


def replacement(uncertainty, criteria, *, keys_complete: bool) -> dict:
    """`cp22-replacement`: R, then M, each against v4 (HGL) on identical rows (§20.6)."""
    out = {'rule': RULES['cp22-replacement'], 'application': APPLICATION, 'candidates': {}}
    for cand in ('R', 'M'):
        eq = _eq(uncertainty, cand, 'HGL')
        c1 = bool(eq.status.eq('resolved').all() and eq.loc['MAE', 'ci_lower'] <= 0 and eq.loc['WIS', 'ci_lower'] <= 0)
        c2, v2 = _section8_met(criteria, cand)
        c4, v4 = _no_fold_worse(uncertainty, cand, 'HGL')
        out['candidates'][cand] = _decide({
            1: {'met': c1, 'values': _values(eq)}, 2: {'met': c2, 'values': v2},
            3: {'met': bool(keys_complete), 'values': {'keys_issued_finite_ordered': bool(keys_complete), 'engineering_pass': ENGINEERING}},
            4: {'met': c4, 'values': v4}})
    winner = next((c for c in ('R', 'M') if out['candidates'][c]['met']), None)
    out['winner'] = winner
    out['verdict'] = {'R': 'R replaces v4 (three-block construction superseded)',
                      'M': 'M replaces v4 (three-block construction superseded)',
                      None: 'no replacement: CP-22 stops at its return and the Owner decides; three-block v4 stays current'}[winner]
    out['first_unmet_condition'] = {c: out['candidates'][c]['first_unmet_condition'] for c in ('R', 'M')}
    return out


def dynamic_layer(uncertainty, criteria, metrics, w: str | None) -> dict:
    if w is None:
        return {'rule': RULES['cp22-dynamic-layer'], 'applies': False, 'verdict': 'not applicable: no winner W',
                'adopted': False, 'first_unmet_condition': None}
    eq = _eq(uncertainty, 'W+DL', w)
    c1 = bool(eq.status.eq('resolved').all() and eq.loc['WIS', 'ci_upper'] < 0)
    c2 = bool(eq.status.eq('resolved').all() and eq.loc['MAE', 'ci_lower'] <= 0)
    c3, v3 = _section8_met(criteria, 'W+DL')
    pooled = metrics.loc[metrics.scope.eq('pooled')].set_index('policy')
    cov_dl, cov_w = float(pooled.loc['W+DL', 'coverage95']), float(pooled.loc[w, 'coverage95'])
    c4 = abs(cov_dl - 0.95) < abs(cov_w - 0.95)
    c5, v5 = _no_fold_worse(uncertainty, 'W+DL', w)
    d = _decide({1: {'met': c1, 'values': _values(eq)}, 2: {'met': c2, 'values': _values(eq)}, 3: {'met': c3, 'values': v3},
                 4: {'met': bool(c4), 'values': {'pooled_coverage95_W+DL': cov_dl, f'pooled_coverage95_{w}': cov_w,
                                                 'distance_W+DL': abs(cov_dl - 0.95), 'distance_W': abs(cov_w - 0.95)}},
                 5: {'met': c5, 'values': v5}})
    return {'rule': RULES['cp22-dynamic-layer'], 'applies': True, 'W': w, **d, 'adopted': d['met'],
            'verdict': f'W+DL adopted: {w} with the dynamic interval layer' if d['met'] else f'W with HG\'s H layer ({w}) stands'}


def fast_component(uncertainty, criteria, dl: dict) -> dict:
    if not dl.get('adopted'):
        return {'rule': RULES['cp22-fast-component'], 'applies': False, 'adopted': False, 'first_unmet_condition': None,
                'verdict': 'not applicable: W+DL was not adopted; (W+DLF)-(W+DL) is descriptive only'
                           if dl.get('applies') else 'not applicable: no winner W'}
    eq = _eq(uncertainty, 'W+DLF', 'W+DL')
    c1 = bool(eq.status.eq('resolved').all() and eq.loc['WIS', 'ci_upper'] < 0)
    c2 = bool(eq.status.eq('resolved').all() and eq.loc['MAE', 'ci_lower'] <= 0)
    c3, v3 = _section8_met(criteria, 'W+DLF')
    c4, v4 = _no_fold_worse(uncertainty, 'W+DLF', 'W+DL')
    d = _decide({1: {'met': c1, 'values': _values(eq)}, 2: {'met': c2, 'values': _values(eq)}, 3: {'met': c3, 'values': v3},
                 4: {'met': c4, 'values': v4}})
    return {'rule': RULES['cp22-fast-component'], 'applies': True, **d, 'adopted': d['met'],
            'verdict': 'W+DLF replaces W+DL' if d['met'] else 'W+DL stands'}


def evaluate(predictions, expected_keys, policies, w: str | None, *, production=True, replicates=BOOTSTRAP_REPLICATES,
             indices=None):
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
    contrast_list = contrasts(w)
    uncertainty, replicate_draws, replicate_scores, meta = bootstrap(daily, policies, contrast_list, replicates=replicates,
                                                                     indices=indices)
    status = {}
    for policy in candidates:
        part = criteria.loc[criteria.policy.eq(policy)]
        status[policy] = 'unassessed' if part.status.eq('unassessed').any() else ('met' if part.passed.all() else 'not_met')
    rep = replacement(uncertainty, criteria, keys_complete=True)
    dl = dynamic_layer(uncertainty, criteria, metrics, w) if w else dynamic_layer(None, None, None, None)
    fast = fast_component(uncertainty, criteria, dl)
    readings = {f'{c}-{b}': {**joint_reading(uncertainty, c, b), 'role': role} for c, b, role in contrast_list}
    summary = dict(
        policies=policies, scores=scores, original_section8_status=status,
        failed_criteria={p: sorted(criteria.loc[criteria.policy.eq(p) & ~criteria.passed, 'criterion'].unique().tolist()) for p in candidates},
        replacement=rep, dynamic_layer=dl, fast_component=fast, contrasts=readings, bootstrap=meta,
        inference='exploratory post-selection; no equivalence, absence-of-benefit or absence-of-harm claim; mixed = no demonstrated joint preference',
        primary_weighting='equal-fold B0-normalized eligible-hour losses', pooled_weighting='eligible hours',
        evidence_class='development_post_selection', product_status='v1 remains the released product and demo',
        engineering_status='not assessed by scoring; bound to the fresh Integration verdict',
        validation=dict(production=production, rows=len(frame), keys_per_policy=len(frame) // len(policies),
                        missing_predictions=0, nonfinite_quantiles=0, crossings=0),
        support_rule=dict(minimum_represented_dates=SUPPORT_DATES, role='reporting only; no product gate'),
        economics='zero new economic runs or thresholds')
    return dict(metrics=metrics, diagnostics=diagnostics, uncertainty=uncertainty, criteria=criteria, daily=daily,
                hourly=hourly, replicates=replicate_draws, replicate_scores=replicate_scores, summary=summary)
