"""Frozen descriptive CP-26 calibration, causal states and moment diagnostics.

No diagnostic changes the scorer's frozen calendar or forecast availability.
The complete stream×24×3 moment family is tested on all-eligible 24-hour dates;
all four masks receive calibration tables. Arrays are yielded cell by cell so
the scheduler can persist compressed draws without retaining a family in RAM.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from cp25 import guard
from cp25.calendar import expected_hours
from .inference import CHUNK_REPLICATES, calendar_runs, draw_plan, hac_mean_covariance, holm
from .scoring import (CP25_MANIFEST_SHA256, DAY_GROUPS, INTERVALS, LEVELS, MASKS,
                      QUANTILES, RESEARCH_STREAMS, _keys, _sha, registered_breakdown_cells)

MOMENT_REPLICATES = 100_000
MOMENT_SEED = 20261013
MINIMUM_MOMENT_DATES = 180
BIN_PROBABILITIES = np.array([.025, .075, .15, .25, .25, .15, .075, .025])
PRICE_PUBLICATION_TEXT = 'auction day D-1 for delivery D: no later than one hour after the 12:00 Europe/Berlin gate, i.e. 13:00 local'


def prospective_specification():
    return {'schema': 'cp26-diagnostic-semantics-v1', 'evidence_class': 'descriptive',
            'coverage': 'inclusive bounds; equal-date means beside pooled-hour rates',
            'quantile_hits': 'both strict Y<q and weak Y<=q plus exact tie rates',
            'histogram': 'eight quantile bins, upper-interval ties; observed/expected probabilities; never exact PIT',
            'histogram_expected': BIN_PROBABILITIES.tolist(),
            'Christoffersen': 'each stream/mask/interval chronological UTC-hour Bernoulli hits; transitions only exact1h; bound ties unavailable; UC1df,IND1df,CC2df',
            'moment_family': 'all-eligible only; complete frozen lexical streams×localhour00..23×interval50/80/95; other masks calibration descriptive',
            'moment_days': '24-hour days only; complete calendar lag1/lag7; n>=180',
            'moments': '(I-c,(I-c)*(I_D-1-c),(I-c)*(I_D-7-c)); constructed originally, carried in blocks',
            'moment_HAC': 'Bartlett14 centered within fixed calendar-year strata, covariance of mean divisor n squared',
            'moment_null': 'centered vector moving calendar14 blocks, same run/stratum/short-run boundaries as primary; replicate re-estimated HAC',
            'moment_replicates': MOMENT_REPLICATES, 'moment_seed': MOMENT_SEED,
            'moment_order': 'lexical frozen stream, localhour00..23, interval50/80/95; unavailable cell no draw',
            'moment_degeneracy': 'ties, fewer180 complete moments, singular/nonpositive HAC, any invalidreplicate =>p1; no floor/drop/redraw',
            'moment_Holm': '5percent over complete frozen stream×24×3 family; unavailable p1 retained; lexical identity breaks ties',
            'moment_MonteCarlo': 'Clopper-Pearson95 interval of upper-tail replicatecount, reported beside+1 p-value',
            'price_availability': PRICE_PUBLICATION_TEXT,
            'negative_state': 'latest7 complete fully-known price days, available prefix, none0; gaps skipped, span/support disclosed',
            'volatility_state': 'sampleSD ddof1 of latest28 complete known dailymeans; strictabove linearp90 of365 preceding rollingSDs; current excluded; startup high0 unknown1',
            'realized_slices': 'Y<0 actualhour and own dailyMAE strictabove linearp95 ofall-eligibleavailable dates perstream; apply four frozenmasks afterward; never selection/nominalcalibration gates'}


def attach_registered_price_availability(history, root):
    """Map CP-25's verbatim tier-2 release deadline, with its local-day offset.

    This is reconstructed documented availability, not an observed publication
    timestamp. The frozen tier register and manifest are verified before use.
    ``history`` must already be guarded raw price rows; the guard is repeated.
    """
    root = Path(root)
    manifest_path = root / 'reports/development-inputs/manifest.json'
    if _sha(manifest_path) != CP25_MANIFEST_SHA256:
        raise ValueError('CP-25 price availability manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    relative = 'reports/development-inputs/tier-register.json'
    actual_sha = _sha(root / relative)
    if manifest['outputs_sha256'].get(relative) != actual_sha:
        raise ValueError('CP-25 price availability tier-register mismatch')
    register = json.loads((root / relative).read_text())
    target = [row for row in register['published_inputs'] if row['input'].startswith('target price (')]
    if len(target) != 1 or target[0]['tier'] != 2 or target[0]['documented_publication'] != PRICE_PUBLICATION_TEXT:
        raise ValueError('registered price availability convention changed')
    out = _keys(history)
    out['available_utc'] = pd.to_datetime([
        datetime.combine(day-timedelta(days=1), time(13), tzinfo=guard.BERLIN).astimezone(guard.UTC)
        for day in out.delivery_date], utc=True)
    out.attrs['price_availability'] = {'tier_register_sha256': actual_sha, 'tier': 2,
                                     'documented_publication': PRICE_PUBLICATION_TEXT,
                                     'reconstructed_deadline': True}
    return out


def known_price_states(history, origins):
    """Complete published price vectors only; no target-day or future errors.

    The latest seven known dates use their available prefix. SD uses 28 known
    actual-hour day means and ddof1; its reference is the previous 365 rolling
    SDs, excluding current SD. Price gaps extend the represented span. Startup
    high-volatility=0/unknown=1 is disclosed and remains a GW observation.
    """
    history = _keys(history)
    column = 'price_eur_mwh' if 'price_eur_mwh' in history else 'y_true'
    if column not in history or 'available_utc' not in history:
        raise ValueError('price states require prices and registered available_utc')
    available = pd.to_datetime(history.available_utc, errors='raise')
    if available.dt.tz is None:
        raise ValueError('price publication instants must be timezone aware')
    history['available_utc'] = available.dt.tz_convert('UTC')
    complete = []
    for day, rows in history.groupby('delivery_date', sort=True):
        index = pd.DatetimeIndex(rows.timestamp_utc)
        prices = pd.to_numeric(rows[column], errors='raise').to_numpy(float)
        if index.equals(expected_hours(day).as_unit('ns')) and np.isfinite(prices).all():
            complete.append({'day': day, 'published': rows.available_utc.max(),
                             'mean': float(prices.mean()), 'negative': bool((prices < 0).any())})
    origins = [pd.Timestamp(origin).date() for origin in origins]
    if origins != sorted(set(origins)):
        raise ValueError('price state origins must be chronological and unique')
    result = []
    for origin in origins:
        guard.admit_delivery_day(origin)
        issue = pd.Timestamp(guard.issue_time(origin))
        known = [row for row in complete if row['day'] < origin and row['published'] < issue]
        last7, last28 = known[-7:], known[-28:]
        negative = any(row['negative'] for row in last7)
        sd, threshold, unknown = np.nan, np.nan, True
        sd_history = pd.Series([row['mean'] for row in known], dtype=float).rolling(28, min_periods=28).std(ddof=1).to_numpy()
        previous = sd_history[:-1]
        previous = previous[np.isfinite(previous)][-365:]
        if len(last28) == 28:
            sd = float(sd_history[-1])
        if np.isfinite(sd) and len(previous) == 365:
            threshold = float(np.quantile(previous, .90, method='linear'))
            unknown = False
        result.append({'delivery_date': origin, 'issue_utc': issue.isoformat(),
            'negative_price_risk': int(negative), 'negative_support_dates': len(last7),
            'negative_oldest_day': str(last7[0]['day']) if last7 else None,
            'negative_latest_day': str(last7[-1]['day']) if last7 else None,
            'negative_span_days': (last7[-1]['day']-last7[0]['day']).days+1 if last7 else 0,
            'high_volatility': int(not unknown and sd > threshold), 'volatility_unknown': int(unknown),
            'volatility': sd, 'volatility_threshold': threshold, 'volatility_support_dates': len(last28),
            'volatility_reference_values': len(previous),
            'volatility_oldest_day': str(last28[0]['day']) if last28 else None,
            'volatility_latest_day': str(last28[-1]['day']) if last28 else None,
            'volatility_span_days': (last28[-1]['day']-last28[0]['day']).days+1 if last28 else 0,
            'latest_fully_known_day': str(known[-1]['day']) if known else None,
            'availability_rule': PRICE_PUBLICATION_TEXT})
    return pd.DataFrame(result)


def calibration_tables(hourly, daily):
    """Inclusive interval coverage, strict/weak quantile hits and upper-tie bins.

    Primary descriptive values average actual hours within date then dates
    equally. Pooled-hour values are shown beside them. The eight-bin histogram
    reports observed/expected probability, never an exact PIT claim. Every
    registered breakdown cell remains present with an unavailable reason and
    undefined estimates when it has no emitted scored observations.
    """
    labels = ['stream', 'delivery_date', *MASKS, *DAY_GROUPS]
    h = (hourly.merge(daily[labels], on=['stream', 'delivery_date'], validate='many_to_one')
         if not hourly.empty else pd.DataFrame(columns=[*labels, 'local_hour', 'y_true', *QUANTILES]))
    y, q = h.y_true.to_numpy(float), h[list(QUANTILES)].to_numpy(float)
    h['quantile_bin'] = (y[:, None] >= q).sum(axis=1)
    h['any_quantile_tie'] = (y[:, None] == q).any(axis=1)
    for j, label in enumerate(QUANTILES):
        h[f'strict_{label}'], h[f'weak_{label}'], h[f'tie_{label}'] = y < q[:, j], y <= q[:, j], y == q[:, j]
    coverage, hits, histogram = [], [], []
    for mask in MASKS:
        selected = h[h[mask].astype(bool)]
        selected_calendar = daily[daily[mask]]
        dimensions = registered_breakdown_cells(include_local_hours=True)
        for stream in sorted(daily.stream.unique()):
            own = selected[selected.stream.eq(stream)]
            own_calendar = selected_calendar[selected_calendar.stream.eq(stream)]
            for name, value in dimensions:
                cell = own if name == 'overall' else own[own[name].eq(value)]
                # A local-hour calendar count describes date membership only;
                # it never reconstructs or fills CP-25's frozen scoring keys.
                calendar_cell = (own_calendar if name in ('overall', 'local_hour') else
                                 own_calendar[own_calendar[name].eq(value)])
                reason = ('no_calendar_support' if calendar_cell.empty else
                          'no_emitted_scored_hours' if cell.empty else None)
                common = {'mask': mask, 'stream': stream, 'group': name, 'value': value,
                          'n_hours': len(cell), 'n_dates': cell.delivery_date.nunique(),
                          'calendar_dates': len(calendar_cell), 'status': 'computable' if reason is None else 'unavailable',
                          'reason': reason, 'evidence_class': 'descriptive'}
                means = cell.groupby('delivery_date', sort=True).mean(numeric_only=True) if not cell.empty else None
                def average(quantity, *, pooled=False):
                    return float((cell if pooled else means)[quantity].mean()) if not cell.empty else np.nan
                for level, _, _, _ in INTERVALS:
                    record = {**common, 'interval': level}
                    for quantity in (f'coverage{level}', f'lower_exceedance{level}', f'upper_exceedance{level}', f'width{level}'):
                        record[quantity] = average(quantity)
                        record[f'pooled_hour_{quantity}'] = average(quantity, pooled=True)
                    for component in ('WIS_dispersion', 'WIS_underprediction', 'WIS_overprediction'):
                        record[component] = average(component)
                    coverage.append(record)
                for j, label in enumerate(QUANTILES):
                    hits.append({**common, 'quantile': label, 'nominal_probability': float(LEVELS[j]),
                                 'strict_hit': average(f'strict_{label}'),
                                 'weak_hit': average(f'weak_{label}'),
                                 'tie_rate': average(f'tie_{label}'),
                                 'pooled_hour_strict_hit': average(f'strict_{label}', pooled=True),
                                 'pooled_hour_weak_hit': average(f'weak_{label}', pooled=True)})
                for bin_number, expected in enumerate(BIN_PROBABILITIES):
                    indicator = cell.quantile_bin.eq(bin_number)
                    observed = float(indicator.groupby(cell.delivery_date).mean().mean()) if not cell.empty else np.nan
                    histogram.append({**common, 'bin': bin_number, 'expected_probability': float(expected),
                                      'observed_probability': observed, 'observed_over_expected': observed/expected,
                                      'pooled_hour_probability': float(indicator.mean()),
                                      'tie_rate': average('any_quantile_tie'),
                                      'tie_convention': 'upper interval', 'interpretation': 'quantile histogram; not exact PIT'})
    return {'coverage': pd.DataFrame(coverage), 'quantile_hits': pd.DataFrame(hits), 'histogram': pd.DataFrame(histogram)}


def _bernoulli_loglik(n0, n1, p):
    if p == 0:
        return 0. if n1 == 0 else -np.inf
    if p == 1:
        return 0. if n0 == 0 else -np.inf
    return float(n0*np.log1p(-p)+n1*np.log(p))


def christoffersen(hourly, daily):
    """UC/IND/CC descriptive LR on genuine adjacent UTC hours, ties unavailable."""
    h = (hourly.merge(daily[['stream', 'delivery_date', *MASKS]], on=['stream', 'delivery_date'], validate='many_to_one')
         if not hourly.empty else pd.DataFrame(columns=['stream', 'delivery_date', 'timestamp_utc', *MASKS]))
    rows = []
    for mask in MASKS:
        for stream in sorted(daily.stream.unique()):
            cell = h[h[mask].astype(bool) & h.stream.eq(stream)].sort_values('timestamp_utc')
            for level, _, lo, hi in INTERVALS:
                common = {'mask': mask, 'stream': stream, 'interval': level, 'n_hours': len(cell),
                          'n_dates': cell.delivery_date.nunique(), 'n_adjacent_transitions': 0,
                          'transition_counts': [[0, 0], [0, 0]],
                          'LR_UC': np.nan, 'LR_IND': np.nan, 'LR_CC': np.nan,
                          'p_UC': 1., 'p_IND': 1., 'p_CC': 1., 'evidence_class': 'descriptive',
                          'assumptions': 'Bernoulli interval hits; first-order Markov independence alternative'}
                if len(cell) < 2:
                    rows.append({**common, 'status': 'insufficient_support'})
                    continue
                ties = cell.y_true.eq(cell[QUANTILES[lo]]) | cell.y_true.eq(cell[QUANTILES[hi]])
                if ties.any():
                    rows.append({**common, 'status': 'forecast_bound_ties', 'bound_ties': int(ties.sum())})
                    continue
                hit = cell[f'coverage{level}'].to_numpy(int)
                n1, n0 = int(hit.sum()), int(len(hit)-hit.sum())
                uc = max(0., 2*(_bernoulli_loglik(n0, n1, n1/len(hit))-_bernoulli_loglik(n0, n1, level/100)))
                adjacent = (cell.timestamp_utc.diff() == pd.Timedelta(hours=1)).to_numpy()[1:]
                previous, current = hit[:-1][adjacent], hit[1:][adjacent]
                counts = np.array([[np.sum((previous == i) & (current == j)) for j in range(2)] for i in range(2)])
                n00, n01, n10, n11 = counts.ravel()
                transitions = int(counts.sum())
                if not transitions:
                    rows.append({**common, 'status': 'no_adjacent_transitions', 'LR_UC': uc,
                                 'p_UC': float(stats.chi2.sf(uc, 1)), 'transition_counts': counts.tolist()})
                    continue
                pi = (n01+n11)/transitions
                independent_ll = _bernoulli_loglik(n00+n10, n01+n11, pi)
                dependent_ll = sum(_bernoulli_loglik(int(a), int(b), b/(a+b)) if a+b else 0.
                                   for a, b in ((n00, n01), (n10, n11)))
                ind = max(0., 2*(dependent_ll-independent_ll))
                rows.append({**common, 'status': 'computable', 'LR_UC': uc, 'LR_IND': ind, 'LR_CC': uc+ind,
                             'p_UC': float(stats.chi2.sf(uc, 1)), 'p_IND': float(stats.chi2.sf(ind, 1)),
                             'p_CC': float(stats.chi2.sf(uc+ind, 2)), 'transition_counts': counts.tolist(),
                             'n_adjacent_transitions': transitions})
    return pd.DataFrame(rows)


def _positive_covariance(covariance):
    if covariance.ndim == 2:
        return bool(np.isfinite(covariance).all() and np.linalg.eigvalsh(covariance).min() > 0)
    finite = np.isfinite(covariance).all(axis=(1, 2))
    result = np.zeros(len(covariance), dtype=bool)
    if finite.any():
        result[finite] = np.linalg.eigvalsh(covariance[finite]).min(axis=1) > 0
    return result


def moment_test(cell, *, interval, rng=None, replicates=MOMENT_REPLICATES, production=True):
    """Memo's three-moment Wald; original calendar-lag products travel in blocks."""
    if production and replicates != MOMENT_REPLICATES:
        raise ValueError('production moment family requires 100,000 replicates')
    definition = next((row for row in INTERVALS if row[0] == interval), None)
    if definition is None:
        raise ValueError('unregistered interval')
    _, _, lo, hi = definition
    cell = cell.sort_values('delivery_date').copy()
    if cell.delivery_date.duplicated().any():
        raise ValueError('moment cell requires one local hour on each 24-hour date')
    bound_ties = cell.y_true.eq(cell[QUANTILES[lo]]) | cell.y_true.eq(cell[QUANTILES[hi]])
    result = {'p_value': 1., 'interval': interval, 'n_available_dates': len(cell), 'n_moment_dates': 0,
              'bound_ties': int(bound_ties.sum()), 'HAC_lag': 14, 'replicates_required': replicates,
              'evidence_class': 'descriptive', 'centering': 'within fixed calendar-year stratum',
              'draw_starts': np.empty((0, 0), np.int32), 'replicate_statistics': np.empty(0),
              'draw_metadata': None}
    if bound_ties.any():
        return {**result, 'status': 'forecast_bound_ties'}
    hits = cell.set_index('delivery_date')[f'coverage{interval}'].astype(float)
    dates, vectors = [], []
    c = interval/100
    for day, hit in hits.items():
        day = pd.Timestamp(day).date()
        lag1, lag7 = day-timedelta(days=1), day-timedelta(days=7)
        if lag1 in hits.index and lag7 in hits.index:
            dates.append(day)
            vectors.append((hit-c, (hit-c)*(hits.at[lag1]-c), (hit-c)*(hits.at[lag7]-c)))
    g = np.asarray(vectors, float).reshape(-1, 3)
    n = len(g)
    result.update(n_moment_dates=n, moment_dates=[str(day) for day in dates], moment_vectors=g)
    if n < MINIMUM_MOMENT_DATES:
        return {**result, 'status': 'fewer_than_180_complete_moment_dates'}
    runs = calendar_runs(dates)
    covariance = hac_mean_covariance(g, runs, centering='stratum')
    if not _positive_covariance(covariance):
        return {**result, 'status': 'singular_or_nonpositive_observed_HAC_covariance'}
    mean = g.mean(axis=0)
    observed = float(mean @ np.linalg.solve(covariance, mean))
    plan = draw_plan(dates, rng=np.random.default_rng(MOMENT_SEED) if rng is None else rng, replicates=replicates)
    draws = np.full(replicates, np.nan)
    for first in range(0, replicates, CHUNK_REPLICATES):
        last = min(first+CHUNK_REPLICATES, replicates)
        sampled = g[plan.indices(first, last)]
        replicate_covariance = hac_mean_covariance(sampled, runs, centering='stratum')
        valid = _positive_covariance(replicate_covariance)
        centered_mean = sampled.mean(axis=1)-mean
        if valid.any():
            solved = np.linalg.solve(replicate_covariance[valid], centered_mean[valid, :, None])[..., 0]
            draws[first:last][valid] = np.einsum('bp,bp->b', centered_mean[valid], solved)
    result.update(draw_starts=plan.starts, draw_metadata=plan.metadata(), replicate_statistics=draws,
                  statistic=observed, observed_HAC_covariance_of_mean=covariance)
    if not np.isfinite(draws).all():
        return {**result, 'status': 'singular_or_nonfinite_replicate_covariance_or_statistic'}
    count = int((draws >= observed).sum())
    p = (1+count)/(replicates+1)
    # Exact binomial interval for Monte Carlo exceedance probability; +1 rule
    # p-value remains separately reported and is what Holm consumes.
    lower = float(stats.beta.ppf(.025, count, replicates-count+1)) if count else 0.
    upper = float(stats.beta.ppf(.975, count+1, replicates-count)) if count < replicates else 1.
    return {**result, 'status': 'computable', 'p_value': p, 'upper_tail_count': count,
            'monte_carlo_probability_ci95': (lower, upper)}


def iter_moment_family(hourly, daily, *, streams=None, replicates=MOMENT_REPLICATES, production=True):
    """Freeze complete lexical stream×hour00..23×interval50/80/95 family.

    Unavailable cells remain p1 and consume no draw. A single fresh seed serves
    the family in that fixed order. Consume each yield into durable storage
    before advancing, then run ``moment_holm`` on its compact p-value rows.
    """
    streams = tuple(sorted(RESEARCH_STREAMS if production else daily.stream.unique())) if streams is None else tuple(sorted(streams))
    if production and streams != tuple(sorted(RESEARCH_STREAMS)):
        raise ValueError('production moment family is exactly 13 CP-26 research streams')
    if len(set(streams)) != len(streams):
        raise ValueError('moment family stream identities must be unique')
    labels = daily[['stream', 'delivery_date', 'all_eligible', 'canonical_hours']]
    h = hourly.merge(labels, on=['stream', 'delivery_date'], validate='many_to_one') if not hourly.empty else hourly
    rng = np.random.default_rng(MOMENT_SEED)
    for stream in streams:
        for hour in range(24):
            cell = (h[h.stream.eq(stream) & h.local_hour.eq(hour) & h.all_eligible & h.canonical_hours.eq(24)]
                    if not h.empty else pd.DataFrame(columns=['delivery_date', 'y_true', *QUANTILES, 'coverage50', 'coverage80', 'coverage95']))
            for interval, *_ in INTERVALS:
                yield {'identity': f'{stream}/hour{hour:02d}/interval{interval}', 'stream': stream,
                       'local_hour': hour, 'mask': 'all_eligible',
                       **moment_test(cell, interval=interval, rng=rng, replicates=replicates, production=production)}


def moment_holm(rows, *, streams=RESEARCH_STREAMS, production=True):
    if production and tuple(sorted(streams)) != tuple(sorted(RESEARCH_STREAMS)):
        raise ValueError('production Holm family is exactly 13 CP-26 research streams×24×3')
    expected = {f'{stream}/hour{hour:02d}/interval{interval}' for stream in streams
                for hour in range(24) for interval, *_ in INTERVALS}
    pvalues = {row['identity']: row['p_value'] for row in rows}
    if len(pvalues) != len(rows) or set(pvalues) != expected:
        raise ValueError('Holm requires every registered moment cell including unavailable p1 cells')
    return holm(pvalues, alpha=.05)


def realized_outcome_slices(hourly, daily):
    """Frozen descriptive negative hours and own-MAE strict upper-p95 dates.

    Per-stream extreme-MAE thresholds use all-eligible available dates and are
    retained before the four masks are applied. No conditional nominal-coverage
    or admission claim is attached to any realized-outcome slice.
    """
    rows, thresholds = [], []
    for stream in sorted(daily.stream.unique()):
        d = daily[daily.stream.eq(stream)]
        valid = d[d.all_eligible & d.available]
        threshold = float(np.quantile(valid.MAE, .95, method='linear')) if len(valid) else np.nan
        thresholds.append({'stream': stream, 'daily_MAE_p95': threshold, 'basis_dates': len(valid),
                           'status': 'computable' if len(valid) else 'unavailable',
                           'reason': None if len(valid) else 'no_available_all_eligible_daily_MAE',
                           'comparison': 'strict >', 'evidence_class': 'realized_outcome_descriptive'})
        extreme_dates = set(valid.loc[valid.MAE > threshold, 'delivery_date'])
        for mask in MASKS:
            measured = d[d[mask]]
            selected = measured[measured.delivery_date.isin(extreme_dates)]
            for metric in ('MAE', 'WIS'):
                rows.append({'stream': stream, 'mask': mask, 'slice': 'extreme_realized_daily_MAE', 'metric': metric,
                             'mean_daily_loss': float(selected[metric].mean()), 'n_dates': len(selected),
                             'status': 'computable' if len(selected) else 'unavailable',
                             'reason': None if len(selected) else 'no_dates_above_own_all_eligible_daily_MAE_p95',
                             'evidence_class': 'realized_outcome_descriptive'})
            h = (hourly[hourly.stream.eq(stream) & hourly.delivery_date.isin(measured.delivery_date) & hourly.y_true.lt(0)]
                 if not hourly.empty else pd.DataFrame(columns=['delivery_date', 'MAE', 'WIS']).astype({'MAE': float, 'WIS': float}))
            for metric in ('MAE', 'WIS'):
                by_date = h.groupby('delivery_date')[metric].mean()
                rows.append({'stream': stream, 'mask': mask, 'slice': 'negative_realized_hour', 'metric': metric,
                             'mean_daily_loss': float(by_date.mean()), 'n_dates': len(by_date), 'n_hours': len(h),
                             'status': 'computable' if len(h) else 'unavailable',
                             'reason': None if len(h) else 'no_negative_realized_scored_hours',
                             'evidence_class': 'realized_outcome_descriptive'})
    return {'slices': pd.DataFrame(rows), 'thresholds': pd.DataFrame(thresholds)}
