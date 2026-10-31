"""CP-26 emitted-forecast scores on the independently verified CP-25 population.

Loading a population computes no metric. The frozen calendar, record masks and
omitted-hour records are verified against CP-25's committed manifest before
caller keys are compared. Partial eligible days retain exactly their recorded
hours; complete-day H/learning eligibility has no role in scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cp25 import guard
from cp25.calendar import expected_hours

QUANTILES = ('p025', 'p10', 'p25', 'p50', 'p75', 'p90', 'p975')
LEVELS = np.array([.025, .1, .25, .5, .75, .9, .975])
INTERVALS = ((50, .5, 2, 4), (80, .2, 1, 5), (95, .05, 0, 6))
MASKS = ('all_eligible', 'never_scored', 'old_fold_2_5', 'newly_opened')
SCORE_FIRST, SCORE_LAST = date(2020, 12, 29), guard.BOUNDARY
CP25_MANIFEST_SHA256 = 'f3e8ca15e4466d371de5b6db9c181f77adaa8823ca2e7e0d325182120db32fd9'
POPULATION_FILES = ('eligibility-calendar.csv', 'record-masks.csv', 'ineligible-hours.csv')
DAY_GROUPS = ('season', 'year', 'quarter', 'weekday_weekend', 'nationwide_holiday',
              'canonical_hours', 'stress_2021_2022', 'negative_price_risk', 'high_volatility')
RESEARCH_STREAMS = ('nominal/B0', 'nominal/v1-R', 'nominal/v2', 'nominal/v3', 'nominal/v4', 'nominal/v5', 'nominal/D2',
                    'noload/v3', 'noload/v4', 'noload/v5', 'gfs6/v3', 'gfs6/v4', 'gfs6/v5')
RELEASED_REFERENCE_PINS = {
    'reports/cp2/holdout_predictions.parquet': '136e178d5ed171520bc05eb2961c1a7ea966f331ad2e126417274095498b29f7',
    'reports/cp2/holdout_report.json': 'c04666cd77b194137ce88baa23f6c78d8f6c75339d546dead8484ad3d10e9ae1',
    'scripts/cp2_final_holdout.py': '95d3d3574ba1b8af1ead403ce5b51a2d1dcdf9f98b342aad184cd3fdaef8b34b',
    'src/delu_forecast/experiment.py': '53ef459cc1e6227c37abda1c8913ce2116ac303833b54ee90502beb66dd5a740'}


def prospective_specification():
    return {'schema': 'cp26-scoring-semantics-v1', 'CP25_manifest_sha256': CP25_MANIFEST_SHA256,
            'population': 'committed eligible-calendar canonicalhours minus committed ineligiblehours; exact counts/masks verified',
            'measurement': [str(SCORE_FIRST), str(SCORE_LAST)], 'masks': list(MASKS),
            'day_weighting': 'mean actual frozeneligiblehours per date, then equal-date mean; never fill an omitted hour',
            'paired_missing': 'whole missing emissiondate retained in calendar and absent from every pairedcomparison involving stream',
            'B0_ratio': 'ratio of paired mean daily losses, never mean daily ratios',
            'fallback': 'only original emitted B0 quantiles, bitwise equality on frozenkeys; count eligible-calendar denominator',
            'partialday': 'score permitted frozenkeys; H and CP28learningcomplete predicates never alter scorekeys',
            'streams': list(RESEARCH_STREAMS), 'released_reference_pins': RELEASED_REFERENCE_PINS,
            'released_reference': 'separate historicalartifact, originaleligiblerow order independentlyreconstructed; issue instants not stored; originalcutoffs/identity preserved; no refit/native/H/learning claim'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dates(values):
    parsed = pd.to_datetime(values, errors='raise')
    if parsed.isna().any() or parsed.dt.tz is not None or not parsed.eq(parsed.dt.normalize()).all():
        raise ValueError('delivery_date must contain naive calendar dates')
    days = parsed.dt.date
    for day in days:
        guard.admit_delivery_day(day)
    return days


def _keys(frame):
    frame = frame.copy()
    if not {'timestamp_utc', 'delivery_date'}.issubset(frame):
        raise ValueError('scoring keys require timestamp_utc and delivery_date')
    ts = pd.to_datetime(frame.timestamp_utc, errors='raise')
    if not len(frame):
        frame['timestamp_utc'] = pd.Series([], dtype='datetime64[ns, UTC]')
        frame['delivery_date'] = _dates(frame.delivery_date)
        return frame.reset_index(drop=True)
    if ts.dt.tz is None:
        raise ValueError('scoring timestamps must be timezone aware')
    frame['timestamp_utc'] = ts.dt.tz_convert('UTC').dt.as_unit('ns')
    guard.admit_rows(frame.timestamp_utc)
    frame['delivery_date'] = _dates(frame.delivery_date)
    local = frame.timestamp_utc.dt.tz_convert('Europe/Berlin')
    if not local.dt.date.eq(frame.delivery_date).all() or not frame.timestamp_utc.eq(frame.timestamp_utc.dt.floor('h')).all():
        raise ValueError('scoring keys are not canonical local delivery hours')
    if frame[['timestamp_utc', 'delivery_date']].isna().any().any() or frame.timestamp_utc.duplicated().any():
        raise ValueError('missing or duplicate scoring keys')
    return frame.sort_values('timestamp_utc').reset_index(drop=True)


def _bools(values, name):
    if not values.isin([True, False]).all():
        raise ValueError(f'{name} must contain boolean flags')
    return values.astype(bool)


@dataclass(frozen=True)
class FrozenPopulation:
    calendar: pd.DataFrame
    keys: pd.DataFrame
    masks: pd.DataFrame
    evidence: dict

    def keys_by_day(self):
        """Explicit scoring keys for every retained date, including zero-key dates."""
        grouped = {day: pd.DatetimeIndex(rows.timestamp_utc).as_unit('ns')
                   for day, rows in self.keys.groupby('delivery_date', sort=True)}
        return {day: grouped.get(day, pd.DatetimeIndex([], tz='UTC'))
                for day in self.calendar.delivery_date}

    def verify_key_mapping(self, supplied):
        observed = {pd.Timestamp(day).date(): pd.DatetimeIndex(pd.to_datetime(index, utc=True)).as_unit('ns')
                    for day, index in supplied.items()}
        expected = self.keys_by_day()
        if len(observed) != len(supplied) or set(observed) != set(expected) or any(not observed[day].equals(index) for day, index in expected.items()):
            raise ValueError('caller per-date keys differ from committed CP-25 records')
        for index in observed.values():
            guard.admit_rows(index)
        return expected

    def verify_keys(self, supplied):
        observed = _keys(supplied)[['timestamp_utc', 'delivery_date']]
        expected = _keys(self.keys)[['timestamp_utc', 'delivery_date']]
        if not observed.equals(expected):
            raise ValueError('caller scoring keys differ from committed CP-25 records')
        return observed

    def verify_masks(self, supplied):
        observed = supplied.copy()
        observed['delivery_date'] = _dates(observed.delivery_date)
        observed = observed.sort_values('delivery_date').reset_index(drop=True)
        for name in MASKS:
            observed[name] = _bools(observed[name], name)
        if not observed[['delivery_date', *MASKS]].equals(self.masks[['delivery_date', *MASKS]]):
            raise ValueError('caller masks differ from committed CP-25 records')
        return observed


def population_from_records(calendar, record_masks, omitted_hours, *, evidence):
    """Validate population metadata, also usable with explicit synthetic fixtures.

    Production callers use ``load_population``; this constructor does not claim
    that arbitrary passed frames are committed CP-25 records.
    """
    cal, masks = calendar.copy(), record_masks.copy()
    cal['delivery_date'], masks['delivery_date'] = _dates(cal.delivery_date), _dates(masks.delivery_date)
    if cal.delivery_date.duplicated().any() or masks.delivery_date.duplicated().any():
        raise ValueError('duplicate frozen calendar or mask date')
    cal = cal.sort_values('delivery_date').reset_index(drop=True)
    masks = masks.sort_values('delivery_date').reset_index(drop=True)
    if not cal.delivery_date.equals(masks.delivery_date):
        raise ValueError('CP-25 calendar and mask dates disagree')
    cal['eligible'] = _bools(cal.eligible, 'eligible')
    if not cal.eligible.equals(_bools(masks.eligible, 'mask eligible')):
        raise ValueError('CP-25 eligibility flags disagree')
    omitted = _keys(omitted_hours)
    eligible_dates = set(cal.loc[cal.eligible, 'delivery_date'])
    if not set(omitted.delivery_date).issubset(eligible_dates):
        raise ValueError('omitted-hour record outside eligible date')
    omitted_index = pd.DatetimeIndex(omitted.timestamp_utc)
    key_rows = []
    for row in cal.itertuples(index=False):
        hours = expected_hours(row.delivery_date).as_unit('ns')
        if row.canonical_hours != len(hours):
            raise ValueError('frozen canonical-hour count disagrees with calendar')
        removed = omitted_index[omitted.delivery_date.eq(row.delivery_date)]
        if not removed.isin(hours).all():
            raise ValueError('omitted hours are not canonical')
        if row.eligible:
            retained = hours[~hours.isin(removed)]
            if len(retained) != row.landed_eligible_hours or len(removed) != row.ineligible_hours_on_eligible_day:
                raise ValueError('CP-25 omitted-hour records disagree with eligible-hour counts')
            key_rows.extend({'timestamp_utc': ts, 'delivery_date': row.delivery_date} for ts in retained)
    keys = _keys(pd.DataFrame(key_rows, columns=['timestamp_utc', 'delivery_date']))
    in_measurement = cal.delivery_date.between(SCORE_FIRST, SCORE_LAST)
    cal = cal.loc[in_measurement].reset_index(drop=True)
    masks = masks.loc[in_measurement].reset_index(drop=True)
    frozen_masks = pd.DataFrame({'delivery_date': cal.delivery_date, 'all_eligible': cal.eligible,
        'never_scored': cal.eligible & _bools(masks.never_scored, 'never_scored'),
        'old_fold_2_5': cal.eligible & _bools(masks.old_fold_2_5, 'old_fold_2_5'),
        'newly_opened': cal.eligible & cal.delivery_date.between(date(2026, 4, 8), SCORE_LAST)})
    keys = keys[keys.delivery_date.isin(cal.delivery_date)].reset_index(drop=True)
    return FrozenPopulation(cal, keys, frozen_masks, dict(evidence))


def load_population(root, *, supplied_keys=None, supplied_masks=None):
    """Read hash-bound CP-25 metadata, behind its development-date guard."""
    root = Path(root)
    manifest_path = root / 'reports/development-inputs/manifest.json'
    if _sha(manifest_path) != CP25_MANIFEST_SHA256:
        raise ValueError('CP-25 manifest identity mismatch')
    manifest = json.loads(manifest_path.read_text())
    hashes, frames = {}, {}
    for name in POPULATION_FILES:
        relative = f'reports/development-inputs/{name}'
        hashes[relative] = _sha(root / relative)
        if manifest['outputs_sha256'].get(relative) != hashes[relative]:
            raise ValueError(f'CP-25 frozen population file identity mismatch: {name}')
        frames[name] = pd.read_csv(root / relative)
    result = population_from_records(frames['eligibility-calendar.csv'], frames['record-masks.csv'],
        frames['ineligible-hours.csv'], evidence={'manifest_sha256': CP25_MANIFEST_SHA256,
                                                'files_sha256': hashes, 'committed_cp25_verified': True})
    if supplied_keys is not None:
        result.verify_keys(supplied_keys)
    if supplied_masks is not None:
        result.verify_masks(supplied_masks)
    return result


def calendar_groups(calendar, known_states=None):
    """Deterministic groups plus explicitly supplied pre-outcome state flags."""
    import holidays
    out = calendar[['delivery_date', 'canonical_hours', 'eligible']].copy()
    d = pd.to_datetime(out.delivery_date)
    out['season'] = d.dt.month.map({12:'DJF', 1:'DJF', 2:'DJF', 3:'MAM', 4:'MAM', 5:'MAM',
                                  6:'JJA', 7:'JJA', 8:'JJA', 9:'SON', 10:'SON', 11:'SON'})
    out['year'], out['quarter'] = d.dt.year, d.dt.to_period('Q').astype(str)
    out['weekday_weekend'] = np.where(d.dt.dayofweek >= 5, 'weekend', 'weekday')
    federal = holidays.country_holidays('DE', years=sorted(set(d.dt.year)))
    out['nationwide_holiday'] = out.delivery_date.map(lambda day: day in federal)
    out['stress_2021_2022'] = out.year.isin([2021, 2022])
    if known_states is not None:
        states = known_states.copy()
        states['delivery_date'] = _dates(states.delivery_date)
        if states.delivery_date.duplicated().any() or set(states.delivery_date) != set(out.delivery_date):
            raise ValueError('known states require exactly one row for every measurement date')
        required = ['negative_price_risk', 'high_volatility']
        if not set(required).issubset(states):
            raise ValueError('both registered issue-time states are required')
        out = out.merge(states[['delivery_date', *required]], on='delivery_date', validate='one_to_one')
    else:
        out['negative_price_risk'], out['high_volatility'] = None, None
    return out


def hourly_losses(truth, quantiles):
    """Exactly §24.7's emitted-p50 MAE and seven-quantile WIS."""
    y, q = np.asarray(truth, float), np.asarray(quantiles, float)
    if q.shape != (len(y), 7) or not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError('truth and emitted quantiles must be finite and aligned')
    if (np.diff(q, axis=1) < 0).any():
        raise ValueError('emitted quantile crossing')
    result = {'MAE': np.abs(y - q[:, 3])}
    wis = .5 * result['MAE']
    dispersion = np.zeros(len(y))
    underprediction = .5*np.maximum(y-q[:, 3], 0)
    overprediction = .5*np.maximum(q[:, 3]-y, 0)
    for level, alpha, lo, hi in INTERVALS:
        width = q[:, hi] - q[:, lo]
        wis += (alpha / 2) * width + np.maximum(q[:, lo] - y, 0) + np.maximum(y - q[:, hi], 0)
        dispersion += (alpha / 2)*width
        underprediction += np.maximum(y-q[:, hi], 0)
        overprediction += np.maximum(q[:, lo]-y, 0)
        result[f'coverage{level}'] = (y >= q[:, lo]) & (y <= q[:, hi])
        result[f'lower_exceedance{level}'] = y < q[:, lo]
        result[f'upper_exceedance{level}'] = y > q[:, hi]
        result[f'width{level}'] = width
    result['WIS'] = wis / 3.5
    result['WIS_dispersion'], result['WIS_underprediction'], result['WIS_overprediction'] = (
        dispersion/3.5, underprediction/3.5, overprediction/3.5)
    return result


def score_records(records, truth, population, *, streams, known_states=None, production=True):
    """Score immutable replay records; retain failed/excluded dates explicitly.

    ``streams`` freezes route/policy identities before evaluation. ``truth`` is a
    frame of the exact scoring keys and ``y_true``. Every stream must supply one
    replay record for every measurement calendar date. Missing output is a
    whole-day missing loss, not a dropped hour. B0 fallback values are verified
    against that date's emitted nominal B0 output.
    """
    if production and not population.evidence.get('committed_cp25_verified'):
        raise ValueError('production scoring requires verified committed CP-25 population')
    if production and known_states is None:
        raise ValueError('production scoring requires both frozen issue-time price states')
    truth = _keys(truth)
    population.verify_keys(truth)
    y = pd.to_numeric(truth.y_true, errors='raise').to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError('frozen scoring truth is nonfinite')
    streams = tuple(streams)
    if not streams or len(set(streams)) != len(streams) or 'nominal/B0' not in streams:
        raise ValueError('freeze unique scoring streams including nominal/B0')
    by_record = {}
    dates = set(population.calendar.delivery_date)
    for record in records:
        day = date.fromisoformat(record['day'])
        guard.admit_delivery_day(day)
        if day < SCORE_FIRST:
            continue
        stream = f"{record['route']}/{record['policy']}"
        key = (stream, day)
        if stream not in streams or day not in dates or key in by_record:
            raise ValueError('unexpected or duplicate measurement replay record')
        if not record['scored']:
            raise ValueError('measurement record is marked unscored')
        if pd.Timestamp(record['issue_utc']) != pd.Timestamp(guard.issue_time(day)):
            raise ValueError('record does not retain frozen issue instant')
        by_record[key] = record
    expected = {(stream, day) for stream in streams for day in dates}
    if set(by_record) != expected:
        raise ValueError('every measurement date and stream must remain in replay accounting')
    groups = calendar_groups(population.calendar, known_states).set_index('delivery_date')
    truth_by_day = {day: rows for day, rows in truth.groupby('delivery_date', sort=True)}
    daily, hourly = [], []
    for stream in streams:
        for day in sorted(dates):
            record = by_record[(stream, day)]
            actual = truth_by_day.get(day, truth.iloc[:0])
            frozen = pd.DatetimeIndex(actual.timestamp_utc)
            scored = pd.DatetimeIndex(pd.to_datetime(record['scoring_index'], utc=True)).as_unit('ns')
            guard.admit_rows(scored)
            if not scored.equals(frozen):
                raise ValueError('emission scoring_index differs from frozen CP-25 hour records')
            source, values = record['output_source'], record['emitted_quantiles']
            if source not in ('native', 'B0', 'missing') or (source == 'missing') != (values is None):
                raise ValueError('emitted output and source status disagree')
            if stream == 'nominal/B0' and source == 'B0':
                raise ValueError('B0 cannot fall back to itself')
            q = None if values is None else np.asarray(values, float)
            if q is not None and not len(frozen) and not q.size:
                q = q.reshape(0, 7)
            if q is not None and (q.shape != (len(frozen), 7) or not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any()):
                raise ValueError('invalid emitted quantile day')
            if source == 'B0':
                baseline = by_record[('nominal/B0', day)]
                baseline_q = None if baseline['emitted_quantiles'] is None else np.asarray(baseline['emitted_quantiles'], float)
                if baseline_q is not None and not len(frozen) and not baseline_q.size:
                    baseline_q = baseline_q.reshape(0, 7)
                if baseline_q is None or not np.array_equal(q, baseline_q):
                    raise ValueError('fallback differs from emitted nominal B0')
            available = bool(groups.at[day, 'eligible'] and len(frozen) and q is not None)
            row = {'stream': stream, 'route': record['route'], 'policy': record['policy'], 'delivery_date': day,
                   'eligible': bool(groups.at[day, 'eligible']), 'available': available,
                   'scoring_hours': len(frozen), 'MAE': np.nan, 'WIS': np.nan,
                   'output_source': source, 'fallback': source == 'B0', 'failed': source == 'missing',
                   'reason': record.get('reason'), 'native_central_valid': record['native_central_valid'],
                   'interval_ready': record['interval_ready'], 'error_eligible': record['error_eligible'],
                   **groups.loc[day].drop(['eligible']).to_dict()}
            if available:
                losses = hourly_losses(actual.y_true.to_numpy(float), q)
                row.update(MAE=float(np.mean(losses['MAE'])), WIS=float(np.mean(losses['WIS'])))
                h = actual[['timestamp_utc', 'delivery_date', 'y_true']].copy()
                h['stream'], h['local_hour'] = stream, h.timestamp_utc.dt.tz_convert('Europe/Berlin').dt.hour
                for name, value in losses.items():
                    h[name] = value
                for j, name in enumerate(QUANTILES):
                    h[name] = q[:, j]
                hourly.append(h)
            daily.append(row)
    daily = pd.DataFrame(daily).merge(population.masks, on='delivery_date', validate='many_to_one')
    hourly = pd.concat(hourly, ignore_index=True) if hourly else pd.DataFrame()
    summary = summaries(daily)
    return {'daily': daily, 'hourly': hourly, 'population_evidence': population.evidence,
            'summaries': summary, 'secondary_summaries': secondary_summaries(summary),
            'local_hour_summaries': local_hour_summaries(hourly, daily), 'fallbacks': fallback_summary(daily)}


def paired_losses(daily, left, right, *, metric, mask='all_eligible', group=None):
    """One fixed calendar paired mask, with both missing counts disclosed."""
    if metric not in ('MAE', 'WIS') or mask not in MASKS:
        raise ValueError('unregistered score metric or mask')
    a = daily[daily.stream.eq(left)].set_index('delivery_date').sort_index()
    b = daily[daily.stream.eq(right)].set_index('delivery_date').sort_index()
    if not a.index.equals(b.index):
        raise ValueError('comparison streams do not share the retained calendar')
    selected = a[mask].astype(bool)
    if group is not None:
        name, value = group
        selected &= a[name].eq(value)
    valid_a, valid_b = np.isfinite(a[metric]), np.isfinite(b[metric])
    paired = selected & valid_a & valid_b
    frame = pd.DataFrame({'delivery_date': a.index[paired], 'a': a.loc[paired, metric].to_numpy(float),
                          'b': b.loc[paired, metric].to_numpy(float)})
    metadata = {'calendar_dates': int(selected.sum()), 'paired_dates': int(paired.sum()),
                'left_missing_dates': int((selected & ~valid_a).sum()),
                'right_missing_dates': int((selected & ~valid_b).sum()),
                'either_missing_dates': int((selected & ~paired).sum())}
    return frame, metadata


def registered_breakdown_cells(*, include_local_hours=False):
    """Outcome-free descriptive grid for the registered measurement calendar.

    Fixed binary levels and DST day lengths remain present even when a mask
    has no such date or a stream emitted no forecast. These identities do not
    imply that an hour is a frozen CP-25 scoring key.
    """
    domains = {
        'season': ('DJF', 'MAM', 'JJA', 'SON'),
        'year': tuple(range(SCORE_FIRST.year, SCORE_LAST.year+1)),
        'quarter': tuple(str(value) for value in pd.period_range(SCORE_FIRST, SCORE_LAST, freq='Q')),
        'weekday_weekend': ('weekday', 'weekend'),
        'nationwide_holiday': (False, True),
        'canonical_hours': (23, 24, 25),
        'stress_2021_2022': (False, True),
        'negative_price_risk': (0, 1),
        'high_volatility': (0, 1),
    }
    cells = [('overall', 'overall')]
    cells.extend((name, value) for name in DAY_GROUPS for value in domains[name])
    if include_local_hours:
        cells.extend(('local_hour', hour) for hour in range(24))
    return tuple(cells)


def summaries(daily):
    """Raw equal-date losses beside ratios using exactly the same paired dates."""
    rows = []
    groups = registered_breakdown_cells()
    for mask in MASKS:
        for stream in sorted(daily.stream.unique()):
            own = daily[daily.stream.eq(stream)]
            for name, value in groups:
                cell = own[own[mask] & (True if name == 'overall' else own[name].eq(value))]
                for metric in ('MAE', 'WIS'):
                    pair, counts = paired_losses(daily, stream, 'nominal/B0', metric=metric, mask=mask,
                                                group=None if name == 'overall' else (name, value))
                    a, b = pair.a.mean(), pair.b.mean()
                    denominator_ok = np.isfinite(b) and b != 0
                    raw = float(cell[metric].mean())
                    reason = ('no_calendar_support' if not counts['calendar_dates'] else
                              'no_available_daily_losses' if not np.isfinite(raw) else None)
                    rows.append({'mask': mask, 'stream': stream, 'metric': metric, 'group': name,
                        'value': value, 'raw_mean_daily_loss': raw,
                        'raw_available_dates': int(cell.available.sum()), 'paired_policy_loss': float(a),
                        'paired_B0_loss': float(b), 'B0_ratio': float(a/b) if denominator_ok else np.nan,
                        'status': 'computable' if reason is None else 'unavailable', 'reason': reason,
                        'ratio_status': 'computable' if denominator_ok else 'unavailable',
                        'ratio_undefined_reason': None if denominator_ok else 'zero_or_nonfinite_B0_denominator', **counts})
    return pd.DataFrame(rows)


def fallback_summary(daily):
    rows = []
    for stream, cell in daily.groupby('stream', sort=True):
        eligible = cell[cell.eligible]
        count = int(eligible.fallback.sum())
        denominator = len(eligible)
        rows.append({'stream': stream, 'calendar_dates': len(cell), 'eligible_dates': denominator,
            'fallback_dates': count, 'failed_dates': int(eligible.failed.sum()),
            'fallback_fraction': count / denominator if denominator else np.nan,
            'exceeds_one_percent': bool(denominator and count / denominator > .01),
            'reason_counts': eligible[eligible.fallback | eligible.failed].reason.fillna('unspecified').value_counts().to_dict()})
    return pd.DataFrame(rows)


def secondary_summaries(summary):
    """Mean of exactly four season or five complete-year paired B0 ratios."""
    rows = []
    for (mask, stream, metric), cell in summary.groupby(['mask', 'stream', 'metric'], sort=True):
        for name, expected in (('season', ('DJF', 'MAM', 'JJA', 'SON')), ('year', tuple(range(2021, 2026)))):
            selected = cell[cell.group.eq(name) & cell.value.isin(expected)].set_index('value').reindex(expected)
            valid = np.isfinite(selected.B0_ratio).all()
            rows.append({'mask': mask, 'stream': stream, 'metric': metric,
                         'summary': 'mean_four_pooled_season_ratios' if name == 'season' else 'mean_five_complete_year_ratios',
                         'ratio': float(selected.B0_ratio.mean()) if valid else np.nan,
                         'status': 'computable' if valid else 'missing_or_undefined_registered_ratio',
                         'periods': list(expected), 'period_ratios': selected.B0_ratio.tolist(),
                         'period_paired_dates': selected.paired_dates.tolist(),
                         'period_B0_losses': selected.paired_B0_loss.tolist()})
    return pd.DataFrame(rows)


def local_hour_summaries(hourly, daily):
    """All 24 registered hours, retaining unavailable raw/paired estimates.

    Duplicate autumn hours are averaged within date, then dates equally.
    ``calendar_dates`` counts mask membership, not frozen eligible hour keys;
    ``n_hours`` and ``n_dates`` count only emitted scored observations.
    """
    h = (hourly.merge(daily[['stream', 'delivery_date', *MASKS]], on=['stream', 'delivery_date'], validate='many_to_one')
         if not hourly.empty else pd.DataFrame(columns=['stream', 'local_hour', 'delivery_date', 'MAE', 'WIS', *MASKS])
         .astype({'MAE': float, 'WIS': float}))
    rows = []
    for mask in MASKS:
        selected = h[h[mask].astype(bool)]
        by_date = selected.groupby(['stream', 'local_hour', 'delivery_date'], sort=True)[['MAE', 'WIS']].mean()
        support = by_date.index.droplevel(2)
        hour_counts = selected.groupby(['stream', 'local_hour']).size()
        for stream in sorted(daily.stream.unique()):
            calendar_dates = int(daily.loc[daily.stream.eq(stream), mask].sum())
            for hour in range(24):
                own = by_date.loc[(stream, hour)] if (stream, hour) in support else by_date.iloc[:0].droplevel([0, 1])
                baseline = (by_date.loc[('nominal/B0', hour)] if ('nominal/B0', hour) in support
                            else own.iloc[:0])
                for metric in ('MAE', 'WIS'):
                    pair = pd.concat({'a': own[metric], 'b': baseline[metric]}, axis=1).dropna()
                    denominator = pair.b.mean()
                    denominator_ok = np.isfinite(denominator) and denominator != 0
                    reason = ('no_calendar_support' if not calendar_dates else
                              'no_emitted_scored_hours' if own.empty else None)
                    rows.append({'mask': mask, 'stream': stream, 'local_hour': hour, 'metric': metric,
                        'raw_mean_daily_loss': float(own[metric].mean()), 'raw_available_dates': len(own),
                        'n_dates': len(own), 'n_hours': int(hour_counts.get((stream, hour), 0)),
                        'calendar_dates': calendar_dates, 'status': 'computable' if reason is None else 'unavailable',
                        'reason': reason, 'ratio_status': 'computable' if denominator_ok else 'unavailable',
                        'ratio_undefined_reason': None if denominator_ok else 'zero_or_nonfinite_B0_denominator',
                        'paired_dates': len(pair), 'paired_B0_loss': float(denominator),
                        'B0_ratio': float(pair.a.mean()/denominator) if denominator_ok else np.nan})
    return pd.DataFrame(rows)


def score_released_frame(saved, historical_keys, population, truth, *, metadata, known_states=None):
    """Pure descriptive reference; never a CP-26 native forecast or H issuance.

    The original artifact omitted timestamps. Its per-date row order is bound
    only when its count and saved truth agree with independently reconstructed
    historical eligible keys. An ambiguous/missing mapping retains a missing
    reference date. Saved truth drift fails closed before any loss is computed.
    """
    first, last = date(2026, 6, 9), date(2026, 9, 6)
    saved = saved.copy()
    saved['delivery_date'] = _dates(saved.delivery_date)
    if not saved.delivery_date.between(first, last).all():
        raise ValueError('released reference contains rows outside original holdout')
    historical = _keys(historical_keys)
    if not historical.delivery_date.between(first, last).all():
        raise ValueError('historical reference keys must belong to original holdout')
    truth = _keys(truth)
    population.verify_keys(truth)
    required = ['y_true', *(f'final_{name}' for name in QUANTILES)]
    if not set(required).issubset(saved):
        raise ValueError('released reference lacks saved final seven-quantile columns')
    groups = calendar_groups(population.calendar, known_states).set_index('delivery_date')
    scored_by_day = {day: rows for day, rows in truth.groupby('delivery_date', sort=True)}
    historical_by_day = {day: rows for day, rows in historical.groupby('delivery_date', sort=True)}
    saved_by_day = {day: rows for day, rows in saved.groupby('delivery_date', sort=False)}
    daily, hourly, differences = [], [], []
    for day in population.calendar.delivery_date:
        target = scored_by_day.get(day, truth.iloc[:0])
        original_keys = historical_by_day.get(day, historical.iloc[:0])
        original = saved_by_day.get(day, saved.iloc[:0])
        requested = pd.DatetimeIndex(target.timestamp_utc)
        original_index = pd.DatetimeIndex(original_keys.timestamp_utc)
        positions = original_index.get_indexer(requested)
        in_holdout = first <= day <= last
        mapping_ok = len(original) == len(original_keys)
        has_timestamp = 'timestamp_utc' in original
        if has_timestamp:
            original = _keys(original)
            original_index = pd.DatetimeIndex(original.timestamp_utc)
            positions = original_index.get_indexer(requested)
            mapping_ok = True
        elif mapping_ok and len(original):
            if not np.array_equal(original.y_true.to_numpy(float), original_keys.y_true.to_numpy(float)):
                raise ValueError('stale released reference truth or ambiguous historical row order')
        reason = None if in_holdout else 'outside_original_released_holdout'
        available = bool(in_holdout and groups.at[day, 'eligible'] and len(requested) and mapping_ok and (positions >= 0).all())
        if in_holdout and not mapping_ok:
            reason = 'historical_row_to_key_mapping_unavailable'
        elif in_holdout and (positions < 0).any():
            reason = 'released_artifact_missing_frozen_CP25_keys'
        q = None
        if available:
            selected = original.iloc[positions]
            if not np.array_equal(selected.y_true.to_numpy(float), target.y_true.to_numpy(float)):
                raise ValueError('stale released reference truth differs from frozen CP-25 targets')
            q = selected[[f'final_{name}' for name in QUANTILES]].to_numpy(float)
            if not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
                available, reason = False, 'released_quantiles_invalid'
        row = {'stream': 'nominal/released-v1', 'route': 'external_reference', 'policy': 'released-v1',
               'delivery_date': day, 'eligible': bool(groups.at[day, 'eligible']), 'available': available,
               'scoring_hours': len(requested), 'MAE': np.nan, 'WIS': np.nan,
               'output_source': 'released_artifact' if available else 'missing_reference',
               'fallback': False, 'failed': bool(in_holdout and not available and groups.at[day, 'eligible']),
               'reason': reason, 'native_central_valid': None, 'interval_ready': None, 'error_eligible': False,
               'original_issue_utc': None, 'evidence_class': 'released_holdout_descriptive',
               **groups.loc[day].drop('eligible').to_dict()}
        if available:
            losses = hourly_losses(target.y_true.to_numpy(float), q)
            row.update(MAE=float(losses['MAE'].mean()), WIS=float(losses['WIS'].mean()))
            h = target[['timestamp_utc', 'delivery_date', 'y_true']].copy()
            h['stream'], h['local_hour'] = 'nominal/released-v1', h.timestamp_utc.dt.tz_convert('Europe/Berlin').dt.hour
            for name, values in losses.items():
                h[name] = values
            for j, name in enumerate(QUANTILES):
                h[name] = q[:, j]
            hourly.append(h)
        if in_holdout:
            differences.append({'delivery_date': day, 'CP25_requested_hours': len(requested),
                                'historical_eligible_hours': len(original_keys), 'saved_artifact_hours': len(original),
                                'mapping_resolved': mapping_ok,
                                'missing_requested_hours': int((positions < 0).sum()) if mapping_ok else None,
                                'unresolved_requested_hours': len(requested) if not mapping_ok else 0,
                                'historical_extra_hours': int((~original_index.isin(requested)).sum()),
                                'reference_available': available, 'reason': reason})
        daily.append(row)
    return {'daily': pd.DataFrame(daily).merge(population.masks, on='delivery_date', validate='one_to_one'),
            'hourly': pd.concat(hourly, ignore_index=True) if hourly else pd.DataFrame(),
            'key_differences': pd.DataFrame(differences),
            'reference_metadata': {**metadata, 'holdout': [str(first), str(last)],
                                   'original_issue_instant_status': 'not retained in saved artifact',
                                   'CP26_native_or_refit': False, 'H_or_learning_intake': False,
                                   'interpretation': 'released artifact on original holdout only; descriptive'}}


def score_released_reference(root, population, truth, *,
                             committed_artifact_sha256=RELEASED_REFERENCE_PINS['reports/cp2/holdout_predictions.parquet'],
                             committed_report_sha256=RELEASED_REFERENCE_PINS['reports/cp2/holdout_report.json'], known_states=None):
    """Guarded loader, bound by Lead-supplied original committed blob hashes.

    The Lead verifies those hashes against the reachable original Git blobs;
    this module runs no Git operation and never regenerates a released forecast.
    Only identity/cutoff metadata is retained from the historical report.
    """
    from cp25.calendar import load_snapshot
    from delu_forecast.features import BASE_FEATURES, build_base_features
    root = Path(root)
    relative = 'reports/cp2/holdout_predictions.parquet'
    report_relative = 'reports/cp2/holdout_report.json'
    if committed_artifact_sha256 != RELEASED_REFERENCE_PINS[relative] or committed_report_sha256 != RELEASED_REFERENCE_PINS[report_relative]:
        raise ValueError('released reference input identity differs from prospective freeze')
    for source in ('scripts/cp2_final_holdout.py', 'src/delu_forecast/experiment.py'):
        if _sha(root / source) != RELEASED_REFERENCE_PINS[source]:
            raise ValueError('released reference historical row-generation source identity mismatch')
    if not committed_artifact_sha256 or _sha(root / relative) != committed_artifact_sha256:
        raise ValueError('released artifact differs from original committed blob identity')
    if not committed_report_sha256 or _sha(root / report_relative) != committed_report_sha256:
        raise ValueError('released metadata differs from original committed blob identity')
    manifest_path = root / 'reports/development-inputs/manifest.json'
    if _sha(manifest_path) != CP25_MANIFEST_SHA256:
        raise ValueError('historical key reconstruction manifest mismatch')
    manifest = json.loads(manifest_path.read_text())
    if _sha(root / 'data/snapshot.parquet') != manifest['inputs_sha256']['data/snapshot.parquet']:
        raise ValueError('historical key reconstruction snapshot mismatch')
    first, last = date(2026, 6, 9), date(2026, 9, 6)
    saved = pd.read_parquet(root / relative, columns=['delivery_date', 'y_true', *(f'final_{name}' for name in QUANTILES)],
                            filters=[('delivery_date', '>=', first), ('delivery_date', '<=', last)])
    # Date guard runs before any target or quantile is used for scoring.
    saved['delivery_date'] = _dates(saved.delivery_date)
    frame = load_snapshot(root)
    index = pd.DatetimeIndex(frame.timestamp_utc).as_unit('ns')
    base = build_base_features(frame)
    historical_valid = np.isfinite(frame.price_eur_mwh.to_numpy(float)) & base[list(BASE_FEATURES)].notna().all(axis=1).to_numpy()
    holdout = frame.delivery_date.between(first, last).to_numpy() & historical_valid
    historical_keys = pd.DataFrame({'timestamp_utc': index[holdout], 'delivery_date': frame.delivery_date[holdout],
                                    'y_true': frame.price_eur_mwh.to_numpy(float)[holdout]})
    report = json.loads((root / report_relative).read_text())
    metadata = {key: report.get(key) for key in ('artifact_fingerprint_sha256', 'cutoffs', 'snapshot_sha256',
                                                'code_sha', 'selected_catalog', 'final_cqr_thresholds')}
    metadata.update(artifact_path=relative, artifact_sha256=committed_artifact_sha256,
                    report_path=report_relative, report_sha256=committed_report_sha256,
                    historical_key_rule='original eligible target and every landed base feature non-null, original UTC row order',
                    original_issue_convention='historical v1 source contract; individual issue instants absent from saved artifact',
                    original_committed_base='647d6eb81b2cf6e0cfaf4152f659d689ff973578',
                    original_source_sha256={source: RELEASED_REFERENCE_PINS[source]
                                            for source in ('scripts/cp2_final_holdout.py', 'src/delu_forecast/experiment.py')})
    return score_released_frame(saved, historical_keys, population, truth, metadata=metadata, known_states=known_states)
