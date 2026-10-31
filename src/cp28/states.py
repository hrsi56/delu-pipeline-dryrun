"""The two issue-time-known price states and the volatility-unknown flag (protocol §5).

A price day is *fully known* at origin D when all its canonical hours are finite and its
documented publication (D−1 13:00 Europe/Berlin for delivery D, CP-25's tier-2 deadline) is
strictly before D's issue instant. No target-day price and no forecast error enters.

* Negative-price risk: 1 if any canonical hour of the latest 7 fully known days is negative;
  available prefix below 7; 0 with none.
* High volatility: the sample SD (ddof=1) of the latest 28 fully known daily means, strictly above
  the linear 90th percentile of the preceding 365 rolling-28-day SDs (the current SD excluded).
  All 28 means and all 365 reference values are required; otherwise high=0 and unknown=1.
Gaps are skipped as days that are not fully known; the represented span is recorded.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from cp25 import guard

from .reader import canonical_hours


def published(day: date) -> datetime:
    return datetime.combine(day - timedelta(days=1), time(13), tzinfo=guard.BERLIN).astimezone(guard.UTC)


def complete_days(prices: pd.Series) -> pd.DataFrame:
    """Every delivery day whose canonical hours are all present and finite, with its mean and negative flag."""
    local = prices.index.tz_convert('Europe/Berlin')
    rows = []
    for day, values in prices.groupby(local.date, sort=True):
        index = values.index
        if index.equals(canonical_hours(day)) and np.isfinite(values.to_numpy()).all():
            v = values.to_numpy(float)
            rows.append({'day': day, 'published': published(day), 'mean': float(v.mean()), 'negative': bool((v < 0).any())})
    return pd.DataFrame(rows)


def rolling_sd(means: np.ndarray) -> np.ndarray:
    """SD (ddof=1) of each trailing window of 28 daily means, two-pass; NaN before 28."""
    out = np.full(len(means), np.nan)
    if len(means) >= 28:
        windows = np.lib.stride_tricks.sliding_window_view(np.asarray(means, float), 28)
        out[27:] = windows.std(axis=1, ddof=1)
    return out


def issue_states(prices: pd.Series, origins) -> pd.DataFrame:
    known_days = complete_days(prices)
    days = list(known_days.day)
    published_at = list(known_days.published)
    means = known_days['mean'].to_numpy(float)
    negative = known_days.negative.to_numpy(bool)
    sd = rolling_sd(means)
    result, k = [], 0
    origins = list(origins)
    if origins != sorted(set(origins)):
        raise ValueError('state origins must be unique and chronological')
    if sorted(published_at) != published_at:
        raise ValueError('publication must be monotone in the delivery day')
    for origin in origins:
        guard.admit_delivery_day(origin)
        issue = guard.issue_time(origin)
        # Known days form a chronological prefix, so the boundary only moves forward.
        while k < len(days) and days[k] < origin and published_at[k] < issue:
            k += 1
        last7 = slice(max(0, k - 7), k)
        neg = bool(negative[last7].any()) if k else False
        current = sd[k - 1] if k >= 28 else np.nan
        previous = sd[:k - 1] if k >= 1 else sd[:0]
        previous = previous[np.isfinite(previous)][-365:]
        unknown = not (np.isfinite(current) and len(previous) == 365)
        threshold = float(np.quantile(previous, .90, method='linear')) if not unknown else np.nan
        result.append({'delivery_date': origin, 'issue_utc': issue.isoformat(),
                       'negative_price_risk': int(neg), 'negative_support_dates': k - last7.start,
                       'negative_oldest_day': days[last7.start] if k else None, 'negative_latest_day': days[k - 1] if k else None,
                       'high_volatility': int(not unknown and current > threshold), 'volatility_unknown': int(unknown),
                       'volatility': float(current), 'volatility_threshold': threshold,
                       'volatility_support_dates': min(k, 28), 'volatility_reference_values': int(len(previous)),
                       'volatility_oldest_day': days[max(0, k - 28)] if k else None,
                       'volatility_latest_day': days[k - 1] if k else None,
                       'known_days': k})
    return pd.DataFrame(result)


SEASONS = {12: 'DJF', 1: 'DJF', 2: 'DJF', 3: 'MAM', 4: 'MAM', 5: 'MAM', 6: 'JJA', 7: 'JJA', 8: 'JJA',
           9: 'SON', 10: 'SON', 11: 'SON'}
SEASON_ORDER = ('DJF', 'MAM', 'JJA', 'SON')


def season(day: date) -> str:
    return SEASONS[day.month]
