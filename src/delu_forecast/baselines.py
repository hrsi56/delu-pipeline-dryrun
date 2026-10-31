"""The three §7 baselines: similar-day naive, seasonal-naive-168h, and Ridge.

Both naive comparators resolve their source by *calendar day and local hour*, not
by a fixed UTC row offset. That is not cosmetic: on the 25-hour fall-back day a
fixed 24-row offset resolves to a price of the *same* delivery day, which is the
§5.2 availability violation CP-1 was refused landing for. An unavailable or
ambiguous source hour fails closed as null, exactly as the champion lags do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import calendar_day_lag
from .ingest import BERLIN

#: Tue-Fri take D-1; Mon/Sat/Sun take D-7 (Lago et al. 2021 EPF convention).
SIMILAR_DAY_LAG_BY_WEEKDAY: dict[int, int] = {0: 7, 1: 1, 2: 1, 3: 1, 4: 1, 5: 7, 6: 7}

RIDGE_ONE_HOT: tuple[str, ...] = ("local_hour", "day_of_week", "month")
RIDGE_ALPHA: float = 1.0


def _price_series(snapshot: pd.DataFrame) -> pd.Series:
    frame = snapshot.set_index("timestamp_utc") if "timestamp_utc" in snapshot.columns else snapshot
    index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
    return pd.Series(frame["price_eur_mwh"].to_numpy(dtype="float64"), index=index)


def similar_day_naive(snapshot: pd.DataFrame) -> pd.Series:
    """Same hour of D-1 for Tue-Fri; same hour of D-7 for Mon/Sat/Sun."""
    price = _price_series(snapshot)
    weekday = pd.Index(price.index.tz_convert(BERLIN).dayofweek)
    lag_1 = calendar_day_lag(price, 1).to_numpy()
    lag_7 = calendar_day_lag(price, 7).to_numpy()
    takes_previous_day = np.isin(weekday.to_numpy(), [key for key, value in SIMILAR_DAY_LAG_BY_WEEKDAY.items() if value == 1])
    return pd.Series(np.where(takes_previous_day, lag_1, lag_7), index=price.index, name="similar_day_naive")


def seasonal_naive_168h(snapshot: pd.DataFrame) -> pd.Series:
    """Same local hour one week earlier (D-7); the 168h week-over-week baseline."""
    price = _price_series(snapshot)
    return pd.Series(calendar_day_lag(price, 7).to_numpy(), index=price.index, name="seasonal_naive_168h")


def build_ridge(seed: int = 42) -> Pipeline:
    """Standardized Ridge with one-hot calendar terms.

    The calendar columns are cyclic integers; feeding them to a linear model raw
    would cripple the classical baseline on an encoding artifact rather than on
    its own merits, so hour/day-of-week/month are one-hot encoded and everything
    else is standardized. Ridge is deterministic; the seed is recorded for the
    experiment record, not consumed by the estimator.
    """
    del seed
    return Pipeline(
        [
            (
                "prepare",
                ColumnTransformer(
                    [("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(RIDGE_ONE_HOT))],
                    remainder=StandardScaler(),
                    verbose_feature_names_out=False,
                ),
            ),
            ("ridge", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


__all__ = [
    "RIDGE_ALPHA",
    "RIDGE_ONE_HOT",
    "SIMILAR_DAY_LAG_BY_WEEKDAY",
    "build_ridge",
    "seasonal_naive_168h",
    "similar_day_naive",
]
