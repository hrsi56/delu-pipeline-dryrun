"""A69 post-gate columns (§7.2) and the Dunkelflaute evaluation stratum (§8.3).

Everything here is built from the *delivery-day* A69 forecast, which is published
after the 12:00 gate. It therefore never reaches the champion: it exists for the
one-number post-gate benchmark and for stratifying errors after the fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import BENCHMARK_ADDITIONS

#: A delivery day is flagged when its day-ahead-forecast VRE output is below this
#: share of its day-ahead load forecast. Pinned as a round constant before any
#: error analysis, so it cannot be tuned against a stratum's result, and constant
#: across folds and arms so it carries no cross-fold information.
DUNKELFLAUTE_VRE_SHARE_THRESHOLD: float = 0.20


def dunkelflaute_days(snapshot: pd.DataFrame) -> pd.Series:
    """Day-level scarcity flag: mean A69 VRE forecast / mean A65 load forecast."""
    daily = snapshot.groupby("delivery_date").agg(
        vre=("vre_forecast_mw", "mean"), load=("load_forecast_mw", "mean")
    )
    share = daily["vre"] / daily["load"]
    return (share < DUNKELFLAUTE_VRE_SHARE_THRESHOLD).astype("int8").rename("dunkelflaute_flag")


def build_benchmark_columns(snapshot: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """The six A69-derived columns the §7.2 augmented arm adds, and nothing else."""
    flags = dunkelflaute_days(snapshot)
    frame = pd.DataFrame(index=index)
    for column in ("wind_onshore_forecast_mw", "wind_offshore_forecast_mw", "solar_forecast_mw", "vre_forecast_mw"):
        frame[column] = snapshot[column].to_numpy(dtype="float64")
    frame["residual_load_fc"] = (
        snapshot["load_forecast_mw"].to_numpy(dtype="float64") - snapshot["vre_forecast_mw"].to_numpy(dtype="float64")
    )
    frame["dunkelflaute_flag"] = (
        pd.Index(snapshot["delivery_date"].to_numpy()).map(flags).to_numpy(dtype="float64")
    )
    missing = [column for column in BENCHMARK_ADDITIONS if column not in frame.columns]
    if missing:
        raise AssertionError(f"benchmark arm must add exactly {list(BENCHMARK_ADDITIONS)}; missing {missing}")
    return frame.loc[:, list(BENCHMARK_ADDITIONS)]


def negative_price_mask(prices: np.ndarray) -> np.ndarray:
    return np.asarray(prices, dtype="float64") < 0.0


__all__ = [
    "DUNKELFLAUTE_VRE_SHARE_THRESHOLD",
    "build_benchmark_columns",
    "dunkelflaute_days",
    "negative_price_mask",
]
