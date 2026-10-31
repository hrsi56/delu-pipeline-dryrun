"""One inference path for every CP-3 surface (§9.2).

The CLI, the marimo app and the static-page builder all call the functions in
this module. There is exactly one place where a forecast frame is assembled, so
a helper cannot quietly reintroduce a row-wise boundary on one surface while the
others stay correct -- the §5.2 delivery-day availability invariant is enforced
here, once, and asserted by `tests/test_18_showcase_gate_boundary.py` with a
positive control.

The champion loads from the image alongside the bundled snapshot. There is no
registry-first path, no weekly refresh and no live API pull during a user
session (§9.2).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .claims import REPO_ROOT
from .model import CHAMPION_INPUT_COLUMNS, ChampionModel
from .postprocess import QUANTILE_LABELS

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "champion"
DEFAULT_SNAPSHOT_PATH = REPO_ROOT / "data" / "snapshot.parquet"

#: Every champion feature is a bounded backward window: 720 canonical hourly
#: price observations (30 delivery days) and a D-7 calendar-day lag. 200 days is
#: a wide margin over both, and `test_18` proves a 200-day truncation reproduces
#: the whole-snapshot result bitwise on every holdout day.
HISTORY_DAYS = 200

#: Nominal level -> (lower label, upper label) for the quantile-level selector.
INTERVAL_LEVELS: dict[int, tuple[str, str]] = {
    50: ("p25", "p75"),
    80: ("p10", "p90"),
    95: ("p025", "p975"),
}


def load_champion(path: Path | str = DEFAULT_MODEL_PATH) -> ChampionModel:
    """Load the bundled `mlflow.pyfunc` champion and unwrap the python model."""
    import mlflow.pyfunc

    loaded = mlflow.pyfunc.load_model(str(path))
    return loaded.unwrap_python_model()


def load_snapshot(path: Path | str = DEFAULT_SNAPSHOT_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def delivery_days(snapshot: pd.DataFrame) -> list[date]:
    return sorted({value for value in snapshot["delivery_date"]})


def default_target_day(snapshot: pd.DataFrame) -> date:
    """The last delivery day the bundled snapshot can forecast.

    A genuine D+1 needs the delivery day's A65 load forecast, which the frozen
    snapshot only carries up to its own cutoff. The bundled demo therefore
    forecasts the last snapshot day from history ending at D-1 -- a historical
    out-of-sample replay, labelled as one, never a live forecast.
    """
    return max(delivery_days(snapshot))


def gate_feasible_frame(
    snapshot: pd.DataFrame,
    target_day: date,
    *,
    catalog: str = "base",
    history_days: int = HISTORY_DAYS,
    load_scale: float = 1.0,
) -> pd.DataFrame:
    """The frame a 12:00 CET D-1 forecast origin actually has.

    Three properties, all enforced here rather than trusted:

    1. **No row dated after `target_day`.** Later rows are dropped, not masked.
    2. **No delivery-day price.** Every `price_eur_mwh` on `target_day` is set to
       NaN, because at the gate those prices have not cleared. The champion's
       features do not consume them either way (§5.2 is satisfied by the feature
       construction); masking makes the frame honest rather than merely harmless,
       and `test_18` asserts the two give bitwise-identical output.
    3. **Only champion-admissible columns.** Post-gate A69 and same-day actual
       columns are dropped before the model ever sees the frame; the model's own
       runtime firewall then rejects them again if a caller forgets.

    `load_scale` multiplies the **target day's** A65 load forecast only -- the
    ceteris-paribus sensitivity probe of §9.2. It touches no other delivery day
    and no price-derived feature.
    """
    columns = list(CHAMPION_INPUT_COLUMNS[catalog])
    lower = target_day - timedelta(days=history_days)
    dates = pd.Index(snapshot["delivery_date"].to_numpy())
    window = (dates >= lower) & (dates <= target_day)
    frame = snapshot.loc[window, columns].copy().reset_index(drop=True)

    on_target = pd.Index(frame["delivery_date"].to_numpy()) == target_day
    frame.loc[on_target, "price_eur_mwh"] = np.nan
    if load_scale != 1.0:
        frame.loc[on_target, "load_forecast_mw"] = (
            frame.loc[on_target, "load_forecast_mw"] * float(load_scale)
        )
    return frame


def forecast_delivery_day(
    model: ChampionModel,
    snapshot: pd.DataFrame,
    target_day: date,
    *,
    load_scale: float = 1.0,
    history_days: int = HISTORY_DAYS,
) -> pd.DataFrame:
    """Nine calibrated, monotone quantiles for every local hour of `target_day`."""
    frame = gate_feasible_frame(
        snapshot,
        target_day,
        catalog=model.catalog,
        history_days=history_days,
        load_scale=load_scale,
    )
    stages = model.predict_stages(frame)
    on_target = pd.Index(frame["delivery_date"].to_numpy()) == target_day
    result = pd.DataFrame(stages["final"][on_target], columns=list(QUANTILE_LABELS))
    result.insert(0, "local_hour", frame.loc[on_target, "local_hour"].to_numpy())
    result.insert(0, "timestamp_utc", frame.loc[on_target, "timestamp_utc"].to_numpy())
    return result.reset_index(drop=True)


def actuals_for_day(snapshot: pd.DataFrame, target_day: date) -> pd.Series:
    """The cleared prices for `target_day` -- outcome, never model input."""
    rows = snapshot.loc[pd.Index(snapshot["delivery_date"].to_numpy()) == target_day]
    return pd.Series(
        rows["price_eur_mwh"].to_numpy(dtype="float64"),
        index=rows["local_hour"].to_numpy(),
        name="price_eur_mwh",
    )


def holdout_replay(path: Path | str = REPO_ROOT / "reports/cp2/holdout_predictions.parquet") -> pd.DataFrame:
    """The committed one-shot holdout predictions, for the labelled replay view."""
    return pd.read_parquet(path)


def quantile_fan(forecast: pd.DataFrame, level: int) -> pd.DataFrame:
    """Median plus the selected interval's two endpoints, per local hour."""
    if level not in INTERVAL_LEVELS:
        raise ValueError(f"level must be one of {sorted(INTERVAL_LEVELS)}, got {level!r}")
    low, high = INTERVAL_LEVELS[level]
    return pd.DataFrame(
        {
            "local_hour": forecast["local_hour"].to_numpy(),
            "lower": forecast[low].to_numpy(),
            "median": forecast["p50"].to_numpy(),
            "upper": forecast[high].to_numpy(),
        }
    )


def empirical_coverage(level: int, claims: dict[str, str] | Any) -> str:
    """The committed holdout empirical coverage for the selected nominal level."""
    getter = claims.get if hasattr(claims, "get") else claims.__getitem__
    return getter(f"holdout_coverage_{level}")


__all__ = [
    "DEFAULT_MODEL_PATH",
    "DEFAULT_SNAPSHOT_PATH",
    "HISTORY_DAYS",
    "INTERVAL_LEVELS",
    "actuals_for_day",
    "default_target_day",
    "delivery_days",
    "empirical_coverage",
    "forecast_delivery_day",
    "gate_feasible_frame",
    "holdout_replay",
    "load_champion",
    "load_snapshot",
    "quantile_fan",
]
