"""Runtime schema firewall separating the strict champion from A69 benchmark."""

from __future__ import annotations

from collections.abc import Iterable

from .features import AUGMENTED_FEATURES, BASE_FEATURES

BENCHMARK_ADDITIONS: tuple[str, ...] = (
    "wind_onshore_forecast_mw",
    "wind_offshore_forecast_mw",
    "solar_forecast_mw",
    "vre_forecast_mw",
    "residual_load_fc",
    "dunkelflaute_flag",
)

SAME_DAY_ACTUAL_COLUMNS: frozenset[str] = frozenset(
    {
        "wind_onshore_actual_mw",
        "wind_offshore_actual_mw",
        "solar_actual_mw",
        "vre_actual_mw",
        "actual_load_mw",
    }
)


def catalog_columns(catalog: str) -> tuple[str, ...]:
    if catalog == "base":
        return BASE_FEATURES
    if catalog == "base_plus_residual_load_proxy":
        return AUGMENTED_FEATURES
    raise ValueError(f"unknown frozen catalog: {catalog}")


def validate_champion_runtime_schema(columns: Iterable[str], catalog: str) -> None:
    supplied = tuple(columns)
    expected = catalog_columns(catalog)
    forbidden = set(supplied) & (set(BENCHMARK_ADDITIONS) | set(SAME_DAY_ACTUAL_COLUMNS))
    if forbidden:
        raise ValueError(f"post-gate or same-day actual columns forbidden for champion: {sorted(forbidden)}")
    if supplied != expected:
        missing = sorted(set(expected) - set(supplied))
        extra = sorted(set(supplied) - set(expected))
        raise ValueError(f"champion schema mismatch; missing={missing}, extra={extra}")


def validate_benchmark_runtime_schema(columns: Iterable[str], selected_catalog: str) -> None:
    supplied = tuple(columns)
    expected = catalog_columns(selected_catalog) + BENCHMARK_ADDITIONS
    if supplied != expected:
        missing = sorted(set(expected) - set(supplied))
        extra = sorted(set(supplied) - set(expected))
        raise ValueError(f"benchmark schema mismatch; missing={missing}, extra={extra}")


def validate_historical_actual_input(column: str, latest_delivery_lag_days: int) -> None:
    """Admit A75 only when its newest observation is no later than D-2."""
    if column not in SAME_DAY_ACTUAL_COLUMNS:
        raise ValueError(f"{column} is not a recognized actual-value source column")
    if column == "actual_load_mw":
        raise ValueError("actual load is not an authorized champion input")
    if latest_delivery_lag_days > -2:
        raise ValueError("actual generation must be bounded at the close of D-2")
