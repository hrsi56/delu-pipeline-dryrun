"""SMARD fallback-primary ingestion with explicit market-series mappings.

SMARD exposes quarter-hour load/generation values as energy per interval. Four
quarters are therefore summed to an hourly MWh value (numerically equal to the
average MW over that hour). Quarter-hour prices are averaged. The differing
rules are deliberately carried by :class:`SeriesSpec`, never inferred from a
column name.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BERLIN = "Europe/Berlin"
SMARD_REGION = "DE-LU"
SMARD_ROOT = "https://www.smard.de/app/chart_data"
TRANSITION_LOCAL_DATE = date(2025, 10, 1)


class IncompleteQuarterHourBinError(ValueError):
    """Raised when an otherwise populated hour contains fewer than 4 quarters."""


@dataclass(frozen=True)
class SeriesSpec:
    filter_id: int
    column: str
    plan_series: str
    smard_label: str
    aggregation: Literal["mean", "sum"]
    output_unit: str
    role: Literal["target", "kft_assumption", "benchmark_only", "lag_input"]


SERIES_SPECS: tuple[SeriesSpec, ...] = (
    SeriesSpec(4169, "price_eur_mwh", "A44", "Day-ahead price", "mean", "EUR/MWh", "target"),
    SeriesSpec(411, "load_forecast_mw", "A65/A01", "Forecast consumption: total grid load", "sum", "MW-hour-average", "kft_assumption"),
    SeriesSpec(123, "wind_onshore_forecast_mw", "A69", "Forecast wind onshore", "sum", "MW-hour-average", "benchmark_only"),
    SeriesSpec(125, "solar_forecast_mw", "A69", "Forecast solar", "sum", "MW-hour-average", "benchmark_only"),
    SeriesSpec(3791, "wind_offshore_forecast_mw", "A69", "Forecast wind offshore", "sum", "MW-hour-average", "benchmark_only"),
    SeriesSpec(4067, "wind_onshore_actual_mw", "A75", "Actual wind onshore", "sum", "MW-hour-average", "lag_input"),
    SeriesSpec(4068, "solar_actual_mw", "A75", "Actual solar", "sum", "MW-hour-average", "lag_input"),
    SeriesSpec(1225, "wind_offshore_actual_mw", "A75", "Actual wind offshore", "sum", "MW-hour-average", "lag_input"),
)


def _session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=24, pool_maxsize=24))
    session.headers["User-Agent"] = "delu-price-forecast-cp1/0.1 (research; CC-BY snapshot)"
    return session


def stitch_chunks(chunks: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Stitch timestamp/value chunks, accepting only identical overlaps."""
    frames = [chunk[["timestamp_utc", "value"]].copy() for chunk in chunks if not chunk.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", "value"])
    joined = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc")
    conflicts = joined.groupby("timestamp_utc", dropna=False)["value"].nunique(dropna=False)
    if (conflicts > 1).any():
        examples = conflicts[conflicts > 1].index[:3].tolist()
        raise ValueError(f"conflicting values at chunk overlap: {examples}")
    return joined.drop_duplicates("timestamp_utc", keep="last").reset_index(drop=True)


def _fetch_bucket(filter_id: int, resolution: str, bucket_ms: int) -> pd.DataFrame:
    url = f"{SMARD_ROOT}/{filter_id}/{SMARD_REGION}/{filter_id}_{SMARD_REGION}_{resolution}_{bucket_ms}.json"
    with _session() as session:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        rows = response.json()["series"]
    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([row[0] for row in rows], unit="ms", utc=True),
            "value": pd.to_numeric([row[1] for row in rows], errors="coerce"),
        }
    )


def fetch_smard_series(
    spec: SeriesSpec,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    *,
    resolution: Literal["hour", "quarterhour"],
    workers: int = 12,
) -> pd.DataFrame:
    """Fetch and stitch all SMARD buckets intersecting ``[start_utc, end_utc)``."""
    index_url = f"{SMARD_ROOT}/{spec.filter_id}/{SMARD_REGION}/index_{resolution}.json"
    with _session() as session:
        response = session.get(index_url, timeout=30)
        response.raise_for_status()
        bucket_starts = response.json()["timestamps"]

    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    eight_days_ms = 8 * 24 * 60 * 60 * 1000
    needed = [value for value in bucket_starts if value < end_ms and value + eight_days_ms > start_ms]
    chunks: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_fetch_bucket, spec.filter_id, resolution, bucket): bucket for bucket in needed
        }
        for future in as_completed(futures):
            chunks.append(future.result())
    stitched = stitch_chunks(chunks)
    return stitched.loc[
        (stitched["timestamp_utc"] >= start_utc) & (stitched["timestamp_utc"] < end_utc)
    ].reset_index(drop=True)


def aggregate_quarterhours(
    raw: pd.DataFrame,
    spec: SeriesSpec,
    *,
    fail_on_partial: bool = True,
) -> pd.Series:
    """Aggregate complete UTC hours and never turn a partial bin into a value."""
    frame = raw.copy()
    frame["hour_utc"] = frame["timestamp_utc"].dt.floor("h")
    grouped = frame.groupby("hour_utc", sort=True)["value"]
    total_rows = grouped.size()
    non_null = grouped.count()
    partial = (non_null > 0) & ((non_null != 4) | (total_rows != 4))
    if fail_on_partial and partial.any():
        examples = [str(value) for value in partial[partial].index[:3]]
        raise IncompleteQuarterHourBinError(
            f"{spec.column}: incomplete quarter-hour bins at {examples}"
        )
    if spec.aggregation == "mean":
        hourly = grouped.mean()
    else:
        hourly = grouped.sum(min_count=4)
    hourly[(non_null != 4) | (total_rows != 4)] = np.nan
    hourly.name = spec.column
    hourly.index.name = "timestamp_utc"
    return hourly


def quarterhour_source_counts(raw: pd.DataFrame) -> pd.Series:
    """Return non-null source-point counts for each UTC hour."""
    frame = raw.copy()
    frame["hour_utc"] = frame["timestamp_utc"].dt.floor("h")
    result = frame.groupby("hour_utc", sort=True)["value"].count().astype("int8")
    result.index.name = "timestamp_utc"
    return result


def assemble_price_series(pre_transition_hourly: pd.DataFrame, post_transition_qh: pd.DataFrame) -> pd.Series:
    """Join the hourly-product and quarter-hour-product A44 eras."""
    price_spec = SERIES_SPECS[0]
    pre = pre_transition_hourly.set_index("timestamp_utc")["value"].rename(price_spec.column)
    post = aggregate_quarterhours(post_transition_qh, price_spec)
    boundary = pd.Timestamp(TRANSITION_LOCAL_DATE, tz=BERLIN).tz_convert("UTC")
    pre = pre.loc[pre.index < boundary]
    post = post.loc[post.index >= boundary]
    result = pd.concat([pre, post]).sort_index()
    if result.index.has_duplicates:
        raise ValueError("duplicate price timestamp at transition")
    if result.isna().any():
        raise ValueError("day-ahead price contains nulls")
    expected = pd.date_range(result.index.min(), result.index.max(), freq="h", tz="UTC")
    if not result.index.equals(expected):
        missing = expected.difference(result.index)[:3].tolist()
        raise ValueError(f"day-ahead price is not continuous; missing {missing}")
    return result


def build_snapshot(series: dict[str, pd.Series]) -> pd.DataFrame:
    """Outer-join mapped hourly series and add canonical UTC/local identities."""
    snapshot = pd.concat(series.values(), axis=1).sort_index()
    snapshot.index.name = "timestamp_utc"
    if snapshot.index.tz is None or str(snapshot.index.tz) != "UTC":
        raise ValueError("snapshot index must be timezone-aware UTC")
    local = snapshot.index.tz_convert(BERLIN)
    snapshot.insert(0, "delivery_date", pd.Index(local.date))
    snapshot.insert(1, "local_hour", local.hour.astype("int8"))
    snapshot.insert(2, "utc_offset_minutes", np.array([int(ts.utcoffset().total_seconds() // 60) for ts in local], dtype="int16"))
    snapshot["vre_forecast_mw"] = snapshot[
        ["wind_onshore_forecast_mw", "wind_offshore_forecast_mw", "solar_forecast_mw"]
    ].sum(axis=1, min_count=3)
    snapshot["vre_actual_mw"] = snapshot[
        ["wind_onshore_actual_mw", "wind_offshore_actual_mw", "solar_actual_mw"]
    ].sum(axis=1, min_count=3)
    if snapshot["price_eur_mwh"].isna().any():
        raise ValueError("snapshot has null day-ahead prices")
    return snapshot


def series_manifest() -> list[dict[str, object]]:
    return [asdict(spec) for spec in SERIES_SPECS]


def write_parquet(snapshot: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.reset_index().to_parquet(path, index=False, compression="zstd")
