"""Row selection over the five pinned development folds and the tail partitions.

`data/partitions.json` was pinned at CP-1 and is frozen: this module reads it and
never re-derives it. Every mask is expressed over *delivery dates*, never over row
offsets, so a 23- or 25-hour day carries its real hour count (§5.1).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .partitions import DevelopmentFold, Window

PARTITION_PATH = Path("data/partitions.json")
TAIL_ORDER: tuple[str, ...] = ("fold_5", "embargo_a", "final_calibration", "embargo_b", "holdout")


def _window(payload: dict[str, str]) -> Window:
    return Window(date.fromisoformat(payload["start"]), date.fromisoformat(payload["end"]))


@dataclass(frozen=True)
class PartitionSpec:
    snapshot_cutoff: date
    eda_cutoff: date
    tail_partitions: dict[str, Window]
    development_folds: tuple[DevelopmentFold, ...]
    diagnostic_windows: tuple[Window, ...]

    @property
    def holdout(self) -> Window:
        return self.tail_partitions["holdout"]

    @property
    def final_calibration(self) -> Window:
        return self.tail_partitions["final_calibration"]

    @property
    def embargo_a(self) -> Window:
        return self.tail_partitions["embargo_a"]

    def fold(self, name: str) -> DevelopmentFold:
        for item in self.development_folds:
            if item.name == name:
                return item
        raise KeyError(name)


def load_partition_spec(path: Path | str = PARTITION_PATH) -> PartitionSpec:
    payload = json.loads(Path(path).read_text())
    folds = tuple(
        DevelopmentFold(
            name=item["name"],
            proper_training=_window(item["proper_training"]),
            calibration_embargo=_window(item["calibration_embargo"]),
            calibration=_window(item["calibration"]),
            evaluation_embargo=_window(item["evaluation_embargo"]),
            evaluation=_window(item["evaluation"]),
        )
        for item in payload["development_folds"]
    )
    return PartitionSpec(
        snapshot_cutoff=date.fromisoformat(payload["snapshot_cutoff"]),
        eda_cutoff=date.fromisoformat(payload["eda_cutoff"]),
        tail_partitions={key: _window(value) for key, value in payload["tail_partitions"].items()},
        development_folds=folds,
        diagnostic_windows=tuple(_window(item) for item in payload["diagnostic_windows"]),
    )


def delivery_dates(index: pd.DatetimeIndex | pd.Index, snapshot: pd.DataFrame | None = None) -> pd.Index:
    """Delivery date per row, taken from the snapshot's own committed column."""
    if snapshot is not None:
        return pd.Index(snapshot["delivery_date"].to_numpy(), name="delivery_date")
    from .ingest import BERLIN

    return pd.Index(pd.DatetimeIndex(index).tz_convert(BERLIN).date, name="delivery_date")


def window_mask(dates: pd.Index, window: Window) -> np.ndarray:
    values = np.asarray(dates)
    return (values >= window.start) & (values <= window.end)


def any_window_mask(dates: pd.Index, windows: Iterable[Window]) -> np.ndarray:
    result = np.zeros(len(dates), dtype=bool)
    for window in windows:
        result |= window_mask(dates, window)
    return result


def excluded_from_development(spec: PartitionSpec) -> tuple[Window, ...]:
    """Partitions that may never enter a development fold, EDA or diagnostic."""
    return (
        spec.tail_partitions["embargo_a"],
        spec.tail_partitions["final_calibration"],
        spec.tail_partitions["embargo_b"],
        spec.tail_partitions["holdout"],
    )


def final_fit_window(spec: PartitionSpec) -> Window:
    """Every delivery day strictly before Embargo A (§7.1 raw-model fit set)."""
    return Window(date(2019, 1, 1), spec.embargo_a.start - pd.Timedelta(days=1).to_pytimedelta())


__all__ = [
    "PARTITION_PATH",
    "PartitionSpec",
    "TAIL_ORDER",
    "any_window_mask",
    "delivery_dates",
    "excluded_from_development",
    "final_fit_window",
    "load_partition_spec",
    "window_mask",
]
