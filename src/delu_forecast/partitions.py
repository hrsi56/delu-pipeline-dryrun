"""Pinned development folds and five complete-delivery-day tail partitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class Window:
    start: date
    end: date

    def dates(self) -> set[date]:
        return {self.start + timedelta(days=offset) for offset in range((self.end - self.start).days + 1)}


@dataclass(frozen=True)
class DevelopmentFold:
    name: str
    proper_training: Window
    calibration_embargo: Window
    calibration: Window
    evaluation_embargo: Window
    evaluation: Window


def _window_ending(end: date, days: int) -> Window:
    return Window(end=end, start=end - timedelta(days=days - 1))


def _development_fold(name: str, eval_start: date) -> DevelopmentFold:
    evaluation = Window(eval_start, eval_start + timedelta(days=89))
    evaluation_embargo = Window(eval_start - timedelta(days=1), eval_start - timedelta(days=1))
    calibration = _window_ending(eval_start - timedelta(days=2), 60)
    calibration_embargo = Window(eval_start - timedelta(days=62), eval_start - timedelta(days=62))
    proper_training = Window(date(2019, 1, 1), eval_start - timedelta(days=63))
    return DevelopmentFold(name, proper_training, calibration_embargo, calibration, evaluation_embargo, evaluation)


def pin_partition_spec(snapshot_cutoff: date) -> dict[str, object]:
    holdout = _window_ending(snapshot_cutoff, 90)
    embargo_b = Window(holdout.start - timedelta(days=1), holdout.start - timedelta(days=1))
    final_calibration = _window_ending(embargo_b.start - timedelta(days=1), 60)
    embargo_a = Window(final_calibration.start - timedelta(days=1), final_calibration.start - timedelta(days=1))
    fold_5_window = _window_ending(embargo_a.start - timedelta(days=1), 90)
    if fold_5_window.start < date(2025, 10, 1):
        raise ValueError("Fold 5 must be fully post-transition")

    folds = [
        _development_fold("fold_1", date(2020, 7, 1)),
        _development_fold("fold_2", date(2021, 4, 1)),
        _development_fold("fold_3", date(2022, 7, 1)),
        _development_fold("fold_4", date(2025, 5, 1)),
        _development_fold("fold_5", fold_5_window.start),
    ]
    if folds[-1].evaluation != fold_5_window:
        raise AssertionError("Fold 5 evaluation and tail partition diverged")
    tail = {
        "fold_5": fold_5_window,
        "embargo_a": embargo_a,
        "final_calibration": final_calibration,
        "embargo_b": embargo_b,
        "holdout": holdout,
    }
    values = list(tail.values())
    for left, right in zip(values, values[1:]):
        if left.end + timedelta(days=1) != right.start or left.dates() & right.dates():
            raise AssertionError("tail partitions must be disjoint and contiguous")
    return {
        "snapshot_cutoff": snapshot_cutoff,
        "tail_partitions": tail,
        "development_folds": folds,
        "eda_cutoff": fold_5_window.end,
        "diagnostic_windows": [fold.evaluation for fold in folds],
    }


def _serialize(value: object) -> object:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (Window, DevelopmentFold)):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


def serializable_partition_spec(snapshot_cutoff: date) -> dict[str, object]:
    return _serialize(pin_partition_spec(snapshot_cutoff))  # type: ignore[return-value]
