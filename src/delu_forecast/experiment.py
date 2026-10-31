"""Fold-running machinery shared by every CP-2 driver script.

One row set, defined once, used by every arm. `eligible_rows` depends only on the
target and the `base` catalog -- never on the augmented or A69 columns -- so the
two-arm comparison (§4.1) and the post-gate benchmark (§7.2) run on *literally the
same rows*, not on comparable ones. Extra columns carry NaN where undefined and
LightGBM handles them natively; that keeps the comparison symmetric instead of
deleting rows from the arm that does not need them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import build_ridge, seasonal_naive_168h, similar_day_naive
from .benchmark import build_benchmark_columns
from .conformal import fit_cqr_thresholds, thresholds_to_json
from .features import build_feature_catalog
from .folds import PartitionSpec, excluded_from_development, window_mask
from .metrics import QUANTILES, crossing_violations, interval_coverage, mae, pooled_mean_pinball
from .model import LGBM_PARAMS, SEED, fit_quantile_heads, predict_raw_heads
from .partitions import DevelopmentFold, Window
from .postprocess import QUANTILE_LABELS, apply_cqr_thresholds, isotonic_last
from .schema import BASE_FEATURES, BENCHMARK_ADDITIONS, catalog_columns

SNAPSHOT_PATH = Path("data/snapshot.parquet")

CATALOG_BASE = "base"
CATALOG_AUGMENTED = "base_plus_residual_load_proxy"


@dataclass(frozen=True)
class Inputs:
    snapshot: pd.DataFrame
    spec: PartitionSpec
    index: pd.DatetimeIndex
    delivery_dates: pd.Index
    target: np.ndarray
    wide_features: pd.DataFrame
    eligible: np.ndarray

    def arm_columns(self, arm: str) -> list[str]:
        if arm.endswith("+a69"):
            return list(catalog_columns(arm.removesuffix("+a69"))) + list(BENCHMARK_ADDITIONS)
        return list(catalog_columns(arm))

    def matrix(self, arm: str, mask: np.ndarray) -> pd.DataFrame:
        return self.wide_features.loc[mask, self.arm_columns(arm)]


def load_inputs(snapshot_path: Path | str = SNAPSHOT_PATH) -> Inputs:
    from .folds import load_partition_spec

    snapshot = pd.read_parquet(snapshot_path)
    spec = load_partition_spec()
    index = pd.DatetimeIndex(snapshot["timestamp_utc"]).tz_convert("UTC")
    delivery_dates = pd.Index(snapshot["delivery_date"].to_numpy(), name="delivery_date")

    augmented = build_feature_catalog(snapshot, CATALOG_AUGMENTED)
    augmented.index = index
    benchmark = build_benchmark_columns(snapshot, index)
    wide = pd.concat([augmented, benchmark], axis=1)

    target = snapshot["price_eur_mwh"].to_numpy(dtype="float64")
    eligible = np.isfinite(target) & wide.loc[:, list(BASE_FEATURES)].notna().all(axis=1).to_numpy()
    return Inputs(snapshot, spec, index, delivery_dates, target, wide, eligible)


def mask_for(inputs: Inputs, window: Window) -> np.ndarray:
    return window_mask(inputs.delivery_dates, window) & inputs.eligible


def assert_no_reserved_rows(inputs: Inputs, mask: np.ndarray, label: str) -> None:
    """Calibration/holdout rows enter no development fold and no diagnostic (§5.1)."""
    for window in excluded_from_development(inputs.spec):
        overlap = int(np.count_nonzero(mask & window_mask(inputs.delivery_dates, window)))
        if overlap:
            raise AssertionError(f"{label} contains {overlap} rows from a reserved tail partition {window}")


@dataclass
class FoldPredictions:
    fold: str
    arm: str
    delivery_dates: pd.Index
    y_true: np.ndarray
    raw: np.ndarray
    post_cqr: np.ndarray
    final: np.ndarray
    thresholds: dict[str, float]
    n_train: int
    n_calibration: int
    calibration_dates: pd.Index | None = None
    calibration_y: np.ndarray | None = None
    calibration_raw: np.ndarray | None = None
    metrics: dict[str, float] = field(default_factory=dict)


def run_fold_arm(inputs: Inputs, fold: DevelopmentFold, arm: str, *, seed: int = SEED) -> FoldPredictions:
    """Fit on proper-training, calibrate on the fold's embargoed slice, predict eval."""
    train = mask_for(inputs, fold.proper_training)
    calibration = mask_for(inputs, fold.calibration)
    evaluation = mask_for(inputs, fold.evaluation)
    for name, mask in (("train", train), ("calibration", calibration), ("evaluation", evaluation)):
        assert_no_reserved_rows(inputs, mask, f"{fold.name}:{arm}:{name}")
    if (train & calibration).any() or (train & evaluation).any() or (calibration & evaluation).any():
        raise AssertionError(f"{fold.name}: fold components overlap")

    heads = fit_quantile_heads(inputs.matrix(arm, train), inputs.target[train], seed=seed)
    raw_calibration = predict_raw_heads(heads, inputs.matrix(arm, calibration))
    thresholds = fit_cqr_thresholds(raw_calibration, inputs.target[calibration])

    raw_eval = predict_raw_heads(heads, inputs.matrix(arm, evaluation))
    post_cqr = apply_cqr_thresholds(raw_eval, thresholds)
    final = isotonic_last(post_cqr)
    y_true = inputs.target[evaluation]

    result = FoldPredictions(
        fold=fold.name,
        arm=arm,
        delivery_dates=inputs.delivery_dates[evaluation],
        y_true=y_true,
        raw=raw_eval,
        post_cqr=post_cqr,
        final=final,
        thresholds=thresholds_to_json(thresholds),
        n_train=int(train.sum()),
        n_calibration=int(calibration.sum()),
        calibration_dates=inputs.delivery_dates[calibration],
        calibration_y=inputs.target[calibration],
        calibration_raw=raw_calibration,
    )
    result.metrics = {
        "n_eval": float(evaluation.sum()),
        "mae_raw_p50": mae(y_true, raw_eval[:, 4]),
        "mae_final_p50": mae(y_true, final[:, 4]),
        "pinball_raw": pooled_mean_pinball(y_true, raw_eval),
        "pinball_post_cqr": pooled_mean_pinball(y_true, post_cqr),
        "pinball_final": pooled_mean_pinball(y_true, final),
        "crossings_raw": float(crossing_violations(raw_eval)),
        "crossings_post_cqr": float(crossing_violations(post_cqr)),
        "crossings_final": float(crossing_violations(final)),
    }
    for stage, matrix in (("raw", raw_eval), ("post_cqr", post_cqr), ("final", final)):
        for level, value in interval_coverage(y_true, matrix).items():
            result.metrics[f"coverage_{stage}_{level}"] = value
    return result


def run_arm(inputs: Inputs, arm: str, *, seed: int = SEED) -> list[FoldPredictions]:
    return [run_fold_arm(inputs, fold, arm, seed=seed) for fold in inputs.spec.development_folds]


def pooled_raw_pinball(results: list[FoldPredictions]) -> float:
    """L_c = sum_f sum_i sum_tau rho / (9 * sum_f n_f) -- observation-weighted (§4.1)."""
    total = 0.0
    count = 0
    for item in results:
        from .metrics import pinball_matrix

        losses = pinball_matrix(item.y_true, item.raw)
        total += float(losses.sum())
        count += int(losses.size)
    return total / count


def stack(results: list[FoldPredictions], attribute: str) -> np.ndarray:
    return np.concatenate([getattr(item, attribute) for item in results], axis=0)


def stacked_frame(results: list[FoldPredictions]) -> pd.DataFrame:
    """Long-format per-row predictions for downstream diagnostics."""
    frames = []
    for item in results:
        frame = pd.DataFrame({"fold": item.fold, "arm": item.arm, "delivery_date": item.delivery_dates})
        frame["y_true"] = item.y_true
        for stage, matrix in (("raw", item.raw), ("cqr", item.post_cqr), ("final", item.final)):
            for position, label in enumerate(QUANTILE_LABELS):
                frame[f"{stage}_{label}"] = matrix[:, position]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def calibration_frame(results: list[FoldPredictions]) -> pd.DataFrame:
    """Persist each fold's raw calibration-set predictions and targets.

    §6.2 requires the Integration Critic to recompute the four real-fold CQR
    thresholds independently, which is only possible if the inputs to that
    computation are on disk. Storing the thresholds alone would ask the reviewer
    to take them on trust.
    """
    frames = []
    for item in results:
        if item.calibration_raw is None:
            continue
        frame = pd.DataFrame({"fold": item.fold, "arm": item.arm, "delivery_date": item.calibration_dates})
        frame["y_true"] = item.calibration_y
        for position, label in enumerate(QUANTILE_LABELS):
            frame[f"raw_{label}"] = item.calibration_raw[:, position]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def baseline_predictions(inputs: Inputs, fold: DevelopmentFold) -> dict[str, np.ndarray]:
    """Similar-day naive, seasonal-naive-168h and Ridge on this fold's eval rows."""
    evaluation = mask_for(inputs, fold.evaluation)
    train = mask_for(inputs, fold.proper_training)
    assert_no_reserved_rows(inputs, evaluation, f"{fold.name}:baseline-eval")
    assert_no_reserved_rows(inputs, train, f"{fold.name}:baseline-train")

    similar = similar_day_naive(inputs.snapshot).to_numpy()[evaluation]
    seasonal = seasonal_naive_168h(inputs.snapshot).to_numpy()[evaluation]

    ridge = build_ridge(seed=SEED)
    ridge.fit(inputs.matrix(CATALOG_BASE, train), inputs.target[train])
    ridge_prediction = ridge.predict(inputs.matrix(CATALOG_BASE, evaluation))
    return {
        "similar_day_naive": similar,
        "seasonal_naive_168h": seasonal,
        "ridge": np.asarray(ridge_prediction, dtype="float64"),
    }


def experiment_params(arm: str, inputs: Inputs, extra: dict[str, object] | None = None) -> dict[str, object]:
    from .tracking import code_sha, snapshot_hash

    payload: dict[str, object] = {
        "arm": arm,
        "catalog": arm,
        "feature_list": ",".join(inputs.arm_columns(arm)),
        "n_features": len(inputs.arm_columns(arm)),
        "seed": SEED,
        "quantiles": ",".join(str(value) for value in QUANTILES),
        "snapshot_sha256": snapshot_hash(),
        "code_sha": code_sha(),
        "fold_spec": ";".join(
            f"{fold.name}:{fold.evaluation.start}..{fold.evaluation.end}" for fold in inputs.spec.development_folds
        ),
        "hyperparameters": ",".join(f"{key}={value}" for key, value in sorted(LGBM_PARAMS.items())),
        "tuning_budget": "zero -- one frozen configuration, identical across every arm",
    }
    payload.update(extra or {})
    return payload


def first_date(dates: pd.Index) -> date:
    return min(dates)


__all__ = [
    "CATALOG_AUGMENTED",
    "CATALOG_BASE",
    "FoldPredictions",
    "Inputs",
    "assert_no_reserved_rows",
    "baseline_predictions",
    "calibration_frame",
    "experiment_params",
    "load_inputs",
    "mask_for",
    "pooled_raw_pinball",
    "run_arm",
    "run_fold_arm",
    "stack",
    "stacked_frame",
]
