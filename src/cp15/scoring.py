"""Frozen CP-15 scores: final emitted quantiles, EUR/MWh, no row deletion.

``evaluate`` accepts all nine arms on the original eligible evaluation rows.
An optional ``expected_keys`` frame independently binds that caller contract.
Bootstrap intervals are exploratory after selection; peak is descriptive only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


POLICIES = ("B0", "B1", "B2", "B3", "A1", "A2", "A3", "A4", "A5")
BASELINES = POLICIES[:4]
CANDIDATES = POLICIES[4:]
FOLDS = tuple(f"fold_{i}" for i in range(1, 6))
QUANTILES = ("p025", "p10", "p25", "p50", "p75", "p90", "p975")
LEVELS = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
INTERVALS = ((50, 0.5, "p25", "p75"), (80, 0.2, "p10", "p90"),
             (95, 0.05, "p025", "p975"))
KEYS = ["fold", "timestamp_utc"]
BOOTSTRAP_SEED = 15042
BOOTSTRAP_REPLICATES = 2000
BLOCK_DAYS = 7
FOLD_WINDOWS = {
    "fold_1": ("2020-07-01", "2020-09-28"),
    "fold_2": ("2021-04-01", "2021-06-29"),
    "fold_3": ("2022-07-01", "2022-09-28"),
    "fold_4": ("2025-05-01", "2025-07-29"),
    "fold_5": ("2026-01-08", "2026-04-07"),
}


class UndefinedComparisonError(ValueError):
    """B0 has a zero denominator: a protocol correction is required."""


def _dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="raise")
    if parsed.dt.tz is not None or not parsed.eq(parsed.dt.normalize()).all():
        raise ValueError("delivery_date must be a timezone-naive calendar date")
    return parsed


def validate_predictions(predictions: pd.DataFrame,
                         expected_keys: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a canonical copy, refusing incomplete/crossed/mismatched rows."""
    required = {*KEYS, "policy", "delivery_date", "y_true", "central", "scale", *QUANTILES}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    frame = predictions.copy()
    if frame[list(required)].isna().any().any():
        raise ValueError("missing predictions or keys; rows must never be dropped")
    if set(frame.policy) != set(POLICIES) or set(frame.fold) != set(FOLDS):
        raise ValueError("exactly nine policies and all five folds are required")
    timestamps = pd.to_datetime(frame.timestamp_utc, errors="raise")
    if timestamps.dt.tz is None:
        raise ValueError("timestamp_utc must be timezone-aware")
    frame["timestamp_utc"] = timestamps.dt.tz_convert("UTC")
    if not frame.timestamp_utc.eq(frame.timestamp_utc.dt.floor("h")).all():
        raise ValueError("evaluation timestamps must be canonical whole hours")
    frame["delivery_date"] = _dates(frame.delivery_date)
    local_date = frame.timestamp_utc.dt.tz_convert("Europe/Berlin").dt.tz_localize(None).dt.normalize()
    if not local_date.eq(frame.delivery_date).all():
        raise ValueError("delivery_date disagrees with Europe/Berlin timestamp")
    if frame.duplicated(["policy", "timestamp_utc"]).any():
        raise ValueError("duplicate evaluation target for a policy")
    numeric = ["y_true", "central", "scale", *QUANTILES]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if not np.isfinite(frame[numeric].to_numpy(dtype=float)).all():
        raise ValueError("nonfinite predictions or target")
    if (frame.scale <= 0).any():
        raise ValueError("scale must be positive")
    if (np.diff(frame[list(QUANTILES)].to_numpy(), axis=1) < 0).any():
        raise ValueError("quantile crossings are forbidden")
    reference = frame.loc[frame.policy.eq("B0")].set_index(KEYS).sort_index()
    for policy in POLICIES:
        arm = frame.loc[frame.policy.eq(policy)].set_index(KEYS).sort_index()
        if not arm.index.equals(reference.index):
            raise ValueError(f"unmatched evaluation row set: {policy}")
        if not arm[["delivery_date", "y_true"]].equals(reference[["delivery_date", "y_true"]]):
            raise ValueError(f"inconsistent truth or delivery date: {policy}")
    if expected_keys is not None:
        expected = expected_keys.copy()
        if not set(KEYS).issubset(expected.columns):
            raise ValueError("expected_keys needs fold and timestamp_utc")
        if expected[KEYS].isna().any().any() or expected.duplicated(KEYS).any():
            raise ValueError("expected_keys contains missing or duplicate keys")
        expected["timestamp_utc"] = pd.to_datetime(expected.timestamp_utc, utc=True)
        if not expected.set_index(KEYS).sort_index().index.equals(reference.index):
            raise ValueError("predictions differ from original eligible evaluation keys")
        expected = expected.set_index(KEYS).sort_index()
        if "delivery_date" in expected:
            expected["delivery_date"] = _dates(expected.delivery_date)
            if not expected.delivery_date.equals(reference.delivery_date):
                raise ValueError("original eligible delivery dates disagree")
        if "y_true" in expected and not np.array_equal(expected.y_true.to_numpy(), reference.y_true.to_numpy()):
            raise ValueError("original eligible target values disagree")
    return frame.sort_values(["policy", *KEYS]).reset_index(drop=True)


def score_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute losses on already validated rows without changing their keys."""
    out = frame[["policy", *KEYS, "delivery_date"]].copy()
    truth = frame.y_true.to_numpy(dtype=float)
    forecast = frame.p50.to_numpy(dtype=float)
    error = forecast - truth
    out["absolute_error"] = np.abs(error)
    out["squared_error"] = error**2
    out["signed_error"] = error
    out["raw_central_absolute_error"] = np.abs(frame.central.to_numpy() - truth)
    out["centering_effect"] = out.absolute_error - out.raw_central_absolute_error
    day_keys = [frame.policy, frame.fold, frame.delivery_date]
    predicted_mean = frame.p50.groupby(day_keys).transform("mean").to_numpy()
    truth_mean = frame.y_true.groupby(day_keys).transform("mean").to_numpy()
    out["level_absolute_error"] = np.abs(predicted_mean - truth_mean)
    out["shape_absolute_error"] = np.abs((forecast - predicted_mean) - (truth - truth_mean))
    wis = 0.5 * np.abs(error)
    for coverage, alpha, lower_name, upper_name in INTERVALS:
        lower = frame[lower_name].to_numpy(dtype=float)
        upper = frame[upper_name].to_numpy(dtype=float)
        width = upper - lower
        out[f"width{coverage}"] = width
        out[f"hit{coverage}"] = (truth >= lower) & (truth <= upper)
        out[f"lower_miss{coverage}"] = truth < lower
        out[f"upper_miss{coverage}"] = truth > upper
        wis += (alpha / 2) * width + np.maximum(lower - truth, 0) + np.maximum(truth - upper, 0)
    out["WIS"] = wis / 3.5
    residual = truth[:, None] - frame[list(QUANTILES)].to_numpy(dtype=float)
    out["mean_pinball_7"] = np.maximum(LEVELS * residual, (LEVELS - 1) * residual).mean(axis=1)
    return out


def _summary(rows: pd.DataFrame) -> dict:
    n = len(rows)
    result = {
        "n_hours": n, "n_days": rows.delivery_date.nunique(),
        "first_delivery_date": rows.delivery_date.min(),
        "last_delivery_date": rows.delivery_date.max(),
        "MAE": rows.absolute_error.mean(), "RMSE": np.sqrt(rows.squared_error.mean()),
        "WIS": rows.WIS.mean(), "mean_pinball_7": rows.mean_pinball_7.mean(),
        "raw_central_MAE": rows.raw_central_absolute_error.mean(),
        "centering_effect": rows.centering_effect.mean(), "bias": rows.signed_error.mean(),
        "daily_mean_level_MAE": rows.groupby(["fold", "delivery_date"]).level_absolute_error.first().mean(),
        "within_day_shape_MAE": rows.shape_absolute_error.mean(),
        "missing_count": 0, "crossings": 0,
    }
    for coverage, *_ in INTERVALS:
        width = rows[f"width{coverage}"]
        hits = int(rows[f"hit{coverage}"].sum())
        result.update({
            f"coverage{coverage}": hits / n if n else np.nan,
            f"hit_count{coverage}": hits,
            f"lower_miss_count{coverage}": int(rows[f"lower_miss{coverage}"].sum()),
            f"upper_miss_count{coverage}": int(rows[f"upper_miss{coverage}"].sum()),
            f"mean_width{coverage}": width.mean(),
            f"median_width{coverage}": width.median(),
            f"p95_width{coverage}": width.quantile(0.95),
        })
    return result


def _summaries(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    records = []
    for key, group in rows.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        records.append(dict(zip(by, key)) | _summary(group))
    return pd.DataFrame(records)


def select_candidates(per_fold: pd.DataFrame, peak: pd.DataFrame,
                      *, integrity_verified: bool = True) -> dict:
    """Apply all six section-8 criteria; preserve actuals and comparator limits."""
    if per_fold.duplicated(["policy", "fold"]).any() or peak.duplicated("policy").any():
        raise ValueError("duplicate summary keys")
    expected = pd.MultiIndex.from_product([POLICIES, FOLDS], names=["policy", "fold"])
    actual = pd.MultiIndex.from_frame(per_fold[["policy", "fold"]])
    if len(actual) != len(expected) or set(actual) != set(expected) or set(peak.policy) != set(POLICIES):
        raise ValueError("selection requires all matched policy/fold and peak summaries")
    fold = per_fold.set_index(["policy", "fold"])
    peak_by_policy = peak.set_index("policy")
    ratios = {}
    for metric in ("MAE", "WIS"):
        reference = fold.loc["B0", metric].reindex(FOLDS)
        if not np.isfinite(reference).all() or (reference <= 0).any():
            raise UndefinedComparisonError(
                f"undefined {metric} comparison: B0 denominator is zero/nonfinite; "
                "protocol correction required before selection")
        ratios[metric] = {policy: float((fold.loc[policy, metric].reindex(FOLDS) / reference).mean())
                         for policy in POLICIES}
    if not np.isfinite(per_fold[["MAE", "WIS", "coverage95"]].to_numpy()).all() or not np.isfinite(
            peak[["MAE", "WIS", "coverage95"]].to_numpy()).all():
        raise ValueError("nonfinite selection summary; selection cannot omit missing scores")
    ranking = pd.DataFrame([{"policy": policy, "S_MAE": ratios["MAE"][policy],
                             "S_WIS": ratios["WIS"][policy], "table_order": i}
                            for i, policy in enumerate(CANDIDATES)])
    ranking = ranking.sort_values(["S_MAE", "S_WIS", "table_order"]).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    criteria = []

    def add(policy, criterion, metric, scope, actual, lower=None, upper=None):
        passed = bool(np.isfinite(actual) and (lower is None or actual >= lower)
                      and (upper is None or actual <= upper))
        criteria.append({"policy": policy, "criterion": criterion, "metric": metric,
                         "scope": scope, "actual": actual, "lower_limit": lower,
                         "upper_limit": upper, "passed": passed})

    for policy in CANDIDATES:
        for criterion, metric in ((1, "MAE"), (2, "WIS")):
            add(policy, criterion, f"S_{metric}", "equal_fold",
                ratios[metric][policy], upper=0.90 * min(ratios[metric][b] for b in BASELINES))
        for fold_name in FOLDS:
            add(policy, 3, "coverage95", fold_name, fold.loc[(policy, fold_name), "coverage95"],
                lower=0.90, upper=0.98)
        add(policy, 4, "coverage95", "peak", peak_by_policy.loc[policy, "coverage95"], lower=0.90)
        for metric in ("MAE", "WIS"):
            add(policy, 4, metric, "peak", peak_by_policy.loc[policy, metric],
                upper=min(peak_by_policy.loc[b, metric] for b in BASELINES))
            for fold_name in FOLDS:
                add(policy, 5, metric, fold_name, fold.loc[(policy, fold_name), metric],
                    upper=1.05 * min(fold.loc[(b, fold_name), metric] for b in ("B2", "B3")))
        add(policy, 6, "complete_finite_ordered", "all_eligible", int(integrity_verified), lower=1)
    criteria_frame = pd.DataFrame(criteria)
    failed = {policy: sorted(criteria_frame.loc[
        criteria_frame.policy.eq(policy) & ~criteria_frame.passed, "criterion"].unique().tolist())
        for policy in CANDIDATES}
    ranking["failed_criteria"] = ranking.policy.map(failed)
    ranking["qualified"] = ranking.policy.map(lambda p: not failed[p])
    ranking["product_feasibility"] = ranking.qualified.map({
        True: "DEMONSTRATED_ON_DEVELOPMENT", False: "NOT_DEMONSTRATED"})
    qualified = ranking.loc[ranking.qualified, "policy"].tolist()
    qualified_policy = qualified[0] if qualified else None
    return {"ranking": ranking, "criteria": criteria_frame,
            "relative_scores": pd.DataFrame([{"policy": p, "S_MAE": ratios["MAE"][p],
                                               "S_WIS": ratios["WIS"][p]} for p in POLICIES]),
            "selection": {"best_observed_policy": ranking.iloc[0].policy,
                          "qualified_policy": qualified_policy,
                          "product_feasibility": "DEMONSTRATED_ON_DEVELOPMENT" if qualified else "NOT_DEMONSTRATED",
                          "failed_criteria": failed}}


def bootstrap_daily(daily: pd.DataFrame) -> dict:
    """Shared seeded noncircular seven-day blocks on each full calendar grid."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    fold_samples = []
    fold_points = []
    offsets = np.arange(BLOCK_DAYS)
    for fold_name in FOLDS:
        part = daily.loc[daily.fold.eq(fold_name)]
        dates = sorted(part.delivery_date.unique())
        if len(dates) != 90 or not np.all(np.diff(np.array(dates, dtype="datetime64[D]")) == np.timedelta64(1, "D")):
            raise ValueError("bootstrap requires each full chronological 90-day calendar grid")
        n = len(dates)
        values = np.stack([part.pivot(index="delivery_date", columns="policy", values=metric)
                           .reindex(index=dates, columns=POLICIES).to_numpy()
                           for metric in ("MAE", "WIS")], axis=2)
        valid = np.isfinite(values).all(axis=(1, 2))
        if not np.array_equal(np.isfinite(values), np.broadcast_to(valid[:, None, None], values.shape)):
            raise ValueError("daily losses must have identically missing calendar days")
        starts = rng.integers(0, n - BLOCK_DAYS + 1,
                              size=(BOOTSTRAP_REPLICATES, int(np.ceil(n / BLOCK_DAYS))))
        indices = (starts[:, :, None] + offsets).reshape(BOOTSTRAP_REPLICATES, -1)[:, :n]
        counts = valid[indices].sum(axis=1)
        if (counts == 0).any():
            raise ValueError("bootstrap replicate has no eligible days; mean is undefined")
        samples = np.where(np.isfinite(values), values, 0)[indices].sum(axis=1) / counts[:, None, None]
        point = values[valid].mean(axis=0)
        fold_samples.append(samples)
        fold_points.append(point)
        rows.extend(_bootstrap_intervals(fold_name, samples, point))
    rows.extend(_bootstrap_intervals("equal_fold", np.mean(fold_samples, axis=0), np.mean(fold_points, axis=0)))
    return {"bootstrap": pd.DataFrame(rows), "bootstrap_metadata": {
        "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES, "block_days": BLOCK_DAYS,
        "confidence": 0.95, "method": "paired noncircular moving-block percentile",
        "pairing": "identical calendar indices across all nine policies and both losses",
        "daily_denominator": "represented eligible delivery days; zero-hour days excluded, not zero losses",
        "aggregate": "equal mean of five fold mean daily losses",
        "evidence_class": "exploratory_post_selection", "peak": "descriptive_only_small_effective_sample"}}


def _bootstrap_intervals(scope, samples, point):
    rows = []
    for ai, candidate in enumerate(CANDIDATES, start=4):
        for bi, baseline in enumerate(BASELINES):
            for mi, metric in enumerate(("MAE", "WIS")):
                difference = samples[:, ai, mi] - samples[:, bi, mi]
                lower, upper = np.quantile(difference, [0.025, 0.975], method="linear")
                rows.append({"scope": scope, "candidate": candidate, "baseline": baseline,
                             "metric": metric, "difference": point[ai, mi] - point[bi, mi],
                             "ci_lower": lower, "ci_upper": upper,
                             "replicates": BOOTSTRAP_REPLICATES})
    return rows


def evaluate(predictions: pd.DataFrame, *, expected_keys: pd.DataFrame | None = None,
             fold_windows: dict[str, tuple[str, str]] | None = None) -> dict[str, pd.DataFrame | dict]:
    """Score all nine arms; return diagnostic tables, criteria, selection and CIs.

    Defaults are the frozen data/partitions.json evaluation calendar windows.
    ``fold_windows`` allows explicit 90-day test windows, never inferred bounds.
    Resource/fit measurements are recorded by the driver, not inferred from scores.
    ``centering_effect`` is final-p50 MAE minus raw-central MAE (negative improves).
    """
    fold_windows = FOLD_WINDOWS if fold_windows is None else fold_windows
    if set(fold_windows) != set(FOLDS):
        raise ValueError("the five preregistered fold_windows are required")
    frame = validate_predictions(predictions, expected_keys)
    grids = {}
    for fold_name in FOLDS:
        start, end = map(pd.Timestamp, fold_windows[fold_name])
        grid = pd.date_range(start, end, freq="D")
        if len(grid) != 90:
            raise ValueError("every preregistered fold must contain 90 calendar days")
        if not frame.loc[frame.fold.eq(fold_name), "delivery_date"].isin(grid).all():
            raise ValueError(f"evaluation date outside {fold_name} bounds")
        grids[fold_name] = grid
    hourly = score_hourly(frame)
    per_fold = _summaries(hourly, ["policy", "fold"])
    per_fold["window_start"] = per_fold.fold.map(lambda f: fold_windows[f][0])
    per_fold["window_end"] = per_fold.fold.map(lambda f: fold_windows[f][1])
    per_fold["calendar_days"] = 90
    pooled = _summaries(hourly, ["policy"])
    represented_daily = _summaries(hourly, ["policy", "fold", "delivery_date"])
    daily_parts = []
    for fold_name in FOLDS:
        grid_index = pd.MultiIndex.from_product([POLICIES, [fold_name], grids[fold_name]],
                                                names=["policy", "fold", "delivery_date"])
        part = represented_daily.set_index(["policy", "fold", "delivery_date"]).reindex(grid_index)
        for column in ["n_hours", "n_days", "missing_count", "crossings"] + [
                c for c in part if "_count" in c]:
            part[column] = part[column].fillna(0).astype(int)
        daily_parts.append(part.reset_index())
    daily = pd.concat(daily_parts, ignore_index=True)
    peak_rows = hourly.loc[hourly.delivery_date.between("2022-08-15", "2022-08-31")]
    if peak_rows.empty or set(peak_rows.fold) != {"fold_3"}:
        raise ValueError("matched August 15–31 peak must be represented in fold_3")
    peak = _summaries(peak_rows, ["policy"])
    peak["window_start"], peak["window_end"] = "2022-08-15", "2022-08-31"
    peak["calendar_days"] = 17
    peak["evidence_class"] = "descriptive_only_small_effective_sample"
    recovery_records = []
    for begin in (1, 8, 15, 22):
        start, end = f"2022-09-{begin:02}", f"2022-09-{begin + 6:02}"
        for policy in POLICIES:
            selected = hourly.loc[hourly.policy.eq(policy) & hourly.fold.eq("fold_3")
                                  & hourly.delivery_date.between(start, end)]
            recovery_records.append({"policy": policy, "fold": "fold_3", "window_start": start,
                                     "window_end": end, "calendar_days": 7, **_summary(selected)})
    result = {"hourly_losses": hourly, "per_fold": per_fold, "pooled": pooled, "daily": daily,
              "peak": peak, "recovery": pd.DataFrame(recovery_records)}
    result.update(select_candidates(per_fold, peak))
    result.update(bootstrap_daily(daily))
    result["scoring_metadata"] = {"point_forecast": "final_emitted_p50", "score_units": "EUR/MWh",
                                  "coverage_units": "fraction", "primary_weighting": "equal_fold",
                                  "pooled_weighting": "eligible_hour", "native_pinball": "preserved untouched",
                                  "eligible_keys_binding": "independent_expected_keys" if expected_keys is not None else "caller_contract"}
    return result
