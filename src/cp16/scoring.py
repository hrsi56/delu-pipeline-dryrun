"""Frozen CP-16 research scores; no fitting, selection search or product promotion.

Production ``evaluate`` binds independent expected keys and always uses the single
seed-15042, 2,000-replicate index set. Private helpers permit synthetic controls.
Runtime, fit and memory measurements belong to the driver resource ledger, not to
predictions; they must accompany these tables in the completed report.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from cp15.scoring import FOLDS, FOLD_WINDOWS, QUANTILES, _summary, score_hourly

POLICIES = ("B0", "B1", "B2", "B3", "A1", "V2-H", "V2-P")
CANDIDATES = ("V2-H", "V2-P")
BASELINES = POLICIES[:4]
KEYS = ["fold", "timestamp_utc"]
FOLD_COUNTS = dict(zip(FOLDS, (2160, 2159, 2112, 2160, 2156)))
BOOTSTRAP_SEED = 15042
BOOTSTRAP_REPLICATES = 2000
BLOCK_DAYS = 7
SUPPORT_DATES = 56
CONTRASTS = (("V2-H", "V2-P"), ("V2-H", "B2"), ("V2-P", "B2"))


def validate_predictions(predictions, expected_keys, *, production=True):
    """Refuse incomplete, mismatched, nonfinite or invalid output populations."""
    required = {*KEYS, "policy", "delivery_date", "y_true", "central", "scale", *QUANTILES}
    if not required.issubset(predictions):
        raise ValueError(f"missing prediction columns: {sorted(required - set(predictions))}")
    frame = predictions.copy(deep=True)
    if frame[list(required)].isna().any().any():
        raise ValueError("missing predictions or keys")
    if set(frame.policy) != set(POLICIES) or set(frame.fold) != set(FOLDS):
        raise ValueError("exactly seven policies and five folds required")
    for data in (frame, expected_keys):
        if not set(KEYS).issubset(data) or data[KEYS].isna().any().any():
            raise ValueError("independent expected_keys and prediction keys are required")
    timestamps = pd.to_datetime(frame.timestamp_utc, errors="raise")
    expected = expected_keys.copy(deep=True)
    expected_times = pd.to_datetime(expected.timestamp_utc, errors="raise")
    if timestamps.dt.tz is None or expected_times.dt.tz is None:
        raise ValueError("timestamp_utc must be timezone-aware")
    frame["timestamp_utc"] = timestamps.dt.tz_convert("UTC")
    expected["timestamp_utc"] = expected_times.dt.tz_convert("UTC")
    if not frame.timestamp_utc.eq(frame.timestamp_utc.dt.floor("h")).all():
        raise ValueError("timestamps must be whole hours")
    dates = pd.to_datetime(frame.delivery_date, errors="raise")
    local = frame.timestamp_utc.dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
    if dates.dt.tz is not None or not dates.eq(local.dt.normalize()).all():
        raise ValueError("delivery_date disagrees with local calendar date")
    frame["delivery_date"] = dates
    for fold, (start, end) in FOLD_WINDOWS.items():
        if not frame.loc[frame.fold.eq(fold), "delivery_date"].between(start, end).all():
            raise ValueError(f"date outside {fold} window")
    if frame.duplicated(["policy", "timestamp_utc"]).any() or expected.duplicated(KEYS).any():
        raise ValueError("duplicate target keys")
    numeric = ["y_true", "central", "scale", *QUANTILES]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise").astype(float)
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError("nonfinite predictions or truth")
    if frame.scale.le(0).any() or (np.diff(frame[list(QUANTILES)], axis=1) < 0).any():
        raise ValueError("nonpositive scale or crossed quantiles")
    expected = expected.set_index(KEYS).sort_index()
    reference = frame.loc[frame.policy.eq("B0")].set_index(KEYS).sort_index()
    if not expected.index.equals(reference.index):
        raise ValueError("predictions differ from independent expected keys")
    for column in ("delivery_date", "y_true"):
        if column in expected:
            values = pd.to_datetime(expected[column]) if column == "delivery_date" else expected[column]
            if not np.array_equal(values.to_numpy(), reference[column].to_numpy()):
                raise ValueError(f"expected {column} disagrees")
    for policy in POLICIES:
        part = frame.loc[frame.policy.eq(policy)].set_index(KEYS).sort_index()
        if not part.index.equals(reference.index):
            raise ValueError(f"unmatched keys: {policy}")
        if not part[["y_true", "delivery_date"]].equals(reference[["y_true", "delivery_date"]]):
            raise ValueError(f"unmatched truth or dates: {policy}")
    h = frame.loc[frame.policy.eq("V2-H")].set_index(KEYS).sort_index()
    p = frame.loc[frame.policy.eq("V2-P")].set_index(KEYS).sort_index()
    if not h[["central", "scale"]].equals(p[["central", "scale"]]):
        raise ValueError("H/P common central and scale parity failed")
    if production:
        counts = reference.reset_index().groupby("fold").size().to_dict()
        peak = reference.loc[reference.delivery_date.between("2022-08-15", "2022-08-31")]
        f3 = reference.xs("fold_3")
        if counts != FOLD_COUNTS or len(reference) != 10747:
            raise ValueError("production requires original 10,747 keys and exact fold counts")
        if len(peak) != 408 or peak.delivery_date.nunique() != 17 or f3.delivery_date.nunique() != 88:
            raise ValueError("production fold-3/peak dates and denominators disagree")
    return frame.sort_values(["policy", *KEYS]).reset_index(drop=True)


def _record(rows, **labels):
    result = _summary(rows)
    result["represented_dates"] = json.dumps(sorted(rows.delivery_date.dt.strftime("%Y-%m-%d").unique().tolist()))
    return labels | result


def _tables(hourly):
    metrics, diagnostics = [], []
    for policy in POLICIES:
        selected = hourly.loc[hourly.policy.eq(policy)]
        metrics.append(_record(selected, policy=policy, scope="pooled", fold="all"))
        for fold in FOLDS:
            part = selected.loc[selected.fold.eq(fold)]
            start, end = FOLD_WINDOWS[fold]
            metrics.append(_record(part, policy=policy, scope="per_fold", fold=fold,
                                   window_start=start, window_end=end, calendar_days=90))
            for scope, groups in (("hour", [(str(h), part.loc[part.local_hour.eq(h)]) for h in range(24)]),
                                  ("block", [(b, part.loc[part.local_block.eq(b)])
                                             for b in ("night", "solar", "shoulder")])):
                for label, group in groups:
                    enough = group.delivery_date.nunique() >= SUPPORT_DATES
                    diagnostics.append(_record(group, policy=policy, fold=fold, scope=scope,
                                               group=label, support_status="eligible" if enough else "support_limited",
                                               uncertainty_claim="none_descriptive_diagnostics", minimum_dates=SUPPORT_DATES))
        peak = selected.loc[selected.fold.eq("fold_3") & selected.delivery_date.between("2022-08-15", "2022-08-31")]
        diagnostics.append(_record(peak, policy=policy, fold="fold_3", scope="peak", group="August15-31",
                                   window_start="2022-08-15", window_end="2022-08-31", calendar_days=17,
                                   support_status="support_limited", evidence_class="descriptive_only_small_effective_sample"))
        for begin in (1, 8, 15, 22):
            start, end = f"2022-09-{begin:02}", f"2022-09-{begin + 6:02}"
            part = selected.loc[selected.fold.eq("fold_3") & selected.delivery_date.between(start, end)]
            diagnostics.append(_record(part, policy=policy, fold="fold_3", scope="recovery", group=start,
                                       window_start=start, window_end=end, calendar_days=7,
                                       evidence_class="descriptive_only"))
    daily = hourly.groupby(["policy", "fold", "delivery_date"]).agg(
        n_hours=("WIS", "size"), MAE=("absolute_error", "mean"), WIS=("WIS", "mean"),
        coverage95=("hit95", "mean"), hit_count95=("hit95", "sum"),
        lower_miss_count95=("lower_miss95", "sum"), upper_miss_count95=("upper_miss95", "sum"),
        mean_width95=("width95", "mean"))
    parts = []
    for fold, (start, end) in FOLD_WINDOWS.items():
        index = pd.MultiIndex.from_product([POLICIES, [fold], pd.date_range(start, end)],
                                          names=["policy", "fold", "delivery_date"])
        part = daily.reindex(index).reset_index()
        part["n_hours"] = part.n_hours.fillna(0).astype(int)
        parts.append(part)
    daily = pd.concat(parts, ignore_index=True)
    daily["scope"] = "daily"
    return pd.DataFrame(metrics), pd.concat([pd.DataFrame(diagnostics), daily], ignore_index=True), daily


def _divide(numerator, denominator):
    """Undefined denominators remain NaN; never an epsilon or a zero outcome."""
    a, b = np.broadcast_arrays(np.asarray(numerator, float), np.asarray(denominator, float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=np.isfinite(b) & (b > 0))


def _selection(metrics, diagnostics, *, integrity_verified=True):
    per_fold = metrics.loc[metrics.scope.eq("per_fold")].set_index(["policy", "fold"])
    peak = diagnostics.loc[diagnostics.scope.eq("peak")].set_index("policy")
    scores = {}
    for policy in POLICIES:
        scores[policy] = {f"S_{metric}": float(np.mean(_divide(
            per_fold.loc[policy, metric].reindex(FOLDS), per_fold.loc["B0", metric].reindex(FOLDS))))
            for metric in ("MAE", "WIS")}
    criteria = []

    def add(policy, criterion, metric, scope, actual, lower=None, upper=None, comparator=None):
        limits = [x for x in (lower, upper) if x is not None]
        assessed = bool(np.isfinite(actual) and np.isfinite(limits).all())
        passed = assessed and (lower is None or actual >= lower) and (upper is None or actual <= upper)
        criteria.append(dict(policy=policy, criterion=criterion, metric=metric, scope=scope,
                             actual=actual, lower_limit=lower, upper_limit=upper, comparator=comparator,
                             status=("met" if passed else "not_met") if assessed else "unassessed", passed=bool(passed)))

    for policy in CANDIDATES:
        for number, metric in ((1, "MAE"), (2, "WIS")):
            add(policy, number, f"S_{metric}", "equal_fold", scores[policy][f"S_{metric}"],
                upper=0.9 * np.min([scores[b][f"S_{metric}"] for b in BASELINES]), comparator="best B0-B3")
        for fold in FOLDS:
            add(policy, 3, "coverage95", fold, per_fold.loc[(policy, fold), "coverage95"], lower=.90, upper=.98)
        add(policy, 4, "coverage95", "peak", peak.loc[policy, "coverage95"], lower=.90)
        for metric in ("MAE", "WIS"):
            add(policy, 4, metric, "peak", peak.loc[policy, metric],
                upper=np.min([peak.loc[b, metric] for b in BASELINES]), comparator="best B0-B3 on matched peak")
            for fold in FOLDS:
                add(policy, 5, metric, fold, per_fold.loc[(policy, fold), metric],
                    upper=1.05 * min(per_fold.loc[(b, fold), metric] for b in ("B2", "B3")), comparator="best rolling B2/B3")
        add(policy, 6, "complete_finite_ordered", "all_eligible", int(integrity_verified), lower=1, upper=1)
    criteria = pd.DataFrame(criteria)
    ranking = sorted(CANDIDATES, key=lambda p: (scores[p]["S_WIS"], scores[p]["S_MAE"], p != "V2-P"))
    defined = all(np.isfinite(list(v.values())).all() for v in scores.values())
    return scores, criteria, ranking if defined else None


def _indices(replicates):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return {fold: (rng.integers(0, 84, size=(replicates, 13))[:, :, None] + np.arange(7))
            .reshape(replicates, -1)[:, :90] for fold in FOLDS}


def _bootstrap(daily, *, replicates=BOOTSTRAP_REPLICATES, indices=None):
    """One shared index set; primary estimand retains all sampled eligible hours.

Daily sums/counts retain variable 23/24/25-hour support. Missing dates have zero
observation counts, not zero observed losses. Per-fold daily CIs are a separate
descriptive estimand. Any undefined draw invalidates that comparison's entire CI.
"""
    indices = _indices(replicates) if indices is None else indices
    if set(indices) != set(FOLDS):
        raise ValueError("indices require exactly all five folds")
    samples, points, rows = [], [], []

    def interval(scope, candidate, baseline, metric, draws, point, estimand):
        unresolved = int((~np.isfinite(draws)).sum())
        valid = unresolved == 0 and np.isfinite(point)
        lo, hi = np.quantile(draws, [.025, .975], method="linear") if valid else (np.nan, np.nan)
        rows.append(dict(scope=scope, candidate=candidate, baseline=baseline, metric=metric,
                         difference=float(point), ci_lower=float(lo), ci_upper=float(hi),
                         status="resolved" if valid else "unresolved_uncertainty", undefined_replicates=unresolved,
                         replicates=replicates, estimand=estimand, confidence=.95,
                         evidence_class="exploratory_post_selection" if scope == "equal_fold" else "descriptive_paired_daily"))

    for fold in FOLDS:
        part = daily.loc[daily.fold.eq(fold)]
        dates = pd.date_range(*FOLD_WINDOWS[fold])
        expected = pd.MultiIndex.from_product([POLICIES, dates], names=["policy", "delivery_date"])
        if part.duplicated(["policy", "delivery_date"]).any() or set(part.set_index(["policy", "delivery_date"]).index) != set(expected):
            raise ValueError("bootstrap requires the full matched 90-calendar-date grid")
        values = np.stack([part.pivot(index="delivery_date", columns="policy", values=m)
                           .reindex(index=dates, columns=POLICIES).to_numpy(float) for m in ("MAE", "WIS")], axis=2)
        counts = part.pivot(index="delivery_date", columns="policy", values="n_hours").reindex(index=dates, columns=POLICIES).to_numpy(float)
        if not np.isfinite(counts).all() or (counts < 0).any() or not (counts == counts[:, :1]).all():
            raise ValueError("daily observation counts must be paired and nonnegative")
        valid = counts[:, 0] > 0
        if not np.array_equal(np.isfinite(values), np.broadcast_to(valid[:, None, None], values.shape)):
            raise ValueError("daily missingness must be identical and agree with zero-hour dates")
        ix = np.asarray(indices[fold])
        if ix.shape != (replicates, 90) or not np.issubdtype(ix.dtype, np.integer) or (ix < 0).any() or (ix >= 90).any():
            raise ValueError("invalid bootstrap index fixture")
        sums = np.where(valid[:, None, None], values, 0) * counts[:, :, None]
        sampled_means = _divide(sums[ix].sum(axis=1), counts[ix].sum(axis=1)[:, :, None])
        point = _divide(sums.sum(axis=0), counts.sum(axis=0)[:, None])
        samples.append(_divide(sampled_means, sampled_means[:, :1, :]))
        points.append(_divide(point, point[:1, :]))
        daily_samples = _divide(np.where(valid[:, None, None], values, 0)[ix].sum(axis=1), valid[ix].sum(axis=1)[:, None, None])
        daily_point = _divide(np.where(valid[:, None, None], values, 0).sum(axis=0), valid.sum())
        for candidate, baseline in CONTRASTS:
            a, b = POLICIES.index(candidate), POLICIES.index(baseline)
            for mi, metric in enumerate(("MAE", "WIS")):
                interval(fold, candidate, baseline, metric, daily_samples[:, a, mi] - daily_samples[:, b, mi],
                         daily_point[a, mi] - daily_point[b, mi], "paired mean daily loss difference")
    aggregate, point = np.mean(samples, axis=0), np.mean(points, axis=0)
    for candidate, baseline in CONTRASTS:
        a, b = POLICIES.index(candidate), POLICIES.index(baseline)
        for mi, metric in enumerate(("MAE", "WIS")):
            interval("equal_fold", candidate, baseline, metric, aggregate[:, a, mi] - aggregate[:, b, mi],
                     point[a, mi] - point[b, mi], "equal-fold B0-normalized eligible-hour mean difference")
    fingerprint = hashlib.sha256(b"".join(np.asarray(indices[f], dtype="<i8").tobytes() for f in FOLDS)).hexdigest()
    return pd.DataFrame(rows), dict(seed=BOOTSTRAP_SEED, replicates=replicates, block_days=BLOCK_DAYS,
                                   calendar_days=90, index_sha256=fingerprint, shared_index_sets=1,
                                   method="paired noncircular calendar moving-block percentile",
                                   undefined_handling="unresolved; no dropping or redrawing")


def _conclusions(uncertainty):
    conclusions = {}
    for candidate, baseline in CONTRASTS:
        part = uncertainty.loc[uncertainty.scope.eq("equal_fold") & uncertainty.candidate.eq(candidate)
                               & uncertainty.baseline.eq(baseline)].set_index("metric")
        resolved = set(part.index) == {"MAE", "WIS"} and part.status.eq("resolved").all()
        joint = resolved and part.loc["WIS", "ci_upper"] < 0 and part.loc["MAE", "ci_upper"] <= 0
        conclusions[f"{candidate}-{baseline}"] = "observed joint improvement" if joint else "no demonstrated joint preference"
    return conclusions


def _evaluate(predictions, expected_keys, *, production, replicates):
    frame = validate_predictions(predictions, expected_keys, production=production)
    hourly = score_hourly(frame)
    hourly["local_hour"] = hourly.timestamp_utc.dt.tz_convert("Europe/Berlin").dt.hour
    hourly["local_block"] = np.select([hourly.local_hour.ge(22) | hourly.local_hour.le(5),
                                       hourly.local_hour.between(10, 16)], ["night", "solar"], default="shoulder")
    metrics, diagnostics, daily = _tables(hourly)
    scores, criteria, ranking = _selection(metrics, diagnostics)
    pooled = metrics.loc[metrics.scope.eq("pooled")].set_index("policy")
    for metric in ("MAE", "WIS"):
        metrics[f"pooled_ratio_{metric}"] = np.nan
        mask = metrics.scope.eq("pooled")
        metrics.loc[mask, f"pooled_ratio_{metric}"] = _divide(metrics.loc[mask, metric], pooled.loc["B0", metric])
    metrics = pd.concat([metrics, pd.DataFrame([dict(policy=p, fold="all", scope="equal_fold", **scores[p]) for p in POLICIES])], ignore_index=True)
    uncertainty, bootstrap_metadata = _bootstrap(daily, replicates=replicates)
    diagnostic_status, failures = {}, {}
    for policy in CANDIDATES:
        part = criteria.loc[criteria.policy.eq(policy)]
        diagnostic_status[policy] = "unassessed" if part.status.eq("unassessed").any() else ("met" if part.passed.all() else "not_met")
        failures[policy] = sorted(part.loc[~part.passed, "criterion"].unique().tolist())
    summary = dict(ranking=ranking, ranking_rule="S_WIS, S_MAE, P on exact ties; descriptive only",
                   selection_status="defined" if ranking else "undefined; protocol correction required before selection",
                   original_section8_status=diagnostic_status, failed_criteria=failures,
                   joint_conclusions=_conclusions(uncertainty), bootstrap=bootstrap_metadata,
                   inference="exploratory post-selection; no equivalence, absence-of-benefit or absence-of-harm claim",
                   primary_contrast="V2-H-V2-P", secondary_contrasts=["V2-H-B2", "V2-P-B2"],
                   primary_weighting="equal-fold B0-normalized eligible-hour losses", pooled_weighting="eligible hours",
                   historical_cp15_product_status="NOT_DEMONSTRATED", product_delivery_eligibility="not authorized by research evaluation",
                   engineering_status="not assessed by scoring", native_v1_pinball="preserved separately; mean_pinball_7 is not native v1",
                   resources="fit/runtime/memory and failed attempts must be joined from driver resource ledger",
                   validation=dict(production=production, rows=len(frame), keys_per_policy=len(frame) // 7,
                                   missing_predictions=0, nonfinite_quantiles=0, crossings=0),
                   support_rule=dict(minimum_represented_dates=SUPPORT_DATES, role="reporting only; no product gate"))
    return dict(metrics=metrics, diagnostics=diagnostics, uncertainty=uncertainty, criteria=criteria, summary=summary)


def evaluate(predictions: pd.DataFrame, expected_keys: pd.DataFrame) -> dict:
    """Evaluate exactly 75,229 production rows using the frozen 2,000 draws."""
    return _evaluate(predictions, expected_keys, production=True, replicates=BOOTSTRAP_REPLICATES)
