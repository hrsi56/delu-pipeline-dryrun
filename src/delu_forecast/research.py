"""Typed evidence records for the research generations (presentation plan §9.1–§9.2).

v1's published claims come from `claims.py`, which reads CP-2's artifacts. Everything a public
surface says about CP-10, CP-15, CP-16 and CP-20 comes from here instead: one typed record per
value, each carrying the file it was read from, the git blob that file must still be, the row it
came from, its unit, aggregation, population, comparator and evidence class.

**This module reads committed files. It never scores, fits or recomputes anything.** A record's
value is the exact text of one cell in one committed row; formatting for display happens once,
in `display()`, so "0.5658" is one rounding of one saved number on every surface.

Provenance is checked from bytes, not from Git: the committed CI checkout is shallow and carries
no tags, so each source file's expected blob SHA is recorded in `SOURCES` next to the evidence
tag that preserves it, and `validate()` recomputes the blob from the file on disk.
`tests/test_29_research_evidence.py` re-derives every record and carries the negative controls.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from . import registry as _registry

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The last development delivery date. No v2+ record may describe anything after it, and no
#: v2+ model is scored on v1's holdout window (plan §6 invariant 10).
EVIDENCE_BOUNDARY = "2026-04-07"

MINUS = "−"


class EvidenceError(ValueError):
    """A record no longer matches its committed source."""


# --------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class Source:
    path: str
    tag: str
    blob: str


def _source(path: str, tag: str, blob: str) -> tuple[str, Source]:
    return path, Source(path, tag, blob)


#: Every file a record may be read from, with the evidence tag that preserves it and the Git
#: blob SHA its committed bytes must hash to. The tag is the checkpoint that produced the file:
#: CP-10's files landed with CP-15, and CP-16's are also reachable from `evidence/cp-20`.
SOURCES: dict[str, Source] = dict(
    [
        # CP-24 (v5, scored attempt 1), registered by PRES-4
        _source("reports/ddnn2/attempt-1/metrics.csv", "evidence/cp-24", "9e9dcec32887a8684930e19ae244b7be317cb087"),
        _source("reports/ddnn2/attempt-1/uncertainty.csv", "evidence/cp-24", "a584caff5121b52ebe4b9f423791aa3b10c7d606"),
        _source("reports/ddnn2/attempt-1/ratio-intervals.csv", "evidence/cp-24", "468ab28d21d2f1d50d00d9e81bf0426394e70fcb"),
        _source("reports/ddnn2/attempt-1/criteria.csv", "evidence/cp-24", "42e0cb41d928e14c12572ad47669978e582461b4"),
        _source("reports/ddnn2/attempt-1/diagnostics.csv", "evidence/cp-24", "3b6e2c1ab5895f6440fb308fe8667055a076df7c"),
        _source("reports/ddnn2/attempt-1/protocol.json", "evidence/cp-24", "51cfe59d5d29f305df10d374dab25d91a75d1a61"),
        _source("reports/ddnn2/attempt-1/decisions.json", "evidence/cp-24", "cb164128f96b9005cedfec790bd029c223f093be"),
        _source("reports/ddnn2/attempt-1/controls.json", "evidence/cp-24", "190d29c3388f4c378828e3e0ebb189947e04d9d3"),
        _source("reports/ddnn2/attempt-1/leakage-controls.json", "evidence/cp-24",
                "a01fdeb5dc85e6bda5e09410618725fa1a1c3314"),
        # CP-23 (the branch "DDNN member on v4")
        _source("reports/distribution-challenger/metrics.csv", "evidence/cp-23", "a9fb20ec0272420091e2f063cbbfa72d5da45348"),
        _source("reports/distribution-challenger/uncertainty.csv", "evidence/cp-23",
                "1eae3fe473bab425ba91f0b51a124fe840244870"),
        _source("reports/distribution-challenger/criteria.csv", "evidence/cp-23", "f23656dc75f944697358db581a27d5f049be6ea8"),
        _source("reports/distribution-challenger/diagnostics.csv", "evidence/cp-23",
                "d7572377978e6e76f08214974fedf053e1183864"),
        _source("reports/distribution-challenger/protocol.json", "evidence/cp-23", "a7018661bc3df169ea90694e12b61b897463d959"),
        _source("reports/distribution-challenger/decisions.json", "evidence/cp-23", "a07073be22b19b6be1149e87eff5acdc4bb8cebc"),
        # CP-22 (the branch "Pooled LightGBM member for v4")
        _source("reports/v4-revision/metrics.csv", "evidence/cp-22", "2b22bf79791ac21095ca5ffa3f4ebc4419faca7b"),
        _source("reports/v4-revision/uncertainty.csv", "evidence/cp-22", "fe4600d74624ea559efc1d5726e534b4f2d6b054"),
        _source("reports/v4-revision/criteria.csv", "evidence/cp-22", "171a4c51b253b2c580236b883dee13e12a0788f6"),
        _source("reports/v4-revision/diagnostics.csv", "evidence/cp-22", "bdaaa93622aed5b511c0686bc8cff0816babbc3d"),
        _source("reports/v4-revision/protocol.json", "evidence/cp-22", "7463370d0d6aa1b0eb1a74077f319246b7153209"),
        _source("reports/v4-revision/decisions.json", "evidence/cp-22", "9c4f1598e9c0ddf20aa3bf5afe3829d864d9782a"),
        # The data's own range, for the coverage limits of the shared population (PRES-4): the first data day and the
        # development boundary, read from these committed records, never from post-boundary rows.
        _source("data/source_manifest.json", "evidence/cp-2", "bd53ae262273280a79ebaba84be5da5e54823530"),
        _source("data/partitions.json", "evidence/cp-2", "6cabcd6674fb426d16514640be0436484ca6df79"),
        # CP-21 (v4)
        _source("reports/block-challenger/metrics.csv", "evidence/cp-21", "677ccf0ce4bf826508a1d195a1dfdf829d8568f6"),
        _source("reports/block-challenger/uncertainty.csv", "evidence/cp-21", "390969f619cf163869f7b2b349bc561e87e05a9a"),
        _source("reports/block-challenger/criteria.csv", "evidence/cp-21", "7e2af1e899f9cfaf393fc3258d3687910831db47"),
        _source("reports/block-challenger/diagnostics.csv", "evidence/cp-21", "e8479ec670b03a813fab7e94c71a119b7b3e84b0"),
        _source("reports/block-challenger/protocol.json", "evidence/cp-21", "3670a30d3d1eb92d498a343787ec8242926c7e06"),
        _source("reports/block-challenger/adoption.json", "evidence/cp-21", "f554bbc76ce4896ded59ba8d77e140498312f4ca"),
        _source("reports/block-challenger/controls.json", "evidence/cp-21", "36f3ac0c59d0728912a55bb4db27f591a13c45fa"),
        _source("reports/block-challenger/hg-parity.json", "evidence/cp-21", "f6bbf3f32e956ebe012911c0416b5bd3ad729968"),
        _source("reports/block-challenger/fit-cost.json", "evidence/cp-21", "33bf81e615b862e9efe0dad4e75a6b489fa9464a"),
        # CP-20 (v3)
        _source("reports/weather-ablation/metrics.csv", "evidence/cp-20", "91be7505842cdec6062fd0bcbc7f88f2f3ec031c"),
        _source("reports/weather-ablation/uncertainty.csv", "evidence/cp-20", "aa3028f17bbe7f6c0dcc65f91266fc00b29b0207"),
        _source("reports/weather-ablation/criteria.csv", "evidence/cp-20", "ec377d647ff5eee68346fd5fcddaa6a2dbd2aa16"),
        _source("reports/weather-ablation/diagnostics.csv", "evidence/cp-20", "e1c8e11b9fccd3e61f7fb9f0ab904c410f89bc72"),
        _source("reports/weather-ablation/protocol.json", "evidence/cp-20", "13095656266eaf9de92e0e0da038fd1a324eee23"),
        _source("reports/weather-ablation/extraction-summary.json", "evidence/cp-20", "22349d9cfd7eba20f4ab5ef93deb1c884a49e01d"),
        _source("reports/weather-ablation/missingness.csv", "evidence/cp-20", "4c1603696964f51819d094c5227a1fe2f946b043"),
        _source("reports/weather-ablation/causal-controls.json", "evidence/cp-20", "adbd9973300b2aaec3175ad46c05c84282e7ccd3"),
        _source("reports/weather-ablation/causal-controls-supplement.json", "evidence/cp-20", "05ea6d8f0b3d7f3538f50e0941a1c88e68414bf9"),
        _source("reports/weather-ablation/weather-features.parquet", "evidence/cp-20", "8704810f2778b8432ebb7289f808abec96730256"),
        _source("docs/track-b/evidence/cp-20/resource-final.json", "evidence/cp-20", "f50cf0e8e5c0ee94495ee4982dc5e4d704de6f72"),
        # CP-16 (v2)
        _source("reports/v2-causal/metrics.csv", "evidence/cp-16", "a194e3c3e46b963eaa2dd3c65c8e79627062b5e5"),
        _source("reports/v2-causal/uncertainty.csv", "evidence/cp-16", "59cef0552f86cf939d507b30c4f46fe83cef5248"),
        _source("reports/v2-causal/criteria.csv", "evidence/cp-16", "29c0b5bf5c3b251e1cde32f0c28310f924edfa3c"),
        _source("reports/v2-causal/diagnostics.csv", "evidence/cp-16", "bd5aed0726dac95e83423ccf4f498edae230eefb"),
        _source("reports/v2-causal/protocol.json", "evidence/cp-16", "53f096ad464a1c27fec8013fffb20c49696f6f74"),
        # CP-15 (the model-comparison study) and CP-10 (landed with it)
        _source("reports/cp15/peak.csv", "evidence/cp-15", "6e36a4fc724753f6a6eef277465f0f5b26446892"),
        _source("reports/cp15/relative_scores.csv", "evidence/cp-15", "854f1b3e865887d39e87a3380ff7f427b927d793"),
        _source("reports/cp15/pooled.csv", "evidence/cp-15", "3e5cb75c3db7e82064a64c0f31971330095858ab"),
        _source("reports/cp15/per_fold.csv", "evidence/cp-15", "56779e3646426f2c7791536cdc509134d85afa51"),
        _source("reports/cp15/daily.csv", "evidence/cp-15", "941b833627d66da19f6ae9955f4406d12cfae46c"),
        _source("reports/cp15/bootstrap.csv", "evidence/cp-15", "8a0fa9d96458ad870c2bd91954df80d594ecdfe8"),
        _source("reports/cp15/bootstrap_metadata.json", "evidence/cp-15", "096047ae36cb6708b5ac739e3bab7af72a621b1d"),
        _source("reports/cp15/ranking.csv", "evidence/cp-15", "68d549bcdf68cb7acb369f471bdb3d73a02744b5"),
        _source("reports/cp15/criteria.csv", "evidence/cp-15", "79aed270c3ef26f76e14134be6e6e4782a0c9ad8"),
        _source("reports/cp15/protocol.json", "evidence/cp-15", "b3a926e93f59a4e4d736e194a56c7ac678d810a9"),
        _source("reports/cp10/metrics.csv", "evidence/cp-15", "058b0f538df6091a17305c414e4b114d1c940b62"),
        _source("reports/cp10/peak_windows.csv", "evidence/cp-15", "4a33ec625812496a8973606dcb02e44856657468"),
        _source("reports/cp10/selection.json", "evidence/cp-15", "f27eee818bea83314df32da55db2e56ee20fb8af"),
        _source("reports/cp10/protocol.json", "evidence/cp-15", "8da7b10e78499a4a89b61e771793da0fef622ce7"),
        # CP-2 (v1's own development record, for the F07 note)
        _source("reports/cp2/dm_development.json", "evidence/cp-2", "05b225ef472fc2368526c95381ae773fdaf92aec"),
        _source("reports/cp2/development_pooled_metrics.csv", "evidence/cp-2", "5ee7d18e3636256a24e72707a9ec1300c825b48d"),
    ]
)

#: The released product's own evaluation and diagnostics (CP-2), for its documentation on the page
#: (PUBLISH_RULES 1.0 §5.1). Checked exactly like `SOURCES`, and kept apart from it because the MLflow
#: export's manifest lists the research mirror's sources, and these files are not part of that mirror.
PRODUCT_SOURCES: dict[str, Source] = dict(
    [
        _source("reports/cp2/holdout_report.json", "evidence/cp-2", "43676b8f3135bbd916a335b8cf8f422d57834def"),
        _source("reports/cp2/regime_table.csv", "evidence/cp-2", "417d2ccc45280ca305118221e775884dc4c5ac47"),
        _source("reports/cp2/reliability_three_stage.csv", "evidence/cp-2", "3994e819870bdb8ce6429ad782f17c51f997246c"),
        _source("reports/cp2/diagnostics.json", "evidence/cp-2", "0b53a8be8af2f8458aca513298fd2048e5c2f6d8"),
    ]
)


def registered_source(path: str) -> Source | None:
    """The registered source of `path`, research or product, or None."""
    return SOURCES.get(path) or PRODUCT_SOURCES.get(path)


def blob_sha(data: bytes) -> str:
    """The Git blob SHA-1 of `data`, computed without Git."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def source_bytes(path: str) -> bytes:
    return (REPO_ROOT / path).read_bytes()


def check_source(path: str) -> None:
    """Raise unless `path` is a registered source whose bytes still hash to its recorded blob."""
    source = registered_source(path)
    if source is None:
        raise EvidenceError(f"{path} is not a registered evidence source")
    actual = blob_sha(source_bytes(path))
    if actual != source.blob:
        raise EvidenceError(
            f"{path} has changed: blob {actual} is not the {source.blob} preserved by {source.tag}"
        )


def _csv(path: str) -> list[tuple[int, dict[str, str]]]:
    """(physical line, row) pairs, every cell as its exact committed text."""
    reader = csv.DictReader(io.StringIO(source_bytes(path).decode("utf-8"), newline=""))
    out = []
    for row in reader:
        out.append((reader.line_num, dict(row)))
    return out


@lru_cache(maxsize=None)
def _csv_cached(path: str) -> tuple[tuple[int, tuple[tuple[str, str], ...]], ...]:
    return tuple((line, tuple(row.items())) for line, row in _csv(path))


def _rows(path: str) -> list[tuple[int, dict[str, str]]]:
    return [(line, dict(items)) for line, items in _csv_cached(path)]


@lru_cache(maxsize=None)
def _json(path: str) -> dict:
    return json.loads(source_bytes(path))


def _json_get(document: dict, dotted: str):
    node = document
    for part in dotted.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


# --------------------------------------------------------------------------- the record


@dataclass(frozen=True)
class Interval:
    """A confidence interval of an estimated difference -- never a forecast interval."""

    method: str
    level: float
    seed: int
    replicates: int
    block_days: int


@dataclass(frozen=True)
class EvidenceRecord:
    record_id: str
    checkpoint: str
    generation: str | None
    policy_code: str | None
    source_path: str
    selector: tuple[tuple[str, str], ...]
    field: str
    metric: str
    unit: str
    aggregation: str
    population_id: str
    comparator: str | None
    evidence_class: str
    window: tuple[str, str] | None
    display_precision: int
    raw: str
    source_line: int | None = None
    interval: Interval | None = None
    ci_low_raw: str | None = None
    ci_high_raw: str | None = None
    detail: tuple[tuple[str, str], ...] = field(default=())
    #: The columns an interval's endpoints are read from: a difference's, or (from CP-21) a ratio's own.
    ci_columns: tuple[str, str] = ("ci_lower", "ci_upper")

    @property
    def value(self) -> float:
        return float(self.raw)

    @property
    def ci_low(self) -> float | None:
        return None if self.ci_low_raw is None else float(self.ci_low_raw)

    @property
    def ci_high(self) -> float | None:
        return None if self.ci_high_raw is None else float(self.ci_high_raw)

    @property
    def source(self) -> Source:
        return registered_source(self.source_path)

    @property
    def source_revision(self) -> str:
        return f"{self.source.tag} (blob {self.source.blob})"

    def selector_dict(self) -> dict[str, str]:
        return dict(self.selector)


# --------------------------------------------------------------------------- units

UNIT_RATIO = "ratio to the similar-day naive (B0)"
UNIT_EUR = "EUR/MWh"
UNIT_FRACTION = "fraction"
UNIT_COUNT = "count"
UNIT_NORM_DIFF = "normalized score difference"
UNIT_EUR_DIFF = "EUR/MWh, paired mean daily loss difference"
UNIT_HOURS = "hours"
UNIT_DAYS = "days"
UNIT_PERCENT = "percent"
UNIT_GIB = "GiB"
UNIT_MACHINE_HOURS = "machine-hours"
UNIT_USD = "USD"
UNIT_RUNS = "runs"
UNIT_MESSAGES = "messages"
UNIT_SEED = "seed"
UNIT_REPLICATES = "replicates"
UNIT_LABEL = "label"
#: A relative change from a checkpoint's own bootstrap draws (CP-21 on): S_policy / S_comparator − 1.
UNIT_RATIO_CHANGE = "change as a share of the comparator's score"

#: CSV column -> (metric name, unit, display precision). One place, so a unit cannot drift
#: between the record that carries it and the check that validates it.
FIELD_SPECS: dict[str, tuple[str, str, int]] = {
    "S_MAE": ("S_MAE", UNIT_RATIO, 4),
    "S_WIS": ("S_WIS", UNIT_RATIO, 4),
    "MAE": ("MAE", UNIT_EUR, 2),
    "WIS": ("WIS", UNIT_EUR, 2),
    "RMSE": ("RMSE", UNIT_EUR, 2),
    "bias": ("bias", UNIT_EUR, 2),
    "raw_central_MAE": ("raw central MAE", UNIT_EUR, 2),
    "daily_mean_level_MAE": ("daily mean-level MAE", UNIT_EUR, 2),
    "within_day_shape_MAE": ("within-day shape MAE", UNIT_EUR, 2),
    "mean_width50": ("mean 50% interval width", UNIT_EUR, 2),
    "mean_width80": ("mean 80% interval width", UNIT_EUR, 2),
    "mean_width95": ("mean 95% interval width", UNIT_EUR, 2),
    "coverage50": ("50% coverage", UNIT_FRACTION, 4),
    "coverage80": ("80% coverage", UNIT_FRACTION, 4),
    "coverage95": ("95% coverage", UNIT_FRACTION, 4),
    "hit_count50": ("hits inside the 50% interval", UNIT_COUNT, 0),
    "hit_count80": ("hits inside the 80% interval", UNIT_COUNT, 0),
    "hit_count95": ("hits inside the 95% interval", UNIT_COUNT, 0),
    "n_hours": ("hours", UNIT_HOURS, 0),
    "n_days": ("days", UNIT_DAYS, 0),
    # CP-10's own column names (native nine-quantile scores)
    "mae": ("MAE", UNIT_EUR, 2),
    "mean_pinball": ("nine-quantile mean pinball loss", UNIT_EUR, 2),
    "coverage_50": ("50% coverage", UNIT_FRACTION, 4),
    "coverage_80": ("80% coverage", UNIT_FRACTION, 4),
    "coverage_95": ("95% coverage", UNIT_FRACTION, 4),
    "covered_95": ("hits inside the 95% interval", UNIT_COUNT, 0),
    "n_obs": ("hours", UNIT_HOURS, 0),
    "mae_p50": ("MAE", UNIT_EUR, 2),
    "rank": ("rank", UNIT_COUNT, 0),
    # CP-2's regime table and three-stage reliability (the released product's development record)
    "mae_ci95_low": ("MAE, lower end of the day-block bootstrap 95% confidence interval", UNIT_EUR, 2),
    "mae_ci95_high": ("MAE, upper end of the day-block bootstrap 95% confidence interval", UNIT_EUR, 2),
    "empirical": ("empirical coverage", UNIT_FRACTION, 4),
}

#: Metric-row fields published as records, per aggregation.
SCORE_FIELDS = ("S_MAE", "S_WIS")
ROW_FIELDS = (
    "MAE", "WIS", "RMSE", "bias", "raw_central_MAE", "daily_mean_level_MAE", "within_day_shape_MAE",
    "coverage50", "coverage80", "coverage95", "hit_count50", "hit_count80", "hit_count95",
    "mean_width50", "mean_width80", "mean_width95", "n_hours", "n_days",
)
PEAK_FIELDS = (
    "MAE", "WIS", "bias", "daily_mean_level_MAE", "within_day_shape_MAE", "coverage95",
    "hit_count95", "mean_width95", "n_hours", "n_days",
)

#: Which adopted generation a policy code is, if any, from the registry: references, study arms
#: and controls have none, and one identity keeps one generation across codes (V2-H = H0). A code whose text names
#: different identities in different experiments (`v5`) is left out here and resolved by `_generation`.
GENERATION_OF = {code: _registry.generation_of(code) for code in _registry.codes() if _registry.generation_of(code)}


def _generation(code: str, checkpoint: str) -> str | None:
    """The generation a policy code is in the experiment whose rows carry it: CP-23's `v5` is the branch's candidate,
    with none, and CP-24's is v5. A code the registry does not scope to that experiment keeps its global reading."""
    try:
        return _registry.by_code(code, checkpoint).version
    except _registry.RegistryError:
        return GENERATION_OF.get(code)

EVIDENCE_CLASS_DEVELOPMENT = "development_post_selection"
EVIDENCE_CLASS_CALIBRATION = "development_calibration_comparison"

#: Scope column value -> aggregation name (plan §9.2).
AGGREGATION_OF_SCOPE = {
    "equal_fold": "equal_fold",
    "pooled": "pooled",
    "per_fold": "per_fold",
    "peak": "peak_window",
    "hour": "hour",
    "daily": "daily",
}

POPULATIONS = {
    "common-10747h": "the 10,747 eligible development hours shared by CP-15, CP-16 and CP-20",
    "crisis-window-408h": "the matched crisis window, delivery 2022-08-15..31",
    "cp10-fold-block": "CP-10's full fold blocks (its own calibration comparison)",
    "cp10-crisis-window-408h": "CP-10's matched crisis window, delivery 2022-08-15..31",
    "v1-development": "v1's own development evaluation (CP-2)",
    "v1-holdout-90d": "v1's pre-specified 90-day holdout, delivery 2026-06-09..2026-09-06, the frozen artifact",
    "v1-fold5-eval": "fold 5's evaluation block, delivery 2026-01-08..2026-04-07, as v1's diagnostics read it (CP-2)",
    "cp20-extraction": "the CP-20 GFS extraction",
    "cp20-resources": "the CP-20 cumulative resource ledger",
    "protocol": "a frozen protocol setting",
}


def _population(scope: str, fold: str | None) -> str:
    if scope in ("per_fold", "hour", "daily") and fold:
        return f"common-10747h/{fold}"
    if scope == "peak":
        return "crisis-window-408h"
    return "common-10747h"


# --------------------------------------------------------------------------- builders


def _metrics_key(row: dict[str, str]) -> str:
    return row["scope"] if row["scope"] in ("pooled", "equal_fold") else row["fold"]


def _windows_by_policy(rows) -> dict[str, tuple[str, str]]:
    return {
        row["policy"]: (row["first_delivery_date"], row["last_delivery_date"])
        for _, row in rows
        if row.get("scope") == "pooled"
    }


def _metric_records(prefix: str, checkpoint: str, path: str) -> list[EvidenceRecord]:
    rows = _rows(path)
    pooled_window = _windows_by_policy(rows)
    records = []
    for line, row in rows:
        policy, scope = row["policy"], row["scope"]
        key = _metrics_key(row)
        fields = SCORE_FIELDS if scope == "equal_fold" else ROW_FIELDS
        window = pooled_window[policy] if scope == "equal_fold" else (
            row["first_delivery_date"], row["last_delivery_date"]
        )
        for column in fields:
            if row.get(column, "") == "":
                continue
            metric, unit, precision = FIELD_SPECS[column]
            records.append(
                EvidenceRecord(
                    record_id=f"{prefix}.metrics.{policy}.{key}.{column}",
                    checkpoint=checkpoint,
                    generation=_generation(policy, checkpoint),
                    policy_code=policy,
                    source_path=path,
                    selector=(("policy", policy), ("scope", scope), ("fold", row["fold"])),
                    field=column,
                    metric=metric,
                    unit=unit,
                    aggregation=AGGREGATION_OF_SCOPE[scope],
                    population_id=_population(scope, row["fold"]),
                    comparator="B0" if column in SCORE_FIELDS else None,
                    evidence_class=EVIDENCE_CLASS_DEVELOPMENT,
                    window=window,
                    display_precision=precision,
                    raw=row[column],
                    source_line=line,
                )
            )
    return records


def _interval(checkpoint: str) -> Interval:
    """The bootstrap settings, read from the checkpoint's own frozen record. CP-16 inherited
    CP-15's recorded settings unchanged (capstone §14.4); CP-21 reused CP-20's index generator."""
    if checkpoint in ("CP-22", "CP-23", "CP-24"):
        # CP-22's and CP-23's frozen protocols carry their bootstrap settings; CP-24's decision record does.
        settings = (_json(CP24_ROWS["decisions"])["bootstrap"] if checkpoint == "CP-24"
                    else _json(CP22_ROWS["protocol"] if checkpoint == "CP-22" else CP23_ROWS["protocol"])["uncertainty"])
        return Interval(
            method="paired noncircular moving-block percentile bootstrap",
            level=0.95,
            seed=int(settings["seed"]),
            replicates=int(settings["replicates"]),
            block_days=int(settings["block_days"]),
        )
    if checkpoint == "CP-21":
        settings = _json("reports/block-challenger/protocol.json")["uncertainty"]
        return Interval(
            method="paired noncircular moving-block percentile bootstrap",
            level=0.95,
            seed=int(settings["seed"]),
            replicates=int(settings["replicates"]),
            block_days=int(settings["block_days"]),
        )
    if checkpoint == "CP-20":
        comparison = _json("reports/weather-ablation/protocol.json")["comparison"]
        return Interval(
            method="paired noncircular moving-block percentile bootstrap",
            level=0.95,
            seed=int(comparison["bootstrap_seed"]),
            replicates=int(comparison["replicates"]),
            block_days=int(comparison["block_days"]),
        )
    meta = _json("reports/cp15/bootstrap_metadata.json")
    return Interval(
        method=meta["method"] + " bootstrap",
        level=float(meta["confidence"]),
        seed=int(meta["seed"]),
        replicates=int(meta["replicates"]),
        block_days=int(meta["block_days"]),
    )


def _uncertainty_records(prefix: str, checkpoint: str, path: str, windows) -> list[EvidenceRecord]:
    records = []
    interval = _interval(checkpoint)
    for line, row in _rows(path):
        scope, candidate, baseline, metric = row["scope"], row["candidate"], row["baseline"], row["metric"]
        equal = scope == "equal_fold"
        confidence = float(row.get("confidence") or interval.level)
        replicates = int(row.get("replicates") or interval.replicates)
        records.append(
            EvidenceRecord(
                record_id=f"{prefix}.uncertainty.{candidate}-{baseline}.{scope}.{metric}",
                checkpoint=checkpoint,
                generation=_generation(candidate, checkpoint),
                policy_code=candidate,
                source_path=path,
                selector=(("scope", scope), ("candidate", candidate), ("baseline", baseline), ("metric", metric)),
                field="difference",
                metric=f"delta_{'S_' if equal else ''}{metric}",
                unit=UNIT_NORM_DIFF if equal else UNIT_EUR_DIFF,
                aggregation="equal_fold_contrast" if equal else "per_fold_contrast",
                population_id="common-10747h" if equal else f"common-10747h/{scope}",
                comparator=baseline,
                evidence_class=EVIDENCE_CLASS_DEVELOPMENT,
                window=windows["all" if equal else scope],
                display_precision=4 if equal else 2,
                raw=row["difference"],
                source_line=line,
                interval=Interval(interval.method, confidence, interval.seed, replicates, interval.block_days),
                ci_low_raw=row["ci_lower"],
                ci_high_raw=row["ci_upper"],
                detail=tuple(
                    (key, row[key]) for key in ("status", "estimand", "evidence_class") if key in row
                ),
            )
        )
    return records


def _ratio_records(prefix: str, checkpoint: str, path: str, windows) -> list[EvidenceRecord]:
    """From CP-21 on, each equal-fold contrast's change as a share of the comparator's score, with the interval
    of that ratio from the checkpoint's own bootstrap draws (PUBLISH_RULES §3.2): never a difference divided
    by a fixed denominator."""
    records = []
    interval = _interval(checkpoint)
    for line, row in _rows(path):
        if row["scope"] != "equal_fold" or row.get("ratio", "") == "":
            continue
        candidate, baseline, metric = row["candidate"], row["baseline"], row["metric"]
        records.append(EvidenceRecord(
            record_id=f"{prefix}.ratio.{candidate}-{baseline}.{metric}", checkpoint=checkpoint,
            generation=_generation(candidate, checkpoint), policy_code=candidate, source_path=path,
            selector=(("scope", "equal_fold"), ("candidate", candidate), ("baseline", baseline), ("metric", metric)),
            field="ratio", metric=f"ratio_S_{metric}", unit=UNIT_RATIO_CHANGE, aggregation="equal_fold_ratio",
            population_id="common-10747h", comparator=baseline, evidence_class=EVIDENCE_CLASS_DEVELOPMENT,
            window=windows["all"], display_precision=4, raw=row["ratio"], source_line=line,
            interval=Interval(interval.method, float(row.get("confidence") or interval.level), interval.seed,
                              int(row.get("replicates") or interval.replicates), interval.block_days),
            ci_low_raw=row["ratio_ci_lower"], ci_high_raw=row["ratio_ci_upper"],
            ci_columns=("ratio_ci_lower", "ratio_ci_upper"),
        ))
    return records


#: CP-24's decision level (capstone v21-r11 §23.9): condition 1 reads two-sided 97.5% intervals of the same stored
#: draws, the 1.25% and 98.75% percentiles; condition 4 stays at 95%. Both levels are published (§23.8).
DECISION_LEVEL_CP24 = 0.975


def _decision_level_records(prefix: str, checkpoint: str, path: str, ratio_path: str, windows) -> list[EvidenceRecord]:
    """CP-24's equal-fold contrasts at the decision level: each paired difference with its 97.5% interval
    (`ci97.5_lower`, `ci97.5_upper` of the uncertainty rows) and each ratio with its 97.5% interval
    (`ratio_ci97.5_lower`, `ratio_ci97.5_upper` of `ratio-intervals.csv`), from the checkpoint's own draws."""
    base = _interval(checkpoint)
    level = float(_json(CP24_ROWS["decisions"])["bootstrap"]["decision_level"])
    records = []
    for line, row in _rows(path):
        if row["scope"] != "equal_fold" or row.get("ci97.5_lower", "") == "":
            continue
        candidate, baseline, metric = row["candidate"], row["baseline"], row["metric"]
        records.append(EvidenceRecord(
            record_id=f"{prefix}.uncertainty97_5.{candidate}-{baseline}.equal_fold.{metric}", checkpoint=checkpoint,
            generation=_generation(candidate, checkpoint), policy_code=candidate, source_path=path,
            selector=(("scope", "equal_fold"), ("candidate", candidate), ("baseline", baseline), ("metric", metric)),
            field="difference", metric=f"delta_S_{metric}", unit=UNIT_NORM_DIFF, aggregation="equal_fold_contrast",
            population_id="common-10747h", comparator=baseline, evidence_class=EVIDENCE_CLASS_DEVELOPMENT,
            window=windows["all"], display_precision=4, raw=row["difference"], source_line=line,
            interval=Interval(base.method, level, base.seed, int(row.get("replicates") or base.replicates), base.block_days),
            ci_low_raw=row["ci97.5_lower"], ci_high_raw=row["ci97.5_upper"], ci_columns=("ci97.5_lower", "ci97.5_upper"),
        ))
    for line, row in _rows(ratio_path):
        if row.get("defined") != "True" or row.get("ratio", "") == "":
            continue
        candidate, baseline, metric = row["candidate"], row["baseline"], row["metric"]
        records.append(EvidenceRecord(
            record_id=f"{prefix}.ratio97_5.{candidate}-{baseline}.{metric}", checkpoint=checkpoint,
            generation=_generation(candidate, checkpoint), policy_code=candidate, source_path=ratio_path,
            selector=(("candidate", candidate), ("baseline", baseline), ("metric", metric)),
            field="ratio", metric=f"ratio_S_{metric}", unit=UNIT_RATIO_CHANGE, aggregation="equal_fold_ratio",
            population_id="common-10747h", comparator=baseline, evidence_class=EVIDENCE_CLASS_DEVELOPMENT,
            window=windows["all"], display_precision=4, raw=row["ratio"], source_line=line,
            interval=Interval(base.method, level, base.seed, int(row.get("replicates") or base.replicates), base.block_days),
            ci_low_raw=row["ratio_ci97.5_lower"], ci_high_raw=row["ratio_ci97.5_upper"],
            ci_columns=("ratio_ci97.5_lower", "ratio_ci97.5_upper"),
        ))
    return records


def _fold_windows(path: str) -> dict[str, tuple[str, str]]:
    """Fold windows as the metrics file records them (the B0 rows)."""
    windows = {}
    for _, row in _rows(path):
        if row["policy"] != "B0":
            continue
        key = "all" if row["scope"] == "pooled" else row["fold"]
        if row["scope"] in ("pooled", "per_fold"):
            windows[key] = (row["first_delivery_date"], row["last_delivery_date"])
    return windows


def _criteria_records(prefix: str, checkpoint: str, path: str, windows) -> list[EvidenceRecord]:
    records = []
    for line, row in _rows(path):
        policy, criterion, metric, scope = row["policy"], row["criterion"], row["metric"], row["scope"]
        base = f"{prefix}.criteria.{policy}.c{criterion}.{scope}.{metric}"
        # criterion 6's "complete_finite_ordered" is the share of eligible targets that pass.
        name, unit, precision = FIELD_SPECS.get(metric, (metric, UNIT_FRACTION, 4))
        if scope == "peak":
            window, population = ("2022-08-15", "2022-08-31"), "crisis-window-408h"
        elif scope.startswith("fold_"):
            window, population = windows[scope], f"common-10747h/{scope}"
        else:
            window, population = windows["all"], "common-10747h"
        selector = (("policy", policy), ("criterion", criterion), ("metric", metric), ("scope", scope))
        for column, suffix in (("actual", ""), ("upper_limit", ".upper_limit"), ("lower_limit", ".lower_limit")):
            if row.get(column, "") == "":
                continue
            records.append(
                EvidenceRecord(
                    record_id=base + suffix,
                    checkpoint=checkpoint,
                    generation=_generation(policy, checkpoint),
                    policy_code=policy,
                    source_path=path,
                    selector=selector,
                    field=column,
                    metric=f"criterion {criterion} {name}",
                    unit=unit,
                    aggregation=AGGREGATION_OF_SCOPE.get(scope, "all_eligible") if scope != "peak" else "peak_window",
                    population_id=population,
                    comparator=row.get("comparator") or None,
                    evidence_class=EVIDENCE_CLASS_DEVELOPMENT,
                    window=window,
                    display_precision=5 if column != "actual" and unit == UNIT_RATIO else precision,
                    raw=row[column],
                    source_line=line,
                )
            )
        status_column = "status" if "status" in row else "passed"
        records.append(
            EvidenceRecord(
                record_id=base + ".status",
                checkpoint=checkpoint,
                generation=_generation(policy, checkpoint),
                policy_code=policy,
                source_path=path,
                selector=selector,
                field=status_column,
                metric=f"criterion {criterion} status",
                unit=UNIT_LABEL,
                aggregation="criterion",
                population_id=population,
                comparator=row.get("comparator") or None,
                evidence_class=EVIDENCE_CLASS_DEVELOPMENT,
                window=window,
                display_precision=0,
                raw=row[status_column],
                source_line=line,
            )
        )
    return records


def _diagnostic_records(prefix: str, checkpoint: str, path: str) -> list[EvidenceRecord]:
    records = []
    for line, row in _rows(path):
        scope, policy, fold = row["scope"], row["policy"], row["fold"]
        if scope == "peak":
            key, fields = "peak", PEAK_FIELDS
        elif scope == "hour":
            key, fields = f"{fold}.hour_{int(row['group']):02d}", ("MAE", "WIS", "n_hours")
        else:
            continue
        for column in fields:
            metric, unit, precision = FIELD_SPECS[column]
            records.append(
                EvidenceRecord(
                    record_id=f"{prefix}.diagnostics.{policy}.{key}.{column}",
                    checkpoint=checkpoint,
                    generation=_generation(policy, checkpoint),
                    policy_code=policy,
                    source_path=path,
                    selector=(("policy", policy), ("fold", fold), ("scope", scope), ("group", row["group"])),
                    field=column,
                    metric=metric,
                    unit=unit,
                    aggregation=AGGREGATION_OF_SCOPE[scope],
                    population_id=_population(scope, fold),
                    comparator=None,
                    evidence_class=EVIDENCE_CLASS_DEVELOPMENT,
                    window=(row["first_delivery_date"], row["last_delivery_date"]),
                    display_precision=precision,
                    raw=row[column],
                    source_line=line,
                    detail=(("row_evidence_class", row.get("evidence_class", "")),
                            ("support_status", row.get("support_status", ""))),
                )
            )
    return records


def _cp15_records() -> list[EvidenceRecord]:
    records = []
    pooled = _rows("reports/cp15/pooled.csv")
    window_of = {row["policy"]: (row["first_delivery_date"], row["last_delivery_date"]) for _, row in pooled}
    for line, row in pooled:
        for column in ROW_FIELDS:
            metric, unit, precision = FIELD_SPECS[column]
            records.append(EvidenceRecord(
                record_id=f"cp15.pooled.{row['policy']}.{column}", checkpoint="CP-15",
                generation=GENERATION_OF.get(row["policy"]), policy_code=row["policy"],
                source_path="reports/cp15/pooled.csv", selector=(("policy", row["policy"]),),
                field=column, metric=metric, unit=unit, aggregation="pooled",
                population_id="common-10747h", comparator=None,
                evidence_class=EVIDENCE_CLASS_DEVELOPMENT, window=window_of[row["policy"]],
                display_precision=precision, raw=row[column], source_line=line,
            ))
    for line, row in _rows("reports/cp15/per_fold.csv"):
        for column in ROW_FIELDS:
            metric, unit, precision = FIELD_SPECS[column]
            records.append(EvidenceRecord(
                record_id=f"cp15.per_fold.{row['policy']}.{row['fold']}.{column}", checkpoint="CP-15",
                generation=GENERATION_OF.get(row["policy"]), policy_code=row["policy"],
                source_path="reports/cp15/per_fold.csv",
                selector=(("policy", row["policy"]), ("fold", row["fold"])),
                field=column, metric=metric, unit=unit, aggregation="per_fold",
                population_id=f"common-10747h/{row['fold']}", comparator=None,
                evidence_class=EVIDENCE_CLASS_DEVELOPMENT,
                window=(row["first_delivery_date"], row["last_delivery_date"]),
                display_precision=precision, raw=row[column], source_line=line,
            ))
    for line, row in _rows("reports/cp15/relative_scores.csv"):
        for column in SCORE_FIELDS:
            metric, unit, precision = FIELD_SPECS[column]
            records.append(EvidenceRecord(
                record_id=f"cp15.relative_scores.{row['policy']}.{column}", checkpoint="CP-15",
                generation=GENERATION_OF.get(row["policy"]), policy_code=row["policy"],
                source_path="reports/cp15/relative_scores.csv", selector=(("policy", row["policy"]),),
                field=column, metric=metric, unit=unit, aggregation="equal_fold",
                population_id="common-10747h", comparator="B0",
                evidence_class=EVIDENCE_CLASS_DEVELOPMENT, window=window_of[row["policy"]],
                display_precision=precision, raw=row[column], source_line=line,
            ))
    for line, row in _rows("reports/cp15/peak.csv"):
        for column in PEAK_FIELDS:
            metric, unit, precision = FIELD_SPECS[column]
            records.append(EvidenceRecord(
                record_id=f"cp15.peak.{row['policy']}.{column}", checkpoint="CP-15",
                generation=GENERATION_OF.get(row["policy"]), policy_code=row["policy"],
                source_path="reports/cp15/peak.csv", selector=(("policy", row["policy"]),),
                field=column, metric=metric, unit=unit, aggregation="peak_window",
                population_id="crisis-window-408h", comparator=None,
                evidence_class=EVIDENCE_CLASS_DEVELOPMENT,
                window=(row["first_delivery_date"], row["last_delivery_date"]),
                display_precision=precision, raw=row[column], source_line=line,
                detail=(("row_evidence_class", row["evidence_class"]),),
            ))
    fold_windows = {}
    for _, row in _rows("reports/cp15/per_fold.csv"):
        fold_windows[row["fold"]] = (row["first_delivery_date"], row["last_delivery_date"])
    fold_windows["all"] = window_of["B0"]
    interval = _interval("CP-15")
    for line, row in _rows("reports/cp15/bootstrap.csv"):
        scope, candidate, baseline, metric = row["scope"], row["candidate"], row["baseline"], row["metric"]
        equal = scope == "equal_fold"
        records.append(EvidenceRecord(
            record_id=f"cp15.bootstrap.{candidate}-{baseline}.{scope}.{metric}", checkpoint="CP-15",
            generation=GENERATION_OF.get(candidate), policy_code=candidate,
            source_path="reports/cp15/bootstrap.csv",
            selector=(("scope", scope), ("candidate", candidate), ("baseline", baseline), ("metric", metric)),
            field="difference", metric=f"delta_{'S_' if equal else ''}{metric}",
            unit=UNIT_NORM_DIFF if equal else UNIT_EUR_DIFF,
            aggregation="equal_fold_contrast" if equal else "per_fold_contrast",
            population_id="common-10747h" if equal else f"common-10747h/{scope}", comparator=baseline,
            evidence_class=EVIDENCE_CLASS_DEVELOPMENT, window=fold_windows["all" if equal else scope],
            display_precision=4 if equal else 2, raw=row["difference"], source_line=line,
            interval=Interval(interval.method, interval.level, interval.seed, int(row["replicates"]), interval.block_days),
            ci_low_raw=row["ci_lower"], ci_high_raw=row["ci_upper"],
        ))
    for line, row in _rows("reports/cp15/ranking.csv"):
        for column in ("rank", "failed_criteria", "product_feasibility"):
            records.append(EvidenceRecord(
                record_id=f"cp15.ranking.{row['policy']}.{column}", checkpoint="CP-15",
                generation=None, policy_code=row["policy"], source_path="reports/cp15/ranking.csv",
                selector=(("policy", row["policy"]),), field=column, metric=column,
                unit=UNIT_COUNT if column == "rank" else UNIT_LABEL, aggregation="equal_fold",
                population_id="common-10747h", comparator="B0",
                evidence_class=EVIDENCE_CLASS_DEVELOPMENT, window=window_of[row["policy"]],
                display_precision=0, raw=row[column], source_line=line,
            ))
    return records


def _cp10_records() -> list[EvidenceRecord]:
    records = []
    for line, row in _rows("reports/cp10/metrics.csv"):
        for column in ("mae", "mean_pinball", "coverage_50", "coverage_80", "coverage_95", "covered_95", "n_obs"):
            metric, unit, precision = FIELD_SPECS[column]
            records.append(EvidenceRecord(
                record_id=f"cp10.metrics.{row['candidate']}.{row['fold']}.{column}", checkpoint="CP-10",
                generation=None, policy_code=row["candidate"], source_path="reports/cp10/metrics.csv",
                selector=(("candidate", row["candidate"]), ("fold", row["fold"])), field=column,
                metric=metric, unit=unit, aggregation="per_fold", population_id="cp10-fold-block",
                comparator=None, evidence_class=EVIDENCE_CLASS_CALIBRATION,
                window=(row["window_start"], row["window_end"]), display_precision=precision,
                raw=row[column], source_line=line,
            ))
    for line, row in _rows("reports/cp10/peak_windows.csv"):
        for column in ("mae", "mean_pinball", "coverage_95", "covered_95", "n_obs"):
            metric, unit, precision = FIELD_SPECS[column]
            records.append(EvidenceRecord(
                record_id=f"cp10.peak_windows.{row['candidate']}.{column}", checkpoint="CP-10",
                generation=None, policy_code=row["candidate"], source_path="reports/cp10/peak_windows.csv",
                selector=(("candidate", row["candidate"]),), field=column, metric=metric, unit=unit,
                aggregation="peak_window", population_id="cp10-crisis-window-408h", comparator=None,
                evidence_class=EVIDENCE_CLASS_CALIBRATION, window=(row["window_start"], row["window_end"]),
                display_precision=precision, raw=row[column], source_line=line,
            ))
    return records


def _json_record(record_id, checkpoint, path, dotted, *, unit, precision, population, window=None,
                 evidence_class=EVIDENCE_CLASS_DEVELOPMENT, metric=None, generation=None) -> EvidenceRecord:
    value = _json_get(_json(path), dotted)
    return EvidenceRecord(
        record_id=record_id, checkpoint=checkpoint, generation=generation, policy_code=None,
        source_path=path, selector=(("json", dotted),), field=dotted.split(".")[-1],
        metric=metric or dotted.split(".")[-1], unit=unit, aggregation="record",
        population_id=population, comparator=None, evidence_class=evidence_class,
        window=window, display_precision=precision, raw=_json_text(value),
    )


def _json_text(value) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def _json_records() -> list[EvidenceRecord]:
    ex = "reports/weather-ablation/extraction-summary.json"
    rf = "docs/track-b/evidence/cp-20/resource-final.json"
    cp20 = "reports/weather-ablation/protocol.json"
    cp15 = "reports/cp15/bootstrap_metadata.json"
    sel10 = "reports/cp10/selection.json"
    dm2 = "reports/cp2/dm_development.json"
    development = next((row["first_delivery_date"], row["last_delivery_date"])
                       for _, row in _rows("reports/cp15/pooled.csv") if row["policy"] == "B1")
    records = [
        _json_record("cp20.extraction.complete_runs", "CP-20", ex, "complete_runs", unit=UNIT_RUNS, precision=0,
                     population="cp20-extraction"),
        _json_record("cp20.extraction.target_messages", "CP-20", ex, "target_messages", unit=UNIT_MESSAGES,
                     precision=0, population="cp20-extraction"),
        _json_record("cp20.resources.transfer_gib", "CP-20", rf, "transfer_gib", unit=UNIT_GIB, precision=1,
                     population="cp20-resources"),
        _json_record("cp20.resources.machine_hours", "CP-20", rf, "machine_hours", unit=UNIT_MACHINE_HOURS,
                     precision=2, population="cp20-resources"),
        _json_record("cp20.resources.external_cost_usd", "CP-20", rf, "external_cost_usd", unit=UNIT_USD,
                     precision=0, population="cp20-resources"),
        _json_record("cp20.protocol.bootstrap_seed", "CP-20", cp20, "comparison.bootstrap_seed", unit=UNIT_SEED,
                     precision=0, population="protocol"),
        _json_record("cp20.protocol.replicates", "CP-20", cp20, "comparison.replicates", unit=UNIT_REPLICATES,
                     precision=0, population="protocol"),
        _json_record("cp20.protocol.block_days", "CP-20", cp20, "comparison.block_days", unit=UNIT_DAYS,
                     precision=0, population="protocol"),
        _json_record("cp15.bootstrap_metadata.seed", "CP-15", cp15, "seed", unit=UNIT_SEED, precision=0,
                     population="protocol"),
        _json_record("cp15.bootstrap_metadata.replicates", "CP-15", cp15, "replicates", unit=UNIT_REPLICATES,
                     precision=0, population="protocol"),
        _json_record("cp15.bootstrap_metadata.block_days", "CP-15", cp15, "block_days", unit=UNIT_DAYS,
                     precision=0, population="protocol"),
        _json_record("cp10.selection.v1_coverage_95", "CP-10", sel10, "same_window_diagnostic.v1_coverage_95",
                     unit=UNIT_FRACTION, precision=4, population="cp10-crisis-window-408h",
                     window=("2022-08-15", "2022-08-31"), evidence_class=EVIDENCE_CLASS_CALIBRATION),
        _json_record("cp10.selection.selected_coverage_95", "CP-10", sel10,
                     "same_window_diagnostic.selected_coverage_95", unit=UNIT_FRACTION, precision=4,
                     population="cp10-crisis-window-408h", window=("2022-08-15", "2022-08-31"),
                     evidence_class=EVIDENCE_CLASS_CALIBRATION),
    ]
    cp21 = "reports/block-challenger/protocol.json"
    adoption = "reports/block-challenger/adoption.json"
    for key, dotted, unit in (("bootstrap_seed", "uncertainty.seed", UNIT_SEED),
                              ("replicates", "uncertainty.replicates", UNIT_REPLICATES),
                              ("block_days", "uncertainty.block_days", UNIT_DAYS)):
        records.append(_json_record(f"cp21.protocol.{key}", "CP-21", cp21, dotted, unit=unit, precision=0,
                                    population="protocol"))
    for key, dotted in (("verdict", "verdict"), ("candidate", "candidate"), ("comparator", "comparator"),
                        ("rule_id", "rule.id"), ("rule_set_on", "rule.set_on"),
                        ("first_unmet_condition", "first_unmet_condition"),
                        ("block_split_reading", "block_split.reading")):
        records.append(_json_record(f"cp21.adoption.{key}", "CP-21", adoption, dotted, unit=UNIT_LABEL, precision=0,
                                    population="protocol"))
    for condition in ("1", "2", "3", "4"):
        records.append(_json_record(f"cp21.adoption.condition_{condition}_met", "CP-21", adoption,
                                    f"conditions.{condition}.met", unit=UNIT_LABEL, precision=0,
                                    population="protocol"))
    records += [
        _json_record("cp21.controls.checks", "CP-21", "reports/block-challenger/controls.json", "checks",
                     unit=UNIT_COUNT, precision=0, population="protocol"),
        _json_record("cp21.controls.all_passed", "CP-21", "reports/block-challenger/controls.json", "all_passed",
                     unit=UNIT_LABEL, precision=0, population="protocol"),
        _json_record("cp21.hg_parity.rows", "CP-21", "reports/block-challenger/hg-parity.json", "rows",
                     unit=UNIT_HOURS, precision=0, population="common-10747h"),
        _json_record("cp21.hg_parity.bitwise_equal", "CP-21", "reports/block-challenger/hg-parity.json",
                     "bitwise_equal", unit=UNIT_LABEL, precision=0, population="common-10747h"),
        _json_record("cp21.fit_cost.fits", "CP-21", "reports/block-challenger/fit-cost.json", "fits",
                     unit=UNIT_RUNS, precision=0, population="protocol", metric="main-run LightGBM fits"),
    ]
    # CP-24 (v5): its mechanical decision, its bootstrap settings, its controls and its full-coverage leakage
    # controls (capstone v21-r11 §23.9-§23.10; claim map C421, C439, C441).
    decisions, controls, leakage = CP24_ROWS["decisions"], CP24_ROWS["controls"], CP24_ROWS["leakage"]
    for key, dotted in (("adopted", "adoption.adopted"), ("candidate", "adoption.candidate"),
                        ("comparator", "adoption.comparator"), ("first_unmet_condition", "adoption.first_unmet_condition")):
        records.append(_json_record(f"cp24.adoption.{key}", "CP-24", decisions, dotted, unit=UNIT_LABEL, precision=0,
                                    population="protocol"))
    for condition in ("1", "2", "3", "4", "5"):
        records.append(_json_record(f"cp24.adoption.condition_{condition}_met", "CP-24", decisions,
                                    f"adoption.conditions.{condition}.met", unit=UNIT_LABEL, precision=0,
                                    population="protocol"))
    for key, dotted, unit in (("bootstrap_seed", "bootstrap.seed", UNIT_SEED),
                              ("replicates", "bootstrap.replicates", UNIT_REPLICATES),
                              ("block_days", "bootstrap.block_days", UNIT_DAYS)):
        records.append(_json_record(f"cp24.protocol.{key}", "CP-24", decisions, dotted, unit=unit, precision=0,
                                    population="protocol"))
    records += [
        _json_record("cp24.protocol.rule_id", "CP-24", CP24_ROWS["protocol"], "rule", unit=UNIT_LABEL, precision=0,
                     population="protocol"),
        _json_record("cp24.controls.checks", "CP-24", controls, "checks", unit=UNIT_COUNT, precision=0,
                     population="protocol"),
        _json_record("cp24.controls.all_passed", "CP-24", controls, "all_passed", unit=UNIT_LABEL, precision=0,
                     population="protocol"),
        _json_record("cp24.leakage.origins", "CP-24", leakage, "blind.origins", unit=UNIT_COUNT, precision=0,
                     population="protocol", metric="warm-up and evaluation origins refitted with the future destroyed"),
        _json_record("cp24.leakage.bitwise", "CP-24", leakage, "blind.bitwise", unit=UNIT_COUNT, precision=0,
                     population="protocol", metric="origins bit-identical to the committed vectors"),
        _json_record("cp24.leakage.keys", "CP-24", leakage, "blind.committed_predictions.keys_compared",
                     unit=UNIT_HOURS, precision=0, population="common-10747h", metric="evaluation keys compared"),
        _json_record("cp24.leakage.all_passed", "CP-24", leakage, "all_passed", unit=UNIT_LABEL, precision=0,
                     population="protocol"),
        _json_record("cp24.leakage.d1_mutation_moves", "CP-24", leakage, "checks.d1_positive_moves_every_pick",
                     unit=UNIT_LABEL, precision=0, population="protocol"),
        _json_record("cp24.leakage.planted_leak_caught", "CP-24", leakage, "checks.planted_leak_detected_in_every_fold",
                     unit=UNIT_LABEL, precision=0, population="protocol"),
    ]
    point = next(i for i, t in enumerate(_json(dm2)["dm_tests"])
                 if t["analysis"] == "point_median_absolute_error" and t["comparator"] == "similar_day_naive")
    for key, unit, precision in (("relative_improvement_pct", UNIT_PERCENT, 2), ("n_days", UNIT_DAYS, 0),
                                 ("mean_loss_differential", UNIT_EUR, 3), ("p_value", UNIT_LABEL, 3)):
        records.append(_json_record(f"cp2.dm_development.point_vs_naive.{key}", "CP-2", dm2,
                                    f"dm_tests.{point}.{key}", unit=unit, precision=precision,
                                    population="v1-development", window=development))
    for line, row in _rows("reports/cp2/development_pooled_metrics.csv"):
        records.append(EvidenceRecord(
            record_id=f"cp2.development_pooled_metrics.{row['model']}.{row['stage']}.MAE", checkpoint="CP-2",
            generation="v1" if row["model"] == "base" else None, policy_code=row["model"],
            source_path="reports/cp2/development_pooled_metrics.csv",
            selector=(("model", row["model"]), ("stage", row["stage"])), field="mae_p50",
            metric="MAE", unit=UNIT_EUR, aggregation="pooled", population_id="v1-development",
            comparator=None, evidence_class=EVIDENCE_CLASS_DEVELOPMENT, window=development,
            display_precision=2, raw=row["mae_p50"], source_line=line,
        ))
    return records


#: The evidence class of an explanation computed on rows the explained model was fitted on: a diagnostic of
#: the frozen artifact, not an out-of-sample result (PUBLISH_RULES 1.0 §5.1, subject 6).
EVIDENCE_CLASS_IN_SAMPLE = "in_sample_diagnostic"
#: v1's one-shot holdout, verbatim in its badge (plan §7.8); the class the frozen artifact's own test carries.
EVIDENCE_CLASS_V1_HOLDOUT = "confirmatory_style_not_power_qualified"
UNIT_CORRELATION = "Spearman rank correlation"

#: CP-2's regime-table strata that describe the released recipe's development folds, by a stable key; the
#: committed label is the row selector. The December-2024 backtest row (a stale fold-3 model's raw heads,
#: outside every evaluation block) is not a product result and has no record here.
CP2_STRATA = (
    ("all", "all validation folds", None),
    ("pre_crisis", "pre-crisis (< 2021-09-01)", None),
    ("crisis", "crisis (2021-09-01 .. 2022-12-31)", None),
    ("post_crisis", "post-crisis (>= 2023-01-01)", None),
    ("negative_price", "negative-price hours", None),
    ("dunkelflaute", "Dunkelflaute days (A69 stratum)", None),
    ("non_flag", "non-flag days", None),
    ("weekday", "weekday", None),
    ("weekend", "weekend", None),
    ("august_2022_peak", "August-2022 peak weeks", ("2022-08-15", "2022-08-31")),
)
CP2_REGIME_FIELDS = ("n_obs", "n_days", "mae", "mean_pinball", "coverage_50", "coverage_80", "coverage_95",
                     "mae_ci95_low", "mae_ci95_high", "read")
#: The three calibration stages, as CP-2's reliability table and holdout report name them.
CP2_STAGES = (("raw", "raw LightGBM", "raw_coverage"), ("post_cqr", "post-CQR / pre-isotonic", "post_cqr_coverage"),
              ("final", "final post-isotonic", "final_coverage"))
CP2_LEVELS = (("50", "0.5"), ("80", "0.8"), ("95", "0.95"))
V1_HOLDOUT_WINDOW = ("2026-06-09", "2026-09-06")
V1_FOLD5_WINDOW = ("2026-01-08", "2026-04-07")


def _cp2_product_records() -> list[EvidenceRecord]:
    """The released product's own evaluation and diagnostics, each with the model, output and rows it
    describes: the frozen artifact on its one-shot holdout; the released recipe's development folds (fold
    models, calibrated per fold); fold 5's development model on its own evaluation block (out of sample); and
    the frozen artifact explained on those same rows, which it was fitted on (in sample)."""
    development = next((row["first_delivery_date"], row["last_delivery_date"])
                       for _, row in _rows("reports/cp15/pooled.csv") if row["policy"] == "B1")
    out: list[EvidenceRecord] = []
    holdout = "reports/cp2/holdout_report.json"
    for stage, _, key in CP2_STAGES:
        for level, _ in CP2_LEVELS:
            out.append(_json_record(f"cp2.holdout.coverage.{stage}.{level}", "CP-2", holdout, f"{key}.{level}",
                                    unit=UNIT_FRACTION, precision=4, population="v1-holdout-90d",
                                    window=V1_HOLDOUT_WINDOW, evidence_class=EVIDENCE_CLASS_V1_HOLDOUT,
                                    metric=f"{level}% coverage, {stage}", generation="v1"))
    for record_id, dotted, unit in (("cp2.holdout.n_days", "n_holdout_days", UNIT_DAYS),
                                    ("cp2.holdout.n_hours", "n_holdout_rows", UNIT_HOURS)):
        out.append(_json_record(record_id, "CP-2", holdout, dotted, unit=unit, precision=0,
                                population="v1-holdout-90d", window=V1_HOLDOUT_WINDOW,
                                evidence_class=EVIDENCE_CLASS_V1_HOLDOUT, generation="v1"))
    regime = "reports/cp2/regime_table.csv"
    by_label = {row["stratum"]: (line, row) for line, row in _rows(regime)}
    for key, label, window in CP2_STRATA:
        if label not in by_label:
            raise EvidenceError(f"{regime} has no stratum {label!r}")
        line, row = by_label[label]
        for column in CP2_REGIME_FIELDS:
            if row.get(column, "") == "":
                continue
            metric, unit, precision = FIELD_SPECS[column] if column != "read" else ("reading", UNIT_LABEL, 0)
            out.append(EvidenceRecord(
                record_id=f"cp2.regime.{key}.{column}", checkpoint="CP-2", generation="v1", policy_code="base",
                source_path=regime, selector=(("stratum", label),), field=column, metric=metric, unit=unit,
                aggregation="stratum", population_id=f"v1-development/{key}", comparator=None,
                evidence_class=EVIDENCE_CLASS_DEVELOPMENT, window=window or development,
                display_precision=precision, raw=row[column], source_line=line))
    reliability = "reports/cp2/reliability_three_stage.csv"
    cells = {(row["stage"], row["nominal"]): (line, row) for line, row in _rows(reliability)}
    for stage, label, _ in CP2_STAGES:
        for level, nominal in CP2_LEVELS:
            line, row = cells[(label, nominal)]
            out.append(EvidenceRecord(
                record_id=f"cp2.reliability.{stage}.{level}", checkpoint="CP-2", generation="v1", policy_code="base",
                source_path=reliability, selector=(("stage", label), ("nominal", nominal)), field="empirical",
                metric=f"{level}% empirical coverage, {stage}", unit=UNIT_FRACTION, aggregation="pooled",
                population_id="v1-development", comparator=None, evidence_class=EVIDENCE_CLASS_DEVELOPMENT,
                window=development, display_precision=4, raw=row["empirical"], source_line=line))
    diagnostics = "reports/cp2/diagnostics.json"
    document = _json(diagnostics)
    for family, key, evidence_class, fields in (
        ("frozen_shap", "frozen_champion_shap_top10_in_sample", EVIDENCE_CLASS_IN_SAMPLE, ("feature", "mean_abs_shap")),
        ("fold5_shap", "shap_top10", EVIDENCE_CLASS_DEVELOPMENT, ("feature", "mean_abs_shap")),
        ("permutation", "permutation_top10", EVIDENCE_CLASS_DEVELOPMENT,
         ("feature", "importance_mean_mae_increase", "importance_std")),
    ):
        for index, item in enumerate(document[key]):
            if int(item["rank"]) != index + 1:
                raise EvidenceError(f"{diagnostics}: {key}[{index}] is rank {item['rank']}, not {index + 1}")
            for field_name in fields:
                unit, precision = (UNIT_LABEL, 0) if field_name == "feature" else (UNIT_EUR, 2)
                out.append(_json_record(f"cp2.diagnostics.{family}.{index + 1}.{field_name}", "CP-2", diagnostics,
                                        f"{key}.{index}.{field_name}", unit=unit, precision=precision,
                                        population="v1-fold5-eval", window=V1_FOLD5_WINDOW,
                                        evidence_class=evidence_class, generation="v1"))
    for record_id, dotted, unit, precision, evidence_class in (
        ("cp2.diagnostics.frozen_vs_fold5.rank_spearman", "frozen_vs_fold5_shap_agreement.rank_spearman",
         UNIT_CORRELATION, 2, EVIDENCE_CLASS_IN_SAMPLE),
        ("cp2.diagnostics.frozen_vs_fold5.top10_overlap", "frozen_vs_fold5_shap_agreement.top10_overlap",
         UNIT_COUNT, 0, EVIDENCE_CLASS_IN_SAMPLE),
        ("cp2.diagnostics.shap_vs_permutation.rank_spearman", "shap_vs_permutation_rank_spearman",
         UNIT_CORRELATION, 2, EVIDENCE_CLASS_DEVELOPMENT),
    ):
        out.append(_json_record(record_id, "CP-2", diagnostics, dotted, unit=unit, precision=precision,
                                population="v1-fold5-eval", window=V1_FOLD5_WINDOW,
                                evidence_class=evidence_class, generation="v1"))
    return out


#: The committed rows of the checkpoints PRES-4 publishes, by role.
CP22_ROWS = {name: f"reports/v4-revision/{name}.{ext}" for name, ext in (
    ("metrics", "csv"), ("uncertainty", "csv"), ("criteria", "csv"), ("diagnostics", "csv"), ("protocol", "json"),
    ("decisions", "json"))}
CP23_ROWS = {name: f"reports/distribution-challenger/{name}.{ext}" for name, ext in (
    ("metrics", "csv"), ("uncertainty", "csv"), ("criteria", "csv"), ("diagnostics", "csv"), ("protocol", "json"),
    ("decisions", "json"))}
CP24_ROWS = {name: f"reports/ddnn2/attempt-1/{file}" for name, file in (
    ("metrics", "metrics.csv"), ("uncertainty", "uncertainty.csv"), ("ratios", "ratio-intervals.csv"),
    ("criteria", "criteria.csv"), ("diagnostics", "diagnostics.csv"), ("protocol", "protocol.json"),
    ("decisions", "decisions.json"), ("controls", "controls.json"), ("leakage", "leakage-controls.json"))}


def _checkpoint_records(prefix: str, checkpoint: str, rows: dict[str, str]) -> list[EvidenceRecord]:
    """One later checkpoint's committed rows, by the same builders as CP-21's: scores, paired differences, ratios,
    the original screening criteria and the diagnostics."""
    windows = _fold_windows(rows["metrics"])
    return (_metric_records(prefix, checkpoint, rows["metrics"])
            + _uncertainty_records(prefix, checkpoint, rows["uncertainty"], windows)
            + _ratio_records(prefix, checkpoint, rows["uncertainty"], windows)
            + _criteria_records(prefix, checkpoint, rows["criteria"], windows)
            + _diagnostic_records(prefix, checkpoint, rows["diagnostics"]))


@lru_cache(maxsize=1)
def records() -> dict[str, EvidenceRecord]:
    """Every published evidence record, keyed by `record_id`."""
    cp20_windows = _fold_windows("reports/weather-ablation/metrics.csv")
    cp16_windows = _fold_windows("reports/v2-causal/metrics.csv")
    cp21_windows = _fold_windows("reports/block-challenger/metrics.csv")
    built: list[EvidenceRecord] = []
    built += _checkpoint_records("cp24", "CP-24", CP24_ROWS)
    built += _decision_level_records("cp24", "CP-24", CP24_ROWS["uncertainty"], CP24_ROWS["ratios"],
                                     _fold_windows(CP24_ROWS["metrics"]))
    built += _checkpoint_records("cp23", "CP-23", CP23_ROWS)
    built += _checkpoint_records("cp22", "CP-22", CP22_ROWS)
    built += _metric_records("cp21", "CP-21", "reports/block-challenger/metrics.csv")
    built += _uncertainty_records("cp21", "CP-21", "reports/block-challenger/uncertainty.csv", cp21_windows)
    built += _ratio_records("cp21", "CP-21", "reports/block-challenger/uncertainty.csv", cp21_windows)
    built += _criteria_records("cp21", "CP-21", "reports/block-challenger/criteria.csv", cp21_windows)
    built += _diagnostic_records("cp21", "CP-21", "reports/block-challenger/diagnostics.csv")
    built += _metric_records("cp20", "CP-20", "reports/weather-ablation/metrics.csv")
    built += _uncertainty_records("cp20", "CP-20", "reports/weather-ablation/uncertainty.csv", cp20_windows)
    built += _criteria_records("cp20", "CP-20", "reports/weather-ablation/criteria.csv", cp20_windows)
    built += _diagnostic_records("cp20", "CP-20", "reports/weather-ablation/diagnostics.csv")
    built += _metric_records("cp16", "CP-16", "reports/v2-causal/metrics.csv")
    built += _uncertainty_records("cp16", "CP-16", "reports/v2-causal/uncertainty.csv", cp16_windows)
    built += _criteria_records("cp16", "CP-16", "reports/v2-causal/criteria.csv", cp16_windows)
    built += _diagnostic_records("cp16", "CP-16", "reports/v2-causal/diagnostics.csv")
    built += _cp15_records()
    cp15_windows = {row["fold"]: (row["first_delivery_date"], row["last_delivery_date"])
                    for _, row in _rows("reports/cp15/per_fold.csv")}
    cp15_windows["all"] = cp20_windows["all"]
    built += _criteria_records("cp15", "CP-15", "reports/cp15/criteria.csv", cp15_windows)
    built += _cp10_records()
    built += _json_records()
    built += _cp2_product_records()
    out: dict[str, EvidenceRecord] = {}
    for record in built:
        if record.record_id in out:
            raise EvidenceError(f"duplicate record id {record.record_id}")
        out[record.record_id] = record
    return out


def get(record_id: str) -> EvidenceRecord:
    try:
        return records()[record_id]
    except KeyError:
        raise EvidenceError(f"no evidence record {record_id!r}") from None


# --------------------------------------------------------------------------- daily series


@dataclass(frozen=True)
class DailyPoint:
    delivery_date: str
    fold: str
    n_hours: int
    raw: str
    source_line: int

    @property
    def value(self) -> float:
        return float(self.raw)


#: Where each policy's committed daily rows are, by the checkpoint that produced the policy.
DAILY_SOURCES = {
    "CP-15": ("reports/cp15/daily.csv", None),
    "CP-16": ("reports/v2-causal/diagnostics.csv", "daily"),
    "CP-20": ("reports/weather-ablation/diagnostics.csv", "daily"),
    "CP-21": ("reports/block-challenger/diagnostics.csv", "daily"),
}


def daily_series(checkpoint: str, policy: str, column: str = "MAE") -> tuple[DailyPoint, ...]:
    """A policy's committed daily rows, in delivery order. Days with no eligible hour carry no
    loss and are left out rather than written as zero."""
    path, scope = DAILY_SOURCES[checkpoint]
    points = []
    for line, row in _rows(path):
        if row["policy"] != policy or (scope and row["scope"] != scope):
            continue
        if int(float(row["n_hours"])) == 0 or row[column] == "":
            continue
        points.append(DailyPoint(row["delivery_date"], row["fold"], int(float(row["n_hours"])), row[column], line))
    points.sort(key=lambda point: point.delivery_date)
    if not points:
        raise EvidenceError(f"no daily rows for {policy} in {path}")
    return tuple(points)


# --------------------------------------------------------------------------- validation


def _expected_unit(record: EvidenceRecord) -> str:
    if record.unit == UNIT_LABEL or record.selector[0][0] == "json":
        return record.unit
    if record.field == "difference":
        return UNIT_NORM_DIFF if record.selector_dict()["scope"] == "equal_fold" else UNIT_EUR_DIFF
    if record.field == "ratio":
        return UNIT_RATIO_CHANGE
    column = record.field
    if record.field in ("actual", "upper_limit", "lower_limit"):
        column = record.selector_dict()["metric"]
    spec = FIELD_SPECS.get(column)
    return spec[1] if spec else UNIT_FRACTION


def _expected_aggregation(record: EvidenceRecord) -> str | None:
    selector = record.selector_dict()
    if record.field == "difference":
        return "equal_fold_contrast" if selector["scope"] == "equal_fold" else "per_fold_contrast"
    if record.field == "ratio":
        return "equal_fold_ratio"
    if "scope" in selector and record.field not in ("actual", "upper_limit", "lower_limit", "status", "passed"):
        return AGGREGATION_OF_SCOPE[selector["scope"]]
    return None


class FreshRead:
    """One fresh read of the committed sources, independent of the registry's caches.

    Each file is read and hashed once, and its rows are indexed once per selector shape, so
    re-deriving thousands of records costs one pass over each file rather than one per record.
    """

    def __init__(self) -> None:
        self._checked: set[str] = set()
        self._rows: dict[str, list[dict[str, str]]] = {}
        self._docs: dict[str, dict] = {}
        self._index: dict[tuple[str, tuple[str, ...]], dict[tuple[str, ...], list[dict[str, str]]]] = {}

    def check(self, path: str) -> None:
        if path not in self._checked:
            check_source(path)
            self._checked.add(path)

    def document(self, path: str) -> dict:
        self.check(path)
        if path not in self._docs:
            self._docs[path] = json.loads(source_bytes(path))
        return self._docs[path]

    def matches(self, path: str, selector: dict[str, str]) -> list[dict[str, str]]:
        self.check(path)
        if path not in self._rows:
            self._rows[path] = [row for _, row in _csv(path)]
        keys = tuple(sorted(selector))
        index = self._index.get((path, keys))
        if index is None:
            index = {}
            for row in self._rows[path]:
                index.setdefault(tuple(row.get(key) for key in keys), []).append(row)
            self._index[(path, keys)] = index
        return index.get(tuple(selector[key] for key in keys), [])


def rederive(record: EvidenceRecord, fresh: FreshRead | None = None) -> dict[str, str]:
    """Read the record's cells afresh from the committed file: value and interval endpoints."""
    fresh = fresh or FreshRead()
    selector = record.selector_dict()
    if "json" in selector:
        return {"value": _json_text(_json_get(fresh.document(record.source_path), selector["json"]))}
    matches = fresh.matches(record.source_path, selector)
    if len(matches) != 1:
        raise EvidenceError(f"{record.record_id}: selector {selector} matches {len(matches)} rows")
    row = matches[0]
    if record.field not in row:
        raise EvidenceError(f"{record.record_id}: no column {record.field!r}")
    out = {"value": row[record.field]}
    if record.interval is not None:
        low, high = record.ci_columns
        out["ci_low"], out["ci_high"] = row[low], row[high]
    return out


def validate(record: EvidenceRecord, fresh: FreshRead | None = None) -> None:
    """Raise `EvidenceError` unless the record still re-derives from its committed source row."""
    derived = rederive(record, fresh)
    if derived["value"] != record.raw:
        raise EvidenceError(f"{record.record_id}: committed value {derived['value']!r} != record {record.raw!r}")
    if record.interval is not None:
        if (derived["ci_low"], derived["ci_high"]) != (record.ci_low_raw, record.ci_high_raw):
            raise EvidenceError(f"{record.record_id}: interval endpoints differ from the committed row")
    expected_unit = _expected_unit(record)
    if record.unit != expected_unit:
        raise EvidenceError(f"{record.record_id}: unit {record.unit!r} but the source column is {expected_unit!r}")
    expected_aggregation = _expected_aggregation(record)
    if expected_aggregation is not None and record.aggregation != expected_aggregation:
        raise EvidenceError(
            f"{record.record_id}: aggregation {record.aggregation!r} but the source row is {expected_aggregation!r}"
        )
    selector = record.selector_dict()
    for key in ("policy", "candidate"):
        if key in selector and record.policy_code is not None and selector[key] != record.policy_code:
            raise EvidenceError(f"{record.record_id}: policy {record.policy_code!r} but the row is {selector[key]!r}")


def validate_all(candidates=None) -> list[str]:
    """Every record that fails to re-derive, as one message each. Empty means all hold."""
    fresh = FreshRead()
    problems = []
    for record in (candidates if candidates is not None else records().values()):
        try:
            validate(record, fresh)
        except EvidenceError as exc:
            problems.append(str(exc))
    return problems


# --------------------------------------------------------------------------- display


def exact(raw: str) -> str:
    """The full decimal expansion of a committed value, never in exponent form.

    `3.857628092332211e-06` -> `0.000003857628092332211`: the H−P endpoint is printed in full.
    """
    return format(Decimal(raw), "f")


def format_number(raw: str, precision: int, *, signed: bool = False, thousands: bool = False) -> str:
    value = Decimal(raw)
    quantized = round(value, precision)
    text = f"{abs(quantized):,.{precision}f}" if thousands else f"{abs(quantized):.{precision}f}"
    negative = quantized < 0
    if negative:
        return MINUS + text
    if signed and quantized > 0:
        return "+" + text
    return text


#: Differences are shown with an explicit sign: "+0.037" is a finding, "0.037" is ambiguous.
SIGNED_UNITS = {UNIT_NORM_DIFF, UNIT_EUR_DIFF}
THOUSANDS_UNITS = {UNIT_COUNT, UNIT_HOURS, UNIT_DAYS, UNIT_RUNS, UNIT_MESSAGES, UNIT_REPLICATES}


def display(record: EvidenceRecord, which: str = "value", *, precision: int | None = None,
            style: str | None = None) -> str:
    """The one display string for a record's value or interval endpoint.

    `style`: None (the record's own precision), "exact" (full expansion), "percent" (a fraction
    shown as a percentage), "int".
    """
    raw = {"value": record.raw, "ci_low": record.ci_low_raw, "ci_high": record.ci_high_raw}[which]
    if raw is None:
        raise EvidenceError(f"{record.record_id} has no {which}")
    if record.unit == UNIT_LABEL:
        return raw
    signed = record.unit in SIGNED_UNITS
    if style == "exact":
        text = exact(raw)
        if text.startswith("-"):
            return MINUS + text[1:]
        return ("+" + text) if signed and Decimal(raw) > 0 else text
    if style == "percent":
        if record.unit != UNIT_FRACTION:
            raise EvidenceError(f"{record.record_id}: percent style needs a fraction, not {record.unit}")
        return format_number(str(Decimal(raw) * 100), 2 if precision is None else precision) + "%"
    places = record.display_precision if precision is None else precision
    if style == "int":
        places = 0
    # Never round toward zero across a sign (standard §4): a value near zero keeps its sign and at
    # least two significant figures, whatever precision was asked for (+0.0000039, not +0.0000).
    if Decimal(raw) != 0 and round(Decimal(raw), places) == 0:
        places = max(places, _places_for_two_significant(raw))
    return format_number(raw, places, signed=signed, thousands=record.unit in THOUSANDS_UNITS)


def _places_for_two_significant(raw: str) -> int:
    """Decimal places that keep two significant digits of a small difference, so an interval
    endpoint just above zero is never displayed as zero (+0.037, not +0.04 or 0.0000)."""
    value = abs(Decimal(raw))
    if value == 0:
        return 0
    return max(0, -value.adjusted() + 1)


__all__ = [
    "DAILY_SOURCES",
    "EVIDENCE_BOUNDARY",
    "EvidenceError",
    "EvidenceRecord",
    "FreshRead",
    "FIELD_SPECS",
    "GENERATION_OF",
    "Interval",
    "MINUS",
    "POPULATIONS",
    "PRODUCT_SOURCES",
    "SOURCES",
    "Source",
    "registered_source",
    "blob_sha",
    "check_source",
    "daily_series",
    "display",
    "exact",
    "format_number",
    "get",
    "records",
    "rederive",
    "validate",
    "validate_all",
]
