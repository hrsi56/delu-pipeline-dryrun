"""Typed derived records: the headline quantities of Publication Standard v1 §3.3 (brief W3).

`delu_forecast.research` publishes committed cells exactly as saved. A headline also needs a few
quantities that no single cell holds: whether a policy met the pre-specified targets, how far it
sits from the targets' comparator, how many policies had been tested against the same rule, when
the rule was set, the change against a comparator as a share of the comparator's score, and the
range of absolute errors over the ordinary test periods. Each one is a **typed derived record**
here: a formula over named evidence records, computed with exact decimals from the committed rows,
carrying its inputs, its unit and the claim that maps it. Nothing is fitted, scored or retrieved.

`validate_all()` recomputes every record from a fresh read of the committed files, independent of
the caches, and `tests/test_36_derived_records.py` re-derives the standard's Appendix values and
shows that a perturbed source row fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from functools import lru_cache
from pathlib import Path

from . import registry as G
from . import research as R

REPO_ROOT = R.REPO_ROOT

# --------------------------------------------------------------------------- units

#: A relative change in whole percent: the only kind of percentage the reading path may carry
#: besides structural levels and v1's protected statements (standard §3.3, §4).
UNIT_RELATIVE = "relative change, percent"
UNIT_COUNT = "count"
UNIT_LABEL = "label"
UNIT_DATE = "date"
UNIT_EUR = R.UNIT_EUR
UNIT_NORM_DIFF = R.UNIT_NORM_DIFF

#: Where the committed rows of each checkpoint's criteria and scores live.
CRITERIA_FILES = {
    "CP-15": ("cp15", "reports/cp15/criteria.csv"),
    "CP-16": ("cp16", "reports/v2-causal/criteria.csv"),
    "CP-20": ("cp20", "reports/weather-ablation/criteria.csv"),
    "CP-21": ("cp21", "reports/block-challenger/criteria.csv"),
    "CP-22": ("cp22", "reports/v4-revision/criteria.csv"),
    "CP-23": ("cp23", "reports/distribution-challenger/criteria.csv"),
    "CP-24": ("cp24", "reports/ddnn2/attempt-1/criteria.csv"),
}
REFERENCE_CODES = ("B0", "B1", "B2", "B3")
FOLDS = ("fold_1", "fold_2", "fold_3", "fold_4", "fold_5")


class DerivedError(ValueError):
    """A derived record no longer follows from its committed inputs."""


@dataclass(frozen=True)
class DerivedRecord:
    record_id: str
    kind: str
    subject: str | None
    unit: str
    value: str
    inputs: tuple[str, ...]
    formula: str
    display_precision: int
    claim: str
    ci_low: str | None = None
    ci_high: str | None = None
    label: str = ""
    sources: tuple[str, ...] = field(default=())

    @property
    def interval(self) -> bool:
        return self.ci_low is not None


# --------------------------------------------------------------------------- the inputs


def _d(record_id: str, which: str = "value") -> Decimal:
    record = R.get(record_id)
    raw = {"value": record.raw, "ci_low": record.ci_low_raw, "ci_high": record.ci_high_raw}[which]
    return Decimal(raw)


def _score_record(experiment: str, code: str, score: str) -> str:
    """The equal-fold score record of a policy in one checkpoint's committed rows."""
    if experiment == "CP-15":
        return f"cp15.relative_scores.{code}.{score}"
    prefix = G.EXPERIMENT_PREFIX[experiment]
    return f"{prefix}.metrics.{code}.equal_fold.{score}"


def _criteria_policies(experiment: str) -> tuple[str, ...]:
    prefix, _ = CRITERIA_FILES[experiment]
    found = []
    for record_id, record in R.records().items():
        if record_id.startswith(f"{prefix}.criteria.") and record_id.endswith(".c1.equal_fold.S_MAE"):
            found.append(record.policy_code)
    return tuple(dict.fromkeys(found))


def _strongest_reference(experiment: str, score: str) -> tuple[str, str]:
    """(code, record) of the lowest score among B0–B3: the rule's comparator (capstone §8)."""
    candidates = [(code, _score_record(experiment, code, score)) for code in REFERENCE_CODES]
    candidates = [(code, record_id) for code, record_id in candidates if record_id in R.records()]
    return min(candidates, key=lambda pair: _d(pair[1]))


def _relative(numerator: Decimal, denominator: Decimal) -> Decimal:
    return (numerator / denominator - 1) * 100


def _share(numerator: Decimal, denominator: Decimal) -> Decimal:
    return numerator / denominator * 100


def _text(value: Decimal) -> str:
    return format(value.normalize(), "f")


# --------------------------------------------------------------------------- the rule (§3.3 a)

RULE = G.RULES["criteria-1-2"]
#: The rule's own date, as its anchor states it, and the committed files that establish it.
RULE_DATE_LINE = "**Active owner-authorized execution plan · 2026-09-15.**"
RULE_DATE_SOURCES = ("capstone_v21.md", "reports/cp15/attempt-1/capstone_v21.md")


def section_8(text: str) -> str:
    """Capstone §8, the product-feasibility criteria, as committed."""
    start = text.index("## 8. Product feasibility decision")
    return text[start:text.index("## 9.", start)]


def _rule_date() -> str:
    for path in RULE_DATE_SOURCES:
        if RULE_DATE_LINE not in (REPO_ROOT / path).read_text():
            raise DerivedError(f"{path} no longer dates the rule 2026-09-15")
    anchor, attempt_1 = ((REPO_ROOT / path).read_text() for path in RULE_DATE_SOURCES)
    if section_8(anchor) != section_8(attempt_1):
        raise DerivedError("§8 of the anchor differs from the anchor preserved from attempt 1")
    return re.search(r"(\d{4}-\d{2}-\d{2})", RULE_DATE_LINE).group(1)


# --------------------------------------------------------------------------- the stress period (§3.3 c)

#: The research protocol names the stress period: fold 3, the 2022 crisis (capstone v21 §3 and §7;
#: CP-10's protocol calls it its diagnostic fold). It is read, never assumed.
STRESS_QUOTES = (
    ("capstone_v21.md", "Report fold 3 and August 15–31 with their own dates and denominators"),
    ("capstone_v21.md", "The equal fold weighting is deliberate so a crisis fold is not diluted"),
)


def stress_period() -> str:
    for path, quote in STRESS_QUOTES:
        if quote not in (REPO_ROOT / path).read_text():
            raise DerivedError(f"{path} no longer names the stress period")
    fold = R._json("reports/cp10/protocol.json")["diagnostic_fold"]
    if fold != "fold_3":
        raise DerivedError(f"the protocol's diagnostic fold is {fold}, not fold_3")
    return fold


# --------------------------------------------------------------------------- building the records


def _decision(entry: G.Entry) -> G.StatusEvent:
    """The first evaluated-policy decision, with its dated committed landing provenance."""
    landing = G.CHECKPOINTS[entry.checkpoint].landing
    events = [event for event in entry.statuses if event.source == landing]
    if not events:
        raise DerivedError(f"{entry.id}: no decision tied to {landing}")
    event = min(events, key=lambda item: item.date)
    if event.date not in (REPO_ROOT / landing).read_text().splitlines()[0]:
        raise DerivedError(f"{entry.id}: decision date differs from {landing}")
    return event


def _decision_counts(census: dict[str, tuple[str, bool]]) -> dict[str, tuple[str, str]]:
    """Same-date policies share a boundary; no CSV order breaks a simultaneous result tie."""
    return {
        subject: (
            str(sum(other_date <= date for other_date, _ in census.values())),
            "yes" if met and not any(other != subject and other_met and other_date <= date
                                    for other, (other_date, other_met) in census.items()) else "no",
        ) for subject, (date, met) in census.items()
    }


def _fresh_decision_counts(fresh: R.FreshRead) -> dict[str, tuple[str, str]]:
    """Independent census from hash-checked CSV rows, not cached derived verdicts/counts."""
    census = {}
    for experiment in sorted(CRITERIA_FILES, key=lambda exp: G.CHECKPOINTS[exp].frozen_on):
        _, path = CRITERIA_FILES[experiment]
        rows = fresh.matches(path, {"scope": "equal_fold"})
        for code in sorted({row["policy"] for row in rows}):
            entry = G.by_code(code, experiment)
            if entry.id in census:
                continue  # H0 is the already evaluated V2-H, never a ninth policy.
            verdict = []
            for criterion, metric in (("1", "S_MAE"), ("2", "S_WIS")):
                matches = [row for row in rows if row["policy"] == code
                           and row["criterion"] == criterion and row["metric"] == metric]
                if len(matches) != 1:
                    raise DerivedError(f"{experiment}/{code}: expected one row for criterion {criterion}")
                row = matches[0]
                verdict.append(Decimal(row["actual"]) <= Decimal(row["upper_limit"]))
            census[entry.id] = (_decision(entry).date, all(verdict))
    # Recompute directly, without calling the builder's count routine.
    passed_dates = [date for date, passed in census.values() if passed]
    first_date = min(passed_dates) if passed_dates else None
    first_subjects = {subject for subject, (date, passed) in census.items() if passed and date == first_date}
    return {subject: (str(len({other for other, (other_date, _) in census.items() if other_date <= date})),
                      "yes" if first_subjects == {subject} else "no")
            for subject, (date, _) in census.items()}


def _criteria_records() -> list[DerivedRecord]:
    """Per evaluated policy: the verdict on criteria 1–2, its distance from the rule's comparator in
    the rule's unit, N tested up to its decision, and whether it was the first to meet the rule."""
    out: list[DerivedRecord] = []
    date = _rule_date()
    out.append(DerivedRecord(
        "derived.rule.set_on", "rule_date", None, UNIT_DATE, date, (), "the anchor's own date line; §8 equal to "
        "the anchor preserved from attempt 1", 0, "P17", sources=RULE_DATE_SOURCES + RULE.provenance[2:]))
    for score, criterion in (("S_MAE", "c1"), ("S_WIS", "c2")):
        code, reference = _strongest_reference("CP-20", score)
        limit = f"cp20.criteria.HG.{criterion}.equal_fold.{score}.upper_limit"
        margin = -_relative(_d(limit), _d(reference))
        out.append(DerivedRecord(
            f"derived.rule.margin.{score}", "rule_margin", G.by_code(code).id, UNIT_RELATIVE, _text(margin),
            (limit, reference), "(1 − limit / lowest reference score) × 100", 0, "P17",
            label=f"below {G.by_code(code).name}'s score"))
    census: dict[str, tuple[str, bool]] = {}
    count_inputs: list[str] = []
    decision_sources: set[str] = set()
    for experiment in sorted(CRITERIA_FILES, key=lambda exp: G.CHECKPOINTS[exp].frozen_on):
        prefix, _ = CRITERIA_FILES[experiment]
        for code in _criteria_policies(experiment):
            entry = G.by_code(code, experiment)
            passed, inputs, distances = [], [], {}
            for score, criterion in (("S_MAE", "c1"), ("S_WIS", "c2")):
                actual = f"{prefix}.criteria.{code}.{criterion}.equal_fold.{score}"
                limit = actual + ".upper_limit"
                scored = _score_record(experiment, code, score)
                ref_code, reference = _strongest_reference(experiment, score)
                if _d(actual) != _d(scored):
                    raise DerivedError(f"{actual} is not the committed score {scored}")
                passed.append(_d(actual) <= _d(limit))
                inputs += [actual, limit, reference, scored]
                distances[score] = (_relative(_d(scored), _d(reference)), (scored, reference), ref_code)
            verdict = "met" if all(passed) else "not met"
            base = f"derived.criteria.{entry.id}"
            if f"{base}.verdict" in {record.record_id for record in out}:
                continue  # one identity, one verdict: CP-20's H0 re-reports v2 (= V2-H), CP-21's HG v3, unchanged
            out.append(DerivedRecord(f"{base}.verdict", "verdict", entry.id, UNIT_LABEL, verdict, tuple(inputs),
                                     "actual ≤ upper limit for criterion 1 and criterion 2", 0, "P16",
                                     label="point comparison"))
            for score, (value, pair, ref_code) in distances.items():
                out.append(DerivedRecord(
                    f"{base}.distance.{score}", "distance", entry.id, UNIT_RELATIVE, _text(value), pair,
                    "(policy score / lowest reference score − 1) × 100", 0, "P16",
                    label=f"point comparison with {G.by_code(ref_code).name}"))
            event = _decision(entry)
            census[entry.id] = (event.date, verdict == "met")
            count_inputs.extend(inputs)
            decision_sources.add(event.source)
    counts = _decision_counts(census)
    for subject in sorted(census):
        tested, first = counts[subject]
        base = f"derived.criteria.{subject}"
        for kind, value, formula in (
            ("tested", tested, "distinct evaluated identities with decision date ≤ this policy's decision date"),
            ("first_to_meet", first, "sole policy meeting the rule at the earliest passing decision date"),
        ):
            out.append(DerivedRecord(f"{base}.{kind}", kind, subject,
                                     UNIT_COUNT if kind == "tested" else UNIT_LABEL, value,
                                     tuple(count_inputs), formula, 0, "P16",
                                     sources=tuple(sorted(decision_sources))))
    return out


#: The change against the comparator as a share of the comparator's score (§3.3 b): the committed
#: paired difference and its 95% interval, each divided by the same fixed denominator.
CHANGES = {
    "v3": ("cp20.uncertainty.HG-H0.equal_fold.{metric}", "cp20.metrics.H0.equal_fold.{score}"),
    "v2": ("cp16.uncertainty.V2-H-B2.equal_fold.{metric}", "cp16.metrics.B2.equal_fold.{score}"),
}


#: From CP-21 on (standard §3.3 b): the change as a share of the comparator's score is the checkpoint's own
#: ratio, S_policy / S_comparator − 1, with the interval of that ratio from its own bootstrap draws.
RATIO_CHANGES = {
    "v4": "cp21.ratio.HGL-HG.{metric}",
    "v5": "cp24.ratio.v5-HGL.{metric}",
    # PRES-4: v5's study arm DDNN-2 alone, against its comparator v4 (descriptive, never eligible).
    "ddnn2-alone": "cp24.ratio.D2-HGL.{metric}",
}

#: The same change at a rule's own decision level, where it is not 95% (capstone v21-r11 §23.8-§23.9): v5's rule decided
#: at 97.5%, so its ratio is published at both levels, from the same stored draws.
DECISION_LEVEL_CHANGES = {
    "v5": ("cp24.ratio97_5.v5-HGL.{metric}", "97.5%"),
}


def _change_records() -> list[DerivedRecord]:
    out = []
    for subject, ratio in RATIO_CHANGES.items():
        comparator = G.get(G.get(subject).comparator)
        for metric, score in (("MAE", "S_MAE"), ("WIS", "S_WIS")):
            ratio_id = ratio.format(metric=metric)
            hundred = Decimal(100)
            out.append(DerivedRecord(
                f"derived.change.{subject}.{score}", "ratio_change", subject, UNIT_RELATIVE,
                _text(_d(ratio_id) * hundred), (ratio_id,),
                "the checkpoint's ratio S_policy / S_comparator − 1 × 100; its interval the 2.5th and 97.5th "
                "percentiles of the same ratio over the bootstrap draws, × 100",
                0, "P18", ci_low=_text(_d(ratio_id, "ci_low") * hundred),
                ci_high=_text(_d(ratio_id, "ci_high") * hundred),
                label=f"as a share of {comparator.short}'s score"))
    for subject, (ratio, level) in DECISION_LEVEL_CHANGES.items():
        comparator = G.get(G.get(subject).comparator)
        for metric, score in (("MAE", "S_MAE"), ("WIS", "S_WIS")):
            ratio_id = ratio.format(metric=metric)
            hundred = Decimal(100)
            out.append(DerivedRecord(
                f"derived.change_decision.{subject}.{score}", "ratio_change", subject, UNIT_RELATIVE,
                _text(_d(ratio_id) * hundred), (ratio_id,),
                f"the checkpoint's ratio S_policy / S_comparator − 1 × 100; its interval at the rule's {level} decision "
                "level, the 1.25th and 98.75th percentiles of the same ratio over the bootstrap draws, × 100",
                0, "P18", ci_low=_text(_d(ratio_id, "ci_low") * hundred),
                ci_high=_text(_d(ratio_id, "ci_high") * hundred),
                label=f"as a share of {comparator.short}'s score, {level} interval"))
    for subject, (difference, denominator) in CHANGES.items():
        comparator = G.get(G.get(subject).comparator)
        for metric, score in (("MAE", "S_MAE"), ("WIS", "S_WIS")):
            diff_id, denom_id = difference.format(metric=metric), denominator.format(score=score)
            denom = _d(denom_id)
            out.append(DerivedRecord(
                f"derived.change.{subject}.{score}", "share_change", subject, UNIT_RELATIVE,
                _text(_share(_d(diff_id), denom)), (diff_id, denom_id),
                "paired difference / comparator's score × 100; the interval divided by the same denominator",
                0, "P18", ci_low=_text(_share(_d(diff_id, "ci_low"), denom)),
                ci_high=_text(_share(_d(diff_id, "ci_high"), denom)),
                label=f"as a share of {comparator.short}'s score"))
    return out


#: Absolute context (§3.3 c): per-period MAE in EUR/MWh for these identities, each from the experiment whose rows it
#: was published from: v5's from CP-24; the others from CP-21, as v4's chapter and the comparison published them. The
#: shared rows are byte-identical across the comparison's experiments (`comparison_parity_problems`).
PERIOD_SOURCES = {"v5": "CP-24", "v4": "CP-21", "v3": "CP-21", "v2": "CP-21", "daily-lear": "CP-21", "naive": "CP-21"}
PERIOD_SUBJECTS = tuple(PERIOD_SOURCES)


def _period_records() -> list[DerivedRecord]:
    out = []
    stress = stress_period()
    out.append(DerivedRecord("derived.periods.stress_period", "stress_period", None, UNIT_LABEL, stress, (),
                             "the protocol's named stress period", 0, "P19",
                             sources=tuple(path for path, _ in STRESS_QUOTES) + ("reports/cp10/protocol.json",)))
    for subject, experiment in PERIOD_SOURCES.items():
        prefix = G.comparison_prefix(experiment)
        code = G.get(subject).code_in(experiment)
        ordinary = [f"{prefix}.metrics.{code}.{fold}.MAE" for fold in FOLDS if fold != stress]
        low = min(ordinary, key=_d)
        high = max(ordinary, key=_d)
        for which, record_id in (("low", low), ("high", high)):
            out.append(DerivedRecord(
                f"derived.periods.{subject}.ordinary_{which}", "period_range", subject, UNIT_EUR,
                R.get(record_id).raw, tuple(ordinary),
                f"{'minimum' if which == 'low' else 'maximum'} MAE over the ordinary test periods", 1, "P19"))
        stress_id = f"{prefix}.metrics.{code}.{stress}.MAE"
        out.append(DerivedRecord(f"derived.periods.{subject}.stress", "period_stress", subject, UNIT_EUR,
                                 R.get(stress_id).raw, (stress_id,), "MAE in the stress period", 1, "P19"))
    return out


# --------------------------------------------------------------------------- an adoption rule (§3.3 a)

#: A generation adopted under its own pre-registered rule (CP-21 on): the rule, the committed record of its
#: mechanical verdict, the paired differences it judged at its own decision level, the anchor text that dates it, and the
#: eligible policies. CP-21's record names the generation in its verdict; CP-24's records whether its candidate, code
#: `v5` in CP-24, was adopted. v5's rule decided at 97.5% (capstone v21-r11 §23.9), so its headline reads those intervals.
ADOPTION_RULES = {
    "v4": {
        "rule": "cp21-adoption",
        "claim": "P51",
        "prefix": "cp21",
        "experiment": "CP-21",
        "conditions": ("1", "2", "3", "4"),
        "set_on": "cp21.adoption.rule_set_on",
        "verbatim": "adoption_rule_verbatim",
        "differences": "cp21.uncertainty.HGL-HG.equal_fold.{metric}",
        "level": "95%",
        "dated_by": (("capstone_v21.md", "Rule `cp21-adoption` was set on 2026-09-29"),
                     ("reports/block-challenger/protocol.json", "Rule `cp21-adoption` was set on 2026-09-29")),
    },
    "v5": {
        "rule": "cp24-adoption",
        "claim": "P53",
        "prefix": "cp24",
        "experiment": "CP-24",
        "conditions": ("1", "2", "3", "4", "5"),
        "set_on": None,
        "verbatim": "rule_verbatim",
        "differences": "cp24.uncertainty97_5.v5-HGL.equal_fold.{metric}",
        "level": "97.5%",
        "dated_by": (("capstone_v21.md", "Pre-registered rule `cp24-adoption`, set 2026-10-05"),
                     ("reports/ddnn2/attempt-1/protocol.json", "Pre-registered rule `cp24-adoption`, set 2026-10-05")),
    },
}


def _subject_of(spec: dict) -> str:
    return next(subject for subject, item in ADOPTION_RULES.items() if item is spec)


def _adoption_date(spec: dict, fresh: R.FreshRead | None = None) -> str:
    """The rule's date, as its adoption record or the registry's rule states it, and as the ratified anchor and the
    protocol frozen before scoring both quote it."""
    if spec["set_on"]:
        record = R.get(spec["set_on"])
        date = R.rederive(record, fresh)["value"] if fresh else record.raw
    else:
        date = G.RULES[spec["rule"]].set_on
    for path, quote in spec["dated_by"]:
        text = (R._json(path)[spec["verbatim"]] if path.endswith(".json") else (REPO_ROOT / path).read_text())
        if quote not in text or not quote.endswith(date):
            raise DerivedError(f"{path} no longer dates rule {spec['rule']} {date}")
    return date


def _adoption_verdict(spec: dict, value) -> str:
    """Met only when every condition is met and the committed mechanical verdict names this generation: CP-21's verdict
    names the version, CP-24's records its candidate's code and that it was adopted."""
    prefix, subject = spec["prefix"], _subject_of(spec)
    conditions = [value(f"{prefix}.adoption.condition_{n}_met") for n in spec["conditions"]]
    if spec["set_on"]:
        named = value(f"{prefix}.adoption.verdict") == G.get(subject).version
    else:
        named = (value(f"{prefix}.adoption.adopted") == "True"
                 and value(f"{prefix}.adoption.candidate") == G.get(subject).code_in(spec["experiment"]))
    return "met" if all(c == "True" for c in conditions) and named else "not met"


def _adoption_eligible(spec: dict, value) -> str:
    """N (§3.3 a): the policies eligible under the rule up to its decision. The rule names one candidate; the
    study arms are never eligible (§17.6, §23.5), so N counts the committed candidate, never the arms."""
    return str(len([value(f"{spec['prefix']}.adoption.candidate")]))


def _adoption_records() -> list[DerivedRecord]:
    out = []
    for subject, spec in ADOPTION_RULES.items():
        prefix, base = spec["prefix"], f"derived.adoption.{subject}"
        raw = lambda record_id: R.get(record_id).raw  # noqa: E731
        conditions = tuple(f"{prefix}.adoption.condition_{n}_met" for n in spec["conditions"])
        named = ((f"{prefix}.adoption.verdict",) if spec["set_on"]
                 else (f"{prefix}.adoption.adopted", f"{prefix}.adoption.candidate"))
        out.append(DerivedRecord(f"{base}.verdict", "adoption_verdict", subject, UNIT_LABEL,
                                 _adoption_verdict(spec, raw), named + conditions,
                                 "met when the committed mechanical verdict names this generation and all "
                                 f"{len(spec['conditions'])} conditions are met", 0, "P51"))
        out.append(DerivedRecord(f"{base}.set_on", "adoption_date", subject, UNIT_DATE, _adoption_date(spec),
                                 ((spec["set_on"],) if spec["set_on"] else (f"{prefix}.protocol.rule_id",)),
                                 "the rule's date, quoted by the ratified anchor and by the protocol frozen before "
                                 "scoring", 0, "P51", sources=tuple(path for path, _ in spec["dated_by"])))
        out.append(DerivedRecord(f"{base}.tested", "adoption_tested", subject, UNIT_COUNT,
                                 _adoption_eligible(spec, raw), (f"{prefix}.adoption.candidate",),
                                 "policies eligible under the rule up to its decision: its one candidate", 0, "P51"))
        for metric, score in (("MAE", "S_MAE"), ("WIS", "S_WIS")):
            difference = spec["differences"].format(metric=metric)
            record = R.get(difference)
            out.append(DerivedRecord(f"{base}.distance.{score}", "adoption_distance", subject, UNIT_NORM_DIFF,
                                     record.raw, (difference,), "the rule's paired difference, in the rule's unit, "
                                     f"with its interval at the rule's decision level, {spec['level']}", 4, "P51",
                                     ci_low=record.ci_low_raw, ci_high=record.ci_high_raw))
    return out


# --------------------------------------------------------------------------- the coverage limits (PRES-4)

#: The shared population's coverage of the calendar, stated beside v5 (the Owner's programme plan of 2026-10-11, stage 2):
#: how many delivery days it scores, out of how many days the data holds up to the development boundary, that none falls
#: in October to December, and how many winters it holds. Counted from committed records: the represented dates of the
#: comparison's naive row, the data's first price hour (`data/source_manifest.json`) and the development boundary
#: (`data/partitions.json`, `eda_cutoff`). No row after the boundary is read.
COVERAGE_POPULATION = ("CP-24", "reports/ddnn2/attempt-1/metrics.csv")
COVERAGE_FIRST_HOUR = ("data/source_manifest.json", "coverage.price_eur_mwh")
COVERAGE_BOUNDARY = ("data/partitions.json", "eda_cutoff")
DATA_TIMEZONE = "Europe/Berlin"


def _represented(row: dict[str, str]) -> list[str]:
    import json

    return sorted(json.loads(row["represented_dates"]))


def _coverage_facts(population_row: dict[str, str], manifest: dict, partitions: dict) -> dict[str, str]:
    """Every coverage fact from its committed inputs; the same function serves the build and the fresh re-derivation."""
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    days = _represented(population_row)
    if len(days) != len(set(days)) or str(len(days)) != str(int(float(population_row["n_days"]))):
        raise DerivedError("the represented dates do not match the population's day count")
    price = manifest["coverage"]["price_eur_mwh"]
    if int(price["null_rows"]) != 0:
        raise DerivedError("the data's price series has missing hours; its days cannot be counted from its range")
    first = datetime.fromisoformat(price["first_valid_utc"]).astimezone(ZoneInfo(DATA_TIMEZONE)).date()
    boundary = date.fromisoformat(partitions["eda_cutoff"])
    if boundary.isoformat() != R.EVIDENCE_BOUNDARY:
        raise DerivedError(f"the partition boundary {boundary} is not the evidence boundary {R.EVIDENCE_BOUNDARY}")
    if days[0] < first.isoformat() or days[-1] > boundary.isoformat():
        raise DerivedError("a scored day lies outside the data's development range")
    winters = {(int(d[:4]) if d[5:7] == "12" else int(d[:4]) - 1) for d in days if d[5:7] in ("12", "01", "02")}
    return {
        "days": str(len(days)),
        "data_days": str((boundary - first).days + 1),
        "first_data_day": first.isoformat(),
        "data_end": boundary.isoformat(),
        "q4_days": str(sum(d[5:7] in ("10", "11", "12") for d in days)),
        "winters": str(len(winters)),
    }


COVERAGE_KINDS = {
    "days": (UNIT_COUNT, "the distinct delivery dates the shared population scores (its naive row's represented dates)"),
    "data_days": (UNIT_COUNT, "calendar days from the data's first day, in the market's time zone, to the development "
                              "boundary, inclusive; the price series has no missing hour"),
    "first_data_day": (UNIT_DATE, "the delivery date of the data's first price hour, in the market's time zone"),
    "data_end": (UNIT_DATE, "the development boundary: no research result reads data after it"),
    "q4_days": (UNIT_COUNT, "scored delivery dates in October, November or December"),
    "winters": (UNIT_COUNT, "distinct winters, December to February, holding at least one scored day"),
}


def _coverage_inputs(fresh: R.FreshRead | None = None) -> tuple[dict[str, str], dict, dict]:
    experiment, path = COVERAGE_POPULATION
    if fresh is None:
        row = next(row for _, row in R._rows(path) if row["policy"] == "B0" and row["scope"] == "pooled")
        return row, R._json(COVERAGE_FIRST_HOUR[0]), R._json(COVERAGE_BOUNDARY[0])
    rows = fresh.matches(path, {"policy": "B0", "scope": "pooled"})
    if len(rows) != 1:
        raise DerivedError(f"{path}: {len(rows)} pooled naive rows")
    return rows[0], fresh.document(COVERAGE_FIRST_HOUR[0]), fresh.document(COVERAGE_BOUNDARY[0])


def _coverage_records() -> list[DerivedRecord]:
    experiment, path = COVERAGE_POPULATION
    facts = _coverage_facts(*_coverage_inputs())
    prefix = G.comparison_prefix(experiment)
    out = []
    for key, (unit, formula) in COVERAGE_KINDS.items():
        out.append(DerivedRecord(f"derived.coverage.{key}", "coverage", None, unit, facts[key],
                                 (f"{prefix}.metrics.B0.pooled.n_days",), formula, 0, "P54",
                                 sources=(path, COVERAGE_FIRST_HOUR[0], COVERAGE_BOUNDARY[0])))
    return out


# --------------------------------------------------------------------------- the comparison's population (PRES-4)


#: The committed cells a comparison row is drawn from. Every experiment of the live comparison that carries an identity
#: must hold exactly these cells for it, and the naive's per-period hours must agree: one population, not a join.
PARITY_FIELDS = (("equal_fold", ("S_MAE", "S_WIS")), ("pooled", ("MAE", "WIS", "n_hours", "n_days")))


def comparison_parity_problems(rows=None) -> list[str]:
    """Why the live comparison's rows are not one population: an identity whose committed cells differ between two of
    the comparison's experiments, or a naive whose per-period hours do. Empty when they are one (PUBLISH_RULES §3.2)."""
    problems = []
    entries = rows if rows is not None else G.comparison_rows()
    experiments = G.COMPARISON_EXPERIMENTS
    for entry in tuple(entries) + (G.get("naive"),):
        source = G.comparison_source(entry)
        for experiment in experiments:
            code = entry.code_in(experiment)
            if code is None or experiment == source:
                continue
            for scope, fields in PARITY_FIELDS:
                for field_name in fields:
                    here = f"{G.comparison_prefix(source)}.metrics.{entry.code_in(source)}.{scope}.{field_name}"
                    there = f"{G.comparison_prefix(experiment)}.metrics.{code}.{scope}.{field_name}"
                    if R.get(here).raw != R.get(there).raw:
                        problems.append(f"{entry.id}: {field_name} ({scope}) differs between {source} and {experiment}")
    for experiment in experiments[1:]:
        for fold in FOLDS:
            here = f"{G.comparison_prefix(experiments[0])}.metrics.B0.{fold}.n_hours"
            there = f"{G.comparison_prefix(experiment)}.metrics.B0.{fold}.n_hours"
            if R.get(here).raw != R.get(there).raw:
                problems.append(f"the naive's {fold} hours differ between {experiments[0]} and {experiment}")
    return problems


@lru_cache(maxsize=1)
def records() -> dict[str, DerivedRecord]:
    built = _criteria_records() + _change_records() + _adoption_records() + _period_records() + _coverage_records()
    out: dict[str, DerivedRecord] = {}
    for record in built:
        if record.record_id in out:
            raise DerivedError(f"duplicate derived record {record.record_id}")
        out[record.record_id] = record
    return out


def get(record_id: str) -> DerivedRecord:
    try:
        return records()[record_id]
    except KeyError:
        raise DerivedError(f"no derived record {record_id!r}") from None


def is_derived(record_id: str) -> bool:
    return record_id.startswith("derived.")


# --------------------------------------------------------------------------- validation


def validate_all() -> list[str]:
    """Recompute every derived record from a fresh read of its committed inputs. Empty means all hold.

    Each input record is re-read from its committed row (and its file's blob checked), then the
    derived value is recomputed from those fresh values, independent of every cache."""
    problems = []
    fresh = R.FreshRead()
    cache: dict[tuple[str, str], Decimal] = {}

    def value(record_id: str, which: str = "value") -> Decimal:
        key = (record_id, which)
        if key not in cache:
            derived = R.rederive(R.get(record_id), fresh)
            cache[key] = Decimal(derived["value" if which == "value" else which])
        return cache[key]

    counts = None
    for record in records().values():
        try:
            if record.kind == "rule_date":
                expected = _rule_date()
            elif record.kind == "stress_period":
                expected = stress_period()
            elif record.kind == "rule_margin":
                limit, reference = record.inputs
                expected = _text(-_relative(value(limit), value(reference)))
            elif record.kind == "distance":
                actual, reference = record.inputs
                expected = _text(_relative(value(actual), value(reference)))
            elif record.kind == "verdict":
                groups = [record.inputs[i:i + 4] for i in range(0, len(record.inputs), 4)]
                for actual, _limit, _reference, scored in groups:
                    if value(actual) != value(scored):
                        problems.append(f"{record.record_id}: {actual} no longer equals the committed score {scored}")
                expected = "met" if all(value(a) <= value(lim) for a, lim, _, _ in groups) else "not met"
            elif record.kind in ("tested", "first_to_meet"):
                if counts is None:
                    counts = _fresh_decision_counts(fresh)
                expected = counts[record.subject][0 if record.kind == "tested" else 1]
            elif record.kind == "share_change":
                diff_id, denom_id = record.inputs
                denom = value(denom_id)
                expected = _text(_share(value(diff_id), denom))
                low, high = _text(_share(value(diff_id, "ci_low"), denom)), _text(_share(value(diff_id, "ci_high"), denom))
                if (low, high) != (record.ci_low, record.ci_high):
                    problems.append(f"{record.record_id}: interval no longer follows from its rows")
            elif record.kind == "ratio_change":
                (ratio_id,) = record.inputs
                hundred = Decimal(100)
                expected = _text(value(ratio_id) * hundred)
                low, high = _text(value(ratio_id, "ci_low") * hundred), _text(value(ratio_id, "ci_high") * hundred)
                if (low, high) != (record.ci_low, record.ci_high):
                    problems.append(f"{record.record_id}: interval no longer follows from its rows")
            elif record.kind == "adoption_verdict":
                spec = ADOPTION_RULES[record.subject]
                expected = _adoption_verdict(spec, lambda rid: R.rederive(R.get(rid), fresh)["value"])
            elif record.kind == "adoption_date":
                expected = _adoption_date(ADOPTION_RULES[record.subject], fresh)
            elif record.kind == "adoption_tested":
                expected = _adoption_eligible(ADOPTION_RULES[record.subject],
                                              lambda rid: R.rederive(R.get(rid), fresh)["value"])
            elif record.kind == "adoption_distance":
                (difference,) = record.inputs
                expected = _text(value(difference))
                if (_text(value(difference, "ci_low")), _text(value(difference, "ci_high"))) != (
                        _text(Decimal(record.ci_low)), _text(Decimal(record.ci_high))):
                    problems.append(f"{record.record_id}: interval no longer follows from its rows")
            elif record.kind == "period_range":
                values = [value(record_id) for record_id in record.inputs]
                pick = min(values) if record.record_id.endswith("_low") else max(values)
                expected = _text(pick)
            elif record.kind == "period_stress":
                expected = _text(value(record.inputs[0]))
            elif record.kind == "coverage":
                expected = _coverage_facts(*_coverage_inputs(fresh))[record.record_id.rsplit(".", 1)[1]]
            else:
                problems.append(f"{record.record_id}: unknown kind {record.kind}")
                continue
            recorded = record.value if record.unit in (UNIT_LABEL, UNIT_DATE) else _text(Decimal(record.value))
            if record.kind == "coverage" and record.unit != UNIT_DATE:
                expected = _text(Decimal(expected))
            if expected != recorded:
                problems.append(f"{record.record_id}: {recorded!r} no longer follows from its rows ({expected!r})")
        except (R.EvidenceError, DerivedError, KeyError) as exc:
            problems.append(f"{record.record_id}: {exc}")
    return problems


# --------------------------------------------------------------------------- display


def display(record: DerivedRecord, which: str = "value", *, style: str | None = None) -> str:
    """The one display string for a derived value (standard §4): whole percent for relative
    changes, one decimal for EUR/MWh, the word itself for labels and dates. `style="abs"` drops the
    sign ("14% below" reads the sign in words)."""
    raw = {"value": record.value, "ci_low": record.ci_low, "ci_high": record.ci_high}[which]
    if raw is None:
        raise DerivedError(f"{record.record_id} has no {which}")
    if record.unit in (UNIT_LABEL, UNIT_DATE):
        return raw
    places = record.display_precision + (2 if style == "exact" else 0)  # value tables carry two more places
    quantized = Decimal(raw).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)
    if quantized == 0:
        quantized = abs(quantized)
    text = f"{abs(quantized):,.{places}f}" if record.unit == UNIT_COUNT else f"{abs(quantized):.{places}f}"
    if style == "abs":
        sign = ""
    else:
        sign = R.MINUS if quantized < 0 else ""
    suffix = "%" if record.unit == UNIT_RELATIVE else ""
    return f"{sign}{text}{suffix}"


__all__ = [
    "ADOPTION_RULES", "CHANGES", "COVERAGE_KINDS", "DECISION_LEVEL_CHANGES", "PERIOD_SOURCES", "RATIO_CHANGES",
    "comparison_parity_problems", "DerivedError", "DerivedRecord", "PERIOD_SUBJECTS", "RULE", "UNIT_COUNT", "UNIT_DATE", "UNIT_EUR",
    "UNIT_LABEL", "UNIT_RELATIVE", "display", "get", "is_derived", "records", "section_8", "stress_period",
    "validate_all",
]
