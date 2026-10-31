"""The registry: one source for every published identity, status and comparison (standard §5).

Publication Standard v1 §5 requires one entry per generation, branch, reference, study arm and
control, and requires every public surface to take its names, statuses and order from it: the
page's status pair, rail, jump row, lineage, comparison rows and chapter headers; the README's
headings; the Space cards' model line; and the `delu-generations` MLflow run names, parents,
descriptions and tags. Before PRES-1's conformance work those were typed in about 45 places in
nine files, and "current" was written into prose that becomes false the day v4 is adopted.

**Status is derived, never typed.** An entry carries a dated status history. A surface never
writes "current", "latest" or "the demo runs"; it asks this module, which answers from the
history -- `current_generation()`, `released()`, `status_sentence()` -- in the past tense and dated.

**One identity across experiment codes.** v2 is `V2-H` in CP-16 and `H0` in CP-20; v1 is `B1` in
CP-15 and `v1_reference` in CP-10. `by_code()` resolves either to the same entry, so a policy is
never renamed by the experiment that happens to evaluate it. A code is an experiment's own label, so
the same text can name different identities in different experiments: `v5` is CP-23's candidate,
the branch "DDNN member on v4", and CP-24's adopted v5. Inside one experiment a code names one
identity; `by_code(code, experiment)` resolves it there, and an unscoped lookup of a shared text
refuses rather than guess.

**Nothing here is a research number.** Dates, codes and names are identity; every score comes from
`delu_forecast.research` and `delu_forecast.derived`. `tests/test_35_registry.py` holds the
consistency checks and their negative controls.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

# --------------------------------------------------------------------------- vocabularies (§5)

KINDS = ("generation", "branch", "reference", "study arm", "control")

ADOPTED_IN_RESEARCH = "adopted in research"
NOT_ADOPTED = "not adopted"
RELEASED = "released"
FINAL_CANDIDATE = "final candidate"
LIVE = "live"
RETIRED = "retired"
STATUSES = (ADOPTED_IN_RESEARCH, NOT_ADOPTED, RELEASED, FINAL_CANDIDATE, LIVE, RETIRED)

#: The hero shows the model with the highest status reached (standard §2): live, then final
#: candidate, then released, then research.
HERO_ORDER = (LIVE, FINAL_CANDIDATE, RELEASED, ADOPTED_IN_RESEARCH)

EVIDENCE_DEVELOPMENT = "development_post_selection"
EVIDENCE_CALIBRATION = "development_calibration_comparison"
EVIDENCE_V1_HOLDOUT = "confirmatory_style_not_power_qualified"

#: Evidence-class badges (plan §7.8, standard §2). The v1 text is verbatim and may wrap.
BADGES = {
    EVIDENCE_DEVELOPMENT: "Development · post-selection",
    EVIDENCE_CALIBRATION: "Development · post-selection",
    EVIDENCE_V1_HOLDOUT: "Confirmatory-style, not power-qualified",
}

#: Each evidence class's caveat, phrased as a property of the class (standard §1, Tone): it
#: never expires, and it appears once, where it applies.
CLASS_CAVEATS = {
    EVIDENCE_DEVELOPMENT: "development evidence, not a test on new data",
    EVIDENCE_CALIBRATION: "development evidence, not a test on new data",
    EVIDENCE_V1_HOLDOUT: "evaluated once, as specified in advance, on days never used for development or selection",
}

MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December")


class RegistryError(ValueError):
    """A surface asked for an identity or status the registry does not hold."""


# --------------------------------------------------------------------------- the entry (§5 fields)


@dataclass(frozen=True)
class Code:
    """One experiment code of an identity, with the population it was evaluated on there."""

    experiment: str
    code: str
    population: str
    note: str = ""


@dataclass(frozen=True)
class StatusEvent:
    """A dated status, with the record that establishes it."""

    status: str
    date: str
    source: str
    reason: str = ""


@dataclass(frozen=True)
class Entry:
    # identity
    id: str
    name: str
    subtitle: str
    kind: str
    # codes
    codes: tuple[Code, ...]
    # status history, dated
    statuses: tuple[StatusEvent, ...]
    # comparison
    comparator: str | None
    population: str | None
    # evidence
    evidence_class: str
    plan: str
    rules: tuple[str, ...]
    sources: tuple[str, ...]
    claim_map: str | None
    run_keys: tuple[str, ...]
    # presentation: the marker and colour role (plan §7.3, §7.8) and the page anchor
    style: str
    anchor: str | None = None
    #: The checkpoint that produced this identity first, and where a branch sits in the lineage.
    checkpoint: str | None = None
    after: str | None = None
    #: For branches: the question, and what the branch informed.
    question: str = ""
    informed: str | None = None
    #: A short form of the name for tight chart labels ("v3", "Daily LEAR").
    short: str = ""
    #: For a generation after the first: the adopted generation it replaced in research (A3). An
    #: explicit, validated relationship; never inferred from the version number or the name.
    predecessor: str | None = None

    @property
    def primary_code(self) -> str:
        return self.codes[0].code

    def code_in(self, experiment: str) -> str | None:
        for code in self.codes:
            if code.experiment == experiment:
                return code.code
        return None

    @property
    def version(self) -> str | None:
        """`v3` for a generation, None otherwise: a number exists only on adoption (plan §16 D2)."""
        match = re.match(r"^(v\d+) · ", self.name)
        return match.group(1) if match and self.kind == "generation" else None

    @property
    def status(self) -> StatusEvent | None:
        return self.statuses[-1] if self.statuses else None

    @property
    def badge(self) -> str:
        return BADGES[self.evidence_class]


# --------------------------------------------------------------------------- rules and checkpoints


@dataclass(frozen=True)
class Rule:
    """A decision rule the governing research plan set before the experiments it judges."""

    id: str
    plan: str
    words: str
    set_on: str
    provenance: tuple[str, ...]


RULES: dict[str, Rule] = {
    "criteria-1-2": Rule(
        id="criteria-1-2",
        plan="capstone_v21.md §8, criteria 1–2",
        words="each error score at least 10% below the strongest benchmark's",
        set_on="2026-09-15",
        provenance=(
            "capstone_v21.md: 'Active owner-authorized execution plan · 2026-09-15.'",
            "reports/cp15/attempt-1/capstone_v21.md: the anchor preserved from attempt 1; its §8 is identical",
            "bb5e678 (2026-09-16 00:17 +03:00): the pre-registration commit, reachable from evidence/cp-15",
            "0be8e56 (2026-09-16 03:15 +03:00): CP-15's results, committed after it",
        ),
    ),
    "joint-improvement": Rule(
        id="joint-improvement",
        plan="capstone_v21.md §14.4 and §15.4",
        words="both paired differences below zero: the interval score's upper 95% endpoint below zero and "
              "the point error's at or below zero",
        set_on="2026-09-23",
        provenance=("capstone_v21.md §14.4 (ratified O2 contract) and §15.4",),
    ),
    "cp21-adoption": Rule(
        id="cp21-adoption",
        plan="capstone_v21.md §17.6 (v21-r6)",
        words="HGL becomes v4 only if both paired differences against v3 improve (the interval score's upper 95% "
              "endpoint below zero, the point error's at or below zero), it meets all six original screening "
              "diagnostics, the evaluation is complete and valid, and no fold is decisively worse",
        set_on="2026-09-29",
        provenance=("capstone_v21.md v21-r6 §17.6 (ratified 2026-09-29)",
                    "reports/block-challenger/protocol.json adoption_rule_verbatim (frozen before scoring)"),
    ),
    "v1-holdout": Rule(
        id="v1-holdout",
        plan="capstone_V6_8.md §7.1",
        words="one pre-specified evaluation on a 90-day holdout, opened once",
        set_on="2026-09-09",
        provenance=("reports/cp2/holdout_report.json", "data/partitions.json"),
    ),
    # CP-22's three rules (capstone v21-r9 §20.6), set together on 2026-10-01 and frozen before any fit.
    "cp22-replacement": Rule(
        id="cp22-replacement",
        plan="capstone_v21.md §20.6 (v21-r9)",
        words="a pooled LightGBM member replaces v4's three-block member only if neither paired score difference "
              "against v4 has a 95% interval entirely above zero, it meets the six original screening diagnostics, "
              "the evaluation is complete and valid, and no fold is decisively worse",
        set_on="2026-10-01",
        provenance=("capstone_v21.md v21-r9 §20.6 (ratified 2026-10-01)",
                    "reports/v4-revision/protocol.json rules_verbatim (frozen before scoring)"),
    ),
    "cp22-dynamic-layer": Rule(
        id="cp22-dynamic-layer",
        plan="capstone_v21.md §20.6 (v21-r9)",
        words="the dynamic interval layer replaces the winner's interval layer only if it improves the interval "
              "score without worsening the point error, meets the screen, brings pooled 95% coverage closer to "
              "nominal and leaves no fold decisively worse",
        set_on="2026-10-01",
        provenance=("capstone_v21.md v21-r9 §20.6 (ratified 2026-10-01)",
                    "reports/v4-revision/protocol.json rules_verbatim (frozen before scoring)"),
    ),
    "cp22-fast-component": Rule(
        id="cp22-fast-component",
        plan="capstone_v21.md §20.6 (v21-r9)",
        words="the fast component is decided only as an add-on to an adopted dynamic layer",
        set_on="2026-10-01",
        provenance=("capstone_v21.md v21-r9 §20.6 (ratified 2026-10-01)",
                    "reports/v4-revision/protocol.json rules_verbatim (frozen before scoring)"),
    ),
    "cp23-adoption": Rule(
        id="cp23-adoption",
        plan="capstone_v21.md §21.6 (v21-r10)",
        words="v4 plus a DDNN member is adopted only if both paired differences against v4 improve (the interval "
              "score's upper 95% endpoint below zero, the point error's at or below zero), it meets the six original "
              "screening diagnostics, the evaluation is complete and valid, and no fold is decisively worse",
        set_on="2026-10-04",
        provenance=("capstone_v21.md v21-r10 §21.6 (ratified 2026-10-04)",
                    "reports/distribution-challenger/protocol.json rule_verbatim (frozen before scoring)"),
    ),
    "cp24-adoption": Rule(
        id="cp24-adoption",
        plan="capstone_v21.md §23.9 (v21-r11)",
        words="v5 is adopted only if, against v4, both paired differences improve at the attempts-adjusted 97.5% "
              "level (the interval score's upper endpoint below zero, the point error's at or below zero), it meets "
              "the six original screening diagnostics, the evaluation is complete and valid, no fold is decisively "
              "worse, and both point estimates improve v4's score by at least 0.5%",
        set_on="2026-10-05",
        provenance=("capstone_v21.md v21-r11 §23.9 (ratified 2026-10-05)",
                    "reports/ddnn2/attempt-1/protocol.json rule_verbatim (frozen before its fits)"),
    ),
}


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint that produced registry entries: its owning entry and its evidence tag."""

    code: str
    run_key: str
    owner: str
    evidence_tag: str
    evidence_sha: str
    frozen_on: str
    report: str
    verdict: str
    landing: str
    #: The MLflow children, in the order the export and the UI list them.
    children: tuple[str, ...] = ()


CHECKPOINTS: dict[str, Checkpoint] = {
    "CP-10": Checkpoint("CP-10", "cp10", "calibration", "evidence/cp-15", "1bdc75b", "2026-09-16",
                        "reports/cp10/report.md", "docs/track-b/evidence/cp-10/integration.md",
                        "docs/track-b/cp-15-landing.md",
                        ("v1_reference", "c1_head_spread", "c1_price_volatility", "c2_aci_gamma_0.000001",
                         "c2_aci_gamma_0.000005", "c2_aci_gamma_0.00001", "c2_aci_gamma_0.00002")),
    "CP-15": Checkpoint("CP-15", "cp15", "model-comparison", "evidence/cp-15", "1bdc75b", "2026-09-16",
                        "reports/cp15/report.md", "docs/track-b/evidence/cp-15/integration.md",
                        "docs/track-b/cp-15-landing.md",
                        ("B0", "B1", "B2", "B3", "A1", "A2", "A3", "A4", "A5")),
    "CP-16": Checkpoint("CP-16", "cp16", "v2", "evidence/cp-16", "5ec8a92", "2026-09-23",
                        "reports/v2-causal/report.md", "docs/track-b/evidence/cp-16/integration.md",
                        "docs/track-b/cp-16-landing-2026-09-23.md", ("V2-P", "V2-H")),
    "CP-20": Checkpoint("CP-20", "cp20", "v3", "evidence/cp-20", "a7a9b2e", "2026-09-24",
                        "reports/weather-ablation/report.md", "docs/track-b/evidence/cp-20/integration.md",
                        "docs/track-b/cp-20-landing-2026-09-24.md", ("HG",)),
    "CP-21": Checkpoint("CP-21", "cp21", "v4", "evidence/cp-21", "1d13f99", "2026-09-30",
                        "reports/block-challenger/report.md", "docs/track-b/evidence/cp-21/integration.md",
                        "docs/track-b/cp-21-landing-2026-09-30.md", ("HGL", "L-P", "L-R", "L-N")),
    # PRES-4 (2026-10-11): CP-22 and CP-23 as branches, CP-24 as v5, from their draft registries and landing records.
    "CP-22": Checkpoint("CP-22", "cp22", "pooled-lightgbm-member-for-v4", "evidence/cp-22", "8daf7d0", "2026-10-04",
                        "reports/v4-revision/report.md", "docs/track-b/evidence/cp-22/integration.md",
                        "docs/track-b/cp-22-landing-2026-10-04.md",
                        ("R", "M", "A-PN-sel", "A-LP", "A-LN", "v4+DL", "v3+DL")),
    "CP-23": Checkpoint("CP-23", "cp23", "ddnn-member-on-v4", "evidence/cp-23", "928bc13", "2026-10-04",
                        "reports/distribution-challenger/report.md", "docs/track-b/evidence/cp-23/integration.md",
                        "docs/track-b/cp-23-landing-2026-10-04.md", ("v5", "D", "v3+D")),
    "CP-24": Checkpoint("CP-24", "cp24", "v5", "evidence/cp-24", "05b7e61", "2026-10-10",
                        "reports/ddnn2/report.md", "docs/track-b/evidence/cp-24/integration.md",
                        "docs/track-b/cp-24-landing-2026-10-11.md", ("v5", "D2", "v3+D2")),
}

#: Every evidence tag a public link may cite, with the date it froze (the date its commit was
#: made, in the author's time zone). An audit-grade link's label carries this date (standard §7).
EVIDENCE_TAGS: dict[str, tuple[str, str]] = {
    "evidence/cp-2": ("e491079", "2026-09-14"),
    "evidence/cp-3": ("73da531", "2026-09-15"),
    "evidence/cp-3b": ("0adc309", "2026-09-15"),
    "evidence/cp-15": ("1bdc75b", "2026-09-16"),
    "evidence/cp-16": ("5ec8a92", "2026-09-23"),
    "evidence/cp-20": ("a7a9b2e", "2026-09-24"),
    "evidence/cp-21": ("1d13f99", "2026-09-30"),
    "evidence/cp-22": ("8daf7d0", "2026-10-04"),
    "evidence/cp-23": ("928bc13", "2026-10-04"),
    "evidence/cp-24": ("05b7e61", "2026-10-10"),
}

POPULATIONS = {
    "common-10747h": "the 10,747 eligible development hours shared by CP-15, CP-16 and CP-20 to CP-24",
    "cp10-fold-block": "CP-10's full fold blocks, v1's nine-quantile scores",
    "v1-holdout-90d": "v1's pre-specified 90-day holdout, delivery 2026-06-09..2026-09-06",
}

_CP15 = "docs/track-b/research-content/cp15-cp16-claims.md"
_CP20 = "docs/track-b/research-content/cp20-claims.md"
_CP21 = "docs/track-b/research-content/cp21-claims.md"
_LANDING21 = "docs/track-b/cp-21-landing-2026-09-30.md"
#: CP-21's committed rows, the sources of its entries (the publication packet's §2).
_CP21_SOURCES = ("reports/block-challenger/metrics.csv", "reports/block-challenger/uncertainty.csv",
                 "reports/block-challenger/criteria.csv", "reports/block-challenger/adoption.json")
_PLAN21 = "capstone_v21.md"

_CP22 = "docs/track-b/research-content/cp22-claims.md"
_CP23 = "docs/track-b/research-content/cp23-claims.md"
_CP24 = "docs/track-b/research-content/cp24-claims.md"
_LANDING22 = "docs/track-b/cp-22-landing-2026-10-04.md"
_LANDING23 = "docs/track-b/cp-23-landing-2026-10-04.md"
_LANDING24 = "docs/track-b/cp-24-landing-2026-10-11.md"
#: Each checkpoint's committed rows, the sources of its entries (each draft registry's `sources`).
_CP22_SOURCES = ("reports/v4-revision/metrics.csv", "reports/v4-revision/uncertainty.csv",
                 "reports/v4-revision/criteria.csv", "reports/v4-revision/decisions.json")
_CP23_SOURCES = ("reports/distribution-challenger/metrics.csv", "reports/distribution-challenger/uncertainty.csv",
                 "reports/distribution-challenger/criteria.csv", "reports/distribution-challenger/decisions.json")
_CP24_SOURCES = ("reports/ddnn2/attempt-1/metrics.csv", "reports/ddnn2/attempt-1/uncertainty.csv",
                 "reports/ddnn2/attempt-1/criteria.csv", "reports/ddnn2/attempt-1/decisions.json")
#: A study arm's status, as the three draft registries word it.
_ATTRIBUTION_ONLY = "a study arm for attribution; never eligible"

#: The codes the CP-22, CP-23 and CP-24 draft registries add to existing entries
#: (`code_additions_to_existing_entries`): each experiment's saved rows of an identity evaluated first elsewhere.
#: None changes a published MLflow record: a run's name reads only its own experiment's code.
_LATER_CODES: dict[str, tuple[Code, ...]] = {
    "v4": (Code("CP-22", "HGL", "common-10747h", "comparator, saved"),
           Code("CP-23", "HGL", "common-10747h", "comparator, saved"),
           Code("CP-24", "HGL", "common-10747h", "comparator, saved")),
    "v3": (Code("CP-22", "HG", "common-10747h", "reference, saved"),
           Code("CP-23", "HG", "common-10747h", "reference, saved"),
           Code("CP-24", "HG", "common-10747h", "reference, saved")),
    "v2": (Code("CP-22", "H0", "common-10747h", "saved reference"),),
    "v1": (Code("CP-22", "B1", "common-10747h", "development replay"),
           Code("CP-23", "B1", "common-10747h", "development replay"),
           Code("CP-24", "B1", "common-10747h", "development replay")),
    "naive": (Code("CP-22", "B0", "common-10747h", "normalizer"), Code("CP-23", "B0", "common-10747h", "normalizer"),
              Code("CP-24", "B0", "common-10747h", "normalizer")),
    "daily-lear": (Code("CP-22", "B2", "common-10747h"), Code("CP-23", "B2", "common-10747h", "reference"),
                   Code("CP-24", "B2", "common-10747h", "reference")),
    "daily-lightgbm": (Code("CP-22", "B3", "common-10747h"),
                       Code("CP-23", "B3", "common-10747h", "section-8 comparator"),
                       Code("CP-24", "B3", "common-10747h", "section-8 comparator")),
    "normalized-lear": (Code("CP-22", "A1", "common-10747h"), Code("CP-23", "A1", "common-10747h", "reference"),
                        Code("CP-24", "A1", "common-10747h", "reference")),
    "ddnn-alone": (Code("CP-24", "D", "common-10747h", "CP-23's DDNN, saved"),),
}


def _later(ident: str) -> tuple[Code, ...]:
    return _LATER_CODES.get(ident, ())


def _arm(ident: str, checkpoint: str, code: str, name: str, subtitle: str, short: str, *, comparator: str,
         reason: str = _ATTRIBUTION_ONLY) -> Entry:
    """A study arm of CP-22, CP-23 or CP-24, as its draft registry enters it: not adopted on the day the checkpoint
    landed, sourced from its landing record."""
    plan, rules, sources, claim_map, landing, date = {
        "CP-22": (f"{_PLAN21} §20 (v21-r9)", ("cp22-replacement",), _CP22_SOURCES, _CP22, _LANDING22, "2026-10-04"),
        "CP-23": (f"{_PLAN21} §21 (v21-r10)", ("cp23-adoption",), _CP23_SOURCES, _CP23, _LANDING23, "2026-10-04"),
        "CP-24": (f"{_PLAN21} §23 (v21-r11)", ("cp24-adoption",), _CP24_SOURCES, _CP24, _LANDING24, "2026-10-11"),
    }[checkpoint]
    note = "scored attempt 1" if checkpoint == "CP-24" else ""
    return Entry(
        id=ident, name=name, subtitle=subtitle, kind="study arm",
        codes=(Code(checkpoint, code, "common-10747h", note), *_later(ident)),
        statuses=(StatusEvent(NOT_ADOPTED, date, landing, reason),),
        comparator=comparator, population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=plan, rules=rules, sources=sources, claim_map=claim_map,
        run_keys=(f"{checkpoint.lower().replace('-', '')}/{code}",), style="study", checkpoint=checkpoint, short=short,
    )


def _study(ident: str, code: str, name: str, subtitle: str, short: str = "") -> Entry:
    # A1 is the study's best challenger; CP-16 and CP-20 carry its saved rows as a comparator.
    later = (Code("CP-16", code, "common-10747h"), Code("CP-20", code, "common-10747h"),
             Code("CP-21", code, "common-10747h"), *_later(ident)) if code == "A1" else ()
    return Entry(
        id=ident, name=name, subtitle=subtitle, kind="study arm",
        codes=(Code("CP-15", code, "common-10747h"), *later),
        statuses=(StatusEvent(NOT_ADOPTED, "2026-09-16", "docs/track-b/cp-15-landing.md",
                              "no study policy met the product criteria"),),
        comparator="daily-lear", population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=f"{_PLAN21} §5–§8 (v21-r1)", rules=("criteria-1-2",),
        sources=("reports/cp15/relative_scores.csv", "reports/cp15/criteria.csv"), claim_map=_CP15,
        run_keys=(f"cp15/{code}",), style="study", checkpoint="CP-15", short=short or name,
    )


def _reference(ident: str, code: str, name: str, subtitle: str, *, comparator: str | None, note: str = "") -> Entry:
    return Entry(
        id=ident, name=name, subtitle=subtitle, kind="reference",
        codes=(Code("CP-15", code, "common-10747h", note), Code("CP-16", code, "common-10747h", note),
               Code("CP-20", code, "common-10747h", note), Code("CP-21", code, "common-10747h", note), *_later(ident)),
        statuses=(), comparator=comparator, population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=f"{_PLAN21} §5 (v21-r1)", rules=(), sources=("reports/cp15/relative_scores.csv",),
        claim_map=_CP15, run_keys=(f"cp15/{code}",), style="reference", checkpoint="CP-15", short=name,
    )


def _cp10(ident: str, code: str, name: str, subtitle: str, note: str = "") -> Entry:
    return Entry(
        id=ident, name=name, subtitle=subtitle, kind="study arm",
        codes=(Code("CP-10", code, "cp10-fold-block", note),),
        statuses=(StatusEvent(NOT_ADOPTED, "2026-09-16", "docs/track-b/cp-15-landing.md",
                              "recalibrating v1 without refitting it was not enough"),),
        comparator="v1", population="cp10-fold-block", evidence_class=EVIDENCE_CALIBRATION,
        plan="capstone_v20.md §4 and §9 CP-10", rules=(), sources=("reports/cp10/metrics.csv",),
        claim_map=_CP20, run_keys=(f"cp10/{code}",), style="study", checkpoint="CP-10", short=name,
    )


def _cp21_arm(ident: str, code: str, name: str, subtitle: str, short: str) -> Entry:
    return Entry(
        id=ident, name=name, subtitle=subtitle, kind="study arm",
        codes=(Code("CP-21", code, "common-10747h"),),
        statuses=(StatusEvent(NOT_ADOPTED, "2026-09-30", _LANDING21,
                              "a study arm for attribution; never eligible for adoption"),),
        comparator="v3", population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=f"{_PLAN21} §17 (v21-r6)", rules=("cp21-adoption",), sources=_CP21_SOURCES, claim_map=_CP21,
        run_keys=(f"cp21/{code}",), style="study", checkpoint="CP-21", short=short,
    )


#: Every published identity. The order within a kind is the order surfaces show it in:
#: generations oldest first (the lineage reads left to right; chapters reverse it), branches in
#: time order, references and study arms in the comparison's fixed order.
_ENTRIES: tuple[Entry, ...] = (
    # ---- generations -------------------------------------------------------------------
    Entry(
        id="v1", name="v1 · released LightGBM", subtitle="Nine-quantile LightGBM with calibrated intervals",
        kind="generation",
        codes=(Code("CP-15", "B1", "common-10747h", "development replay"),
               Code("CP-16", "B1", "common-10747h", "development replay"),
               Code("CP-20", "B1", "common-10747h", "development replay"),
               Code("CP-10", "v1_reference", "cp10-fold-block", "unscaled CQR reference"),
               Code("CP-21", "B1", "common-10747h", "development replay"), *_later("v1")),
        statuses=(StatusEvent(RELEASED, "2026-09-15", "land/cp-3 and land/cp-3b, 2026-09-15"),),
        comparator="naive", population="v1-holdout-90d", evidence_class=EVIDENCE_V1_HOLDOUT,
        plan="capstone_V6_8.md", rules=("v1-holdout",),
        sources=("reports/cp2/holdout_report.json", "reports/cp15/relative_scores.csv"),
        claim_map=None, run_keys=("cp15/B1", "cp10/v1_reference"), style="v1", anchor="#v1",
        checkpoint="CP-3", short="v1",
    ),
    Entry(
        id="v2", name="v2 · blended LEAR, hour-aware intervals",
        subtitle="Two LEAR forecasts blended, hour-aware intervals", kind="generation",
        codes=(Code("CP-16", "V2-H", "common-10747h"), Code("CP-20", "H0", "common-10747h"),
               Code("CP-21", "H0", "common-10747h", "saved reference"), *_later("v2")),
        statuses=(StatusEvent(ADOPTED_IN_RESEARCH, "2026-09-23", "docs/track-b/cp-16-landing-2026-09-23.md"),),
        comparator="daily-lear", population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=f"{_PLAN21} §14 (v21-r3)", rules=("criteria-1-2", "joint-improvement"),
        sources=("reports/v2-causal/metrics.csv", "reports/v2-causal/uncertainty.csv", "reports/v2-causal/criteria.csv"),
        claim_map=_CP15, run_keys=("cp16", "cp16/V2-H"), style="v2", anchor="#v2", checkpoint="CP-16", short="v2",
        predecessor="v1",
    ),
    Entry(
        id="v3", name="v3 · weather features", subtitle="v2 plus three weather forecast inputs",
        kind="generation",
        codes=(Code("CP-20", "HG", "common-10747h"), Code("CP-21", "HG", "common-10747h", "comparator, saved"),
               *_later("v3")),
        statuses=(StatusEvent(ADOPTED_IN_RESEARCH, "2026-09-24", "docs/track-b/cp-20-landing-2026-09-24.md"),),
        comparator="v2", population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=f"{_PLAN21} §15 (v21-r4)", rules=("criteria-1-2", "joint-improvement"),
        sources=("reports/weather-ablation/metrics.csv", "reports/weather-ablation/uncertainty.csv",
                 "reports/weather-ablation/criteria.csv"),
        claim_map=_CP20, run_keys=("cp20", "cp20/HG"), style="v3", anchor="#v3", checkpoint="CP-20", short="v3",
        predecessor="v2",
    ),
    Entry(
        id="v4", name="v4 · three-block LightGBM added", subtitle="v3 plus a three-block LightGBM member",
        kind="generation",
        codes=(Code("CP-21", "HGL", "common-10747h"), *_later("v4")),
        statuses=(StatusEvent(ADOPTED_IN_RESEARCH, "2026-09-30", _LANDING21),),
        comparator="v3", population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=f"{_PLAN21} §17 (v21-r6)", rules=("cp21-adoption",), sources=_CP21_SOURCES,
        claim_map=_CP21, run_keys=("cp21", "cp21/HGL"), style="v4", anchor="#v4", checkpoint="CP-21", short="v4",
        predecessor="v3",
    ),
    # PRES-4: the packet's draft entry (reports/ddnn2/draft-registry.json) as drafted, with the Owner's name of
    # 2026-10-11 and its status dated and sourced from the CP-24 landing record.
    Entry(
        id="v5", name="v5 · DDNN-2 added", subtitle="v4 plus a DDNN-2 member", kind="generation",
        codes=(Code("CP-24", "v5", "common-10747h", "scored attempt 1"),),
        statuses=(StatusEvent(ADOPTED_IN_RESEARCH, "2026-10-11", _LANDING24),),
        comparator="v4", population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=f"{_PLAN21} §23 (v21-r11)", rules=("cp24-adoption",), sources=_CP24_SOURCES,
        claim_map=_CP24, run_keys=("cp24", "cp24/v5"), style="v5", anchor="#v5", checkpoint="CP-24", short="v5",
        predecessor="v4",
    ),
    # ---- branches ------------------------------------------------------------------------
    Entry(
        id="calibration", name="Calibration experiment", subtitle="Recalibrating v1 without refitting it",
        kind="branch", codes=(Code("programme", "CP-10", "cp10-fold-block"),),
        statuses=(StatusEvent(NOT_ADOPTED, "2026-09-16", "docs/track-b/cp-15-landing.md",
                              "recalibrating v1 without refitting it was not enough"),),
        comparator="v1", population="cp10-fold-block", evidence_class=EVIDENCE_CALIBRATION,
        plan="capstone_v20.md §4 and §9 CP-10", rules=(), sources=("reports/cp10/peak_windows.csv",),
        claim_map=_CP20, run_keys=("cp10",), style="branch", anchor="#branch-calibration",
        checkpoint="CP-10", after="v1",
        question="Can recalibrating v1, without refitting it, repair its coverage in the 2022 crisis?",
        short="Calibration experiment",
    ),
    Entry(
        id="model-comparison", name="Model comparison study", subtitle="Nine forecasting policies on identical hours",
        kind="branch", codes=(Code("programme", "CP-15", "common-10747h"),),
        statuses=(StatusEvent(NOT_ADOPTED, "2026-09-16", "docs/track-b/cp-15-landing.md",
                              "no policy met the product criteria"),),
        comparator="daily-lear", population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=f"{_PLAN21} §5–§8 (v21-r1)", rules=("criteria-1-2",),
        sources=("reports/cp15/relative_scores.csv", "reports/cp15/peak.csv", "reports/cp15/criteria.csv"),
        claim_map=_CP15, run_keys=("cp15",), style="branch", anchor="#branch-model-comparison",
        checkpoint="CP-15", after="calibration", informed="v2",
        question="Which forecasting approach copes best with the 2022 crisis?",
        short="Model comparison study",
    ),
    # PRES-4: CP-22's and CP-23's branches, from their draft registries; each was tested against v4 and not adopted.
    Entry(
        id="pooled-lightgbm-member-for-v4", name="Pooled LightGBM member for v4",
        subtitle="The block split removed, under non-inferiority", kind="branch",
        codes=(Code("CP-22", "CP-22", "common-10747h"),),
        statuses=(StatusEvent(NOT_ADOPTED, "2026-10-04", _LANDING22,
                              "cp22-replacement yielded no replacement: for R and for M, condition 4 of rule "
                              "cp22-replacement, no resolved per-fold degradation, was not met; the Owner kept v4 "
                              "unchanged"),),
        comparator="v4", population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=f"{_PLAN21} §20 (v21-r9)", rules=("cp22-replacement",), sources=_CP22_SOURCES, claim_map=_CP22,
        run_keys=("cp22",), style="branch", anchor="#branch-pooled-lightgbm-member-for-v4", checkpoint="CP-22",
        after="v4", question="Can one pooled LightGBM member replace v4's three-block member without losing accuracy?",
        short="Pooled member for v4",
    ),
    Entry(
        id="ddnn-member-on-v4", name="DDNN member on v4", subtitle="v4 plus a DDNN member, under rule cp23-adoption",
        kind="branch", codes=(Code("CP-23", "v5", "common-10747h"),),
        statuses=(StatusEvent(NOT_ADOPTED, "2026-10-04", _LANDING23,
                              "condition 1 of rule cp23-adoption, a joint improvement over v4, with the upper 95% "
                              "endpoint of the interval-score difference below zero and that of the point-error "
                              "difference at or below zero, was not met"),),
        comparator="v4", population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=f"{_PLAN21} §21 (v21-r10)", rules=("cp23-adoption",), sources=_CP23_SOURCES, claim_map=_CP23,
        run_keys=("cp23", "cp23/v5"), style="branch", anchor="#branch-ddnn-member-on-v4", checkpoint="CP-23",
        after="v4", question="Does a distributional neural network, added to v4 as a one-third member, improve on v4 "
                             "jointly in point and interval accuracy?",
        short="DDNN member on v4",
    ),
    # ---- references ----------------------------------------------------------------------
    _reference("naive", "B0", "Similar-day naive", "The forecast every error score divides by",
               comparator=None, note="normalizer"),
    _reference("daily-lear", "B2", "Daily LEAR", "The strongest benchmark: a daily linear model",
               comparator="naive"),
    _reference("daily-lightgbm", "B3", "Daily LightGBM", "Gradient-boosted trees refitted every day",
               comparator="naive"),
    # ---- study arms (CP-15) --------------------------------------------------------------
    _study("normalized-lear", "A1", "Normalized LEAR", "LEAR on a normalized price target"),
    _study("normalized-lightgbm", "A2", "Normalized LightGBM", "LightGBM on a normalized price target"),
    _study("component-mean", "A3", "Normalized component mean", "Mean of the normalized components"),
    _study("normalized-lear-84d", "A4", "84-day normalized LEAR", "Normalized LEAR with a shorter history"),
    _study("component-mean-variant", "A5", "Normalized component mean, variant",
           "A variant of the normalized component mean"),
    # ---- control (CP-16) -----------------------------------------------------------------
    Entry(
        id="pooled-control", name="Pooled-interval control", subtitle="v2's blend with pooled residual intervals",
        kind="control", codes=(Code("CP-16", "V2-P", "common-10747h"),),
        statuses=(StatusEvent(NOT_ADOPTED, "2026-09-23", "docs/track-b/cp-16-landing-2026-09-23.md",
                              "the control that isolates hour-aware intervals"),),
        comparator="daily-lear", population="common-10747h", evidence_class=EVIDENCE_DEVELOPMENT,
        plan=f"{_PLAN21} §14 (v21-r3)", rules=("criteria-1-2", "joint-improvement"),
        sources=("reports/v2-causal/metrics.csv", "reports/v2-causal/uncertainty.csv"),
        claim_map=_CP15, run_keys=("cp16/V2-P",), style="control", checkpoint="CP-16",
        short="Pooled-interval control",
    ),
    # ---- study arms (CP-21): the ladder that attributes v4's change; never eligible for adoption ----------
    _cp21_arm("pooled-lightgbm-weather", "L-P", "Pooled LightGBM with weather", "One 24-hour LightGBM on v3 information",
              "Pooled LightGBM"),
    _cp21_arm("block-lightgbm", "L-R", "Three-block LightGBM", "Night, solar and peak models, raw price",
              "Block LightGBM"),
    _cp21_arm("normalized-block-lightgbm", "L-N", "Normalized three-block LightGBM",
              "The block models on a normalized price", "Normalized block LightGBM"),
    # ---- study arms (CP-22, CP-23, CP-24): from their draft registries; never adopted -----------------------
    _arm("v3-plus-pooled-normalized-lightgbm-averaged", "CP-22", "R", "v3 plus a capacity-averaged pooled LightGBM",
         "v3 with a normalized pooled LightGBM, capacities averaged", "Pooled member, averaged", comparator="v4",
         reason="condition 4 of rule cp22-replacement, no resolved per-fold degradation, was not met"),
    _arm("v3-plus-pooled-lightgbm-pair", "CP-22", "M", "v3 plus pooled normalized and raw LightGBM",
         "v4 with the block split removed", "Pooled pair", comparator="v4",
         reason="condition 4 of rule cp22-replacement, no resolved per-fold degradation, was not met"),
    _arm("v3-plus-pooled-normalized-lightgbm-selected", "CP-22", "A-PN-sel", "v3 plus a pooled normalized LightGBM",
         "The raw half dropped; daily capacity selection", "Pooled normalized member", comparator="v4"),
    _arm("v3-plus-pooled-raw-lightgbm", "CP-22", "A-LP", "v3 plus a pooled raw LightGBM",
         "v3 with the pooled raw LightGBM member", "Pooled raw member", comparator="v4"),
    _arm("v3-plus-normalized-block-lightgbm", "CP-22", "A-LN", "v3 plus normalized block LightGBM",
         "v3 with the normalized three-block member", "Normalized block member", comparator="v4"),
    _arm("v4-with-dynamic-intervals", "CP-22", "v4+DL", "v4 with dynamic intervals",
         "Three-block v4 with the dynamic interval layer", "v4 + dynamic layer", comparator="v4"),
    _arm("v3-with-dynamic-intervals", "CP-22", "v3+DL", "v3 with dynamic intervals",
         "v3 with the dynamic interval layer", "v3 + dynamic layer", comparator="v4"),
    # CP-23's draft subtitle had thirteen words; the registry's field holds at most eight (standard §5), so it is
    # shortened to its first clause and its last: no other field of the draft entry changes.
    _arm("ddnn-alone", "CP-23", "D", "DDNN alone",
         "A distributional neural network with its own quantiles", "DDNN alone", comparator="v4"),
    _arm("v3-plus-ddnn", "CP-23", "v3+D", "v3 plus a DDNN member", "DDNN as v3's third member, in LightGBM's place",
         "v3 + DDNN", comparator="v3"),
    _arm("ddnn2-alone", "CP-24", "D2", "DDNN-2 alone", "A day-level distributional network with its own quantiles",
         "DDNN-2 alone", comparator="v4"),
    _arm("v3-plus-ddnn2", "CP-24", "v3+D2", "v3 plus a DDNN-2 member", "DDNN-2 as v3's third member, in LightGBM's place",
         "v3 + DDNN-2", comparator="v3"),
    # ---- the calibration experiment's arms (CP-10) ---------------------------------------
    _cp10("cp10-head-spread", "c1_head_spread", "Scaled conformal, head spread",
          "Conformal scores scaled by the heads' spread"),
    _cp10("cp10-price-volatility", "c1_price_volatility", "Scaled conformal, price volatility",
          "Conformal scores scaled by recent price volatility", "selected scale"),
    _cp10("cp10-aci-0.000001", "c2_aci_gamma_0.000001", "Adaptive conformal, gamma 0.000001",
          "Adaptive conformal intervals, smallest step"),
    _cp10("cp10-aci-0.000005", "c2_aci_gamma_0.000005", "Adaptive conformal, gamma 0.000005",
          "Adaptive conformal intervals, second step"),
    _cp10("cp10-aci-0.00001", "c2_aci_gamma_0.00001", "Adaptive conformal, gamma 0.00001",
          "Adaptive conformal intervals, third step"),
    _cp10("cp10-aci-0.00002", "c2_aci_gamma_0.00002", "Adaptive conformal, gamma 0.00002",
          "Adaptive conformal intervals, largest step", "selected step"),
)

#: The order of the shared comparison's rows (plan §8.2): generations newest first, then the candidates of the branches
#: tested against v4 and not adopted, as study points (the Owner's decision D2 of 2026-10-11), then the benchmarks and
#: the study's best challenger by score, v1's development replay, and the naive.
COMPARISON_ORDER = ("v5", "v4", "v3", "v2", "v3-plus-pooled-normalized-lightgbm-averaged", "v3-plus-pooled-lightgbm-pair",
                    "ddnn-member-on-v4", "daily-lear", "normalized-lear", "daily-lightgbm", "v1", "naive")

#: The shared comparison as v4's chapter published it, from CP-21's rows (PRES-3). It redraws the overview that the
#: `cp21/HGL` MLflow run carries as an artifact, byte for byte: a published record never changes (§17.9).
V4_COMPARISON_ORDER = ("v4", "v3", "v2", "daily-lear", "normalized-lear", "daily-lightgbm", "v1", "naive")

#: The shared comparison as v3's chapter published it, from CP-20's rows. It redraws the overview that the
#: `cp20/HG` MLflow run carries as an artifact, byte for byte: a published record never changes (§17.9).
V3_COMPARISON_ORDER = ("v3", "v2", "daily-lear", "normalized-lear", "daily-lightgbm", "v1", "naive")

#: The marker-key words for the non-generation styles.
STYLE_LABELS = {"reference": "reference", "study": "study arm", "control": "control arm"}


#: The order of the scores chart at the time of v2 (the v2 chapter's protocol detail).
V2_SCORES_ORDER = ("v2", "pooled-control", "daily-lear", "normalized-lear", "daily-lightgbm", "v1")

#: The crisis window's rows in v3's chapter: v1, the study's best challenger, then the generations after it.
#: Pinned, as the chapter and the `cp20/HG` MLflow run published them.
V3_CRISIS_ORDER = ("v1", "normalized-lear", "v2", "v3")

#: The crisis window's rows in v4's chapter, which shows the window (runbook §2 step 6): the adopted
#: generations CP-21 scored on it, oldest first, so v4's peak error sits beside v3's on one readable scale.
CRISIS_ORDER = ("v2", "v3", "v4")

#: The release rule, stated once (standard §5): the demo runs the released model.
RELEASE_RULE = (
    "Research generations are not released one by one; only the final model, after its one-shot test "
    "and live run, replaces the released one."
)


# --------------------------------------------------------------------------- lookups


@lru_cache(maxsize=1)
def entries() -> tuple[Entry, ...]:
    ids = [entry.id for entry in _ENTRIES]
    if len(ids) != len(set(ids)):
        raise RegistryError("duplicate registry id")
    return _ENTRIES


@lru_cache(maxsize=1)
def _by_id() -> dict[str, Entry]:
    return {entry.id: entry for entry in entries()}


def get(ident: str) -> Entry:
    try:
        return _by_id()[ident]
    except KeyError:
        raise RegistryError(f"no registry entry {ident!r}") from None


@lru_cache(maxsize=1)
def _by_scoped_code() -> dict[tuple[str, str], Entry]:
    """(experiment, code) -> identity. Inside one experiment a code names one identity, or the registry refuses."""
    out: dict[tuple[str, str], Entry] = {}
    for entry in entries():
        for code in entry.codes:
            key = (code.experiment, code.code)
            if key in out and out[key].id != entry.id:
                raise RegistryError(f"code {code.code!r} names two identities in {code.experiment}")
            out[key] = entry
    return out


@lru_cache(maxsize=1)
def _by_code() -> dict[str, Entry]:
    """code -> identity, for the codes that name one identity in every experiment that uses them."""
    named: dict[str, set[str]] = {}
    for (_, code), entry in _by_scoped_code().items():
        named.setdefault(code, set()).add(entry.id)
    return {code: get(next(iter(ids))) for code, ids in named.items() if len(ids) == 1}


def by_code(code: str, experiment: str | None = None) -> Entry:
    """The one identity behind an experiment code: `H0` and `V2-H` are both v2. A code whose text names different
    identities in different experiments (`v5`) resolves only with its experiment."""
    if experiment is not None:
        try:
            return _by_scoped_code()[(experiment, code)]
        except KeyError:
            raise RegistryError(f"no registry entry carries the code {code!r} in {experiment}") from None
    try:
        return _by_code()[code]
    except KeyError:
        if any(key[1] == code for key in _by_scoped_code()):
            raise RegistryError(f"the code {code!r} names different identities in different experiments; "
                                "name the experiment") from None
        raise RegistryError(f"no registry entry carries the code {code!r}") from None


def codes() -> tuple[str, ...]:
    """Every experiment code in the registry: the lint's generated denylist (standard §4)."""
    return tuple(sorted({code.code for entry in entries() for code in entry.codes}))


def of_kind(kind: str) -> tuple[Entry, ...]:
    if kind not in KINDS:
        raise RegistryError(f"unknown kind {kind!r}")
    return tuple(entry for entry in entries() if entry.kind == kind)


def generations(*, newest_first: bool = False) -> tuple[Entry, ...]:
    ordered = sorted(of_kind("generation"), key=lambda entry: int(entry.version[1:]))
    return tuple(reversed(ordered)) if newest_first else tuple(ordered)


def branches() -> tuple[Entry, ...]:
    return of_kind("branch")


def generation_of(code: str, experiment: str | None = None) -> str | None:
    """The adopted generation a code belongs to, or None: references and study arms have none. In CP-23, `v5` is
    the branch's candidate and has none; in CP-24 it is v5."""
    entry = (_by_scoped_code().get((experiment, code)) if experiment is not None else _by_code().get(code))
    return entry.version if entry is not None else None


def with_status(status: str) -> tuple[Entry, ...]:
    return tuple(entry for entry in entries() if entry.status is not None and entry.status.status == status)


def current_generation() -> Entry:
    """The newest generation adopted in research or beyond: the headline's subject (§3.4)."""
    candidates = [entry for entry in generations()
                  if entry.status is not None and entry.status.status in HERO_ORDER]
    if not candidates:
        raise RegistryError("no adopted generation")
    return candidates[-1]


def released() -> Entry:
    """The released model: the one the demo runs (the release rule, §5)."""
    found = [entry for entry in generations() if entry.status is not None and entry.status.status == RELEASED]
    if len(found) != 1:
        raise RegistryError(f"exactly one released model is required; found {len(found)}")
    return found[0]


def hero() -> Entry:
    """The model with the highest status reached: live, final candidate, released, research (§2)."""
    for status in HERO_ORDER:
        found = [entry for entry in generations() if entry.status is not None and entry.status.status == status]
        if found:
            return found[-1]
    raise RegistryError("no generation has a status")


def population_in(entry: Entry, experiment: str) -> str:
    """The comparability ID of an identity's evaluated population in one experiment."""
    for code in entry.codes:
        if code.experiment == experiment:
            return code.population
    raise RegistryError(f"{entry.id} was not evaluated in {experiment}")


#: The newest experiment of the shared comparison (plan §8.2): its rows carry the targets' line and the population's
#: counts.
COMPARISON_EXPERIMENT = "CP-24"

#: The experiments whose committed rows the live comparison draws, newest first. Each row comes from the newest one
#: that evaluated its identity on the shared population: CP-24 carries no v2 and no branch candidate, so v2 comes from
#: CP-22, CP-22's candidates from CP-22 and CP-23's from CP-23. They are one population, not a join of incompatible
#: results: every identity two of them carry has byte-identical committed rows there
#: (`delu_forecast.derived.comparison_parity_problems`), and the build refuses otherwise (PUBLISH_RULES §3.2).
COMPARISON_EXPERIMENTS = ("CP-24", "CP-23", "CP-22", "CP-21")

#: Each experiment's record prefix in `delu_forecast.research`.
EXPERIMENT_PREFIX = {"CP-15": "cp15", "CP-16": "cp16", "CP-20": "cp20", "CP-21": "cp21", "CP-22": "cp22",
                     "CP-23": "cp23", "CP-24": "cp24"}


def comparison_prefix(experiment: str | None = None) -> str:
    """The record prefix of the experiment whose rows a comparison draws: `cp24.metrics.…`."""
    return EXPERIMENT_PREFIX[experiment or COMPARISON_EXPERIMENT]


def comparison_source(entry: Entry, experiment: str | None = None) -> str:
    """The experiment a comparison row is drawn from: a pinned comparison's own experiment, or, for the live one,
    the newest of `COMPARISON_EXPERIMENTS` that carries the identity."""
    if experiment is not None:
        return experiment
    for candidate in COMPARISON_EXPERIMENTS:
        if entry.code_in(candidate) is not None:
            return candidate
    raise RegistryError(f"{entry.id} was not evaluated in any of {COMPARISON_EXPERIMENTS}")


def comparison_rows(order: tuple[str, ...] | None = None, experiment: str | None = None) -> tuple[Entry, ...]:
    """The shared comparison's rows, refusing a mix of evaluated populations (standard §5). Without arguments, the
    live comparison, each row from its `comparison_source`; v3's pinned one is
    `comparison_rows(V3_COMPARISON_ORDER, "CP-20")` and v4's `comparison_rows(V4_COMPARISON_ORDER, "CP-21")`."""
    order = order or COMPARISON_ORDER
    rows = tuple(get(ident) for ident in order)
    populations = {population_in(entry, comparison_source(entry, experiment)) for entry in rows}
    if len(populations) != 1:
        raise RegistryError(f"one comparison chart holds one comparability ID; got {sorted(populations)}")
    return rows


def checkpoint_of_run_key(run_key: str) -> Checkpoint:
    prefix = run_key.split("/")[0]
    for checkpoint in CHECKPOINTS.values():
        if checkpoint.run_key == prefix:
            return checkpoint
    raise RegistryError(f"no checkpoint owns the run key {run_key!r}")


def entry_for_run_key(run_key: str) -> Entry:
    for entry in entries():
        if run_key in entry.run_keys:
            return entry
    raise RegistryError(f"no registry entry owns the run key {run_key!r}")


# --------------------------------------------------------------------------- derived wording


def month(iso: str) -> str:
    day = date.fromisoformat(iso)
    return f"{MONTHS[day.month - 1]} {day.year}"


def status_sentence(entry: Entry) -> str:
    """The dated, past-tense status (standard §5): 'In September 2026, v3 was adopted in research.'"""
    event = entry.status
    if event is None:
        raise RegistryError(f"{entry.id} has no status")
    subject = entry.version or f"the {inline_name(entry)}"  # keeps an acronym's case: "the DDNN member on v4"
    verb = {"adopted in research": "was adopted in research", "not adopted": "was not adopted",
            "released": "was released", "final candidate": "became the final candidate",
            "live": "went live", "retired": "was retired"}[event.status]
    return f"In {month(event.date)}, {subject} {verb}."


def run_role(run_key: str) -> str:
    """A run's role inside its checkpoint (plan §10.5): an identity first produced elsewhere is a
    saved reference there; otherwise its kind decides."""
    entry, checkpoint = entry_for_run_key(run_key), checkpoint_of_run_key(run_key)
    if entry.checkpoint != checkpoint.code or entry.kind == "reference":
        return "reference"
    return "control" if entry.kind == "control" else "candidate"


def adopted_flag(entry: Entry) -> str:
    """`delu.adopted`: true once adopted in research or beyond, false when not adopted, n/a otherwise."""
    if entry.status is None:
        return "n/a"
    return "false" if entry.status.status in (NOT_ADOPTED, RETIRED) else "true"


def parent_run_keys() -> tuple[str, ...]:
    """The `delu-generations` parents, one per checkpoint, in time order."""
    return tuple(checkpoint.run_key for checkpoint in CHECKPOINTS.values())


def expected_run_keys() -> tuple[str, ...]:
    """Every `delu-generations` run the registry expects: each checkpoint's parent, then its children."""
    out = []
    for checkpoint in CHECKPOINTS.values():
        out.append(checkpoint.run_key)
        out += [f"{checkpoint.run_key}/{code}" for code in checkpoint.children]
    return tuple(out)


@dataclass(frozen=True)
class Transition:
    """One adopted generation replacing its predecessor in research (PUBLISH_RULES 1.0, A3)."""

    predecessor: Entry
    successor: Entry

    @property
    def id(self) -> str:
        return f"{self.predecessor.id}-{self.successor.id}"

    @property
    def comparator(self) -> Entry:
        """The comparator the governing protocol named for the successor, which may differ from the
        predecessor: v2 was judged against daily LEAR, v3 against v2."""
        return get(self.successor.comparator)

    @property
    def comparator_is_predecessor(self) -> bool:
        return self.successor.comparator == self.predecessor.id


def first_status(entry: Entry) -> StatusEvent:
    """The event that put an identity on the main line: its adoption or release, dated."""
    if not entry.statuses:
        raise RegistryError(f"{entry.id} has no status")
    return entry.statuses[0]


def transition_problems(entries_: tuple[Entry, ...] | None = None) -> list[str]:
    """Why the predecessor relationships are not one dated chain of adopted generations."""
    pool = entries_ if entries_ is not None else entries()
    by_id = {entry.id: entry for entry in pool}
    gens = [entry for entry in pool if entry.kind == "generation"]
    problems = []
    for entry in pool:
        if entry.predecessor is not None and entry.kind != "generation":
            problems.append(f"{entry.id}: only a generation has a predecessor")
    firsts = [entry.id for entry in gens if entry.predecessor is None]
    if len(firsts) != 1:
        problems.append(f"exactly one generation starts the main line; found {firsts}")
    successors: dict[str, str] = {}
    for entry in gens:
        if entry.predecessor is None:
            continue
        before = by_id.get(entry.predecessor)
        if before is None or before.kind != "generation":
            problems.append(f"{entry.id}: predecessor {entry.predecessor!r} is not a registered generation")
            continue
        if entry.predecessor in successors:
            problems.append(f"{entry.predecessor} precedes both {successors[entry.predecessor]} and {entry.id}")
        successors[entry.predecessor] = entry.id
        if not (entry.statuses and before.statuses) or first_status(before).date > first_status(entry).date:
            problems.append(f"{entry.id}: adopted before its predecessor {before.id}")
    # one chain: following predecessors from every generation reaches the first without a cycle
    for entry in gens:
        seen, node = set(), entry
        while node is not None and node.predecessor is not None:
            if node.id in seen:
                problems.append(f"{entry.id}: the predecessor chain loops")
                break
            seen.add(node.id)
            node = by_id.get(node.predecessor)
    return problems


def transitions(*, newest_first: bool = True) -> tuple[Transition, ...]:
    """Every adopted transition, from the validated predecessor relationships (A3)."""
    problems = transition_problems()
    if problems:
        raise RegistryError("; ".join(problems))
    out = [Transition(get(entry.predecessor), entry) for entry in generations() if entry.predecessor]
    out.sort(key=lambda item: first_status(item.successor).date, reverse=newest_first)
    return tuple(out)


def transition_into(entry: Entry) -> Transition | None:
    return next((item for item in transitions() if item.successor.id == entry.id), None)


def inline_name(entry: Entry) -> str:
    """The name inside a sentence: 'daily LEAR', 'the similar-day naive'; a version keeps its case."""
    if entry.version or entry.name[:2].isupper():
        return entry.name
    return entry.name[0].lower() + entry.name[1:]


ADOPTION_LABELS = {ADOPTED_IN_RESEARCH: "Adopted in research", NOT_ADOPTED: "Not adopted", RELEASED: "Released",
                   FINAL_CANDIDATE: "Final candidate", LIVE: "Live", RETIRED: "Retired"}


def adoption_label(entry: Entry) -> str:
    """The adoption label (plan §7.8): text, separate from the badge, from the dated status."""
    if entry.status is None:
        raise RegistryError(f"{entry.id} has no status")
    return ADOPTION_LABELS[entry.status.status]


def release_sentence() -> str:
    """The release rule, stated once (standard §5), with the released model named from the history."""
    return f"The demo runs the released model, {released().version}. {RELEASE_RULE}"


def resolve(ident: str) -> Entry:
    """A registry id, or one of the derived aliases `current` and `released`."""
    if ident == "current":
        return current_generation()
    if ident == "released":
        return released()
    return get(ident)


def common_run_key(entry: Entry) -> str:
    """An identity's run on the shared population (CP-15/16/20): v2's is CP-16's V2-H, never H0."""
    for key in entry.run_keys:
        if "/" in key and not key.startswith("cp10/"):
            return key
    raise RegistryError(f"{entry.id} has no run on the shared population")


def expected_routes() -> dict[str, tuple[str, tuple[str, ...]]]:
    """Every MLflow reader route the surfaces advertise once it is verified (standard §9), as
    `{route id: (kind, run keys)}`: the experiment, the shared comparison, each research
    generation against its comparator, and each branch's own runs. Nothing else is advertised."""
    routes: dict[str, tuple[str, tuple[str, ...]]] = {"experiment": ("experiment", ())}
    routes["compare:overview"] = ("compare", tuple(common_run_key(entry) for entry in comparison_rows()))
    for entry in generations():
        checkpoint = CHECKPOINTS.get(entry.checkpoint)
        if checkpoint is None:
            continue  # v1's runs are its own experiment's, delu-cp2
        runs = tuple(f"{checkpoint.run_key}/{code}" for code in checkpoint.children)
        routes[f"compare:{entry.id}"] = ("compare", runs + (common_run_key(get(entry.comparator)),))
    for entry in branches():
        checkpoint = CHECKPOINTS[entry.checkpoint]
        routes[f"compare:{entry.id}"] = ("compare", tuple(f"{checkpoint.run_key}/{code}" for code in checkpoint.children))
    return routes


def mlflow_run_name(run_key: str) -> str:
    """A run's public name in `delu-generations`: its registry name, then its code in that experiment."""
    if "/" not in run_key:
        checkpoint = checkpoint_of_run_key(run_key)
        return f"{get(checkpoint.owner).name} ({checkpoint.code})"
    entry = entry_for_run_key(run_key)
    checkpoint = checkpoint_of_run_key(run_key)
    code = run_key.split("/", 1)[1]
    note = next((c.note for c in entry.codes if c.code == code and c.experiment == checkpoint.code), "")
    if entry.code_in(checkpoint.code) != code:
        raise RegistryError(f"{run_key}: {entry.id} has no code {code} in {checkpoint.code}")
    return f"{entry.name} ({code}{', ' + note if note else ''})"


__all__ = [
    "ADOPTED_IN_RESEARCH", "BADGES", "CHECKPOINTS", "CLASS_CAVEATS", "COMPARISON_ORDER", "CRISIS_ORDER", "Checkpoint", "Code",
    "EVIDENCE_CALIBRATION", "EVIDENCE_DEVELOPMENT", "EVIDENCE_TAGS", "EVIDENCE_V1_HOLDOUT", "Entry",
    "FINAL_CANDIDATE", "HERO_ORDER", "KINDS", "LIVE", "NOT_ADOPTED", "POPULATIONS", "RELEASED", "RELEASE_RULE",
    "RETIRED", "RULES", "STYLE_LABELS", "RegistryError", "Rule", "STATUSES", "StatusEvent", "Transition",
    "V2_SCORES_ORDER", "V3_COMPARISON_ORDER", "V4_COMPARISON_ORDER", "V3_CRISIS_ORDER", "EXPERIMENT_PREFIX", "COMPARISON_EXPERIMENTS", "comparison_prefix", "comparison_source", "branches", "by_code", "codes", "first_status", "transition_into", "transition_problems",
    "transitions",
    "adopted_flag", "common_run_key", "comparison_rows", "expected_routes", "expected_run_keys", "run_role", "COMPARISON_EXPERIMENT", "current_generation", "entries", "entry_for_run_key", "generation_of", "generations",
    "get", "hero", "mlflow_run_name", "month", "adoption_label", "inline_name", "of_kind", "parent_run_keys", "release_sentence", "resolve", "population_in", "released", "status_sentence", "with_status",
]
