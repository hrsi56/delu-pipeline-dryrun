"""The research claims every public surface may render, and how (presentation plan §9.1, §9.3).

Each piece of research wording is a **block**: an approved template bound to one claim ID from
the claim maps (`docs/track-b/research-content/cp15-cp16-claims.md` and `cp20-claims.md`) and to
the evidence records its numbers come from. A generator never types a research number; it asks
for a block, and the block fills its numbers from `delu_forecast.research`.

Template tokens:

* `{r:RECORD}` a record's value; `{r:RECORD|ci}` its interval "[low, high]"; `|lo`, `|hi` one
  endpoint; `|pct` a fraction as a percentage; `|exact` / `|exact_hi` the full decimal
  expansion; `|p=N` N decimal places.
* `{s:KIND:TEXT}` a structural numeral -- a version label, date, fold index, section number or
  control value -- declared as such so it cannot carry a research result.
* `{v1:KEY}` a v1 claim from `claims.py`, the one v1 claim set.
* `**bold**` and `` `code` `` as in Markdown.

In HTML a value becomes `<data value=… data-claim=… data-record=…>`; inside SVG the renderer puts
the same `data-claim`/`data-record` attributes on the `<text>` or mark itself (`<data>` is not
valid SVG). In Markdown the same template renders plain text, so the README and the page cannot
disagree about a number. `tests/test_30_research_claims_rendered.py` holds the guards.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

from . import derived as D
from . import registry as G
from . import research as R

REPO_ROOT = R.REPO_ROOT
CLAIM_MAPS = (
    REPO_ROOT / "docs" / "track-b" / "research-content" / "cp15-cp16-claims.md",
    REPO_ROOT / "docs" / "track-b" / "research-content" / "cp20-claims.md",
    REPO_ROOT / "docs" / "track-b" / "research-content" / "cp21-claims.md",
    REPO_ROOT / "docs" / "track-b" / "research-content" / "cp22-claims.md",
    REPO_ROOT / "docs" / "track-b" / "research-content" / "cp23-claims.md",
    REPO_ROOT / "docs" / "track-b" / "research-content" / "cp24-claims.md",
    REPO_ROOT / "docs" / "track-b" / "research-content" / "publication-claims.md",
)

# --------------------------------------------------------------------------- names and labels

# Names, kinds and statuses are not typed here: they come from `delu_forecast.registry`
# (standard §5), which replaced the hand-written POLICY_NAMES and POLICY_ROLES maps.

#: Evidence badges (plan §7.8). The v1 text is verbatim and may wrap; it is never shortened.
BADGE_DEVELOPMENT = "Development · post-selection"
BADGE_V1_HOLDOUT = "Confirmatory-style, not power-qualified"
BADGE_PROSPECTIVE = "Prospective"  # reserved for the live model; not rendered yet (plan §15)

#: Adoption labels: text, separate from the badge, and never a success colour or check mark.
ADOPTED = "Adopted"
NOT_ADOPTED = "Not adopted"
PLANNED = "Planned, not evaluated"

#: The Owner's contribution statement (plan §8.9), displayed exactly as written. It is not a
#: research claim, and only the Owner changes it.
#: The Owner's contribution statement. Plan §8.9 carried the first English rendering; at Stop 1
#: (2026-09-25) the Owner approved the wording proposed in the D1 editorial review, shown as written.
CONTRIBUTION_STATEMENT = (
    "I led the project's problem definition, evaluation criteria and research direction, and made "
    "the decisions on model adoption and product presentation. AI agents assisted with "
    "implementation, analysis and documentation. Automated tests and reviews separate from "
    "implementation supported verification. I retained responsibility for approving deliverables "
    "and publication."
)

#: The Owner's approved public name (Stop 1, 2026-09-25), used once as a byline near the opening
#: and to sign the contribution statement.
OWNER_PUBLIC_NAME = "Yarden Viktor Dejorno"

# --------------------------------------------------------------------------- blocks


@dataclass(frozen=True)
class Block:
    key: str
    claim_id: str
    template: str
    surfaces: frozenset[str]
    status: str = "published"

    def records(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(match.group(1) for match in _RECORD_TOKEN.finditer(self.template)))


def _block(key: str, claim_id: str, template: str, *surfaces: str) -> Block:
    return Block(key, claim_id, " ".join(template.split()), frozenset(surfaces or ("page",)))


PAGE, README = "page", "readme"

_HG_MAE = "cp20.uncertainty.HG-H0.equal_fold.MAE"
_HG_WIS = "cp20.uncertainty.HG-H0.equal_fold.WIS"
_HP_MAE = "cp16.uncertainty.V2-H-V2-P.equal_fold.MAE"
_HP_WIS = "cp16.uncertainty.V2-H-V2-P.equal_fold.WIS"
_HB_MAE = "cp16.uncertainty.V2-H-B2.equal_fold.MAE"
_HB_WIS = "cp16.uncertainty.V2-H-B2.equal_fold.WIS"
_PB_MAE = "cp16.uncertainty.V2-P-B2.equal_fold.MAE"
_PB_WIS = "cp16.uncertainty.V2-P-B2.equal_fold.WIS"

#: The generation the headline is about, and the one it was compared with, read from the registry
#: when this module loads: no template types "the latest research model" (standard §5).
_CUR = G.current_generation()
_CMP = G.get(_CUR.comparator)
#: The current generation's own adoption rule, when it has one (CP-21 on): its decision level and its conditions.
_RULE = D.ADOPTION_RULES.get(_CUR.id)
_COUNT_WORDS = {4: "four", 5: "five", 6: "six"}
#: The shared comparison's record prefix: the experiment whose rows the overview draws (`cp21.metrics.…`).
_OV = G.comparison_prefix()

BLOCKS: tuple[Block, ...] = (
    # ---- the terms the headline introduces, directly below it (standard §1 ii, §3.1) ----------------
    _block("terms.error_scores", "P21", """**Error scores:** error relative to a simple naive forecast, averaged over the
           test periods; lower is better.""", PAGE, README),
    _block("terms.targets", "P21", """**Accuracy targets:** each error score at least {r:derived.rule.margin.S_MAE}
           below the strongest benchmark's, set on {r:derived.rule.set_on}; a policy's distance from them is a point
           comparison.""", PAGE, README),
    _block("terms.benchmark", "P21", """**{g:daily-lear.name}:** the strongest benchmark, a linear model refitted
           daily.""", PAGE, README),
    _block("terms.policies", "P21", """**Policies:** the models and variants tested.""", PAGE, README),
    _block("terms.not_adopted", "P57", """**Not adopted:** the candidates of the two experiments tested against
           {g:v4.version} and not adopted, drawn as open circles. {g:v3-plus-pooled-normalized-lightgbm-averaged.short}:
           one LightGBM for all {s:hours:24} hours, averaged over four sizes, in place of {g:v4.version}'s three blocks.
           {g:v3-plus-pooled-lightgbm-pair.short}: {g:v4.version}'s normalized and raw-price LightGBM pair, each pooled over
           all {s:hours:24} hours. {g:ddnn-member-on-v4.short}: a distributional neural network with one third of the
           blend's weight. Their cards are under the experiments not adopted between {g:v4.version} and
           {g:v5.version}.""", PAGE),
    # The terms of a headline that leads with a generation's own adoption rule (standard §3.3 a; PUBLISH_RULES A1).
    # They follow the current generation and its rule: its date, its number of conditions and its decision level.
    _block("terms.adoption_rule", _RULE["claim"] if _RULE else "P51", f"""**Adoption rule:** set on
           {{r:derived.adoption.{_CUR.id}.set_on}}; its {_COUNT_WORDS[len(_RULE["conditions"]) if _RULE else 4]}
           conditions are in the {{g:{_CUR.id}.version}} chapter.""", PAGE, README),
    # A rule that decided at another level than 95% names it as the rule's (W47), in the wording P53 quotes; the phone
    # placement (A2) keeps this term to two lines.
    _block("terms.differences", _RULE["claim"] if _RULE else "P51", (f"""**Paired difference:**
           {{g:{_CUR.id}.version}}'s score minus {{g:{_CMP.id}.version}}'s, with the rule's
           {{s:level:{_RULE["level"]}}} interval; below zero favours {{g:{_CUR.id}.version}}.""" if _RULE and
           _RULE["level"] != "95%" else f"""**Paired difference:** {{g:{_CUR.id}.version}}'s error score minus
           {{g:{_CMP.id}.version}}'s on identical hours, with its {{s:level:95%}} confidence interval; below zero favours
           {{g:{_CUR.id}.version}}."""), PAGE, README),
    _block("terms.class", "P21", """**{g:current.badge}:** development evidence, not a test on new data.""",
           PAGE, README),
    # The model the current generation's name introduces (PRES-4, editorial V1): defined beneath the headline, on one
    # line at phone width (A2).
    _block("terms.ddnn2", "C403", """**{s:name:DDNN-2}:** a distributional neural network (DDNN).""",
           PAGE, README),
    # ---- the comparison: the change against the comparator, and the main caveat (§1 i) ---------------
    _block("comparison.finding", "P24", f"""{{g:{_CUR.id}.version}} against {{g:{_CMP.id}.version}}, as a share of
           {{g:{_CMP.id}.version}}'s error scores: {{r:derived.change.{_CUR.id}.S_MAE}}
           {{r:derived.change.{_CUR.id}.S_MAE|ci}} on the point-error score and {{r:derived.change.{_CUR.id}.S_WIS}}
           {{r:derived.change.{_CUR.id}.S_WIS|ci}} on the interval score, with {{s:level:95%}} confidence
           intervals.""", PAGE, README),
    _block("comparison.caveat", "P24", """Development evidence, not a test on new data. Meeting the targets is a
           development diagnostic, not a product qualification.""", PAGE, README),
    # The visible fairness population (plan §7.9, review F04): hours and the days they fall on, together, from the
    # same committed row; the row count comes from the registry's comparison, never typed.
    _block("comparison.sub", "P07", f"""{{s:count:{len(G.comparison_rows())}}} policies · the same
           {{r:{_OV}.metrics.B0.pooled.n_hours}} historical hours over {{r:{_OV}.metrics.B0.pooled.n_days}} days ·
           error scores averaged with equal weight over five test periods · lower is better"""),
    _block("comparison.howto", "C68", f"""Each score divides a model's error by that of a simple similar-day forecast in
           the same period, which therefore scores {{r:{_OV}.metrics.B0.equal_fold.S_MAE|p=3}}. The interval score
           accounts for both interval width and missed outcomes."""),
    _block("comparison.scale", "P26", f"""For scale, as mean absolute error per test period: {{g:{_CUR.id}.version}}
           {{r:derived.periods.{_CUR.id}.ordinary_low}}–{{r:derived.periods.{_CUR.id}.ordinary_high}} EUR/MWh in the
           four ordinary periods and {{r:derived.periods.{_CUR.id}.stress}} EUR/MWh in the {{s:date:2022}} crisis
           period; the similar-day naive {{r:derived.periods.naive.ordinary_low}}–{{r:derived.periods.naive.ordinary_high}}
           and {{r:derived.periods.naive.stress}} EUR/MWh. Context only: the error scores rank the policies.""",
           PAGE, README),
    _block("overview.v1pointer", "P09", """{g:v1.version} appears here as its development replay; its one-shot holdout
           results are in the {g:v1.version} chapter."""),
    _block("overview.fairness", "P08", """The five test periods are called folds below; the third covers the {s:date:2022}
           price crisis. These are development results, not a test on new data."""),
    # README-only: the page states the shared hours in the comparison's subtitle.
    _block("overview.fairness.readme", "P08", """Every policy was evaluated on the same {r:cp20.metrics.B0.pooled.n_hours}
           historical hours over {r:cp20.metrics.B0.pooled.n_days} days, across five test periods (folds); the third
           covers the {s:date:2022} price crisis. Scores are
           normalized within each period, then averaged with equal weight. These are development results, not a test on
           new data.""", README),
    _block("overview.fairness.detail", "P08", """Paired intervals come from a moving-block bootstrap with seed
           {r:cp20.protocol.bootstrap_seed}, {r:cp20.protocol.replicates} replicates and
           {r:cp20.protocol.block_days}-day blocks, resampled identically for every policy. They are confidence
           intervals of an estimated difference, not forecast intervals. A change's interval as a share of the
           comparator's score is, from {g:v4.version} on, the percentile interval of the ratio of the two scores over
           the same draws; for {g:v2.version} and {g:v3.version}, it is the difference's interval divided by the
           comparator's score."""),
    _block("overview.f07", "P09", """**Why {s:version:v1} scores {r:cp20.metrics.B1.equal_fold.S_MAE|p=3} in the shared
           development comparison but {r:cp2.dm_development.point_vs_naive.relative_improvement_pct|abs}% worse in its own
           report.** Both figures describe the same {r:cp2.dm_development.point_vs_naive.n_days} development days.
           {s:version:v1}'s report compared its daily absolute error with the raw similar-day naive (MAE
           {r:cp2.development_pooled_metrics.similar_day_naive.point.MAE} EUR/MWh) and pooled all days, so the
           {s:date:2022} crisis dominates. This comparison gives each of the five folds equal weight, and its naive
           reference is the forecast's emitted median after the common residual layer (MAE
           {r:cp20.metrics.B0.pooled.MAE} EUR/MWh). {s:version:v1}'s own error is {r:cp20.metrics.B1.pooled.MAE} EUR/MWh
           in both. {s:version:v1}'s original nine-quantile pinball is a different score from the seven-quantile interval
           score used here.""", PAGE, README),
    _block("overview.definitions", "C68", """**The point-error score (S_MAE)** is the mean over the five historical test
           periods (folds) of a policy's MAE divided by the similar-day naive's MAE in the same period; **the interval
           score (S_WIS)** does the same for the weighted interval score, which rewards narrow intervals and penalizes
           missed outcomes. MAE uses the emitted median."""),
    # ---- the v4 chapter (the chapter grammar, standard §6; publication packet §5, CP-21 claim map) ----------
    _block("v4.question", "C104", """Does adding a nonlinear forecaster, LightGBM, to {g:v3.version} improve both of
           its error scores?"""),
    _block("v4.change", "C104", """{g:v4.version} adds a LightGBM forecaster, a nonlinear model built from decision
           trees, with the same information as {g:v3.version}, to its blend of two linear forecasts, and re-estimates the
           hour-aware intervals on the new forecast's own errors."""),
    _block("v4.chart_headline", "C107", """Against {g:v3.version}, the point-error score changed by
           {r:derived.change.v4.S_MAE} {r:derived.change.v4.S_MAE|ci} and the interval score by
           {r:derived.change.v4.S_WIS} {r:derived.change.v4.S_WIS|ci}, each as a share of the comparator's score,
           with {s:level:95%} confidence intervals."""),
    _block("v4.reading", "C109", """All four conditions of the rule set in advance held: both paired differences'
           {s:level:95%} confidence intervals lie below zero; {g:v4.version} passed the six screening criteria set before
           the experiments (the two accuracy targets, interval coverage in each test period, the August {s:date:2022}
           peak against the benchmarks, each test period against the daily benchmarks, and complete forecasts); the
           evaluation was complete and valid; and no test period's paired interval lies wholly above zero.""",
           PAGE, README),
    # README-only: the page shows these differences once, in the main chart and its table.
    _block("v4.outcome.head", "C107", """Difference in error score ({s:version:v4} − {s:version:v3}), with {s:level:95%}
           confidence intervals. Below zero favours {s:version:v4}.""", README),
    _block("v4.outcome.mae", "C107", """Point-error score: {r:cp21.uncertainty.HGL-HG.equal_fold.MAE}
           {r:cp21.uncertainty.HGL-HG.equal_fold.MAE|ci}""", README),
    _block("v4.outcome.wis", "C108", """Interval score: {r:cp21.uncertainty.HGL-HG.equal_fold.WIS}
           {r:cp21.uncertainty.HGL-HG.equal_fold.WIS|ci}""", README),
    _block("v4.caveat.class", "C125", """**Performance on new data.** This is development evidence after selection, not a
           test on new data.""", PAGE, README),
    _block("v4.caveat.blocks", "C111", """**That separate hour-block models help.** The three-block against the pooled
           LightGBM showed no demonstrated joint preference: point-error score
           {r:cp21.uncertainty.L-R-L-P.equal_fold.MAE} {r:cp21.uncertainty.L-R-L-P.equal_fold.MAE|ci}, interval score
           {r:cp21.uncertainty.L-R-L-P.equal_fold.WIS} {r:cp21.uncertainty.L-R-L-P.equal_fold.WIS|ci}.""",
           PAGE, README),
    _block("v4.caveat.peak", "C120", """**A gain in the August {s:date:2022} peak.** Over those
           {r:cp21.diagnostics.HGL.peak.n_days} days its point error is higher than {g:v3.version}'s: MAE
           {r:cp21.diagnostics.HGL.peak.MAE|p=1} against {r:cp21.diagnostics.HG.peak.MAE|p=1} EUR/MWh. Descriptive, on a
           small sample; no inference is drawn.""", PAGE, README),
    _block("v4.decision", "C127", """{g:v4.status} It met the adoption rule set before the experiment against
           {g:v3.version}; the released model, {g:released.version}, did not change. The three study arms were built
           only to separate the change into steps, and were never candidates for adoption.""", PAGE, README),
    _block("v4.method.blend", "C104", """**The blend.** {g:v4.version}'s central forecast gives {g:v3.version}'s two LEAR
           forecasts, bit for bit, two thirds of the weight, and the mean of the raw and normalized three-block
           LightGBM forecasts one third. The weights were fixed before any result and never tuned. The hour-aware
           intervals keep {g:v3.version}'s recipe, re-estimated on {g:v4.version}'s own errors."""),
    _block("v4.method.inputs", "C105", """**The same information as {g:v3.version}.** Every LightGBM arm uses the model
           comparison study's LightGBM inputs plus the three weather forecasts and their missing-data indicators. The
           blocks are the night, the solar hours and the shoulder and peak hours, in local time. At every forecast
           origin each block picks one of {s:count:4} model sizes on the training window's last
           {s:days:28} days, by MAE, then refits."""),
    _block("v4.method.ladder", "C106", """**The ladder.** Three study arms separate the change into steps, one at a time. Daily
           LightGBM to the pooled LightGBM adds weather, bundled with the training-only choice of model size and the
           missing-input rule, so that step is not a weather effect on its own. The pooled to the three-block
           LightGBM adds the block split alone: the same rows, target, inputs, size grid, selection rule and interval
           recipe. {g:v4.version} then blends the block models into {g:v3.version}."""),
    _block("v4.method.split", "C111", """**The block split** (three-block minus pooled LightGBM): point-error score
           {r:cp21.uncertainty.L-R-L-P.equal_fold.MAE} {r:cp21.uncertainty.L-R-L-P.equal_fold.MAE|ci}, interval score
           {r:cp21.uncertainty.L-R-L-P.equal_fold.WIS} {r:cp21.uncertainty.L-R-L-P.equal_fold.WIS|ci}: no demonstrated
           joint preference. It was tested on the raw price only, with one seed."""),
    _block("v4.method.arms", "C115", """No study arm on its own shows a demonstrated joint preference over
           {g:v3.version}: each arm's intervals against {g:v3.version} span zero, except the three-block LightGBM's interval score, whose interval lies
           wholly above zero, worse: {r:cp21.uncertainty.L-R-HG.equal_fold.WIS}
           {r:cp21.uncertainty.L-R-HG.equal_fold.WIS|ci}."""),
    _block("v4.folds", "C110", """Every test period's point estimate favours {g:v4.version}, and no period's paired
           interval lies wholly above zero. In fold {s:fold:3}, the {s:date:2022} crisis, both intervals cross zero: point error
           {r:cp21.uncertainty.HGL-HG.fold_3.MAE|p=1} {r:cp21.uncertainty.HGL-HG.fold_3.MAE|ci1} EUR/MWh."""),
    _block("v4.absolute", "C118", """Mean absolute error per test period: {g:v4.version}
           {r:derived.periods.v4.ordinary_low}–{r:derived.periods.v4.ordinary_high} EUR/MWh in the four ordinary
           periods and {r:derived.periods.v4.stress} EUR/MWh in the {s:date:2022} crisis period, where
           {g:v3.version} had {r:derived.periods.v3.stress}."""),
    _block("v4.peak", "C120", """**The August {s:date:2022} peak**, {r:cp21.diagnostics.HGL.peak.n_days} days inside
           the crisis period: {g:v4.version}'s point error is higher than {g:v3.version}'s, MAE
           {r:cp21.diagnostics.HGL.peak.MAE|p=1} against {r:cp21.diagnostics.HG.peak.MAE|p=1} EUR/MWh, and so is its
           interval score, {r:cp21.diagnostics.HGL.peak.WIS|p=1} against {r:cp21.diagnostics.HG.peak.WIS|p=1} EUR/MWh;
           {r:cp21.diagnostics.HGL.peak.hit_count95} against {r:cp21.diagnostics.HG.peak.hit_count95} of its
           {r:cp21.diagnostics.HGL.peak.n_hours} hours fell inside the {s:level:95%} interval. Descriptive, on a small
           sample: no inference is drawn."""),
    _block("v4.coverage", "C121", """{g:v4.version}'s pooled intervals are narrower than {g:v3.version}'s at every level. Its
           pooled coverage is {r:cp21.metrics.HGL.pooled.coverage50}, {r:cp21.metrics.HGL.pooled.coverage80} and
           {r:cp21.metrics.HGL.pooled.coverage95} at the {s:level:50%}, {s:level:80%} and {s:level:95%} levels, against
           {g:v3.version}'s {r:cp21.metrics.HG.pooled.coverage50}, {r:cp21.metrics.HG.pooled.coverage80} and
           {r:cp21.metrics.HG.pooled.coverage95}. No hour or block effect is claimed."""),
    _block("v4.rule", "C101", """**The rule**, set on {r:derived.adoption.v4.set_on} and frozen in the protocol before
           any fit: {g:v4.version} would be adopted only if, against {g:v3.version}, the interval score's paired
           difference had its upper {s:level:95%} confidence limit below zero and the point-error score's at or below
           zero; it met the six original screening criteria; the evaluation was complete and valid; and no test
           period was decisively worse, that is, none had a paired interval wholly above zero. All four held.""", PAGE,
           README),
    _block("v4.criteria", "C119", """As a diagnostic, {g:v4.version} meets all six original screening criteria, as
           {g:v3.version} does. Among the study arms, the normalized three-block LightGBM meets them; the pooled
           LightGBM misses criterion {s:criterion:4} (the August {s:date:2022} peak against the benchmarks), and the
           three-block LightGBM misses criteria {s:criterion:4} and {s:criterion:5} (each test period against the daily
           benchmarks). This is a development diagnostic, not a product qualification."""),
    _block("v4.parity", "C122", """Run through this experiment's code, {g:v3.version} reproduced its accepted forecasts
           bit for bit on all {r:cp21.hg_parity.rows} hours, so the comparison is with {g:v3.version} itself."""),
    _block("v4.controls", "C123", """All {r:cp21.controls.checks} frozen causal and integrity controls passed: masking the
           delivery day or future weather changed the forecast by exactly zero, a day-before price change moved every
           arm, and the pooled and block arms saw the same rows."""),
    _block("v4.cost", "C124", """{r:cp21.fit_cost.fits} LightGBM fits in the main run. The time of one daily cycle was
           measured as a diagnostic, not as a qualification for daily operation."""),
    # ---- the v5 chapter (the chapter grammar, standard §6; publication packet §5, CP-24 claim map) ----------
    _block("v5.question", "C403", """Does a distributional neural network that forecasts the whole next day at once,
           added to {g:v4.version} with a fixed one-sixth weight, improve both of its error scores?"""),
    _block("v5.change", "C403", """{g:v5.version} adds {s:name:DDNN-2}, a distributional neural network: it forecasts the {s:hours:24} hours
           of the next day together, each as a full probability distribution, from the same information as
           {g:v4.version}. It takes half of the third that {g:v4.version}'s LightGBM held; the two LEAR forecasts keep two
           thirds, and the hour-aware intervals are re-estimated on the new forecast's own errors."""),
    _block("v5.chart_headline", "C420", """Against {g:v4.version}, the point-error score changed by
           {r:derived.change.v5.S_MAE} {r:derived.change.v5.S_MAE|ci} and the interval score by
           {r:derived.change.v5.S_WIS} {r:derived.change.v5.S_WIS|ci}, each as a share of the comparator's score, with
           {s:level:95%} confidence intervals; at the rule's {s:level:97.5%} decision level the intervals are
           {r:derived.change_decision.v5.S_MAE|ci} and {r:derived.change_decision.v5.S_WIS|ci}."""),
    _block("v5.reading", "C421", """All five conditions of the rule set in advance held: at the rule's {s:level:97.5%}
           level both paired differences' intervals lie below zero; {g:v5.version} passed the six screening criteria set
           before the experiments (the two accuracy targets, interval coverage in each test period, the August
           {s:date:2022} peak against the benchmarks, each test period against the daily benchmarks, and complete
           forecasts); the evaluation was complete and valid; no test period's paired interval lies wholly above zero; and
           both scores improved on {g:v4.version}'s by more than the practical size set in advance, half a percent.""",
           PAGE, README),
    # README-only: the page shows these differences once, in the main chart and its table.
    _block("v5.outcome.head", "C420", """Difference in error score ({s:version:v5} − {s:version:v4}), with its
           {s:level:95%} confidence interval and its interval at the rule's {s:level:97.5%} decision level. Below zero
           favours {s:version:v5}.""", README),
    _block("v5.outcome.mae", "C420", """Point-error score: {r:cp24.uncertainty.v5-HGL.equal_fold.MAE}
           {r:cp24.uncertainty.v5-HGL.equal_fold.MAE|ci} at {s:level:95%};
           {r:cp24.uncertainty97_5.v5-HGL.equal_fold.MAE|ci} at {s:level:97.5%}""", README),
    _block("v5.outcome.wis", "C420", """Interval score: {r:cp24.uncertainty.v5-HGL.equal_fold.WIS}
           {r:cp24.uncertainty.v5-HGL.equal_fold.WIS|ci} at {s:level:95%};
           {r:cp24.uncertainty97_5.v5-HGL.equal_fold.WIS|ci} at {s:level:97.5%}""", README),
    _block("v5.caveat.coverage", "P54", """**That the gain holds across the year.** The comparison scores
           {r:derived.coverage.days} of the {r:derived.coverage.data_days} days of data up to the end of the development
           period, {r:derived.coverage.first_data_day} to {r:derived.coverage.data_end}; {r:derived.coverage.q4_days} of
           them fall in October to December, and only {r:derived.coverage.winters} winter (December to February) contains
           any of them.""", PAGE, README),
    _block("v5.caveat.class", "C449", """**Performance on new data.** This is development evidence after selection,
           and the second decision on a distributional neural network over the same five test periods, after the
           {g:ddnn-member-on-v4.inline}; only a one-shot test on days never used for development or selection protects
           against that reuse.""", PAGE, README),
    _block("v5.caveat.alone", "C427", """**That {s:name:DDNN-2} is best used as {g:v5.version} uses it.**
           {g:ddnn2-alone.name} had lower, that is better, error scores than {g:v4.version}:
           {r:derived.change.ddnn2-alone.S_MAE} {r:derived.change.ddnn2-alone.S_MAE|ci} on the point-error score and
           {r:derived.change.ddnn2-alone.S_WIS} {r:derived.change.ddnn2-alone.S_WIS|ci} on the interval score, as shares
           of {g:v4.version}'s score, with {s:level:95%} confidence intervals; {g:v3-plus-ddnn2.inline} also scored better
           than {g:v5.version}. Descriptive only: both were study arms, never eligible for adoption, and no rule tested
           them.""", PAGE, README),
    _block("v5.decision", "C422", """{g:v5.status} It met the adoption rule set before the experiment against
           {g:v4.version}; the released model, {g:released.version}, did not change. The two study arms,
           {g:ddnn2-alone.name} and {g:v3-plus-ddnn2.inline}, were built only to show where the change came from, and
           were never candidates for adoption.""", PAGE, README),
    _block("v5.method.blend", "C403", """**The blend.** {g:v5.version}'s central forecast gives {g:v4.version}'s two LEAR
           forecasts, bit for bit, two thirds of the weight, {g:v4.version}'s LightGBM one sixth and {s:name:DDNN-2} one sixth.
           The weights were fixed before any fit and never tuned. The hour-aware intervals keep {g:v4.version}'s recipe,
           re-estimated on {g:v5.version}'s own errors."""),
    _block("v5.method.network", "C403", """**{s:name:DDNN-2}.** A feed-forward network written in NumPy, on one row per delivery
           day, with a Johnson SU distribution for each of the {s:hours:24} hours, and exactly {g:v4.version}'s information. Each
           member trains on up to two years before the forecast day, from {s:date:2019-01-01} at the earliest, minus a
           random fifth of whole weeks held out to stop its training; four configurations, chosen per test period by a search on training data only, times two seeds,
           are combined by the median of their forecasts."""),
    _block("v5.method.arms", "C427", """**The study arms.** {g:ddnn2-alone.name}, against {g:v4.version}:
           {r:derived.change.ddnn2-alone.S_MAE} {r:derived.change.ddnn2-alone.S_MAE|ci} on the point-error score and
           {r:derived.change.ddnn2-alone.S_WIS} {r:derived.change.ddnn2-alone.S_WIS|ci} on the interval score, as shares
           of {g:v4.version}'s score: descriptive, never eligible."""),
    _block("v5.method.arms_lightgbm", "C432", """{g:v3-plus-ddnn2.name}, {s:name:DDNN-2} in the LightGBM's place, scored
           better than {g:v5.version} itself: {g:v5.version} minus it is
           {r:cp24.uncertainty.v5-v3+D2.equal_fold.MAE} {r:cp24.uncertainty.v5-v3+D2.equal_fold.MAE|ci} on the
           point-error score and {r:cp24.uncertainty.v5-v3+D2.equal_fold.WIS}
           {r:cp24.uncertainty.v5-v3+D2.equal_fold.WIS|ci} on the interval score, both intervals wholly above zero: with
           {s:name:DDNN-2} present, adding the LightGBM member made both scores worse. It too was a study arm, never
           eligible, so the rule's candidate was {g:v5.version}."""),
    _block("v5.folds", "C421", """Every test period favours {g:v5.version}: in each, the paired {s:level:95%} intervals
           of both errors lie below zero. The largest change is in fold {s:fold:3}, the {s:date:2022} crisis: point error
           {r:cp24.uncertainty.v5-HGL.fold_3.MAE|p=1} {r:cp24.uncertainty.v5-HGL.fold_3.MAE|ci1} EUR/MWh."""),
    _block("v5.absolute", "C435", """Mean absolute error per test period: {g:v5.version}
           {r:derived.periods.v5.ordinary_low}–{r:derived.periods.v5.ordinary_high} EUR/MWh in the four ordinary
           periods and {r:derived.periods.v5.stress} EUR/MWh in the {s:date:2022} crisis period, where {g:v4.version} had
           {r:derived.periods.v4.stress}."""),
    _block("v5.peak", "P59", """**The August {s:date:2022} peak**, {r:cp24.diagnostics.v5.peak.n_days} days inside the
           crisis period: {g:v5.version}'s point error is {r:cp24.diagnostics.v5.peak.MAE|p=1} EUR/MWh against
           {g:v4.version}'s {r:cp24.diagnostics.HGL.peak.MAE|p=1}. Descriptive, on a small sample: no inference is
           drawn."""),
    _block("v5.coverage", "P59", """{g:v5.version}'s pooled intervals are narrower than {g:v4.version}'s at every level:
           mean widths {r:cp24.metrics.v5.pooled.mean_width50|p=1}, {r:cp24.metrics.v5.pooled.mean_width80|p=1} and
           {r:cp24.metrics.v5.pooled.mean_width95|p=1} EUR/MWh at the {s:level:50%}, {s:level:80%} and {s:level:95%} levels,
           against {r:cp24.metrics.HGL.pooled.mean_width50|p=1}, {r:cp24.metrics.HGL.pooled.mean_width80|p=1} and
           {r:cp24.metrics.HGL.pooled.mean_width95|p=1}. Its pooled coverage is {r:cp24.metrics.v5.pooled.coverage50},
           {r:cp24.metrics.v5.pooled.coverage80} and {r:cp24.metrics.v5.pooled.coverage95}, against
           {g:v4.version}'s {r:cp24.metrics.HGL.pooled.coverage50}, {r:cp24.metrics.HGL.pooled.coverage80} and
           {r:cp24.metrics.HGL.pooled.coverage95}."""),
    _block("v5.rule", "C401", """**The rule**, set on {r:derived.adoption.v5.set_on} and frozen in the protocol before any
           of its fits: {g:v5.version} would be adopted only if, against {g:v4.version}, the paired differences' intervals
           at {s:level:97.5%} lay below zero for the interval score and at or below zero for the point-error score; it met
           the six original screening criteria; the evaluation was complete and valid; no test period was decisively
           worse, that is, none had a {s:level:95%} paired interval wholly above zero; and both point estimates improved
           on {g:v4.version}'s score by at least half a percent. The {s:level:97.5%} level splits a {s:level:5%} error rate
           across the two attempts the plan allowed; one ran. All five conditions held.""", PAGE, README),
    _block("v5.criteria", "C434", """As a diagnostic, {g:v5.version} meets all six original screening criteria, as
           {g:v4.version}, {g:ddnn2-alone.name} and {g:v3-plus-ddnn2.inline} do. This is a development diagnostic, not a
           product qualification."""),
    _block("v5.controls", "C439", """All {r:cp24.controls.checks} frozen causal and integrity controls passed, each
           negative check paired with a positive one: masking the delivery day and future inputs changed the forecast by
           exactly zero, and a day-before price change, weather permutations and held-out weeks moved {s:name:DDNN-2}."""),
    _block("v5.leakage", "C441", """**No outcome from the delivery day on is used, at any origin.** With every such
           outcome destroyed, the frozen network refitted at all {r:cp24.leakage.origins} warm-up and evaluation origins
           reproduced the committed forecasts bit for bit at {r:cp24.leakage.bitwise} of them, over
           {r:cp24.leakage.keys} hours; a change to the previous day's prices moved them, and a planted one-day leak was
           caught in every test period. The input vintages are inherited from the earlier experiments, not re-tested
           here."""),
    # ---- the v3 chapter (the chapter grammar, standard §6) ------------------------------------------
    _block("v3.question", "P29", """Do weather forecasts available before the auction improve {g:v2.version}?"""),
    _block("v3.change.short", "C65", """{g:v3.version} adds weather forecasts available before the auction to
           {g:v2.version}'s inputs; the modelling setup is otherwise unchanged."""),
    _block("v3.chart_headline", "C71", """Weather inputs lowered both error scores against {g:v2.version}."""),
    _block("v3.reading", "P30", """Both confidence intervals lie below zero and every test period favours
           {g:v3.version}; its prediction intervals are narrower in every period, with pooled {s:level:95%} coverage
           slightly lower, {r:cp20.metrics.HG.pooled.coverage95} against {r:cp20.metrics.H0.pooled.coverage95}.""",
           PAGE, README),
    # README-only: the page shows these differences once, in the main chart and its table.
    _block("v3.outcome.head", "C69", """Difference in error score ({s:version:v3} − {s:version:v2}), with {s:level:95%}
           confidence intervals. Below zero favours {s:version:v3}.""", README),
    _block("v3.outcome.mae", "C69", """Point-error score: {r:%s} {r:%s|ci}""" % (_HG_MAE, _HG_MAE), README),
    _block("v3.outcome.wis", "C70", """Interval score: {r:%s} {r:%s|ci}""" % (_HG_WIS, _HG_WIS), README),
    _block("v3.caveat.bundle", "C83", """**Which weather feature drives the gain.** The three were evaluated together, so
           no single feature's contribution is isolated.""", PAGE, README),
    _block("v3.caveat.fold3", "C73", """**A point-error gain within the {s:date:2022} crisis period on its own.** There
           its confidence interval crosses zero: {r:cp20.uncertainty.HG-H0.fold_3.MAE|p=1}
           {r:cp20.uncertainty.HG-H0.fold_3.MAE|ci1} EUR/MWh.""", PAGE, README),
    _block("v3.caveat.class", "C84", """**Performance on new data.** This is development evidence, not a test on new
           data.""", PAGE, README),
    _block("v3.decision", "C95", """{g:v3.status} It met the joint improvement rule against {g:v2.version} and both
           accuracy targets.""", PAGE, README),
    _block("v3.criteria", "C78", """As a diagnostic, {s:version:v3} is the first policy evaluated against the original
           criteria to meet all six; {s:version:v2} misses criteria {s:criterion:1} and {s:criterion:2}. This is a
           development diagnostic, not a product qualification."""),
    _block("v3.helps", "C79", """**During the {s:date:2022} price peak**, a {r:cp20.diagnostics.HG.peak.n_days}-day
           window inside the crisis period, MAE went from {r:cp20.criteria.H0.c4.peak.MAE|p=1} to
           {r:cp20.criteria.HG.c4.peak.MAE|p=1} EUR/MWh. Descriptive only; the hours inside the interval are in the
           chart."""),
    _block("v3.hurts", "C80", """The prediction intervals are narrower in every period, while pooled {s:level:95%}
           coverage is slightly lower, {r:cp20.metrics.HG.pooled.coverage95} against
           {r:cp20.metrics.H0.pooled.coverage95} for {s:version:v2}."""),
    _block("v3.recipe", "C66", """Operational NCEP GFS {s:grid:0.25°}, the {s:time:00 UTC} run of the
           day before delivery. Wind components are interpolated to each delivery hour; wind speed is
           computed per grid cell before a cos(latitude)-weighted average over the fixed box
           {s:coordinates:47–55.25°N, 5.5–15.5°E}; radiation is de-averaged from the forecast's running
           means. Each feature has a missing indicator. A regional weather proxy: not the DE-LU zone's exact
           outline and not a generation forecast."""),
    _block("v3.missing", "C67", """Only delivery {s:date:2019-01-01} has no weather, structurally: its
           run would be {s:date:2018-12-31}, before the input floor. Nothing was imputed from a failed
           retrieval."""),
    _block("v3.availability", "C85", """Availability of each run before the {s:time:D−1 11:00 UTC}
           forecast origin is reconstructed from dated NCEP production records with an assumed
           dissemination lag. It is not a per-day delivery guarantee."""),
    _block("v3.controls", "C89", """Frozen causal controls ran on {s:date:2020-07-01} and
           {s:date:2025-05-01}: masking the delivery day or the future changed the forecast by exactly
           zero, an available day-before price change moved it, and future weather changed
           nothing."""),
    _block("v3.controls.supplement", "C90", """One frozen check could not fail: multiplying all
           weather by a constant is undone by the training-only scaler. It was kept as an invariance
           check, and two controls that can fail were added, a shift of the target day's weather and a
           permutation of the training weather. Both passed (repair {s:repair:r13})."""),
    _block("v3.review", "C91", """The first independent Integration review within this project's
           process failed: one frozen input's recorded hash described its bytes with Windows line
           endings, so the reproduction commands refused to run in a clean checkout. The bytes were
           restored to the recorded identity (repair {s:repair:r14}) and a fresh Integration review
           passed."""),
    _block("v3.cost", "C92", """{r:cp20.extraction.complete_runs} GFS runs and
           {r:cp20.extraction.target_messages} decoded messages; {r:cp20.resources.transfer_gib} GiB
           transferred; {r:cp20.resources.machine_hours} machine-hours; external cost
           ${r:cp20.resources.external_cost_usd}."""),
    _block("v3.dependency", "C93", """{s:version:v3} needs one GFS retrieval per delivery day."""),
    # ---- the v2 chapter ------------------------------------------------------------------------------
    _block("v2.question", "P29", """Could a different forecasting model repair {g:v1.version}'s failure in the
           {s:date:2022} crisis?"""),
    _block("v2.problem", "C53", """{s:version:v1} substantially underestimated prices during the
           {s:date:2022} crisis. Recalibrating its intervals improved coverage, but the improvement was
           insufficient."""),
    _block("v2.change", "C10", """{g:v2.version} combines two LEAR forecasts (LEAR: a regularized linear model fitted for
           each hour) with intervals that account for the hour of the day. A control uses the same blend with pooled
           intervals, so the interval methods can be compared."""),
    _block("v2.chart_headline", "C37", """{g:v2.version} lowered both error scores against {g:daily-lear.inline}."""),
    _block("v2.reading", "C34", """Against {g:daily-lear.inline} both confidence intervals lie below zero; against its
           pooled-interval control only the interval score improved, and the point-error interval ends just above
           zero, at {r:%s|hi}: not equivalence.""" % _HP_MAE),
    _block("v2.caveat.attribution", "C41", """**That hour-aware intervals caused the gain over
           {g:daily-lear.inline}.** The pooled-interval control also passes the per-period criterion.""", PAGE, README),
    _block("v2.caveat.split", "C41", """**How the gain splits between the blend and the interval layer.** The experiment
           does not separate them.""", PAGE, README),
    _block("v2.caveat.class", "P31", """**Performance on new data.** This is development evidence, not a test on new
           data.""", PAGE, README),
    _block("v2.decision", "P32", """{g:v2.status} {g:v3.version} was built on it.""", PAGE, README),
    # README-only: the page states these in its reading and carries them in the chart and table.
    _block("v2.result.hb2", "C37", """Against {g:daily-lear.inline} ({s:version:v2} − {g:daily-lear.inline}),
           {s:version:v2} meets the joint improvement rule as exploratory evidence: point-error score difference
           {r:%s} {r:%s|ci}, interval-score difference {r:%s} {r:%s|ci}.""" % (_HB_MAE, _HB_MAE, _HB_WIS, _HB_WIS),
           README),
    _block("v2.result.hp", "C34", """Against its pooled-interval control ({s:version:v2} − control), there is
           **no demonstrated joint preference**: the interval-score difference is {r:%s} {r:%s|ci}, but the
           point-error difference's interval ends just above zero, {r:%s} [{r:%s|lo}, {r:%s|hi}]. This is
           not equivalence.""" % (_HP_WIS, _HP_WIS, _HP_MAE, _HP_MAE, _HP_MAE), README),
    _block("v2.result.pb2", "C38", """The pooled-interval control against {g:daily-lear.inline} does not meet the rule:
           its interval-score interval, {r:%s|ci}, spans zero.""" % _PB_WIS),
    _block("v2.result.criteria", "C39", """Both {s:version:v2} arms miss criteria {s:criterion:1–2}, the
           point-error and interval-score thresholds, and meet criteria {s:criterion:3–6}."""),
    # ---- the branch cards (standard §6) ------------------------------------------------------------
    _block("branch.calibration.question", "P27", """Could recalibrating {g:v1.version}, without refitting it, repair its
           coverage in the {s:date:2022} crisis?""", PAGE, README),
    _block("branch.calibration.result", "P12", """Against {g:v1.version}, on the matched crisis window: the hours inside
           the {s:level:95%} interval rose from {r:cp10.peak_windows.v1_reference.covered_95} to
           {r:cp10.peak_windows.c1_price_volatility.covered_95} of {r:cp10.peak_windows.c1_price_volatility.n_obs}.""",
           PAGE, README),
    _block("branch.calibration.reason", "P27", """Recalibration alone was not enough: the model itself had to adapt.""",
           PAGE, README),
    _block("branch.model-comparison.question", "P27", """Which forecasting approach copes best with the {s:date:2022}
           crisis?""", PAGE, README),
    _block("branch.model-comparison.result", "C21", """Nine policies on the same hours, none meeting the product
           criteria. {g:normalized-lear.name} cut crisis-window MAE to {r:cp15.peak.A1.MAE|p=1} EUR/MWh, but
           {g:daily-lear.inline} kept the better error scores: {r:cp15.relative_scores.B2.S_MAE|p=3} and
           {r:cp15.relative_scores.B2.S_WIS|p=3} against {r:cp15.relative_scores.A1.S_MAE|p=3} and
           {r:cp15.relative_scores.A1.S_WIS|p=3}.""", PAGE, README),
    _block("branch.model-comparison.reason", "P27", """No policy met the criteria; the study informed {g:v2.version}'s
           blend of two LEAR forecasts.""", PAGE, README),
    # ---- the branch cards after v4 (PRES-4; runbook §3): CP-22's and CP-23's, tested against v4 and not adopted ----
    _block("branch.pooled-lightgbm-member-for-v4.question", "P56", """Can one pooled LightGBM, for all {s:hours:24} hours, replace
           {g:v4.version}'s three-block LightGBM member without losing accuracy against {g:v4.version} itself?""",
           PAGE, README),
    _block("branch.pooled-lightgbm-member-for-v4.result", "C208", """Against {g:v4.version}, over the five test periods
           together, neither candidate's {s:level:95%} interval lay wholly above zero: the averaged member's paired
           difference is {r:cp22.uncertainty.R-HGL.equal_fold.MAE} {r:cp22.uncertainty.R-HGL.equal_fold.MAE|ci} in the
           point-error score and {r:cp22.uncertainty.R-HGL.equal_fold.WIS} {r:cp22.uncertainty.R-HGL.equal_fold.WIS|ci}
           in the interval score; the pair's {r:cp22.uncertainty.M-HGL.equal_fold.MAE}
           {r:cp22.uncertainty.M-HGL.equal_fold.MAE|ci} and {r:cp22.uncertainty.M-HGL.equal_fold.WIS}
           {r:cp22.uncertainty.M-HGL.equal_fold.WIS|ci}. In fold {s:fold:4}, both were decisively worse: point error
           {r:cp22.uncertainty.R-HGL.fold_4.MAE|p=1} {r:cp22.uncertainty.R-HGL.fold_4.MAE|ci1} and
           {r:cp22.uncertainty.M-HGL.fold_4.MAE|p=1} {r:cp22.uncertainty.M-HGL.fold_4.MAE|ci1} EUR/MWh, interval score
           {r:cp22.uncertainty.R-HGL.fold_4.WIS|p=1} {r:cp22.uncertainty.R-HGL.fold_4.WIS|ci1} and
           {r:cp22.uncertainty.M-HGL.fold_4.WIS|p=1} {r:cp22.uncertainty.M-HGL.fold_4.WIS|ci1} EUR/MWh. Development
           evidence after selection, not a test on new data.""", PAGE, README),
    _block("branch.pooled-lightgbm-member-for-v4.reason", "C212", """Both candidates were decisively worse than
           {g:v4.version} in one test period, with a {s:level:95%} interval wholly above zero; the rule allowed none, so
           {g:v4.version} was kept as it was.""", PAGE, README),
    _block("branch.ddnn-member-on-v4.question", "P56", """Does a distributional neural network, added to
           {g:v4.version} as a one-third member, improve on {g:v4.version} in point and interval accuracy together?""",
           PAGE, README),
    _block("branch.ddnn-member-on-v4.result", "C308", """Against {g:v4.version}, the paired difference is
           {r:cp23.uncertainty.v5-HGL.equal_fold.MAE} {r:cp23.uncertainty.v5-HGL.equal_fold.MAE|ci} in the point-error
           score and {r:cp23.uncertainty.v5-HGL.equal_fold.WIS} {r:cp23.uncertainty.v5-HGL.equal_fold.WIS|ci} in the
           interval score, with {s:level:95%} intervals; in fold {s:fold:3}, the {s:date:2022} crisis, its point error was
           decisively worse, {r:cp23.uncertainty.v5-HGL.fold_3.MAE|p=1} {r:cp23.uncertainty.v5-HGL.fold_3.MAE|ci1}
           EUR/MWh. Development evidence after selection, not a test on new data.""", PAGE, README),
    _block("branch.ddnn-member-on-v4.reason", "C310", """The improvement over {g:v4.version} that the rule required was
           not shown, on either score.""", PAGE, README),
    # ---- the v1 chapter (standard §4; its archive is preserved as published) ------------------------
    _block("v1.what", "P11", """A LightGBM ensemble forecasting nine quantiles for every hour, calibrated
           into {s:level:50 / 80 / 95%} prediction intervals, and shipped exactly as it was evaluated.""", PAGE, README),
    _block("v1.holdout", "P11", """On its pre-specified {v1:holdout_days}-day holdout, {s:version:v1} beat the
           similar-day naive: MAE {v1:holdout_mae_champion|p=1} against {v1:holdout_mae_naive|p=1} EUR/MWh, and mean
           pinball loss {v1:holdout_pinball_champion|p=1} against {v1:holdout_pinball_naive|p=1} EUR/MWh. A one-sided
           Diebold–Mariano test on the daily pinball-loss vectors gave {v1:holdout_dm_p_value|pfloor}; it tests the
           probabilistic forecast, not the MAE difference.""", PAGE, README),
    _block("v1.unflattering", "P11", """**Where it falls short.** On the development folds its point
           error is {v1:development_dm_point_relative} than the naive's, pooled over all days (a one-sided test
           for an improvement gives p = {v1:development_dm_point_p_value}: no evidence of an advantage). Over the
           August {s:date:2022} peak weeks its {s:level:95%} interval covered {r:cp15.peak.B1.coverage95|p=3}
           of outcomes.""", PAGE, README),
    _block("v1.lesson", "P11", """**Why:** its crisis errors were about price level, not shape. On the
           crisis window the bias was {r:cp15.peak.B1.bias|p=1} EUR/MWh, and the daily mean-level error was
           {r:cp15.peak.B1.daily_mean_level_MAE|p=1} EUR/MWh against a within-day shape error of
           {r:cp15.peak.B1.within_day_shape_MAE|p=1} EUR/MWh.""", PAGE, README),
    # ---- adopted transitions (PUBLISH_RULES 1.0 A3): one summary per adopted generation after the first ---------
    _block("transition.v4-v5.title", "P55", """From {g:v4.version} to {g:v5.version}: adding a distributional neural
           network""", PAGE, README),
    _block("transition.v4-v5.change", "P55", """{g:v5.version} kept {g:v4.version}'s inputs, its two LEAR forecasts and its
           LightGBM, and added {s:name:DDNN-2}, a distributional neural network that forecasts the whole next day at once, with one
           sixth of the blend's weight, taken from the LightGBM's third; the hour-aware intervals were re-estimated on the
           new forecast's own errors.""", PAGE, README),
    _block("transition.v4-v5.comparator", "P55", """Set in advance: {g:v4.version} itself, on identical hours, so the
           protocol's comparator is also the predecessor and the comparison is direct.""", PAGE, README),
    _block("transition.v4-v5.result", "P55", """Point-error score {r:derived.change.v5.S_MAE}
           {r:derived.change.v5.S_MAE|ci} and interval score {r:derived.change.v5.S_WIS}
           {r:derived.change.v5.S_WIS|ci}, as a share of {g:v4.version}'s score, with {s:level:95%} confidence intervals
           of that ratio from the experiment's own bootstrap draws; at the rule's {s:level:97.5%} decision level,
           {r:derived.change_decision.v5.S_MAE|ci} and {r:derived.change_decision.v5.S_WIS|ci}. Development evidence
           after selection, not a test on new data.""", PAGE, README),
    _block("transition.v4-v5.limits", "P54", """The comparison scores {r:derived.coverage.days} of the
           {r:derived.coverage.data_days} days of data up to the end of the development period,
           {r:derived.coverage.first_data_day} to {r:derived.coverage.data_end}; {r:derived.coverage.q4_days} of them fall
           in October to December, and only {r:derived.coverage.winters} winter (December to February) contains any of
           them. It is the second decision on a distributional neural network over the same
           five test periods; only a one-shot test on days never used for development or selection protects against
           that reuse.""", PAGE),
    _block("transition.v3-v4.title", "P52", """From {g:v3.version} to {g:v4.version}: adding a LightGBM
           forecaster""", PAGE, README),
    _block("transition.v3-v4.change", "P52", """{g:v4.version} kept {g:v3.version}'s inputs and its two LEAR forecasts,
           added a LightGBM forecaster with one third of the blend's weight, and re-estimated the hour-aware intervals on
           the new forecast's own errors. The gain comes from the combination: on its own, no LightGBM arm showed an
           improvement over {g:v3.version}, but the blend improves both error scores, because a nonlinear model built
           from decision trees makes different errors from the two linear forecasts. The LightGBM's split into hour
           blocks is not shown to contribute.""", PAGE, README),
    _block("transition.v3-v4.comparator", "P52", """Set in advance: {g:v3.version} itself, on identical hours, so the
           protocol's comparator is also the predecessor and the comparison is direct.""", PAGE, README),
    _block("transition.v3-v4.result", "P52", """Point-error score {r:derived.change.v4.S_MAE}
           {r:derived.change.v4.S_MAE|ci} and interval score {r:derived.change.v4.S_WIS}
           {r:derived.change.v4.S_WIS|ci}, as a share of {g:v3.version}'s score, with {s:level:95%} confidence
           intervals of that ratio from the experiment's own bootstrap draws; development evidence, not a test on new data.""",
           PAGE, README),
    _block("transition.v3-v4.limits", "P52", """Separate hour-block models are not shown to help: the three-block
           against the pooled LightGBM shows no demonstrated joint preference. A gain over the August {s:date:2022} peak
           is not shown either: over those {r:cp21.diagnostics.HGL.peak.n_days} days, {g:v4.version}'s point error is
           higher than {g:v3.version}'s.""", PAGE),
    _block("transition.v2-v3.title", "P35", """From {g:v2.version} to {g:v3.version}: adding weather forecasts""",
           PAGE, README),
    _block("transition.v2-v3.change", "P35", """{g:v3.version} kept {g:v2.version}'s blend of two LEAR forecasts and
           its hour-aware intervals, and added three weather forecasts available before the auction: wind speed at
           {s:height:10 m} and {s:height:100 m} and solar radiation, each with a missing-data indicator.""",
           PAGE, README),
    _block("transition.v2-v3.comparator", "P35", """Set in advance: {g:v2.version} itself, on identical hours, so the
           protocol's comparator is also the predecessor and the comparison is direct.""", PAGE, README),
    _block("transition.v2-v3.result", "P35", """Point-error score {r:derived.change.v3.S_MAE}
           {r:derived.change.v3.S_MAE|ci} and interval score {r:derived.change.v3.S_WIS}
           {r:derived.change.v3.S_WIS|ci}, as a share of {g:v2.version}'s score, with {s:level:95%} confidence
           intervals; development evidence, not a test on new data.""", PAGE, README),
    _block("transition.v2-v3.limits", "P35", """The three weather inputs were tested together, so no single input's
           contribution is isolated; within the {s:date:2022} crisis period the point-error gain is not demonstrated
           on its own.""", PAGE),
    _block("transition.v1-v2.title", "P36", """From {g:v1.version} to {g:v2.version}: a blended linear model with
           hour-aware intervals""", PAGE, README),
    _block("transition.v1-v2.change", "P36", """After {g:v1.version} failed in the {s:date:2022} crisis,
           {g:v2.version} replaced its LightGBM quantile model and fixed-window calibration with an equal blend of
           two LEAR forecasts refitted every day, and intervals built from recent errors by hour of the day.""",
           PAGE, README),
    _block("transition.v1-v2.comparator", "P36", """Set in advance: {g:daily-lear.inline}, the strongest benchmark,
           not the predecessor {g:v1.version}; a pooled-interval control isolated the interval method.""",
           PAGE, README),
    _block("transition.v1-v2.result", "P36", """Against {g:daily-lear.inline}: point-error score
           {r:derived.change.v2.S_MAE} {r:derived.change.v2.S_MAE|ci} and interval score {r:derived.change.v2.S_WIS}
           {r:derived.change.v2.S_WIS|ci}, as a share of {g:daily-lear.inline}'s score, with {s:level:95%}
           confidence intervals; development evidence, not a test on new data.""", PAGE, README),
    _block("transition.v1-v2.predecessor", "P36", """No paired interval for {g:v2.version} against {g:v1.version}
           exists: the protocol did not test that pair. As descriptive context on the same development hours,
           without an interval, the point-error scores are {r:cp20.metrics.B1.equal_fold.S_MAE|p=3} for
           {g:v1.version} and {r:cp20.metrics.H0.equal_fold.S_MAE|p=3} for {g:v2.version}, and the interval scores
           {r:cp20.metrics.B1.equal_fold.S_WIS|p=3} and {r:cp20.metrics.H0.equal_fold.S_WIS|p=3}.
           {g:v1.version}'s one-shot holdout covers a different window and evidence class and is not compared.""",
           PAGE, README),
    _block("transition.v1-v2.limits", "P36", """The gain over {g:daily-lear.inline} is not attributed to hour-aware
           intervals alone, the blend and the interval layer are not separated, and against its pooled-interval
           control there is no demonstrated joint preference.""", PAGE),
    # ---- the released product, documented (PUBLISH_RULES 1.0 §5 and A4): the twelve subjects, for the model the
    # registry names as released. Each block states which model, output and rows its evidence describes.
    _block("product.lede", "P37", """The released model, {g:released.name}, topic by topic: the artifact its one-shot
           test evaluated and the demo computes.""", PAGE, README),
    _block("product.data.target", "P38", """**What it forecasts.** The German–Luxembourg day-ahead price of each
           hour of the next delivery day, in EUR/MWh: {s:hours:24} values on most days and {s:hours:23} or
           {s:hours:25} on daylight-saving transition days, issued together as one curve before the
           {s:time:12:00 CET} auction gate."""),
    _block("product.data.sources", "P38", """**Where the data comes from.** One committed hourly snapshot, from
           {s:date:2019-01-01} to {v1:snapshot_cutoff}, of the day-ahead price and the day-ahead load forecast,
           pinned by its {s:name:SHA-256} hash. It was pulled from SMARD.de, which republishes the ENTSO-E series, and a sample
           spanning every regime agreed with the ENTSO-E Transparency Platform to the cent. {v1:attribution}"""),
    _block("product.data.cutoff", "P38", """**The information cutoff.** Every price-derived input uses only prices
           of delivery days before the forecast day, and rolling price statistics are frozen at the day before. The
           day-ahead wind and solar forecast is published after the gate, so the model never uses it. That the load
           forecast exists before the gate is an assumption, not a measurement."""),
    _block("product.data.windows", "P38", """**Fitting, calibration and test data.** The model was fitted on data up
           to {v1:raw_model_fit_cutoff}, calibrated once on {v1:final_calibration_window} and tested once on
           {v1:holdout_window}. The demo and this page replay the committed snapshot; no live data feed is read."""),
    _block("product.data.preprocessing", "P38", """**Preprocessing.** A quarter-hour series becomes an hour only from
           four complete quarter-hours; from {s:date:2025-10-01} the hourly price is the mean of four quarter-hour
           prices. The price is not log-transformed, because it is often negative."""),
    _block("product.regimes.periods", "P39", """Three regimes shape the development period: before
           {s:date:2021-09-01}, the energy crisis from {s:date:2021-09-01} to {s:date:2022-12-31}, and from
           {s:date:2023-01-01} a market with frequent negative prices. The development hours split into
           {r:cp2.regime.pre_crisis.n_obs}, {r:cp2.regime.crisis.n_obs} and {r:cp2.regime.post_crisis.n_obs} hours
           across them; {r:cp2.regime.negative_price.n_obs} hours, on {r:cp2.regime.negative_price.n_days} days, had a
           negative price."""),
    _block("product.regimes.why", "P39", """Why it matters: the model learns from calm, crisis and negative-price
           periods together, so its errors and interval coverage are reported per regime, and the calibration's
           assumption that recent errors resemble future ones breaks when the regime shifts. {v1:floor_change}"""),
    _block("product.inputs.catalog", "P40", """{g:released.version} uses {v1:champion_features} inputs, a catalog
           frozen before fitting: calendar, holiday and seasonal features; two date-based regime flags; the
           day-ahead load forecast for the hour and for the day; the price of the same hour one, two and seven days
           earlier; rolling price statistics over the previous week and month; and a weekly count of negative-price
           hours."""),
    _block("product.inputs.excluded", "P40", """**What was left out, and why.** The one candidate domain input, a
           residual-load proxy built from actual wind and solar generation, had to lower the pooled pinball loss in a
           comparison fixed before fitting; it did not ({v1:catalog_pct}). The day-ahead wind and solar forecast is
           excluded because it is published after the gate; a separate benchmark measured what it would have been
           worth ({v1:benchmark_pct})."""),
    _block("product.inputs.seasonal", "P40", """**Why hour-of-day and day-of-week inputs.** The price spectrum shows
           pronounced cycles at {s:period:24} and {s:period:168} hours, with a {s:period:12}-hour harmonic, and their
           amplitude changed in the crisis; the calendar inputs follow that structure rather than assuming it."""),
    _block("product.validation.design", "P41", """**Before release.** Five development folds of
           {s:days:90} days each, spanning all three regimes, were scored walk-forward: each fold's model was fitted
           on earlier data only, with a gap of one delivery day before its test block. The similar-day naive, a
           {s:hours:168}-hour seasonal naive and ridge regression were scored on the same rows. The folds hold
           {r:cp2.regime.all.n_obs} hours on {r:cp2.regime.all.n_days} days, the same hours the research comparison
           uses."""),
    _block("product.validation.holdout", "P41", """**The one-shot test.** After every choice was frozen, the model
           was fitted once, calibrated once on {v1:final_calibration_window} and frozen; only then was the
           {v1:holdout_days}-day holdout, {v1:holdout_window}, opened and scored, once. {v1:holdout_dm_label}"""),
    _block("product.validation.classes", "P41", """**Evidence classes.** The development folds also chose the input
           catalog, so their results are post-selection evidence; the holdout is the only test specified in advance;
           prospective live evidence is a third class, reserved for a future final model."""),
    _block("product.validation.leakage", "P41", """**Leakage checks, on the browser path.**
           {v1:wasm_availability_statement} The model's input schema also refuses the post-gate forecast."""),
    _block("product.results.coverage", "P42", """Its prediction intervals covered {v1:holdout_coverage_50},
           {v1:holdout_coverage_80} and {v1:holdout_coverage_95} of the holdout's hours at the
           {s:level:50, 80 and 95%} levels."""),
    _block("product.results.shared", "P42", """In the shared development comparison, whose error scores divide by the
           naive's error in each period, {g:v1.version}'s development replay scores
           {r:cp20.metrics.B1.equal_fold.S_MAE|p=3} on the point-error score and
           {r:cp20.metrics.B1.equal_fold.S_WIS|p=3} on the interval score, where the naive scores
           {r:cp20.metrics.B0.equal_fold.S_MAE|p=3}; lower is better."""),
    _block("product.attribution.headline", "P43", """The price of the same hour one day earlier dominates the median
           forecast."""),
    _block("product.attribution.reading", "P43", """The weekly lag, the rolling weekly price quantiles, the two-day lag
           and the day of the week follow, close to one another."""),
    _block("product.reliability.headline", "P46", """Calibration raised coverage, which stayed below nominal at every
           level, on the holdout and on the development folds."""),
    _block("product.attribution.identity", "P43", """**Which model and rows.** These are the released artifact's own
           SHAP values for its median head on fold {s:fold:5}'s test block, {s:date:2026-01-08} to
           {s:date:2026-04-07}: rows it was fitted on, so an in-sample diagnostic. Fold {s:fold:5}'s development
           model, explained out of sample on the same rows, ranks the inputs almost identically (Spearman rank
           correlation {r:cp2.diagnostics.frozen_vs_fold5.rank_spearman};
           {r:cp2.diagnostics.frozen_vs_fold5.top10_overlap} of the top {s:count:10} shared), but agreement does not
           make them the same evidence."""),
    _block("product.attribution.scope", "P43", """SHAP on the median explains its central tendency, not the interval
           width, and it is not a test of incremental value: that was the comparison fixed before fitting, in the
           inputs topic."""),
    _block("product.importance.permutation", "P44", """**Permutation importance** shuffles one input at a time and
           measures how much the median forecast's MAE rises: here for fold {s:fold:5}'s development model on its
           own test block, out of sample. {r:cp2.diagnostics.permutation.1.feature} adds
           {r:cp2.diagnostics.permutation.1.importance_mean_mae_increase|p=1} EUR/MWh,
           {r:cp2.diagnostics.permutation.2.feature} {r:cp2.diagnostics.permutation.2.importance_mean_mae_increase|p=1}
           and {r:cp2.diagnostics.permutation.3.feature}
           {r:cp2.diagnostics.permutation.3.importance_mean_mae_increase|p=1}. Its ranking agrees with SHAP's at the
           top and differs in the middle (Spearman rank correlation
           {r:cp2.diagnostics.shap_vs_permutation.rank_spearman})."""),
    _block("product.importance.limits", "P44", """Neither ranking measures incremental value or causation; the only
           incremental-value test was the comparison fixed before fitting."""),
    _block("product.importance.sensitivity", "P44", """**Scenario sensitivity.** The load-forecast control in the
           replay and the demo multiplies the day's load forecast and holds every other input fixed.
           {v1:sensitivity_probe_label}"""),
    _block("product.failures.reading", "P45", """Errors and coverage are worst in the crisis: MAE
           {r:cp2.regime.crisis.mae|p=1} EUR/MWh [{r:cp2.regime.crisis.mae_ci95_low|p=1},
           {r:cp2.regime.crisis.mae_ci95_high|p=1}], with the {s:level:95%} interval covering
           {r:cp2.regime.crisis.coverage_95|p=3} of {r:cp2.regime.crisis.n_obs} hours, against
           {r:cp2.regime.pre_crisis.mae|p=1} EUR/MWh and {r:cp2.regime.pre_crisis.coverage_95|p=3} before it and
           {r:cp2.regime.post_crisis.mae|p=1} EUR/MWh and {r:cp2.regime.post_crisis.coverage_95|p=3} after. Over the
           August {s:date:2022} peak weeks the {s:level:95%} interval covered
           {r:cp2.regime.august_2022_peak.coverage_95|p=3} of {r:cp2.regime.august_2022_peak.n_obs} hours; on
           negative-price hours, {r:cp2.regime.negative_price.coverage_95|p=3}."""),
    _block("product.failures.identity", "P45", """**Which model.** These strata come from the five development folds,
           each scored by the model fitted for that fold and calibrated on its own window, not by the frozen
           artifact, whose holdout covers only the most recent regime. Thin subsets carry day-block bootstrap
           {s:level:95%} confidence intervals and are read qualitatively. The crisis failure's mechanism is in the
           limitations: shrinkage toward the training price level, with calibration thresholds estimated at a lower
           level."""),
    _block("product.reliability.stages", "P46", """The intervals come from nine quantile heads, then conformalized
           quantile regression (CQR), which shifts them by thresholds calibrated on a recent window, then an
           isotonic step that keeps the quantiles in order. On the one-shot holdout the {s:level:95%} interval
           covered {r:cp2.holdout.coverage.raw.95} of hours from the raw heads, {r:cp2.holdout.coverage.post_cqr.95}
           after CQR and {r:cp2.holdout.coverage.final.95} after the isotonic step; on the development folds,
           {r:cp2.reliability.raw.95}, {r:cp2.reliability.post_cqr.95} and {r:cp2.reliability.final.95}."""),
    _block("product.reliability.width", "P46", """Coverage is read with width: over the development hours the final
           intervals averaged {r:cp20.metrics.B1.pooled.mean_width50|p=1},
           {r:cp20.metrics.B1.pooled.mean_width80|p=1} and {r:cp20.metrics.B1.pooled.mean_width95|p=1} EUR/MWh at
           the {s:level:50, 80 and 95%} levels. The holdout report records coverage but not width."""),
    _block("product.reliability.crossings", "P46", """The one hard gate was correctness, not a favourable number:
           adjacent quantiles crossed {v1:crossings_development_raw} times from the raw heads and
           {v1:crossings_development_post_cqr} times after CQR on the development folds, and
           {v1:crossings_development_final} times after the isotonic step; {v1:crossings_holdout_final} on the
           holdout."""),
    _block("product.reliability.guarantee", "P46", """CQR's finite-sample marginal guarantee applies to its output
           before the isotonic step, and only when past and future errors are exchangeable; regime shifts break that,
           so every coverage figure here is empirical."""),
    _block("product.forecast.read", "P47", """**How to read it.** The line is the median forecast for each local
           hour; the band is the prediction interval at the level you choose; the dashed line is the price that
           cleared, an outcome that was never an input. {v1:replay_label}"""),
    _block("product.forecast.controls", "P47", """**The controls.** The level selector reads other quantiles of the
           same forecast, and the frozen model's empirical coverage at that level over the holdout is shown beside
           the chart. The load-forecast scenario multiplies the day's load forecast with every other input held
           fixed, and hides the cleared price, which belongs to the unperturbed day."""),
    _block("product.forecast.demo", "P47", """**Computed in your browser.** The demo forecasts historical delivery days
           with the released model's own boosters, so a Python runtime and the model download first, about
           {v1:wasm_cold_load_mb} MB on a first visit; on a committed fixture of {v1:wasm_fixture_days} delivery days
           its output equals the frozen artifact's bitwise. Change the interval level or the load scenario there as
           here."""),
    _block("product.limits.summary", "P48", """The main limits, each in full below: the development results are
           post-selection evidence; the intervals under-cover when the regime shifts, severely in the
           {s:date:2022} crisis; the model is frozen, fitted {v1:staleness_days} delivery days before its snapshot
           ends; the load forecast's availability before the gate is an assumption, not a measurement; and it is a
           portfolio artifact, not an operations system."""),
    _block("product.run.local", "P49", """**Run the released model on your machine.** Install the pinned
           dependencies with {s:command:uv sync}, then
           {s:command:uv run python predict_next_day.py --level 80 --self-check} forecasts a delivery day offline
           from the bundled snapshot and re-proves the gate boundary."""),
    _block("product.run.identity", "P49", """**Check its identity.** The artifact fingerprint is
           {v1:champion_fingerprint}, the same value in the model card, the one-shot holdout report and the
           registered model's alias; the snapshot's {s:name:SHA-256} hash is {v1:snapshot_sha256}."""),
    _block("product.run.demo", "P49", """**Check the browser computation.** {s:command:make wasm} rebuilds the browser
           payload from the committed model, proves its output equal to the frozen artifact's on the fixture and
           exports the demo; {s:command:uv run pytest tests/test_22_wasm_equivalence.py -q} reruns that proof
           alone."""),
    _block("product.run.historical", "P49", """The archived report's container instructions are historical: the demo
           is a static Hugging Face Space that computes in the browser, and publishing it is the Owner's action."""),
)

BLOCKS_BY_KEY: dict[str, Block] = {block.key: block for block in BLOCKS}

#: Chart claims: the records each figure may plot, beyond those its caption names.
CHART_CLAIMS: dict[str, str] = {
    "overview": "P07",
    "v5.c2a": "C420",
    "v4.c2a": "C107",
    "v4.ladder": "C106",
    "v4.c2b": "C110",
    "v4.c3": "C118",
    "v4.c4": "C120",
    "v4.c6": "C121",
    "v3.c2a": "C69",
    "v3.c2b": "C72",
    "v3.c3": "C74",
    "v3.c4": "C79",
    "v3.c5": "C82",
    "v3.c6": "C80",
    "v2.chart1": "C31",
    "v2.chart2": "C37",
    "preview": "P05",
    "targets": "P25",
    "product.shap": "P43",
    "product.coverage": "P46",
    "product.forecast": "P47",
}

#: Claims bound outside a block: single chart rows where a chart mixes claims (v2's three
#: contrasts), the target verdicts in the comparison, and the demo's startup figure (P13).
ROW_CLAIMS: tuple[str, ...] = ("C33", "C34", "C38", "C70", "C108", "C112", "C113", "C114", "C116", "P13", "P16", "P20", "P22", "P23", "P28",
                               "P33", "P34", "P45", "P48", "P50")

#: README research blocks (plan §9.4), each rendered from the same template as the page.
README_BLOCKS: tuple[str, ...] = tuple(block.key for block in BLOCKS if README in block.surfaces)

#: Records shown in full wherever they appear in a value table (standard §4; plan §6 invariant 9 as amended).
EXACT_FIELDS: set[tuple[str, str]] = set()


# --------------------------------------------------------------------------- the headline (standard §3.4)


def headline_template(entry: G.Entry | None = None) -> tuple[str, str]:
    """(claim, template) of the headline block, generated from the registry's current generation and
    its derived records. It leads with the pre-specified verdict when a rule existed (§3.3 a), and with
    the change against the comparator otherwise (§3.3 b)."""
    entry = entry or G.current_generation()
    verdict_id = f"derived.criteria.{entry.id}.verdict"
    if entry.id in D.ADOPTION_RULES:
        # The generation's own pre-registered adoption rule (CP-21 on): it leads, with the rule's paired differences
        # in the rule's unit, each metric named beside its value (A1), and N (§3.3 a). The change as a share of the
        # comparator's score belongs to the orientation layer (§3.4).
        base = f"derived.adoption.{entry.id}"
        comparator = G.get(entry.comparator)
        if D.get(f"{base}.verdict").value != "met":
            raise ClaimError(f"{entry.id} is a generation whose adoption rule was not met")
        tested = D.get(f"{base}.tested").value
        level = D.ADOPTION_RULES[entry.id]["level"]
        at_level = " at {s:level:%s}" % level if level != "95%" else ""  # the rule's decision level, defined beneath
        template = (
            "Met the adoption rule set before the experiment: both error scores improved on {g:%s.version}'s "
            "({g:%s.version} − {g:%s.version}%s: {r:%s.distance.S_MAE} "
            "{r:%s.distance.S_MAE|ci} on the point-error score, {r:%s.distance.S_WIS} {r:%s.distance.S_WIS|ci} on "
            "the interval score; {r:%s.tested} %s tested against the rule)."
            % (comparator.id, entry.id, comparator.id, at_level, base, base, base, base, base,
               "policy" if tested == "1" else "policies"))
        return D.ADOPTION_RULES[entry.id]["claim"], template
    if "criteria-1-2" in entry.rules and verdict_id in D.records():
        margins = {D.get(f"derived.rule.margin.{score}").value for score in ("S_MAE", "S_WIS")}
        if len({D.display(D.get(f"derived.rule.margin.{s}")) for s in ("S_MAE", "S_WIS")}) != 1:
            raise ClaimError(f"the two targets' margins differ: {sorted(margins)}")
        benchmark = D.get("derived.rule.margin.S_MAE").subject
        base = f"derived.criteria.{entry.id}"
        # PUBLISH_RULES 1.0 A1: every headline value names its metric beside it, so no later sentence, chart
        # or disclosure is needed to tell which number is which.
        if D.get(f"{base}.verdict").value == "met":
            first = D.get(f"{base}.first_to_meet").value == "yes"
            tail = ("the first of {r:%s.tested} policies tested to meet them" % base if first
                    else "one of the {r:%s.tested} policies tested against them" % base)
            template = (
                "Met both accuracy targets set before the experiments: error scores at least "
                "{r:derived.rule.margin.S_MAE} below the strongest benchmark, {g:%s.inline} "
                "({g:%s.version}: {r:%s.distance.S_MAE|abs} below on the point-error score and "
                "{r:%s.distance.S_WIS|abs} below on the interval score; %s)."
                % (benchmark, entry.id, base, base, tail))
        else:
            template = (
                "Did not meet the accuracy targets set before the experiments: error scores at least "
                "{r:derived.rule.margin.S_MAE} below the strongest benchmark, {g:%s.inline} "
                "({g:%s.version}: {r:%s.distance.S_MAE} on the point-error score and {r:%s.distance.S_WIS} on "
                "the interval score, against it; {r:%s.tested} policies tested)." % (benchmark, entry.id, base, base, base))
        return "P20", template
    change = f"derived.change.{entry.id}"
    comparator = G.get(entry.comparator)
    return "P20", (
        "Against {g:%s.version}, the point-error score changed by {r:%s.S_MAE} {r:%s.S_MAE|ci} and the "
        "interval score by {r:%s.S_WIS} {r:%s.S_WIS|ci}, as a share of {g:%s.version}'s score."
        % (comparator.id, change, change, change, change, comparator.id))


def target_sentence(target: str = "html", *, surface: str = PAGE) -> str:
    """The target line in words, with its date, and N (standard §15 amending plan §8.2): what the
    comparison's dashed lines mark, how many policies were tested against it and who met it. The README, which draws
    no chart, states the same targets without the lines."""
    entry = G.current_generation()
    base = f"derived.criteria.{entry.id}"
    benchmark = D.get("derived.rule.margin.S_MAE").subject
    lead = "The dashed lines mark the targets" if surface == PAGE else "The accuracy targets"
    template = ("%s: each error score at least {r:derived.rule.margin.S_MAE} below the strongest benchmark, "
                "{g:%s.inline} (set {r:derived.rule.set_on}). {r:%s.tested} policies were tested against them; "
                % (lead, benchmark, base))
    met = [record.subject for record in D.records().values() if record.kind == "verdict" and record.value == "met"]
    first = [record.subject for record in D.records().values() if record.kind == "first_to_meet" and record.value == "yes"]
    if D.get(f"{base}.verdict").value == "met" and D.get(f"{base}.first_to_meet").value == "yes" and met == [entry.id]:
        template += "{g:%s.version} met both, the first and only one to do so." % entry.id
    elif D.get(f"{base}.verdict").value == "met" and first and first != [entry.id]:
        # Named from the verdict records, never typed: the first to meet them, then every later policy that did. Once
        # more than four did, they are counted and only the adopted generations among them are named.
        later = [subject for subject in met if subject not in first]
        if len(later) <= LISTED_LATER:
            names = ["{g:%s.version}" % s if G.get(s).version else "{g:%s.inline}" % s for s in later]
            listed = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
            template += "{g:%s.version} was the first to meet both; %s met them too." % (first[0], listed)
        else:
            names = ["{g:%s.version}" % s for s in later if G.get(s).version]
            listed = ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else "".join(names)
            template += ("{g:%s.version} was the first to meet both; {s:count:%d} later policies met them too, among "
                         "them %s." % (first[0], len(later), listed))
    elif D.get(f"{base}.verdict").value == "met":
        template += "{g:%s.version} met both." % entry.id
    else:
        template += "{g:%s.version} did not meet them." % entry.id
    return render_template("P25", template, target)


#: The most later policies the target sentence names one by one; beyond it they are counted (PRES-4: seventeen met).
LISTED_LATER = 4

#: Every term a headline form may introduce, in reading order.
ALL_TERMS = ("terms.error_scores", "terms.targets", "terms.benchmark", "terms.policies", "terms.adoption_rule",
             "terms.differences", "terms.class", "terms.ddnn2", "terms.not_adopted")


def comparison_terms(entry: G.Entry | None = None) -> tuple[str, ...]:
    """The terms the headline does not introduce, defined where the comparison uses them: its targets, benchmark and
    policies are still drawn there when the headline leads with a generation's own rule."""
    used = headline_terms(entry)
    comparison = tuple(key for key in ALL_TERMS if key not in used
                       and key not in ("terms.adoption_rule", "terms.differences", "terms.ddnn2", "terms.not_adopted"))
    # The comparison's rows include the candidates of branches tested and not adopted (PRES-4): named where they are drawn.
    points = [row for row in G.comparison_rows() if row.kind == "branch" or (
        row.kind == "study arm" and row.checkpoint in G.COMPARISON_EXPERIMENTS
        and G.get(G.CHECKPOINTS[row.checkpoint].owner).kind == "branch")]
    return comparison + (("terms.not_adopted",) if points else ())


def headline_terms(entry: G.Entry | None = None) -> tuple[str, ...]:
    """The terms the headline introduces, defined directly below it (standard §1 ii): they follow the headline's
    form, so a term is never defined for a headline that does not use it."""
    entry = entry or G.current_generation()
    # A model name the generation's own name introduces is defined beneath the headline too (PRES-4: DDNN-2).
    named = ("terms.ddnn2",) if "DDNN-2" in entry.name else ()
    if entry.id in D.ADOPTION_RULES:
        return ("terms.error_scores", "terms.adoption_rule", "terms.differences", "terms.policies", "terms.class") + named
    return ("terms.error_scores", "terms.targets", "terms.benchmark", "terms.policies", "terms.class") + named


def _tested(entry: G.Entry) -> bool:
    return f"derived.criteria.{entry.id}.verdict" in D.records()


def census_sentence(target: str = "html") -> str:
    """The headline's N and the comparison's rows are different counts (PUBLISH_RULES 1.0 §3.3; plan §5.2): said
    where they meet, with both derived -- N from the decision records, the rows from the registry's comparison."""
    rows = G.comparison_rows()
    tested = [entry for entry in rows if _tested(entry)]
    references = []
    for entry in rows:
        if entry in tested:
            continue
        if entry.version:
            references.append("{g:%s.version}'s development replay" % entry.id)
        else:  # the normalizer reads "the similar-day naive"; a named benchmark keeps its bare name
            references.append(("the " if entry.comparator is None else "") + "{g:%s.inline}" % entry.id)
    names = ", ".join(references[:-1]) + " and " + references[-1] if len(references) > 1 else "".join(references)
    template = ("The chart's {s:count:%d} rows are not that census: {s:count:%d} of them were tested against the "
                "targets, and the others are references that never were: %s." % (len(rows), len(tested), names))
    return render_template("P33", template, target)


def census_detail(target: str = "html") -> str:
    """The exact census behind N, by decision date (plan §5.2): every identity tested against the targets, once."""
    subjects = sorted({record.subject for record in D.records().values() if record.kind == "verdict"},
                      key=lambda ident: (G.first_status(G.get(ident)).date, G.entries().index(G.get(ident))))
    groups: dict[str, list[str]] = {}
    for ident in subjects:
        entry = G.get(ident)
        name = "{g:%s.name}" % ident if entry.version else "{s:name:%s}" % entry.name
        groups.setdefault(G.first_status(entry).date, []).append(name)
    # Names can carry commas ("Normalized component mean, variant"), so a semicolon separates them.
    parts = ["Decided {s:date:%s}: %s." % (day, "; ".join(names)) for day, names in groups.items()]
    current = G.current_generation()
    template = ("**The {r:derived.criteria.%s.tested} policies tested against the targets**, each identity counted "
                "once, by decision date. %s" % (current.id, " ".join(parts)))
    return render_template("P33", template, target)


def headline(target: str = "html") -> str:
    """The headline block: one sentence plus the evidence-class badge, identical on the page and in
    the README (§3.4)."""
    claim_id, template = headline_template()
    sentence = render_template(claim_id, template, target)
    badge = G.current_generation().badge
    if target == "md":
        return f"{sentence} `{badge}`"
    return f'{sentence} <span class="badge badge-development">{html.escape(badge)}</span>'


# --------------------------------------------------------------------------- rendering

_TOKEN = re.compile(r"\{(r|s|g|v1):([^{}]+)\}")
_RECORD_TOKEN = re.compile(r"\{r:([^|{}]+)(?:\|[^{}]*)?\}")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")


class ClaimError(ValueError):
    """A template, binding or rendering request that the claim layer refuses."""


def _html_value(claim_id: str, record_id: str, which: str, text: str, raw: str) -> str:
    field = "" if which == "value" else f' data-field="{which}"'
    return (
        f'<data value="{html.escape(raw)}" data-claim="{claim_id}" data-record="{record_id}"{field}>'
        f"{html.escape(text)}</data>"
    )


def structural(kind: str, text: str, target: str = "html") -> str:
    """A declared structural numeral: labels, dates, fold indices, control values."""
    if target == "md":
        return text
    return f'<span data-structural="{html.escape(kind)}">{html.escape(text)}</span>'


def _value_text(record: R.EvidenceRecord, which: str, option: str | None) -> str:
    if (record.record_id, which) in EXACT_FIELDS or option == "exact":
        return R.display(record, which, style="exact")
    if option == "pct":
        return R.display(record, which, style="percent")
    if option and option.startswith("p="):
        return R.display(record, which, precision=int(option[2:]))
    if option == "abs":
        text = R.display(record, which)
        return text.lstrip("+" + R.MINUS)
    return R.display(record, which)


def _render_derived(claim_id: str, record_id: str, option: str | None, target: str) -> str:
    """A typed derived record (standard §3.3) in HTML or Markdown: whole percent, one decimal for
    EUR/MWh, the word itself for a label or a date. It carries `data-derived` beside its claim."""
    record = D.get(record_id)

    def one(which: str, style: str | None) -> str:
        text = D.display(record, which, style=style)
        raw = {"value": record.value, "ci_low": record.ci_low, "ci_high": record.ci_high}[which]
        if target == "md":
            return text
        field = "" if which == "value" else f' data-field="{which}"'
        return (f'<data value="{html.escape(raw)}" data-claim="{claim_id}" data-record="{record_id}" '
                f'data-derived="{record.kind}"{field}>{html.escape(text)}</data>')

    if option in ("ci", "ciexact"):
        if not record.interval:
            raise ClaimError(f"{record_id} has no interval")
        style = "exact" if option == "ciexact" else None
        return f"[{one('ci_low', style)}, {one('ci_high', style)}]"
    if option in ("lo", "hi"):
        return one("ci_low" if option == "lo" else "ci_high", None)
    if option not in (None, "abs", "exact"):
        raise ClaimError(f"{record_id}: unknown option {option!r} for a derived record")  # never the point instead
    return one("value", option)


def _render_record(claim_id: str, spec: str, target: str) -> str:
    record_id, _, option = spec.partition("|")
    option = option or None
    if D.is_derived(record_id):
        return _render_derived(claim_id, record_id, option, target)
    record = R.get(record_id)

    def one(which: str, opt: str | None) -> str:
        text = _value_text(record, which, opt)
        raw = {"value": record.raw, "ci_low": record.ci_low_raw, "ci_high": record.ci_high_raw}[which]
        return text if target == "md" else _html_value(claim_id, record_id, which, text, raw)

    if option in ("ci", "ci1", "ciexact"):
        if record.interval is None:
            raise ClaimError(f"{record_id} has no interval")
        places = {"ci1": "p=1", "ciexact": "exact"}.get(option)
        return f"[{one('ci_low', places)}, {one('ci_high', places)}]"
    if option == "lo":
        return one("ci_low", None)
    if option == "hi":
        return one("ci_high", None)
    if option == "exact_hi":
        return one("ci_high", "exact")
    return one("value", option)


#: The p-value floor on the reading path (standard §4): below it, "p < 10⁻⁶"; the exact value stays
#: in the value table.
P_FLOOR = 1e-6
P_FLOOR_TEXT = "p < 10⁻⁶"


def v1_display(key: str, option: str | None = None) -> str:
    """A v1 claim as the reading path shows it: `p=N` rounds to N decimals (standard §4), `pfloor`
    floors a p-value at 10⁻⁶. With no option it is the claim set's own string."""
    from .claims import build_claims

    text = build_claims()[key]
    if option and option.startswith("p="):
        return f"{float(text):.{int(option[2:])}f}".replace("-", R.MINUS)
    if option == "pfloor":
        return P_FLOOR_TEXT if float(text) < P_FLOOR else f"p = {text}"
    return text


def _v1(claim_id: str, spec: str, target: str) -> str:
    from .claims import build_claims

    key, _, option = spec.partition("|")
    raw = build_claims()[key]
    text = v1_display(key, option or None)
    if target == "md":
        return text
    shown = f' data-display="{html.escape(option)}"' if option else ""
    return (
        f'<data value="{html.escape(raw)}" data-claim="{claim_id}" data-v1="{key}"{shown}>'
        f"{html.escape(text)}</data>"
    )


#: Model names that carry a digit as part of the name, declared structural wherever a name or fixed copy uses them.
STRUCTURAL_NAMES = ("DDNN-2",)
_NAME_PARTS = re.compile(r"\b(v\d+)\b|(" + "|".join(re.escape(name) for name in STRUCTURAL_NAMES) + r")(?![\w-])")


def _structural_name(name: str, target: str) -> str:
    """A registry name with its version labels and digit-bearing model names declared structural: 'v3 · weather
    features', 'v5 · DDNN-2 added', 'DDNN member on v4'."""
    if target == "md":
        return name
    out, last = [], 0
    for match in _NAME_PARTS.finditer(name):
        out.append(html.escape(name[last:match.start()]))
        out.append(structural("version", match.group(1)) if match.group(1) else structural("name", match.group(2)))
        last = match.end()
    return "".join(out) + html.escape(name[last:])


def _registry(spec: str, target: str) -> str:
    """A registry token (standard §5): a name, version, badge or dated status, never typed in a template.

    `{g:ID.name}`, `{g:ID.inline}`, `{g:ID.short}`, `{g:ID.version}`, `{g:ID.badge}`, `{g:ID.adoption}`,
    `{g:ID.status}` and `{g:release_rule}`; `ID` may be `current` or `released`."""
    if spec == "release_rule":
        sentence = G.release_sentence()
        if target == "md":
            return sentence
        version = G.released().version
        body = html.escape(sentence).replace(f" {version}.", " " + structural("version", version) + ".", 1)
        return f'<span data-status="release-rule">{body}</span>'
    ident, _, attribute = spec.partition(".")
    entry = G.resolve(ident)
    if attribute in ("name", "inline", "short"):
        name = {"name": entry.name, "inline": G.inline_name(entry), "short": entry.short}[attribute]
        text = _structural_name(name, target)
        return text if target == "md" else f'<span data-registry="{entry.id}">{text}</span>'
    if attribute == "version":
        if entry.version is None:
            raise ClaimError(f"{entry.id} has no version")
        return structural("version", entry.version, target)
    if attribute == "badge":
        return entry.badge if target == "md" else html.escape(entry.badge)
    if attribute == "adoption":
        return G.adoption_label(entry) if target == "md" else html.escape(G.adoption_label(entry))
    if attribute == "status":
        sentence = G.status_sentence(entry)
        if target == "md":
            return sentence
        month = G.month(entry.status.date)
        body = html.escape(sentence).replace(month, structural("date", month), 1)
        if entry.version:
            body = body.replace(f" {entry.version} ", " " + structural("version", entry.version) + " ", 1)
        return f'<span data-status="{entry.id}">{body}</span>'
    raise ClaimError(f"unknown registry token {spec!r}")


def render_template(claim_id: str, template: str, target: str = "html") -> str:
    """Render one template to HTML or Markdown. Every number comes from a record or a declared
    structural token; everything else is escaped text."""
    if target not in ("html", "md"):
        raise ClaimError(f"unknown target {target!r}")
    out, position = [], 0
    for match in _TOKEN.finditer(template):
        literal = template[position:match.start()]
        out.append(literal if target == "md" else html.escape(literal, quote=False))
        kind, body = match.groups()
        if kind == "r":
            out.append(_render_record(claim_id, body, target))
        elif kind == "s":
            structural_kind, _, text = body.partition(":")
            out.append(structural(structural_kind, text, target))
        elif kind == "g":
            out.append(_registry(body, target))
        else:
            out.append(_v1(claim_id, body, target))
        position = match.end()
    tail = template[position:]
    out.append(tail if target == "md" else html.escape(tail, quote=False))
    text = "".join(out)
    if target == "html":
        text = _BOLD.sub(r"<strong>\1</strong>", text)
        text = _CODE.sub(r"<code>\1</code>", text)
    return text


def render(key: str, target: str = "html", *, surface: str = PAGE) -> str:
    """Render a published block for a surface; a withheld or off-surface block is refused."""
    block = BLOCKS_BY_KEY.get(key)
    if block is None:
        raise ClaimError(f"no block {key!r}")
    if block.status != "published":
        raise ClaimError(f"block {key!r} is {block.status}")
    if surface not in block.surfaces:
        raise ClaimError(f"block {key!r} may not appear on the {surface}")
    return render_template(block.claim_id, block.template, target)


def svg_binding(claim_id: str, record_id: str, which: str = "value") -> str:
    """The attributes an SVG `<text>` or mark carries for a research value (no `<data>` in SVG)."""
    if claim_id not in all_claim_ids():
        raise ClaimError(f"unknown claim {claim_id!r}")
    if D.is_derived(record_id):
        record = D.get(record_id)
        field = "" if which == "value" else f' data-field="{which}"'
        return f' data-claim="{claim_id}" data-record="{record_id}" data-derived="{record.kind}"{field}'
    R.get(record_id)
    field = "" if which == "value" else f' data-field="{which}"'
    return f' data-claim="{claim_id}" data-record="{record_id}"{field}'


def svg_value(record_id: str, which: str = "value", option: str | None = None) -> str:
    if D.is_derived(record_id):
        return D.display(D.get(record_id), which, style="abs" if option == "abs" else None)
    return _value_text(R.get(record_id), which, option)


# --------------------------------------------------------------------------- the maps


def all_claim_ids() -> set[str]:
    return {block.claim_id for block in BLOCKS} | set(CHART_CLAIMS.values()) | set(ROW_CLAIMS)


_MAP_ID = re.compile(r"^\|\s*((?:C|P|W)\d+)\s*\|", re.MULTILINE)


def claim_map_ids() -> set[str]:
    """Every claim ID defined in a claim map's tables."""
    found: set[str] = set()
    for path in CLAIM_MAPS:
        found |= set(_MAP_ID.findall(path.read_text()))
    return found


# --------------------------------------------------------------------------- phrase guards

#: Withheld claims (W1–W21) as phrases a surface must not contain. Each is written to catch the
#: claim, not an honest denial of it; `tests/test_30_research_claims_rendered.py` pins both.
WITHHELD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (wid, re.compile(pattern, re.IGNORECASE))
    for wid, pattern in (
        ("W1", r"\b(hour-aware intervals?|V2-H|H) (are|is) (better|preferred)\b"),
        ("W2", r"\b(H and P are equivalent|hour awareness has no effect)\b"),
        ("W3", r"\+?0\.0000\]|≈\s?0\b"),
        ("W4", r"hour-aware intervals fixed criterion"),
        ("W5", r"\bthe blend alone fixed\b"),
        ("W6", r"\b(product[- ]ready|production[- ]ready|qualified for production|promoted to production)\b"),
        ("W7", r"\bstatistically significant|significantly (beats|better|outperform)"),
        ("W8", r"\b(revenue|battery gain|annuali[sz]ed (value|gain|revenue))\b"),
        ("W9", r"\bbetter (solar|night|shoulder)[- ]hour\b"),
        ("W11", r"\bA1 beats B2\b"),
        ("W14", r"\b(Chronos-2|TabPFN|DDNN|VRE) (outperforms|beats|improves)\b"),
        ("W15", r"(?<!no )(?<!no conformal )\bcoverage guarantee\b|\bguaranteed coverage\b"),
        ("W17", r"\b(wind|radiation|solar)[^.]{0,40}\b(drives|explains|is responsible for) the (gain|improvement)\b"),
        ("W18", r"\bpeer[- ]review|external validation|externally validated\b"),
        ("W19", r"\b(v3 runs in|runs v3|try the v3|v3 demo|live v3)\b"),
        # A nominal interval level ("the 95% interval") is not a relative gain.
        ("W20", r"\bv3\b[^.]{0,60}\b\d+(\.\d+)?\s?(%(?!\s+(prediction |confidence )?interval)|x\b|times\b)[^.]{0,30}\bv1\b"),
        ("W21", r"(?<!never )(?<!not )\bconfirmatory\b(?!-style)"),
        # CP-21 (claim map W22–W27). W22: the block split beyond its endpoint reading, or a mechanism for it.
        ("W22", r"(?<!that )(?<!whether )\b(the block split|separate (hour-)?block models)\b"
                r" (helps?|helped|hurts?|hurt|improved?s?|is beneficial|is harmful|drives|explains)\b"),
        # W23: the daily-to-pooled LightGBM step as weather alone.
        ("W23", r"(?<!not )\b(a|the) weather effect on its own\b|\bweather alone (improved|lowered|explains)"),
        # W24: v4, HGL or a CP-21 arm as the demo, the product or live.
        ("W24", r"\b(v4 runs in|runs v4|try the v4|v4 demo|live v4|v4 is (the )?(product|live|released))\b"),
        # W25: a mixed or non-significant contrast called equivalent, or no effect, benefit or harm.
        ("W25", r"(?<!not )(?<!never )\b(is|are|was|were) equivalent\b|\bno (benefit|harm)\b"),
        # W26: the daily-cycle or fit-cost figures as qualification for daily operation.
        ("W26", r"(?<!not as a )\bqualif(y|ies|ied|ication) for daily operation\b"),
        # CP-24 (claim map W42-W48). W42: distributional networks "do not work" or "cannot help" in general.
        ("W42", r"\b(distributional (neural )?networks?|neural networks?|DDNN(-2)?)\b[^.]{0,20}"
                r"\b(do not|don't|does not|doesn't|cannot|can't) (work|help)\b"),
        # W43: v5 and v4 called equivalent, or as good as each other.
        ("W43", r"\bv5\b[^.]{0,40}\b(equivalent|as good as)\b[^.]{0,20}\bv4\b"),
        # W44: a forecast or metric attributed to PyTorch, a test-only correctness reference.
        ("W44", r"\bPyTorch\b(?![- ]reference)"),
        # W45: v5 or DDNN-2 as the product, live, or in the demo.
        ("W45", r"\b(v5 runs in|runs v5|try the v5|v5 demo|live v5|v5 is (the )?(product|live|released)|"
                r"DDNN-2 (runs in|is) the (demo|product))\b"),
        # W46: the pre-fold gate presented as evidence of improvement.
        ("W46", r"\bgate (shows?|showed|proves?|proved|demonstrates?|demonstrated)\b[^.]{0,30}\bimprove"),
        # W47: a 95% interval quoted as v5's decision interval (the rule decided at 97.5%).
        ("W47", r"\b95% decision (level|interval)\b|\bdecided at (the )?95%"),
    )
)

#: Stale status text the reconciliation removes (plan §8.7). The phrase guard runs on every
#: public surface, including the archived v1 report and the Space cards.
STALE_PHRASES: tuple[str, ...] = (
    "delu-m4",
    "No v2 run exists yet",
    "When M4 is ratified",
    "the defect the planned v2 targets",
    "the planned v2",
    "Development update · 2026-09-16",
    "CP-16 needs a complete new acceptance bar",
    "CP-16 requires a complete acceptance bar",
    "Current development — 2026-09-16",
)


#: Absolute network and hosting claims that are not true as stated (final audit F07): opening the
#: report still downloads it, independence from a CDN after loading is not an availability guarantee,
#: and the Space is hosted. Checked on the active surfaces; v1's preserved archive keeps its history.
NETWORK_ABSOLUTES: tuple[str, ...] = (
    "fetches nothing",
    "fetches **nothing**",
    "without any download",
    "cannot fail when a CDN",
    "no server;",
    "there is no server",
    "instant report",
)


def network_absolute_findings(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in NETWORK_ABSOLUTES if phrase.lower() in lowered]


def withheld_findings(text: str) -> list[str]:
    """Withheld claims found in `text`, as `W<n>: <match>`."""
    flat = " ".join(text.split())
    return [f"{wid}: {match.group(0)!r}" for wid, pattern in WITHHELD_PATTERNS for match in pattern.finditer(flat)]


def stale_findings(text: str) -> list[str]:
    flat = " ".join(text.split())
    return [phrase for phrase in STALE_PHRASES if phrase in flat]


__all__ = [
    "ADOPTED",
    "BADGE_DEVELOPMENT",
    "BADGE_PROSPECTIVE",
    "BADGE_V1_HOLDOUT",
    "BLOCKS",
    "BLOCKS_BY_KEY",
    "Block",
    "CHART_CLAIMS",
    "CLAIM_MAPS",
    "CONTRIBUTION_STATEMENT",
    "ClaimError",
    "OWNER_PUBLIC_NAME",
    "EXACT_FIELDS",
    "NOT_ADOPTED",
    "PLANNED",
    "README_BLOCKS",
    "STALE_PHRASES",
    "WITHHELD_PATTERNS",
    "all_claim_ids",
    "claim_map_ids",
    "render",
    "render_template",
    "stale_findings",
    "structural",
    "target_sentence",
    "census_sentence",
    "census_detail",
    "headline",
    "headline_template",
    "headline_terms",
    "comparison_terms",
    "svg_binding",
    "svg_value",
    "withheld_findings",
]
