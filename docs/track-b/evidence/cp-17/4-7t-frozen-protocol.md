# 4.7T's frozen protocol: registered before the window's data exist

**Status: registered at CP-17, outcome-blind.**
- No delivery day after 2026-09-06 exists on this machine, or has been downloaded, read or inferred.
- This protocol was written without any 4.7T outcome. It produces no 4.7T forecast.

**Authority:**
- `capstone_v21.md` v21-r17, §24.9's "registered before 4.7T opens" list, with §24.2 (the audit standard), §24.3
  (the window), §24.4 (the issue time) and §24.7 (the scores);
- the Owner's decisions 4, 5 and 8–12 (`docs/track-b/pre-cp-17-rulings-2026-10-25.md`, `d29b0abc…`);
- the brief's item 8.

**The anchor amendment (r18) ratifies these terms before 4.7T opens** (decision 5: "the exact endpoints and
tests are written in the amendment").

**Executable form:**
- `src/cp17/tools/fresh_test.py`, tested on synthetic losses only (`tests/cp17/test_fresh_test.py`);
- the boundary guard refuses the real window until the gate moves it.

## 1. The candidate, the window and the one look

- **The candidate is A1:** DDNN-2 alone, the nominal D2 that CP-28 selected, as the frozen product policy.
  - A1 issues when its own daily fit succeeds.
  - Otherwise the frozen failure policy issues (decisions 11 and 12; see the freeze record).
  - The candidate's score is the product's emitted forecast.
- **The window** is delivery days 2026-09-07 to 2026-12-05: 90 days (§24.3).
  - It opens no earlier than 2026-12-04, after the price for 2026-12-05 is published (decision 4).
  - It is never extended, shortened or rolled after any result.
- **One look.** No runner-up exists to move up: the candidate is CP-28's selection, and only one candidate is
  tested.

## 2. The comparators

| Role | Policy | How its window forecasts are made |
|---|---|---|
| Control (registered gate) | **v5** (nominal; the control under §24.5) | natively, under the rules CP-26 ran it by: v4's four members and D2 fitted daily, `v5_central`, v5's own H layer continued from CP-26's final state |
| Reference (registered gate) | **B0**, the similar-day naive | natively, as CP-26 ran it: the naive central, and its own raw-error layer continued from CP-26's final state |
| Descriptive | **v3** (HG with weather) and **v4** | natively, as CP-26 ran them, H layers continued |
| Descriptive | **v1's released artifact** | the released model as released, not refitted (CP-26's descriptive rule) |

The comparators' H and raw-error layers continue CP-26's final replay state from 2026-09-06. That state is
rebuilt and checked by `cp17.tools.states`, and the freeze manifest records it by digest.

**Comparator failures follow §24.7's rule:** a failed comparator fit is scored with B0's forecast and counted.
B0 has no fallback; a missing B0 is missing.

## 3. How the window's forecasts are generated

- **The product policy:** the frozen package (`src/cp17/`, by the freeze manifest's file hashes) under the
  issuing host, Pyodide 0.28.0 in Node.js v26.7.0 (decision 10), one worker.
  - It runs daily, at each origin D−1 11:00 Europe/Berlin (§24.4).
  - It uses only the data available strictly before that instant (the tier register's vintages and
    availability).
  - Each day is fitted from its own training window, in origin order. The issued artifact is written before
    D's outcome is opened.
- **The comparators:** natively, with BLAS at one thread, under CP-26's rules.
- **Vintages:** per the tier register (`reports/development-inputs/tier-register.json`) and the Owner's tier,
  margin and GFS rulings, which are unchanged.
- **The fallbacks' interval states:** the freeze record proposes, for r18, that the fallback chain's policies
  issue every day as shadow issuances, never published, to keep their own H buffers current (gap G1). 4.7T
  follows the rule r18 ratifies.
- **Counting:** every fallback, skipped fit, failed fit and failed issuance is counted by cause, and stays in
  the record.
- **No DDNN-2 search falls in the window.** The next search is S_14 = 2027-01-06, after 2026-12-05. A1 uses
  S_13's frozen ensemble throughout.

## 4. The primary scores

- **p50 MAE and the seven-quantile WIS.** The WIS is §24.7's: 50/80/95% intervals, weights α/2, median weight
  1/2, divided by 3.5 (`cp28.scoring.daily`).
- **Daily losses:**
  - a date's loss is the mean over its scored keys;
  - the scored keys of a paired comparison are CP-26's keys: the canonical hours (23, 24 or 25, with CP-26's
    DST rules) that CP-25's eligibility rules admit, applied to the window's days unchanged (a published price,
    and the frozen exclusion rules), intersected with the hours both policies emitted;
  - the mean daily loss weighs dates equally.
- **The product's loss is its emitted forecast's.** That is A1 on a successful fit, the frozen fallback
  otherwise. A failed issuance is missing in every pairing that involves it.

## 5. The endpoints, the compound test and the decision

Following decision 5, which mirrors CP-28's gates:

| # | Comparison | Daily differential | Null hypothesis |
|---|---|---|---|
| 1 | A1 vs v5, MAE | `d = L_A1 − 0.995·L_v5` | `E[d] ≥ 0` |
| 2 | A1 vs v5, WIS | `d = L_A1 − 0.995·L_v5` | `E[d] ≥ 0` |
| 3 | A1 vs B0, MAE | `d = L_A1 − L_B0` | `E[d] ≥ 0` |
| 4 | A1 vs B0, WIS | `d = L_A1 − L_B0` | `E[d] ≥ 0` |

- **The compound p-value** is the largest of the four one-sided p-values.
- **4.7T passes** when the compound p-value is ≤ α = 0.05. There is one candidate, so there is no Holm.
- **Boundary equality** is not a rejection.
- **The descriptive references** v3, v4 and v1's artifact are reported with the same scores. They are never
  tested and never move a decision.

## 6. The dependence treatment, suited to 90 days

- **The bootstrap** is CP-26's paired, non-circular, moving-block studentized bootstrap, as CP-28 landed it
  (`cp28.inference.test_endpoint`). Its parameters:
  - **7-day blocks;**
  - **10,000 replicates;**
  - **Bartlett HAC at lag 14** (weights 1 − l/15, divisor n²), unchanged;
  - **one calendar stratum,** all of 2026;
  - runs split at any missing date, and never crossed by a block.
- **The RNG** is a fresh `numpy.random.default_rng(20261204)`. It visits endpoints 1 to 4 in the table's
  order; an uncomputable endpoint consumes no draw.
- **The statistic:** T = mean(d)/√V, against replicate T_b = (mean(d_b) − mean(d))/√V_b. The one-sided
  p-value is (1 + #{T_b ≤ T}) / 10,001.
- **An uncomputable endpoint has p = 1.** That covers fewer than two dates, a non-finite or non-positive
  variance, or a non-finite replicate.
- **No sensitivity variant moves the decision.** The 14-day-block result is reported beside it, descriptively.

## 7. The minimum scorable support

- **The rule:** each comparison, against v5 and against B0, needs **at least 60 of the 90 dates paired and
  scored**. Below that, 4.7T is **inconclusive**: the window is spent, and v1 stays the product.
- **Why 60.**
  - With 7-day blocks, 60 dates give at least eight blocks to resample.
  - It allows up to 30 failed issuances or unpublished days (a third of the window) before the test loses its
    meaning. CP-26's development failure rates make that margin generous for a working pipeline.
  - A pipeline that loses more than a third of the window has failed operationally. That is a finding in
    itself, not a test of the policy.
- **The rule was set without any 4.7T outcome,** and is not adjusted after one.

## 8. The failure rules

- **No runner-up moves up.**
- **The window is spent:** whether the test passes, fails or is inconclusive, it is never rerun or extended.
- **v1 stays the product until CP-19.** It also stays if 4.7T passes, until CP-18's activation (§16.1).
- **A failed 4.7T** makes no policy the final product by this protocol. Any later candidate needs a new
  window and a new evaluation.
- **A null result is valid** (§24.3).

## 9. The §24.2 audit, before opening

Before 4.7T opens, the Lead and an independent reviewer check four things:
- **the download logs** of every retrieval that fed development: the snapshot pull of 2026-09-08, CP-15's,
  CP-20's and CP-25's, against their request bounds;
- **the boundary guard's tests** at 2026-09-06: `cp25.guard` and its tests, including the GFS-initialization
  rule;
- **the exposure ledger** (`docs/track-b/exposure-ledger.md`), with every entry's impact assessment;
- **the Owner's declaration** that his decisions used no information after 2026-09-06. That covers the tier
  ruling, the CP-28 decision, the margin rulings and CP-17's decisions 8 to 12.

**The test** (§24.2): no outcome after 2026-09-06 entered any fit, tuning, selection, registration or recorded
Owner decision. Incidental exposures block 4.7T only if they could have informed a registered choice.

If the standard is not met, 4.7T returns BLOCKED.

CP-17 recorded no exposure.

## 10. The retrieval plan: prepared, not run

| Source | What | Date bounds | Estimated bytes |
|---|---|---|---|
| SMARD `chart_data` (the snapshot's route, CC BY 4.0) | day-ahead price (filter 4169), load forecast (411), VRE forecast and actual components | delivery days 2026-09-07 to 2026-12-05 | about 1–5 MB of JSON; the snapshot holds 7.7 years in 3.1 MB |
| NOAA GFS on AWS (CP-25's route; NCAR d084001 as the one same-product alternate) | u10, v10, u100, v100, dswrf; leads f021–f048 every 3 h; byte-range reads | runs initialized 2026-09-06 00 UTC to 2026-12-04 00 UTC, one per delivery day (run = D − 1, 00 UTC) | 90 runs × 50 messages × about 0.93 MB ≈ **4.2 GB** read; retained as the three derived aggregates |

- **Every downloader refuses rows outside the bounds,** in code.
- **The guard moves from 2026-09-06 to 2026-12-05 only at the gate** (§24.2), after the audit and the Owner's
  declaration. CP-17 does not move it.
- **Every retrieval is logged** with its request bounds, sizes and hashes, for the audit.

## 11. The resource and calendar plan

- **Calendar:**
  - the audit and the Owner's declaration come first;
  - on or after 2026-12-04, once the price for 2026-12-05 is published, the retrieval runs;
  - then the guard moves, and the 90 daily product issuances run in origin order;
  - then the comparators run, then the scoring, then the analysis (§5–7);
  - then the report.
- **The product run under the issuing host:**
  - measured at CP-17: about 70 s per A1 fit (146 s at an origin whose ensemble has a member at the
    1,000-epoch cap), about 55 s per v4 fit and about 45 s per v2 fit;
  - with the proposed shadow issuances, about 3 minutes per origin, so about **4.5 machine-hours** for 90
    origins on one worker.
- **The comparators natively:** about 45 s per origin for v5's members and D2, about 1.2 machine-hours.
- **The proposed ceiling is 20 machine-hours,** CPU only: at most 4 workers, BLAS at one thread, peak memory
  12 GB, no cloud.
- **Daily operation (CP-18):** about 3 minutes on the issuing host per delivery day, within the success
  definition's 15-minute ceiling, issued at 11:00 local, before the 12:00 gate. CP-18 fixes the deadline and
  the late-forecast rule (§24.4).

## 12. PUBLISH_RULES §14, the "before final-candidate test" items, proposed for the Owner

**The badge:** "One-shot fresh-data test (4.7T) · delivery days 2026-09-07 to 2026-12-05 · 90 days ·
registered 2026-10-28 before the data existed". Its evidence class is PUBLISH_RULES §3.1's one-shot test.

**The wording:**
- "A1 was frozen and its test registered before any of these 90 days existed. It was tested once, against
  the control (v5) and the similar-day naive."
- On a pass: "It passed the pre-registered test (compound p ≤ 0.05)."
- On a fail or inconclusive result: "It did not pass; the window is spent, and v1 remains the product."
- Development results stay labelled "development, post-selection". They are never pooled with 4.7T.
- v1's holdout keeps its own label, "confirmatory-style, not power-qualified".

**The fresh headline window:**
- Until 4.7T is fully scored, the headline shows development, post-selection results with their badge.
- After 4.7T is scored, the headline's test numbers are 4.7T's window and nothing else, with the badge
  above.
- The prospective record stays separate, "evaluation in progress" until CP-19.

**The hero-switch rule:**
- The hero keeps PUBLISH_RULES' priority: live, final candidate, released, research.
- A1 appears as "final candidate" only while it is the frozen, designated candidate. Before the result it is
  labelled "pending 4.7T".
- It switches to "live" only at CP-18's activation, which requires a passed 4.7T.
- On a failed or inconclusive 4.7T, the hero keeps v1 as the released product and shows A1's result as a
  tested candidate that did not pass.
