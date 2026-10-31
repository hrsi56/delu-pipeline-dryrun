# 4.7T's frozen protocol, as CP-17b revises it: partial days and shadow issuances

**Status: registered at CP-17b, outcome-blind.**
- No delivery day after 2026-09-06 exists on this machine, or has been downloaded, read or inferred.
- This revision was written without any 4.7T outcome. It produces no 4.7T forecast.

**What this document is.** CP-17 registered 4.7T's protocol in
[`../cp-17/4-7t-frozen-protocol.md`](../cp-17/4-7t-frozen-protocol.md), SHA-256
`337a102d6bbbf6d9335fe6ed701a4c80ea1b9172344ffc0f63909b155b567c58`. That file stays as committed.
- **This revision changes only what the Owner's decisions 13 and 14 touch:** how partial days and shadow
  issuances enter the generation of the window's forecasts and their counting.
- **Everything else stands exactly as CP-17 froze it, by that hash:** §1 (the candidate, the window and the one
  look), §2 (the comparators), §4's scores and their aggregation, §5 (the endpoints and the decision), §6 (the
  dependence treatment), §8 (the failure rules), §9 (the §24.2 audit), §10 (the retrieval plan) and §12 (PUBLISH_RULES
  §14's items).
- **The sections below replace CP-17's §3 and §11, and add to §4 and §7.** Where a sentence of CP-17's §3 or §11 is
  not restated here, it is replaced.

**Authority:**
- `capstone_v21.md` v21-r17, §24.9's "registered before 4.7T opens" list, as CP-17's protocol cites it;
- the Owner's decisions 13 (C2, partial days) and 14 (G1, shadow issuances), and 15 to 18
  (`docs/track-b/cp-17-owner-decisions-2026-10-28.md`, `4d049770…`; frozen copy in [`pinned/`](pinned/));
- the brief `b9aeeee4…`, item 7.

**r18 ratifies these terms before 4.7T opens,** with CP-17b's freeze identity (decision 19).

**Executable form, unchanged:** `src/cp17/tools/fresh_test.py`. The boundary guard still refuses the real window
until the gate moves it (`tests/cp17/test_fresh_test.py::test_the_real_window_is_refused_until_the_gate`, and
`tests/cp17b/test_records.py`).

## 3. How the window's forecasts are generated (replaces CP-17's §3)

**The product policy.**
- It is the frozen package and its issuance driver, as CP-17b's freeze manifest identifies them
  ([`freeze-manifest.json`](freeze-manifest.json)): `src/cp17/` by its file hashes, and `cp17.daily`.
- It runs under the issuing host: Pyodide 0.28.0 in Node.js v26.7.0 (decision 10), one worker.
- It runs daily, at each origin D−1 11:00 Europe/Berlin (§24.4), from data available strictly before that instant
  (the tier register's vintages and availability).
- Each day is fitted from its own training window, in origin order.
- **Each attempt** of the frozen chain runs in its own fresh issuing-host process:
  - with a fresh output path and a 900 s timeout;
  - a killed, timed-out or non-zero process, or a missing, unreadable or foreign record, is a failed fit, and the
    chain moves on;
  - its artifact is written and verified inside the chain, before the chain moves on.
- **The issued artifact is written before D's outcome is opened.** A failed issuance is written too.

**Partial days (decision 13).**
- **The forecastable hours of D** are the frozen design's: a row of D is a forecast key when its inputs are valid
  (`cp15.data.Inputs.rows`; `cp24.member.Keys.from_data`).
- **A fit that emits seven finite, ordered quantiles on every forecastable hour** succeeds on those hours.
  - It is issued and displayed as partial, with the missing hours named.
  - This holds for A1 and for every fallback alike.
- **A day with no forecastable hour** is a failed issuance for every policy, B0 included. It is displayed as
  "failed issuance (no hour of the day is forecastable by the frozen design)".
- **In the window:** 2026-10-25 is the autumn clock change (25 hours). Days after a clock change can have an
  unforecastable hour, as 43 of CP-26's 2,078 scored development days had incomplete keys (decision 13's reason).
  These are calendar facts, not outcomes.

**The shadow issuances (decision 14).**
- **Every window day, whether or not A1 succeeds,** v4, v2 and B0 are fitted under the issuing host as shadow
  issuances. Each runs in its own process, as an attempt does.
- **A shadow is never published, displayed or scored as the product's forecast.**
- **Its central and scale are registered before the outcome.** Each policy's own errors from complete delivery
  days enter its own buffer once, at the first origin after D−2's release.
- **CP-26's intake rule applies.** Only a complete canonical issuance (23, 24 or 25 keys) with a finite central and
  a valid scale enters a buffer. So a partial day's shadow enters none, and a failed shadow enters none.
- **A fallback the chain fitted is the same fit as its shadow.** It is fitted once.
- **On a day whose GFS is late or missing at the issue time,** v4's shadow is skipped: its inputs are unavailable.
- **B0 is shadow-issued too.** Its buffer, rebuilt from published prices alone, does not equal CP-26's
  (`reports/cp17b/b0-rebuild.json`). It costs a few seconds.
- **The states start from CP-17's freeze-origin state:** CP-26's final replay state for v4, v2 and B0 at
  2026-09-06, as CP-17's freeze manifest records it by digest and CP-17b's carries it.
  - So the first window origin, 2026-09-07, releases CP-26's registration of 2026-09-05, and the second releases
    2026-09-06's.
  - From 2026-09-09 on, the buffers take the window's own shadow errors.
- **A fallback that issues in the window** issues from its shadow-kept state.

**The comparators** are unchanged (CP-17's §2): natively, with BLAS at one thread, under CP-26's rules, their
layers continued from CP-26's final state. The descriptive v4 comparator keeps its own native state. It is not the
product's v4 shadow, whose errors come from the issuing host and differ from native by rounding.

**Vintages:** per the tier register (`reports/development-inputs/tier-register.json`) and the Owner's tier, margin and
GFS rulings, which are unchanged.

**Counting.** Each window day is counted once, in exactly one of these classes, and stays in the record:
- **A1 issued:** complete, or partial with its emitted and missing hours;
- **a fallback issued:** by cause (A1's fit failed; GFS late or missing), by policy, and complete or partial;
- **a failed issuance:** by cause (no forecastable hour; every policy in the chain failed).

The shadows are counted apart from the product. For each day and policy, the record holds the shadow's status, its
emitted keys and its intake. A shadow never enters the product's count or score.

**No DDNN-2 search falls in the window.** The next search is S_14 = 2027-01-06, a protocol date. A1 uses S_13's
frozen ensemble throughout.

## 4. The primary scores (CP-17's §4 stands; this is added)

- **A partial product day** is scored on the hours it emitted. Its paired scored keys are CP-17's: CP-26's keys,
  intersected with the hours both policies emitted. So an hour the product did not emit is scored for neither side
  of that pairing, and the date's loss is the mean over its remaining scored keys.
- **A partial comparator day** is treated the same way.
- **A failed issuance,** including a day with no forecastable hour, is missing in every pairing that involves it,
  as CP-17's §4 says.

## 7. The minimum scorable support (CP-17's §7 stands; this is added)

- **A partial date counts** as a paired and scored date when its pairing keeps at least one scored key.
- **A failed issuance does not count,** whatever its cause.
- The rule of at least 60 of the 90 dates per comparison, and its reason, are unchanged.

## 11. The resource and calendar plan (replaces CP-17's §11)

**The calendar is unchanged:**
1. the audit and the Owner's declaration;
2. on or after 2026-12-04, once the price for 2026-12-05 is published, the retrieval;
3. the guard's move;
4. the 90 daily product issuances in origin order, each with its shadows;
5. the comparators, the scoring and the analysis (§5–7);
6. the report.

**The product run under the issuing host, measured at CP-17b** (`reports/cp17b/shadow-run.json`):
- A1 plus the three shadows on 14 consecutive development days took **255 s** of process time per delivery
  day (largest **288 s**), within the 15-minute ceiling on every day.
- That run had two days in flight at once, among other jobs, so these figures are conservative.
- So about **6.4 machine-hours** for the 90 origins on one worker.

**The comparators natively:** about 45 s per origin for v5's members and D2, about 1.2 machine-hours (unchanged).

**The proposed ceiling stays 20 machine-hours,** CPU only: at most 4 workers, BLAS at one thread, peak memory 12 GB,
no cloud.

**Daily operation (CP-18):** about 4.2 minutes on the issuing host per delivery day, with the shadows,
within the success definition's ceiling. It is issued at 11:00 local, before the 12:00 gate. CP-18 fixes the deadline
and the late-forecast rule (§24.4).
