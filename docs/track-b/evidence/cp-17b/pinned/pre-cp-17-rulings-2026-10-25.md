# The Owner's decisions before CP-17 — 2026-10-25

**Authority.** The Owner, 2026-10-25, on the Orchestrator's list of decisions before CP-17 (programme
plan, "Before CP-17"): "אני מקבל את כל ההחלטות שלך". He accepted every recommendation below.

CP-28 selected A1, DDNN-2 alone with its own Johnson SU quantiles
([CP-28 landing record](cp-28-landing-2026-10-25.md)). Research v21-r17 §24.9 freezes it, together
with terms registered before 4.7T opens. These decisions set those terms. The comparators, pass
conditions and failure rules enter the anchor by amendment, under the Owner's suspension, before
4.7T opens. Until then they bind the programme as decisions.

1. **The candidate is A1, DDNN-2 alone,** as CP-28's pre-registered rule selected it. v1 stays the
   product until CP-19.
2. **When GFS is late or missing at the issue time,** the fallback is v2, which uses no weather.
   - **The measured options** (CP-26 and r17's note, measured on v5):
     - a route trained on 6-hour-older GFS runs, about 14–16% worse than on time;
     - v2, which on the affected dates is within about 1% of the delayed route.
   - **The rule is fixed only after a check.** Before it is fixed, CP-17 checks the same comparison
     for A1 itself, descriptively, from CP-26's stored forecasts and without training.
   - **No new switching experiment.** The fixed chain for a missing member (v5, then B0) stays.
3. **The lock-price idea is not adopted in the freeze.** 4.7T stays a pure forecast-accuracy test.
   The idea is registered separately for the live stage, where quotes can be collected
   (`docs/track-b/final-product-lock-price-notes-2026-10-11.md`).
4. **4.7T keeps its 90-day window,** 2026-09-07 to 2026-12-05. It opens no earlier than 2026-12-04.
5. **4.7T's terms mirror CP-28's gates:**
   - **A1 against the control,** nominal v5: an improvement of more than 0.5% on both MAE and WIS;
   - **A1 against B0:** better on both;
   - **descriptive references:** v3, v4 and v1's released artifact;
   - **the dependence treatment:** 7-day blocks;
   - **on failure,** no runner-up moves up, the window is spent, and v1 stays the product.

   The exact endpoints and tests are written in the amendment.
6. **CP-26's and CP-28's results are published (PRES-5)** while CP-17 is prepared, labelled
   "development, post-selection, pending 4.7T", under PUBLISH_RULES 1.5's invariant 10.
   - **Still open:** the page has 3,733 bytes left under its 2.0 MB budget (A-PRES4-5), so PRES-5
     needs a compaction or a budget change first.
   - **The Orchestrator's recommendation is a compaction,** which stays within PUBLISH_RULES 1.5. A
     budget change would amend a publication anchor.
7. **CP-18's preparation starts alongside:** security hardening, the daily pipeline and the
   final-product Space, with dry runs on data to 2026-09-06 only. Nothing downloads or reads data
   after the red line before 4.7T opens.
8. **The percentage measure's tolerance, τ = 10 EUR/MWh** (§19.5, which leaves the tolerance to the
   Owner). The Owner, 2026-10-25: "τ = 10, תכתוב את הבריף ל־CP-17". The measure is the share of
   scored hours whose p50 lies within ±10 EUR/MWh of the published price. It was set before any 4.7T
   outcome exists, and is never tuned on one.
9. **The equality tolerance, τ_eq = 0.01 EUR/MWh** (§19.3 item 2, which leaves it to the Owner).
   The Owner, 2026-10-28, on CP-17-probe's measurement
   ([landing record](cp-17-probe-landing-2026-10-27.md)): "τ_eq = 0.01".
   - **The status a visitor's retrain reports:**
     - "identical bit for bit" when every emitted quantile is IEEE-754 equal to the issued one;
     - "within ±0.01 EUR/MWh" when every data-driven selection matched and every emitted quantile
       is within τ_eq;
     - "different" otherwise, with the numbers.
   - **The measurement it rests on:** the largest deviation was 6.3e-7 EUR/MWh, from rounding
     only, with every selection matched.
10. **The official daily fit is issued under the pinned Pyodide runtime in Node.js,** the same
    WebAssembly runtime the browser runs (Space plan §8). The Owner, 2026-10-28: "Node מאושר".
    - **Why:** Chromium, WebKit and Node.js gave bit-identical results on every probed fit, so a
      visitor's retrain can report "identical bit for bit".
    - **The cost:** about 70 s per A1 fit instead of 15 s natively.
    - **The issued values differ from CP-26's native arithmetic by rounding.** The freeze records
      this, and keeps the native parity with CP-26 as the check of code identity.
11. **When GFS is late or missing at the issue time, the product issues v2,** then B0.
    - **The check.** CP-17's item 1 (`reports/cp17/gfs-check.md`, on `gauntlet/cp-17` at `2d570e0`)
      met the stop rule of decision 2 above. On CP-25's 1,354 paired GFS +6 h affected dates, v2 is
      +2.27% on MAE and +6.68% on WIS against A1's GFS +6 h route.
    - **The Owner kept v2 on 2026-10-28,** approving the Orchestrator's recommendation ("מאושר"). The
      excess is disclosed.
      - v2 needs neither weather nor DDNN-2.
      - The measured route is a separately searched DDNN-2. Adopting it would add a second daily
        model.
      - A1 with weather missing was not measured.
12. **When A1's own daily fit fails and the weather is available, the product issues v4,** then v2,
    then B0. The Owner, 2026-10-28: "מאושר", on his own proposal to split the two causes.
    - **The reason.** v5 contains DDNN-2, so CP-28's chain for a missing member (v5, then B0) ended
      at B0 whenever D2 failed.
    - **What happened on the 18 development dates** between 2022-12-04 and 2023-06-20 when a D2
      member failed to train (`nonfinite_loss`):

      | | MAE | WIS |
      |---|---:|---:|
      | B0, the forecast issued | 58.56 | 33.63 |
      | v2 | 27.69 | 16.36 |
      | v4 | 26.90 | 15.33 |

      v4 was native on all 18.
    - **v4 is v5 without DDNN-2,** and the best policy without it on every eligible date (MAE
      18.58, against v2's 22.58). It uses weather, so it is not the GFS-late route.
    - **The product's chain replaces decision 2's** "v5, then B0" for the product. v5 stays the
      control in 4.7T. A1 issues whenever its own fit succeeds. The validity of other policies'
      members does not trigger a fallback; CP-28's common-member check was a comparison device.
    - **v4 joins the one code path** (§19.2). If it cannot run under Pyodide within a visitor's
      limits, CP-17 reports it under §19.3's exception route, for the Owner.
