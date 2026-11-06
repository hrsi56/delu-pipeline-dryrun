# CP-17: the freeze of A1 as a complete daily policy on one code path

- **Evidence class.** Everything here is `development_post_selection` and descriptive, on development data to
  2026-09-06 only.
- **No 4.7T forecast.** None exists, and none was produced.
- **The frozen identities** are in [`freeze-manifest.json`](../../docs/track-b/evidence/cp-17/freeze-manifest.json)
  and the [freeze record](../../docs/track-b/evidence/cp-17/freeze-record.md).
- **The tables below are generated** from this folder's records by `cp17.tools.report`.

## What CP-17 established

1. **The GFS check (item 1)** met its stop rule. The Owner then ruled:
   - **decision 11:** GFS late or missing → v2 → B0, with the excess disclosed;
   - **decision 12:** A1's own fit failed while the weather is available → v4 → v2 → B0.

   The check: on the 1,354 paired GFS +6 h affected dates, v2 is +2.27% (MAE) and +6.68% (WIS) against A1's
   GFS +6 h route. See [`gfs-check.md`](gfs-check.md).
2. **One code path (item 2).** `src/cp17/` is one package of 33 files that runs unchanged natively, under the
   issuing host, and in Chromium and WebKit. It fits A1, v4, v2 and B0 from their shipped payloads (and v5,
   the 4.7T control, natively).
   - Its imports are the standard library, NumPy, pandas, scikit-learn and LightGBM through the landed model
     modules, plus two import-only stand-ins that fail if reached.
   - No component needs §19.3's exception route.
3. **Native parity (item 3)** is bitwise for every policy at every origin, against CP-26's stored nominal
   forecasts:
   - five policies at nine origins: the last seven development origins and the two DST days;
   - every member's central and every recorded selection;
   - the interval layers' buffers, rebuilt from the ledger and checked against CP-26's own consumption hashes.
4. **Issuing-host parity (item 3).** Every issuing-host fit matched every data-driven selection of the native
   fit: A1 at nine origins, v4 and v2 at two. Every one is within τ_eq of the native fit except A1 at the
   25-hour DST origin 2025-10-26, which **misses τ_eq**.
   - **The miss:** its largest deviation is 0.028 EUR/MWh at the p50, so the status is "different".
   - **The cause** is one configuration, `cp26-k11-t032`, whose two members both train to the 1,000-epoch cap
     with no early stop.
     - Member 5 (seed 27190032), whose best epoch is the last, accumulates rounding over that long run to
       0.77 EUR/MWh in its own quantiles. Member 4 (seed 27190031) reaches 3.6e-05.
     - The other six members agree to about 1e-12, and the eight-member median carries 0.028 of it.
   - **Every selection is identical:** epochs run, best epoch and stop reason.
   - **This is reported, not tuned** (brief item 3).
   - **What it bounds, and what it does not.**
     - It bounds rounding between native CPython and WebAssembly.
     - It does not affect a visitor's retrain: the visitor's result is compared with the issued artifact,
       which the same WebAssembly runtime produced, and that comparison is bit for bit (item 5 below).
   - **Elsewhere,** the deviations come from rounding only, in the LEAR solver and in DDNN-2's arithmetic.
     LightGBM's members are bit-identical across runtimes.
5. **Browser identity (item 3).** A1 retrained in Chromium and in WebKit is bit-identical to the issuing
   host's artifact. v4, retrained in both browsers, is bit-identical to the issuing host's v4 as well.
6. **The failure policy** runs on the issuing host, on real payloads, taking exactly the path it takes
   natively:
   - the freeze origin's own A1 issuance;
   - a GFS-late issuance (v2);
   - a real A1 training failure from development, 2022-12-04, where v4 issues.

   **Each attempt runs in its own fresh issuing-host process,** so its wall time and peak memory are its own.
   The success definition's resource ceiling is judged on each attempt before the chain moves on: a fit over
   15 minutes or 4 GB is never issued (`tests/cp17/test_units.py`).
7. **§19.5's records** (item 5) recompute under the issuing host bit for bit. §19.3's status rule and its
   negative controls (item 6) behave as frozen. The reference bundle (item 7) validates, and every tampering
   control is refused.

<!-- generated tables: begin -->

### Native parity (the code-identity check)

| Origin (delivery) | A1 | v5 | v4 | v2 | B0 | A1 fit (s) | v4 fit (s) |
|---|---|---|---|---|---|---:|---:|
| 2025-10-26 | bitwise | bitwise | bitwise | bitwise | bitwise | 28.4 | 25.7 |
| 2026-03-29 | bitwise | bitwise | bitwise | bitwise | bitwise | 15.7 | 29.8 |
| 2026-08-31 | bitwise | bitwise | bitwise | bitwise | bitwise | 16.3 | 31.9 |
| 2026-09-01 | bitwise | bitwise | bitwise | bitwise | bitwise | 15.0 | 29.9 |
| 2026-09-02 | bitwise | bitwise | bitwise | bitwise | bitwise | 13.7 | 31.7 |
| 2026-09-03 | bitwise | bitwise | bitwise | bitwise | bitwise | 14.9 | 32.2 |
| 2026-09-04 | bitwise | bitwise | bitwise | bitwise | bitwise | 14.0 | 30.9 |
| 2026-09-05 | bitwise | bitwise | bitwise | bitwise | bitwise | 17.1 | 30.3 |
| 2026-09-06 | bitwise | bitwise | bitwise | bitwise | bitwise | 14.8 | 30.5 |

Status PASS; BLAS threads ['1']; interval-state days checked against CP-26's consumption trace: {'B0': 428, 'v2': 428, 'v4': 428, 'v5': 428}.

### Issuing-host parity (Pyodide 0.28.0 in Node.js)

| Run | Status vs native | Largest deviation (EUR/MWh) | Selections matched | Fit (s) | Native fit (s) | Wasm heap (MiB) | Peak RSS (MiB) |
|---|---|---:|---|---:|---:|---:|---:|
| A1-2025-10-26 | different | 2.8e-02 | yes | 148.6 | 28.4 | 187 | 611 |
| A1-2026-03-29 | within ±0.01 EUR/MWh | 6.3e-07 | yes | 73.4 | 15.7 | 189 | 631 |
| A1-2026-08-31 | within ±0.01 EUR/MWh | 9.4e-09 | yes | 72.6 | 16.3 | 189 | 782 |
| A1-2026-09-01 | within ±0.01 EUR/MWh | 8.2e-10 | yes | 67.9 | 15.0 | 189 | 803 |
| A1-2026-09-02 | within ±0.01 EUR/MWh | 1.3e-11 | yes | 61.0 | 13.7 | 189 | 770 |
| A1-2026-09-03 | within ±0.01 EUR/MWh | 1.6e-09 | yes | 65.6 | 14.9 | 189 | 771 |
| A1-2026-09-04 | within ±0.01 EUR/MWh | 3.0e-11 | yes | 63.5 | 14.0 | 189 | 802 |
| A1-2026-09-05 | within ±0.01 EUR/MWh | 7.5e-10 | yes | 78.2 | 17.1 | 187 | 765 |
| A1-2026-09-06 | within ±0.01 EUR/MWh | 9.4e-13 | yes | 65.6 | 14.8 | 225 | 814 |
| v4-2026-03-29 | within ±0.01 EUR/MWh | 3.3e-13 | yes | 52.5 | 29.8 | 323 | 883 |
| v4-2026-09-06 | within ±0.01 EUR/MWh | 4.3e-13 | yes | 53.9 | 30.5 | 274 | 877 |
| v2-2026-03-29 | within ±0.01 EUR/MWh | 5.4e-13 | yes | 35.9 | 20.1 | 274 | 860 |
| v2-2026-09-06 | within ±0.01 EUR/MWh | 5.1e-13 | yes | 44.7 | 24.9 | 274 | 856 |
| B0-2026-03-29 | identical bit for bit | 0 | yes | 0.0 | 0.0 | 187 | 735 |
| B0-2026-09-06 | identical bit for bit | 0 | yes | 0.0 | 0.0 | 189 | 796 |

| Run | Member deviations, issuing host vs native (EUR/MWh) |
|---|---|
| A1-2025-10-26 | largest: member 5 (cp26-k11-t032, seed 27190032, 1000 epochs, max_epochs) 7.7e-01; the other seven at most 3.6e-05 |
| A1-2026-03-29 | largest: member 6 (cp26-k12-t096, seed 27290041, 338 epochs, patience) 1.8e-06; the other seven at most 6.0e-07 |
| A1-2026-08-31 | largest: member 3 (cp26-k13-t094, seed 27390022, 262 epochs, patience) 3.8e-08; the other seven at most 1.6e-12 |
| A1-2026-09-01 | largest: member 2 (cp26-k13-t094, seed 27390021, 210 epochs, patience) 1.9e-09; the other seven at most 2.3e-12 |
| A1-2026-09-02 | largest: member 2 (cp26-k13-t094, seed 27390021, 177 epochs, patience) 2.5e-11; the other seven at most 2.8e-13 |
| A1-2026-09-03 | largest: member 2 (cp26-k13-t094, seed 27390021, 226 epochs, patience) 3.7e-09; the other seven at most 1.7e-12 |
| A1-2026-09-04 | largest: member 3 (cp26-k13-t094, seed 27390022, 127 epochs, patience) 2.4e-10; the other seven at most 1.5e-11 |
| A1-2026-09-05 | largest: member 3 (cp26-k13-t094, seed 27390022, 230 epochs, patience) 1.6e-08; the other seven at most 4.7e-10 |
| A1-2026-09-06 | largest: member 2 (cp26-k13-t094, seed 27390021, 147 epochs, patience) 1.5e-11; the other seven at most 8.5e-13 |
| v4-2026-03-29 | A1_w 5.4e-13, B2_w 1.0e-12, L-N 0, L-R 0 |
| v4-2026-09-06 | A1_w 9.0e-13, B2_w 5.1e-13, L-N 0, L-R 0 |
| v2-2026-03-29 | A1 1.1e-12, B2 1.1e-12 |
| v2-2026-09-06 | A1 3.0e-13, B2 1.1e-12 |

Issuing host: Node.js v26.7.0 (SHA-256 1ef99ea25fe7…), Pyodide 0.28.0; τ_eq = 0.01 EUR/MWh. Non-local requests: 0. Executed files equal the package: yes. Attribution was computed at 2026-09-06 and 2026-03-29.

### Browser identity and v4 in the browser

| Browser | Run | vs the issuing host | vs native (largest, EUR/MWh) | Selections matched | Fit (s) | Wasm heap (MiB) | Browser RSS peak (MiB) | Training payload (MB, gzip) | First visit (KB) |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| chromium 153.0.8010.12 | A1-2026-09-06 | identical bit for bit | within ±0.01 EUR/MWh (9.4e-13) | yes | 65.6 | 215 | 1,659 | 45.6 (38.2) | 3.4 |
| chromium 153.0.8010.12 | v4-2026-09-06 | identical bit for bit | within ±0.01 EUR/MWh (4.3e-13) | yes | 54.2 | 274 | 1,746 | 48.6 (40.8) | 3.4 |
| webkit 26.6 | A1-2026-09-06 | identical bit for bit | within ±0.01 EUR/MWh (9.4e-13) | yes | 62.0 | 215 | 1,374 | 45.6 (38.2) | 3.4 |
| webkit 26.6 | v4-2026-09-06 | identical bit for bit | within ±0.01 EUR/MWh (4.3e-13) | yes | 57.9 | 309 | 1,481 | 48.6 (40.8) | 3.4 |

### What the browser loaded

| Browser | Run | Runtime (MB) | Packages (MB) | Code (KB) | Data (MB) | Ready to fit (s) | Attribution (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| chromium | A1-2026-09-06 | 12.3 | 32.7 | 86.7 | 0.5 | 2.1 | 0.3 |
| chromium | v4-2026-09-06 | 12.3 | 32.7 | 86.7 | 3.5 | 2.1 | 14.9 |
| webkit | A1-2026-09-06 | 12.3 | 32.7 | 86.7 | 0.5 | 1.8 | 0.3 |
| webkit | v4-2026-09-06 | 12.3 | 32.7 | 86.7 | 3.5 | 1.8 | 16.1 |

### The failure policy, exercised under the issuing host

| Case | Path (issuing host) | Same path natively | Display | Issued vs native | Wall (s) | Peak (MiB) |
|---|---|---|---|---|---:|---:|
| 2026-09-06, GFS available | A1 ok | yes | successful daily fit | within ±0.01 EUR/MWh (9.4e-13) | 69.8 | 802 |
| 2026-09-06, GFS declared unavailable | A1 skipped → v2 ok | yes | skipped fit (GFS late or missing); fallback v2 issued | within ±0.01 EUR/MWh (5.1e-13) | 61.0 | 868 |
| 2022-12-04, GFS available | A1 failed → v4 ok | yes | failed fit; fallback v4 issued | within ±0.01 EUR/MWh (5.1e-13) | 62.0 | 887 |

### Retrain negative controls

| Control | Change | Status | Quantiles and selections alone | Largest deviation (EUR/MWh) | Expected |
|---|---|---|---|---:|---|
| Changed seed | member 0 seed 27390011 -> 27390012 | different | different | 2.4e+01 | different |
| Changed training-window value | price of 2026-08-20 slot 12 +1 EUR/MWh | different | within ±0.01 EUR/MWh | 9.1e-03 | different |
| Changed target-day outcome | every price of 2026-09-06 +50 EUR/MWh | payloads unchanged: A1, v2, v4, B0 | — | — | unchanged |

### §19.5 records, native against the issuing host

| Window / stream | Days | Hours | Partial days | Negative-price hours | Items compared | Native = issuing host |
|---|---:|---:|---:|---:|---:|---|
| autumn-2025/A1 | 61 | 1,462 | 3 | 51 | 2,536 | bit for bit |
| autumn-2025/B0 | 61 | 1,462 | 3 | 51 | 2,536 | bit for bit |
| spring-summer-2026/A1 | 190 | 4,556 | 3 | 438 | 7,696 | bit for bit |
| spring-summer-2026/B0 | 190 | 4,556 | 3 | 438 | 7,696 | bit for bit |

<!-- generated tables: end -->

## Reading the tables

- **"within ±0.01 EUR/MWh"** is the Owner's status for a retrain that matched every data-driven selection and
  stayed within τ_eq of the issued artifact (decision 9). It also requires the retrain to have run on the issued
  fit's own training window: a retrain from any other window is "different" (the second control). "identical bit for bit" is stricter: it requires
  every emitted quantile to be IEEE-754 identical.
- **Issuing host against native.** These deviations are between the issuing host and the native fit. They are
  not the visitor's check: a visitor's retrain is compared with the issued artifact, which the issuing host
  produced, and the browser identity rows show that comparison is bit for bit.
- **Timing.** Times are on one Apple M3 (8 cores, 16 GB, macOS 26.5.2), one worker per run, with at most two
  runs at once; the browser runs ran alone.
  - Each time includes attribution where it was computed (A1, v4 and v2 at 2026-09-06 and 2026-03-29).
  - The native fits ran with BLAS at one thread.
- **The issuances.** Their wall time and peak are the issued attempt's own process: the runtime load, the
  packages and the fit, with attribution.
- **Memory.**
  - "Wasm heap" is the WebAssembly linear memory, which only grows, so its final size is the peak.
  - "Peak RSS" is the Node.js process's maximum resident set.
  - "Browser RSS peak" is the summed resident memory of the browser's processes, sampled every 0.25 s.
- **The training payload** is what a visitor's browser loads on "Train it yourself": the runtime, the
  packages, the package code and the day's training window. The first visit is the page alone.

## §19.7's attributions, frozen

| Component class | Method | Checked |
|---|---|---|
| A1 (neural) | exact baseline Shapley over DDNN-2's input groups on the issued p50, each member's training-row mean as the baseline, every coalition (at most 2,048) | the values add up to the issued p50 minus the baseline in every hour; equal across runtimes within rounding |
| LEAR members (v2, v4) | coefficient × centered, standardized input, from the refitted final Lasso | the refit reproduces the recorded coefficient hash; contributions plus intercept equal the member's forecast |
| LightGBM members (v4) | TreeSHAP per prediction, and split gain, from the refitted booster | the refit reproduces the recorded tree hash; contributions plus expected value equal the member's forecast |
| Blends (v2, v4) | each component's weight × its forecast | the decomposition equals the central forecast within rounding |
| Interval layers (H, B0) | explained in words; no per-input attribution | — |

All of it is attribution within one fitted artifact, relative to its stated baseline, never causality or
incremental value (§19.7).

## The reference runtime, and the visitor wording proposed for the Owner

**The reference runtime:**
- Pyodide 0.28.0;
- Chromium 153.0.8010.12 and WebKit 26.6, through Playwright 1.63.0;
- Node.js v26.7.0;
- the Apple M3 machine class, 8 cores, 16 GB, macOS 26.5.2.

**Proposed wording** (for the Owner's later publication decision; not frozen here): "Retraining this day's
model downloads about the training payload shown above, once, and took about the measured fit time on an
Apple M3 laptop in Chromium and WebKit (measured 2026-10-28). Your result is compared with the
issued forecast: identical bit for bit, within ±0.01 EUR/MWh, or different, with the numbers. It never
replaces the issued forecast."

## The Owner's summary for the anchor amendment (r18)

- **The GFS check and the rulings:**
  - item 1's table;
  - decision 11 (GFS late → v2 → B0, with the excess disclosed);
  - decision 12 (A1 failed → v4 → v2 → B0).
- **The freeze identities:**
  - the policy identity and the freeze manifest (their SHA-256 values are in the return);
  - the package `ac45f033…`;
  - the issuing host: Pyodide 0.28.0 by its 23 file hashes, and Node.js v26.7.0 `1ef99ea2…`.
- **Parity:** the tables above.
- **The failure policy and the success definition:** freeze record §10.
- **The protocol's key choices:**
  - the endpoints and the compound test at α = 0.05;
  - 7-day blocks, 10,000 replicates, seed 20261204;
  - the minimum support of 60 paired days;
  - the retrieval plan, about 4.2 GB, prepared not run;
  - the proposed ceiling of 20 machine-hours;
  - PUBLISH_RULES §14's four items.
- **The open items** (freeze record §11):
  - **G1:** daily shadow issuances for the fallbacks' interval states;
  - **C1:** "labels ≤ D−2" read as released errors, with training targets published before the issue time;
  - **C2:** partial issuance on days the design cannot forecast in full;
  - **G2:** GFS missing in a training window.
- **§19.3 exceptions:** none needed.

## Compute, retrieval and the red line

- **Compute** is in [`compute-ledger.jsonl`](compute-ledger.jsonl), summed over workers' wall time, within the
  15-machine-hour ceiling.
- **Retrieval:** none. No data, software or remote byte was retrieved
  ([`retrieval-manifest.json`](retrieval-manifest.json)).
- **Network:** every Node.js and browser run refused and recorded non-local requests, and none was made.
- **The red line:**
  - no delivery day after 2026-09-06 exists on this machine, or was read;
  - the boundary guard stays at 2026-09-06;
  - no committed record carries a later date (`tests/cp17/test_records.py`). The 4.7T protocol names its
    future window as protocol dates only.

## Reproduction

- **Every record:** `src/cp17/tools/reproduce.sh <project root> <out>` from a checkout. It needs no network.
  It takes about 40 minutes on the reference machine, and uses the pinned runtimes and the `evidence/cp-26`
  extract in `.local/`.
- **Committed:** `python -m cp17.tools.records --run <out>` collects the records, and
  `python -m cp17.tools.report` renders these tables.
- **Tests:** `tests/cp17/`. The records tests and units are CI-safe. The real-input tests skip without the
  extract, and the host and browser tests are marked `stress` and skip without the runtimes.
