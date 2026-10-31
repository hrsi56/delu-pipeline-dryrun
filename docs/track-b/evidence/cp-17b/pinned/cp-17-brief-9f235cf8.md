# Track B Checkpoint Brief — CP-17 (the freeze: A1 as a complete daily policy on one code path, §19's freeze fields, and 4.7T's protocol written blind)

*Issued by the Orchestrator on 2026-10-28, Asia/Jerusalem, under the Owner's delegation of
2026-10-27. This is CP-17's second brief. The first (`64758101…`) was withdrawn on 2026-10-27 before
any repository work, because it left out §19's freeze fields. CP-17-probe then supplied §19.3's
measurement.*

*Amended on 2026-10-28, after item 1's stop. The second brief (`9e522b9a…`) stopped at item 1, as
its rule required, and returned BLOCKED. The Owner then ruled on the product's fallbacks (decisions
11 and 12). This amendment carries those rulings into items 1–4, adds v4 to the one code path and
the parity checks, and resumes CP-17 on `gauntlet/cp-17` from `2d570e0`. Nothing else changes.*

## Target
- Repository: DE-LU day-ahead forecasting, `/Users/djourno/Downloads/PJM` (origin
  `hrsi56/delu-day-ahead-forecast`). The checkpoint runs locally, in the Lead's own worktree.
- Authorized checkpoint: CP-17, exactly one.
- Ratified plan anchor: `capstone_v21.md`, revision v21-r17, SHA-256
  `3124451da84c3cd99fae728c371207aace6ae537474f7769d0ff6333ee53e4fc`. The bar's sections, by their
  identities on `main`:

  | Section | Lines | Bytes | Section SHA-256 |
  |---|---:|---:|---|
  | §24.9 "CP-17 and 4.7T" | 40 | 2,007 | `6675e907ccad916e02059b12479daec712f0e720efd4c4056e656b863027a0f1` |
  | §16.2 "What freezes and what updates every day" | 32 | 2,408 | `c54723eff2f113a18efaef5884c3c2a93876c755edacf3b2fb70dc3af44ac5ec` |
  | §19.2 "One code path" | 6 | 335 | `538342106768fedad8201f44bd4aadc3de02415cebf321cd0c1584eca75f6f7c` |
  | §19.3 "Train it yourself" is required | 25 | 1,354 | `5d1f869e153bac43b8f049ee8395b317f48a28e65282109eb107821ef28e582e` |
  | §19.5 "Definitions frozen at CP-17" | 19 | 1,170 | `1d47a50558cc7b9a9df1a657b7552a50df728457af6dd3dea8835e3403d7b521` |
  | §19.7 "Attribution and importance" | 12 | 494 | `10060bd2f7315143d9e0a3a6a5529df324b76b9d8a757c87605f17b79148821c` |

  - **These sections are governed by:**
    - §16.1 and §16.4's CP-17 row;
    - §19 as a whole, which binds the CP-17 and CP-18 briefs and which §24.1 keeps in force;
    - §24.2: the red line and the audit standard;
    - §24.3: 4.7T's window, 2026-09-07 to 2026-12-05;
    - §24.4: the issue time;
    - §24.5: the tiers and margins;
    - §24.8: CP-28's selection.
  - **The Owner's decisions before CP-17:** `docs/track-b/pre-cp-17-rulings-2026-10-25.md`, SHA-256
    `d29b0abc6a17dcb6bca2cbd464c1dcc5858c50e2b653a42e646d3dc66ea8f6b0`. In summary:
    1. the candidate is A1;
    2. the GFS fallback is v2, after this checkpoint's check;
    3. no lock-price policy in the freeze;
    4. 4.7T keeps 90 days;
    5. 4.7T's terms mirror CP-28's gates;
    6. PRES-5 runs alongside;
    7. CP-18's preparation runs alongside;
    8. τ = 10 EUR/MWh;
    9. **τ_eq = 0.01 EUR/MWh,** with the three-status rule;
    10. **the official daily fit is issued under the pinned Pyodide runtime in Node.js;**
    11. **GFS late or missing → v2, then B0,** with item 1's measured excess disclosed;
    12. **A1's own fit failed while the weather is available → v4, then v2, then B0.** A1 issues
        whenever its own fit succeeds.
  - **CP-28's selection:** `docs/track-b/cp-28-landing-2026-10-25.md`
    (`7c2eecf23bb3928fa08b0331c42345c4468f89c633cbd4afeae8bf113413f8d8`),
    `reports/cp28/results/selection.json`
    (`22a3dacfc06303a42db132ae3ba8ee8fc7e82606a886585fc40d02b82ae930b0`) and
    `reports/cp28/owner-summary.json`
    (`ea9edc50f9fa1c1d08199876f1c48a83ca516687f8be5054b5024d495d7a6e0d`).
  - **CP-17-probe's measurement and proposals:**
    - `docs/track-b/cp-17-probe-landing-2026-10-27.md`
      (`3071d60a876ab13087490e85b5e09077bfabf78f0385f0f7044fa5f93cb6c00d`);
    - `reports/cp17-probe/report.md` (`e53aad22a30de6edf98a11b76999101b53851742e7aa49c6a1057fb7de4d39a2`);
    - `reports/cp17-probe/retrieval-manifest.json`
      (`37bf451ff50af2699978cc408e2e5ea1da6539abb0a648ac6a11b2d909cfdfa2`);
    - `reports/cp17-probe/bundle-manifest.json`
      (`93e5054a953a022c7f158b38354d0b0e48e182b89145553719d308207b0198f7`);
    - `reports/cp17-probe/measurements.json`
      (`289dbeabe2b6168b500d9cf3ce55ef4746ecfbeb8c87476dc4203296ec36437f`).
  - **The Space plan:** `docs/track-b/final-product-space-plan-2026-09-30.md`
    (`ad3568c8997a08aa291bd3a7f29046c24ce61f0c1fd992622f921b5d23f9174c`): §4.4, §4.6–§4.8, §4.12,
    §7 step 3 and §8.
  - **The registry naming:** `docs/track-b/mlflow-tracking-spec.md`
    (`8912e2284a5dac0d6eef90b40a755237c3303841523fae3d10485b4ceb379114`). Names are local; nothing
    is uploaded.
  - **The publication anchor** is PUBLISH_RULES 1.5, SHA-256
    `1bc0c52351656c5a499291bf15ccb82b0ae8b4e6475d1879b0a4d307daaef94e`, for §14's "before
    final-candidate test" trigger, which you draft for the Owner. CP-17 publishes nothing.

## Orchestrator-reported expected state
- **Branch / commit:** `main` = `origin/main` at the commit that records this amendment.
  - It is a direct child of `be504ab` (the second brief's issue record), and changes only
    `progress.md` and the decisions record (adding items 11 and 12).
  - **Tags:**
    - `land/cp-17-probe` = `42e27c7` and `evidence/cp-17-probe` = `07da82a`, both pushed;
    - `land/cp-28` = `35e841b` and `evidence/cp-28` = `1a86c54`, both pushed;
    - `land/cp-26` = `61658c7`;
    - `evidence/cp-26` = `ca20331`, which is **local-only and never pushed**.
  - **Branches and worktrees:** `gauntlet/cp-17` at `2d570e06f0e6cd4a2a2950f2bc31ec3361377b2d`,
    a child of `be504ab` by two commits: the packaged second brief (`b990688`) and item 1
    (`2d570e0`). Its worktree is `.local/worktrees/cp-17/lead`, clean. There are no other branches or
    worktrees besides `main` and the primary checkout.
- **Working tree:** the primary checkout `/Users/djourno/Downloads/PJM` is on `main` and clean
  apart from ignored material.
  - **The Orchestrator uses it during CP-17.** Never switch its branch, and never write a tracked
    file in it. Your only writes there are under:
    - `.local/artifacts/cp-17/`;
    - `.local/tmp/cp-17/`;
    - the worktrees under `.local/worktrees/cp-17/`.
  - **The canonical brief** is `.local/artifacts/cp-17/issued-brief.md`, this amended brief.
    Package that file byte for byte, not the pasted text.
    - The second brief is kept as `.local/artifacts/cp-17/issued-brief-v1-9e522b9a.md`.
    - The withdrawn first brief is kept in `.local/artifacts/cp-17/withdrawn-64758101/`. It binds
      nothing.
  - **Your local material from item 1:** the verified `evidence/cp-26` extract in
    `.local/artifacts/cp-17/inputs/` (2.2 GB), `start.json`, `compute-ledger.jsonl` and
    `return-blocked.md`. Continue the same ledger.
  - **`.local/tmp/cp-17/`** already holds two section extracts from your validation of the first
    brief (`s195.txt`, `s249.txt`). They are yours to keep or delete.
- **What already exists:**
  - **A1 is a policy CP-26 already ran:** the nominal DDNN-2 member D2, refit daily at every
    development origin under §24.7, with DDNN-2's own Johnson SU quantiles. CP-28 evaluated it from
    CP-26's stored ledger.
    - **Its code:** the landed DDNN-2 implementation in `src/cp24/` (`land/cp-24`), and CP-26's daily
      refit harness in `src/cp26/`.
    - **CP-26's live identity guards refuse under r17 by design.** Never call them, and never edit
      `src/cp26/`.
  - **CP-17-probe's landed code,** `src/cp17probe/`: `dailyfit.py`, one module that runs the landed
    fit calls under native CPython and under Pyodide alike, with about 30 lines of restated glue
    proven by bitwise CP-26 parity; `payload.py`, `attribution.py`, `native.py`, `compare.py`,
    `measure.py`, `node_run.mjs`, the browser page and worker, and two import-only stand-ins
    (`holidays`, `resource`). `src/cp17/` may start from it. `src/cp17probe/` itself stays unchanged.
  - **CP-26's stored state,** read-only, bound by the hashes in CP-26's committed manifests:
    - in the local-only tag `evidence/cp-26`:
      - search results and plans for the `nominal`, `gfs6` and `noload` routes, k00–k13
        (`reports/rebaseline/cold-rerun-2026-10-15/search-results/`, `search-plans/`);
      - the replay state and the learning ledger;
      - the forecast ledger.
    - in `.local/artifacts/cp-26/cold-rerun-2026-10-15/` of the primary checkout: `fits/`,
      `searches/` and `state/`.

    Extract what you need into `.local/artifacts/cp-17/inputs/` with
    `git -C /Users/djourno/Downloads/PJM archive evidence/cp-26 <path> | tar -x -C <dir>`. Verify it,
    and never commit it. CP-17-probe's extract in `.local/artifacts/cp-17-probe/inputs/` may be
    reused read-only after you verify it against the tag.
  - **The pinned runtimes, already on this machine:**
    - Pyodide 0.28.0 in `.local/tools/pyodide-0.28.0/`, 23 files, as `retrieval-manifest.json`
      records them. Its lock file reports `0.28.0.dev0`; the runtime's identity is its file hashes;
    - Chromium 153.0.8010.12 (`.local/tools/ms-playwright/chromium-1243/`) and WebKit 26.6
      (`webkit-2359`), driven by Playwright 1.63.0 in `.local/tools/playwright/`;
    - Node.js v26.7.0 at `/opt/homebrew/Cellar/node/26.7.0/bin/node`, SHA-256
      `1ef99ea25fe70c9b67e7efe768ef8ee22148d3cabc703db6131b57aeb617d040`.
  - **CP-28's outputs** are in `reports/cp28/` and `src/cp28/`, on `main`.
  - **The development inputs:**
    - `data/snapshot.parquet`, to 2026-09-06;
    - CP-25's `reports/development-inputs/`: the calendar, the masks, the tier register and the margin
      table, with the GFS +6 h affected dates;
    - CP-20's committed weather features and their retained grids;
    - the boundary guard `src/cp25/guard.py`. **Its boundary stays at 2026-09-06 throughout CP-17.**
  - **The exposure ledger** is `docs/track-b/exposure-ledger.md`. It is append-only, and the
    Orchestrator keeps it; report any exposure in your return.
  - **The backup register** is locked; never edit it.
  - **The tools:** `scripts/gauntlet.py` and `scripts/bar.py`. Timing-sensitive tests carry the
    `stress` marker.
- Verify this yourself before relying on it, and report any material mismatch.

## Observable outcome
CP-17 closes with the following, bound by one fresh Integration-Critic PASS. Items 1 to 8 are built
in this order where one depends on another.

1. **The A1 GFS check: done, at `2d570e0`.** It is descriptive, uses CP-26's stored forecasts, and
   trains nothing.
   - **What it compared,** for MAE and WIS, with paired counts:
     - A1 on nominal weather;
     - A1's GFS +6 h route (D2 in CP-26's `gfs6` ledger);
     - v2 on nominal inputs.
   - **Where:** on CP-25's GFS +6 h affected dates, on the unaffected dates, and on every eligible
     date.
   - **The rule:** if v2 is more than 2% worse than A1's GFS +6 h route on the affected dates on
     *both* MAE and WIS, stop and return BLOCKED for the Owner's decision.
   - **The result:** the rule was met (+2.27% MAE, +6.68% WIS on 1,354 paired dates), and CP-17
     stopped as it required.
   - **The Owner's ruling** (decision 11): the GFS-late fallback stays v2, with the excess disclosed.
     The record stays as committed. The Integration Critic reviews it with the rest.
2. **One code path, `src/cp17/`** (§19.2). One browser-safe daily-fit package that the issuing
   pipeline imports and that the Space will run byte for byte.
   - **Its scope:** the daily fit, issuance and scoring of every policy that can issue the product's
     forecast under item 4's failure policy (A1, v4, v2 and B0), §19.5's metrics and the
     recomputation check, the §19.7 attributions, and the retrain comparison of item 6.
   - **Its imports:** the standard library, NumPy and pandas, plus the landed model modules it calls.
     The two stand-ins (`holidays`, `resource`) are frozen with the package if they are kept. Do not
     edit landed code to trim the import chain; the 44.6 MB training payload, loaded on request, is
     accepted.
   - **Its only input** is the shipped payload of item 7, with the embargo rule.
   - **A component that cannot run under Pyodide** is reported under §19.3's exception route, with
     the numbers, for the Owner. It is never replaced silently.
3. **The issuing host and its parity** (the Owner's decision 10).
   - **The issuing host** is Pyodide 0.28.0 under Node.js v26.7.0, pinned by the file hashes above.
     The freeze-origin artifact and every 4.7T forecast of the product policy are issued there.
   - **Native parity, the code-identity check.** The same package, run natively with BLAS at one
     thread, reproduces CP-26's stored nominal forecasts for A1, v5, v4, v2 and B0 at the last seven
     development origins (deliveries 2026-08-31 to 2026-09-06) and at two DST origins (deliveries
     2025-10-26 and 2026-03-29). The standard is CP-26's recorded parity: bitwise where CP-26 was
     bitwise, otherwise within its recorded tolerance.
   - **Issuing-host parity:**
     - A1 at all nine origins, and v4 and v2 at two of them, one a DST origin, under the issuing
       host;
     - every data-driven selection matches the native fit, and every emitted quantile is within
       τ_eq = 0.01 EUR/MWh of it;
     - record the largest deviation per origin. It is reported, never tuned.
   - **Browser identity:** A1 at one origin, retrained in Chromium and in WebKit, is bit-identical to
     the issuing host's artifact.
   - **v4 in the browser** (§19.3, for a fallback that can issue). v4 is retrained once, at one
     origin, in Chromium and in WebKit.
     - Record its time, memory, payload, deviation and selection agreement, as CP-17-probe recorded
       them for A1 and v2.
     - If v4 cannot run under Pyodide, or not within limits a visitor can meet, report it under §19.3's
       exception route with the numbers, for the Owner. Do not drop it from the chain yourself.
   - **A miss is a finding, not a failure to repair.** An origin that misses issuing-host parity or
     browser identity is reported with its numbers and selections, for the Owner and r18. Nothing is
     tuned to make it pass.
   - **A data check:** no input after 2026-09-06 exists or is read.
4. **The freeze record,** `docs/track-b/evidence/cp-17/freeze-record.md`, plus a machine-readable
   `freeze-manifest.json`.
   - **What it covers:** every item of §16.2's freeze list and §16.4's CP-17 row, for A1 and for each
     policy that 4.7T runs: the control v5, B0, the fallbacks v4 and v2, and v3 and v1's released
     artifact as descriptive references. v4 is also a descriptive reference.
   - **What it records:**
     - code and dependency identities: the package's file hashes, the `uv.lock` hash, and the
       issuing host's identity (the Pyodide file hashes and the Node.js binary);
     - feature and source-availability contracts: the issue time, D−1 11:00 Europe/Berlin (§24.4), and
       the tier register with the Owner's rulings (§24.5);
     - hyperparameters, including DDNN-2's nominal search state from S_13 = 2026-07-08, by hash;
     - training-window and label-delay rules, including §2's D−2 release;
     - the initial artifacts and state at the freeze origin, issued under the issuing host;
     - randomness and seeds;
     - the refit schedule: daily, with the next search at S_14 = 2027-01-06, a protocol date only;
     - the uncertainty layers: DDNN-2's own quantiles for A1, v4's and v2's interval layers as
       fallbacks, and v5's H layer for the control;
     - **the failure policy,** as the Owner ruled it (decisions 11 and 12):
       - **GFS late or missing at the issue time** → v2 → B0 → a failed issuance;
       - **A1's own fit fails** (the success definition below) **while the weather is available** →
         v4 → v2 → B0 → a failed issuance;
       - **a fallback whose own fit fails** passes to the next one in its chain;
       - A1 issues whenever its own fit succeeds. The validity of other policies' members does not
         trigger a fallback. CP-28's common-member check was a comparison device, and v5 is not in
         the product's chain;
       - every fallback is complete: never renormalize a subset or manufacture quantiles;
       - a failed or skipped fit is displayed as such, even when a fallback issues (§16.2);
       - the disclosure of item 1's measured excess for the GFS-late route;
       - **any gap you find,** for example a cause that the two rulings do not cover. Propose its
         rule with the reason, written without any 4.7T outcome, for r18;
     - **the definition of a successful daily fit,** exactly as this brief fixes it (below);
     - the registry identities, which are local names and versions under the tracking spec;
     - the resource and calendar plan for 4.7T's run and for daily operation.
   - **The Owner's designation:** A1 is designated the final version (§16.1, decision 1), pending 4.7T.
     v1 remains the product until CP-19.
5. **§19.5's definitions frozen,** in `docs/track-b/evidence/cp-17/definitions.md`, implemented in
   `src/cp17/` and tested under native CPython and under the issuing host:
   - **the percentage measure:**
     - the share of scored hours with |p50 − published price| ≤ 10 EUR/MWh (τ = 10, the Owner);
     - the denominator is the scored hours with a published price;
     - partial days score their published hours, and show completeness;
     - zero and negative prices count as they are;
     - missing prices are unscored;
     - it is labelled by its tolerance, and never as accuracy relative to a benchmark;
   - **coverage and width** per nominal band, with a rolling window of the last 28 scored delivery
     days;
   - **the reliability diagram and the quantile-band PIT,** with counts, over the eight bands that
     seven emitted levels define;
   - **the recomputation check:** counts match exactly, and a real value a matches its committed
     value b when |a − b| ≤ 1e-9 · max(1, |b|). A mismatch is shown and blocks that day's
     publication.
6. **§19.3's retrain comparison and §19.7's attributions, frozen.**
   - **The status rule** (the Owner's decision 9), implemented and tested:
     - "identical bit for bit" when every emitted quantile is IEEE-754 equal to the issued artifact's;
     - "within ±0.01 EUR/MWh" when every data-driven selection matched and every emitted quantile is
       within τ_eq;
     - "different" otherwise, with the numbers.
     - **The selections compared** are those CP-17-probe defined, by policy. Diagnostics such as
       weight hashes are reported beside them and are not selections. Records compared across
       runtimes hash canonical int64 indices.
     - **Negative controls:** a changed seed reports "different"; a changed training-window value
       reports "different"; a changed outcome of the target day leaves the payload unchanged.
   - **The attributions, by component class:**
     - **A1, a neural component:** exact baseline Shapley over DDNN-2's input groups, on the issued
       p50, with each member's training-row mean as the baseline, as `src/cp17probe/attribution.py`
       defines it, at most 2,048 coalitions;
     - **v2:** the linear rule, coefficient times the centered, standardized input; its H layer is
       explained in words;
     - **v4:** each component's weight times its forecast, then the linear rule for its linear
       components and TreeSHAP for its tree components. Its interval layer is explained in words;
     - **the wording:** attribution within one fitted artifact, relative to its stated baseline, never
       causality or incremental value.
   - **The reference runtime:** Pyodide 0.28.0; Chromium 153.0.8010.12 and WebKit 26.6; Node.js
     v26.7.0; the Apple M3 machine class, 8 cores, 16 GB, macOS 26.5.2. Record the measured time,
     memory and payload figures. The visitor wording is proposed for the Owner's later publication
     decision, not frozen here.
7. **The payload schema and manifest** (Space plan §4.12 and §7 step 3), frozen in
   `docs/track-b/evidence/cp-17/payload-schema.md` with machine-readable schemas.
   - **The payload kinds:** the manifest, the issued forecasts, the outcomes, the evaluation records,
     the fitted artifact, the training window and the series.
   - **The manifest** lists every served file with its SHA-256, the product and policy identity, the
     daily artifact's identity, the three times (last training, data refresh, issuance) and the
     freshness state.
   - **The embargo:** the training window never carries the target day's outcome.
   - **A reference bundle** for the freeze origin, built from development data by `src/cp17/` and
     validated against the schemas. It stays in `.local/artifacts/cp-17/`, listed in a committed
     manifest. A tampered file fails validation, as a tested negative control.
   - Building the daily pipeline that publishes bundles is CP-18's, not CP-17's.
8. **4.7T's frozen protocol,** `docs/track-b/evidence/cp-17/4-7t-frozen-protocol.md`, written
   outcome-blind. No data after 2026-09-06 exists. It follows §24.9's "registered before 4.7T opens"
   list and the Owner's decision 5:
   - **the comparators:** A1 against the control, nominal v5, and against B0; v3, v4 and v1's
     released artifact are descriptive;
   - **the primary scores:** p50 MAE and seven-quantile WIS, as mean daily losses with equal date
     weights, CP-26's keys and its DST rules;
   - **the endpoints:**
     - against v5: `H0: E[L_A1 − 0.995·L_v5] ≥ 0`, rejected for both MAE and WIS;
     - against B0: `H0: E[L_A1 − L_B0] ≥ 0`, rejected for both;
     - a compound p-value, the largest over the four endpoints, at α = 0.05. There is one candidate,
       so there is no Holm;
   - **the dependence treatment:** a block bootstrap with 7-day blocks, 10,000 replicates and a fixed
     seed;
   - **how the window's forecasts are generated:**
     - the product policy (A1 with its frozen failure policy) by the frozen package under the issuing
       host, daily, at each origin, with only the data available at that origin;
     - the comparators natively, under the rules CP-26 ran them by;
     - vintages per the tier register, and every fallback and failed fit counted;
   - **the minimum scorable support,** and what happens below it, proposed with its reason;
   - **the failure rules:** no runner-up moves up, the window is spent, and v1 stays the product;
   - **the §24.2 audit procedure:** the download logs, the guard's tests, the exposure ledger and the
     Owner's declaration;
   - **the retrieval plan:** sources, date bounds, an estimate of bytes, and the guard moving to
     2026-12-05 only at the gate. It is prepared, not run;
   - **PUBLISH_RULES §14's "before final-candidate test" items,** proposed for the Owner: the badge,
     the wording, the fresh headline window and the hero-switch rule.

**A successful daily fit (§16.2), as this brief fixes it.** For delivery day D, issued at the
origin D−1 11:00 Europe/Berlin:

1. **Inputs.** Every input the policy uses passes the frozen availability check: published strictly
   before the origin, under the tier register, and with the target-day embargo. Labels come only
   from complete delivery days ≤ D−2.
2. **Training.** A1's eight DDNN-2 members each train under the frozen search state, seed,
   held-out weeks, preprocessing and early-stopping rules, with no guard refusal and no non-finite
   value.
3. **Emission.** The policy emits all seven quantiles for every delivery hour of D (23, 24 or 25),
   finite and ordered after the frozen crossing restoration.
4. **The artifact.** It is written with its training window, available labels, seed, timestamps and
   parent, input and state hashes, and its manifest verifies.
5. **The resource ceiling.** Under the issuing host on the reference machine, with one worker, the
   fit and its emission finish within 15 minutes of wall time and 4 GB of peak memory. CP-17-probe
   measured about 70 s and 0.25 GB.
6. **Anything else is a failed fit.** The frozen fallback then issues, and the day is displayed as
   a failed fit. A fallback forecast never counts as a successful A1 fit. On a GFS-late day, A1's fit
   is recorded as skipped for an unavailable input. A v4 or v2 fallback fit is judged by the same
   five criteria, with that policy's own frozen training rules in item 2.

If the landed code's own rules conflict with this definition, for example on a partial ensemble,
report the conflict with its file and line and propose the resolution for r18. Do not change landed
code.

**The return carries the Owner's summary for the anchor amendment (r18),** which ratifies these
terms before 4.7T opens:
- the GFS check's result, and the Owner's rulings 11 and 12;
- the freeze identities, including the issuing host;
- the parity results, native and issuing-host, and v4's browser measurement;
- the product's failure policy and the success definition;
- the protocol's key choices;
- every open item, and any §19.3 exception the Owner must decide.

## Complete authoritative checkpoint bar
`capstone_v21.md` v21-r17 §24.9, "CP-17 and 4.7T", every item, together with §16.2, §19.2, §19.3,
§19.5 and §19.7, and §16.4's CP-17 row. The Owner's decisions record binds them.

Verbatim, extracted with `scripts/bar.py bar` from the anchor at `146dcd7`:

> ### 24.9 CP-17 and 4.7T
>
> **CP-17 freezes one candidate,** CP-28's selection or the control (§24.5), together with:
>
> - its daily retraining rule (§16.2), the issue time, and its inputs and their tiers;
> - its DDNN-2 search state and its combiner state at the freeze;
> - **DDNN-2's search schedule in operation:** the 182-day searches continue, the next being
>   S_14 = 2027-01-06, after S_13 = 2026-07-08, under the same phase and rules, as part of the
>   frozen refit schedule. These are protocol dates only, never permission to read future data;
> - the tier and margin rulings (§24.5);
> - its fallback and failure rules;
> - §19.5's definitions;
> - the lock-price policy, if the final product adopts it
>   (`docs/track-b/final-product-lock-price-notes-2026-10-11.md`).
>
> **§16.4's "before fresh-data test" row** is met by CP-26's daily refits of the control and CP-28's
> of the arms.
>
> **Registered before 4.7T opens:**
>
> - **the comparators:**
>   - the candidate against the control, when they differ;
>   - the candidate against B0, in every case;
>   - v3, v4 and v1's released artifact, as descriptive references;
> - the primary scores and their aggregation, from §24.7;
> - the pass conditions and margins;
> - a dependence treatment suited to 90 days;
> - the failure rules;
> - PUBLISH_RULES §14's "before final-candidate test" trigger.
>
> **The audit (§24.2's standard).** Before opening, the Lead and an independent reviewer check the
> download logs, the boundary guard's tests, the exposure ledger and the Owner's declaration. If the
> standard is not met, 4.7T returns BLOCKED.
>
> **Then:** CP-18, with security hardening (§22's reminder), a controlled live trial whose length the
> Owner sets, and site activation; then CP-19.
>
> **On the timeline.** §16.4 requires CP-19 to cover at least 90 consecutive delivery days after the
> final freeze. With 4.7T opening no earlier than 2026-12-04, CP-19 cannot finish before about March
> 2027. CP-18 may still publish live observations labelled "prospective evaluation in progress".

> ### 16.2 What freezes and what updates every day
>
> Freeze code and dependencies, algorithm/architecture, feature and source-availability
> contracts, hyperparameters, training-window and label-delay rules, initial artifact/state,
> randomness, refit schedule, uncertainty/calibration updates, scoring/decision rules and failure
> policy. Store numeric registry versions and initialization fingerprints as §9 requires.
>
> For the selected final product, **daily retraining, eligible input refresh and forecast
> issuance are required on every delivery day**. Each successful fit produces a versioned
> artifact with its training window, available labels, seed, timestamps and parent/input/state
> hashes. Use only information available at the fixed origin. Daily fitting changes weights/state
> within the frozen policy, not the adopted generation; discretionary tuning or changing that
> policy requires a new freeze and evaluation. Never overwrite an issued forecast after outcomes.
>
> The future brief fixes numeric origin/deadlines, windows, resource ceilings and the exact
> meaning of a successful daily fit before the fresh-data replay/final freeze. Input refresh,
> residual-state maintenance or TabPFN context replacement alone is not weight training. A
> candidate unable to meet daily retraining within admitted licensing/resources needs an explicit
> Owner-approved exception specifying its real update mechanism and public wording before live
> admission; no alternative cadence or exemption is silently inferred from model family.
> Immutable historical v1/reference artifacts remain untouched; this requirement does not
> retroactively retrain them or force every research comparator to train daily.
>
> Daily operation covers all delivery days, including DST. The operational brief supplies an
> explicitly authorized unattended schedule and incident/fallback policy. Publication authority,
> operator ownership, monitoring, permissions and resources must be settled before launch. This
> amendment creates no automation or standing authority to publish externally.
>
> For missing inputs/labels, failed training, resource exhaustion or source outage, apply the
> frozen fallback/staleness limits or record failed issuance. Display a failed/skipped fit as
> such, even if a valid fallback produces a forecast. Do not count retained weights as a new
> successful training run or backfill a late forecast as timely. Failures remain in the record.

> ### 19.2 One code path
>
> The code that performs the daily fit, issuance and scoring is the code the Space runs in the
> browser for §19.3 and §19.5. No second implementation stands in for the product. A component
> that cannot run in the browser runtime is reported at CP-17 under §19.3's exception route; it is
> never replaced silently.

> ### 19.3 "Train it yourself" is required
>
> The Owner requires an in-browser retraining action for the final product.
>
> 1. **What it does.** In the visitor's browser, it retrains the daily fit of one issued delivery
>    day from that fit's shipped training window, with the same code, seed and selection rules.
>    It then predicts that day's quantiles.
> 2. **What it reports.**
>    - whether every data-driven selection matched the issued artifact;
>    - the largest absolute difference per emitted quantile, in EUR/MWh;
>    - one status that states only what was measured: identical bit for bit, within the frozen
>      tolerance τ_eq, or different.
> 3. **What it never does.** It never replaces, rescores or relabels an issued forecast.
> 4. **The probe.** Before the CP-17 freeze, a feasibility probe on development data measures, for
>    the designated policy:
>    - browser time and memory;
>    - payload sizes;
>    - the deviation from the native fit;
>    - selection agreement.
>
>    CP-17 freezes τ_eq and the reference runtime from that measurement.
> 5. **If the policy cannot meet this requirement** within the measured limits, CP-17 returns a
>    blocker. Only an explicit Owner exception, with its public wording, lets the product ship
>    without the action. The requirement is never dropped silently, and never met by retraining a
>    different implementation.

> ### 19.5 Definitions frozen at CP-17, before the fresh-data test
>
> CP-17's freeze fixes the following before 4.7T:
>
> - **The percentage measure beside MAE:** its formula; the tolerance, which the Owner sets; the
>   denominator; eligible and partial-day hours; and zero, negative and missing prices.
>   - The proposed form is the share of scored hours within ±τ EUR/MWh.
>   - A figure relative to a benchmark is improvement, not correctness.
> - **Coverage and width** per nominal band, and the rolling window over scored delivery days.
> - **The reliability diagram:** for each emitted quantile level, the observed share of outcomes
>   at or below it, with counts.
> - **The quantile-band PIT:** the share of outcomes between adjacent emitted quantiles, against
>   each band's nominal share.
> - **The recomputation check.** The Space recomputes its evaluation metrics from the shipped
>   issued forecasts and outcomes, and compares them with the committed records.
>   - Counts match exactly; real values match within a frozen tolerance.
>   - A mismatch is shown, and blocks that day's publication as a product failure.
>
> Nothing in this list is chosen after seeing evaluation outcomes.

> ### 19.7 Attribution and importance
>
> Methods are fixed by component class:
>
> - **a blend:** each component's weight times its forecast, an exact decomposition;
> - **linear components:** coefficient times the centered, standardized input;
> - **tree components:** per-prediction contributions (TreeSHAP), and gain read from the fitted
>   model;
> - **a neural component:** a method frozen at CP-17.
>
> All are presented as attribution within one fitted artifact, never as causality or incremental
> value.

**§16.4's CP-17 row:** "Owner's final-version designation, prerequisites/disposition,
policy/initialization and registry identities, daily fit and failure rules, metric formulas,
resource and calendar plan".

**Further items,** which bind this candidate as well:

1. **Blind.** The boundary stays at 2026-09-06. No byte dated after it is downloaded, read or
   inferred. 4.7T's protocol is fixed without any 4.7T outcome, and no 4.7T forecast is produced.
2. **GitHub-landable as is.** No committed file is over 50 MB, so that the evidence tip itself can
   be pushed. Extracted inputs, reference bundles and bulk stay in `.local/artifacts/cp-17/`, listed
   in a committed manifest.
3. **CI-ready.** The workflow's seven steps (`.github/workflows/tests.yml`) are green on a
   history-free single-commit copy, with `CI=true` and Python 3.12, and `pytest -m stress` passes
   locally.
   - **No test depends on** branch history, tags or local-only inputs. Tests over real stored inputs,
     and tests that need Pyodide, Node.js or a browser, skip with a stated reason when those are
     absent, as they are on CI.
   - **Timing-sensitive tests** carry the `stress` marker.
4. **No pinned file changes.** New code and records live under `src/cp17/`, `tests/` (new CP-17
   files), `reports/cp17/` and `docs/track-b/evidence/cp-17/`. Before editing any other tracked file,
   check that no earlier evidence binds its bytes. That includes `src/cp26/harness.py`'s `PINS`.
5. **The Integration Critic runs the whole suite,** with the history-free `CI=true` Python 3.12
   sequence plus `pytest -m stress`. It also reruns the GFS check, one native parity origin, one
   issuing-host parity origin, one browser identity run, and v4's browser run in one browser.

## Task-specific supporting extract
- **The ceiling is 15 machine-hours,** summed over workers' wall time, including item 1's charge.
  That covers the native parity refits (nine origins, five policies), the issuing-host fits (about
  70 s per A1 fit), the browser runs and the tests.
  - At most 4 concurrent workers, BLAS 1 for native fits, CPU only, peak memory 12 GB, no cloud and
    no GPU.
  - Keep a ledger, and stop before the ceiling.
- **The freeze origin** is the last development origin: issuance for delivery 2026-09-06. 4.7T's
  first forecast, delivery 2026-09-07, is issued at 2026-09-06 11:00 Europe/Berlin from data
  available then. CP-17 prepares it but does not produce it.
- **"Frozen"** means: after this checkpoint lands, any change to the policy, its code, its runtime
  or its state needs a new freeze and evaluation (§16.2).
- **The live schedule, its deadline before 12:00 local and its late-forecast rule** are CP-18's
  (§24.4). CP-17's success definition binds them.

## Applicable constraints
- **No retrieval of any kind:** 0 data bytes, 0 software bytes and 0 remote writes. That includes no
  `git fetch`, `ls-remote` or network probe, and no package installation. Any MLflow record is a
  local draft, never uploaded. Every browser or Node.js run blocks and records non-local requests.
- **The red line (§24.2).** No delivery day after 2026-09-06 is read. Do not read, record or mention
  any market value or event after 2026-09-06.
- **Files that never change:**
  - `data/` and `reports/development-inputs/`;
  - every earlier checkpoint's `reports/` outputs, and every file that earlier evidence binds;
  - the landed code in `src/delu_forecast/`, `src/cp15/` to `src/cp28/` and `src/cp17probe/`;
  - the root `pyproject.toml` (`aa35326abbb13dfa4ea8ba1b4b367343b77f895e7724d5dc94e0bdb7637b4e4a`)
    and `uv.lock` (`5d2aecc3d173fef6191cb2bb9d14cd008c0e9cc764700f1352b81d1924465a94`). Runtimes live
    under `.local/tools/`, not in the lock;
  - the anchors and amendment records, `AGENTS.md`, the role files, the exposure ledger and the
    backup register.
- **Long computation** may run unattended in the background, within the ceiling, under your
  ledger. Commit work in progress on `gauntlet/cp-17` at every stage boundary, and resume from your
  last commit.
- **Credentials** follow `AGENTS.md` § Credentials; none is needed. Export
  `MLFLOW_DISABLE_TELEMETRY=true` and `DO_NOT_TRACK=1`.
- **Pinned identities:** every hash above.
- **Tools:**
  1. run `scripts/gauntlet.py start cp-17` first, from the primary checkout;
  2. create your worktree with
     `git -C /Users/djourno/Downloads/PJM worktree add -b gauntlet/cp-17 .local/worktrees/cp-17/lead main`;
  3. launch the one Critic with `critic-open`, `critic-brief` and `critic-close`. If you cannot
     start a subagent, end your turn with `critic-open`'s output and ask the Orchestrator to launch
     the Critic with exactly the printed prompt;
  4. check the return with `scripts/gauntlet.py return`.
- **Memory:** do not open or edit the files in the Claude memory directory
  (`~/.claude/projects/-Users-djourno-Downloads-PJM/memory/`). Its auto-loaded index informs no
  decision.
- **Owner-facing Git commands** must be non-interactive: `git --no-pager …` and
  `git commit -F <file>`.

## Timebox
About 30 active hours from the second brief's orientation to the terminal return, including the
hour item 1 took. Unattended computation within the 15-machine-hour ceiling runs alongside.

## Owner-only actions already authorized
- **CP-17's execution** under v21-r17 §24.9, §16, §19, the decisions record and this brief. The
  Orchestrator delivers this brief under the Owner's delegation of 2026-10-27, and that delivery is
  the grant. It includes the 15-machine-hour ceiling, the parity refits, and the issuing-host and
  browser runs on the pinned runtimes already present. It covers local `gauntlet/cp-17` candidate
  and evidence commits in your worktree, and the exact packaging of this brief.
- **Nothing else:** no retrieval, no 4.7T run, no boundary move, no mainline operation, no push, tag,
  publication or remote write, no MLflow upload, and no anchor or governance edit.

## Stop and return
- **The resumption's first commit** on `gauntlet/cp-17`, after `2d570e0`:
  - moves the packaged second brief to `docs/track-b/evidence/cp-17/issued-brief-v1-9e522b9a.md`,
    unchanged;
  - copies this amended canonical brief byte for byte to `docs/track-b/evidence/cp-17/issued-brief.md`.
- **Questions.** If the brief is unclear or wrong, ask the Orchestrator before repository work, or
  stop at the conflict. A conflict between this brief and the anchor is resolved by the anchor.
- **The return.** Return exactly one of PASS / BLOCKED / INCOMPLETE, using templates §3. It carries
  both terminal SHAs and the verdict-only delta, checked with `scripts/gauntlet.py return`.
  - Its "Open risk or exact owner action" section gives the Owner's summary for r18, from Observable
    outcome.
  - Name any exposure (§24.2), and any interview-answer trigger, in one line each.
- **Do not** open 4.7T, move the boundary, or download anything. Do not commit to `main`, publish or
  push. Leave your worktree and branch for the LAND and the Orchestrator's reclamation.
