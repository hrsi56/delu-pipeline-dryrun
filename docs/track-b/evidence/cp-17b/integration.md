# Verdict — CP-17b — Integration — PASS

- Candidate SHA: 2366ea64137bdd38046ae2091bc4ff9c4c3e80b4
- Plan / version / bar: `capstone_v21.md`, v21-r17 (SHA-256 `3124451da84c3cd99fae728c371207aace6ae537474f7769d0ff6333ee53e4fc`),
  `### 16.2 What freezes and what updates every day` (section 32 lines, 2408 bytes, SHA-256
  `c54723eff2f113a18efaef5884c3c2a93876c755edacf3b2fb70dc3af44ac5ec`), with §24.9's failure rules, §19.2 (section 6 lines,
  335 bytes, SHA-256 `538342106768fedad8201f44bd4aadc3de02415cebf321cd0c1584eca75f6f7c`) and the Owner's decisions 13–19
  (`docs/track-b/cp-17-owner-decisions-2026-10-28.md`, `4d049770…`), as the brief `b9aeeee4…` binds them.
- Verbatim bar excerpt (confirmed in the file at the candidate with `scripts/bar.py bar … --at 2366ea64…`, exit 0):

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

  (The excerpt is the citation. Any line number is a courtesy and is non-binding.)
- Worktree clean before and after: yes. `git -C …/critic-2 status --porcelain` was empty at the start and at the end;
  `HEAD` stayed `2366ea64137bdd38046ae2091bc4ff9c4c3e80b4`. I made no Git write in the repository. The only commits were
  the CI runner's own, inside its throwaway `git init` copies under `.local/tmp/cp-17b/critic2-{ci312,amended,stress}`.
  The runner removed the first two (`scratch_removed: true`). I removed `critic2-stress` myself after interrupting its
  second attempt (see row 24). My scratch copies (a `git archive` extract, a copied Pyodide folder, pytest temp
  folders) were under `.local/artifacts/cp-17b/critic-2/` and are deleted. The run outputs I kept are listed at the end.
- Compute: about **1.73 machine-hours** (summed worker wall time), never more than 4 workers at once, BLAS 1 for every
  native fit, no network, no retrieval. Itemised at the end.

## Commands actually run

All from `/Users/djourno/Downloads/PJM/.local/worktrees/cp-17b/critic-2` unless a scratch copy is named, with
`MLFLOW_DISABLE_TELEMETRY=true DO_NOT_TRACK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src`, the five `*_NUM_THREADS=1`,
`CP17_PROJECT_ROOT=/Users/djourno/Downloads/PJM` (except where it is hidden for CI), and
`/Users/djourno/Downloads/PJM/.venv/bin/python` (3.13.15). Outputs are under `.local/artifacts/cp-17b/critic-2/` (`$O`).

| # | Command | Exit | Observed |
|---|---|---|---|
| 1 | `git status --porcelain`; `git rev-parse HEAD` | 0 | empty; `2366ea64…80b4` |
| 2 | `git show main:scripts/bar.py \| python3 -I - bar capstone_v21.md '### 16.2 What freezes and what updates every day' --at 2366ea64…` | 0 | the excerpt above, verbatim; plan `3124451d…`, section 32 lines, 2408 bytes, `c54723ef…` |
| 3 | `… bar capstone_v21.md '### 19.2 One code path' --at 2366ea64…` | 0 | §19.2 verbatim as the brief quotes it; section `53834210…` |
| 4 | `shasum -a 256` of the anchor, the brief, both decisions records, the CP-25 and CP-26 rulings, CP-17's two briefs, the 8 `pinned/` copies, CP-17's four evidence files and its landing record; `cmp` of the packaged brief with `.local/artifacts/cp-17b/issued-brief.md` | 0 | every hash equals the brief's and `refreeze.GOVERNANCE`'s; brief byte-equal (`b9aeeee4…`) |
| 5 | `git diff --name-status 6abbce5f 2366ea64…`, filtered against the allowlist; `git diff --stat 6abbce5f 2366ea64…` over the protected paths | 0 | 46 added, 6 modified; nothing outside `src/cp17/`, `tests/cp17/`, `tests/cp17b/`, `reports/cp17b/`, `docs/track-b/evidence/cp-17b/`. In `tests/cp17/` only `test_hosts.py` and `test_records.py`. Zero changes under `reports/cp17/`, `docs/track-b/evidence/cp-17/`, `src/cp17probe/`, `src/delu_forecast/`, `data/`, `reports/development-inputs/`, `pyproject.toml`, `uv.lock`, the anchor, both decisions records, `AGENTS.md`, the exposure ledger and the backup register. `dailyfit.py`, `scoring.py`, `retrain.py`, `attribution.py`, `host/`, `browser/`, `stubs/`, `tools/freeze.py`, `tools/fresh_test.py`, `src/cp15`, `src/cp24`, `src/cp25` and `src/cp26` are unchanged |
| 6 | `git ls-tree -r -l 2366ea64…` for files over 50 MB | 0 | none; the largest is 30.8 MB (`models/champion/python_model.pkl`) |
| 7 | red-line scan of every added line (pinned copies, brief and attempt-1 verdict excluded) for dates after 2026-09-06, and for UTC keys past `2026-09-06T21:00Z` | 0 | only protocol and calendar dates (2026-09-07 window origin, 2026-10-25 clock change, 2026-12-04/05, S_14 = 2027-01-06), file names (`…-2026-10-25.md`, `cold-rerun-2026-10-15/…`), ledger timestamps and decision dates. The latest UTC key is `2026-09-06T21:00:00`, the last hour of delivery 2026-09-06 |
| 8 | `python -m cp17.tools.bundle --out $O/bundle/src.zip` | 0 | `34 files 91480 bytes 41256a486bae6477`; `src.json` equals `reports/cp17b/bundle-manifest.json` in every key; zip `41256a48…a8aa` |
| 9 | `python -m cp17.tools.refreeze verify docs/track-b/evidence/cp-17b/freeze-manifest.json` | 0 | `verified`. Identity `439115fb…`, content `261c6283…`, package `682c5833…`, driver `18e15893…`; `reference_runtime` has Chromium 153.0.8010.12, WebKit 26.6, Playwright 1.63.0, Node v26.7.0, Pyodide 0.28.0 and the machine class |
| 10 | `python -m cp17.tools.freeze verify docs/track-b/evidence/cp-17/freeze-manifest.json` (CP-17's verifier, live tree) | 1 | exactly `src/cp17/issue.py: missing or altered`, `src/cp17/schema.py: missing or altered` |
| 11 | `python3 …/ci-runner.py critic2 2366ea64… ci312 amended` (read in full first), from the primary checkout | 0 | **ci312:** tree `631804a0…` = the candidate's, Python 3.12.14, `CI=true`, local inputs hidden. Steps 01 `uv sync` 0; 02 `build_wasm_payload.py` 0 (85 s); 03 `pytest -q` 0, **2427 passed, 42 skipped** (635 s); 04 CQR fixture 0, 8 passed; 05 `verify_release.py` 0, PASS; 06 WASM equivalence 0, 15 passed; 07 `publication_guard.py tree` 0, clean; 08 stress collection 0, 19/2468. **amended** (the same, after appending a section to `capstone_v21.md` and adding `docs/track-b/cp-17b-owner-decisions-new.md`): every step exit 0, 03 `pytest -q` **2427 passed, 42 skipped** (855 s). `complete: true` and `scratch_removed: true` in both. Logs are in `.local/artifacts/cp-17b/critic2-validation/{ci312,amended}/` |
| 12 | `python3 …/ci-runner.py critic2 2366ea64… stress` (attempt 1), run alongside three of my own workers while an external macOS process (Storage.appex, about 250% CPU) was also running, load average about 27 | 1 | `pytest -m stress -q`: `....FFF.F..........`, **4 failed, 15 passed, 1 skipped** (577 s). The four failures are all in `tests/test_cp26_ledger_liveness.py` (`wedged_supervisor_lock…`, `supervisor_startup_accounting_wait…`, `duplicate_live_request…`, `rich_closed_jobs_three_busy_owners…`), each `wait_until(…, timeout=12)`: "monitored test process did not reach its expected state". By collection order (row 25), positions 1–3 passed: `tests/cp17/test_hosts.py::test_issuing_host_reproduces_its_record`, `::test_browser_reproduces_the_issuing_host` and the CP-17-probe browser test. Logs are kept in `$O/stress-attempt-1/` and `critic2-validation/stress/` |
| 13 | `python -m cp17.tools.exercise partial --day 2026-03-30 --bundle $O/bundle/src.zip --out $O/partial/2026-03-30` | 0 | `[('A1','ok')] successful daily fit (partial: 23 of 24 hours; missing 02:00 Europe/Berlin) True` (359 s) |
| 14 | `python -m cp17.tools.exercise partial --day 2023-08-31 …` | 0 | every policy `failed` at its inputs with "no hour of the day is forecastable by the frozen design"; display `failed issuance (no hour of the day is forecastable by the frozen design)`; `process_seconds == {}`; artifact verifies (31 s) |
| 15 | `python -m cp17.tools.exercise shadow-run --mode native --first 2026-08-24 --out $O/shadow-native` | 0 | 14 days, each `{'v4','v2','B0'}: registered`, buffers bitwise; `final {'v4': True, 'v2': True, 'B0': True}` (623 s) |
| 16 | `python -m cp17.tools.exercise shadow-run --mode host --first 2026-09-05 --bundle $O/bundle/src.zip --out $O/shadow-host` (the brief's shorter span, two workers) | 0 | A1 ok both days; buffers bitwise; `final {'v4': True, 'v2': True, 'B0': True}` (322 s). In this span no host-fitted error is released yet (see row 17) |
| 17 | `… shadow-run --mode host --first 2026-09-03 … --out $O/shadow-host4` (four days, two workers, so that the host's own shadow errors of 2026-09-03 and 2026-09-04 enter the buffers) | 0 | A1 ok every day; every shadow registered. Buffers bitwise on 09-03 and 09-04, then within 7.4e-15 (09-05) and 1.42e-14 (09-06) for v2 and v4; B0 bitwise. `final {'v4': False, 'v2': False, 'B0': True}` against CP-17's bitwise digests, so within rounding (811 s) |
| 18 | `python -m cp17.tools.exercise control --day 2026-09-06 --payloads $O/shadow-native/payloads --bundle $O/bundle/src.zip --stale /Users/djourno/Downloads/PJM/.local/artifacts/cp-17/final/nodes/node-A1-2026-09-06.json --out $O/control` | 0 | `[('A1','failed'),('v4','ok')] failed fit; fallback v4 issued PASS` (222 s) |
| 19 | `python -m cp17.tools.parity --out $O/parity --origins 2026-03-29` | 0 | `2026-03-29 PASS {A1,v2,v4,v5,B0: True}`; `PASS 98s` |
| 20 | scratch `git archive 2366ea64… \| tar -x -C $O/scratch` → `refreeze verify … --root $O/scratch` | 0 | `verified` |
| 21 | the same scratch with r18 simulated: `capstone_v21.md` appended, decisions 13–19, decisions 1–12 and the CP-25 rulings appended, a new decisions record `docs/track-b/r18-owner-decisions-critic2.md` added. Then (a) `refreeze verify`; (b) CP-17's `freeze verify` on the raw tree; (c) CP-17's `freeze.verify` on `refreeze.cp17_view` | 0 / 1 / 0 | (a) `verified`; (b) `capstone_v21.md: altered`, decisions 1–12 altered, CP-25 rulings altered, plus `issue.py`, `schema.py` (CP-17's own verifier still binds live governance, as landed); (c) exactly `['src/cp17/issue.py: missing or altered', 'src/cp17/schema.py: missing or altered']` |
| 22 | in that amended scratch, `CI=true`, local inputs hidden: `pytest -q -m 'not stress' tests/cp17/test_records.py tests/cp17b` | 0 | **128 passed, 1 skipped** (the skip: "the pinned Pyodide runtime or the evidence/cp-26 extract is absent"); `test_freeze_manifest_verifies` passes |
| 23 | in the amended scratch, one change at a time, then restored: tier register +1 byte, `margin-table.csv` +1 byte, `src/cp17/dailyfit.py`, `src/cp17/retrain.py`, `src/cp17/host/node_run.mjs`, `src/cp17/daily.py`, `src/cp24/member.py` appended. Each time `refreeze verify` and (with `CI=true`) `pytest tests/cp17/test_records.py::test_freeze_manifest_verifies`. Then: a copied Pyodide 0.28.0 under a stand-in `CP17_PROJECT_ROOT` (unaltered, `pyodide.asm.js` appended, then removed); the frozen anchor copy `pinned/anchor-capstone-v21-r17.md` +1 byte; finally everything restored with the r18 changes kept | see observed | `refreeze verify` exit 1 for every change, naming the file (and, for package files, "the rebuilt package's files/identity/zip differ"). The superseded CP-17 test **failed** for every input and every CP-17 code file. It **passed** for `daily.py`, which CP-17's manifest never listed; `refreeze` refuses it. Runtime copy: `verified` (0), then `…/pyodide.asm.js: altered` (1), then `…: absent` (1). Altered frozen copy: `refreeze` "the frozen governance copy is missing or altered" (1), and the CP-17 test fails. Restored: `verified` (0) |
| 24 | `ci-runner.py critic2 2366ea64… stress` (attempt 2), stopped by me after 85 s, during `uv sync`/version steps, to stay within the compute allowance | — (stopped) | no test ran; its partial logs are in `$O/stress-attempt-2-interrupted/`; I removed its leftover scratch after row 25 |
| 25 | in that attempt's history-free 3.12 copy (tree `631804a0…`, its synced `.venv`, the runner's stress environment): `uv run pytest -m stress -rA` on the four tests that failed in row 12; then `uv run pytest --collect-only -m stress -q` | 0; 0 | **4 passed** in 37.6 s (load average about 16); the collection order puts the CP-17 host and browser tests first and the CP-26 liveness tests at positions 4–10 |
| 26 | final `git status --porcelain`; `git rev-parse HEAD` | 0 | empty; `2366ea64…80b4` |

## Evidence actually inspected

- **The brief** `docs/track-b/evidence/cp-17b/issued-brief.md`, all of it, and the Owner's decisions 13–19.
- **The first verdict** `integration-attempt-1.md`, and the repair `git diff 2529c62f 2366ea64…`: `refreeze.cp17_view`
  and `cp17_governance_live`; the new `test_freeze_manifest_verifies`; the three new `tests/cp17b/test_refreeze.py`
  tests; the freeze record's and report's new paragraphs; the three new ledger rows.
- **The whole diff** `6abbce5f..2366ea64` for `src/cp17/`. I read in full `issue.py`'s changes (`forecastable_keys`,
  `check_inputs`, `check_emission`, `partial`, `attempt`, `decide`'s `accept`, the failed-issuance record,
  `resources_ok`, `artifact`'s measurement guard, `verify_artifact`), `schema.py`'s changes (`MEASUREMENT`, `PARTIAL`,
  the two artifact forms, `maximum`, `verify_bundle`'s partial check), `daily.py`, `shadow.py`, `tools/refreeze.py`,
  `tools/bundle.py`, `tools/refbundle.py`, and the parts of `tools/exercise.py` my reruns use. CP-17's
  `tools/freeze.verify` is unchanged.
- **Tests:** the `tests/cp17/` supersessions in `test_hosts.py` (the `package` fixture) and `test_records.py` (two
  tests); `tests/cp17b/test_daily.py` (the real-process controls), `test_units.py` (partial-day boundary cases),
  `test_records.py` and `test_refreeze.py` (the test lists and the amendment controls).
- **The evidence:** `freeze-record.md` (all of it), `freeze-manifest.json` (identity, content, `reference_runtime`,
  `pipeline_files_sha256`, `freeze_origin`); `artifact-2026-09-06.json` (file `c1c3dab6…`, artifact `91e00fea…`, quantiles
  `4e6abb02…`); `4-7t-frozen-protocol.md`, diffed against CP-17's `337a102d…`; the 8 `pinned/` copies.
- **The records:** `reports/cp17b/report.md` (its Open items, all of them); `partial-days.json`, `shadow-run.json`,
  `control.json`, `native-parity.json` (`PASS`, `all_equal_to_cp17: true`, nine origins); `host-parity.json`
  (`all_equal_to_cp17: true`, Chromium and WebKit equal at A1 2026-09-06); `b0-rebuild.json`; `retrieval-manifest.json`
  (0/0/0); `compute-ledger.jsonl`.
- **My run outputs, compared field by field with the committed records:**
  - **Partial, 2026-03-30.**
    - Equal to `partial-days.json` in every field except timings and memory: forecastable hours `{A1,v4,v2,B0: 23}`; the
      missing hour `02:00 Europe/Berlin (2026-03-30T00:00:00+00:00)`; payloads; start state; path; display; host and
      native the same path and the same display.
    - The issued quantiles are `718f0c0d…`, as recorded. The artifact's `partial` names the missing hour and it verifies.
      A1's own measurement is 114.4 s and 595 MiB.
    - Issued against native: `different`, 0.0823 EUR/MWh, selections matched (F1, as the record says). A1 native is
      bitwise CP-26's D2.
    - The native comparison is labelled "not judged: the native comparison path, which can never issue".
    - Every shadow is `ok` on 23 keys with "no intake: not a complete canonical issuance", with quantile hashes equal
      to the record's. There were no non-local requests.
  - **Partial, 2023-08-31.** Equal to the record except the attempts' elapsed seconds.
  - **Native shadow run.**
    - The start state and every day's payload hashes equal the record's.
    - Every day's buffers are bitwise CP-26's rebuilt state (largest difference 0.0). Every shadow's quantiles are
      bitwise CP-26's stored forecast, and every shadow quantile hash equals the record's.
    - The final digests equal CP-17's recorded freeze-origin digests for v4, v2 and B0.
  - **Host shadow runs** (rows 16–17).
    - A1 issued every day with "successful daily fit", and every artifact verifies.
    - Each day's payloads equal my native run's until host-fitted errors enter (09-05).
    - Every executed file hash equals the package's.
    - The per-day total of A1 plus three shadows was 260–475 s, within 900 s on every day. The largest totals ran
      under heavy concurrent load. The largest process RSS was 927 MiB.
    - No non-local request in any of the 32 issuing-host run records.
    - A1's 2026-09-06 quantiles are `4e6abb02…`, bit for bit the freeze-origin artifact's.
  - **Control.**
    - A1's process was killed at 15.0 s with "the issuing-host process exceeded the 15.0 s timeout and was killed".
      Its fresh folder `A1-2026-09-06-5x8z32m7/` is empty.
    - The three planted stale records (a valid successful A1 record, quantiles `4e6abb02…`) were never read.
    - v4 issued from its own fit (`76599de7…`, 98.6 s, 873 MiB), and the artifact verifies.
    - The v4/v2/B0 shadow quantile hashes equal `control.json`'s. v4 is `fitted_by: chain`, fitted once.
  - **Parity spot 2026-03-29.** Checks, the five payload hashes, the five quantile hashes and the ledger file hash
    are equal to CP-17's committed `reports/cp17/native-parity.json`.

## Checklist verdict

| # | Checklist item | Verdict | Evidence |
|---|---|---|---|
| G | **The previous review's gap, by its own acceptance test.** On a history-free single-commit copy with `capstone_v21.md` changed and a new decisions record added, the seven steps pass under `CI=true` Python 3.12, including `test_freeze_manifest_verifies`. `refreeze verify` still prints `verified`. In the same copy, a changed computational input, or a CP-17 code file other than `issue.py`/`schema.py`, fails both | PASS | Row 11 (amended): all seven steps exit 0, 2427 passed, the same count as unamended. Rows 21–22: with a broader r18 (three governance records amended as well), `refreeze` verified, CP-17's check on the view reports exactly the two re-frozen files, and 128 cp17/cp17b tests pass under `CI=true`. Row 23: tier register, margin table, `dailyfit.py`, `retrain.py`, `node_run.mjs` and `cp24/member.py` each fail both. `daily.py` fails `refreeze` only, correctly, since CP-17 never froze it |
| O1 | Partial days (decision 13): criterion 3 on the forecastable hours by the design's rule; the partial display naming the missing hours; a no-forecastable-hour day a failed issuance for every policy; exercised under the host and natively on a post-DST day and an input-gap day; tested with boundary cases | PASS | `issue.forecastable_keys` applies `Inputs.rows` and the payload keys, then `check_inputs`, `check_emission(expected)`, `partial`. Rows 13–14 reproduce `partial-days.json`. `test_units.py` covers complete 23/24/25-hour days, one missing hour, 24 of 25, a single hour, zero hours for every policy, extra/missing/reordered/duplicated keys, a key outside the day, and a fallback on a partial day |
| O2 | Shadow issuances (decision 14): v4 and v2 every day under the host, stored before the outcome, their own errors into their own buffer once after D−2, never published; B0 from prices shown equal to CP-26's or else shadow-issued; the consecutive-day exercise ending at CP-17's recorded state (bitwise natively, within rounding under the host); measured per-day total within 15 min | PASS | `daily.issue_day` runs the shadows after the chain whether or not A1 succeeds, without fitting a chained fallback twice, and registers them through `ShadowStates.issue` on CP-26's classes. `b0-rebuild.json`: not equal (36 days, 524 origins), so B0 is shadow-issued as the brief prescribes. Row 15: native bitwise every day and at the end. Rows 16–17: under the host, buffers within 1.42e-14 once host errors enter, B0 bitwise, per day ≤ 475 s. The full 14-day host record (`shadow-run.json`, 2.18e-14) is consistent with these |
| O3 | N1–N7 fixed | PASS | **N1:** one fresh process per attempt, a `mkdtemp` output path and a 900 s timeout. A killed, timed-out, non-zero, recordless, unreadable, foreign or unmeasured record is a failed fit (`test_daily.py` with real processes; row 18 on the issuing host). The driver is in `pipeline_files_sha256` and refused when altered (row 23). **N2:** `fitted_artifact` v2 has two forms with `MEASUREMENT` ≤ 900 s / 4 GiB; `artifact()` raises and `verify_artifact` refuses an issued artifact without a measurement; the native path is labelled and never issues (row 13). **N3:** `decide`'s `accept` builds and verifies the artifact inside the chain. **N4:** `refreeze verify` recomputes the package (rebuilt), the 23 Pyodide files, Node.js, `k13.json`, the schemas and the governance copies (rows 9, 20–23). **N5:** the report's Open items. **N6:** `reference_runtime` (row 9). **N7:** `resources_ok` prints `{wall_seconds!r}` |
| O4 | Governance pins that survive amendment (decision 19): governance documents bound by frozen copies with SHA-256; computational inputs, code and runtimes verified live; tests prove a changed live anchor or a new decisions record leaves `freeze verify` green, and a changed computational input, code file or runtime file fails it | PASS | Row 4: the 8 frozen copies are at the pinned hashes. Rows 21–23 and `tests/cp17b/test_refreeze.py`. The first review's gap is closed (item G): CP-17's superseded check now reads governance from the frozen copies through `cp17_view` and still binds every input, code file, runner and the freeze-origin artifact live |
| O5 | The re-freeze record and manifest: complete on its own; decisions 13–18, the success definition with the partial-day rule, the shadow rule, C1, G2 and F1 stated; the new policy identity and the reissued freeze-origin artifact; CP-17's freeze superseded, not edited | PASS | `freeze-record.md` §§1–13, read in full. Identity `439115fb…`. The artifact `91e00fea…` was issued under the host with the new code, cites that identity and package `682c5833…`, and verifies (row 9). My host run reproduces its quantiles `4e6abb02…` (row 16). CP-17's four evidence files are unchanged at the cited hashes (rows 4–5) and checked by `refreeze` |
| O6 | The fit is unchanged, shown: every changed `src/cp17/` file listed with why; `schema.py` changed, so CP-17's full native parity and issuing-host parity rerun, equal to CP-17's records | PASS | The record's table lists `issue.py`, `schema.py`, `shadow.py`, the driver `daily.py` and the native-only tools, matching the diff (row 5). Fit, emission, scoring, retrain and attribution modules are unchanged. `native-parity.json` and `host-parity.json` are all equal to CP-17's. Row 19 reproduces 2026-03-29. Row 12: the issuing host and Chromium reproduce CP-17's A1 quantiles on CP-17b's package |
| O7 | 4.7T's protocol revised only where decisions 13 and 14 touch it, outcome-blind; everything else by CP-17's hash; the analysis tool still refused by the guard | PASS | The diff against CP-17's protocol replaces §3 and §11, adds to §4 and §7, and cites `337a102d…` for the rest. It holds no outcome. `fresh_test.py` is unchanged. The guard tests (`tests/cp17b/test_records.py`, CP-17's) pass in row 11 |
| O8 | The Owner's summary for r18 in the return | PASS as far as the candidate carries it | The return postdates this verdict and is checked by `scripts/gauntlet.py return`. The report's Open items carry the r18 terms (C2, G1 with B0's shadow, the chains, C1, G2, F1 with the larger 0.082 EUR/MWh figure, the GFS-late disclosure, the protocol revision), the first review's failure and its fix, CI's unchecked local runtimes, the concurrent-timing caveat, §19.2, 4.7T and CP-18. My reruns confirm the figures they reach (F1's 0.0823, the per-day totals, B0's 36/524) |
| F1 | Blind: boundary 2026-09-06; no later byte read or inferred; no 4.7T forecast | PASS | Row 7. Every exercise reads development data through 2026-09-06; my runs touched no later day. `test_no_committed_record_holds_a_date_after_the_boundary` passes in row 11 |
| F2 | GitHub-landable: no committed file over 50 MB | PASS | Row 6 |
| F3 | CI-ready: the seven steps green history-free with `CI=true` Python 3.12; `pytest -m stress` passes; no test depends on history, tags or local-only inputs; local-input tests skip with a reason | PASS | Row 11, unamended and amended. Stress: 15 of 19 passed under heavy load in row 12, including both CP-17 host and browser tests. The 4 failures are timing waits in `tests/test_cp26_ledger_liveness.py`, unchanged since CP-26 and untouched by CP-17b, and all 4 pass in the same history-free 3.12 copy once the load eased (row 25). I judge them load-induced, not a candidate defect. Under `CI=true` the cp17b local-input test skips with its stated reason (row 22) |
| F4 | What may change, all listed | PASS | Row 5; the record lists every `src/cp17/` change and the three `tests/cp17/` supersessions |
| F5 | What never changes | PASS | Row 5 |
| F6 | The Critic's reruns: CI 3.12 history-free and stress; one partial-day issuance; the shadow run; the killed-process control; `freeze verify` after a simulated anchor change | PASS (all performed; results as expected) | Rows 11–25. The shadow run was rerun natively in full, and under the host on the brief's shorter span, plus a four-day span so that host-fitted errors enter the buffers |

### Observations (not blocking)

- **N1. The stress run is load-sensitive.** CP-26's liveness tests wait 12 s for a monitored process. With three other
  workers and an external process loading the machine, four of them timed out (row 12); alone, they pass (row 25). The
  Lead's timing caveat applies to this suite too: run `pytest -m stress` on a quiet machine.
- **N2. The brief's shorter host span does not test host intake.** `--first 2026-09-05` releases only CP-26's
  registrations by 2026-09-06, so its buffers are trivially bitwise (row 16). A span of at least three days (row 17) is
  the shortest that puts the host's own shadow errors into the buffers.
- **N3. CP-17's own verifier still binds live governance on the raw tree** (row 21b), as landed and by design. The
  workflow and the tests run it only on `cp17_view`. Anyone running `python -m cp17.tools.freeze verify` by hand after
  r18 will see the governance files reported. CP-17b's `refreeze verify` is the freeze check that survives amendment.
- **N4. Outside this checkpoint (for r18 to plan; read, not run).** `tests/test_cp26_repair_contract.py` hashes the
  live anchor's §§1–23 against r13. The runner's `amended` mode only appends a section, so it does not exercise that
  test. An r18 edit inside §§1–23 would turn that CP-26 test red. CP-17b may not change it, and its bar does not
  reach it.
- **N5. The assignment calls `src/cp17/daily.py` "changed".** Against `6abbce5f` it is new. The freeze record treats
  it correctly, as the new driver. No effect.

## Compute

| Job | Workers | Wall (s) |
|---|---:|---:|
| CI 3.12 history-free, `ci312` then `amended` (row 11) | 1 | 1,744 |
| shadow run, native, 14 days (row 15) | 1 | 625 |
| partial days, both (rows 13–14) | 1 | 390 |
| shadow run, host, 2 days (row 16) | 2 | 324 |
| killed-process control (row 18) | 1 | 222 |
| shadow run, host, 4 days (row 17) | 2 | 815 |
| parity spot (row 19) | 1 | 101 |
| stress attempt 1 (row 12) | 1 | 615 |
| stress attempt 2, stopped (row 24) | 1 | 85 |
| the four stress tests and the collection (row 25) | 1 | 46 |
| bundle, `verify` runs, scratch extract, amendment tests and controls (rows 8–10, 20–23) | 1 | about 120 |
| **Total** | ≤ 4 at once | **about 6,230 worker-seconds, 1.73 machine-hours** |

Retained outputs, about 141 MB, in `.local/artifacts/cp-17b/critic-2/`: `bundle/`, `partial/`, `shadow-native/`,
`shadow-host/`, `shadow-host4/`, `control/`, `parity/`, `logs/`, `stress-attempt-1/`, `stress-attempt-2-interrupted/`.
The runner's logs are in `.local/artifacts/cp-17b/critic2-validation/`. All of it is disposable reproduction material;
nothing in it is the sole copy of required evidence.
