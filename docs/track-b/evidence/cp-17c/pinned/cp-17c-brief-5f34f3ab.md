# Track B Checkpoint Brief — CP-17c (the re-freeze of A1 under the `macos-26` issuing host)

*Issued by the Orchestrator on 2026-10-30, Asia/Jerusalem, under the Owner's decisions 24 and 25.*

## Target
- Repository: DE-LU day-ahead forecasting, `/Users/djourno/Downloads/PJM` (origin
  `hrsi56/delu-day-ahead-forecast`). The checkpoint runs locally, in the Lead's own worktree, plus runs on
  GitHub-hosted `macos-26` runners in one new private scratch repository (below).
- Authorized checkpoint: CP-17c, exactly one.
- **What it does.** It re-freezes A1 under the issuing host that r19 §24.13 pins: the official Node.js
  v26.7.0 darwin-arm64 binary, on a GitHub-hosted `macos-26` runner and on the Owner's Mac. A new runtime
  is a change under §16.2, so the re-freeze comes before 4.7T opens, and 4.7T is its evaluation.
- Ratified plan anchor: `capstone_v21.md`, revision v21-r19, SHA-256 `acdeeab18a17ecdb83ade83d369a5031f3180f8fdb618a23af5b37836663996e`.
  - **The bar** is §24.13's "The issuing host (decision 24; H1)" and "What CP-17c re-freezes, and what
    4.7T then tests", quoted below.
  - **§24.12** ratifies everything CP-17c keeps, with CP-17b's identities.
- **What this round re-freezes:** CP-17b's freeze.
  - its freeze record `docs/track-b/evidence/cp-17b/freeze-record.md`
    (`aaaa93047dece943db8f5d19f78927ac2f1dfb2da94136425570506e3da1eaca`);
  - its manifest `docs/track-b/evidence/cp-17b/freeze-manifest.json`
    (`1fe07bddda14035ff44e9a8e6571c12bd58320d501395cbacc185d902f855502`), policy identity
    `439115fb984359c0c68f79b77a6da4b85a5e75677cfbeb24e043876793ca9893`;
  - its landing record `docs/track-b/cp-17b-landing-2026-10-29.md`;
  - its 4.7T protocol `docs/track-b/evidence/cp-17b/4-7t-frozen-protocol.md`
    (`04aac76a1fc1166a367eb212c19f802abf2b2ba46f7e63345ed72e46508ae22a`), whose text does not change.
- **The probe this round builds on:** CP-17c-probe. Its landing record is
  `docs/track-b/cp-17c-probe-landing-2026-10-30.md` (`350ff44cc1f72f7f1c18873d8d0cc5a4e4df70ccf9cf3b7881911f10ce15aeff`).
  Its harness is `src/cp17cprobe/`, and its proposals are `reports/cp17c-probe/proposals.md`.
- **The Owner's decisions:** 1–19, bound by CP-17b's frozen copies; 20–21 in
  `docs/track-b/owner-decisions-2026-10-29.md`; 22–25 in `docs/track-b/owner-decisions-2026-10-30.md`
  (`1286548362b5618e3fb01c616d673c0af83dd79effeaf102bb9727ac219aabab`).

## Orchestrator-reported expected state
- **Branch / commit:** `main` = `origin/main` at the commit that records this brief's issue.
  - It is a direct child of `bd8ad22ad8f1fe73819b7b813f77746c77d39ced`, r19's ratification, and changes only `progress.md`.
  - **Tags:**
    - `land/cp-17c-probe` = `a92e0525` and `evidence/cp-17c-probe` = `909f552e`, pushed, with CP-17b's,
      CP-17's and CP-17-probe's;
    - `evidence/cp-26` = `ca20331`, which is **local-only and never pushed**.
  - **Branches and worktrees:** none besides `main` and the primary checkout.
- **Working tree:** the primary checkout is on `main` and clean apart from ignored material.
  - **The Orchestrator uses it.** Never switch its branch, and never write a tracked file in it.
    Your only writes there are under `.local/artifacts/cp-17c/`, `.local/tmp/cp-17c/`, `.local/tools/`
    and the worktrees under `.local/worktrees/cp-17c/`.
  - **The canonical brief** is `.local/artifacts/cp-17c/issued-brief.md`. Package that file byte for
    byte.
- **What already exists:**
  - **CP-17b's frozen package** `src/cp17/`, with `python -m cp17.tools.refreeze verify` as its check;
  - **the official binary** in `.local/tools/node-v26.7.0-official/`, retrieved by CP-17c-probe and
    recorded in `reports/cp17c-probe/retrieval-local.json`;
  - **Pyodide 0.28.0** in `.local/tools/pyodide-0.28.0/`;
  - **Chromium 153 and WebKit 26.6** under `.local/tools/ms-playwright/`, driven by Playwright 1.63.0;
  - **the Homebrew binary** at `/opt/homebrew/Cellar/node/26.7.0/bin/node`, pinned. It stays the frozen
    host until this round lands;
  - **CP-17's local material** in `.local/artifacts/cp-17/`, usable read-only, with the verified
    `evidence/cp-26` extract in `inputs/` (A1's `k13.json` among it);
  - **the boundary guard** `src/cp25/guard.py`. Its boundary stays at 2026-09-06;
  - **GitHub:** `gh` is authenticated as `hrsi56`, with the `repo` and `workflow` scopes. The private
    repository `hrsi56/delu-host-probe` is CP-17c-probe's. Leave it untouched.
- Verify this yourself before relying on it, and report any material mismatch.

## Observable outcome
CP-17c closes with the following, bound by one fresh Integration-Critic PASS.

1. **The re-freeze record,** `docs/track-b/evidence/cp-17c/freeze-record.md`, plus
   `freeze-manifest.json`.
   - **It is complete on its own.** CP-17b's unchanged terms are carried by reference with their hashes,
     and every change is stated.
   - **It records:**
     - the new policy identity;
     - the freeze-origin artifact, reissued under the new host;
     - the canary's frozen quantile hash;
     - every value §24.12's rows "The freeze", "Where it landed", "The code", "The freeze origin", "The
       initial state" and "The check that survives amendment" now take.

     4.7T's brief cites them from this record and the landing record.
   - **CP-17b's freeze is superseded, not edited.** Its records stay as committed.
   - **Governance is pinned by frozen copies** under `docs/track-b/evidence/cp-17c/pinned/`, as CP-17b
     did. That covers the anchor at r19, this brief, and the decisions records 20–21 and 22–25. A later
     anchor revision must leave `refreeze verify` green.
2. **The issuing host, frozen,** by §24.13's list "What CP-17c changes":
   - the binary, by its tarball and `bin/node` hashes, retrieved on each run with the release's
     `SHASUMS256.txt` and checked by hash;
   - the binary's path, in place of the Homebrew defaults `cp17.daily.NODE` and
     `cp17.tools.hosts.NODE`;
   - the Pyodide folder, and how the 23 files reach the runner;
   - `refreeze verify` with no problem and no unchecked item on the runner and on the Owner's Mac. That
     includes the zip check, under the zlib runtime the new manifest records;
   - the CPython patch version, pinned without changing the root `uv.lock` or `pyproject.toml`;
   - where the runner's image version and CPU model are recorded.
3. **The checks before each run's first issuance (H2),** frozen as code in the package, with tests:
   - the pins, the binary, the machine (`ImageOS` `macos26` on the runner), and the canary fit, in that
     order;
   - any failure issues nothing in that run;
   - for 4.7T, a failure is a blocker, never a day's outcome;
   - each check is tested failing on its own.
4. **The states' route.** It is defined, and exercised on the runner:
   - A1's `k13.json`;
   - the freeze-origin states of v4, v2 and B0;
   - the shadows' buffers;
   - the issued records;
   - the canary's payload.

   State how each reaches the runner and persists between runs. Persistence across live runs is
   CP-18's to build, so here propose its form, and exercise it across the 14 shadow days.
5. **R7's `schema.py` defect, fixed on its error path only (H3).**
   - A structurally malformed day in a re-hashed issued-forecasts record is listed as a fault, not
     raised.
   - Tested. Every fit, quantile and artifact field stays bit-identical.
   - **If the fix cannot be that narrow,** do not make it. Leave R7 standing, and say so in the return.
6. **The parity, shown.**
   - **On a GitHub-hosted `macos-26` runner in the private scratch repository, and on the Owner's M3:**
     - `refreeze verify` clean;
     - the issuing host's parity at the nine origins, bit for bit, against CP-17b's records;
     - the shadows' buffers over the 14 days, 2026-08-24 to 2026-09-06, against CP-17b's
       `reports/cp17b/shadow-run.json`;
     - the H2 checks passing, and one forced failure of each;
     - the timing and peak memory per fit, against 15 minutes and 4 GB.
   - **On the Owner's M3 only:**
     - native parity with CP-26: five policies, nine origins, since `schema.py` changes;
     - Chromium's and WebKit's parity for A1 at the freeze origin, if the package's zip changed.
7. **The cost,** measured on the scratch repository's `macos-26`, from the shadow run's sequential
   issuances: one issuance, and decision 23's four issuances with the canary. The public class's timing
   is CP-18's (§24.13).
8. **The values 4.7T's brief cites,** in the return, in one table.

## Complete authoritative checkpoint bar
`capstone_v21.md` v21-r19 §24.13. Verbatim, from the anchor at `bd8ad22a`:

> #### The issuing host (decision 24; H1)
>
> **From CP-17c's landing on,** the issuing host is the following.
>
> | Item | Value |
> |---|---|
> | The runtime | Pyodide 0.28.0, by its 23 pinned file hashes, in the official Node.js v26.7.0 darwin-arm64 binary, one worker. The tarball is `node-v26.7.0-darwin-arm64.tar.gz` from `https://nodejs.org/dist/v26.7.0/`, 57,622,243 bytes, SHA-256 `7ee659a7768e641bbfd5360940660b8e8fd0052f77488f365562bac522fc15d4`. Its `bin/node` is 145,238,560 bytes, SHA-256 `a9bd0630891c2dcdee70de88270fee2cc0c4a9e76495039dd3b4f91c5e6b71df`. The chain's judgement and the shadow states run in CPython inside `cp17.daily`, in the environment `uv.lock` pins, under one CPython patch version that CP-17c pins |
> | The machines | **The daily product:** a GitHub-hosted standard runner, label `macos-26` (Apple silicon), in the product's repository, `hrsi56/delu-day-ahead-forecast`, which is public. **4.7T and any local run:** the Owner's Apple-silicon Mac. The runner's image version and CPU model are recorded, and are not pinned |
> | The reference runtime (§19.3) | the issuing host above, plus Chromium 153.0.8010.12 and WebKit 26.6 through Playwright 1.63.0, on the Apple M3 machine class, as §24.12 records |
> | Resources | unchanged: criterion 5's one worker, 15 minutes of wall time and 4 GB of peak memory |
>
> **What this supersedes in §24.12, from CP-17c's landing:**
> - **its "The issuing host" row,** and the issuing-host part of its "reference runtime" row;
> - **in its list "For CP-18":**
>   - the first item's "deadline and late-forecast rule (§24.4)". The publication time below governs;
>   - the item "where the pipeline runs". The issuing host is the binary and runtime pinned above.
>     - **A machine that passes the checks below (H2) is not a runtime change.** That covers any
>       `macos-26` runner VM, image version or CPU model, and the Owner's Apple-silicon Mac.
>     - **Any other binary, runtime file, operating system or architecture is a runtime change** (§16.2).
>
> **Until CP-17c lands,** §24.12's issuing host stays the frozen host. That is the Homebrew binary
> `1ef99ea25fe70c9b67e7efe768ef8ee22148d3cabc703db6131b57aeb617d040`, and Homebrew's `node` stays pinned
> on the Owner's Mac.
>
> **Why macOS and not Linux (H1).** Both runners passed cleanly, and decision 24 leaves the choice to the
> Orchestrator. CP-17c-probe proposed Linux.
> - **One binary for the whole product.**
>   - 4.7T runs on the Owner's Mac with no cloud (§24.12).
>   - With the macOS runner, the daily product, 4.7T and any local run all use the same binary,
>     `a9bd0630…`, on the same architecture. So 4.7T tests the binary and runtime the product issues
>     with. The probe found the arithmetic identical on every machine it measured, and the canary checks
>     it on every run.
>   - Linux's binary, `ad19784f…`, cannot run on the Owner's Mac. 4.7T would then have tested another
>     binary, or moved to the cloud against its registered terms.
> - **The machine moved no bit.** The same binary reproduced every fit on the runners' M2 Pro, across two
>   runner images, and the parity and F1 fits on the M3.
> - **The cost is the same.** The product's repository is public, where standard runners cost nothing on
>   either OS, by GitHub's published terms. The probe's ten-times macOS rate is a private repository's.
> - **What Linux would have given:**
>   - reproduction on two runner CPU generations, one of them from the cancelled run's records. macOS
>     showed two Apple generations, the M2 Pro and the M3;
>   - more plentiful runners. Decisions 22 and 23 already absorb a delayed or dropped run.
>
> **The runner class.** The probe measured a private repository's `macos-26` class, and the public
> repository's class may differ.
> - **The arithmetic** did not change across machines. The canary checks it on every run.
> - **The timing** did change. Each fit's ceiling is enforced in its own process, so a slower class fails
>   closed, never silently.
> - **So CP-17c measures on a private repository's `macos-26`.** CP-18 measures the public class's timing
>   in the product's repository, on development days, before the first live issuance.
> - **Why not a public scratch repository.** Its runs would need A1's search state and CP-17c's material.
>   That would publish them before decision 25's one complete publication.
>
> **The checks before each run's first issuance (H2).**
> - They run once per run, on either machine, in the order below. They run again whenever the machine or
>   the runtime changes within a run.
> - **Any failure means nothing in that run is issued,** and each of its days is not issued.
> - **For 4.7T,** they run once before the window's first issuance, as part of §24.12's runtime
>   precondition. A failure there is not a day's outcome: no window issuance runs, nothing is counted, and
>   4.7T's brief returns it as a blocker.
>
> The checks:
> 1. **The pins.** `refreeze verify` against CP-17c's manifest must report no problem and no unchecked
>    item. It exits 0 in both cases, so the run reads its output, as §24.12 requires for 4.7T.
> 2. **The binary.** The tarball and `bin/node` must match the hashes above.
> 3. **The machine.** It must be macOS on ARM64. On the runner, `ImageOS` must be `macos26`, and the image
>    version is recorded.
> 4. **A canary fit.** A1 at the freeze origin, 2026-09-06, is reissued under the host. Its quantiles must
>    reproduce the frozen SHA-256 that CP-17c records for the freeze origin.
>    - The probe found CP-17b's value,
>      `4e6abb02fbb02aa8ef202207f21b76bc33b34c37d025a0a93a491135ea43cea5` (`reports/cp17b/host-parity.json`),
>      and CP-17c shows it unchanged.
>    - It took 98.8 s on `macos-26`.

and

> #### What CP-17c re-freezes, and what 4.7T then tests
>
> **A new runtime is a change under §16.2,** so CP-17c re-freezes A1 before 4.7T opens. 4.7T is the new
> evaluation. CP-17c records a new policy identity, and 4.7T tests that identity under the host above.
> CP-17c's brief is the Orchestrator's, under decision 25.
>
> **What CP-17c changes:**
> - **the issuing host** above, by its hashes;
> - **how a run gets the binary.** The tarball (57,622,243 bytes) is over the project's 50 MB limit for a
>   committed file, and `bin/node` (145,238,560 bytes) is over GitHub's 100 MB limit, by GitHub's published
>   terms. So each run retrieves
>   the tarball and checks both by hash. A failed retrieval or a failed hash means the run issues nothing;
> - **the binary's path.** The frozen defaults `cp17.daily.NODE` and `cp17.tools.hosts.NODE` name
>   Homebrew's path;
> - **the Pyodide folder** (`cp17.tools.freeze.RUNTIME_DIR`, now `.local/tools/pyodide-0.28.0`), and how
>   the 23 files reach the runner. The probe proposed committing them;
> - **`refreeze verify`'s zip check.** It must report no unchecked item on the runner and on the Owner's
>   Mac. The zip is compared only where the zlib runtime equals the one the manifest records. CP-17b's is
>   `1.2.12`, and CP-17c records its own, so the check must pass on both machines;
> - **where `refreeze verify` runs:** on the runner and on the Owner's Mac, before the first fit;
> - **how the states reach the runner and persist between runs:**
>   - A1's search state `k13.json`, now only in the local `evidence/cp-26` extract;
>   - the freeze-origin states of v4, v2 and B0;
>   - the shadows' buffers;
>   - the issued records;
>   - the canary's payload;
> - **the CPython patch version,** pinned without changing the root `uv.lock` or `pyproject.toml`, which
>   CP-26's tests pin;
> - **where the image version and CPU model are recorded:** in the artifact's lineage, or in CP-18's
>   records;
> - **the checks above** (H2);
> - **R7's `schema.py` defect, fixed (H3).** §24.12's R7 fixes it "at the next freeze". CP-17c is that
>   freeze, and it comes before 4.7T, so 4.7T's result covers the fixed identity.
>   - **The fix changes only that error path.** Every fit, quantile and artifact field stays
>     bit-identical, and CP-17c's parity shows it.
>   - **R7's acceptance and its instruction end at CP-17c's landing.**
>   - **If the fix cannot be that narrow,** R7 stands unchanged, and the Owner is told.
>
> **What CP-17c shows.**
> - **On a GitHub-hosted `macos-26` runner in a private repository, and on the Owner's M3:**
>   - `refreeze verify` with no problem and no unchecked item;
>   - the issuing host's parity at the nine origins, bit for bit;
>   - the shadows' buffers over the 14 days, as CP-17b did;
>   - the timing.
> - **On the Owner's M3 only:**
>   - native parity with CP-26, which is the check of code identity (§24.12);
>   - Chromium's and WebKit's parity at the freeze origin, if the package's zip changed. The reference
>     runtime's machine class is the M3's.
>
> **What it keeps.** Every other term of CP-17b's freeze, as §24.12 ratifies it:
> - the candidate;
> - the success definition and the two chains;
> - partial days and the shadows;
> - τ and τ_eq, and F1's disclosure;
> - the seeds and the refit schedule.
>
> **4.7T's registered terms do not change in their text.**
> - **Its protocol** (`337a102d…` as revised by `04aac76a…`) names CP-17b's freeze manifest and freeze
>   identity, in the revision's preamble and §3, and the issuing host.
>   - From CP-17c's landing, these read as CP-17c's freeze manifest, its policy identity and the issuing
>     host above.
>   - So do §24.12's rows "The freeze", "Where it landed", "The code", "The freeze origin" and "The check
>     that survives amendment".
>   - **CP-17c's landing record states each value,** and 4.7T's brief cites them from it. No further
>     anchor revision is needed for them.
> - **4.7T still runs on the Owner's Mac,** with no cloud, within 20 machine-hours. The official binary ran
>   the probe's local control at 0.90 times the Homebrew binary's time.
> - **Decision 20's badge and wording stay true under CP-17c.** The protocol's text is unchanged, and the
>   re-freeze also comes before any of the window's data is opened. Any change to them stays the Owner's.
> - **Two frozen sentences on daily operation** are superseded for CP-18 by the publication time below.
>   Neither is one of 4.7T's terms:
>   - the protocol's "It is issued at 11:00 local, before the 12:00 gate. CP-18 fixes the deadline and the
>     late-forecast rule (§24.4).";
>   - CP-17b's freeze record §11, "issued at 11:00 local".
>
> **If CP-17c cannot land before 4.7T opens,** including when its parity fails on the product's runner
> class or on the Owner's Mac:
> - the frozen host and CP-17b's identity stay;
> - 4.7T does not open under either until the Owner decides. Decision 24 reads: "Any other outcome is a
>   new decision, brought to the Owner."

**Further items,** which bind this candidate as well:

1. **Blind.** The boundary stays at 2026-09-06. No byte dated after it is downloaded, read or inferred,
   on this machine or on a runner. No 4.7T forecast is produced.
2. **GitHub-landable as is.** No committed file is over 50 MB.
3. **CI-ready.** The workflow's seven steps are green on a history-free single-commit copy, with
   `CI=true` and Python 3.12, and `pytest -m stress` passes locally.
   - No test depends on branch history, tags or local-only inputs.
   - Tests over real stored inputs, runtimes or GitHub skip with a stated reason when those are absent.
4. **What may change:**
   - `src/cp17/`, the package this round re-freezes. List every change and why;
   - `tests/cp17/` and `tests/cp17b/`, only where the re-freeze supersedes a test. List them;
   - new files under `tests/cp17c/`, `reports/cp17c/` and `docs/track-b/evidence/cp-17c/`;
   - new files under `src/cp17cprobe/` or a new `src/cp17c/` for the runner harness, outside the frozen
     package unless the H2 checks need them inside it.

   **Nothing is added under `.github/`** in this repository. The scratch workflow is kept under the new
   paths.
5. **What never changes:**
   - `reports/cp17/`, `reports/cp17b/`, `reports/cp17c-probe/`, `docs/track-b/evidence/cp-17/`,
     `docs/track-b/evidence/cp-17b/` and every earlier checkpoint's records;
   - the landed code in `src/delu_forecast/`, `src/cp15/` to `src/cp28/` and `src/cp17probe/`;
   - `data/`, `reports/development-inputs/`, the root `pyproject.toml` and `uv.lock`;
   - the anchors and amendment records, the decisions records, `AGENTS.md`, the role files, the exposure
     ledger and the backup register.

   Before editing any tracked file, check that no earlier evidence binds its bytes. That includes
   `src/cp26/harness.py`'s `PINS`, and CP-17's and CP-17b's freeze checks.
6. **The Integration Critic** reruns, locally:
   - the suite in the history-free `CI=true` Python 3.12 sequence, and `pytest -m stress`;
   - `refreeze verify` on the new manifest, and after a simulated anchor change;
   - the H2 checks with one forced failure;
   - the issuing host's parity at one origin, and native parity at one DST origin.

   It recomputes every runner comparison from the downloaded results, verified by their hashes. It runs
   nothing on GitHub.

## Task-specific supporting extract
- **The private scratch repository:** one new **private** repository, `hrsi56/delu-host-refreeze`,
  created for this round.
  - **Why private.** §24.13 measures CP-17c on a private repository's `macos-26`, so that A1's search
    state and this round's material are not published before decision 25's one complete publication.
    **It is never made public.**
  - **It holds only what the runs need:** the package's files, the harness, the workflow, the pinned
    Pyodide files, A1's `k13.json`, the shadows' states, and development payloads or their inputs. No
    data dated after 2026-09-06.
  - **It creates no secret and no variable.** The workflow token is read-only apart from what uploading
    results needs.
  - **It is left in place after the return.** Deleting it is the Owner's, and the return names it.
- **Retrieval, authorized for software only:**
  - Node.js v26.7.0's official darwin-arm64 tarball and `SHASUMS256.txt` from
    `https://nodejs.org/dist/v26.7.0/`, verified against §24.13's hashes;
  - **on the runners:** `uv` and the Python packages `uv.lock` pins; the pinned CPython; GitHub's own
    actions, pinned by commit SHA;
  - Record each URL, the version, the bytes and the SHA-256. Make no other network contact.
- **The ceilings:**
  - **Local:** 8 machine-hours, summed over workers' wall time. At most 4 concurrent workers, BLAS 1 for
    native fits, CPU only, peak memory 12 GB, no GPU.
  - **GitHub:** standard hosted `macos-26` runners only, no larger runners.
    - **At most 1,000 minutes** as GitHub's usage report counts them for a private repository, with macOS
      at ten times. That is about 100 job-minutes.
    - **CP-17c-probe already used 863 of this account's minutes this month,** and the remaining quota is
      unknown. No paid usage: a quota or billing refusal stops the runs.
    - **The runs, in this priority:**
      1. `refreeze verify`, the H2 checks with one forced failure each, and the canary;
      2. A1's issuing-host parity at the nine origins;
      3. the 14 shadow days, latest first. Their sequential issuances also give item 7's cost, so no
         separate cost run is needed.
    - **If the quota stops the runs** before all three are done, return BLOCKED with what was measured.
      The Orchestrator resumes when the quota resets.
  - Keep a ledger, and stop before a ceiling.
- **"Frozen"** keeps §16.2's meaning. After CP-17c lands, any change to the policy, its code, its runtime
  or its state needs a new freeze and a new evaluation, except a machine that passes the H2 checks
  (§24.13).

## Applicable constraints
- **The red line (§24.2).** No delivery day after 2026-09-06 is read, recorded or mentioned, except as
  a protocol or calendar date.
- **Homebrew's `node` stays pinned and installed.** Never upgrade, unpin or remove it.
- **Credentials** follow `AGENTS.md` § Credentials; none is needed, and no secret is created anywhere.
  Export `MLFLOW_DISABLE_TELEMETRY=true` and `DO_NOT_TRACK=1`.
- **Long computation** may run unattended in the background, within the ceilings, under your ledger.
  Commit work in progress on `gauntlet/cp-17c` at every stage boundary.
- **Tools:**
  1. run `scripts/gauntlet.py start cp-17c` first, from the primary checkout;
  2. create the worktree with
     `git -C /Users/djourno/Downloads/PJM worktree add -b gauntlet/cp-17c .local/worktrees/cp-17c/lead main`;
  3. run one Critic, through `critic-open`, `critic-brief` and `critic-close`. **The first line of the
     Critic's launch assignment** forbids opening the Claude memory directory;
  4. check the return with `scripts/gauntlet.py return`.
- **Memory:** do not open or edit the files in the Claude memory directory
  (`~/.claude/projects/-Users-djourno-Downloads-PJM/memory/`).
- **Owner-facing Git commands** must be non-interactive.

## Timebox
About 12 active hours from orientation to the terminal return.

## Owner-only actions already authorized
- **CP-17c's execution** under decisions 24 and 25 and this brief. The Orchestrator delivers this brief
  under the Owner's grant, and that delivery is the grant. It covers:
  - creating the private scratch repository `hrsi56/delu-host-refreeze`, pushing to it, and running its
    workflows within the GitHub ceiling;
  - the software retrievals above, and the local ceiling;
  - local `gauntlet/cp-17c` commits and the brief's packaging.
- **Nothing else:** no push, branch or tag on `origin`; no secret; no data retrieval; no 4.7T run; no
  boundary move; no anchor or governance edit; no publication; no mainline operation.

## Stop and return
- **First commit.** The first commit copies the canonical brief byte for byte to
  `docs/track-b/evidence/cp-17c/issued-brief.md`.
- **Questions.** If the brief is unclear or wrong, ask the Orchestrator before repository work, or stop
  at the conflict. The anchor and the Owner's decisions prevail over this brief.
- **The return.** Return exactly one of PASS / BLOCKED / INCOMPLETE, using templates §3, with both
  terminal SHAs and the verdict-only delta, checked with `scripts/gauntlet.py return`.
  - **Its "Open risk or exact owner action" section** gives the values 4.7T's brief cites, the runner's
    measurements, and the proposed state route for CP-18.
  - **If a parity check fails** on either machine, return BLOCKED with the numbers. §24.13 then sends
    the outcome to the Owner.
  - Name the scratch repository and the GitHub minutes used.
  - Name any exposure (§24.2), and any interview-answer trigger, in one line each.
- **Do not** open 4.7T or begin CP-18. Leave your worktree and branch for the LAND.
