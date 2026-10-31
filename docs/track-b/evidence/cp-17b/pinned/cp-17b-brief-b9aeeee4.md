# Track B Checkpoint Brief — CP-17b (the re-freeze before r18: partial days, shadow issuances, the review's notes, and governance pins that survive amendment)

*Issued by the Orchestrator on 2026-10-28, Asia/Jerusalem, under the Owner's delegation of
2026-10-27 and his approval of 2026-10-28: "יש לך אישור מלא עם כל ההשעיות וכל מה שאתה צריך".*

## Target
- Repository: DE-LU day-ahead forecasting, `/Users/djourno/Downloads/PJM` (origin
  `hrsi56/delu-day-ahead-forecast`). The checkpoint runs locally, in the Lead's own worktree.
- Authorized checkpoint: CP-17b, exactly one.
- **What it does:** it amends the earlier freeze before 4.7T opens. That is allowed, because no 4.7T
  outcome exists, and §16.2 asks for a new freeze for any change to the policy or its code.
- Ratified plan anchor: `capstone_v21.md`, revision v21-r17, SHA-256
  `3124451da84c3cd99fae728c371207aace6ae537474f7769d0ff6333ee53e4fc`, unchanged.
  - **The bar** is CP-17's, for the items this round touches: §24.9's failure rules, §16.2 (the
    success definition and failure policy) and §19.2 (one code path), together with the Owner's
    decisions 13–19 below.
  - **The anchor's sections stand as CP-17's brief quoted them,** with the identities it recorded.
- **CP-17, which this round amends:**
  - its landing record `docs/track-b/cp-17-landing-2026-10-28.md`
    (`7f90cb87c8d64ec3faecb0b8a45ab2d5e027ace3cc1824785bcbffa5f1ff267b`);
  - its freeze record `docs/track-b/evidence/cp-17/freeze-record.md`
    (`4a47623718441cca1eb7210dfa3b47dcfcc5c421b8e3a57b858282281b5e0cec`), §11 in particular;
  - its freeze manifest `docs/track-b/evidence/cp-17/freeze-manifest.json`
    (`4af62410c2d08442b85cd11cd271f0eb367bfb019196784d1e048a46a908a544`), policy identity
    `b304d6479a8628a26014422c6682f1e6543bd0d9ff656ad34b2d87b163bc254e`;
  - its Integration verdict `docs/track-b/evidence/cp-17/integration.md`
    (`7083bfb7176a9b7fbb4bfc4e03d00317f0c5601ec5360c006de1a2b4e066319d`), whose notes N1–N7 this
    round fixes;
  - its 4.7T protocol `docs/track-b/evidence/cp-17/4-7t-frozen-protocol.md`
    (`337a102d6bbbf6d9335fe6ed701a4c80ea1b9172344ffc0f63909b155b567c58`).
- **The Owner's decisions:**
  - 1–12 in `docs/track-b/pre-cp-17-rulings-2026-10-25.md`
    (`d29b0abc6a17dcb6bca2cbd464c1dcc5858c50e2b653a42e646d3dc66ea8f6b0`), which CP-17's freeze pins;
  - 13–19 in `docs/track-b/cp-17-owner-decisions-2026-10-28.md`
    (`4d049770e51a37a806c52a4ae681f70aae62ac1847a67c7c0c3109385da3ac37`). In summary:
    - 13. **C2:** a fit that emits every hour the frozen design can forecast succeeds on those hours,
      displayed as partial, with the missing hours named. A day with no forecastable hour is a failed
      issuance.
    - 14. **G1:** v4 and v2 issue every day under the issuing host as shadow issuances, never
      published, to keep their own buffers current. B0 needs no fit.
    - 15. **The chains stay** as decisions 11 and 12 set them.
    - 16. **C1:** released errors come from days ≤ D−2. Training targets are the days before D
      published strictly before the issue time.
    - 17. **G2** is recorded as the frozen behaviour.
    - 18. **F1** is disclosed, and τ_eq is unchanged.
    - 19. **This round,** then r18.

## Orchestrator-reported expected state
- **Branch / commit:** `main` = `origin/main` at the commit that records this brief's issue.
  - It is a direct child of `ee0ea9d2`, and adds only the decisions record
    `docs/track-b/cp-17-owner-decisions-2026-10-28.md` and a `progress.md` change.
  - **Tags:**
    - `land/cp-17` = `d7619ea` and `evidence/cp-17` = `3876bcd`, both pushed;
    - `land/cp-17-probe`, `evidence/cp-17-probe`, `land/cp-28` and `evidence/cp-28`, all pushed;
    - `evidence/cp-26` = `ca20331`, which is **local-only and never pushed**.
  - **Branches and worktrees:** none besides `main` and the primary checkout.
- **Working tree:** the primary checkout is on `main` and clean apart from ignored material.
  - **The Orchestrator uses it.** Never switch its branch, and never write a tracked file in it.
    Your only writes there are under `.local/artifacts/cp-17b/`, `.local/tmp/cp-17b/` and the
    worktrees under `.local/worktrees/cp-17b/`.
  - **The canonical brief** is `.local/artifacts/cp-17b/issued-brief.md`. Package that file byte for
    byte.
- **What already exists:**
  - **CP-17's landed package,** `src/cp17/`, with its tests `tests/cp17/`, records `reports/cp17/`
    and evidence `docs/track-b/evidence/cp-17/`;
  - **CP-17's local material** in `.local/artifacts/cp-17/`, usable read-only:
    - the verified `evidence/cp-26` extract in `inputs/` (2.1 GB);
    - the final reproduction in `final/`, with its payloads and the reference bundle;
  - **the pinned runtimes:** Pyodide 0.28.0 in `.local/tools/pyodide-0.28.0/`; Chromium 153 and
    WebKit 26.6 under `.local/tools/ms-playwright/`, driven by Playwright 1.63.0; Node.js v26.7.0 at
    `/opt/homebrew/Cellar/node/26.7.0/bin/node`;
  - **the boundary guard** `src/cp25/guard.py`. Its boundary stays at 2026-09-06.
- Verify this yourself before relying on it, and report any material mismatch.

## Observable outcome
CP-17b closes with the following, bound by one fresh Integration-Critic PASS.

1. **Partial days** (decision 13).
   - **Success criterion 3 becomes:** seven finite, ordered quantiles for every hour of D that the
     frozen design can forecast. The design's own rule decides which hours those are
     (`src/cp15/data.py:79-80`, `src/cp24/member.py:43-45`).
   - **The display:** such a fit is displayed as partial, with the missing hours named.
   - **A day with no forecastable hour** is a failed issuance for every policy.
   - **Exercised** under the issuing host and natively, on real development days:
     - one day after a DST change with one unforecastable hour;
     - one input-gap day with no forecastable hour.
   - **Tested,** with the boundary cases.
2. **Shadow issuances** (decision 14).
   - **The rule:** the daily issuance runs v4 and v2 under the issuing host as shadow issuances every
     day, whether or not A1 succeeds.
     - Their central and scale are stored before the outcome.
     - Each policy's own released errors enter its own buffer once, after D−2's release.
     - They are never published or displayed as the product's forecast.
   - **B0's raw-error buffer** comes from published prices. Show that rebuilding it this way equals
     CP-26's `RawResidualState` exactly. If it does not, shadow-issue B0 too; that costs under a
     second.
   - **Exercised** on consecutive development days:
     - starting from the buffers rebuilt from CP-26's ledger at an earlier origin;
     - shadowing day by day through delivery 2026-09-06 under the issuing host;
     - ending with buffers equal to the freeze-origin state CP-17 recorded, bitwise natively and
       within rounding under the host.
   - **The measured total per delivery day:** A1 plus the shadows, within the 15-minute ceiling.
3. **The review's notes, fixed** (CP-17's verdict, N1–N7).
   - **N1:** the issuance driver joins the frozen identity.
     - Each attempt runs with a timeout and writes to its own fresh output path.
     - A dead, timed-out or unreadable attempt is a failed fit, and the chain moves on.
     - Tested with a killed process and with a stale record.
   - **N2:** an issued artifact must carry a resource measurement. The schema and the verifier refuse
     one without it. The native comparison path is labelled as such and can never issue.
   - **N3:** criterion 4 is judged inside the chain, before it moves on.
   - **N4:** `freeze verify` recomputes every hash it records:
     - the package;
     - the 23 Pyodide files and the Node.js binary;
     - the search state;
     - the payload schemas;
     - the governance copies of item 4.
   - **N5:** the report lists every open item.
   - **N6:** the machine-readable manifest carries the browser part of the reference runtime:
     Chromium 153.0.8010.12, WebKit 26.6, Playwright 1.63.0 and the machine class.
   - **N7:** the ceiling's message reports wall time exactly.
4. **Governance pins that survive amendment** (decision 19).
   - **Governance documents** are bound by frozen copies under
     `docs/track-b/evidence/cp-17b/pinned/`, with their SHA-256 values recorded:
     - the anchor at v21-r17;
     - CP-17's two briefs and this brief;
     - both decisions records;
     - `cp-25-rulings-2026-10-12.md` and `cp-26-gfs-ruling-2026-10-19.md`.
   - **Computational inputs, code and runtimes stay verified live:** the snapshot, CP-25's and
     CP-26's manifests, the tier register, the margin tables, the weather features, `uv.lock`,
     `pyproject.toml`, the package and the runtimes.
   - **Tests prove:**
     - a changed live anchor or a new decisions record leaves `freeze verify` green;
     - a changed computational input, code file or runtime file fails it.
5. **The re-freeze record,** `docs/track-b/evidence/cp-17b/freeze-record.md`, plus
   `freeze-manifest.json`.
   - **It is complete on its own,** so that r18 cites one document. CP-17's unchanged sections are
     carried by reference with their hashes, and the changes are stated:
     - decisions 13–18;
     - the success definition with the partial-day rule;
     - the shadow rule;
     - C1's wording;
     - G2's behaviour;
     - F1's disclosure.
   - **It records** the new policy identity and the freeze-origin artifact, reissued under the
     issuing host with the new code.
   - **CP-17's freeze is superseded, not edited.** Its records stay as committed.
6. **The fit is unchanged, shown.** List every file of `src/cp17/` that changed, and why.
   - **If any fit, emission, scoring, retrain, attribution or schema module changed:** rerun CP-17's
     full native parity (five policies, nine origins) and its issuing-host parity.
   - **Otherwise:**
     - native parity at two origins, one of them a DST origin;
     - issuing-host parity at one origin;
     - browser identity for A1 at one origin;
     - all equal to CP-17's committed records.
7. **4.7T's protocol, revised only where decisions 13 and 14 touch it,** and written outcome-blind,
   as `docs/track-b/evidence/cp-17b/4-7t-frozen-protocol.md`.
   - **What can change:** how partial days and shadow issuances enter generation and counting.
   - **Everything else stays as CP-17 froze it,** stated by its hash.
   - **The window's analysis tool** stays refused by the guard on the real window.
8. **The Owner's summary for r18,** in the return: the final identities, the terms r18 ratifies, and
   every open item.

## Complete authoritative checkpoint bar
The items of CP-17's bar that this round touches, as CP-17's brief (`9f235cf8…`) quoted them from
`capstone_v21.md` v21-r17: §24.9's "its fallback and failure rules", §16.2 and §19.2. The Owner's
decisions 13–19 bind them. Verbatim, from the anchor at `ee0ea9d2`:

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

**Further items,** which bind this candidate as well:

1. **Blind.** The boundary stays at 2026-09-06. No byte dated after it is downloaded, read or
   inferred. No 4.7T forecast is produced.
2. **GitHub-landable as is.** No committed file is over 50 MB.
3. **CI-ready.** The workflow's seven steps are green on a history-free single-commit copy, with
   `CI=true` and Python 3.12, and `pytest -m stress` passes locally.
   - No test depends on branch history, tags or local-only inputs. Tests over real stored inputs or
     runtimes skip with a stated reason when those are absent.
4. **What may change:**
   - `src/cp17/`, CP-17's package, which this round re-freezes. List every change.
   - `tests/cp17/`, only where the re-freeze supersedes a test. List them.
   - New files under `tests/cp17b/`, `reports/cp17b/` and `docs/track-b/evidence/cp-17b/`.
5. **What never changes:**
   - `reports/cp17/`, `docs/track-b/evidence/cp-17/` and every earlier checkpoint's records;
   - `src/cp17probe/` and the landed code in `src/delu_forecast/` and `src/cp15/` to `src/cp28/`;
   - `data/`, `reports/development-inputs/`, the root `pyproject.toml` and `uv.lock`;
   - the anchors, both decisions records, `AGENTS.md`, the role files, the exposure ledger and the
     backup register.

   Before editing any tracked file, check that no earlier evidence binds its bytes. That includes
   `src/cp26/harness.py`'s `PINS` and CP-17's `freeze.py`.
6. **The Integration Critic** reruns:
   - the suite in the history-free `CI=true` Python 3.12 sequence, and `pytest -m stress`;
   - one partial-day issuance;
   - the shadow run;
   - the killed-process control;
   - `freeze verify` after a simulated anchor change.

## Task-specific supporting extract
- **The ceiling is 8 machine-hours,** summed over workers' wall time.
  - At most 4 concurrent workers, BLAS 1 for native fits, CPU only, peak memory 12 GB, no cloud and
    no GPU.
  - Keep a ledger, and stop before the ceiling.
- **"Frozen"** keeps CP-17's meaning. After CP-17b lands, any change to the policy, its code, its
  runtime or its state needs a new freeze and evaluation (§16.2).

## Applicable constraints
- **No retrieval of any kind:** 0 data bytes, 0 software bytes and 0 remote writes. That includes no
  `git fetch`, `ls-remote` or network probe. Every browser or Node.js run blocks and records
  non-local requests.
- **The red line (§24.2).** No delivery day after 2026-09-06 is read, recorded or mentioned, except
  as a protocol or calendar date.
- **Credentials** follow `AGENTS.md` § Credentials; none is needed. Export
  `MLFLOW_DISABLE_TELEMETRY=true` and `DO_NOT_TRACK=1`.
- **Long computation** may run unattended in the background, within the ceiling, under your
  ledger. Commit work in progress on `gauntlet/cp-17b` at every stage boundary.
- **Tools:**
  1. run `scripts/gauntlet.py start cp-17b` first, from the primary checkout;
  2. create the worktree with
     `git -C /Users/djourno/Downloads/PJM worktree add -b gauntlet/cp-17b .local/worktrees/cp-17b/lead main`;
  3. run one Critic, through `critic-open`, `critic-brief` and `critic-close`;
  4. check the return with `scripts/gauntlet.py return`.
- **Memory:** do not open or edit the files in the Claude memory directory
  (`~/.claude/projects/-Users-djourno-Downloads-PJM/memory/`).
- **Owner-facing Git commands** must be non-interactive.

## Timebox
About 12 active hours from orientation to the terminal return.

## Owner-only actions already authorized
- **CP-17b's execution** under the decisions above and this brief, including the 8-machine-hour
  ceiling and the issuing-host and browser runs on the pinned runtimes already present. The
  Orchestrator delivers this brief under the Owner's delegation, and that delivery is the grant. It
  covers local `gauntlet/cp-17b` commits and the brief's packaging.
- **Nothing else:** no retrieval, no 4.7T run, no boundary move, no mainline operation, no push or
  tag, no publication, no MLflow upload, and no anchor or governance edit. r18 is the
  Orchestrator's, after this round.

## Stop and return
- **First commit.** The first commit copies the canonical brief byte for byte to
  `docs/track-b/evidence/cp-17b/issued-brief.md`.
- **Questions.** If the brief is unclear or wrong, ask the Orchestrator before repository work, or
  stop at the conflict. The anchor and the Owner's decisions prevail over this brief.
- **The return.** Return exactly one of PASS / BLOCKED / INCOMPLETE, using templates §3, with both
  terminal SHAs and the verdict-only delta, checked with `scripts/gauntlet.py return`.
  - Its "Open risk or exact owner action" section gives the Owner's summary for r18.
  - Name any exposure (§24.2), and any interview-answer trigger, in one line each.
- **Do not** open 4.7T or begin r18. Leave your worktree and branch for the LAND.
