# CP-17c-probe landing — 2026-10-30

**CP-17c-probe is closed. Both GitHub-hosted runners reproduce A1's frozen issuing host bit for bit.**
- It is the probe of the Owner's decision 24 (`docs/track-b/owner-decisions-2026-10-30.md`).
- **The host chosen:** a GitHub-hosted macOS runner, `macos-26` (Apple silicon), with the official Node.js
  v26.7.0 darwin-arm64 binary. The Orchestrator made the choice under decision 24, and the reasons are below.
- **Next:** r19 pins that host, and the CP-17c re-freeze follows. Both come before 4.7T opens, no earlier than
  2026-12-04.

## Identities

| Item | Value |
|---|---|
| Brief | `docs/track-b/evidence/cp-17c-probe/issued-brief.md`, SHA-256 `e0528d5c81ede078003cbc76ee4b7d8b0d3caa1a573b9f16ab3d95a3c2d4c3c1`. Issued at `b903aefe`, packaged byte-identical from its canonical file |
| Anchor | `capstone_v21.md` v21-r18, SHA-256 `9caf83393a08df6851f7e829722583a142ebc2354ee99a3bbaa015d929f3d401`, unchanged |
| The Owner's mandate | Decision 24, `docs/track-b/owner-decisions-2026-10-30.md`, SHA-256 `1286548362b5618e3fb01c616d673c0af83dd79effeaf102bb9727ac219aabab` |
| Final candidate (the Integration PASS binds here) | `3bba29495db26de28a45a20ab05f09a1d3bcb313`. Verdict: `docs/track-b/evidence/cp-17c-probe/integration.md`, SHA-256 `63b5130aa0386f918516ba8c5dbaaa7a374616c65ef4c90719d7faafba3e3541` |
| First review (FAIL, preserved) | `7ee1cf07`. Verdict: `integration-attempt-1.md` |
| Evidence tip / `evidence/cp-17c-probe` | `909f552e3a815786b8140d2ddbe2cdfaccec8066` |
| Squash landing / `land/cp-17c-probe` | `a92e0525`. Single parent `b903aefe`. Its tree equals the evidence tip's |
| Outputs | New only: `src/cp17cprobe/` (the harness, input builder, comparison, report renderer and the scratch workflow `probe.yml`), `tests/cp17cprobe/`, `reports/cp17c-probe/`, `docs/track-b/evidence/cp-17c-probe/`. No frozen or landed file changed, and nothing was added under `.github/` |

## The return and its receipt

- **The Lead returned PASS.** It took about 2.5 active hours of a 6-hour timebox, about 1.8 of 3 local
  machine-hours, and 863 of 1,500 GitHub minutes.
  - GitHub's timing API reported 0 billable ms, so the minutes are counted from the jobs' own times, with
    macOS at ten times, as a private repository is charged.
- **The Integration reviews.**
  - **The first review returned FAIL** on `7ee1cf07`, and is preserved. The report gave `macos-26` one
    image and one CPython. In fact the two shadow jobs ran image `20261026.0455.2` with CPython 3.13.17.
  - **The repair (`3bba2949`):** the hosts table has one row per job, and a test fails when a job is
    missing from it. The measurement did not change.
  - **The second review passed `3bba2949`.** It reran one local-control day (A1, 2026-09-06, bit-identical),
    ran the whole suite in CI mode, and recomputed `measurement.json` byte for byte from the downloaded
    results, verified by their hashes.
- **The Orchestrator's receipt:**
  - `gauntlet.py return` and `receipt`: 0 problems. The delta between the candidate and the evidence tip
    is the verdict only.
  - **22 files**, all new, all inside the brief's allowed paths. The largest is `measurement.json`, 283 KB.
  - **Dates after 2026-09-06** in the added content are job timestamps, runner image versions, record
    names and S_14's protocol date only. No market day after 2026-09-06 was read.
  - **The history-free CI** (`CI=true`, Python 3.12) on the squash commit `a92e0525`, with tree
    `6e01e22a43c24f256beb894dcee0e2896d3583ad`:
    - every step exits 0;
    - the main suite has 2,446 passed and 46 skipped;
    - the log is in `.local/artifacts/r18/land-cp17c-probe-validation/`.

    The push followed.
  - **The secret guard** passed at the commit and at the push.

## What the probe established

- **What ran.** CP-17b's own reference fits went through the frozen chain `cp17.daily`, unchanged. Each
  runner used the frozen package, Pyodide 0.28.0's 23 pinned files, and the official Node.js v26.7.0
  binary for its platform in place of Homebrew's. The fits were:
  - A1 at the nine parity origins, and v4, v2 and B0 at two of them;
  - all four policies on F1's day, 2026-03-30;
  - the 14 shadow days, 2026-08-24 to 2026-09-06, with the shadow states carried on the runner.
- **The results:**

| | Fits bit-identical | Shadow buffers | Slowest attempt | Largest peak | One issuance (A1 + 3 shadows, sequential) |
|---|---|---|---|---|---|
| `ubuntu-24.04` (EPYC 7763, 2 vCPU, 7.7 GiB) | 75 / 75 | equal at all 14 origins and in the final state | 284.6 s | 844 MiB | 349–366 s |
| `macos-26` (M2 Pro virtual, 5 cores, 14 GiB; two images) | 75 / 75 | equal at all 14 origins and in the final state | 228.9 s | 917 MiB | 270–284 s |
| Local control (this M3, official darwin-arm64 binary `a9bd0630…`) | 19 / 19, twice | — | 153.3 s | 894 MiB | 197–212 s |

- **More evidence.** Fourteen records left by the cancelled run 1 are bit-identical as well, 10 of them on
  a different Linux CPU (EPYC 9V45).
- **No selection differed, and no quantile moved.** τ_eq and F1 are untouched.
- **Every fit stayed inside criterion 5's ceilings:** 900 s and 4 GB.

## The host choice (decision 24)

Both runners passed cleanly. Under decision 24 the choice between them is the Orchestrator's.

**Chosen: the GitHub-hosted standard runner `macos-26` (Apple silicon).** It runs Pyodide 0.28.0 in the
official Node.js v26.7.0 darwin-arm64 binary:
- the tarball `node-v26.7.0-darwin-arm64.tar.gz` is 57,622,243 bytes, SHA-256
  `7ee659a7768e641bbfd5360940660b8e8fd0052f77488f365562bac522fc15d4`;
- its `bin/node` has SHA-256 `a9bd0630891c2dcdee70de88270fee2cc0c4a9e76495039dd3b4f91c5e6b71df`;
- the release's `SHASUMS256.txt` has SHA-256
  `4533f0a43b9ba7f78a48230a0511b9dd5c931f20c3b3cac281ff9b7a2080fb2e`.

**Why macOS, against the probe's proposal of Linux.**
- **One binary for the whole product.**
  - 4.7T runs on the Owner's Mac with no cloud (§24.12's registered terms). It needs about 6.4 machine-hours
    under the issuing host.
  - With the macOS runner, the daily product, 4.7T and any local fallback or rerun all run the same binary,
    `a9bd0630…`, on the same architecture. So 4.7T tests the product exactly as it issues.
  - With Linux, the product would issue under `ad19784f…`, a binary 4.7T cannot run on this Mac. 4.7T would
    then either test a different binary or move to the cloud, which would change a registered term.
- **The machine does not move a bit.** The same binary reproduced every fit:
  - on the runners' M2 Pro, across two runner images;
  - on this M3, twice.
- **Cost is no difference.** The production repository is public, and GitHub's standard runners cost
  nothing there on either OS. The probe's ten-times macOS rate applies to private repositories only.
- **Time is well inside the ceilings.**
  - The slowest attempt was 228.9 s, 25% of 900 s. The peak was 917 MiB of 4 GB.
  - One issuance with the canary is about 7 job-minutes. Decision 23's four issuances are about 21.
- **What Linux would have had.** It reproduced on two CPU generations, and Linux runners are more
  plentiful. Decisions 22 and 23 already absorb a delayed or dropped run, so availability is not decisive.

**Risks carried into r19 and CP-17c.**
- **The runner class.** The probe measured a private repository's macOS class (5 cores, 14 GiB). The public
  repository's class may differ. Arithmetic did not change across machines, but timing did. CP-17c measures
  the timing on the class the product will use.
- **Images change.** GitHub updates its runner images. Each run records the image version and the CPU model
  in the lineage. The run's checks catch any change that matters:
  - the pins (`refreeze verify` with no problem and no unchecked file);
  - the runner's OS, architecture and image;
  - a canary fit, A1 at the freeze origin (`4e6abb02…`, 98.8 s on `macos-26`), which must reproduce its
    frozen hash before anything is issued.

  If any check fails, the day is not issued.
- **The Homebrew binary stays the issuing host until CP-17c's re-freeze.** It stays installed and pinned.

## For r19 and CP-17c

- **r19 pins the host above.** The new "where the pipeline runs" row names:
  - the runner label and architecture;
  - the binary, by its tarball and `bin/node` hashes, retrieved and verified on each run;
  - the Owner's Mac, with the same binary, for 4.7T and for local runs.
- **r19 also carries decision 22.** It changes §24.4's 12:00 and late sentence and §16.4's "timely" count.
- **What CP-17c re-freezes** (from proposal 2, applied to macOS):
  - `cp17.daily.NODE` and `cp17.tools.hosts.NODE` default to Homebrew's path. The new manifest names the
    binary by a fixed path on the runner and on the Mac.
  - One pinned CPython patch version. The runners had 3.13.16 and 3.13.17; the frozen host has 3.13.15.
  - `refreeze verify` runs on the runner before the first fit.
  - A1's search state `k13.json`, now only in the local `evidence/cp-26` extract, must reach the runner by
    a route CP-17c decides.
  - The canary and the runner checks above, in that order, before anything is issued.
- **S_14 (2027-01-06) is after 4.7T, and r19 decides where it runs.**
  - Native, as S_13 ran, it fits one hosted macOS job: about 0.9 h with four workers and the 2× margin.
  - Under the issuing host it is about 3.9 h on the measured class.
  - A native search on another machine is not this M3's arithmetic. The probe measured only the Pyodide
    path.

## Disclosures

- **Both Integration Critics opened one file** in the Claude memory directory before they read their brief,
  which forbids it. Each disclosed it in its verdict, and neither verdict rests on it.
  - The cause: a subagent sees the memory index before any brief.
  - **From now on,** the Critic's launch assignment carries the memory rule in its first line.
- **The second verdict's observations are all non-gaps:**
  - the "weekly" image cadence in proposal 2 is not from a record;
  - setup-python's interpreter is recorded by version only;
  - seven A1 references are bound by payload identity only;
  - the compute ledger omits the Lead's CI run (about 0.17 machine-hours) and the second review's (0.27).

## Interview answers

- How did we show that moving the daily retraining to a GitHub runner changes not a single forecast?
  150 runner fits and 19 local fits were bit-identical to the frozen host, across four CPU models and two
  architectures, by the frozen comparison.

## Reclamation

- **`gauntlet.py reclaim cp-17c-probe --disposition land`:**
  - a verified bundle, `.local/artifacts/cp-17c-probe-reclaim-20261030/gauntlet-cp-17c-probe.bundle`
    (169,994 bytes);
  - a ref snapshot, `refs-before.txt`;
  - the Lead worktree, removed;
  - the branch, deleted. It was `909f552e` and is kept by `evidence/cp-17c-probe`.

  The two Critic worktrees had already been removed by `critic-close`.
- **Live citations of the branch:** none. The brief and the two verdicts name `gauntlet/cp-17c-probe`.
  They are historical and never repointed.
- **Both tags are on GitHub.**
- **Outside this repository:** the private scratch repository `hrsi56/delu-host-probe` is left in place,
  with no secret. Deleting it is the Owner's decision. Its artifacts expire 30 days after the runs, and
  their zips are kept under `.local/artifacts/cp-17c-probe/downloads/`.
- **Retained locally:**
  - `.local/artifacts/cp-17c-probe/`, about 64 MB;
  - `.local/tools/node-v26.7.0-official/`, about 275 MB. It holds the chosen binary, CP-17c's input.
  - the scratch clone `.local/tmp/cp-17c-probe/`, about 291 MB. It can be deleted, because
    `cp17cprobe.inputs` rebuilds it byte for byte.
