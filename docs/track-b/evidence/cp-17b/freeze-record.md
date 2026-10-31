# CP-17b's freeze record: A1 re-frozen before r18, superseding CP-17's freeze

**The machine-readable freeze** is [`freeze-manifest.json`](freeze-manifest.json). It holds:
- **the policy identity:** SHA-256 `439115fb984359c0c68f79b77a6da4b85a5e75677cfbeb24e043876793ca9893`. Every issued artifact cites it.
- **the freeze-origin artifact,** reissued under the issuing host with the new code;
- **the initial state,** CP-17's, carried by reference;
- **the reference bundle's manifest,** in the re-frozen payload kinds;
- **`verification`:** how each hash it records is checked.

Its content digest is `261c6283b50c051a71d7a98da6e8ab9912f0b33e79c00ba2c180ab224b97cca1`.

**Verifying it.** `python -m cp17.tools.refreeze verify docs/track-b/evidence/cp-17b/freeze-manifest.json` recomputes
every hash the manifest records, or names how a hash is checked where it is local reproduction material (N4). It
refuses a missing or altered file, a wrong registry version, a wrong set of governance pins, or a broken digest. Its
controls are in `tests/cp17b/test_refreeze.py`.

**This record is complete on its own,** so that r18 cites one document. Each section below is either restated with
its change, or carried from CP-17's freeze record unchanged, by that record's hash.

**What it supersedes.** CP-17's freeze is superseded, not edited. Its records stay as committed, and this
manifest's verifier checks that they do:
- the freeze record `../cp-17/freeze-record.md`, `4a47623718441cca1eb7210dfa3b47dcfcc5c421b8e3a57b858282281b5e0cec`;
- the freeze manifest `../cp-17/freeze-manifest.json`, `4af62410c2d08442b85cd11cd271f0eb367bfb019196784d1e048a46a908a544`,
  policy identity `b304d6479a8628a26014422c6682f1e6543bd0d9ff656ad34b2d87b163bc254e`;
- the 4.7T protocol `../cp-17/4-7t-frozen-protocol.md`, `337a102d6bbbf6d9335fe6ed701a4c80ea1b9172344ffc0f63909b155b567c58`;
- the Integration verdict `../cp-17/integration.md`, `7083bfb7176a9b7fbb4bfc4e03d00317f0c5601ec5360c006de1a2b4e066319d`,
  whose notes N1–N7 this round fixes;
- CP-17's landing record `docs/track-b/cp-17-landing-2026-10-28.md`, `7f90cb87c8d64ec3faecb0b8a45ab2d5e027ace3cc1824785bcbffa5f1ff267b`
  (cited here; it is a governance record, not pinned).

CP-17's own verifier, `cp17.tools.freeze`, stays as landed. CP-17's manifest pins governance documents live (the
anchor, CP-17's brief, decisions 1–12 and the two rulings), so its check is run on a view whose governance documents
are this freeze's frozen copies and whose computational inputs, code and artifact are live
(`cp17.tools.refreeze.cp17_view`). There it reports exactly the two CP-17 package files this round changed,
`src/cp17/issue.py` and `src/cp17/schema.py`, and nothing else (`tests/cp17/test_records.py::test_freeze_manifest_verifies`).
An amendment of the anchor or a new decisions record leaves it green; a changed input or code file still fails it
(`tests/cp17b/test_refreeze.py`). The first Integration review found this check reading the governance documents live,
which r18 would have turned red; that is fixed here
([`integration-attempt-1.md`](integration-attempt-1.md)).

**Authority:**
- `capstone_v21.md` v21-r17 (`3124451d…`; frozen copy `pinned/anchor-capstone-v21-r17.md`): §24.9's "its fallback and
  failure rules", §16.2 and §19.2, as CP-17's brief quoted them;
- the Owner's decisions 1–12 (`d29b0abc…`) and 13–19 (`4d049770…`), frozen copies in `pinned/`;
- the brief `b9aeeee4…` (`issued-brief.md`, and its frozen copy in `pinned/`).

**"Frozen"** keeps CP-17's meaning. After CP-17b lands, any change to the policy, its code, its runtime or its state
needs a new freeze and a new evaluation (§16.2). Daily fitting changes weights and state within the frozen policy,
and is not a change to it.

## 1. The decisions applied (13–18)

| Decision | What it rules | How the freeze implements it |
|---|---|---|
| 13, C2 | A fit that emits every forecastable hour succeeds on those hours, displayed as partial with the missing hours named; a day with no forecastable hour is a failed issuance | the success definition's criteria 1 and 3 (§4 below); `cp17.issue.forecastable_keys`, `check_inputs`, `check_emission`, `partial`; the artifact and the issued forecasts carry the partial record (`schema` v2 kinds) |
| 14, G1 | v4 and v2 issue every day under the issuing host as shadows, never published, to keep their buffers current | `cp17.daily.issue_day`'s shadows and `cp17.shadow.ShadowStates` (§5 below); B0 too, because its buffer from prices alone is not CP-26's |
| 15 | the chains stay as decisions 11 and 12 set them | `cp17.issue.CHAINS`, unchanged |
| 16, C1 | released errors from complete days ≤ D−2; training targets the days before D published strictly before the issue time | the wording, §6 below; the code already followed it (CP-17's parity) |
| 17, G2 | GFS missing in A1's training window fails A1's fit; v4 fails for the same reason, and v2 issues | recorded as the frozen behaviour, §4 below; the landed rule (`uncovered_fails=True`) and v4's weather-row check |
| 18, F1 | disclosed; τ_eq unchanged | §7 below |

Decision 19 is this round, then r18.

## 2. Code and dependencies (§16.2, §19.2): restated

- **The package** is `src/cp17/`'s import closure, measured (`cp17.tools.bundle`): **34 files**, SHA-256 of
  the file list `682c5833fce8fdb5390a49794d83e11a515cb2b89db24d9e9945fcbad714642e`, deterministic zip 91,480 bytes, `41256a486bae64774c74b18a95d85877bc035d61683ffd8a93e5054b7f38a8aa`.
- **What changed in it (item 6), and why:**

  | File | Change | Why |
  |---|---|---|
  | `cp17/issue.py` | the forecastable hours (`forecastable_keys`, by the landed `Inputs.rows`); criterion 1 fails a day with no forecastable hour; criterion 3 on those hours; the partial record and display; the artifact built and verified inside the chain, and a failed issuance written; an issued artifact must carry its measurement; the ceiling's message exact; artifact schema `cp17b-issued-artifact-v2` | decision 13; N3; N2; N7 |
  | `cp17/schema.py` | `fitted_artifact` v2 in two forms (issued with a measurement within the ceiling, or a failed issuance); `issued_forecasts` v2 with partial days and the partial record; `verify_bundle` checks the partial record; `validate` supports `maximum` | N2; decision 13 |
  | `cp17/shadow.py` (new) | the shadow states on CP-26's own classes (`cp16.residuals.SharedResidualState`, `cp26.state.RawResidualState`) | decision 14 |

  No fit, emission, scoring, retrain or attribution module changed: `dailyfit.py`, `scoring.py`, `retrain.py` and
  `attribution.py` are CP-17's bytes, as are the runners `host/node_run.mjs`, `browser/worker.js` and
  `browser/index.html` and the host driver `host/driver.py`.
- **The issuance driver** `cp17.daily` (N1), SHA-256 `18e1589307e26336ed9eb05561b0b17485a065eda9c7da1d75f2940c4dc473be`: frozen with the package (`pipeline_files_sha256`), never
  shipped. It runs natively because it starts the issuing host's processes; it holds no fit, judgement or scoring of
  its own.
- **The native-only tools** in `cp17.tools` changed or were added for this round's exercises and records
  (`bundle`, `refbundle`, `refreeze`, `exercise`, `b0check`, `records17b`, `report17b`). They never ship and are not
  part of the policy.
- **One code path (§19.2).** The daily fit (`dailyfit`), the chain's judgement (`issue`), the shadow states
  (`shadow`) and the scoring (`scoring`) are package files. The issuing host runs them for the product, and the
  browser runs the same files for §19.3 and §19.5. The judgement and the states run in CPython inside `cp17.daily`,
  from the same files the package ships. No second implementation stands in for the product, and no component needs
  §19.3's exception route.
- **Dependencies** are unchanged: the root `uv.lock` `5d2aecc3…` and `pyproject.toml` `aa35326a…`; the pinned runtime's
  packages.
- **The issuing host** is unchanged (decision 10): Pyodide 0.28.0 (its 23 file hashes) in Node.js v26.7.0
  (`1ef99ea2…d040`), one worker.
- **The reference runtime, now in the manifest (N6):** Pyodide 0.28.0; Chromium 153.0.8010.12 and WebKit 26.6 through
  Playwright 1.63.0; Node.js v26.7.0; the Apple M3 machine class, 8 cores, 16 GB, macOS 26.5.2. This round's browser
  run measured those browser versions (chromium 153.0.8010.12, webkit 26.6).

## 3. The governance pins that survive amendment (decision 19; item 4)

**Governance documents are bound by frozen copies** in [`pinned/`](pinned/), each with its SHA-256. `verify` reads
the copies, never the live files.

| Document | Frozen copy | SHA-256 |
|---|---|---|
| the anchor `capstone_v21.md` at v21-r17 (`capstone_v21.md`) | `anchor-capstone-v21-r17.md` | `3124451da84c3cd99fae728c371207aace6ae537474f7769d0ff6333ee53e4fc` |
| CP-17's brief (reissue) (`docs/track-b/evidence/cp-17/issued-brief-v1-9e522b9a.md`) | `cp-17-brief-9e522b9a.md` | `9e522b9a42ea3c02574bd2cde4fad97266bdfd8dc38a104c13566e05c80164fa` |
| CP-17's brief (amended, the one that controlled) (`docs/track-b/evidence/cp-17/issued-brief.md`) | `cp-17-brief-9f235cf8.md` | `9f235cf84876db4975930f1fce8fb62dbe188b8be4130e692eda9b5b7e1b1a73` |
| this brief (`docs/track-b/evidence/cp-17b/issued-brief.md`) | `cp-17b-brief-b9aeeee4.md` | `b9aeeee4131cadd060d35e05d8d0c04f114a8ec764b5c141dc893863fdd75034` |
| the CP-25 rulings (`docs/track-b/cp-25-rulings-2026-10-12.md`) | `cp-25-rulings-2026-10-12.md` | `c7af67f88157277aafd8dd2b1b49e12a1d9ff4b747ca5bdf178fac664fba53c1` |
| the CP-26 GFS ruling (`docs/track-b/cp-26-gfs-ruling-2026-10-19.md`) | `cp-26-gfs-ruling-2026-10-19.md` | `3b97ff6abdf9a30ddd8c950cf3c459301d757aa1af08f6d8121bc73f92ba2373` |
| the Owner's decisions 13–19 (`docs/track-b/cp-17-owner-decisions-2026-10-28.md`) | `cp-17-owner-decisions-2026-10-28.md` | `4d049770e51a37a806c52a4ae681f70aae62ac1847a67c7c0c3109385da3ac37` |
| the Owner's decisions 1–12 (`docs/track-b/pre-cp-17-rulings-2026-10-25.md`) | `pre-cp-17-rulings-2026-10-25.md` | `d29b0abc6a17dcb6bca2cbd464c1dcc5858c50e2b653a42e646d3dc66ea8f6b0` |

**Computational inputs, code and runtimes stay verified live:** the snapshot, CP-25's manifest, CP-26's scores
manifest, CP-28's selection, the tier register, the partitions, the margin tables, the weather features and their
gaps, CP-17-probe's runtime retrieval manifest, CP-15's protocol, `uv.lock`, `pyproject.toml`; every package file,
runner and driver, and the rebuilt zip; the payload schemas; the 23 Pyodide files, the Node.js binary and the search
state `k13.json`. A local runtime or state file present is always checked; on the issuing machine each is required;
elsewhere (CI) an absent one is reported unchecked, never passed.

**The controls** (`tests/cp17b/test_refreeze.py`): a changed live anchor, a changed live decisions record and a new
decisions record leave `verify` green; a changed computational input, code file, schema, runtime file, frozen copy,
CP-17 record or freeze-origin artifact fails it.

## 4. The failure policy and the success definition (decisions 11–13, 15, 17; §16.2): restated

**The chains** are CP-17's (`cp17.issue.CHAINS`; decision 15 keeps them):

| Cause | Chain |
|---|---|
| GFS late or missing at the issue time | A1 recorded as skipped for an unavailable input → **v2** → **B0** → failed issuance |
| A1's own fit fails while the weather is available | **v4** → **v2** → **B0** → failed issuance |
| a fallback's own fit fails | the next one in its chain |

**G2, the frozen behaviour (decision 17).** GFS missing in A1's training window but not at the issue time fails A1's
fit (the landed rule, `uncovered_fails=True`), so the A1-failure chain applies. v4 checks its weather rows and fails
for the same reason, so v2 issues.

**The rules of the chain** stand as CP-17 froze them: A1 issues whenever its own fit succeeds; every fallback is
complete (no subset renormalized, no quantile manufactured); a failed or skipped fit is displayed as such, even when
a fallback issues; a fallback forecast never counts as a successful A1 fit. The GFS-late route's measured excess stays
disclosed: v2 is +2.27% MAE and +6.68% WIS against A1's GFS +6 h route on 1,354 paired affected development dates.

**The success definition.** For D, issued at D−1 11:00 Europe/Berlin, a successful daily fit passes all of:
1. **Inputs:** the payload's identity verifies; the target day's outcome is absent; released errors are from complete
   days ≤ D−2; training targets end before D. **At least one hour of D is forecastable by the frozen design:** a row of
   D is a forecast key when its inputs are valid (`cp15.data.Inputs.rows`, `src/cp15/data.py:79-80`;
   `cp24.member.Keys.from_data`, `src/cp24/member.py:43-45`). A day with no forecastable hour fails here for every
   policy, so it is a failed issuance (decision 13).
2. **Training:** the members train under the frozen rules with no guard refusal and no non-finite value. One failed
   DDNN-2 member fails A1.
3. **Emission:** seven finite, ordered quantiles for **every hour of D the frozen design can forecast**, and only
   those, in order. When they are fewer than the day's 23, 24 or 25 hours, the fit succeeds on those hours and is
   issued and displayed as partial, with the missing hours named (decision 13): for example "successful daily fit
   (partial: 23 of 24 hours; missing 02:00 Europe/Berlin)".
4. **The artifact:** written with its training window, labels, seeds, timestamps and parent, input and state hashes,
   and its partial record, and it verifies. **This is judged inside the chain, before it moves on** (N3): an artifact
   that does not verify, or cannot be written, fails its fit, and the chain moves on.
5. **Resources:** under the issuing host, one worker, ≤ 15 minutes of wall time and ≤ 4 GB peak memory, each attempt
   measured in its own fresh process. **An issued artifact always carries its fit's measurement:** the schema
   (`fitted_artifact` v2) and the verifier refuse one without it, or over the ceiling (N2). The native comparison is
   labelled "not judged: the native comparison path, which can never issue", and cannot become an issued artifact.
   The ceiling's message reports wall time exactly (N7).

Anything else is a failed fit: its forecast is never issued, and the chain moves on. v4, v2 and B0 are judged by the
same criteria with their own frozen training rules.

**The driver (N1).** `cp17.daily.issue_day` runs each attempt in its own fresh issuing-host process, with a fresh
output path and a 900 s timeout. A process that is killed, times out or exits non-zero, or leaves no record, an
unreadable record or another run's record, is a failed fit, and the chain moves on. A failed issuance is written and
verified too.

**Exercised under the issuing host on real development days,** each the same path natively:

| Case | Path | Display | Issued forecast vs native |
|---|---|---|---|
| 2026-03-30, the day after the spring change: 23 of 24 hours forecastable (missing 02:00 Europe/Berlin, 2026-03-30T00:00:00+00:00: its same-hour price lag falls on the change day's absent 02:00) | A1 ok | successful daily fit (partial: 23 of 24 hours; missing 02:00 Europe/Berlin) | different (largest 0.082 EUR/MWh, every selection matched: F1's mechanism, §7); A1 natively bitwise equal to CP-26's D2 |
| 2023-08-31, the load forecast absent on all 24 hours: no hour forecastable | every policy failed at its inputs; no process started | failed issuance (no hour of the day is forecastable by the frozen design) | — (a failed issuance, written and verified) |
| 2026-09-06 with A1's issuing-host process killed after 15 s and a stale A1 record planted (N1) | A1 failed → v4 ok | failed fit; fallback v4 issued | v4's own fit; the stale record never read |
| 2026-09-06, the freeze-origin issuance | A1 ok | successful daily fit | — (§9) |

Records: `reports/cp17b/partial-days.json`, `control.json`, `issuance.json`. The shadows on 2026-03-30 fitted 23 hours each and entered no buffer, as CP-26's did not; on 2023-08-31 none fitted.

## 5. The shadow rule (decision 14): new

- **Every delivery day, whether or not A1 succeeds,** v4, v2 and B0 are fitted under the issuing host as shadow
  issuances, each in its own process (`cp17.daily.issue_day`).
- **A shadow is never published, displayed or scored as the product's forecast.**
- **Its central and scale are registered before the outcome** (`cp17.shadow.ShadowStates.issue`). Each policy's own
  errors from complete delivery days enter its own buffer once, at the first origin after D−2's release
  (`ShadowStates.release`).
- **CP-26's intake rule:** only a complete canonical issuance (23, 24 or 25 keys) with a finite central and, for v4
  and v2, a scale floored at 1 and the single-central identity. A partial day's shadow or a failed shadow enters no
  buffer. The states are CP-26's own classes, so the arithmetic is CP-26's.
- **A fallback the chain fitted is the same fit as its shadow,** fitted once.
- **A shadow passes the same input checks as an attempt:** a day with no forecastable hour fits nothing. On a day
  whose GFS is late or missing at the issue time, v4's shadow is skipped, because its inputs are unavailable.
- **B0 is shadow-issued too.** Decision 14 expected B0's raw-error buffer to come from published prices. The brief
  asked to show that a rebuild so equals CP-26's `RawResidualState` exactly, or else to shadow-issue B0
  (`cp17.tools.b0check`, `reports/cp17b/b0-rebuild.json`):
  the 2,390 days both hold are bit for bit equal, but the price rebuild also takes 36 days CP-26 never took into the buffer (26 on which CP-26 issued 23 of 24 keys, a partial issuance it never takes in; 10 on which CP-26 issued no B0 key, the design having no forecastable hour), so the 28-day buffer differs at 524 of 2,381 origins. They are not equal, so B0 is shadow-issued, as the brief prescribes. It costs a few seconds a day.
- **The initial state** is CP-17's freeze-origin state, carried by reference (§9 below).

**Exercised on 14 consecutive development days** (`cp17.tools.exercise shadow-run`, `reports/cp17b/shadow-run.json`):
- **From CP-26's state at 2026-08-24** (its released buffers and the issuance of 2026-08-23, registered and not yet released), day by day through delivery 2026-09-06, under the issuing host and natively.
- **Natively:** every day's buffers equal CP-26's rebuilt state bitwise; every shadow's quantiles equal CP-26's stored forecast bitwise; the final buffers equal CP-17's recorded freeze-origin state bitwise for v4, v2 and B0.
- **Under the issuing host:** A1 issued every day; every shadow registered; the buffers within rounding of CP-26's (largest standardized difference 2.2e-14), B0's bitwise; the final buffers within that rounding against CP-17's recorded state (B0 bitwise).
- **The measured total per delivery day,** A1 plus the three shadows, each in its own process: median 255 s, largest 288 s, within the 900 s ceiling on every day; peak resident memory of any process 902 MiB. Two days ran at once among other jobs, so these figures are conservative.

## 6. Training windows and label rules (decision 16, C1): restated

The windows stand as CP-17 froze them (its §5). **The labels, in decision 16's wording:**
- **released errors** in every interval buffer come from complete delivery days ≤ D−2, consumed once (§2);
- **training targets** are the delivery days before D whose prices were published strictly before the issue time.
  D−1's prices are published on D−2, so D−1 is a training target.

The landed code and CP-26's parity already followed this; only the wording of CP-17's brief was wrong.

## 7. F1, disclosed (decision 18)

At the 25-hour DST origin 2025-10-26, A1 under the issuing host differs from the native fit by up to 0.028 EUR/MWh
(p50), with every data-driven selection identical. The cause is one configuration whose two members train to the
1,000-epoch cap. τ_eq = 0.01 EUR/MWh stays the visitor's tolerance against the issued artifact, which the browsers
reproduce bit for bit. The native parity with CP-26 stays the check of code identity. This round's rerun reproduces
F1 exactly (§8).

**Observed again, and larger, at CP-17b** (`reports/cp17b/f1-2026-03-30.json`). On item 1's spring day 2026-03-30
(S_12's ensemble), A1 under the issuing host differs from native by up to 0.082 EUR/MWh (p2.5; 0.028 at the
p50), every selection identical. The cause is one member, member 7 (cp26-k12-t096, seed 27290042, 351 epochs, patience), whose own quantiles differ by 0.203 EUR/MWh. The rest agree to about 1e-9 or better. So the mechanism is not
confined to members at the epoch cap, and 0.028 EUR/MWh is not a bound: r18's F1 disclosure should state the larger
figure observed. Nothing is changed: decision 18's reasoning holds, since τ_eq governs the visitor's retrain against
the issued artifact, which is bit for bit.

## 8. The fit is unchanged, shown (item 6)

`schema.py` is a schema module and changed, so CP-17's full native parity and its issuing-host parity were rerun:
- **Native parity, five policies at nine origins:** PASS, every check bitwise against CP-26, and every check, payload identity and quantile hash equal to CP-17's committed record (all equal; `reports/cp17b/native-parity.json`).
- **Issuing-host parity, 15 runs on CP-17b's package:** every issued quantile hash, status, deviation and selection equal to CP-17's record (all equal; F1 reproduced exactly); no non-local request; the executed files are the package's (`reports/cp17b/host-parity.json`).
- **Browser identity, A1 at 2026-09-06:** Chromium and WebKit bit for bit equal to the issuing host, and to CP-17's record (yes).

## 9. The freeze-origin artifact and the initial state: restated

**The freeze origin** is the issuance for delivery 2026-09-06 at 2026-09-05 09:00 UTC.

**The freeze-origin artifact** is the issuing host's, reissued with the new code and citing this freeze's identity:
- `91e00fea32af466127c0e2bd41404259f4652d26acfe9f9aa9f423cb9bc12cd2`, kept in [`artifact-2026-09-06.json`](artifact-2026-09-06.json) (file SHA-256 `c1c3dab694ac828867087d2028c0f5aaeda21b00d58e1e7bf35b2c7c7bbc273d`), with its arrays in the reference bundle;
- displayed as "successful daily fit", the issued policy A1, citing this freeze's policy identity and package;
- issued in 80.4 s of wall time and 795 MiB of peak memory (its own process), within the 15-minute and 4 GB ceilings; with its three shadows, 236 s in all.
- **The reference bundle** in the re-frozen kinds verifies, and its negative controls are refused, a wrong partial record included (`reports/cp17b/reference-bundle.json`; manifest `abb65542fa3a8688…`).

**The initial state is CP-17's, carried by reference.** v4's, v2's and B0's interval states at the freeze origin are
CP-26's final replay state (`d04a4e4d…`), rebuilt from its ledger and checked (`cp17.tools.states`), with the released
buffers recorded by digest in CP-17's manifest (2026-08-08 to 2026-09-04, 28 days each). The shadow run ends at the
freeze origin with exactly those buffers natively, and within rounding under the issuing host (§5).
- **For 4.7T,** the shadow states continue from it on 2026-09-07. The registrations of 2026-09-05 and 2026-09-06 it
  holds are CP-26's, released at the first two window origins.
- **v5's state,** the 4.7T control's, is in it too, unchanged.

## 10. CP-17's sections carried unchanged

These sections of CP-17's freeze record (`4a476237…`) stand as written there:
- **§1,** the designation and its prerequisites;
- **§3,** features and source availability, with the tier, margin and GFS rulings (now pinned by copy, §3 above);
- **§4,** the hyperparameters and search state: S_13's eight members, the landed recipes, the fixed blends;
- **§5's windows** (its labels are restated in §6 above);
- **§7,** randomness and seeds;
- **§8,** the refit schedule. Every delivery day, every policy that issues is refitted, and now the shadows too
  (§5 above); the next search is S_14 = 2027-01-06, a protocol date;
- **§9,** the uncertainty layers;
- **§12,** the registry identities: local names and versions only, nothing uploaded or registered;
- **§14,** §19.5's definitions, §19.3's status rule and τ_eq with its reference runtime, and §19.7's attribution
  methods.

Its §6, §10, §11 and §13 are replaced by §§4, 5, 9 and 11 here, and its §2 by §2 here.

## 11. The resource and calendar plan: restated

- **For 4.7T:** [`4-7t-frozen-protocol.md`](4-7t-frozen-protocol.md) §11, revised only where decisions 13 and 14
  touch the protocol.
- **For daily operation (CP-18):** about 4.2 minutes on the issuing host per delivery day with the shadows (median 255 s measured, largest 288 s), issued at 11:00 local, within the success definition's 15-minute and 4 GB ceilings.

## 12. CP-17's review notes, fixed

| Note | Fix | Where it is shown |
|---|---|---|
| N1, the driver | `cp17.daily` frozen in the identity; one fresh process per attempt, a fresh output path, a 900 s timeout; a killed, timed-out, non-zero, recordless, unreadable or foreign attempt is a failed fit | `tests/cp17b/test_daily.py` (a real killed process, a stale record); `reports/cp17b/control.json` (the issuing host, A1's process killed, a stale record planted) |
| N2, the measurement | the artifact schema's two forms and the verifier refuse an issued artifact without a measurement, or over the ceiling; the native comparison is labelled and never issues | `tests/cp17b/test_units.py` |
| N3, criterion 4 | judged inside the chain, before it moves on | `tests/cp17b/test_units.py` |
| N4, `freeze verify` | recomputes every hash it records (the package rebuilt, the 23 Pyodide files, Node.js, the search state, the schemas, the governance copies), and the manifest's `verification` says how each is checked | `tests/cp17b/test_refreeze.py` |
| N5, the open items | the report lists every open item, F1 included | `reports/cp17b/report.md` |
| N6, the reference runtime | in the manifest, with its browser part, checked against a measured run | the manifest's `policy_identity.reference_runtime` |
| N7, the ceiling's message | reports wall time exactly | `tests/cp17b/test_units.py` |

## 13. Open items

The report lists them all: [`reports/cp17b/report.md`](../../../../reports/cp17b/report.md) § Open items. The return
carries the Owner's summary for r18.
