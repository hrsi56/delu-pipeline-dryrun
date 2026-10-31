"""CP-18a item 2: the live-input payload builder (Track B, DE-LU day-ahead).

It turns retrieved source data into the frozen payload kinds `cp17.daily.issue_day` takes (A1, v4, v2, B0),
sets each day's GFS flag under the tier register and the Owner's tier, margin and GFS rulings, and keeps an
archive that makes every input's availability verifiable against D's cutoff (D-1 11:00 Europe/Berlin, strictly
before). **The only red line on this path is `cp25.guard`**: retiring its boundary is the only change needed
for days after 2026-09-06. (Ruling 1's exception keeps its own scope, the stored development history.)

Layout:

| Module | What it holds |
|---|---|
| `transport` | `Request` (scope + `admit()` through `cp25.guard`), `Response`, `ReplayTransport` (the only transport used in CP-18a), `ReplayFolder`, `HttpTransport` (live; guard before any socket; never instantiated in CP-18a outside its refusal test) |
| `smard` | SMARD `chart_data` client (filter 4169 price, 411 load forecast), the snapshot's aggregation rules |
| `entsoe` | ENTSO-E A44 / A65 client; the token is a variable name the live transport reads inside `send` |
| `gfs` | GFS client: AWS listing (`LastModified`), `.idx`, byte ranges, checks; CP-20's decoder as a subprocess |
| `decode_worker` | the decoder subprocess (ecCodes environment only) |
| `archive` | the per-day input archive and the writers that turn client results into records |
| `availability` | the cutoff, the publication bases each tier may rest on, ruling 1's scope, the judgement |
| `history` | the committed base through `base_end` + the archive's increments to D; the GFS flag per day |
| `payloads` | the context from landed building blocks and the four payload kinds (`cp17.tools.payloads`' logic) |
| `build` | `build_day` and `published_prices` |
| `recorded` | development responses (rendered SMARD/ENTSO-E; recorded GFS), the parse check, dry-run inputs |
| `equality` | the 9 + 14 byte-equality run against the development builder |

**The contract** (`cp18.live.build.build_day`)::

    build_day(day: date, *, archive: Archive | str | Path, base_end: date,
              buffers: {'v4': list, 'v2': list, 'B0': list}, search_dir: str | Path,
              out: str | Path, root: str | Path) -> dict

* `archive` - the input archive root (`root/days/<D>/manifest.json`); only archive days <= D are read.
* `base_end` - the last day of the committed base (`data/snapshot.parquet` and the committed weather features);
  every day in (base_end, D] comes from the archive. `base_end < day`, both admitted by `cp25.guard`.
* `buffers` - the released buffers the shadow states hold after D-2's release
  (`cp17.shadow.ShadowStates.buffer(p)` for p in v4, v2, B0): 28 days each, all <= D-2.
* `search_dir` - the folder holding A1's search states `k{NN}.json`; the one used is
  `cp26.search.serving_search(day)`, validated by `cp26.search.validate_ensemble(..., 'nominal')`.
* `out` - where `A1-<D>.{npz,json}`, `v4-<D>...`, `v2-<D>...`, `B0-<D>...` are written.
* `root` - the repository root (committed snapshot, weather features, `reports/cp15/protocol.json`).

It raises only `cp25.guard.BoundaryRefused` (day or base_end past the boundary, before anything is read); every
other failure is a result. The returned dict (JSON-able; `build.write_record` writes it):

| Key | Value |
|---|---|
| `schema` | `'cp18-live-build-v1'` |
| `day` | ISO date |
| `status` | `'built'`, `'search_state_missing'` (the pipeline's "S_14's state not in place"; nothing built), `'not_shown'` (the pipeline's "a live day's inputs that cannot be shown to precede its cutoff"; nothing built), or `'failed'` |
| `cause` | `None` when built; otherwise the reason (redacted) |
| `payloads` | `{policy: {'meta': path, 'npz': path, 'payload_sha256', 'npz_bytes', 'meta_bytes'}}` for A1, v4, v2, B0 when built |
| `gfs_available` | the flag to pass to `cp17.daily.issue_day` (payloads are built either way; on False D's weather is missing under §15.3) |
| `gfs` | D's flag record: `day`, `run`, `available`, `status`, `reason`, `vector_valid`, `judgement` |
| `availability` | `shown` (every required input shown, or admitted under a recorded exception valid for that day), `reasons`, `by_exception` (inputs admitted under ruling 1), `records` (one judgement per input and data day: `input`, `tier`, `data_day`, `for_day`, `cutoff_utc`, `archive_day`, `record`, `sha256`, `provenance`, `publication_utc`, `basis_id`, `basis`, `admitted`, `by_exception`, `reason`, `required`), `base` (what the committed base is admitted as) |
| `search` | `k`, `path`, `present`, `search_day`, `content_sha256`, `file_sha256` |
| `history` | `base_end`, `increment_days`, `frame_rows`, `weather_rows` |
| `problems` | non-blocking findings (partial bins, disagreeing price records, ...) |
| `timings` | `history_seconds`, `context_seconds`, `total_seconds` |
| `traceback` | only when `failed` (redacted) |

The required inputs for `shown` are the prices of every increment day in (base_end, D-1] and the load forecast
of D. The search state is checked before `shown` decides, so `search_state_missing` is reported first; the
availability record is filled either way.

`published_prices(day, *, archive, base_end, root) -> truth(keys) -> array` gives
`ShadowStates.release(day, truth)` the published prices of delivery days <= D-1 only, NaN unless every key
is present.
"""
