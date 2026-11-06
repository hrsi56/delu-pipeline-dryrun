# delu-pipeline-dryrun

Public scratch repository for CP-18a and CP-18c of the DE-LU day-ahead forecasting project
([hrsi56/delu-day-ahead-forecast](https://github.com/hrsi56/delu-day-ahead-forecast)): dry runs of the daily pipeline
around the frozen A1 package on GitHub-hosted `macos-26` runners, on development days through 2026-09-06 only. The
workflow's live jobs run only in the product repository; here, only its dry run is dispatched.

It holds only what the runs need: the candidate's code and the files its freeze manifest verifies, the guards, the
workflow, A1's search state, the freeze-origin states, the canary's payload, a development state for the store and the
recorded responses the pipeline replays. `scratch-files.json` lists every file's SHA-256. Pyodide's files and the
Node.js binary are retrieved by each run from their official URLs and checked by hash.

It is not a publication: it has no Pages site, no secret and no variable, and its Git history is not the product.
Data: ENTSO-E Transparency Platform; Bundesnetzagentur | SMARD.de - CC BY 4.0. NOAA GFS (public domain).
