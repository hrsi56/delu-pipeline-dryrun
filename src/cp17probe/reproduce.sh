#!/bin/zsh
# The CP-17-probe measurement from a checkout, end to end, with no network contact:
#   1. the native reference (payloads, native fits, the parity check against CP-26's stored forecasts);
#   2. the source bundle; 3. Chromium and WebKit; 4. Node.js under the same Pyodide; 5. the table.
#
#   src/cp17probe/reproduce.sh <project root holding .local/> <output folder>
# Run from the checkout. BROWSERS and RUNS narrow step 3 (defaults: chromium,webkit and all three runs).
set -euo pipefail
PROJECT=$1; OUT=$2
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MLFLOW_DISABLE_TELEMETRY=true DO_NOT_TRACK=1 PYTHONPATH=src
PY=$PROJECT/.venv/bin/python
PW=$PROJECT/.local/tools/playwright/bin/python
PYODIDE=$PROJECT/.local/tools/pyodide-0.28.0
CP26=$PROJECT/.local/artifacts/cp-26/cold-rerun-2026-10-15
mkdir -p $OUT
$PY -m cp17probe.native --root . --inputs $PROJECT/.local/artifacts/cp-17-probe/inputs \
    --fits $CP26/fits/195857d79c60b5573ae2238989be1f3042d71a67df9831fc36471a81e9736f3b/nominal \
    --state $CP26/state/e17f95f72a67b8213d6faa7b5105bf325756b01c14eba071aa82c6c4357f4b85/replay-state.json --out $OUT
$PY -m cp17probe.bundle --src src --out $OUT/bundle/src.zip
PLAYWRIGHT_BROWSERS_PATH=$PROJECT/.local/tools/ms-playwright $PW src/cp17probe/measure.py --pyodide $PYODIDE \
    --bundle $OUT/bundle --payloads $OUT/payloads --out $OUT/browser-results.json \
    --browsers ${BROWSERS:-chromium,webkit} --runs ${RUNS:-A1-2026-09-06,A1-2026-03-29,v2-2026-09-06}
NODES=()
for stem in A1-2026-09-06 A1-2026-03-29 v2-2026-09-06; do
  packages=numpy,pandas,scikit-learn,requests,tzdata
  [[ $stem == v2-* ]] && packages=$packages,lightgbm
  node src/cp17probe/node_run.mjs $PYODIDE $OUT/bundle/src.zip $OUT/payloads $stem $packages $OUT/node-$stem.json
  NODES+=($OUT/node-$stem.json)
done
$PY -m cp17probe.compare --native $OUT/native-reference.json --browser $OUT/browser-results.json \
    --bundle-manifest $OUT/bundle/src.json --pyodide $PYODIDE --bundle $OUT/bundle --payloads $OUT/payloads \
    --out $OUT/measurements.json --node $NODES
