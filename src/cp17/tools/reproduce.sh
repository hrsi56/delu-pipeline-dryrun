#!/bin/zsh
# CP-17's records from a checkout, end to end, with no network contact (brief items 3-7):
#   1. the frozen package; 2. native parity at the nine origins (A1, v5, v4, v2, B0); 3. the policy identity;
#   4. the issuing host (A1 at nine origins; v4, v2 and B0 at two); 5. the issuances (the freeze-origin
#   artifact, GFS late, and the A1 failure of 2022-12-04); 6. the retrain controls; 7. the scoring check;
#   8. the comparison table; 9. the initial state, the reference bundle and the freeze manifest;
#   10. Chromium and WebKit (A1 and v4 at 2026-09-06), run last and alone, then the table again.
#
#   src/cp17/tools/reproduce.sh <project root holding .local/> <output folder>
# Run from the checkout. STEPS narrows it (default: all), e.g. STEPS="browsers compare".
set -euo pipefail
PROJECT=$1; OUT=$2
STEPS=${STEPS:-"package native identity hosts issuance controls scoring compare freeze browsers"}
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MLFLOW_DISABLE_TELEMETRY=true DO_NOT_TRACK=1 PYTHONPATH=src
PY=$PROJECT/.venv/bin/python
PW=$PROJECT/.local/tools/playwright/bin/python
PYODIDE=$PROJECT/.local/tools/pyodide-0.28.0
ORIGINS="2025-10-26 2026-03-29 2026-08-31 2026-09-01 2026-09-02 2026-09-03 2026-09-04 2026-09-05 2026-09-06"
mkdir -p $OUT
step() { [[ " $STEPS " == *" $1 "* ]]; }
if step package; then $PY -m cp17.tools.bundle --out $OUT/bundle/src.zip; fi
if step native; then $PY -m cp17.tools.parity --out $OUT/native; fi
if step identity; then
  $PY -m cp17.tools.freeze identity --bundle-manifest $OUT/bundle/src.json \
      --protocol-from $OUT/native/payloads/v2-2026-09-06.json --out $OUT/policy-identity.json
fi
if step hosts; then
  RUNS=$(for d in ${=ORIGINS}; do printf 'A1-%s,' $d; done)v4-2026-09-06,v4-2026-03-29,v2-2026-09-06,v2-2026-03-29,B0-2026-09-06,B0-2026-03-29
  $PY -m cp17.tools.hosts node --bundle $OUT/bundle/src.zip --payloads $OUT/native/payloads --out $OUT/nodes --runs $RUNS \
      --attribution A1-2026-09-06,A1-2026-03-29,v4-2026-09-06,v4-2026-03-29,v2-2026-09-06,v2-2026-03-29 --workers 2
fi
if step issuance; then
  for args in "2026-09-06 1 native/payloads" "2026-09-06 0 native/payloads" "2022-12-04 1 payloads-2022-12-04"; do
    set -- ${=args}
    mkdir -p $OUT/$3
    $PY -m cp17.tools.issuance --day $1 --gfs $2 --payloads $OUT/$3 --bundle $OUT/bundle/src.zip \
        --identity $OUT/policy-identity.json --out $OUT/issuance
  done
fi
if step controls; then
  $PY -c "import json,sys; r=json.load(open(sys.argv[1]))['python']['result']; json.dump(r, open(sys.argv[2],'w'))" \
      $OUT/nodes/node-A1-2026-09-06.json $OUT/issued-A1-2026-09-06.json
  $PY -m cp17.tools.controls --payloads $OUT/native/payloads --issued $OUT/issued-A1-2026-09-06.json --out $OUT/controls.json
fi
if step scoring; then $PY -m cp17.tools.scorecheck --bundle $OUT/bundle/src.zip --out $OUT/score --record $OUT/scoring-check.json; fi
if step compare; then
  BROWSER=(); [[ -f $OUT/browser-results.json ]] && BROWSER=(--browser $OUT/browser-results.json)
  $PY -m cp17.tools.hosts compare --native $OUT/native/native-parity.json --nodes $OUT/nodes --bundle-manifest $OUT/bundle/src.json \
      --payloads $OUT/native/payloads $BROWSER --out $OUT/host-parity.json
fi
if step freeze; then
  $PY -m cp17.tools.freeze states --out $OUT/states.json
  $PY -m cp17.tools.refbundle --issuance $OUT/issuance/issuance-2026-09-06-gfs1.json --artifact-dir $OUT/issuance \
      --payloads $OUT/native/payloads --identity $OUT/policy-identity.json --out $OUT/reference-bundle \
      --record $OUT/reference-bundle.json
  $PY -m cp17.tools.freeze manifest --identity $OUT/policy-identity.json --issuance $OUT/issuance/issuance-2026-09-06-gfs1.json \
      --bundle $OUT/reference-bundle/manifest.json --states $OUT/states.json --out $OUT/freeze-manifest.json
fi
if step browsers; then
  PLAYWRIGHT_BROWSERS_PATH=$PROJECT/.local/tools/ms-playwright $PW src/cp17/tools/measure.py --pyodide $PYODIDE \
      --bundle $OUT/bundle --payloads $OUT/native/payloads --out $OUT/browser-results.json \
      --runs ${RUNS_BROWSER:-A1-2026-09-06,v4-2026-09-06} --browsers ${BROWSERS:-chromium,webkit} --attribution 1
  $PY -m cp17.tools.hosts compare --native $OUT/native/native-parity.json --nodes $OUT/nodes --bundle-manifest $OUT/bundle/src.json \
      --payloads $OUT/native/payloads --browser $OUT/browser-results.json --out $OUT/host-parity.json
fi
