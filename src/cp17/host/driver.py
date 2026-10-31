"""The Python every Pyodide host executes: the issuing host (Node.js) and the browser load this file as text
from the unpacked package, so its hash is in the package manifest like every other shipped file.

Globals set by the host: MODE ('fit' or 'score'), STEM, POLICY, ATTRIBUTION and EXPORT. The payload files are at
/home/pyodide/<policy>-<day>.npz and .json. The last expression is the run as JSON.

- **fit**: one policy's daily fit from its payload, as `cp17.dailyfit.run`. The issuing host runs each attempt of
  an issuance in its own fresh process in this mode, so that each fit's wall time and peak memory are its own
  (the success definition's criterion 5); `cp17.issue` then judges the attempts and walks the chain. With EXPORT,
  the fitted arrays (A1's members' parameters) are written to /home/pyodide/fit-arrays.npz for the artifact.
- **score**: §19.5's evaluation record (`cp17.scoring.evaluate`) of /home/pyodide/score-input.json, as the
  Space recomputes it.
"""
import hashlib, json, os, sys, time
SRC = '/home/pyodide/src'
sys.path.insert(0, SRC)
stubs = []
try:
    import resource
except ImportError:
    sys.path.insert(0, SRC + '/cp17/stubs/fallback')
    import resource
    stubs.append('resource')
sys.path.append(SRC + '/cp17/stubs/absent')
import numpy as np
from cp17 import dailyfit


def _paths(stem):
    npz = f'/home/pyodide/{stem}.npz'
    return (npz if os.path.exists(npz) else None), f'/home/pyodide/{stem}.json'


started = time.perf_counter()
if MODE == 'fit':
    npz, meta = _paths(STEM)
    result = dailyfit.run(POLICY, npz, meta, attribution=bool(ATTRIBUTION), export=bool(globals().get('EXPORT')))
    fitted = result.pop('_arrays', None)
    if fitted:
        dailyfit.save_npz('/home/pyodide/fit-arrays.npz', fitted)
    out = {'result': result, 'exported_arrays': sorted(fitted or {})}
elif MODE == 'score':
    from cp17 import scoring
    with open('/home/pyodide/score-input.json', encoding='utf-8') as handle:
        item = json.load(handle)
    outcomes = {k: (float('nan') if v is None else v) for k, v in item['outcomes'].items()}
    out = {'records': {name: scoring.evaluate(issued, outcomes) for name, issued in item['windows'].items()}}
else:
    raise ValueError(f'unknown MODE {MODE}')
seconds = time.perf_counter() - started
loaded = sorted(n for n, m in sys.modules.items() if getattr(m, '__file__', None) and m.__file__.startswith(SRC + '/'))
files = {}
for name in loaded:
    path = sys.modules[name].__file__
    with open(path, 'rb') as handle:
        files[os.path.relpath(path, SRC)] = hashlib.sha256(handle.read()).hexdigest()
with open(SRC + '/cp17/host/driver.py', 'rb') as handle:
    files['cp17/host/driver.py'] = hashlib.sha256(handle.read()).hexdigest()
if 'holidays' in sys.modules and getattr(sys.modules['holidays'], '__file__', '').startswith(SRC + '/'):
    stubs.append('holidays')
import pandas, platform
out.update({'seconds': seconds, 'stubs': stubs, 'source_sha256': files, 'python': sys.version.split()[0],
            'numpy': np.__version__, 'pandas': pandas.__version__, 'platform': platform.platform()})
json.dumps(out, allow_nan=False)
