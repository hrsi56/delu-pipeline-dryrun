"""The Python run every host executes: the browser worker and the Node.js runner load this file as text.

It expects the globals POLICY and STEM, and the payload files at /home/pyodide/<STEM>.npz and .json.
Its last expression is the JSON text of the run.
"""
import hashlib, json, os, sys, time
sys.path.insert(0, '/home/pyodide/src')
stubs = []
try:
    import resource
except ImportError:
    sys.path.insert(0, '/home/pyodide/src/cp17probe/stubs/fallback')
    import resource
    stubs.append('resource')
sys.path.append('/home/pyodide/src/cp17probe/stubs/absent')
import numpy as np
from cp17probe import dailyfit
started = time.perf_counter()
result = dailyfit.run(POLICY, '/home/pyodide/' + STEM + '.npz', '/home/pyodide/' + STEM + '.json', attribution=POLICY == 'A1')
fit_seconds = time.perf_counter() - started
loaded = sorted(n for n, m in sys.modules.items() if getattr(m, '__file__', None) and m.__file__.startswith('/home/pyodide/src/'))
files = {}
for name in loaded:
    path = sys.modules[name].__file__
    with open(path, 'rb') as handle:
        files[os.path.relpath(path, '/home/pyodide/src')] = hashlib.sha256(handle.read()).hexdigest()
if 'holidays' in sys.modules and getattr(sys.modules['holidays'], '__file__', '').startswith('/home/pyodide/src/'):
    stubs.append('holidays')
import pandas, platform
json.dumps({'result': result, 'fit_seconds': fit_seconds, 'stubs': stubs, 'source_sha256': files,
            'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pandas.__version__,
            'platform': platform.platform()}, allow_nan=False)
