// The product worker: the pinned Pyodide, the frozen package and the shared glue (src/cp18/spaceglue.py), run in
// the visitor's browser. One worker serves every view. It loads /pyodide/pyodide.js and the same six packages as
// the frozen runner, unpacks /bundle/src.zip to /home/pyodide/src, writes /app/glue/spaceglue.py to
// /home/pyodide/glue/ after checking both against the build record's SHA-256, and then runs spaceglue.<op> on files
// the page has already verified against the manifest. It reports only the stages it reaches.
//
// in:  {id, op: 'boot' | 'recompute' | 'explain' | 'features' | 'retrain_report', files: {name: Uint8Array},
//       expected: {zip_sha256, glue_sha256, packages}, retrained: <JSON text, retrain_report only>}
// out: {id, type: 'stage', stage, ms} | {id, type: 'result', result, ms} | {id, type: 'error', message, stage}
'use strict';
importScripts('/pyodide/pyodide.js');

let booted = null;
const t0 = performance.now();

function hex(buffer) {
  return Array.from(new Uint8Array(buffer)).map(b => b.toString(16).padStart(2, '0')).join('');
}

async function fetchChecked(url, expected, what) {
  let response;
  try { response = await fetch(url); } catch (error) { throw new Error(`${what}: the request for ${url} failed (${error.message})`); }
  if (!response.ok) throw new Error(`${what}: HTTP ${response.status} for ${url}`);
  const bytes = await response.arrayBuffer();
  const got = hex(await crypto.subtle.digest('SHA-256', bytes));
  if (got !== expected) throw new Error(`${what} failed its check: SHA-256 ${got} is not the build record's ${expected}`);
  return bytes;
}

async function boot(id, expected) {
  const post = (stage) => self.postMessage({id, type: 'stage', stage, ms: performance.now() - t0});
  const py = await loadPyodide({indexURL: '/pyodide/'});
  post('runtime loaded');
  await py.loadPackage(expected.packages, {messageCallback: () => {}});
  post('packages loaded');
  const zip = await fetchChecked('/bundle/src.zip', expected.zip_sha256, 'the frozen package zip');
  py.unpackArchive(zip, 'zip', {extractDir: '/home/pyodide/src'});
  const glue = await fetchChecked('/app/glue/spaceglue.py', expected.glue_sha256, 'the shared glue');
  py.FS.mkdirTree('/home/pyodide/glue');
  py.FS.writeFile('/home/pyodide/glue/spaceglue.py', new Uint8Array(glue));
  py.runPython(`
import sys
SRC = '/home/pyodide/src'
sys.path.insert(0, SRC)
try:
    import resource
except ImportError:
    sys.path.insert(0, SRC + '/cp17/stubs/fallback')
    import resource
sys.path.append(SRC + '/cp17/stubs/absent')
sys.path.insert(0, '/home/pyodide/glue')
import json, math, os, shutil
import spaceglue
_explained = {}

def _clean(v):
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if hasattr(v, 'tolist'):
        return _clean(v.tolist())
    return v

def _files():
    folder = '/home/pyodide/in'
    return {n: open(os.path.join(folder, n), 'rb').read() for n in sorted(os.listdir(folder))}

def _run(op, retrained):
    files = _files()
    if op == 'recompute':
        out = spaceglue.recompute(files)
    elif op == 'explain':
        out = spaceglue.explain(files)
        _explained[out['artifact_sha256']] = out
    elif op == 'features':
        sha = json.loads(files['artifact.json'])['artifact_sha256']
        out = spaceglue.features(files, _explained.get(sha))
        out['explained_reused'] = sha in _explained
    elif op == 'retrain_report':
        out = spaceglue.retrain_report(json.loads(files['artifact.json']), json.loads(retrained))
    else:
        raise ValueError('unknown op ' + op)
    return json.dumps(_clean(out), allow_nan=False)
`);
  post('package unpacked');
  return py;
}

self.onmessage = async (event) => {
  const {id, op, files, expected, retrained} = event.data;
  let stage = 'starting';
  try {
    if (!booted) booted = boot(id, expected);
    let py;
    try {
      py = await booted;
    } catch (error) {
      booted = null;
      throw error;
    }
    if (op === 'boot') {
      self.postMessage({id, type: 'result', result: {booted: true}, ms: performance.now() - t0});
      return;
    }
    stage = 'computing';
    self.postMessage({id, type: 'stage', stage, ms: performance.now() - t0});
    try { py.FS.mkdirTree('/home/pyodide/in'); } catch (e) { /* exists */ }
    for (const name of py.FS.readdir('/home/pyodide/in')) {
      if (name !== '.' && name !== '..') py.FS.unlink('/home/pyodide/in/' + name);
    }
    for (const [name, bytes] of Object.entries(files || {})) {
      if (!/^[A-Za-z0-9._-]+$/.test(name)) throw new Error('unexpected file name ' + name);
      py.FS.writeFile('/home/pyodide/in/' + name, bytes);
    }
    py.globals.set('OP', op);
    py.globals.set('RETRAINED', retrained || null);
    const text = await py.runPythonAsync('_run(OP, RETRAINED)');
    self.postMessage({id, type: 'stage', stage: 'done', ms: performance.now() - t0});
    self.postMessage({id, type: 'result', result: JSON.parse(text), ms: performance.now() - t0});
  } catch (error) {
    self.postMessage({id, type: 'error', stage,
                      message: String(error && error.message ? error.message : error)});
  }
};
