// CP-17's issuing host: the pinned Pyodide runtime in Node.js, running the frozen package (the Owner's
// decision 10). The same WebAssembly runtime the browser runs, so a visitor's retrain can be compared bit
// for bit with the issued artifact.
//
// Everything is read from local files. Any network call is refused and recorded: fetch, http(s) and raw
// sockets are replaced before Pyodide loads, and only file paths and file: URLs pass.
//
// An issuance runs each attempt in its own fresh process in fit mode, so that its wall time and peak memory
// (the success definition's criterion 5) are its own; `cp17.issue` then judges the attempts.
//
//   node node_run.mjs fit   <pyodide dir> <package zip> <payload dir> <stem> <attribution 0|1> <packages> <out.json>
//   node node_run.mjs score <pyodide dir> <package zip> <input dir> <input.json> 0 <packages> <out.json>
// CP17_EXPORT=1 makes fit mode write the fitted arrays to <out>-arrays.npz.
import {readFileSync, readdirSync, writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';
import {join, resolve} from 'node:path';
import {performance} from 'node:perf_hooks';

const require = createRequire(import.meta.url);
const [mode, pyodideDir, packageZip, payloadDir, target, flag, packages, outPath] = process.argv.slice(2);
const blocked = [];
const originalFetch = globalThis.fetch;
globalThis.fetch = async (input, init) => {
  const url = String(input && input.url ? input.url : input);
  if (url.startsWith('file:') || url.startsWith('/') || url.startsWith('.')) return originalFetch(input, init);
  blocked.push(url.slice(0, 200));
  throw new Error(`network request refused: ${url.slice(0, 200)}`);
};
for (const name of ['http', 'https', 'net', 'tls']) {
  const mod = require(name);
  for (const fn of ['request', 'get', 'connect', 'createConnection']) {
    if (typeof mod[fn] === 'function') {
      mod[fn] = (...args) => { blocked.push(`${name}.${fn}`); throw new Error(`network call refused: ${name}.${fn}`); };
    }
  }
}

const t0 = performance.now();
const out = {host: 'node', node: process.version, node_execpath: process.execPath, mode, target, marks: {}, errors: []};
try {
  const {loadPyodide} = await import(join(resolve(pyodideDir), 'pyodide.mjs'));
  const py = await loadPyodide({indexURL: resolve(pyodideDir) + '/'});
  out.pyodide_version = py.version;
  out.marks.runtime_ms = performance.now() - t0;
  await py.loadPackage(packages.split(','), {messageCallback: () => {}});
  out.marks.packages_ms = performance.now() - t0;
  py.unpackArchive(new Uint8Array(readFileSync(packageZip)), 'zip', {extractDir: '/home/pyodide/src'});
  if (mode === 'score') {
    py.FS.writeFile('/home/pyodide/score-input.json', new Uint8Array(readFileSync(join(payloadDir, target))));
  } else {
    const day = target.split('-').slice(1).join('-');
    for (const name of readdirSync(payloadDir)) {
      if (name.endsWith(`-${day}.npz`) || name.endsWith(`-${day}.json`)) {
        py.FS.writeFile(`/home/pyodide/${name}`, new Uint8Array(readFileSync(join(payloadDir, name))));
      }
    }
  }
  out.marks.payload_ms = performance.now() - t0;
  py.globals.set('MODE', mode);
  if (mode === 'fit') {
    py.globals.set('STEM', target);
    py.globals.set('POLICY', target.split('-')[0]);
    py.globals.set('ATTRIBUTION', flag === '1');
    py.globals.set('EXPORT', process.env.CP17_EXPORT === '1');
  }
  const driver = py.FS.readFile('/home/pyodide/src/cp17/host/driver.py', {encoding: 'utf8'});
  out.python = JSON.parse(await py.runPythonAsync(driver));
  out.marks.done_ms = performance.now() - t0;
  if (mode === 'fit' && py.FS.analyzePath('/home/pyodide/fit-arrays.npz').exists) {
    writeFileSync(outPath.replace(/\.json$/, '') + '-arrays.npz', py.FS.readFile('/home/pyodide/fit-arrays.npz'));
  }
  out.wasm_memory_bytes = py._module.HEAP8.buffer.byteLength;
} catch (error) {
  out.errors.push(String(error && error.message ? error.message : error));
}
out.total_ms = performance.now() - t0;
out.process_rss_peak_bytes = process.resourceUsage().maxRSS * 1024;
out.blocked_non_local_requests = blocked;
writeFileSync(outPath, JSON.stringify(out, null, 1) + '\n');
console.log(mode, target, out.errors.length ? 'FAILED ' + out.errors[0].slice(0, 400) : 'ok', (out.total_ms / 1000).toFixed(1) + 's');
process.exit(out.errors.length ? 1 : 0);
