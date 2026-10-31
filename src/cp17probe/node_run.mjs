// CP-17-probe: the same daily fit in the same WebAssembly runtime, hosted by Node.js instead of a browser.
//
// It measures the one-code-path option (§19.2; Space plan §8): if the issuing pipeline ran its daily fit
// under the pinned Pyodide, would a visitor's browser retrain reproduce the issued artifact bit for bit?
// Everything is read from local files; there is no network contact.
//
//   node src/cp17probe/node_run.mjs <pyodide dir> <bundle zip> <payload dir> <stem> <packages> <out.json>
import {readFileSync, writeFileSync} from 'node:fs';
import {dirname, join, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {performance} from 'node:perf_hooks';

const [pyodideDir, bundleZip, payloadDir, stem, packages, outPath] = process.argv.slice(2);
const here = dirname(fileURLToPath(import.meta.url));
const {loadPyodide} = await import(join(resolve(pyodideDir), 'pyodide.mjs'));
const t0 = performance.now();
const out = {host: 'node', node: process.version, stem, marks: {}, errors: []};
try {
  const py = await loadPyodide({indexURL: resolve(pyodideDir) + '/'});
  out.pyodide_version = py.version;
  out.marks.runtime_ms = performance.now() - t0;
  await py.loadPackage(packages.split(','), {messageCallback: () => {}});
  out.marks.packages_ms = performance.now() - t0;
  py.unpackArchive(new Uint8Array(readFileSync(bundleZip)), 'zip', {extractDir: '/home/pyodide/src'});
  for (const ext of ['npz', 'json']) {
    py.FS.writeFile(`/home/pyodide/${stem}.${ext}`, new Uint8Array(readFileSync(join(payloadDir, `${stem}.${ext}`))));
  }
  py.globals.set('POLICY', stem.split('-')[0]);
  py.globals.set('STEM', stem);
  out.python = JSON.parse(await py.runPythonAsync(readFileSync(join(here, 'browser', 'driver.py'), 'utf8')));
  out.marks.fit_done_ms = performance.now() - t0;
  out.wasm_memory_bytes = py._module.HEAP8.buffer.byteLength;
} catch (error) {
  out.errors.push(String(error && error.message ? error.message : error));
}
out.total_ms = performance.now() - t0;
out.process_rss_bytes = process.memoryUsage().rss;
writeFileSync(outPath, JSON.stringify(out, null, 1) + '\n');
console.log(stem, out.errors.length ? 'FAILED ' + out.errors[0].slice(0, 300) : 'ok', (out.total_ms / 1000).toFixed(1) + 's');
process.exit(out.errors.length ? 1 : 0);
