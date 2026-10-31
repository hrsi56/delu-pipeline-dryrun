// CP-17's "Train it yourself" worker: load the pinned Pyodide and its packages, unpack the frozen package,
// write the day's training-window payload into the in-memory file system, and run the package's own host
// driver (`cp17/host/driver.py`) in fit mode -- the same file the issuing host runs in Node.js. Times are
// performance.now() marks; memory is the WebAssembly linear memory, which only grows, so its final size is
// the run's peak.
importScripts('/pyodide/pyodide.js');


self.onmessage = async (event) => {
  const {stem, packages, attribution} = event.data;
  const t0 = performance.now();
  const out = {stem, marks: {}, errors: []};
  try {
    const py = await loadPyodide({indexURL: '/pyodide/'});
    out.marks.runtime_ms = performance.now() - t0;
    out.pyodide_version = py.version;
    await py.loadPackage(packages.split(','), {messageCallback: () => {}});
    out.marks.packages_ms = performance.now() - t0;
    const bundle = await (await fetch('/bundle/src.zip')).arrayBuffer();
    py.unpackArchive(bundle, 'zip', {extractDir: '/home/pyodide/src'});
    for (const ext of ['npz', 'json']) {
      const response = await fetch(`/payloads/${stem}.${ext}`);
      if (!response.ok) throw new Error(`payload ${stem}.${ext}: HTTP ${response.status}`);
      py.FS.writeFile(`/home/pyodide/${stem}.${ext}`, new Uint8Array(await response.arrayBuffer()));
    }
    out.marks.payload_ms = performance.now() - t0;
    py.globals.set('MODE', 'fit');
    py.globals.set('STEM', stem);
    py.globals.set('POLICY', stem.split('-')[0]);
    py.globals.set('ATTRIBUTION', attribution === '1');
    const driver = py.FS.readFile('/home/pyodide/src/cp17/host/driver.py', {encoding: 'utf8'});
    const text = await py.runPythonAsync(driver);
    out.marks.fit_done_ms = performance.now() - t0;
    out.python = JSON.parse(text);
    out.wasm_memory_bytes = py._module.HEAP8.buffer.byteLength;
  } catch (error) {
    // WebKit's stack omits the message; Pyodide's message carries the Python traceback.
    out.errors.push(String(error && error.message ? error.message : error) + '\n' + String((error && error.stack) || ''));
  }
  out.total_ms = performance.now() - t0;
  self.postMessage(out);
};
