"""The frozen package as the Pyodide hosts load it: the import closure of `src/cp17`'s product modules, byte
for byte, the host driver and the two stand-ins, in one deterministic zip with a manifest of file hashes.

The file list is measured, not declared: a fresh native interpreter imports the entry modules, and every
loaded module whose file lies under `src/` ships (`cp17.tools` is never imported, so it never ships). The zip
has sorted names and fixed timestamps; the hosts record the hash of every file they executed, which the
comparison checks against this manifest.

    PYTHONPATH=src python -m cp17.tools.bundle --out <folder>/src.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ENTRY = ('cp17.dailyfit', 'cp17.attribution', 'cp17.retrain', 'cp17.scoring', 'cp17.issue', 'cp17.schema', 'cp17.shadow',
         'cp24.design', 'cp24.ddnn2', 'cp24.member', 'cp15.data', 'cp15.models', 'cp21.lgbm', 'cp16.residuals',
         'cp26.state', 'cp25.guard', 'cp28.scoring', 'delu_forecast.baselines')
EXTRA = ('cp17/host/driver.py', 'cp17/stubs/absent/holidays.py', 'cp17/stubs/fallback/resource.py')
from ..daily import PACKAGES  # noqa: E402  (the pinned Pyodide packages the closure needs)
FIXED_TIME = (2026, 9, 6, 0, 0, 0)
#: The runners that load the package: the issuing host's Node.js script and the browser page and worker. They
#: run outside Pyodide, so they are not in the zip; their hashes are frozen with it.
RUNNERS = ('cp17/host/node_run.mjs', 'cp17/browser/worker.js', 'cp17/browser/index.html')
#: The native pipeline modules that start the issuing host (CP-17's review, N1), its pins and the checks before a
#: run's first issuance (CP-17c, r19 §24.13): frozen with the package, never shipped to the browser.
PIPELINE = ('cp17/daily.py', 'cp17/runtime.py', 'cp17/hostcheck.py')

#: The closure is measured by importing the entry modules natively. LightGBM's native library needs Homebrew's
#: libomp, which a GitHub-hosted macos-26 runner lacks (CP-17c). Where it cannot load, a placeholder stands in for
#: its names, and the measurement says so (`stubbed`): the closure is the set of `src/` files the entry modules
#: import, which LightGBM's internals never change, and `refreeze verify` still compares the rebuilt files, package
#: identity and zip with the frozen ones, bit for bit.
PROBE = """
import importlib, json, sys, types
from pathlib import Path
src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src))
stubbed = []
try:
    import lightgbm
except OSError:
    stub = types.ModuleType('lightgbm')
    def placeholder(name):
        if name.startswith('__'):
            raise AttributeError(name)
        return type(name, (), {})
    stub.__getattr__ = placeholder
    sys.modules['lightgbm'] = stub
    stubbed.append('lightgbm')
for name in sys.argv[2:]:
    importlib.import_module(name)
files = sorted({str(Path(m.__file__).resolve().relative_to(src)) for m in list(sys.modules.values())
                if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(src)})
third = sorted({n.split('.')[0] for n, m in sys.modules.items() if getattr(m, '__file__', None)
                and 'site-packages' in str(m.__file__)})
print(json.dumps({'files': files, 'third_party': third, 'stubbed': stubbed}))
"""


def closure(src: Path) -> dict:
    out = subprocess.check_output([sys.executable, '-c', PROBE, str(src), *ENTRY], env={'PATH': '/usr/bin:/bin'}, text=True)
    return json.loads(out)


def build(src: Path, out: Path) -> dict:
    src, out = Path(src).resolve(), Path(out)
    found = closure(src)
    if any(f.startswith('cp17/tools/') for f in found['files']):
        raise ValueError('a native-only driver entered the shipped closure')
    names = sorted(set(found['files']) | set(EXTRA))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in names:
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (src / name).read_bytes())
    files = {n: hashlib.sha256((src / n).read_bytes()).hexdigest() for n in names}
    manifest = {'schema': 'cp17-bundle-v1', 'entry_modules': list(ENTRY), 'extra_files': list(EXTRA),
                'third_party_imported_natively': found['third_party'], 'pyodide_packages': list(PACKAGES),
                'third_party_stubbed': found.get('stubbed', []),
                'files_sha256': files, 'package_sha256': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
                'runner_files_sha256': {n: hashlib.sha256((src / n).read_bytes()).hexdigest() for n in RUNNERS},
                'pipeline_files_sha256': {n: hashlib.sha256((src / n).read_bytes()).hexdigest() for n in PIPELINE},
                'zip_bytes': out.stat().st_size, 'zip_sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix('.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='src', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    m = build(args.src, args.out)
    print(len(m['files_sha256']), 'files', m['zip_bytes'], 'bytes', m['zip_sha256'][:16])
