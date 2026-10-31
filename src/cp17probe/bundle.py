"""The browser's source bundle: the landed files the daily fit imports, byte for byte, and the two stubs.

The file list is the import closure, measured: a fresh native interpreter imports the fit's entry
modules and every loaded module whose file lies under `src/` is shipped. The zip is deterministic
(sorted names, fixed timestamps), and its manifest records each file's SHA-256 so the browser's own
hashes of what it executed can be checked against the checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ENTRY = ('cp17probe.dailyfit', 'cp17probe.attribution', 'cp24.design', 'cp24.ddnn2', 'cp24.member', 'cp15.data', 'cp15.models',
         'cp21.lgbm', 'cp16.residuals')
STUBS = ('cp17probe/stubs/absent/holidays.py', 'cp17probe/stubs/fallback/resource.py')
#: Pyodide packages (with their lock dependencies) each policy's landed import closure needs.
PACKAGES = {'A1': ('numpy', 'pandas', 'scikit-learn', 'requests', 'tzdata'),
            'v2': ('numpy', 'pandas', 'scikit-learn', 'requests', 'tzdata', 'lightgbm')}
FIXED_TIME = (2026, 9, 6, 0, 0, 0)

PROBE = """
import importlib, json, sys
from pathlib import Path
src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src))
for name in sys.argv[2:]:
    importlib.import_module(name)
files = sorted({str(Path(m.__file__).resolve().relative_to(src)) for m in list(sys.modules.values())
                if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(src)})
third = sorted({n.split('.')[0] for n, m in sys.modules.items() if getattr(m, '__file__', None)
                and 'site-packages' in str(m.__file__)})
print(json.dumps({'files': files, 'third_party': third}))
"""


def closure(src: Path) -> dict:
    out = subprocess.check_output([sys.executable, '-c', PROBE, str(src), *ENTRY],
                                  env={'PATH': '/usr/bin:/bin'}, text=True)
    return json.loads(out)


def build(src: Path, out: Path) -> dict:
    src, out = Path(src).resolve(), Path(out)
    found = closure(src)
    names = sorted(set(found['files']) | set(STUBS))
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name in names:
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (src / name).read_bytes())
    manifest = {'schema': 'cp17-probe-bundle-v1', 'entry_modules': list(ENTRY), 'stubs': list(STUBS),
                'third_party_imported_natively': found['third_party'], 'packages': {k: list(v) for k, v in PACKAGES.items()},
                'files_sha256': {n: hashlib.sha256((src / n).read_bytes()).hexdigest() for n in names},
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
