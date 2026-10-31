"""The issuing host as CP-17c re-freezes it (r19 §24.13, H1): its pins, where its files live, and how a run gets
the binary.

**The runtime.** Pyodide 0.28.0, by its 23 pinned file hashes, in the official Node.js v26.7.0 darwin-arm64
binary, one worker. The tarball is over the project's 50 MB limit for a committed file, and `bin/node` is over
GitHub's 100 MB limit, so each run retrieves the tarball with the release's `SHASUMS256.txt` and checks all three
by hash (`retrieve`). A failed retrieval or a failed hash means the run issues nothing. The hashes are the pin, not
the URL.

**The machines.** A GitHub-hosted standard runner, label `macos-26` (Apple silicon), and the Owner's Apple-silicon
Mac. Both are macOS on ARM64; on a runner, `ImageOS` is `macos26`. The image version and the CPU model are recorded
by each run's checks (`cp17.hostcheck`), never pinned.

**Where the files live.** Every path is relative to the project root, the checkout that holds `.local/`, so one
manifest names the same files on the runner and on the Owner's Mac:
- the binary: `NODE_RELATIVE`, extracted from the tarball in `NODE_DIR/dl/`;
- Pyodide's 23 files: `PYODIDE_DIR` (`cp17.tools.freeze.RUNTIME_DIR`, unchanged);
- the states (§24.13's list): A1's search state `K13`, the freeze-origin states of v4, v2 and B0 `FREEZE_STATE`,
  the canary's payload `CANARY_DIR`, and a run's shadow buffers and issued records under a state store
  (`cp17c.store`).

**CPython.** The chain's judgement and the shadow states run in CPython inside `cp17.daily`, in the environment
`uv.lock` pins, under one patch version, `CPYTHON`: uv's own build of CPython 3.13.15 for macOS on ARM64. Its zlib
runtime is `ZLIB_RUNTIME`, which the package zip's bytes depend on. It is pinned here, not in `uv.lock` or
`pyproject.toml`, which CP-26's tests pin.

This module is frozen with the policy identity (`cp17.tools.bundle.PIPELINE`) and never ships to a browser.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request

NODE_VERSION = 'v26.7.0'
NODE_DIST = f'https://nodejs.org/dist/{NODE_VERSION}/'
TARBALL = {'name': f'node-{NODE_VERSION}-darwin-arm64.tar.gz', 'bytes': 57_622_243,
           'sha256': '7ee659a7768e641bbfd5360940660b8e8fd0052f77488f365562bac522fc15d4'}
BINARY = {'bytes': 145_238_560, 'sha256': 'a9bd0630891c2dcdee70de88270fee2cc0c4a9e76495039dd3b4f91c5e6b71df'}
#: The release folder's `SHASUMS256.txt`, as retrieved on 2026-10-30 (CP-17c-probe).
SHASUMS = {'name': 'SHASUMS256.txt', 'bytes': 2_943,
           'sha256': '4533f0a43b9ba7f78a48230a0511b9dd5c931f20c3b3cac281ff9b7a2080fb2e'}
NODE_DIR = '.local/tools/node-v26.7.0-official'
NODE_RELATIVE = f'{NODE_DIR}/node-{NODE_VERSION}-darwin-arm64/bin/node'
TARBALL_RELATIVE = f"{NODE_DIR}/dl/{TARBALL['name']}"
SHASUMS_RELATIVE = f"{NODE_DIR}/dl/{SHASUMS['name']}"
PYODIDE_DIR = '.local/tools/pyodide-0.28.0'
CPYTHON = '3.13.15'
CPYTHON_BUILD = "uv's CPython 3.13.15 for macOS on ARM64 (cpython-3.13.15-macos-aarch64-none, as uv 0.12.6 installs it)"
ZLIB_RUNTIME = '1.2.12'
MACHINE = {'system': 'Darwin', 'machine': 'arm64', 'runner_label': 'macos-26', 'image_os': 'macos26'}
STATE_DIR = '.local/artifacts/cp-17c/state'
K13 = '.local/artifacts/cp-17/inputs/reports/rebaseline/cold-rerun-2026-10-15/search-results/nominal/k13.json'
FREEZE_STATE = f'{STATE_DIR}/freeze-origin-state.json'
CANARY_DIR = f'{STATE_DIR}/canary'


def project_root(start=None) -> Path:
    """The checkout that holds `.local/`: `$CP17_PROJECT_ROOT`, or the git common dir's parent, or this file's."""
    if os.environ.get('CP17_PROJECT_ROOT'):
        return Path(os.environ['CP17_PROJECT_ROOT'])
    here = Path(start) if start is not None else Path(__file__).resolve().parent
    try:
        common = subprocess.run(['git', '-C', str(here), 'rev-parse', '--path-format=absolute', '--git-common-dir'],
                                capture_output=True, text=True, check=True).stdout.strip()
        return Path(common).parent
    except (OSError, subprocess.CalledProcessError):
        return Path(__file__).resolve().parents[2]


def node_path(project=None) -> Path:
    return (project_root() if project is None else Path(project)) / NODE_RELATIVE


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def check_file(path, pin: dict) -> list[str]:
    """`path` against a pin's bytes and SHA-256: [] when it holds."""
    path = Path(path)
    if not path.is_file():
        return [f'{path}: absent']
    size, actual = path.stat().st_size, sha256_file(path)
    if size != pin['bytes'] or actual != pin['sha256']:
        return [f"{path}: {size} bytes, SHA-256 {actual}; pinned {pin['bytes']} bytes, {pin['sha256']}"]
    return []


def check_binary(project=None) -> dict:
    """The tarball and `bin/node` against their pins (H2's second check)."""
    project = project_root() if project is None else Path(project)
    problems = check_file(project / TARBALL_RELATIVE, TARBALL) + check_file(project / NODE_RELATIVE, BINARY)
    return {'problems': problems, 'tarball': str(project / TARBALL_RELATIVE), 'node': str(project / NODE_RELATIVE)}


def _context():
    """TLS against certifi's roots where the environment has them (a runner's CPython may ship without a store)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _get(url: str, target: Path) -> dict:
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=300, context=_context()) as response, open(target, 'wb') as handle:
        while block := response.read(1 << 20):
            handle.write(block)
    return {'url': url, 'bytes': target.stat().st_size, 'sha256': sha256_file(target),
            'seconds': time.perf_counter() - started}


def retrieve(project=None, fetch=_get) -> dict:
    """Retrieve the tarball and `SHASUMS256.txt`, check both and the release's listing by hash, and extract
    `bin/node` to `NODE_RELATIVE`, checked by hash. Raises on any failure, so the run issues nothing.

    The download lands in a fresh folder, and the binary is moved into place only after every hash holds."""
    project = project_root() if project is None else Path(project)
    with tempfile.TemporaryDirectory(dir=project / '.local' if (project / '.local').is_dir() else None) as tmp:
        tmp = Path(tmp)
        sums = fetch(NODE_DIST + SHASUMS['name'], tmp / SHASUMS['name'])
        problems = check_file(tmp / SHASUMS['name'], SHASUMS)
        if problems:
            raise RuntimeError(f'the release listing is not the pinned one: {problems}')
        listed = dict(reversed(line.split()) for line in (tmp / SHASUMS['name']).read_text().splitlines() if line.strip())
        if listed.get(TARBALL['name']) != TARBALL['sha256']:
            raise RuntimeError('the release listing does not list the pinned tarball')
        tarball = fetch(NODE_DIST + TARBALL['name'], tmp / TARBALL['name'])
        problems = check_file(tmp / TARBALL['name'], TARBALL)
        if problems:
            raise RuntimeError(f'the tarball is not the pinned one: {problems}')
        member = f"{TARBALL['name'].removesuffix('.tar.gz')}/bin/node"
        with tarfile.open(tmp / TARBALL['name']) as archive:
            archive.extractall(tmp / 'x', members=[archive.getmember(member)], filter='data')
        problems = check_file(tmp / 'x' / member, BINARY)
        if problems:
            raise RuntimeError(f'bin/node is not the pinned one: {problems}')
        for source, rel in ((tmp / SHASUMS['name'], SHASUMS_RELATIVE), (tmp / TARBALL['name'], TARBALL_RELATIVE),
                            (tmp / 'x' / member, NODE_RELATIVE)):
            (project / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), project / rel)
    (project / NODE_RELATIVE).chmod(0o755)
    return {'version': NODE_VERSION, 'shasums': sums, 'tarball': {**tarball, 'listed_sha256': listed[TARBALL['name']]},
            'binary': {'path': NODE_RELATIVE, 'bytes': (project / NODE_RELATIVE).stat().st_size,
                       'sha256': sha256_file(project / NODE_RELATIVE)}, 'machine': platform.machine()}
