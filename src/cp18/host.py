"""The issuing host's files, in place before the H2 gate: the states by hash, the binary and Pyodide by hash, the zip.

- **The states** the freeze manifest pins (A1's search state S_13 `cp17.runtime.K13`, the freeze-origin states of v4,
  v2 and B0 `cp17.runtime.FREEZE_STATE`, the canary's payload `cp17.runtime.CANARY_DIR`) are committed in the
  repository, which decisions 26 and 27 let the public repository hold, at `<pinned>/<path>`: each one's
  project-relative runtime path with its leading `.local/` written `local/`. The product's folder is `live/pinned`
  (placed by CP-18's activation); this round's dry runs use `reports/cp18a/dryrun/pinned`. `materialize` copies each
  to its runtime path and checks it against the freeze manifest's hash; any absent or altered file refuses the run.
- **Pyodide 0.28.0's 23 files** are retrieved from their official release URLs (`PYODIDE_SOURCE`, the CDN that
  CP-17-probe retrieved them from) and each is checked against the manifest's hash before it is placed (`pyodide`).
- **The binary** is never committed: each run retrieves the tarball and `SHASUMS256.txt` and checks all three by hash
  (`cp17.runtime.retrieve`).
- **The package zip** is rebuilt from `src/` and compared with the manifest (`cp17.tools.bundle.build`).

The H2 gate then checks all of it again, independently. Any failed hash means the run issues nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

PINNED = 'live/pinned'
DRY_RUN_PINNED = 'reports/cp18a/dryrun/pinned'
PYODIDE_SOURCE = 'https://cdn.jsdelivr.net/pyodide/v0.28.0/full/'
MANIFEST = 'docs/track-b/evidence/cp-17c/freeze-manifest.json'
PACKAGE_ZIP = '.local/artifacts/cp-17c/bundle/src.zip'


def pyodide_files(root) -> dict[str, str]:
    """{file name: frozen SHA-256} for Pyodide's 23 pinned files."""
    identity = json.loads((Path(root) / MANIFEST).read_text())['policy_identity']
    return dict(sorted(identity['issuing_host']['pyodide']['files_sha256'].items()))


def pinned_files(root) -> dict[str, str]:
    """{runtime path: frozen SHA-256} for every pinned state the manifest names (A1's S_13, the freeze-origin states,
    the canary's payload)."""
    identity = json.loads((Path(root) / MANIFEST).read_text())['policy_identity']
    route = identity['state_route']
    files = {route['k13']['path']: route['k13']['file_sha256']}
    files[route['freeze_origin_state']['path']] = route['freeze_origin_state']['file_sha256']
    files.update(route['canary_payload']['files_sha256'])          # keyed by their runtime paths already
    return dict(sorted(files.items()))


def committed_path(runtime_path: str, pinned: str = PINNED) -> str:
    if not runtime_path.startswith('.local/'):
        raise ValueError(f'{runtime_path}: a pinned runtime file lives under .local/')
    return f"{pinned}/local/{runtime_path.removeprefix('.local/')}"


def materialize(root, project, pinned: str = PINNED) -> dict:
    """Copy each committed pinned state to its runtime path in `project`, checked by hash. Raises on any mismatch."""
    from cp17 import runtime
    root, project = Path(root), Path(project)
    out = {}
    for rel, sha in pinned_files(root).items():
        source, target = root / committed_path(rel, pinned), project / rel
        if target.is_file() and runtime.sha256_file(target) == sha:
            out[rel] = {'sha256': sha, 'action': 'already in place'}
            continue
        if not source.is_file() or runtime.sha256_file(source) != sha:
            raise RuntimeError(f'{committed_path(rel, pinned)}: absent or not the pinned file')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        if runtime.sha256_file(target) != sha:
            raise RuntimeError(f'{rel}: the copy is not the pinned file')
        out[rel] = {'sha256': sha, 'action': 'copied', 'from': committed_path(rel, pinned)}
    return out


def _fetch(url: str, target: Path) -> dict:
    import time
    import urllib.request
    from cp17 import runtime
    started = time.perf_counter()
    with urllib.request.urlopen(url, timeout=300, context=runtime._context()) as response, open(target, 'wb') as handle:
        while block := response.read(1 << 20):
            handle.write(block)
    return {'url': url, 'bytes': target.stat().st_size, 'sha256': runtime.sha256_file(target),
            'seconds': time.perf_counter() - started}


def pyodide(root, project, retrieve: bool, fetch=_fetch, source: str = PYODIDE_SOURCE) -> dict:
    """Pyodide's 23 files at `cp17.runtime.PYODIDE_DIR`: retrieved from the official URLs and placed only after each
    hash holds (`retrieve=True`), or checked in place. Raises on any absent or altered file."""
    import tempfile
    from cp17 import runtime
    root, project = Path(root), Path(project)
    folder = project / runtime.PYODIDE_DIR
    out = {}
    for name, sha in pyodide_files(root).items():
        target = folder / name
        if target.is_file() and runtime.sha256_file(target) == sha:
            out[name] = {'sha256': sha, 'action': 'already in place'}
            continue
        if not retrieve:
            raise RuntimeError(f'{runtime.PYODIDE_DIR}/{name}: absent or not the pinned file')
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            got = fetch(source + name, Path(tmp) / name)
            if got['sha256'] != sha:
                raise RuntimeError(f'{source + name}: SHA-256 {got["sha256"]}, pinned {sha}; nothing is placed')
            shutil.move(str(Path(tmp) / name), target)
        out[name] = {**got, 'action': 'retrieved'}
    return out


def package(root, project) -> dict:
    """Rebuild the frozen package zip from `src/` and compare it with the manifest."""
    from cp17.tools import bundle
    root, project = Path(root), Path(project)
    built = bundle.build(root / 'src', project / PACKAGE_ZIP)
    code = json.loads((root / MANIFEST).read_text())['policy_identity']['code']
    equal = built['zip_sha256'] == code['zip_sha256'] and built['package_sha256'] == code['package_sha256']
    return {'zip_sha256': built['zip_sha256'], 'package_sha256': built['package_sha256'], 'zip_bytes': built['zip_bytes'],
            'equal_to_manifest': equal}


def node(project, retrieve: bool) -> dict:
    """The official binary: retrieved and checked by hash (`retrieve=True`, the runner), or checked in place."""
    from cp17 import runtime
    if retrieve:
        return runtime.retrieve(project)
    found = runtime.check_binary(project)
    if found['problems']:
        raise RuntimeError('the official binary is not in place: ' + '; '.join(found['problems']))
    return {'version': runtime.NODE_VERSION, 'in_place': True, 'node': found['node']}
