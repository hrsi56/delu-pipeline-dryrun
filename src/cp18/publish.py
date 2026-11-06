"""The publication step: a verified bundle and the Space, published to a static host, then the served identity checked.

**Prepared, not run against Hugging Face.** This round exercises it against a local static server only
(`LocalTarget`, served from `127.0.0.1`). The Hugging Face upload is CP-18's activation, with the Owner's `HF_TOKEN`;
`HubTarget` holds that path and refuses to write unless the activation passes `authorized=True` (nothing in this round
does), and it reads the token only inside its one commit, never printing, logging or writing it.

**One publication, in order** (`publish`), failing closed at the first failure:

1. **The gate** (`cp18.bundle.gate`): R7's verification, the frozen identity and the red line, and the native
   recomputation check. A refused bundle is never published: a mismatch is a product failure for that day.
2. **The Space build carries exactly that bundle** at `payloads/`, byte for byte, and its own build record
   (`app/space-build.json`) lists every served file with its SHA-256.
3. **The upload** of the whole build to the target, in one operation.
4. **The served identity** (`verify_served`): the served `payloads/manifest.json` is fetched as a visitor would fetch
   it, and must equal the intended manifest byte for byte; every file it lists, and every file of the Space's build
   record, is fetched and must hash as intended; the served manifest must name the intended daily artifact. Any
   difference fails the publication.

The record of each publication is appended to `live/bundles/` (`cp18.records`).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import shutil
import threading
import urllib.parse
import urllib.request

from .bundle import gate, read
from .records import RecordLog

SCHEMA = 'cp18b-publication-v1'
LOOPBACK = ('127.0.0.1', 'localhost', '[::1]')


class PublicationRefused(RuntimeError):
    pass


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree(folder: Path) -> dict[str, bytes]:
    folder = Path(folder)
    return {str(p.relative_to(folder)): p.read_bytes() for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name != '.DS_Store'}


# ------------------------------------------------------------------ targets
class LocalTarget:
    """A static host on this machine: the build is copied into `folder`, which a local server serves."""

    kind = 'local static server'

    def __init__(self, folder):
        self.folder = Path(folder)

    def upload(self, build: Path) -> dict:
        staging = self.folder.with_name(self.folder.name + '.staging')
        if staging.exists():
            shutil.rmtree(staging)
        shutil.copytree(build, staging)
        if self.folder.exists():
            shutil.rmtree(self.folder)
        staging.rename(self.folder)              # one operation: the old build or the new, never a mixture
        return {'target': self.kind, 'files': len(tree(self.folder))}


class HubTarget:
    """The Hugging Face Static Space (CP-18's activation; CP-18c): the Space made to serve exactly the build, in one Hub
    commit. Refuses unless `authorized`. In order, as v1's `scripts/deploy_space.py` deployed (its pure helpers are
    reused): the Space's identity and file tree read anonymously; the plan (every build file missing or different there,
    every remote file the build does not carry, which is deleted: the build replaces v1's demo, which it carries at
    `history/`); ONE `create_commit` against the revision just read, so a concurrent change fails rather than being
    overwritten; then the tree at the new revision must equal the build, path by path and hash by hash. `HF_TOKEN` is
    read inside the commit only (`os.environ.get`), never printed, logged or written; a failure is recorded by its type
    and HTTP status only."""

    kind = 'Hugging Face Static Space'

    def __init__(self, repo: str, *, authorized: bool = False, hub=None, committer=None):
        if not authorized:
            raise PublicationRefused('the Hugging Face upload is CP-18\'s activation; nothing was uploaded')
        self.repo = repo
        self.hub = hub                    # tests: a recording fake of deploy_space.Hub's reads
        self.committer = committer        # tests: a recording fake of the one write

    def _hub(self):
        if self.hub is None:
            self.hub = _deploy().Hub(api=f'{_deploy().HUB}api/spaces/{self.repo}')
        return self.hub

    def commit(self, build: Path, adds: list[str], deletes: list[str], parent: str, message: str) -> dict:  # pragma: no cover
        token = os.environ.get('HF_TOKEN')
        if not token:
            raise PublicationRefused('HF_TOKEN is unset; no upload was attempted')
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi   # `uv run --with`, pinned
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
        operations = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(Path(build) / path)) for path in adds]
        operations += [CommitOperationDelete(path_in_repo=path) for path in deletes]
        info = HfApi(token=token).create_commit(repo_id=self.repo, repo_type='space', operations=operations,
                                                commit_message=message, parent_commit=parent)
        return {'commit': info.oid, 'commit_url': info.commit_url}

    def upload(self, build: Path) -> dict:
        D = _deploy()
        hub, build = self._hub(), Path(build)
        identity = hub.identity()
        if identity.get('sdk') != 'static' or identity.get('private') is not False or not identity.get('revision'):
            raise PublicationRefused(f'not a public Static Space at a known revision: {identity}')
        local = D.local_files(build)
        planned = D.plan(local, hub.tree(identity['revision']))
        guard = D.secret_guard
        files = sorted(p.relative_to(build).as_posix() for p in build.rglob('*') if p.is_file())
        findings = guard._scan([(f, (build / f).read_bytes()) for f in files], guard.credentials())   # noqa: SLF001
        if findings:                      # the guard names the variable and the file, never the value
            raise PublicationRefused('the credential-value guard blocked the build: ' + '; '.join(findings))
        adds = sorted(planned['planned_adds'] + planned['planned_changes'], key=str.encode)
        out = {'target': self.kind, 'repo': self.repo, 'before_revision': identity['revision'],
               'planned_adds': len(planned['planned_adds']), 'planned_changes': len(planned['planned_changes']),
               'planned_deletes': planned['planned_deletes'], 'commit': None, 'credential_guard': 'passed'}
        after = identity['revision']
        if adds or planned['planned_deletes']:
            committed = (self.committer or self.commit)(build, adds, planned['planned_deletes'], identity['revision'],
                                                        'Daily bundle and the final-product Space (data-only update)')
            out.update(commit=committed['commit'], commit_url=committed.get('commit_url'))
            after = committed['commit']
        out['after_revision'] = after
        out['tree'] = D.verify(local, hub.tree(after))
        if not out['tree']['verified']:
            raise PublicationRefused(f"the Space's tree at {after} is not the build: {out['tree']}")
        return out


def _deploy():
    """v1's `scripts/deploy_space.py` (landed; imported, never changed): its tree parse, plan and verify."""
    import sys
    from delu_forecast.claims import REPO_ROOT
    scripts = str(Path(REPO_ROOT) / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import deploy_space
    return deploy_space


def space_origin(repo: str) -> str:
    """A Static Space's own direct origin: https://<owner>-<name>.static.hf.space/ (lowercase; '_' and '.' as '-')."""
    owner, name = repo.split('/')
    host = f'{owner}-{name}'.lower().replace('_', '-').replace('.', '-')
    return f'https://{host}.static.hf.space/'


def listed(build: Path) -> dict[str, dict]:
    """Every file the Space's build record lists ({path: {'sha256', 'bytes'}}), with History's own listing when the build
    carries one (`app/history-files.json`, itself listed and so bound by its hash)."""
    record = json.loads((Path(build) / 'app' / 'space-build.json').read_text())
    files = record['files']
    out = {f['path']: f for f in files} if isinstance(files, list) else dict(files)
    history = Path(build) / 'app' / 'history-files.json'
    if 'app/history-files.json' in out and history.is_file():
        out.update({f['path']: f for f in json.loads(history.read_text())['files']})
    return out


# ------------------------------------------------------------------ reading what is served
#: What Hugging Face's static host injects into every served HTML file, right after its first `<head>`: one script
#: naming the Space's creator, then Open Graph and Twitter card tags. Nothing else is accepted as an injection.
HF_INJECTION = re.compile(rb'<script>window\.huggingface=\{[^<]*\};</script>'
                          rb'(?:<meta (?:property|name)="(?:og|twitter):[a-z_:]+" content="[^"<>]*">)*')


def normalized(path: str, data: bytes) -> tuple[bytes, bytes | None]:
    """A served HTML file without the host's injection (A6: "raw/normalized hashes as applicable"); else as served."""
    if not path.endswith('.html'):
        return data, None
    i = data.find(b'<head>')
    m = HF_INJECTION.match(data, i + 6) if i >= 0 else None
    if not m or m.end() == i + 6:
        return data, None
    return data[:i + 6] + data[m.end():], data[i + 6:m.end()]


def matches(path: str, data: bytes | None, sha256: str, size: int | None = None) -> tuple[bool, str]:
    """Raw equality first; for HTML, equality after removing exactly the host's injection."""
    if data is None:
        return False, 'not served'
    if sha(data) == sha256 and (size is None or len(data) == size):
        return True, 'raw'
    body, injected = normalized(path, data)
    if injected is not None and sha(body) == sha256 and (size is None or len(body) == size):
        return True, 'normalized'
    return False, 'different'


def https_fetch(origin: str):
    """A visitor's GET of the Space's own direct origin over HTTPS (certificate verified), uncached."""
    parsed = urllib.parse.urlparse(origin)
    if parsed.scheme != 'https' or not (parsed.hostname or '').endswith('.static.hf.space'):
        raise PublicationRefused(f'{origin}: only a Static Space\'s own HTTPS origin is read')
    from .live.transport import tls_context
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=tls_context()))

    def get(path: str) -> bytes | None:
        url = origin.rstrip('/') + '/' + urllib.parse.quote(path)
        req = urllib.request.Request(url, headers={'User-Agent': 'delu-day-ahead-forecast-cp18/1 (served identity)',
                                                   'Cache-Control': 'no-cache'})
        try:
            with opener.open(req, timeout=60) as response:      # noqa: S310 - https, the Space's own origin
                return response.read()
        except urllib.error.HTTPError:
            return None
    return get


def http_fetch(base: str):
    """A visitor's GET of `base`/<path>. In this round only a loopback host is allowed."""
    parsed = urllib.parse.urlparse(base)
    if parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise PublicationRefused(f'{parsed.hostname}: only a local static server is read in this round')

    def get(path: str) -> bytes | None:
        url = base.rstrip('/') + '/' + urllib.parse.quote(path)
        try:
            with urllib.request.urlopen(url, timeout=30) as response:   # noqa: S310 - loopback only
                return response.read()
        except urllib.error.HTTPError:
            return None
    return get


def verify_served(fetch, build: Path, bundle: Path) -> dict:
    """The served identity against the intended build and bundle; every difference is listed."""
    intended = read(bundle)
    problems = []
    served_manifest = fetch('payloads/manifest.json')
    if served_manifest is None:
        return {'served_identity_equal': False, 'problems': ['payloads/manifest.json is not served'], 'checked': 0}
    if served_manifest != intended['manifest.json']:
        problems.append('the served manifest is not the intended one')
    manifest = json.loads(served_manifest)
    want = json.loads(intended['manifest.json'])
    if manifest.get('artifact', {}).get('artifact_sha256') != want['artifact']['artifact_sha256']:
        problems.append('the served manifest names another daily artifact')
    checked = 1
    how = {'raw': 0, 'normalized': 0}
    for f in manifest.get('files', []):
        data = fetch('payloads/' + f['path'])
        checked += 1
        ok, kind = matches('payloads/' + f['path'], data, f['sha256'], f['bytes'])
        how[kind] = how.get(kind, 0) + 1
        if not ok:
            problems.append(f"payloads/{f['path']}: not served as the manifest lists it")
    for path, item in sorted(listed(build).items()):
        data = fetch(path)
        checked += 1
        ok, kind = matches(path, data, item['sha256'])
        how[kind] = how.get(kind, 0) + 1
        if not ok:
            problems.append(f'{path}: not served as the Space build lists it')
    served_artifact = fetch('payloads/artifact.json')
    if served_artifact is None or json.loads(served_artifact).get('artifact_sha256') != want['artifact']['artifact_sha256']:
        problems.append('the served artifact is not the intended daily artifact')
    return {'served_identity_equal': not problems, 'problems': problems[:50], 'checked': checked, 'how': how,
            'served_manifest_sha256': sha(served_manifest), 'intended_manifest_sha256': sha(intended['manifest.json']),
            'artifact_sha256': want['artifact']['artifact_sha256']}


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def publish(build: Path, bundle: Path, target, fetch, root=None, *, retries: int = 0, retry_seconds: float = 30.0,
            sleep=None) -> dict:
    """Gate, check the build carries the bundle, upload, verify the served identity. Never raises for a failed step:
    the record says where it stopped."""
    import time
    sleep = sleep or time.sleep
    build, bundle = Path(build), Path(bundle)
    out = {'schema': SCHEMA, 'target': target.kind, 'bundle': str(bundle), 'build': str(build), 'published': False}
    files = read(bundle)
    verdict = gate(files, root)
    out['gate'] = {'publishable': verdict['publishable'], 'reason': verdict['reason'],
                   'verify_problems': verdict['verify']['problems'][:10],
                   'recompute_match': (verdict['recompute'] or {}).get('match')}
    if not verdict['publishable']:
        out['stopped'] = 'the gate refused the bundle: nothing was uploaded'
        return out
    carried = {k: v for k, v in tree(build).items() if k.startswith('payloads/')}
    if carried != {f'payloads/{k}': v for k, v in files.items()}:
        out['stopped'] = 'the Space build does not carry exactly this bundle: nothing was uploaded'
        return out
    built = tree(build)
    if any(sha(built.get(p, b'')) != i['sha256'] for p, i in listed(build).items()):
        out['stopped'] = 'the Space build differs from its own build record: nothing was uploaded'
        return out
    try:
        out['upload'] = target.upload(build)
    except Exception as exc:          # a failed upload is recorded by its type (and HTTP status) only
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
        out['stopped'] = f'the upload failed ({type(exc).__name__}' + (f', HTTP {status}' if status else '') + ')'
        if isinstance(exc, PublicationRefused) and 'HF_TOKEN' not in str(exc):
            out['stopped'] += f': {exc}'
        return out
    out['served'] = verify_served(fetch, build, bundle)
    out['served_attempts'] = [{'at': _now(), 'equal': out['served']['served_identity_equal'],
                               'problems': out['served']['problems'][:5]}]
    while not out['served']['served_identity_equal'] and len(out['served_attempts']) < retries + 1:
        sleep(retry_seconds)                  # a CDN may serve the previous revision for a while: retried, recorded
        out['served'] = verify_served(fetch, build, bundle)
        out['served_attempts'].append({'at': _now(), 'equal': out['served']['served_identity_equal'],
                                       'problems': out['served']['problems'][:5]})
    out['published'] = out['served']['served_identity_equal']
    if not out['published']:
        out['stopped'] = 'the served identity is not the intended one: the publication failed'
    return out


@contextmanager
def local_server(folder: Path):
    """A static server on 127.0.0.1 for `folder` (WebAssembly with its media type), in a thread."""
    folder = Path(folder)

    class Handler(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.wasm': 'application/wasm',
                          '.mjs': 'text/javascript', '.json': 'application/json'}

        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(folder), **kwargs)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}/'
    finally:
        server.shutdown()
        server.server_close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog='python -m cp18.publish')
    ap.add_argument('--build', required=True, type=Path, help='the built Space (scripts/build_product_space.py)')
    ap.add_argument('--bundle', required=True, type=Path)
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument('--target', type=Path, help='the local folder the local static server serves (a dry run)')
    where.add_argument('--hub', help='the Hugging Face Static Space, owner/name (CP-18c: the live publication); '
                                     'HF_TOKEN is read inside its one commit')
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--record', type=Path)
    ap.add_argument('--live', type=Path, help='append the publication to <live>/bundles/')
    args = ap.parse_args(argv)
    if args.hub:
        origin = space_origin(args.hub)
        value = publish(args.build, args.bundle, HubTarget(args.hub, authorized=True), https_fetch(origin), args.root,
                        retries=20, retry_seconds=30.0)
        value['served_from'] = origin
    else:
        target = LocalTarget(args.target)
        args.target.parent.mkdir(parents=True, exist_ok=True)
        args.target.mkdir(exist_ok=True)
        with local_server(args.target) as base:
            value = publish(args.build, args.bundle, target, http_fetch(base), args.root)
            value['served_from'] = base
    if args.live and value['published']:
        manifest = json.loads((args.bundle / 'manifest.json').read_text())
        as_of = manifest['freshness']['as_of_day']
        RecordLog(args.live / 'bundles').append('publication', f'{as_of}/publication.json', value, as_of_day=as_of,
                                                manifest_sha256=value['served']['served_manifest_sha256'])
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
    print('published' if value['published'] else f"NOT published: {value.get('stopped')}",
          (value.get('served') or {}).get('checked', 0), 'files checked')
    return 0 if value['published'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
