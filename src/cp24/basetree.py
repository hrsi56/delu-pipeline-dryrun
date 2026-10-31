"""The tree CP-24 started from, file by file (§23.13 items 1 and 16; §23.14's "never written" and "preserve").

`reports/ddnn2/base-tree.json` records the SHA-256 of every file tracked at CP-24's base commit (the issued
`main`, `522d7ea`), read from the Git objects. CP-24 writes only its own paths (`src/cp24/`, `tests/cp24/`,
`scripts/cp24_*.py`, `reports/ddnn2/`, `docs/track-b/evidence/cp-24/`, `docs/track-b/research-content/cp24-claims.md`),
none of which existed at the base, so every recorded file must be byte-identical in the candidate: the public
surfaces, the published export set, `scripts/mlflow_export.py`, the root `pyproject.toml` and `uv.lock`, every
earlier checkpoint's code, reports and evidence, every locked document and `progress.md`.

**Where it is proved.** "Unchanged" is a property of a candidate commit, so it is checked by explicit,
candidate-scoped commands, run by the Lead before the review and rerun by the Integration Critic:

    python -m cp24.basetree --check-rev <candidate>   # Git objects: every recorded file byte-identical, and
                                                      # every other file in the commit under CP-24's paths
    python -m cp24.basetree --check                   # the recorded files in a checkout of the candidate

The default test suite does not hash the recorded files. After the Owner's LAND, authorized work on `main`
changes many of them (`progress.md`, the anchors, `AGENTS.md`, the public surfaces), and a collected test that
bound them would turn CI red on `main` for good. `tests/cp24/test_base_tree.py` checks the record and these
checkers on fixtures instead.

Run as ``python -m cp24.basetree`` (writes once), ``--check`` or ``--check-rev <rev>``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = '522d7ea722b5d215d827d7aed9c56b7a9dec066e'
PATH = ROOT / 'reports/ddnn2/base-tree.json'
CP24_PATHS = ('src/cp24/', 'tests/cp24/', 'scripts/cp24_', 'reports/ddnn2/', 'docs/track-b/evidence/cp-24/',
              'docs/track-b/research-content/cp24-claims.md')


def _tree(rev: str, root: Path) -> list[tuple[str, str, str]]:
    """(name, kind, object id) of every entry in `rev`'s tree, recursively."""
    listing = subprocess.check_output(['git', 'ls-tree', '-r', '-z', rev], cwd=root).split(b'\0')
    out = []
    for item in filter(None, listing):
        meta, name = item.split(b'\t', 1)
        _, kind, obj = meta.split()
        out.append((name.decode('utf-8'), kind.decode(), obj.decode()))
    return out


def _blob_sha256(objects: list[str], root: Path) -> list[str]:
    """SHA-256 of each blob's content, in order, read from the Git objects."""
    batch = subprocess.run(['git', 'cat-file', '--batch'], cwd=root, input=''.join(f'{o}\n' for o in objects).encode(),
                           capture_output=True, check=True).stdout
    out, pos = [], 0
    for _ in objects:
        header_end = batch.index(b'\n', pos)
        size = int(batch[pos:header_end].split()[2])
        out.append(hashlib.sha256(batch[header_end + 1:header_end + 1 + size]).hexdigest())
        pos = header_end + 1 + size + 1
    return out


def build(base: str = BASE, root: Path = ROOT) -> dict:
    entries = []
    for name, kind, obj in _tree(base, root):
        if kind != 'blob':
            continue
        if name.startswith(CP24_PATHS):
            raise ValueError(f'{name}: a CP-24 path existed at the base')
        entries.append((name, obj))
    digests = _blob_sha256([o for _, o in entries], root)
    out = {name: digest for (name, _), digest in zip(entries, digests)}
    return {'schema': 'cp24-base-tree-v1', 'base_commit': base, 'files': len(out), 'sha256': out}


def check(record: dict, root: Path = ROOT) -> list[str]:
    """The recorded files in a checkout: each present and byte-identical."""
    bad = []
    for name, digest in record['sha256'].items():
        p = Path(root) / name
        if not p.is_file():
            bad.append(f'missing: {name}')
            continue
        h = hashlib.sha256()
        with p.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b''):
                h.update(chunk)
        if h.hexdigest() != digest:
            bad.append(f'changed: {name}')
    return bad


def check_rev(record: dict, rev: str, root: Path = ROOT) -> list[str]:
    """A commit, from the Git objects: every recorded file present and byte-identical, and every other entry
    a file under CP-24's write paths (§23.14)."""
    entries = {name: (kind, obj) for name, kind, obj in _tree(rev, root)}
    bad = []
    names = [n for n in record['sha256'] if n in entries and entries[n][0] == 'blob']
    digests = dict(zip(names, _blob_sha256([entries[n][1] for n in names], root)))
    for name, digest in record['sha256'].items():
        if name not in entries:
            bad.append(f'missing: {name}')
        elif entries[name][0] != 'blob':
            bad.append(f'not a file: {name}')
        elif digests[name] != digest:
            bad.append(f'changed: {name}')
    for name, (kind, _) in entries.items():
        if name in record['sha256']:
            continue
        if kind != 'blob':
            bad.append(f'not a file: {name}')
        elif not name.startswith(CP24_PATHS):
            bad.append(f'outside CP-24 paths: {name}')
    return bad


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ['--check-rev'] and len(args) == 2:
        rev = subprocess.check_output(['git', 'rev-parse', '--verify', f'{args[1]}^{{commit}}'], cwd=ROOT, text=True).strip()
        record = json.loads(PATH.read_text())
        bad = check_rev(record, rev)
        added = sum(1 for name, kind, _ in _tree(rev, ROOT) if name not in record['sha256'])
        print(json.dumps({'commit': rev, 'base_commit': record['base_commit'], 'recorded_files': record['files'],
                          'unchanged': not bad, 'added_under_cp24_paths': added if not bad else None, 'problems': bad[:20]}))
        return 0 if not bad else 1
    if args == ['--check']:
        bad = check(json.loads(PATH.read_text()))
        print(json.dumps({'unchanged': not bad, 'problems': bad[:20]}))
        return 0 if not bad else 1
    if args:
        raise SystemExit('usage: python -m cp24.basetree [--check | --check-rev <rev>]')
    if PATH.exists():
        raise SystemExit('base-tree.json exists; it is written once')
    PATH.write_text(json.dumps(build(), indent=1, sort_keys=True) + '\n')
    print(f'wrote {PATH}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
