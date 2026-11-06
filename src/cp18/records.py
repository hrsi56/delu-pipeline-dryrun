"""CP-18b's data-only record logs: append-only, hash-chained, each file written once.

The daily pipeline commits these to `main` beside CP-18a's `live/store/` and `live/inputs/` (AGENTS.md's standing
exception: the day's generated data-only records, in the paths the CP-18 brief names):

| Path | What it holds |
|---|---|
| `live/outcomes/` | each reconciliation of the issued days' published prices (`cp18.outcomes`) |
| `live/evaluation/` | the committed evaluation records the Space's recomputation must match (`cp18.evaluation`) |
| `live/bundles/` | each dated bundle's manifest and its verification, and each publication's served-identity check (`cp18.bundle`, `cp18.publish`) |

A log is a folder holding `ledger.jsonl` and the files it names. Each ledger line names one file, its SHA-256 and the
previous line's hash; a line's own hash covers everything else in it. Nothing is ever rewritten: a file path is
written once, a replaced or dropped line breaks the chain, and `verify()` refuses either.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUTCOMES_PATH = 'live/outcomes'
EVALUATION_PATH = 'live/evaluation'
BUNDLES_PATH = 'live/bundles'
SCHEMA = 'cp18b-record-log-v1'


class LogRefused(ValueError):
    pass


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump_bytes(value) -> bytes:
    """A record file's bytes: sorted keys, one space indent, a final newline (deterministic)."""
    return (json.dumps(value, indent=1, sort_keys=True, allow_nan=False) + '\n').encode()


class RecordLog:
    def __init__(self, folder):
        self.folder = Path(folder)

    @property
    def ledger(self) -> Path:
        return self.folder / 'ledger.jsonl'

    def entries(self) -> list[dict]:
        if not self.ledger.exists():
            return []
        return [json.loads(line) for line in self.ledger.read_text().splitlines() if line.strip()]

    def verify(self) -> list[dict]:
        """Every line's own hash, the chain, and every file by hash; returns the entries."""
        entries = self.entries()
        previous = None
        seen = set()
        for i, entry in enumerate(entries):
            body = {k: v for k, v in entry.items() if k != 'entry_sha256'}
            if entry.get('schema') != SCHEMA or entry.get('seq') != i:
                raise LogRefused(f'{self.ledger}: line {i} is not entry {i} of a record log')
            if sha_bytes(canonical(body)) != entry.get('entry_sha256'):
                raise LogRefused(f'{self.ledger}: line {i} differs from its own hash')
            if entry.get('previous_entry_sha256') != previous:
                raise LogRefused(f'{self.ledger}: line {i} does not follow line {i - 1}')
            path = self.folder / entry['file']
            if entry['file'] in seen:
                raise LogRefused(f'{self.ledger}: {entry["file"]} is named twice')
            seen.add(entry['file'])
            if not path.is_file() or sha_bytes(path.read_bytes()) != entry['file_sha256']:
                raise LogRefused(f'{self.ledger}: {entry["file"]} is absent or altered')
            previous = entry['entry_sha256']
        return entries

    def append(self, kind: str, relative: str, value, **fields) -> dict:
        """Write `value` once at `relative` and chain a line naming it."""
        entries = self.verify()
        target = self.folder / relative
        if target.exists():
            raise LogRefused(f'{target}: a record file is never replaced')
        data = dump_bytes(value)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        body = {'schema': SCHEMA, 'seq': len(entries), 'kind': kind, 'file': relative, 'file_sha256': sha_bytes(data),
                'previous_entry_sha256': entries[-1]['entry_sha256'] if entries else None, **fields}
        entry = {**body, 'entry_sha256': sha_bytes(canonical(body))}
        with self.ledger.open('a') as handle:
            handle.write(json.dumps(entry, sort_keys=True, separators=(',', ':')) + '\n')
        return entry

    def read(self, entry: dict):
        return json.loads((self.folder / entry['file']).read_text())

    def latest(self, kind: str | None = None) -> dict | None:
        found = [e for e in self.verify() if kind is None or e['kind'] == kind]
        return found[-1] if found else None
