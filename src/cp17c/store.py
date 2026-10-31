"""The state store: how the shadows' buffers and the issued records persist between runs (proposed for CP-18).

A store is one folder. It is the single source of what has been issued, and every run reads it, checks it, and
appends to it:

    head.json                 the last day, its state file and its artifact, and the ledger's last entry hash
    ledger.jsonl              one entry per issued delivery day, each chained to the one before by SHA-256
    states/<label>.json       the shadow states after a day's registrations (`cp17.shadow.ShadowStates`)
    issued/<day>/artifact.json    the day's issued artifact (`cp17.issue.artifact`)
    issued/<day>/record.json      the day's chain, shadows, intakes, buffers and measurements

**The rules.**
- **Loading checks everything.** Each ledger entry's own hash, the chain, every state file and artifact file by
  hash, and that the head names the last entry. A store that fails any of these is refused, so nothing issues.
- **One issuance per delivery day.** A day already in the ledger is refused, and no issued record is replaced.
- **Days go forward.** A day earlier than the last one is refused. A gap is allowed (decision 23's catch-up and
  its lookback are CP-18's).
- **The genesis entry** records where the states started: the freeze-origin state for 4.7T and the live product,
  or a development day's state for an exercise, with its source.

CP-18 decides where the folder lives between runs (§24.13's "how the states are carried"). CP-17c exercises it on
the runner across the 14 shadow days, one process per day.
"""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path

SCHEMA = 'cp17c-state-store-v1'


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(Path(path).read_bytes())


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _write(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=str).encode() + b'\n'
    path.write_bytes(data)
    return _sha_bytes(data)


class StoreRefused(Exception):
    """The store does not verify, or the requested write would break a rule."""


class Store:
    def __init__(self, folder):
        self.folder = Path(folder)

    # ---- reading ---------------------------------------------------------------------------------------------
    def entries(self) -> list[dict]:
        path = self.folder / 'ledger.jsonl'
        if not path.exists():
            raise StoreRefused(f'{path}: absent')
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def verify(self) -> dict:
        """Every hash the store records, and its chain. Raises `StoreRefused`; returns the head."""
        entries = self.entries()
        if not entries or entries[0].get('kind') != 'genesis':
            raise StoreRefused('the ledger does not start with its genesis entry')
        previous, days = None, []
        for i, entry in enumerate(entries):
            body = {k: v for k, v in entry.items() if k != 'entry_sha256'}
            if entry.get('entry_sha256') != _sha_bytes(_canonical(body)):
                raise StoreRefused(f'ledger entry {i}: its own hash does not hold')
            if entry.get('previous_entry_sha256') != previous:
                raise StoreRefused(f'ledger entry {i}: it does not chain to the entry before it')
            for rel, key in (('state_file', 'state_file_sha256'), ('artifact_file', 'artifact_file_sha256'),
                             ('record_file', 'record_file_sha256')):
                if entry.get(rel) is not None:
                    path = self.folder / entry[rel]
                    if not path.is_file() or _sha(path) != entry[key]:
                        raise StoreRefused(f'ledger entry {i}: {entry[rel]} is absent or altered')
            if entry['kind'] == 'issuance':
                if days and entry['day'] <= days[-1]:
                    raise StoreRefused(f"ledger entry {i}: {entry['day']} does not follow {days[-1]}")
                if entry.get('parent_state_sha256') != entries[i - 1]['state_sha256']:
                    raise StoreRefused(f'ledger entry {i}: its parent state is not the state before it')
                days.append(entry['day'])
            previous = entry['entry_sha256']
        head = json.loads((self.folder / 'head.json').read_text())
        if head.get('schema') != SCHEMA or head.get('entry_sha256') != previous or head.get('entries') != len(entries):
            raise StoreRefused('head.json does not name the ledger\'s last entry')
        last, issued = entries[-1], [e for e in entries if e['kind'] == 'issuance']
        named = {'state_file': last['state_file'], 'state_sha256': last['state_sha256'],
                 'last_day': issued[-1]['day'] if issued else None,
                 'last_artifact_sha256': issued[-1]['artifact_sha256'] if issued else None}
        if any(head.get(k) != v for k, v in named.items()):
            raise StoreRefused('head.json names another state, day or artifact than the ledger\'s last entry')
        return head

    def state(self):
        """The current shadow states, after verifying the store."""
        from cp17.shadow import ShadowStates
        head = self.verify()
        state = ShadowStates.from_dict(json.loads((self.folder / head['state_file']).read_text()))
        if state.digest() != head['state_sha256']:
            raise StoreRefused(f"{head['state_file']}: its buffers are not the recorded state")
        return state, head

    # ---- writing ---------------------------------------------------------------------------------------------
    def _append(self, entry: dict, head_extra: dict) -> dict:
        entries = self.entries() if (self.folder / 'ledger.jsonl').exists() else []
        entry['previous_entry_sha256'] = entries[-1]['entry_sha256'] if entries else None
        entry['entry_sha256'] = _sha_bytes(_canonical(entry))
        with (self.folder / 'ledger.jsonl').open('a') as handle:
            handle.write(json.dumps(entry, sort_keys=True, allow_nan=False) + '\n')
        head = {'schema': SCHEMA, 'entries': len(entries) + 1, 'entry_sha256': entry['entry_sha256'],
                'state_file': entry['state_file'], 'state_sha256': entry['state_sha256'], **head_extra}
        _write(self.folder / 'head.json', head)
        return entry

    @classmethod
    def init(cls, folder, state, origin: date, source: str) -> 'Store':
        """A new store whose states start from `state` at `origin` (after its release); refuses an existing one."""
        store = cls(folder)
        if (store.folder / 'ledger.jsonl').exists():
            raise StoreRefused(f'{store.folder}: a store exists there; it is never re-initialized')
        rel = f'states/genesis-{origin}.json'
        file_sha = _write(store.folder / rel, state.to_dict())
        store._append({'kind': 'genesis', 'origin': str(origin), 'source': source, 'state_file': rel,
                       'state_file_sha256': file_sha, 'state_sha256': state.digest()},
                      {'last_day': None, 'last_artifact_sha256': None})
        return store

    def last_artifact_sha256(self) -> str | None:
        return json.loads((self.folder / 'head.json').read_text()).get('last_artifact_sha256')

    def issued_days(self) -> list[str]:
        return [e['day'] for e in self.entries() if e['kind'] == 'issuance']

    def check_day(self, day: date) -> None:
        """Refuse a day already issued, or one earlier than the last (one issuance per delivery day)."""
        days = self.issued_days()
        if str(day) in days:
            raise StoreRefused(f'{day} is already issued; an issued record is never replaced')
        if days and str(day) < days[-1]:
            raise StoreRefused(f'{day} is earlier than the last issued day {days[-1]}')

    def record_day(self, day: date, state, artifact: dict, record: dict, parent_state_sha256: dict) -> dict:
        """Append a day's issuance: its artifact, its record, and the states after its registrations."""
        self.check_day(day)
        art_rel, rec_rel, state_rel = f'issued/{day}/artifact.json', f'issued/{day}/record.json', f'states/{day}.json'
        art_sha = _write(self.folder / art_rel, artifact)
        rec_sha = _write(self.folder / rec_rel, record)
        state_sha = _write(self.folder / state_rel, state.to_dict())
        entry = {'kind': 'issuance', 'day': str(day), 'issued_policy': artifact.get('issued_policy'),
                 'display': artifact.get('display'), 'artifact_sha256': artifact.get('artifact_sha256'),
                 'parent_artifact_sha256': artifact.get('lineage', {}).get('parent_artifact_sha256'),
                 'artifact_file': art_rel, 'artifact_file_sha256': art_sha, 'record_file': rec_rel,
                 'record_file_sha256': rec_sha, 'state_file': state_rel, 'state_file_sha256': state_sha,
                 'state_sha256': state.digest(), 'parent_state_sha256': parent_state_sha256}
        return self._append(entry, {'last_day': str(day), 'last_artifact_sha256': artifact.get('artifact_sha256')})
