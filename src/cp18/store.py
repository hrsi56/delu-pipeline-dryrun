"""The live state store: CP-17c's store (`cp17c.store.Store`), imported and extended, never weakened.

Everything CP-17c's store checks still holds, by calling its own `verify` first: each ledger entry's own hash, the
chain, every state, artifact and record file by hash, days forward, and the fix for its first Critic's F1 —
`head.json`'s `state_file`, `state_sha256`, `last_day` and `last_artifact_sha256` equal the ledger's last entry, so a
repointed head fails `verify()`. This module adds four entry kinds, each carrying the state unchanged:

- `run`: every run's record (`runs/<run_id>.json`), whatever it issued, so "no run within the lookback" is
  verifiable from the store itself;
- `not_issued`: a delivery day closed as not issued, with one of r19's five causes (`cp18.causes`) and the runs
  that attempted it;
- `search`: a search state that serves later days (S_14's `k14.json`, H4), copied into `search/` and bound by hash;
- `carry`: the carry from 4.7T's last window day to the first live day, naming the days between them, which are
  outside both and never issued.

And it adds four checks: every non-issuance entry carries the state of the entry before it; the delivery days of
issuance and not-issued entries strictly increase together; no day precedes the store's first day; and a search
file hashes as recorded and its content is the search it names (`cp26.search.validate_ensemble`).

**Where the live store lives (the paths CP-18's daily commits to `main` will use, AGENTS.md's daily exception).**
`live/store/` in the product repository: `head.json`, `ledger.jsonl`, `states/<day>.json`, `issued/<day>/artifact.json`,
`issued/<day>/record.json`, `runs/<run_id>.json`, `search/k<NN>.json`; and the day's input archive under
`live/inputs/<day>/` (`cp18.live`). Nothing in them is ever rewritten: the chain makes a replaced or dropped record
visible, and the store refuses it.
"""
from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

from cp17c.store import SCHEMA as CP17C_SCHEMA, Store, StoreRefused, _canonical, _sha, _sha_bytes, _write

from . import causes

STORE_PATH = 'live/store'
INPUTS_PATH = 'live/inputs'
DAY_KINDS = ('issuance', 'not_issued')


class LiveStore(Store):
    """CP-17c's store plus the run, not-issued, search and carry entries."""

    # ---- reading ---------------------------------------------------------------------------------------------
    def verify(self) -> dict:
        head = super().verify()                                  # every CP-17c check, F1's included
        entries = self.entries()
        genesis = entries[0]
        first = date.fromisoformat(genesis.get('first_day') or genesis['origin'])
        previous_day = None
        for i, entry in enumerate(entries[1:], start=1):
            kind = entry.get('kind')
            if kind not in DAY_KINDS + ('run', 'search', 'carry'):
                raise StoreRefused(f'ledger entry {i}: unknown kind {kind!r}')
            if kind != 'issuance' and (entry.get('state_sha256') != entries[i - 1]['state_sha256']
                                       or entry.get('state_file') != entries[i - 1]['state_file']):
                raise StoreRefused(f'ledger entry {i}: a {kind} entry must carry the state before it unchanged')
            if kind in DAY_KINDS:
                day = date.fromisoformat(entry['day'])
                if day < first:
                    raise StoreRefused(f'ledger entry {i}: {day} precedes the store\'s first day {first}')
                if previous_day is not None and day <= previous_day:
                    raise StoreRefused(f'ledger entry {i}: {day} does not follow {previous_day}')
                previous_day = day
            if kind == 'not_issued':
                try:
                    causes.text(entry.get('cause'))
                except ValueError as exc:
                    raise StoreRefused(f'ledger entry {i}: {exc}') from None
            if kind == 'carry':
                last, nxt = date.fromisoformat(entry['last_window_day']), date.fromisoformat(entry['first_live_day'])
                if previous_day is not None and last != previous_day or nxt <= last:
                    raise StoreRefused(f'ledger entry {i}: the carry does not start at the last recorded day')
                previous_day = nxt - timedelta(days=1)            # the days between are outside both, never issued
            if kind == 'search':
                path = self.folder / entry['search_file']
                if not path.is_file() or _sha(path) != entry['search_file_sha256']:
                    raise StoreRefused(f'ledger entry {i}: {entry["search_file"]} is absent or altered')
                from cp26.search import validate_ensemble
                value = json.loads(path.read_text())
                try:
                    validate_ensemble(value, entry['k'], 'nominal')
                except Exception as exc:
                    raise StoreRefused(f'ledger entry {i}: {entry["search_file"]} is not search {entry["k"]} '
                                       f'({type(exc).__name__})') from None
                if value.get('content_sha256') != entry['content_sha256']:
                    raise StoreRefused(f'ledger entry {i}: the search content is not the recorded one')
        return head

    def first_day(self) -> date:
        genesis = self.entries()[0]
        return date.fromisoformat(genesis.get('first_day') or genesis['origin'])

    def days(self) -> dict[str, dict]:
        """Every recorded delivery day: issued (with its policy, or a failed issuance) or not issued, with a cause."""
        out = {}
        for entry in self.entries():
            if entry['kind'] == 'issuance':
                failed = entry.get('issued_policy') is None
                out[entry['day']] = {'status': 'not issued' if failed else 'issued', 'policy': entry.get('issued_policy'),
                                     'cause': causes.CHAIN if failed else None, 'display': entry.get('display'),
                                     'artifact_sha256': entry.get('artifact_sha256')}
            elif entry['kind'] == 'not_issued':
                out[entry['day']] = {'status': 'not issued', 'policy': None, 'cause': entry['cause'],
                                     'attempts': entry.get('attempts', [])}
            elif entry['kind'] == 'carry':
                last, nxt = date.fromisoformat(entry['last_window_day']), date.fromisoformat(entry['first_live_day'])
                for i in range(1, (nxt - last).days):
                    out[str(last + timedelta(days=i))] = {'status': 'outside', 'policy': None, 'cause': None,
                                                          'note': entry['reason']}
        return out

    def last_closed(self) -> date | None:
        """The last day the store has decided (issued, not issued, or carried over); later days only go forward."""
        days = self.days()
        return date.fromisoformat(max(days)) if days else None

    def runs(self) -> list[dict]:
        return [json.loads((self.folder / e['record_file']).read_text()) for e in self.entries() if e['kind'] == 'run']

    def search_file(self, k: int) -> Path | None:
        found = [e for e in self.entries() if e['kind'] == 'search' and e['k'] == k]
        return self.folder / found[-1]['search_file'] if found else None

    # ---- writing ---------------------------------------------------------------------------------------------
    @classmethod
    def init(cls, folder, state, origin: date, source: str, first_day: date | None = None) -> 'LiveStore':
        """A new store whose states start from `state` at `origin`; `first_day` is the first day it may issue."""
        store = cls(folder)
        if (store.folder / 'ledger.jsonl').exists():
            raise StoreRefused(f'{store.folder}: a store exists there; it is never re-initialized')
        rel = f'states/genesis-{origin}.json'
        file_sha = _write(store.folder / rel, state.to_dict())
        store._append({'kind': 'genesis', 'origin': str(origin), 'first_day': str(first_day or origin), 'source': source,
                       'state_file': rel, 'state_file_sha256': file_sha, 'state_sha256': state.digest()},
                      {'last_day': None, 'last_artifact_sha256': None})
        return store

    def _head_now(self) -> dict:
        head = json.loads((self.folder / 'head.json').read_text())
        return {'last_day': head.get('last_day'), 'last_artifact_sha256': head.get('last_artifact_sha256')}

    def _carry_state(self) -> dict:
        last = self.entries()[-1]
        return {k: last[k] for k in ('state_file', 'state_file_sha256', 'state_sha256')}

    def check_day(self, day: date) -> None:
        super().check_day(day)                                    # issued days: never twice, never backwards
        last = self.last_closed()
        if last is not None and day <= last:
            raise StoreRefused(f'{day} is not after the last decided day {last}; days go forward')
        if day < self.first_day():
            raise StoreRefused(f'{day} precedes the store\'s first day {self.first_day()}')

    def record_run(self, run_id: str, record: dict) -> dict:
        rel = f'runs/{run_id}.json'
        if (self.folder / rel).exists():
            raise StoreRefused(f'{rel} exists; a run record is never replaced')
        sha = _write(self.folder / rel, record)
        return self._append({'kind': 'run', 'run_id': run_id, 'record_file': rel, 'record_file_sha256': sha,
                             **self._carry_state()}, self._head_now())

    def record_not_issued(self, day: date, cause: str, attempts: list[str]) -> dict:
        causes.text(cause)
        self.check_day(day)
        return self._append({'kind': 'not_issued', 'day': str(day), 'cause': cause, 'cause_text': causes.text(cause),
                             'attempts': list(attempts), **self._carry_state()}, self._head_now())

    def record_search(self, k: int, source: Path, provenance: str) -> dict:
        from cp26.search import validate_ensemble
        value = json.loads(Path(source).read_text())
        validate_ensemble(value, k, 'nominal')
        rel = f'search/k{k:02d}.json'
        if (self.folder / rel).exists():
            raise StoreRefused(f'{rel} exists; a search state is never replaced')
        target = self.folder / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())
        return self._append({'kind': 'search', 'k': k, 'search_file': rel, 'search_file_sha256': _sha(target),
                             'content_sha256': value['content_sha256'], 'provenance': provenance,
                             **self._carry_state()}, self._head_now())

    def record_carry(self, last_window_day: date, first_live_day: date, reason: str) -> dict:
        last = self.last_closed()
        if last is None or last != last_window_day or first_live_day <= last_window_day:
            raise StoreRefused(f'the carry must start at the store\'s last decided day ({last})')
        return self._append({'kind': 'carry', 'last_window_day': str(last_window_day),
                             'first_live_day': str(first_live_day), 'reason': reason, **self._carry_state()},
                            self._head_now())


def head_sha256(folder) -> str:
    """The store's identity for a carry: its head file's SHA-256 (the head names the ledger's last entry)."""
    return _sha(Path(folder) / 'head.json')


def ledger_sha256(folder) -> str:
    return _sha(Path(folder) / 'ledger.jsonl')


__all__ = ['LiveStore', 'StoreRefused', 'STORE_PATH', 'INPUTS_PATH', 'CP17C_SCHEMA', 'head_sha256', 'ledger_sha256',
           '_canonical', '_sha_bytes']
