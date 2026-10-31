"""The CP-25 ceilings ledger (capstone v21-r13 §24.6), enforced from the first job.

Standard library only. It has the interface CP-20's `Fetcher`, `Attempts` and `Extractor` call
(`reserve`, `headroom`, `event`, `read`), so CP-20's extraction code runs unchanged against
CP-25's own caps:

* 24 machine-hours, summed over workers' wall time (every job is charged wall time x workers);
* at most 4 concurrent workers;
* at most 40 GiB of transient transfer (reserved before each request is sent);
* $0 (no paid service is reachable: the fetcher admits only the two anonymous archives).

Every capped counter is reserved under an exclusive file lock before the operation it pays for,
so a restart or a second process can neither reset nor bypass the cumulative account.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import time

from cp20.budget import CapExceeded  # the class CP-20's fetcher and extractor catch

GIB = 1024 ** 3
HOUR = 3600
CAPS = {'machine_seconds': 24 * HOUR, 'workers': 4, 'transfer_bytes': 40 * GIB, 'cost_usd': 0,
        # §15.5's per-message rule: three attempts per target message, review included.
        'message_attempts': 3 * 329 * 50}
GAUGES = {'workers'}


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False, default=str) + '\n')
    os.replace(temp, path)


class Ledger:
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def initialise(cls, path) -> 'Ledger':
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                raise FileExistsError('ledger exists; cumulative accounting is never reset')
            atomic(path, {'schema': 'cp25-ledger-v1', 'caps': CAPS, 'counts': {}, 'peaks': {}, 'jobs': [],
                          'events': [], 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
        return cls(path)

    @contextlib.contextmanager
    def transaction(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.path.exists():
                raise FileNotFoundError(f'CP-25 ledger not initialised: {self.path}')
            state = json.loads(self.path.read_text())
            if state['caps'] != CAPS:
                raise ValueError('CP-25 resource contract changed; refuse to continue')
            yield state
            atomic(self.path, state)

    def read(self) -> dict:
        return json.loads(self.path.read_text())

    def reserve(self, limits=None, **increments):
        """Charge before use; refuse if any capped counter would pass its cap or stop line."""
        limits = limits or {}
        with self.transaction() as s:
            counts = s['counts']
            for key, value in increments.items():
                if value < 0 or key in GAUGES:
                    raise ValueError(f'invalid resource counter {key}={value}')
                if key in CAPS:
                    limit = min(CAPS[key], limits.get(key, CAPS[key]))
                    if counts.get(key, 0) + value > limit:
                        raise CapExceeded(f'CP-25 cap/stop line refuses next operation: {key} '
                                          f'{counts.get(key, 0)}+{value}>{limit}')
            for key, value in increments.items():
                counts[key] = counts.get(key, 0) + value
            return dict(counts)

    def headroom(self, **increments):
        counts = self.read()['counts']
        for key, value in increments.items():
            if key in CAPS and counts.get(key, 0) + value > CAPS[key]:
                raise CapExceeded(f'CP-25 cap refuses next operation: {key}')

    def event(self, name, **fields):
        with self.transaction() as s:
            s['events'].append({'event': name, 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields})

    def begin_job(self, name: str, workers: int, command: str = '') -> int:
        """Register a job; refuse it if it would exceed the worker cap or no machine time is left."""
        if not 1 <= workers <= CAPS['workers']:
            raise CapExceeded(f'workers={workers} outside 1..{CAPS["workers"]}')
        with self.transaction() as s:
            active = sum(j['workers'] for j in s['jobs'] if j.get('end_epoch') is None and _alive(j.get('pid')))
            if active + workers > CAPS['workers']:
                raise CapExceeded(f'{active} workers already active; {workers} more would exceed {CAPS["workers"]}')
            if s['counts'].get('machine_seconds', 0) >= CAPS['machine_seconds']:
                raise CapExceeded('machine-hour cap exhausted')
            s['jobs'].append({'name': name, 'workers': workers, 'command': command, 'pid': os.getpid(),
                              'start_epoch': time.time(), 'start_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                              'end_epoch': None, 'charged_seconds': 0.0})
            s['peaks']['workers'] = max(s['peaks'].get('workers', 0), active + workers)
            return len(s['jobs']) - 1

    def charge_job(self, index: int, final: bool = False, exit_code=None) -> float:
        """Charge the job's wall time x workers since its last charge; return machine seconds used."""
        with self.transaction() as s:
            job = s['jobs'][index]
            now = time.time()
            due = (now - job['start_epoch']) * job['workers'] - job['charged_seconds']
            job['charged_seconds'] += max(0.0, due)
            s['counts']['machine_seconds'] = s['counts'].get('machine_seconds', 0) + max(0.0, due)
            if final:
                job['end_epoch'], job['exit_code'] = now, exit_code
                job['end_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                job['wall_seconds'] = now - job['start_epoch']
            return s['counts']['machine_seconds']


def _alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
