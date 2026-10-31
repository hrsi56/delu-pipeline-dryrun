"""Persistent CP-20 ceilings shared by every job, both environments and the review.

Standard library only: the GFS extraction environment (ecCodes) and the modelling
environment both import this module. Every capped counter is reserved *before* the
operation it pays for, under an exclusive file lock, so a restart, a crash or a second
process can neither reset nor bypass the cumulative ledger. Caps are the Owner-approved
v21-r4 section 15.5 ceilings; they are maxima, not feasibility guarantees or targets.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
from pathlib import Path
import time

GIB = 1024**3
HOUR = 3600

# Section 15.5 hard ceilings (distinct caps are simultaneous, not additive).
CAPS = {
    'component_attempts': 4000, 'main_component_attempts': 3000,
    'primitive_fits': 480000, 'inner_fits': 384000, 'final_fits': 96000,
    'policy_days': 9000, 'reference_passes': 3, 'analysis_passes': 3,
    'machine_seconds': 120 * HOUR, 'active_seconds': 80 * HOUR,
    'rss_bytes': 10 * GIB, 'additional_disk_bytes': 40 * GIB,
    'transfer_bytes': 160 * GIB, 'message_attempts': 3 * 123800,
    'workers': 4,
}
TIMEBOX_ACTIVE_SECONDS = 64 * HOUR
ATTEMPTS_PER_MESSAGE = 3
REQUIRED_RUNS = 2476
TARGET_MESSAGES = 123800
# Uncapped informational counters that are still recorded cumulatively.
TRACKED = {'requests', 'failed_requests', 'runs_completed', 'decoded_messages',
           'prerun_replacement_attempts',  # Owner decision O1; also inside message_attempts
           'message_tries', 'uncounted_message_tries', 'network_retries'}  # Owner decision O2
# Gauges are peaks or live sums, not cumulative consumption.
GAUGES = {'rss_bytes', 'additional_disk_bytes', 'workers'}
# Conservative effort start: the CP-20 session began before the ledger existed.
SESSION_START_EPOCH = 1790178600.0  # 2026-09-23T15:50:00Z


def atomic(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj, indent=1, sort_keys=True, allow_nan=False, default=str) + '\n')
    os.replace(temp, path)


def dir_bytes(paths):
    """Apparent bytes of regular files; tolerate atomic renames during the walk."""
    total = 0
    for base in paths:
        base = Path(base)
        if not base.exists():
            continue
        for root, _, files in os.walk(base):
            for name in files:
                try:
                    st = os.lstat(os.path.join(root, name))
                except FileNotFoundError:
                    continue
                if not os.path.islink(os.path.join(root, name)):
                    total += st.st_size
    return total


class CapExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def transaction(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.path.exists():
                raise FileNotFoundError(f'CP-20 ledger not initialised: {self.path}')
            state = json.loads(self.path.read_text())
            if state['caps'] != CAPS:
                raise ValueError('CP-20 resource contract changed; refuse to continue')
            yield state
            atomic(self.path, state)

    def initialise(self, baseline):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                raise FileExistsError('ledger exists; never reset cumulative accounting')
            atomic(self.path, {'schema': 'cp20-ledger-v1', 'caps': CAPS, 'counts': {},
                               'peaks': {}, 'jobs': [], 'events': [], 'created_epoch': time.time(),
                               'effort': {'session_start_epoch': SESSION_START_EPOCH, 'pauses': []},
                               'disk_baseline': baseline})

    def read(self):
        return json.loads(self.path.read_text())

    def reserve(self, limits=None, **increments):
        """Charge increments before use; refuse if any would exceed its cap or stop line."""
        limits = limits or {}
        with self.transaction() as s:
            counts = s['counts']
            for key, value in increments.items():
                if value < 0 or (key not in CAPS and key not in TRACKED) or key in GAUGES:
                    raise ValueError(f'invalid resource counter {key}={value}')
                if key in CAPS:
                    limit = min(CAPS[key], limits.get(key, CAPS[key]))
                    if counts.get(key, 0) + value > limit:
                        raise CapExceeded(f'CP-20 cap/stop line refuses next operation: {key} '
                                          f'{counts.get(key, 0)}+{value}>{limit}')
            for key, value in increments.items():
                counts[key] = counts.get(key, 0) + value
            return dict(counts)

    def headroom(self, **increments):
        """Refuse (without charging) if a later charge of these increments would exceed a cap."""
        counts = self.read()['counts']
        for key, value in increments.items():
            if key in CAPS and counts.get(key, 0) + value > CAPS[key]:
                raise CapExceeded(f'CP-20 cap refuses next operation: {key}')

    def add_time(self, seconds):
        """Machine time is charged as it elapses; the monitor aborts at the cap."""
        with self.transaction() as s:
            s['counts']['machine_seconds'] = s['counts'].get('machine_seconds', 0) + seconds
            return s['counts']['machine_seconds']

    def event(self, name, **fields):
        with self.transaction() as s:
            s['events'].append({'event': name, 'epoch': time.time(), **fields})

    def peak(self, key, value):
        with self.transaction() as s:
            s['peaks'][key] = max(s['peaks'].get(key, 0), value)

    @staticmethod
    def active_seconds(state, now=None):
        now = time.time() if now is None else now
        effort = state['effort']
        paused = sum(b - a for a, b in effort['pauses'])
        if effort.get('paused_since'):
            paused += now - effort['paused_since']
        return now - effort['session_start_epoch'] - paused


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
