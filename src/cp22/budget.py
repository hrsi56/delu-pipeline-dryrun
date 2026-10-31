"""Persistent CP-22 ceilings (capstone v21-r9 §20.8; §17.8's counting rules apply).

Every capped counter is reserved *before* the operation it pays for, under an exclusive file lock,
so a restart, a crash or a second process can neither reset nor bypass the cumulative ledger. The
caps are the Owner-approved maxima of §20.8, not estimates or targets. Nothing transfers from
CP-16, CP-20 or CP-21; their debits stay in their own ledgers.
"""
from __future__ import annotations

import contextlib
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from cp20.budget import atomic, dir_bytes, pid_alive  # noqa: F401  (pure helpers, re-exported)

GIB = 1024**3
HOUR = 3600

#: §20.8 hard ceilings. Distinct caps are simultaneous, not additive.
CAPS = {
    'lgbm_fits': 9000, 'main_lgbm_fits': 6000,
    'component_attempts': 1600, 'primitive_fits': 192000,
    'policy_days': 16000,
    'reference_passes': 3, 'analysis_passes': 3,
    'machine_seconds': 30 * HOUR, 'active_seconds': 32 * HOUR,
    'rss_bytes': 10 * GIB, 'additional_disk_bytes': 10 * GIB,
    'workers': 4,
    'download_bytes': 0, 'remote_writes': 0,
}
TIMEBOX_ACTIVE_SECONDS = 24 * HOUR
#: Uncapped informational counters, recorded cumulatively (every LightGBM fit is also inside
#: `lgbm_fits`; these split them by purpose and by inner/full-window role).
TRACKED = {
    'lgbm_inner_fits', 'lgbm_final_fits',
    'lgbm_fits_main', 'lgbm_fits_benchmark', 'lgbm_fits_control', 'lgbm_fits_daily_cycle',
    'lgbm_fits_review', 'lgbm_fits_repair', 'lgbm_fits_reproduction',
    'component_attempts_daily_cycle', 'component_attempts_control', 'component_attempts_review',
    'component_attempts_reproduction',
    'lasso_inner_fits', 'lasso_final_fits',
    'policy_days_admission', 'policy_days_evaluation', 'policy_days_parity', 'policy_days_control',
    'policy_days_daily_cycle', 'policy_days_review', 'policy_days_w_admission', 'policy_days_w_evaluation',
    'synthetic_fixture_fits',
}
GAUGES = {'rss_bytes', 'additional_disk_bytes', 'workers'}
#: Conservative effort start: the Lead's first command, 2026-10-01 15:37:39 IDT (12:37:39 UTC).
SESSION_START_EPOCH = datetime(2026, 10, 1, 12, 37, 39, tzinfo=ZoneInfo('UTC')).timestamp()


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
                raise FileNotFoundError(f'CP-22 ledger not initialised: {self.path}')
            state = json.loads(self.path.read_text())
            if state['caps'] != CAPS:
                raise ValueError('CP-22 resource contract changed; refuse to continue')
            yield state
            atomic(self.path, state)

    def initialise(self, baseline):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                raise FileExistsError('ledger exists; never reset cumulative accounting')
            atomic(self.path, {'schema': 'cp22-ledger-v1', 'caps': CAPS, 'counts': {}, 'peaks': {}, 'jobs': [],
                               'events': [], 'created_epoch': time.time(),
                               'effort': {'session_start_epoch': SESSION_START_EPOCH, 'pauses': []},
                               'disk_baseline': baseline})

    def read(self):
        return json.loads(self.path.read_text())

    def reserve(self, **increments):
        """Charge increments before use; refuse if any would exceed its cap."""
        with self.transaction() as s:
            counts = s['counts']
            for key, value in increments.items():
                if value < 0 or (key not in CAPS and key not in TRACKED) or key in GAUGES:
                    raise ValueError(f'invalid resource counter {key}={value}')
                if key in CAPS and counts.get(key, 0) + value > CAPS[key]:
                    raise CapExceeded(f'CP-22 cap refuses next operation: {key} {counts.get(key, 0)}+{value}>{CAPS[key]}')
            for key, value in increments.items():
                counts[key] = counts.get(key, 0) + value
            return dict(counts)

    def headroom(self, **increments):
        """Refuse (without charging) if these increments would exceed a cap."""
        counts = self.read()['counts']
        for key, value in increments.items():
            if key in CAPS and counts.get(key, 0) + value > CAPS[key]:
                raise CapExceeded(f'CP-22 cap refuses next operation: {key}')

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


def ledger_path() -> Path:
    project = Path(os.environ.get('CP22_PROJECT_ROOT', '/Users/djourno/Downloads/PJM'))
    return Path(os.environ.get('CP22_LEDGER', project / '.local/artifacts/cp-22/ledger/budget.json'))


def ledger() -> Budget:
    return Budget(ledger_path())
