"""Persistent CP-23 ceilings (capstone v21-r10 §21.8; §17.8's counting rules apply).

Every capped counter is reserved *before* the operation it pays for, under an exclusive file lock,
so a restart, a crash or a second process can neither reset nor bypass the cumulative ledger. The
caps are the maxima of §21.8, not estimates or targets. Nothing transfers from CP-16, CP-20, CP-21
or CP-22; their debits stay in their own ledgers.

One DDNN member fit is one network (one configuration, one seed) trained at one origin with early
stopping; every such fit counts, whatever its purpose (§21.8: "including 4.6R, controls,
reproduction, failures and review"). Main-run fits (the per-fold configuration selection and the
fits at the 638 origins) also count against the 4,000 main cap.
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

#: §21.8 hard ceilings. Distinct caps are simultaneous, not additive.
CAPS = {
    'ddnn_fits': 6000, 'main_ddnn_fits': 4000,
    'policy_days': 8000,
    'reference_passes': 3, 'analysis_passes': 3,
    'machine_seconds': 60 * HOUR, 'active_seconds': 40 * HOUR,
    'rss_bytes': 10 * GIB, 'additional_disk_bytes': 10 * GIB,
    'workers': 4,
    'data_download_bytes': 0, 'remote_writes': 0,
}
TIMEBOX_ACTIVE_SECONDS = 30 * HOUR
#: Uncapped informational counters, recorded cumulatively. Every DDNN fit is also inside
#: `ddnn_fits`; these split them by purpose. The one permitted download (the pinned PyTorch CPU
#: test dependency from PyPI, §21.8) is recorded in `test_dependency_download_bytes`, never in
#: `data_download_bytes`.
TRACKED = {
    'ddnn_fits_selection', 'ddnn_fits_warmup', 'ddnn_fits_evaluation',
    'ddnn_fits_resource_admission', 'ddnn_fits_control', 'ddnn_fits_reproduction', 'ddnn_fits_daily_cycle',
    'ddnn_fits_review', 'ddnn_fits_repair',
    'ddnn_epochs',
    'lgbm_fits_reproduction', 'component_attempts_reproduction', 'lasso_fits_reproduction',
    'policy_days_admission', 'policy_days_evaluation', 'policy_days_parity', 'policy_days_control',
    'policy_days_daily_cycle', 'policy_days_review',
    'synthetic_fixture_fits', 'reference_check_runs', 'test_dependency_download_bytes',
}
GAUGES = {'rss_bytes', 'additional_disk_bytes', 'workers'}
#: Conservative effort start: before the Lead's first command, 2026-10-04 16:30 IDT (13:30 UTC),
#: the time the issued brief's canonical copy was written.
SESSION_START_EPOCH = datetime(2026, 10, 4, 13, 30, 0, tzinfo=ZoneInfo('UTC')).timestamp()


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
                raise FileNotFoundError(f'CP-23 ledger not initialised: {self.path}')
            state = json.loads(self.path.read_text())
            if state['caps'] != CAPS:
                raise ValueError('CP-23 resource contract changed; refuse to continue')
            yield state
            atomic(self.path, state)

    def initialise(self, baseline):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                raise FileExistsError('ledger exists; never reset cumulative accounting')
            atomic(self.path, {'schema': 'cp23-ledger-v1', 'caps': CAPS, 'counts': {}, 'peaks': {}, 'jobs': [],
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
                    raise CapExceeded(f'CP-23 cap refuses next operation: {key} {counts.get(key, 0)}+{value}>{CAPS[key]}')
            for key, value in increments.items():
                counts[key] = counts.get(key, 0) + value
            return dict(counts)

    def headroom(self, **increments):
        """Refuse (without charging) if these increments would exceed a cap."""
        counts = self.read()['counts']
        for key, value in increments.items():
            if key in CAPS and counts.get(key, 0) + value > CAPS[key]:
                raise CapExceeded(f'CP-23 cap refuses next operation: {key}')

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
    project = Path(os.environ.get('CP23_PROJECT_ROOT', '/Users/djourno/Downloads/PJM'))
    return Path(os.environ.get('CP23_LEDGER', project / '.local/artifacts/cp-23/ledger/budget.json'))


def ledger() -> Budget:
    return Budget(ledger_path())


def charge_fits(budget: Budget, purpose: str, *, main: bool):
    """A callable that reserves one DDNN member fit (and its purpose counter) before the fit runs."""
    if purpose not in {'selection', 'warmup', 'evaluation', 'resource_admission', 'control', 'reproduction',
                       'daily_cycle', 'review', 'repair'}:
        raise ValueError(f'unknown DDNN fit purpose {purpose}')
    if main != (purpose in ('selection', 'warmup', 'evaluation')):
        raise ValueError('only selection, warm-up and evaluation fits are main-run fits')

    def charge(n: int = 1):
        extra = {'main_ddnn_fits': n} if main else {}
        budget.reserve(ddnn_fits=n, **{f'ddnn_fits_{purpose}': n}, **extra)
    return charge
