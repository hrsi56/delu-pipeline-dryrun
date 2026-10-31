"""Persistent CP-24 ceilings (capstone v21-r11 §23.11; §17.8's counting rules apply; §23.6's raises).

Every capped counter is reserved *before* the operation it pays for, under an exclusive file lock, so
a restart, a crash or a second process can neither reset nor bypass the cumulative ledger. The caps
are §23.11's maxima, not estimates or targets. Nothing transfers from CP-16 or CP-20 to CP-23; their
debits stay in their own ledgers.

**Raises (§23.6).** Only three ceilings can be raised, each by a stated amount, by an S1 or S2 answer
committed under `docs/track-b/evidence/cp-24/steering/` before use: DDNN-2 member fits up to 60,000,
machine-hours up to 200 and active hours up to 70. A raise is recorded in the ledger with the
steering file's path, its SHA-256 and the commit that holds it; the effective cap is the raised one.
Every other ceiling never changes.

**Units.**

* One DDNN-2 member fit is one network (one configuration, one seed) trained once with early
  stopping, at one origin or for one validation batch. Every fit counts, whatever its purpose:
  4.6R', the search, the gate, the attempts, controls, reproduction, failures and review.
* v4's members on the gate days: one pass at the 280 gate origins (one origin = A1_w, B2_w, L-N and
  L-R), reused across rounds; the parity checks are counted separately and are not a second pass.
* Policy-days: one policy issued (or replayed) for one delivery day, gate days included.
* A bootstrap pass is one run over all of a scored attempt's contrasts (2,000 replicates).
"""
from __future__ import annotations

import contextlib
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

from cp20.budget import atomic, dir_bytes, pid_alive  # noqa: F401  (pure helpers, re-exported)

GIB = 1024**3
HOUR = 3600

#: §23.11 hard ceilings. Distinct caps are simultaneous, not additive.
CAPS = {
    'ddnn2_fits': 40000,
    'v4_gate_origins': 280,
    'policy_days': 12000,
    'reference_passes': 3, 'bootstrap_passes': 6,
    'scored_attempts': 2, 'rounds_before_attempt_1': 3, 'rounds_before_attempt_2': 1,
    'machine_seconds': 150 * HOUR, 'active_seconds': 50 * HOUR,
    'rss_bytes': 10 * GIB, 'additional_disk_bytes': 10 * GIB,
    'workers': 4,
    'data_download_bytes': 0, 'remote_writes': 0, 'cost_usd': 0,
}
#: The only raisable ceilings and their §23.6 maxima.
RAISABLE = {'ddnn2_fits': 60000, 'machine_seconds': 200 * HOUR, 'active_seconds': 70 * HOUR}
TIMEBOX_ACTIVE_SECONDS = 35 * HOUR
#: Uncapped informational counters, recorded cumulatively. Every DDNN-2 fit is also inside
#: `ddnn2_fits`; these split it by purpose. The one permitted download (CP-23's pinned PyTorch CPU
#: wheel set, only if the local cache lacks it) would be recorded in `test_dependency_download_bytes`,
#: never in `data_download_bytes`.
PURPOSES = ('resource_admission', 'search', 'gate', 'warmup', 'evaluation', 'control', 'reproduction',
            'daily_cycle', 'review', 'repair', 'correctness')
TRACKED = {
    *(f'ddnn2_fits_{p}' for p in PURPOSES),
    'ddnn2_epochs',
    'v4_parity_origins', 'v4_control_origins', 'v4_review_origins',
    'lgbm_fits', 'lgbm_fits_gate', 'lgbm_fits_parity', 'lgbm_fits_control', 'lgbm_fits_review',
    'component_attempts', 'component_attempts_gate', 'component_attempts_parity', 'component_attempts_control',
    'component_attempts_review', 'primitive_fits', 'lasso_inner_fits', 'lasso_final_fits',
    'policy_days_gate', 'policy_days_admission', 'policy_days_evaluation', 'policy_days_parity',
    'policy_days_control', 'policy_days_daily_cycle', 'policy_days_review',
    'rounds', 'synthetic_fixture_fits', 'reference_check_runs', 'test_dependency_download_bytes',
}
GAUGES = {'rss_bytes', 'additional_disk_bytes', 'workers'}
#: Conservative effort start: before the Lead's first command, 2026-10-05 06:14 IDT (03:14 UTC), the
#: time the issued brief's canonical copy was written.
SESSION_START_EPOCH = datetime(2026, 10, 5, 3, 14, 0, tzinfo=ZoneInfo('UTC')).timestamp()
STEERING = 'docs/track-b/evidence/cp-24/steering/'


class CapExceeded(RuntimeError):
    pass


def effective_caps(state: dict) -> dict:
    """§23.11's caps with every recorded, committed §23.6 raise applied (never above its maximum)."""
    caps = dict(CAPS)
    for raise_ in state.get('raises', []):
        key, value = raise_['counter'], raise_['new_cap']
        if key not in RAISABLE or value > RAISABLE[key] or value < CAPS[key]:
            raise ValueError(f'invalid recorded raise {raise_}')
        caps[key] = max(caps[key], value)
    return caps


class Budget:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def transaction(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.path.exists():
                raise FileNotFoundError(f'CP-24 ledger not initialised: {self.path}')
            state = json.loads(self.path.read_text())
            if state['caps'] != CAPS:
                raise ValueError('CP-24 resource contract changed; refuse to continue')
            yield state
            atomic(self.path, state)

    def initialise(self, baseline):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                raise FileExistsError('ledger exists; never reset cumulative accounting')
            atomic(self.path, {'schema': 'cp24-ledger-v1', 'caps': CAPS, 'raisable': RAISABLE, 'raises': [],
                               'counts': {}, 'peaks': {}, 'jobs': [], 'events': [], 'created_epoch': time.time(),
                               'effort': {'session_start_epoch': SESSION_START_EPOCH, 'pauses': []},
                               'disk_baseline': baseline})

    def read(self):
        return json.loads(self.path.read_text())

    def caps(self) -> dict:
        return effective_caps(self.read())

    def reserve(self, **increments):
        """Charge increments before use; refuse if any would exceed its (effective) cap."""
        with self.transaction() as s:
            caps = effective_caps(s)
            counts = s['counts']
            for key, value in increments.items():
                if value < 0 or (key not in CAPS and key not in TRACKED) or key in GAUGES:
                    raise ValueError(f'invalid resource counter {key}={value}')
                if key in caps and counts.get(key, 0) + value > caps[key]:
                    raise CapExceeded(f'CP-24 cap refuses next operation: {key} {counts.get(key, 0)}+{value}>{caps[key]}')
            for key, value in increments.items():
                counts[key] = counts.get(key, 0) + value
            return dict(counts)

    def headroom(self, **increments):
        """Refuse (without charging) if these increments would exceed a cap."""
        state = self.read()
        caps, counts = effective_caps(state), state['counts']
        for key, value in increments.items():
            if key in caps and counts.get(key, 0) + value > caps[key]:
                raise CapExceeded(f'CP-24 cap refuses next operation: {key} {counts.get(key, 0)}+{value}>{caps[key]}')

    def event(self, name, **fields):
        with self.transaction() as s:
            s['events'].append({'event': name, 'epoch': time.time(), **fields})

    def peak(self, key, value):
        with self.transaction() as s:
            s['peaks'][key] = max(s['peaks'].get(key, 0), value)

    def record_raise(self, root: Path, counter: str, new_cap, steering_file: str) -> dict:
        """Apply a §23.6 raise stated in a steering answer committed at HEAD under `steering/`."""
        root = Path(root)
        if counter not in RAISABLE:
            raise ValueError(f'{counter} is not raisable (§23.6: only member fits, machine-hours and active hours)')
        if not steering_file.startswith(STEERING):
            raise ValueError('a raise must cite a committed steering answer under ' + STEERING)
        committed = subprocess.check_output(['git', 'show', f'HEAD:{steering_file}'], cwd=root)
        if committed != (root / steering_file).read_bytes():
            raise ValueError('the steering answer is not committed at HEAD')
        commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', steering_file], cwd=root, text=True).strip()
        with self.transaction() as s:
            caps = effective_caps(s)
            if new_cap > RAISABLE[counter] or new_cap <= caps[counter]:
                raise ValueError(f'raise of {counter} to {new_cap} is outside ({caps[counter]}, {RAISABLE[counter]}]')
            item = {'counter': counter, 'old_cap': caps[counter], 'new_cap': new_cap, 'steering_file': steering_file,
                    'steering_sha256': hashlib.sha256(committed).hexdigest(), 'commit': commit, 'epoch': time.time()}
            s['raises'].append(item)
            s['events'].append({'event': 'cap_raise', **item})
        return item

    @staticmethod
    def active_seconds(state, now=None):
        now = time.time() if now is None else now
        effort = state['effort']
        paused = sum(b - a for a, b in effort['pauses'])
        if effort.get('paused_since'):
            paused += now - effort['paused_since']
        return now - effort['session_start_epoch'] - paused


def project() -> Path:
    return Path(os.environ.get('CP24_PROJECT_ROOT', '/Users/djourno/Downloads/PJM'))


def ledger_path() -> Path:
    return Path(os.environ.get('CP24_LEDGER', project() / '.local/artifacts/cp-24/ledger/budget.json'))


def ledger() -> Budget:
    return Budget(ledger_path())


def charge_fits(budget: Budget, purpose: str):
    """A callable that reserves `n` DDNN-2 member fits (and the purpose counter) before they run."""
    if purpose not in PURPOSES:
        raise ValueError(f'unknown DDNN-2 fit purpose {purpose}')

    def charge(n: int = 1):
        budget.reserve(ddnn2_fits=n, **{f'ddnn2_fits_{purpose}': n})
    return charge


class LedgerAdapter:
    """The budget interface `cp21.components.refit` expects, charging CP-24's counters.

    CP-21's counted LEAR wrapper reserves `component_attempts`, `component_attempts_<purpose>`,
    `primitive_fits` and `lasso_inner_fits` / `lasso_final_fits`; all of them are tracked here."""

    def __init__(self, budget: Budget):
        self.budget = budget

    def reserve(self, **increments):
        return self.budget.reserve(**increments)
