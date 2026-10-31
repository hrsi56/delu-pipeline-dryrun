"""Independent local worker supervisor for CP-26 resource receipts.

The launcher, supervisor and detached watchdog check independent progress. The
watchdog can stop bound workers and receipt their duration when a supervisor is
stopped, wedged or killed, including after launcher loss. Unknown duration still
blocks admission; a restart never substitutes its current clock for worker time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait

_SUPERVISOR_BEGAN = time.monotonic()
from .limits import (STARTUP_GRACE_SECONDS, HEARTBEAT_TIMEOUT_SECONDS,
                     POLL_SECONDS, LEDGER_TIMEOUT_SECONDS)

SUPERVISION_POLICY = {'startup_grace_seconds': STARTUP_GRACE_SECONDS,
                      'heartbeat_timeout_seconds': HEARTBEAT_TIMEOUT_SECONDS,
                      'poll_seconds': POLL_SECONDS,
                      'ledger_timeout_seconds': LEDGER_TIMEOUT_SECONDS}

import psutil

from cp25.guard import redact
from .budget import Budget, CAPS, CapExceeded, add_seconds, atomic, limit_reason, unresolved_reserves
from .budget import total_seconds
from .limits import shutdown_exposure_seconds
from .harness import art


def local_path(path: Path) -> Path:
    path = path.resolve()
    if not path.is_relative_to(art().resolve()):
        raise ValueError('job receipts must stay inside primary .local/artifacts/cp-26')
    return path


def process_identity(pid: int | None = None) -> dict | None:
    try:
        proc = psutil.Process(os.getpid() if pid is None else pid)
        return {'pid': proc.pid, 'create_time': proc.create_time()}
    except psutil.NoSuchProcess:
        return None


def identity_status(identity: dict | None) -> str:
    """A reused PID is never the old worker; a missing identity is not proof of death."""
    if not identity or identity.get('create_time') is None:
        return 'unknown'
    try:
        proc = psutil.Process(identity['pid'])
        if proc.create_time() != identity['create_time']:
            return 'reused'
        return 'gone' if proc.status() == psutil.STATUS_ZOMBIE else 'same'
    except psutil.NoSuchProcess:
        return 'gone'
    except psutil.AccessDenied:
        return 'unknown'


def group_members(pgid: int) -> list[dict]:
    """Read numeric membership as untrusted observations; it does not confer ownership."""
    members = []
    for proc in psutil.process_iter(['pid', 'create_time', 'status']):
        try:
            if os.getpgid(proc.pid) == pgid and proc.info['status'] != psutil.STATUS_ZOMBIE:
                members.append({'pid': proc.pid, 'create_time': proc.info['create_time']})
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
        except (PermissionError, psutil.AccessDenied) as exc:
            raise RuntimeError('cannot verify the CP-26 worker process group') from exc
    return members


def identities_rss(identities: list[dict]) -> int:
    total = 0
    for identity in identities:
        if identity_status(identity) != 'same':
            continue
        try:
            total += psutil.Process(identity['pid']).memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def _finite(value, *, minimum=0.0) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= minimum)


def freshness_reason(job: dict, role: str, *, now: float | None = None) -> str | None:
    """A live PID is insufficient: bind fresh advancing samples to elapsed/RSS accounting."""
    now = time.monotonic() if now is None else now
    observed = job.get(f'{role}_observed_monotonic')
    sequence = job.get(f'{role}_sample_sequence')
    wall = job.get(f'{role}_sample_wall_seconds')
    began = job.get('accounting_began_monotonic')
    charged = job.get('charged_wall_seconds')
    accounting_observed = job.get('accounting_observed_monotonic')
    accounting_sequence = job.get('accounting_sample_sequence')
    if (job.get('supervision_policy') != SUPERVISION_POLICY
            or not _finite(observed) or not isinstance(sequence, int) or isinstance(sequence, bool)
            or sequence < 1 or not _finite(wall) or not _finite(began)
            or not _finite(charged) or not _finite(accounting_observed)
            or not isinstance(accounting_sequence, int) or isinstance(accounting_sequence, bool)
            or accounting_sequence < 1 or not _finite(job.get('live_rss_bytes'))):
        return f'{role}_progress_not_evidenced'
    if (not 0 <= now - observed <= HEARTBEAT_TIMEOUT_SECONDS
            or not 0 <= now - accounting_observed <= HEARTBEAT_TIMEOUT_SECONDS):
        return f'{role}_heartbeat_stale'
    # Every heartbeat is generated only after a charged time/RSS sample. Advancing
    # an epoch alone, or copying an old wall sample, cannot establish supervision.
    if (abs(observed - began - wall) > POLL_SECONDS
            or charged + 1e-8 < wall
            or charged > accounting_observed - began + POLL_SECONDS):
        return f'{role}_accounting_not_evidenced'
    return None


def require_fresh_supervision(job: dict, *, status=None, now=None) -> None:
    status = identity_status if status is None else status
    for role in ('supervisor', 'watchdog'):
        if status(job.get(f'{role}_identity')) != 'same':
            raise RuntimeError(f'CP-26 {role} identity is not evidenced; compute is stopped')
        reason = freshness_reason(job, role, now=now)
        if reason:
            raise RuntimeError(f'CP-26 supervision is stale or unmeasured: {reason}; compute is stopped')


def startup_is_fresh(job: dict, *, now=None) -> bool:
    now = time.monotonic() if now is None else now
    reserved = job.get('reserved_monotonic')
    return (job.get('phase') in ('reserved', 'starting') and job.get('child_identity') is None
            and job.get('supervision_policy') == SUPERVISION_POLICY and _finite(reserved)
            and 0 <= now - reserved <= STARTUP_GRACE_SECONDS)


def register_worker(budget: Budget | None = None) -> dict:
    """Register before detached/parallel worker computation, while ancestry exists.

    The supervised child counts as a compute worker conservatively, even while it
    coordinates others. Each additional live registered process consumes one of the
    declared slots. A denied identity is persisted before raising, so it remains
    discoverable and cannot disappear from ending evidence.
    """
    job_id, index_text = os.environ.get('CP26_JOB_ID'), os.environ.get('CP26_JOB_INDEX')
    if not job_id or not index_text:
        raise RuntimeError('CP26_JOB_ID and CP26_JOB_INDEX are required for worker registration')
    identity = process_identity()
    if identity is None:
        raise RuntimeError('worker registration requires its process identity')
    budget = budget or Budget()
    violation = None
    with budget.transaction(timeout_seconds=LEDGER_TIMEOUT_SECONDS) as state:
        job = _job(state, int(index_text), job_id)
        if not job['running'] or job.get('phase') != 'supervising':
            raise RuntimeError('CP-26 worker registration closed; refuse to compute')
        require_fresh_supervision(job)
        workers = job.setdefault('worker_identities', [])
        if identity not in workers:
            workers.append(identity)
        proposed = list(job.get('registered_worker_identities', []))
        if identity not in proposed:
            proposed.append(identity)
        own_active = active_compute_identities(job, registered=proposed)
        declared = sum(j['workers'] for j in state['jobs'] if j['running'])
        global_active = len(own_active) + sum(len(active_compute_identities(j))
                                             for j in state['jobs'] if j['running'] and j is not job)
        if (job.get('child_identity') is None or len(own_active) > job['workers']
                or declared > CAPS['workers'] or global_active > CAPS['workers']):
            violation = {'job_allocation': job['workers'], 'job_active_compute_workers': len(own_active),
                         'global_declared_workers': declared, 'global_active_compute_workers': global_active}
            denied = job.setdefault('denied_worker_identities', [])
            if identity not in denied:
                denied.append(identity)
            job['worker_ceiling_error'] = violation
            state['events'].append({'event': 'worker_registration_denied', 'epoch': time.time(),
                                    'job_id': job_id, 'identity': identity, **violation})
        else:
            job['registered_worker_identities'] = proposed
    if violation is not None:
        raise CapExceeded('CP-26 registered compute-worker ceiling refuses worker before computation')
    return identity


def active_compute_identities(job: dict, *, registered: list | None = None) -> dict:
    """Count the root plus registered/denied workers; unknown identities reserve a slot."""
    active = {}
    identities = [job.get('child_identity'),
                  *(job.get('registered_worker_identities', []) if registered is None else registered),
                  *job.get('denied_worker_identities', [])]
    for identity in identities:
        if identity is not None and identity_status(identity) in ('same', 'unknown'):
            active[(identity['pid'], identity.get('create_time'))] = identity
    return active


def root_owns_group(pgid: int, child_identity: dict) -> bool:
    """Numeric PGID is ownership only while its original session root is verified."""
    if identity_status(child_identity) != 'same':
        return False
    try:
        owned = (child_identity['pid'] == pgid and os.getpgid(pgid) == pgid and os.getsid(pgid) == pgid)
    except ProcessLookupError:
        return False
    if not owned:
        raise RuntimeError('CP-26 original worker no longer owns its recorded session; compute is stopped')
    return identity_status(child_identity) == 'same'


def worker_members(pgid: int, child_identity: dict, known: dict) -> list[dict]:
    """Discover only under bound ancestry/session; never adopt a reused numeric group."""
    found = {}
    for identity in [child_identity, *list(known.values())]:
        observed = identity_status(identity)
        if observed == 'unknown':
            raise RuntimeError('CP-26 worker identity cannot be inspected; compute is stopped')
        if observed != 'same':
            continue
        found[(identity['pid'], identity['create_time'])] = identity
        try:
            proc = psutil.Process(identity['pid'])
            if proc.create_time() != identity['create_time']:
                continue
            descendants = []
            for child in proc.children(recursive=True):
                if child.status() != psutil.STATUS_ZOMBIE:
                    descendants.append({'pid': child.pid, 'create_time': child.create_time()})
            if identity_status(identity) == 'same':
                found.update({(m['pid'], m['create_time']): m for m in descendants})
        except psutil.NoSuchProcess:
            pass
    owns_group = root_owns_group(pgid, child_identity)
    group = group_members(pgid)
    if owns_group and root_owns_group(pgid, child_identity):
        found.update({(m['pid'], m['create_time']): m for m in group})
    else:
        bound = set(known) | set(found) | {(child_identity['pid'], child_identity['create_time'])}
        if any((m['pid'], m['create_time']) not in bound for m in group):
            raise RuntimeError('unowned numeric CP-26 process group after root exit/PID reuse; compute is stopped')
        found.update({(m['pid'], m['create_time']): m for m in group})
    known.update(found)
    return list(found.values())


def ending_evidence(pgid: int | None, child_identity: dict | None, known: dict,
                    previously_stopped: bool) -> dict:
    """An ownership problem cannot become a positive terminal ending assertion."""
    try:
        stopped = previously_stopped and (pgid is None or not worker_members(pgid, child_identity, known))
        return {'workers_stopped': stopped, 'group_empty': stopped, 'ending_error': None}
    except RuntimeError as exc:
        return {'workers_stopped': False, 'group_empty': False, 'ending_error': redact(str(exc))}


def check_completion(job: dict, receipt: dict) -> float:
    if (receipt.get('schema') != 'cp26-completion-v2'
            or receipt.get('job_id') != job.get('job_id')
            or receipt.get('job_index') != job.get('job_index')
            or receipt.get('started_epoch') != job.get('started_epoch')
            or receipt.get('request_sha256') != job.get('request_sha256')
            or receipt.get('workers') != job.get('workers')):
        raise RuntimeError('completion receipt does not identify the interrupted CP-26 job')
    if receipt.get('supervisor_identity') != job.get('supervisor_identity'):
        raise RuntimeError('completion receipt has a different supervisor process identity')
    if receipt.get('child_identity') != job.get('child_identity'):
        raise RuntimeError('completion receipt has a different worker process identity')
    if job.get('watchdog_identity') is not None:
        if receipt.get('watchdog_identity') != job['watchdog_identity']:
            raise RuntimeError('completion receipt has a different watchdog process identity')
        issuer = receipt.get('completion_issuer')
        if (issuer not in ('supervisor', 'watchdog', 'monitor')
                or receipt.get('completion_issuer_identity') != job.get(f'{issuer}_identity')):
            raise RuntimeError('completion receipt has no bound independent issuer')
    if not receipt.get('workers_stopped') or not receipt.get('group_empty'):
        raise RuntimeError('completion receipt has no sufficient worker-ending evidence; compute is stopped')
    expected_workers = {(i['pid'], i['create_time']) for i in job.get('worker_identities', [])}
    receipted_workers = {(i['pid'], i['create_time']) for i in receipt.get('worker_identities', [])}
    if not expected_workers.issubset(receipted_workers):
        raise RuntimeError('completion receipt omits recorded workers; compute is stopped')
    wall, charged = receipt.get('wall_seconds'), job.get('charged_wall_seconds')
    if (isinstance(wall, bool) or isinstance(charged, bool)
            or not isinstance(wall, (float, int)) or not isinstance(charged, (float, int))
            or not math.isfinite(wall) or not math.isfinite(charged)
            or wall < 0 or charged < 0 or charged > wall + 1e-8):
        raise RuntimeError('completion receipt does not bound the already measured worker duration')
    return max(0.0, wall - charged) * job['workers']


def apply_completion(state: dict, job: dict, receipt: dict) -> None:
    """Receipt reconciliation is exactly once; never use a restart's current time."""
    tail = check_completion(job, receipt)
    if not job['running']:
        if job.get('wall_seconds') != receipt['wall_seconds']:
            raise RuntimeError('completed CP-26 job conflicts with its immutable receipt')
        return
    add_seconds(state, tail)
    job.update(running=False, live_rss_bytes=0, exit_code=receipt['exit_code'],
               abort_reason=receipt['abort_reason'], ended_epoch=receipt['ended_epoch'],
               wall_seconds=receipt['wall_seconds'], charged_wall_seconds=receipt['wall_seconds'],
               workers_stopped=True, group_empty=True,
               peak_rss_bytes=receipt['peak_rss_bytes'], completion_schema=receipt['schema'])


def completion_receipt(job: dict) -> dict:
    path = local_path(Path(job.get('completion_clock', art() / 'jobs' / 'missing-clock.json')))
    if not path.exists():
        raise RuntimeError('dead CP-26 supervisor has no completion clock: worker runtime is uncertain; '
                           'new compute is stopped, and restart idle time is not charged')
    return json.loads(path.read_text())


def reconcile_running(state: dict, *, status=identity_status, load_receipt=completion_receipt,
                      now=None) -> None:
    """Refuse orphan/unknown processes and reconcile only independently completed jobs."""
    for job in state['jobs']:
        if not job['running']:
            continue
        monitor_identity = job.get('monitor_identity')
        if monitor_identity is None and job.get('monitor_create_time') is not None:
            monitor_identity = {'pid': job['monitor_pid'], 'create_time': job['monitor_create_time']}
        monitor = status(monitor_identity)
        if monitor == 'same':
            if startup_is_fresh(job, now=now):
                # This grace reserves slots only. The exec gate prevents compute
                # before both independent accounting loops have sampled it.
                continue
            supervisor = status(job.get('supervisor_identity'))
            if supervisor == 'same':
                require_fresh_supervision(job, status=status, now=now)
                continue
            if supervisor == 'unknown':
                raise RuntimeError('CP-26 supervisor identity is not evidenced; compute is stopped')
        elif monitor == 'unknown':
            raise RuntimeError('legacy/unknown CP-26 monitor identity; compute is stopped')
        elif status(job.get('supervisor_identity')) == 'same':
            require_fresh_supervision(job, status=status, now=now)
            raise RuntimeError('orphaned CP-26 monitor: supervised worker remains alive; await its completion')
        identities = [job.get('child_identity'), *job.get('worker_identities', [])]
        for identity in identities:
            if identity is None:
                continue
            observed = status(identity)
            if observed == 'same':
                raise RuntimeError('orphaned CP-26 worker remains alive; compute is stopped')
            if observed == 'unknown':
                raise RuntimeError('CP-26 worker ending is uncertain; compute is stopped')
        receipt = load_receipt(job)
        apply_completion(state, job, receipt)


def _job(state: dict, index: int, job_id: str) -> dict:
    job = state['jobs'][index]
    if job.get('job_id') != job_id:
        raise RuntimeError('CP-26 job identity changed; refuse resource mutation')
    return job


def _resource_state(budget) -> dict:
    # Keep the read-only snapshot test-double interface; production Budget always
    # uses the verified resource view rather than decoding cold task history.
    read = getattr(budget, 'resource_read', None)
    return read() if read is not None else budget.read()


def _completion_clock_token(state: dict, index: int, job_id: str):
    job = _job(state, index, job_id)
    if not job.get('completion_clock'):
        return None
    path = local_path(Path(job['completion_clock']))
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return str(path), None
    return str(path), hashlib.sha256(raw).hexdigest()


def _apply_recorded_completion(state: dict, job: dict) -> bool:
    if not job.get('completion_clock'):
        return False
    path = local_path(Path(job['completion_clock']))
    try:
        receipt = json.loads(path.read_bytes())
    except FileNotFoundError:
        return False
    # A positive bound receipt is authoritative for ending time even if its
    # writer died before the ledger rename. Invalid/uncertain clocks fail closed.
    apply_completion(state, job, receipt)
    return True


def _update_live_job(budget: Budget, index: int, job_id: str, callback):
    """Persist startup hot fields without serializing cold history."""
    def mutate(state):
        job = _job(state, index, job_id)
        if _apply_recorded_completion(state, job) or not job['running']:
            return False
        callback(job)
        return True
    budget.resource_read()
    live = budget.resource_update(mutate, timeout_seconds=LEDGER_TIMEOUT_SECONDS,
                                 guard=lambda state: _completion_clock_token(state, index, job_id))
    if not live:
        raise RuntimeError('CP-26 job already ended; refuse worker admission or launch mutation')


def _bind_supervisor_start(job: dict, request_sha: str, own_identity: dict, began: float):
    """Consume a reserved launch exactly once; never rebase a live elapsed clock."""
    if request_sha != job['request_sha256'] or not job['running']:
        raise RuntimeError('CP-26 supervisor request does not bind the running job')
    if (not own_identity or not isinstance(own_identity.get('pid'), int)
            or isinstance(own_identity['pid'], bool) or own_identity['pid'] <= 0
            or not _finite(own_identity.get('create_time')) or not _finite(began)):
        raise RuntimeError('CP-26 supervisor startup identity or elapsed origin is not evidenced')
    if (job.get('supervisor_identity') not in (None, own_identity)
            or job.get('supervisor_pid') not in (None, own_identity['pid'])):
        raise RuntimeError('CP-26 supervisor startup identity changed; refuse duplicate launch')
    if (job.get('phase') != 'reserved' or 'accounting_began_monotonic' in job
            or job.get('child_identity') is not None or job.get('child_pid') is not None):
        raise RuntimeError('CP-26 supervisor startup already consumed; refuse duplicate launch')
    # All guards precede every mutation, including a same-process request replay.
    job.update(supervisor_identity=own_identity, supervisor_pid=own_identity['pid'],
               phase='starting', accounting_began_monotonic=began)


def tick(budget: Budget, index: int, job_id: str, wall: float, rss: int,
         *, sample_role: str | None = None, **updates) -> str | None:
    def sample(state):
        job = _job(state, index, job_id)
        if _apply_recorded_completion(state, job):
            return limit_reason(state)
        if not job['running'] or not _finite(wall) or not _finite(rss):
            raise RuntimeError('CP-26 elapsed clock regressed or job already ended')
        if sample_role is None and wall < job['charged_wall_seconds']:
            raise RuntimeError('CP-26 elapsed clock regressed or job already ended')
        if sample_role and wall + 1e-8 < job.get(f'{sample_role}_sample_wall_seconds', 0):
            raise RuntimeError('CP-26 observer elapsed clock regressed')
        # Independent samplers share one monotonic origin. A sample captured just
        # before the other loop commits cannot regress or double-charge the clock.
        sampled_rss = max(rss, job['live_rss_bytes']) if wall < job['charged_wall_seconds'] else rss
        charged = max(wall, job['charged_wall_seconds'])
        add_seconds(state, (charged - job['charged_wall_seconds']) * job['workers'])
        sample_updates = dict(updates)
        if 'worker_identities' in sample_updates:
            merged = {(i['pid'], i['create_time']): i for i in job.get('worker_identities', [])}
            merged.update({(i['pid'], i['create_time']): i for i in sample_updates['worker_identities']})
            sample_updates['worker_identities'] = list(merged.values())
        if 'peak_rss_bytes' in sample_updates:
            sample_updates['peak_rss_bytes'] = max(sample_updates['peak_rss_bytes'], job.get('peak_rss_bytes', 0))
        if sample_role is not None:
            if sample_role not in ('supervisor', 'watchdog'):
                raise ValueError('unknown CP-26 accounting sampler')
            observed = job['accounting_began_monotonic'] + wall
            sample_updates.update({f'{sample_role}_observed_monotonic': observed,
                            f'{sample_role}_heartbeat_epoch': time.time(),
                            f'{sample_role}_sample_wall_seconds': wall,
                            f'{sample_role}_sample_child_identity': job.get('child_identity'),
                            f'{sample_role}_sample_sequence': job.get(f'{sample_role}_sample_sequence', 0) + 1,
                            f'{sample_role}_ledger_lock_telemetry': budget.telemetry(),
                            'accounting_observed_monotonic': job['accounting_began_monotonic'] + charged,
                            'accounting_sample_sequence': job.get('accounting_sample_sequence', 0) + 1})
        job.update(charged_wall_seconds=charged, last_charge_epoch=time.time(), live_rss_bytes=sampled_rss,
                   **sample_updates)
        aggregate = sum(j['live_rss_bytes'] for j in state['jobs'] if j['running'])
        state['peaks']['rss_bytes'] = max(aggregate, state['peaks'].get('rss_bytes', 0))
        reason = limit_reason(state)
        if aggregate >= CAPS['rss_bytes']:
            reason = 'aggregate_memory_cap'
        if unresolved_reserves(state):
            reason = 'unreconciled_reserve'
        if job.get('worker_ceiling_error'):
            reason = 'registered_worker_ceiling'
        return reason
    budget.resource_read()
    return budget.resource_update(sample, timeout_seconds=LEDGER_TIMEOUT_SECONDS,
                                  guard=lambda state: _completion_clock_token(state, index, job_id))


def observer_tick(budget, snapshot, known, *, sample_role, wall, rss, **updates):
    """Keep direct ownership/time/RSS enforcement live while the ledger refreshes.

    The one I/O thread does no fitting. Its finite transaction is still exactly
    the normal tick; independent checks never acquire the ledger lock. Workers
    are stopped before waiting for an in-flight accounting operation to finish.
    """
    state = _resource_state(budget)
    early_reason = None
    with ThreadPoolExecutor(max_workers=1) as writer:
        future = writer.submit(tick, budget, snapshot['job_index'], snapshot['job_id'],
                               wall, rss, sample_role=sample_role, **updates)
        while not future.done():
            root = snapshot.get('child_identity')
            if root is not None:
                members = worker_members(snapshot['worker_pgid'], root, known)
                for identity in members:
                    known[(identity['pid'], identity['create_time'])] = identity
            live_rss, uncharged, allocation = 0, 0., 0
            now = time.monotonic()
            for job in state['jobs']:
                if not job['running']:
                    continue
                allocation += job['workers']
                began = job.get('accounting_began_monotonic')
                if _finite(began):
                    uncharged += max(0., now - began - job['charged_wall_seconds']) * job['workers']
                identities = { (i['pid'], i['create_time']): i for i in
                    [job.get('child_identity'), *job.get('worker_identities', []),
                     job.get('monitor_identity'), job.get('supervisor_identity'),
                     job.get('watchdog_identity')] if i is not None }
                if job['job_id'] == snapshot['job_id']:
                    identities.update(known)
                live_rss += identities_rss(list(identities.values()))
            if total_seconds(state) + uncharged + shutdown_exposure_seconds(allocation) >= CAPS['machine_seconds']:
                early_reason = 'machine_hours_cap'
            elif live_rss >= CAPS['rss_bytes']:
                early_reason = 'aggregate_memory_cap'
            peer = 'watchdog' if sample_role == 'supervisor' else 'supervisor'
            if identity_status(snapshot.get(peer + '_identity')) != 'same':
                early_reason = early_reason or peer + '_identity_lost'
            if early_reason:
                if root is not None:
                    terminate_workers(snapshot['worker_pgid'], root, known)
                break
            wait([future], timeout=POLL_SECONDS)
        result = future.result()
    return early_reason or result


def signal_identity(identity: dict | None, sig: int) -> bool:
    """No group signal: psutil also rechecks reuse before sending to its bound process."""
    if not identity or identity.get('create_time') is None:
        return False
    try:
        proc = psutil.Process(identity['pid'])
        if proc.create_time() != identity['create_time'] or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        proc.send_signal(sig)
        return True
    except psutil.NoSuchProcess:
        return False


def terminate_workers(pgid: int, child_identity: dict, known: dict, *, grace_seconds=5.0) -> None:
    ownership_error = None
    try:
        worker_members(pgid, child_identity, known)
    except RuntimeError as exc:
        ownership_error = exc
    bound = dict(known)
    bound[(child_identity['pid'], child_identity['create_time'])] = child_identity
    # An uncertain numeric group cannot confer ownership. Still stop every process
    # whose original identity is known, including workers outside the old group.
    for identity in bound.values():
        signal_identity(identity, signal.SIGTERM)
    def live_bound():
        live = []
        for identity in bound.values():
            status = identity_status(identity)
            if status == 'unknown':
                raise RuntimeError('CP-26 bound worker ending is uncertain; compute is stopped')
            if status == 'same':
                live.append(identity)
        return live
    deadline = time.monotonic() + grace_seconds
    while live_bound() and time.monotonic() < deadline:
        time.sleep(0.05)
    for identity in live_bound():
        signal_identity(identity, signal.SIGKILL)
    deadline = time.monotonic() + 5
    while live_bound() and time.monotonic() < deadline:
        time.sleep(0.05)
    if live_bound():
        raise RuntimeError('CP-26 escaped worker did not end; compute is stopped')
    if ownership_error is not None:
        raise ownership_error
    if worker_members(pgid, child_identity, known):
        raise RuntimeError('CP-26 worker ending is not evidenced; compute is stopped')


def load_request(request_path: Path) -> tuple[dict, str, Budget]:
    request_path = local_path(request_path)
    raw = request_path.read_bytes()
    request = json.loads(raw)
    request_sha = hashlib.sha256(raw).hexdigest()
    budget = Budget(request['ledger'])
    job = _job(_resource_state(budget), request['job_index'], request['job_id'])
    if (request_sha != job['request_sha256'] or not job['running']
            or request.get('supervision_policy') != SUPERVISION_POLICY):
        raise RuntimeError('CP-26 supervision request does not bind the running job')
    return request, request_sha, budget


def worker_gate(request_path: Path) -> int:
    """The new session cannot exec compute until both accounting loops are ready."""
    request, _, budget = load_request(request_path)
    own = process_identity()
    deadline = time.monotonic() + STARTUP_GRACE_SECONDS
    while time.monotonic() < deadline:
        job = _job(_resource_state(budget), request['job_index'], request['job_id'])
        if not job['running'] or job.get('phase') not in ('starting', 'supervising'):
            raise RuntimeError('CP-26 worker admission closed before computation')
        sampled = all(job.get(f'{role}_sample_child_identity') == own
                      for role in ('supervisor', 'watchdog'))
        if job.get('phase') == 'supervising' and job.get('child_identity') == own and sampled:
            require_fresh_supervision(job)
            os.execvpe(request['command'][0], request['command'], os.environ.copy())
        time.sleep(0.025)
    raise RuntimeError('CP-26 worker startup grace expired before accounting was established')


def discover_unrecorded_root(job: dict, watchdog_identity: dict) -> dict | None:
    """Cover death between gate spawn and ledger registration, using bound ancestry."""
    supervisor_identity = job.get('supervisor_identity')
    if identity_status(supervisor_identity) != 'same':
        return None
    roots = []
    try:
        supervisor = psutil.Process(supervisor_identity['pid'])
        if supervisor.create_time() != supervisor_identity['create_time']:
            return None
        for proc in supervisor.children():
            identity = {'pid': proc.pid, 'create_time': proc.create_time()}
            if identity == watchdog_identity or proc.status() == psutil.STATUS_ZOMBIE:
                continue
            if os.getpgid(proc.pid) == proc.pid and os.getsid(proc.pid) == proc.pid:
                roots.append(identity)
        if identity_status(supervisor_identity) != 'same':
            return None
    except (ProcessLookupError, psutil.NoSuchProcess):
        return None
    if len(roots) > 1:
        raise RuntimeError('CP-26 startup has multiple unrecorded session roots')
    return roots[0] if roots else None


def watchdog_run(request_path: Path) -> int:
    """Detached fail-safe; samples caps independently and never needs launcher life."""
    request, request_sha, budget = load_request(request_path)
    index, job_id = request['job_index'], request['job_id']
    own = process_identity()
    snapshot = _job(_resource_state(budget), index, job_id)
    def bind_watchdog(job):
        if job.get('watchdog_identity') not in (None, own):
            raise RuntimeError('CP-26 watchdog identity changed during launch')
        job.update(watchdog_identity=own, watchdog_pid=own['pid'])
    _update_live_job(budget, index, job_id, bind_watchdog)
    interrupted = []
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    known, peak = {}, 0
    reason = None
    try:
        while True:
            snapshot = _job(_resource_state(budget), index, job_id)
            if not snapshot['running']:
                return 0
            for identity in snapshot.get('worker_identities', []):
                known[(identity['pid'], identity['create_time'])] = identity
            root = snapshot.get('child_identity')
            members = worker_members(snapshot['worker_pgid'], root, known) if root else []
            rss = identities_rss([*members, own, snapshot.get('supervisor_identity'),
                                  request['monitor_identity']])
            peak = max(peak, rss, snapshot.get('peak_rss_bytes', 0))
            wall = time.monotonic() - snapshot['accounting_began_monotonic']
            reason = observer_tick(budget, snapshot, known, wall=wall, rss=rss, sample_role='watchdog',
                          worker_identities=list(known.values()), peak_rss_bytes=peak)
            # Process inspection and the ledger commit may overlap a newer peer
            # sample. Judge the persisted peer proof, rather than the old read.
            snapshot = _job(_resource_state(budget), index, job_id)
            if not snapshot['running']:
                return 0
            # A stopped live supervisor and a hard-dead supervisor both lose their
            # proof of supervision. Startup grace applies only before worker exec.
            if interrupted:
                reason = f'watchdog_signal_{interrupted[0]}'
            elif identity_status(snapshot.get('supervisor_identity')) != 'same':
                reason = 'supervisor_identity_lost'
            elif not startup_is_fresh(snapshot):
                reason = reason or freshness_reason(snapshot, 'supervisor')
            if reason:
                break
            time.sleep(POLL_SECONDS)
    except BaseException as exc:
        # A stopped supervisor can hold flock. Timeout must reach the independent
        # termination path, without waiting for permission from that same lock.
        reason = redact(f'watchdog_{type(exc).__name__}: {exc}')

    return failsafe_finish(request, request_sha, budget, snapshot, known, peak, reason,
                           issuer='watchdog', own=own)


def _check_completion_observation(job: dict, observed: dict) -> None:
    """External ending observations cannot be applied to a replaced binding."""
    fields = ('job_id', 'job_index', 'request_sha256', 'started_epoch', 'workers',
              'accounting_began_monotonic', 'supervisor_identity', 'watchdog_identity',
              'monitor_identity', 'child_identity', 'worker_pgid')
    if any(job.get(field) != observed.get(field) for field in fields):
        raise RuntimeError('CP-26 completion binding changed after ending observation')


def _commit_completion(state: dict, job: dict, receipt: dict, clock: Path) -> None:
    """Keep the first bound clock immutable before its exactly-once charge.

    The short clock write stays serialized with competing completion issuers.
    Process traversal is already complete outside this accounting lock. A crash
    after the clock rename still leaves a receipt that reconciliation can apply.
    """
    if clock.exists():
        previous = json.loads(clock.read_text())
        check_completion(job, previous)
        apply_completion(state, job, previous)
        return
    if receipt['workers_stopped'] and receipt['group_empty']:
        check_completion(job, receipt)
    atomic(clock, receipt)
    if receipt['workers_stopped'] and receipt['group_empty']:
        apply_completion(state, job, receipt)


def failsafe_finish(request: dict, request_sha: str, budget: Budget, snapshot: dict,
                    known: dict, peak: int, reason: str, *, issuer: str, own: dict) -> int:
    """Stop only recorded identities before taking over one immutable completion."""
    index, job_id = request['job_index'], request['job_id']
    latest = _job(_resource_state(budget), index, job_id)
    if not latest['running']:
        return 0
    if (any(latest.get(f'{role}_identity') != snapshot.get(f'{role}_identity')
            for role in ('supervisor', 'watchdog')) or latest.get(f'{issuer}_identity') != own):
        raise RuntimeError('CP-26 failsafe identities changed; refuse to signal')
    snapshot = latest
    for identity in snapshot.get('worker_identities', []):
        known[(identity['pid'], identity['create_time'])] = identity
    supervisor_identity = snapshot.get('supervisor_identity')
    root = snapshot.get('child_identity')
    pgid = snapshot.get('worker_pgid')
    # Freeze the old writer before stopping workers; a verified SIGKILL then
    # releases any lock it retained. Numeric groups are never signal targets.
    signal_identity(supervisor_identity, signal.SIGSTOP)
    if issuer == 'monitor':
        signal_identity(snapshot.get('watchdog_identity'), signal.SIGSTOP)
    stopped = False
    ending_error = None
    try:
        if root is None:
            root = discover_unrecorded_root(snapshot, snapshot.get('watchdog_identity'))
            pgid = root['pid'] if root else None
        if root is not None:
            known[(root['pid'], root['create_time'])] = root
            terminate_workers(pgid, root, known, grace_seconds=POLL_SECONDS)
            evidence = ending_evidence(pgid, root, known, previously_stopped=True)
            stopped, ending_error = evidence['workers_stopped'], evidence['ending_error']
        else:
            # If the writer died between spawn intent and identity capture, lack
            # of a PID is uncertainty, not evidence that no gate was spawned.
            stopped = not known and not snapshot.get('worker_spawn_intent')
            if not stopped:
                ending_error = 'worker spawn ending is uncertain; compute is stopped'
    except Exception as exc:
        ending_error = redact(str(exc))
    finally:
        ended_epoch = time.time()
        ended_monotonic = time.monotonic()
        signal_identity(supervisor_identity, signal.SIGKILL)
        if issuer == 'monitor':
            signal_identity(snapshot.get('watchdog_identity'), signal.SIGKILL)
    # Sending a signal is not evidence that the old accounting writer has ended.
    # Confirm the identity outside flock before publishing a takeover receipt.
    # Otherwise a fast resource commit can become visible while SIGKILL is still
    # pending and the prior writer retains its original process identity.
    retired = [supervisor_identity]
    if issuer == 'monitor':
        retired.append(snapshot.get('watchdog_identity'))
    deadline = time.monotonic() + HEARTBEAT_TIMEOUT_SECONDS
    while True:
        observed = [identity_status(identity) for identity in retired if identity is not None]
        if 'unknown' in observed:
            raise RuntimeError('CP-26 previous accounting writer ending is uncertain; compute is stopped')
        if 'same' not in observed:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('CP-26 previous accounting writer did not end; compute is stopped')
        time.sleep(min(POLL_SECONDS, 0.025))
    began = snapshot.get('accounting_began_monotonic')
    if not _finite(began) or ended_monotonic < began:
        raise RuntimeError('watchdog has no bound elapsed origin; worker duration remains uncertain')
    wall = max(ended_monotonic - began, snapshot['charged_wall_seconds'])
    receipt = {'schema': 'cp26-completion-v2', 'job_id': job_id, 'job_index': index,
               'started_epoch': request['started_epoch'], 'ended_epoch': ended_epoch,
               'request_sha256': request_sha, 'workers': request['workers'],
               'supervisor_identity': supervisor_identity, 'watchdog_identity': snapshot.get('watchdog_identity'),
               'completion_issuer': issuer, 'completion_issuer_identity': own,
               'child_identity': root, 'worker_identities': list(known.values()),
               'worker_pgid': pgid, 'workers_stopped': stopped, 'group_empty': stopped,
               'ending_error': ending_error, 'wall_seconds': wall, 'exit_code': 70,
               'abort_reason': reason, 'peak_rss_bytes': peak,
               'ledger_lock_telemetry': budget.telemetry()}
    def finish(state):
        job = _job(state, index, job_id)
        if not job['running']:
            return 0
        if (job.get('child_identity') not in (None, root)
                or job.get(f'{issuer}_identity') != own):
            raise RuntimeError('watchdog completion identities changed; compute is stopped')
        _check_completion_observation(job, snapshot)
        job.update(phase='closing', child_identity=root, child_pid=root['pid'] if root else None)
        _commit_completion(state, job, receipt, local_path(Path(request['completion_clock'])))
        return 70
    return budget.resource_update(finish, timeout_seconds=STARTUP_GRACE_SECONDS)


def monitor_failsafe(request_path: Path) -> int:
    """Third local check when neither independent accounting loop is evidenced."""
    request, request_sha, budget = load_request(request_path)
    snapshot = _job(_resource_state(budget), request['job_index'], request['job_id'])
    own = process_identity()
    failed = []
    for role in ('supervisor', 'watchdog'):
        failed.append(f'{role}_identity_lost' if identity_status(snapshot.get(f'{role}_identity')) != 'same'
                      else freshness_reason(snapshot, role))
    if not all(failed):
        raise RuntimeError('one independent accounting loop remains fresh; await its ending receipt')
    return failsafe_finish(request, request_sha, budget, snapshot, {}, snapshot.get('peak_rss_bytes', 0),
                           'monitor_failsafe:' + ','.join(failed), issuer='monitor', own=own)


def run(request_path: Path) -> int:
    began = _SUPERVISOR_BEGAN
    request_path = local_path(request_path)
    request, request_sha, budget = load_request(request_path)
    index, job_id = request['job_index'], request['job_id']
    clock = local_path(Path(request['completion_clock']))
    progress = local_path(Path(request['progress_clock']))
    log_path = local_path(Path(request['log']))
    own_identity = process_identity()
    _update_live_job(budget, index, job_id,
                     lambda job: _bind_supervisor_start(job, request_sha, own_identity, began))
    interrupted = []
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, frame: interrupted.append(signum))
    child = None
    child_identity = None
    pgid = None
    known = {}
    peak = 0
    code = 70
    reason = None
    stopped = False
    caffeinate = None
    watchdog = None
    watchdog_identity = None
    try:
        watchdog = subprocess.Popen([sys.executable, '-B', '-m', 'cp26.supervisor',
                                     '--watchdog', str(request_path)],
                                    cwd=request['cwd'], stdin=subprocess.DEVNULL,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    start_new_session=True)
        watchdog_identity = process_identity(watchdog.pid)
        if watchdog_identity is None:
            raise RuntimeError('new CP-26 watchdog identity could not be captured')
        def bind_watchdog(job):
            if job.get('watchdog_identity') not in (None, watchdog_identity):
                raise RuntimeError('CP-26 watchdog identity changed during launch')
            job.update(watchdog_identity=watchdog_identity, watchdog_pid=watchdog.pid)
        _update_live_job(budget, index, job_id, bind_watchdog)
        deadline = time.monotonic() + STARTUP_GRACE_SECONDS
        while True:
            wall = time.monotonic() - began
            rss = identities_rss([own_identity, watchdog_identity, request['monitor_identity']])
            cap_reason = tick(budget, index, job_id, wall, rss, sample_role='supervisor')
            job = _job(_resource_state(budget), index, job_id)
            if cap_reason or interrupted:
                raise RuntimeError(cap_reason or f'signal_{interrupted[0]}')
            if freshness_reason(job, 'watchdog') is None and identity_status(watchdog_identity) == 'same':
                break
            if watchdog.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError('CP-26 watchdog startup grace expired; refuse to compute')
            # Startup accounting uses the declared cadence too. Faster writes
            # can starve a peer's first sample on a full historical ledger.
            time.sleep(POLL_SECONDS)
        if sys.platform == 'darwin':
            caffeinate = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(os.getpid())],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('a') as log:
            log.write(f'# CP-26 supervised job {request["name"]}; workers={request["workers"]}\n')
            log.flush()
            _update_live_job(budget, index, job_id,
                             lambda job: job.update(worker_spawn_intent=True))
            child = subprocess.Popen([sys.executable, '-B', '-m', 'cp26.supervisor',
                                      '--worker', str(request_path)],
                                     cwd=request['cwd'], stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            child_identity = process_identity(child.pid)
            if child_identity is None:
                raise RuntimeError('new CP-26 worker identity could not be captured')
            pgid = child.pid
            _update_live_job(budget, index, job_id,
                             lambda job: job.update(child_identity=child_identity, child_pid=child.pid,
                                                    worker_pgid=pgid, phase='supervising'))
            known[(child_identity['pid'], child_identity['create_time'])] = child_identity
            while True:
                for identity in _resource_state(budget)['jobs'][index].get('worker_identities', []):
                    known[(identity['pid'], identity['create_time'])] = identity
                members = worker_members(pgid, child_identity, known)
                for identity in members:
                    known[(identity['pid'], identity['create_time'])] = identity
                rss = identities_rss([*members, own_identity, watchdog_identity, request['monitor_identity']])
                peak = max(peak, rss)
                wall = time.monotonic() - began
                cap_reason = observer_tick(budget, _job(_resource_state(budget), index, job_id), known,
                                  wall=wall, rss=rss, sample_role='supervisor',
                                  worker_identities=list(known.values()), peak_rss_bytes=peak)
                sampled = _job(_resource_state(budget), index, job_id)
                atomic(progress, {'schema': 'cp26-progress-v3', 'job_id': job_id, 'job_index': index,
                                  'request_sha256': request_sha,
                                  'supervisor_identity': own_identity, 'child_identity': child_identity,
                                  'worker_identities': list(known.values()), 'worker_pgid': pgid,
                                  'wall_seconds': wall, 'observed_epoch': time.time(),
                                  'observed_monotonic': began + wall, 'rss_bytes': rss,
                                  'sample_sequence': sampled['supervisor_sample_sequence']})
                if interrupted:
                    reason = f'signal_{interrupted[0]}'
                elif cap_reason:
                    reason = cap_reason
                elif identity_status(watchdog_identity) != 'same':
                    reason = 'watchdog_identity_lost'
                elif freshness_reason(sampled, 'watchdog'):
                    reason = freshness_reason(sampled, 'watchdog')
                result = child.poll()
                if result is not None:
                    # The root can acknowledge registration and exit between two
                    # samples. Reload that durable handshake before ending evidence.
                    latest = _resource_state(budget)['jobs'][index]
                    for identity in latest.get('worker_identities', []):
                        known[(identity['pid'], identity['create_time'])] = identity
                    members = worker_members(pgid, child_identity, known)
                    if latest.get('worker_ceiling_error'):
                        reason = 'registered_worker_ceiling'
                if result is not None and members and reason is None:
                    # Reaping the root never means its background descendants stopped.
                    remaining = [m for m in members if m != child_identity]
                    if remaining:
                        reason = 'orphaned_descendants'
                if reason:
                    terminate_workers(pgid, child_identity, known)
                    code = child.wait()
                    break
                if result is not None:
                    code = result
                    break
                time.sleep(POLL_SECONDS)
            stopped = not worker_members(pgid, child_identity, known)
            if not stopped:
                raise RuntimeError('CP-26 worker ending is not evidenced')
    except BaseException as exc:
        reason = redact(f'{type(exc).__name__}: {exc}')
        code = 70
        if pgid is not None:
            try:
                for identity in _resource_state(budget)['jobs'][index].get('worker_identities', []):
                    known[(identity['pid'], identity['create_time'])] = identity
                terminate_workers(pgid, child_identity, known)
                if child is not None:
                    child.wait(timeout=5)
                stopped = not worker_members(pgid, child_identity, known)
            except Exception:
                stopped = False
        elif child is None:
            # Spawn never happened; there are no compute workers to orphan.
            stopped = True
    finally:
        ended_epoch = time.time()
        wall = time.monotonic() - began
        ending_snapshot = _job(_resource_state(budget), index, job_id)
        for identity in ending_snapshot.get('worker_identities', []):
            known[(identity['pid'], identity['create_time'])] = identity
        # Process inspection must not block every other accounting observer.
        # Revalidate the exact job/process binding under the lock before commit.
        evidence = (ending_evidence(pgid, child_identity, known, stopped)
                    if ending_snapshot['running'] else {})
        receipt = {'schema': 'cp26-completion-v2', 'job_id': job_id, 'job_index': index,
                   'started_epoch': request['started_epoch'], 'ended_epoch': ended_epoch,
                   'request_sha256': request_sha, 'workers': request['workers'],
                   'supervisor_identity': own_identity, 'child_identity': child_identity,
                   'watchdog_identity': watchdog_identity,
                   'completion_issuer': 'supervisor', 'completion_issuer_identity': own_identity,
                   'worker_identities': list(known.values()), 'worker_pgid': pgid,
                   'workers_stopped': stopped, 'group_empty': stopped,
                   'wall_seconds': wall, 'exit_code': code, 'abort_reason': reason,
                   'peak_rss_bytes': peak, 'ledger_lock_telemetry': budget.telemetry(),
                   **evidence}
        def finish(state):
            nonlocal stopped
            job = _job(state, index, job_id)
            if job['running']:
                _check_completion_observation(job, ending_snapshot)
                if (job.get('child_identity') not in (None, child_identity)
                        or job.get('supervisor_identity') != own_identity):
                    raise RuntimeError('supervisor completion process identities changed; compute is stopped')
                job['phase'] = 'closing'
                stopped = evidence['workers_stopped']
                receipt.update(worker_identities=list(known.values()), **evidence,
                               worker_ceiling_error=job.get('worker_ceiling_error'),
                               wall_seconds=max(wall, job['charged_wall_seconds']),
                               peak_rss_bytes=max(peak, job.get('peak_rss_bytes', 0)))
                job['child_identity'] = child_identity
                _commit_completion(state, job, receipt, clock)
        budget.resource_update(finish, timeout_seconds=STARTUP_GRACE_SECONDS)
        if watchdog is not None:
            try:
                watchdog.wait(timeout=HEARTBEAT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                signal_identity(watchdog_identity, signal.SIGKILL)
                watchdog.wait(timeout=5)
        if caffeinate is not None:
            caffeinate.terminate()
            caffeinate.wait()
    return code if not reason else 70


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--watchdog', action='store_true')
    modes.add_argument('--worker', action='store_true')
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    if args.watchdog:
        return watchdog_run(args.request)
    if args.worker:
        return worker_gate(args.request)
    return run(args.request)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(redact(f'{type(exc).__name__}: {exc}'), file=sys.stderr)
        raise SystemExit(70)
