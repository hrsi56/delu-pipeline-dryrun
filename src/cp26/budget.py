"""Persistent CP-26 accounting, including reported time and bounded reserves.

The r16 cumulative ceiling is 500 hours; admission is an absolute, run-scoped limit.
Machine seconds are worker wall seconds, including failed work. An independent
supervisor records elapsed duration, so a monitor restart never measures an idle gap.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import MappingProxyType

from cp20.budget import atomic, pid_alive  # noqa: F401 (unchanged pure helpers)
from cp25.guard import redact
from .harness import art
from .limits import LEDGER_TIMEOUT_SECONDS, shutdown_exposure_seconds

HOUR = 3600
GB = 10**9
CAPS = {
    'machine_seconds': 500 * HOUR, 'workers': 4, 'rss_bytes': 12 * GB,
    'data_download_bytes': 0, 'remote_writes': 0, 'cost_usd': 0,
    'gpu_jobs': 0, 'cloud_jobs': 0,
}
GENERAL_ADMISSION_HOURS = 436.5
COLD_RUN_ID = 'cp26-cold-rerun-2026-10-15-01'
COLD_RUN_ADMISSION_HOURS = 440.0


def admission_hours(run_id=None):
    return COLD_RUN_ADMISSION_HOURS if run_id == COLD_RUN_ID else GENERAL_ADMISSION_HOURS
TIMEBOX_ACTIVE_HOURS = 60
REPAIR_SECONDS = 10 * HOUR
HISTORICAL_SECONDS = 547.5647564170067
EARLY_REPORTED_SECONDS = 6.125  # Already inside HISTORICAL_SECONDS.
AUDIT_REPORTED_SECONDS = 90.127
AUDIT_RESERVE_SECONDS = 2826.0
PARITY_END_EPOCH = 1791905827.8090382
AUDIT_RECEIPT_EPOCH = 1791908653.442842
REPAIR_EVENT = 'repair-v21-r14-historical-reconciliation'


class CapExceeded(RuntimeError):
    pass


class AccountingAbort(BaseException):
    """An operational accounting stop, never a cacheable numerical fit failure."""
    def __init__(self, cause):
        self.cause = cause
        super().__init__(redact(f'CP-26 accounting aborted: {type(cause).__name__}: {cause}'))


def ledger_path() -> Path:
    return Path(os.environ.get('CP26_LEDGER', art() / 'ledger' / 'budget.json'))


def accounting(state: dict) -> dict:
    """Read legacy totals conservatively; only explicit reconciliation reclassifies them."""
    return state.get('accounting', {
        'measured_seconds': state['counts'].get('machine_seconds', 0.0),
        'reported_seconds': 0.0, 'reserved_seconds': 0.0,
    })


def total_seconds(state: dict) -> float:
    return sum(accounting(state).get(f'{kind}_seconds', 0.0)
               for kind in ('measured', 'reported', 'reserved'))


def _finite_nonnegative(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError('resource duration must be a finite nonnegative number')
    return float(value)


def add_seconds(state: dict, seconds: float, kind: str = 'measured') -> None:
    seconds = _finite_nonnegative(seconds)
    if kind not in ('measured', 'reported', 'reserved'):
        raise ValueError('resource classification must be measured, reported or reserved')
    state.setdefault('accounting', accounting(state).copy())
    state['accounting'][f'{kind}_seconds'] += seconds
    state['counts']['machine_seconds'] = total_seconds(state)


def unresolved_reserves(state: dict) -> list:
    return [r for r in state.get('reserves', []) if r.get('status') != 'authorized']


def limit_reason(state: dict) -> str | None:
    allocated = sum(j['workers'] for j in state.get('jobs', []) if j.get('running'))
    if total_seconds(state) + shutdown_exposure_seconds(allocated) >= CAPS['machine_seconds']:
        return 'machine_hours_cap'
    scope = state.get('repair_scope')
    if scope and scope.get('active'):
        if total_seconds(state) - scope['baseline_seconds'] >= scope['limit_seconds']:
            return 'repair_machine_hours_cap'
    return None


def require_compute_ready(state: dict) -> None:
    scope = state.get('repair_scope')
    if scope and scope.get('active') is False and scope.get('terminal_snapshot') is None:
        raise RuntimeError('closed repair scope has no hash-bound terminal snapshot; new compute is stopped')
    if unresolved_reserves(state):
        raise RuntimeError('unreconciled CP-26 reserve: new compute is stopped')
    if limit_reason(state):
        raise CapExceeded(f'CP-26 {limit_reason(state)} exhausted')


def resource_summary(state: dict) -> dict:
    total = total_seconds(state)
    result = {**accounting(state), 'total_seconds': total,
              'machine_hours': total / HOUR,
              'headroom_hours': (CAPS['machine_seconds'] - total) / HOUR,
              'pilot_stop_hours': admission_hours(),
              'run_admission_hours': {COLD_RUN_ID: admission_hours(COLD_RUN_ID)},
              'unreconciled_reserves': unresolved_reserves(state)}
    result['live_shutdown_exposure_seconds'] = shutdown_exposure_seconds(
        sum(j['workers'] for j in state.get('jobs', []) if j.get('running')))
    scope = state.get('repair_scope')
    if scope:
        if scope['active']:
            used = total - scope['baseline_seconds']
        else:
            terminal = scope.get('terminal_snapshot')
            if terminal is None:
                raise RuntimeError('closed repair scope has no hash-bound terminal snapshot; '
                                   'freeze its original evidence before continuation')
            used = terminal['repair_seconds_used']
        result.update(repair_seconds_used=used,
                      repair_seconds_remaining=scope['limit_seconds'] - used,
                      repair_scope_active=scope['active'])
    continuation = state.get('continuation_scope')
    if continuation:
        result.update(continuation_seconds_used=total - continuation['baseline_seconds'],
                      post_repair_seconds=total - scope['terminal_snapshot']['total_seconds'],
                      continuation_authority_sha256=continuation['authority_sha256'])
    if state.get('scientific_runs'):
        result['scientific_runs'] = {run_id: scientific_run_summary(state, run_id)
                                     for run_id in state['scientific_runs']}
    return result


def scientific_run_summary(state: dict, run_id: str) -> dict:
    """A view of attributed existing receipts, never another cumulative charge."""
    registration = state.get('scientific_runs', {}).get(run_id)
    if registration is None:
        raise ValueError('unregistered CP-26 scientific run')
    components = {f'{kind}_seconds': 0.0 for kind in ('measured', 'reported', 'reserved')}
    for job in state.get('jobs', []):
        if job.get('run_id') == run_id:
            components['measured_seconds'] += job['charged_wall_seconds'] * job['workers']
    for charge in state.get('charges', []):
        if charge.get('source', {}).get('run_id') == run_id:
            components[f"{charge['kind']}_seconds"] += charge['seconds']
    total = sum(components.values())
    return {'run_id': run_id, **components, 'total_seconds': total,
            'machine_hours': total / HOUR,
            'historical_baseline_hours': registration['cumulative_baseline']['machine_hours'],
            'cumulative_machine_hours': total_seconds(state) / HOUR}


def projected_cost(state: dict, searches: list[dict], remaining: dict[str, dict], *, run_id=None) -> dict:
    """A future pilot must price all 42 searches separately and include prior reserves.

    Each estimate carries its basis and whether it is measured or conservative.
    This helper prices a proposed continuation; it does not authorize/run one.
    """
    require_compute_ready(state)
    if len(searches) != 42:
        raise ValueError('cost projection requires exactly 42 separately costed searches')
    by_run = {}
    costs = []
    for row in searches:
        run, k = row.get('run'), row.get('k')
        if not isinstance(run, str) or not run or isinstance(k, bool) or not isinstance(k, int) or k not in range(14):
            raise ValueError('search cost identity requires a run and k=0..13')
        if k in by_run.setdefault(run, set()):
            raise ValueError('duplicate search cost identity')
        by_run[run].add(k)
        if row.get('classification') not in ('measured', 'conservative') or not row.get('basis'):
            raise ValueError('search estimate requires a measured/conservative classification and basis')
        value = _finite_nonnegative(row.get('machine_seconds'))
        if value == 0:
            raise ValueError('a future search cannot be priced at zero')
        completed = row.get('completed_evidence')
        if completed is not None:
            if (not isinstance(completed, dict) or not completed.get('result_sha256')
                    or not completed.get('path') or not completed.get('task_id')):
                raise ValueError('completed search estimate needs terminal evidence identity')
            task = next((t for t in state.get('execution_tasks', []) if t['id'] == completed['task_id']), None)
            if task is None or task.get('evidence', {}).get('result_sha256') != completed['result_sha256']:
                raise ValueError('completed search estimate has no matching cumulative-ledger task')
        costs.append(0.0 if completed is not None else value)
    if set(by_run) != {'nominal', 'noload', 'gfs6'} or any(v != set(range(14)) for v in by_run.values()):
        raise ValueError('cost projection requires 14 searches in each of three runs')
    required = {'daily_fits', 'parity', 'tests', 'bootstrap', 'diagnostics', 'failed_jobs_and_resume'}
    if not required.issubset(remaining):
        raise ValueError('cost projection is missing daily fits, validation, diagnostics or failure/resume costs')
    remainder = 0.0
    for row in remaining.values():
        if row.get('classification') not in ('measured', 'conservative') or not row.get('basis'):
            raise ValueError('remaining cost estimate requires a classification and basis')
        remainder += _finite_nonnegative(row.get('machine_seconds'))
    searches_total = sum(costs)
    projected = total_seconds(state) + searches_total + remainder
    stop = admission_hours(run_id) * HOUR
    return {**resource_summary(state), 'searches_priced': 42,
            'remaining_search_seconds': searches_total, 'remaining_other_seconds': remainder,
            'projected_total_seconds': projected, 'projected_total_hours': projected / HOUR,
            'within_pilot_stop': projected <= stop, 'pilot_stop_seconds': stop,
            'pilot_stop_hours': stop / HOUR}


def _ledger_json(state: dict) -> str:
    # No indent or sort_keys: the C encoder handles this growing hot ledger.
    # This changes representation only; all logical fields remain present.
    return json.dumps(state, separators=(',', ':'), allow_nan=False, default=str) + '\n'


_COMPACT_LEDGER_FORMAT = 'cp26-compact-json-v1'


def _format_path(path: Path) -> Path:
    return path.with_suffix('.format.json')


def _recognized_compact(path: Path, raw: bytes) -> bool:
    """Use exact writer provenance, never guess from whitespace or JSON shape.

    This small optional cache has no accounting authority. A missing, stale or
    corrupt cache makes us canonicalize the input for logical no-op comparison.
    """
    try:
        with _format_path(path).open('rb') as handle:
            cached = handle.read(257)
        if len(cached) > 256:
            return False
        identity = json.loads(cached)
        return (isinstance(identity, dict)
                and identity.get('format') == _COMPACT_LEDGER_FORMAT
                and identity.get('sha256') == hashlib.sha256(raw).hexdigest())
    except (OSError, ValueError, UnicodeError):
        return False


def _write_ledger(path: Path, serialized: str) -> None:
    raw = serialized.encode('utf-8')
    _replace_ledger(path, raw, hashlib.sha256(raw).hexdigest())


def _replace_ledger(path: Path, raw: bytes, sha256: str, *, canonical=True) -> None:
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_bytes(raw)
    os.replace(temp, path)
    try:
        atomic(_format_path(path), {'format': (_COMPACT_LEDGER_FORMAT if canonical
                                             else 'cp26-preserved-json-spans-v1'),
                                   'sha256': sha256})
    except OSError:
        # The ledger is already committed. Cache failure only disables the fast
        # comparison path; it cannot turn a committed charge into a failed write.
        pass


def _validate_ledger(state: dict, frozen: dict | None) -> None:
    if state['caps'] != CAPS:
        raise ValueError('CP-26 resource contract changed; refuse to continue')
    if frozen is not None and state.get('repair_scope') != frozen:
        raise ValueError('closed repair scope is immutable; continuation cannot change it')
    if not math.isclose(state['counts'].get('machine_seconds', 0.0), total_seconds(state),
                        rel_tol=0, abs_tol=1e-8):
        raise ValueError('CP-26 classified accounting disagrees with cumulative total')


_RESOURCE_MUTABLE = frozenset(('counts', 'accounting', 'peaks', 'jobs'))
_RESOURCE_POLICY = frozenset(('schema', 'caps', 'repair_scope', 'reserves', 'charges', 'continuation_scope',
                              'scientific_runs'))
_RESOURCE_FIELDS = _RESOURCE_MUTABLE | _RESOURCE_POLICY
_JSON_WHITESPACE = ' \t\r\n'


def _reject_json_constant(value):
    raise ValueError(f'CP-26 ledger contains nonfinite JSON constant {value}')


_RESOURCE_DECODER = json.JSONDecoder(parse_constant=_reject_json_constant)


def _immutable_job(*args, **kwargs):
    raise ValueError('CP-26 closed job history is immutable in resource view')


class _FrozenJobList(tuple):
    """Immutable tuple storage with JSON-list equality, including nested maps."""
    __slots__ = ()

    def __new__(cls, values):
        return tuple.__new__(cls, (_freeze_job(value) for value in values))

    def __eq__(self, other):
        if not isinstance(other, (list, tuple, _FrozenJobList)):
            return NotImplemented
        return len(self) == len(other) and all(left == right for left, right in zip(self, other))

    def __ne__(self, other):
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __deepcopy__(self, memo):
        return self

    __setitem__ = __delitem__ = append = extend = insert = remove = reverse = sort = clear = pop = _immutable_job
    __iadd__ = __imul__ = _immutable_job


def _freeze_job(value):
    if isinstance(value, dict):
        # No mutable dictionary is retained or exposed. Unlike a dict subclass,
        # even dict.__setitem__(view, ...) cannot bypass this immutability.
        return MappingProxyType({key: _freeze_job(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenJobList(value)
    return value


def _job_binding(job):
    return tuple(job.get(key) for key in ('job_id', 'job_index', 'request_sha256', 'started_epoch', 'workers'))


def _jobs_layout(text, jobs):
    """Index complete authoritative array items once, outside the ledger lock."""
    if not isinstance(jobs, list):
        raise ValueError('CP-26 jobs must be a JSON array')
    offset = _skip_json_space(text, 0)
    if offset >= len(text) or text[offset] != '[':
        raise ValueError('CP-26 jobs must be a JSON array')
    offset = _skip_json_space(text, offset + 1)
    previous_end = 0
    entries = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict) or not isinstance(job.get('running'), bool):
            raise ValueError('CP-26 job must have an explicit boolean running state')
        value_start = offset
        parsed, value_end = _RESOURCE_DECODER.raw_decode(text, value_start)
        if parsed != job:
            raise ValueError('CP-26 job span differs from authoritative decoded job')
        frozen = _freeze_job(job) if job['running'] is False else None
        entries.append((text[previous_end:value_start], text[value_start:value_end], frozen,
                        _job_binding(job)))
        previous_end = value_end
        offset = _skip_json_space(text, value_end)
        if index < len(jobs) - 1:
            if offset >= len(text) or text[offset] != ',':
                raise ValueError('CP-26 jobs array has invalid item delimiter')
            offset = _skip_json_space(text, offset + 1)
    if offset >= len(text) or text[offset] != ']' or _skip_json_space(text, offset + 1) != len(text):
        raise ValueError('CP-26 jobs array has invalid closing delimiter')
    return tuple(entries), text[previous_end:]


def _jobs_match(text, offset, layout):
    jobs = []
    entries, suffix = layout
    for prefix, cold, frozen, binding in entries:
        if not text.startswith(prefix, offset):
            return None
        offset += len(prefix)
        if frozen is not None:
            if not text.startswith(cold, offset):
                return None
            jobs.append(frozen)
            offset += len(cold)
        else:
            job, offset = _RESOURCE_DECODER.raw_decode(text, offset)
            if (not isinstance(job, dict) or job.get('running') is not True
                    or _job_binding(job) != binding):
                return None
            jobs.append(job)
    if not text.startswith(suffix, offset):
        return None
    return jobs, offset + len(suffix)


def _resource_job_guard(state, layout):
    jobs = state.get('jobs')
    job_spans = layout[3]
    if not isinstance(jobs, list) or job_spans is None or len(jobs) < len(job_spans[0]):
        raise ValueError('CP-26 resource mutation cannot change job inventory or closed history')
    for job, (_, _, frozen, binding) in zip(jobs, job_spans[0]):
        if frozen is not None:
            if job is not frozen:
                raise ValueError('CP-26 resource mutation cannot change closed job history')
        elif not isinstance(job, dict) or _job_binding(job) != binding or not isinstance(job.get('running'), bool):
            raise ValueError('CP-26 resource mutation cannot change active job binding or inventory')
    ids = ({job.get('job_id') for job in jobs[:len(job_spans[0])]}
           if len(jobs) > len(job_spans[0]) else set())
    for index in range(len(job_spans[0]), len(jobs)):
        job = jobs[index]
        if (not isinstance(job, dict) or job.get('running') is not True
                or job.get('job_index') != index or not isinstance(job.get('job_id'), str)
                or not job['job_id'] or job['job_id'] in ids):
            raise ValueError('CP-26 resource mutation can append only uniquely bound live jobs')
        ids.add(job['job_id'])


def _resource_signature(state, layout):
    _resource_job_guard(state, layout)
    # Never ask the JSON encoder to walk verified immutable closed-job history.
    active = [(index, state['jobs'][index]) for index, entry in enumerate(layout[3][0]) if entry[2] is None]
    active.extend((index, state['jobs'][index]) for index in range(len(layout[3][0]), len(state['jobs'])))
    return _ledger_json({**{key: value for key, value in state.items() if key != 'jobs'},
                         'active_jobs': active})


def _jobs_splice(jobs, layout):
    entries, suffix = layout
    pieces = []
    for job, (prefix, cold, frozen, _) in zip(jobs, entries):
        pieces.append(prefix)
        pieces.append(cold if frozen is not None else json.dumps(
            job, separators=(',', ':'), allow_nan=False, default=str))
    if len(jobs) > len(entries):
        close = suffix.index(']')
        pieces.append(suffix[:close])
        for index in range(len(entries), len(jobs)):
            pieces.append((',' if index else '') + json.dumps(
                jobs[index], separators=(',', ':'), allow_nan=False, default=str))
        pieces.append(suffix[close:])
    else:
        pieces.append(suffix)
    return ''.join(pieces)


def _skip_json_space(text, offset):
    while offset < len(text) and text[offset] in _JSON_WHITESPACE:
        offset += 1
    return offset


def _resource_layout(raw):
    """Parse authoritative JSON once, retaining grammar-bound exact value spans.

    No disk index or format sidecar chooses these spans. The prefixes and suffix
    come from a complete root-object parse; repeated hot reads prove each cold
    value and every intervening delimiter against those exact bytes. A changed
    root order, key, cold value, or suffix requires another complete parse.
    """
    text = raw.decode('utf-8')
    offset = _skip_json_space(text, 0)
    if offset >= len(text) or text[offset] != '{':
        raise ValueError('CP-26 ledger must be a JSON root object')
    offset += 1
    previous_end = 0
    entries = []
    state = {}
    offset = _skip_json_space(text, offset)
    if offset < len(text) and text[offset] == '}':
        offset += 1
    else:
        while True:
            if offset >= len(text) or text[offset] != '"':
                raise ValueError('CP-26 ledger has an invalid root key')
            key, offset = _RESOURCE_DECODER.raw_decode(text, offset)
            if key in state:
                raise ValueError('CP-26 ledger has a duplicate root key')
            offset = _skip_json_space(text, offset)
            if offset >= len(text) or text[offset] != ':':
                raise ValueError('CP-26 ledger has an invalid root delimiter')
            value_start = _skip_json_space(text, offset + 1)
            value, value_end = _RESOURCE_DECODER.raw_decode(text, value_start)
            entries.append((key, text[previous_end:value_start], text[value_start:value_end]))
            state[key] = value
            previous_end = value_end
            offset = _skip_json_space(text, value_end)
            if offset >= len(text):
                raise ValueError('CP-26 ledger has an unterminated root object')
            delimiter = text[offset]
            offset = _skip_json_space(text, offset + 1)
            if delimiter == '}':
                break
            if delimiter != ',':
                raise ValueError('CP-26 ledger has an invalid root delimiter')
    if _skip_json_space(text, offset) != len(text):
        raise ValueError('CP-26 ledger has trailing JSON content')
    # Besides canonical provenance, this rejects overflow-to-inf numbers anywhere
    # in cold metadata, as generic ledger writes do with allow_nan=False.
    canonical = text == _ledger_json(state)
    job_spans = None
    if 'jobs' in state:
        job_text = next(value for key, _, value in entries if key == 'jobs')
        job_spans = _jobs_layout(job_text, state['jobs'])
        state['jobs'] = [entry[2] if entry[2] is not None else job
                         for job, entry in zip(state['jobs'], job_spans[0])]
    array_lengths = {}
    for field in ('execution_tasks', 'events'):
        if field not in state:
            continue
        if not isinstance(state[field], list):
            raise ValueError('CP-26 task/event history must be a JSON array')
        array_lengths[field] = len(state[field])
    return state, (tuple(entries), text[previous_end:], canonical, job_spans,
                   (None, MappingProxyType(array_lengths)))


def _execution_index(records):
    task_index = {}
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get('id'), str)
                or not record['id'] or record['id'] in task_index):
            raise ValueError('CP-26 execution task inventory has duplicate or missing identity')
        task_index[record['id']] = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return MappingProxyType(task_index)


def _execution_evidence_sha256(evidence):
    # Same canonical JSON identity used by execution.state.digest, without
    # importing scientific state machinery into the accounting hot path.
    raw = json.dumps(evidence, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _prepare_execution_append(layout, records):
    """Prepare cold array bytes/index outside flock; later exact proof binds them."""
    proposed, proposed_json, event_json, proposed_identity = records
    items = {'execution_tasks': proposed_json, 'events': event_json}
    return _prepare_cold_append(layout, items, (proposed['id'], proposed_identity))


def _prepare_cold_append(layout, items, index_update=None):
    lengths = dict(layout[4][1])
    appends = {}
    entries = []
    for key, prefix, cold in layout[0]:
        if key in items:
            value = cold[:-1] + (',' if lengths[key] else '') + items[key] + ']'
            appends[key] = value
            lengths[key] += 1
            entries.append((key, prefix, value))
        else:
            entries.append((key, prefix, cold))
    suffix = layout[1]
    for key in sorted(items.keys() - appends.keys()):
        value = '[' + items[key] + ']'
        close = suffix.index('}')
        prefix = suffix[:close] + ',' + json.dumps(key) + ':'
        suffix = suffix[close:]
        entries.append((key, prefix, value))
        appends[key] = value
        lengths[key] = 1
    index = None if layout[4][0] is None else dict(layout[4][0])
    if index_update is not None:
        index[index_update[0]] = index_update[1]
    advanced = (tuple(entries), suffix, layout[2], layout[3],
                (None if index is None else MappingProxyType(index), MappingProxyType(lengths)))
    return appends, advanced


def _resource_match(raw, layout):
    """Return fresh resource fields only when all cached cold spans still match."""
    text = raw.decode('utf-8')
    offset = 0
    state = {}
    entries, suffix = layout[:2]
    for key, prefix, cold in entries:
        if not text.startswith(prefix, offset):
            return None
        offset += len(prefix)
        if key == 'jobs':
            matched = _jobs_match(text, offset, layout[3])
            if matched is None:
                return None
            state[key], offset = matched
        elif key in _RESOURCE_MUTABLE:
            value, offset = _RESOURCE_DECODER.raw_decode(text, offset)
            state[key] = value
        else:
            if not text.startswith(cold, offset):
                return None
            if key in _RESOURCE_POLICY:
                # Policy is included in the view, but is immutable for this
                # cached layout. A foreign/policy writer forces full validation;
                # the next hot splice must never restore an earlier policy value.
                state[key], _ = _RESOURCE_DECODER.raw_decode(text, offset)
            offset += len(cold)
    if text[offset:] != suffix:
        return None
    return state


def _resource_splice(layout, state, appends=None):
    entries, suffix = layout[:2]
    appends = appends or {}
    pieces = []
    for key, prefix, cold in entries:
        pieces.append(prefix)
        if key in appends:
            # Already prepared outside flock; exact old cold bytes were just
            # re-proven under flock before this fresh hot-resource merge.
            pieces.append(appends[key])
        elif key == 'jobs':
            pieces.append(_jobs_splice(state[key], layout[3]))
        else:
            pieces.append(json.dumps(state[key], separators=(',', ':'), allow_nan=False,
                                     default=str) if key in _RESOURCE_MUTABLE else cold)
    present = {entry[0] for entry in entries}
    missing = sorted(((_RESOURCE_MUTABLE & state.keys()) | appends.keys()) - present)
    if missing:
        close = suffix.index('}')
        pieces.append(suffix[:close])
        for key in missing:
            value = (appends[key] if key in appends else json.dumps(
                state[key], separators=(',', ':'), allow_nan=False, default=str))
            pieces.append(',' + json.dumps(key) + ':' + value)
        pieces.append(suffix[close:])
    else:
        pieces.append(suffix)
    return ''.join(pieces).encode('utf-8')


class Budget:
    def transition_r16_cap(self, root: Path, backup: Path) -> dict:
        """One stopped-ledger cap transition under the existing exclusive lock.

        No charge, receipt, reserve or closed scope is changed. The exact old
        bytes are retained before the atomic replacement; repeated calls are no-ops.
        """
        from . import harness as H
        authority = H.continuation_authority(root)
        if (authority.get('actual_ceiling_hours') != 500
                or authority.get('run_id') != COLD_RUN_ID
                or authority.get('admission_hours') != 440):
            raise RuntimeError('r16 cap transition lacks the named Owner authority')
        backup = Path(backup).resolve()
        if not backup.is_relative_to(H.art().resolve()):
            raise ValueError('cap backup must remain inside primary CP-26 artifacts')
        deadline = time.monotonic() + LEDGER_TIMEOUT_SECONDS
        with self.path.with_suffix('.lock').open('a') as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('r16 cap transition lock deadline expired')
                    time.sleep(.025)
            raw = self.path.read_bytes()
            state = json.loads(raw)
            if any(j.get('running') for j in state['jobs']):
                raise RuntimeError('cap transition requires all owned jobs stopped')
            previous = state.get('cap_transitions', [])
            if previous and previous[-1].get('authority_sha256') == H.PINS[H.AUTHORITY_PATH]:
                _validate_ledger(state, None)
                if hashlib.sha256(backup.read_bytes()).hexdigest() != previous[-1]['before_sha256']:
                    raise RuntimeError('r16 cap backup changed')
                return previous[-1]
            old_caps = {**CAPS, 'machine_seconds': 485 * HOUR}
            if state['caps'] != old_caps or not math.isclose(
                    state['counts']['machine_seconds'], total_seconds(state), abs_tol=1e-8, rel_tol=0):
                raise RuntimeError('r16 cap transition requires the intact 485-hour account')
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                if backup.read_bytes() != raw:
                    raise RuntimeError('cap backup already exists with different bytes')
            else:
                with backup.open('xb') as handle:
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
            receipt = {'event': 'owner-r16-cap-transition', 'epoch': time.time(),
                       'old_caps': old_caps, 'new_caps': CAPS.copy(),
                       'before_sha256': hashlib.sha256(raw).hexdigest(), 'backup': str(backup),
                       'authority_sha256': H.PINS[H.AUTHORITY_PATH],
                       'run_id': COLD_RUN_ID, 'charges_unchanged': True}
            state['caps'] = CAPS.copy()
            state.setdefault('cap_transitions', []).append(receipt)
            _validate_ledger(state, None)
            if time.monotonic() >= deadline:
                raise TimeoutError('r16 cap transition elapsed deadline expired')
            _write_ledger(self.path, _ledger_json(state))
            self._resource_spans = None
            return receipt

    def __init__(self, path=None):
        self.path = Path(path or ledger_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.last_transaction = {'wait_seconds': 0.0, 'hold_seconds': 0.0,
                                 'total_seconds': 0.0, 'acquired': False,
                                 'written': False, 'read_bytes': 0,
                                 'written_bytes': 0, 'canonical_input': False,
                                 'outcome': 'not_started', 'attempts': 0,
                                 'conflicts': 0, 'body_seconds': 0.0,
                                 'serialized_fallback': False}
        self._transactions = self._timeouts = 0
        self._peak_wait_seconds = self._peak_hold_seconds = 0.0
        self._resource_spans = None
        self._resource_closed = None
        self._task_index_needed = False

    def _validate_resources(self, state):
        closed = state.get('repair_scope', {})
        if self._resource_closed is None and not closed.get('active', True) and closed.get('terminal_snapshot'):
            self._resource_closed = json.loads(_ledger_json(closed))
        _validate_ledger(state, self._resource_closed)

    def _warm_resources(self, raw):
        # Complete cold parsing and canonical validation happen outside flock.
        # Cache contents are derived from these authoritative bytes only.
        full, spans = _resource_layout(raw)
        self._validate_resources(full)
        if self._task_index_needed:
            spans = (*spans[:4], (_execution_index(full.get('execution_tasks', [])), spans[4][1]))
        self._resource_spans = spans
        return {key: value for key, value in full.items() if key in _RESOURCE_FIELDS}

    def resource_read(self) -> dict:
        """Read validated accounting/policy/jobs without decoding unchanged history.

        The full budget.json remains authoritative and generic read() still returns
        every field. This restricted view includes caps, reserves and closed scopes;
        it cannot be used for execution-task or evidence-history reconciliation.
        Its outer jobs list is ordinary; closed rows contain deeply immutable
        mappings/sequences, and active rows are fresh plain dictionaries. For a
        JSON export or mutable deepcopy of the full ledger, use generic read().
        """
        raw = self.path.read_bytes()
        state = (_resource_match(raw, self._resource_spans)
                 if self._resource_spans is not None else None)
        if state is None:
            return self._warm_resources(raw)
        # This small encoding rejects nonfinite values in fresh resource metadata.
        _resource_signature(state, self._resource_spans)
        self._validate_resources(state)
        return state

    def resource_update(self, callback, *, timeout_seconds: float = LEDGER_TIMEOUT_SECONDS, guard=None):
        """Merge one pure resource sample under a short, bounded exclusive lock.

        No whole-ledger CAS conflicts occur between independent samplers. Every
        mutation starts from freshly decoded hot fields under flock. Exact cold
        spans preserve all history; changed cold bytes are reparsed outside flock
        before retrying within the same absolute acquisition deadline. A valid
        refresh does not exhaust a separate retry-count allowance.
        The callback cannot inspect or mutate cold fields or resource policy.
        """
        return self._resource_update(callback, timeout_seconds=timeout_seconds, guard=guard)

    def append_execution_task(self, proposed, event, *, timeout_seconds: float = LEDGER_TIMEOUT_SECONDS):
        """Append one hash-bound evidence task/event and its counters exactly once.

        Duration remains exclusively supervisor-accounted. Existing records are
        indexed only from complete authoritative parsing, then re-proven by exact
        cold bytes under flock; concurrent append conflicts refresh outside flock.
        """
        started = time.monotonic()
        proposed_json = json.dumps(proposed, separators=(',', ':'), allow_nan=False)
        event_json = json.dumps(event, separators=(',', ':'), allow_nan=False)
        proposed, event = json.loads(proposed_json), json.loads(event_json)
        proposed_identity = json.dumps(proposed, sort_keys=True, separators=(',', ':'), allow_nan=False)
        if (not isinstance(proposed, dict) or set(proposed) != {'id', 'evidence', 'counts'}
                or not isinstance(proposed['id'], str) or not proposed['id']
                or not isinstance(proposed['counts'], dict)
                or any(not isinstance(key, str) or isinstance(value, bool)
                       or not isinstance(value, int) or value < 0
                       for key, value in proposed['counts'].items())):
            raise ValueError('completed-evidence task requires identity, evidence and nonnegative integer counts')
        if 'machine_seconds' in proposed['counts']:
            raise ValueError('execution task evidence cannot charge machine duration')
        if (not isinstance(event, dict) or event.get('event') != 'cp26_execution_task_terminal'
                or event.get('task_id') != proposed['id']
                or event.get('evidence_sha256') != _execution_evidence_sha256(proposed['evidence'])):
            raise ValueError('execution task event does not bind its evidence')
        return self._resource_update(None, timeout_seconds=timeout_seconds,
                                     append_records=(proposed, proposed_json, event_json, proposed_identity),
                                     started=started)

    def _resource_update(self, callback, *, timeout_seconds: float = LEDGER_TIMEOUT_SECONDS, guard=None,
                         append_records=None, append_event=None, started=None):
        timeout_seconds = _finite_nonnegative(timeout_seconds)
        started = time.monotonic() if started is None else started
        deadline = started + timeout_seconds
        wait = hold = body_seconds = 0.0
        acquired = False
        read_bytes = written_bytes = attempts = conflicts = 0
        canonical_input = False
        outcome = 'error'
        refresh_raw = None
        try:
            if append_records is not None:
                self._task_index_needed = True
            if self._resource_spans is None:
                raw = self.path.read_bytes()
                read_bytes = len(raw)
                self._warm_resources(raw)
            elif append_records is not None and self._resource_spans[4][0] is None:
                # Only evidence writers build this index. Samplers need exact
                # cold bytes, not a second decoded copy of all task history.
                cached = next((value for key, _, value in self._resource_spans[0]
                               if key == 'execution_tasks'), '[]')
                index = _execution_index(_RESOURCE_DECODER.decode(cached))
                self._resource_spans = (*self._resource_spans[:4],
                                        (index, self._resource_spans[4][1]))
            while True:
                if refresh_raw is not None:
                    if time.monotonic() >= deadline:
                        outcome = 'timeout'
                        raise TimeoutError('CP-26 ledger accounting lock did not progress: '
                                           'absolute deadline expired before cold-history refresh')
                    self._warm_resources(refresh_raw)
                    refresh_raw = None
                if time.monotonic() >= deadline:
                    outcome = 'timeout'
                    raise TimeoutError('CP-26 ledger accounting lock did not progress')
                attempts += 1
                prepared_appends = prepared_layout = None
                if append_records is not None:
                    prepared_appends, prepared_layout = _prepare_execution_append(self._resource_spans, append_records)
                elif append_event is not None:
                    prepared_appends, prepared_layout = _prepare_cold_append(self._resource_spans, {'events': append_event})
                began_wait = time.monotonic()
                acquired_at = None
                try:
                    with self.path.with_suffix('.lock').open('a') as lock:
                        while True:
                            if time.monotonic() >= deadline:
                                outcome = 'timeout'
                                raise TimeoutError('CP-26 ledger accounting lock did not progress')
                            try:
                                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                break
                            except BlockingIOError:
                                time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
                        acquired_at = time.monotonic()
                        acquired = True
                        raw = self.path.read_bytes()
                        read_bytes = len(raw)
                        state = _resource_match(raw, self._resource_spans)
                        if state is None:
                            conflicts += 1
                            refresh_raw = raw
                            continue
                        canonical_input = self._resource_spans[2]
                        before = _resource_signature(state, self._resource_spans)
                        self._validate_resources(state)
                        policy = _ledger_json({key: value for key, value in state.items()
                                               if key in _RESOURCE_POLICY})
                        original_fields = set(state)
                        token = guard(state) if guard is not None else None
                        began_body = time.monotonic()
                        appends = None
                        try:
                            if append_event is not None:
                                appends, result = prepared_appends, None
                            elif append_records is None:
                                result = callback(state)
                            else:
                                proposed, proposed_json, event_json, proposed_identity = append_records
                                previous = self._resource_spans[4][0].get(proposed['id'])
                                if previous is not None:
                                    if previous != proposed_identity:
                                        raise ValueError('execution task identity was reused with different evidence')
                                    result = False
                                else:
                                    for key, value in proposed['counts'].items():
                                        state['counts'][key] = state['counts'].get(key, 0) + value
                                    appends = prepared_appends
                                    result = True
                        finally:
                            body_seconds += time.monotonic() - began_body
                        if (set(state) - _RESOURCE_FIELDS
                                or original_fields - set(state)
                                or _ledger_json({key: value for key, value in state.items()
                                                 if key in _RESOURCE_POLICY}) != policy):
                            raise ValueError('CP-26 resource mutation cannot change cold fields or policy')
                        self._validate_resources(state)
                        after = _resource_signature(state, self._resource_spans)
                        # Receipt writers share flock; this also covers an external
                        # clock appearing during the callback in fault fixtures.
                        if guard is not None and guard(state) != token:
                            raise RuntimeError('CP-26 terminal receipt changed during serialized mutation')
                        if after != before or appends:
                            candidate = (_resource_splice(self._resource_spans, state, appends)
                                         if appends else _resource_splice(self._resource_spans, state))
                            candidate_sha = hashlib.sha256(candidate).hexdigest()
                            if guard is not None and guard(state) != token:
                                raise RuntimeError('CP-26 terminal receipt changed during serialized mutation')
                            if time.monotonic() >= deadline:
                                outcome = 'timeout'
                                raise TimeoutError('CP-26 ledger accounting lock did not progress')
                            _replace_ledger(self.path, candidate, candidate_sha,
                                            canonical=canonical_input)
                            written_bytes = len(candidate)
                            if appends:
                                # Only our successful authoritative rename lets
                                # us advance this process-local representation.
                                # Every later operation still proves cold bytes.
                                self._resource_spans = prepared_layout
                            outcome = 'committed'
                        else:
                            if time.monotonic() >= deadline:
                                outcome = 'timeout'
                                raise TimeoutError('CP-26 ledger accounting lock did not progress')
                            outcome = 'unchanged'
                        return result
                finally:
                    finished = time.monotonic()
                    wait += (finished if acquired_at is None else acquired_at) - began_wait
                    if acquired_at is not None:
                        hold = max(hold, finished - acquired_at)
        finally:
            self._record_transaction(started=started, wait=wait, hold=hold, acquired=acquired,
                                     read_bytes=read_bytes, written_bytes=written_bytes,
                                     canonical_input=canonical_input, outcome=outcome,
                                     attempts=attempts, conflicts=conflicts, body_seconds=body_seconds)

    def telemetry(self) -> dict:
        """A bounded process-local summary, including failed acquisitions.

        Supervisors persist one latest summary per observer, rather than adding
        an event on every sample. The previous completed transaction's hold time
        is known only after releasing its lock.
        """
        return {**self.last_transaction, 'transactions': self._transactions,
                'timeouts': self._timeouts, 'peak_wait_seconds': self._peak_wait_seconds,
                'peak_hold_seconds': self._peak_hold_seconds}

    def _record_transaction(self, *, started, wait, hold, acquired, read_bytes,
                            written_bytes, canonical_input, outcome, attempts,
                            conflicts, body_seconds, serialized_fallback=False):
        self.last_transaction = {'wait_seconds': wait, 'hold_seconds': hold,
                                 'total_seconds': time.monotonic() - started,
                                 'acquired': acquired, 'written': written_bytes > 0,
                                 'read_bytes': read_bytes, 'written_bytes': written_bytes,
                                 'canonical_input': canonical_input, 'outcome': outcome,
                                 'attempts': attempts, 'conflicts': conflicts,
                                 'body_seconds': body_seconds,
                                 'serialized_fallback': serialized_fallback}
        self._transactions += 1
        self._timeouts += int(outcome == 'timeout')
        self._peak_wait_seconds = max(wait, self._peak_wait_seconds)
        self._peak_hold_seconds = max(hold, self._peak_hold_seconds)

    def update(self, callback, *, timeout_seconds: float = LEDGER_TIMEOUT_SECONDS, guard=None):
        """Prepare outside flock, with one bounded serialized conflict fallback.

        The callback may run more than once and must have no external side effects.
        Each attempt gets a fresh complete state. A read-only guard can bind an
        external immutable receipt; its token is checked again under the lock.
        Exact original bytes, never the optional format cache, decide the CAS.
        After one conflict, the fresh mutation is prepared while holding flock.
        This prevents repeatedly serializing stale history until starvation.
        The deadline covers both preparations and acquisition together.
        """
        timeout_seconds = _finite_nonnegative(timeout_seconds)
        started = time.monotonic()
        deadline = started + timeout_seconds
        wait = hold = body_seconds = 0.0
        acquired = False
        read_bytes = written_bytes = attempts = conflicts = 0
        canonical_input = False
        outcome = 'error'
        original_closed = None
        serialized_fallback = False

        def prepare(original):
            nonlocal read_bytes, canonical_input, original_closed, body_seconds
            read_bytes = len(original)
            # Capture optional writer provenance before the relatively slow parse.
            canonical_input = _recognized_compact(self.path, original)
            state = json.loads(original)
            closed = state.get('repair_scope', {})
            frozen = (json.loads(_ledger_json(closed))
                      if not closed.get('active', True) and closed.get('terminal_snapshot') else None)
            if original_closed is None:
                original_closed = frozen
            elif state.get('repair_scope') != original_closed:
                raise ValueError('closed repair scope is immutable; continuation cannot change it')
            frozen = original_closed
            _validate_ledger(state, frozen)
            before = original.decode('utf-8') if canonical_input else _ledger_json(state)
            token = guard(state) if guard is not None else None
            began_body = time.monotonic()
            try:
                result = callback(state)
            finally:
                body_seconds += time.monotonic() - began_body
            _validate_ledger(state, frozen)
            after = _ledger_json(state)
            candidate = after.encode('utf-8')
            return state, before, after, candidate, hashlib.sha256(candidate).hexdigest(), token, result

        try:
            while True:
                if time.monotonic() >= deadline:
                    outcome = 'timeout'
                    raise TimeoutError('CP-26 ledger accounting lock did not progress')
                attempts += 1
                if not serialized_fallback:
                    original = self.path.read_bytes()
                    state, before, after, candidate, candidate_sha, token, result = prepare(original)
                began_wait = time.monotonic()
                acquired_at = None
                try:
                    with self.path.with_suffix('.lock').open('a') as lock:
                        while True:
                            if time.monotonic() >= deadline:
                                outcome = 'timeout'
                                raise TimeoutError('CP-26 ledger accounting lock did not progress')
                            try:
                                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                break
                            except BlockingIOError:
                                time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
                        acquired_at = time.monotonic()
                        acquired = True
                        if serialized_fallback:
                            # Exactly one fresh preparation, protected from all
                            # cooperating ledger and completion-clock writers.
                            original = self.path.read_bytes()
                            state, before, after, candidate, candidate_sha, token, result = prepare(original)
                            if guard is not None and guard(state) != token:
                                raise RuntimeError('CP-26 terminal receipt changed during serialized mutation')
                        elif self.path.read_bytes() != original or (guard is not None and guard(state) != token):
                            conflicts += 1
                            serialized_fallback = True
                            continue
                        if time.monotonic() >= deadline:
                            outcome = 'timeout'
                            raise TimeoutError('CP-26 ledger accounting lock did not progress')
                        if after != before:
                            _replace_ledger(self.path, candidate, candidate_sha)
                            written_bytes = len(candidate)
                            outcome = 'committed'
                        else:
                            outcome = 'unchanged'
                        return result
                finally:
                    finished = time.monotonic()
                    wait += (finished if acquired_at is None else acquired_at) - began_wait
                    if acquired_at is not None:
                        hold = max(hold, finished - acquired_at)
        finally:
            self._record_transaction(started=started, wait=wait, hold=hold, acquired=acquired,
                                     read_bytes=read_bytes, written_bytes=written_bytes,
                                     canonical_input=canonical_input, outcome=outcome,
                                     attempts=attempts, conflicts=conflicts, body_seconds=body_seconds,
                                     serialized_fallback=serialized_fallback)

    @contextlib.contextmanager
    def transaction(self, *, timeout_seconds: float | None = None):
        started = time.monotonic()
        acquired_at = None
        read_bytes = written_bytes = 0
        canonical_input = False
        body_seconds = 0.0
        outcome = 'error'
        try:
            with self.path.with_suffix('.lock').open('a') as lock:
                if timeout_seconds is None:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                else:
                    timeout_seconds = _finite_nonnegative(timeout_seconds)
                    deadline = time.monotonic() + timeout_seconds
                    while True:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            if time.monotonic() >= deadline:
                                outcome = 'timeout'
                                raise TimeoutError('CP-26 ledger accounting lock did not progress')
                            time.sleep(min(0.025, max(0.0, deadline - time.monotonic())))
                acquired_at = time.monotonic()
                if not self.path.exists():
                    raise FileNotFoundError(f'CP-26 ledger not initialised: {self.path}')
                raw = self.path.read_bytes()
                read_bytes = len(raw)
                state = json.loads(raw)
                if state['caps'] != CAPS:
                    raise ValueError('CP-26 resource contract changed; refuse to continue')
                canonical_input = _recognized_compact(self.path, raw)
                before = raw.decode('utf-8') if canonical_input else _ledger_json(state)
                closed = state.get('repair_scope', {})
                frozen = json.loads(_ledger_json(closed)) if not closed.get('active', True) and closed.get('terminal_snapshot') else None
                began_body = time.monotonic()
                try:
                    yield state
                finally:
                    body_seconds = time.monotonic() - began_body
                _validate_ledger(state, frozen)
                after = _ledger_json(state)
                if after != before:
                    _write_ledger(self.path, after)
                    written_bytes = len(after)  # The encoder's default ensure_ascii is true.
                    outcome = 'committed'
                else:
                    outcome = 'unchanged'
        finally:
            finished = time.monotonic()
            wait = (finished if acquired_at is None else acquired_at) - started
            hold = 0.0 if acquired_at is None else finished - acquired_at
            self._record_transaction(started=started, wait=wait, hold=hold,
                                     acquired=acquired_at is not None, read_bytes=read_bytes,
                                     written_bytes=written_bytes, canonical_input=canonical_input,
                                     outcome=outcome, attempts=int(acquired_at is not None),
                                     conflicts=0, body_seconds=body_seconds)

    def initialise(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                raise FileExistsError('ledger exists; cumulative accounting is never reset')
            _write_ledger(self.path, _ledger_json({
                'schema': 'cp26-ledger-v2', 'caps': CAPS, 'counts': {'machine_seconds': 0.0},
                'accounting': {'measured_seconds': 0.0, 'reported_seconds': 0.0, 'reserved_seconds': 0.0},
                'peaks': {}, 'jobs': [], 'events': [], 'charges': [], 'reserves': [],
                'created_epoch': time.time(),
                'rule': 'total = measured + reported + authorized reserves; machine time is the sum '
                        'of supervisor-observed worker wall seconds x declared workers',
            }))

    def read(self) -> dict:
        return json.loads(self.path.read_text())

    def freeze_closed_repair(self, resources_path: Path, ledger_path: Path, *,
                             authority_sha256: str) -> dict:
        """Bind closed usage to the original handover, preserving every later charge.

        The two historical evidence files are read-only. Their hashes identify
        the closure; the live cumulative total is never reduced to that snapshot.
        Repeated invocation is an identity check, not another charge or allocation.
        """
        if len(authority_sha256) != 64 or any(c not in '0123456789abcdef' for c in authority_sha256):
            raise ValueError('continuation requires a SHA-256 authority identity')
        resources_raw, ledger_raw = Path(resources_path).read_bytes(), Path(ledger_path).read_bytes()
        resources, historical = json.loads(resources_raw), json.loads(ledger_raw)
        old_scope = historical.get('repair_scope', {})
        if old_scope.get('active') is not False or any(j['running'] for j in historical['jobs']):
            raise ValueError('original repair evidence is not a closed terminal snapshot')
        terminal_total = total_seconds(historical)
        used = terminal_total - old_scope['baseline_seconds']
        if (resources.get('repair_scope_active') is not False
                or not math.isclose(resources['total_seconds'], terminal_total, abs_tol=1e-8, rel_tol=0)
                or not math.isclose(resources['repair_seconds_used'], used, abs_tol=1e-8, rel_tol=0)):
            raise ValueError('original repair resource and ledger evidence disagree')
        snapshot = {'repair_seconds_used': used, 'total_seconds': terminal_total,
                    'accounting': accounting(historical),
                    'resources_sha256': hashlib.sha256(resources_raw).hexdigest(),
                    'ledger_sha256': hashlib.sha256(ledger_raw).hexdigest(),
                    'terminal_candidate_sha': old_scope['terminal_candidate_sha']}
        with self.transaction() as state:
            scope = state.get('repair_scope', {})
            if scope.get('active') is not False:
                raise ValueError('continuation cannot reopen or close an active repair')
            for key in ('id', 'baseline_seconds', 'limit_seconds', 'terminal_candidate_sha'):
                if scope.get(key) != old_scope.get(key):
                    raise ValueError('live closed repair differs from its original evidence')
            if total_seconds(state) < terminal_total:
                raise ValueError('live cumulative charges precede the terminal repair evidence')
            if scope.get('terminal_snapshot') not in (None, snapshot):
                raise ValueError('closed repair snapshot identity changed')
            continuation = state.get('continuation_scope')
            if continuation is not None:
                if continuation['authority_sha256'] != authority_sha256:
                    raise ValueError('continuation authority identity changed')
                return resource_summary(state)
            if any(j['running'] for j in state['jobs']):
                raise RuntimeError('freeze continuation accounting before starting workers')
            scope['terminal_snapshot'] = snapshot
            state['continuation_scope'] = {'id': 'continuation-v21-r14',
                'authority_sha256': authority_sha256, 'baseline_seconds': total_seconds(state),
                'opened_epoch': time.time(), 'ceiling_seconds': CAPS['machine_seconds']}
            state['events'].append({'event': 'continuation-v21-r14-accounting-opened',
                'epoch': time.time(), 'authority_sha256': authority_sha256,
                'repair_terminal_snapshot': snapshot, 'cumulative_seconds': total_seconds(state)})
            return resource_summary(state)

    def count(self, **increments):
        """Uncapped informational counters; machine duration uses classified charges."""
        if 'machine_seconds' in increments:
            raise ValueError('machine_seconds requires a classified resource charge')
        # Argument misuse is distinct from corrupt persisted counter state.
        # Validate arguments before opening the ledger; failures after that point
        # are operational and must bypass the search's scientific-failure cache.
        for value in increments.values():
            if not isinstance(value, (int, float)):
                raise TypeError('counter increments must be finite numbers')
            if not math.isfinite(value):
                raise ValueError('counter increments must be finite numbers')
        def increment(state):
            for key, value in increments.items():
                previous = state['counts'].get(key, 0)
                if not isinstance(previous, (int, float)) or not math.isfinite(previous):
                    raise ValueError('persisted counter is not a finite number')
                state['counts'][key] = previous + value
        try:
            self.resource_update(increment)
        except Exception as exc:
            # A search evaluator catches Exception to cache scientific failures.
            # Retain the underlying exception for inspection while suppressing
            # raw cause chaining in an unhandled child-process traceback.
            raise AccountingAbort(exc) from None

    def event(self, name, **fields):
        started = time.monotonic()
        event = json.dumps({'event': name, 'epoch': time.time(), **fields},
                           separators=(',', ':'), allow_nan=False, default=str)
        self._resource_update(None, append_event=event, started=started)

    def record_charge(self, charge_id: str, seconds: float, kind: str, source: dict,
                      reserve_status: str = 'unreconciled') -> bool:
        """Exactly-once external duration/reserve import, with immutable provenance."""
        seconds = _finite_nonnegative(seconds)
        proposed = {'id': charge_id, 'seconds': seconds, 'kind': kind, 'source': source}
        if kind == 'reserved':
            proposed['status'] = reserve_status
        with self.transaction() as state:
            for charge in state.get('charges', []):
                if charge['id'] == charge_id:
                    if {k: charge[k] for k in proposed} != proposed:
                        raise ValueError('resource charge id was reused with different evidence')
                    return False
            add_seconds(state, seconds, kind)
            state.setdefault('charges', []).append({**proposed, 'epoch': time.time()})
            if kind == 'reserved':
                state.setdefault('reserves', []).append(proposed.copy())
        return True

    def reconcile_repair_history(self, bound: dict, *, evidence_sha256: str,
                                 preservation_seconds: float = 0.055875) -> dict:
        """Owner-authorized classification/additions; never infer exact failed runtime.

        The sole audit turn encloses the admitted failed test and lies inside parity
        completion .. attachment receipt. The 2826-second reserve remains a reserve;
        the 15.01-second lower bound is not charged separately.
        """
        preservation_seconds = _finite_nonnegative(preservation_seconds)
        if bound.get('schema') != 'cp26-audit-time-bound-v1':
            raise ValueError('missing audit enclosing-time evidence')
        began, ended = bound.get('turn_started_epoch'), bound.get('turn_completed_epoch')
        if not (isinstance(began, (float, int)) and isinstance(ended, (float, int))
                and PARITY_END_EPOCH <= began <= ended <= AUDIT_RECEIPT_EPOCH):
            raise RuntimeError('audit reserve window does not bound the failed action; compute is stopped')
        if not evidence_sha256 or len(evidence_sha256) != 64:
            raise ValueError('audit time-bound evidence requires its SHA-256')
        window = AUDIT_RECEIPT_EPOCH - PARITY_END_EPOCH
        if window > AUDIT_RESERVE_SECONDS:
            raise RuntimeError('authorized audit reserve is smaller than its verified enclosing window')
        with self.transaction() as state:
            previous = next((e for e in state['events'] if e['event'] == REPAIR_EVENT), None)
            if previous:
                if previous['evidence_sha256'] != evidence_sha256 or previous['preservation_seconds'] != preservation_seconds:
                    raise ValueError('historical reconciliation evidence changed')
                return resource_summary(state)
            if not math.isclose(total_seconds(state), HISTORICAL_SECONDS, rel_tol=0, abs_tol=1e-8):
                raise RuntimeError('historical ledger changed before reconciliation; inspect, never reset')
            if any(j['running'] for j in state['jobs']):
                raise RuntimeError('historical reconciliation requires all prior jobs finished')
            measured = sum(j['wall_seconds'] * j['workers'] for j in state['jobs'])
            if not math.isclose(measured + EARLY_REPORTED_SECONDS, HISTORICAL_SECONDS, rel_tol=0, abs_tol=1e-8):
                raise RuntimeError('historical job receipts do not reconcile to the ledger')
            parity = next((j for j in state['jobs'] if j['name'] == 'parity'), None)
            if not parity or parity.get('ended_epoch') != PARITY_END_EPOCH:
                raise RuntimeError('parity endpoint does not match the authorized reserve evidence')
            early = next((e for e in state['events'] if e['event'] == 'builder_lightweight_tests_before_monitor'), None)
            if not early or early.get('observed_wall_seconds') != [4.13, 1.995] or early.get('workers') != 1:
                raise RuntimeError('already charged early tests have no matching source event')
            state['schema'] = 'cp26-ledger-v2'
            state['rule'] = ('total = measured + reported + authorized reserves; measured machine time is '
                             'the sum of supervisor-observed worker wall seconds x declared workers')
            state['accounting'] = {'measured_seconds': measured, 'reported_seconds': EARLY_REPORTED_SECONDS,
                                   'reserved_seconds': 0.0}
            source = {'evidence_sha256': evidence_sha256, 'thread_id': bound.get('thread_id'),
                      'parity_ended_epoch': PARITY_END_EPOCH, 'audit_received_epoch': AUDIT_RECEIPT_EPOCH,
                      'audit_turn_started_epoch': began, 'audit_turn_completed_epoch': ended}
            imports = [
                {'id': 'pre-monitor-tests-already-in-legacy', 'seconds': EARLY_REPORTED_SECONDS,
                 'kind': 'reported', 'already_in_legacy_total': True,
                 'source': {'event': 'builder_lightweight_tests_before_monitor', 'durations': [4.13, 1.995]}},
                {'id': 'independent-audit-successful-tests', 'seconds': AUDIT_REPORTED_SECONDS,
                 'kind': 'reported', 'source': {**source, 'durations': [29.958, 17.646, 21.423, 21.10]}},
                {'id': 'independent-audit-failed-test', 'seconds': AUDIT_RESERVE_SECONDS,
                 'kind': 'reserved', 'status': 'authorized',
                 'source': {**source, 'failed_test_lower_bound_seconds': 15.01,
                            'enclosing_window_seconds': window, 'exact_runtime_recovered': False}},
            ]
            state.setdefault('charges', []).extend(imports)
            state.setdefault('reserves', []).append(imports[-1].copy())
            add_seconds(state, AUDIT_REPORTED_SECONDS, 'reported')
            add_seconds(state, AUDIT_RESERVE_SECONDS, 'reserved')
            state['repair_scope'] = {'id': 'repair-v21-r14', 'active': True,
                                     'baseline_seconds': total_seconds(state),
                                     'limit_seconds': REPAIR_SECONDS, 'opened_epoch': time.time()}
            add_seconds(state, preservation_seconds, 'measured')
            state['charges'].append({'id': 'repair-initial-preservation', 'seconds': preservation_seconds,
                                     'kind': 'measured', 'source': {'operation': 'Owner-brief state preservation'}})
            state['events'].append({'event': REPAIR_EVENT, 'epoch': time.time(),
                                    'legacy_total_seconds': HISTORICAL_SECONDS,
                                    'evidence_sha256': evidence_sha256,
                                    'preservation_seconds': preservation_seconds})
            return resource_summary(state)


def machine_hours(state: dict) -> float:
    return total_seconds(state) / HOUR


def headroom_hours(state: dict) -> float:
    return (CAPS['machine_seconds'] - total_seconds(state)) / HOUR
