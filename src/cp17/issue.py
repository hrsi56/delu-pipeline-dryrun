"""The product's daily issuance: the frozen failure policy, the success definition, and the issued artifact.

**The failure policy** (the Owner's decisions 11 and 12, 2026-10-28):

- GFS late or missing at the issue time: A1's fit is recorded as skipped for an unavailable input, and the
  chain is v2, then B0, then a failed issuance;
- A1's own fit fails while the weather is available: v4, then v2, then B0, then a failed issuance;
- a fallback whose own fit fails passes to the next one in its chain;
- A1 issues whenever its own fit succeeds; the validity of other policies' members never triggers a fallback;
- every fallback is complete: no subset of members is renormalized and no quantile is manufactured;
- a failed or skipped fit is displayed as such, even when a fallback issues (§16.2).

**A successful daily fit** (§16.2, as the brief fixes it), for delivery day D issued at D-1 11:00
Europe/Berlin, is one that passes all of:

1. *inputs*: the payload's identity verifies; the target day's outcome is absent (the embargo); released
   errors in any interval buffer are from complete delivery days <= D-2; training targets end before D;
2. *training*: the policy's members train under the frozen rules with no guard refusal and no non-finite
   value (a `dailyfit` exception is a failed fit);
3. *emission* (decision 13): seven finite, ordered quantiles for every hour of D that the frozen design can
   forecast: the payload's forecast keys, which the landed rule sets (`cp15.data.Inputs.rows`, whose rows are
   `feature_valid`; `cp24.member.Keys.from_data`). A fit on fewer hours than the day's 23, 24 or 25 is successful
   on those hours and displayed as partial, with the missing hours named; a day with no forecastable hour fails
   every policy at its inputs, so it is a failed issuance;
4. *the artifact*: written with its training window, labels, seeds, timestamps and parent, input and state
   hashes, and it verifies (`artifact`, `verify_artifact`), judged inside the chain before it moves on (`decide`);
5. *resources*: on the issuing host, one worker, within 15 minutes of wall time and 4 GB of peak memory, each
   attempt measured in its own fresh process (`attempt`'s `resources`). An issued artifact always carries its
   fit's measurement; an unmeasured attempt (the native comparison) can never issue.

Released errors come from complete delivery days <= D-2 (§2); training targets are the delivery days before D
published strictly before the issue time, so D-1 is one (decision 16). Anything else is a failed fit, and the
chain moves on: a fit that fails any criterion is never issued. A fallback forecast never counts as a successful
A1 fit.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import time
import traceback

import numpy as np

from .dailyfit import FitFailure, array_hash, canonical

PRODUCT = 'delu-day-ahead-policy'
DESIGNATED = 'A1'
CHAINS = {'gfs_unavailable': ('v2', 'B0'), 'a1_failed': ('v4', 'v2', 'B0')}
WALL_LIMIT_SECONDS = 15 * 60
MEMORY_LIMIT_BYTES = 4 * 1024 ** 3
SCHEMA = 'cp17b-issued-artifact-v2'
COMPARISON = 'not judged: the native comparison path, which can never issue'
NO_FORECASTABLE_HOUR = 'no hour of the day is forecastable by the frozen design'


def issue_utc(day: date) -> str:
    """D-1 11:00 Europe/Berlin as UTC (§24.4), by the landed `cp25.guard.issue_time`."""
    from cp25.guard import issue_time
    return issue_time(day).isoformat()


def canonical_hours(day: date) -> list[str]:
    import pandas as pd
    local = pd.date_range(pd.Timestamp(day, tz='Europe/Berlin'), pd.Timestamp(day + timedelta(days=1), tz='Europe/Berlin'),
                          freq='h', inclusive='left')
    return [t.isoformat() for t in local.tz_convert('UTC')]


def forecastable_keys(policy: str, arrays: dict, meta: dict) -> list[str]:
    """The hours of D the frozen design can forecast, as the payload carries them (decision 13). The landed rule
    decides: a row of D is a forecast key when its inputs are valid (`cp15.data.Inputs.rows`, `feature_valid`).
    v2's and v4's payloads carry the rows, so the rule is applied to them here; A1's keys are the day table's
    (`cp24.member.Keys.from_data`, the same rows) and B0's are those rows' timestamps, both written by the
    payload builder from the same rule."""
    import pandas as pd
    day = date.fromisoformat(meta['day'])
    if policy == 'A1':
        ns = arrays['key_ts']
    elif policy in ('v2', 'v4'):
        from .dailyfit import rows_inputs
        ns = arrays['index_ns'][rows_inputs(arrays, meta).rows(day)]
    elif policy == 'B0':
        ns = arrays['key_ns']
    else:
        raise ValueError(f'unknown policy {policy}')
    return [pd.Timestamp(int(t), tz='UTC').isoformat() for t in ns]


def check_inputs(policy: str, arrays: dict, meta: dict) -> list[str]:
    """Criterion 1 on the shipped payload; returns the failed checks (empty when it passes)."""
    from .dailyfit import payload_digest
    day = date.fromisoformat(meta['day'])
    problems = []
    if meta.get('payload_sha256') != payload_digest(arrays, {k: v for k, v in meta.items() if k != 'payload_sha256'}):
        return ['the payload differs from its recorded identity']
    if not forecastable_keys(policy, arrays, meta):
        return [NO_FORECASTABLE_HOUR]
    if policy == 'A1':
        if np.isfinite(arrays['price'][-1]).any() or np.isfinite(arrays['y'][-1]).any() or arrays['mask'][-1].any():
            problems.append("the target day's outcome is present in the day table")
        if int(arrays['days'][-1]) != (day - date(1970, 1, 1)).days:
            problems.append('the day table does not end at the target day')
    elif policy in ('v2', 'v4'):
        on_day = arrays['dates'] == (day - date(1970, 1, 1)).days
        if np.isfinite(arrays['y'][on_day]).any():
            problems.append("the target day's outcome is present in the rows")
    elif policy == 'B0':
        if np.isfinite(arrays['price'][np.isin(arrays['price_ns'], arrays['key_ns'])]).any():
            problems.append("the target day's prices are present")
    for key in ('h_buffer', 'b0_buffer'):
        if key in meta and any(date.fromisoformat(b['day']) > day - timedelta(days=2) for b in meta[key]):
            problems.append(f'{key}: an error day is not released under the complete D-2 rule')
    return problems


def check_emission(result: dict, day: date, expected: list[str] | None = None) -> list[str]:
    """Criterion 3: seven finite, ordered quantiles on exactly the forecastable hours (`expected`, from the
    payload; every delivery hour of D when no payload is given)."""
    q = np.asarray(result['quantiles'], float)
    problems = []
    expected = canonical_hours(day) if expected is None else expected
    if not set(expected) <= set(canonical_hours(day)):
        problems.append(f'a forecastable key is not an hour of {day}')
    if result['index'] != expected:
        problems.append(f"the keys are not the forecastable hours of {day} ({len(result['index'])} emitted, "
                        f"{len(expected)} forecastable)")
    if q.ndim != 2 or q.shape[1] != 7 or q.shape[0] != len(result['index']):
        problems.append('not seven quantiles per key')
    elif not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
        problems.append('the quantiles are not finite and ordered')
    return problems


def partial(day: date, keys: list[str]) -> dict | None:
    """The partial-day record (decision 13): None for a complete day, else the missing hours named."""
    import pandas as pd
    full = canonical_hours(day)
    missing = [k for k in full if k not in set(keys)]
    if not missing:
        return None
    names = [f"{pd.Timestamp(k).tz_convert('Europe/Berlin'):%H:%M} Europe/Berlin ({k})" for k in missing]
    return {'issued_hours': len(keys), 'canonical_hours': len(full), 'missing_hours': names}


def _partial_text(item: dict | None) -> str:
    if not item:
        return ''
    short = ', '.join(n.split(' ')[0] for n in item['missing_hours'])
    return f" (partial: {item['issued_hours']} of {item['canonical_hours']} hours; missing {short} Europe/Berlin)"


def attempt(policy: str, fit, day: date, arrays: dict | None = None, meta: dict | None = None,
            resources=None) -> dict:
    """One policy's fit at the day, judged by criteria 1, 2, 3 and 5; `fit()` returns the `dailyfit` result.

    `resources()` returns this fit's own measured {'wall_seconds', 'peak_bytes'} on the issuing host, where each
    attempt runs in its own fresh process, so the peak is the fit's own. A fit over the ceiling is a failed fit:
    its quantiles are never issued, and the chain moves on. With `resources=None` (the native comparison)
    criterion 5 is not judged, the record says so, and `artifact` refuses to issue it (N2)."""
    started = time.perf_counter()
    item = {'policy': policy, 'status': 'failed', 'cause': None}
    try:
        expected = None
        if arrays is not None and meta is not None:
            problems = check_inputs(policy, arrays, meta)
            if problems:
                item['cause'] = 'inputs: ' + '; '.join(problems)
                return item
            expected = forecastable_keys(policy, arrays, meta)
        result = fit()
        problems = check_emission(result, day, expected)
        if problems:
            item['cause'] = 'emission: ' + '; '.join(problems)
            return item
        item['partial'] = partial(day, result['index'])
        if resources is not None:
            measured = resources()
            item['resources'] = {**measured, 'limits': {'wall_seconds': WALL_LIMIT_SECONDS, 'peak_bytes': MEMORY_LIMIT_BYTES}}
            problems = resources_ok(measured['wall_seconds'], measured['peak_bytes'])
            if problems:
                item['cause'] = 'resources: ' + '; '.join(problems)
                return item
        else:
            item['resources'] = COMPARISON
        item.update(status='ok', result=result)
    except Exception as exc:     # a failed fit is recorded with its cause, never retried with other settings
        item['cause'] = f'training: {type(exc).__name__}: {exc}'
        item['traceback'] = traceback.format_exc(limit=4)
    finally:
        item['seconds'] = time.perf_counter() - started
    return item


def decide(day: date, gfs_available: bool, attempts, finalize=None) -> dict:
    """Walk the frozen chain. `attempts(policy)` returns `attempt(...)`'s record; it is called only for the
    policies the chain reaches, in order, so the chain fits A1 alone on a normal day.

    `finalize(decision)` builds the artifact a successful attempt would issue; criterion 4 is judged there, inside
    the chain (N3): if the artifact does not verify, that fit fails and the chain moves on. A failed issuance is
    written too, and must verify, because failures remain in the record (§16.2). Without `finalize` (the native
    comparison, the unit tests) no artifact is built."""
    record = []

    def accept(item, policy, route, display):
        decision = {'day': str(day), 'issued_policy': policy, 'route': route, 'attempts': record,
                    'display': display + _partial_text(item.get('partial'))}
        if finalize is None:
            return decision
        try:
            built = finalize(decision)
            problems = verify_artifact(built)
        except Exception as exc:      # an artifact that cannot be written is criterion 4 failing
            built, problems = None, [f'{type(exc).__name__}: {exc}']
        if problems:
            item.update(status='failed', cause='artifact: ' + '; '.join(problems))
            return None
        decision['artifact'] = built
        return decision

    if not gfs_available:
        record.append({'policy': 'A1', 'status': 'skipped', 'cause': 'GFS late or missing at the issue time'})
        chain, reason = CHAINS['gfs_unavailable'], 'gfs_unavailable'
    else:
        first = attempts('A1')
        record.append(first)
        if first['status'] == 'ok':
            done = accept(first, 'A1', 'designated', 'successful daily fit')
            if done is not None:
                return done
        chain, reason = CHAINS['a1_failed'], 'a1_failed'
    for policy in chain:
        item = attempts(policy)
        record.append(item)
        if item['status'] == 'ok':
            head = 'skipped fit (GFS late or missing)' if reason == 'gfs_unavailable' else 'failed fit'
            done = accept(item, policy, reason, f'{head}; fallback {policy} issued')
            if done is not None:
                return done
    display = 'failed issuance'
    if all(a['status'] == 'skipped' or (a.get('cause') or '').endswith(NO_FORECASTABLE_HOUR) for a in record):
        display += f' ({NO_FORECASTABLE_HOUR})'
    decision = {'day': str(day), 'issued_policy': None, 'route': reason, 'attempts': record, 'display': display}
    if finalize is not None:
        built = finalize(decision)
        problems = verify_artifact(built)
        if problems:
            raise ValueError('the failed issuance\'s record does not verify: ' + '; '.join(problems))
        decision['artifact'] = built
    return decision


def resources_ok(wall_seconds: float, peak_bytes: int) -> list[str]:
    problems = []
    if wall_seconds > WALL_LIMIT_SECONDS:
        problems.append(f'wall time {wall_seconds!r} s over the {WALL_LIMIT_SECONDS} s (15-minute) ceiling')
    if peak_bytes > MEMORY_LIMIT_BYTES:
        problems.append(f'peak memory {peak_bytes} bytes over the {MEMORY_LIMIT_BYTES}-byte (4 GiB) ceiling')
    return problems


def artifact(decision: dict, *, identity: dict, payloads: dict, parent_sha256: str | None, data_refresh_utc: str,
             host: dict) -> dict:
    """The issued artifact (criterion 4). Criterion 5 was judged per attempt (`attempt`), so the issued policy is
    the first in its chain whose fit met every criterion; its own measured resources are recorded."""
    day = date.fromisoformat(decision['day'])
    issued = next((a for a in decision['attempts'] if a.get('policy') == decision['issued_policy']), None)
    if issued is not None and not _measured(issued.get('resources')):
        raise ValueError('an issued artifact needs its fit\'s resource measurement; the native comparison never issues')
    result = issued['result'] if issued else None
    attempts = [{k: v for k, v in a.items() if k not in ('result', 'traceback')} for a in decision['attempts']]
    body = {
        'schema': SCHEMA, 'product': PRODUCT, 'designated_policy': DESIGNATED, 'delivery_day': str(day),
        'issue_utc': issue_utc(day), 'issued_policy': decision['issued_policy'], 'route': decision['route'],
        'display': decision['display'], 'attempts': attempts,
        'keys': result['index'] if result else None, 'quantiles': result['quantiles'] if result else None,
        'quantiles_sha256': result['quantiles_sha256'] if result else None,
        'times': {'last_training_utc': issue_utc(day), 'data_refresh_utc': data_refresh_utc, 'issuance_utc': issue_utc(day)},
        'lineage': {'parent_artifact_sha256': parent_sha256, 'policy_identity_sha256': identity.get('policy_identity_sha256'),
                    'code_sha256': identity.get('package_sha256'), 'host': host,
                    'inputs': {p: payloads[p].get('payload_sha256') for p in payloads},
                    'training': _training(result) if result else None,
                    'state': _state(result) if result else None},
        'resources': issued.get('resources') if issued else None,
        'partial': issued.get('partial') if issued else None}
    body['artifact_sha256'] = hashlib.sha256(canonical(body)).hexdigest()
    return body


def _measured(value) -> bool:
    return isinstance(value, dict) and isinstance(value.get('wall_seconds'), (int, float)) \
        and isinstance(value.get('peak_bytes'), int) and not isinstance(value.get('peak_bytes'), bool)


def _training(result: dict) -> dict:
    if result['policy'] == 'A1':
        return {'search': result['search'], 'ensemble_content_sha256': result['ensemble_content_sha256'],
                'seeds': [m['seed'] for m in result['members']],
                'members': [{k: m[k] for k in ('member', 'seed', 'train_days_sha256', 'holdout_days_sha256',
                                               'latest_training_day', 'params_sha256')} for m in result['members']]}
    if result['policy'] in ('v2', 'v4'):
        return {'train_rows': result['train_rows'], 'seeds': {'lear': None, 'lightgbm_random_state': 42}}
    return {'source_day': result.get('source_day')}


def _state(result: dict) -> dict | None:
    return result.get('h_layer') or result.get('b0_layer')


def verify_artifact(item: dict) -> list[str]:
    """The artifact's own identity; its keys and quantiles; its resource measurement within the ceiling; its
    issue time. Empty when it verifies."""
    problems = []
    body = {k: v for k, v in item.items() if k != 'artifact_sha256'}
    if hashlib.sha256(canonical(body)).hexdigest() != item.get('artifact_sha256'):
        problems.append('artifact identity differs')
    if item.get('issued_policy') is not None:
        q = np.asarray(item['quantiles'], float)
        if q.shape != (len(item['keys']), 7) or array_hash(q) != item['quantiles_sha256']:
            problems.append('quantiles differ from their recorded hash')
        if not _measured(item.get('resources')):
            problems.append('an issued artifact without its fit\'s resource measurement')
        elif resources_ok(item['resources']['wall_seconds'], item['resources']['peak_bytes']):
            problems.append('an issued fit over the resource ceiling')
        day = date.fromisoformat(item['delivery_day'])
        full = canonical_hours(day)
        if not item['keys'] or item['keys'] != [k for k in full if k in set(item['keys'])]:
            problems.append(f'the keys are not ordered, distinct hours of {day}')
        elif item.get('partial') != partial(day, item['keys']):
            problems.append('the partial-day record does not name exactly the missing hours')
    if item.get('issue_utc') != issue_utc(date.fromisoformat(item['delivery_day'])):
        problems.append('the issue time is not D-1 11:00 Europe/Berlin')
    return problems
