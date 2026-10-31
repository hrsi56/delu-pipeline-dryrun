"""Pure conservative CP-26 pricing from sealed, identity-bound timing evidence.

This module performs no fit, reads no data and grants no experiment authority.
Custody of committed plans/protocols remains the execution gate's responsibility.
"""
from __future__ import annotations

import copy
from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path

from .budget import CAPS, admission_hours, CapExceeded, atomic, projected_cost, total_seconds

SCHEMA = 'cp26-cost-projection-v1'
ROUTES = ('nominal', 'noload', 'gfs6')
ORIGINS = tuple(str(date(2020, 1, 15) + timedelta(days=182 * k)) for k in range(14))
DAILY_MEMBERS = {'nominal': ('B0', 'A1', 'B2', 'A1_w', 'B2_w', 'L-N', 'L-R', 'v1-R'),
                 'noload': ('A1_w', 'B2_w', 'L-N', 'L-R'),
                 'gfs6': ('A1_w', 'B2_w', 'L-N', 'L-R')}
ADDITIONAL = frozenset(('parity', 'tests', 'bootstrap', 'diagnostics', 'failed_jobs_and_resume'))


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _signed(value: dict) -> dict:
    return {**value, 'content_sha256': digest(value)}


def _verify(value: dict, schema: str) -> None:
    if (not isinstance(value, dict) or value.get('schema') != schema
            or value.get('content_sha256') != digest({k: v for k, v in value.items() if k != 'content_sha256'})):
        raise ValueError(f'corrupt or unsigned {schema} evidence')


def _positive(value, label: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError(f'{label} must be finite and positive')
    return float(value)


def _integer(value, label, upper):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= upper:
        raise ValueError(f'{label} is outside the frozen domain')
    return value


def _identity(row):
    if not isinstance(row, dict) or row.get('route') not in ROUTES:
        raise ValueError('timing/plan route is outside the frozen domain')
    return row['route'], _integer(row.get('k'), 'search k', 13)


def _inventory(rows, label):
    expected = {(route, k) for route in ROUTES for k in range(14)}
    if not isinstance(rows, list) or len(rows) != 42:
        raise ValueError(f'{label} requires all 42 route/search identities')
    result = {}
    for row in rows:
        key = _identity(row)
        if key in result:
            raise ValueError(f'duplicate {label} identity')
        result[key] = row
    if set(result) != expected:
        raise ValueError(f'{label} omits frozen searches')
    return result


def halving_fit_count(batches: int) -> int:
    batches = _integer(batches, 'retained batches B', 11)
    return 128 * min(4, batches) + 43 * max(0, batches - 4)


def _cost_basis(row, label):
    if (not isinstance(row, dict) or row.get('classification') not in ('measured', 'conservative')
            or not isinstance(row.get('basis'), str) or not row['basis'].strip()
            or not _sha(row.get('evidence_sha256')) or row.get('completed_evidence') is not None):
        raise ValueError(f'{label} requires an identified measured/conservative basis')
    _positive(row.get('machine_seconds'), label)
    return copy.deepcopy(row)


def _protocol(raw: bytes, identity: dict, code: dict):
    if not isinstance(raw, bytes):
        raise ValueError('protocol bytes are required to verify the frozen document hash')
    protocol = json.loads(raw)
    if (protocol.get('schema') != 'cp26-execution-protocol-v21-r14'
            or identity.get('schema') != protocol['schema']
            or identity.get('document_sha256') != hashlib.sha256(raw).hexdigest()
            or identity.get('authority_sha256') != protocol.get('authority_sha256')
            or identity.get('research_sha256') != protocol.get('research_sha256')
            or not all(_sha(identity.get(key)) for key in ('document_sha256', 'authority_sha256',
                                                         'research_sha256', 'cp28_protocol_sha256'))):
        raise ValueError('costing protocol/authority identity mismatch')
    calendar = protocol.get('calendar', {})
    pricing = protocol.get('pilot', {}).get('projection', {})
    if (calendar.get('origins') != 2441 or calendar.get('routes') != list(ROUTES)
            or calendar.get('first') != '2020-01-01' or calendar.get('last') != '2026-09-06'
            or protocol.get('pilot', {}).get('origins') != list(ORIGINS)
            or pricing.get('ceiling_hours') != 485 or pricing.get('stop_hours') not in (388, 436.5)
            or CAPS['machine_seconds'] != 500 * 3600 or admission_hours() != 436.5):
        raise ValueError('costing protocol differs from frozen calendar/resource choices')
    if (not isinstance(code, dict) or not code
            or any(not isinstance(path, str) or not path or not _sha(sha) for path, sha in code.items())):
        raise ValueError('projection requires current code file identities')
    return protocol


def _plans(plans, manifest, pilot, protocol, code, recipe_code=None):
    _verify(manifest, 'cp26-execution-manifest-v1')
    if (manifest.get('protocol_identity') != protocol or manifest.get('protocol_sha256') != digest(protocol)
            or pilot.get('execution_manifest_sha256') != manifest['content_sha256']):
        raise ValueError('pilot does not bind the frozen execution manifest')
    indexed, references = _inventory(plans, 'plans'), _inventory(manifest.get('searches'), 'manifest')
    for key, plan in indexed.items():
        _verify(plan, 'cp26-search-plan-v2')
        support = plan.get('support')
        if not isinstance(support, dict) or support.get('content_sha256') != digest(
                {k: v for k, v in support.items() if k != 'content_sha256'}):
            raise ValueError('search support is missing or corrupt')
        b = _integer(plan.get('B'), 'retained batches B', 11)
        route, k = key
        ref = references[key]
        if (plan.get('protocol') != 'cp26-ddnn2-search-v21-r14' or support.get('B') != b
                or support.get('k') != k or support.get('search_day') != ORIGINS[k]
                or plan.get('search_day') != ORIGINS[k] or len(support.get('batches', [])) != b
                or len(plan.get('trials', [])) != 128 or ref.get('content_sha256') != plan['content_sha256']
                or ref.get('B') != b or ref.get('search_day') != ORIGINS[k]
                or ref.get('identity') != plan.get('identity')):
            raise ValueError('plan/manifest support binding differs')
        identity = plan.get('identity', {})
        bound_code = identity.get('code_sha256')
        if (identity.get('route') != route or identity.get('k') != k
                or identity.get('protocol') != plan['protocol'] or not isinstance(bound_code, dict) or not bound_code
                or any(not _sha(value) for value in bound_code.values())
                or any(code.get(path) != sha and not (
                    path in ('src/cp26/harness.py', 'src/cp26/search.py')
                    and (recipe_code or {}).get(path) == sha)
                    for path, sha in bound_code.items() if path.startswith('src/cp26/'))):
            raise ValueError('search plan differs from current code identity')
        expected = [26_000_000 + k * 100_000 + 90_000 + 10 * rank + repeat
                    for rank in range(1, 5) for repeat in (1, 2)]
        if plan.get('member_seeds') != expected:
            raise ValueError('plan member seeds differ from frozen domains')
    return indexed


def _selected(plan):
    seen, selected = set(), []
    for trial, config in enumerate(plan['trials']):
        effective = {k: v for k, v in config.items() if k != 'id'}
        if plan['route'] == 'noload' and 'groups' in effective:
            effective['groups'] = [g for g in effective['groups'] if g not in ('load_d1', 'load_d7')]
        key = digest(effective)
        if key not in seen:
            seen.add(key)
            selected.append((trial, config))
        if len(selected) == 4:
            return selected
    raise ValueError('pilot cannot price eight daily D2 members from fewer than four configurations')


def _probe_group(rows, plan, protocol):
    route, k = _identity(plan)
    expected = {f'daily-{member}' for member in DAILY_MEMBERS[route]}
    expected |= {f'daily-D2-r{rank}-s{repeat}' for rank in range(1, 5) for repeat in (1, 2)}
    if plan['B']:
        expected |= {f'search-t{trial:03d}-b00' for trial in range(12)}
    found, daily, search, failed = {}, [], {}, 0
    selected = _selected(plan)
    for probe in rows:
        _verify(probe, 'cp26-timing-probe-v1')
        key = probe.get('key', {})
        task = key.get('task')
        if (task not in expected or task in found or _identity(key) != (route, k)
                or key.get('plan_sha256') != plan['content_sha256']
                or key.get('protocol_sha256') != digest(protocol) or key.get('identity') != plan['identity']
                or probe.get('status') not in ('ok', 'failed')):
            raise ValueError('timing probe identity/task/status differs from frozen plan')
        wall = _positive(probe.get('wall_seconds'), 'outer observed probe wall time')
        found[task] = probe
        failed += int(probe['status'] == 'failed')
        if task.startswith('search-'):
            trial = _integer(key.get('trial'), 'probe trial', 11)
            if (task != f'search-t{trial:03d}-b00' or key.get('batch') != 0
                    or key.get('config') != plan['trials'][trial]
                    or key.get('seed') != 26_000_000 + k * 100_000 + trial * 100
                    or key.get('origin') != plan['support']['batches'][0]['first']):
                raise ValueError('search probe config/seed/batch binding differs')
            search[trial] = wall
        else:
            if key.get('origin') != ORIGINS[k]:
                raise ValueError('daily timing probe origin differs from frozen selection')
            if task.startswith('daily-D2-'):
                rank, repeat = key.get('rank'), key.get('repeat')
                if (type(rank) is not int or not 1 <= rank <= 4 or type(repeat) is not int or repeat not in (1, 2)
                        or task != f'daily-D2-r{rank}-s{repeat}'
                        or (key.get('trial'), key.get('config')) != selected[rank - 1]
                        or key.get('seed') != plan['member_seeds'][2 * (rank - 1) + repeat - 1]):
                    raise ValueError('daily D2 probe draw/seed binding differs')
            elif key.get('member') != task.removeprefix('daily-'):
                raise ValueError('daily timing probe member differs')
            daily.append(wall)
    if set(found) != expected:
        raise ValueError('timing pilot omits required raw probes; failed fits must remain represented')
    return sum(daily), [search[trial] for trial in range(12)] if plan['B'] else [], failed


def price_pilot(state: dict, pilot: dict, *, plans: list[dict], execution_manifest: dict,
                protocol_bytes: bytes, protocol: dict, code_sha256: dict,
                additional_costs: dict, zero_support_overheads: dict | None = None,
                recipe_code_sha256: dict | None = None) -> dict:
    """Price all 42 searches and three full daily routes without adding a ledger charge."""
    _protocol(protocol_bytes, protocol, code_sha256)
    _verify(pilot, 'cp26-timing-pilot-v1')
    if (pilot.get('protocol') != protocol or pilot.get('protocol_identity') != protocol
            or pilot.get('origins') != list(ORIGINS) or pilot.get('routes') != list(ROUTES)
            or pilot.get('evaluation_scores_computed') is not False
            or pilot.get('serving_ensembles_created') is not False):
        raise ValueError('timing pilot is not bound sealed resource evidence')
    indexed = _plans(plans, execution_manifest, pilot, protocol, code_sha256, recipe_code_sha256)
    summaries = _inventory(pilot.get('summary'), 'pilot summaries')
    probes = pilot.get('probes')
    if not isinstance(probes, list) or not probes:
        raise ValueError('retained raw timing probes are required')
    grouped = {key: [] for key in indexed}
    for probe in probes:
        key = _identity(probe.get('key', {}))
        grouped[key].append(probe)
    zero_support_overheads = zero_support_overheads or {}
    searches, daily_by_route = [], {route: [] for route in ROUTES}
    for route in ROUTES:
        for k in range(14):
            plan, summary, rows = indexed[route, k], summaries[route, k], grouped[route, k]
            daily, samples, failed = _probe_group(rows, plan, protocol)
            if (summary.get('origin') != ORIGINS[k] or summary.get('probe_sha256') != [p['content_sha256'] for p in rows]
                    or summary.get('daily_wall_seconds') != daily or summary.get('search_probe_wall_seconds') != samples
                    or type(summary.get('failed_probes')) is not int or summary['failed_probes'] != failed):
                raise ValueError('timing summary differs from retained raw probe details')
            fits = halving_fit_count(plan['B'])
            if fits:
                maximum = max(samples)
                seconds = maximum * fits * 1.5
                basis = (f'maximum of all 12 outer observed search-probe wall times, including failures: '
                         f'{maximum:.17g}s × (128×min(4,{plan["B"]}) + 43×max(0,{plan["B"]}−4)) '
                         f'= {fits} frozen fits × 1.5; {failed} total failed route/origin probes retained; '
                         'pilot compute already remains in cumulative ledger')
            else:
                overhead = _cost_basis(zero_support_overheads.get((route, k)), 'zero-support search overhead')
                seconds, maximum = overhead['machine_seconds'], None
                basis = 'B=0 retains prepriced search admission/failure-record overhead: ' + overhead['basis']
            row = {'run': route, 'k': k, 'origin': ORIGINS[k], 'plan_sha256': plan['content_sha256'],
                   'classification': 'conservative', 'basis': basis, 'machine_seconds': _positive(seconds, 'search price'),
                   'B': plan['B'], 'halving_fit_count': fits, 'maximum_probe_wall_seconds': maximum,
                   'probe_sha256': summary['probe_sha256'], 'failed_probes': failed, 'margin': 1.5}
            if not fits:
                row['zero_support_overhead'] = overhead
            searches.append(row)
            daily_by_route[route].append((daily, k, summary['probe_sha256'], failed))
    if not isinstance(additional_costs, dict) or set(additional_costs) != ADDITIONAL:
        raise ValueError('identified parity/tests/bootstrap/diagnostics/failed-resume costs are all required')
    remaining = {name: _cost_basis(additional_costs[name], name) for name in sorted(ADDITIONAL)}
    daily_routes = []
    for route, observations in daily_by_route.items():
        maximum, k, probe_hashes, failed = max(observations, key=lambda row: (row[0], -row[1]))
        daily_routes.append({'run': route, 'maximum_origin_wall_seconds': maximum, 'maximum_origin': ORIGINS[k],
                             'pilot_origins': 14, 'total_origins': 2441, 'margin': 1.25,
                             'machine_seconds': _positive(maximum * 2441 * 1.25, 'daily route price'),
                             'probe_sha256': probe_hashes, 'failed_probes_at_maximum': failed})
    remaining['daily_fits'] = {'classification': 'conservative', 'total_origins': 2441,
        'route_costs': daily_routes, 'machine_seconds': sum(row['machine_seconds'] for row in daily_routes),
        'basis': 'each of three routes: maximum of all 14 full-origin outer wall costs including failed fits '
                 '× 2441 frozen origins × 1.25; pilot compute already remains in cumulative ledger',
        'evidence_sha256': digest(pilot)}
    estimate = projected_cost(state, searches, remaining, run_id=protocol.get('run_id'))
    return _signed({'schema': SCHEMA, 'protocol': copy.deepcopy(protocol), 'code_sha256': copy.deepcopy(code_sha256),
                    'pilot_sha256': digest(pilot), 'execution_manifest_sha256': execution_manifest['content_sha256'],
                    'ledger_sha256_at_pricing': digest(state), 'ledger_seconds_at_pricing': total_seconds(state),
                    'searches': searches, 'remaining': remaining, **estimate})


def require_projection(projection: dict, state: dict) -> dict:
    """Executable absolute admission refusal, repriced against all cumulative charges."""
    _verify(projection, SCHEMA)
    baseline = projection.get('ledger_seconds_at_pricing')
    if (not isinstance(baseline, (float, int)) or isinstance(baseline, bool)
            or not math.isfinite(baseline) or baseline < 0 or total_seconds(state) < baseline):
        raise RuntimeError('projection ledger baseline regressed or is invalid')
    current = projected_cost(state, projection['searches'], projection['remaining'],
                             run_id=projection.get('protocol', {}).get('run_id'))
    if not current['within_pilot_stop']:
        raise CapExceeded(f'CP-26 projection {current["projected_total_hours"]:.6f} hours exceeds '
                          f'the {current["pilot_stop_hours"]:g}-hour cumulative admission line; Owner resource/protocol decision required')
    return current


def persist_projection(root: Path, projection: dict, *, output=Path(
        'reports/rebaseline/continuation-v21-r14-accounting-3/cost-projection.json')) -> Path:
    """Retain the whole bound projection before applying its executable gate."""
    _verify(projection, SCHEMA)
    root = Path(root).resolve()
    path = (root / output).resolve()
    if not path.is_relative_to(root / 'reports/rebaseline'):
        raise ValueError('cost projection must remain in reports/rebaseline')
    if path.exists() and json.loads(path.read_text()) != projection:
        raise ValueError('frozen pilot projection differs; preserve its original identity')
    atomic(path, projection)
    return path
