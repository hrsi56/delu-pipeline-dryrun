"""Production CP-26 task scheduling and restart-safe forecast materialization.

The Lead supplies a bound authority/preparation/pilot gate and independently
verified CP-25 scoring keys. This module cannot authorize an experiment, choose
pilot origins, change a method, or compute evaluation scores. Searches, native
fit caches and causal replay use their existing production implementations.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from . import harness as H
from . import search as S
from .budget import Budget
from .fits import FitCache, _plain, fit_members_cached, frozen_fit_identity
from .members import Context, RUN_MEMBERS, fit_members
from .replay import POLICIES, bundles_from_fits, replay_origin
from .state import ReplayState, _save, digest, timestamps

from .gates import OUTPUT, LOCAL_NAME
SCHEMA = 'cp26-execution-manifest-v1'
DAY_SCHEMA = 'cp26-forecast-day-v1'
ROUTES = tuple(S.ROUTES)


@dataclass(frozen=True)
class Paths:
    root: Path
    output: Path
    local: Path

    @classmethod
    def build(cls, root, output_root=None):
        root = Path(root).resolve()
        output = Path(output_root or root / OUTPUT)
        if not output.is_absolute():
            output = root / output
        output = output.resolve()
        if not output.is_relative_to(root / 'reports/rebaseline'):
            raise ValueError('durable execution outputs must remain in reports/rebaseline')
        return cls(root, output, H.art() / LOCAL_NAME)

    def plan(self, route, k):
        return self.output / 'search-plans' / route / f'k{k:02d}.json'

    def result(self, route, k):
        return self.output / 'search-results' / route / f'k{k:02d}.json'

    def trial_cache(self, route, k):
        return self.local / 'searches' / route / f'k{k:02d}'

    def state(self, identity):
        return self.local / 'state' / digest(identity) / 'replay-state.json'

    def day(self, day):
        return self.output / 'forecast-ledger' / f'{day}.json'


def calendar(first=H.WARMUP_START, last=H.NEW_BOUNDARY):
    if type(first) is not date or type(last) is not date or not H.WARMUP_START <= first <= last <= H.NEW_BOUNDARY:
        raise ValueError('execution dates must stay inside the frozen warm-up/development calendar')
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def _gate(gate, stage, *, fitting=False):
    if not callable(gate):
        raise ValueError('a bound Lead authority/preparation/pilot gate is required')
    if not os.environ.get('CP26_MONITORED_JOB'):
        raise RuntimeError('execution requires the cumulative CP-26 monitor')
    H.check_blas()
    gate(stage)
    if fitting:
        S._require_fit_authorization()


def _signed(value):
    return {**value, 'content_sha256': digest(value)}


def _read(path):
    value = json.loads(Path(path).read_text())
    if value.get('content_sha256') != digest({k: v for k, v in value.items() if k != 'content_sha256'}):
        raise ValueError(f'corrupt execution artifact: {Path(path).name}')
    return value


def _immutable(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise ValueError(f'immutable execution artifact differs: {path.name}')
    else:
        _save(path, value)
    return value


def record_evidence_once(budget, task_id, evidence, counts=None):
    """Reconcile immutable task evidence atomically; never charge machine time.

    Supervisor receipts own duration. These counters describe completed cache
    evidence, including failed entries, rather than pretending to measure all
    attempted solver calls in an interrupted process.
    """
    proposed = {'id': task_id, 'evidence': _plain(evidence), 'counts': counts or {}}
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in proposed['counts'].values()):
        raise ValueError('completed-evidence counts must be nonnegative integers')
    return budget.append_execution_task(proposed, {
        'event': 'cp26_execution_task_terminal', 'task_id': task_id,
        'evidence_sha256': digest(proposed['evidence'])})


def prepare(root, protocol_identity, *, gate, output_root=None):
    """Freeze all 42 support/sampler plans; the Lead commits them before fitting."""
    _gate(gate, 'prepare')
    if not isinstance(protocol_identity, dict) or not protocol_identity:
        raise ValueError('a frozen protocol identity is required')
    paths = Paths.build(root, output_root)
    plans = []
    for route in ROUTES:
        for k in range(S.SEARCH_COUNT):
            plan = S.prepare_search(paths.root, k, route, paths.plan(route, k))
            plans.append({'route': route, 'k': k, 'path': str(paths.plan(route, k).relative_to(paths.root)),
                          'content_sha256': plan['content_sha256'], 'B': plan['B'],
                          'search_day': plan['search_day'], 'identity': plan['identity']})
    manifest = _signed({'schema': SCHEMA, 'protocol_identity': protocol_identity,
                        'protocol_sha256': digest(protocol_identity), 'routes': list(ROUTES),
                        'warmup_first': str(H.WARMUP_START), 'scored_first': str(H.SCORED_START),
                        'calendar_last': str(H.NEW_BOUNDARY), 'searches': plans,
                        'inputs': H.identities(paths.root), 'input_fingerprint': H.input_fingerprint(paths.root)})
    return _immutable(paths.output / 'execution-manifest.json', manifest)


def manifest_for(paths, protocol_identity):
    value = _read(paths.output / 'execution-manifest.json')
    if value.get('schema') != SCHEMA or value.get('protocol_identity') != protocol_identity \
            or value.get('protocol_sha256') != digest(protocol_identity) or value.get('routes') != list(ROUTES) \
            or value.get('warmup_first') != str(H.WARMUP_START) or value.get('scored_first') != str(H.SCORED_START) \
            or value.get('calendar_last') != str(H.NEW_BOUNDARY):
        raise ValueError('execution manifest differs from frozen protocol/calendar')
    expected = {(r, k) for r in ROUTES for k in range(S.SEARCH_COUNT)}
    rows = value.get('searches', [])
    if len(rows) != len(expected) or {(r['route'], r['k']) for r in rows} != expected:
        raise ValueError('execution manifest must contain exactly 42 route-specific searches')
    if value['input_fingerprint'] != H.input_fingerprint(paths.root):
        raise ValueError('execution input/landed-code lineage changed')
    return value


def terminal_search(paths, route, k, manifest):
    """A missing search is a scheduler defect, not a scientific unavailable result."""
    row = next(r for r in manifest['searches'] if r['route'] == route and r['k'] == k)
    plan = _read(paths.plan(route, k))
    S.validate_plan(plan)
    if row['content_sha256'] != plan['content_sha256'] or row['identity'] != plan['identity'] \
            or plan['identity'] != S.search_identity(paths.root, k, route):
        raise ValueError('serving search plan/current lineage mismatch')
    path = paths.result(route, k)
    if not path.exists():
        raise RuntimeError(f'serving search {route}/k{k:02d} has no terminal result; no daily fit may start')
    result = _read(path)
    if result.get('schema') != 'cp26-search-result-v2' or result.get('protocol') != S.PROTOCOL \
            or result.get('k') != k or result.get('route') != route or result.get('identity') != plan['identity'] \
            or result.get('plan_sha256') != plan['content_sha256'] or result.get('B') != plan['B'] \
            or result.get('search_day') != str(S.search_day(k)) \
            or result.get('service_last') != str(S.search_day(k + 1) - timedelta(days=1)):
        raise ValueError('serving search result differs from frozen plan')
    if result.get('status') == 'available':
        S.validate_ensemble(result, k, route)
    elif result.get('status') != 'unavailable' or result.get('complete') or result.get('members') \
            or result.get('failure_reason') not in ('zero_eligible_batches', 'fewer_than_four_valid_effective_configurations'):
        raise ValueError('invalid unavailable serving search result')
    elif (result['failure_reason'] == 'zero_eligible_batches') != (plan['B'] == 0):
        raise ValueError('search failure reason disagrees with frozen support')
    return result


def run_searches(root, protocol_identity, *, gate, output_root=None, budget=None):
    """Execute/reuse every scheduled route-specific search, without evaluation scores."""
    _gate(gate, 'search', fitting=True)
    paths = Paths.build(root, output_root)
    manifest = manifest_for(paths, protocol_identity)
    budget = budget or Budget()
    out = {route: {} for route in ROUTES}
    for k in range(S.SEARCH_COUNT):
        for route in ROUTES:
            S.execute_search(paths.root, paths.plan(route, k), paths.trial_cache(route, k), paths.result(route, k))
            result = terminal_search(paths, route, k, manifest)
            record_evidence_once(budget, f"search/{digest(protocol_identity)}/{route}/{k}",
                                 {'result_sha256': result['content_sha256'], 'status': result['status']},
                                 {'completed_search_results': 1,
                                  'unavailable_search_results': int(result['status'] == 'unavailable')})
            out[route][str(k)] = result
    return out


def contexts_and_caches(paths, protocol_identity):
    contexts, caches = {}, {}
    for route in ROUTES:
        ctx = Context.build(paths.root, H.GFS6 if route == 'gfs6' else H.NEW, route)
        contexts[route] = ctx
        caches[route] = FitCache(paths.local / 'fits', route, frozen_fit_identity(ctx, protocol_identity))
    return contexts, caches


def _probe(paths, key, run, budget):
    """Retain complete timing evidence, including failure; never retry a completed probe."""
    path = paths.output / 'timing-pilot' / key['route'] / f"k{key['k']:02d}" / f"{key['task']}.json"
    if path.exists():
        result = _read(path)
        if result.get('key') != key:
            raise ValueError('timing probe differs from frozen protocol/plan/code')
    else:
        began = time.perf_counter()
        try:
            result = _plain(run())
        except Exception as exc:
            result = {'status': 'failed', 'cause': H.G25.redact(f'{type(exc).__name__}: {exc}')}
        result['wall_seconds'] = time.perf_counter() - began
        result = _signed({'schema': 'cp26-timing-probe-v1', 'key': key,
                          'operational_lineage': H.runtime_lineage(paths.root),
                          'issue_utc': H.issue_instant(date.fromisoformat(key['origin']), H.NEW).isoformat(),
                          'training_target_cutoff': key['origin'], **result})
        _immutable(path, result)
    record_evidence_once(budget, f"pilot/{digest(key)}", {'probe_sha256': result['content_sha256']},
                         {'completed_timing_probes': 1, 'failed_timing_probes': int(result['status'] == 'failed')})
    return result


def _raw_probe(ctx, origin, forecast_days, config, seed, budget):
    from cp24.member import fit_member
    fit = fit_member(ctx.dd, origin, config, seed, forecast_days, exclude_uncovered=True,
                     charge=lambda: budget.count(ddnn2_timing_fit_attempts=1))
    values = np.asarray(fit['eur'], float)
    if fit['window']['end_exclusive'] != str(origin) or (ctx.dd.days[forecast_days] < np.datetime64(origin)).any():
        raise ValueError('timing fit violated its origin/forecast date binding')
    return {'status': 'ok', 'origin': str(origin), 'issue_utc': H.issue_instant(origin, ctx.settings).isoformat(),
            'training_target_cutoff': str(origin), 'config': config, 'seed': int(seed),
            'window': fit['window'], 'seconds': float(fit['seconds']), 'maxrss_bytes': int(fit['maxrss_bytes']),
            'epochs_run': int(fit['member']['epochs_run']), 'best_epoch': fit['member'].get('best_epoch'),
            'stop_reason': fit['member'].get('stop_reason'),
            'forecast_first': str(ctx.dd.date_of(int(forecast_days[0]))),
            'forecast_last': str(ctx.dd.date_of(int(forecast_days[-1]))), 'forecast_days': len(forecast_days),
            'internal_refits': 0, 'finite_values': int(np.isfinite(values).sum()),
            'nonfinite_values': int((~np.isfinite(values)).sum()),
            'crossed_slots': int((np.diff(values, axis=-1) < 0).any(axis=-1).sum())}


def timing_pilot(root, protocol_identity, *, gate, output_root=None, budget=None, pairs=None):
    """Registered independent timing probes; no serving ensemble, replay or scores.

    Origins are S_k. Daily D2 probes use the first four distinct effective
    configurations from committed draws and their two frozen member seeds.
    Search probes use drawn trials 0..11 on retained batch 0, forecast all its
    28 days from one fit. The Lead's frozen protocol and price gate own these
    prospective choices and the conservative projection; this emits timings.
    """
    _gate(gate, 'pilot', fitting=True)
    paths = Paths.build(root, output_root)
    manifest = manifest_for(paths, protocol_identity)
    budget = budget or Budget()
    selected_pairs = set(pairs) if pairs is not None else {(r, k) for r in ROUTES for k in range(S.SEARCH_COUNT)}
    if not selected_pairs or not selected_pairs <= {(r, k) for r in ROUTES for k in range(S.SEARCH_COUNT)}:
        raise ValueError('pilot subset must use registered route/search pairs')
    contexts = {route: Context.build(paths.root, H.GFS6 if route == 'gfs6' else H.NEW, route)
                for route in ROUTES if any(r == route for r, _ in selected_pairs)}
    probes = []
    for k in range(S.SEARCH_COUNT):
        for route in ROUTES:
            if (route, k) not in selected_pairs:
                continue
            plan = _read(paths.plan(route, k))
            S.validate_plan(plan)
            if plan['identity'] != S.search_identity(paths.root, k, route):
                raise ValueError('timing plan differs from current frozen lineage')
            # Custody is a Lead commit, not merely a JSON signature.
            rel = str(paths.plan(route, k).relative_to(paths.root))
            if S.subprocess.check_output(['git', 'show', f'HEAD:{rel}'], cwd=paths.root) != paths.plan(route, k).read_bytes():
                raise ValueError('timing pilot requires prospectively committed support/sampler plans')
            ctx, origin = contexts[route], S.search_day(k)
            common = {'protocol_sha256': digest(protocol_identity), 'plan_sha256': plan['content_sha256'],
                      'identity': plan['identity'], 'route': route, 'k': k}
            for member in RUN_MEMBERS[route]:
                if member == 'D2':
                    continue
                def regular(member=member):
                    result = fit_members(ctx, origin, (member,))[member]
                    return {key: result[key] for key in ('status', 'cause', 'seconds', 'lineage', 'fits') if key in result}
                probes.append(_probe(paths, {**common, 'task': f'daily-{member}', 'member': member,
                                             'origin': str(origin)}, regular, budget))
            distinct, selected = set(), []
            for trial, config in enumerate(plan['trials']):
                effective = S.effective_key(config, route)
                if effective not in distinct:
                    distinct.add(effective)
                    selected.append((trial, config))
                if len(selected) == 4:
                    break
            if len(selected) != 4:
                raise ValueError('frozen pilot draws lack four distinct effective configurations')
            for rank, (trial, config) in enumerate(selected, 1):
                for repeat in (1, 2):
                    seed = S.member_seeds(k)[2 * (rank - 1) + repeat - 1]
                    key = {**common, 'task': f'daily-D2-r{rank}-s{repeat}', 'origin': str(origin),
                           'trial': trial, 'rank': rank, 'repeat': repeat, 'config': config, 'seed': seed}
                    def daily(config=config, seed=seed):
                        with H.ddnn2_without_load(ctx.noload_active):
                            return _raw_probe(ctx, origin, np.asarray([ctx.dd.ix(origin)]), config, seed, budget)
                    probes.append(_probe(paths, key, daily, budget))
            if not plan['B']:
                raise ValueError('pilot search pricing needs a supported newest batch; report the support gap')
            first = date.fromisoformat(plan['support']['batches'][0]['first'])
            forecast = np.asarray([ctx.dd.ix(first + timedelta(days=i)) for i in range(28)])
            for trial in range(12):
                config, seed = plan['trials'][trial], S.trial_fit_seed(k, trial, 0)
                key = {**common, 'task': f'search-t{trial:03d}-b00', 'origin': str(first),
                       'trial': trial, 'batch': 0, 'config': config, 'seed': seed}
                def search_probe(config=config, seed=seed):
                    with H.ddnn2_without_load(ctx.noload_active):
                        return _raw_probe(ctx, first, forecast, config, seed, budget)
                probes.append(_probe(paths, key, search_probe, budget))
    if pairs is not None:
        return {'status': 'SEALED_PILOT_SUBSET_COMPLETE', 'pairs': sorted(selected_pairs),
                'probes': len(probes), 'evaluation_scores_computed': False}
    summary = []
    for k in range(S.SEARCH_COUNT):
        for route in ROUTES:
            rows = [p for p in probes if p['key']['route'] == route and p['key']['k'] == k]
            summary.append({'route': route, 'k': k, 'origin': str(S.search_day(k)),
                            'daily_wall_seconds': sum(p['wall_seconds'] for p in rows if p['key']['task'].startswith('daily-')),
                            'search_probe_wall_seconds': [p['wall_seconds'] for p in rows if p['key']['task'].startswith('search-')],
                            'failed_probes': sum(p['status'] == 'failed' for p in rows),
                            'probe_sha256': [p['content_sha256'] for p in rows]})
    report = _signed({'schema': 'cp26-timing-pilot-v1', 'protocol_identity': protocol_identity,
                      'protocol': protocol_identity,
                      'execution_manifest_sha256': manifest['content_sha256'],
                      'origins': [str(S.search_day(k)) for k in range(S.SEARCH_COUNT)],
                      'routes': list(ROUTES), 'summary': summary, 'probes': probes,
                      'evaluation_scores_computed': False, 'serving_ensembles_created': False})
    return _immutable(paths.output / 'timing-pilot.json', report)


def _check_contexts(contexts, caches, identity):
    if set(contexts) != set(ROUTES) or set(caches) != set(ROUTES):
        raise ValueError('daily scheduling requires nominal, noload and gfs6 contexts/caches')
    for route in ROUTES:
        if contexts[route].variant != route or contexts[route].restore or caches[route].route != route \
                or caches[route].identity['protocol'] != identity:
            raise ValueError('daily route/context/cache identity mismatch')


def _keys_for(day, bundle, frozen_keys):
    prepared = timestamps(day, bundle['scoring_index'])
    if day < H.SCORED_START:
        return prepared
    key = day if day in frozen_keys else str(day)
    if key not in frozen_keys:
        raise ValueError(f'frozen CP-25 scoring keys lack an explicit entry for {day}')
    expected = timestamps(day, pd.to_datetime(frozen_keys[key], utc=True))
    if not expected.equals(prepared):
        raise ValueError(f'prepared scoring keys differ from frozen CP-25 records for {day}')
    return expected


def truth_from_context(ctx):
    """Causal replay requests only released dates; each requested row passes CP-25."""
    def truth(index):
        index = pd.DatetimeIndex(index)
        H.G25.admit_rows(index)
        positions = ctx.data.index.get_indexer(index)
        if (positions < 0).any():
            return np.full(len(index), np.nan)
        return ctx.data.y[positions].copy()
    return truth


def _reconcile_members(budget, caches, day, bundles):
    counts, evidence = {}, {'fit_identity_sha256': {}, 'members': {}}
    for route in ROUTES:
        evidence['fit_identity_sha256'][route] = digest(caches[route].identity)
        for member in RUN_MEMBERS[route]:
            result = bundles[route]['members'][member]
            counts['completed_member_cache_records'] = counts.get('completed_member_cache_records', 0) + 1
            counts['failed_member_cache_records'] = counts.get('failed_member_cache_records', 0) + int(result['status'] == 'failed')
            for kind, count in result.get('fits', {}).items():
                key = f'completed_member_{kind}'
                counts[key] = counts.get(key, 0) + int(count)
            evidence['members'][f'{route}/{member}'] = {'cache_sha256': result['cache_content_sha256'],
                                                      'status': result['status']}
    # One transaction per complete origin avoids repeatedly rewriting a growing
    # machine ledger for each of its 19 native member records.
    record_evidence_once(budget, f"member-day/{digest(caches['nominal'].identity['protocol'])}/{day}", evidence, counts)


def execute_day(state, day, contexts, caches, ensembles, frozen_keys, truth, *, budget):
    """Fit/reuse native members and issue all routes exactly once at this origin.

    ``ensembles`` contains verified terminal results, including explicit search
    failures. The caller must persist state before materializing daily output.
    """
    _check_contexts(contexts, caches, state.identity)
    expected = H.WARMUP_START if state.last_origin is None else state.last_origin + timedelta(days=1)
    if day != expected or day > H.NEW_BOUNDARY:
        raise ValueError('daily replay must advance one frozen calendar day at a time')
    k = S.serving_search(day)
    if k is not None and any(str(k) not in ensembles.get(route, {}) for route in ROUTES):
        raise RuntimeError('all three serving searches require terminal results before daily fits')
    if k is not None:
        for route in ROUTES:
            result = ensembles[route][str(k)]
            if result.get('schema') != 'cp26-search-result-v2' or result.get('protocol') != S.PROTOCOL \
                    or result.get('k') != k or result.get('route') != route \
                    or result.get('identity') != S.search_identity(Path(contexts[route].root), k, route) \
                    or result.get('content_sha256') != S.digest(result):
                raise ValueError('daily serving search/code/route identity mismatch')
            if result.get('status') == 'available':
                S.validate_ensemble(result, k, route)
            elif result.get('status') != 'unavailable' or result.get('complete') or result.get('members'):
                raise ValueError('daily serving search is not a verified terminal result')
    bundles = bundles_from_fits(contexts, caches, ensembles, day)
    for route in ROUTES:
        bundles[route]['scoring_index'] = _keys_for(day, bundles[route], frozen_keys)
    _reconcile_members(budget, caches, day, bundles)
    rows = replay_origin(state, day, bundles, truth, scored=day >= H.SCORED_START)
    expected_policies = {f'{route}/{policy}' for route in ROUTES for policy in POLICIES[route]}
    if set(rows) != expected_policies:
        raise ValueError('daily issuance omitted a registered policy')
    return rows


def _validate_state_calendar(state):
    if state.last_origin is None:
        if state.policies or state.ledgers:
            raise ValueError('empty execution state contains policy/learning history')
        return
    days = {str(d) for d in calendar(last=state.last_origin)}
    expected = {f'{route}/{policy}' for route in ROUTES for policy in POLICIES[route]}
    if set(state.policies) != expected or set(state.ledgers) != set(ROUTES) \
            or any(set(policy.records) != days for policy in state.policies.values()):
        raise ValueError('saved execution state omits a calendar date, policy or route')


def materialize_day(paths, state, day, caches, ensembles, manifest):
    """Rebuild an immutable daily ledger from saved issuances and cached natives.

    Native fit payloads are embedded so local recovery caches are never their
    sole durable copy. Outcome releases do not alter original daily issuance.
    """
    policies = {key: policy.records[str(day)] for key, policy in sorted(state.policies.items())}
    native = {}
    for route in ROUTES:
        row = policies[f'{route}/{POLICIES[route][0]}']
        index = pd.to_datetime(row['index'], utc=True)
        k = S.serving_search(day)
        ensemble = ensembles[route].get(str(k)) if k is not None else None
        native[route] = {}
        for member in RUN_MEMBERS[route]:
            result = caches[route].load(day, member, index, row['issue_utc'], ensemble=ensemble)
            if result is None:
                raise RuntimeError('saved issuance lacks its immutable native member cache')
            expected_hash = row['provenance']['member_cache_sha256'][member]
            if result['cache_content_sha256'] != expected_hash:
                raise ValueError('saved issuance and native cache provenance differ')
            native[route][member] = _plain(result)
    lineage = {route: {'fit_identity': caches[route].identity,
                      'input_vintages': {name: value for name, value in manifest['inputs'].items()
                                        if name.startswith(('data/', 'reports/development-inputs/', 'reports/weather-ablation/'))},
                      'code_identity': caches[route].identity['code'],
                      'target_day_embargo': str(day), 'feedback_last_permitted_day': str(day - timedelta(days=2)),
                      'issue_utc': policies[f'{route}/{POLICIES[route][0]}']['issue_utc']}
               for route in ROUTES}
    value = _signed({'schema': DAY_SCHEMA, 'day': str(day), 'protocol_identity': state.identity,
                     'operational_lineage': H.runtime_lineage(paths.root),
                     'protocol': state.identity,
                     'execution_manifest_sha256': manifest['content_sha256'],
                     'members': native, 'policies': policies, 'lineage': lineage})
    return _immutable(paths.day(day), value)


def materialize_outputs(paths, state, caches, ensembles, manifest, *, budget=None):
    """Recover missing exports even when the preceding process saved before export."""
    _validate_state_calendar(state)
    if state.last_origin is not None:
        for day in calendar(last=state.last_origin):
            path = paths.day(day)
            if not path.exists():
                value = materialize_day(paths, state, day, caches, ensembles, manifest)
            else:
                value = _read(path)
                if value.get('schema') != DAY_SCHEMA or value.get('day') != str(day) \
                        or value.get('protocol_identity') != state.identity \
                        or value.get('execution_manifest_sha256') != manifest['content_sha256'] \
                        or value.get('policies') != {key: policy.records[str(day)] for key, policy in sorted(state.policies.items())}:
                    raise ValueError('durable forecast ledger differs from original saved issuance')
            if budget is not None:
                record_evidence_once(budget, f"issuance/{digest(state.identity)}/{day}",
                                     {'ledger_sha256': value['content_sha256']}, {'completed_daily_issuances': 1})
    # These are state snapshots, not changes to any historical daily output.
    state.save(paths.output / 'replay-state.json')
    _save(paths.output / 'learning-ledger.json', _signed({'schema': 'cp26-learning-export-v1',
          'protocol_identity': state.identity, 'last_origin': str(state.last_origin) if state.last_origin else None,
          'routes': {route: ledger.to_dict() for route, ledger in sorted(state.ledgers.items())}}))


def run_forecasts(root, protocol_identity, frozen_keys, *, gate, through=H.NEW_BOUNDARY,
                  output_root=None, budget=None, contexts=None, caches=None, truth=None):
    """Resume the production calendar through a dated slice, with scores sealed.

    The required Lead gate checks bound authority, passed preparation/current
    parity and a sealed pilot projection at or below 388 hours. The CLI must
    enforce the same gate; this function never treats an environment flag as
    sufficient authority.
    """
    _gate(gate, 'forecast', fitting=True)
    if not isinstance(frozen_keys, dict):
        raise ValueError('independently verified frozen CP-25 scoring keys are required')
    calendar(last=through)
    paths = Paths.build(root, output_root)
    manifest = manifest_for(paths, protocol_identity)
    budget = budget or Budget()
    if contexts is None and caches is None:
        contexts, caches = contexts_and_caches(paths, protocol_identity)
    elif contexts is None or caches is None:
        raise ValueError('contexts and caches must be supplied together')
    _check_contexts(contexts, caches, protocol_identity)
    if through >= H.SCORED_START:
        for day in calendar(H.SCORED_START, through):
            for route in ROUTES:
                data = contexts[route].data
                rows = data.rows(day)
                scoring = rows[data.eligible[rows]]
                _keys_for(day, {'scoring_index': data.index[scoring]}, frozen_keys)
    ensembles = {route: {} for route in ROUTES}
    # Require every serving search needed by this slice, including those needed
    # to recover earlier durable output after a crash.
    for k in range(S.SEARCH_COUNT):
        if S.search_day(k) <= through:
            for route in ROUTES:
                ensembles[route][str(k)] = terminal_search(paths, route, k, manifest)
    save_path = paths.state(protocol_identity)
    durable = paths.output / 'replay-state.json'
    state = ReplayState.load(save_path, protocol_identity) if save_path.exists() else \
            ReplayState.load(durable, protocol_identity) if durable.exists() else ReplayState(protocol_identity)
    _validate_state_calendar(state)
    if state.last_origin is not None and state.last_origin > through:
        raise ValueError('requested slice cannot move behind persisted execution')
    truth = truth or truth_from_context(contexts['nominal'])
    materialize_outputs(paths, state, caches, ensembles, manifest, budget=budget)
    first = H.WARMUP_START if state.last_origin is None else state.last_origin + timedelta(days=1)
    if first <= through:
        for day in calendar(first, through):
            execute_day(state, day, contexts, caches, ensembles, frozen_keys, truth, budget=budget)
            state.save(save_path)
            value = materialize_day(paths, state, day, caches, ensembles, manifest)
            record_evidence_once(budget, f"issuance/{digest(protocol_identity)}/{day}",
                                 {'ledger_sha256': value['content_sha256']}, {'completed_daily_issuances': 1})
    materialize_outputs(paths, state, caches, ensembles, manifest, budget=budget)
    if state.last_origin == H.NEW_BOUNDARY:
        _immutable(paths.output / 'forecast-completion.json', _signed({
            'schema': 'cp26-forecast-completion-v1', 'status': 'COMPLETE', 'protocol': protocol_identity,
            'origin_count': len(calendar()), 'last_origin': str(H.NEW_BOUNDARY),
            'execution_manifest_sha256': manifest['content_sha256'],
            'replay_state_sha256': state.to_dict()['content_sha256']}))
    return {'schema': 'cp26-forecast-slice-v1', 'protocol_sha256': digest(protocol_identity),
            'last_origin': str(state.last_origin), 'calendar_dates': len(calendar(last=through)),
            'forecast_ledger': str(paths.output / 'forecast-ledger'),
            'replay_state': str(paths.output / 'replay-state.json'),
            'learning_ledger': str(paths.output / 'learning-ledger.json'), 'evaluation_scores_computed': False}
