"""CP-26 chronological issuance replay, with no fitting or scoring side effect.

The caller supplies identity-verified native member fits. State survives every
calendar and search transition. Learning rows use original c_HG/L/D2 members;
H history and a scored operational fallback do not decide learning validity.
"""
from __future__ import annotations

from datetime import date

from cp21.lgbm import hg_central, hgl_central
from cp24.execution import v5_central
from .state import ReplayState, _vector, complete, digest, timestamps

POLICIES = {'nominal': ('B0', 'v1-R', 'v2', 'v3', 'v4', 'v5', 'D2'),
            'noload': ('v3', 'v4', 'v5'), 'gfs6': ('v3', 'v4', 'v5')}
LAYERS = {'B0': 'B0', 'v1-R': 'direct', 'D2': 'direct'}


def _member(members, name, n):
    item = members.get(name)
    if item is None or item.get('status') != 'ok':
        raise ValueError(f'native member {name} missing or failed')
    return _vector(item['central'], n, f'native member {name}')


def central_for(policy, members, n):
    """Use the landed arithmetic and fixed weights, with no subset renormalization."""
    get = lambda name: _member(members, name, n)
    if policy in ('B0', 'v1-R', 'D2'):
        return get(policy)
    if policy == 'v2':
        return hg_central(get('A1'), get('B2'))
    if policy == 'v3':
        return hg_central(get('A1_w'), get('B2_w'))
    if policy == 'v4':
        return hgl_central(get('A1_w'), get('B2_w'), get('L-N'), get('L-R'))
    if policy == 'v5':
        return v5_central(get('A1_w'), get('B2_w'), get('L-N'), get('L-R'), get('D2'))
    raise ValueError('unknown CP-26 policy')


def _learning_members(members, n):
    return {'c_HG': central_for('v3', members, n),
            'L': _member(members, 'L-N', n) / 2 + _member(members, 'L-R', n) / 2,
            'D2': _member(members, 'D2', n)}


def replay_origin(state: ReplayState, day: date, bundles: dict, truth, *, scored=False):
    """Release through D−2, then issue every route on its frozen scored keys.

    Each bundle supplies ``index``, ``members``, ``scale``, ``issue_utc`` and
    ``identity``; optional ``scoring_index`` is independent of complete native
    issuance validity. No target-day or D−1 outcome is requested.
    """
    if not bundles or any(route not in POLICIES for route in bundles):
        raise ValueError('replay requires a registered CP-26 route')
    for route in bundles:
        for policy in POLICIES[route]:
            state.policy(route, policy, LAYERS.get(policy, 'H'))
        state.learning(route)
    state.release(day, truth)
    result = {}
    fallback = None
    for route in sorted(bundles, key=lambda item: (item != 'nominal', item)):
        bundle = bundles[route]
        index = timestamps(day, bundle['index'])
        score_index = bundle.get('scoring_index', index)
        members = bundle['members']
        identity_valid = bundle.get('identity') == state.identity
        try:
            learning = _learning_members(members, len(index)) if identity_valid and complete(day, index) else None
        except (TypeError, ValueError, KeyError):
            learning = None
        provenance = {'member_cache_sha256': {k: v.get('cache_content_sha256') for k, v in members.items()},
                      'search': bundle.get('search'), 'ensemble_sha256': bundle.get('ensemble_sha256'),
                      'learning_eligible': learning is not None, 'learning_support': state.learning(route).support(day)}
        for policy in POLICIES[route]:
            try:
                native = central_for(policy, members, len(index))
                cause = None
                native_q = members[policy].get('quantiles') if policy in ('v1-R', 'D2') else None
            except (KeyError, TypeError, ValueError) as exc:
                native, native_q, cause = None, None, str(exc)
            row = state.policy(route, policy, LAYERS.get(policy, 'H')).issue(
                day, index, native, issue_utc=bundle['issue_utc'], scale=bundle.get('scale'),
                native_quantiles=native_q, scoring_index=score_index, scored=scored,
                fallback=fallback, central_failure=cause, identity_valid=identity_valid,
                provenance=provenance)
            result[f'{route}/{policy}'] = row
            if route == 'nominal' and policy == 'B0' and row['native_quantiles'] is not None:
                fallback = {'index': row['index'], 'quantiles': row['native_quantiles']}
        # Eligibility comes only from three original complete member vectors.
        # Missing scale prevents H intake, but does not erase a native learning row.
        if learning is not None:
            try:
                try:
                    scale = _vector(bundle.get('scale'), len(index), 'learning scale', scale=True)
                except (ValueError, TypeError):
                    scale = None
                provenance['scale_available'] = scale is not None
                state.learning(route).issue(day, index, learning, issue_utc=bundle['issue_utc'],
                                            scale=scale, provenance=provenance)
            except (TypeError, ValueError, KeyError):
                # Failed or partial original member day stays absent in learning.
                pass
    return result


def bundles_from_fits(contexts, caches, ensembles, day, *, charge=None):
    """Budgeted future-run fit orchestration. Does not perform a search.

    All ensembles are CP-26's frozen route-specific search results. Before the
    first search or after a failed search D2 fails closed through members.py.
    Nominal B0 is the fallback source for every registered route.
    """
    from .fits import fit_members_cached
    from .harness import issue_instant
    from .search import serving_search
    bundles = {}
    for route, ctx in contexts.items():
        k = serving_search(day)
        ensemble = ensembles.get(route, {}).get(str(k)) if k is not None else None
        member_fits = fit_members_cached(ctx, day, caches[route], ensemble=ensemble, k=k, charge=charge)
        rows = ctx.data.rows(day)
        scoring = rows[ctx.data.eligible[rows]]
        bundles[route] = {'index': ctx.data.index[rows], 'scoring_index': ctx.data.index[scoring],
                          'members': member_fits, 'scale': ctx.data.scale[rows],
                          'issue_utc': issue_instant(day, ctx.settings), 'identity': caches[route].identity['protocol'],
                          'search': k, 'ensemble_sha256': digest(ensemble) if ensemble is not None else None}
    return bundles


def replay_days(state, days, source, truth, *, score_start, save_path=None):
    """Explicit future-run primitive; no invocation occurs in the repair task.

    Resume with the next unissued origin and the complete saved ReplayState.
    No fresh state is created at score_start, search changes or member failures.
    """
    for day in days:
        if state.last_origin is not None and day <= state.last_origin:
            raise ValueError('resume must start after the last persisted origin')
        records = replay_origin(state, day, source(day), truth, scored=day >= score_start)
        if save_path is not None:
            state.save(save_path)
        yield records
