"""CP-26 own-fit cache. No CP-24 forecast cache is accepted here.

An entry binds the frozen input/protocol/code identity, route, origin, native
keys and frozen search ensemble. Failed entries are retained too; loading never
selects a substitute seed or silently refits under changed identity.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from cp15.data import array_hash, sha
from . import harness as H
from .state import _instant, _save, digest, quantiles, timestamps

SCHEMA = 'cp26-member-cache-v1'


def frozen_fit_identity(ctx, protocol_identity):
    """Call after protocol/pins are committed; explicit document identity is required."""
    if not isinstance(protocol_identity, dict) or not protocol_identity:
        raise ValueError('a frozen active protocol identity is required')
    root = Path(ctx.root)
    code = {str(path.relative_to(root)): sha(path) for path in sorted((root / 'src/cp26').glob('*.py'))}
    return {'protocol': protocol_identity, 'inputs': H.input_fingerprint(root), 'code': code,
            'route': ctx.variant, 'settings': {'name': ctx.settings.name, 'boundary': str(ctx.settings.boundary),
                                             'weather': ctx.settings.weather, 'issue_label': ctx.settings.issue_label},
            'restore': bool(ctx.restore)}


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _payload_hash(item):
    return digest({k: v for k, v in item.items() if k != 'content_sha256'})


class FitCache:
    def __init__(self, base, route, identity):
        if route not in ('nominal', 'noload', 'gfs6') or not identity:
            raise ValueError('invalid fit-cache route or identity')
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        self.route = route
        self.base = Path(base) / digest(self.identity) / route

    def path(self, day, member):
        if member not in ('B0', 'A1', 'B2', 'A1_w', 'B2_w', 'L-N', 'L-R', 'D2', 'v1-R'):
            raise ValueError('unknown CP-26 member')
        return self.base / str(day) / f'{member}.json'

    def load(self, day, member, index, issue_utc, *, ensemble=None):
        path = self.path(day, member)
        if not path.exists():
            return None
        item = json.loads(path.read_text())
        keys = timestamps(day, index)
        expected = {'schema': SCHEMA, 'route': self.route, 'day': str(day), 'member': member,
                    'identity': self.identity, 'issue_utc': _instant(day, issue_utc),
                    'index': [t.isoformat() for t in keys], 'ensemble_sha256': digest(ensemble) if member == 'D2' else None}
        if any(item.get(k) != v for k, v in expected.items()) or item.get('content_sha256') != _payload_hash(item):
            raise ValueError('stale/wrong member cache identity or content')
        result = item['result']
        if result['status'] == 'ok':
            if member == 'D2':
                from .search import serving_search, validate_ensemble
                validate_ensemble(ensemble, serving_search(day), self.route)
            central = np.asarray(result['central'], float)
            if central.shape != (len(keys),) or not np.isfinite(central).all() or array_hash(central) != item['central_sha256']:
                raise ValueError('cached native central is invalid')
            result['central'] = central
            if member in ('D2', 'v1-R') and 'quantiles' not in result:
                raise ValueError('native D2/v1-R cache requires quantiles')
            if 'quantiles' in result:
                result['quantiles'] = quantiles(result['quantiles'], len(keys))
                if array_hash(result['quantiles']) != item['quantiles_sha256']:
                    raise ValueError('cached quantiles differ from their hash')
                if member == 'D2' and not np.array_equal(central, result['quantiles'][:, 3]):
                    raise ValueError('cached D2 p50 differs from native central')
        elif result['status'] != 'failed':
            raise ValueError('unknown cached member status')
        result['cache_content_sha256'] = item['content_sha256']
        return result

    def save(self, day, member, index, issue_utc, result, *, ensemble=None):
        path = self.path(day, member)
        if path.exists():
            raise FileExistsError('retain member cache; no silent overwrite or retry')
        index = timestamps(day, index)
        result = _plain(result)
        if result.get('status') not in ('ok', 'failed'):
            raise ValueError('cache requires an explicit fit status')
        central_hash, quantile_hash = None, None
        if result['status'] == 'ok':
            if member == 'D2':
                from .search import serving_search, validate_ensemble
                validate_ensemble(ensemble, serving_search(day), self.route)
            central = np.asarray(result['central'], float)
            if central.shape != (len(index),) or not np.isfinite(central).all():
                raise ValueError('invalid native fit central')
            central_hash = array_hash(central)
            if member in ('D2', 'v1-R') and 'quantiles' not in result:
                raise ValueError('native D2/v1-R cache requires quantiles')
            if 'quantiles' in result:
                q = quantiles(result['quantiles'], len(index))
                quantile_hash = array_hash(q)
                if member == 'D2' and not np.array_equal(central, q[:, 3]):
                    raise ValueError('D2 p50 differs from native central')
        item = {'schema': SCHEMA, 'route': self.route, 'day': str(day), 'member': member, 'identity': self.identity,
                'issue_utc': _instant(day, issue_utc), 'index': [t.isoformat() for t in index],
                'ensemble_sha256': digest(ensemble) if member == 'D2' else None,
                'central_sha256': central_hash, 'quantiles_sha256': quantile_hash, 'result': result}
        item['content_sha256'] = _payload_hash(item)
        _save(path, item)
        return item['content_sha256']


def fit_members_cached(ctx, day, cache, *, ensemble=None, k=None, members=None, charge=None):
    """Run only inside the resource monitor. The caller owns compute authority.

    This is a resumable fit primitive, not an experiment runner. It does not
    perform searches, pilot fits, replay, scores, or CP-28 arms by itself.
    """
    from .members import RUN_MEMBERS, fit_members
    if not os.environ.get('CP26_MONITORED_JOB'):
        raise RuntimeError('CP-26 fits require the cumulative resource monitor')
    if os.environ.get('CP26_EXPERIMENT_AUTHORIZED') != '1':
        raise RuntimeError('full CP-26 experiment requires a new authorized continuation brief')
    if cache.route != ctx.variant:
        raise ValueError('fit cache and context routes differ')
    if cache.identity != frozen_fit_identity(ctx, cache.identity['protocol']):
        raise ValueError('fit cache identity differs from current frozen input/protocol/code')
    index = ctx.data.index[ctx.data.rows(day)]
    instant = H.issue_instant(day, ctx.settings)
    out = {}
    for member in members or RUN_MEMBERS[ctx.variant]:
        result = cache.load(day, member, index, instant, ensemble=ensemble)
        if result is None:
            result = fit_members(ctx, day, (member,), ensemble=ensemble, k=k)[member]
            cache.save(day, member, index, instant, result, ensemble=ensemble)
            if charge is not None:
                charge(member, result)
            result = cache.load(day, member, index, instant, ensemble=ensemble)
        out[member] = result
    return out
