"""CP-26's support-only batch freeze and training-only DDNN-2 search.

The landed sampler, space, successive halving, metric and ties are unchanged. The r14
repair moves S_0 to 2020-01-15 and admits a batch only if *every* landed whole-week
holdout choice meets its 250/14 row guards. This admission never fits a model, reads
a loss, or chooses a seed. Search and daily-fit failures never extend an older ensemble.

Actual fits require the cumulative monitor and a subsequent experiment authorization;
the repair task may use the support functions and synthetic evaluator only.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from cp24 import design as G
from cp24 import sampler as SP

SEARCH_START = date(2020, 1, 15)
SEARCH_SPACING_DAYS = 182
SEARCH_COUNT = 14
EARLIEST_BATCH_START = date(2019, 7, 1)
TRIALS = 128
ROUTES = ('nominal', 'noload', 'gfs6')
SEED_BASE = 26_000_000
SEED_SEARCH_STRIDE = 100_000
SEED_TRIAL_STRIDE = 100
SEED_MEMBER_OFFSET = 90_000
SEED_RANK_STRIDE = 10
PROTOCOL = 'cp26-ddnn2-search-v21-r14'
MODEL_CODE = ('src/cp26/search.py', 'src/cp26/harness.py', 'src/cp26/members.py', 'src/cp24/design.py',
              'src/cp24/sampler.py', 'src/cp24/member.py', 'src/cp24/ddnn2.py')


class SearchUnavailable(ValueError):
    """The serving search has no complete native CP-26 ensemble."""


def digest(item: dict) -> str:
    return hashlib.sha256(json.dumps({k: v for k, v in item.items() if k != 'content_sha256'},
                                    sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _atomic(path: Path, item: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(item, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def _signed(item: dict) -> dict:
    return {**item, 'content_sha256': digest(item)}


def _route(route: str) -> str:
    if route not in ROUTES:
        raise ValueError(f'unknown CP-26 route: {route}')
    return route


def search_day(k: int) -> date:
    """The protocol continues at k >= 14; this function reads no data."""
    if not isinstance(k, int) or k < 0:
        raise ValueError('search index must be a nonnegative integer')
    return SEARCH_START + timedelta(days=SEARCH_SPACING_DAYS * k)


def serving_search(day: date) -> int | None:
    """No native D2 before S_0. A failed search still owns its full service period."""
    days = (day - SEARCH_START).days
    return None if days < 0 else days // SEARCH_SPACING_DAYS


def candidate_blocks(k: int) -> list[tuple[date, date]]:
    blocks = []
    end = search_day(k) - timedelta(days=1)
    while (start := end - timedelta(days=SP.BATCH_DAYS - 1)) >= EARLIEST_BATCH_START:
        blocks.append((start, end))
        end = start - timedelta(days=1)
    return blocks


def week_pool(lower: date, origin: date) -> list[date]:
    """Exactly cp24.design.holdout_weeks' complete Monday-Sunday pool, excluding D-7..D-1."""
    end = origin - timedelta(days=G.RECENT_EXCLUDED_DAYS)
    monday = lower + timedelta(days=(7 - lower.weekday()) % 7)
    pool = []
    while monday + timedelta(days=6) < end:
        pool.append(monday)
        monday += timedelta(days=7)
    return pool


def window_support(dd: G.DayData, origin: date) -> dict:
    """All-choice extrema under the *landed* support mask, without trial-seed filtering.

    For N supported days and m held weeks, min(train) = N - sum(m heaviest
    weeks), and min(hold) = sum(m lightest weeks). The most recent seven days
    remain trainable because they are outside the whole-week holdout pool.
    """
    lower, ix, excluded = G.window(dd, origin, exclude_uncovered=True)
    weeks = week_pool(lower, origin)
    m = max(1, int(math.floor(G.HOLDOUT_SHARE * len(weeks) + 0.5))) if weeks else 0
    supported = dd.days[ix]
    weights = [int(((supported >= np.datetime64(w, 'D'))
                    & (supported < np.datetime64(w, 'D') + 7)).sum()) for w in weeks]
    ordered = sorted(weights)
    lightest = sum(ordered[:m]) if m else 0
    heaviest = sum(ordered[-m:]) if m else 0
    minimum_train = int(len(ix) - heaviest)
    minimum_hold = int(lightest)
    support_sha = hashlib.sha256(np.ascontiguousarray(supported.astype('int64')).tobytes()).hexdigest()
    return {'first': str(origin), 'window_first': str(lower), 'support_days_before_holdout': int(len(ix)),
            'support_days_sha256': support_sha, 'excluded_uncovered_days': len(excluded),
            'whole_week_pool': len(weeks), 'selected_heldout_weeks': m,
            'week_support': [{'first': str(w), 'days': n} for w, n in zip(weeks, weights)],
            'week_support_distribution': {str(n): weights.count(n) for n in sorted(set(weights))},
            'minimum_possible_train_days': minimum_train,
            'maximum_possible_train_days': int(len(ix) - lightest),
            'minimum_possible_hold_days': minimum_hold,
            'maximum_possible_hold_days': int(heaviest),
            'every_possible_split_sufficient': bool(weeks and minimum_train >= G.MIN_TRAIN_DAYS
                                                   and minimum_hold >= G.MIN_HOLD_DAYS)}


def batch_support(dd: G.DayData, k: int) -> dict:
    """Retain up to eleven newest eligible blocks. Rejections remain auditable."""
    candidates, kept = [], []
    for tile, (first, last) in enumerate(candidate_blocks(k)):
        item = {**window_support(dd, first), 'tile': tile, 'last': str(last)}
        eligible = item['every_possible_split_sufficient']
        item['retained'] = bool(eligible and len(kept) < SP.MAX_BATCHES)
        if item['retained']:
            item['batch'] = len(kept)
            kept.append(item)
        candidates.append(item)
    return _signed({'schema': 'cp26-support-freeze-v2', 'protocol': PROTOCOL, 'k': k,
                    'search_day': str(search_day(k)), 'service_last': str(search_day(k + 1) - timedelta(days=1)),
                    'B': len(kept), 'minimum_train_days': G.MIN_TRAIN_DAYS,
                    'minimum_hold_days': G.MIN_HOLD_DAYS, 'batches': kept, 'candidates': candidates,
                    'admission': 'support-only: all possible landed whole-week splits must meet 250/14; '
                                 'newest up to 11; independent of seed, configuration, loss and convergence'})


def sampler_seed(k: int) -> np.random.SeedSequence:
    search_day(k)
    return np.random.SeedSequence([26, k, 0x5A3])


def trial_fit_seed(k: int, trial: int, batch: int) -> int:
    search_day(k)
    if not 0 <= trial < TRIALS or not 0 <= batch < SP.MAX_BATCHES:
        raise ValueError('trial/batch outside frozen seed domains')
    return SEED_BASE + k * SEED_SEARCH_STRIDE + trial * SEED_TRIAL_STRIDE + batch


fit_seed = trial_fit_seed


def member_seeds(k: int) -> list[int]:
    search_day(k)
    return [SEED_BASE + k * SEED_SEARCH_STRIDE + SEED_MEMBER_OFFSET + SEED_RANK_STRIDE * rank + repeat
            for rank in range(1, SP.ENSEMBLE_CONFIGS + 1) for repeat in range(1, SP.ENSEMBLE_SEEDS + 1)]


def effective_config(config: dict, route: str) -> dict:
    """The effective settings used in fits; no-load ignores removed flags and never id."""
    _route(route)
    out = {k: v for k, v in config.items() if k != 'id'}
    if route == 'noload':
        out['groups'] = [g for g in out['groups'] if g not in ('load_d1', 'load_d7')]
    return out


def effective_key(config: dict, route: str) -> str:
    return json.dumps(effective_config(config, route), sort_keys=True, separators=(',', ':'))


def trials(k: int, route: str, space: dict | None = None) -> list[dict]:
    """Exactly 128 draws, in the landed draw order; duplicate effective settings stay drawn."""
    _route(route)
    rng = np.random.Generator(np.random.PCG64(sampler_seed(k)))
    sampled = [SP.sample(rng, space or SP.SPACE_ROUND_1, f'cp26-k{k:02d}-t{t:03d}') for t in range(TRIALS)]
    return [{**effective_config(c, route), 'id': c['id']} for c in sampled]


def make_plan(dd: G.DayData, k: int, route: str, identity: dict) -> dict:
    """Freeze and sign support before drawing trials; fits consume the signed bytes."""
    support = batch_support(dd, k)
    return _signed({'schema': 'cp26-search-plan-v2', 'protocol': PROTOCOL, 'k': k, 'route': _route(route),
                    'search_day': str(search_day(k)), 'identity': identity, 'support': support,
                    'B': support['B'], 'space': SP.SPACE_ROUND_1, 'trials': trials(k, route),
                    'sampler_seed': [26, k, 0x5A3],
                    'trial_seed_formula': '26_000_000 + k*100_000 + trial*100 + batch; trial=0..127,batch=0..10',
                    'member_seed_formula': '26_000_000 + k*100_000 + 90_000 + 10*rank + repeat; '
                                           'rank=1..4,repeat=1..2',
                    'member_seeds': member_seeds(k)})


def validate_plan(plan: dict) -> None:
    k, route = plan['k'], _route(plan['route'])
    support = plan['support']
    if plan.get('schema') != 'cp26-search-plan-v2' or plan.get('protocol') != PROTOCOL \
            or plan.get('content_sha256') != digest(plan) \
            or support.get('content_sha256') != digest(support) \
            or support['k'] != k or support['B'] != plan['B'] \
            or support['search_day'] != str(search_day(k)) or plan['search_day'] != str(search_day(k)) \
            or plan['space'] != SP.SPACE_ROUND_1 or plan['trials'] != trials(k, route) \
            or plan['member_seeds'] != member_seeds(k):
        raise ValueError('search plan signature, protocol or fixed sampler mismatch')
    if plan['B'] != len(support['batches']) or not 0 <= plan['B'] <= SP.MAX_BATCHES:
        raise ValueError('search plan has an invalid batch count')
    if any(not b['every_possible_split_sufficient'] for b in support['batches']):
        raise ValueError('search plan includes an unsupported batch')


def _pooled(items: list[dict], batch_ids: list[int]) -> tuple[float, float]:
    by = {item['batch']: item for item in items}
    if any(b not in by or by[b]['status'] != 'ok' or by[b].get('pinball_count', 0) <= 0
           or by[b].get('hours', 0) <= 0 for b in batch_ids):
        return math.inf, math.inf
    if not batch_ids:
        return math.inf, math.inf
    pin = sum(by[b]['pinball_sum'] for b in batch_ids) / sum(by[b]['pinball_count'] for b in batch_ids)
    mae = sum(by[b]['abs_error_sum'] for b in batch_ids) / sum(by[b]['hours'] for b in batch_ids)
    return (pin, mae) if math.isfinite(pin) and math.isfinite(mae) else (math.inf, math.inf)


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


class TrialCache:
    """Immutable task records, keyed by signed plan/config/seed/batch lineage."""
    def __init__(self, root: Path, plan: dict):
        validate_plan(plan)
        self.root, self.plan = Path(root), plan

    def path(self, trial: int, batch: int) -> Path:
        return self.root / f't{trial:03d}-b{batch:02d}.json'

    def key(self, trial: int, batch: int) -> dict:
        b = self.plan['support']['batches'][batch]
        return {'protocol': PROTOCOL, 'plan_sha256': self.plan['content_sha256'], 'k': self.plan['k'],
                'route': self.plan['route'], 'trial': trial, 'batch': batch,
                'batch_first': b['first'], 'batch_last': b['last'],
                'config': self.plan['trials'][trial], 'seed': trial_fit_seed(self.plan['k'], trial, batch),
                'identity': self.plan['identity']}

    def load(self, trial: int, batch: int) -> dict | None:
        path = self.path(trial, batch)
        if not path.exists():
            return None
        item = json.loads(path.read_text())
        if item.get('content_sha256') != digest(item) or item.get('key') != self.key(trial, batch):
            raise ValueError(f'stale or mismatched CP-26 search cache: {path.name}')
        if item.get('status') not in ('ok', 'failed'):
            raise ValueError('invalid cached search status')
        return item

    def save(self, trial: int, batch: int, result: dict) -> dict:
        if self.path(trial, batch).exists():
            raise FileExistsError('CP-26 task evidence is immutable; resume by verified reuse')
        item = _signed({**result, 'key': self.key(trial, batch), 'batch': batch})
        _atomic(self.path(trial, batch), item)
        return item


def run_search(plan: dict, evaluate, *, cache: TrialCache | None = None) -> dict:
    """Operational halving engine; evaluate(trial,batch,config,seed) performs one fit.

    Synthetic tests supply an evaluator with no fit. Production uses execute_search,
    which checks experiment authorization and the resource monitor. A resumed run
    reuses failed records too; it never adds trials or changes a seed.
    """
    validate_plan(plan)
    k, route, nb, configs = plan['k'], plan['route'], plan['B'], plan['trials']
    records = {t: [] for t in range(TRIALS)}
    order = [c['id'] for c in configs]

    def task(t, b):
        item = cache.load(t, b) if cache else None
        if item is None:
            began = time.perf_counter()
            try:
                item = dict(evaluate(t, b, configs[t], trial_fit_seed(k, t, b)))
                item['batch'] = b
                if item.get('status') != 'ok':
                    item['status'] = 'failed'
                elif not math.isfinite(_pooled([item], [b])[0]):
                    item = {'status': 'failed', 'batch': b, 'cause': 'invalid/nonfinite search metric'}
            except Exception as exc:
                from cp25.guard import redact
                item = {'status': 'failed', 'batch': b, 'cause': redact(f'{type(exc).__name__}: {exc}'),
                        'seconds': time.perf_counter() - began}
            if cache:
                item = cache.save(t, b, item)
        records[t].append(item)

    first = list(range(min(SP.HALVING_BATCHES, nb)))
    if nb:
        for t in range(TRIALS):
            for b in first:
                task(t, b)
    metric = {configs[t]['id']: _pooled(records[t], first)[0] for t in range(TRIALS)}
    sizes = {configs[t]['id']: SP.n_params(configs[t], next((r['n_inputs'] for r in records[t]
              if r['status'] == 'ok' and r.get('n_inputs') is not None), 10**6)) for t in range(TRIALS)}
    ranked_a = SP.rank(metric, sizes, order)
    survivors = (ranked_a[:SP.halving_survivors(TRIALS)] if nb > SP.HALVING_BATCHES else ranked_a) if nb else []
    for cid in survivors:
        t = order.index(cid)
        for b in range(SP.HALVING_BATCHES, nb):
            task(t, b)
    allb = list(range(nb))
    eligible = {configs[t]['id']: _pooled(records[t], allb)[0] for t in range(TRIALS)
                if configs[t]['id'] in survivors and len(records[t]) == nb}
    ranked = [cid for cid in SP.rank(eligible, sizes, order) if math.isfinite(eligible[cid])]
    distinct, chosen = set(), []
    for cid in ranked:
        config = configs[order.index(cid)]
        key = effective_key(config, route)
        if key not in distinct:
            chosen.append(cid)
            distinct.add(key)
        if len(chosen) == SP.ENSEMBLE_CONFIGS:
            break
    complete = len(chosen) == SP.ENSEMBLE_CONFIGS
    members = []
    if complete:
        for rank, cid in enumerate(chosen, 1):
            t = order.index(cid)
            pin, mae = _pooled(records[t], allb)
            for repeat in range(1, SP.ENSEMBLE_SEEDS + 1):
                members.append({'rank': rank, 'repeat': repeat, 'config': configs[t],
                                'effective_config_sha256': hashlib.sha256(effective_key(configs[t], route).encode()).hexdigest(),
                                'seed': member_seeds(k)[2 * (rank - 1) + repeat - 1],
                                'validation_pinball_all_batches': pin, 'validation_mae_all_batches': mae})
    rows = []
    for t, config in enumerate(configs):
        pin, mae = _pooled(records[t], allb)
        rows.append({'trial': t, 'config': config, 'n_parameters': sizes[config['id']],
                     'ran_all_batches': bool(nb and len(records[t]) == nb),
                     'pinball_first_batches': _finite(_pooled(records[t], first)[0]),
                     'pinball_all_batches': _finite(pin), 'mae_all_batches': _finite(mae),
                     'batches': records[t]})
    return _signed({'schema': 'cp26-search-result-v2', 'protocol': PROTOCOL, 'k': k, 'route': route,
                    'search_day': str(search_day(k)), 'service_last': str(search_day(k + 1) - timedelta(days=1)),
                    'plan_sha256': plan['content_sha256'], 'identity': plan['identity'], 'B': nb,
                    'complete': complete, 'status': 'available' if complete else 'unavailable',
                    'failure_reason': None if complete else ('zero_eligible_batches' if not nb
                                                            else 'fewer_than_four_valid_effective_configurations'),
                    'survivors': survivors, 'ranking_all_batches': ranked, 'chosen': chosen,
                    'members': members, 'trials': rows,
                    'failed_fits': sum(r['status'] == 'failed' for rs in records.values() for r in rs),
                    'next_search_day': str(search_day(k + 1)), 'older_ensemble_extended': False})


def validate_ensemble(ensemble: dict | None, k: int | None, route: str | None = None) -> None:
    if k is None or ensemble is None or ensemble.get('schema') != 'cp26-search-result-v2' \
            or ensemble.get('protocol') != PROTOCOL or ensemble.get('k') != k \
            or not ensemble.get('complete') or ensemble.get('status') != 'available' \
            or ensemble.get('search_day') != str(search_day(k)) \
            or ensemble.get('content_sha256') != digest(ensemble):
        raise SearchUnavailable(f'search k={k} has no verified native CP-26 ensemble')
    actual_route = _route(ensemble['route'])
    if route is not None and actual_route != _route(route):
        raise SearchUnavailable('CP-26 ensemble belongs to a different route')
    members = ensemble.get('members', [])
    if len(members) != 8 or [m.get('seed') for m in members] != member_seeds(k):
        raise SearchUnavailable('D2 requires exactly eight frozen members; no subset median')
    keys = []
    for rank in range(1, 5):
        pair = members[2 * (rank - 1):2 * rank]
        if [m.get('rank') for m in pair] != [rank, rank] \
                or [m.get('repeat') for m in pair] != [1, 2] \
                or effective_key(pair[0]['config'], actual_route) != effective_key(pair[1]['config'], actual_route):
            raise SearchUnavailable('each ensemble rank requires two seeds of one effective configuration')
        keys.append(effective_key(pair[0]['config'], actual_route))
    if len(set(keys)) != 4:
        raise SearchUnavailable('D2 requires four distinct effective configurations')


def load_search_for_origin(base: Path, day: date, route: str, identity: dict | None = None) -> dict:
    k = serving_search(day)
    if k is None:
        raise SearchUnavailable(f'no CP-26 search serves {day} before {SEARCH_START}')
    path = Path(base) / _route(route) / f'k{k:02d}' / 'result.json'
    if not path.exists():
        raise SearchUnavailable(f'serving search k={k} missing; no extension of an earlier ensemble')
    item = json.loads(path.read_text())
    validate_ensemble(item, k, route)
    if identity is not None and item.get('identity') != identity:
        raise SearchUnavailable('serving search identity differs from the fit-cache lineage')
    return item


def search_identity(root: Path, k: int, route: str) -> dict:
    from . import harness as H
    from .gates import RUN_ID
    return {'protocol': PROTOCOL, 'run_id': RUN_ID, 'k': k, 'route': _route(route),
            'cp26_input_fingerprint': H.input_fingerprint(root),
            'code_sha256': {name: H.recipe_code_sha256(root, name) for name in MODEL_CODE}}


def prepare_search(root: Path, k: int, route: str, path: Path) -> dict:
    """Support-only signed preparation; safe before fit authorization, under the monitor."""
    from . import harness as H
    if not os.environ.get('CP26_MONITORED_JOB'):
        raise RuntimeError('search preparation requires the cumulative CP-26 monitor')
    if not 0 <= k < SEARCH_COUNT:
        raise ValueError('CP-26 only prepares k=0..13; continuation dates are protocol-only')
    settings = H.GFS6 if _route(route) == 'gfs6' else H.NEW
    cutoff = search_day(k)
    data = H.load(root, settings, before=cutoff)
    dd = G.build(data, H.weather_design(root, settings))
    plan = make_plan(dd, k, route, search_identity(root, k, route))
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text())
        validate_plan(existing)
        if existing != plan:
            raise ValueError('signed search support/plan changed; refuse to overwrite')
        return existing
    _atomic(path, plan)
    return plan


def _require_fit_authorization() -> None:
    if not os.environ.get('CP26_MONITORED_JOB') or os.environ.get('CP26_EXPERIMENT_AUTHORIZED') != '1':
        raise RuntimeError('DDNN-2 fitting needs the cumulative monitor and the subsequent authorized experiment brief')


def execute_search(root: Path, plan_path: Path, cache_root: Path, result_path: Path) -> dict:
    """Future experiment entry point. The repair never calls this function."""
    _require_fit_authorization()
    from . import harness as H
    from .members import Context
    from .budget import Budget
    from cp24 import ddnn2 as M
    from cp24.member import emit_rows, fit_member, pinball_rows
    root, plan_path, result_path = Path(root), Path(plan_path), Path(result_path)
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if plan['identity'] != search_identity(root, plan['k'], plan['route']):
        raise ValueError('signed search plan differs from current lineage')
    # Freezing before fits is a committed artifact, not an after-the-fact signature.
    rel = str(plan_path.relative_to(root))
    if subprocess.check_output(['git', 'show', f'HEAD:{rel}'], cwd=root) != plan_path.read_bytes():
        raise ValueError('support plan and sampled trials must be committed before fitting')
    if result_path.exists():
        item = json.loads(result_path.read_text())
        if item.get('plan_sha256') != plan['content_sha256'] or item.get('content_sha256') != digest(item) \
                or item.get('identity') != plan['identity'] or item.get('k') != plan['k'] \
                or item.get('route') != plan['route'] or item.get('schema') != 'cp26-search-result-v2' \
                or item.get('protocol') != PROTOCOL or item.get('status') not in ('available', 'unavailable'):
            raise ValueError('existing search result differs from the signed plan')
        if item['status'] == 'available':
            validate_ensemble(item, plan['k'], plan['route'])
        elif item.get('complete') or item.get('members') or item.get('failure_reason') not in (
                'zero_eligible_batches', 'fewer_than_four_valid_effective_configurations'):
            raise ValueError('existing failed search result is inconsistent')
        return item
    settings = H.GFS6 if plan['route'] == 'gfs6' else H.NEW
    ctx = Context.build(root, settings, plan['route'], before=search_day(plan['k']))
    if batch_support(ctx.dd, plan['k']) != plan['support']:
        raise ValueError('actual support differs from frozen batch support')
    budget = Budget()

    def evaluate(t, b, config, seed):
        first = date.fromisoformat(plan['support']['batches'][b]['first'])
        days = np.asarray([ctx.dd.ix(first + timedelta(days=i)) for i in range(SP.BATCH_DAYS)])
        with H.ddnn2_without_load(ctx.noload_active):
            fit = fit_member(ctx.dd, first, config, seed, days, exclude_uncovered=True,
                             charge=lambda: budget.count(ddnn2_search_fits=1))
        budget.count(ddnn2_epochs=int(fit['member']['epochs_run']))
        rows = ctx.keys.of_days(days)
        rows = rows[ctx.keys.eligible[rows]]
        q = emit_rows(ctx.keys, rows, days, fit['eur'])
        y = ctx.keys.y[rows]
        pin = pinball_rows(q, y)
        lower, window, _ = G.window(ctx.dd, first, exclude_uncovered=True)
        train, hold = G.split(ctx.dd, window,
                             [date.fromisoformat(w) for w in fit['window']['held_out_weeks']])
        return {'status': 'ok', 'pinball_sum': float(pin.sum()), 'pinball_count': int(pin.size),
                'abs_error_sum': float(np.abs(q[:, M.MEDIAN] - y).sum()), 'hours': int(len(rows)),
                'n_inputs': fit['n_inputs'], 'n_parameters': SP.n_params(config, fit['n_inputs']),
                'seconds': fit['seconds'], 'maxrss_bytes': fit['maxrss_bytes'],
                'window': fit['window'], 'preprocessor': fit['preprocessor'],
                'member': fit['member'], 'guards': fit['guards'], 'seed': seed,
                'origin': str(first), 'issue_utc': H.issue_instant(first, settings).isoformat(),
                'training_target_cutoff': str(first),
                'latest_training_delivery_day': str(ctx.dd.date_of(int(train[-1]))) if len(train) else None,
                'latest_holdout_delivery_day': str(ctx.dd.date_of(int(hold[-1]))) if len(hold) else None,
                'holdout_days_sha256': hashlib.sha256(np.ascontiguousarray(hold).tobytes()).hexdigest(),
                'forecast_first': str(first), 'forecast_last': str(first + timedelta(days=SP.BATCH_DAYS - 1)),
                'forecast_days': SP.BATCH_DAYS, 'internal_refits': 0,
                'input_fingerprint': plan['identity']['cp26_input_fingerprint'],
                'input_vintages_reference': 'reports/development-inputs/tier-register.json',
                'code_identity': plan['identity']['code_sha256'],
                'operational_lineage': H.runtime_lineage(root)}

    result = run_search(plan, evaluate, cache=TrialCache(cache_root, plan))
    result = _signed({**{key: value for key, value in result.items() if key != 'content_sha256'},
                      'operational_lineage': H.runtime_lineage(root)})
    _atomic(result_path, result)
    budget.event('ddnn2_search_terminal', k=plan['k'], route=plan['route'], status=result['status'],
                 plan_sha256=plan['content_sha256'], result_sha256=result['content_sha256'])
    return result


def fit_origin(ctx, day: date, ensemble: dict, k: int, charge=None) -> dict:
    """Fresh eight-member daily fit; any missing/failed member fails the whole D2 issuance."""
    _require_fit_authorization()
    from . import harness as H
    from cp24.execution import fit_origin as landed_fit_origin
    if serving_search(day) != k:
        raise SearchUnavailable('attempted daily use outside ensemble service period')
    validate_ensemble(ensemble, k, ctx.variant)
    if ensemble['identity'] != search_identity(Path(ctx.root), k, ctx.variant):
        raise SearchUnavailable('daily ensemble identity differs from current lineage')
    with H.ddnn2_without_load(ctx.noload_active):
        return landed_fit_origin(ctx.dd, ctx.keys, ctx.data, day, ensemble['members'], charge,
                                 uncovered_fails=True)
