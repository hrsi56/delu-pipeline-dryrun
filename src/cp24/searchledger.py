"""A round's search ledger and ensembles, from the search's identity-verified task records (a recorded repair).

Round 1's `job_search` completed every one of its 3,464 trial-batch fits, then failed while writing its ledger:
a trial outside the halving survivors (or with a failed fit) has an infinite pooled metric, and the ledger
writer refuses non-JSON floats. `src/cp24/search.py` is part of the search identity that binds every task
record, so it is left byte-identical. This job performs `job_search`'s post-processing on the same records,
unchanged, and writes an undefined (infinite) metric as null:

* every expected task record is loaded with the search identity (`cp24.search._load_task`); a missing or
  mismatched record refuses (it is never refitted here);
* the halving survivors are recomputed from the stage-A records with `cp24.sampler`'s rule, and the stage-B
  records must be exactly the survivors on the remaining batches;
* the ranking, the distinct top four, the member seeds and the ledger rows are `job_search`'s code.

No fit runs and nothing is charged but the monitor's machine time.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import sampler as SP
from .budget import atomic, ledger
from .evaluate import clean
from .jobs import stamp
from .member import config_hash
from .preflight import fold_table
from .search import _load_task, load_design, pooled_metric, round_dir, search_identity, task_path


def job_search_ledger(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, required=True)
    args = ap.parse_args(rest)
    root, r = Path(root), args.round
    rd = round_dir(root, r)
    if (rd / 'search-ledger.json').exists():
        raise ValueError('the search ledger exists; it is written once')
    design = load_design(root, r)
    n, space = design['trials_per_fold'], design['space']
    identity = search_identity(root, r)
    plan = {f['fold']: {'fold': f, 'trials': SP.trials(r, f['index'], n, space)} for f in fold_table(root)}
    results, survivors, missing, stage_b_mismatch = {}, {}, [], {}
    for name, p in plan.items():
        f = p['fold']
        nb = len(f['batches'])
        first = list(range(min(SP.HALVING_BATCHES, nb)))
        results[name] = {t: [] for t in range(n)}
        for t in range(n):
            for j in first:
                item = _load_task(task_path(r, name, t, j), identity)
                if item is None:
                    missing.append((name, t, j))
                else:
                    results[name][t].append(item)
        metric = {p['trials'][t]['id']: pooled_metric(results[name][t], first)[0] for t in range(n)}
        size = {p['trials'][t]['id']: SP.n_params(p['trials'][t], next((i['n_inputs'] for i in results[name][t]
                                                                         if i['status'] == 'ok'), 10**6)) for t in range(n)}
        order = [c['id'] for c in p['trials']]
        ranked = SP.rank(metric, size, order)
        keep = ranked[:SP.halving_survivors(n)] if nb > SP.HALVING_BATCHES else ranked
        survivors[name] = keep
        if nb > SP.HALVING_BATCHES:
            expected_b = {(order.index(cid), j) for cid in keep for j in range(SP.HALVING_BATCHES, nb)}
            present_b = set()
            for t in range(n):
                for j in range(SP.HALVING_BATCHES, nb):
                    path = task_path(r, name, t, j)
                    if path.exists():
                        present_b.add((t, j))
            if present_b != expected_b:
                stage_b_mismatch[name] = {'missing': sorted(expected_b - present_b), 'extra': sorted(present_b - expected_b)}
            for t, j in sorted(expected_b):
                item = _load_task(task_path(r, name, t, j), identity)
                if item is None:
                    missing.append((name, t, j))
                else:
                    results[name][t].append(item)
    if missing or stage_b_mismatch:
        raise ValueError(f'search records incomplete or inconsistent: missing {missing[:10]}, stage B {stage_b_mismatch}')
    # ---- job_search's post-processing, unchanged
    ledger_out, ensembles = {}, {}
    failed_total = 0
    for name, p in plan.items():
        f = p['fold']
        nb = len(f['batches'])
        allb = list(range(nb))
        first = list(range(min(SP.HALVING_BATCHES, nb)))
        order = [c['id'] for c in p['trials']]
        rows = []
        for t, cfg in enumerate(p['trials']):
            its = sorted(results[name][t], key=lambda i: i['batch'])
            failed_total += sum(i['status'] == 'failed' for i in its)
            pin_a, mae_a = pooled_metric(its, first)
            pin_all, mae_all = pooled_metric(its, allb) if cfg['id'] in survivors[name] else (math.inf, math.inf)
            n_inputs = next((i['n_inputs'] for i in its if i['status'] == 'ok'), None)
            rows.append({'trial': t, 'config': cfg, 'config_sha256': config_hash(cfg), 'n_inputs': n_inputs,
                         'n_parameters': SP.n_params(cfg, n_inputs) if n_inputs else None,
                         'ran_all_batches': cfg['id'] in survivors[name] and len(its) == nb,
                         'pinball_first_batches': pin_a, 'mae_first_batches': mae_a,
                         'pinball_all_batches': pin_all, 'mae_all_batches': mae_all,
                         'batches': [{'batch': i['batch'], 'start': i['batch_start'], 'seed': i['seed'], 'status': i['status'],
                                      'pinball': (i['pinball_sum'] / i['pinball_count']) if i['status'] == 'ok' else None,
                                      'mae': (i['abs_error_sum'] / i['hours']) if i['status'] == 'ok' else None,
                                      'epochs_run': i.get('member', {}).get('epochs_run'),
                                      'best_epoch': i.get('member', {}).get('best_epoch'),
                                      'stop_reason': i.get('member', {}).get('stop_reason'),
                                      'seconds': i.get('seconds'), 'cap_activations': i.get('guards', {}).get('cap_forecast_slot_levels'),
                                      'excluded_uncovered_days': i.get('window', {}).get('excluded_uncovered_days'),
                                      'error': i.get('error'), 'record_sha256': i['content_sha256']} for i in its]})
        eligible = {row['config']['id']: row['pinball_all_batches'] for row in rows if row['ran_all_batches']}
        size = {row['config']['id']: row['n_parameters'] or 10**9 for row in rows}
        ranked = [cid for cid in SP.rank(eligible, size, order) if math.isfinite(eligible[cid])]
        distinct, chosen = set(), []
        for cid in ranked:
            cfg = p['trials'][order.index(cid)]
            key = json.dumps({k: v for k, v in cfg.items() if k != 'id'}, sort_keys=True)
            if key not in distinct:
                distinct.add(key)
                chosen.append(cid)
            if len(chosen) == SP.ENSEMBLE_CONFIGS:
                break
        seeds = SP.member_seeds(r, f['index'])
        members = []
        for j, cid in enumerate(chosen):
            row = rows[order.index(cid)]
            for s in range(SP.ENSEMBLE_SEEDS):
                members.append({'rank': j + 1, 'config': row['config'], 'seed': seeds[j * SP.ENSEMBLE_SEEDS + s],
                                'validation_pinball_all_batches': row['pinball_all_batches'],
                                'validation_mae_all_batches': row['mae_all_batches']})
        ensembles[name] = {'fold': name, 'd0': str(f['d0']), 'B': nb, 'members': members, 'complete': len(chosen) == SP.ENSEMBLE_CONFIGS}
        ledger_out[name] = {'fold': name, 'd0': str(f['d0']), 'data_cutoff_exclusive': str(f['search_cutoff']),
                            'batches': [[str(a), str(b)] for a, b in f['batches']], 'B': nb,
                            'sampler_seed': [24, r, f['index'], 0x5A3], 'trials': rows,
                            'survivors': survivors[name], 'ranking_all_batches': ranked, 'ensemble': chosen}
    repair = {'what': 'job_search completed every fit and failed while writing this ledger (an infinite metric is not JSON); '
                      'written by cp24.searchledger from the identity-verified task records with job_search\'s post-processing; '
                      'an undefined (infinite) metric is null; no fit repeated', 'written_utc': stamp()}
    atomic(rd / 'search-ledger.json', clean({'schema': 'cp24-search-ledger-v1', 'round': r, 'written_utc': stamp(),
                                             'identity': identity, 'trials_per_fold': n, 'space': space, 'folds': ledger_out,
                                             'failed_fits': failed_total, 'repair': repair,
                                             'procedure': 'cp24.sampler: N trials on the min(4, B_f) most recent batches; the best '
                                                          'ceil(N/3) on all B_f; ensemble = four best distinct configurations '
                                                          'among trials that ran all B_f batches; ties to fewer parameters'}))
    atomic(rd / 'ensembles.json', clean({'schema': 'cp24-ensembles-v1', 'round': r, 'written_utc': stamp(), 'identity': identity,
                                         'folds': ensembles, 'repair': repair,
                                         'combination': 'per-level median of the eight members\' EUR/MWh quantiles'}))
    ledger().event('search_ledger_written', round=r, failed=failed_total, repair=repair['what'])
    print(json.dumps({'round': r, 'failed_fits': failed_total,
                      'ensembles': {k: [m['config']['id'] for m in v['members'][::2]] for k, v in ensembles.items()}}), flush=True)
    return 0 if all(e['complete'] for e in ensembles.values()) else 5
