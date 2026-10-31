"""4.6R: DDNN's training-only resource entry (capstone v21-r10 §21.4; handoff 4.6R).

**Order.** It runs only after the §21.3 checks pass, and refuses otherwise
(`cp23.reference.require_passing_record`).

**Data.** Training partitions only. Each origin is a fold's genuine warm-up start, whose data is
materialised strictly before the fold's first evaluation day. No evaluation-fold or reserved
outcome is read, and nothing is scored.

**What it measures.** At two origins -- fold 1's (the shortest windows) and fold 5's (a full 728-day
window, the representative scale) -- every frozen configuration's four-seed ensemble is fitted
exactly as the main run would fit it. It records:

* peak memory;
* fit and prediction time;
* epochs to the early stop;
* that the seven quantiles and the p50 are finite and ordered.

It is accuracy-blind: no validation or holdout loss value is reported.

**Projection.** The full run is projected against every §21.8 ceiling, conservatively assuming the
largest configuration everywhere. A projection above any ceiling is NOT_ADMITTED, with the
concrete blocker; it never leads to a smaller experiment.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np

from . import ddnn as D
from .budget import CAPS, GIB, HOUR, charge_fits, ledger
from .features import encode, target
from .inputs import load, origin_manifest, weather_design, weather_matrix
from .jobs import art, stamp, write_json
from .member import fit_origin, maxrss
from .reference import require_passing_record

OUT = Path('reports/distribution-challenger')
ORIGINS_WITH_ROWS = 636      # CP-22's committed E1: 638 origins, two without eligible hours
SELECTION_FITS = 5 * len(D.CONFIGS) * len(D.SEEDS)
#: Planned non-main DDNN fits (all inside the 6,000 total).
PLANNED_NON_MAIN = {'resource_admission': 2 * len(D.CONFIGS) * len(D.SEEDS), 'controls': 240, 'reproduction': 24,
                    'daily_cycle': 25 * len(D.SEEDS), 'review': 40, 'repair_reserve': 1500}
#: Non-fit machine time: replay, scoring, the bootstrap, controls' non-fit parts, the daily cycle's
#: other steps, tests, the Critic's reproduction and slack (aggregate hours).
OTHER_MACHINE_HOURS = 10.0
PROJECTION_MARGIN = 1.5


def job_resource_admission(root: Path, rest) -> int:
    root = Path(root)
    reference = require_passing_record(root)
    budget = ledger()
    charge = charge_fits(budget, 'resource_admission', main=False)
    m = origin_manifest(root)
    design_w = weather_design(root)
    picked = (m['folds'][0], m['folds'][4])
    out = {'schema': 'cp23-resource-admission-v1', 'written_utc': stamp(), 'machine': platform.platform(),
           'reference_record': {'passed': reference['passed'], 'ddnn_sha256': reference['ddnn_sha256'],
                                'written_utc': reference['written_utc']},
           'blas_threads': 1, 'workers_in_this_job': 1, 'origins': [], 'fits': []}
    t0 = time.perf_counter()
    for fold in picked:
        first = date.fromisoformat(fold['evaluation_start'])
        day = date.fromisoformat(fold['warmup_start'])
        if day >= first:
            raise ValueError('4.6R origins must precede the fold\'s evaluation window')
        t_load = time.perf_counter()
        data = load(root, before=first)
        if max(data.dates) >= np.datetime64(first):
            raise ValueError('4.6R materialised an evaluation-fold date')
        wx, present = weather_matrix(design_w, data)
        x, labels = encode(data)
        z = target(data)
        load_seconds = time.perf_counter() - t_load
        origin = {'fold': fold['fold'], 'origin': str(day), 'materialised_through': str(max(data.dates)),
                  'load_and_design_seconds': load_seconds, 'n_inputs_before_weather': int(x.shape[1]),
                  'input_labels': labels, 'configs': {}}
        for config in D.CONFIGS:
            fit = fit_origin(data, x, wx, z, present, day, config['id'], charge=charge)
            q = fit['quantiles']
            ordered = bool(np.all(np.diff(q, axis=1) > 0))
            finite = bool(np.isfinite(q).all() and np.isfinite(fit['central']).all())
            p50_is_median = bool(np.array_equal(fit['central'], q[:, D.MEDIAN]))
            for rec in fit['members']:
                out['fits'].append({'fold': fold['fold'], 'origin': str(day), 'config': config['id'], 'seed': rec['seed'],
                                    'n_train': fit['rows']['n_train'], 'n_stop': fit['rows']['n_stop'],
                                    'best_epoch': rec['best_epoch'], 'epochs_run': rec['epochs_run'],
                                    'fit_wall_seconds': rec['fit_wall_seconds'], 'fit_cpu_seconds': rec['fit_cpu_seconds'],
                                    'params_sha256': rec['params_sha256']})
            origin['configs'][config['id']] = {
                'n_parameters': D.n_parameters(fit['n_inputs'], config['hidden']), 'n_inputs': fit['n_inputs'],
                'rows': fit['rows'], 'n_forecast_hours': len(fit['timestamp_utc']),
                'ensemble_fit_wall_seconds': sum(r['fit_wall_seconds'] for r in fit['members']),
                'predict_seconds': fit['predict_seconds'], 'origin_seconds': fit['seconds'],
                'emission': {'finite': finite, 'ordered': ordered, 'p50_is_ensemble_median_and_central': p50_is_median,
                             'crossed_rows_rearranged': fit['crossed_rows'], 'levels': list(D.LEVELS)},
                'maxrss_bytes_after': fit['maxrss_bytes']}
        out['origins'].append(origin)
    out['job_seconds'] = time.perf_counter() - t0
    out['peak_rss_bytes_single_process'] = maxrss()
    out['projection'] = project(out, budget.read())
    out['verdict'] = 'PASS' if out['projection']['within_every_ceiling'] and all(
        c['emission']['finite'] and c['emission']['ordered'] and c['emission']['p50_is_ensemble_median_and_central']
        for o in out['origins'] for c in o['configs'].values()) else 'NOT_ADMITTED'
    out['cause'] = None if out['verdict'] == 'PASS' else out['projection']['blockers'] or ['emission requirement failed']
    write_json(art() / 'resource-admission.json', out)
    write_json(root / OUT / 'resource-admission.json', out)
    print(json.dumps({'verdict': out['verdict'], 'projection': {k: v for k, v in out['projection'].items() if k != 'by_config'}},
                     default=str), flush=True)
    return 0


def project(out: dict, state: dict) -> dict:
    fits = out['fits']
    by_config = {}
    for config in D.CONFIGS:
        rows = [f for f in fits if f['config'] == config['id']]
        full = [f for f in rows if f['fold'] == 'fold_5']
        by_config[config['id']] = {
            'mean_fit_wall_seconds_all': statistics.fmean(f['fit_wall_seconds'] for f in rows),
            'mean_fit_wall_seconds_full_window': statistics.fmean(f['fit_wall_seconds'] for f in full),
            'max_fit_wall_seconds': max(f['fit_wall_seconds'] for f in rows),
            'mean_epochs_run': statistics.fmean(f['epochs_run'] for f in rows),
            'max_epochs_run': max(f['epochs_run'] for f in rows)}
    worst = max(v['mean_fit_wall_seconds_full_window'] for v in by_config.values())
    worst_max = max(v['max_fit_wall_seconds'] for v in by_config.values())
    main_fits = ORIGINS_WITH_ROWS * len(D.SEEDS) + SELECTION_FITS
    total_fits = main_fits + sum(PLANNED_NON_MAIN.values())
    fit_hours = total_fits * worst * PROJECTION_MARGIN / HOUR
    machine_hours = fit_hours + OTHER_MACHINE_HOURS
    spent_hours = state['counts'].get('machine_seconds', 0) / HOUR
    peak = out['peak_rss_bytes_single_process']
    rss_projection = 4 * peak + 1.0 * GIB  # four workers plus a parent and the monitor
    ledger_rss = state['peaks'].get('rss_bytes', 0)
    disk_projection = 1.0 * GIB  # caches, states, logs, the local MLflow store and two worktrees (bounded below)
    blockers = []
    if main_fits > CAPS['main_ddnn_fits']:
        blockers.append(f'main fits {main_fits} > {CAPS["main_ddnn_fits"]}')
    if total_fits > CAPS['ddnn_fits']:
        blockers.append(f'total fits {total_fits} > {CAPS["ddnn_fits"]}')
    if machine_hours + spent_hours > CAPS['machine_seconds'] / HOUR:
        blockers.append(f'machine hours {machine_hours + spent_hours:.1f} > {CAPS["machine_seconds"] / HOUR:.0f}')
    if rss_projection > CAPS['rss_bytes']:
        blockers.append(f'aggregate RSS {rss_projection / GIB:.2f} GiB > 10 GiB')
    wall_hours_four_workers = main_fits * worst * PROJECTION_MARGIN / HOUR / 4
    return {'main_fits': main_fits, 'main_fit_cap': CAPS['main_ddnn_fits'], 'selection_fits': SELECTION_FITS,
            'origins_with_eligible_hours': ORIGINS_WITH_ROWS, 'seeds': len(D.SEEDS),
            'planned_non_main_fits': PLANNED_NON_MAIN, 'total_fits': total_fits, 'total_fit_cap': CAPS['ddnn_fits'],
            'assumed_seconds_per_fit': worst, 'worst_single_fit_seconds': worst_max, 'margin': PROJECTION_MARGIN,
            'assumption': 'the largest-cost configuration\'s mean full-window fit time for every fit, x1.5',
            'fit_machine_hours': fit_hours, 'other_machine_hours': OTHER_MACHINE_HOURS,
            'projected_machine_hours': machine_hours, 'machine_hours_spent_before': spent_hours,
            'machine_hour_cap': CAPS['machine_seconds'] / HOUR,
            'main_run_wall_hours_with_four_workers': wall_hours_four_workers,
            'peak_rss_bytes_single_process': peak, 'projected_aggregate_rss_bytes': rss_projection,
            'ledger_peak_aggregate_rss_bytes': ledger_rss, 'rss_cap_bytes': CAPS['rss_bytes'],
            'projected_added_disk_bytes': disk_projection, 'disk_cap_bytes': CAPS['additional_disk_bytes'],
            'policy_days_planned': {'v5_and_v3+D_admission_and_evaluation': 2 * 638, 'D_issuance': 450,
                                    'hg_v4_parity': 2 * 638, 'controls': 300, 'daily_cycle': 100, 'review': 1500,
                                    'total': 2 * 638 + 450 + 2 * 638 + 300 + 100 + 1500, 'cap': CAPS['policy_days']},
            'data_bytes': 0, 'remote_writes': 0, 'by_config': by_config,
            'within_every_ceiling': not blockers, 'blockers': blockers}
