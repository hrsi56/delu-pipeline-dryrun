"""What the Space computes from a served bundle, and what the pipeline computes before publishing it: one file.

This module runs in two places, unchanged: natively in the pipeline (`cp18.bundle`'s gate before publication) and
in the visitor's browser under Pyodide, next to the frozen package's zip (`space-product/`'s worker loads this file
as text and checks its SHA-256 against the build record). It imports only the frozen package (`cp17`, `cp24`), NumPy
and the standard library, and it never fits the product: every number it returns is the frozen package's own
function applied to the served files.

- `recompute(files)`: §19.5's recomputation check. Each recomputed window (the development rehearsal, 4.7T and the
  prospective record; the development folds are static from evidence) is evaluated again with `cp17.scoring.evaluate`
  from the served issued forecasts and outcomes (B0, the context, from its served training windows with
  `cp17.dailyfit.run`, which fits nothing for B0), and compared with the committed record by
  `cp17.scoring.recompute_check`. Any mismatch is a ✗ with both values, and blocks that day's publication.
- `explain(files)`: §19.7's A1 method (`cp17.attribution.group_shapley`) on the served fitted artifact. Each member's
  fit is rebuilt from the served arrays (parameters, target centre and scale) and the served training window with the
  landed design functions; the rebuilt members must reproduce the issued quantiles bit for bit, or nothing is shown.
- `features(files)`: the day's inputs against their training-window distribution, and each input group's mean
  absolute attribution on the day ("how much this fitted model relies on each input").
- `retrain_report(artifact, retrained)`: §19.3's comparison of a visitor's retrain with the issued artifact, by
  `cp17.retrain`'s deviation and status rule. The issued artifact records four of A1's per-member selections (the
  seed, the training and held-out day sets, the latest training day) and each member's weights' hash; those are the
  selections compared, and the report says so.
"""
from __future__ import annotations

from datetime import date
import io
import json
import os
import tempfile

import numpy as np

BOUNDARY = date(2026, 9, 6)
DEVELOPMENT = 'development · post-selection'
REHEARSAL = ('development rehearsal · CP-18 dry runs on development days · not prospective and not 4.7T')
FRESH_TEST = 'one-shot fresh-data test (4.7T) · not yet scored'
PROSPECTIVE = 'prospective · evaluation in progress until CP-19'
#: The selections the issued A1 artifact records per member (`cp17.issue.artifact`'s lineage), compared one by one.
RECORDED_A1_SELECTIONS = ('seed', 'train_days_sha256', 'holdout_days_sha256', 'latest_training_day')
RECORDED_A1_DIAGNOSTICS = ('params_sha256',)


def recomputed(entry: dict) -> bool:
    """Whether the Space recomputes a window (Space plan §4.6): 4.7T, the prospective record and the rehearsal."""
    return entry['window'] in ('4.7T', 'prospective') or entry['evidence_class'].startswith('development rehearsal')


def window_days(entry: dict, days: list[str]) -> list[str]:
    """The issued days a recomputed window scores: the rehearsal holds development days, the prospective record any
    later issued day; 4.7T's days are not served before it is scored."""
    if entry['evidence_class'].startswith('development rehearsal'):
        return [d for d in days if date.fromisoformat(d) <= BOUNDARY]
    if entry['window'] == 'prospective':
        return [d for d in days if date.fromisoformat(d) > BOUNDARY]
    return []


def _json(files: dict, name: str):
    return json.loads(files[name])


def outcome_map(outcomes: dict) -> dict:
    return {p['timestamp_utc']: (float('nan') if p['price'] is None else float(p['price'])) for p in outcomes['prices']}


def product_items(forecasts: dict, days: list[str]) -> list[dict]:
    from cp17.issue import canonical_hours
    wanted = set(days)
    return [{'day': d['delivery_day'], 'keys': d['keys'], 'quantiles': d['quantiles'],
             'canonical_hours': len(canonical_hours(date.fromisoformat(d['delivery_day'])))}
            for d in forecasts['days'] if d['delivery_day'] in wanted]


def _fit_from_files(files: dict, stem: str) -> dict:
    """`cp17.dailyfit.run` on a served training window (B0 fits nothing: the similar-day naive and its raw errors)."""
    from cp17 import dailyfit
    folder = tempfile.mkdtemp(prefix='glue-')
    npz, meta = os.path.join(folder, stem + '.npz'), os.path.join(folder, stem + '.json')
    with open(npz, 'wb') as handle:
        handle.write(files[stem + '.npz'])
    with open(meta, 'wb') as handle:
        handle.write(files[stem + '.json'])
    return dailyfit.run(stem.split('-')[0], npz, meta)


def b0_items(files: dict, days: list[str]) -> tuple[list[dict], list[str]]:
    """B0's forecasts of `days`, from its served training windows; days without one are named, never invented."""
    from cp17.issue import canonical_hours
    items, absent = [], []
    for d in days:
        stem = f'B0-{d}'
        if stem + '.json' not in files:
            absent.append(d)
            continue
        result = _fit_from_files(files, stem)
        items.append({'day': d, 'keys': result['index'], 'quantiles': result['quantiles'],
                      'canonical_hours': len(canonical_hours(date.fromisoformat(d))), 'quantiles_sha256':
                      result['quantiles_sha256']})
    return items, absent


SHOWN = ('pooled/scored_days', 'pooled/scored_hours', 'pooled/mae_eur', 'pooled/wis_eur', 'pooled/percentage/share')


def recompute(files: dict) -> dict:
    """§19.5's recomputation check over the served bundle; every row, with both values of the headline items."""
    from cp17 import scoring
    forecasts = _json(files, 'issued-forecasts.json')
    outcomes = outcome_map(_json(files, 'outcomes.json'))
    evaluation = _json(files, 'evaluation.json')
    issued_days = sorted(d['delivery_day'] for d in forecasts['days'])
    rows = []
    for i, entry in enumerate(evaluation['windows']):
        if not recomputed(entry):
            rows.append({'index': i, 'window': entry['window'], 'evidence_class': entry['evidence_class'],
                         'policy': entry['policy'], 'recomputed': False,
                         'note': 'static, from evidence: the development folds are not recomputed in the browser'})
            continue
        days = window_days(entry, issued_days)
        if entry['policy'] == 'product':
            items, absent = product_items(forecasts, days), []
        else:
            items, absent = b0_items(files, days)
        record = scoring.evaluate([{k: v for k, v in it.items() if k != 'quantiles_sha256'} for it in items], outcomes)
        check = scoring.recompute_check(record, entry['record'])
        a, b = scoring.flatten(record), scoring.flatten(entry['record'])
        rows.append({'index': i, 'window': entry['window'], 'evidence_class': entry['evidence_class'],
                     'policy': entry['policy'], 'recomputed': True, 'days': [it['day'] for it in items],
                     'absent_days': absent, 'compared': check['compared'], 'match': check['match'],
                     'blocks_publication': check['blocks_publication'], 'mismatches': check['mismatches'][:20],
                     'rule': check['rule'],
                     'shown': [{'path': p, 'recomputed': a.get(p), 'committed': b.get(p)} for p in SHOWN]})
    done = [r for r in rows if r['recomputed']]
    return {'schema': 'cp18b-recompute-v1', 'rows': rows, 'match': all(r['match'] for r in done),
            'blocks_publication': any(r['blocks_publication'] for r in done), 'recomputed_windows': len(done)}


# ------------------------------------------------------------------ the fitted artifact: A1's members rebuilt
def _npz(data: bytes) -> dict:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _training_window(files: dict, artifact: dict):
    from cp17 import dailyfit
    stem = f"{artifact['issued_policy']}-{artifact['delivery_day']}"
    arrays = _npz(files[stem + '.npz'])
    meta = json.loads(files[stem + '.json'])
    if meta['payload_sha256'] != artifact['lineage']['inputs'][artifact['issued_policy']]:
        raise ValueError("the served training window is not the issued fit's")
    if dailyfit.payload_digest(arrays, {k: v for k, v in meta.items() if k != 'payload_sha256'}) != meta['payload_sha256']:
        raise ValueError('the served training window differs from its identity')
    return arrays, meta


def rebuild_a1(files: dict) -> dict:
    """A1's eight member fits from the served arrays and training window, checked against the issued quantiles."""
    from cp17.dailyfit import a1_daydata, a1_keys, array_hash
    from cp24 import design as G
    from cp24 import ddnn2 as M
    from cp24.member import emit_rows
    artifact = _json(files, 'artifact.json')
    if artifact.get('issued_policy') != 'A1':
        raise ValueError('the served artifact is not an issued A1 fit')
    arrays, meta = _training_window(files, artifact)
    fitted = _npz(files['artifact-arrays.npz'])
    day = date.fromisoformat(artifact['delivery_day'])
    dd = a1_daydata(arrays, meta)
    keys = a1_keys(arrays, dd, day)
    di = np.array([dd.ix(day)])
    rows_k = keys.of_days(di)
    lower, ix, _ = G.window(dd, day, exclude_uncovered=False)
    fits, eurs = [], []
    recorded = artifact['lineage']['training']['members']
    for j, m in enumerate(meta['members']):
        config, seed = m['config'], int(m['seed'])
        if recorded[j]['seed'] != seed:
            raise ValueError(f'member {j}: the training window and the artifact name different seeds')
        weeks, _ = G.holdout_weeks(dd, lower, day, seed)
        train, _ = G.split(dd, ix, weeks)
        if _index_hash(train, meta['offset']) != recorded[j]['train_days_sha256']:
            raise ValueError(f"member {j}: the rebuilt training days are not the issued fit's")
        form = config['transform']
        stat = M.statistics_of(form)
        z_tr = G.target_z(dd, train, stat)
        mask = dd.mask[train]
        cap = M.CAP_MULTIPLE * float(np.max(np.abs(z_tr[mask])))
        n = len([k for k in fitted if k.startswith(f'member{j}_param')])
        params = [fitted[f'member{j}_param{k}'] for k in range(n)]
        centre, scale = fitted[f'member{j}_centre'], fitted[f'member{j}_scale']
        if not (np.array_equal(centre, dd.centre[stat][di]) and np.array_equal(scale, dd.scale[stat][di])):
            raise ValueError(f"member {j}: the served centre and scale are not the day's")
        groups = tuple(config['groups'])
        C_tr, B_tr, _, _ = G.raw_inputs(dd, train, groups, form)
        pre = G.Preprocessor(C_tr, B_tr)
        C_fc, B_fc, _, _ = G.raw_inputs(dd, di, groups, form)
        zq, _ = M.member_quantiles_z(params, pre.transform(C_fc, B_fc), config['activation'], form, cap)
        eur = centre[:, None, None] + scale[:, None, None] * zq
        fits.append({'config': config, 'window': {'held_out_weeks': [str(w) for w in weeks]}, 'params': params,
                     'cap_z': cap, 'centre': centre, 'scale': scale, 'eur': eur})
        eurs.append(emit_rows(keys, rows_k, di, eur))
    med, _ = M.ensemble_median(np.stack(eurs)[:, :, None, :])
    q = med[:, 0, :]
    issued_q = np.asarray(artifact['quantiles'], float)
    if array_hash(q) == artifact['quantiles_sha256'] and q.tolist() == artifact['quantiles']:
        reproduced = 'bit for bit'
    else:
        from cp17.scoring import RELATIVE_TOLERANCE
        gap = float(np.abs(q - issued_q).max()) if q.shape == issued_q.shape else float('inf')
        if not gap <= RELATIVE_TOLERANCE * max(1.0, float(np.abs(issued_q).max())):
            raise ValueError('the rebuilt members do not reproduce the issued quantiles')
        reproduced = f'within the frozen recomputation tolerance ({gap:.1e} EUR/MWh; runtime rounding)'
    return {'artifact': artifact, 'dd': dd, 'day': day, 'di': di, 'fits': fits, 'meta': meta, 'lower': lower, 'ix': ix,
            'keys': keys, 'rows_k': rows_k, 'reproduced': reproduced}


def _index_hash(ix, offset):
    from cp17.dailyfit import index_hash
    return index_hash(ix, offset)


def explain(files: dict) -> dict:
    """§19.7's A1 attribution on the served fitted artifact: exact baseline Shapley over the input groups."""
    from cp17.attribution import WORDING, group_shapley
    built = rebuild_a1(files)
    out = group_shapley(built['dd'], built['day'], built['di'], built['fits'])
    artifact = built['artifact']
    slots = built['keys'].slot[built['rows_k']].tolist()
    return {'schema': 'cp18b-explain-v1', 'artifact_sha256': artifact['artifact_sha256'],
            'delivery_day': artifact['delivery_day'], 'issued_policy': artifact['issued_policy'],
            'keys': artifact['keys'], 'slots': slots, 'method': out['method'], 'players': out['players'],
            'coalitions': out['coalitions'], 'phi_eur': out['phi_eur'], 'baseline_p50': out['baseline_p50'],
            'full_p50': out['full_p50'], 'efficiency_max_abs': out['efficiency_max_abs'], 'wording': WORDING,
            'issued_p50': [row[3] for row in artifact['quantiles']], 'reproduced': built['reproduced']}


def raw_design(dd, ix: np.ndarray) -> tuple[np.ndarray, list[str], list[str]]:
    """Every A1 input of days `ix` in its source unit, before the members' own transforms and standardization: the
    price lags (EUR/MWh), the load-forecast lags (MW), the GFS aggregates, the rolling statistics, the weekday and
    calendar flags. Columns are named as `cp24.design.raw_inputs` names them."""
    from cp24 import design as G
    cols, names, units = [], [], []
    for g, lag in G.PRICE_LAGS.items():
        cols.append(G.shifted(dd.price, lag)[ix])
        names += [f'{g}_h{h:02d}' for h in range(24)]
        units += ['EUR/MWh'] * 24
    for g, lag in G.LOAD_LAGS.items():
        cols.append(G.shifted(dd.load, lag)[ix])
        names += [f'{g}_h{h:02d}' for h in range(24)]
        units += ['MW'] * 24
    cols.append(dd.weather[ix].reshape(len(ix), 72))
    for h in range(24):
        names += [f'gfs_h{h:02d}_{k}' for k in ('wind10', 'wind100', 'dswrf')]
        units += ['m/s', 'm/s', 'W/m²']
    cols.append(dd.stats[ix])
    names += list(G.STAT_LOCATION + G.STAT_SCALE + G.STAT_COUNT)
    units += ['EUR/MWh'] * (len(G.STAT_LOCATION) + len(G.STAT_SCALE)) + ['hours']
    cols.append(dd.weekday[ix])
    names += [f'weekday_{k}' for k in range(7)]
    units += ['flag'] * 7
    cols.append(dd.calendar[ix])
    names += list(G.CAL_FLAGS) + [f'day_type_{k}' for k in range(3)] + [f'month_{k}' for k in range(1, 13)]
    units += ['flag'] * (len(G.CAL_FLAGS) + 15)
    return np.column_stack(cols), names, units


def features(files: dict, explained: dict | None = None) -> dict:
    """The day's inputs against the fit's training window, in their source units, and each group's reliance."""
    from cp17.attribution import column_group
    built = rebuild_a1(files)
    dd, day, di, ix = built['dd'], built['day'], built['di'], built['ix']
    window, names, units = raw_design(dd, ix)
    today = raw_design(dd, di)[0][0]
    used = set()
    for m in built['meta']['members']:
        used |= set(m['config']['groups'])
    rows = []
    for k, name in enumerate(names):
        col = window[:, k]
        finite = col[np.isfinite(col)]
        v = float(today[k])
        group = column_group(name)
        rows.append({'name': name, 'group': group, 'unit': units[k], 'value': v if np.isfinite(v) else None,
                     'missing': not np.isfinite(v),
                     'used_by_members': group in used or group in ('price_d1', 'load_d0', 'weekday'),
                     'percentile': (float((finite <= v).mean()) if finite.size and np.isfinite(v) else None),
                     'window_min': float(finite.min()) if finite.size else None,
                     'window_median': float(np.median(finite)) if finite.size else None,
                     'window_max': float(finite.max()) if finite.size else None, 'window_values': int(finite.size)})
    explained = explained or explain(files)
    reliance = []
    for g, phi in zip(explained['players'], explained['phi_eur']):
        a = np.abs(np.asarray(phi, float))
        reliance.append({'group': g, 'mean_abs_eur': float(a.mean()), 'max_abs_eur': float(a.max())})
    reliance.sort(key=lambda r: -r['mean_abs_eur'])
    return {'schema': 'cp18b-features-v1', 'artifact_sha256': built['artifact']['artifact_sha256'],
            'delivery_day': str(day), 'window_first_day': str(built['lower']), 'window_days': int(len(ix)),
            'members': [{'member': j, 'config': m['config'].get('id'), 'groups': list(m['config']['groups']),
                         'transform': m['config']['transform']} for j, m in enumerate(built['meta']['members'])],
            'inputs': rows, 'reliance': reliance,
            'wording': 'how much this fitted model relies on each input group on this day; attribution within one '
                       'fitted artifact, not causality and not incremental value'}


# ------------------------------------------------------------------ Train it yourself
def retrain_report(artifact: dict, retrained: dict, tau_eq: float | None = None) -> dict:
    """§19.3 item 2 for one visitor retrain against the issued artifact (cp17.retrain's deviation and status rule)."""
    from cp17 import retrain
    tau = retrain.TAU_EQ if tau_eq is None else tau_eq
    policy = artifact['issued_policy']
    if retrained.get('day') != artifact['delivery_day'] or retrained.get('index') != artifact['keys']:
        return {'policy': policy, 'day': artifact['delivery_day'], 'status': retrain.STATUS[2],
                'reason': 'different delivery day or keys', 'deviation': {'shape_equal': False},
                'selections': [], 'selections_match': False, 'payload_sha256_equal': False, 'tau_eq_eur': tau}
    dev = retrain.deviation(artifact['quantiles'], retrained['quantiles'])
    issued = artifact['lineage']['training']['members']
    rows, diagnostics = [], []
    for a, b in zip(issued, retrained.get('members', [])):
        rows.append({'member': a['member'], 'seed': a['seed'],
                     'compared': list(RECORDED_A1_SELECTIONS),
                     'differences': [f for f in RECORDED_A1_SELECTIONS if a.get(f) != b.get(f)]})
        diagnostics.append({'member': a['member'], 'differences': [f for f in RECORDED_A1_DIAGNOSTICS
                                                                   if a.get(f) != b.get(f)]})
    training = artifact['lineage']['training']
    search_match = (retrained.get('search') == training['search']
                    and retrained.get('ensemble_content_sha256') == training['ensemble_content_sha256'])
    match = (len(rows) == len(issued) == len(retrained.get('members', [])) and search_match
             and all(not r['differences'] for r in rows))
    window = retrained.get('payload_sha256') == artifact['lineage']['inputs'][policy]
    measured = retrain.status(dev, match, tau)
    out = {'policy': policy, 'day': artifact['delivery_day'], 'status': measured if window else retrain.STATUS[2],
           'measured_status': measured, 'deviation': dev, 'selections': rows, 'selections_match': match,
           'search_match': search_match, 'diagnostics': diagnostics, 'payload_sha256_equal': window, 'tau_eq_eur': tau,
           'selections_compared': ('the selections the issued artifact records per member: ' +
                                   ', '.join(RECORDED_A1_SELECTIONS) + "; each member's weights' hash beside them")}
    if not window:
        out['reason'] = "the retrain did not run on the issued fit's shipped training window"
    return out
