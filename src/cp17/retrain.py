"""§19.3's retrain comparison: a retrained fit against the issued artifact, and the status it reports.

**The status rule** (the Owner's decision 9, τ_eq = 0.01 EUR/MWh):

- "identical bit for bit" when every emitted quantile is IEEE-754 identical, bit for bit, to the issued one;
- "within ±0.01 EUR/MWh" when every data-driven selection matched and every emitted quantile is within τ_eq;
- "different" otherwise, with the numbers.

**The training window is bound first.** A retrain is of the issued fit only if it ran on that fit's shipped
training window (§19.3 item 1), so its payload identity must equal the issued artifact's; otherwise the status
is "different". (CP-17's first run of the training-window control showed why: a +1 EUR/MWh change to one
training hour moved the quantiles by under 0.01 EUR/MWh with every selection matched.)

The report also states whether every data-driven selection matched and the largest absolute difference per
emitted quantile level, in EUR/MWh (§19.3 item 2). It never replaces, rescores or relabels an issued forecast
(item 3): it reads two results and returns a comparison.

**The selections compared**, by policy, are those CP-17-probe defined (diagnostics such as weight hashes
are reported beside them and are not selections; row and day sets are compared by their int64 index hashes):

- **A1**, per member: the seeded held-out weeks and the week pool; the training and held-out day sets
  (counts and hashes); the input count; the winsorisation activations on the training, held-out and
  forecast rows; early stopping's best epoch, epochs run and stop reason; the cap's activations on the
  forecast; the guard events. Across members: the crossing restorations of the ensemble median.
- **LEAR members** (v2's A1 and B2, v4's A1_w and B2_w): the selected relative alpha of every local hour and
  the training rows.
- **LightGBM members** (v4's L-N and L-R): the selected capacity configuration of each block model, and the
  window, inner and validation rows.
- **H layers** (v2, v4): the buffer's released days, their errors' hash and the issuing scale.
- **B0**: the similar day and the raw-error buffer's days and errors' hash.
"""
from __future__ import annotations

import numpy as np

from .dailyfit import array_hash

TAU_EQ = 0.01
LEVELS = ('p025', 'p10', 'p25', 'p50', 'p75', 'p90', 'p975')
STATUS = ('identical bit for bit', 'within ±0.01 EUR/MWh', 'different')
A1_SELECTIONS = ('held_out_weeks', 'week_pool', 'n_days', 'n_train_days', 'n_held_out_days', 'train_days_sha256',
                 'holdout_days_sha256', 'n_inputs', 'winsor_train', 'winsor_hold', 'winsor_forecast', 'best_epoch',
                 'epochs_run', 'stop_reason', 'cap_forecast_slot_levels', 'events')
A1_DIAGNOSTICS = ('params_sha256', 'best_stopping_metric_eur', 'cap_z', 'preprocessor')
H_SELECTION = ('buffer_start', 'buffer_end', 'buffer_days', 'buffer_hours', 'buffer_sha256', 'scale_sha256')
LEAR_DIAGNOSTICS = ('alpha', 'final_n_iter', 'final_solver_passes', 'n_iter_by_alpha', 'solver_passes_by_alpha',
                    'model_sha256', 'validation_mae_by_alpha', 'scaler_sha256', 'imputer_sha256')
B0_SELECTION = ('buffer_start', 'buffer_end', 'buffer_days', 'buffer_hours', 'buffer_sha256')


def deviation(issued, retrained) -> dict:
    a, b = np.asarray(issued, float), np.asarray(retrained, float)
    if a.shape != b.shape:
        return {'shape_equal': False, 'issued_shape': list(a.shape), 'retrained_shape': list(b.shape)}
    diff = np.abs(a - b)
    return {'shape_equal': True, 'bitwise_equal': bool(np.array_equal(a.view(np.uint64), b.view(np.uint64))),
            'max_abs_eur': float(diff.max()), 'max_abs_eur_by_level': dict(zip(LEVELS, map(float, diff.max(axis=0)))),
            'mean_abs_eur': float(diff.mean()), 'issued_sha256': array_hash(a), 'retrained_sha256': array_hash(b)}


def a1_selections(issued: dict, retrained: dict) -> dict:
    rows, diagnostics = [], []
    for n, b in zip(issued['members'], retrained['members']):
        rows.append({'member': n['member'], 'seed': n['seed'], 'differences': [f for f in A1_SELECTIONS if n[f] != b[f]]})
        diagnostics.append({'member': n['member'], 'differences': [f for f in A1_DIAGNOSTICS if n[f] != b[f]]})
    crossed = issued['crossed_rows'] == retrained['crossed_rows']
    ok = len(rows) == 8 == len(retrained['members']) and crossed and all(not r['differences'] for r in rows)
    return {'all_match': ok, 'crossed_rows_match': crossed, 'members': rows, 'diagnostics': diagnostics}


def _lear(issued: list[dict], retrained: list[dict]) -> tuple[list, list]:
    hours = [n['local_hour'] for n, b in zip(issued, retrained)
             if n['selected_relative_alpha'] != b['selected_relative_alpha'] or n['local_hour'] != b['local_hour']]
    if len(issued) != len(retrained):
        hours.append('hour count')
    diagnostics = sorted({f for n, b in zip(issued, retrained) for f in LEAR_DIAGNOSTICS if n.get(f) != b.get(f)})
    return hours, diagnostics


def _lgbm(issued: dict, retrained: dict) -> tuple[dict, list]:
    out = {}
    for model, selected in issued['selected'].items():
        diff = []
        if retrained['selected'].get(model) != selected:
            diff.append('selected configuration')
        for key in ('window_sha256', 'inner_sha256', 'validation_sha256', 'forecast_sha256'):
            if issued['rows'][model][key] != retrained['rows'][model][key]:
                diff.append(key)
        out[model] = diff
    diagnostics = sorted({f"{n['model']}/{n['role']}/{n['config']}" for n, b in zip(issued['fits'], retrained['fits'])
                          if n.get('model_sha256') != b.get('model_sha256') or n.get('validation_mae') != b.get('validation_mae')})
    return out, diagnostics


def blend_selections(issued: dict, retrained: dict) -> dict:
    """v2 and v4: every member's selections, the training rows, and the H layer's buffer."""
    members, diagnostics = {}, {}
    for name, sel in issued['selections'].items():
        other = retrained['selections'].get(name)
        if other is None:
            members[name] = ['member missing']
            continue
        if isinstance(sel, list):
            members[name], diagnostics[name] = _lear(sel, other)
        else:
            members[name], diagnostics[name] = _lgbm(sel, other)
    h = all(issued['h_layer'][k] == retrained['h_layer'][k] for k in H_SELECTION)
    rows = issued['train_rows_sha256'] == retrained['train_rows_sha256']
    lgbm_ok = all(not d for v in members.values() if isinstance(v, dict) for d in v.values())
    lear_ok = all(not v for v in members.values() if isinstance(v, list))
    return {'all_match': bool(h and rows and lgbm_ok and lear_ok), 'members': members, 'h_layer_match': h,
            'train_rows_match': rows, 'diagnostics': diagnostics}


def b0_selections(issued: dict, retrained: dict) -> dict:
    layer = all(issued['b0_layer'][k] == retrained['b0_layer'][k] for k in B0_SELECTION)
    source = issued['source_day'] == retrained['source_day']
    return {'all_match': bool(layer and source), 'b0_layer_match': layer, 'source_day_match': source}


def selections(policy: str, issued: dict, retrained: dict) -> dict:
    if policy == 'A1':
        return a1_selections(issued, retrained)
    if policy in ('v2', 'v4', 'v5'):
        return blend_selections(issued, retrained)
    if policy == 'B0':
        return b0_selections(issued, retrained)
    raise ValueError(f'unknown policy {policy}')


def status(dev: dict, selections_match: bool, tau_eq: float = TAU_EQ) -> str:
    if not dev.get('shape_equal'):
        return STATUS[2]
    if dev['bitwise_equal']:
        return STATUS[0]
    if selections_match and dev['max_abs_eur'] <= tau_eq:
        return STATUS[1]
    return STATUS[2]


def compare(policy: str, issued: dict, retrained: dict, tau_eq: float = TAU_EQ) -> dict:
    """The full §19.3 item 2 report for one retrain.

    A retrain is of the issued fit only if it ran on that fit's shipped training window (§19.3 item 1): its
    payload identity must equal the issued result's. Otherwise the status is "different", whatever the quantiles
    and selections show; `measured_status` keeps what they alone would say."""
    if issued['day'] != retrained['day'] or issued.get('index') != retrained.get('index'):
        dev = {'shape_equal': False, 'reason': 'different delivery day or keys'}
        return {'policy': policy, 'day': issued['day'], 'status': STATUS[2], 'deviation': dev,
                'selections': {'all_match': False}, 'selections_match': False, 'tau_eq_eur': tau_eq,
                'payload_sha256_equal': False, 'measured_status': STATUS[2]}
    dev = deviation(issued['quantiles'], retrained['quantiles'])
    sel = selections(policy, issued, retrained)
    window = issued.get('payload_sha256') is not None and issued.get('payload_sha256') == retrained.get('payload_sha256')
    measured = status(dev, sel['all_match'], tau_eq)
    out = {'policy': policy, 'day': issued['day'], 'status': measured if window else STATUS[2], 'deviation': dev,
           'selections': sel, 'selections_match': sel['all_match'], 'tau_eq_eur': tau_eq,
           'payload_sha256_equal': window, 'measured_status': measured}
    if not window:
        out['reason'] = "the retrain did not run on the issued fit's shipped training window"
    return out
