"""A pre-fold round's design, projection and report (capstone v21-r11 §23.6).

A round is one search per fold, then the gate, then a pre-fold report. Before a round the Lead
projects it against every remaining ceiling, including the review reserve, and starts it only if it
fits (`job_round_design` refuses otherwise). The design -- the trial count, the space and any change S1
named, with the steering answer it rests on -- is committed before the search runs.

The pre-fold report covers the search ledger with each fold's ensemble and validation scores; the
gate table by fold and pooled; D2's error correlations with HG and L on the gate days; the excluded
training days; and the costs. Before attempt 1's scores exist it carries no warm-up or evaluation
outcome.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import ddnn2 as M
from . import design as G
from . import sampler as SP
from .admission import ATTEMPT_FITS, GATE_FITS, OTHER_PER_ATTEMPT, REVIEW_RESERVE, search_fits
from .budget import HOUR, atomic, effective_caps, ledger
from .jobs import stamp
from .preflight import fold_table

OUT = Path('reports/ddnn2')

RECIPE = {
    'window': '[max(2019-01-01, D-728), D) day rows with an eligible target slot and finite origin statistics; pre-fold fits '
              'leave out every delivery day without a frozen weather record (§23.6)',
    'held_out_weeks': f'the member\'s seeded random {G.HOLDOUT_SHARE:.0%} (rounded, at least one) of the whole Monday-Sunday weeks '
                      f'inside [window start, D-{G.RECENT_EXCLUDED_DAYS}); PCG64(SeedSequence([seed, 0x5EED]))',
    'minimum_rows': {'training_days': G.MIN_TRAIN_DAYS, 'held_out_days': G.MIN_HOLD_DAYS},
    'loss': f'first {M.WARM_EPOCHS} epochs NLL only; then kappa*NLL + (1-kappa)*mean pinball over the 19-level grid {list(M.GRID)}; '
            '+ l1*sum|W| + l2*sum W^2 over weight matrices; present slots only; rows weighted by recency (mean 1)',
    'stopping': f'mean pinball over the seven scored levels in EUR/MWh after inversion and the cap, on the held-out weeks, after '
                f'every epoch from {M.WARM_EPOCHS + 1}; patience {M.PATIENCE}; maximum {M.MAX_EPOCHS} epochs; best epoch kept '
                '(strict improvement); a nonfinite loss stops training and keeps the best epoch (recorded)',
    'optimizer': f'Adam {M.ADAM} in PyTorch\'s update order; Glorot-uniform init, zero biases, output biases at lambda = delta = 1',
    'guards': f'winsorisation of continuous inputs at the training rows\' {G.WINSOR} quantiles; the cap at +-{M.CAP_MULTIPLE} x max|z| '
              'on the member\'s training rows (before any asinh); sorting of crossings; every activation counted',
    'head': 'xi = o1, lambda = softplus(o2) + 1e-3, gamma = o3, delta = softplus(o4) + 0.05, per local-hour slot',
    'ensemble': 'four best distinct configurations x two seeds; per-level median of the eight members\' EUR/MWh quantiles '
                '(mean of the two middle values); p50 = central = D2',
    'emission': 'the day\'s feature-valid keys; on a 25-hour day both keys of the repeated hour take that slot\'s forecast',
}


def design_path(root: Path, r: int) -> Path:
    return Path(root) / OUT / 'rounds' / f'round-{r}' / 'design.json'


def projection(root: Path, n_trials: int, before_attempt: int) -> dict:
    """This round and the rest of the maximal remaining route against every remaining ceiling."""
    adm = json.loads((Path(root) / OUT / 'resource-admission.json').read_text())
    mu = adm['measured']['mean_random_fit_seconds']
    p75 = adm['measured']['p75_random_fit_seconds']
    worst = adm['measured']['largest_extreme_fit_seconds']
    state = ledger().read()
    caps = effective_caps(state)
    counts = state['counts']
    b = [len(f['batches']) for f in fold_table(root)]
    sf = search_fits(n_trials, b)
    attempts_left = 2 - counts.get('scored_attempts', 0)
    rounds_left_after = (3 - counts.get('rounds_before_attempt_1', 0) - 1 if before_attempt == 1 else 0) + (1 if before_attempt == 1 else 0)
    out = {}
    for label, member_s in (('expected', p75), ('worst_case', worst)):
        this_fits = sf + GATE_FITS
        this_hours = (sf * mu + GATE_FITS * member_s) / HOUR
        rest_fits = rounds_left_after * (sf + GATE_FITS) + attempts_left * (ATTEMPT_FITS + OTHER_PER_ATTEMPT['ddnn2_fits']) \
            + REVIEW_RESERVE['ddnn2_fits']
        rest_hours = rounds_left_after * this_hours + attempts_left * (ATTEMPT_FITS * member_s / HOUR + OTHER_PER_ATTEMPT['machine_hours']) \
            + REVIEW_RESERVE['machine_hours']
        fits_total = counts.get('ddnn2_fits', 0) + this_fits + rest_fits
        hours_total = counts.get('machine_seconds', 0) / HOUR + this_hours + rest_hours
        out[label] = {'member_seconds': member_s, 'this_round_fits': this_fits, 'this_round_machine_hours': round(this_hours, 2),
                      'maximal_remaining_route_fits_total': int(fits_total),
                      'maximal_remaining_route_machine_hours_total': round(hours_total, 2),
                      'fits_cap': caps['ddnn2_fits'], 'machine_hours_cap': caps['machine_seconds'] / HOUR,
                      'fits': bool(fits_total <= caps['ddnn2_fits'] and hours_total <= caps['machine_seconds'] / HOUR)}
    out['this_round_fits_alone'] = bool(counts.get('ddnn2_fits', 0) + sf + GATE_FITS + REVIEW_RESERVE['ddnn2_fits']
                                        <= caps['ddnn2_fits'])
    out['spent'] = {k: counts.get(k, 0) for k in ('ddnn2_fits', 'machine_seconds', 'rounds_before_attempt_1',
                                                    'rounds_before_attempt_2', 'scored_attempts', 'policy_days')}
    out['active_hours_so_far'] = round((__import__('time').time() - state['effort']['session_start_epoch']
                                        - sum(b - a for a, b in state['effort']['pauses'])) / HOUR, 2)
    return out


def job_round_design(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--before-attempt', type=int, choices=(1, 2), required=True)
    ap.add_argument('--changes', type=Path, default=None, help='JSON: the S1/S2-named design changes and the space')
    ap.add_argument('--steering', default=None, help='the committed steering answer this round rests on')
    args = ap.parse_args(rest)
    root = Path(root)
    path = design_path(root, args.round)
    if path.exists():
        raise ValueError('the round design exists; it is written once')
    adm = json.loads((root / OUT / 'resource-admission.json').read_text())
    n = adm['fixed']['trials_per_fold_per_round']
    space, changes = SP.SPACE_ROUND_1, None
    if args.changes:
        changes = json.loads(args.changes.read_text())
        space = changes.get('space', space)
    proj = projection(root, n, args.before_attempt)
    if not proj['expected']['fits']:
        raise ValueError(f'the round does not fit the remaining ceilings: {proj}')
    design = {'schema': 'cp24-round-design-v1', 'round': args.round, 'before_attempt': args.before_attempt,
              'trials_per_fold': n, 'space': space, 'changes_from_previous': changes, 'steering': args.steering,
              'recipe': RECIPE, 'sampler': 'cp24.sampler (seed SeedSequence([24, round, fold_index, 0x5A3])); fit seeds '
                                           'cp24.sampler.fit_seed; ensemble seeds cp24.sampler.member_seeds',
              'projection': proj, 'written_utc': stamp()}
    atomic(path, design)
    print(json.dumps({'round': args.round, 'trials': n, 'projection': proj['expected']}, default=str), flush=True)
    return 0


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return 'n/a'
    return f'{x:.{nd}f}' if isinstance(x, float) else str(x)


def job_round_report(root: Path, rest) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, required=True)
    args = ap.parse_args(rest)
    root, r = Path(root), args.round
    rd = Path(root) / OUT / 'rounds' / f'round-{r}'
    design = json.loads((rd / 'design.json').read_text())
    search = json.loads((rd / 'search-ledger.json').read_text())
    ens = json.loads((rd / 'ensembles.json').read_text())
    gate = json.loads((rd / 'gate.json').read_text())
    state = ledger().read()
    L = [f'# Pre-fold round {r} report (CP-24)\n',
         f'- **Before attempt:** {design["before_attempt"]}. **Trials per fold:** {design["trials_per_fold"]}. '
         f'**Design:** [`design.json`](design.json); **search ledger:** [`search-ledger.json`](search-ledger.json); '
         f'**ensembles:** [`ensembles.json`](ensembles.json); **gate:** [`gate.json`](gate.json).',
         '- **Evidence class:** pre-fold, training-only. This report carries no warm-up or evaluation outcome: every fit '
         'here trained and forecast before its fold\'s D0 (the search before D0 − 56).',
         f'- **Gate: {"PASS" if gate["passed"] else "FAIL"}.**\n', '## The gate (§23.6), pooled over the 280 gate days\n',
         '| Condition | Met | Values |', '|---|---|---|']
    c = gate['conditions']
    L.append(f'| G0 finite and ordered | {c["G0"]["met"]} | every DDNN-2 and v4-member forecast |')
    L.append(f'| G1 MAE(v5) ≤ MAE(v4) | {c["G1"]["met"]} | v5 {_fmt(c["G1"]["v5"])}, v4 {_fmt(c["G1"]["v4"])}, '
             f'difference {_fmt(c["G1"]["difference"], 4)} EUR/MWh |')
    L.append(f'| G2 MAE(D2) ≤ 1.10 × MAE(L) | {c["G2"]["met"]} | D2 {_fmt(c["G2"]["D2"])}, L {_fmt(c["G2"]["L"])}, '
             f'ratio {_fmt(c["G2"]["ratio"], 4)} |')
    L.append(f'| G3 cap binds on < 0.1% | {c["G3"]["met"]} | {c["G3"]["binding"]:,} of {c["G3"]["emitted_hour_levels"]:,} '
             f'member hour-levels ({100 * c["G3"]["share"]:.3f}%) |\n')
    L += ['## By fold\n', '| Fold | Gate window | Days | Hours | MAE v4 | MAE v5 | MAE D2 | MAE L | MAE HG | Cap share | '
          'corr(D2, HG) | corr(D2, L) | Excluded days (v4 / DDNN-2, max) |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for fold, p in gate['by_fold'].items():
        L.append(f'| {fold} | {p["gate"][0]}..{p["gate"][1]} | {p["days"]} | {p["hours"]} | {_fmt(p["MAE_v4"])} | '
                 f'{_fmt(p["MAE_v5"])} | {_fmt(p["MAE_D2"])} | {_fmt(p["MAE_L"])} | {_fmt(p["MAE_HG"])} | '
                 f'{100 * p["cap_share"]:.3f}% | {_fmt(p["corr_D2_HG"])} | {_fmt(p["corr_D2_L"])} | '
                 f'{p["excluded_training_days_v4_max"]} / {p["excluded_training_days_d2_max"]} |')
    po = gate['pooled']
    L.append(f'| pooled | | {po["days"]} | {po["hours"]} | {_fmt(po["MAE_v4"])} | {_fmt(po["MAE_v5"])} | {_fmt(po["MAE_D2"])} | '
             f'{_fmt(po["MAE_L"])} | {_fmt(po["MAE_HG"])} | {100 * c["G3"]["share"]:.3f}% | {_fmt(po["corr_D2_HG"])} | '
             f'{_fmt(po["corr_D2_L"])} | |\n')
    L += ['## The search and each fold\'s ensemble\n']
    for fold, f in search['folds'].items():
        trials = f['trials']
        ok = [t for t in trials if math.isfinite(t['pinball_first_batches'] or math.inf)]
        failed = sum(b['status'] == 'failed' for t in trials for b in t['batches'])
        L.append(f'### {fold} (D0 {f["d0"]}, B = {f["B"]}, batches {f["batches"][-1][0]}..{f["batches"][0][1]})\n')
        L.append(f'- {len(trials)} trials, {sum(len(t["batches"]) for t in trials)} fits, {failed} failed; '
                 f'{len(f["survivors"])} ran all {f["B"]} batches.')
        if ok:
            vals = np.array([t['pinball_first_batches'] for t in ok])
            L.append(f'- Pinball on the {min(4, f["B"])} most recent batches: best {vals.min():.3f}, median {np.median(vals):.3f}, '
                     f'worst {vals.max():.3f} EUR/MWh.')
        L.append('\n| Rank | Trial | Layers | Activation | Transform | κ | Half-life | Batch | lr | Dropout | L1 | L2 | Groups | '
                 'Validation pinball | Validation MAE |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for j, m in enumerate(ens['folds'][fold]['members'][::2], start=1):
            cfg = m['config']
            L.append(f'| {j} | {cfg["id"]} | {cfg["hidden"]} | {cfg["activation"]} | {cfg["transform"]} | {cfg["kappa"]} | '
                     f'{cfg["half_life"]} | {cfg["batch_size"]} | {cfg["lr"]:.2e} | {_fmt(cfg["input_dropout"], 2)} | '
                     f'{"off" if cfg["l1"] is None else f"{cfg["l1"]:.1e}"} | {"off" if cfg["l2"] is None else f"{cfg["l2"]:.1e}"} | '
                     f'{", ".join(cfg["groups"]) or "none"} | {_fmt(m["validation_pinball_all_batches"])} | '
                     f'{_fmt(m["validation_mae_all_batches"])} |')
        L.append('')
    jobs = [j for j in state['jobs'] if j['name'] in (f'search-r{r}', f'gate-r{r}', 'v4-gate')]
    L += ['## Costs\n', f'- DDNN-2 member fits so far: {state["counts"].get("ddnn2_fits", 0):,} (search '
          f'{state["counts"].get("ddnn2_fits_search", 0):,}, gate {state["counts"].get("ddnn2_fits_gate", 0):,}, 4.6R′ '
          f'{state["counts"].get("ddnn2_fits_resource_admission", 0):,}).',
          f'- Machine-hours so far: {state["counts"].get("machine_seconds", 0) / HOUR:.2f} of {effective_caps(state)["machine_seconds"] / HOUR:.0f}.',
          f'- This round\'s jobs: ' + '; '.join(f'{j["name"]} {j.get("charged_seconds", 0) / HOUR:.2f} machine-hours' for j in jobs) + '.',
          f'- Gate fit seconds: DDNN-2 {gate["costs"]["ddnn2_fit_seconds"]:.0f}, v4 members {gate["costs"]["v4_fit_seconds"]:.0f} '
          '(the v4 pass is one pass, reused across rounds).',
          '\n## Excluded training days (§23.6)\n',
          '- Every pre-fold fit leaves out the delivery days without a frozen weather record (2022-09-29..2023-03-24). Per gate '
          'day, the number left out is in `gate.json` (`excluded_training_days`); per search fit, in the ledger.',
          f'- v4\'s members on fold 4\'s gate days: {gate["by_fold"]["fold_4"]["excluded_training_days_v4_max"]} days at most; '
          f'DDNN-2: {gate["by_fold"]["fold_4"]["excluded_training_days_d2_max"]} at most. Other folds\' gate days: none.']
    (rd / 'report.md').write_text('\n'.join(L) + '\n')
    print(f'wrote {rd / "report.md"}', flush=True)
    return 0
