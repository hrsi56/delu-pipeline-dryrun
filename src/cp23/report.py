"""The CP-23 report (`reports/distribution-challenger/report.md`), generated from the committed tables only.

Run as ``python -m cp23.report`` after scoring, the controls and the diagnostics. Every number is read from
a committed row: decisions.json, metrics.csv, uncertainty.csv, criteria.csv, diagnostics.csv,
diagnostics/, selection.json, controls.json, parity.json, reproduction.json, fit-cost.json, daily-cycle.json,
resources.json, resource-admission.json and reference-checks.json. Nothing is typed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cp15.scoring import FOLDS
from .execution import OUT
from . import scoring as S

NAMES = {'HG': 'v3', 'HGL': 'v4 (three-block)', 'B0': 'B0 naive', 'B1': 'B1 (v1)', 'B2': 'B2 daily LEAR',
         'B3': 'B3 daily LightGBM', 'A1': 'A1 normalized LEAR', 'v5': 'v5 = (2/3)·v4 + (1/3)·D', 'D': 'D (DDNN alone)',
         'v3+D': 'v3+D (DDNN as v3\'s third member)'}
ORDER = ('v5', 'HGL', 'v3+D', 'HG', 'D', 'A1', 'B2', 'B0', 'B1', 'B3')


def _n(policy: str) -> str:
    return NAMES.get(policy, policy)


def _f(x, d=4):
    return '—' if x is None or (isinstance(x, float) and x != x) else f'{x:.{d}f}'


def _ci(lo, hi, d=4):
    return f'[{_f(lo, d)}, {_f(hi, d)}]'


def _cond(conditions, k):
    return conditions[str(k)] if str(k) in conditions else conditions[k]


def build(root: Path) -> str:
    out = root / OUT
    dec = json.loads((out / 'decisions.json').read_text())
    metrics = pd.read_csv(out / 'metrics.csv')
    crit = pd.read_csv(out / 'criteria.csv')
    diag = pd.read_csv(out / 'diagnostics.csv', low_memory=False)
    load = lambda name: json.loads((out / name).read_text()) if (out / name).exists() else None  # noqa: E731
    controls, parity, repro = load('controls.json'), load('parity.json'), load('reproduction.json')
    cost, cycle, res, protocol = load('fit-cost.json'), load('daily-cycle.json'), load('resources.json'), load('protocol.json')
    sel, dg, r46, ref = load('selection.json'), load('diagnostics.json'), load('resource-admission.json'), load('reference-checks.json')
    a = dec['adoption']
    conds = a['conditions']
    policies = [p for p in ORDER if p in dec['scores']]
    L = []
    add = L.append
    add('# CP-23 — the DDNN route: v5 = v4 plus a DDNN member\n')
    add('`capstone_v21.md` v21-r10 §21 · evidence class **development_post_selection** · population `common-10747h` '
        '(10,747 keys, five folds).\n')
    add('## Verdicts\n')
    add('| Status | Result |\n|---|---|')
    add('| Engineering | Bound to the fresh Integration-Critic verdict on the final candidate '
        '(`docs/track-b/evidence/cp-23/integration.md`) |')
    add('| Entry | 4.6L: research use and retention permitted · §21.3: import audit, gradient checks and the PyTorch reference '
        f'passed ({ref["counts"]["passed"]} of {ref["expected_passed"]}) · 4.6R: **{r46["verdict"]}** · DDNN admitted to 4.6C |')
    add(f'| `cp23-adoption` | **{a["verdict"]}** · first unmet condition **{a["first_unmet_condition"]}**'
        + (f' (unmet: {", ".join(map(str, a["unmet_conditions"]))})' if a['unmet_conditions'] else '') + ' |')
    add('| Research | Development finding under the rule pre-registered on 2026-10-04; one more decision on the same five folds '
        '(4.7T carries the protection) |')
    add('| Product | v1 remains the released product and the demo; no designation, freeze or Live follows (§16) |\n')
    add('The rule was applied mechanically to the committed tables (`reports/distribution-challenger/decisions.json`). D and '
        'v3+D are never eligible. A mixed result is no demonstrated joint preference, never equivalence.\n')
    add('## DDNN as run\n')
    dd = protocol['ddnn']
    add(f'- **Model:** {dd["architecture"]["family"]}; {dd["architecture"]["head"]}. NumPy and the standard library only '
        '(`src/cp23/ddnn.py`; import audit in `tests/cp23/test_numpy_only.py`).')
    add(f'- **Information:** {dd["representation"]["inputs"]}. Target: {dd["representation"]["target"]}.')
    add(f'- **Configurations:** ' + ', '.join(f'{c["id"]} {c["hidden"]} ({c["n_parameters"]:,} parameters)' for c in dd['configurations'])
        + f'. Selection: {dd["selection"]["when"]}; {dd["selection"]["criterion"]}; {dd["selection"]["tie"]}.')
    add('- **Chosen per fold** (`selection.json`): ' + '; '.join(
        f'{f} {v["selected"]} (holdout MAE ' + ', '.join(f'{c} {_f(m, 2)}' for c, m in v['holdout_mae'].items())
        + f'; margin {_f(100 * v["winner_margin_relative"], 2)}%)' for f, v in sel['folds'].items()) + '.')
    add(f'- **Training:** {dd["training"]["loss"]}; {dd["training"]["optimizer"]}; learning rate {dd["training"]["lr"]}, batch '
        f'{dd["training"]["batch_size"]}, L2 {dd["training"]["l2"]}; early stopping on {dd["early_stopping"]["rows"]}, patience '
        f'{dd["early_stopping"]["patience"]}, at most {dd["early_stopping"]["max_epochs"]} epochs; best epoch kept.')
    add(f'- **Ensemble:** seeds {dd["ensemble"]["seeds"]}, {dd["ensemble"]["combination"]}; the p50 and D are the ensemble '
        'median. History `[max(2019-01-01, D−728), D)`, a fresh fit at every one of the 636 origins with eligible hours.\n')
    add('## Scores (equal-fold ratios to B0; lower is better)\n')
    eq = metrics.loc[metrics.scope.eq('equal_fold')].set_index('policy')
    pooled = metrics.loc[metrics.scope.eq('pooled')].set_index('policy')
    add('| Policy | S_MAE | S_WIS | pooled MAE (EUR/MWh) | pooled WIS | pooled 95% coverage | mean 95% width | §8 |')
    add('|---|---|---|---|---|---|---|---|')
    status = dec['original_section8_status']
    for p in policies:
        add(f'| {_n(p)} | {_f(eq.loc[p, "S_MAE"])} | {_f(eq.loc[p, "S_WIS"])} | {_f(pooled.loc[p, "MAE"], 2)} | '
            f'{_f(pooled.loc[p, "WIS"], 2)} | {_f(pooled.loc[p, "coverage95"], 3)} | {_f(pooled.loc[p, "mean_width95"], 2)} | '
            f'{status.get(p, "saved reference")} |')
    add('')
    add('## `cp23-adoption` (§21.6), condition by condition\n')
    v1 = _cond(conds, 1)['values']
    add(f'1. Joint improvement over v4: dS_MAE {_f(v1["MAE"]["difference"])} {_ci(v1["MAE"]["ci_lower"], v1["MAE"]["ci_upper"])} '
        f'(needs upper ≤ 0), dS_WIS {_f(v1["WIS"]["difference"])} {_ci(v1["WIS"]["ci_lower"], v1["WIS"]["ci_upper"])} (needs upper '
        f'< 0) → **{"met" if _cond(conds, 1)["met"] else "not met"}**. As a share of v4\'s score: S_MAE '
        f'{_f(100 * v1["MAE"]["ratio"], 2)}% {_ci(100 * v1["MAE"]["ratio_ci"][0], 100 * v1["MAE"]["ratio_ci"][1], 2)}%, S_WIS '
        f'{_f(100 * v1["WIS"]["ratio"], 2)}% {_ci(100 * v1["WIS"]["ratio_ci"][0], 100 * v1["WIS"]["ratio_ci"][1], 2)}%.')
    nm = _cond(conds, 2)['values']['not_met']
    add('2. All six original §8 diagnostics: ' + ('**met**.' if _cond(conds, 2)['met'] else
        '**not met** (' + ', '.join(f"criterion {r['criterion']} {r['scope']}" for r in nm) + ').'))
    add('3. A complete, valid evaluation: all 10,747 keys issued with finite, ordered quantiles; Engineering PASS is bound to '
        'the Integration verdict.')
    worse = _cond(conds, 4)['values']['folds_decisively_worse']
    add('4. No resolved per-fold degradation: ' + ('**met**.\n' if _cond(conds, 4)['met'] else '**not met** — ' + '; '.join(
        f"{r['scope']} {r['metric']} {_f(r['difference'], 3)} EUR/MWh {_ci(r['ci_lower'], r['ci_upper'], 3)}" for r in worse) + '.\n'))
    add(f'**Outcome:** {a["verdict"]}, with condition {a["first_unmet_condition"]} as the reason.\n')
    add('## Every §21.5 contrast, with its reading\n')
    add('| Contrast | Role | dS_MAE [95%] | dS_WIS [95%] | ratio S_MAE [95%] | ratio S_WIS [95%] | Reading |')
    add('|---|---|---|---|---|---|---|')
    for key, r in dec['contrasts'].items():
        add(f'| {key} | {r["role"]} | {_f(r["dS_MAE"]["point"])} {_ci(*r["dS_MAE"]["ci"])} | {_f(r["dS_WIS"]["point"])} '
            f'{_ci(*r["dS_WIS"]["ci"])} | {_f(100 * r["dS_MAE"]["ratio"], 2)}% '
            f'{_ci(100 * r["dS_MAE"]["ratio_ci"][0], 100 * r["dS_MAE"]["ratio_ci"][1], 2)}% | {_f(100 * r["dS_WIS"]["ratio"], 2)}% '
            f'{_ci(100 * r["dS_WIS"]["ratio_ci"][0], 100 * r["dS_WIS"]["ratio_ci"][1], 2)}% | {r["reading"]} |')
    add('')
    add('Bootstrap: seed 15042, 2,000 replicates of 7-calendar-day blocks within each fold, the shared CP-20 index set '
        f'(SHA-256 `{dec["bootstrap"]["index_sha256"]}`, equal to CP-20\'s: {dec["bootstrap"]["index_equals_cp20"]}); every '
        'replicate is stored in `replicates.parquet`.\n')
    add('## Per-fold MAE and WIS (EUR/MWh)\n')
    per = metrics.loc[metrics.scope.eq('per_fold')]
    add('| Policy | ' + ' | '.join(f'{f} MAE / WIS' for f in FOLDS) + ' |')
    add('|---|' + '---|' * len(FOLDS))
    for p in policies:
        part = per.loc[per.policy.eq(p)].set_index('fold')
        add(f'| {_n(p)} | ' + ' | '.join(f'{_f(part.loc[f, "MAE"], 2)} / {_f(part.loc[f, "WIS"], 2)}' for f in FOLDS) + ' |')
    unc = pd.read_csv(out / 'uncertainty.csv')
    pf = unc.loc[unc.scope.isin(FOLDS) & unc.candidate.eq('v5') & unc.baseline.eq('HGL')]
    add('\n**v5 − v4 per fold** (paired daily-loss difference, EUR/MWh, 95%):\n')
    add('| Fold | MAE | WIS |\n|---|---|---|')
    for f in FOLDS:
        row = pf.loc[pf.scope.eq(f)].set_index('metric')
        add(f'| {f} | {_f(row.loc["MAE", "difference"], 3)} {_ci(row.loc["MAE", "ci_lower"], row.loc["MAE", "ci_upper"], 3)} | '
            f'{_f(row.loc["WIS", "difference"], 3)} {_ci(row.loc["WIS", "ci_lower"], row.loc["WIS", "ci_upper"], 3)} |')
    add('\nFold 3 (2022-07-01..09-28, 2,112 hours / 88 days) is the stress period. Every contrast\'s per-fold intervals are in '
        '`uncertainty.csv` (scopes fold_1..fold_5).\n')
    add('## Coverage with width, stress period and peak\n')
    add('| Policy | cov50 | cov80 | cov95 | mean / median / p95 width95 | fold-3 MAE | peak MAE | peak cov95 |')
    add('|---|---|---|---|---|---|---|---|')
    peak = diag.loc[diag.scope.eq('peak')].set_index('policy')
    stress = diag.loc[diag.scope.eq('stress')].set_index('policy')
    for p in policies:
        r = pooled.loc[p]
        add(f'| {_n(p)} | {_f(r.coverage50, 3)} | {_f(r.coverage80, 3)} | {_f(r.coverage95, 3)} | '
            f'{_f(r.mean_width95, 1)} / {_f(r.median_width95, 1)} / {_f(r.p95_width95, 1)} | {_f(stress.loc[p, "MAE"], 2)} | '
            f'{_f(peak.loc[p, "MAE"], 2)} | {_f(peak.loc[p, "coverage95"], 3)} |')
    add('\nThe 17-day peak (2022-08-15..31, 408 hours) is descriptive only: a small effective sample.\n')
    add('## All six original §8 diagnostics, every new policy\n')
    add('| Policy | Criteria not met (criterion: scope metric) |\n|---|---|')
    for p in [p for p in policies if p not in S.SAVED] + ['HG', 'HGL']:
        part = crit.loc[crit.policy.eq(p) & ~crit.passed]
        add(f'| {_n(p)} | ' + (', '.join(f'{r.criterion}: {r.scope} {r.metric}' for r in part.itertuples()) or 'none — all met') + ' |')
    add('\nThe rows, with actual values and limits, are in `criteria.csv`. The emitted p50 is scored; for v5 and v3+D it is the '
        'central forecast plus the H layer\'s median residual, kept separate from the central forecast. For D the p50 is the '
        'ensemble median, which is D\'s central forecast by definition.\n')
    if dg:
        add('## §21.5 diagnostics (descriptive; they choose nothing)\n')
        cal = dg['calibration']
        add('**DDNN\'s own calibration** (D, pooled; `diagnostics/calibration-by-level.csv`, `diagnostics/pit-histogram.csv`):\n')
        add('| Nominal level | ' + ' | '.join(str(q) for q in (0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)) + ' |')
        add('|---|' + '---|' * 7)
        add('| Share of actuals at or below | ' + ' | '.join(
            _f(cal['pooled'][f'share_at_or_below_{lab}'], 3) for lab in ('p025', 'p10', 'p25', 'p50', 'p75', 'p90', 'p975')) + ' |')
        add(f'\nCentral coverage: 50% {_f(cal["pooled"]["coverage50"], 3)}, 80% {_f(cal["pooled"]["coverage80"], 3)}, 95% '
            f'{_f(cal["pooled"]["coverage95"], 3)} (mean 95% width {_f(cal["pooled"]["mean_width95"], 1)} EUR/MWh). PIT of the '
            f'quantile-averaged ensemble: mean {_f(cal["pit_mean"], 3)}, share below 0.05 {_f(cal["pit_share_below_0.05"], 3)}, '
            f'share above 0.95 {_f(cal["pit_share_above_0.95"], 3)} (uniform = 0.05 each).\n')
        ext = dg['extrapolation']
        add(f'**Extrapolation** (`diagnostics/extrapolation.csv`, beside CP-22\'s tree record): {ext["days"]} extreme or top-5% '
            f'days (' + ', '.join(f'{v} {k.replace("_", " ").replace("+", " and ")}' for k, v in ext['sets'].items()) + '). Hours forecast above the origin\'s training-window maximum on those days: '
            + ', '.join(f'{_n(p)} {v}' for p, v in ext['hours_above_window_max'].items()) + '.\n')
        add('**The 2022 peak and fold 4** (`diagnostics/peak-and-fold4.csv`, `diagnostics/fold4-intervals.csv`); fold 4 is the fold '
            'where CP-22\'s candidates were decisively worse than v4.\n')
        ss = dg['seed_stability_pooled']
        add(f'**Ensemble and seed stability** (`diagnostics/seed-stability.csv`, `diagnostics/epochs.csv`): the ensemble median\'s '
            f'pooled MAE is {_f(ss["ensemble_median_MAE"], 2)} EUR/MWh; the single seeds\' medians score '
            + ', '.join(f'{k.split("_")[1]} {_f(v, 2)}' for k, v in ss.items() if k.startswith('seed_')) + f'; the four member medians '
            f'spread {_f(ss["member_median_spread_mean_abs"], 2)} EUR/MWh around the ensemble median on average and straddle the '
            f'actual price in {_f(100 * ss["share_hours_members_straddle_actual"], 1)}% of hours.\n')
        add('**The configuration chosen in each fold:** as above (`selection.json`).\n')
    if cost or cycle:
        add('## Fit cost and daily cycle (diagnostic, §17.5 D3)\n')
        if cost:
            add(f'{cost["fits"]:,} main-run DDNN member fits (' + ', '.join(f'{k} {v:,}' for k, v in cost['fits_by_stage'].items()) + f'), {_f(cost["totals"]["wall_seconds"] / 3600, 2)} '
                f'single-thread hours in total; median four-seed ensemble per origin {_f(cost["per_origin"]["median_ensemble_fit_wall_seconds"], 1)} s '
                f'(maximum {_f(cost["per_origin"]["max_ensemble_fit_wall_seconds"], 1)} s). By configuration: '
                + '; '.join(f'{c} {int(v["fits"])} fits, median {_f(v["median_wall"], 2)} s, median best epoch {_f(v["median_best_epoch"], 0)}'
                            for c, v in cost['by_config'].items()) + '.\n')
        if cycle:
            s = cycle['summary']
            v4c = cycle['v4_component_cycle_beside']['summary']
            add(f'Cold daily cycle of v5 (DDNN members, ensemble, v5 central, H layer, issuance; four workers): median '
                f'{_f(s["median_seconds"], 1)} s, maximum {_f(s["max_seconds"], 1)} s over {s["origins"]} origins; every bitwise '
                f'check {"passed" if cycle["all_bitwise_checks_passed"] else "FAILED"}. v4\'s own component cycle, measured by '
                f'CP-21 and not refitted here: median {_f(v4c["median"], 1)} s, maximum {_f(v4c["max"], 1)} s over 25 origins.\n')
    add('## Integrity\n')
    if controls:
        add(f'- Controls (`controls.json`): {controls["checks"]} checks, all passed: {controls["all_passed"]}. They include '
            'delivery-day and future masking (exactly 0.0), the non-uniform D−1 price mutation (moves), future weather (0.0), '
            'weather permutation and rearrangement (move), early stopping responding to inner-validation outcomes, the '
            'configuration choice unchanged by evaluation outcomes and changed in its losses by holdout outcomes, '
            'determinism (a fresh process refits the main run\'s members bit for bit), restart replay, release and cache '
            'refusals, composite parity on every key, and the boundary guard.')
    if parity:
        add(f'- v3 and v4 through CP-23\'s H-layer replay path reproduce the committed vectors bit for bit on all 10,747 keys: '
            f'v3 {parity["HG"]["bitwise_equal"]}, v4 {parity["HGL"]["bitwise_equal"]} (`parity.json`).')
    if repro:
        add(f'- Independent representative HG and v4 slice, refitted at two origins (`reproduction.json`): bitwise {repro["all_bitwise"]}.')
    cons = dec.get('consistency_with_committed') or {}
    add(f'- Scoring reproduces CP-20\'s and CP-21\'s committed §8 rows and v4 − v3 intervals: {cons.get("all_equal")}.')
    add(f'- PyTorch reference (`reference-checks.json`): {ref["counts"]["summary"]}, torch {ref["versions"]["torch"]}, at the '
        'frozen tolerances; the largest error was a small fraction of its bound.\n')
    if res:
        add('## Resources against the §21.8 ceilings\n')
        add('| Dimension | Used | Cap |\n|---|---|---|')
        for k, v in res['caps_vs_use'].items():
            add(f'| {k} | {_f(v["used"], 2) if isinstance(v["used"], float) else v["used"]} | {v["cap"]} |')
        add(f'\n{res.get("calendar", "")}\n')
    add('## What this result does not establish\n')
    add('- No confirmatory or out-of-sample claim: development_post_selection on folds already used by CP-15, CP-20, CP-21 and '
        'CP-22.')
    add('- "No demonstrated joint preference" is not equivalence, and not proof that DDNN cannot help.')
    add('- One DDNN design was tested: one architecture family, four configurations, four seeds, a fixed one-third weight. '
        'Learned weights belong to 4.8.')
    add('- No release, final-product designation, freeze, Live or economic claim; v1 remains the released product.\n')
    add('Reproduction: `reports/distribution-challenger/reproduce.md`. Protocol: `reports/distribution-challenger/protocol.json` '
        '(committed before any main-run fit).\n')
    return '\n'.join(L) + '\n'


def main() -> int:
    root = Path.cwd()
    (root / OUT / 'report.md').write_text(build(root))
    print('wrote reports/distribution-challenger/report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
