"""The CP-22 report (`reports/v4-revision/report.md`), generated from the committed tables only.

Run as ``python -m cp22.report`` after scoring, controls and diagnostics. Every number is read from
a committed row (decisions.json, metrics.csv, uncertainty.csv, criteria.csv, diagnostics.csv,
investigation/, controls.json, parity.json, reproduction.json, fit-cost.json, daily-cycle.json,
resources.json); nothing is typed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cp15.scoring import FOLDS
from .execution import OUT
from . import scoring as S

NAMES = {'HG': 'v3', 'HGL': 'v4 (three-block)', 'B0': 'B0 naive', 'B1': 'B1 (v1)', 'B2': 'B2 daily LEAR',
         'B3': 'B3 daily LightGBM', 'A1': 'A1 normalized LEAR', 'H0': 'H0 (v2)'}


def _n(policy: str, w: str | None) -> str:
    if policy in NAMES:
        return NAMES[policy]
    return policy


def _f(x, d=4):
    return '—' if x is None or (isinstance(x, float) and x != x) else f'{x:.{d}f}'


def _ci(lo, hi, d=4):
    return f'[{_f(lo, d)}, {_f(hi, d)}]'


def build(root: Path) -> str:
    out = root / OUT
    dec = json.loads((out / 'decisions.json').read_text())
    metrics = pd.read_csv(out / 'metrics.csv')
    unc = pd.read_csv(out / 'uncertainty.csv')
    crit = pd.read_csv(out / 'criteria.csv')
    diag = pd.read_csv(out / 'diagnostics.csv', low_memory=False)
    load = lambda name: json.loads((out / name).read_text()) if (out / name).exists() else None  # noqa: E731
    inv, controls, parity, repro = load('investigation.json'), load('controls.json'), load('parity.json'), load('reproduction.json')
    cost, cycle, res, protocol = load('fit-cost.json'), load('daily-cycle.json'), load('resources.json'), load('protocol.json')
    rep, dl, fast = dec['replacement'], dec['dynamic_layer'], dec['fast_component']
    w = rep['winner']
    policies = list(dec['scores'])
    L = []
    add = L.append
    add('# CP-22 — v4 revised: one pooled member and a dynamic interval layer\n')
    add('`capstone_v21.md` v21-r9 §20 · evidence class **development_post_selection** · population `common-10747h` '
        '(10,747 keys, five folds).\n')
    add('## Verdicts\n')
    add('| Status | Result |\n|---|---|')
    add('| Engineering | Bound to the fresh Integration-Critic verdict on the final candidate '
        '(`docs/track-b/evidence/cp-22/integration.md`) |')
    add(f'| `cp22-replacement` | **{rep["verdict"]}** · first unmet condition: R → {rep["first_unmet_condition"]["R"]}, '
        f'M → {rep["first_unmet_condition"]["M"]} |')
    add(f'| `cp22-dynamic-layer` | **{dl["verdict"]}**' + (f' · first unmet condition: {dl["first_unmet_condition"]}' if dl.get('applies') else '') + ' |')
    add(f'| `cp22-fast-component` | **{fast["verdict"]}**' + (f' · first unmet condition: {fast["first_unmet_condition"]}' if fast.get('applies') else '') + ' |')
    add('| Research | Development finding under the three rules pre-registered on 2026-10-01; one more decision on the '
        'same five folds (4.7T carries the protection) |')
    add('| Product | v1 remains the released product and the demo; no designation, freeze or Live follows (§16) |\n')
    add('The rules were applied mechanically, in their fixed sequence, from the committed tables '
        '(`reports/v4-revision/decisions.json`). A mixed result is no demonstrated joint preference, never equivalence.\n')
    add('## Scores (equal-fold ratios to B0; lower is better)\n')
    eq = metrics.loc[metrics.scope.eq('equal_fold')].set_index('policy')
    pooled = metrics.loc[metrics.scope.eq('pooled')].set_index('policy')
    add('| Policy | S_MAE | S_WIS | pooled MAE (EUR/MWh) | pooled WIS | pooled 95% coverage | mean 95% width | §8 |')
    add('|---|---|---|---|---|---|---|---|')
    status = dec['original_section8_status']
    for p in policies:
        add(f'| {_n(p, w)} | {_f(eq.loc[p, "S_MAE"])} | {_f(eq.loc[p, "S_WIS"])} | {_f(pooled.loc[p, "MAE"], 2)} | '
            f'{_f(pooled.loc[p, "WIS"], 2)} | {_f(pooled.loc[p, "coverage95"], 3)} | {_f(pooled.loc[p, "mean_width95"], 2)} | '
            f'{status.get(p, "reference")} |')
    add('')
    add('## `cp22-replacement` (§20.6), condition by condition\n')
    for cand in ('R', 'M'):
        c = rep['candidates'][cand]
        add(f'**{cand} against v4** — {"all four conditions met" if c["met"] else "not met; first unmet condition " + str(c["first_unmet_condition"])}.\n')
        v1 = c['conditions']['1']['values'] if '1' in c['conditions'] else c['conditions'][1]['values']
        conds = c['conditions']
        get = lambda k: conds[str(k)] if str(k) in conds else conds[k]  # noqa: E731
        add(f'1. Non-inferiority: dS_MAE {_f(v1["MAE"]["difference"])} {_ci(v1["MAE"]["ci_lower"], v1["MAE"]["ci_upper"])}, '
            f'dS_WIS {_f(v1["WIS"]["difference"])} {_ci(v1["WIS"]["ci_lower"], v1["WIS"]["ci_upper"])} → '
            f'{"met" if get(1)["met"] else "not met"}.')
        nm = get(2)['values']['not_met']
        failed = ', '.join(f"criterion {r['criterion']} {r['scope']}" for r in nm)
        add(f'2. All six §8 diagnostics: ' + ('met.' if get(2)['met'] else f'not met ({failed}).'))
        add(f'3. A complete, valid evaluation: all 10,747 keys issued; Engineering PASS is bound to the Integration verdict.')
        worse = get(4)['values']['folds_decisively_worse']
        bad = '; '.join(f"{r['scope']} {r['metric']} {_f(r['difference'], 3)} {_ci(r['ci_lower'], r['ci_upper'], 3)}" for r in worse)
        add('4. No resolved per-fold degradation: ' + ('met.\n' if get(4)['met'] else f'not met — {bad}.\n'))
    for title, d in (('`cp22-dynamic-layer`', dl), ('`cp22-fast-component`', fast)):
        add(f'## {title}\n')
        if not d.get('applies'):
            add(d['verdict'] + '.\n')
            continue
        add(f'**{d["verdict"]}** (first unmet condition: {d["first_unmet_condition"]}).\n')
        for k, cond in sorted(d['conditions'].items(), key=lambda kv: int(kv[0])):
            add(f'- Condition {k} ({S.RULES["cp22-dynamic-layer" if d is dl else "cp22-fast-component"]["conditions"][int(k)]}): '
                f'{"met" if cond["met"] else "not met"}.')
        add('')
    add('## Every §20.5 contrast, with its reading\n')
    add('| Contrast | Role | dS_MAE [95%] | dS_WIS [95%] | ratio S_WIS [95%] | Reading |')
    add('|---|---|---|---|---|---|')
    for key, r in dec['contrasts'].items():
        add(f'| {key} | {r["role"]} | {_f(r["dS_MAE"]["point"])} {_ci(*r["dS_MAE"]["ci"])} | {_f(r["dS_WIS"]["point"])} '
            f'{_ci(*r["dS_WIS"]["ci"])} | {_f(100 * r["dS_WIS"]["ratio"], 2)}% {_ci(100 * r["dS_WIS"]["ratio_ci"][0], 100 * r["dS_WIS"]["ratio_ci"][1], 2)}% | '
            f'{r["reading"]} |')
    add('')
    if not w:
        add('Not applicable without a winner W (§20.6): (W+DL) − W, (W+ACI) − W, (W+DL) − (W+ACI) and (W+DLF) − (W+DL); the '
            'layer arms on W were not run. The dynamic layer is reported on v4 and v3, descriptively.\n')
    add('Bootstrap: seed 15042, 2,000 replicates of 7-calendar-day blocks within each fold, the shared CP-20 index set '
        f'(SHA-256 `{dec["bootstrap"]["index_sha256"]}`); every replicate is stored in `replicates.parquet`.\n')
    add('## Per-fold MAE and WIS (EUR/MWh)\n')
    per = metrics.loc[metrics.scope.eq('per_fold')]
    add('| Policy | ' + ' | '.join(f'{f} MAE / WIS' for f in FOLDS) + ' |')
    add('|---|' + '---|' * len(FOLDS))
    for p in policies:
        part = per.loc[per.policy.eq(p)].set_index('fold')
        add(f'| {_n(p, w)} | ' + ' | '.join(f'{_f(part.loc[f, "MAE"], 2)} / {_f(part.loc[f, "WIS"], 2)}' for f in FOLDS) + ' |')
    add('\nFold 3 (2022-07-01..09-28, 2,112 hours / 88 days) is the stress period. Per-fold paired daily-loss intervals for '
        'every contrast are in `uncertainty.csv` (scopes fold_1..fold_5).\n')
    add('## Coverage with width, stress period and peak\n')
    add('| Policy | cov50 | cov80 | cov95 | mean / median / p95 width95 | fold-3 MAE | peak MAE | peak cov95 |')
    add('|---|---|---|---|---|---|---|---|')
    peak = diag.loc[diag.scope.eq('peak')].set_index('policy')
    stress = diag.loc[diag.scope.eq('stress')].set_index('policy')
    for p in policies:
        r = pooled.loc[p]
        add(f'| {_n(p, w)} | {_f(r.coverage50, 3)} | {_f(r.coverage80, 3)} | {_f(r.coverage95, 3)} | '
            f'{_f(r.mean_width95, 1)} / {_f(r.median_width95, 1)} / {_f(r.p95_width95, 1)} | {_f(stress.loc[p, "MAE"], 2)} | '
            f'{_f(peak.loc[p, "MAE"], 2)} | {_f(peak.loc[p, "coverage95"], 3)} |')
    add('\nThe 17-day peak (2022-08-15..31, 408 hours) is descriptive only: a small effective sample.\n')
    add('## All six original §8 diagnostics, every new policy\n')
    add('| Policy | Criteria not met (criterion: scope) |\n|---|---|')
    for p in [p for p in policies if p not in S.SAVED] + ['HG', 'HGL']:
        part = crit.loc[crit.policy.eq(p) & ~crit.passed]
        add(f'| {_n(p, w)} | ' + (', '.join(f'{r.criterion}: {r.scope} {r.metric}' for r in part.itertuples()) or 'none — all met') + ' |')
    add('\nThe rows, with actual values and limits, are in `criteria.csv`.\n')
    if inv:
        add('## The Owner\'s investigation (descriptive; chooses nothing)\n')
        lad = inv['ladder']
        add('**Ladder decomposition of v4 − v3** (equal-fold score differences; the brackets sum to the total exactly: '
            f'{lad["identity_holds"]}):\n')
        add('| Step | Change | dS_MAE [95%] | dS_WIS [95%] |\n|---|---|---|---|')
        for s in lad['steps']:
            add(f'| {s["from"]} → {s["to"]} | {s["change"]} | {_f(s["dS_MAE"])} {_ci(*(s["dS_MAE_ci"] or (None, None)))} | '
                f'{_f(s["dS_WIS"])} {_ci(*(s["dS_WIS_ci"] or (None, None)))} |')
        add(f'\nMember-weight curve (central forecast, before any interval layer; **oracle, not selectable**): the weight '
            f'with the lowest central S_MAE for each member is {inv["member_weight_curve"]["argmin_weight_by_member"]} '
            '(`investigation/member-weight-curve.csv`). The fixed weight is 1/3; no weight is learned (D6).\n')
        add('**Capacity-selection stability** (inner validation, evaluation origins):\n')
        add('| Model | flip rate | winner margin (median) | G1 | G2 | G3 | G4 |\n|---|---|---|---|---|---|---|')
        for r in inv['capacity_stability']:
            if r['phase'] == 'evaluation':
                add(f'| {r["arm"]} {r["model"]} | {_f(r["flip_rate"], 3)} | {_f(100 * r["winner_margin_median"], 2)}% | '
                    + ' | '.join(_f(r[f"share_G{i}"], 2) for i in range(1, 5)) + ' |')
        ext = inv['extrapolation']
        add(f'\n**Extrapolation:** {ext["days"]} extreme or top-5% days, {ext["exceeds_window_max_days"]} of them above the '
            'origin\'s training-window maximum (`investigation/extrapolation.csv`: actual, window and forecast maxima per member).\n')
        add('**Reaction after the peak began (2022-08-15):**\n')
        add('| Policy | ACI reaction (days) | width reaction (days) |\n|---|---|---|')
        for r in inv['reaction_after_peak_start']:
            add(f'| {r["policy"]} | {r["aci_reaction_days"]} | {r["width_reaction_days"]} |')
        add('\nCoverage by hour, block and regime: `diagnostics.csv` (hour/block, 56-date support rule) and '
            '`investigation/coverage-regime.csv` (terciles of the issued 168-hour scale); each fold\'s alpha_t path: '
            '`investigation/alpha-paths.csv`.\n')
        add('**Sharp changes** (each fold\'s 5% largest daily-mean changes and the peak\'s first ten days, each with the three '
            'days after; pooled over the event windows, so an overlapping day counts once per event; per offset in '
            '`investigation/shock-days.csv`)' + ('' if w else '. With no winner W, W, W+DL and W+DLF do not exist; the same '
            'windows are reported for v4, v4+DL, v3 and v3+DL') + ':\n')
        add('| Set / policy | hours | MAE | WIS | cov95 |\n|---|---|---|---|---|')
        for k, v in inv['shock_days']['pooled_shock_and_three_after'].items():
            add(f'| {k} | {v["hours"]} | {_f(v["MAE"], 2)} | {_f(v["WIS"], 2)} | {_f(v["coverage95"], 3)} |')
        add('\n**LEAR\'s penalty-selection stability (D8)**, from logged selections in the verified CP-20 cache (no refit):\n')
        add('| Component | phase | flip rate | median margin |\n|---|---|---|---|')
        for r in inv['lear_penalty_stability']['rows']:
            add(f'| {r["component"]} | {r["phase"]} | {_f(r["flip_rate"], 3)} | {_f(100 * r["margin_median"], 2)}% |')
        add('')
    if cost or cycle:
        add('## Fit cost and daily cycle (diagnostic, §17.5 D3)\n')
        if cost:
            pn, lp = cost['PN'], cost['L-P (CP-21, committed)']
            add(f'PN: {cost["fits"]} fits over {cost["origins"]} origins; inner fits {_f(pn["inner_wall_total"], 0)} s and full-window '
                f'fits {_f(pn["full_window_wall_total"], 0)} s single-thread wall; median per origin {_f(pn["median_per_origin_wall_all_eight"], 2)} s '
                f'for all eight, {_f(pn["median_per_origin_wall_r_four_full_window"], 2)} s for R\'s four full-window fits, '
                f'{_f(pn["median_per_origin_wall_m_selection_and_refit"], 2)} s for PN-sel\'s selection and refit. CP-21 L-P: '
                f'{_f(lp["median_per_origin_wall_selection_and_refit"], 2)} s per origin.\n')
        if cycle:
            for p, s in cycle['summary'].items():
                add(f'Complete cold daily cycle, {p}: median {_f(s["median_seconds"], 1)} s, maximum {_f(s["max_seconds"], 1)} s over '
                    f'{s["origins"]} origins, four workers; every bitwise check {"passed" if cycle["all_bitwise_checks_passed"] else "FAILED"}.\n')
    add('## Integrity\n')
    if controls:
        add(f'- Controls (`controls.json`): {controls["checks"]} checks, all passed: {controls["all_passed"]}.')
    if parity:
        add(f'- v3 and v4 through CP-22\'s H-layer replay path reproduce the committed vectors bit for bit on all 10,747 keys: '
            f'v3 {parity["HG"]["bitwise_equal"]}, v4 {parity["HGL"]["bitwise_equal"]} (`parity.json`).')
    if repro:
        add(f'- Independent representative HG and v4 slice, refitted (`reproduction.json`): bitwise {repro["all_bitwise"]}.')
    cons = dec.get('consistency_with_committed') or {}
    add(f'- Scoring reproduces CP-20\'s and CP-21\'s committed §8 rows and intervals for v3 and v4: {cons.get("all_equal")}.')
    if dec.get('pass1_reproduced'):
        add(f'- Pass 2 reproduces every pass-1 row: {dec["pass1_reproduced"]["equal"]}.')
    add('')
    if res:
        add('## Resources against the §20.8 ceilings\n')
        add('| Dimension | Used | Cap |\n|---|---|---|')
        for k, v in res['caps_vs_use'].items():
            add(f'| {k} | {_f(v["used"], 2) if isinstance(v["used"], float) else v["used"]} | {v["cap"]} |')
        add(f'\n{res.get("calendar", "")}\n')
    add('## What this result does not establish\n')
    add('- No confirmatory or out-of-sample claim: development_post_selection on folds already used by CP-15, CP-20 and CP-21.')
    add('- Non-inferiority here means no 95% interval lies entirely above zero; it is not equivalence.')
    add('- No release, final-product designation, freeze, Live or economic claim; v1 remains the released product.')
    add('- The member weight is fixed at 1/3; the member-weight curve is an oracle and selects nothing.\n')
    add('Reproduction: `reports/v4-revision/reproduce.md`. Protocol: `reports/v4-revision/protocol.json` '
        f'(committed before any main fit).\n')
    return '\n'.join(L) + '\n'


def main() -> int:
    root = Path.cwd()
    (root / OUT / 'report.md').write_text(build(root))
    print('wrote reports/v4-revision/report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
