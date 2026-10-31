"""CP-24's report, `reports/ddnn2/report.md`, generated from the committed evidence in every outcome. Run as
``python -m cp24.report`` (``--check`` compares with the committed file)."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .packet import outcome

OUT = Path('reports/ddnn2')
NAMES = {'v5': 'v5 = (2/3)·HG + (1/6)·L + (1/6)·D2', 'HGL': 'v4 (three-block)', 'v3+D2': 'v3 + DDNN-2', 'HG': 'v3',
         'D2': 'DDNN-2 alone', 'D': 'CP-23\'s DDNN (saved)', 'v3+D': 'v3 + CP-23\'s DDNN (saved)', 'L': 'L = mean(L-N, L-R), point only',
         'A1': 'A1 normalized LEAR', 'B2': 'B2 daily LEAR', 'B0': 'B0 naive', 'B1': 'B1 (v1)', 'B3': 'B3 daily LightGBM'}


def f(x, d=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return 'n/a'
    return f'{float(x):.{d}f}'


def build(root: Path) -> str:
    import pandas as pd
    root = Path(root)
    o = outcome(root)
    load = lambda p: json.loads((root / p).read_text())  # noqa: E731
    ref, r46, parity = load(OUT / 'reference-checks.json'), load(OUT / 'resource-admission.json'), load(OUT / 'v4-parity.json')
    L = ['# CP-24 — DDNN-2: a literature-faithful DDNN, and v5 = v4 plus a DDNN-2 member\n',
         '`capstone_v21.md` v21-r11 §23 · evidence class **development_post_selection** · population `common-10747h` '
         '(10,747 keys, five folds) · the second DDNN decision on these folds (4.7T carries the protection).\n',
         '## Verdicts\n', '| Status | Result |', '|---|---|',
         '| Engineering | Bound to the fresh Integration-Critic verdict on the final candidate (`docs/track-b/evidence/cp-24/integration.md`) |',
         f'| Entry | 4.6L′ PASS · §23.7: import audit, finite-difference checks, PyTorch reference {ref["counts"]["passed"]} of '
         f'{ref["expected_passed"]} · 4.6R′ **{r46["verdict"]}** ({r46["fixed"]["trials_per_fold_per_round"]} trials per fold per round) |']
    for r, p in o['gates_passed'].items():
        L.append(f'| Pre-fold round {r} | gate **{"PASS" if p else "FAIL"}** (`rounds/round-{r}/gate.json`, `rounds/round-{r}/report.md`) |')
    for k in o['scored_attempts']:
        a = load(OUT / f'attempt-{k}' / 'decisions.json')['adoption']
        L.append(f'| `cp24-adoption`, attempt {k} | **{a["verdict"]}** · first unmet condition **{a["first_unmet_condition"]}** '
                 f'(unmet: {", ".join(map(str, a["unmet_conditions"])) or "none"}) |')
    outcome_text = {'adopted': f'v5 adopted in research in scored attempt {o["deciding_attempt"]}',
                    'branch': 'the branch "DDNN-2 member on v4"',
                    'stopped_at_the_pre_fold_gate': 'stopped at the pre-fold gate (no fold look)'}[o['kind']]
    L += [f'| CP-24\'s outcome | **{outcome_text}** |',
          '| Research | Development finding under the rule pre-registered on 2026-10-05; DDNN-2 vectors stored for 4.8 |',
          '| Product | v1 remains the released product and the demo; no designation, freeze or Live follows (§16) |\n']
    L += ['## DDNN-2 as designed (§23.3–§23.4)\n']
    for r in o['rounds']:
        d = load(OUT / 'rounds' / f'round-{r}' / 'design.json')
        L.append(f'- **Round {r} design** (`rounds/round-{r}/design.json`): {d["trials_per_fold"]} trials per fold; '
                 f'changes from the previous round: {json.dumps(d["changes_from_previous"]) if d["changes_from_previous"] else "none"}.')
    rec = load(OUT / 'rounds' / f'round-{o["rounds"][0]}' / 'design.json')['recipe'] if o['rounds'] else {}
    for key, text in rec.items():
        L.append(f'- **{key.replace("_", " ").capitalize()}:** {text}.')
    L.append('')
    L += ['## The pre-fold rounds\n', '| Round | G0 | G1 MAE v5 / v4 | G2 D2 / L | G3 cap share | Gate |', '|---|---|---|---|---|---|']
    for r in o['rounds']:
        g = load(OUT / 'rounds' / f'round-{r}' / 'gate.json')
        c = g['conditions']
        L.append(f'| {r} | {c["G0"]["met"]} | {f(c["G1"]["v5"], 3)} / {f(c["G1"]["v4"], 3)} | {f(c["G2"]["ratio"], 4)} | '
                 f'{100 * c["G3"]["share"]:.3f}% | {"PASS" if g["passed"] else "FAIL"} |')
    L.append(f'\nv4\'s gate code path: bit-for-bit parity at {len(parity["origins"])} covered origins and the unwrapped code\'s refusal '
             f'of a fold-4 gate day (`v4-parity.json`).\n')
    for k in o['scored_attempts']:
        A = OUT / f'attempt-{k}'
        dec = load(A / 'decisions.json')
        metrics = pd.read_csv(root / A / 'metrics.csv')
        unc = pd.read_csv(root / A / 'uncertainty.csv')
        L += [f'## Scored attempt {k}\n', f'Frozen protocol `attempt-{k}/protocol.json`; vectors `attempt-{k}/predictions.parquet`, '
              f'`attempt-{k}/members.parquet`; tables `metrics.csv`, `uncertainty.csv`, `criteria.csv`.\n',
              '### Scores (equal-fold ratios to B0; lower is better)\n',
              '| Policy | S_MAE | S_WIS | pooled MAE | pooled WIS | pooled 95% coverage | mean 95% width | §8 |', '|---|---|---|---|---|---|---|---|']
        eqm = metrics.loc[metrics.scope.eq('equal_fold')].set_index('policy')
        pooled = metrics.loc[metrics.scope.eq('pooled')].set_index('policy')
        for p in ('v5', 'HGL', 'v3+D2', 'HG', 'D2', 'D', 'v3+D', 'A1', 'B2', 'B0', 'B1', 'B3'):
            s8 = dec['original_section8_status'].get(p, 'saved reference')
            L.append(f'| {NAMES.get(p, p)} | {f(eqm.loc[p, "S_MAE"])} | {f(eqm.loc[p, "S_WIS"])} | {f(pooled.loc[p, "MAE"], 2)} | '
                     f'{f(pooled.loc[p, "WIS"], 2)} | {f(pooled.loc[p, "coverage95"], 3)} | {f(pooled.loc[p, "mean_width95"], 2)} | {s8} |')
        L.append(f'| {NAMES["L"]} | {f(eqm.loc["L", "S_MAE"])} | n/a | {f(pooled.loc["L", "MAE"], 2)} | n/a | n/a | n/a | n/a |\n')
        a = dec['adoption']
        L.append('### `cp24-adoption` (§23.9), condition by condition\n')
        for c, v in sorted(a['conditions'].items(), key=lambda kv: int(kv[0])):
            L.append(f'{c}. {a["rule"]["conditions"][str(c)] if str(c) in a["rule"]["conditions"] else a["rule"]["conditions"][int(c)]} '
                     f'→ **{"met" if v["met"] else "not met"}**.')
        L.append(f'\n**Decision:** {a["verdict"]}.\n')
        ri = pd.read_csv(root / A / 'ratio-intervals.csv', float_precision='round_trip').set_index(['candidate', 'baseline', 'metric'])

        def ratio(c_, b_, m):
            x = ri.loc[(c_, b_, m)]
            if not bool(x.defined):
                return 'n/a'
            return (f'{100 * x.ratio:+.2f}% [{100 * x.ratio_ci95_lower:+.2f}%, {100 * x.ratio_ci95_upper:+.2f}%] '
                    f'[{100 * x["ratio_ci97.5_lower"]:+.2f}%, {100 * x["ratio_ci97.5_upper"]:+.2f}%]')
        L += ['### Every §23.8 contrast, with its reading\n',
              'Ratios are R = S_candidate / S_comparator − 1, with their 95% and 97.5% (decision-level) intervals from the stored '
              f'equal-fold draws (`attempt-{k}/ratio-intervals.csv`).\n',
              '| Contrast | Role | dS_MAE [95%] [97.5%] | dS_WIS [95%] [97.5%] | ratio S_MAE [95%] [97.5%] | ratio S_WIS [95%] [97.5%] | '
              'Reading |', '|---|---|---|---|---|---|---|']
        for key, r in dec['contrasts'].items():
            e = unc.loc[unc.scope.eq('equal_fold')]
            c, b = key.split('-', 1) if key.count('-') == 1 else (key.rsplit('-', 1)[0], key.rsplit('-', 1)[1])
            rows = {m: e.loc[(e.candidate + '-' + e.baseline).eq(key) & e.metric.eq(m)].iloc[0] for m in ('MAE', 'WIS')}
            m_, w_ = rows['MAE'], rows['WIS']
            wis = (f'{f(w_.difference)} [{f(w_.ci_lower)}, {f(w_.ci_upper)}] [{f(w_["ci97.5_lower"])}, {f(w_["ci97.5_upper"])}]'
                   if bool(w_.defined) else 'not defined')
            L.append(f'| {key} | {r["role"]} | {f(m_.difference)} [{f(m_.ci_lower)}, {f(m_.ci_upper)}] [{f(m_["ci97.5_lower"])}, '
                     f'{f(m_["ci97.5_upper"])}] | {wis} | {ratio(m_.candidate, m_.baseline, "MAE")} | '
                     f'{ratio(w_.candidate, w_.baseline, "WIS")} | {r["reading"]} |')
        diag = load(A / 'diagnostics.json')
        g = diag['guards']
        L += ['', '### Diagnostics (descriptive; `attempt-%d/diagnostics/`)\n' % k,
              f'- D2\'s calibration (pooled): 50/80/95% coverage {f(diag["calibration"]["pooled"]["coverage50"], 3)} / '
              f'{f(diag["calibration"]["pooled"]["coverage80"], 3)} / {f(diag["calibration"]["pooled"]["coverage95"], 3)}; PIT below '
              f'0.05 {f(diag["calibration"]["pit_share_below_0.05"], 3)}, above 0.95 {f(diag["calibration"]["pit_share_above_0.95"], 3)}.',
              '- Pooled error correlations: ' + ', '.join(f'{x} {f(v, 3)}' for x, v in diag['error_correlations_pooled'].items()) + '.',
              f'- Guards: {g["cap_slot_levels_forecast"]:,} capped slot-levels, {g["winsor_values_forecast"]:,} winsorised forecast inputs, '
              f'{g["ensemble_crossings_restored"]} crossings restored, {g["nonfinite_loss_stops"]} nonfinite-loss stops, over '
              f'{g["member_fits"]:,} member fits.',
              '- Guards by fold (every attempt fit; the cap\'s share of the members\' emitted hour-levels, the gate\'s G3 '
              'measure and never a condition, beside the round\'s gate share; `diagnostics/guards-by-fold.csv`, '
              '`diagnostics/guards-by-member.csv`): '
              + '; '.join(f'{fold} {100 * v["cap_share"]:.3f}% (gate {100 * v["gate_cap_share_round"]:.3f}%), '
                          f'{v["member_fits_with_cap"]:,} of {v["member_fits"]:,} member fits capped'
                          for fold, v in diag['guards_by_fold'].items()) + '.',
              f'- Extrapolation: {diag["extrapolation"]["days"]} extreme or top-5% days; hours above the window maximum '
              + ', '.join(f'{p} {v}' for p, v in diag['extrapolation']['hours_above_window_max'].items()) + '.',
              '- Shape blend (descriptive, never eligible): ' + '; '.join(f'{x["policy"]} WIS {f(x["WIS"], 2)}, 95% coverage '
                                                                         f'{f(x["coverage95"], 3)}' for x in diag['shape_blend']['pooled']) + '.',
              '- Also: MAE by local hour, member stability, the 2022 peak, folds 3 and 4, the search beside the folds '
              '(`attempt-%d/diagnostics/*.csv`); fit cost and the cold daily cycle (`fit-cost.json`, `daily-cycle.json`).\n' % k]
        if (A / 'controls.json').exists():
            ctl = load(A / 'controls.json')
            L += ['### Causal and integrity controls (§23.10; verification only, nothing rescored)\n',
                  f'- `attempt-{k}/controls.json`: {ctl["checks"]} checks, {"all passed" if ctl["all_passed"] else "NOT all passed"}. The masks, the weather and '
                  'the recency controls at two representative origins (' + ', '.join(f'{o["fold"]} {o["day"]}' for o in ctl['origins'])
                  + '); the search and gate controls for one fold each; pre-registration, states, caches, the population and the code.']
            if (A / 'leakage-controls.json').exists():
                lk = load(A / 'leakage-controls.json')
                sg, st, pos = lk['search_and_gate_by_fold'], lk['structure'], lk['positives']
                L += [f'- `attempt-{k}/leakage-controls.json` (`leakage-by-origin.csv`), extended because D2 alone is far ahead of v4: '
                      f'at all {lk["blind"]["origins"]} origins ({lk["blind"]["by_stage"]["evaluation"]["origins"]} evaluation, '
                      f'{lk["blind"]["by_stage"]["warmup"]["origins"]} warm-up), the frozen eight-member ensemble refitted with every '
                      'outcome on or after the delivery day destroyed reproduces the committed D2 vectors bit for bit at '
                      f'{lk["blind"]["bitwise"]} of {lk["blind"]["origins"]}, every member at its recorded weights; '
                      f'{lk["blind"]["committed_predictions"]["keys_compared"]:,} evaluation keys equal `predictions.parquet`.',
                      f'- Paired positives in every fold: a D-1 evening price mutation moves the ensemble at {len(pos["d1_evening_prices"])} '
                      f'of {len(pos["d1_evening_prices"])} origins (smallest move '
                      f'{f(min(r["max_abs_difference"] for r in pos["d1_evening_prices"]), 2)} EUR/MWh); a planted one-day leak is '
                      f'detected in {sum(r["leak_detected_max_abs_difference"] > lk["threshold_eur_mwh"] for r in pos["planted_one_day_leak"])} '
                      f'of {len(pos["planted_one_day_leak"])} folds.',
                      f'- Search and gate, fold by fold: outcomes from D0-56 change no rank-1 search result in '
                      f'{sum(v["search"]["negative_identical"] for v in sg.values())} of {len(sg)} folds, outcomes from D0 change no gate '
                      f'result in {sum(v["gate"]["negative_identical"] for v in sg.values())} of {len(sg)}, and each paired positive moves. '
                      f'Structure: {st["members_checked"]:,} member windows, {st["search_batches_checked"]:,} search batches and the gate '
                      f'days, problems {len(st["window_problems"]) + len(st["search_batch_problems"]) + len(st["gate_problems"])}.',
                      f'- {len(lk["checks"])} leakage checks, {"all passed" if lk["all_passed"] else "NOT all passed"}. The frozen code is guarded as '
                      '`check_protocol` guards it, except for the Owner\'s recorded work-availability maintenance of '
                      + ' and '.join(f'`{n}`' for n in lk['frozen_guard']['maintenance_exceptions']) + ' (calendar gate removed).\n']
            else:
                L[-1] += '\n'
    if (root / OUT / 'resources.json').exists():
        res = load(OUT / 'resources.json')
        cv = res['caps_vs_use']
        L += ['## Resources (§23.11; at the candidate, before the review)\n', '| Ceiling | Cap | Used |', '|---|---|---|']
        for key in ('ddnn2_fits', 'v4_gate_origins', 'policy_days', 'reference_passes', 'bootstrap_passes', 'scored_attempts',
                    'rounds_before_attempt_1', 'rounds_before_attempt_2', 'machine_seconds', 'active_seconds', 'rss_bytes',
                    'additional_disk_bytes', 'workers', 'data_download_bytes', 'remote_writes'):
            v = cv[key]
            if key.endswith('seconds'):
                L.append(f'| {key.replace("_seconds", "_hours")} | {v["cap"] / 3600:.0f} | {v["used"] / 3600:.2f} |')
            elif key.endswith('bytes') and key not in ('data_download_bytes',):
                L.append(f'| {key} (GiB) | {v["cap"] / 1024**3:.0f} | {v["used"] / 1024**3:.2f} |')
            else:
                L.append(f'| {key} | {v["cap"]} | {v["used"]} |')
        L.append(f'\nRaises (§23.6): {res["raises"] or "none"}. Authorized work may run at any time; '
                 'resource accounting and hard caps remain enforced.\n')
    L += ['## Reproduction\n', 'See [`reproduce.md`](reproduce.md).\n']
    return '\n'.join(L) + '\n'


def main() -> int:
    import sys
    root = Path.cwd()
    text = build(root)
    path = root / OUT / 'report.md'
    if '--check' in sys.argv[1:]:
        same = path.exists() and path.read_text() == text
        print(json.dumps({'report_identical_to_committed': same}), flush=True)
        return 0 if same else 11
    path.write_text(text)
    print(f'wrote {path}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
