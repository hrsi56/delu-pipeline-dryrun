"""CP-24's claim map and publication packet (capstone v21-r11 §23.12; packet template sections 1-8), in every
outcome: a stop at the pre-fold gate is recorded as such.

Run as ``python -m cp24.claims`` after the draft registry (``python -m cp24.packet``) and, if any attempt was
scored, its diagnostics and the draft export (``--check`` regenerates and compares, writing nothing). Every value
and row reference (`L<n>`, the physical line; `L1` is a CSV's header) is read from the committed files at
generation time. Draft slot texts carry registry tokens (`{g:<id>.<field>}`) and evidence tokens
(`{r:<record>}`), checked with the publication lint's code and status-word rules. Only the outcome the rule
yields is drafted; identities that exist only at landing stay explicit pending fields.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from cp22.claims import Rows, num, pct, resolved_note, _source_row
from delu_forecast import publication_lint as PL
from .packet import outcome

OUT = Path('reports/ddnn2')
CLAIMS = Path('docs/track-b/research-content/cp24-claims.md')
PACKET = Path('docs/track-b/evidence/cp-24/publication-packet.md')
FOLDS = ('fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5')
INTERNAL_CODES = ('HGL', 'HG', 'H0', 'B0', 'B1', 'B2', 'B3', 'A1', 'A2', 'L-P', 'L-R', 'L-N', 'PN', 'D', 'D2', 'v3+D', 'v3+D2',
                  'JSU')
DISPLAY = {'HG': 'v3', 'HGL': 'v4', 'D2': 'DDNN-2 alone', 'D': 'CP-23\'s DDNN', 'v3+D2': 'v3 + DDNN-2', 'v3+D': 'v3 + CP-23\'s DDNN'}


def lint_slot(text: str) -> list[str]:
    plain = re.sub(r'\{[gr]:[^{}]+\}', '', text)
    findings = PL.code_findings(PL.reading_path(f'<p>{plain}</p>'))
    findings += [f'status word {m.group(0)!r}' for m in PL.STATUS_WORDS.finditer(plain)]
    findings += [f'internal code {c!r}' for c in INTERNAL_CODES if re.search(rf'(?<![\w+-]){re.escape(c)}(?![\w+-])', plain)]
    return findings


def _load(root, path):
    return json.loads((Path(root) / path).read_text())


def build(root: Path):
    root = Path(root)
    o = outcome(root)
    registry = _load(root, OUT / 'draft-registry.json')
    ref, r46 = _load(root, OUT / 'reference-checks.json'), _load(root, OUT / 'resource-admission.json')
    parity = _load(root, OUT / 'v4-parity.json')
    claims = []

    def claim(ident, text, source, status):
        claims.append(f'| {ident} | {text} | {source} | {status} |')

    # ---- identity, rule, scope, entry
    claim('C400', 'CP-24\'s Integration verdict and the reviewed final candidate are recorded in I24 and the CP-24 checkpoint '
                  'return; the evidence tag and landing record are pending at landing.', 'I24; RET24', 'S (pending)')
    claim('C401', 'The rule `cp24-adoption` was set on 2026-10-05 (v21-r11, ratified under the Owner\'s delegation) and frozen, '
                  'verbatim, in each scored attempt\'s protocol before any of its warm-up or evaluation fits.',
          'CAP §23.9; P24 `rule_verbatim`', 'S')
    claim('C402', 'No product change, promotion, freeze, Live, final-product designation or economic claim follows; v1 is the '
                  'released product and the demo.', 'CAP §23.9 (what adoption means)', 'S')
    claim('C403', 'DDNN-2 is a NumPy-only feed-forward network on one row per delivery day with 24 Johnson SU heads, exactly v4\'s '
                  'sources in day-level form; every member trains on [max(2019-01-01, D−728), D) minus its own random 20% of whole '
                  'calendar weeks (never the last 7 days), early-stopped on the seven-level pinball loss; NLL for 20 epochs, then '
                  'κ·NLL + (1−κ)·pinball over 19 levels; four tuned configurations × two seeds combined by the per-level median.',
          'CAP §23.3; RD24 `recipe`', 'Def')
    claim('C404', f'Entry: 4.6L′ PASS (provenance; CP-23\'s reference reused); the import audit, the finite-difference checks and the '
                  f'PyTorch reference ({ref["counts"]["passed"]} of {ref["expected_passed"]} checks, torch {ref["versions"]["torch"]}, '
                  f'tolerances frozen before the first run) passed; 4.6R′ on pre-fold data: **{r46["verdict"]}**, fixing '
                  f'{r46["fixed"]["trials_per_fold_per_round"]} trials per fold per round.', 'LIC24; REF24; RA24; CAP §23.7', 'S')
    claim('C405', 'v4\'s members at the gate days run their unchanged code; their code path, with the weather-coverage wrapper, '
                  'reproduces the committed A1_w, B2_w, L-N, L-R, v3 and v4 vectors bit for bit at '
                  f'{len(parity["origins"])} covered origins, and the unwrapped code refuses a fold-4 gate day.', 'PAR24', 'S')
    # ---- rounds and gates
    for r in o['rounds']:
        gate = _load(root, OUT / 'rounds' / f'round-{r}' / 'gate.json')
        c = gate['conditions']
        claim(f'C41{r}', f'Pre-fold round {r}: gate **{"passed" if gate["passed"] else "not passed"}** on 280 pre-fold days — '
                         f'G0 {"met" if c["G0"]["met"] else "not met"}; G1 MAE(v5) {num(c["G1"]["v5"], 3)} vs MAE(v4) '
                         f'{num(c["G1"]["v4"], 3)} EUR/MWh ({"met" if c["G1"]["met"] else "not met"}); G2 MAE(DDNN-2)/MAE(L) '
                         f'{num(c["G2"]["ratio"], 3)} vs 1.10 ({"met" if c["G2"]["met"] else "not met"}); G3 cap share '
                         f'{100 * c["G3"]["share"]:.3f}% vs 0.1% ({"met" if c["G3"]["met"] else "not met"}). A screen, not a claim.',
              f'GATE24-{r}', 'S-d')
    steering = sorted((root / 'docs/track-b/evidence/cp-24/steering').glob('*.md')) if (root / 'docs/track-b/evidence/cp-24/steering').exists() else []
    claim('C419', f'Steering: {len(steering)} committed exchange file(s) under `docs/track-b/evidence/cp-24/steering/`; '
                  f'{len(o["rounds"])} pre-fold round(s) and {len(o["scored_attempts"])} scored attempt(s).', 'STEER24', 'S')
    ctx = {'o': o, 'registry': registry}
    if o['scored_attempts']:
        k = o['deciding_attempt']
        A = OUT / f'attempt-{k}'
        M, U, C = Rows(root, str(A / 'metrics.csv')), Rows(root, str(A / 'uncertainty.csv')), Rows(root, str(A / 'criteria.csv'))
        D = Rows(root, str(A / 'diagnostics.csv'))
        dec = _load(root, A / 'decisions.json')
        a = dec['adoption']

        def eq(c, b, m):
            return U.find(scope='equal_fold', candidate=c, baseline=b, metric=m)

        RI = Rows(root, str(A / 'ratio-intervals.csv'))

        def ri(c, b, m):
            return RI.find(candidate=c, baseline=b, metric=m)

        def pf(p, fo):
            return M.find(policy=p, scope='per_fold', fold=fo)

        def score(p):
            return M.find(policy=p, scope='equal_fold')

        def contrast_claim(ident, c, b, what, status='S-d'):
            (l1, r1), (l2, r2) = eq(c, b, 'MAE'), eq(c, b, 'WIS')
            reading = dec['contrasts'][f'{c}-{b}']['reading']
            if b == 'L':
                claim(ident, f'{what}: ΔS_MAE {num(r1["difference"])} [{num(r1["ci_lower"])}, {num(r1["ci_upper"])}] '
                             f'({pct(r1["ratio"])}); WIS not defined (L has no interval forecast): **{reading}**.',
                      f'U24 L{l1}; DEC24 `contrasts`', status)
                return
            (i1, x1), (i2, x2) = ri(c, b, 'MAE'), ri(c, b, 'WIS')
            claim(ident, f'{what}: ΔS_MAE {num(r1["difference"])} [{num(r1["ci_lower"])}, {num(r1["ci_upper"])}] '
                         f'({pct(r1["ratio"])} [{pct(r1["ratio_ci_lower"])}, {pct(r1["ratio_ci_upper"])}]; 97.5% '
                         f'[{pct(x1["ratio_ci97.5_lower"])}, {pct(x1["ratio_ci97.5_upper"])}]), ΔS_WIS '
                         f'{num(r2["difference"])} [{num(r2["ci_lower"])}, {num(r2["ci_upper"])}] ({pct(r2["ratio"])} '
                         f'[{pct(r2["ratio_ci_lower"])}, {pct(r2["ratio_ci_upper"])}]; 97.5% [{pct(x2["ratio_ci97.5_lower"])}, '
                         f'{pct(x2["ratio_ci97.5_upper"])}]): **{reading}**{resolved_note(r1, r2)}.',
                  f'U24 L{l1}, L{l2}; RI24 L{i1}, L{i2}; DEC24 `contrasts`', status)

        (lm, rm), (lw, rw) = eq('v5', 'HGL', 'MAE'), eq('v5', 'HGL', 'WIS')
        claim('C420', f'Scored attempt {k}, v5 − v4: ΔS_MAE {num(rm["difference"])} [95% {num(rm["ci_lower"])}, {num(rm["ci_upper"])}; '
                      f'97.5% {num(rm["ci97.5_lower"])}, {num(rm["ci97.5_upper"])}] ({pct(rm["ratio"])} of v4\'s score, 95% '
                      f'[{pct(rm["ratio_ci_lower"])}, {pct(rm["ratio_ci_upper"])}], 97.5% [{pct(ri("v5", "HGL", "MAE")[1]["ratio_ci97.5_lower"])}, '
                      f'{pct(ri("v5", "HGL", "MAE")[1]["ratio_ci97.5_upper"])}]); ΔS_WIS '
                      f'{num(rw["difference"])} [95% {num(rw["ci_lower"])}, {num(rw["ci_upper"])}; 97.5% {num(rw["ci97.5_lower"])}, '
                      f'{num(rw["ci97.5_upper"])}] ({pct(rw["ratio"])}, 95% [{pct(rw["ratio_ci_lower"])}, {pct(rw["ratio_ci_upper"])}], '
                      f'97.5% [{pct(ri("v5", "HGL", "WIS")[1]["ratio_ci97.5_lower"])}, {pct(ri("v5", "HGL", "WIS")[1]["ratio_ci97.5_upper"])}]).',
              f'U24 L{lm}, L{lw}; RI24 L{ri("v5", "HGL", "MAE")[0]}, L{ri("v5", "HGL", "WIS")[0]}', 'S')
        conds = a['conditions']
        cond_text = '; '.join(f'condition {x} {"met" if v["met"] else "not met"}' for x, v in sorted(conds.items(), key=lambda kv: int(kv[0])))
        claim('C421', f'Rule `cp24-adoption` for v5 in attempt {k}: ' + ('all five conditions met' if a['adopted'] else
                      f'first unmet condition {a["first_unmet_condition"]}') + f' ({cond_text}; condition 3\'s Engineering PASS is '
                      'the fresh Integration verdict).', 'DEC24 `adoption`; C24; U24 per-fold rows', 'S')
        claim('C422', f'The mechanical result of `cp24-adoption` in attempt {k}: **{a["verdict"]}**.', 'DEC24 `adoption.verdict`', 'S')
        if len(o['scored_attempts']) > 1:
            for kk in o['scored_attempts']:
                if kk != k:
                    other = _load(root, OUT / f'attempt-{kk}' / 'decisions.json')['adoption']
                    claim('C423', f'Scored attempt {kk}: **{other["verdict"]}** (first unmet condition {other["first_unmet_condition"]}).',
                          f'reports/ddnn2/attempt-{kk}/decisions.json', 'S')
        contrast_claim('C424', 'v5', 'HG', 'v5 against v3 (reference)', status='S')
        contrast_claim('C425', 'D2', 'L', 'DDNN-2 alone against L, its same-information twin (point only)')
        contrast_claim('C426', 'D2', 'HG', 'DDNN-2 alone against v3 (LEAR)')
        contrast_claim('C427', 'D2', 'HGL', 'DDNN-2 alone against v4')
        contrast_claim('C428', 'D2', 'D', 'DDNN-2 against CP-23\'s DDNN')
        contrast_claim('C429', 'v3+D2', 'HGL', 'DDNN-2 in LightGBM\'s place ((v3 + DDNN-2) − v4)')
        contrast_claim('C430', 'v3+D', 'HGL', 'Beside it, CP-23\'s DDNN in LightGBM\'s place ((v3 + CP-23\'s DDNN) − v4)')
        contrast_claim('C431', 'v3+D2', 'HG', 'DDNN-2 as v3\'s third member ((v3 + DDNN-2) − v3)')
        contrast_claim('C432', 'v5', 'v3+D2', 'Whether LightGBM still adds once DDNN-2 is present (v5 − (v3 + DDNN-2))')
        parts = ', '.join(f'{DISPLAY.get(p, p)} {num(score(p)[1]["S_MAE"])}/{num(score(p)[1]["S_WIS"])}'
                          for p in ('v5', 'HGL', 'v3+D2', 'HG', 'D2', 'D'))
        claim('C433', f'Equal-fold S_MAE/S_WIS (attempt {k}): {parts}.', 'M24 equal-fold rows', 'S')
        claim('C434', 'Original §8 screen (diagnostic): ' + ', '.join(f'{DISPLAY.get(p, p)} {s}' for p, s in dec['original_section8_status'].items())
              + '.', 'C24', 'S')
        f3, ordinary = pf('v5', 'fold_3'), [float(pf('v5', fo)[1]['MAE']) for fo in ('fold_1', 'fold_2', 'fold_4', 'fold_5')]
        claim('C435', f'v5 per-fold MAE: ordinary folds {num(min(ordinary), 1)}–{num(max(ordinary), 1)} EUR/MWh; stress fold 3 '
                      f'{num(f3[1]["MAE"], 1)} EUR/MWh (v4: {num(pf("HGL", "fold_3")[1]["MAE"], 1)}; v3: {num(pf("HG", "fold_3")[1]["MAE"], 1)}).',
              f'M24 L{pf("v5", "fold_1")[0]}–L{pf("v5", "fold_5")[0]}', 'S')
        diag = _load(root, A / 'diagnostics.json')
        cal = diag['calibration']
        claim('C436', f'DDNN-2 alone, calibration (pooled): 50/80/95% coverage {num(cal["pooled"]["coverage50"], 3)} / '
                      f'{num(cal["pooled"]["coverage80"], 3)} / {num(cal["pooled"]["coverage95"], 3)}; PIT shares below 0.05 and '
                      f'above 0.95 {num(cal["pit_share_below_0.05"], 3)} and {num(cal["pit_share_above_0.95"], 3)} (0.05 each if calibrated).',
              'DG24 `calibration`', 'S-d')
        corr = diag['error_correlations_pooled']
        claim('C437', 'Pooled error correlations: ' + ', '.join(f'{a_.replace("~", " with ")} {num(v, 2)}' for a_, v in corr.items()
                                                              if a_ in ('D2~L', 'D2~HG', 'D2~D', 'D~L', 'HG~L')) + '.',
              'DG24 `error_correlations_pooled`', 'S-d')
        g = diag['guards']
        claim('C438', f'Guards over {g["member_fits"]:,} attempt member fits: cap activations {g["cap_slot_levels_forecast"]:,} '
                      f'emitted slot-levels; winsorised forecast inputs {g["winsor_values_forecast"]:,}; ensemble crossings restored '
                      f'{g["ensemble_crossings_restored"]}; nonfinite-loss stops {g["nonfinite_loss_stops"]}.', 'GU24', 'S')
        controls = _load(root, A / 'controls.json')
        claim('C439', f'All §23.10 controls and the inherited ones passed ({controls["checks"]} checks), each negative paired with a '
                      'positive: masking and future inputs exactly 0.0; a non-uniform D−1 mutation, weather permutations, held-out '
                      'weeks and recent training days move DDNN-2; search and gate outcomes after their cut-offs change nothing; '
                      'pre-registration by ancestry; composite parity on every key; restart replay and cache refusals.', 'CT24', 'S')
        cost, cycle = _load(root, A / 'fit-cost.json'), _load(root, A / 'daily-cycle.json')
        claim('C440', f'Fit cost and the daily cycle (diagnostic only): {cost["attempt_fits"]:,} attempt member fits, median eight-member '
                      f'ensemble {num(cost["per_origin"]["median_ensemble_fit_wall_seconds"], 1)} s per origin; v5\'s cold daily cycle '
                      f'median {num(cycle["summary"]["median_seconds"], 0)} s, maximum {num(cycle["summary"]["max_seconds"], 0)} s at '
                      f'{cycle["summary"]["origins"]} origins.', 'FC24; DC24', 'S')
        if (root / A / 'leakage-controls.json').exists():
            lk = _load(root, A / 'leakage-controls.json')
            claim('C441', f'No outcome dated on or after the delivery day is used, at any origin: with every such outcome destroyed, the frozen '
                          f'eight-member ensemble refitted at all {lk["blind"]["origins"]} warm-up and evaluation origins reproduces the '
                          f'committed DDNN-2 vectors bit for bit ({lk["blind"]["bitwise"]} of {lk["blind"]["origins"]}); a D−1 price '
                          'mutation moves them and a planted one-day leak is detected in every fold; the search and gate controls hold '
                          f'in all {len(lk["search_and_gate_by_fold"])} folds. The input vintages (A65 load forecasts, CP-20\'s GFS '
                          'availability rule) are inherited from CP-15 and CP-20, not re-tested here.', 'LK24', 'S')
        ctx.update(dict(dec=dec, eq=eq, ri=ri, pf=pf, k=k, adopted=bool(a['adopted'])))
    claim('C449', 'Development evidence after selection on the same five folds CP-15 and CP-20 to CP-23 used; DDNN-2 is the second '
                  'DDNN decision on them; not a test on new data; 4.7T carries the protection.', 'CAP §23.1', 'S')
    withheld = [
        ('W42', 'Saying that distributional neural networks "do not work" or "cannot help" in general',
         'One design family under one rule (C403, C421 or the gate claims); no demonstrated preference is not a proof of absence.'),
        ('W43', 'Calling v5 and v4 equivalent', 'A mixed or spanning result is no demonstrated joint preference (CAP §23.9).'),
        ('W44', 'Attributing any forecast or metric to PyTorch', 'PyTorch is a test-only correctness reference (CAP §18.3).'),
        ('W45', 'Stating or implying that v5 or DDNN-2 is the product, is live, runs in the demo or retrains in the browser',
         'v1 is the released product and demo (C402); runtimes other than the evaluated one are unmeasured.'),
        ('W46', 'Presenting the gate as evidence of improvement', 'The gate is a screen on pre-fold days, not a claim (CAP §23.6).'),
        ('W47', 'Quoting a 95% interval as the decision interval for v5 − v4', 'Condition 1 reads the attempts-adjusted 97.5% interval.'),
        ('W48', 'Quoting any CP-24 number without its evidence class, or as a confirmatory or significance result',
         'Development after selection; exploratory intervals.'),
    ]
    src = [('R24', 'reports/ddnn2/report.md'), ('LIC24', 'reports/ddnn2/licence-admission.md'),
           ('REF24', 'reports/ddnn2/reference-checks.json'), ('RA24', 'reports/ddnn2/resource-admission.md'),
           ('PAR24', 'reports/ddnn2/v4-parity.json'), ('STEER24', 'docs/track-b/evidence/cp-24/steering/'),
           ('DR24', 'reports/ddnn2/draft-registry.json'), ('I24', 'docs/track-b/evidence/cp-24/integration.md'),
           ('RET24', 'docs/track-b/evidence/cp-24/checkpoint-return.md'), ('CAP', 'capstone_v21.md (v21-r11), §23')]
    for r in o['rounds']:
        src.append((f'GATE24-{r}', f'reports/ddnn2/rounds/round-{r}/gate.json'))
        src.append((f'RD24-{r}', f'reports/ddnn2/rounds/round-{r}/design.json'))
    if o['scored_attempts']:
        k = o['deciding_attempt']
        base = f'reports/ddnn2/attempt-{k}'
        src += [('M24', f'{base}/metrics.csv'), ('U24', f'{base}/uncertainty.csv'), ('C24', f'{base}/criteria.csv'),
                ('D24', f'{base}/diagnostics.csv'), ('DEC24', f'{base}/decisions.json'), ('P24', f'{base}/protocol.json'),
                ('RI24', f'{base}/ratio-intervals.csv'), ('CT24', f'{base}/controls.json'), ('DG24', f'{base}/diagnostics.json'), ('GU24', f'{base}/guards.json'),
                ('DC24', f'{base}/daily-cycle.json'), ('FC24', f'{base}/fit-cost.json'),
                *([('LK24', f'{base}/leakage-controls.json')] if (root / base / 'leakage-controls.json').exists() else []),
                ('X24', 'reports/ddnn2/mlflow-export-draft/cp24.json')]
    claims_md = '\n'.join([
        '# CP-24 research result: claim-to-evidence map', '',
        '**2026-10 · Draft for the next publication (PUBLISH_RULES 1.3 §11; capstone v21-r11 §23.12).** Each claim the',
        'publication may render is mapped to a committed file and row, in the format of the [CP-23](cp23-claims.md) map: `L<n>`',
        'is the physical line in the file, and for a CSV `L1` is the header. Values here are rounded from the saved values; the',
        'rendered surfaces bind to evidence records, never to these digits. Publication follows the Owner\'s landing and the',
        'Owner\'s decisions (§23.12).', '',
        f'**Outcome:** {o["kind"].replace("_", " ")}. **Default evidence class:** `development_post_selection`, exploratory.', '',
        '## Source keys', '', '| Key | File |', '|---|---|', *[_source_row(x, p) for x, p in src], '',
        '## Claim map: CP-24', '',
        'Status codes: **S** supported by a saved value or verdict; **S-d** supported, descriptive only; **Def** a',
        'definition; **O** an Owner decision.', '',
        '| ID | Claim | Source → table/row | Status |', '|---|---|---|---|', *claims, '',
        '## Withheld claims: do not use', '', 'W1–W41 in the earlier maps stay in force. CP-24 adds:', '',
        '| ID | Withheld claim | Reason / controlling source |', '|---|---|---|', *[f'| {x} | {y} | {z} |' for x, y, z in withheld], '',
        '## Gaps: documented, not resolved', '', '| ID | Gap | Why it stays open |', '|---|---|---|',
        '| G30 | Learned weights, new information, other families | Excluded: 4.8 and stage 7 (CAP §23.5, §22). |',
        '| G31 | Hosted-service and product uses | Unresolved in 4.6L; carried to 4.7 and 4.10 (LIC24). |',
        '| G32 | DDNN-2 in another runtime and its equality with the evaluated code | Unmeasured (CAP §18.2 item 4). |',
        '| G33 | Status dates, the evidence tag and the landing record | They exist only at landing (pending fields). |', '',
    ]) + '\n'
    return claims_md, registry, ctx


def packet(root: Path, registry: dict, ctx: dict, claims_sha: str) -> tuple[str, list[str]]:
    root = Path(root)
    o = ctx['o']
    L, findings = [], []
    add = L.append
    add('# CP-24 publication packet (draft)\n')
    add('**PUBLISH_RULES 1.3 §11; capstone v21-r11 §23.12; template `docs/track-b/publication-packet-template.md` '
        '(SHA-256 4efb0185…829cf).** Filled from inside CP-24, in every outcome; a stop is recorded as such. Numbers are never typed '
        'into a surface: each value names the committed row it comes from. Identities that exist only at landing are marked '
        '**pending-at-landing**. Nothing here is published.\n')
    add('Pinned rules (the issued brief also records them): PUBLISH_RULES 1.3 '
        '`5a660864f8b82943174741320c71087a3d0508f707edeb997cb2c1395c1c73b4`; packet template '
        '`4efb01855ae2d9a6f54b5624607a93046b1fffa881e34af0c6798c513c7829cf`. Both verified by `cp24.inputs.identities`.\n')
    add('## 1. Identity\n')
    add('| Item | Value |\n|---|---|')
    add('| Checkpoint | CP-24, brief `docs/track-b/evidence/cp-24/issued-brief.md` at `a3f11470dde0a36e9bdf84fbf02de0597ae3d0eca4c4c75f18e79ab173c34bd0` |')
    add('| Governing research plan and version | `capstone_v21.md` v21-r11 §23 (SHA-256 `11068e3f…ed2d2`); rule `cp24-adoption` |')
    add('| Evidence tag and tip | `evidence/cp-24` at **pending-at-landing** (the evidence tip SHA is in the CP-24 return), frozen **pending-at-landing** |')
    add('| Final reviewed candidate (model code) | recorded in the CP-24 checkpoint return; a commit cannot contain its own SHA |')
    add('| Report, review verdict, landing record | `reports/ddnn2/report.md`; `docs/track-b/evidence/cp-24/integration.md`; landing record **pending-at-landing** |\n')
    add(f'**Outcome:** {o["kind"].replace("_", " ")}; pre-fold rounds {len(o["rounds"])} (gates passed: '
        + ', '.join(f'round {r} {"yes" if p else "no"}' for r, p in o['gates_passed'].items()) + f'); scored attempts '
        f'{len(o["scored_attempts"])}.\n')
    add('## 2. The draft registry entries (standard §5)\n')
    if not registry['entries']:
        add('None: no scored attempt ran (the route stopped at the pre-fold gate), so no identity was evaluated on the folds. '
            'CP-24 registers nothing.\n')
    for e in registry['entries']:
        add(f'### `{e["id"]}` — {e["kind"]}\n')
        add('| Field | Value |\n|---|---|')
        for field in ('id', 'name', 'subtitle', 'short', 'kind'):
            add(f'| `{field}` | {e[field]} |')
        add('| `codes` | ' + '; '.join(f'{c["experiment"]}: {c["code"]} ({c["population"]}' + (f', {c["note"]}' if c['note'] else '') + ')'
                                     for c in e['codes']) + ' |')
        add('| `statuses` | ' + '; '.join(f'{s["status"]}, {s["date"]}, {s["source"]}' + (f', {s["reason"]}' if s['reason'] else '')
                                        for s in e['statuses']) + ' |')
        add(f'| `comparator` | `{e["comparator"]}` (named by §23.5 before results) |')
        add(f'| `population` | `{e["population"]}` |')
        add(f'| `evidence_class` | Development (`{e["evidence_class"]}`) |')
        add(f'| `plan`, `rules` | {e["plan"]}; `{", ".join(e["rules"])}` |')
        add('| `sources` | ' + ', '.join(f'`{s}`' for s in e['sources']) + ' |')
        add(f'| `claim_map` | `{e["claim_map"]}` |')
        add('| `run_keys` | ' + (', '.join(f'`{x}`' for x in e['run_keys']) or 'none (not the deciding attempt)') + ' |')
        add(f'| `style`, `anchor` | {e["style"]}; {e["anchor"] or "none"}'
            + (' — v5\'s colour is the Owner\'s decision before the publication brief (PUBLISH_RULES §14)' if e['id'] == 'v5' else '') + ' |')
        add(f'| `checkpoint`, `after` | CP-24; {e["after"] or "n/a"} |')
        add(f'| `question`, `informed` | {e["question"] or "n/a"}; {e["informed"] or "none recorded"} |')
        add(f'| `predecessor` | {e["predecessor"] or "n/a"} |\n')
    add('## 3. The claim map\n')
    add(f'`docs/track-b/research-content/cp24-claims.md` (SHA-256 `{claims_sha}`): every claim with its committed rows, the entry '
        'gates, every round\'s gate, the steering record, ' + ('the adoption decision, every §23.8 contrast and diagnostic, ' if o['scored_attempts'] else '')
        + 'and the withheld claims W42–W48 (W1–W41 stay in force).\n')
    add('## 4. The derived headline quantities (standard §3.3), computed inside the checkpoint\n')
    add('| Quantity | Value | From |\n|---|---|---|')
    gates = '; '.join(f'round {r}: {"passed" if p else "not passed"}' for r, p in o['gates_passed'].items())
    add(f'| The gate results | {gates} | `reports/ddnn2/rounds/round-<r>/gate.json` |')
    add(f'| The number of rounds and scored attempts | {len(o["rounds"])} pre-fold round(s); {len(o["scored_attempts"])} scored attempt(s) | '
        '`reports/ddnn2/rounds/`, `reports/ddnn2/attempt-<k>/`, `steering/` |')
    add('| The adjusted level | condition 1 of `cp24-adoption` at two-sided 97.5% (the 1.25% and 98.75% percentiles of the shared '
        'replicates), splitting 5% across the cap of two attempts; per-fold condition 4 at 95% | CAP §23.9 |')
    slots = {}
    if o['scored_attempts']:
        k, dec, eq = ctx['k'], ctx['dec'], ctx['eq']
        a = dec['adoption']
        (lm, rm), (lw, rw) = eq('v5', 'HGL', 'MAE'), eq('v5', 'HGL', 'WIS')
        add(f'| (a) The pre-specified verdict (deciding attempt {k}) | `cp24-adoption`: **{a["verdict"]}** (first unmet condition '
            f'{a["first_unmet_condition"]}; unmet {a["unmet_conditions"]}) | `reports/ddnn2/attempt-{k}/decisions.json` |')
        add('| The rule, in words, with the date it was set | v5 is adopted in research only if, against v4, both paired score '
            'differences improve at 97.5% (interval score upper endpoint below zero, point error at or below zero), it meets all '
            'six original screening diagnostics, the evaluation is complete and valid with every guard activation reported, no fold '
            'is decisively worse, and both point estimates improve v4 by at least 0.5%. Set **2026-10-05** | CAP §23.9; '
            '`protocol.json` `rule_verbatim` |')
        add(f'| Distance from v4, in the rule\'s unit | ΔS_MAE {num(rm["difference"])} (97.5% upper {num(rm["ci97.5_upper"])}); ΔS_WIS '
            f'{num(rw["difference"])} (97.5% upper {num(rw["ci97.5_upper"])}) | `uncertainty.csv` L{lm}, L{lw} |')
        add(f'| N: policies tested against the same rule up to this decision | {len(o["scored_attempts"])} v5 candidate(s), one per '
            'scored attempt; DDNN-2 alone and v3 + DDNN-2 are never eligible | `protocol.json` `arms` |')
        add('| "Point comparison" label needed? | no — every difference has an interval | `uncertainty.csv` |')
        add(f'| (b) The change against v4, as a share of v4\'s score | S_MAE {pct(rm["ratio"])}; S_WIS {pct(rw["ratio"])} | '
            f'`uncertainty.csv` L{lm}, L{lw} (`ratio`) |')
        add(f'| Its 95% interval | S_MAE [{pct(rm["ratio_ci_lower"])}, {pct(rm["ratio_ci_upper"])}]; S_WIS [{pct(rw["ratio_ci_lower"])}, '
            f'{pct(rw["ratio_ci_upper"])}] — CP-24\'s own draws: seed 15042, 2,000 replicates, 7-calendar-day blocks | '
            f'`uncertainty.csv`; `replicates.parquet` |')
        (li_m, xm), (li_w, xw) = ctx['ri']('v5', 'HGL', 'MAE'), ctx['ri']('v5', 'HGL', 'WIS')
        add(f'| Its 97.5% interval, the decision level (§23.8) | S_MAE [{pct(xm["ratio_ci97.5_lower"])}, {pct(xm["ratio_ci97.5_upper"])}]; '
            f'S_WIS [{pct(xw["ratio_ci97.5_lower"])}, {pct(xw["ratio_ci97.5_upper"])}] — the 1.25% and 98.75% percentiles of the same '
            f'stored draws | `ratio-intervals.csv` L{li_m}, L{li_w} |')
        ordinary = [float(ctx['pf']('v5', fo)[1]['MAE']) for fo in ('fold_1', 'fold_2', 'fold_4', 'fold_5')]
        add(f'| (c) Mean absolute error per period, EUR/MWh (v5) | ordinary periods {num(min(ordinary), 1)}–{num(max(ordinary), 1)}; '
            f'stress period, fold 3: {num(ctx["pf"]("v5", "fold_3")[1]["MAE"], 1)} | `metrics.csv` per-fold rows |\n')
        owner = registry['checkpoint']['owner']
        if ctx['adopted']:
            slots = {
                'v5.question': 'Does a day-level distributional neural network, added to {g:v4.name} at a fixed one-sixth weight, '
                               'improve on it jointly in point and interval accuracy?',
                'v5.change': '{g:v5.name} adds a distributional neural network, written in NumPy and tuned on training data only, '
                             'inside the nonlinear third of {g:v4.name}.',
                'v5.main_chart.headline': 'Against {g:v4.name}, the point-error score changed by {r:cp24.ratio.v5-HGL.MAE} and the '
                                          'interval score by {r:cp24.ratio.v5-HGL.WIS}, each as a share of the comparator\'s score.',
                'v5.reading': 'Both scores improved against the comparator at the adjusted level, by more than the practical size, '
                              'and no test period was decisively worse.',
                'v5.not_established': 'Development evidence after selection, not a test on new data.',
                'v5.decision': 'Adopted in research on {g:v5.status.date}; the released model is unchanged.',
                'v5.evidence': 'the report, the review verdict, the claim map, the MLflow comparison',
                'transition.v4-v5.title': 'From v4 to v5: adding a distributional neural network'}
        else:
            slots = {
                f'branch.{owner}.question': '{g:' + owner + '.question}',
                f'branch.{owner}.comparator': '{g:v4.name}',
                f'branch.{owner}.result': 'Against {g:v4.name}, adding the network at a one-sixth weight changed the point-error '
                                          'score by {r:cp24.ratio.v5-HGL.MAE} and the interval score by {r:cp24.ratio.v5-HGL.WIS}, '
                                          'each as a share of the comparator\'s score.',
                f'branch.{owner}.reason': '{g:' + owner + '.status.reason}',
                f'branch.{owner}.decision': 'The Owner decides whether this branch is published; {g:v4.name} is unchanged.',
                f'branch.{owner}.evidence': 'the report, the review verdict, the claim map, the MLflow comparison'}
    else:
        add('| (a)–(c) The verdict, the distance from v4, the change and the per-period MAE | none: no scored attempt, so no fold was '
            'looked at; the stop is the result | the gate results above |\n')
    for key, text in slots.items():
        findings += [f'{key}: {f}' for f in lint_slot(text)]
    add('## 5. Draft slot texts (standard §6)\n')
    if slots:
        add(f'Only the outcome the rule yields is drafted. The §4 lint\'s code and status-word rules on every draft: **{len(findings)} findings**.\n')
        add('| Slot key | Draft |\n|---|---|')
        for key, text in slots.items():
            add(f'| `{key}` | ' + text.replace('|', '\\|') + ' |')
        add('')
    else:
        add('None: a stop at the pre-fold gate yields no chapter, branch card or transition; the Owner decides whether the stop is '
            'mentioned in a publication.\n')
    add('### 5a. The adopted transition (A3)\n\n' + ('See the slot `transition.v4-v5.title`; the comparator is the predecessor, v4.'
                                                     if ctx.get('adopted') else 'Not applicable: no generation is adopted.') + '\n')
    add('### 5b. The released model\'s documentation (A4)\n\nNot applicable: the released model does not change (v1; §23.9).\n')
    add('### 5c. The chart routes (A5)\n\n' + ('The v5 chapter\'s paired differences against v4 (`#v5`).' if ctx.get('adopted') else
                                            'None drafted.') + '\n')
    add('### 5d. Final-product transition / daily operation (A7/A8)\n\nNot applicable: no final-product designation, rollout or Live.\n')
    add('## 6. The MLflow export\n')
    xpath = root / OUT / 'mlflow-export-draft' / 'cp24.json'
    if xpath.exists():
        xsha = hashlib.sha256(xpath.read_bytes()).hexdigest()
        add(f'- **Draft export:** `reports/ddnn2/mlflow-export-draft/cp24.json` (SHA-256 `{xsha}`), built by `python -m cp24.export` '
            'importing `scripts/mlflow_export.py`\'s functions unchanged, from CP-24\'s committed rows and `draft-registry.json`; '
            'experiment `delu-generations`; pending fields as listed in the draft registry.')
        add('- **Local tracking:** the same runs in `.local/mlruns/cp24`, read back equal (`reports/ddnn2/mlflow-local.json`). No upload, '
            'no network call; MLflow telemetry disabled.')
    else:
        add('- **Draft export:** none — no scored attempt, so there are no CP-24 runs to export. The published export set is unchanged.')
    add('- **The published export set** (`reports/presentation/mlflow-export/`) and `scripts/mlflow_export.py` are unchanged; '
        '`scripts/mlflow_export.py --check` passes.\n')
    add('## 7. Checks the checkpoint ran\n')
    add('- `tests/cp24/` (re-derivation, the rule on synthetic fixtures, DST fixtures, the search procedure, the import audit, the '
        'finite-difference checks, the reference record) and the full default suite, in a clean detached worktree.')
    add(f'- The §4 lint on every draft slot text: {len(findings)} findings.\n')
    add('## 8. Publication completion receipt — intended identities (publisher completes the rest)\n')
    add('| Surface | Intended identity | Action / unchanged rationale | Observed | Verification | Outstanding |\n|---|---|---|---|---|---|')
    for surface in ('GitHub source / README', 'GitHub Pages', 'Public MLflow', 'Hugging Face Space card', 'Hugging Face direct demo'):
        add(f'| {surface} | after the Owner\'s landing and decisions (§23.12) | pending the Owner | pending | pending | pending |')
    add('\n- **Authority:** none exercised; every external action needs the Owner\'s instruction for that action.\n')
    return '\n'.join(L) + '\n', findings


def main() -> int:
    import sys
    root = Path.cwd()
    check = '--check' in sys.argv[1:]
    claims_md, registry, ctx = build(root)
    claims_sha = hashlib.sha256(claims_md.encode()).hexdigest()
    text, findings = packet(root, registry, ctx, claims_sha)
    if check:
        same = {str(CLAIMS): (root / CLAIMS).read_text() == claims_md, str(PACKET): (root / PACKET).read_text() == text}
        print(json.dumps({'identical_to_committed': same, 'lint_findings': findings}), flush=True)
        return 0 if all(same.values()) and not findings else 11
    (root / CLAIMS).parent.mkdir(parents=True, exist_ok=True)
    (root / CLAIMS).write_text(claims_md)
    (root / PACKET).parent.mkdir(parents=True, exist_ok=True)
    (root / PACKET).write_text(text)
    print(json.dumps({'claims': str(CLAIMS), 'packet': str(PACKET), 'lint_findings': findings}), flush=True)
    return 0 if not findings else 10


if __name__ == '__main__':
    raise SystemExit(main())
