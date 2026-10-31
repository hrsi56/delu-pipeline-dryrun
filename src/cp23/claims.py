"""CP-23's claim map and publication packet (capstone v21-r10 §21.9; packet template sections 1-8).

Run as ``python -m cp23.claims`` after scoring, the diagnostics, the draft registry and the draft export
(``--check`` regenerates and compares, writing nothing). Every value and every row reference (`L<n>`, the
physical line; `L1` is a CSV's header) is read from the committed files at generation time, so neither
document can drift from the evidence. Draft slot texts carry registry tokens (`{g:<id>.<field>}`) for names
and statuses and evidence tokens (`{r:<record>}`) for numbers. They are checked with the publication lint's
code and status-word rules. Only the outcome the rule yields is drafted. Identities that exist only at
landing stay explicit pending fields.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from cp22.claims import Rows, num, pct, resolved_note, _source_row
from delu_forecast import publication_lint as PL

OUT = Path('reports/distribution-challenger')
CLAIMS = Path('docs/track-b/research-content/cp23-claims.md')
PACKET = Path('docs/track-b/evidence/cp-23/publication-packet.md')
FOLDS = ('fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5')
INTERNAL_CODES = ('HGL', 'HG', 'H0', 'B0', 'B1', 'B2', 'B3', 'A1', 'A2', 'L-P', 'L-R', 'L-N', 'PN', 'D', 'v3+D', 'C1', 'C2',
                  'C3', 'C4', 'JSU')
DISPLAY = {'HG': 'v3', 'HGL': 'v4'}


def lint_slot(text: str) -> list[str]:
    plain = re.sub(r'\{[gr]:[^{}]+\}', '', text)
    findings = PL.code_findings(PL.reading_path(f'<p>{plain}</p>'))
    findings += [f'status word {m.group(0)!r}' for m in PL.STATUS_WORDS.finditer(plain)]
    findings += [f'internal code {c!r}' for c in INTERNAL_CODES if re.search(rf'(?<![\w+-]){re.escape(c)}(?![\w+-])', plain)]
    return findings


def build(root: Path):
    M, U, C = Rows(root, str(OUT / 'metrics.csv')), Rows(root, str(OUT / 'uncertainty.csv')), Rows(root, str(OUT / 'criteria.csv'))
    D = Rows(root, str(OUT / 'diagnostics.csv'))
    load = lambda name: json.loads((root / OUT / name).read_text())  # noqa: E731
    dec, registry, controls, parity = load('decisions.json'), load('draft-registry.json'), load('controls.json'), load('parity.json')
    repro, cost, cycle, protocol = load('reproduction.json'), load('fit-cost.json'), load('daily-cycle.json'), load('protocol.json')
    diag, sel, ref, r46 = load('diagnostics.json'), load('selection.json'), load('reference-checks.json'), load('resource-admission.json')
    a = dec['adoption']
    adopted = bool(a['adopted'])

    def eq(c, b, m):
        return U.find(scope='equal_fold', candidate=c, baseline=b, metric=m)

    def score(p):
        return M.find(policy=p, scope='equal_fold')

    def pf(p, fo):
        return M.find(policy=p, scope='per_fold', fold=fo)

    claims = []

    def claim(ident, text, source, status):
        claims.append(f'| {ident} | {text} | {source} | {status} |')

    def contrast_claim(ident, c, b, what, status='S-d'):
        (l1, r1), (l2, r2) = eq(c, b, 'MAE'), eq(c, b, 'WIS')
        reading = dec['contrasts'][f'{c}-{b}']['reading']
        claim(ident, f'{what}: ΔS_MAE {num(r1["difference"])} [{num(r1["ci_lower"])}, {num(r1["ci_upper"])}] '
                     f'({pct(r1["ratio"])} [{pct(r1["ratio_ci_lower"])}, {pct(r1["ratio_ci_upper"])}]), ΔS_WIS '
                     f'{num(r2["difference"])} [{num(r2["ci_lower"])}, {num(r2["ci_upper"])}] ({pct(r2["ratio"])} '
                     f'[{pct(r2["ratio_ci_lower"])}, {pct(r2["ratio_ci_upper"])}]): **{reading}**{resolved_note(r1, r2)}.',
              f'U23 L{l1}, L{l2}; DEC23 `contrasts`', status)

    # ---- identity, rule, scope
    claim('C300', 'CP-23\'s Integration verdict and the reviewed final candidate are recorded in I23 and the CP-23 checkpoint '
                  'return; the evidence tag and landing record are pending at landing.', 'I23; RET23', 'S (pending)')
    claim('C301', 'The rule `cp23-adoption` was set on 2026-10-04 (the Owner\'s ratification of v21-r10) and frozen, verbatim, in '
                  'the pre-run protocol before any main-run fit or score.', 'P23 `rule_verbatim`; CAP §21.6', 'S')
    claim('C302', 'No product change, promotion, freeze, Live, final-product designation or economic claim follows; v1 is the '
                  'released product and the demo.', 'CAP §21.6 (what adoption means); DEC23 `statuses`', 'S')
    claim('C303', 'The same 10,747 eligible hours in five folds (2,160 / 2,159 / 2,112 / 2,160 / 2,156), 448 represented days; '
                  'fold 3 (the stress period) has 2,112 hours on 88 days; the peak 2022-08-15..31 has 408 hours.',
          f'M23 L{pf("B0", "fold_1")[0]}–L{pf("B0", "fold_5")[0]} (`n_hours`); D23 peak rows', 'S')
    dd = protocol['ddnn']
    claim('C304', 'DDNN is a feed-forward network (ELU) with a Johnson SU distributional head, written in NumPy and the Python '
                  'standard library only. It has exactly v4\'s information on the normalized price target, one row per '
                  'delivery hour; four configurations ('
          + ', '.join(f'{c["id"]} {c["hidden"]}' for c in dd['configurations'])
          + '), one chosen per fold before its first origin on training data only; early stopping on the window\'s last 28 '
            'days; four seeds combined by averaging quantiles; the median is both the p50 and the central forecast.',
          'CAP §21.2; P23 `ddnn`', 'Def')
    claim('C305', f'The entry gates passed in order: 4.6L permits research use and retention (hosted service and product '
                  f'unresolved, carried to 4.7); the import audit, the finite-difference gradient checks and the PyTorch '
                  f'reference ({ref["counts"]["passed"]} of {ref["expected_passed"]} checks, torch {ref["versions"]["torch"]}, at '
                  f'tolerances frozen before the first run) passed; 4.6R on training data only: **{r46["verdict"]}** '
                  f'(projected {num(r46["projection"]["projected_machine_hours"], 1)} of 60 machine-hours; '
                  f'{r46["projection"]["main_fits"]:,} of 4,000 main fits).',
          'LIC23; REF23; RA23; CAP §21.3–§21.4', 'S')
    claim('C306', 'v5 = (2/3)·v4 + (1/3)·DDNN, and v3 plus a DDNN member = (2/3)·v3 + (1/3)·DDNN, each with v3\'s hour-aware '
                  'interval layer re-estimated on its own errors; DDNN alone uses its own Johnson SU quantiles. The member '
                  'weight is fixed at 1/3, as LightGBM\'s was in v4.',
          f'CAP §21.2; P23 `policies`; CT23 `composite_parity_max_abs_eur_mwh` '
          f'({controls["population"]["composite_parity_max_abs_eur_mwh"]["v5"]:.1e} EUR/MWh)', 'Def/S')
    claim('C307', 'Configuration chosen per fold, before its first origin: ' + '; '.join(
        f'{f.replace("_", " ")} {v["selected"]}' for f, v in sel['folds'].items()) + '.', 'SEL23', 'S')
    # ---- the adoption decision
    (lm, rm), (lw, rw) = eq('v5', 'HGL', 'MAE'), eq('v5', 'HGL', 'WIS')
    claim('C308', f'v5 − v4 (three-block): ΔS_MAE {num(rm["difference"])} [{num(rm["ci_lower"])}, {num(rm["ci_upper"])}] '
                  f'({pct(rm["ratio"])} [{pct(rm["ratio_ci_lower"])}, {pct(rm["ratio_ci_upper"])}] of v4\'s score); ΔS_WIS '
                  f'{num(rw["difference"])} [{num(rw["ci_lower"])}, {num(rw["ci_upper"])}] ({pct(rw["ratio"])} '
                  f'[{pct(rw["ratio_ci_lower"])}, {pct(rw["ratio_ci_upper"])}]).', f'U23 L{lm}, L{lw}', 'S')
    conds = a['conditions']
    worse = (conds.get('4') or conds.get(4))['values']['folds_decisively_worse']
    worse_folds = {}
    for row in worse:
        worse_folds.setdefault(row['scope'], []).append(row['metric'])
    worse_text = (f'{len(worse_folds)} (' + '; '.join(f'{f.replace("_", " ")}: {" and ".join(m)}' for f, m in sorted(worse_folds.items()))
                  + ')' if worse_folds else '0')
    cond_text = '; '.join(f'condition {k} {"met" if v["met"] else "not met"}' for k, v in sorted(conds.items(), key=lambda kv: int(kv[0])))
    claim('C309', 'Rule `cp23-adoption` for v5: ' + ('all four conditions met' if adopted else f'first unmet condition {a["first_unmet_condition"]}')
          + f' ({cond_text}; condition 3\'s Engineering PASS is the fresh Integration verdict). Folds decisively worse in MAE or WIS '
            f'(lower endpoint above zero): {worse_text}.', 'DEC23 `adoption`; C23; U23 per-fold rows', 'S')
    claim('C310', f'The mechanical result of `cp23-adoption`: **{a["verdict"]}**.', 'DEC23 `adoption.verdict`', 'S')
    # ---- the other §21.5 contrasts
    contrast_claim('C311', 'v5', 'HG', 'v5 against v3 (reference)', status='S')
    contrast_claim('C312', 'D', 'HGL', 'DDNN alone against v4, descriptive')
    contrast_claim('C313', 'D', 'HG', 'DDNN alone against v3, descriptive')
    contrast_claim('C314', 'v3+D', 'HG', 'DDNN as v3\'s third member ((v3 plus a DDNN member) − v3)')
    contrast_claim('C315', 'HGL', 'HG', 'Beside it, LightGBM as v3\'s third member (v4 − v3, CP-21\'s contrast recomputed on the same index set)')
    contrast_claim('C316', 'v5', 'v3+D', 'Whether LightGBM still adds once DDNN is present (v5 − (v3 plus a DDNN member))')
    parts = ', '.join(f'{DISPLAY.get(p, p)} {num(score(p)[1]["S_MAE"])}/{num(score(p)[1]["S_WIS"])}'
                      for p in ('v5', 'HGL', 'v3+D', 'HG', 'D'))
    claim('C317', f'Equal-fold S_MAE/S_WIS: {parts}.', 'M23 equal-fold rows', 'S')
    claim('C318', 'Original §8 screen (diagnostic): ' + ', '.join(f'{DISPLAY.get(p, p)} {s}' for p, s in dec['original_section8_status'].items())
          + '.', 'C23', 'S')
    f3, ordinary = pf('v5', 'fold_3'), [float(pf('v5', fo)[1]['MAE']) for fo in ('fold_1', 'fold_2', 'fold_4', 'fold_5')]
    claim('C319', f'v5 per-fold MAE: ordinary folds {num(min(ordinary), 1)}–{num(max(ordinary), 1)} EUR/MWh; stress fold 3 '
                  f'{num(f3[1]["MAE"], 1)} EUR/MWh (v4 three-block: {num(pf("HGL", "fold_3")[1]["MAE"], 1)}; v3: '
                  f'{num(pf("HG", "fold_3")[1]["MAE"], 1)}).',
          f'M23 L{pf("v5", "fold_1")[0]}–L{pf("v5", "fold_5")[0]}, L{pf("HGL", "fold_3")[0]}, L{pf("HG", "fold_3")[0]}', 'S')
    pl, pk = D.find(policy='v5', scope='peak')
    vl, vk = D.find(policy='HGL', scope='peak')
    claim('C320', f'Peak (descriptive, 17 days, small effective sample): v5 MAE {num(pk["MAE"], 1)}, WIS {num(pk["WIS"], 1)} EUR/MWh, '
                  f'95% coverage {num(pk["coverage95"], 3)}; v4 three-block MAE {num(vk["MAE"], 1)}, WIS {num(vk["WIS"], 1)}, '
                  f'coverage {num(vk["coverage95"], 3)}. No inference is drawn.', f'D23 L{pl}, L{vl}', 'S-d')
    cal = diag['calibration']
    claim('C321', f'DDNN alone, calibration (pooled): 50/80/95% coverage {num(cal["pooled"]["coverage50"], 3)} / '
                  f'{num(cal["pooled"]["coverage80"], 3)} / {num(cal["pooled"]["coverage95"], 3)}; share of actuals below its '
                  f'2.5% and above its 97.5% quantile {num(cal["pooled"]["share_at_or_below_p025"], 3)} and '
                  f'{num(1 - cal["pooled"]["share_at_or_below_p975"], 3)}; PIT shares below 0.05 and above 0.95 '
                  f'{num(cal["pit_share_below_0.05"], 3)} and {num(cal["pit_share_above_0.95"], 3)} (0.05 each if calibrated).',
          'DG23 `calibration`; diagnostics/calibration-by-level.csv, pit-histogram.csv', 'S-d')
    ext = diag['extrapolation']
    claim('C322', f'Extrapolation: on {ext["days"]} extreme or top-5% days, hours forecast above the origin\'s training-window maximum: '
          + ', '.join(f'{DISPLAY.get(p, p) if p != "D" else "DDNN alone"} {v}' for p, v in ext['hours_above_window_max'].items()) + '.',
          'DG23 `extrapolation`; diagnostics/extrapolation.csv', 'S-d')
    ss = diag['seed_stability_pooled']
    claim('C323', f'Ensemble and seeds: the ensemble median\'s MAE {num(ss["ensemble_median_MAE"], 2)} EUR/MWh against the single seeds\' '
          + ', '.join(num(v, 2) for k, v in ss.items() if k.startswith('seed_')) + f'; member medians spread '
          f'{num(ss["member_median_spread_mean_abs"], 2)} EUR/MWh around the ensemble median.', 'DG23 `seed_stability_pooled`', 'S-d')
    claim('C324', f'Fit cost and the daily cycle (diagnostic only): {cost["fits"]:,} main-run member fits, median four-seed ensemble '
                  f'{num(cost["per_origin"]["median_ensemble_fit_wall_seconds"], 1)} s per origin; v5\'s cold daily cycle (DDNN, '
                  f'ensemble, interval layer, issuance) median {num(cycle["summary"]["median_seconds"], 0)} s, maximum '
                  f'{num(cycle["summary"]["max_seconds"], 0)} s at {cycle["summary"]["origins"]} origins, beside v4\'s own component '
                  f'cycle measured by CP-21 (median {num(cycle["v4_component_cycle_beside"]["summary"]["median"], 0)} s).', 'FC23; DC23', 'S')
    claim('C325', 'v3 and v4 through CP-23\'s interval-layer path reproduce their committed vectors bit for bit on all 10,747 keys; '
                  'an independent representative v3 and v4 slice, refitted, matches bit for bit; every reused vector matches its '
                  'committed blob.', 'PAR23; REP23; P23 `cache_identities.verification`', 'S')
    claim('C326', f'All §21.7, §17.7 and applicable §20.7 controls passed ({controls["checks"]} checks), each negative paired with a '
                  'positive control: delivery-day masking and future weather exactly 0.0; a non-uniform D−1 mutation and '
                  'non-monotone weather perturbations move DDNN; early stopping responds to inner-validation outcomes and the '
                  'configuration choice ignores evaluation outcomes; a fresh process refits the main run\'s members bit for bit; '
                  'composite parity on every key; restart replay and cache refusals.', 'CT23', 'S')
    claim('C327', 'Development evidence after selection on the same five folds CP-15, CP-20, CP-21 and CP-22 used; not a test on new '
                  'data; 4.7T carries the protection.', 'CAP §21.1', 'S')
    decision = ('v5 is adopted in research as "v5 · DDNN member added", with predecessor v4' if adopted else
                'v5 is not adopted; CP-23 becomes the branch "DDNN member on v4", and the Owner decides whether the branch is published')
    claim('C328', f'The decision, dated at landing: {decision}.', 'DEC23; DR23; landing record (pending)', 'O (pending)')

    withheld = [
        ('W35', 'Saying that DDNN, or neural networks, "do not work", "cannot help" or are worse than LightGBM in general',
         'One design was tested against one rule (C304, C310); no demonstrated joint preference is not a proof of absence.'),
        ('W36', 'Calling v5 and v4 equivalent, or v5 "as good as" v4', 'A mixed or spanning result is no demonstrated joint preference, '
                'never equivalence (CAP §21.6).'),
        ('W37', 'Attributing any forecast, quantile or metric to PyTorch, or presenting PyTorch as part of the model',
         'PyTorch is a host-side, test-only correctness reference (CAP §18.3, §21.3).'),
        ('W38', 'Stating or implying that v5 or DDNN runs in the demo, is the product, is live, or retrains in the browser',
         'v1 is the released product and demo (C302); browser runtime and equality between runtimes are unmeasured (CAP §18.2 item 4).'),
        ('W39', 'Describing DDNN\'s coverage or its PIT histogram as calibrated or guaranteed', 'Descriptive diagnostics only (C321).'),
        ('W40', 'Presenting the fit-cost or daily-cycle figures as qualification for daily operation',
         'A diagnostic only (§17.5 D3); v4\'s components were not refitted in CP-23 (C324).'),
        ('W41', 'Quoting any CP-23 number without its evidence class, or as a confirmatory or significance result',
         'Development after selection; exploratory intervals.'),
    ]
    src = [('R23', 'reports/distribution-challenger/report.md'), ('M23', 'reports/distribution-challenger/metrics.csv'),
           ('U23', 'reports/distribution-challenger/uncertainty.csv'), ('C23', 'reports/distribution-challenger/criteria.csv'),
           ('D23', 'reports/distribution-challenger/diagnostics.csv'), ('DEC23', 'reports/distribution-challenger/decisions.json'),
           ('P23', 'reports/distribution-challenger/protocol.json'), ('SEL23', 'reports/distribution-challenger/selection.json'),
           ('CT23', 'reports/distribution-challenger/controls.json'), ('PAR23', 'reports/distribution-challenger/parity.json'),
           ('REP23', 'reports/distribution-challenger/reproduction.json'), ('DG23', 'reports/distribution-challenger/diagnostics.json'),
           ('DC23', 'reports/distribution-challenger/daily-cycle.json'), ('FC23', 'reports/distribution-challenger/fit-cost.json'),
           ('LIC23', 'reports/distribution-challenger/licence-admission.md'),
           ('REF23', 'reports/distribution-challenger/reference-checks.json'),
           ('RA23', 'reports/distribution-challenger/resource-admission.md'),
           ('RP23', 'reports/distribution-challenger/replicates.parquet'),
           ('DR23', 'reports/distribution-challenger/draft-registry.json'),
           ('X23', 'reports/distribution-challenger/mlflow-export-draft/cp23.json'),
           ('I23', 'docs/track-b/evidence/cp-23/integration.md'), ('RET23', 'docs/track-b/evidence/cp-23/checkpoint-return.md'),
           ('CAP', 'capstone_v21.md (v21-r10), §21')]
    claims_md = '\n'.join([
        '# CP-23 research result: claim-to-evidence map',
        '',
        '**2026-10-04 · Draft for the next publication (PUBLISH_RULES 1.3 §11; capstone v21-r10 §21.9).** Each claim the',
        'publication may render is mapped to a committed file and row, in the format of the [CP-22](cp22-claims.md) map: `L<n>`',
        'is the physical line in the file, and for a CSV `L1` is the header. Values here are rounded from the saved values; the',
        'rendered surfaces bind to evidence records, never to these digits. The publication block registers this map after the',
        'Owner\'s landing' + ('' if adopted else ', and only if the Owner decides to publish the branch') + '.',
        '',
        '**Default evidence class.** Every CP-23 result is **`development_post_selection`** and exploratory.',
        '',
        '## Source keys', '', '| Key | File |', '|---|---|', *[_source_row(k, p) for k, p in src], '',
        '## Claim map: CP-23', '',
        'Status codes: **S** supported by a saved value or verdict; **S-d** supported, descriptive only; **Def** a',
        'definition; **O** an Owner decision.', '',
        '| ID | Claim | Source → table/row | Status |', '|---|---|---|---|', *claims, '',
        '## Withheld claims: do not use', '',
        'W1–W34 in the earlier maps stay in force. CP-23 adds:', '',
        '| ID | Withheld claim | Reason / controlling source |', '|---|---|---|',
        *[f'| {x} | {y} | {z} |' for x, y, z in withheld], '',
        '## Gaps: documented, not resolved', '',
        '| ID | Gap | Why it stays open |', '|---|---|---|',
        '| G26 | Learned or per-block member weights, other DDNN designs, new information | Excluded: 4.8 and stage 7 (CAP §21.2, §22). |',
        '| G27 | Hosted-service and product uses of DDNN | Unresolved in 4.6L; carried to 4.7 and 4.10 (LIC23). |',
        '| G28 | DDNN in another runtime (browser) and its equality with the evaluated code | Unmeasured (CAP §18.2 item 4). |',
        '| G29 | Status dates, the evidence tag and the landing record | They exist only at landing (pending fields). |', '',
    ]) + '\n'
    return claims_md, registry, dict(dec=dec, protocol=protocol, eq=eq, pf=pf, adopted=adopted)


def packet(root: Path, registry: dict, ctx: dict, claims_sha: str) -> tuple[str, list[str]]:
    dec, eq, pf, adopted = ctx['dec'], ctx['eq'], ctx['pf'], ctx['adopted']
    a = dec['adoption']
    (lm, rm), (lw, rw) = eq('v5', 'HGL', 'MAE'), eq('v5', 'HGL', 'WIS')
    (vm, rvm), (vw, rvw) = eq('v5', 'HG', 'MAE'), eq('v5', 'HG', 'WIS')
    ordinary = [(fo, pf('v5', fo)) for fo in ('fold_1', 'fold_2', 'fold_4', 'fold_5')]
    lo = min(ordinary, key=lambda x: float(x[1][1]['MAE']))
    hi = max(ordinary, key=lambda x: float(x[1][1]['MAE']))
    stress = pf('v5', 'fold_3')
    owner = registry['checkpoint']['owner']
    if adopted:
        slots = {
            'v5.question': 'Does a distributional neural network, added to {g:v4.name} as a one-third member, improve on it jointly '
                           'in point and interval accuracy?',
            'v5.change': '{g:v5.name} adds a distributional neural network, written in NumPy, as a fixed one-third member of the '
                         'blend of {g:v4.name}.',
            'v5.main_chart.headline': 'Against {g:v4.name}, the point-error score changed by {r:cp23.ratio.v5-HGL.MAE} and the '
                                      'interval score by {r:cp23.ratio.v5-HGL.WIS}, each as a share of the comparator\'s score, '
                                      'with 95% intervals.',
            'v5.reading': 'Both scores improved against the comparator, the six screening diagnostics hold and no test period was '
                          'decisively worse, so the rule set in advance adopted the generation.',
            'v5.not_established': 'Development evidence after selection, not a test on new data. One network design and a fixed '
                                  'weight were tested.',
            'v5.decision': 'Adopted in research on {g:v5.status.date}; the released model is unchanged.',
            'v5.evidence': 'the report, the review verdict, the claim map, the MLflow comparison',
            'transition.v4-v5.title': 'From v4 to v5: adding a distributional neural network',
        }
    else:
        slots = {
            f'branch.{owner}.question': '{g:' + owner + '.question}',
            f'branch.{owner}.comparator': '{g:v4.name}',
            f'branch.{owner}.result': 'Against {g:v4.name}, adding the distributional neural network as a one-third member changed '
                                      'the point-error score by {r:cp23.ratio.v5-HGL.MAE} and the interval score by '
                                      '{r:cp23.ratio.v5-HGL.WIS}, each as a share of the comparator\'s score, with 95% intervals.',
            f'branch.{owner}.reason': '{g:' + owner + '.status.reason}',
            f'branch.{owner}.decision': 'The Owner decides whether this branch is published; {g:v4.name} is unchanged.',
            f'branch.{owner}.evidence': 'the report, the review verdict, the claim map, the MLflow comparison',
        }
    findings = []
    for key, text in slots.items():
        findings += [f'{key}: {f}' for f in lint_slot(text)]
    L = []
    add = L.append
    add('# CP-23 publication packet (draft)\n')
    add('**PUBLISH_RULES 1.3 §11; capstone v21-r10 §21.9; template `docs/track-b/publication-packet-template.md` '
        '(SHA-256 4efb0185…829cf).** Filled from inside CP-23. Numbers are never typed into a surface: each value names the '
        'committed row it comes from, and the evidence layer re-derives it. Identities that exist only at landing are marked '
        '**pending-at-landing**. Publication follows the Owner\'s landing, as the next publication under PUBLISH_RULES 1.3'
        + ('' if adopted else ', and only if the Owner decides to publish the branch') + '; nothing here is published.\n')
    add('Pinned rules: PUBLISH_RULES 1.3 `5a660864f8b82943174741320c71087a3d0508f707edeb997cb2c1395c1c73b4`. The issued brief '
        'omitted the hash §21.9 says it records; the Owner ruled on 2026-10-04 to pin 1.3\'s single identity. It incorporates '
        'Publication Standard v1 `01d721c2…478cc` and presentation plan revision 3 `28119374…c812c`; all verified at the pre-run '
        'freeze (protocol `issued_and_inherited_sha256`).\n')
    add('## 1. Identity\n')
    add('| Item | Value |\n|---|---|')
    add('| Checkpoint | CP-23, brief `docs/track-b/evidence/cp-23/issued-brief.md` at `33f6b41202586b2fc4f1d87624c5978e1fd5bde1185ac05271429bf719502d0c` |')
    add('| Governing research plan and version | `capstone_v21.md` v21-r10 §21 (SHA-256 `6873c250…f6709`); rule `cp23-adoption` |')
    add('| Evidence tag and tip | `evidence/cp-23` at **pending-at-landing** (the evidence tip SHA is in the CP-23 return), frozen **pending-at-landing** |')
    add('| Final reviewed candidate (model code) | recorded in the CP-23 checkpoint return; a commit cannot contain its own SHA |')
    add('| Report, review verdict, landing record | `reports/distribution-challenger/report.md`; `docs/track-b/evidence/cp-23/integration.md`; '
        'landing record **pending-at-landing** |\n')
    add('## 2. The draft registry entries (standard §5)\n')
    add(f'Machine-readable: `reports/distribution-challenger/draft-registry.json`, derived mechanically from `decisions.json` '
        f'(adopted **{adopted}**, first unmet condition **{a["first_unmet_condition"]}**). Statuses are dated at landing; CP-23 '
        'registers nothing.\n')
    for e in registry['entries']:
        add(f'### `{e["id"]}` — {e["kind"]}\n')
        add('| Field | Value |\n|---|---|')
        for field in ('id', 'name', 'subtitle', 'short', 'kind'):
            add(f'| `{field}` | {e[field]} |')
        add('| `codes` | ' + '; '.join(f'{c["experiment"]}: {c["code"]} ({c["population"]}' + (f', {c["note"]}' if c['note'] else '') + ')'
                                     for c in e['codes']) + ' |')
        add('| `statuses` | ' + '; '.join(f'{s["status"]}, {s["date"]}, {s["source"]}' + (f', {s["reason"]}' if s['reason'] else '')
                                        for s in e['statuses']) + ' |')
        add(f'| `comparator` | `{e["comparator"]}` (named by §21.2 before results) |')
        add(f'| `population` | `{e["population"]}` |')
        add(f'| `evidence_class` | Development (`{e["evidence_class"]}`) |')
        add(f'| `plan`, `rules` | {e["plan"]}; `{", ".join(e["rules"])}` |')
        add('| `sources` | ' + ', '.join(f'`{s}`' for s in e['sources']) + ' |')
        add(f'| `claim_map` | `{e["claim_map"]}` |')
        add('| `run_keys` | ' + ', '.join(f'`{k}`' for k in e['run_keys']) + ' |')
        add(f'| `style`, `anchor` | {e["style"]}; {e["anchor"] or "none (a study arm has no anchor of its own)"}'
            + (' — v5\'s colour is the Owner\'s decision before the publication brief (PUBLISH_RULES §14)' if e['id'] == 'v5' else '') + ' |')
        add(f'| `checkpoint`, `after` | CP-23; {e["after"] or "n/a"} |')
        add(f'| `question`, `informed` | {e["question"] or "n/a"}; {e["informed"] or "none recorded"} |')
        add(f'| `predecessor` | {e["predecessor"] or "n/a"} |\n')
    add('Proposed code additions to existing entries: ' + '; '.join(
        f'`{x["entry"]}` gains CP-23 {x["code"]["code"]}' for x in registry['code_additions_to_existing_entries']) + '.\n')
    add('## 3. The claim map\n')
    add(f'`docs/track-b/research-content/cp23-claims.md` (SHA-256 `{claims_sha}`): every claim with its committed rows, the entry '
        'gates, the adoption decision, every §21.5 contrast and diagnostic, and the withheld claims W35–W41 (W1–W34 stay in force).\n')
    add('## 4. The derived headline quantities (standard §3.3), computed inside the checkpoint\n')
    add('| Quantity | Value | From |\n|---|---|---|')
    add(f'| (a) The pre-specified verdict | `cp23-adoption`: **{a["verdict"]}** (first unmet condition {a["first_unmet_condition"]}; '
        f'unmet {a["unmet_conditions"]}) | `reports/distribution-challenger/decisions.json` |')
    add('| The rule, in words, with the date it was set | v5 is adopted in research only if both paired score differences '
        'against v4 improve (the interval score\'s upper 95% endpoint below zero, the point error\'s at or below zero), it meets '
        'all six original screening diagnostics, the evaluation is complete and valid, and no fold is decisively worse. Set '
        '**2026-10-04** | `capstone_v21.md` v21-r10 §21.6; `protocol.json` `rule_verbatim` (committed before any main-run fit) |')
    add(f'| Distance from v4 (the rule\'s comparator), in the rule\'s unit | ΔS_MAE {num(rm["difference"])} (upper endpoint '
        f'{num(rm["ci_upper"])}, rule: at or below zero); ΔS_WIS {num(rw["difference"])} (upper endpoint {num(rw["ci_upper"])}, rule: '
        f'below zero) | `uncertainty.csv` L{lm}, L{lw} |')
    add('| N: policies tested against the same rule up to this decision | one eligible candidate, v5; DDNN alone and v3 plus a DDNN '
        'member are never eligible | `protocol.json` `policies`; §21.2 |')
    add('| "Point comparison" label needed? | no — every difference has a 95% interval | `uncertainty.csv` |')
    add(f'| (b) The change against v4, as a share of v4\'s score | S_MAE {pct(rm["ratio"])} (full {rm["ratio"]}); S_WIS '
        f'{pct(rw["ratio"])} (full {rw["ratio"]}) | `uncertainty.csv` L{lm}, L{lw} (`ratio`) |')
    add(f'| Its 95% interval | S_MAE [{pct(rm["ratio_ci_lower"])}, {pct(rm["ratio_ci_upper"])}]; S_WIS [{pct(rw["ratio_ci_lower"])}, '
        f'{pct(rw["ratio_ci_upper"])}] — CP-23\'s own draws: seed 15042, 2,000 replicates, 7-calendar-day blocks | '
        f'`uncertainty.csv` L{lm}, L{lw}; every draw in `replicates.parquet` |')
    add(f'| (b′) v5 against v3, for reference | S_MAE {pct(rvm["ratio"])}, S_WIS {pct(rvw["ratio"])} | `uncertainty.csv` L{vm}, L{vw} |')
    add(f'| (c) Mean absolute error per period, EUR/MWh (v5) | ordinary periods (folds 1, 2, 4, 5): {num(lo[1][1]["MAE"], 1)} '
        f'({lo[0].replace("_", " ")}) to {num(hi[1][1]["MAE"], 1)} ({hi[0].replace("_", " ")}); stress period, fold 3: '
        f'{num(stress[1]["MAE"], 1)} | `metrics.csv` per-fold rows |\n')
    add('## 5. Draft slot texts (standard §6)\n')
    add(('Only the outcome the rule yields is drafted: v5\'s chapter and the A3 transition "From v4 to v5: adding a distributional '
         'neural network".' if adopted else 'Only the outcome the rule yields is drafted: a branch card after v4, published only '
         'if the Owner so decides; no generation and no transition.')
        + ' Tokens: `{g:<id>.<field>}` registry fields; `{r:<record>}` evidence records. The §4 lint\'s code and status-word rules, '
        f'applied to every draft: **{len(findings)} findings**.\n')
    add('| Slot key | Draft |\n|---|---|')
    for key, text in slots.items():
        add(f'| `{key}` | ' + text.replace('|', '\\|') + ' |')
    add('')
    if adopted:
        add('### 5a. The adopted transition (A3)\n')
        add('| Field | Value |\n|---|---|')
        add('| Title | "From v4 to v5: adding a distributional neural network" |')
        add('| Predecessor and dates | `v4` (adopted in research 2026-09-30); v5 adopted **pending-at-landing** |')
        add('| The change | model: a NumPy distributional neural network (Johnson SU head) as a fixed one-third member |')
        add('| The comparator set in advance | v4, which is also the predecessor |')
        add(f'| The result, with its uncertainty and evidence class | S_MAE {pct(rm["ratio"])} [{pct(rm["ratio_ci_lower"])}, '
            f'{pct(rm["ratio_ci_upper"])}], S_WIS {pct(rw["ratio"])} [{pct(rw["ratio_ci_lower"])}, {pct(rw["ratio_ci_upper"])}]; development |')
        add('| Against the predecessor | the comparator is the predecessor |')
        add('| What it does not establish | performance on new data; equivalence; a learned weight |')
        add('| The decision, dated, and the route | adopted in research, **pending-at-landing**; `#v5` and `compare:v5` |\n')
    else:
        add('### 5a. The adopted transition (A3)\n\nNot applicable: v5 is not adopted, so no generation or transition exists.\n')
    add('### 5b. The released model\'s documentation (A4)\n\nNot applicable: the released model does not change. v1 is the released '
        'product and the demo; CP-23 is research only (§21.6).\n')
    add('### 5c. The chart routes (A5)\n')
    if adopted:
        add('| Chart | Its heading | The route\'s label | Where the route starts |\n|---|---|---|---|')
        add('| v5 paired differences against v4 | `#v5` | v5 against v4 | v5\'s chapter "Explore these results" and the transition summary |\n')
    else:
        add('None drafted: the branch card carries its deciding difference and interval in text; its reader route is the MLflow '
            f'comparison `compare:{owner}`, advertised only after the authorized upload and both route checks.\n')
    add('### 5d. Final-product transition / daily operation (A7/A8)\n\nNot applicable: the trigger is unmet — no final-product '
        'designation, rollout or Live (§16 unchanged).\n')
    add('## 6. The MLflow export\n')
    xsha = hashlib.sha256((root / OUT / 'mlflow-export-draft' / 'cp23.json').read_bytes()).hexdigest()
    add(f'- **Draft export:** `reports/distribution-challenger/mlflow-export-draft/cp23.json` (SHA-256 `{xsha}`), built by '
        '`scripts/mlflow_export.py --draft cp23` from CP-23\'s committed rows and `draft-registry.json`: parent `cp23` and one child '
        'per new policy (' + ', '.join(f'`cp23/{c}`' for c in registry['checkpoint']['children']) + '), experiment '
        '`delu-generations`. Pending fields: ' + ', '.join(f'`{k}`' for k in registry['pending_fields']) + '.')
    add('- **Local tracking:** the same runs, names and tags in `.local/mlruns/cp23`, read back equal '
        '(`reports/distribution-challenger/mlflow-local.json`). No upload, no network call; MLflow telemetry disabled.')
    diff = json.loads((root / OUT / 'published-export-diff.json').read_text())
    add('- **Record-level diff against the last published export:** the published set is unchanged — '
        f'`scripts/mlflow_export.py --diff-against {diff["against"]}` over {diff["runs_old"]} runs: only_identity '
        f'{diff["only_identity"]}, identity changes on {len(diff["identity_changes"])} runs, substantive changes '
        f'{diff["substantive_changes"] or "none"} (`reports/distribution-challenger/published-export-diff.json`); `--check` passes.')
    add(f'- **Expected routes:** `experiment`; `compare:{"v5" if adopted else owner}` — verified by REST and in Chromium and WebKit '
        'before it is linked, at publication.\n')
    add('## 7. Checks the checkpoint ran\n')
    add('- Re-derivation tests with negative controls: `tests/cp23/test_saved_evidence.py` (new-policy metrics from committed '
        'predictions, every interval from its stored replicates, the rule re-applied, byte-exact storage) and '
        '`tests/cp23/test_draft_export.py` (every exported value from a committed row; a renamed run is caught; the earlier drafts '
        'and the published set unchanged).')
    add('- The export contract: the draft matches its draft entries (`draft_problems`); the published export still matches the '
        'registry (`--check`).')
    add(f'- The §4 lint (code and status-word rules) on every draft slot text: {len(findings)} findings.')
    add('- Transition contract tests (`tests/test_43_publish_rules_migration.py`): pass unchanged; '
        + ('the publication block extends them with the v4 → v5 transition.' if adopted else 'not applicable (no transition).') + '\n')
    add('## 8. Publication completion receipt — intended identities (publisher completes the rest)\n')
    add('| Surface | Intended identity | Action / unchanged rationale | Observed public identity and UTC time | Verification record and result | Outstanding action |\n|---|---|---|---|---|---|')
    if adopted:
        rows = [('GitHub source / README', 'the Owner\'s landing SHA and the publication commit; README glance naming v5', 'push by the Owner'),
                ('GitHub Pages', 'rebuilt report: v5 chapter and the A3 transition (colour set by the Owner first)', 'deploy on the Owner\'s instruction'),
                ('Public MLflow', 'regenerated `cp23.json` equal to this draft apart from pending fields', 'authorized upload, then verification'),
                ('Hugging Face Space card', 'registry-derived model line; released model unchanged (v1)', 'deploy only if the card changes'),
                ('Hugging Face direct demo', 'v1 bundle unchanged', 'verified unchanged')]
    else:
        rows = [('GitHub source / README', 'after the Owner\'s decision on the branch', 'pending the Owner\'s decision'),
                ('GitHub Pages', 'after the Owner\'s decision; v4 unchanged meanwhile', 'pending the Owner\'s decision'),
                ('Public MLflow', 'regenerated `cp23.json` equal to this draft apart from pending fields', 'pending the Owner\'s decision'),
                ('Hugging Face Space card', 'unchanged', 'verified unchanged'),
                ('Hugging Face direct demo', 'v1 bundle unchanged', 'verified unchanged')]
    for surface, identity, action in rows:
        add(f'| {surface} | {identity} | {action} | pending (publisher) | pending (publisher) | pending |')
    add('')
    add('- **Publication-to-MLflow mapping:** pending — filled by the publisher.')
    add('- **Final product and daily updates:** not applicable (no final-product designation).')
    add('- **Completion disposition:** incomplete by construction until the publication; CP-23 publishes nothing.')
    add('- **Authority:** none exercised; every external action needs the Owner\'s instruction for that action.\n')
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
    (root / PACKET).write_text(text)
    print(json.dumps({'claims': str(CLAIMS), 'packet': str(PACKET), 'lint_findings': findings}), flush=True)
    return 0 if not findings else 10


if __name__ == '__main__':
    raise SystemExit(main())
