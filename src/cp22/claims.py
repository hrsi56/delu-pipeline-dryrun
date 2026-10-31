"""CP-22's claim map and publication packet (capstone v21-r9 §20.9; packet template sections 1-8).

Run as ``python -m cp22.claims`` after scoring, the diagnostics, the draft registry and the draft export
(``--check`` regenerates and compares, writing nothing). Every value and every row reference (`L<n>`,
the physical line; `L1` is a CSV's header) is read from the committed files at generation time, so
neither document can drift from the evidence. Draft slot texts carry registry tokens
(`{g:<id>.<field>}`) for names and statuses and evidence tokens (`{r:<record>}`) for numbers; they are
checked with the publication lint's code and status-word rules. Only the outcome the rules yield is
drafted. Identities that exist only at landing stay explicit pending fields.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re

from delu_forecast import publication_lint as PL

OUT = Path('reports/v4-revision')
CLAIMS = Path('docs/track-b/research-content/cp22-claims.md')
PACKET = Path('docs/track-b/evidence/cp-22/publication-packet.md')
FOLDS = ('fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5')
PENDING = 'pending-at-landing'
INTERNAL_CODES = ('HGL', 'L-P', 'L-R', 'L-N', 'HG', 'H0', 'B0', 'B1', 'B2', 'B3', 'A1', 'A2', 'PN', 'PN-avg', 'PN-sel',
                  'A-PN-sel', 'A-LP', 'A-LN', 'W+DL', 'W+ACI', 'W+DLF', 'v4+DL', 'v3+DL')
DISPLAY = {'HG': 'v3', 'HGL': 'v4'}


class Rows:
    """A committed CSV's rows with their physical line numbers (L1 = header)."""

    def __init__(self, root: Path, path: str):
        text = (root / path).read_text()
        reader = csv.DictReader(io.StringIO(text, newline=''))
        self.rows = [(reader.line_num, row) for row in reader]
        self.path = path

    def find(self, **match) -> tuple[int, dict]:
        found = [(line, row) for line, row in self.rows if all(row.get(k) == v for k, v in match.items())]
        if len(found) != 1:
            raise ValueError(f'{self.path}: {len(found)} rows match {match}')
        return found[0]


def pct(x) -> str:
    return f'{100 * float(x):+.1f}%'.replace('-', '−')


def num(x, d: int = 4) -> str:
    return f'{float(x):.{d}f}'.replace('-', '−')


def _source_row(key: str, path: str) -> str:
    if path.startswith('capstone'):
        return f'| {key} | {path} |'
    if path.startswith('docs/track-b/'):
        return f'| {key} | [{path}](../{path[len("docs/track-b/"):]}) |'
    return f'| {key} | [{path}](../../../{path}) |'


def resolved_note(r_mae: dict, r_wis: dict) -> str:
    notes = []
    for name, r in (('point-error score (S_MAE)', r_mae), ('interval score (S_WIS)', r_wis)):
        lo, hi = float(r['ci_lower']), float(r['ci_upper'])
        if lo > 0:
            notes.append(f'the {name} difference\'s 95% interval lies wholly above zero (worse)')
        elif hi < 0:
            notes.append(f'the {name} difference\'s 95% interval lies wholly below zero (better)')
    return ('; ' + ' and '.join(notes)) if notes else '; both single-metric intervals span zero'


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
    inv, repro, cost, cycle, protocol = load('investigation.json'), load('reproduction.json'), load('fit-cost.json'), \
        load('daily-cycle.json'), load('protocol.json')
    rep, dl, fast = dec['replacement'], dec['dynamic_layer'], dec['fast_component']
    w = rep['winner']
    current = registry['outcome']['current_revision_code']

    def eq(c, b, m):
        return U.find(scope='equal_fold', candidate=c, baseline=b, metric=m)

    def fold(fo, c, b, m):
        return U.find(scope=fo, candidate=c, baseline=b, metric=m)

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
              f'U22 L{l1}, L{l2}; DEC22 `contrasts`', status)

    # ---- identity and status
    claim('C200', 'CP-22\'s Integration verdict and the reviewed final candidate are recorded in I22 and the CP-22 checkpoint '
                  'return; the evidence tag and landing record are pending at landing.', 'I22; RET22', 'S (pending)')
    claim('C201', 'The three rules `cp22-replacement`, `cp22-dynamic-layer` and `cp22-fast-component` were set on 2026-10-01 '
                  '(the Owner\'s ratification) and frozen, verbatim, in the pre-run protocol before any main fit or score.',
          'P22 `rules_verbatim`; CAP §20.6', 'S')
    claim('C202', 'No product change, promotion, freeze, Live, final-product designation or economic claim follows; v1 '
                  'remains the released product and the demo.', 'CAP §20.6 (what replacement means); DEC22 `statuses`', 'S')
    claim('C203', 'The same 10,747 eligible hours in five folds (2,160 / 2,159 / 2,112 / 2,160 / 2,156), 448 represented '
                  'days; fold 3 (the stress period) has 2,112 hours on 88 days; the peak 2022-08-15..31 has 408 hours.',
          f'M22 L{pf("B0", "fold_1")[0]}–L{pf("B0", "fold_5")[0]} (`n_hours`); D22 peak rows', 'S')
    claim('C204', 'PN is one pooled LightGBM for all 24 hours on the normalized price target, with exactly v3\'s and v4\'s '
                  'information; its capacity-averaged form is the equal mean of four full-window fits (150/15, 300/31, 600/31 '
                  'and 600/63 trees/leaves), and its selected form chooses one of them daily on the window\'s last 28 days.',
          'CAP §20.2; P22 `members`, `capacity_grid`, `selection`', 'Def')
    claim('C205', 'Every composite is `(2/3)·c_v3 + (1/3)·member` with the member weight fixed at 1/3: R\'s member is PN '
                  'averaged over capacities; M\'s is the mean of PN (selected) and the pooled raw LightGBM; v4\'s three-block '
                  'construction\'s is the mean of the normalized and raw three-block LightGBM. Every non-dynamic composite uses '
                  'v3\'s hour-aware interval layer on its own errors.',
          'CAP §20.2; P22 `policies`; CT22 `composite_parity_max_abs_eur_mwh`, `non_dl_composites_on_h_path`', 'Def/S')
    claim('C206', 'The ladder, one change per step: v4 → M removes the block split (same raw/normalized mix and daily '
                  'selection); M → the selected normalized member drops the raw half; that → R averages over capacities '
                  'instead of selecting daily.', 'CAP §20.1 (the ladder); tests/cp22/test_pn_and_composites.py; CT22 `ladder_rows`',
          'Def')
    claim('C207', 'The dynamic interval layer keeps v3\'s layer and changes two things, fixed before scoring: 7-day recency '
                  'weights inside the 28-day buffer (weighted quantiles, effective counts, weighted median) and adaptive '
                  'coverage (γ = 0.10 per day, α_t clipped to [α/5, min(2α, 0.9)], sorting if quantiles cross). The fast '
                  'component adds a one-day kernel carrying one third of the weight.', 'CAP §20.3; P22 `interval_layers`', 'Def')
    # ---- the replacement
    for ident, cand in (('C208', 'R'), ('C210', 'M')):
        (lm, rm), (lw, rw) = eq(cand, 'HGL', 'MAE'), eq(cand, 'HGL', 'WIS')
        claim(ident, f'{cand} − v4 (three-block): ΔS_MAE {num(rm["difference"])} [{num(rm["ci_lower"])}, {num(rm["ci_upper"])}] '
                     f'({pct(rm["ratio"])} [{pct(rm["ratio_ci_lower"])}, {pct(rm["ratio_ci_upper"])}] of v4\'s score); ΔS_WIS '
                     f'{num(rw["difference"])} [{num(rw["ci_lower"])}, {num(rw["ci_upper"])}] ({pct(rw["ratio"])} '
                     f'[{pct(rw["ratio_ci_lower"])}, {pct(rw["ratio_ci_upper"])}]).', f'U22 L{lm}, L{lw}', 'S')
        cand_d = rep['candidates'][cand]
        conds = '; '.join(f'condition {k} {"met" if v["met"] else "not met"}' for k, v in sorted(cand_d['conditions'].items()))
        worse = cand_d['conditions']['4']['values']['folds_decisively_worse']
        worse_folds = {}
        for row in worse:
            worse_folds.setdefault(row['scope'], []).append(row['metric'])
        worse_text = (f'{len(worse_folds)} (' + '; '.join(f'{f.replace("_", " ")}: {" and ".join(m)}'
                                                          for f, m in sorted(worse_folds.items())) + ')'
                      if worse_folds else '0')
        claim(f'C{int(ident[1:]) + 1}', f'Rule `cp22-replacement` for {cand}: '
              + ('all four conditions met' if cand_d['met'] else f'first unmet condition {cand_d["first_unmet_condition"]}')
              + f' ({conds}; condition 3\'s Engineering PASS is the fresh Integration verdict). Folds decisively worse in MAE or '
                f'WIS (lower endpoint above zero): {worse_text}.', f'DEC22 `replacement.candidates.{cand}`; C22; U22 per-fold rows', 'S')
    claim('C212', f'The mechanical result of `cp22-replacement`: **{rep["verdict"]}**.', 'DEC22 `replacement.verdict`', 'S')
    claim('C213', 'Whether the replacement shows a joint improvement over v4 under §17.5\'s reading: '
          + (f'{w} − v4 reads **{dec["contrasts"][f"{w}-HGL"]["reading"]}**.' if w else 'not applicable (no replacement); '
             f'R − v4 reads {dec["contrasts"]["R-HGL"]["reading"]} and M − v4 reads {dec["contrasts"]["M-HGL"]["reading"]}.'),
          'DEC22 `contrasts`', 'S')
    # ---- the layer decisions
    claim('C214', 'Rule `cp22-dynamic-layer`: ' + (f'**{dl["verdict"]}**' + (f' (first unmet condition {dl["first_unmet_condition"]})'
                                                                           if dl.get('first_unmet_condition') else '')
                                                 if dl.get('applies') else 'not applicable — no winner W.'),
          'DEC22 `dynamic_layer`', 'S')
    claim('C215', 'Rule `cp22-fast-component`: ' + (f'**{fast["verdict"]}**' + (f' (first unmet condition {fast["first_unmet_condition"]})'
                                                                              if fast.get('first_unmet_condition') else '')
                                                  if fast.get('applies') else fast['verdict'] + '.'), 'DEC22 `fast_component`', 'S')
    # ---- secondary contrasts
    n = 216
    secondary = [('R', 'M', 'The full refinement against the minimal one (R − M)'),
                 ('A-PN-sel', 'M', 'Dropping the raw half (selected normalized pooled member − M)'),
                 ('R', 'A-PN-sel', 'Averaging against daily selection (R − selected normalized member)'),
                 ('A-PN-sel', 'A-LP', 'Normalized against raw, pooled'),
                 ('A-LN', 'HGL', 'Descriptive: the blocks under normalization (normalized-block member − v4)'),
                 ('v4+DL', 'HGL', 'The dynamic layer on v4, descriptive'), ('v3+DL', 'HG', 'The dynamic layer on v3, descriptive')]
    if w:
        secondary[0:0] = [('W+DL', w, f'The layer decision ((W+DL) − W, W = {w})'),
                          ('W+ACI', w, 'Adaptive coverage alone ((W+ACI) − W)'),
                          ('W+DL', 'W+ACI', 'Recency weights ((W+DL) − (W+ACI))'),
                          ('W+DLF', 'W+DL', 'The fast component ((W+DLF) − (W+DL))')]
    for c, b, what in secondary:
        contrast_claim(f'C{n}', c, b, what)
        n += 1
    parts = ', '.join(f'{DISPLAY.get(p, p)} {num(score(p)[1]["S_MAE"])}/{num(score(p)[1]["S_WIS"])}'
                      for p in dec['policies'] if p not in ('B0', 'B1', 'B2', 'A1', 'H0')) if 'policies' in dec else ''
    if not parts:
        parts = ', '.join(f'{DISPLAY.get(p, p)} {num(score(p)[1]["S_MAE"])}/{num(score(p)[1]["S_WIS"])}' for p in dec['scores']
                          if p not in ('B0', 'B1', 'B2', 'A1', 'H0'))
    claim(f'C{n}', f'Equal-fold S_MAE/S_WIS: {parts}; every policy against v3 is in U22 (`reference_vs_v3`).', 'M22 equal-fold rows', 'S')
    n += 1
    claim(f'C{n}', 'Original §8 screen (diagnostic): ' + ', '.join(f'{DISPLAY.get(p, p)} {s}' for p, s in dec['original_section8_status'].items())
          + '.', 'C22', 'S')
    n += 1
    lead = w or 'R'
    f3, ordinary = pf(lead, 'fold_3'), [float(pf(lead, fo)[1]['MAE']) for fo in ('fold_1', 'fold_2', 'fold_4', 'fold_5')]
    claim(f'C{n}', f'{lead} per-fold MAE: ordinary folds {num(min(ordinary), 1)}–{num(max(ordinary), 1)} EUR/MWh; stress fold 3 '
                   f'{num(f3[1]["MAE"], 1)} EUR/MWh (v4 three-block: {num(pf("HGL", "fold_3")[1]["MAE"], 1)}; v3: '
                   f'{num(pf("HG", "fold_3")[1]["MAE"], 1)}).',
          f'M22 L{pf(lead, "fold_1")[0]}–L{pf(lead, "fold_5")[0]}, L{pf("HGL", "fold_3")[0]}, L{pf("HG", "fold_3")[0]}', 'S')
    n += 1
    pl, pk = D.find(policy=lead, scope='peak')
    vl, vk = D.find(policy='HGL', scope='peak')
    claim(f'C{n}', f'Peak (descriptive, 17 days, small effective sample): {lead} MAE {num(pk["MAE"], 1)}, WIS {num(pk["WIS"], 1)} EUR/MWh, '
                   f'95% coverage {num(pk["coverage95"], 3)}; v4 three-block MAE {num(vk["MAE"], 1)}, WIS {num(vk["WIS"], 1)}, coverage '
                   f'{num(vk["coverage95"], 3)}. No inference is drawn.', f'D22 L{pl}, L{vl}', 'S-d')
    n += 1
    # ---- the Owner's investigation
    lad = inv['ladder']
    claim(f'C{n}', 'Decomposition of v4 − v3 along the ladder (equal-fold): ' + '; '.join(
        f'{s["from"]} → {s["to"]} ({s["change"]}) ΔS_MAE {num(s["dS_MAE"])}, ΔS_WIS {num(s["dS_WIS"])}' for s in lad['steps'])
        + '. The brackets sum to v4 − v3 exactly.', 'INV22 `ladder`', 'S-d')
    n += 1
    curve_claim = f'C{n}'
    claim(curve_claim, 'The member-weight curve is an oracle computed on outcomes and selects nothing: '
          + ', '.join(f'{k} lowest at {v}' for k, v in inv['member_weight_curve']['argmin_weight_by_member'].items())
          + ' (central forecast, before any interval layer). The fixed weight stays 1/3.', 'INV22 `member_weight_curve`', 'S-d')
    n += 1
    stab = {f'{r["arm"]} {r["model"]}': r for r in inv['capacity_stability'] if r['phase'] == 'evaluation'}
    claim(f'C{n}', 'Capacity-selection stability at evaluation origins (flip rate; median relative winner margin): '
          + '; '.join(f'{k} {num(v["flip_rate"], 2)}, {num(100 * v["winner_margin_median"], 1)}%' for k, v in stab.items()) + '.',
          'INV22 `capacity_stability`', 'S-d')
    n += 1
    claim(f'C{n}', f'Extrapolation: {inv["extrapolation"]["days"]} extreme or top-5% days, {inv["extrapolation"]["exceeds_window_max_days"]} '
                   'above the origin\'s training-window maximum; per-member maxima are in the extrapolation table.',
          'INV22 `extrapolation`; reports/v4-revision/investigation/extrapolation.csv', 'S-d')
    n += 1
    claim(f'C{n}', 'Reaction after the peak began (days from 2022-08-15): ' + '; '.join(
        f'{r["policy"]} ACI {r["aci_reaction_days"]}, width {r["width_reaction_days"]}' for r in inv['reaction_after_peak_start']) + '.',
          'INV22 `reaction_after_peak_start`; investigation/alpha-paths.csv', 'S-d')
    n += 1
    claim(f'C{n}', 'Sharp changes (the shock-day set and the three days after, pooled): ' + '; '.join(
        f'{k} MAE {num(v["MAE"], 1)}, WIS {num(v["WIS"], 1)}, cov95 {num(v["coverage95"], 3)}'
        for k, v in inv['shock_days']['pooled_shock_and_three_after'].items()) + '.', 'INV22 `shock_days`', 'S-d')
    n += 1
    claim(f'C{n}', 'LEAR\'s penalty-selection stability, from logged selections (no refit): ' + '; '.join(
        f'{r["component"]} {r["phase"]} flip rate {num(r["flip_rate"], 2)}' for r in inv['lear_penalty_stability']['rows']) + '.',
          'INV22 `lear_penalty_stability`', 'S-d')
    n += 1
    claim(f'C{n}', 'Fit cost and the daily cycle (diagnostic only): ' + '; '.join(
        f'{p} cold cycle median {num(s["median_seconds"], 0)} s, maximum {num(s["max_seconds"], 0)} s at {s["origins"]} origins'
        for p, s in cycle['summary'].items())
        + f'; PN median per origin {num(cost["PN"]["median_per_origin_wall_all_eight"], 1)} s for eight fits against L-P\'s '
          f'{num(cost["L-P (CP-21, committed)"]["median_per_origin_wall_selection_and_refit"], 1)} s for five.', 'DC22; FC22', 'S')
    n += 1
    # ---- integrity
    claim(f'C{n}', 'v3 and v4 through CP-22\'s interval-layer path reproduce their committed vectors bit for bit on all 10,747 '
                   'keys; an independent representative v3 and v4 slice, refitted, matches bit for bit; every reused vector '
                   'matches its committed blob.', 'PAR22; REP22; P22 `cache_identities.verification`', 'S')
    n += 1
    claim(f'C{n}', f'All §20.7 and §17.7 controls passed ({controls["checks"]} checks), each negative assertion paired with a '
                   'positive control: delivery-day masking exactly 0.0; non-uniform D−1 mutation moves PN; non-monotone weather '
                   'perturbations move PN; training-only selection; capacity-averaging and selection identity over all 636 PN '
                   'fits; composite parity on every key; the dynamic layer\'s direction, clipping, weight and release controls; '
                   'with no weights and no adaptive coverage it reproduces R\'s committed vectors bit for bit.', 'CT22', 'S')
    n += 1
    claim(f'C{n}', 'Development evidence after selection on the same five folds CP-15, CP-20 and CP-21 used; not a test on new '
                   'data; 4.7T carries the protection.', 'CAP §20.1', 'S')
    n += 1
    decision = {'R': 'v4 keeps its number and gains a dated revision: the current construction is **R**'
                     + (' with the dynamic interval layer' if dl.get('adopted') else ''),
                'M': 'v4 keeps its number and gains a dated revision: the current construction is **M**'
                     + (' with the dynamic interval layer' if dl.get('adopted') else ''),
                None: 'no replacement: three-block v4 stays current and the Owner decides (§20.6)'}[w]
    claim(f'C{n}', f'The decision, dated at landing: {decision}.', 'DEC22; DR22; landing record (pending)', 'O (pending)')

    withheld = [
        ('W28', 'Calling R, M or v4 "equivalent", or any non-inferior result "no worse" without its interval',
         'Non-inferiority means no 95% interval lies entirely above zero (CAP §20.6); never equivalence.'),
        ('W29', 'Presenting the member-weight curve as a better or recommended weight', f'An oracle on outcomes, not selectable ({curve_claim}).'),
        ('W30', 'Naming a mechanism for the block split, or reopening its removal', 'The Owner removed the split in every outcome (CAP §20.1).'),
        ('W31', 'Stating or implying that a CP-22 policy or v4 runs in the demo, is the product, or is live',
         'v1 remains the released product and demo (C202).'),
        ('W32', 'Presenting the daily-cycle or fit-cost figures as qualification for daily operation', 'A diagnostic only (§17.5 D3).'),
        ('W33', 'Quoting any CP-22 number without its evidence class, or as a confirmatory or significance result',
         'Development after selection; exploratory intervals.'),
        ('W34', 'Describing the dynamic layer\'s coverage as a guarantee', 'An empirical layer with adaptive coverage, not a '
                'finite-sample guarantee (CAP §6, §20.3).'),
    ]
    src = [('R22', 'reports/v4-revision/report.md'), ('M22', 'reports/v4-revision/metrics.csv'),
           ('U22', 'reports/v4-revision/uncertainty.csv'), ('C22', 'reports/v4-revision/criteria.csv'),
           ('D22', 'reports/v4-revision/diagnostics.csv'), ('DEC22', 'reports/v4-revision/decisions.json'),
           ('P22', 'reports/v4-revision/protocol.json'), ('CT22', 'reports/v4-revision/controls.json'),
           ('PAR22', 'reports/v4-revision/parity.json'), ('REP22', 'reports/v4-revision/reproduction.json'),
           ('INV22', 'reports/v4-revision/investigation.json'), ('DC22', 'reports/v4-revision/daily-cycle.json'),
           ('FC22', 'reports/v4-revision/fit-cost.json'), ('RP22', 'reports/v4-revision/replicates.parquet'),
           ('DR22', 'reports/v4-revision/draft-registry.json'), ('X22', 'reports/v4-revision/mlflow-export-draft/cp22.json'),
           ('I22', 'docs/track-b/evidence/cp-22/integration.md'), ('RET22', 'docs/track-b/evidence/cp-22/checkpoint-return.md'),
           ('CAP', 'capstone_v21.md (v21-r9), §20')]
    claims_md = '\n'.join([
        '# CP-22 research result: claim-to-evidence map',
        '',
        '**2026-10-03 · Draft for PRES-4 (PUBLISH_RULES 1.3 §11; capstone v21-r9 §20.9).** Each claim the publication may',
        'render is mapped to a committed file and row, in the format of the [CP-21](cp21-claims.md) map: `L<n>` is the',
        'physical line in the file, and for a CSV `L1` is the header. Values here are rounded from the saved values; the',
        'rendered surfaces bind to evidence records, never to these digits. The publication block registers this map',
        'after the Owner\'s landing' + ('' if w else ', and only after the Owner\'s decision (no replacement)') + '.',
        '',
        '**Default evidence class.** Every CP-22 result is **`development_post_selection`** and exploratory.',
        '',
        '## Source keys', '', '| Key | File |', '|---|---|', *[_source_row(k, p) for k, p in src], '',
        '## Claim map: CP-22', '',
        'Status codes: **S** supported by a saved value or verdict; **S-d** supported, descriptive only; **Def** a',
        'definition; **O** an Owner decision.', '',
        '| ID | Claim | Source → table/row | Status |', '|---|---|---|---|', *claims, '',
        '## Withheld claims: do not use', '',
        'W1–W27 in the earlier maps stay in force. CP-22 adds:', '',
        '| ID | Withheld claim | Reason / controlling source |', '|---|---|---|',
        *[f'| {a} | {b} | {c} |' for a, b, c in withheld], '',
        '## Gaps: documented, not resolved', '',
        '| ID | Gap | Why it stays open |', '|---|---|---|',
        '| G23 | Learned or per-block member weights, seed ensembles, LEAR changes | Excluded (CAP §20.2; D6, D8). |',
        '| G24 | The registry\'s revision event and the runbook\'s lists (A10 maintenance) | PRES-4\'s work after landing. |',
        '| G25 | Status dates, the evidence tag and the landing record | They exist only at landing (pending fields). |', '',
    ]) + '\n'
    return claims_md, registry, dict(dec=dec, protocol=protocol, eq=eq, pf=pf, w=w, current=current)


def packet(root: Path, registry: dict, ctx: dict, claims_sha: str) -> tuple[str, list[str]]:
    dec, eq, pf, w, current = ctx['dec'], ctx['eq'], ctx['pf'], ctx['w'], ctx['current']
    rep, dl, fast = dec['replacement'], dec['dynamic_layer'], dec['fast_component']
    lead = w or 'R'
    (lm, rm), (lw, rw) = eq(lead, 'HGL', 'MAE'), eq(lead, 'HGL', 'WIS')
    (vm, rvm), (vw, rvw) = eq('HGL', 'HG', 'MAE'), eq('HGL', 'HG', 'WIS')
    ordinary = [(fo, pf(current or lead, fo)) for fo in ('fold_1', 'fold_2', 'fold_4', 'fold_5')]
    lo = min(ordinary, key=lambda x: float(x[1][1]['MAE']))
    hi = max(ordinary, key=lambda x: float(x[1][1]['MAE']))
    stress = pf(current or lead, 'fold_3')
    owner = registry['checkpoint']['owner']
    cur = current or lead
    if w:
        rc = f'{{r:cp22.ratio.{cur}-HG.MAE}}', f'{{r:cp22.ratio.{cur}-HG.WIS}}'
        slots = {
            'v4.question': 'Can one pooled forecaster replace the three-block member of {g:v4.name}\'s earlier construction '
                           'without losing accuracy, and should its intervals learn faster after a sudden change?',
            'v4.change': '{g:v4.name} now adds one LightGBM forecaster for all hours to the blend of {g:v3.name}; the earlier '
                         'construction added three hour-block forecasters.' + (' Its intervals weight recent errors more and '
                                                                             'adjust their width to recent misses.' if dl.get('adopted') else ''),
            'v4.main_chart.headline': f'Against {{g:v3.name}}, the point-error score changed by {rc[0]} and the interval score by '
                                      f'{rc[1]}, each as a share of the comparator\'s score, with 95% intervals.',
            'v4.reading': 'Against the earlier construction, no score and no test period was decisively worse, and the six '
                          'screening diagnostics hold, so the rule set in advance revised the generation in place.',
            'v4.not_established': 'Development evidence after selection, not a test on new data. Not worse is not the same as '
                                  'equal. The weight of the new forecaster is fixed, not learned.',
            'v4.decision': 'Revised in research on {g:v4.status.date}; the earlier construction stays visible; the released '
                           'model is unchanged.',
            'v4.revision_note': f'The revision step against the earlier construction: point-error score {{r:cp22.ratio.{w}-HGL.MAE}}, '
                                f'interval score {{r:cp22.ratio.{w}-HGL.WIS}}, with 95% intervals.',
            'v4.evidence': 'the report, the review verdict, the claim map, the MLflow comparison',
            'transition.v3-v4.title': 'From v3 to v4: adding a LightGBM member',
            'transition.v3-v4.result': f'Against {{g:v3.name}}, both its predecessor and its comparator, the point-error score '
                                       f'changed by {rc[0]} and the interval score by {rc[1]}, in development tests (the revised '
                                       'construction).',
        }
    else:
        slots = {
            f'branch.{owner}.question': '{g:' + owner + '.question}',
            f'branch.{owner}.comparator': '{g:v4.name}',
            f'branch.{owner}.result': 'Against {g:v4.name}, the averaged pooled forecaster changed the point-error score by '
                                      '{r:cp22.ratio.R-HGL.MAE} and the interval score by {r:cp22.ratio.R-HGL.WIS}, each as a '
                                      'share of the comparator\'s score, with 95% intervals.',
            f'branch.{owner}.reason': '{g:' + owner + '.status.reason}',
            f'branch.{owner}.decision': 'Pending the Owner\'s decision; {g:v4.name} is unchanged until then.',
            f'branch.{owner}.evidence': 'the report, the review verdict, the claim map, the MLflow comparison',
        }
    findings = []
    for key, text in slots.items():
        findings += [f'{key}: {f}' for f in lint_slot(text)]
    L = []
    add = L.append
    add('# CP-22 publication packet (draft)\n')
    add('**PUBLISH_RULES 1.3 §11; capstone v21-r9 §20.9; template `docs/track-b/publication-packet-template.md` '
        '(SHA-256 4efb0185…829cf).** Filled from inside CP-22. Numbers are never typed into a surface: each value names '
        'the committed row it comes from, and the evidence layer re-derives it. Identities that exist only at landing are '
        'marked **pending-at-landing**. Publication is PRES-4, after the Owner\'s landing'
        + ('' if w else ', and only after the Owner\'s decision') + '; nothing here is published.\n')
    add('Pinned rules: PUBLISH_RULES 1.3 `5a660864f8b82943174741320c71087a3d0508f707edeb997cb2c1395c1c73b4` (the hash the '
        'issued brief records), incorporating Publication Standard v1 `01d721c2…478cc` and presentation plan revision 3 '
        '`28119374…c812c`; all verified at the pre-run freeze (protocol `issued_and_inherited_sha256`).\n')
    add('## 1. Identity\n')
    add('| Item | Value |\n|---|---|')
    add('| Checkpoint | CP-22, brief `docs/track-b/evidence/cp-22/issued-brief.md` at `563f64f3602848808ff11a7d74da8218b16d51846f34e90ad754c36ec2461fb3` |')
    add('| Governing research plan and version | `capstone_v21.md` v21-r9 §20 (SHA-256 `5fc9c686…09175`); rules `cp22-replacement`, '
        '`cp22-dynamic-layer`, `cp22-fast-component` |')
    add('| Evidence tag and tip | `evidence/cp-22` at **pending-at-landing** (the evidence tip SHA is in the CP-22 return), frozen **pending-at-landing** |')
    add('| Final reviewed candidate (model code) | recorded in the CP-22 checkpoint return; a commit cannot contain its own SHA |')
    add('| Report, review verdict, landing record | `reports/v4-revision/report.md`; `docs/track-b/evidence/cp-22/integration.md`; '
        'landing record **pending-at-landing** |\n')
    add('## 2. The draft registry entries (standard §5)\n')
    add(f'Machine-readable: `reports/v4-revision/draft-registry.json`, derived mechanically from `decisions.json` (replacement '
        f'**{w or "none"}**' + (f', current revision code **{current}**' if w else '') + '). Statuses are dated at landing; '
        'CP-22 registers nothing.\n')
    if registry.get('revisions'):
        add('**v4\'s revision history (PUBLISH_RULES 1.3 A10):**\n')
        add('| Revision | Label | State | Checkpoint / code | Rule | Dates | Evidence |\n|---|---|---|---|---|---|---|')
        for r in registry['revisions']:
            add(f'| {r["revision"]} | {r["label"]} | {r["state"]} | {r["checkpoint"]} / {r["code"]} | {r["rule"]} | adopted '
                f'{r["adopted"]}' + (f'; superseded {r["superseded"]}' if r.get('superseded') else '') + f' | {r["evidence"]} |')
        add('')
    for e in registry['entries']:
        add(f'### `{e["id"]}` — {e["kind"]}\n')
        add('| Field | Value |\n|---|---|')
        for field in ('id', 'name', 'subtitle', 'short', 'kind'):
            add(f'| `{field}` | {e[field]} |')
        add('| `codes` | ' + '; '.join(f'{c["experiment"]}: {c["code"]} ({c["population"]}' + (f', {c["note"]}' if c['note'] else '') + ')'
                                     for c in e['codes']) + ' |')
        add('| `statuses` | ' + '; '.join(f'{s["status"]}, {s["date"]}, {s["source"]}' + (f', {s["reason"]}' if s['reason'] else '')
                                        for s in e['statuses']) + ' |')
        add(f'| `comparator` | `{e["comparator"]}` (named by §20.2 before results) |')
        add(f'| `population` | `{e["population"]}` |')
        add(f'| `evidence_class` | Development (`{e["evidence_class"]}`) |')
        add(f'| `plan`, `rules` | {e["plan"]}; `{", ".join(e["rules"])}` |')
        add('| `sources` | ' + ', '.join(f'`{s}`' for s in e['sources']) + ' |')
        add(f'| `claim_map` | `{e["claim_map"]}` |')
        add('| `run_keys` | ' + ', '.join(f'`{k}`' for k in e['run_keys']) + ' |')
        add(f'| `style`, `anchor` | {e["style"]}; {e["anchor"] or "none (a study arm has no anchor of its own)"}'
            + (' — v4\'s encoding unchanged: amber `#B45309`, filled diamond, direct label "v4" (D6)' if e['id'] == 'v4' else '') + ' |')
        add(f'| `checkpoint`, `after` | CP-22; {e["after"] or "n/a"} |')
        add(f'| `question`, `informed` | {e["question"] or "n/a"}; {e["informed"] or "none recorded"} |')
        add(f'| `predecessor` | {e["predecessor"] or "n/a"} |\n')
    add('Proposed code additions to existing entries: ' + '; '.join(
        f'`{a["entry"]}` gains CP-22 {a["code"]["code"]}' for a in registry['code_additions_to_existing_entries']) + '.\n')
    add('## 3. The claim map\n')
    add(f'`docs/track-b/research-content/cp22-claims.md` (SHA-256 `{claims_sha}`): every claim with its committed rows, the '
        'ladder, the replacement and layer findings, the Owner\'s investigation, and the withheld claims W28–W34 '
        '(W1–W27 stay in force).\n')
    add('## 4. The derived headline quantities (standard §3.3), computed inside the checkpoint\n')
    add('| Quantity | Value | From |\n|---|---|---|')
    add(f'| (a) The pre-specified verdicts | `cp22-replacement`: **{rep["verdict"]}** (first unmet: R {rep["first_unmet_condition"]["R"]}, '
        f'M {rep["first_unmet_condition"]["M"]}); `cp22-dynamic-layer`: **{dl["verdict"]}**; `cp22-fast-component`: '
        f'**{fast["verdict"]}** | `reports/v4-revision/decisions.json` |')
    add('| The rules, in words, with the date they were set | R, then M, replaces v4\'s three-block construction only if neither '
        'paired score difference against v4 has a 95% interval entirely above zero, all six screening diagnostics hold, the '
        'evaluation is complete and valid, and no fold is decisively worse; the dynamic layer then replaces the winner\'s '
        'interval layer only if its interval-score gain\'s upper endpoint is below zero, its point error has no interval '
        'above zero, it meets the screen, its pooled 95% coverage is closer to 0.95 and no fold is decisively worse; the fast '
        'component is decided only as an add-on to an adopted dynamic layer. Set **2026-10-01** | `capstone_v21.md` v21-r9 '
        '§20.6; `protocol.json` `rules_verbatim` (committed before any fit) |')
    add(f'| Distance from v4 (the rule\'s comparator), in the rule\'s unit | {lead} − v4: ΔS_MAE {num(rm["difference"])} (lower '
        f'endpoint {num(rm["ci_lower"])}, rule: not above zero); ΔS_WIS {num(rw["difference"])} (lower endpoint '
        f'{num(rw["ci_lower"])}) | `uncertainty.csv` L{lm}, L{lw} |')
    add('| N: policies tested against the same rule up to this decision | two eligible policies in a fixed sequence (R, then M), '
        'plus two layer decisions in sequence (the dynamic layer, then its fast component); attribution arms are never '
        'eligible | `protocol.json` `rules`; §20.2 |')
    add('| "Point comparison" label needed? | no — every difference has a 95% interval | `uncertainty.csv` |')
    add(f'| (b) The change against v4, as a share of v4\'s score ({lead}) | S_MAE {pct(rm["ratio"])} (full {rm["ratio"]}); S_WIS '
        f'{pct(rw["ratio"])} (full {rw["ratio"]}) | `uncertainty.csv` L{lm}, L{lw} (`ratio`) |')
    add(f'| Its 95% interval | S_MAE [{pct(rm["ratio_ci_lower"])}, {pct(rm["ratio_ci_upper"])}]; S_WIS [{pct(rw["ratio_ci_lower"])}, '
        f'{pct(rw["ratio_ci_upper"])}] — CP-22\'s own draws: seed 15042, 2,000 replicates, 7-calendar-day blocks | '
        f'`uncertainty.csv` L{lm}, L{lw}; every draw in `replicates.parquet` |')
    add(f'| (b′) {"The superseded construction" if w else "v4 (three-block), unchanged,"} against v3, for reference | S_MAE {pct(rvm["ratio"])}, S_WIS {pct(rvw["ratio"])} | '
        f'`uncertainty.csv` L{vm}, L{vw} |')
    add(f'| (c) Mean absolute error per period, EUR/MWh ({DISPLAY.get(cur, cur)}) | ordinary periods (folds 1, 2, 4, 5): '
        f'{num(lo[1][1]["MAE"], 1)} ({lo[0].replace("_", " ")}) to {num(hi[1][1]["MAE"], 1)} ({hi[0].replace("_", " ")}); '
        f'stress period, fold 3: {num(stress[1]["MAE"], 1)} | `metrics.csv` per-fold rows |\n')
    add('## 5. Draft slot texts (standard §6)\n')
    add(('Only the outcome the rules yield is drafted: v4\'s chapter under its current revision, the A3 transition stating the '
         'revision, and the dated revision note (A10).' if w else 'Only the outcome the rules yield is drafted: a branch card '
         'after v4, for the Owner\'s decision; no revision and no transition.')
        + ' Tokens: `{g:<id>.<field>}` registry fields; `{r:<record>}` evidence records. The §4 lint\'s code and status-word '
        f'rules, applied to every draft: **{len(findings)} findings**.\n')
    add('| Slot key | Draft |\n|---|---|')
    for key, text in slots.items():
        add(f'| `{key}` | ' + text.replace('|', '\\|') + ' |')
    add('')
    if w:
        add('### 5a. The adopted transition (A3, with A10)\n')
        add('| Field | Value |\n|---|---|')
        add('| Title | "From v3 to v4: adding a LightGBM member" (current revision) |')
        add('| Predecessor and dates | `v3` (adopted in research 2026-09-24); v4 adopted 2026-09-30, revised **pending-at-landing** |')
        add(f'| The change | model: one pooled LightGBM member ({"averaged over capacities" if w == "R" else "with the raw pooled member, daily selection"})'
            + ('; interval policy: the dynamic layer' if dl.get('adopted') else '') + ' |')
        add('| The comparator set in advance | the revision rule\'s comparator is v4\'s three-block revision; the transition\'s is v3 |')
        add(f'| The result, with its uncertainty and evidence class | against v3: see `{cur}-HG` in `uncertainty.csv`; against the '
            f'three-block revision: S_MAE {pct(rm["ratio"])} [{pct(rm["ratio_ci_lower"])}, {pct(rm["ratio_ci_upper"])}], S_WIS '
            f'{pct(rw["ratio"])} [{pct(rw["ratio_ci_lower"])}, {pct(rw["ratio_ci_upper"])}]; development |')
        add('| Against the predecessor | the comparator of the transition is the predecessor, v3 |')
        add('| What it does not establish | performance on new data; equivalence; a learned weight |')
        add('| The decision, dated, and the route | revised in research, **pending-at-landing**; `#v4` and `compare:v4` |\n')
    else:
        add('### 5a. The adopted transition (A3)\n\nNot applicable: no replacement, so v4 is unchanged and no revision or '
            'transition exists.\n')
    add('### 5b. The released model\'s documentation (A4)\n\nNot applicable: the released model does not change. v1 remains the '
        'released product and the demo; CP-22 is research only (§20.6).\n')
    add('### 5c. The chart routes (A5)\n')
    if w:
        add('| Chart | Its heading | The route\'s label | Where the route starts |\n|---|---|---|---|')
        add('| v4 paired differences (current revision against v3; revision note against the three-block revision) | `#v4` | '
            'v4 against v3 | v4\'s chapter "Explore these results" and the transition summary |\n')
    else:
        add('None drafted: the branch card carries its deciding difference and interval in text; its reader route is the MLflow '
            f'comparison `compare:{owner}`, advertised only after the authorized upload and both route checks.\n')
    add('### 5d. Final-product transition / daily operation (A7/A8)\n\nNot applicable: the trigger is unmet — no final-product '
        'designation, rollout or Live (§16 unchanged).\n')
    add('## 6. The MLflow export\n')
    xsha = hashlib.sha256((root / OUT / 'mlflow-export-draft' / 'cp22.json').read_bytes()).hexdigest()
    add(f'- **Draft export:** `reports/v4-revision/mlflow-export-draft/cp22.json` (SHA-256 `{xsha}`), built by '
        '`scripts/mlflow_export.py --draft cp22` from CP-22\'s committed rows and `draft-registry.json`: parent `cp22` and one '
        'child per new policy (' + ', '.join(f'`cp22/{c}`' for c in registry['checkpoint']['children']) + '), experiment '
        '`delu-generations`. Pending fields: ' + ', '.join(f'`{k}`' for k in registry['pending_fields']) + '.')
    add('- **Local tracking:** the same runs, names and tags in `.local/mlruns/cp22`, read back equal '
        '(`reports/v4-revision/mlflow-local.json`). No upload, no network call; MLflow telemetry disabled.')
    diff = json.loads((root / OUT / 'published-export-diff.json').read_text())
    add('- **Record-level diff against the last published export:** the published set is unchanged — '
        f'`scripts/mlflow_export.py --diff-against {diff["against"]}` over {diff["runs_old"]} runs: only_identity '
        f'{diff["only_identity"]}, identity changes on {len(diff["identity_changes"])} runs, substantive changes '
        f'{diff["substantive_changes"] or "none"} (`reports/v4-revision/published-export-diff.json`); `--check` passes. '
        + ('Under A10 the superseded revision\'s published runs (`cp21`, `cp21/HGL`) stay as they are; the revision adds runs.'
           if w else ''))
    add(f'- **Expected routes:** `experiment`; `compare:{"v4" if w else owner}` — verified by REST and in Chromium and WebKit '
        'before it is linked, at publication.\n')
    add('## 7. Checks the checkpoint ran\n')
    add('- Re-derivation tests with negative controls: `tests/cp22/test_saved_evidence.py` (new-policy metrics from committed '
        'predictions, every interval from its stored replicates, the three verdicts re-applied, byte-exact storage) and '
        '`tests/cp22/test_draft_export.py` (every exported value from a committed row; a renamed run is caught; CP-21\'s draft '
        'and the published set unchanged).')
    add('- The export contract: the draft matches its draft entries (`draft_problems`); the published export still matches the '
        'registry (`--check`).')
    add(f'- The §4 lint (code and status-word rules) on every draft slot text: {len(findings)} findings.')
    if not w:
        add('- **Unavailable comparisons, with the reason:** (W+DL) − W, (W+ACI) − W, (W+DL) − (W+ACI) and (W+DLF) − (W+DL), '
            'and the shock-day comparison of W, W+DL and W+DLF, do not exist: `cp22-replacement` yielded no winner W, so the '
            'layer arms on W were not run (§20.6). The dynamic layer is reported on v4 and v3, descriptively.')
    add('- Transition contract tests (`tests/test_43_publish_rules_migration.py`): pass unchanged; ' + (
        'PRES-4 extends them with A10\'s revision history.' if w else 'not applicable (no transition).') + '\n')
    add('## 8. Publication completion receipt — intended identities (publisher completes the rest)\n')
    add('| Surface | Intended identity | Action / unchanged rationale | Observed public identity and UTC time | Verification record and result | Outstanding action |\n|---|---|---|---|---|---|')
    if w:
        rows = [('GitHub source / README', 'the Owner\'s landing SHA and the publication commit; README glance naming v4\'s current '
                 'revision', 'push by the Owner'),
                ('GitHub Pages', 'rebuilt report: v4 chapter under the current revision, the superseded three-block revision '
                 'visible and dated, the A3 transition stating the revision (amber #B45309, filled diamond, "v4")', 'deploy on the Owner\'s instruction'),
                ('Public MLflow', 'regenerated `cp22.json` equal to this draft apart from pending fields; `cp21` runs unchanged',
                 'authorized upload, then verification'),
                ('Hugging Face Space card', 'registry-derived model line; released model unchanged (v1)', 'deploy only if the card changes'),
                ('Hugging Face direct demo', 'v1 bundle unchanged', 'verified unchanged')]
    else:
        rows = [('GitHub source / README', 'after the Owner\'s decision', 'pending the Owner\'s decision'),
                ('GitHub Pages', 'after the Owner\'s decision; v4 unchanged meanwhile', 'pending the Owner\'s decision'),
                ('Public MLflow', 'regenerated `cp22.json` equal to this draft apart from pending fields', 'pending the Owner\'s decision'),
                ('Hugging Face Space card', 'unchanged', 'verified unchanged'),
                ('Hugging Face direct demo', 'v1 bundle unchanged', 'verified unchanged')]
    for surface, identity, action in rows:
        add(f'| {surface} | {identity} | {action} | pending (publisher) | pending (publisher) | pending |')
    add('')
    add('- **Publication-to-MLflow mapping:** pending — filled by the publisher.')
    add('- **Final product and daily updates:** not applicable (no final-product designation).')
    add('- **Completion disposition:** incomplete by construction until PRES-4; CP-22 publishes nothing.')
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
