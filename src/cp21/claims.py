"""CP-21's claim map and publication packet (capstone v21-r6 §17.9; packet template sections 1-8).

Run under the monitor as ``python -m cp21.claims`` after scoring, the draft registry and the draft
export. Every value and every row reference (`L<n>`, the physical line; `L1` is a CSV's header) is
read from the committed files at generation time, so neither document can drift from the evidence.
Draft slot texts carry registry tokens (`{g:<id>.<field>}`) for names and statuses and evidence
tokens (`{r:<record>}`) for numbers; they are checked with the publication lint's code and
status-word rules. Identities that exist only at landing stay explicit pending fields.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re

from delu_forecast import publication_lint as PL

OUT = Path('reports/block-challenger')
CLAIMS = Path('docs/track-b/research-content/cp21-claims.md')
PACKET = Path('docs/track-b/evidence/cp-21/publication-packet.md')
FOLDS = ('fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5')
PENDING = 'pending-at-landing'
INTERNAL_CODES = ('HGL', 'L-P', 'L-R', 'L-N', 'HG', 'H0', 'B0', 'B1', 'B2', 'B3', 'A1', 'A2')


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


def sig(x: float, n: int = 4) -> str:
    return f'{float(x):.{n}f}'


def pct(x: float) -> str:
    return f'{100 * float(x):+.0f}%'.replace('-', '−')


def num(x: float, d: int = 4) -> str:
    return f'{float(x):.{d}f}'.replace('-', '−')


def _source_row(key: str, path: str) -> str:
    """A source-key row, linked relative to docs/track-b/research-content/."""
    if path.startswith('capstone'):
        return f'| {key} | {path} |'
    if path.startswith('docs/track-b/'):
        return f'| {key} | [{path}](../{path[len("docs/track-b/"):]}) |'
    return f'| {key} | [{path}](../../../{path}) |'


def resolved_note(r_mae: dict, r_wis: dict) -> str:
    """Single-metric intervals that exclude zero, stated so a mixed result is never hidden."""
    notes = []
    for name, r in (('point-error score (S_MAE)', r_mae), ('interval score (S_WIS)', r_wis)):
        lo, hi = float(r['ci_lower']), float(r['ci_upper'])
        if lo > 0:
            notes.append(f'the {name} difference\'s 95% interval lies wholly above zero (worse)')
        elif hi < 0:
            notes.append(f'the {name} difference\'s 95% interval lies wholly below zero (better)')
    return ('; ' + ' and '.join(notes)) if notes else '; both single-metric intervals span zero'


def lint_slot(text: str) -> list[str]:
    """The publication lint's code and status-word rules on one draft, tokens removed."""
    plain = re.sub(r'\{[gr]:[^{}]+\}', '', text)
    findings = PL.code_findings(PL.reading_path(f'<p>{plain}</p>'))
    findings += [f'status word {m.group(0)!r}' for m in PL.STATUS_WORDS.finditer(plain)]
    findings += [f'internal code {c!r}' for c in INTERNAL_CODES if re.search(rf'(?<![\w-]){re.escape(c)}(?![\w-])', plain)]
    return findings


def build(root: Path) -> tuple[str, str, dict]:
    M, U, C = Rows(root, str(OUT / 'metrics.csv')), Rows(root, str(OUT / 'uncertainty.csv')), Rows(root, str(OUT / 'criteria.csv'))
    D = Rows(root, str(OUT / 'diagnostics.csv'))
    load = lambda name: json.loads((root / OUT / name).read_text())  # noqa: E731
    adoption, registry, controls = load('adoption.json'), load('draft-registry.json'), load('controls.json')
    daily, cost, parity, protocol = load('daily-cycle.json'), load('fit-cost.json'), load('hg-parity.json'), load('protocol.json')
    lineage = load('lineage.json')
    summary = lineage['research_summary']
    adopted = adoption['verdict'] == 'v4'
    owner = registry['checkpoint']['owner']
    first = adoption['first_unmet_condition']

    def eq(c, b, m):
        return U.find(scope='equal_fold', candidate=c, baseline=b, metric=m)

    def fold(fo, c, b, m):
        return U.find(scope=fo, candidate=c, baseline=b, metric=m)

    def score(p):
        return M.find(policy=p, scope='equal_fold')

    def pf(p, fo):
        return M.find(policy=p, scope='per_fold', fold=fo)

    (lm, rm), (lw, rw) = eq('HGL', 'HG', 'MAE'), eq('HGL', 'HG', 'WIS')
    (sm, rsm), (sw, rsw) = eq('L-R', 'L-P', 'MAE'), eq('L-R', 'L-P', 'WIS')
    claims = []

    def claim(ident, text, source, status):
        claims.append(f'| {ident} | {text} | {source} | {status} |')

    # ---- identity and status
    claim('C100', 'CP-21 Integration verdict and the reviewed final candidate are recorded in I21 and the CP-21 checkpoint '
                  'return; the evidence tag and landing record are pending at landing.', 'I21; RET21', 'S (pending)')
    claim('C101', f'Rule `cp21-adoption` was set on {adoption["rule"]["set_on"]} (the Owner\'s ratification) and frozen, '
                  'verbatim, in the pre-run protocol before any fit or score.', 'P21 `adoption_rule_verbatim`; CAP §17.6', 'S')
    claim('C102', 'No product change, promotion, freeze, Live, final-product designation or economic claim follows; v1 '
                  'remains the released product and demo.', 'CAP §17.1, §17.6; A21 `status`', 'S')
    # ---- population and design
    l2 = pf('B0', 'fold_3')[0]
    claim('C103', 'The same 10,747 eligible hours in five folds (2,160 / 2,159 / 2,112 / 2,160 / 2,156), 448 represented '
                  'days; fold 3 (the stress period) has 2,112 hours on 88 days; the peak 2022-08-15..31 has 408 hours.',
          f'M21 L{pf("B0", "fold_1")[0]}–L{pf("B0", "fold_5")[0]} (`n_hours`); M21 L{l2}; D21 peak rows', 'S')
    claim('C104', 'HGL\'s central forecast is `A1_w/3 + B2_w/3 + L-N/6 + L-R/6`: v3\'s two LEAR components (bit for bit) with '
                  'weight 2/3 in total and the mean of the raw and normalized three-block LightGBM forecasts with weight 1/3; '
                  'v3\'s hour-aware interval layer is re-estimated on HGL\'s own errors. The weights were fixed before any '
                  'result and never tuned.', 'CAP §17.2; P21 `blend`, `h_layer`; CT21 `hgl_blend_parity_max_abs_eur_mwh`', 'Def/S')
    claim('C105', 'Every LightGBM arm has exactly v3\'s information: CP-15\'s 23 LightGBM features plus the three frozen GFS '
                  'columns and their missing indicators; blocks night 22–05, solar 10–16, shoulder/peak 06–09 and 17–21 '
                  '(Europe/Berlin); four capacity configurations (150/15, 300/31, 600/31, 600/63 trees/leaves) selected at '
                  'every origin on the window\'s last 28 days by MAE in EUR/MWh, ties to the smaller, then refitted.',
          'CAP §17.3; P21 `features`, `capacity_grid`, `selection`', 'Def')
    claim('C106', 'The ladder B3 → pooled LightGBM → three-block LightGBM → HGL adds, in turn, weather (bundled with '
                  'training-only capacity selection and the missing-input rule, so it is not an isolated weather effect), the '
                  'block split (controlled: same rows, target, features, grid, selection rule and interval recipe), and the '
                  'blend into v3.', 'CAP §17.1; CT21 `pooled_block_rows_equal`', 'Def')
    # ---- primary contrast and verdict
    claim('C107', f'ΔS_MAE (HGL − v3) = {num(rm["difference"])}, 95% interval [{num(rm["ci_lower"])}, {num(rm["ci_upper"])}]; '
                  f'as a share of v3\'s score {pct(rm["ratio"])} [{pct(rm["ratio_ci_lower"])}, {pct(rm["ratio_ci_upper"])}].',
          f'U21 L{lm}', 'S')
    claim('C108', f'ΔS_WIS (HGL − v3) = {num(rw["difference"])}, 95% interval [{num(rw["ci_lower"])}, {num(rw["ci_upper"])}]; '
                  f'as a share of v3\'s score {pct(rw["ratio"])} [{pct(rw["ratio_ci_lower"])}, {pct(rw["ratio_ci_upper"])}].',
          f'U21 L{lw}', 'S')
    conds = '; '.join(f'condition {k} {"met" if adoption["conditions"][k]["met"] else "not met"}' for k in ('1', '2', '3', '4'))
    claim('C109', f'The mechanical verdict of rule `cp21-adoption`: **{adoption["verdict"]}**'
                  + ('' if adopted else f' — the first unmet condition is condition {first}') + f' ({conds}; condition 3\'s '
                  'Engineering PASS is the fresh Integration verdict).', 'A21 `verdict`, `conditions`; C21 HGL rows; U21 HGL rows',
          'S')
    fold_lines = ', '.join(f'L{fold(fo, "HGL", "HG", m)[0]}' for fo in FOLDS for m in ('MAE', 'WIS'))
    worse = adoption['conditions']['4']['values']['folds_decisively_worse']
    claim('C110', 'Per fold (EUR/MWh, paired mean daily loss difference, HGL − v3): ' + '; '.join(
        f'{fo} MAE {num(fold(fo, "HGL", "HG", "MAE")[1]["difference"], 2)} [{num(fold(fo, "HGL", "HG", "MAE")[1]["ci_lower"], 2)}, '
        f'{num(fold(fo, "HGL", "HG", "MAE")[1]["ci_upper"], 2)}], WIS {num(fold(fo, "HGL", "HG", "WIS")[1]["difference"], 2)} '
        f'[{num(fold(fo, "HGL", "HG", "WIS")[1]["ci_lower"], 2)}, {num(fold(fo, "HGL", "HG", "WIS")[1]["ci_upper"], 2)}]'
        for fo in FOLDS)
        + f'. Folds decisively worse (lower endpoint above zero): {len(worse)}.', f'U21 {fold_lines}', 'S-d')
    # ---- block split
    claim('C111', f'Block split (three-block minus pooled LightGBM): ΔS_MAE {num(rsm["difference"])} [{num(rsm["ci_lower"])}, '
                  f'{num(rsm["ci_upper"])}], ΔS_WIS {num(rsw["difference"])} [{num(rsw["ci_lower"])}, {num(rsw["ci_upper"])}]: '
                  f'**{summary["block_split"]["reading"]}** under §17.5\'s endpoint reading (a development finding, not an '
                  'adoption criterion).', f'U21 L{sm}, L{sw}; A21 `block_split`', 'S')
    for ident, (c, b, what) in zip(('C112', 'C113', 'C114', 'C115', 'C116'),
                                   (('L-P', 'B3', 'weather bundled with capacity selection'), ('L-N', 'L-R', 'target representation'),
                                    ('L-P', 'HG', 'pooled LightGBM against v3'), ('L-R', 'HG', 'three-block LightGBM against v3'),
                                    ('L-N', 'HG', 'normalized three-block LightGBM against v3'))):
        (l1, r1), (l2_, r2) = eq(c, b, 'MAE'), eq(c, b, 'WIS')
        claim(ident, f'{what}: ΔS_MAE {num(r1["difference"])} [{num(r1["ci_lower"])}, {num(r1["ci_upper"])}], ΔS_WIS '
                     f'{num(r2["difference"])} [{num(r2["ci_lower"])}, {num(r2["ci_upper"])}]: '
                     f'{summary["contrasts"][f"{c}-{b}"]["reading"]}{resolved_note(r1, r2)} (descriptive).',
              f'U21 L{l1}, L{l2_}', 'S-d')
    # ---- scores and diagnostics
    parts = ', '.join(f'{p} {sig(score(p)[1]["S_MAE"])}/{sig(score(p)[1]["S_WIS"])}' for p in ('HGL', 'HG', 'L-P', 'L-R', 'L-N', 'B3'))
    claim('C117', f'Equal-fold S_MAE/S_WIS: {parts}; all eleven policies in M21.',
          f'M21 L{score("HGL")[0]}–L{score("L-N")[0]}, L{score("HG")[0]}, L{score("B3")[0]}', 'S')
    f3 = pf('HGL', 'fold_3')
    ordinary = [float(pf('HGL', fo)[1]['MAE']) for fo in ('fold_1', 'fold_2', 'fold_4', 'fold_5')]
    claim('C118', f'HGL per-fold MAE: ordinary folds {num(min(ordinary), 1)}–{num(max(ordinary), 1)} EUR/MWh; stress fold 3 '
                  f'{num(f3[1]["MAE"], 1)} EUR/MWh (v3: {num(pf("HG", "fold_3")[1]["MAE"], 1)}).',
          f'M21 L{pf("HGL", "fold_1")[0]}–L{pf("HGL", "fold_5")[0]}, L{pf("HG", "fold_3")[0]}', 'S')
    claim('C119', 'Original §8 screen (diagnostic): ' + ', '.join(f'{p} {s}' for p, s in summary['original_section8_status'].items())
          + '.', 'C21; A21 `conditions.2`', 'S')
    peak_line, peak = D.find(policy='HGL', scope='peak')
    hg_line, hg_peak = D.find(policy='HG', scope='peak')
    claim('C120', f'Peak (descriptive, 17 days, small effective sample): HGL MAE {num(peak["MAE"], 1)} and WIS '
                  f'{num(peak["WIS"], 1)} EUR/MWh, 95% coverage {num(peak["coverage95"], 3)} '
                  f'({int(float(peak["hit_count95"]))}/{int(float(peak["n_hours"]))} hours); v3 MAE {num(hg_peak["MAE"], 1)} and WIS '
                  f'{num(hg_peak["WIS"], 1)}, coverage {num(hg_peak["coverage95"], 3)}. On the peak, HGL\'s point error is '
                  + ('higher' if float(peak['MAE']) > float(hg_peak['MAE']) else 'lower') + ' than v3\'s; no inference is drawn.',
          f'D21 L{peak_line}, L{hg_line}', 'S-d')
    claim('C121', 'Coverage is reported with mean, median and 95th-percentile width and tail misses, per fold, hour and block, '
                  'with the 56-date support rule for block and hour statements; no hour or block effect is claimed.',
          'M21 per-fold rows; D21 `hour`, `block` rows', 'S-d')
    # ---- identity, controls, cost
    claim('C122', 'v3 through CP-21\'s interval-layer path reproduces the accepted CP-20 vectors bit for bit on all 10,747 '
                  'keys; the cached v3 components match their CP-20 fingerprints; the frozen weather regenerates bit for bit '
                  'from the 2,476 retained grids.', 'HP21; P21 `cache_identities`, `weather.regenerated_from_grids`', 'S')
    claim('C123', f'All §17.7 controls passed ({controls["checks"]} checks): delivery-day masking exactly 0.0; a non-uniform '
                  'D−1 price mutation moves every arm; future weather exactly 0.0; cross-date permutation and within-day '
                  'rearrangement of weather move every LightGBM arm (non-monotone, so not absorbed by the trees); training-only '
                  'selection; pooled–block row parity; DST blocks; state and cache refusals; the 2026-04-07 guard.',
          'CT21 `all_passed`, `origins`, `state`, `population`', 'S')
    claim('C124', f'Main-run LightGBM fits: {cost["fits"]:,}; block against pooled wall time '
                  f'{cost["block_vs_pooled"]["ratio_L-R_to_L-P"]:.2f}×; HGL\'s cold daily cycle on the M3 with four workers: '
                  f'median {daily["total_cold_cycle_seconds"]["median"]:.0f} s, maximum {daily["total_cold_cycle_seconds"]["max"]:.0f} s '
                  f'at {daily["origins"]} origins (diagnostic only, not a selection criterion).', 'FC21; DC21', 'S')
    # ---- limitations
    claim('C125', 'Development evidence after selection on the same five folds; not a test on new data; the fresh-data test '
                  '4.7T stays reserved.', 'CAP §17.1', 'S')
    claim('C126', 'No contrast isolates individual weather features, and the block split is tested on the raw target only, '
                  'with one seed.', 'CAP §17.2 (out of scope)', 'Def')
    claim('C127', 'The decision, dated at landing: ' + ('HGL becomes **v4 · three-block LightGBM added**, with predecessor v3.'
                                                        if adopted else 'the branch **Three-block LightGBM on v3** is "Not adopted", '
                                                        f'with condition {first} as the one-line reason.'),
          'A21; DR21; landing record (pending)', 'O (pending)')

    withheld = [
        ('W22', 'Calling the block split beneficial or harmful beyond its §17.5 reading, or naming a mechanism for it',
         'The finding is exactly C111\'s reading; no mechanism was tested.'),
        ('W23', 'Attributing the change from daily LightGBM to the pooled arm to weather alone',
         'That step also changes capacity selection and the missing-input rule (C106).'),
        ('W24', 'Stating or implying that HGL, a CP-21 arm or v4 runs in the demo, is the product, or is live',
         'v1 remains the released product and demo (C102).'),
        ('W25', 'Calling a mixed or non-significant contrast "equivalent", "no effect", "no benefit" or "no harm"',
         'Not established by the rule (CAP §14.4, §17.6).'),
        ('W26', 'Presenting the daily-cycle or fit-cost figures as qualification for daily operation',
         'A diagnostic only (Owner decision D3); §16 governs a final product (C124).'),
        ('W27', 'Quoting any CP-21 number without its evidence class, or as a confirmatory or significance result',
         'Development after selection; exploratory intervals (C125).'),
    ]
    src = [('R21', 'reports/block-challenger/report.md'), ('M21', 'reports/block-challenger/metrics.csv'),
           ('U21', 'reports/block-challenger/uncertainty.csv'), ('C21', 'reports/block-challenger/criteria.csv'),
           ('D21', 'reports/block-challenger/diagnostics.csv'), ('A21', 'reports/block-challenger/adoption.json'),
           ('P21', 'reports/block-challenger/protocol.json'), ('CT21', 'reports/block-challenger/controls.json'),
           ('HP21', 'reports/block-challenger/hg-parity.json'), ('DC21', 'reports/block-challenger/daily-cycle.json'),
           ('FC21', 'reports/block-challenger/fit-cost.json'), ('RP21', 'reports/block-challenger/replicates.parquet'),
           ('DR21', 'reports/block-challenger/draft-registry.json'),
           ('X21', 'reports/block-challenger/mlflow-export-draft/cp21.json'),
           ('I21', 'docs/track-b/evidence/cp-21/integration.md'), ('RET21', 'docs/track-b/evidence/cp-21/checkpoint-return.md'),
           ('CAP', 'capstone_v21.md (v21-r6), §17')]
    claims_md = '\n'.join([
        '# CP-21 research result: claim-to-evidence map',
        '',
        '**2026-09-29 · Draft for the publication block (PUBLISH_RULES 1.1 §11; capstone v21-r6 §17.9).** Each claim',
        'the publication may render is mapped to a committed file and row. The format follows the',
        '[CP-15/CP-16](cp15-cp16-claims.md) and [CP-20](cp20-claims.md) maps: `L<n>` is the physical line in the',
        'file, and for a CSV `L1` is the header. Values here are rounded from the saved values; the rendered',
        'surfaces bind to evidence records, never to these digits. This map is not yet one of the claim maps',
        'the page reads: the publication block registers it after the Owner\'s landing.',
        '',
        '**Default evidence class.** Every CP-21 result is **`development_post_selection`** and exploratory.',
        '',
        '## Source keys',
        '',
        '| Key | File |', '|---|---|',
        *[_source_row(k, p) for k, p in src],
        '',
        '## Claim map: CP-21',
        '',
        'Status codes, as in the earlier maps: **S** supported by a saved value or verdict; **S-d** supported,',
        'descriptive only; **I** an inference from design plus saved values; **Def** a definition; **O** an Owner',
        'decision.',
        '',
        '| ID | Claim | Source → table/row | Status |', '|---|---|---|---|',
        *claims,
        '',
        '## Withheld claims: do not use',
        '',
        'W1–W21 in the [CP-15/CP-16](cp15-cp16-claims.md#withheld-claims-do-not-use) and [CP-20](cp20-claims.md) maps stay in',
        'force. CP-21 adds:',
        '',
        '| ID | Withheld claim | Reason / controlling source |', '|---|---|---|',
        *[f'| {a} | {b} | {c} |' for a, b, c in withheld],
        '',
        '## Gaps: documented, not resolved',
        '',
        '| ID | Gap | Why it stays open |', '|---|---|---|',
        '| G21 | No normalized pooled arm, residual-correction model or learned blend weights | Out of scope by design (CAP §17.2); programme 4.8 owns learned weights. |',
        '| G22 | Adoption and status dates, the evidence tag and the landing record | They exist only at landing; the publication block fills them (pending fields). |',
        '',
    ]) + '\n'
    return claims_md, registry, dict(adoption=adoption, summary=summary, cost=cost, daily=daily, parity=parity,
                                     protocol=protocol, controls=controls, M=M, U=U, eq=eq, fold=fold, pf=pf, score=score)


def packet(root: Path, registry: dict, ctx: dict, claims_sha: str) -> tuple[str, list[str]]:
    adoption, summary, protocol = ctx['adoption'], ctx['summary'], ctx['protocol']
    eq, fold, pf = ctx['eq'], ctx['fold'], ctx['pf']
    adopted = adoption['verdict'] == 'v4'
    owner = registry['checkpoint']['owner']
    first = adoption['first_unmet_condition']
    (lm, rm), (lw, rw) = eq('HGL', 'HG', 'MAE'), eq('HGL', 'HG', 'WIS')
    ordinary = [(fo, pf('HGL', fo)) for fo in ('fold_1', 'fold_2', 'fold_4', 'fold_5')]
    lo = min(ordinary, key=lambda x: float(x[1][1]['MAE']))
    hi = max(ordinary, key=lambda x: float(x[1][1]['MAE']))
    stress = pf('HGL', 'fold_3')
    hgl_id = 'v4' if adopted else 'blend-with-block-lightgbm'
    reason = '' if adopted else f'condition {first} of rule cp21-adoption was not met'
    # draft slot texts: tokens for names/statuses ({g:}) and numbers ({r:}); no typed number or code
    split_words = adoption['block_split']['reading']
    if adopted:
        slots = {
            'v4.question': 'Does adding a nonlinear, block-structured forecaster to {g:v3.name} improve both of its error scores?',
            'v4.change': '{g:v4.name} adds a three-block LightGBM forecaster, with the same inputs as {g:v3.name}, to its blend '
                         'of two linear forecasts, and re-estimates the hour-aware intervals on the new forecast\'s own errors.',
            'v4.main_chart.headline': 'Against {g:v3.name}, the point-error score changed by {r:cp21.ratio.HGL-HG.MAE} and the '
                                      'interval score by {r:cp21.ratio.HGL-HG.WIS}, each as a share of the comparator\'s score, '
                                      'with 95% intervals.',
            'v4.reading': 'Both paired differences lie below zero, no test period is decisively worse, and the six screening '
                          'diagnostics hold, so the rule set in advance adopted the change in research.',
            'v4.not_established': 'Development evidence after selection, not a test on new data. It does not show that separate '
                                  f'hour-block models help: block against pooled models showed {split_words}. No single weather '
                                  'feature is isolated.',
            'v4.decision': 'Adopted in research on {g:v4.status.date}; the released model is unchanged.',
            'v4.evidence': 'the report, the review verdict, the claim map, the MLflow comparison',
            'transition.v3-v4.title': 'From v3 to v4: adding a three-block LightGBM',
            'transition.v3-v4.result': 'Against {g:v3.name}, both of its predecessor and its comparator, the point-error score '
                                       'changed by {r:cp21.ratio.HGL-HG.MAE} and the interval score by {r:cp21.ratio.HGL-HG.WIS}, '
                                       'in development tests.',
        }
    else:
        slots = {
            'branch.block-lightgbm-on-v3.question': '{g:block-lightgbm-on-v3.question}',
            'branch.block-lightgbm-on-v3.comparator': '{g:v3.name}',
            'branch.block-lightgbm-on-v3.result': 'Against {g:v3.name}, the point-error score changed by '
                                                  '{r:cp21.ratio.HGL-HG.MAE} and the interval score by '
                                                  '{r:cp21.ratio.HGL-HG.WIS}, each as a share of its score, with 95% intervals.',
            'branch.block-lightgbm-on-v3.reason': '{g:block-lightgbm-on-v3.status.reason}',
            'branch.block-lightgbm-on-v3.decision': '{g:block-lightgbm-on-v3.status.status}, {g:block-lightgbm-on-v3.status.date}.',
            'branch.block-lightgbm-on-v3.block_split': 'Separate block models against one pooled model, with the same inputs '
                                                       'and rows: point-error score difference '
                                                       '{r:cp21.uncertainty.L-R-L-P.equal_fold.MAE|ci}, interval score '
                                                       'difference {r:cp21.uncertainty.L-R-L-P.equal_fold.WIS|ci}.',
            'branch.block-lightgbm-on-v3.ladder': 'The steps were daily LightGBM, then weather with capacity selection '
                                                  '(together, not weather alone), then block models, then the blend into '
                                                  '{g:v3.name}.',
            'branch.block-lightgbm-on-v3.evidence': 'the report, the review verdict, the claim map, the MLflow comparison',
        }
    findings = []
    for key, text in slots.items():
        findings += [f'{key}: {f}' for f in lint_slot(text)]
    lines = []
    add = lines.append
    add('# CP-21 publication packet (draft)\n')
    add('**Publication Standard v1 §12; PUBLISH_RULES 1.1 §11; capstone v21-r6 §17.9; template '
        '`docs/track-b/publication-packet-template.md` (SHA-256 4efb0185…829cf).** Filled from inside CP-21. '
        'Numbers are never typed into a surface: each value below names the committed row it comes from, and the '
        'evidence layer re-derives it. Identities that exist only at landing are marked **pending-at-landing**. '
        'Publication is a separate block after the Owner\'s landing; nothing here is published.\n')
    add('Pinned rules: PUBLISH_RULES 1.1 `91eea445…7584f3`, incorporating Publication Standard v1 `01d721c2…478cc` '
        'and presentation plan revision 3 `28119374…c812c` (all verified at the pre-run freeze, protocol '
        '`issued_and_inherited_sha256`).\n')
    add('## 1. Identity\n')
    add('| Item | Value |\n|---|---|')
    add('| Checkpoint | CP-21, brief `docs/track-b/evidence/cp-21/issued-brief.md` at `813fb8a476a7b6d10ecf0a5519e97ec8efc4ff36e0f13868676ad34534fc020f` |')
    add('| Governing research plan and version | `capstone_v21.md` v21-r6 §17 (SHA-256 `ee402c47…67344`), rule `cp21-adoption` |')
    add('| Evidence tag and tip | `evidence/cp-21` at **pending-at-landing** (the evidence tip SHA is in the CP-21 return), frozen **pending-at-landing** |')
    add('| Final reviewed candidate (model code) | recorded in the CP-21 checkpoint return (`docs/track-b/evidence/cp-21/checkpoint-return.md`); a commit cannot contain its own SHA |')
    add('| Report, review verdict, landing record | `reports/block-challenger/report.md`; `docs/track-b/evidence/cp-21/integration.md`; landing record **pending-at-landing** |')
    add('')
    add('## 2. The draft registry entries (standard §5)\n')
    add(f'Machine-readable: `reports/block-challenger/draft-registry.json` (outcome **{registry["outcome"]}**, derived '
        'mechanically from `adoption.json`). Statuses are dated at landing; CP-21 registers nothing.\n')
    for e in registry['entries']:
        add(f'### `{e["id"]}` — {e["kind"]}\n')
        add('| Field | Value |\n|---|---|')
        for field in ('id', 'name', 'subtitle', 'short', 'kind'):
            add(f'| `{field}` | {e[field]} |')
        add('| `codes` | ' + '; '.join(f'{c["experiment"]}: {c["code"]} ({c["population"]})' for c in e['codes']) + ' |')
        add('| `statuses` | ' + '; '.join(f'{s["status"]}, {s["date"]}, {s["source"]}' + (f', {s["reason"]}' if s['reason'] else '')
                                        for s in e['statuses']) + ' |')
        add(f'| `comparator` | `{e["comparator"]}` (v3, named by §17.2 before results) |')
        add(f'| `population` | `{e["population"]}` |')
        add(f'| `evidence_class` | Development (`{e["evidence_class"]}`) |')
        add(f'| `plan`, `rules` | {e["plan"]}; `{", ".join(e["rules"])}` |')
        add(f'| `sources` | ' + ', '.join(f'`{s}`' for s in e['sources']) + ' |')
        add(f'| `claim_map` | `{e["claim_map"]}` |')
        add(f'| `run_keys` | ' + ', '.join(f'`{k}`' for k in e['run_keys']) + ' |')
        add(f'| `style`, `anchor` | {e["style"]}; {e["anchor"] or "none (a study arm has no anchor of its own)"}'
            + (' — v4\'s encoding: amber `#B45309`, filled diamond, direct label "v4" (Owner decision D6)' if e['id'] == 'v4' else '') + ' |')
        add(f'| `checkpoint`, `after` | CP-21; {e["after"] or "n/a"} |')
        add(f'| `question`, `informed` | {e["question"] or "n/a"}; {e["informed"] or "none recorded"} |')
        add(f'| `predecessor` | {e["predecessor"] or "n/a"} |')
        add('')
    add('Proposed code additions to existing entries (the saved references appear in CP-21 too): '
        + '; '.join(f'`{a["entry"]}` gains CP-21 {a["code"]["code"]}' for a in registry['code_additions_to_existing_entries']) + '.\n')
    add('## 3. The claim map\n')
    add(f'`docs/track-b/research-content/cp21-claims.md` (SHA-256 `{claims_sha}`): every claim with its committed rows, '
        'the block-split finding (C111), the ladder B3 → pooled → block → HGL with the bundling in the first step disclosed '
        '(C106), and the withheld claims W22–W27 (W1–W21 stay in force). Status words come only through registry tokens.\n')
    add('## 4. The derived headline quantities (standard §3.3), computed inside the checkpoint\n')
    add('| Quantity | Value | From |\n|---|---|---|')
    add(f'| (a) The pre-specified verdict | **{"met" if adopted else "not met"}**: {adoption["verdict"]}'
        + ('' if adopted else f'; first unmet condition {first}') + ' | `reports/block-challenger/adoption.json` (`verdict`, `conditions`) |')
    add('| The rule, in words, with the date it was set | ' + registry['rule']['words'] + '; set **2026-09-29** | '
        '`capstone_v21.md` v21-r6 §17.6; `protocol.json` `adoption_rule_verbatim` (committed before scoring) |')
    add(f'| Distance from the rule\'s comparator (v3), in the rule\'s unit | ΔS_WIS {num(rw["difference"])} (upper endpoint '
        f'{num(rw["ci_upper"])}, rule < 0); ΔS_MAE {num(rm["difference"])} (upper endpoint {num(rm["ci_upper"])}, rule ≤ 0) | '
        f'`uncertainty.csv` L{lw}, L{lm} |')
    add('| N: policies tested against the same rule up to this decision | 1 — HGL (the pooled and block study arms are never '
        'eligible under the rule) | `protocol.json` `adoption_rule`; §17.2 |')
    add('| "Point comparison" label needed? | no — both differences have 95% intervals | `uncertainty.csv` |')
    add(f'| (b) The change against v3, as a share of v3\'s score | S_MAE {pct(rm["ratio"])} (full precision {rm["ratio"]}); '
        f'S_WIS {pct(rw["ratio"])} (full precision {rw["ratio"]}) | `uncertainty.csv` L{lm}, L{lw} (`ratio`) |')
    add(f'| Its 95% interval | S_MAE [{pct(rm["ratio_ci_lower"])}, {pct(rm["ratio_ci_upper"])}]; S_WIS '
        f'[{pct(rw["ratio_ci_lower"])}, {pct(rw["ratio_ci_upper"])}] — from CP-21\'s own bootstrap draws: seed 15042, '
        '2,000 replicates, 7-calendar-day blocks, percentiles of `S_HGL,b / S_v3,b − 1` | `uncertainty.csv` '
        f'L{lm}, L{lw} (`ratio_ci_lower`, `ratio_ci_upper`); every draw in `replicates.parquet` |')
    add(f'| (c) Mean absolute error per period, EUR/MWh (HGL) | ordinary periods (folds 1, 2, 4, 5): {num(lo[1][1]["MAE"], 1)} '
        f'({lo[0].replace("_", " ")}) to {num(hi[1][1]["MAE"], 1)} ({hi[0].replace("_", " ")}); stress period, fold 3 '
        f'(2022-07-01..09-28): {num(stress[1]["MAE"], 1)} | `metrics.csv` per-fold rows L{pf("HGL", "fold_1")[0]}–L{pf("HGL", "fold_5")[0]} |')
    add('')
    add('## 5. Draft slot texts (standard §6)\n')
    add('Only the outcome the rule yields is drafted: ' + ('the v4 chapter and its A3 transition.' if adopted else
        'the branch card. No v4 chapter or transition exists for a not-adopted result.') + ' Tokens: `{g:<id>.<field>}` '
        'registry fields; `{r:<record>}` evidence records (rendered with the §3.4 precision rules). The §4 lint\'s code '
        f'and status-word rules, applied to every draft: **{len(findings)} findings**.\n')
    add('| Slot key | Draft |\n|---|---|')
    for key, text in slots.items():
        add(f'| `{key}` | ' + text.replace('|', '\\|') + ' |')  # a token's |ci would split the table cell
    add('')
    if adopted:
        add('### 5a. The adopted transition (A3)\n')
        add('| Field | Value |\n|---|---|')
        add('| Title | "From v3 to v4: adding a three-block LightGBM" |')
        add('| Predecessor and dates | `v3` (adopted in research 2026-09-24); v4 adoption date **pending-at-landing** |')
        add('| The change | model: a three-block LightGBM member added to the blend; the interval layer re-estimated on the new errors |')
        add('| The comparator set in advance | `v3`; it is also the predecessor, so one comparison serves both |')
        add(f'| The result, with its uncertainty and evidence class | S_MAE {pct(rm["ratio"])} [{pct(rm["ratio_ci_lower"])}, '
            f'{pct(rm["ratio_ci_upper"])}], S_WIS {pct(rw["ratio"])} [{pct(rw["ratio_ci_lower"])}, {pct(rw["ratio_ci_upper"])}]; development |')
        add('| Against the predecessor | the comparator is the predecessor |')
        add('| What it does not establish | performance on new data; that separate hour-block models help (block against '
            f'pooled: {adoption["block_split"]["reading"]}); the contribution of any single weather feature |')
        add('| The decision, dated, and the route | adopted in research, **pending-at-landing**; `#v4` and `compare:v4` |')
        add('')
    else:
        add('### 5a. The adopted transition (A3)\n\nNot applicable: HGL was not adopted, so there is no new generation and '
            'no transition. The branch card above is the whole record, attached after v3.\n')
    add('### 5b. The released model\'s documentation (A4)\n\nNot applicable: the released model does not change. v1 remains '
        'the released product and the demo; CP-21 is research only (§17.1, §17.6).\n')
    add('### 5c. The chart routes (A5)\n')
    if adopted:
        add('| Chart | Its heading | The route\'s label | Where the route starts |\n|---|---|---|---|')
        add('| v4 paired differences (both scores, 95% intervals) | `#v4` | v4 against v3 | the v4 chapter\'s "Explore these results" and the transition summary |')
        add('')
    else:
        add('None drafted: a branch card carries its deciding difference and interval in text; its reader route is the MLflow '
            'comparison `compare:block-lightgbm-on-v3`, advertised only after the authorized upload and both route checks.\n')
    add('### 5d. Final-product transition / daily operation (A7/A8)\n\nNot applicable: the trigger is unmet — no final-product '
        'designation, rollout or Live. The final-product designation does not change (§16).\n')
    add('## 6. The MLflow export\n')
    xsha = hashlib.sha256((root / OUT / 'mlflow-export-draft' / 'cp21.json').read_bytes()).hexdigest()
    add(f'- **Draft export:** `reports/block-challenger/mlflow-export-draft/cp21.json` (SHA-256 `{xsha}`), built by '
        '`scripts/mlflow_export.py --draft cp21` from CP-21\'s committed rows and `draft-registry.json`: parent `cp21` and '
        'children `cp21/HGL`, `cp21/L-P`, `cp21/L-R`, `cp21/L-N`, experiment `delu-generations`. Pending fields: '
        + ', '.join(f'`{k}`' for k in registry['pending_fields']) + '.')
    add('- **Local tracking:** the same runs, names and tags in `.local/mlruns/cp21` (read-back equal; '
        '`reports/block-challenger/mlflow-local.json`). No upload, no network call; MLflow telemetry disabled.')
    diff = json.loads((root / OUT / 'published-export-diff.json').read_text())
    add('- **Record-level diff against the last published export:** the published set is unchanged — '
        f'`scripts/mlflow_export.py --diff-against {diff["against"]}` over {diff["runs_old"]} runs: only_identity '
        f'{diff["only_identity"]}, identity changes on {len(diff["identity_changes"])} runs, substantive changes '
        f'{diff["substantive_changes"] or "none"} (`reports/block-challenger/published-export-diff.json`); '
        '`--check` passes. The publication block '
        'registers the entries, regenerates the final export and must show it equals this draft apart from the pending '
        'fields, with no previously published record changed.')
    add(f'- **Expected routes:** `experiment`; `compare:{"v4" if adopted else owner}` — verified by REST and in Chromium and '
        'WebKit before it is linked, at publication.\n')
    add('## 7. Checks the checkpoint ran\n')
    add('- Re-derivation tests with negative controls: `tests/cp21/test_saved_evidence.py` (new-arm metrics from committed '
        'predictions, every interval from its stored replicates, the §17.6 verdict re-applied, byte-exact storage) and '
        '`tests/cp21/test_draft_export.py` (every exported value from a committed row; a changed row changes the draft; a '
        'renamed run is caught).')
    add('- The export contract: the draft matches its draft entries (`draft_problems`); the published export still matches '
        'the registry (`contract_problems`, `--check`).')
    add(f'- The §4 lint (code and status-word rules) on every draft slot text: {len(findings)} findings.')
    add('- Transition contract tests (`tests/test_43_publish_rules_migration.py`): ' + ('to be extended by the publication '
        'block with the v4 transition.' if adopted else 'not applicable (no transition); they pass unchanged in the suite.') + '\n')
    add('## 8. Publication completion receipt — intended identities (publisher completes the rest)\n')
    add('| Surface | Intended identity | Action / unchanged rationale | Observed public identity and UTC time | Verification record and result | Outstanding action |\n|---|---|---|---|---|---|')
    if adopted:
        rows = [('GitHub source / README', 'the Owner\'s landing SHA and the publication commit; README glance and generation list naming v4',
                 'push by the Owner'),
                ('GitHub Pages', 'rebuilt report with the v4 chapter, A3 transition and v4 headline (amber #B45309, filled diamond, "v4")',
                 'deploy on the Owner\'s instruction'),
                ('Public MLflow', 'regenerated `cp21.json` equal to this draft apart from pending fields; parent `cp21`, four children',
                 'authorized upload, then verification'),
                ('Hugging Face Space card', 'registry-derived model line; released model unchanged (v1)', 'deploy only if the card changes'),
                ('Hugging Face direct demo', 'v1 bundle unchanged', 'verified unchanged or redeployed with the same v1 artifact')]
    else:
        rows = [('GitHub source / README', 'the Owner\'s landing SHA and the publication commit; README branch list gains '
                 '"Three-block LightGBM on v3"', 'push by the Owner'),
                ('GitHub Pages', 'rebuilt report with the branch card after v3; headline stays on v3', 'deploy on the Owner\'s instruction'),
                ('Public MLflow', 'regenerated `cp21.json` equal to this draft apart from pending fields; parent `cp21`, four children',
                 'authorized upload, then verification'),
                ('Hugging Face Space card', 'unchanged if the bundle and card are unchanged', 'verified unchanged'),
                ('Hugging Face direct demo', 'v1 bundle unchanged', 'verified unchanged')]
    for surface, identity, action in rows:
        add(f'| {surface} | {identity} | {action} | pending (publisher) | pending (publisher) | pending |')
    add('')
    add('- **Publication-to-MLflow mapping:** pending — the publication SHA, export identity and run references are filled by '
        'the publisher.')
    add('- **Final product and daily updates:** not applicable (no final-product designation).')
    add('- **Completion disposition:** incomplete by construction until the publication block; CP-21 publishes nothing.')
    add('- **Authority:** none exercised; every external action needs the Owner\'s instruction for that action.\n')
    return '\n'.join(lines) + '\n', findings


def main() -> int:
    import sys
    root = Path.cwd()
    check = '--check' in sys.argv[1:]
    claims_md, registry, ctx = build(root)
    claims_sha = hashlib.sha256(claims_md.encode()).hexdigest()
    text, findings = packet(root, registry, ctx, claims_sha)
    if check:  # regenerate and compare, writing nothing (the reviewer's clean checkout stays clean)
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
