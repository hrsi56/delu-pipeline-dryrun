"""CP-21 report (reports/block-challenger/report.md), generated from committed rows only.

Run under the monitor as ``python -m cp21.report``. Every number is read from a committed table;
the text around it states the pre-registered rules. Full precision stays in the CSV files.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path('reports/block-challenger')
POLICY_NAMES = {'B0': 'similar-day naive (normalizer)', 'B1': 'v1 development replay', 'B2': 'daily LEAR',
                'B3': 'daily LightGBM', 'A1': 'normalized LEAR', 'H0': 'v2', 'HG': 'v3 (comparator)',
                'HGL': 'HGL: v3 + block LightGBM member (candidate)', 'L-P': 'L-P: pooled LightGBM (study arm)',
                'L-R': 'L-R: three-block LightGBM (study arm)', 'L-N': 'L-N: normalized three-block LightGBM (study arm)'}
ORDER = ('HGL', 'HG', 'L-P', 'L-R', 'L-N', 'H0', 'A1', 'B3', 'B2', 'B1', 'B0')
FOLDS = ('fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5')


def f(x, digits=4):
    return f'{x:.{digits}f}'


def ci(row, lo='ci_lower', hi='ci_upper', digits=4):
    return f'[{row[lo]:.{digits}f}, {row[hi]:.{digits}f}]'


def pct(x):
    return f'{100 * x:+.1f}%'


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(out)


def build(root: Path) -> str:
    out = root / OUT
    load = lambda name: json.loads((out / name).read_text())  # noqa: E731
    metrics = pd.read_csv(out / 'metrics.csv')
    unc = pd.read_csv(out / 'uncertainty.csv')
    crit = pd.read_csv(out / 'criteria.csv')
    diag = pd.read_csv(out / 'diagnostics.csv', low_memory=False)
    adoption, protocol, lineage = load('adoption.json'), load('protocol.json'), load('lineage.json')
    controls, parity, daily = load('controls.json'), load('hg-parity.json'), load('daily-cycle.json')
    cost, resources = load('fit-cost.json'), load('resources.json')
    fallback = pd.read_csv(out / 'fallback.csv')
    failures = pd.read_csv(out / 'failures.csv')
    summary = lineage['research_summary']
    equal = metrics.loc[metrics.scope.eq('equal_fold')].set_index('policy')
    pooled = metrics.loc[metrics.scope.eq('pooled')].set_index('policy')
    per_fold = metrics.loc[metrics.scope.eq('per_fold')].set_index(['policy', 'fold'])
    eq = unc.loc[unc.scope.eq('equal_fold')].set_index(['candidate', 'baseline', 'metric'])
    folds = unc.loc[unc.scope.isin(FOLDS)].set_index(['scope', 'candidate', 'baseline', 'metric'])
    verdict = adoption['verdict']
    first = adoption['first_unmet_condition']
    split = summary['block_split']
    lines = []
    add = lines.append
    add('# CP-21 — three-block LightGBM on top of v3 (capstone v21-r6 §17)\n')
    add('**Evidence class: `development_post_selection`.** Every result is a development comparison on CP-20\'s '
        '10,747 hours after earlier selection on the same five folds; nothing here is a test on new data. '
        'Nothing dated after 2026-04-07 was read, scored or used.\n')
    add('## Status at a glance\n')
    add(table(['Status', 'Value'], [
        ['Research verdict (rule `cp21-adoption`, §17.6, set 2026-09-29)',
         f'**{verdict}**' + ('' if verdict == 'v4' else f' — first unmet condition: **{first}** ({adoption["rule"]["conditions"][str(first)]})')],
        ['Block-split finding (L-R − L-P, §17.5 reading)', f'**{split["reading"]}**'],
        ['Engineering status', 'bound to the fresh Integration verdict at `docs/track-b/evidence/cp-21/integration.md` '
                               '(not part of this candidate); an INCOMPLETE or BLOCKED return yields no adoption decision'],
        ['Product status', 'unchanged: v1 remains the released product and demo; no promotion, freeze, Live, '
                           'final-product designation or economic claim follows from this research result'],
        ['Original §8 screen (diagnostic)', ', '.join(f'{p}: {s}' for p, s in summary['original_section8_status'].items())],
    ]))
    add('')
    add('## The question and the ladder\n')
    add('HG (v3) blends two linear LEAR forecasts. CP-21 asks whether adding a three-block LightGBM member, built '
        'with exactly HG\'s information (CP-15\'s 23 LightGBM features plus the three frozen GFS columns and their '
        'missing indicators), improves on HG jointly in point and interval accuracy. HGL\'s central forecast is '
        '`A1_w/3 + B2_w/3 + L-N/6 + L-R/6` with HG\'s own components bit for bit; HG\'s hour-aware interval layer is '
        're-estimated on HGL\'s own errors. The weights were fixed before any result and never tuned.\n')
    add(table(['Step', 'What it adds', 'Contrast', 'Reading'], [
        ['B3 → L-P', 'Weather — **bundled with training-only capacity selection and the §15.3 missing-input rule** '
                     '(B3 is a saved CP-15 reference with fixed CP-2 hyperparameters), so not an isolated weather effect',
         'L-P − B3', summary['contrasts']['L-P-B3']['reading']],
        ['L-P → L-R', 'The block split (controlled: same rows, target, features, grid, selection rule and H recipe)',
         'L-R − L-P', split['reading']],
        ['L-R → HGL', 'The blend into v3', 'HGL − HG', summary['contrasts']['HGL-HG']['reading']],
    ]))
    add('')
    add('## Primary contrast and the adoption rule (HGL − HG)\n')
    rows = []
    for m in ('MAE', 'WIS'):
        r = eq.loc[('HGL', 'HG', m)]
        rows.append([f'ΔS_{m}', f(r.difference), ci(r), pct(r.ratio), f'[{pct(r.ratio_ci_lower)}, {pct(r.ratio_ci_upper)}]'])
    add(table(['Score', 'Paired difference', '95% interval', 'Change as a share of HG', '95% interval of the share'], rows))
    add('')
    add(f'Bootstrap: seed {summary["bootstrap"]["seed"]}, {summary["bootstrap"]["replicates"]:,} replicates of '
        f'{summary["bootstrap"]["block_days"]}-calendar-day blocks within each fold, one shared index set (the same as CP-20\'s, '
        f'SHA-256 `{summary["bootstrap"]["index_sha256"][:16]}…`); share intervals are percentiles of '
        '`S_HGL,b / S_HG,b − 1` over the same stored replicates.\n')
    cond_rows = []
    for k in ('1', '2', '3', '4'):
        c = adoption['conditions'][k]
        result = '**met**' if c['met'] else '**not met**'
        if k == '3':
            result += (' — every key issued with finite, ordered quantiles; its Engineering-PASS half is the fresh '
                       'Integration verdict on this exact candidate')
        cond_rows.append([k, adoption['rule']['conditions'][k], result])
    add(table(['#', 'Condition (verbatim rule text in protocol.json)', 'Result'], cond_rows))
    add('')
    add('Per-fold paired daily-loss differences, HGL − HG (condition 4: a fold is decisively worse only if its lower '
        'endpoint is above zero):\n')
    add(table(['Fold', 'ΔMAE (EUR/MWh)', '95% interval', 'ΔWIS (EUR/MWh)', '95% interval'],
              [[fo + (' (stress)' if fo == 'fold_3' else ''), f(folds.loc[(fo, 'HGL', 'HG', 'MAE')].difference, 3),
                ci(folds.loc[(fo, 'HGL', 'HG', 'MAE')], digits=3), f(folds.loc[(fo, 'HGL', 'HG', 'WIS')].difference, 3),
                ci(folds.loc[(fo, 'HGL', 'HG', 'WIS')], digits=3)] for fo in FOLDS]))
    add('')
    add('## Secondary contrasts (descriptive)\n')
    rows = []
    for key, role in (('L-R-L-P', 'block split'), ('L-P-B3', 'weather, bundled with capacity selection'),
                      ('L-N-L-R', 'target representation'), ('L-P-HG', 'standalone vs v3'),
                      ('L-R-HG', 'standalone vs v3'), ('L-N-HG', 'standalone vs v3')):
        cand, base = {'L-R-L-P': ('L-R', 'L-P'), 'L-P-B3': ('L-P', 'B3'), 'L-N-L-R': ('L-N', 'L-R'), 'L-P-HG': ('L-P', 'HG'),
                      'L-R-HG': ('L-R', 'HG'), 'L-N-HG': ('L-N', 'HG')}[key]
        mae, wis = eq.loc[(cand, base, 'MAE')], eq.loc[(cand, base, 'WIS')]
        single = [f'{m} interval above zero (worse)' if r.ci_lower > 0 else f'{m} interval below zero (better)'
                  for m, r in (('S_MAE', mae), ('S_WIS', wis)) if r.ci_lower > 0 or r.ci_upper < 0]
        rows.append([f'{cand} − {base}', role, f'{f(mae.difference)} {ci(mae)}', f'{f(wis.difference)} {ci(wis)}',
                     f'{pct(mae.ratio)} / {pct(wis.ratio)}', summary['contrasts'][f'{cand}-{base}']['reading'],
                     '; '.join(single) or 'both span zero'])
    add(table(['Contrast', 'Shows', 'ΔS_MAE [95%]', 'ΔS_WIS [95%]', 'Share of comparator (MAE / WIS)', 'Joint reading',
               'Single-metric intervals'], rows))
    add('\nA mixed result is reported as no demonstrated joint preference, never as equivalence; where one metric\'s '
        'interval excludes zero it is stated in the last column.\n')
    add('')
    add('## Scores, all eleven policies\n')
    add(table(['Policy', 'S_MAE', 'S_WIS', 'Pooled MAE (EUR/MWh)', 'Pooled WIS (EUR/MWh)', 'Pooled 95% coverage'],
              [[POLICY_NAMES[p], f(equal.loc[p, 'S_MAE']), f(equal.loc[p, 'S_WIS']), f(pooled.loc[p, 'MAE'], 2),
                f(pooled.loc[p, 'WIS'], 2), f(pooled.loc[p, 'coverage95'], 4)] for p in ORDER]))
    add('\nS scores are equal-fold ratios to B0 (lower is better); pooled scores are secondary descriptions.\n')
    add('### Per-fold MAE (EUR/MWh); fold 3 is the stress period (2022-07-01..09-28, 2,112 hours over 88 days)\n')
    add(table(['Policy', *[fo + (' (stress)' if fo == 'fold_3' else '') for fo in FOLDS]],
              [[p, *[f(per_fold.loc[(p, fo), 'MAE'], 2) for fo in FOLDS]] for p in ORDER]))
    add('')
    peak = diag.loc[diag.scope.eq('peak')].set_index('policy')
    add('### The 17-day peak, 2022-08-15..31 (408 hours) — descriptive only, small effective sample\n')
    add(f'On the peak, HGL\'s MAE is {f(peak.loc["HGL", "MAE"], 2)} EUR/MWh against v3\'s {f(peak.loc["HG", "MAE"], 2)} '
        '— ' + ('higher' if peak.loc['HGL', 'MAE'] > peak.loc['HG', 'MAE'] else 'lower') + '; with 17 days no inference '
        'is drawn (the fold-3 paired interval, which contains the peak, is in the primary section).\n')
    add(table(['Policy', 'MAE', 'WIS', '95% coverage', 'Hits / hours'],
              [[p, f(peak.loc[p, 'MAE'], 2), f(peak.loc[p, 'WIS'], 2), f(peak.loc[p, 'coverage95'], 4),
                f'{int(peak.loc[p, "hit_count95"])} / {int(peak.loc[p, "n_hours"])}'] for p in ORDER]))
    add('')
    add('## Coverage with width (per fold), central versus emitted MAE\n')
    rows = []
    for p in ('HGL', 'HG', 'L-P', 'L-R', 'L-N'):
        for fo in FOLDS:
            r = per_fold.loc[(p, fo)]
            rows.append([p, fo, f(r.coverage50, 3), f(r.coverage80, 3), f(r.coverage95, 3),
                         f'{r.mean_width95:.1f} / {r.median_width95:.1f} / {r.p95_width95:.1f}',
                         f'{int(r.lower_miss_count95)} / {int(r.upper_miss_count95)}',
                         f(r.raw_central_MAE, 2), f(r.MAE, 2), f(r.centering_effect, 3), f(r.bias, 2),
                         f(r.daily_mean_level_MAE, 2), f(r.within_day_shape_MAE, 2)])
    add(table(['Policy', 'Fold', 'Cov 50', 'Cov 80', 'Cov 95', '95% width mean / median / p95', '95% misses low / high',
               'Central MAE', 'Emitted MAE', 'Centering effect', 'Bias', 'Level MAE', 'Shape MAE'], rows))
    add('')
    add('## Blocks (per fold; §14.3 support rule: at least 56 represented dates)\n')
    blocks = diag.loc[diag.scope.eq('block') & diag.policy.isin(['HGL', 'HG', 'L-P', 'L-R', 'L-N'])]
    rows = [[r.policy, r.fold, r.group, int(r.n_hours), int(r.n_days), r.support_status, f(r.MAE, 2), f(r.WIS, 2),
             f(r.coverage95, 3)] for r in blocks.sort_values(['group', 'fold', 'policy']).itertuples()]
    add(table(['Policy', 'Fold', 'Block', 'Hours', 'Represented dates', 'Support', 'MAE', 'WIS', 'Cov 95'], rows))
    add('\nPer-hour diagnostics for every policy and fold are in `diagnostics.csv` (scope `hour`).\n')
    add('## Original §8 diagnostics (all six, saved B0–B3 comparators; diagnostic only)\n')
    rows = []
    for p in ('HGL', 'HG', 'L-P', 'L-R', 'L-N'):
        part = crit.loc[crit.policy.eq(p)]
        failed = sorted(part.loc[~part.passed, 'criterion'].unique().tolist())
        rows.append([p, summary['original_section8_status'][p], ', '.join(map(str, failed)) or 'none'])
    add(table(['Policy', 'Status', 'Criteria not met'], rows))
    add('\nEvery row with actual values and limits is in `criteria.csv`.\n')
    add('## Failures and interval-layer fallback\n')
    add(f'LightGBM fit failures: {len(failures)}. Every one of the 10,747 keys was issued by every new arm with '
        'finite, ordered quantiles; the emitted p50 is kept separate from the central forecast.\n')
    fb = fallback.groupby('arm')[['hour_cells', 'fallback_cells']].sum()
    add(table(['Arm', 'Hour cells issued', 'Pooled-fallback cells (w_h = 0)'],
              [[a, int(fb.loc[a, 'hour_cells']), int(fb.loc[a, 'fallback_cells'])] for a in fb.index]))
    add('')
    add('## Identity, parity and controls (§17.7)\n')
    add(table(['Check', 'Result'], [
        ['HG through CP-21\'s H-layer path vs accepted CP-20 vectors (10,747 keys)',
         f'bitwise equal: {parity["bitwise_equal"]}, largest absolute difference {parity["max_abs_difference"]}'],
        ['HG component cache (638 entries) vs CP-20 fingerprints; evaluation centrals vs accepted HG',
         'verified (preflight/input-verification.json)'],
        ['Frozen weather regenerated from the 2,476 retained grids', 'bitwise equal (preflight/weather-regeneration.json)'],
        ['HGL blend parity on every key (largest absolute value of c_HGL − ⅔·c_HG − ⅓·mean(L-N, L-R))', f'{controls["population"]["hgl_blend_parity_max_abs_eur_mwh"]:.2e} EUR/MWh'],
        ['Controls (delivery-day mask, non-uniform D−1 mutation, future/permuted/rearranged weather, training-only '
         'selection, pooled–block parity, DST blocks, state/cache refusals, boundary guard, thread count)',
         f'all passed: {controls["all_passed"]} ({controls["checks"]} checks; controls.json)'],
        ['Cold daily cycle refits (A1_w/B2_w vs CP-20; blocks and issued HGL vector vs main run)',
         f'all bitwise: {daily["all_bitwise_checks_passed"]} at {daily["origins"]} origins (daily-cycle.json)'],
    ]))
    add('')
    add('## Fit cost and daily retraining (diagnostic only, Owner decision D3)\n')
    add(f'Main-run LightGBM fits: {cost["fits"]:,} ({cost["fits_by_role"].get("inner", 0):,} inner, '
        f'{cost["fits_by_role"].get("final", 0):,} final) over {cost["origins"]} origins. Single-thread fit wall time by arm '
        f'(seconds): ' + ', '.join(f'{a} {v:,.0f}' for a, v in cost['wall_seconds_by_arm'].items()) +
        f'. Block (L-R) versus pooled (L-P) wall time: {cost["block_vs_pooled"]["ratio_L-R_to_L-P"]:.2f}×. '
        f'Peak worker memory {cost["peak_worker_maxrss_bytes"] / 1024**3:.2f} GiB. Details: `fit-cost.csv`, '
        '`fit-cost-by-origin.csv`, `fits.parquet`, `fit-cost.json`.\n')
    t = daily['total_cold_cycle_seconds']
    add(f'HGL\'s complete daily cycle, measured cold on the M3 with 4 workers at {daily["origins"]} origins (five per '
        f'fold): median {t["median"]:.1f} s, maximum {t["max"]:.1f} s — features, the A1_w/B2_w refits, the six block '
        'selections and fits, the H layer and issuance. v21-r5 §16 still requires any final-product candidate to retrain '
        'daily; this measurement is a diagnostic, not a selection criterion.\n')
    add('## Resources at this candidate (§17.8 caps; final totals including review are in the evidence directory)\n')
    rows = [[k, f'{v["used"]:,.0f}' if isinstance(v['used'], (int, float)) else v['used'], f'{v["cap"]:,}']
            for k, v in resources['caps_vs_use'].items()]
    add(table(['Counter', 'Used', 'Cap'], rows))
    add(f'\nMachine-hours {resources["machine_hours"]:.2f} of {resources["machine_hours_cap"]:.0f}; active hours (upper bound) '
        f'{resources["active_hours_upper_bound"]:.1f} of the 32-hour timebox; 0 bytes downloaded; 0 remote writes.\n')
    add('## What this result does not establish\n')
    add('- It is development evidence after selection on the same five folds; it is not a test on new data, and the '
        'fresh-data test 4.7T stays reserved.\n'
        '- A mixed or non-significant result is not equivalence, absence of benefit or absence of harm.\n'
        '- The B3 → L-P step bundles weather with capacity selection; no contrast isolates individual weather features.\n'
        '- The block split is tested on the raw target only (no normalized pooled arm), with one seed.\n'
        '- No economic, product, promotion or Live claim follows; v1 remains the released product.\n')
    add('## Defects and repairs\n')
    add('Every defect found during the checkpoint, its effect and its repair are in `defects-and-repairs.md`; no committed '
        'research output was invalidated and no frozen forecast-path file changed after the pre-run freeze.\n')
    add('## Files\n')
    add('`protocol.json` (frozen pre-run protocol) · `lineage.json` · `predictions.parquet` (four new arms) · '
        '`metrics.csv` · `diagnostics.csv` · `uncertainty.csv` · `replicates.parquet` · `replicate-scores.parquet` · '
        '`criteria.csv` · `adoption.json` · `fallback.csv` · `failures.csv` · `controls.json` · `hg-parity.json` · '
        '`daily-cycle.json` · `fit-cost.*` · `fits.parquet` · `resources.json` · `draft-registry.json` · '
        '`mlflow-export-draft/cp21.json` · `mlflow-local.json` · `artifact-manifest.json` · `defects-and-repairs.md` · '
        '`fit-cost.md` · `reproduce.md` · `preflight/`.\n')
    return '\n'.join(lines)


def fit_cost_report(root: Path) -> str:
    """§17.5's fit-cost and daily-retrain diagnostic, as one readable report (Owner decision D3)."""
    out = root / OUT
    cost = json.loads((out / 'fit-cost.json').read_text())
    daily = json.loads((out / 'daily-cycle.json').read_text())
    resources = json.loads((out / 'resources.json').read_text())
    table_ = pd.read_csv(out / 'fit-cost.csv')
    lines = ['# CP-21 fit cost and daily retraining — a diagnostic, not a selection criterion\n',
             '**Owner decision D3 (capstone v21-r6 §17.5).** v21-r5 §16 still requires any final-product candidate to retrain '
             'daily, or to obtain an Owner-approved exception before live admission. Measured on the Apple M3 (16 GB, CPU only), '
             'LightGBM with one thread per process and at most four processes, BLAS 1.\n',
             '## LightGBM fits by arm and model (main run, every origin)\n']
    cols = [c for c in table_.columns if c.startswith('selected_')]
    lines.append(table(['Arm', 'Model', 'Origins', 'Window rows (mean / min / max)', 'Inner fit wall s (total)',
                        'Final fit wall s (total)', 'CPU s (inner + final)', 'Median final fit s',
                        'Peak worker RSS (GiB)', 'Selected capacity (count)', 'Exact ties'],
                       [[r.arm, r.model, int(r.origins), f'{r.mean_window_rows:,.0f} / {int(r.min_window_rows):,} / {int(r.max_window_rows):,}',
                         f'{r.inner_wall_total:,.0f}', f'{r.final_wall_total:,.0f}', f'{r.inner_cpu_total + r.final_cpu_total:,.0f}',
                         f'{r.median_model_wall:.2f}', f'{r.peak_worker_maxrss_bytes / 1024**3:.2f}',
                         ', '.join(f'{c[len("selected_"):]}: {int(getattr(r, c))}' for c in cols), int(r.ties)]
                        for r in table_.itertuples()]))
    lines.append('')
    lines.append(f'Fits: {cost["fits"]:,} ({cost["fits_by_role"].get("inner", 0):,} inner selection fits on each window minus its '
                 f'last 28 days, {cost["fits_by_role"].get("final", 0):,} final refits). Block against pooled: the three L-R block '
                 f'models took {cost["block_vs_pooled"]["ratio_L-R_to_L-P"]:.2f}× the single L-P pooled model\'s wall time '
                 f'({cost["block_vs_pooled"]["blocks_L-R_wall_total"]:,.0f} s against {cost["block_vs_pooled"]["pooled_L-P_wall_total"]:,.0f} s). '
                 'Per origin and model: `fit-cost-by-origin.csv`; per fit (rows, capacity, validation MAE, wall and CPU seconds, '
                 'worker memory, tree hash): `fits.parquet`.\n')
    comp = {k: v for k, v in resources['tracked'].items() if k.startswith('component_attempts_')}
    lines.append('## HG component regeneration\n')
    lines.append('HG\'s A1_w and B2_w were reused from the verified CP-20 cache in the main run (no regeneration). They were '
                 'refitted only as checks: ' + ', '.join(f'{k[len("component_attempts_"):]} {v}' for k, v in sorted(comp.items()))
                 + ' component-days, each equal to the CP-20 cache bit for bit (`daily-cycle.json`, `controls.json`).\n')
    t = daily['total_cold_cycle_seconds']
    lines.append('## HGL\'s complete daily cycle, measured cold\n')
    lines.append(f'{daily["origins"]} origins, five per fold ({daily["selection_rule"]}). Each cycle starts cold: the parent loads '
                 'its data, then a fresh pool of four processes loads its own data and refits A1_w, B2_w and the six block '
                 'models (L-R and L-N, with capacity selection); the parent blends HGL, loads the persisted interval-layer state, '
                 f'releases, predicts and issues. **Median {t["median"]:.1f} s, maximum {t["max"]:.1f} s** (minimum {t["min"]:.1f} s).\n')
    lines.append(table(['Fold', 'Day', 'Total s', 'Components s (A1 / B2)', 'Slowest block s', 'H layer + issue s', 'Bitwise checks'],
                       [[r['fold'], r['day'], f'{r["seconds"]["total_cold_cycle"]:.1f}',
                         f'{r["seconds"]["components"]["A1"]:.1f} / {r["seconds"]["components"]["B2"]:.1f}',
                         f'{max(r["seconds"]["blocks"].values()):.1f}', f'{r["seconds"]["h_layer_and_issuance"]:.2f}',
                         'pass' if all(r['components_bitwise_vs_cp20'].values()) and all(r['blocks_bitwise_vs_main_run'].values())
                         and r['hgl_vector_bitwise_vs_committed'] else 'FAIL'] for r in daily['records']]))
    lines.append('\nBitwise checks: the refitted A1_w/B2_w equal CP-20\'s cached components, the six block fits and their trees '
                 'equal the main run\'s, and the issued HGL central and quantile vector equal the committed prediction.\n')
    return '\n'.join(lines) + '\n'


def main() -> int:
    import sys
    root = Path.cwd()
    text, cost_text = build(root), fit_cost_report(root)
    if '--check' in sys.argv[1:]:
        same = {'report.md': (root / OUT / 'report.md').read_text() == text,
                'fit-cost.md': (root / OUT / 'fit-cost.md').read_text() == cost_text}
        print(json.dumps({'identical_to_committed': same}), flush=True)
        return 0 if all(same.values()) else 11
    (root / OUT / 'report.md').write_text(text)
    (root / OUT / 'fit-cost.md').write_text(cost_text)
    print('wrote reports/block-challenger/report.md and fit-cost.md', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
