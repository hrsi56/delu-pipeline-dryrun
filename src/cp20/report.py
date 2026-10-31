"""Render reports/weather-ablation/report.md strictly from saved CP-20 tables (no computation
of new results beyond formatting)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

OUT = Path('reports/weather-ablation')


def _table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for row in frame.itertuples(index=False):
        lines.append('| ' + ' | '.join('' if (isinstance(v, float) and pd.isna(v)) else str(v) for v in row) + ' |')
    return '\n'.join(lines)


def render(root: Path, resources: dict) -> str:
    out = Path(root) / OUT
    lineage = json.loads((out / 'lineage.json').read_text())
    s = lineage['research_summary']
    ext = json.loads((out / 'extraction-summary.json').read_text())
    metrics = pd.read_csv(out / 'metrics.csv')
    unc = pd.read_csv(out / 'uncertainty.csv')
    crit = pd.read_csv(out / 'criteria.csv')
    fallback = pd.read_csv(out / 'fallback.csv')
    controls = json.loads((out / 'causal-controls.json').read_text())['controls']
    h0 = lineage['h0_verification']
    eq = metrics.loc[metrics.scope.eq('equal_fold'), ['policy', 'S_MAE', 'S_WIS']]
    primary = unc.loc[unc.scope.eq('equal_fold'), ['candidate', 'baseline', 'metric', 'difference', 'ci_lower', 'ci_upper', 'status']]
    per_fold = unc.loc[unc.scope.ne('equal_fold'), ['scope', 'metric', 'difference', 'ci_lower', 'ci_upper', 'status']]
    conclusion = s['joint_conclusions']['HG-H0']
    wis = primary.set_index('metric').loc['WIS']
    mae = primary.set_index('metric').loc['MAE']
    fold_metrics = metrics.loc[metrics.scope.eq('per_fold') & metrics.policy.isin(['H0', 'HG', 'B2', 'B0']),
                               ['policy', 'fold', 'n', 'MAE', 'WIS', 'coverage95', 'mean_width95']] \
        if 'n' in metrics.columns else metrics.loc[metrics.scope.eq('per_fold') & metrics.policy.isin(['H0', 'HG', 'B2', 'B0'])]
    lines = [
        '# CP-20 direct-GFS paired ablation (HG − H0) — development, post-selection',
        '',
        'Engineering status: **pending fresh exact-candidate Integration review**; the terminal verdict and both SHAs are '
        'recorded in the canonical checkpoint return. Historical CP-15 product status: **NOT_DEMONSTRATED**, unchanged. '
        f'Original-§8 diagnostic status (unchanged criteria, saved B0–B3 comparators): {s["original_section8_status"]}. '
        'Product/delivery eligibility: **not authorized by this research evaluation**. No promotion, VRE experiment, '
        'CP-17–19, public surface, live selection or prospective clock follows from any result here.',
        '',
        f'**Primary conclusion (HG − H0): {conclusion}.** ΔS_WIS = {wis.difference:.10f} '
        f'[{wis.ci_lower:.10f}, {wis.ci_upper:.10f}]; ΔS_MAE = {mae.difference:.10f} [{mae.ci_lower:.10f}, {mae.ci_upper:.10f}] '
        '(95% paired percentile intervals, seed 15042, one shared 2,000-replicate 7-calendar-day block index set). '
        'Observed joint improvement requires upper CI(ΔS_WIS) < 0 AND upper CI(ΔS_MAE) ≤ 0, applied to the paired '
        'score differences, never to marginal intervals or point estimates. Otherwise the result is *no demonstrated joint '
        'preference*: not equivalence, absence of benefit or absence of harm. Both metrics, directions and magnitudes are '
        'reported without an invented trade-off weight. All intervals are exploratory post-selection '
        '(`development_post_selection`).',
        '',
        f'Descriptive order (lower S_WIS, then S_MAE, H0 on exact ties): {s["ranking"]} — not promotion.',
        '',
        '## Equal-fold B0-normalised scores (all seven scored policies)',
        '',
        _table(eq),
        '',
        '## Paired HG − H0 uncertainty',
        '',
        _table(primary),
        '',
        'Per-fold paired daily-loss intervals (descriptive only):',
        '',
        _table(per_fold),
        '',
        '## Per-fold point and interval metrics (H0, HG and references B0/B2)',
        '',
        _table(fold_metrics),
        '',
        'All CP-15 §7 / §14.3 metrics for every policy (RMSE, central vs emitted MAE, centering effect, 50/80/95% coverage '
        'and width summaries, tail misses, bias, daily level/shape, crisis recovery) are in `metrics.csv`; all 24 local '
        'hours, the exhaustive night 22–05 / solar 10–16 / shoulder 06–09 and 17–21 blocks with represented dates, hits '
        'and widths, daily losses, the 17-day August 15–31 peak (descriptive only) and September recovery slices are in '
        '`diagnostics.csv`. Hour/block comparisons with fewer than 56 represented dates are support-limited.',
        '',
        '## All six original §8 diagnostics, both arms',
        '',
        _table(crit[['policy', 'criterion', 'metric', 'scope', 'actual', 'lower_limit', 'upper_limit', 'comparator', 'status']]),
        '',
        '## H layer fallback incidence (w_h = 0 → pooled layer)',
        '',
        _table(fallback),
        '',
        '## Population, arms and causal construction',
        '',
        f'Both arms cover the original 10,747 eligible keys (2,160 / 2,159 / 2,112 / 2,160 / 2,156): 21,494 contrast rows; '
        f'with the five saved references B0/B1/B2/B3/A1, 75,229 scored rows. Fold 3 has 2,112 hours on 88 represented dates; '
        f'the peak 408 hours / 17 dates. No denominator changed and no eligible key was dropped. Both arms use the frozen '
        f'638 CP-16 fold/date origins, the same 35 training-only admission dates (frozen and committed before outer '
        f'scoring) and the same genuine warm-up starts. H0 is the no-weather V2-H replayed from identity-verified CP-15/CP-16 '
        f'component caches; it reproduces the accepted CP-16 V2-H vectors: {h0}. HG refits the identical A1/B2 recipes '
        f'with only three same-local-hour weather columns appended; selected penalties may differ because inputs differ. '
        f'Each arm consumes only its own genuinely issued errors under identical release/support/update rules.',
        '',
        'Real-data controls (`causal-controls.json`): ' + '; '.join(
            f"{c['fold']} {c['day']}: HG refit reproduction {c['hg_reproduction_max_abs']}, delivery-day/future mask "
            f"{c['delivery_day_and_future_mask']}, available D−1 price mutation {c['available_d1_price_mutation']}, future "
            f"weather {c['future_weather_mutation']}, available weather {c['available_weather_mutation']}, neutralised "
            f"weather vs H0 {c['neutralised_weather_vs_h0']}" for c in controls) + '.',
        '',
        '## Weather input (regional proxy, not a DE-LU polygon or generation forecast)',
        '',
        f'GFS 0.25° operational D−1 00 UTC, 2,476 runs, {ext["target_messages"]} target messages decoded and validated '
        f'({ext["messages_by_endpoint"]}); payload {ext["payload_bytes"] / 2**30:.2f} GiB. Missingness classes: '
        f'{ext["feature_status"]}; structural-missing delivery days: {ext["structural_missing_days"]}; classified missing runs: '
        f'{ext["classified_missing_runs"] or "none"}. Local-hour cells {ext["local_hour_cells"]}, missing '
        f'{ext["local_hour_cells_missing"]}, repeated autumn hours averaged {ext["repeated_local_hours"]}. Admission retained raw '
        f'samples byte-identical: {ext["sample_hash_identical"]}/{ext["sample_hash_comparisons"]}. The eight added '
        f'2023-03-24..31 runs passed the targeted field/lead/availability checks: {ext["added_runs_passed"]}/{ext["added_runs_checked"]}. '
        'Radiation clipping by version is in `radiation-clipping.csv`.',
        '',
        'Disclosed assumptions carried from admission: NCAR-only 2019–2020 field presence was inferred at admission (every '
        'CP-20 message was decoded and validated); availability before D−1 11:00 UTC is reconstructed from dated NCEP '
        'production-status averages with an assumed, unevidenced dissemination lag, not a per-day delivery guarantee; delivery '
        '2019-01-01 weather is structurally missing; the radiation-precision criterion was defined after admission sampling and '
        'v14/v15 DSWRF is packed at 10 (or 1) W m⁻²; AWS lacks 2021-02-02 and NCAR 2024-09-15 f039 is truncated. ICON is not '
        'admitted and was not used. Admission does not certify live use.',
        '',
        '## Economics and limits',
        '',
        'No new economic calculation, optimiser, annualisation or threshold. Historical economics may be cited only with its '
        'original population, assumptions and post-selection limits. Inherited A65 vintage/revision assumptions remain '
        'limitations. The 2022 crisis fold informs exploratory comparison and cannot become unseen evidence.',
        '',
        '## Resources',
        '',
        '```json',
        json.dumps(resources, indent=1, sort_keys=True),
        '```',
    ]
    return '\n'.join(lines) + '\n'
