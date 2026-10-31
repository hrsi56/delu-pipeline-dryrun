"""Render completed CP-15 evidence without choosing or modifying a policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

POLICIES = ('B0', 'B1', 'B2', 'B3', 'A1', 'A2', 'A3', 'A4', 'A5')
COMPONENTS = {'A3': ('A1', 'A2'), 'A5': ('A1', 'A2', 'A4')}


def table(frame):
    def cell(value):
        if isinstance(value, float):
            return f'{value:.5f}'
        return str(value).replace('|', '\\|').replace('\n', ' ')
    rows = ['| ' + ' | '.join(frame.columns) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows.extend('| ' + ' | '.join(cell(x) for x in row) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(rows)


def resources(output):
    records, lineage = [], []
    for number in range(1, 6):
        fold = f'fold_{number}'
        fits = pd.read_parquet(output / 'folds' / f'{fold}-fits.parquet')
        run = json.loads((output / 'folds' / f'{fold}-run.json').read_text())
        origins = json.loads((output / 'folds' / f'{fold}-origins.json').read_text())
        timing = re.search(r'^\s*([0-9.]+)\s+real\s+', (output / 'execution' / f'{fold}.log').read_text(), re.MULTILINE)
        if timing is None:
            raise ValueError(f'missing completed whole-command timing: {fold}')
        for policy in POLICIES:
            own = fits.loc[fits.policy.eq(policy)]
            component_names = COMPONENTS.get(policy, ())
            component = fits.loc[fits.policy.isin(component_names)]
            records.append({
                'fold': fold, 'policy': policy,
                'direct_logical_fit_calls': int(own.fit_calls.sum()),
                'direct_solver_calls': int(own.solver_calls.fillna(own.fit_calls).sum()),
                'numerical_continuation_calls': int((own.solver_calls.fillna(own.fit_calls) - own.fit_calls).sum()),
                'direct_fit_seconds': float(own.fit_seconds.sum()),
                'component_policies': '+'.join(component_names) or 'none',
                'component_logical_fit_calls': int(component.fit_calls.sum()),
                'component_fit_seconds': float(component.fit_seconds.sum()),
                'shared_fold_wall_seconds': run['runtime_seconds'],
                'shared_full_command_wall_seconds': float(timing.group(1)),
                'shared_process_peak_rss_bytes': run['max_rss_bytes'],
                'origin_count': run['origin_count'], 'cache_hit_origins': run['cache_hits'],
                'memory_scope': 'whole fold process, includes every policy; not per-estimator allocation',
                'runtime_scope': 'original fit durations include cache-reused fits; fold loop excludes initial input/features and warmup-boundary discovery; full command wall time includes setup and teardown',
            })
        for policy in ('B2', 'B3', 'A1', 'A2', 'A4'):
            for phase in ('warmup', 'evaluation'):
                selected = fits.loc[fits.policy.eq(policy) & fits.phase.eq(phase)]
                lineage.append({
                    'fold': fold, 'policy': policy, 'phase': phase,
                    'first_delivery_day': str(selected.delivery_date.min()),
                    'last_delivery_day': str(selected.delivery_date.max()),
                    'origins': selected.delivery_date.nunique(),
                    'model_records': len(selected), 'logical_fit_calls': int(selected.fit_calls.sum()),
                    'minimum_training_rows': int(selected.n_train.min()),
                    'maximum_training_rows': int(selected.n_train.max()),
                    'first_history_start': str(selected.history_start.min()),
                    'last_history_start': str(selected.history_start.max()),
                    'unique_model_hashes': selected.model_sha256.nunique(),
                })
        evaluation = [o for o in origins if o['eligible_evaluation_hours']]
        assert all(v['buffer_days'] == 28 for o in evaluation for v in o['buffer'].values())
    resource_frame, lineage_frame = pd.DataFrame(records), pd.DataFrame(lineage)
    resource_frame.to_csv(output / 'resources.csv', index=False)
    lineage_frame.to_csv(output / 'lineage_summary.csv', index=False)
    return resource_frame, lineage_frame


def render(output):
    selected = json.loads((output / 'selection.json').read_text())
    resource_frame, lineage = resources(output)
    names = ['ranking', 'relative_scores', 'per_fold', 'peak', 'pooled', 'criteria', 'recovery']
    data = {name: pd.read_csv(output / f'{name}.csv') for name in names}
    best = selected['best_observed_policy']
    qualified = selected['qualified_policy'] or 'none'
    counts = data['per_fold'].loc[data['per_fold'].policy.eq('B0'),
        ['fold', 'window_start', 'window_end', 'n_hours', 'n_days']]
    summary = data['relative_scores'].merge(data['pooled'][['policy', 'MAE', 'WIS']], on='policy')
    peak = data['peak'][['policy', 'n_hours', 'n_days', 'MAE', 'WIS', 'coverage95', 'hit_count95',
                         'lower_miss_count95', 'upper_miss_count95', 'mean_width95']]
    fold_cols = ['policy', 'fold', 'n_hours', 'MAE', 'WIS', 'coverage95', 'hit_count95',
                 'raw_central_MAE', 'centering_effect', 'daily_mean_level_MAE', 'within_day_shape_MAE']
    failures = data['criteria'].loc[~data['criteria'].passed]
    paragraphs = [
        '# CP-15 — adaptive forecasting comparison, v21-r1',
        '**Engineering status: pending fresh exact-candidate Integration.** The binding review is committed '
        'after this report under `docs/track-b/evidence/cp-15/integration.md`; the terminal packet reports its result. '
        'The preserved attempt-1 FAIL binds only its original candidate.',
        f'**product_feasibility: {selected["product_feasibility"]}. Best observed candidate: {best}. Qualified policy: {qualified}.** '
        'These are development screening results after selection; they do not authorize promotion.',
        'Data: ENTSO-E Transparency Platform; Bundesnetzagentur | SMARD.de — CC BY 4.0.',
        '## Scope and original evaluation hours',
        'The nine-policy comparison uses the original target hours and the fixed 2019 boundary. Long histories '
        'expand from 2019-01-01 until 728 preceding calendar days are available, then roll; A4 always uses 84 days. '
        'Every training row uses its own causal 168-hour level/scale. The protocol was committed at '
        '`bb5e67882fcfdf65b963d25ce785a3999816dfc2` before comparison. No outer score chose a parameter, window, '
        'feature or solver tolerance.',
        table(counts),
        'Each of the nine arms has 10,747 original eligible targets (96,723 predictions altogether). '
        'The 17-day peak is 2022-08-15 through 2022-08-31, 408 hours; full fold 3 is 2,112 hours on 88 represented days '
        'within its unchanged 90-calendar-day window. Ineligible original days/hours remain absent from scoring, '
        'and remain explicit empty days in the bootstrap calendar.',
        '## Equal-fold comparison and secondary pooled scores',
        'S_MAE and S_WIS are equal-weight averages of five ratios to B0. MAE and WIS below are secondary '
        'observation-weighted pooled values in EUR/MWh. Primary MAE uses the final emitted p50, after signed-error centering.',
        table(summary),
        '## Candidate ranking and qualification',
        table(data['ranking']),
        'All six criteria were applied mechanically. `criteria.csv` includes every actual value and limit, including '
        'each fold separately. B2/B3 best-per-metric comparisons are diagnostic oracles, not deployable policies. '
        'A candidate must pass every criterion to qualify.',
        '### Failed checks, with actual values',
        table(failures),
        '## Every arm, every full fold',
        table(data['per_fold'][fold_cols]),
        '`per_fold.csv` also contains RMSE, 50/80/95% coverage, exact hits and lower/upper misses, '
        'mean/median/95th-percentile widths, signed bias, missing predictions and crossings. '
        '`hourly_losses.csv` retains individual losses; `daily.csv` retains daily level, shape, centering and coverage diagnostics.',
        '## Peak stress, separate from the full crisis fold',
        table(peak),
        'The peak has only 17 delivery days. Coverage is descriptive with exact counts, not an independent-hour '
        'significance claim. Aggregate metrics include all stress outcomes.',
        '## Recovery and dependence-aware uncertainty',
        '`recovery.csv` reports all nine arms on four consecutive seven-day windows from September 1 through 28, 2022; '
        '`daily.csv` also permits inspection without choosing a recovery threshold after outcomes. These diagnostics do not select the winner.',
        '`bootstrap.csv` reports 95% percentile intervals for paired mean daily MAE/WIS differences for every A1–A5 '
        'versus B0–B3, by fold and equally across folds. Seed 15042; 2,000 replicates; noncircular blocks of seven '
        'consecutive calendar days within each full 90-day fold; all arms share each resample. Hourly observations '
        'stay together. Empty original days contribute no loss rather than zero loss. These daily-loss intervals '
        'are distinct from the hourly-weighted primary S scores. They are exploratory and subject to selection; '
        'no confirmatory p-value is claimed.',
        '## Fit provenance and resources',
        table(resource_frame[['fold', 'policy', 'direct_logical_fit_calls', 'direct_fit_seconds',
                              'component_policies', 'component_fit_seconds', 'shared_fold_wall_seconds',
                              'shared_full_command_wall_seconds',
                              'shared_process_peak_rss_bytes', 'cache_hit_origins']]),
        'Direct model-fit counts include four chronological validation penalties and one final refit per hourly LEAR '
        'model; LGBM has one fit per day. A3/A5 reuse their fitted components; B0/B1 have no fit. Component costs are '
        'shown separately, so summing them again would double count. Process RSS and fold wall time are shared '
        'measurements repeated for each arm, not invented estimator-specific allocations. '
        'Initial input/feature preparation and warm-up boundary discovery are excluded from the fold-loop timer; '
        'the separate whole-command wall time includes them, plus process startup/teardown. '
        'Ensemble arithmetic and residual emission are included in fold-loop runtime but not separately timed. '
        'Cached central forecasts keep original fitting durations; cache counts distinguish execution reuse. '
        'Lasso continuation calls are separately recorded in each fit record and do not add statistical grid choices.',
        '`lineage_summary.csv` separates genuine warm-up and evaluation fits. Each `folds/*-fits.parquet` records '
        'training/validation dates, row and normalization hashes, selected penalties, model fingerprints, convergence '
        'and timing. `*-issued.parquet` preserves the central forecasts and scales that generated errors; '
        '`*-feedback.parquet` records one-time D-2 releases; `*-origins.json` binds the latest 28 complete released '
        'days and residual hashes used at every evaluation origin. No in-sample fitted residual seeds the buffer.',
        '## Feasibility deliverables',
        'The pinned Chronos-2 probe completed two local CPU inference calls at one proper-training origin, '
        'with future load support and a positive control. An offline reproduction matched forecast bytes. '
        'It was unscored and does not enter these nine policies. See `feasibility/README.md` for exact revision, '
        'license verification, dependency freeze, inputs, process memory, runtime and reproduction. '
        '`feasibility/structural_inputs.md` covers fuel, EUA, load, renewables, capacity, outages, cross-border and '
        'weather sources with dated primary-source retrievals and explicit vintage, coverage and reuse gaps.',
        '## Preserved history and limitations',
        'The first attempt remains reachable at evidence tip `193d9cf48c586b9c4b1f43d7a5677b2d5f400832`. '
        'Its candidate `f8d0ed2a5f0737f0d088c3474b5a9fe77a406f37` retains its original FAIL. '
        '`attempt-1-preservation.json` maps byte-identical copies of its plan, protocol and reports, and '
        '`docs/track-b/evidence/cp-15/attempt-1-integration.md` preserves the old verdict. The revised history '
        'rule does not rescore or relabel that attempt.',
        'Prior v1 development point-MAE evidence remains p=0.948, statistic +1.6228, median 28.58% worse. '
        'Original v1 peak coverage was 79/408; CP-10 peak coverage was 131/408, while its full-fold result '
        'was 1,515/2,112. Those records remain distinct and unchanged. Native v1 pinball remains in its '
        'historical reports; CP-15 WIS is not renamed as that metric.',
        'The inherited A65 load-forecast availability assumption is preserved; the snapshot does not independently '
        'prove every historical issue vintage. Day-ahead D-1 prices are already published at the prior auction. '
        'D-2 error feedback is the prescribed conservative policy restriction, not a claim that D-1 prices were unavailable. '
        'Origin time is literal fixed CET noon (11:00 UTC), separate from Berlin delivery-day DST. '
        'No pre-2019 inputs, spent holdout, reserved-tail outcomes, A69 or target-day actual predictors entered fitting or selection. '
        'Residual intervals are empirical benchmarks without a finite-sample conformal guarantee. '
        'See `implementation-notes.md` for the recorded fixed-tolerance numerical repair and `reproduction.md` for commands.',
    ]
    (output / 'report.md').write_text('\n\n'.join(paragraphs) + '\n')


def manifest(root, output):
    paths = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'artifact-manifest.json')
    paths += sorted((root / 'src/cp15').glob('*.py'))
    paths += sorted((root / 'tests/cp15').glob('*.py'))
    paths += [root / 'scripts/cp15_forecasting.py', root / 'pyproject.toml', root / 'uv.lock']
    hashes = {}
    for path in paths:
        with path.open('rb') as stream:
            hashes[str(path.relative_to(root))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (output / 'artifact-manifest.json').write_text(json.dumps({'artifact_sha256': hashes,
        'evidence_class': 'development_post_selection', 'excludes': ['this manifest', 'post-candidate Integration evidence']},
        indent=2, sort_keys=True) + '\n')


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    output = root / 'reports/cp15'
    if not args.manifest_only:
        render(output)
    manifest(root, output)


if __name__ == '__main__':
    main()
