"""CP-28's stages, each a separate command, so that the one look is visible in the history.

    python -m cp28.pipeline validate-scoring <root>   # v5/B0 only: the scorer against CP-26's daily table
    python -m cp28.pipeline forecast <root>           # every arm's causal forecasts; no outcome is scored
    python -m cp28.pipeline score <root>              # the one look: losses, 4×20×7 endpoints, Holm, selection
    python -m cp28.pipeline report <root>             # report and draft publication packet (cp28.report)

`forecast` refuses to run if scores exist; `score` refuses unless the committed forecast manifest
matches the files on disk, and refuses to overwrite its own outputs. Bulk retention (bootstrap
draws) goes to `.local/artifacts/cp-28/`, listed with sizes and SHA-256 in a committed manifest.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

from . import engine, grid, inference, reader, scoring, states

OUT = 'reports/cp28'
FORECAST = f'{OUT}/forecast'
RESULTS = f'{OUT}/results'
PRE_SCORE = 'docs/track-b/evidence/cp-28/pre-score'
BULK = 'artifacts/cp-28/bulk'
CODE = ('src/cp28/__init__.py', 'src/cp28/reader.py', 'src/cp28/states.py', 'src/cp28/grid.py', 'src/cp28/meta.py',
        'src/cp28/engine.py', 'src/cp28/scoring.py', 'src/cp28/inference.py', 'src/cp28/pipeline.py',
        'src/cp16/residuals.py', 'src/cp25/guard.py', 'src/cp15/data.py')


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def plain(value):
    """JSON-safe: NumPy scalars to Python, NaN/inf to null, dates and tuples kept readable."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (date, pd.Timestamp)):
        return str(value)
    return value


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plain(value), indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')


def file_entry(path: Path, root: Path) -> dict:
    return {'path': str(path.relative_to(root)), 'bytes': path.stat().st_size, 'sha256': reader.sha256(path)}


def code_identity(root: Path) -> dict:
    return {path: reader.sha256(root / path) for path in CODE}


def context(root: Path):
    local = reader.local_root(root)
    if local is None:
        raise SystemExit('CP-26 stored inputs are absent: extract evidence/cp-26 into .local/artifacts/cp-28/inputs first')
    return local


def mask_flags(root: Path, local: Path) -> pd.DataFrame:
    """CP-26's calendar labels (masks, year, season), read from its v5 rows, for descriptive breakdowns."""
    daily = reader.read_table(root, local, 'daily')
    part = daily[daily.stream == 'nominal/v5'].set_index('delivery_date')
    return part[['all_eligible', 'never_scored', 'old_fold_2_5', 'newly_opened', 'year', 'season', 'quarter',
                 'weekday_weekend', 'nationwide_holiday', 'canonical_hours']].copy()


def stored_losses(days, prices):
    """The control's and B0's daily losses from their CP-26 emitted quantiles on the frozen keys."""
    out = {'v5': {}, 'B0': {}}
    for day in days:
        if not day.scored or not len(day.scoring_index):
            continue
        y = reader.truth_for(prices, day.scoring_index)
        for name, q in (('v5', day.v5_quantiles), ('B0', day.b0_quantiles)):
            out[name][day.day] = scoring.daily(q, y) if q is not None else (np.nan, np.nan)
    return out


# ------------------------------------------------------------------------------- validate-scoring

def validate_scoring(root: Path):
    local = context(root)
    days = reader.load_days(root, local)
    prices = reader.prices(root)
    ours = stored_losses(days, prices)
    daily = reader.read_table(root, local, 'daily')
    report = {'checked_at_utc': utc_now(), 'claim': 'the CP-28 scorer reproduces CP-26 stored v5 and B0 daily losses; '
              'no CP-28 arm exists yet and none is scored here', 'streams': {}}
    for name, stream in (('v5', 'nominal/v5'), ('B0', 'nominal/B0')):
        part = daily[(daily.stream == stream) & daily.eligible & daily.available & ~daily.failed].set_index('delivery_date')
        worst = 0.0
        missing = 0
        for d, row in part.iterrows():
            if d not in ours[name] or not np.isfinite(ours[name][d][0]):
                missing += 1
                continue
            for k, metric in enumerate(('MAE', 'WIS')):
                worst = max(worst, abs(ours[name][d][k] - row[metric]) / max(1.0, abs(row[metric])))
        extra = sum(1 for d, v in ours[name].items() if np.isfinite(v[0]) and d not in part.index)
        report['streams'][name] = {'cp26_dates': int(len(part)), 'missing_here': missing, 'extra_here': extra,
                                   'max_relative_difference': worst}
        if missing or extra or worst > 1e-12:
            raise SystemExit(f'scorer disagrees with CP-26 on {name}: {report["streams"][name]}')
    write_json(root / PRE_SCORE / 'scoring-validation.json', report)
    print(json.dumps(report, indent=2))


# --------------------------------------------------------------------------------------- forecast

QUANTILE_COLUMNS = tuple(f'q{lab}' for lab in ('025', '10', '25', '50', '75', '90', '975'))


def write_forecast(out: Path, result, frame: pd.DataFrame):
    """Persist one engine run: records, fits, features, states, hourly centrals and emitted quantiles."""
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([i.record for i in result.issues]).to_parquet(out / 'arm-days.parquet', index=False)
    fits = pd.DataFrame(result.fits)
    fits['day'] = fits.day.astype(str)
    fits.to_parquet(out / 'fits.parquet', index=False)
    from .meta import COLUMNS
    feats = pd.DataFrame([{'day': str(d), **{c: float(v) for c, v in zip(COLUMNS[:13], f['features'])},
                           'support_last7': f['support']['last7'], 'support_last28': f['support']['last28']}
                          for d, f in result.features.items()])
    feats.to_parquet(out / 'features.parquet', index=False)
    frame.astype({c: str for c in ('delivery_date', 'negative_oldest_day', 'negative_latest_day',
                                   'volatility_oldest_day', 'volatility_latest_day')}).to_parquet(out / 'issue-states.parquet', index=False)
    pd.DataFrame({'day': [str(d) for d in result.learning_days]}).to_csv(out / 'learning-days.csv', index=False)
    hourly_rows = []
    for issue in result.issues:
        if not len(issue.index):
            continue
        n = len(issue.index)
        central = issue.central if issue.central is not None else np.full(n, np.nan)
        emitted = np.full((n, 7), np.nan)
        scoring_flag = np.zeros(n, bool)
        if len(issue.scoring_index):
            positions = issue.index.get_indexer(issue.scoring_index)
            scoring_flag[positions] = True
            if issue.emitted is not None:
                emitted[positions] = issue.emitted
        hourly_rows.append(pd.DataFrame({'arm': issue.arm, 'day': str(issue.day), 'timestamp_utc': issue.index,
                                         'scoring_key': scoring_flag, 'native_central': central,
                                         **{c: emitted[:, k] for k, c in enumerate(QUANTILE_COLUMNS)}}))
    pd.concat(hourly_rows, ignore_index=True).to_parquet(out / 'hourly.parquet', index=False, compression='zstd')
    traces = [{'arm': arm, **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()}}
              for arm, rows in result.h_traces.items() for row in rows]
    pd.DataFrame(traces).to_parquet(out / 'h-traces.parquet', index=False)
    networks = {f'{d}/{key}': np.asarray(array) for d, value in result.networks.items() for key, array in value.items()}
    np.savez_compressed(out / 'networks.npz', **networks)


def forecast(root: Path):
    if (root / RESULTS).exists():
        raise SystemExit('scores exist: the forecasts are frozen; no rerun after the first score')
    local = context(root)
    started, t0 = utc_now(), time.time()
    days = reader.load_days(root, local)
    prices = reader.prices(root)
    frame = states.issue_states(prices, [d.day for d in days])
    config = engine.Config()
    if not config.production():
        raise SystemExit('the production replay uses 364/728/28')
    result = engine.run(days, prices, frame, config=config,
                        progress=lambda step, day, n: print(f'{utc_now()} origin {day} ({step + 1}/{len(days)}), learning dates {n}', flush=True))
    out = root / FORECAST
    write_forecast(out, result, frame)
    write_json(out / 'cross-checks.json', cross_checks(root, local, result, frame))
    files = [file_entry(p, root) for p in sorted(out.iterdir()) if p.name != 'manifest.json']
    manifest = {'schema': 'cp28-forecast-manifest-v1', 'stage': 'forecast', 'started_utc': started, 'finished_utc': utc_now(),
                'wall_seconds': round(time.time() - t0, 1), 'config': config.__dict__, 'code_sha256': code_identity(root),
                'inputs': {'scores_manifest': reader.SCORES_MANIFEST_SHA256, 'snapshot': reader.SNAPSHOT_SHA256,
                           'entry_identity': reader.sha256(root / 'docs/track-b/evidence/cp-28/entry-identity.json')},
                'origins': len(days), 'issues': len(result.issues),
                'latest_truth_requested_at_last_origin': str(result.latest_truth_requested),
                'files': files, 'scores_computed': False}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'files'}, indent=2))


def cross_checks(root, local, result, frame):
    """Identity checks against CP-26's own records: learning rows and the issue-time states."""
    out = {}
    path = local / reader.INPUTS / reader.RUN / 'learning-ledger.json'
    if path.exists():
        cp26 = json.loads(path.read_text())['routes']['nominal']
        theirs = [date.fromisoformat(r['day']) for r in cp26['rows']]
        ours = [d for d in result.learning_days if d <= max(theirs)]
        out['learning_rows'] = {'cp26_released_rows': len(theirs), 'cp28_rows_through_same_date': len(ours),
                                'identical_dates': ours == theirs}
    states_cp26 = reader.read_table(root, local, 'known_price_states').set_index('delivery_date')
    ours = frame.set_index('delivery_date').loc[states_cp26.index]
    agree = {column: int((ours[column].to_numpy() == states_cp26[column].to_numpy()).sum())
             for column in ('negative_price_risk', 'high_volatility', 'volatility_unknown')}
    out['issue_states'] = {'dates': int(len(states_cp26)), 'agreeing': agree,
                           'max_volatility_relative_difference': float(np.nanmax(np.abs(ours.volatility - states_cp26.volatility)
                                                                                 / states_cp26.volatility.abs()))}
    return out


# ------------------------------------------------------------------------------------------ score

def verify_forecast(root: Path) -> dict:
    manifest = json.loads((root / FORECAST / 'manifest.json').read_text())
    for entry in manifest['files']:
        if reader.sha256(root / entry['path']) != entry['sha256']:
            raise SystemExit(f'forecast file changed after its manifest: {entry["path"]}')
    return manifest


def arm_losses(forecast_dir: Path, prices) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Each arm's daily (MAE, WIS) on its scored dates; NaN for a failed issuance."""
    records = pd.read_parquet(forecast_dir / 'arm-days.parquet')
    scored = records[records.scored & (records.scoring_keys > 0)]
    hourly = pd.read_parquet(forecast_dir / 'hourly.parquet')
    hourly = hourly[hourly.scoring_key]
    groups = {key: frame for key, frame in hourly.groupby(['arm', 'day'], sort=False)}
    losses = {arm: {} for arm in engine.ARMS}
    rows = []
    for arm, day_text in zip(scored.arm, scored.day):
        part = groups[(arm, day_text)]
        q = part[list(QUANTILE_COLUMNS)].to_numpy(float)
        y = prices.reindex(pd.DatetimeIndex(part.timestamp_utc)).to_numpy(float)
        ok = np.isfinite(q).all() and np.isfinite(y).all()
        value = scoring.daily(q, y) if ok else (np.nan, np.nan)
        day = date.fromisoformat(day_text)
        losses[arm][day] = value
        rows.append({'stream': arm, 'day': day_text, 'MAE': value[0], 'WIS': value[1], 'scoring_hours': len(part),
                     **(scoring.coverage(q, y) if ok else {})})
    return losses, pd.DataFrame(rows), records


def analyze(losses: dict, records: pd.DataFrame, flags: pd.DataFrame, out: Path, bulk: Path, *, replicates=inference.REPLICATES):
    """The one look's statistics: four variants of 7×20 endpoints, Holm, the pass rule, selection, descriptives."""
    out.mkdir(parents=True, exist_ok=False)
    bulk.mkdir(parents=True, exist_ok=True)
    endpoint_rows, holm_rows, compound = [], [], {}
    bulk_files, overall_draws = [], {}
    for variant in inference.VARIANTS:
        print(f'{utc_now()} variant {variant[0]}', flush=True)
        results = inference.run_variant(losses, variant, replicates=replicates)
        if variant[0] == 'primary14':
            overall_draws = {(arm, r['metric']): r for arm in inference.ARMS for r in results[arm]['endpoints']
                             if r['group'] == 'overall'}
        compound[variant[0]] = {arm: results[arm]['compound_p'] for arm in inference.ARMS}
        for row in inference.holm(compound[variant[0]]):
            holm_rows.append({'variant': variant[0], **row})
        for arm in inference.ARMS:
            arrays = {}
            for r in results[arm]['endpoints']:
                if r['starts'] is not None:
                    arrays[f'{r["endpoint"]:02d}_starts'] = r['starts']
                    arrays[f'{r["endpoint"]:02d}_statistics'] = r['replicate_statistics']
                endpoint_rows.append({k: v for k, v in r.items() if k not in ('starts', 'replicate_statistics', 'runs')}
                                     | {'runs': json.dumps(r['runs'])})
            path = bulk / f'{variant[0]}-{arm}.npz'
            np.savez_compressed(path, **arrays)
            bulk_files.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': reader.sha256(path), 'arrays': len(arrays)})
    endpoints = pd.DataFrame(endpoint_rows)
    endpoints.to_parquet(out / 'endpoints.parquet', index=False)
    holm = pd.DataFrame(holm_rows)
    holm.to_csv(out / 'holm.csv', index=False)
    rejected = {(v, a): bool(r) for v, a, r in zip(holm.variant, holm.arm, holm.rejected)}
    passes = {arm: all(rejected[(v[0], arm)] for v in inference.VARIANTS) for arm in inference.ARMS}
    primary = {arm: rejected[('primary14', arm)] for arm in inference.ARMS}
    wis, counts, control_wis = {}, {}, {}
    for arm in inference.ARMS:
        paired = inference.paired(losses, arm)
        counts[arm] = {'paired_dates': len(paired),
                       'control_dates_without_arm_loss': sum(1 for d, v in losses['v5'].items() if np.isfinite(v[1])
                                                             and not np.isfinite(losses[arm].get(d, (np.nan, np.nan))[1]))}
        wis[arm] = float(np.mean([losses[arm][d][1] for d in paired])) if paired else None
        control_wis[arm] = float(np.mean([losses['v5'][d][1] for d in paired])) if paired else None
    passing = [arm for arm in inference.ARMS if passes[arm]]
    selection = inference.select(passing, wis)
    selection.update({'wis_paired_with_control': wis, 'control_wis_on_each_arm_pairs': control_wis, 'paired_counts': counts,
                      'primary_rejections': primary, 'passes_all_four': passes,
                      'inconclusive_primary_pass_failed_variant': [a for a in inference.ARMS if primary[a] and not passes[a]]})
    write_json(out / 'selection.json', selection)
    write_json(out / 'compound.json', compound)
    descriptive, pivots = describe(losses, records, flags, overall_draws)
    write_json(out / 'descriptive.json', descriptive)
    path = bulk / 'descriptive-ratio-pivots.npz'
    np.savez_compressed(path, **pivots)
    bulk_files.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': reader.sha256(path), 'arrays': len(pivots)})
    write_json(out / 'bulk-manifest.json', {'schema': 'cp28-bulk-manifest-v1', 'location': f'.local/{BULK}/bootstrap',
                                            'retained_locally_only': True, 'files': bulk_files, 'replicates': replicates,
                                            'contents': 'per variant and arm: every computable endpoint\'s drawn block starts '
                                                        '(replicate × block, zero-based within its run) and its replicate '
                                                        'statistics; and the descriptive ratio pivots'})
    return selection


def score(root: Path):
    out = root / RESULTS
    if out.exists():
        raise SystemExit('scores already exist: there is one look')
    verify_forecast(root)
    local = context(root)
    started, t0 = utc_now(), time.time()
    prices = reader.prices(root)
    days = reader.load_days(root, local)
    losses, daily_rows, records = arm_losses(root / FORECAST, prices)
    stored = stored_losses(days, prices)
    losses.update(stored)
    daily_rows = pd.concat([daily_rows, pd.DataFrame([{'stream': name, 'day': str(d), 'MAE': v[0], 'WIS': v[1]}
                                                      for name in ('v5', 'B0') for d, v in stored[name].items()])], ignore_index=True)
    selection = analyze(losses, records, mask_flags(root, local), out, local / BULK / 'bootstrap')
    daily_rows.to_parquet(out / 'daily-losses.parquet', index=False)
    files = [file_entry(p, root) for p in sorted(out.iterdir()) if p.name != 'manifest.json']
    write_json(out / 'manifest.json', {'schema': 'cp28-results-manifest-v1', 'stage': 'score', 'started_utc': started,
                                       'finished_utc': utc_now(), 'wall_seconds': round(time.time() - t0, 1),
                                       'forecast_manifest_sha256': reader.sha256(root / FORECAST / 'manifest.json'),
                                       'code_sha256': code_identity(root), 'files': files})
    print(json.dumps({k: selection[k] for k in ('selected', 'tie_set', 'best_arm', 'reason', 'passes_all_four')}, indent=2))


def describe(losses, records, flags, overall_draws):
    """Descriptive only: raw losses, ratios to the control and to B0, masks, startup and fallbacks.

    Ratio intervals reuse the primary overall endpoint's own paired resamples (its stored block
    starts), recompute the delta-method influence in each replicate, and keep every pivot.
    """
    out = {'arms': {}, 'fallbacks': {}, 'startup': {}}
    pivots = {}
    for arm in inference.ARMS:
        paired = inference.paired(losses, arm)
        entry = {'paired_dates': len(paired)}
        for k, metric in enumerate(inference.METRICS):
            a = np.array([losses[arm][d][k] for d in paired])
            c = np.array([losses['v5'][d][k] for d in paired])
            b = np.array([losses['B0'][d][k] for d in paired])
            entry[metric] = {'arm': float(a.mean()), 'control': float(c.mean()), 'B0': float(b.mean()),
                             'ratio_to_control': float(a.mean() / c.mean()), 'ratio_to_B0': float(a.mean() / b.mean()),
                             'difference_to_control': float(a.mean() - c.mean())}
            draws = overall_draws.get((arm, metric))
            if draws is not None and draws['starts'] is not None and np.isfinite(b).all():
                runs = tuple(tuple(r) for r in draws['runs'])
                plan = inference.DrawPlan(runs, 14, draws['starts'], plan_columns(runs, 14))
                total = len(draws['starts'])
                indices = [plan.indices(i, min(i + inference.CHUNK, total)) for i in range(0, total, inference.CHUNK)]
                for name, other in (('control', c), ('B0', b)):
                    interval = inference.ratio_interval(a, other, indices, runs)
                    pivots[f'{arm}_{metric}_ratio_to_{name}'] = interval.pop('pivots')
                    entry[metric][f'ratio_to_{name}_ci95'] = interval
            else:
                entry[metric]['ratio_intervals'] = 'undefined: the primary overall endpoint drew no resamples'
            masks = {}
            for mask in ('all_eligible', 'never_scored', 'old_fold_2_5', 'newly_opened'):
                sel = [i for i, d in enumerate(paired) if d in flags.index and bool(flags.loc[d, mask])]
                masks[mask] = {'dates': len(sel), 'arm': float(a[sel].mean()) if sel else None,
                               'control': float(c[sel].mean()) if sel else None,
                               'ratio_to_control': float(a[sel].mean() / c[sel].mean()) if sel else None}
            entry[metric]['masks'] = masks
            for group in ('year', 'season'):
                cells = {}
                for value in sorted({str(flags.loc[d, group]) for d in paired if d in flags.index}):
                    sel = [i for i, d in enumerate(paired) if d in flags.index and str(flags.loc[d, group]) == value]
                    cells[value] = {'dates': len(sel), 'arm': float(a[sel].mean()), 'control': float(c[sel].mean()),
                                    'ratio_to_control': float(a[sel].mean() / c[sel].mean())}
                entry[metric][group] = cells
        out['arms'][arm] = entry
        part = records[(records.arm == arm) & records.scored & (records.scoring_keys > 0)]
        out['fallbacks'][arm] = {'scored_dates': int(len(part)),
                                 'by_source': {str(k): int(v) for k, v in part.output_source.value_counts().items()},
                                 'fallback_dates': int(part.fallback.sum()),
                                 'by_reason': {str(k): int(v) for k, v in part[part.fallback].reasons.value_counts().items()}}
        rules = records[records.arm == arm][['day', 'weight_rule', 'learning_dates', 'learning_oldest', 'learning_latest',
                                              'learning_span_days']]
        changes = rules[rules.weight_rule != rules.weight_rule.shift()]
        out['startup'][arm] = changes.to_dict('records')
    return out, pivots


def plan_columns(runs, block_days):
    columns, col = [], 0
    for begin, end, _ in runs:
        blocks = -(-(end - begin) // min(block_days, end - begin))
        columns.append((col, col + blocks))
        col += blocks
    return tuple(columns)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    stage, root = argv[0], Path(argv[1]).resolve()
    {'validate-scoring': validate_scoring, 'forecast': forecast, 'score': score}[stage](root)


if __name__ == '__main__':
    main()
