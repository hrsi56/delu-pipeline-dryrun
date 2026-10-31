"""Render CP-26's local report/publication packet from stored evidence only.

No score, estimator, inference or deployment function is called here. Every
research value retains its source-row identifier, and unavailable endpoints
remain unavailable. The local packet is preparation for Owner review, not
publication authority or an Integration verdict.
"""
from __future__ import annotations

from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path

from cp25 import guard
from .state import _save, digest

MASKS = ('all_eligible', 'never_scored', 'old_fold_2_5', 'newly_opened')
TRANSITIONS = (('v2', 'v3', 'Adding weather forecasts'),
               ('v3', 'v4', 'Adding block LightGBM members'),
               ('v4', 'v5', 'Adding the DDNN-2 ensemble'))
BREAKDOWNS = ('season', 'year', 'quarter', 'local_hour', 'weekday_weekend', 'nationwide_holiday',
              'canonical_hours', 'stress_2021_2022', 'negative_price_risk', 'high_volatility',
              'realized_outcome_slices')
EVIDENCE_CLASS = 'development_post_selection'
from .gates import OUTPUT, RUN_EVIDENCE
DOCS = ('capstone_v21.md', 'docs/track-b/evidence/cp-26/issued-continuation-brief-v21-r14.md',
        'docs/track-b/evidence/cp-26/cp28-frozen-protocol-v21-r14.md', 'docs/PUBLISH_RULES.md',
        'docs/track-b/publication-standard-v1.md', 'docs/track-b/publication-packet-template.md',
        str(RUN_EVIDENCE / 'issued-brief.md'), str(RUN_EVIDENCE / 'registration.json'))
ARRAY_FIELDS = ('draw_starts', 'draw_indices')


def _plain(value):
    if type(value).__name__ in ('NAType', 'NaTType'):
        return None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, 'tolist'):
        return _plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f'unsupported report evidence type: {type(value).__name__}')


def _records(value):
    if value is None:
        return []
    if hasattr(value, 'to_dict'):
        return _plain(value.to_dict('records'))
    if isinstance(value, list):
        return _plain(value)
    if isinstance(value, dict) and 'rows' in value:
        return _records(value['rows'])
    if isinstance(value, dict) and all(isinstance(row, dict) for row in value.values()):
        return _plain([{'identity': key, **row} for key, row in value.items()])
    raise ValueError('report tables must be DataFrames or lists of row dictionaries')


def _catalog(name, value, *, compact=False):
    result, ids = [], set()
    for record in _records(value):
        for key in ('delivery_date', 'day'):
            if record.get(key):
                guard.admit_delivery_day(date.fromisoformat(str(record[key])[:10]))
        if compact:
            record = {k: v for k, v in record.items()
                      if not k.startswith('replicate_') and not k.endswith('_pivots') and k not in ARRAY_FIELDS}
        row_id = record.get('row_id') or record.get('endpoint_id') or f'{name}/{digest(record)[:24]}'
        if row_id in ids:
            raise ValueError(f'duplicate evidence row id in {name}')
        ids.add(row_id)
        result.append({**record, 'row_id': row_id})
    return sorted(result, key=lambda row: row['row_id'])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _artifact_path(root, relative):
    path = Path(relative)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root / 'reports/rebaseline'):
        raise ValueError('numeric evidence must remain in approved rebaseline outputs')
    return path


def _verify_references(root, rows):
    for row in rows:
        if 'metadata_path' not in row:
            continue
        path = _artifact_path(root, row['metadata_path'])
        if row.get('metadata_sha256') != _sha(path):
            raise ValueError('numeric endpoint metadata hash mismatch')
        source = json.loads(path.read_text())
        if source.get('endpoint_id') != row.get('endpoint_id') or source.get('schema') != 'cp26-numeric-artifact-v1':
            raise ValueError('numeric endpoint artifact identity mismatch')
        if row.get('arrays_path'):
            arrays = _artifact_path(root, row['arrays_path'])
            if row.get('arrays_sha256') != _sha(arrays) or source.get('arrays_sha256') != row['arrays_sha256']:
                raise ValueError('numeric endpoint array hash mismatch')
        omitted = {'endpoint_id', 'row_id', 'metadata_path', 'metadata_sha256', 'arrays_path', 'arrays_sha256'}
        for key, value in row.items():
            if key not in omitted and _plain(source['result'].get(key)) != value:
                raise ValueError('numeric endpoint differs from its committed metadata row')


def _load_table(root, value):
    if not isinstance(value, dict) or 'path' not in value:
        return value
    if 'json_columns' in value and 'row_ids_sha256' in value:
        from .analysis import _load_table as load_analysis_table
        return load_analysis_table(root, value)
    path = Path(value['path'])
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if not path.is_relative_to(root / 'reports/rebaseline'):
        raise ValueError('report source artifact must remain in approved rebaseline outputs')
    if value.get('sha256') != _sha(path):
        raise ValueError('report source table differs from its manifest hash')
    if path.suffix == '.json':
        loaded = json.loads(path.read_text())
        return loaded['rows'] if isinstance(loaded, dict) and 'rows' in loaded else loaded
    import pandas as pd
    if path.suffix == '.csv':
        return pd.read_csv(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        guard.admit_delivery_day(guard.BOUNDARY)
        columns = pq.read_schema(path).names
        filters = [('delivery_date', '<=', guard.BOUNDARY)] if 'delivery_date' in columns else None
        return pd.read_parquet(path, filters=filters)
    raise ValueError('unsupported serialized report table')


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _ci(value):
    return value if isinstance(value, list) and len(value) == 2 and all(_number(x) for x in value) else None


def _find(rows, **criteria):
    found = [row for row in rows if all(row.get(k) == v for k, v in criteria.items())]
    if len(found) > 1:
        raise ValueError('ambiguous evidence endpoint; cannot choose a result for presentation')
    return found[0] if found else None


def _comparison(endpoints, left, right, mask, metric):
    row = _find(endpoints, kind='paired_comparison', variant='primary14', left=left, right=right,
                mask=mask, metric=metric, group='overall', value='overall')
    if row is None:
        return {'left': left, 'right': right, 'mask': mask, 'metric': metric, 'status': 'unavailable',
                'reason': 'registered comparison endpoint absent', 'source_rows': [], 'point_comparison': True}
    ratio, difference = row.get('ratio'), row.get('difference')
    ratio_ci = _ci(row.get('ratio_ci95')) if row.get('ratio_status') == 'computable' else None
    difference_ci = _ci(row.get('difference_ci95')) if row.get('difference_status') == 'computable' else None
    return {'left': left, 'right': right, 'mask': mask, 'metric': metric,
            'status': 'point_available' if _number(ratio) or _number(difference) else 'unavailable',
            'difference': difference, 'difference_ci95': difference_ci,
            'ratio': ratio, 'ratio_ci95': ratio_ci,
            'change_percent': 100 * (ratio - 1) if _number(ratio) else None,
            'change_percent_ci95': [100 * (x - 1) for x in ratio_ci] if ratio_ci else None,
            'difference_status': row.get('difference_status'), 'ratio_status': row.get('ratio_status'),
            'point_comparison': ratio_ci is None,
            'reason': row.get('ratio_status') if ratio_ci is None else None,
            'calendar_dates': row.get('calendar_dates'), 'paired_dates': row.get('paired_dates'),
            'either_missing_dates': row.get('either_missing_dates'), 'source_rows': [row['row_id']]}


def _endpoint_label(value, *, eur=False):
    if not _number(value):
        return 'unavailable'
    return f'{value:.1f}' if eur else f'{value:.4g}'


def _percent(value):
    if not _number(value):
        return 'unavailable'
    if value != 0 and abs(value) < .5:
        return 'positive, under 1%' if value > 0 else 'negative, under 1%'
    return f'{value:+.0f}%'


def _comparison_text(row):
    if row['status'] == 'unavailable':
        return f"unavailable ({row['reason']})"
    text = f"{_percent(row.get('change_percent'))} as a share of {row['right']}'s score"
    interval = row.get('change_percent_ci95')
    if interval:
        text += f"; 95% ratio interval [{_percent(interval[0])}, {_percent(interval[1])}]"
    else:
        text += f"; point comparison, interval unavailable ({row.get('reason') or 'not supplied'})"
    text += f"; paired dates {row.get('paired_dates')}, missing pairs {row.get('either_missing_dates')}"
    return text


def _summary_table(rows):
    columns = ('mask', 'stream', 'metric', 'raw_mean_daily_loss', 'B0_ratio', 'raw_available_dates', 'paired_dates', 'row_id')
    text = ['| Mask | Policy | Metric | Raw mean daily loss | B0 ratio | Available dates | Paired dates | Source row |',
            '|---|---|---|---:|---:|---:|---:|---|']
    for row in rows:
        text.append('| ' + ' | '.join(_endpoint_label(row.get(k), eur=k == 'raw_mean_daily_loss')
                                     if k in ('raw_mean_daily_loss', 'B0_ratio') else str(row.get(k)) for k in columns) + ' |')
    return '\n'.join(text)


def _zero_support(row):
    """Use stored support, without recomputing any diagnostic or score."""
    support = next((row[name] for name in ('n_hours', 'n_dates', 'raw_available_dates')
                    if _number(row.get(name))), None)
    return support == 0


def _score_unavailable(name, row):
    parts = []
    if row.get('status') not in (None, 'computable') or not _number(row.get('raw_mean_daily_loss')):
        parts.append('raw_loss')
    if row.get('ratio_status') not in (None, 'computable') or not _number(row.get('B0_ratio')):
        parts.append('B0_ratio')
    return {'table': name, **row, 'unavailable_parts': parts} if parts else None


def _breakdown_support(tables, diagnostics):
    from . import scoring as S
    registry = S.registered_breakdown_cells(include_local_hours=True)
    result = []
    for mask in MASKS:
        for name in BREAKDOWNS:
            unexpected, duplicates = 0, 0
            if name == 'local_hour':
                rows = [r for r in tables['local_hour_summaries'] if r['mask'] == mask]
                expected = 24 * len(S.RESEARCH_STREAMS) * 2
            elif name == 'realized_outcome_slices':
                rows = [r for r in diagnostics.get('slices', []) + diagnostics.get('realized_outcome_slices', [])
                        if r.get('mask') == mask]
                keys = ('stream', 'slice', 'metric')
                identities = {digest(dict(zip(keys, (stream, kind, metric))))
                              for stream in S.RESEARCH_STREAMS
                              for kind in ('negative_realized_hour', 'extreme_realized_daily_MAE')
                              for metric in ('MAE', 'WIS')}
                observed = [digest({key: row.get(key) for key in keys}) for row in rows]
                expected = len(identities)
                unexpected, duplicates = len(set(observed) - identities), len(observed) - len(set(observed))
            else:
                rows = [r for r in tables['summaries'] if r.get('group') == name and r['mask'] == mask]
                expected = sum(group == name for group, _ in registry) * len(S.RESEARCH_STREAMS) * 2
            zero = sum(_zero_support(row) for row in rows)
            supported = sum(not _zero_support(row) and row.get('status') in (None, 'computable') for row in rows)
            absent = len(identities - set(observed)) if name == 'realized_outcome_slices' else max(0, expected - len(rows))
            result.append({'name': name, 'mask': mask, 'status': 'complete registered inventory'
                           if not (absent or unexpected or duplicates) else 'unresolved inventory',
                           'expected_cells': expected, 'supplied_cells': len(rows), 'supported_cells': supported,
                           'zero_support_cells': zero, 'other_unavailable_cells': len(rows) - zero - supported,
                           'unexpected_cells': unexpected, 'duplicate_cells': duplicates,
                           'absent_cells': absent, 'source_rows': [row['row_id'] for row in rows]})
    return result


def _empty_breakdowns(tables, diagnostics):
    groups = {}
    sources = {name: tables[name] for name in ('summaries', 'local_hour_summaries')}
    sources.update({name: diagnostics.get(name, []) for name in ('coverage', 'quantile_hits', 'histogram')})
    for table, rows in sources.items():
        for row in rows:
            if not _zero_support(row):
                continue
            group = 'local_hour' if table == 'local_hour_summaries' else row['group']
            value = row['local_hour'] if table == 'local_hour_summaries' else row['value']
            identity = digest({'mask': row['mask'], 'group': group, 'value': value})
            entry = groups.setdefault(identity, {'mask': row['mask'], 'group': group, 'value': value,
                                                 'reasons': [], 'source_rows': []})
            if row.get('reason') and row['reason'] not in entry['reasons']:
                entry['reasons'].append(row['reason'])
            entry['source_rows'].append({'table': table, 'row_id': row['row_id']})
    return sorted(groups.values(), key=lambda row: (row['mask'], row['group'], str(row['value'])))


def _compact_sources(rows):
    if not rows:
        return 'none'
    return rows[0] + (f' (+{len(rows)-1}; full refs in report.json)' if len(rows) > 1 else '')


def _check_fallbacks(rows):
    for row in rows:
        n, fallback, failed = row.get('eligible_dates'), row.get('fallback_dates'), row.get('failed_dates')
        if not all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (n, fallback, failed)) \
                or fallback + failed > n:
            raise ValueError('invalid fallback accounting source counts')
        fraction = fallback / n if n else None
        supplied = row.get('fallback_fraction')
        if (_number(fraction) and (not _number(supplied) or not math.isclose(fraction, supplied, rel_tol=1e-12, abs_tol=1e-12))) \
                or fraction is None and supplied is not None:
            raise ValueError('fallback fraction differs from source counts')
        if row.get('exceeds_one_percent') != bool(n and fraction > .01):
            raise ValueError('fallback threshold flag differs from source counts')


def _render_report(report):
    lines = ['# Daily development re-baseline', '',
             'Development · post-selection. This control audit changes no adoption, replacement or demotion decision.', '',
             'Point error is p50 MAE in EUR/MWh. Interval error is seven-quantile WIS in EUR/MWh. '
             'Actual hours are averaged within each date, and dates receive equal weight. '
             'B0 ratios use the same paired dates as their numerator. Lower scores are better.', '',
             'The measurement calendar is 2020-12-29 through 2026-09-06, with CP-25 exclusions and partial scoring keys preserved. '
             'Warm-up forecasts from 2020-01-01 through 2020-12-28 are unscored. '
             'Issuance is D-1 at 11:00 Europe/Berlin; own errors release only through D-2.', '',
             '## Four registered masks', '', _summary_table(report['primary']), '',
             'Each raw-loss population and paired B0 population is disclosed separately. '
             'Historical five-fold scores remain separate from these daily scores.', '',
             '## Generation control audits', '']
    for transition in report['transitions']:
        lines.extend([f"### {transition['predecessor']} → {transition['successor']}: {transition['change']}", '',
                      'Descriptive paired re-baseline; the comparison does not isolate a causal feature contribution.'])
        for row in transition['comparisons']:
            lines.extend(['', f"- {row['mask']} · {row['metric']}: {_comparison_text(row)}. Sources: {', '.join(row['source_rows']) or 'none; endpoint unavailable'}."])
        lines.extend(['', f"Year/season detail rows: {', '.join(transition['breakdown_rows']) or 'unavailable'}. ", ''])
    lines.extend(['## Load-forecast bound and delay sensitivity', '',
                  'The nominal policy retains the load forecast under the recorded Owner ruling. '
                  'The no-load route is the declared variant, with its own searches and residual history.'])
    for row in report['load_bound']:
        lines.extend(['', f"- {row['metric']}: {_comparison_text(row)}. Sources: {', '.join(row['source_rows']) or 'none'}. "])
    lines.extend(['', 'The GFS+6-hour check is fragile when either primary paired score worsens by more than 2% against nominal, '
                  'under the rule recorded on 2026-10-12.'])
    for row in report['gfs6']['comparisons']:
        lines.extend(['', f"- {row['metric']}: {_comparison_text(row)}. Sources: {', '.join(row['source_rows']) or 'none'}. "])
    lines.extend(['', f"Fragility disposition: {report['gfs6']['status']}. This is reported for the Owner's later CP-28 decision.", '',
                  '## Fallbacks and missing outputs', '',
                  '| Policy | Calendar dates | Eligible dates | B0 fallback dates | Missing dates | Above 1% | Source row |',
                  '|---|---:|---:|---:|---:|---|---|'])
    for row in report['fallbacks']:
        lines.append('| ' + ' | '.join(str(row.get(k)) for k in ('stream', 'calendar_dates', 'eligible_dates',
                     'fallback_dates', 'failed_dates', 'exceeds_one_percent', 'row_id')) + ' |')
    lines.extend(['', 'Emitted B0 fallbacks retain their scored losses. A valid native central can still seed its own error '
                  'after D-2 when insufficient interval history is the sole fallback cause. Partial scoring keys, complete '
                  'own-error issuances and complete native learning rows are separate predicates.', '',
                  '## Descriptive references', '', _summary_table(report['references']), '',
                  'DDNN-2 alone is descriptive here. The released-v1 artifact is described only on its original '
                  '2026-06-09 through 2026-09-06 holdout. That historical window is development data in this audit.', '',
                  'Historical released-v1 records retain their original issue/source contract and the protected '
                  '“confirmatory-style, not power-qualified” label. Frozen-key differences remain in the reference metadata. '
                  'This reference does not add a contrast to the thirteen-stream inference family.', '',
                  '| Released reference delivery day | MAE, EUR/MWh | WIS, EUR/MWh | Available | Source row |',
                  '|---|---:|---:|---|---|'])
    for row in report['released_reference']['daily_rows']:
        lines.append(f"| {row.get('delivery_date', row.get('day'))} | {_endpoint_label(row.get('MAE'), eur=True)} | "
                     f"{_endpoint_label(row.get('WIS'), eur=True)} | {row.get('available')} | {row['row_id']} |")
    lines.extend(['', 'Absolute MAE context on the all-eligible mask:', '',
                  '| Policy | Complete ordinary-year range, EUR/MWh | Registered stress-label MAE, EUR/MWh | Source rows |',
                  '|---|---:|---:|---|'])
    for row in report['absolute_context']:
        span = row['ordinary_complete_year_range']
        shown = f'{span[0]:.1f}–{span[1]:.1f}' if span else 'unavailable'
        stress = row['stress_rows']
        lines.append(f"| {row['stream']} | {shown} | {', '.join(_endpoint_label(r.get('raw_mean_daily_loss'), eur=True) for r in stress) or 'unavailable'} | "
                     f"{', '.join(row['ordinary_source_rows'] + [r['row_id'] for r in stress]) or 'none'} |")
    lines.extend(['', 'The ordinary range uses complete non-stress years 2023–2025. Partial 2020 and 2026 values '
                  'are retained separately in report.json. This absolute context does not rank policies.', '',
                  '## Uncertainty, calibration and breakdowns', '',
                  'Primary uncertainty uses paired, non-circular 14-calendar-day moving blocks, calendar-year strata, '
                  '10,000 replicates and seed 20261012, with Bartlett HAC at lag 14 and bootstrap-t intervals. '
                  'The 7-day, 28-day and unstratified sensitivity artifacts remain linked. '
                  'Degenerate observed or replicate variance is disclosed; no interval is invented.', '',
                  'Coverage is reported beside interval width; quantile hits and the eight-bin histogram are descriptive. '
                  'Christoffersen, moment-family diagnostics, Diebold–Mariano, Giacomini–White and the 90% MCS '
                  'do not create an adoption claim.', '',
                  '| Mask | Registered breakdown | Supported | Zero support | Other unavailable | Absent | Source rows |',
                  '|---|---|---:|---:|---:|---:|---|'])
    for row in report['breakdown_coverage']:
        lines.append(f"| {row['mask']} | {row['name']} | {row['supported_cells']} | {row['zero_support_cells']} | "
                     f"{row['other_unavailable_cells']} | {row['absent_cells']} | {_compact_sources(row['source_rows'])} |")
    lines.extend(['', 'Registered cells with zero emitted-hour or date support retain unavailable estimates. '
                  'Counts above describe stored score cells; the calibration tables retain their own support counts.', '',
                  '| Mask | Zero-support group | Value | Stored cells | Reason | Source row |',
                  '|---|---|---|---:|---|---|'])
    for row in report['empty_breakdown_cells']:
        refs = [reference['row_id'] for reference in row['source_rows']]
        lines.append(f"| {row['mask']} | {row['group']} | {row['value']} | {len(refs)} | "
                     f"{', '.join(row['reasons']) or 'zero recorded support'} | {_compact_sources(refs)} |")
    lines.extend(['', 'Coverage and width on the overall cells:', '',
                  '| Mask | Policy | Interval level | Coverage | Width, EUR/MWh | Dates | Source row |',
                  '|---|---|---:|---:|---:|---:|---|'])
    for row in report['diagnostic_tables'].get('coverage', []):
        if row.get('group') == 'overall':
            level = row.get('interval')
            lines.append('| ' + ' | '.join((str(row.get('mask')), str(row.get('stream')), str(level),
                         _endpoint_label(row.get(f'coverage{level}')), _endpoint_label(row.get(f'width{level}'), eur=True),
                         str(row.get('n_dates')), row['row_id'])) + ' |')
    lines.extend(['', f"Unavailable or degenerate inference endpoints: {len(report['unavailable_inference'])}. "
                  'Their exact statuses and artifact references are retained in report.json.', '',
                  f"Unavailable diagnostic cells: {len(report['unavailable_diagnostics'])}. "
                  'An accounting p-value of one for an unavailable cell does not establish calibration.', '',
                  f"Score cells with unavailable raw losses or B0 ratios: {len(report['unavailable_score_cells'])}. "
                  'Their recorded statuses, reasons and source IDs remain in report.json; a supported raw loss can have an unavailable ratio.', '',
                  '## Resources and limitations', '',
                  'The supplied resource snapshot preserves its measured, reported and reserved worker time fields. '
                  'Closed-repair and continuation/review identities are displayed only when supplied by the accounting source.', '',
                  '```json', json.dumps(report['resources'], indent=2, sort_keys=True), '```', ''])
    lines.extend(f"- {text}" for text in report['limitations'])
    lines.extend(['', '## CP-28 custody', '', report['cp28_custody'], '',
                  'The local publication packet and MLflow export remain drafts. Review, Owner landing and publication '
                  'have no completion claim in this report.', '', f"Evidence fingerprint: `{report['candidate_fingerprint']}`.", ''])
    return '\n'.join(lines)


def build_report(root, output_root, scores, inference_manifest, diagnostics_manifest, resource_summary, protocol_identity):
    """Write deterministic report, local draft packet and local MLflow export.

    Scores use score_records' dictionary. Inference's ``endpoints`` are compact
    iter_inference rows with stable endpoint_id and metadata/array references.
    Diagnostics' ``tables`` map names to DataFrames, row lists or hash-bound
    path/sha256 references; ``moment_family`` and ``mcs`` carry compact rows.
    No input row's missing or degenerate status is replaced by an estimate.
    """
    root = Path(root).resolve()
    output = Path(output_root or root / OUTPUT)
    output = (output if output.is_absolute() else root / output).resolve()
    if not output.is_relative_to(root / 'reports/rebaseline'):
        raise ValueError('local report outputs must remain in reports/rebaseline')
    if not isinstance(protocol_identity, dict) or not protocol_identity:
        raise ValueError('report requires the shared frozen protocol identity')
    required = {'daily', 'hourly', 'summaries', 'secondary_summaries', 'local_hour_summaries', 'fallbacks', 'population_evidence'}
    if not required.issubset(scores):
        raise ValueError('report requires the scoring builder complete output schema')
    from .analysis import validate_score_tables, validate_calibration_tables
    score_inventory = validate_score_tables(scores)
    tables = {name: _catalog(name, scores[name]) for name in ('summaries', 'secondary_summaries', 'local_hour_summaries', 'fallbacks')}
    endpoints = _catalog('inference', inference_manifest.get('endpoints', []), compact=True)
    _verify_references(root, endpoints)
    diagnostics = {name: _catalog(f'diagnostics/{name}', _load_table(root, value), compact=True)
                   for name, value in diagnostics_manifest.get('tables', {}).items()}
    calibration_inventory = validate_calibration_tables(diagnostics)
    moment = diagnostics_manifest.get('moment_family', {})
    diagnostics['moment_family'] = _catalog('diagnostics/moment_family', moment.get('cells', []) if isinstance(moment, dict) else moment, compact=True)
    diagnostics['moment_holm'] = _catalog('diagnostics/moment_holm', moment.get('holm', []) if isinstance(moment, dict) else [], compact=True)
    diagnostics['mcs'] = _catalog('diagnostics/mcs', diagnostics_manifest.get('mcs', {}).get('cells', [])
                                if isinstance(diagnostics_manifest.get('mcs', {}), dict) else diagnostics_manifest['mcs'], compact=True)
    for rows in diagnostics.values():
        _verify_references(root, rows)
    diagnostic_sources = {name: {'rows_sha256': digest(rows),
                         'artifact': _plain(diagnostics_manifest.get('tables', {}).get(name))
                         if isinstance(diagnostics_manifest.get('tables', {}).get(name), dict)
                         and 'path' in diagnostics_manifest['tables'][name] else None}
                          for name, rows in diagnostics.items()}
    sources = {'score_tables': {name: digest(rows) for name, rows in tables.items()},
               'scores_manifest': _plain(scores.get('manifest', {})),
               'population_evidence': _plain(scores['population_evidence']),
               'inference_manifest': {**_plain({k: v for k, v in inference_manifest.items() if k != 'endpoints'}),
                                      'endpoints': endpoints},
               'diagnostics_manifest': {**_plain({k: v for k, v in diagnostics_manifest.items()
                                                if k not in ('tables', 'moment_family', 'mcs')}),
                                        'tables': diagnostic_sources},
               'generator': {'path': 'src/cp26/reporting.py', 'sha256': _sha(root / 'src/cp26/reporting.py')}
                            if (root / 'src/cp26/reporting.py').exists() else {'status': 'source absent in supplied root'},
               'documents': {name: {'sha256': _sha(root / name)} if (root / name).exists()
                             else {'status': 'unavailable', 'reason': 'source document absent'} for name in DOCS}}
    primary = []
    for source in tables['summaries']:
        if source.get('group') != 'overall' or source.get('value') != 'overall':
            continue
        row = dict(source)
        own = _find(endpoints, kind='own_raw_loss', variant='primary14', stream=row['stream'],
                    mask=row['mask'], metric=row['metric'], group='overall', value='overall')
        row['raw_ci95'] = _ci(own.get('raw_left_ci95')) if own and own.get('raw_left_status') == 'computable' else None
        row['raw_uncertainty_source'] = own['row_id'] if own else None
        row['raw_uncertainty_status'] = own.get('raw_left_status') if own else 'endpoint unavailable'
        primary.append(row)
    streams = sorted({r['stream'] for r in primary})
    transitions = []
    for before, after, change in TRANSITIONS:
        left, right = f'nominal/{after}', f'nominal/{before}'
        transitions.append({'predecessor': before, 'successor': after, 'change': change,
                            'evidence_class': EVIDENCE_CLASS, 'decision': 'descriptive control audit; no new adoption',
                            'comparisons': [_comparison(endpoints, left, right, mask, metric) for mask in MASKS for metric in ('MAE', 'WIS')],
                            'breakdown_rows': [r['row_id'] for r in endpoints if r.get('kind') == 'paired_comparison'
                                and r.get('variant') == 'primary14' and r.get('left') == left and r.get('right') == right
                                and r.get('group') in ('year', 'season')]})
    load = [_comparison(endpoints, 'noload/v5', 'nominal/v5', 'all_eligible', metric) for metric in ('MAE', 'WIS')]
    delay = [_comparison(endpoints, 'gfs6/v5', 'nominal/v5', 'all_eligible', metric) for metric in ('MAE', 'WIS')]
    finite_delay = [r['ratio'] for r in delay if _number(r.get('ratio'))]
    fragile = any(r > 1.02 for r in finite_delay)
    delay_status = 'fragile' if fragile else 'not fragile under observed paired scores' if len(finite_delay) == 2 else 'unavailable; one or both primary comparisons missing'
    _check_fallbacks(tables['fallbacks'])
    main = set(protocol_identity.get('main_policy_streams', [f'nominal/{p}' for p in ('B0', 'v1-R', 'v2', 'v3', 'v4', 'v5')]))
    fallback = [{**row, 'main_policy': row['stream'] in main,
                 'run_role': 'delay sensitivity' if row['stream'].startswith('gfs6/') else
                             'no-load descriptive variant' if row['stream'].startswith('noload/') else 'nominal'}
                for row in tables['fallbacks']]
    unknown = [r for r in endpoints if any(r.get(name) not in (None, 'computable')
                                         for name in ('difference_status', 'ratio_status', 'raw_left_status', 'raw_right_status', 'status'))]
    diagnostic_unknown = [r for rows in diagnostics.values() for r in rows if r.get('status') not in (None, 'computable')]
    unavailable_scores = [value for name in ('summaries', 'local_hour_summaries')
                          for row in tables[name] if (value := _score_unavailable(name, row)) is not None]
    coverage = _breakdown_support(tables, diagnostics)
    empty_breakdowns = _empty_breakdowns(tables, diagnostics)
    released = scores.get('released_reference', {})
    released_rows = _catalog('released-v1/daily', released.get('daily', []))
    released_differences = _catalog('released-v1/key-differences', released.get('key_differences', []))
    sources['released_reference'] = {'daily_sha256': digest(released_rows), 'key_differences_sha256': digest(released_differences),
                                    'metadata': _plain(released.get('metadata', {}))}
    references = [r for r in primary if r.get('stream') == 'nominal/D2']
    missing_reference = []
    if not references:
        missing_reference.append('nominal/D2')
    if not released_rows:
        missing_reference.append('released-v1 original holdout')
    absolute = []
    for stream in streams:
        own = [r for r in tables['summaries'] if r.get('stream') == stream and r.get('mask') == 'all_eligible'
               and r.get('metric') == 'MAE']
        ordinary = [r for r in own if r.get('group') == 'year' and r.get('value') in (2023, 2024, 2025)]
        values = [r['raw_mean_daily_loss'] for r in ordinary if _number(r.get('raw_mean_daily_loss'))]
        stress = [r for r in own if r.get('group') == 'stress_2021_2022' and r.get('value') is True]
        partial = [r for r in own if r.get('group') == 'year' and r.get('value') in (2020, 2026)]
        absolute.append({'stream': stream, 'ordinary_complete_year_range': [min(values), max(values)] if len(values) == 3 else None,
                         'range_basis': 'complete non-stress calendar years 2023–2025; descriptive absolute context',
                         'ordinary_source_rows': [r['row_id'] for r in ordinary], 'stress_rows': stress, 'partial_year_rows': partial})
    limitations = ['Development comparisons do not establish performance on unused data, product qualification or live eligibility.',
                   'The bundled generation changes do not isolate a causal contribution from an individual feature.',
                   'A null or unavailable comparison does not establish equivalence; fallback losses and missing pairs stay disclosed.',
                   'No CP-28 arm is executed here. Business value, trading profit and live reliability are not evaluated by this audit.']
    limitations.extend(f'{name} reference is unavailable: no stored summary rows were supplied.' for name in missing_reference)
    absent_masks = [mask for mask in MASKS if mask not in {r.get('mask') for r in primary}]
    limitations.extend(f'Registered mask {mask} has no primary summary rows.' for mask in absent_masks)
    cp28 = ('The frozen CP-28 protocol stays unchanged. Its specific A4 rule governs the historical wording discrepancy: '
            'before 364 complete released native learning days, A3=w0 and A4=A3; from 364 through 727, A4 uses learned A3; '
            'at 728 the registered block-fit and shrink rule applies. These learning gates are separate from H history. '
            'CP-28 requires the Owner confirmation after CP-26 lands.')
    fingerprint = digest({'protocol': protocol_identity, 'sources': sources, 'resources': _plain(resource_summary)})
    population_id = digest({'population_evidence': sources['population_evidence'],
                            'calendar': ['2020-12-29', '2026-09-06'], 'masks': MASKS, 'aggregation': 'equal delivery-date weight'})
    report = {'schema': 'cp26-report-v1', 'protocol': _plain(protocol_identity), 'evidence_class': EVIDENCE_CLASS,
              'candidate_fingerprint': fingerprint, 'population_id': population_id, 'sources': sources, 'primary': primary,
              'score_tables': tables, 'transitions': transitions, 'load_bound': load,
              'gfs6': {'status': delay_status, 'fragile': True if fragile else False if len(finite_delay) == 2 else None,
                       'threshold_ratio': 1.02, 'comparisons': delay},
              'fallbacks': fallback, 'main_fallback_flags': [r['row_id'] for r in fallback if r.get('exceeds_one_percent') and r['main_policy']],
              'all_fallback_flags': [r['row_id'] for r in fallback if r.get('exceeds_one_percent')],
              'references': references, 'released_reference': {'daily_rows': released_rows,
                  'key_differences': released_differences, 'metadata': _plain(released.get('metadata', {})),
                  'evidence_class': 'historical released holdout reference; reused development window'},
              'missing_references': missing_reference, 'absolute_context': absolute,
              'breakdown_coverage': coverage, 'inference_endpoints': endpoints, 'diagnostic_tables': diagnostics,
              'registered_inventories': {'scores': score_inventory, 'calibration': calibration_inventory},
              'empty_breakdown_cells': empty_breakdowns, 'unavailable_score_cells': unavailable_scores,
              'unavailable_inference': unknown, 'unavailable_diagnostics': diagnostic_unknown,
              'resources': _plain(resource_summary), 'limitations': limitations, 'cp28_custody': cp28,
              'adoption_decision': None, 'integration_verdict': None, 'publication_status': 'local draft'}
    packet, export = _packet(report, streams)
    report['content_sha256'] = digest(report)
    packet['content_sha256'], export['content_sha256'] = digest(packet), digest(export)
    _save(output / 'report.json', report)
    (output / 'report.md').write_text(_render_report(report), encoding='utf-8')
    _save(output / 'publication-packet.json', packet)
    (output / 'publication-packet.md').write_text(_render_packet(packet), encoding='utf-8')
    _save(output / 'MLflow-export.json', export)
    return {'report': report, 'publication_packet': packet, 'mlflow_export': export,
            'paths': {name: str(output / name) for name in ('report.md', 'report.json', 'publication-packet.md',
                                                         'publication-packet.json', 'MLflow-export.json')}}


def _packet(report, streams):
    registry = []
    runs = []
    for stream in streams:
        route, policy = stream.split('/', 1)
        kind = 'reference' if policy in ('B0', 'v1-R', 'D2', 'released-v1') else 'control' if policy == 'v5' else 'generation'
        if route == 'noload':
            kind = 'branch'
        registry.append({'id': stream, 'name': f'{route} · {policy} daily re-baseline', 'subtitle': 'Daily development control audit',
                         'short': policy, 'kind': kind, 'codes': [stream], 'statuses': [],
                         'status_reason': 'CP-26 records no new adoption or product lifecycle decision',
                         'comparator': 'nominal/B0', 'population': report['population_id'],
                         'evidence_class': EVIDENCE_CLASS, 'plan': 'capstone_v21.md §24.7',
                         'rules': 'descriptive control audit', 'sources': [r['row_id'] for r in report['primary'] if r['stream'] == stream],
                         'claim_map': 'publication-packet.json#claim_map', 'run_keys': [f'cp-26/{stream}/{mask}' for mask in MASKS],
                         'style': None, 'style_reason': 'No new generation is adopted; public styling requires registry integration',
                         'anchor': f"cp26-{route}-{policy.lower().replace(' ', '-')}", 'checkpoint': 'CP-26',
                         'after': None, 'question': 'What does the daily re-baseline show?', 'informed': 'Owner CP-28 decision',
                         'predecessor': next((t['predecessor'] for t in report['transitions'] if t['successor'] == policy), None)})
        for mask in MASKS:
            rows = [r for r in report['primary'] if r['stream'] == stream and r['mask'] == mask]
            metrics = {f"{r['metric']}/raw_mean_daily_loss": r.get('raw_mean_daily_loss') for r in rows}
            metrics.update({f"{r['metric']}/B0_ratio": r.get('B0_ratio') for r in rows})
            runs.append({'key': f'cp-26/{stream}/{mask}', 'parent': 'cp-26', 'name': f'{route} · {policy} · {mask}',
                         'description': 'Daily development post-selection control audit; no adoption decision.',
                         'tags': {'checkpoint': 'CP-26', 'route': route, 'policy': policy, 'mask': mask,
                                  'evidence_class': EVIDENCE_CLASS, 'protocol_sha256': digest(report['protocol']),
                                  'population_id': report['population_id'],
                                  'candidate_fingerprint': report['candidate_fingerprint']},
                         'metrics': metrics, 'metric_source_rows': [r['row_id'] for r in rows]})
    pending = {'status': 'not performed', 'reason': 'This is a local draft; Owner publication and public verification are outside CP-26 execution'}
    surfaces = {name: dict(pending) for name in ('GitHub source/README', 'GitHub Pages', 'Public MLflow',
                                               'Hugging Face Space card', 'Hugging Face direct demo')}
    claim_map = [{'id': f"control-{t['predecessor']}-{t['successor']}/{row['mask']}/{row['metric']}",
                  'text': _comparison_text(row), 'metric': row['metric'], 'evidence_class': EVIDENCE_CLASS,
                  'source_rows': row['source_rows'], 'formula': '100 × (stored paired ratio − 1)',
                  'withheld': ['unseen-data performance', 'product qualification', 'causal feature benefit', 'economic value', 'live eligibility']}
                 for t in report['transitions'] for row in t['comparisons']]
    claim_map.extend({'id': f"{name}/{row['metric']}", 'text': _comparison_text(row),
                      'source_rows': row['source_rows'], 'evidence_class': EVIDENCE_CLASS,
                      'formula': '100 × (stored paired ratio − 1)'}
                     for name, rows in (('load-bound', report['load_bound']), ('gfs6', report['gfs6']['comparisons'])) for row in rows)
    claim_map.extend({'id': f"fallback/{row['stream']}",
                      'text': f"{row['fallback_dates']} B0 fallback dates and {row['failed_dates']} missing dates over {row['eligible_dates']} eligible dates.",
                      'source_rows': [row['row_id']], 'evidence_class': EVIDENCE_CLASS}
                     for row in report['fallbacks'])
    packet = {'schema': 'cp26-local-publication-packet-v1', 'status': 'DRAFT_LOCAL',
              'identity': {'checkpoint': 'CP-26', 'protocol': report['protocol'], 'sources': report['sources']['documents'],
                           'candidate_fingerprint': report['candidate_fingerprint'], 'final_candidate_sha': None,
                           'final_candidate_reason': 'Assigned after outputs are frozen; no self-referential SHA',
                           'evidence_tag': None, 'evidence_tag_reason': 'Owner disposition has not occurred',
                           'review_verdict': None, 'landing_record': None},
              'registry_entries': registry, 'claim_map': claim_map,
              'headline_quantities': {'pre_specified_verdict': None, 'eligible_N': None,
                                     'reason': 'CP-26 has no adoption rule or same-rule decision census',
                                     'comparisons': report['transitions'], 'absolute_context': report['absolute_context']},
              'descriptive_support': {'registered_inventories': report['registered_inventories'],
                                      'breakdown_coverage': report['breakdown_coverage'],
                                      'empty_breakdown_cells': report['empty_breakdown_cells'],
                                      'unavailable_score_cells': [{key: row.get(key) for key in
                                          ('table', 'row_id', 'status', 'reason', 'ratio_status',
                                           'ratio_undefined_reason', 'unavailable_parts')}
                                          for row in report['unavailable_score_cells']]},
              'slots': {'question': 'How do the landed policies compare under one daily development protocol?',
                        'change': 'Daily refits, registered input vintages, the corrected issue time and route-specific DDNN-2 searches.',
                        'chart': 'Paired differences and ratio intervals, with all four masks and unavailable endpoints disclosed.',
                        'reading': 'The report preserves favorable, unfavorable and unavailable control comparisons.',
                        'limitations': report['limitations'], 'decision': 'No adoption decision follows from CP-26.',
                        'evidence': ['report.json', 'report.md', 'MLflow-export.json'],
                        'details': ['method', 'period consistency', 'stress period', 'coverage and width', 'protocol and review']},
              'adopted_transitions': {'status': 'inapplicable', 'reason': 'No generation is newly adopted at CP-26',
                                      'historical_control_audits': report['transitions']},
              'product_documentation': [{'subject': name, 'disposition': 'inapplicable',
                                         'reason': 'CP-26 does not replace the released product', 'evidence': []}
                                        for name in ('Data', 'Regimes', 'Inputs', 'Seasonal rationale', 'Validation', 'Results',
                                                     'Attribution', 'Importance and sensitivity', 'Failures', 'Reliability', 'Forecast', 'Limitations', 'Running it')],
              'chart_routes': [{'comparison': f"{t['predecessor']}-to-{t['successor']}",
                               'heading': f"{t['predecessor']} → {t['successor']}: {t['change']}",
                               'route_label': f"Inspect the {t['predecessor']} to {t['successor']} daily control audit",
                               'status': 'local report route; public routes unverified'} for t in report['transitions']],
              'final_product_trigger': {'status': 'unmet', 'reason': 'No final product is selected, frozen or deployed by this checkpoint'},
              'cp28_custody': report['cp28_custody'], 'mlflow_export': 'MLflow-export.json',
              'checks': {'source_row_binding': 'implemented by generator; independent verification pending',
                         'editorial_review': 'pending', 'fresh_reader': 'pending', 'integration_review': 'pending',
                         'registry_integration': 'pending publication task', 'export_contract': 'pending supervised tests'},
              'publication_completion_receipt': {'status': 'INCOMPLETE', 'surfaces': surfaces,
                                                 'publication_to_mlflow_mapping': None,
                                                 'authority': 'No external action authorized by this packet'},
              'expected_routes': [r['key'] for r in runs], 'resources': report['resources']}
    export = {'schema': 'cp26-local-mlflow-export-v1', 'destination': 'local-only', 'remote_writes': 0,
              'experiment': 'cp-26-rebaseline-v21-r14', 'protocol': report['protocol'],
              'candidate_fingerprint': report['candidate_fingerprint'], 'evidence_class': EVIDENCE_CLASS,
              'registry_entries': registry, 'parents': [{'key': 'cp-26', 'name': 'Daily development re-baseline'}],
              'runs': runs, 'expected_routes': [r['key'] for r in runs],
              'published_export_diff': {'status': 'not performed', 'reason': 'No remote/publication action in this local draft'}}
    return packet, export


def _render_packet(packet):
    lines = ['# Local draft publication packet', '',
             'DRAFT_LOCAL. This packet grants no publication, MLflow upload or product activation authority.', '',
             f"Evidence fingerprint: `{packet['identity']['candidate_fingerprint']}`.", '',
             '## Identity and registry', '',
             'The final candidate, evidence tag, review verdict and landing record await their authorized lifecycle steps. '
             'The JSON packet binds document hashes, source rows and named local MLflow routes.', '',
             '| Identity | Kind | Evidence class | Local run keys |', '|---|---|---|---|']
    for row in packet['registry_entries']:
        lines.append(f"| {row['name']} | {row['kind']} | {row['evidence_class']} | {', '.join(row['run_keys'])} |")
    lines.extend(['', '## Claim map and chapter slots', '', packet['slots']['question'], '', packet['slots']['change'], '',
                  packet['slots']['reading'], '', packet['slots']['decision'], ''])
    lines.extend(f"- {text}" for text in packet['slots']['limitations'])
    lines.extend(['', 'No pre-specified adoption verdict or eligible same-rule policy count applies. '
                  'Derived comparison values and their source rows are in headline_quantities in the JSON packet.', '',
                  f"Score cells with unavailable raw losses or B0 ratios: {len(packet['descriptive_support']['unavailable_score_cells'])}. "
                  f"Zero-support breakdown groups: {len(packet['descriptive_support']['empty_breakdown_cells'])}. "
                  'Per-mask support counts, unavailable statuses and exact source IDs are retained in descriptive_support '
                  'in publication-packet.json and displayed in report.md.', '',
                  '## Product documentation and future lifecycle', '',
                  'Product replacement, final-version designation and live-operation fields are inapplicable: '
                  'their trigger has not occurred. The complete subject map is retained in the JSON packet.', '',
                  packet['cp28_custody'], '', '## Checks and publication completion', '',
                  'Editorial, fresh-reader and independent review remain pending. Public source, Pages, MLflow, '
                  'Space-card and demo identities have no verified receipt. Completion is INCOMPLETE.', '',
                  'The deterministic MLflow-export.json names expected routes but advertises no verified public URL. '
                  'The publication-to-MLflow mapping and published-export diff await an authorized publication task.', ''])
    return '\n'.join(lines)
