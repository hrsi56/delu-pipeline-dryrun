"""Historical original-v21 preflight; not the v21-r1 execution rule.

Retained for attempt-1 helper fixtures. Reproduce the complete old audit from
its original candidate SHA and original protocol, preserved in attempt-1 evidence.
The current driver uses the owner-ratified capped expanding-history correction.

Audit exact calendar-window support without fitting or selecting a model.

The 728-day requirement cannot be replaced by the available 547 days. This
audit distinguishes absent leading archive support from ordinary inherited
row exclusions. It does not invent a minimum-row threshold after seeing scores.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from delu_forecast.features import build_feature_catalog
from delu_forecast.folds import load_partition_spec, window_mask
from delu_forecast.ingest import BERLIN

COLUMNS = ['timestamp_utc', 'delivery_date', 'price_eur_mwh', 'load_forecast_mw']


def canonical_hours(start: date, end_exclusive: date) -> int:
    start_utc = pd.Timestamp(start, tz=BERLIN).tz_convert('UTC')
    end_utc = pd.Timestamp(end_exclusive, tz=BERLIN).tz_convert('UTC')
    return int((end_utc - start_utc).total_seconds() // 3600)


def window_support(origin: date, history_days: int, archive_start: date) -> dict:
    if history_days not in (84, 728):
        raise ValueError('only the two ratified history lengths are admitted')
    start = origin - timedelta(days=history_days)
    missing_end = min(origin, archive_start)
    missing_days = max(0, (missing_end - start).days)
    return {
        'origin': origin.isoformat(),
        'history_days': history_days,
        'required_start': start.isoformat(),
        'required_end': (origin - timedelta(days=1)).isoformat(),
        'required_canonical_hours': canonical_hours(start, origin),
        'absent_leading_calendar_days': missing_days,
        'absent_leading_canonical_hours': canonical_hours(start, missing_end) if missing_days else 0,
        'archive_reaches_required_start': missing_days == 0,
    }


def load_admissible(root: Path) -> tuple[pd.DataFrame, object]:
    spec = load_partition_spec(root / 'data/partitions.json')
    # Predicate and projection are passed into the reader, before materializing
    # model inputs. Do not call inherited experiment.load_inputs (unfiltered).
    frame = pd.read_parquet(root / 'data/snapshot.parquet', columns=COLUMNS,
                            filters=[('delivery_date', '<=', spec.eda_cutoff)])
    if frame.empty or (frame.delivery_date > spec.eda_cutoff).any():
        raise ValueError('empty admissible input or reserved outcomes returned by reader')
    return frame, spec


def audit(root: Path) -> tuple[dict, pd.DataFrame]:
    protocol = json.loads((root / 'reports/cp15/protocol.json').read_text())
    for name, expected in protocol['input_sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'input hash mismatch: {name}')
    frame, spec = load_admissible(root)
    index = pd.DatetimeIndex(frame.timestamp_utc).tz_convert('UTC')
    days = pd.Index(frame.delivery_date)
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError('snapshot must have unique chronological canonical hours')
    features = build_feature_catalog(frame, 'base')
    # The date flags are used only to reconstruct inherited eligibility. No
    # fitting occurs and no crisis flag is admitted to a new predictor matrix.
    eligible = features.notna().all(axis=1).to_numpy() & np.isfinite(frame.price_eur_mwh.to_numpy())
    saved = pd.read_parquet(root / 'reports/cp2/development_predictions.parquet')
    saved = saved.loc[saved.arm.eq('base')]
    records, folds = [], []
    archive_start = min(days)
    for fold in spec.development_folds:
        mask = window_mask(days, fold.evaluation) & eligible
        reference = saved.loc[saved.fold.eq(fold.name)]
        if len(reference) != int(mask.sum()) or not np.array_equal(reference.delivery_date.to_numpy(), days[mask].to_numpy()) or not np.array_equal(reference.y_true.to_numpy(), frame.price_eur_mwh.to_numpy()[mask]):
            raise ValueError(f'original evaluation lineage mismatch: {fold.name}')
        # This is an optimistic bound, not a claim of completed warm-up. At D,
        # the latest allowed error day is D-2. 28 consecutive complete days
        # therefore begin at D-29. Missing eligible hours can move this earlier.
        warm_start = fold.evaluation.start - timedelta(days=29)
        for origin in pd.date_range(warm_start, fold.evaluation.end, freq='D').date:
            for history in (728, 84):
                row = window_support(origin, history, archive_start)
                lower = origin - timedelta(days=history)
                train_mask = (days >= lower) & (days < origin)
                row.update(fold=fold.name,
                           phase='evaluation' if origin >= fold.evaluation.start else 'optimistic_warmup',
                           available_snapshot_rows=int(train_mask.sum()),
                           inherited_eligible_training_rows=int((train_mask & eligible).sum()))
                records.append(row)
        ev_days = days[mask]
        peak = mask & (days >= date(2022, 8, 15)) & (days <= date(2022, 8, 31))
        folds.append({'fold': fold.name, 'evaluation_start': str(fold.evaluation.start),
                      'evaluation_end': str(fold.evaluation.end),
                      'original_eligible_hours': int(mask.sum()),
                      'represented_delivery_days': len(ev_days.unique()),
                      'inherited_excluded_hours': int(window_mask(days, fold.evaluation).sum() - mask.sum()),
                      'matched_peak_hours': int(peak.sum()),
                      'optimistic_first_warmup_origin': str(warm_start),
                      'original_saved_dates_and_targets_match': True})
    windows = pd.DataFrame(records)
    bad = windows.loc[~windows.archive_reaches_required_start]
    summary = {
        'engineering_status': 'BLOCKED' if len(bad) else 'PREFLIGHT_ONLY',
        'product_feasibility': 'NOT_DEMONSTRATED',
        'product_criteria_status': 'all six unassessed: comparison not run; no measured product failures claimed',
        'best_observed_policy': None, 'qualified_policy': None,
        'model_fit_count': 0, 'comparison_prediction_count': 0,
        'archive_first_delivery_date': str(archive_start),
        'admissible_last_delivery_date': str(max(days)),
        'admissible_snapshot_rows': len(frame),
        'first_inherited_eligible_delivery_date': str(min(days[eligible])),
        'folds': folds,
        'origins_without_leading_archive_support_by_fold_phase_history':
            bad.groupby(['fold', 'phase', 'history_days']).size().reset_index(name='origin_count').to_dict('records'),
        'warmup_caveat': 'D-29 is an optimistic earliest issuance bound only; no warm-up forecast or error buffer has been produced.',
        'interpretation': 'Do not silently substitute a shorter available archive for the requested window. Earlier admissible inputs or an owner-ratified clarification/change are needed before completing the comparison protocol.',
        'protocol_sha256': hashlib.sha256((root / 'reports/cp15/protocol.json').read_bytes()).hexdigest(),
    }
    return summary, windows
