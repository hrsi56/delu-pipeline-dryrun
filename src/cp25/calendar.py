"""The new partition specification, the eligibility calendar and the record masks
(capstone v21-r13 §24.6 items 6–8; §24.2's boundary).

**The eligibility calendar** lists every development delivery day 2019-01-01..2026-09-06 with its
eligibility and, for every exclusion, its reason. Exclusions follow only target completeness and
the inputs whose absence makes the as-landed v5 fail: prices and the load forecast. Weather gaps
follow §15.3 and exclude no day. No model, candidate or score enters it.

The as-landed verdict is the landed code's own: `cp15.data.prepare` (src/cp15/data.py:93–127),
through which every v2–v5 member (A1_w, B2_w, L-N, L-R and DDNN-2's targets) takes its eligible
rows, marks a row feature-valid only when all of v1's base features are non-null (line 104) and
eligible only when its target is also finite (line 126). A day without one eligible row produces
no v5 forecast. Independent reasons, computed here from the preserved snapshot, must explain
exactly those days, or the build refuses.

**The record masks** are fixed from existing records before any CP-26 score. A day counts as
scored if a forecast's loss on it entered a reported result, a selection or an admission
decision. The never-scored mask subtracts §24.6's four named sources: folds 1–5's evaluation days;
v1's holdout; CP-16's training-only admission slices; CP-24's gate days and search batches. Days
used only to fit or calibrate are listed apart and are not subtracted.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import guard

DEV_FIRST, DEV_LAST = guard.FLOOR, guard.BOUNDARY
RAW_COLUMNS = ['timestamp_utc', 'delivery_date', 'price_eur_mwh', 'load_forecast_mw']
SOURCES = {
    'snapshot': 'data/snapshot.parquet',
    'partitions': 'data/partitions.json',
    'cp15_protocol': 'reports/cp15/protocol.json',
    'cp16_input_manifest': 'reports/v2-causal/input-manifest.json',
    'cp24_search_ledger': 'reports/ddnn2/rounds/round-1/search-ledger.json',
    'cp24_gate': 'reports/ddnn2/rounds/round-1/gate.json',
    'cp24_protocol': 'reports/ddnn2/attempt-1/protocol.json',
    'cp23_selection': 'reports/distribution-challenger/selection.json',
}
# §23.6's gate windows G_f = [D0_f - 56, D0_f), as the anchor lists them.
CP24_GATES = {'fold_1': ('2020-03-30', '2020-05-24'), 'fold_2': ('2020-12-29', '2021-02-22'),
              'fold_3': ('2022-03-30', '2022-05-24'), 'fold_4': ('2025-01-25', '2025-03-21'),
              'fold_5': ('2025-10-07', '2025-12-01')}
# §24.7's windows, as references for the mask counts only (CP-25 plans nothing of CP-26).
ANCHOR_REFERENCE = {'cp26_warm_up': ('2020-01-01', '2020-12-28'), 'cp26_scored_calendar': ('2020-12-29', '2026-09-06'),
                    'newly_opened': ('2026-04-08', '2026-09-06'), 'fresh_test_4_7T_closed': ('2026-09-07', '2026-12-05')}


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def days(first: date = DEV_FIRST, last: date = DEV_LAST) -> list[date]:
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def span(a: str, b: str) -> list[date]:
    return days(date.fromisoformat(a), date.fromisoformat(b))


def expected_hours(day: date) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(day, tz='Europe/Berlin'), pd.Timestamp(day + timedelta(days=1), tz='Europe/Berlin'),
                         freq='h', inclusive='left').tz_convert('UTC')


# ------------------------------------------------------------------------------------ data

def load_snapshot(root: Path) -> pd.DataFrame:
    """The preserved snapshot's price and load columns, development days only, behind the guard."""
    frame = pd.read_parquet(Path(root) / SOURCES['snapshot'], columns=RAW_COLUMNS,
                            filters=[('delivery_date', '>=', DEV_FIRST), ('delivery_date', '<=', DEV_LAST)])
    guard.admit_rows(pd.DatetimeIndex(frame.timestamp_utc).to_pydatetime())
    return frame


def landed_eligibility(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-row feature validity and eligibility from the landed `cp15.data.prepare`, unchanged."""
    from cp15.data import prepare
    p = json.loads((Path(root) / SOURCES['cp15_protocol']).read_text())
    data = prepare(frame, p, None)
    return pd.DataFrame({'timestamp_utc': data.index, 'delivery_date': pd.to_datetime(data.dates).date,
                         'feature_valid': data.feature_valid, 'eligible': data.eligible})


def price_history_complete(frame: pd.DataFrame) -> dict[date, bool]:
    """The price inputs of the as-landed members: the 720 canonical hours before D's local midnight
    (v1's rolling statistics, which contain the D-1, D-2, D-3 and D-7 curves and the 168-hour scale)."""
    price = pd.Series(frame.price_eur_mwh.to_numpy(float), index=pd.DatetimeIndex(frame.timestamp_utc).tz_convert('UTC'))
    out = {}
    for d in days():
        boundary = pd.Timestamp(d, tz='Europe/Berlin').tz_convert('UTC')
        window = price.reindex(pd.date_range(end=boundary - pd.Timedelta(hours=1), periods=720, freq='h'))
        out[d] = bool(np.isfinite(window.to_numpy()).all())
    return out


def eligibility_calendar(root: Path) -> pd.DataFrame:
    frame = load_snapshot(root)
    rows = landed_eligibility(root, frame)
    frame = frame.assign(ts=pd.DatetimeIndex(frame.timestamp_utc).tz_convert('UTC'))
    history = price_history_complete(frame)
    by_day = frame.groupby('delivery_date')
    landed = rows.groupby('delivery_date')
    out = []
    for d in days():
        hours = expected_hours(d)
        g = by_day.get_group(d) if d in by_day.groups else frame.iloc[0:0]
        present = pd.DatetimeIndex(g.ts).equals(hours)
        price_ok = bool(np.isfinite(g.price_eur_mwh.to_numpy(float)).all()) and present
        load_null = int(g.load_forecast_mw.isna().sum()) + (len(hours) - len(g))
        lg = landed.get_group(d) if d in landed.groups else rows.iloc[0:0]
        target_complete = present and price_ok
        reasons = []
        if not target_complete:
            reasons.append('target_incomplete')
        if not history[d]:
            reasons.append('price_inputs_absent:720h_history_before_2019_floor' if d < date(2019, 2, 1)
                           else 'price_inputs_absent')
        if load_null:
            reasons.append(f'load_forecast_absent:{load_null}_of_{len(hours)}_hours_null')
        n_eligible = int(lg.eligible.sum())
        out.append({'delivery_date': d, 'canonical_hours': len(hours), 'rows_present': len(g),
                    'target_complete': target_complete, 'price_history_complete': history[d],
                    'load_forecast_null_hours': load_null, 'landed_feature_valid_hours': int(lg.feature_valid.sum()),
                    'landed_eligible_hours': n_eligible, 'eligible': not reasons,
                    'exclusion_reason': ';'.join(reasons)})
    cal = pd.DataFrame(out)
    # The independent reasons must explain exactly the days the landed code leaves without an eligible row.
    landed_out = set(cal.delivery_date[cal.landed_eligible_hours == 0])
    reasoned_out = set(cal.delivery_date[~cal.eligible])
    if landed_out != reasoned_out:
        raise ValueError(f'calendar disagrees with the landed eligibility: landed-only {sorted(landed_out - reasoned_out)[:5]}, '
                         f'reasoned-only {sorted(reasoned_out - landed_out)[:5]}')
    partial = cal[cal.eligible & (cal.landed_eligible_hours < cal.canonical_hours)]
    cal['ineligible_hours_on_eligible_day'] = 0
    cal.loc[partial.index, 'ineligible_hours_on_eligible_day'] = (partial.canonical_hours - partial.landed_eligible_hours)
    return cal


def ineligible_hour_reasons(root: Path, cal: pd.DataFrame) -> pd.DataFrame:
    """Hours the landed code leaves out on otherwise eligible days, with the null base features."""
    from delu_forecast.features import build_base_features
    frame = load_snapshot(root)
    base = build_base_features(frame)
    rows = landed_eligibility(root, frame).set_index('timestamp_utc')
    keep = set(cal.delivery_date[cal.eligible & (cal.ineligible_hours_on_eligible_day > 0)])
    out = []
    for ts, r in rows.iterrows():
        if r.delivery_date in keep and not r.eligible:
            nulls = [c for c in base.columns if pd.isna(base.at[ts, c])]
            out.append({'delivery_date': r.delivery_date, 'timestamp_utc': ts,
                        'local_hour': ts.tz_convert('Europe/Berlin').hour, 'null_base_features': ','.join(nulls)})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------ masks

def scored_sources(root: Path) -> dict[str, dict[date, str]]:
    """§24.6 item 8's four subtracted sources, each day labelled with its record."""
    root = Path(root)
    parts = json.loads((root / SOURCES['partitions']).read_text())
    out: dict[str, dict[date, str]] = {'folds_1_5_evaluation': {}, 'v1_holdout': {}, 'cp16_admission_slices': {},
                                       'cp24_gate_days': {}, 'cp24_search_batches': {}}
    for f in parts['development_folds']:
        for d in span(f['evaluation']['start'], f['evaluation']['end']):
            out['folds_1_5_evaluation'][d] = f['name']
    h = parts['tail_partitions']['holdout']
    for d in span(h['start'], h['end']):
        out['v1_holdout'][d] = 'holdout'
    m = json.loads((root / SOURCES['cp16_input_manifest']).read_text())
    for f in m['folds']:
        if len(f['admission']) != 7:
            raise ValueError('CP-16 admission slice must hold seven dates per fold')
        for a in f['admission']:
            out['cp16_admission_slices'][date.fromisoformat(a['day'])] = f['fold']
    for fold, (a, b) in CP24_GATES.items():
        for d in span(a, b):
            out['cp24_gate_days'][d] = fold
    s = json.loads((root / SOURCES['cp24_search_ledger']).read_text())
    for fold, x in s['folds'].items():
        if len(x['batches']) != x['B']:
            raise ValueError(f'{fold}: batch count disagrees with B')
        for k, (a, b) in enumerate(x['batches']):
            for d in span(a, b):
                out['cp24_search_batches'][d] = f'{fold}:batch_{k}'
    return out


def fit_or_calibrate_sources(root: Path) -> dict[str, dict[date, str]]:
    """Days used only to fit or calibrate, listed apart (§24.6 item 8): v1's CQR calibration windows
    and the genuine warm-up days whose issued errors only filled interval buffers."""
    root = Path(root)
    parts = json.loads((root / SOURCES['partitions']).read_text())
    out: dict[str, dict[date, str]] = {'v1_fold_calibration': {}, 'v1_final_calibration': {}, 'genuine_warm_up_buffers': {}}
    for f in parts['development_folds']:
        for d in span(f['calibration']['start'], f['calibration']['end']):
            out['v1_fold_calibration'][d] = f['name']
    c = parts['tail_partitions']['final_calibration']
    for d in span(c['start'], c['end']):
        out['v1_final_calibration'][d] = 'final_calibration'
    m = json.loads((root / SOURCES['cp16_input_manifest']).read_text())
    for f in m['folds']:
        first_eval = date.fromisoformat(f['evaluation_start'])
        for d in days(date.fromisoformat(f['warmup_start']), first_eval - timedelta(days=1)):
            out['genuine_warm_up_buffers'][d] = f['fold']
    return out


def cp23_selection_windows(root: Path) -> dict[str, list[str]]:
    """CP-23's per-fold training-only configuration selection (holdout and early-stopping days)."""
    s = json.loads((Path(root) / SOURCES['cp23_selection']).read_text())
    return {fold: [x['rows']['stop_first'], x['rows']['holdout_last']] for fold, x in s['folds'].items()}


def masks(root: Path, cal: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    scored = scored_sources(root)
    fitcal = fit_or_calibrate_sources(root)
    rows = []
    for d in days():
        hit = {k: v[d] for k, v in scored.items() if d in v}
        fc = {k: v[d] for k, v in fitcal.items() if d in v}
        fold = scored['folds_1_5_evaluation'].get(d, '')
        rows.append({'delivery_date': d, 'scored': bool(hit), 'scored_sources': ';'.join(f'{k}={v}' for k, v in hit.items()),
                     'never_scored': not hit, 'old_fold_2_5': fold in ('fold_2', 'fold_3', 'fold_4', 'fold_5'),
                     'fold_1_evaluation': fold == 'fold_1',
                     'fit_or_calibrate_only': bool(fc) and not hit,
                     'fit_or_calibrate_sources': ';'.join(f'{k}={v}' for k, v in fc.items())})
    mask = pd.DataFrame(rows).merge(cal[['delivery_date', 'eligible']], on='delivery_date', how='left')
    # CP-23's training-only selection days also meet the "scored" definition; record that each lies
    # inside CP-24's gate days, so the four named subtractions already cover them.
    cp23 = {}
    for fold, (a, b) in cp23_selection_windows(root).items():
        window = span(a, b)
        cp23[fold] = {'first': a, 'last': b, 'days': len(window),
                      'inside_cp24_gate_days': all(d in scored['cp24_gate_days'] for d in window)}
    if not all(v['inside_cp24_gate_days'] for v in cp23.values()):
        raise ValueError('a CP-23 selection day lies outside the subtracted sources; the mask needs an Owner ruling')
    counts = {k: len(v) for k, v in scored.items()}
    summary = {'definition': ('a day counts as scored if a forecast\'s loss on it entered a reported result, a selection '
                              'or an admission decision (capstone v21-r13 §24.6 item 8)'),
               'subtracted_sources_days': counts,
               'scored_union_days': int(mask.scored.sum()),
               'never_scored_days': int(mask.never_scored.sum()),
               'never_scored_eligible_days': int((mask.never_scored & mask.eligible).sum()),
               'old_fold_2_5_days': int(mask.old_fold_2_5.sum()),
               'old_fold_2_5_eligible_days': int((mask.old_fold_2_5 & mask.eligible).sum()),
               'fold_1_evaluation_days': int(mask.fold_1_evaluation.sum()),
               'fit_or_calibrate_only_days': int(mask.fit_or_calibrate_only.sum()),
               'fit_or_calibrate_sources_days': {k: len(v) for k, v in fitcal.items()},
               'cp23_selection_windows': cp23,
               'within_anchor_reference_windows': {}}
    for name, (a, b) in ANCHOR_REFERENCE.items():
        if name == 'fresh_test_4_7T_closed':
            continue
        sel = (mask.delivery_date >= date.fromisoformat(a)) & (mask.delivery_date <= date.fromisoformat(b))
        summary['within_anchor_reference_windows'][name] = {
            'window': [a, b], 'days': int(sel.sum()), 'eligible': int((sel & mask.eligible).sum()),
            'never_scored_eligible': int((sel & mask.eligible & mask.never_scored).sum()),
            'old_fold_2_5_eligible': int((sel & mask.eligible & mask.old_fold_2_5).sum())}
    return mask, summary


def partition_spec(root: Path, cal_sha: str, mask_sha: str) -> dict:
    root = Path(root)
    parts = json.loads((root / SOURCES['partitions']).read_text())
    return {
        'schema': 'cp25-partition-spec-v1',
        'anchor': 'capstone_v21.md v21-r13 §24.2 and §24.6 item 6 (SHA-256 173799d818387b9d82db16b9ccd7195afaebbea8e11259809e4a8b704bd525a8)',
        'calendar_timezone': 'Europe/Berlin (23/24/25-hour delivery days)',
        'input_floor': DEV_FIRST.isoformat(),
        'development': {'first_delivery_day': DEV_FIRST.isoformat(), 'last_delivery_day': DEV_LAST.isoformat(),
                        'days': len(days()), 'note': '2026-04-08..2026-09-06 joins development by v21-r13 (Owner, 2026-10-12)'},
        'boundary': {'rows': 'refused when the Europe/Berlin delivery day is after 2026-09-06',
                     'gfs': 'runs initialized at or after 2026-09-06 00 UTC refused; leads outside f021-f048 refused',
                     'code': 'src/cp25/guard.py', 'moves': 'only at the 4.7T gate (to 2026-12-05) and, for the live pipeline only, at CP-18'},
        'closed': {'fresh_test_4_7T': list(ANCHOR_REFERENCE['fresh_test_4_7T_closed']),
                   'status': 'closed until 4.7T opens (§24.3, §24.9); never extended after a result'},
        'issue_time': {'rule': 'D-1 11:00 Europe/Berlin; an input is usable only if published strictly before it',
                       'utc': '10:00 UTC when D-1 is on CET, 09:00 UTC on CEST; D-1\'s own offset at 11:00, including transition days',
                       'code': 'cp25.guard.issue_time'},
        'eligibility_calendar': {'file': 'reports/development-inputs/eligibility-calendar.csv', 'sha256': cal_sha},
        'record_masks': {'file': 'reports/development-inputs/record-masks.csv', 'sha256': mask_sha},
        'historical_partitions': {
            'file': SOURCES['partitions'], 'sha256': sha(root / SOURCES['partitions']), 'unchanged': True,
            'role': "v1's historical record: folds 1-5, v1's calibration windows and holdout; never rewritten (§1, §24.2)",
            'eda_cutoff': parts['eda_cutoff'], 'snapshot_cutoff': parts['snapshot_cutoff'],
            'folds_evaluation': {f['name']: [f['evaluation']['start'], f['evaluation']['end']] for f in parts['development_folds']},
            'v1_final_calibration': [parts['tail_partitions']['final_calibration']['start'], parts['tail_partitions']['final_calibration']['end']],
            'v1_holdout': [parts['tail_partitions']['holdout']['start'], parts['tail_partitions']['holdout']['end']]},
        'anchor_reference_windows': {k: list(v) for k, v in ANCHOR_REFERENCE.items()},
        'snapshot': {'file': SOURCES['snapshot'], 'sha256': sha(root / SOURCES['snapshot'])},
    }
