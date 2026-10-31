"""The tier register, the missing-input rules and the margin table (capstone v21-r13 §24.5, §24.6
items 4–5), against the issue time D−1 11:00 Europe/Berlin, strictly before (§24.4).

Three categories (§24.5): published inputs, each with its tier and evidence; deterministic calendar
features known in advance (admissible); hindsight labels (inadmissible). An input whose originality
cannot be established is treated as tier 3. Every documentary source cited here is recorded, with
its retrieval date, in `DOCUMENTS`; nothing was fetched from them but their text.

The margin table: for each tier-2 group and each shift (+1, +6, +24 hours), the delivery days on
which the shifted availability is no longer strictly before the issue time. A shift that affects
no day is "none"; every day, "structural"; a non-empty proper subset, "partial" (§24.5: a delay
run in CP-26). The load forecast is tier 3, but its row is computed as well, for the Owner's ruling.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path

import pandas as pd

from . import guard, runs
from .calendar import days

UTC = timezone.utc
SHIFTS = (1, 6, 24)

DOCUMENTS = {
    'REG543-6': {'title': 'Commission Regulation (EU) No 543/2013, Article 6 "Information on total load", paragraphs 1(b) and 2(b), as adopted',
                 'url': 'https://www.legislation.gov.uk/eur/2013/543/article/6/adopted', 'read_utc': '2026-10-12',
                 'says': ('a day-ahead forecast of the total load per market time unit is published "no later than two hours before '
                          'the gate closure of the day-ahead market in the bidding zone" and updated "when significant changes occur"')},
    'REG543-12': {'title': 'Commission Regulation (EU) No 543/2013, Article 12, paragraphs 1(d) and 2(d), as adopted',
                  'url': 'https://www.legislation.gov.uk/eur/2013/543/article/12/adopted', 'read_utc': '2026-10-12',
                  'says': ('the day-ahead prices in each bidding zone for every market time unit are "published no later than one hour '
                           'after gate closure"; the provision carries no update or revision wording')},
    'CACM-47': {'title': 'Commission Regulation (EU) 2015/1222 (CACM), Article 47 "Operation of single day-ahead coupling", as adopted',
                'url': 'https://www.legislation.gov.uk/eur/2015/1222/article/47/adopted', 'read_utc': '2026-10-12',
                'says': ('day-ahead gate closure is "noon market time day-ahead" (47(2)); "Orders matched in single day-ahead coupling '
                         'shall be considered firm" (47(5))')},
    'CACM-39': {'title': 'CACM Article 39 "Inputs and results of the price coupling algorithm", paragraph 2, as adopted',
                'url': 'https://www.legislation.gov.uk/eur/2015/1222/article/39/adopted', 'read_utc': '2026-10-12',
                'says': 'the algorithm produces "a single clearing price for each bidding zone and market time unit in EUR/MWh" (39(2)(a))'},
    'CACM-48': {'title': 'CACM Article 48 "Delivery of results", as adopted', 'url': 'https://www.legislation.gov.uk/eur/2015/1222/article/48/adopted',
                'read_utc': '2026-10-12',
                'says': ('each TSO verifies the results against allocation constraints and validated capacity, and each NEMO against '
                         'the orders, before market participants are informed (48(2)-(4))')},
    'CACM-50': {'title': 'CACM Article 50 "Initiation of fallback procedures", as adopted', 'url': 'https://www.legislation.gov.uk/eur/2015/1222/article/50/adopted',
                'read_utc': '2026-10-12',
                'says': 'if results cannot be delivered by the deadline, the fallback procedures apply and a notice is published'},
    'SMARD-B1.3': {'title': 'SMARD.de Benutzerhandbuch, April 2026, §B 1.3 "Aktualisierung der Daten" (p. 7)',
                   'url': 'https://www.smard.de/resource/blob/220052/9d526adf4b948599da4a956dfae6dab9/smard-benutzerhandbuch-04-2026-data.pdf',
                   'read_utc': '2026-10-12', 'sha256': '331df7ccba31bf53d35e4d392481e33b5949881baaee998043ada0eacc0b88bd',
                   'says': ('all SMARD data are updated on new information, without a fixed revision process; series follow TSO updates on '
                            'the ENTSO-E platform; provisional data can later be replaced by quality-assured data')},
    'SMARD-D2.2.1': {'title': 'SMARD.de Benutzerhandbuch, April 2026, §D 2.2.1 "Prognostizierter Stromverbrauch: Gesamt" (p. 52)',
                     'url': 'same PDF', 'read_utc': '2026-10-12',
                     'says': ('TSOs forecast next-day total load under Art. 6(1)(b) Reg. 543/2013, sent to ENTSO-E at the latest two hours '
                              'before the day-ahead closure; an update of published data is required when values change by at least 10%')},
    'SMARD-D3.1': {'title': 'SMARD.de Benutzerhandbuch, April 2026, §D 3.1 "Großhandelspreise" (p. 54)', 'url': 'same PDF',
                   'read_utc': '2026-10-12',
                   'says': ('day-ahead prices under Art. 12(1)(d) Reg. 543/2013 are published at the latest one hour after the bid '
                            'closure; "Eine nachträgliche Aktualisierung der veröffentlichten Daten ist möglich"; DE-LU prices reach '
                            'ENTSO-E through a third-party provider for the German TSOs')},
    'DOSSIER-6': {'title': 'reports/weather-admission/dossier.md §6 and availability/gfs_ncep_production_status.csv',
                  'url': 'repository (SHA-256 0782677030db49aa7976343372a26225bc0154915f6e605d24e64495c0625881)', 'read_utc': '2026-10-12',
                  'says': ('dated NCEP production-status captures 2017-05..2026-09 put the 00 UTC GFS "48hr PRODUCTS" completion at '
                           '03:42-04:00 UTC (30-day running averages); reconstructed, not a per-day delivery proof')},
    'MOP-A69': {'title': 'reports/cp15/feasibility/structural_inputs.md (ENTSO-E MoP Ref2 pp.69-70)', 'url': 'repository', 'read_utc': '2026-10-12',
                'says': 'the A69 day-ahead wind/solar forecast is the forecast fixed at 18:00 Brussels time on D-1: after the gate'},
    'RECON': {'title': 'data/reconciliation_summary.json (snapshot pull, 2026-09-08)', 'url': 'repository', 'read_utc': '2026-10-12',
              'says': 'SMARD day-ahead prices equal ENTSO-E A44 on all 120 sampled overlapping hours (max abs diff 0.0 EUR/MWh)'},
}

# --------------------------------------------------------------------------- missing-input rules
# (file, line, text that must appear on that line) — checked by tests/cp25/test_tiers.py.
CITES = {
    'v1_lag_null': ('src/delu_forecast/features.py', 60, 'even a partially observed ambiguous hour is null'),
    'v1_window_nan': ('src/delu_forecast/features.py', 82, 'Reindex the expected hours'),
    'v1_load_day': ('src/delu_forecast/features.py', 143, 'Missing rows and present-but-null values both invalidate the entire day'),
    'v1_complete': ('src/delu_forecast/model.py', 159, 'complete = features.notna().all(axis=1)'),
    'v1_eligible': ('src/delu_forecast/experiment.py', 70, 'eligible = np.isfinite(target)'),
    'b0_naive': ('src/delu_forecast/baselines.py', 42, 'np.where(takes_previous_day, lag_1, lag_7)'),
    'cp15_valid': ('src/cp15/data.py', 104, 'valid=base.notna().all(axis=1)'),
    'cp15_eligible': ('src/cp15/data.py', 126, 'valid & np.isfinite(y)'),
    'cp15_impute': ('src/cp15/models.py', 17, 'fill=np.nanmedian(train,axis=0)'),
    'cp15_indicator': ('src/cp15/models.py', 19, '~np.isfinite(x)'),
    'cp15_no_rows': ('src/cp15/models.py', 47, 'if not len(rows):return'),
    'cp15_nonfinite': ('src/cp15/models.py', 98, 'nonfinite model forecast'),
    'cp16_buffer': ('src/cp16/residuals.py', 141, 'if len(self._buffer) != 28'),
    'wx_missing_day': ('src/cp20/weather.py', 133, 'def missing_day'),
    'wx_record_refusal': ('src/cp20/weather.py', 219, 'a training/forecast row has no frozen weather record'),
    'wx_hg_refusal': ('src/cp20/components.py', 98, 'lacks a frozen weather record'),
    'wx_lgbm_impute': ('src/cp21/lgbm.py', 97, 'fill = np.nanmedian(train, axis=0)'),
    'wx_lgbm_refusal': ('src/cp21/lgbm.py', 185, 'lacks a frozen weather record'),
    'l_no_rows': ('src/cp21/lgbm.py', 178, 'no forecast rows'),
    'v4_no_rows': ('src/cp24/v4members.py', 52, 'no forecast rows'),
    'd2_impute': ('src/cp24/design.py', 265, 'np.nanmedian(clipped, axis=0)'),
    'd2_stats': ('src/cp24/member.py', 78, 'a forecast day lacks its origin statistics'),
    'd2_window': ('src/cp24/design.py', 309, 'dd.mask[ix].any(axis=1)'),
    'd2_uncovered': ('src/cp24/execution.py', 151, 'lacks a frozen weather record'),
    'v5_finite': ('src/cp24/execution.py', 75, 'v5 needs five finite vectors of one shape'),
}


def cite(*keys) -> list[str]:
    return [f'{CITES[k][0]}:{CITES[k][1]}' for k in keys]


MISSING_INPUT_RULES = [
    {'input': 'prices (lags, rolling statistics, normalisation level/scale)', 'member': 'v1-R and the CP-15 base features of every v2-v5 member',
     'rule': ('a calendar-day/local-hour lag whose source hour is unavailable or ambiguous is null; a 168/720-hour window with any '
              'missing hour is NaN; a row with any null base feature gets no forecast and is not eligible'),
     'consequence': 'failure: no forecast for that row; a day without a feature-valid row fails (the day falls back, §24.7)',
     'cites': cite('v1_lag_null', 'v1_window_nan', 'v1_complete', 'v1_eligible', 'cp15_valid', 'cp15_eligible')},
    {'input': 'prices (D-1 / D-7 same local hour)', 'member': 'B0 (similar-day naive)',
     'rule': 'Tue-Fri take D-1, Mon/Sat/Sun take D-7 by calendar day and local hour; an unavailable or ambiguous source hour is null',
     'consequence': 'failure: B0 has no forecast for that hour', 'cites': cite('b0_naive', 'v1_lag_null')},
    {'input': 'price curves D-1, D-2, D-3, D-7 (LEAR cross-hour design)', 'member': 'A1, B2, A1_w, B2_w (LEAR)',
     'rule': 'a missing local-hour value takes the training-partition median with a missing indicator (fitted on training rows only)',
     'consequence': 'imputed inside the design; the row itself still needs every base feature (cp15/data.py:104)',
     'cites': cite('cp15_impute', 'cp15_indicator', 'cp15_valid')},
    {'input': 'prices: origin statistics (level/scale, median/MAD)', 'member': 'DDNN-2',
     'rule': 'a forecast day without finite origin statistics raises a training failure; training days without them leave the window',
     'consequence': 'failure', 'cites': cite('d2_stats', 'd2_window')},
    {'input': 'prices: released errors for the H layer (<= D-2)', 'member': 'v2-v5 interval layer',
     'rule': 'issuance needs 28 complete released days; fewer raises', 'consequence': 'failure (issuance)', 'cites': cite('cp16_buffer')},
    {'input': 'load forecast of D (load_forecast_mw, load_forecast_day_mean_mw)', 'member': 'v1-R and every v2-v5 member through the base features',
     'rule': ('any missing or null hour of D\'s load forecast makes the day mean NaN, so every row of D lacks a base feature, is not '
              'feature-valid and gets no forecast'),
     'consequence': 'failure of the whole delivery day (the day falls back, §24.7); it is an eligibility exclusion in the calendar',
     'cites': cite('v1_load_day', 'v1_complete', 'cp15_valid', 'cp15_no_rows', 'l_no_rows', 'v4_no_rows', 'v5_finite')},
    {'input': 'load forecast lags D-1, D-7 (load_d1, load_d7) and D (load_d0) inside LEAR / DDNN-2 designs', 'member': 'LEAR, DDNN-2',
     'rule': 'training-partition median imputation with missing indicators (LEAR); winsorise, then median imputation (DDNN-2)',
     'consequence': 'imputed; the row still needs D\'s own base features', 'cites': cite('cp15_impute', 'd2_impute')},
    {'input': 'GFS weather (three derived columns)', 'member': 'v3, v4, v5 (A1_w, B2_w, L-N, L-R, DDNN-2 gfs group)',
     'rule': ('§15.3: a confirmed missing, late or invalid weather vector sets all three columns missing and their indicators; '
              'training-only median imputation. A row with no frozen weather record at all is refused (BLOCKED, never imputed)'),
     'consequence': 'imputed, never a failure, once every development day has a weather record (CP-25 fills the gaps)',
     'cites': cite('wx_missing_day', 'wx_record_refusal', 'wx_hg_refusal', 'wx_lgbm_impute', 'wx_lgbm_refusal', 'd2_impute', 'd2_uncovered')},
    {'input': 'calendar features', 'member': 'all', 'rule': 'deterministic; computed for every hour, never missing',
     'consequence': 'none', 'cites': []},
]

# --------------------------------------------------------------------------- the register
PUBLISHED = [
    {'input': 'target price (SDAC DE-LU day-ahead clearing price; training targets and scoring truth)',
     'members': 'B0 (truth), v1-R, v2-v5 (training targets, truth)', 'tier': 2,
     'documented_publication': 'auction day D-1 for delivery D: no later than one hour after the 12:00 Europe/Berlin gate, i.e. 13:00 local',
     'against_issue_time': ("a training target of delivery day D-1 was published by D-2 13:00, at least 22 h before D-1 11:00; "
                            "day D's own price is published after the issue time and is the embargoed target (§24.4)"),
     'finality': ('documented: matched SDAC orders are firm (CACM 47(5)); the algorithm produces one clearing price per zone and MTU '
                  '(39(2)(a)), verified by TSOs and NEMOs before delivery (48(2)-(4)); the publication duty (Reg. 543/2013 Art. 12) has '
                  'no update wording, unlike the load forecast\'s (Art. 6). Caveat, disclosed: SMARD states that a later update of the '
                  'published price data is possible (SMARD-D3.1) and that its series follow TSO corrections (SMARD-B1.3); no historical '
                  'price vintage exists to show none occurred. The snapshot equals ENTSO-E A44 on 120 sampled hours (RECON).'),
     'evidence': ['CACM-47', 'CACM-39', 'CACM-48', 'CACM-50', 'REG543-12', 'SMARD-D3.1', 'SMARD-B1.3', 'RECON'],
     'basis': 'tier 2: preserved original values (SDAC results are firm), availability reconstructed from the documented deadline'},
    {'input': "price lags and price statistics (B0's D-1/D-7; v1-R's 24/48/168 h lags and 168/720 h statistics; LEAR's D-1, D-2, D-3, "
              "D-7 curves; DDNN-2's D-1 curve, optional D-2/D-3/D-7 and statistics; the 168 h normalisation; the H layer's released "
              "errors <= D-2; v1-R's CQR calibration D-61..D-2)",
     'members': 'B0, v1-R, v2, v3, v4, v5', 'tier': 2,
     'documented_publication': "the latest price used, delivery day D-1's, is published by D-2 13:00 Europe/Berlin",
     'against_issue_time': 'strictly before D-1 11:00 by at least 22 h on every day under the documented deadline',
     'finality': 'as the target price', 'evidence': ['CACM-47', 'REG543-12', 'SMARD-D3.1', 'RECON'],
     'basis': 'tier 2, the same values and evidence as the target price'},
    {'input': "day-ahead load forecast (v1-R's load_forecast_mw and load_forecast_day_mean_mw; LEAR's and LightGBM's load of D, "
              "LEAR's D-1 and D-7 load lags; DDNN-2's load_d0, optional load_d1/load_d7)",
     'members': 'v1-R, v2, v3, v4, v5', 'tier': 3,
     'documented_publication': 'no later than two hours before the 12:00 day-ahead gate: D-1 10:00 Europe/Berlin (Reg. 543/2013 Art. 6(2)(b))',
     'against_issue_time': 'strictly before D-1 11:00 by one hour, if the published value were the original issue',
     'finality': ('not established: the forecast must be updated when values change significantly (Art. 6(2)(b)), by at least 10% '
                  '(SMARD-D2.2.1); SMARD follows TSO updates without a fixed revision process (SMARD-B1.3). The snapshot holds SMARD\'s '
                  'values as pulled on 2026-09-08, with no issue or update timestamps, so whether any value is the original D-1 10:00 '
                  'issue cannot be established'),
     'evidence': ['REG543-6', 'SMARD-D2.2.1', 'SMARD-B1.3'],
     'basis': 'tier 3: originality cannot be established (§24.5), the known risk the anchor names; no tier-1/2 substitute exists in the preserved data'},
    {'input': 'GFS 0.25° D-1 00 UTC weather (u10, v10, u100, v100, DSWRF -> three derived columns)',
     'members': 'v3, v4, v5', 'tier': 2,
     'documented_publication': ('NCEP 00 UTC "48hr PRODUCTS" completion 03:42-04:00 UTC on D-1 (dated 30-day averages, dossier §6); '
                                'per run, the later of the two bracketing captures'),
     'against_issue_time': 'strictly before D-1 11:00 (09:00 UTC CEST, 10:00 UTC CET) by about 5-6 h on every day (margin table)',
     'finality': ('operational forecasts as issued; each run file is immutable once published; for CP-25\'s 329 runs the AWS objects\' '
                  'LastModified times are on the run morning (per-run evidence, gfs-targeted-checks.csv)'),
     'evidence': ['DOSSIER-6'],
     'basis': 'tier 2 on CP-20\'s dated NCEP schedule evidence, extended to the new issue time, with per-run evidence where it exists'},
]
REFUSED = [{'input': 'A69 day-ahead wind and solar forecasts (wind_onshore/offshore_forecast_mw, solar_forecast_mw)',
            'status': 'refused (§2 stands): the day-ahead A69 value is fixed at 18:00 Brussels on D-1, after the gate', 'evidence': ['MOP-A69']},
           {'input': 'A75 actual generation (wind/solar actuals)', 'status': 'not an input of B0, v1-R or v2-v5', 'evidence': []}]
CALENDAR = [
    ('local_hour', 'v1-R, LightGBM (B3/A2, L-N, L-R)'), ('day_of_week', 'v1-R, LightGBM'), ('month', 'v1-R, LightGBM, DDNN-2 calendar'),
    ('is_federal_holiday', 'v1-R, LightGBM, DDNN-2 calendar'), ('is_day_after_holiday', 'v1-R, LightGBM, DDNN-2 calendar'),
    ('is_bridge_day', 'v1-R, LightGBM, DDNN-2 calendar'), ('day_type', 'v1-R, LightGBM, DDNN-2 calendar'),
    ('dst_transition_day', 'v1-R, LightGBM, DDNN-2 calendar'), ('summer_peak (months 6-8)', 'v1-R, LightGBM'),
    ('winter_peak (months 12-2)', 'v1-R, LightGBM'), ('weekday_0..6 (seven dummies)', 'LEAR (A1, B2, A1_w, B2_w), DDNN-2'),
    ('B0 weekday rule (Tue-Fri D-1, Mon/Sat/Sun D-7)', 'B0'), ('DDNN-2 recency weights (half-life by day difference)', 'DDNN-2'),
]
HINDSIGHT = [
    {'label': 'crisis_period (2021-09-01..2022-12-31)', 'where': 'src/delu_forecast/features.py:156', 'member': 'v1 (released artifact)',
     'status': 'inadmissible: a regime boundary known only later; v1-R removes it (§24.5, §24.7)'},
    {'label': 'post_crisis (from 2023-01-01)', 'where': 'src/delu_forecast/features.py:159', 'member': 'v1 (released artifact)',
     'status': 'inadmissible: a regime boundary known only later; v1-R removes it (§24.5, §24.7)'},
]


# --------------------------------------------------------------------------- margin table

def _local(day: date, hh: int, mm: int = 0) -> datetime:
    return datetime.combine(day, time(hh, mm), tzinfo=guard.BERLIN).astimezone(UTC)


def availability(root: Path, cp25_evidence: dict | None = None) -> pd.DataFrame:
    """Per development delivery day: the issue instant and each group's reconstructed availability."""
    captures = runs.capture_table(root)
    rows = []
    for d in days():
        issue = guard.issue_time(d)
        run = d - timedelta(days=1)
        row = {'delivery_date': d, 'issue_utc': issue,
               'price_d1_available_utc': _local(d - timedelta(days=2), 13),     # D-1's auction, published by 13:00 on D-2
               'price_d2_available_utc': _local(d - timedelta(days=3), 13),     # D-2's prices: released errors and older lags
               'load_d0_available_utc': _local(d - timedelta(days=1), 10),     # deadline two hours before the gate
               'load_d1_available_utc': _local(d - timedelta(days=2), 10)}
        if run >= guard.FLOOR:
            a = runs.reconstructed_availability(run, captures)
            row['gfs_available_utc'] = datetime.strptime(a['available_utc'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=UTC)
            row['gfs_capture_before'], row['gfs_capture_after'] = a['capture_before'], a['capture_after']
        else:
            row['gfs_available_utc'] = pd.NaT
        if cp25_evidence and run.isoformat() in cp25_evidence:
            row['gfs_aws_last_modified_max_utc'] = cp25_evidence[run.isoformat()]
        rows.append(row)
    return pd.DataFrame(rows)


GROUPS = [
    ('prices', 'D-1 prices: lags, statistics, normalisation, the D-1 training target', 'price_d1_available_utc', 2),
    ('prices', 'D-2 and older prices: released errors, older lags, CQR calibration', 'price_d2_available_utc', 2),
    ('gfs', 'GFS D-1 00 UTC run, leads f021-f048', 'gfs_available_utc', 2),
    ('load_forecast', "load forecast of D (tier 3; computed for the Owner's ruling)", 'load_d0_available_utc', 3),
    ('load_forecast', 'load forecast of D-1 (lag; tier 3)', 'load_d1_available_utc', 3),
]


def margin_table(avail: pd.DataFrame, eligible: set | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_day, table = [], []
    for group, subgroup, col, tier in GROUPS:
        sel = avail[avail[col].notna()]
        margins = (pd.to_datetime(sel.issue_utc, utc=True) - pd.to_datetime(sel[col], utc=True)).dt.total_seconds() / 3600
        base = {'group': group, 'subgroup': subgroup, 'tier': tier}
        for s in SHIFTS:
            affected = margins <= s          # shifted availability >= issue time: not strictly before
            n, total = int(affected.sum()), int(len(sel))
            cls = 'none' if n == 0 else ('structural' if n == total else 'partial')
            days_aff = sel.delivery_date[affected.to_numpy()]
            entry = {**base, 'shift_hours': s, 'days_considered': total, 'affected_days': n, 'class': cls,
                     'first_affected': str(min(days_aff)) if n else '', 'last_affected': str(max(days_aff)) if n else '',
                     'min_margin_hours': round(float(margins.min()), 4), 'max_margin_hours': round(float(margins.max()), 4)}
            if eligible is not None:
                el = sel.delivery_date.isin(eligible).to_numpy()
                ne = int((affected.to_numpy() & el).sum())
                entry.update({'eligible_days_considered': int(el.sum()), 'eligible_affected_days': ne,
                              'class_on_eligible_days': 'none' if ne == 0 else ('structural' if ne == int(el.sum()) else 'partial')})
            table.append(entry)
        nominal = margins > 0
        if not nominal.all():
            raise ValueError(f'{subgroup}: nominal availability is not strictly before the issue time on some day')
        for d, m in zip(sel.delivery_date, margins):
            per_day.append({'delivery_date': d, 'group': group, 'subgroup': subgroup, 'margin_hours': round(float(m), 4),
                            **{f'affected_plus_{s}h': bool(m <= s) for s in SHIFTS}})
    return pd.DataFrame(table), pd.DataFrame(per_day)


def per_run_evidence(listing_path: Path) -> dict:
    """CP-25's per-run evidence: the latest AWS LastModified among the run's 10 target objects and indexes."""
    out = {}
    for line in Path(listing_path).read_text().splitlines():
        e = json.loads(line)
        stamps = []
        for v in e['leads'].values():
            for k in ('object', 'index'):
                if v.get(k):
                    stamps.append(datetime.strptime(v[k]['last_modified'], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=UTC))
        out[e['run']] = max(stamps) if len(stamps) == 20 else None
    return out


def register() -> dict:
    return {'schema': 'cp25-tier-register-v1', 'anchor': 'capstone_v21.md v21-r13 §24.4, §24.5, §24.6 items 4-5',
            'issue_time': 'D-1 11:00 Europe/Berlin, strictly before (10:00 UTC CET, 09:00 UTC CEST)',
            'published_inputs': PUBLISHED, 'refused_or_unused': REFUSED,
            'calendar_features': [{'feature': f, 'members': m, 'status': 'admissible: deterministic, known in advance'} for f, m in CALENDAR],
            'hindsight_labels': HINDSIGHT, 'missing_input_rules': MISSING_INPUT_RULES, 'citations': {k: list(v) for k, v in CITES.items()},
            'documents': DOCUMENTS,
            'tier3_inputs_of_b0_v1_v5': [p['input'] for p in PUBLISHED if p['tier'] == 3]}
