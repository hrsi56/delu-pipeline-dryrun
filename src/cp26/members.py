"""Every CP-26 member at one origin, through its landed code (capstone v21-r14 candidate §24.7).

| Member | Landed code | Inputs |
|---|---|---|
| B0 | `delu_forecast.baselines.similar_day_naive` via `cp15.data.prepare` (`data.naive`) | prices |
| A1, B2 | `cp15.models.fit_day` (v2's LEAR components) | CP-15's LEAR design |
| A1_w, B2_w | `cp15.models.fit_day` on `cp20.components.augment` (v3's components) | + the three GFS columns |
| L-N, L-R | `cp21.lgbm.fit_arm` (v4's three-block member) | CP-15's LightGBM design + GFS |
| D2 | `cp24.execution.fit_origin` (eight `cp24.member.fit_member` fits, per-level median) | DDNN-2's day table |
| v1-R | `delu_forecast.model.fit_quantile_heads`, `predict_raw_heads`, `conformal.fit_cqr_thresholds`, `postprocess.apply_cqr_thresholds`, `isotonic_last` | v1's `base` catalog without `crisis_period`/`post_crisis` |

**v1-R** (§24.7, the Orchestrator's O1): the released recipe refit daily with two declared method
changes -- the hindsight flags removed, and a 728-day window. For origin D the heads train on
`[max(2019-01-01, D-728), D-62)`, D-62 is an embargo, CQR calibrates on D-61..D-2, and D-1 is an
embargo. The thread count follows the worker's allotment (one thread); deterministic row-wise
LightGBM makes the trees independent of it (`v1_recipe` reproduces B1's preserved vectors, which v1
fitted with four threads, as a check).

Every member fails closed: an exception is recorded with its cause, and the replay applies §24.7's
failure rule. Nothing here substitutes or retries with other settings. Training windows exclude D;
replay's synthetic causality controls check D/D−1 error exclusion and D−2 release separately from
the old-settings model parity gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import time
import traceback

import numpy as np
import pandas as pd

from cp15.data import array_hash
from . import harness as H

LEAR_POLICIES = {'A1': ('A1', False), 'B2': ('B2', False), 'A1_w': ('A1', True), 'B2_w': ('B2', True)}
LGBM_ARMS = ('L-N', 'L-R')
V1_EMBARGO_DAYS = 62      # heads end before D-62
V1_CAL_FIRST = 61         # CQR on D-61..D-2
V1_CAL_LAST = 2
SEVEN = ('p025', 'p10', 'p25', 'p50', 'p75', 'p90', 'p975')


@dataclass
class Context:
    """One process's inputs for one run: the prepared data, the weather design and DDNN-2's day table."""
    root: object
    settings: H.Settings
    variant: str                       # 'nominal', 'noload' or 'gfs6'
    restore: bool = False              # the no-load variant's parity mode (inputs restored)
    before: date | None = None
    data: object = None
    data_m: object = None              # the members' inputs (load removed for the no-load variant)
    data_aug: object = None            # data_m plus the three weather columns (LEAR)
    design: object = None
    wx: np.ndarray = None
    present: np.ndarray = None
    dd: object = None
    keys: object = None
    timings: dict = field(default_factory=dict)

    @classmethod
    def build(cls, root, settings: H.Settings, variant: str, *, restore: bool = False, before=None, ddnn2=True):
        from cp20.components import augment
        t0 = time.perf_counter()
        H.install_issue_label(settings)
        ctx = cls(root, settings, variant, restore, before)
        ctx.data = H.load(root, settings, before)
        ctx.design = H.weather_design(root, settings)
        ctx.data_m = H.without_load(ctx.data, restore=restore) if variant == 'noload' else ctx.data
        ctx.data_aug, present_l = augment(ctx.data_m, ctx.design)
        ctx.wx, ctx.present = H.weather_matrix(ctx.design, ctx.data)
        if not np.array_equal(present_l, ctx.present):
            raise H.HarnessError('weather presence differs between the LEAR and LightGBM designs')
        if ddnn2:
            from cp24 import design as G
            from cp24.member import Keys
            ctx.dd = G.build(ctx.data, ctx.design)
            ctx.keys = Keys.from_data(ctx.data, ctx.dd)
        ctx.timings['build_seconds'] = time.perf_counter() - t0
        return ctx

    @property
    def noload_active(self) -> bool:
        return self.variant == 'noload' and not self.restore


def _ok(name, day, central, extra=None, seconds=0.0):
    central = np.asarray(central, float)
    if not np.isfinite(central).all():
        raise ValueError(f'{name} {day}: nonfinite central forecast')
    return {'status': 'ok', 'central': central, 'seconds': seconds, **(extra or {})}


def target_provenance(data, rows, first, end):
    """Describe the landed selector's actual target rows; never select different rows."""
    rows = np.asarray(rows, dtype=int)
    return {'first_delivery_day': str(first), 'end_exclusive': str(end),
            'latest_delivery_day': str(data.dates[rows].max()) if len(rows) else None,
            'rows_sha256': array_hash(rows), 'target_sha256': array_hash(data.y[rows]),
            'rows': int(len(rows))}


def lineage(ctx, day, targets, seeds):
    return {'issue_utc': H.issue_instant(day, getattr(ctx, 'settings', H.NEW)).isoformat(),
            'training_target_cutoff': str(day), 'targets': targets, 'seeds': seeds,
            'availability_rule': 'published strictly before issue_utc; target delivery day embargoed',
            'input_vintages_reference': 'reports/development-inputs/tier-register.json',
            'rulings_reference': 'docs/track-b/cp-25-rulings-2026-10-12.md'}


def fail(name: str, exc: BaseException, seconds: float) -> dict:
    return {'status': 'failed', 'cause': H.G25.redact(f'{type(exc).__name__}: {exc}'),
            'traceback': H.G25.redact(traceback.format_exc(limit=6)),
            'seconds': seconds}


def fit_b0(ctx: Context, day: date) -> dict:
    from delu_forecast.baselines import SIMILAR_DAY_LAG_BY_WEEKDAY
    rows = ctx.data.rows(day)
    source = day - timedelta(days=SIMILAR_DAY_LAG_BY_WEEKDAY[day.weekday()])
    return _ok('B0', day, ctx.data.naive[rows],
               {'records': {'rule': 'similar-day naive: D-1 Tue-Fri, D-7 Mon/Sat/Sun',
                            'source_delivery_day': str(source)},
                'lineage': lineage(ctx, day, {'source_delivery_day': str(source)}, {'random_seed': None})})


def _lear_record(log: list[dict]) -> list[dict]:
    keep = ('local_hour', 'n_train', 'n_hour_train', 'history_start', 'history_end_exclusive',
            'latest_training_delivery_date', 'selected_relative_alpha', 'alpha', 'final_n_iter', 'final_solver_passes',
            'solver_calls', 'train_rows_sha256', 'train_target_sha256', 'hour_train_rows_sha256', 'model_sha256',
            'imputer_sha256', 'scaler_sha256', 'fit_seconds', 'origin_utc')
    return [{k: r[k] for k in keep if k in r} for r in log]


def fit_lear(ctx: Context, day: date, name: str) -> dict:
    import cp15.models as m
    from cp20.components import training_rows
    policy, weather = LEAR_POLICIES[name]
    data = ctx.data_aug if weather else ctx.data_m
    rows = data.rows(day)
    started = time.perf_counter()
    if weather:
        train = training_rows(data, day, policy)
        if not ctx.present[train].all() or not ctx.present[rows].all():
            raise ValueError(f'{day}: a training/forecast row lacks a weather record')
    pred, log = m.fit_day(data, day, policy, rows=rows)
    train = training_rows(data, day, policy)
    return _ok(name, day, pred, {'records': _lear_record(log), 'fits': {'lasso': sum(r.get('solver_calls', 0) for r in log),
                                                                       'hours': len(log)},
                                'lineage': lineage(ctx, day, target_provenance(data, train,
                                    m.history_start(day, policy), day),
                                    {'random_seed': None, 'protocol_seed': int(data.p['seed']),
                                     'solver_selection': data.p['lear']['selection']})},
               time.perf_counter() - started)


def fit_lgbm(ctx: Context, day: date, arm: str) -> dict:
    from cp21.lgbm import fit_arm, model_rows
    started = time.perf_counter()
    count = {'n': 0}

    def charge(role):
        count['n'] += 1
    central, records, selection = fit_arm(ctx.data_m, ctx.wx, ctx.present, day, arm, charge=charge, n_jobs=1)
    targets = {}
    for record in records:
        key = f"{record['model']}/{record['role']}"
        if key not in targets:
            lower, window, inner, _, _ = model_rows(ctx.data_m, day, record['model'], ctx.present)
            selected = inner if record['role'] == 'inner' else window
            end = day - timedelta(days=28) if record['role'] == 'inner' else day
            targets[key] = target_provenance(ctx.data_m, selected, lower, end)
    keep = ('model', 'role', 'config', 'n_train', 'validation_mae', 'history_start', 'history_end_exclusive',
            'window_rows_sha256', 'model_sha256', 'fit_wall_seconds', 'origin_utc')
    return _ok(arm, day, central, {'records': [{k: r[k] for k in keep if k in r} for r in records],
                                   'selection': selection, 'fits': {'lightgbm': count['n']},
                                   'lineage': lineage(ctx, day, targets, {'random_state': int(ctx.data_m.p['seed'])})},
               time.perf_counter() - started)


def ensemble_for(ensembles: dict, day: date):
    """The search that serves `day` (§24.7: search k serves S_k .. S_(k+1) - 1) and its members."""
    from .search import serving_search
    k = serving_search(day)
    item = ensembles.get(str(k))
    return k, item


def fit_d2(ctx: Context, day: date, ensemble: dict | None, k: int | None) -> dict:
    """DDNN-2's eight members fitted fresh at `day` and their per-level median (cp24.execution.fit_origin)."""
    from cp24.execution import fit_origin
    started = time.perf_counter()
    if ensemble is None or not ensemble.get('complete'):
        raise ValueError(f'search k={k} produced no complete ensemble (four distinct configurations); '
                         f'no DDNN-2 forecast exists for {day}')
    if len(ensemble.get('members', [])) != 8:
        raise ValueError('D2 requires exactly eight frozen members; no subset median')
    count = {'n': 0}

    def charge():
        count['n'] += 1
    if k is None:
        if ctx.settings != H.OLD:
            raise ValueError('CP-24 ensembles are parity fixtures only')
        with H.ddnn2_without_load(ctx.noload_active):
            out = fit_origin(ctx.dd, ctx.keys, ctx.data, day, ensemble['members'], charge, uncovered_fails=True)
    else:
        from .search import fit_origin as fit_cp26_origin
        out = fit_cp26_origin(ctx, day, ensemble, k, charge)
    if len(out.get('members', [])) != 8 or count['n'] != 8:
        raise ValueError('one or more daily D2 members failed; no subset median')
    q = np.asarray(out['quantiles'], float)
    from cp24 import design as G
    lower, window, _ = G.window(ctx.dd, day, exclude_uncovered=True)
    targets = {}
    for record in out['members']:
        weeks = [date.fromisoformat(w) for w in record['window']['held_out_weeks']]
        train, hold = G.split(ctx.dd, window, weeks)
        targets[str(record['seed'])] = {
            'first_delivery_day': str(lower), 'end_exclusive': str(day),
            'latest_training_delivery_day': str(ctx.dd.date_of(int(train[-1]))) if len(train) else None,
            'latest_holdout_delivery_day': str(ctx.dd.date_of(int(hold[-1]))) if len(hold) else None,
            'train_days_sha256': array_hash(train), 'holdout_days_sha256': array_hash(hold)}
    members = [{'member': r['member'], 'rank': r['rank'], 'config': r['config'], 'seed': r['seed'],
                'params_sha256': r['record'].get('params_sha256'), 'best_epoch': r['record'].get('best_epoch'),
                'epochs_run': r['record'].get('epochs_run'), 'stop_reason': r['record'].get('stop_reason'),
                'window': r['window'], 'cap_slot_levels': r['guards'].get('cap_forecast_slot_levels'),
                'n_inputs': r['n_inputs'], 'transform': r['transform'], 'seconds': r['seconds']}
               for r in out['members']]
    return _ok('D2', day, out['central'], {'quantiles': q, 'records': members, 'crossed_rows': out['crossed_rows'],
                                           'search': k, 'fits': {'ddnn2': count['n']},
                                           'lineage': lineage(ctx, day, targets,
                                               {'ensemble_seeds': [int(r['seed']) for r in out['members']]})},
               time.perf_counter() - started)


# ------------------------------------------------------------------ v1-R
def v1_features(data, columns: list[str] | None = None) -> pd.DataFrame:
    """v1's `base` catalog as a frame (the CP-15 LightGBM list is v1's base catalog without the two
    hindsight flags, in v1's own order)."""
    names = list(data.p['lgbm']['features'])
    return pd.DataFrame(data.lgbm_raw, columns=names) if columns is None else columns


def v1_recipe(x: pd.DataFrame, y: np.ndarray, train: np.ndarray, cal: np.ndarray, rows: np.ndarray, *, n_jobs: int = 1):
    """The released v1 recipe: nine LightGBM quantile heads, CQR on the calibration rows, isotonic
    ordering. Returns (raw, post-CQR, final) nine-quantile matrices for `rows` and the thresholds."""
    from delu_forecast.conformal import fit_cqr_thresholds, thresholds_to_json
    from delu_forecast.model import LGBM_PARAMS, SEED, fit_quantile_heads, predict_raw_heads
    from delu_forecast.postprocess import apply_cqr_thresholds, isotonic_last
    params = dict(LGBM_PARAMS, n_jobs=n_jobs)
    heads = fit_quantile_heads(x.iloc[train], y[train], seed=SEED, params=params)
    thresholds = fit_cqr_thresholds(predict_raw_heads(heads, x.iloc[cal]), y[cal])
    raw = predict_raw_heads(heads, x.iloc[rows])
    post = apply_cqr_thresholds(raw, thresholds)
    final = isotonic_last(post)
    trees = {label: array_hash(np.frombuffer(h.booster_.model_to_string().encode(), dtype=np.uint8))
             for label, h in heads.items()}
    return raw, post, final, thresholds_to_json(thresholds), trees


def v1r_windows(data, day: date):
    lower = max(H.FLOOR, day - timedelta(days=728))
    dates = data.dates
    train = np.flatnonzero((dates >= np.datetime64(lower)) & (dates < np.datetime64(day - timedelta(days=V1_EMBARGO_DAYS)))
                           & data.eligible)
    cal = np.flatnonzero((dates >= np.datetime64(day - timedelta(days=V1_CAL_FIRST)))
                         & (dates <= np.datetime64(day - timedelta(days=V1_CAL_LAST))) & data.eligible)
    return lower, train, cal


def fit_v1r(ctx: Context, day: date) -> dict:
    from delu_forecast.model import SEED
    started = time.perf_counter()
    data = ctx.data
    rows = data.rows(day)
    lower, train, cal = v1r_windows(data, day)
    raw, post, final, thresholds, trees = v1_recipe(v1_features(data), data.y, train, cal, rows, n_jobs=1)
    from delu_forecast.postprocess import QUANTILE_LABELS
    seven = [QUANTILE_LABELS.index(q) for q in SEVEN]
    q = final[:, seven]
    return _ok('v1-R', day, q[:, 3], {'quantiles': q, 'quantiles9': final,
                                      'records': {'heads_window': [str(lower), str(day - timedelta(days=V1_EMBARGO_DAYS))],
                                                  'calibration_window': [str(day - timedelta(days=V1_CAL_FIRST)),
                                                                         str(day - timedelta(days=V1_CAL_LAST))],
                                                  'n_train': int(len(train)), 'n_calibration': int(len(cal)),
                                                  'thresholds': thresholds, 'trees_sha256': trees,
                                                  'train_rows_sha256': array_hash(train), 'cal_rows_sha256': array_hash(cal),
                                                  'raw_crossings': int((np.diff(raw, axis=1) < 0).any(axis=1).sum())},
                                      'fits': {'lightgbm_heads': 9},
                                      'lineage': {**lineage(ctx, day, {
                                          'heads': target_provenance(data, train, lower, day - timedelta(days=V1_EMBARGO_DAYS)),
                                          'calibration': target_provenance(data, cal, day - timedelta(days=V1_CAL_FIRST),
                                                                           day - timedelta(days=V1_CAL_LAST - 1))},
                                          {'random_state': int(SEED)}),
                                          'training_target_cutoff': str(day - timedelta(days=V1_EMBARGO_DAYS)),
                                          'calibration_target_cutoff': str(day - timedelta(days=V1_CAL_LAST - 1))}},
               time.perf_counter() - started)


FITTERS = {
    'B0': lambda ctx, day, **kw: fit_b0(ctx, day),
    'A1': lambda ctx, day, **kw: fit_lear(ctx, day, 'A1'),
    'B2': lambda ctx, day, **kw: fit_lear(ctx, day, 'B2'),
    'A1_w': lambda ctx, day, **kw: fit_lear(ctx, day, 'A1_w'),
    'B2_w': lambda ctx, day, **kw: fit_lear(ctx, day, 'B2_w'),
    'L-N': lambda ctx, day, **kw: fit_lgbm(ctx, day, 'L-N'),
    'L-R': lambda ctx, day, **kw: fit_lgbm(ctx, day, 'L-R'),
    'D2': lambda ctx, day, ensemble=None, k=None, **kw: fit_d2(ctx, day, ensemble, k),
    'v1-R': lambda ctx, day, **kw: fit_v1r(ctx, day),
}
#: The members each run fits (§24.7; ruling 1 for the variant; ruling 3 for the delay run).
RUN_MEMBERS = {
    'nominal': ('B0', 'A1', 'B2', 'A1_w', 'B2_w', 'L-N', 'L-R', 'D2', 'v1-R'),
    'noload': ('A1_w', 'B2_w', 'L-N', 'L-R', 'D2'),
    'gfs6': ('A1_w', 'B2_w', 'L-N', 'L-R', 'D2'),
}


def fit_members(ctx: Context, day: date, members, *, ensemble=None, k=None) -> dict:
    """Every requested member at `day`; each fails closed independently."""
    out = {}
    for name in members:
        started = time.perf_counter()
        try:
            out[name] = FITTERS[name](ctx, day, ensemble=ensemble, k=k)
        except Exception as exc:   # recorded; the replay applies the failure rule
            out[name] = fail(name, exc, time.perf_counter() - started)
    return out
