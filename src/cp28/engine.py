"""CP-28's causal replay: the seven arms, issued daily over CP-26's nominal ledger (protocol §§2–7).

At each origin D, in this order and never otherwise:

1. **Release.** Every pending issuance for a date s ≤ D−2 whose complete outcome exists is consumed
   once: into the CP-28 learning ledger (only complete all-member-valid dates) and into each combined
   arm's own H buffer (only that arm's own native central, issued before its outcome). Outcomes are
   requested through `Truth`, which refuses any delivery date after D−2.
2. **Fit.** The window is the latest at most 728 released learning dates. A3/A6 share one grid fit and
   A5/A7 one network fit (minimum 364 dates); A4 fits each block (minimum 728).
3. **Issue.** Each arm's deployed weights and native central, then its emitted quantiles or the
   prescribed fallback: a missing member → the control's (v5's) complete forecast, then B0's; a
   solver failure → A2's complete forecast; an unsupported H history, missing scale or failed H
   construction → the control, then B0. A1 is DDNN-2's own quantiles.
4. **Register.** Only now does D's native central enter its pending state, and D's members enter the
   pending learning ledger if the date is complete and all three members are valid.

The H layer is the landed `cp16.residuals.SharedResidualState`, fed one central as both arguments,
exactly as CP-26's v5 used it: residuals (y−c)/s with the issuance's own A1 scale, pooled and
local-hour quantiles, hour shrinkage n_h/(n_h+56) from 14 distinct dates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import hashlib
import json

import numpy as np
import pandas as pd

from cp16.residuals import SharedResidualState

from . import grid, meta
from .reader import Day
from .states import SEASON_ORDER, season

ARMS = ('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7')
COMBINED = ('A2', 'A3', 'A4', 'A5', 'A6', 'A7')
W0 = grid.W0
W_EQUAL = np.array([1 / 3, 1 / 3, 1 / 3])
PAD = 9                                   # the longest block: night or shoulder/peak on a 25-hour date


@dataclass(frozen=True)
class Config:
    minimum: int = 364
    full: int = 728
    h_days: int = 28                      # fixed by cp16's H layer; recorded, never varied

    def production(self) -> bool:
        return (self.minimum, self.full, self.h_days) == (364, 728, 28)


class CausalityError(RuntimeError):
    """An outcome requested for a delivery date the origin may not see."""


class Truth:
    """Outcomes for complete released dates only: every requested key must lie on or before D−2."""
    def __init__(self, prices: pd.Series):
        self.prices = prices
        self.origin = None
        self.latest_requested = None

    def at(self, origin: date):
        self.origin = origin
        return self

    def __call__(self, index) -> np.ndarray:
        index = pd.DatetimeIndex(index)
        days = sorted(set(index.tz_convert('Europe/Berlin').date))
        if self.origin is None or (days and days[-1] > self.origin - timedelta(days=2)):
            raise CausalityError(f'outcome for {days[-1] if days else None} requested at origin {self.origin}')
        if days:
            self.latest_requested = max(self.latest_requested or days[-1], days[-1])
        return self.prices.reindex(index).to_numpy(float)


def local_hours(index) -> np.ndarray:
    return np.asarray(pd.DatetimeIndex(index).tz_convert('Europe/Berlin').hour)


def blocks_of(index) -> list[str]:
    return [grid.HOUR_BLOCK[int(h)] for h in local_hours(index)]


@dataclass
class LearningRow:
    day: date
    members: np.ndarray            # (H, 3): c_HG, L, D2
    truth: np.ndarray              # (H,)
    blocks: list
    member_mae: np.ndarray         # (3,) daily mean absolute central error
    padded_c: np.ndarray           # (3 blocks, PAD, 3)
    padded_y: np.ndarray           # (3 blocks, PAD)
    padded_mask: np.ndarray        # (3 blocks, PAD)


class LearningLedger:
    """CP-28's own learning ledger: complete all-member-valid dates, released by D−2, once."""
    def __init__(self):
        self.pending: dict[date, tuple] = {}
        self.consumed: set[date] = set()
        self.rows: list[LearningRow] = []
        self.global_losses: list[np.ndarray] = []
        self.block_losses = {b: [] for b in grid.BLOCKS}
        self.last_origin = None

    def register(self, day: Day):
        if day.day in self.pending or day.day in self.consumed:
            raise ValueError('learning date already registered')
        if not (day.complete and day.all_members_valid):
            return False
        members = np.stack([day.members[name] for name in ('c_HG', 'L', 'D2')], axis=1)
        self.pending[day.day] = (day.index.copy(), members.copy())
        return True

    def release(self, origin: date, truth: Truth):
        if self.last_origin is not None and origin < self.last_origin:
            raise ValueError('learning origin cannot move backwards')
        self.last_origin = origin
        for d in sorted(self.pending):
            if d > origin - timedelta(days=2):
                break
            index, members = self.pending[d]
            y = truth(index)
            if y.shape != (len(index),) or not np.isfinite(y).all():
                continue
            blocks = blocks_of(index)
            mae = np.abs(members - y[:, None]).mean(axis=0)
            c = np.zeros((3, PAD, 3))
            yy = np.zeros((3, PAD))
            mask = np.zeros((3, PAD))
            for b, name in enumerate(grid.BLOCKS):
                positions = [h for h, x in enumerate(blocks) if x == name]
                c[b, :len(positions)] = members[positions]
                yy[b, :len(positions)] = y[positions]
                mask[b, :len(positions)] = 1.0
            overall, per_block = grid.date_losses(members[:, 0], members[:, 1], members[:, 2], y, blocks)
            self.rows.append(LearningRow(d, members, y, blocks, mae, c, yy, mask))
            self.global_losses.append(overall)
            for name in grid.BLOCKS:
                self.block_losses[name].append(per_block[name])
            self.consumed.add(d)
        for d in self.consumed:
            self.pending.pop(d, None)

    def window(self, origin: date, full: int) -> list[int]:
        cutoff = origin - timedelta(days=2)
        eligible = [i for i, row in enumerate(self.rows) if row.day <= cutoff]
        return eligible[-full:]


def error_features(ledger: LearningLedger, positions: list[int]) -> tuple[np.ndarray, dict]:
    """Six member MAE features from all released learning dates (≤ D−2): last 7 and last 28, prefix, else 0."""
    out = np.zeros(6)
    support = {}
    for k, span in enumerate((7, 28)):
        chosen = positions[-span:]
        support[f'last{span}'] = len(chosen)
        if chosen:
            mean = np.mean([ledger.rows[i].member_mae for i in chosen], axis=0)
            out[[0 + k, 2 + k, 4 + k]] = mean
    return out, support


def date_features(errors: np.ndarray, state: dict, day: date) -> np.ndarray:
    seasons = [1.0 if season(day) == s else 0.0 for s in SEASON_ORDER]
    unknown = int(state['volatility_unknown'])
    high = 0 if unknown else int(state['high_volatility'])
    return np.concatenate([errors, [float(high), float(state['negative_price_risk']), float(unknown)], seasons])


def block_rows(features: np.ndarray) -> np.ndarray:
    """Three model rows for one date: the 13 date features plus each block's one-hot."""
    return np.stack([np.concatenate([features, np.eye(3)[b]]) for b in range(3)])


def h_metadata(state: SharedResidualState, origin: date) -> dict:
    dates = [d for d, _ in state._buffer]
    return {'buffer_days': len(dates), 'buffer_hours': int(sum(len(e) for _, e in state._buffer)),
            'buffer_oldest': str(dates[0]) if dates else None, 'buffer_latest': str(dates[-1]) if dates else None,
            'buffer_span_days': (dates[-1] - dates[0]).days + 1 if dates else 0,
            'buffer_age_days': (origin - dates[-1]).days if dates else None}


def window_hash(ledger, positions) -> str:
    return hashlib.sha256(','.join(str(ledger.rows[i].day) for i in positions).encode()).hexdigest()


@dataclass
class Issue:
    arm: str
    day: date
    scored: bool
    record: dict
    index: pd.DatetimeIndex
    central: np.ndarray | None
    scoring_index: pd.DatetimeIndex
    emitted: np.ndarray | None


@dataclass
class Result:
    issues: list = field(default_factory=list)
    fits: list = field(default_factory=list)
    features: dict = field(default_factory=dict)
    networks: dict = field(default_factory=dict)
    learning_days: list = field(default_factory=list)
    h_traces: dict = field(default_factory=dict)
    config: Config = field(default_factory=Config)


def fallback(day: Day, reasons: list[str]):
    """The control's complete emitted forecast, else B0's, else a failed issuance."""
    if not len(day.scoring_index):
        return 'no_scoring_keys', None
    if day.v5_quantiles is not None:
        return 'v5', day.v5_quantiles
    if day.b0_quantiles is not None:
        return 'B0', day.b0_quantiles
    reasons.append('no_complete_fallback')
    return 'missing', None


def run(days: list[Day], prices: pd.Series, states: pd.DataFrame, *, config: Config = Config(), progress=None) -> Result:
    days = sorted(days, key=lambda d: d.day)
    if [d.day for d in days] != sorted({d.day for d in days}):
        raise ValueError('ledger dates must be unique')
    state_by_day = {row['delivery_date']: row for row in states.to_dict('records')}
    truth = Truth(prices)
    ledger = LearningLedger()
    h = {arm: SharedResidualState() for arm in COMBINED}
    result = Result(config=config)
    for step, day in enumerate(days):
        D = day.day
        truth.at(D)
        ledger.release(D, truth)
        for arm in COMBINED:
            h[arm].release(D, truth)
        window = ledger.window(D, config.full)
        n = len(window)
        support = {'learning_dates': n, 'learning_oldest': str(ledger.rows[window[0]].day) if n else None,
                   'learning_latest': str(ledger.rows[window[-1]].day) if n else None,
                   'learning_span_days': (ledger.rows[window[-1]].day - ledger.rows[window[0]].day).days + 1 if n else 0,
                   'learning_first_position': window[0] if n else None, 'learning_last_position': window[-1] if n else None,
                   'window_sha256': window_hash(ledger, window)}
        released = [i for i, row in enumerate(ledger.rows) if row.day <= D - timedelta(days=2)]
        errors, error_support = error_features(ledger, released)
        features = date_features(errors, state_by_day[D], D)
        result.features[D] = {'features': features, 'support': error_support}
        fit = {'day': D, **support, 'feature_support_last7': error_support['last7'],
               'feature_support_last28': error_support['last28']}
        failures = {}
        a3 = None
        a4 = {}
        network = None
        p_deploy = None
        if n >= config.minimum:
            try:
                a3 = grid.choose(grid.window_objective(ledger.global_losses, window))
                fit.update({f'A3_{k}': v for k, v in a3.items() if k != 'weights'})
            except FloatingPointError as exc:
                failures['A3'] = failures['A6'] = f'grid: {exc}'
            try:
                x = np.concatenate([block_rows(result.features[ledger.rows[i].day]['features']) for i in window])
                c = np.concatenate([ledger.rows[i].padded_c for i in window])
                y = np.concatenate([ledger.rows[i].padded_y for i in window])
                mask = np.concatenate([ledger.rows[i].padded_mask for i in window])
                hours = np.repeat([float(len(ledger.rows[i].truth)) for i in window], 3)
                network = meta.fit(D, x, c, y, mask, hours, n)
                p_deploy = network.predict(block_rows(features))
                result.networks[D] = {'W1': network.params['W1'], 'b1': network.params['b1'], 'W2': network.params['W2'],
                                      'b2': network.params['b2'], 'mean': network.mean, 'sd': network.sd, 'output': p_deploy}
                fit.update({'net_seed': json.dumps(network.seed), 'net_initial_sha256': network.initial_sha256,
                            'net_final_sha256': network.final_sha256})
            except (meta.SolverFailure, FloatingPointError, ValueError) as exc:
                failures['A5'] = failures['A7'] = f'network: {exc}'
        if n >= config.full and 'A3' not in failures:
            try:
                for b in grid.BLOCKS:
                    a4[b] = grid.choose(grid.window_objective(ledger.block_losses[b], window))
                    fit.update({f'A4_{b}_{k}': v for k, v in a4[b].items() if k != 'weights'})
            except FloatingPointError as exc:
                failures['A4'] = f'grid: {exc}'
                a4 = {}
        if 'A3' in failures:
            failures.setdefault('A4', failures['A3'])
        # Deployed weights for each arm, one row per block (night, solar, shoulder/peak).
        startup = n < config.minimum
        deployed, rule = {}, {}
        deployed['A2'], rule['A2'] = np.tile(W_EQUAL, (3, 1)), 'equal'
        if startup:
            for arm in ('A3', 'A5', 'A6', 'A7'):
                deployed[arm], rule[arm] = np.tile(W0, (3, 1)), 'startup_w0'
        else:
            if a3 is not None:
                deployed['A3'], rule['A3'] = np.tile(0.5 * a3['weights'] + 0.5 * W0, (3, 1)), 'fitted_shrunk'
                deployed['A6'], rule['A6'] = np.tile(a3['weights'], (3, 1)), 'fitted'
            if p_deploy is not None:
                deployed['A5'], rule['A5'] = 0.5 * p_deploy + 0.5 * W0, 'fitted_shrunk'
                deployed['A7'], rule['A7'] = p_deploy.copy(), 'fitted'
        if 'A3' in deployed and 'A4' not in failures:
            if n >= config.full and len(a4) == 3:
                deployed['A4'] = np.stack([0.25 * a4[b]['weights'] + 0.75 * deployed['A3'][k] for k, b in enumerate(grid.BLOCKS)])
                rule['A4'] = 'block_fitted_shrunk'
            else:
                deployed['A4'], rule['A4'] = deployed['A3'].copy(), 'a3_deployed' if not startup else 'startup_w0'
        for arm, w in list(deployed.items()):
            if not np.isfinite(w).all() or (w < 0).any() or np.any(np.abs(w.sum(axis=1) - 1) > 1e-12):
                failures[arm] = 'invalid deployed weights'
                deployed.pop(arm)
        for arm in COMBINED:
            if arm not in deployed and arm not in failures:
                failures[arm] = 'no deployable weights'
        fit['failures'] = json.dumps(failures, sort_keys=True)
        fit['deployed'] = json.dumps({a: deployed[a].tolist() for a in sorted(deployed)}, sort_keys=True)
        result.fits.append(fit)
        # Issue A1, then A2 (the solver-failure fallback source), then the learned arms.
        issued = {}
        blocks = blocks_of(day.index) if len(day.index) else []
        block_position = np.array([grid.BLOCKS.index(b) for b in blocks], dtype=int)
        scoring_positions = day.index.get_indexer(day.scoring_index) if len(day.index) else np.array([], dtype=int)
        if (scoring_positions < 0).any():
            raise ValueError(f'{D}: scoring keys are not issuance keys')
        common = {'day': str(D), 'scored': day.scored, 'issue_keys': len(day.index), 'scoring_keys': len(day.scoring_index),
                  'complete': day.complete, 'members_valid': day.all_members_valid, 'scale_valid': day.scale is not None,
                  **support}
        for arm in ARMS:
            reasons = []
            record = {'arm': arm, **common, 'weight_rule': rule.get(arm, 'none'),
                      'weights': json.dumps(deployed[arm].tolist()) if arm in deployed else None,
                      'solver_failure': arm in failures, 'native_central_valid': False, 'interval_ready': False,
                      'numeric_failure': False, 'error_eligible': False}
            central, emitted, source = None, None, None
            if not day.all_members_valid:
                reasons.append('member_invalid' if day.identity_ok else 'identity_invalid')
            elif arm == 'A1':
                central = day.members['D2'].copy()
                record['native_central_valid'] = True
                if day.d2_quantiles is None:
                    reasons.append('native_quantile_failure')
                else:
                    record['interval_ready'] = True
                    source, emitted = 'native', day.d2_quantiles[scoring_positions]
            elif arm in failures:
                reasons.append('solver_failure')
                a2 = issued['A2']
                source = 'A2:' + a2.record['output_source']
                emitted = None if a2.emitted is None else a2.emitted.copy()
            elif day.scale is None:
                reasons.append('scale_unavailable')
            else:
                weights = deployed[arm][block_position]
                central = grid.weighted_central(weights, day.members['c_HG'], day.members['L'], day.members['D2'])
                if not np.isfinite(central).all():
                    reasons.append('central_failure')
                    central = None
                else:
                    record['native_central_valid'] = True
                    state = h[arm]
                    numeric = not np.array_equal(central / 2 + central / 2, central)
                    if numeric:
                        reasons.append('h_identity_numeric_failure')
                    elif len(state._buffer) == config.h_days:
                        try:
                            predictions, _ = state.predict(D, day.index, central, central, day.scale)
                            q = predictions['V2-H']
                            record['interval_ready'] = True
                            source, emitted = 'native', q[scoring_positions]
                        except (ValueError, FloatingPointError):
                            numeric = True
                            reasons.append('interval_numeric_failure')
                    else:
                        reasons.append('interval_history_shortage')
                    record['numeric_failure'] = numeric
                    if not numeric and day.complete:
                        try:
                            state.issue(D, day.index, central, central, day.scale)
                            record['error_eligible'] = True
                        except (ValueError, FloatingPointError):
                            record['numeric_failure'] = True
                            reasons.append('interval_numeric_failure')
                            if source == 'native':
                                source, emitted = None, None
            if arm in COMBINED:
                record.update(h_metadata(h[arm], D))
            if source is None:
                if day.scored:
                    source, emitted = fallback(day, reasons)
                else:
                    source = 'central_only' if record['native_central_valid'] else 'missing'
            if not day.scored and source != 'native':
                emitted = None
            record['output_source'] = source
            record['fallback'] = source not in ('native', 'central_only', 'no_scoring_keys', 'missing')
            record['reasons'] = ','.join(reasons)
            record['startup_equal_to_v5_weights'] = rule.get(arm) == 'startup_w0'
            issue = Issue(arm, D, day.scored, record, day.index, central, day.scoring_index, emitted)
            issued[arm] = issue
            result.issues.append(issue)
        ledger.register(day)
        if progress is not None and (step % 100 == 0 or step == len(days) - 1):
            progress(step, D, n)
    result.learning_days = [row.day for row in ledger.rows]
    result.h_traces = {arm: h[arm].trace for arm in COMBINED}
    result.latest_truth_requested = truth.latest_requested
    return result
