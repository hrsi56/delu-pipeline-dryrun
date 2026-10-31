"""Causal CP-26 issuance state. This module fits no model and computes no score.

Scoring keys, full native error issuances, and full three-member learning rows
are deliberately separate. A valid native central can warm its own history at
a scored origin while the immutable scored output is B0. Only a shortage of
released history has that exemption: central/scale/identity/numeric failures do
not seed H. Each route owns independent policy states and a learning ledger.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cp15.data import LEVELS, array_hash, day_hours
from cp16.residuals import SharedResidualState, _vector

SCHEMA = 'cp26-causal-state-v1'
ROUTES = ('nominal', 'noload', 'gfs6')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def timestamps(day, index):
    if type(day) is not date:
        raise ValueError('delivery day must be a datetime.date')
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        raise ValueError('timestamps must be timezone aware')
    index = index.tz_convert('UTC').as_unit('ns').copy()
    if index.has_duplicates or not index.is_monotonic_increasing or not index.isin(day_hours(day)).all():
        raise ValueError('invalid canonical delivery-day keys')
    return index


def complete(day, index):
    return timestamps(day, index).equals(day_hours(day).as_unit('ns'))


def quantiles(value, n):
    value = np.array(value, dtype=float, copy=True)
    if value.shape != (n, 7) or not np.isfinite(value).all() or (np.diff(value, axis=1) < 0).any():
        raise ValueError('invalid seven-quantile vector')
    return value


def _instant(day, instant):
    from cp25.guard import issue_time
    instant = pd.Timestamp(instant)
    if instant.tzinfo is None:
        raise ValueError('issue instant must be timezone aware')
    instant = instant.tz_convert('UTC')
    if instant != pd.Timestamp(issue_time(day)).tz_convert('UTC'):
        raise ValueError('native issuance must use the frozen CP-26 issue instant')
    return instant.isoformat()


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


class RawResidualState:
    """B0's unchanged CP-15 raw, pooled error recipe; no A1 scale."""
    def __init__(self):
        self.pending, self.consumed, self.buffer = {}, set(), []
        self.last_origin, self.trace = None, []

    def issue(self, day, index, central):
        if day in self.pending or day in self.consumed:
            raise ValueError('issued day already registered')
        index = timestamps(day, index)
        if not complete(day, index):
            raise ValueError('B0 residual intake requires a full canonical issuance')
        self.pending[day] = (index, _vector(central, len(index), 'B0 central'))

    def release(self, origin, truth):
        if self.last_origin is not None and origin < self.last_origin:
            raise ValueError('feedback origin cannot move backwards')
        self.last_origin = origin
        for day in sorted(self.pending):
            if day > origin - timedelta(days=2):
                break
            index, central = self.pending[day]
            actual = np.asarray(truth(index.copy()), float)
            if actual.shape != central.shape or not np.isfinite(actual).all():
                self.trace.append({'origin': str(origin), 'day': str(day), 'status': 'truth_unavailable'})
                continue
            errors = _vector(actual - central, len(index), 'B0 raw errors')
            self.buffer = sorted([*self.buffer, (day, errors)], key=lambda item: item[0])[-28:]
            self.consumed.add(day)
            self.trace.append({'origin': str(origin), 'day': str(day), 'status': 'consumed',
                               'central_sha256': array_hash(central), 'errors_sha256': array_hash(errors)})
        for day in self.consumed:
            self.pending.pop(day, None)

    def predict(self, central):
        if len(self.buffer) != 28:
            raise ValueError('B0 requires 28 complete released raw-error days')
        errors = np.concatenate([e for _, e in self.buffer])
        return quantiles(np.asarray(central)[:, None] + np.quantile(errors, LEVELS, method='linear'), len(central))

    def to_dict(self):
        return {'last_origin': str(self.last_origin) if self.last_origin else None,
                'pending': [{'day': str(d), 'index': [t.isoformat() for t in i], 'central': c.tolist()}
                            for d, (i, c) in sorted(self.pending.items())],
                'consumed': [str(d) for d in sorted(self.consumed)],
                'buffer': [{'day': str(d), 'errors': e.tolist()} for d, e in self.buffer], 'trace': self.trace}

    @classmethod
    def from_dict(cls, item):
        result = cls()
        result.last_origin = date.fromisoformat(item['last_origin']) if item['last_origin'] else None
        consumed = [date.fromisoformat(d) for d in item['consumed']]
        if len(set(consumed)) != len(consumed):
            raise ValueError('duplicate B0 consumed day')
        result.consumed = set(consumed)
        for row in item['pending']:
            result.issue(date.fromisoformat(row['day']), pd.to_datetime(row['index'], utc=True), row['central'])
        for row in item['buffer']:
            day = date.fromisoformat(row['day'])
            result.buffer.append((day, _vector(row['errors'], len(day_hours(day)), 'B0 saved raw errors')))
        dates = [d for d, _ in result.buffer]
        if len(dates) > 28 or dates != sorted(set(dates)) or not set(dates).issubset(result.consumed):
            raise ValueError('invalid B0 residual history')
        if result.consumed and (result.last_origin is None or any(d > result.last_origin - timedelta(days=2) for d in result.consumed)):
            raise ValueError('B0 saved state contains unreleased errors')
        result.trace = json.loads(json.dumps(item['trace'], allow_nan=False))
        return result


class PolicyState:
    """One policy on one route, with immutable original scored daily outputs."""
    def __init__(self, policy, route, identity, layer='H'):
        if route not in ROUTES or layer not in ('H', 'B0', 'direct') or not isinstance(identity, dict) or not identity:
            raise ValueError('invalid policy state identity, route or layer')
        if (policy == 'B0') != (layer == 'B0'):
            raise ValueError('only B0 uses raw errors')
        self.policy, self.route, self.layer = policy, route, layer
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        self.errors = SharedResidualState() if layer == 'H' else RawResidualState() if layer == 'B0' else None
        self.records, self.last_origin = {}, None

    @property
    def buffer_days(self):
        return 0 if self.errors is None else len(self.errors._buffer if self.layer == 'H' else self.errors.buffer)

    def release(self, day, truth):
        if self.last_origin is not None and day < self.last_origin:
            raise ValueError('origin cannot move backwards')
        self.last_origin = day
        if self.errors is not None:
            self.errors.release(day, truth)

    def buffer_metadata(self, origin):
        buffer = [] if self.errors is None else self.errors._buffer if self.layer == 'H' else self.errors.buffer
        dates = [day for day, _ in buffer]
        return {'days': len(dates), 'hours': sum(len(errors) for _, errors in buffer),
                'oldest_day': str(dates[0]) if dates else None, 'latest_day': str(dates[-1]) if dates else None,
                'calendar_span_days': (dates[-1] - dates[0]).days + 1 if dates else 0,
                'age_days': (origin - dates[-1]).days if dates else None,
                'oldest_age_days': (origin - dates[0]).days if dates else None}

    def issue(self, day, index, central, *, issue_utc, scale=None, native_quantiles=None,
              scoring_index=None, scored=False, fallback=None, central_failure=None, identity_valid=True,
              provenance=None):
        """Issue before truth; H shortage alone permits own-central recovery intake.

        ``fallback`` contains B0's quantiles and canonical keys, never its central
        for intake. A scored missing B0 remains missing and can still warm itself.
        """
        if self.last_origin != day:
            raise ValueError('release must precede issuance at this origin')
        if str(day) in self.records:
            raise ValueError('daily issuance already recorded')
        index = timestamps(day, index)
        scoring_index = timestamps(day, index if scoring_index is None else scoring_index)
        if not scoring_index.isin(index).all():
            raise ValueError('scoring keys must be contained in issuance keys')
        instant = _instant(day, issue_utc)
        native, issued_scale, q, cause = None, None, None, central_failure
        valid = identity_valid and central_failure is None and len(index) > 0
        usable = valid
        if not len(index):
            cause = 'no_native_forecast_keys'
        if not identity_valid:
            cause = 'identity_failure'
        if valid:
            try:
                native = _vector(central, len(index), 'native central')
            except (TypeError, ValueError):
                valid, cause = False, 'central_failure'
            usable = valid
            if valid:
                try:
                    if self.layer == 'H':
                        issued_scale = _vector(scale, len(index), 'issued scale', scale=True)
                    if self.layer == 'direct':
                        q = quantiles(native_quantiles, len(index))
                except (TypeError, ValueError):
                    usable, cause = False, 'scale_failure' if self.layer == 'H' else 'native_quantile_failure'
        ready = usable and (self.layer == 'direct' or self.buffer_days == 28)
        numeric_failure = False
        if usable and self.layer == 'H':
            try:
                if not np.array_equal(native / 2 + native / 2, native):
                    raise FloatingPointError('single-central H identity underflowed')
            except FloatingPointError:
                q, ready, numeric_failure, cause = None, False, True, 'h_identity_numeric_failure'
        if ready and self.layer != 'direct' and not numeric_failure:
            try:
                if self.layer == 'H':
                    predictions, _ = self.errors.predict(day, index, native, native, issued_scale)
                    q = quantiles(predictions['V2-H'], len(index))
                else:
                    q = self.errors.predict(native)
            except (TypeError, ValueError, FloatingPointError):
                q, ready, numeric_failure, cause = None, False, True, 'interval_numeric_failure'
        elif usable and self.layer != 'direct' and not numeric_failure:
            cause = 'interval_history_shortage'
        intake = bool(usable and not numeric_failure and self.layer != 'direct' and complete(day, index))
        if intake:
            if self.layer == 'H':
                # Duplicate/corrupt state is a hard invariant failure, distinct
                # from numerical construction of a validated own-error issuance.
                if day in self.errors._pending or day in self.errors._consumed:
                    raise ValueError('issued day already registered')
                try:
                    self.errors.issue(day, index, native, native, issued_scale)
                except (ValueError, FloatingPointError):
                    q, ready, intake, cause = None, False, False, 'interval_numeric_failure'
            else:
                self.errors.issue(day, index, native)
        source, emitted = ('native', q) if q is not None else ('central_only', None) if not scored and usable else ('missing', None)
        if scored and q is None and self.policy != 'B0' and fallback is not None:
            try:
                fallback_index = timestamps(day, pd.to_datetime(fallback['index'], utc=True))
                fallback_q = quantiles(fallback['quantiles'], len(fallback_index))
                positions = fallback_index.get_indexer(scoring_index)
                if (positions < 0).any():
                    raise ValueError('fallback lacks frozen scored keys')
                emitted, source = fallback_q[positions], 'B0'
            except (TypeError, ValueError, KeyError):
                emitted, source = None, 'missing'
        elif emitted is not None:
            emitted = emitted[index.get_indexer(scoring_index)]
        row = {'day': str(day), 'policy': self.policy, 'route': self.route, 'issue_utc': instant,
               'identity_sha256': digest(self.identity), 'index': [t.isoformat() for t in index],
               'scoring_index': [t.isoformat() for t in scoring_index], 'scored': bool(scored),
               'native_central_valid': bool(valid), 'native_scale_valid': bool(issued_scale is not None) if self.layer == 'H' else None,
               'interval_ready': bool(ready), 'output_source': source,
               'full_canonical_issuance': complete(day, index), 'error_eligible': intake,
               'reason': cause, 'buffer_days_at_issue': self.buffer_days, 'buffer': self.buffer_metadata(day),
               'native_central': native.tolist() if valid else None,
               'issued_scale': issued_scale.tolist() if valid and issued_scale is not None else None,
               'native_quantiles': q.tolist() if q is not None else None,
               'emitted_quantiles': emitted.tolist() if emitted is not None else None,
               'provenance': provenance or {}}
        self.records[str(day)] = json.loads(json.dumps(row, allow_nan=False))
        return json.loads(json.dumps(row, allow_nan=False))

    def to_dict(self):
        return {'policy': self.policy, 'route': self.route, 'identity': self.identity, 'layer': self.layer,
                'last_origin': str(self.last_origin) if self.last_origin else None,
                'errors': self.errors.to_dict() if self.errors is not None else None, 'records': self.records}

    @classmethod
    def from_dict(cls, row):
        result = cls(row['policy'], row['route'], row['identity'], row['layer'])
        result.last_origin = date.fromisoformat(row['last_origin']) if row['last_origin'] else None
        if result.layer == 'H':
            result.errors = SharedResidualState.from_dict(row['errors'])
        elif result.layer == 'B0':
            result.errors = RawResidualState.from_dict(row['errors'])
        elif row['errors'] is not None:
            raise ValueError('direct CQR/JSU policy cannot have an H buffer')
        if result.errors is not None and result.errors.last_origin != result.last_origin:
            raise ValueError('policy and error origins differ')
        result.records = json.loads(json.dumps(row['records'], allow_nan=False))
        for day_string, record in result.records.items():
            day = date.fromisoformat(day_string)
            if record['day'] != day_string or record['policy'] != result.policy or record['route'] != result.route \
                    or record['identity_sha256'] != digest(result.identity) or result.last_origin is None or day > result.last_origin:
                raise ValueError('invalid saved issuance identity')
            _instant(day, record['issue_utc'])
            index = timestamps(day, pd.to_datetime(record['index'], utc=True))
            scoring = timestamps(day, pd.to_datetime(record['scoring_index'], utc=True))
            if not scoring.isin(index).all() or record['full_canonical_issuance'] != complete(day, index):
                raise ValueError('invalid saved issuance keys')
            if record['error_eligible'] and (not record['native_central_valid'] or not complete(day, index)):
                raise ValueError('saved invalid native intake')
            if record['native_central_valid']:
                _vector(record['native_central'], len(index), 'saved native central')
                if result.layer == 'H' and record['native_scale_valid']:
                    _vector(record['issued_scale'], len(index), 'saved historical scale', scale=True)
                if result.layer == 'H' and record['error_eligible'] and not record['native_scale_valid']:
                    raise ValueError('saved H intake lacks historical scale')
            if record['emitted_quantiles'] is not None:
                quantiles(record['emitted_quantiles'], len(scoring))
        if result.errors is not None:
            pending = result.errors._pending if result.layer == 'H' else result.errors.pending
            consumed = result.errors._consumed if result.layer == 'H' else result.errors.consumed
            eligible = {date.fromisoformat(day) for day, record in result.records.items() if record['error_eligible']}
            if set(pending) | consumed != eligible or set(pending) & consumed:
                raise ValueError('saved error state differs from native eligible issuances')
            for day, item in pending.items():
                index, central = (item.index, item.central) if result.layer == 'H' else item
                record = result.records[str(day)]
                if not index.equals(pd.to_datetime(record['index'], utc=True).as_unit('ns')) \
                        or not np.array_equal(central, record['native_central']):
                    raise ValueError('saved pending central differs from original native issuance')
                if result.layer == 'H' and not np.array_equal(item.scale, record['issued_scale']):
                    raise ValueError('saved pending scale differs from historical issuance scale')
            consumed_traces = [row for row in result.errors.trace if row['status'] == 'consumed']
            by_day = {date.fromisoformat(row['feedback_day'] if result.layer == 'H' else row['day']): row
                      for row in consumed_traces}
            if set(by_day) != consumed or len(by_day) != len(consumed_traces):
                raise ValueError('saved error consumption lacks unique native provenance')
            for day, trace in by_day.items():
                record = result.records[str(day)]
                central_hash = trace['issued_center_sha256'] if result.layer == 'H' else trace['central_sha256']
                if central_hash != array_hash(np.asarray(record['native_central'], float)):
                    raise ValueError('saved released errors differ from native central provenance')
                if result.layer == 'H' and trace['issued_scale_sha256'] != array_hash(np.asarray(record['issued_scale'], float)):
                    raise ValueError('saved released errors differ from historical scale provenance')
            buffer = result.errors._buffer if result.layer == 'H' else result.errors.buffer
            for day, errors in buffer:
                trace = by_day[day]
                expected = trace['error_sha256'] if result.layer == 'H' else trace['errors_sha256']
                if array_hash(errors) != expected:
                    raise ValueError('saved residual buffer differs from consumed native errors')
        return result


class LearningLedger:
    """CP-28 compatibility only: native c_HG/L/D2 days, separate from H.

    This ledger implements no CP-28 arm, solver, feature or selection procedure.
    A full original member row may enter it while v5's scored output was B0 due
    to an H-history shortage. A fallback member never creates a learning row.
    """
    MEMBERS = ('c_HG', 'L', 'D2')

    def __init__(self, route, identity):
        if route not in ROUTES or not identity:
            raise ValueError('invalid learning ledger identity')
        self.route, self.identity = route, json.loads(json.dumps(identity, allow_nan=False))
        self.pending, self.rows, self.consumed = {}, {}, set()
        self.last_origin = None

    def issue(self, day, index, members, *, issue_utc, scale=None, provenance=None):
        if self.last_origin is not None and day != self.last_origin:
            raise ValueError('learning issuance must belong to the current origin')
        accepted = self._register_pending(day, index, members, issue_utc=issue_utc, scale=scale, provenance=provenance)
        if accepted and self.last_origin is None:
            self.last_origin = day
        return accepted

    def _register_pending(self, day, index, members, *, issue_utc, scale=None, provenance=None):
        """Validated restoration helper, never a public historical issuance API."""
        if day in self.pending or day in self.consumed:
            raise ValueError('learning day already registered')
        index = timestamps(day, index)
        if not complete(day, index) or set(members) != set(self.MEMBERS):
            return False
        vectors = {name: _vector(members[name], len(index), f'learning member {name}').tolist() for name in self.MEMBERS}
        try:
            scale = _vector(scale, len(index), 'learning issued scale', scale=True).tolist() if scale is not None else None
        except (TypeError, ValueError):
            scale = None
        self.pending[day] = {'day': str(day), 'route': self.route, 'identity_sha256': digest(self.identity),
                             'issue_utc': _instant(day, issue_utc), 'index': [t.isoformat() for t in index],
                             'members': vectors, 'issued_scale': scale, 'scale_available': scale is not None,
                             'learning_support_at_issue': self.support(day), 'provenance': provenance or {}}
        return True

    def release(self, origin, truth):
        if self.last_origin is not None and origin < self.last_origin:
            raise ValueError('learning origin cannot move backwards')
        self.last_origin = origin
        for day in sorted(self.pending):
            if day > origin - timedelta(days=2):
                break
            row = self.pending[day]
            index = pd.to_datetime(row['index'], utc=True)
            actual = np.asarray(truth(index.copy()), float)
            if actual.shape != (len(index),) or not np.isfinite(actual).all():
                continue
            released = json.loads(json.dumps(row, allow_nan=False))
            released.update(truth=actual.tolist(), released_origin=str(origin))
            self.rows[day], self.consumed = released, self.consumed | {day}
        for day in self.consumed:
            self.pending.pop(day, None)

    def available(self, origin, limit=728):
        if limit not in (364, 728):
            raise ValueError('registered learning windows are 364 or 728 days')
        cutoff = origin - timedelta(days=2)
        rows = [row for day, row in sorted(self.rows.items())
                if day <= cutoff and date.fromisoformat(row['released_origin']) <= origin]
        return [json.loads(json.dumps(row, allow_nan=False)) for row in rows[-limit:]]

    def support(self, origin):
        rows = self.available(origin)
        days = [date.fromisoformat(row['day']) for row in rows]
        return {'available_dates': len(days), 'minimum_364_met': len(days) >= 364, 'minimum_728_met': len(days) >= 728,
                'oldest_day': str(days[0]) if days else None, 'latest_day': str(days[-1]) if days else None,
                'calendar_span_days': (days[-1] - days[0]).days + 1 if days else 0,
                'age_days': (origin - days[-1]).days if days else None}

    def to_dict(self):
        return {'route': self.route, 'identity': self.identity, 'last_origin': str(self.last_origin) if self.last_origin else None,
                'pending': [row for _, row in sorted(self.pending.items())],
                'rows': [row for _, row in sorted(self.rows.items())], 'consumed': [str(d) for d in sorted(self.consumed)]}

    @classmethod
    def from_dict(cls, value):
        result = cls(value['route'], value['identity'])
        result.last_origin = date.fromisoformat(value['last_origin']) if value['last_origin'] else None
        for row in value['pending'] + value['rows']:
            if row['route'] != result.route or row['identity_sha256'] != digest(result.identity):
                raise ValueError('learning row identity mismatch')
            day = date.fromisoformat(row['day'])
            if result.last_origin is None or day > result.last_origin:
                raise ValueError('saved learning issuance is later than its current origin')
            if not result._register_pending(day, pd.to_datetime(row['index'], utc=True), row['members'], issue_utc=row['issue_utc'],
                                            scale=row['issued_scale'], provenance=row['provenance']):
                raise ValueError('saved learning issuance lacks full original members')
            prepared = result.pending[day]
            if row['issued_scale'] != prepared['issued_scale'] or row['scale_available'] != prepared['scale_available']:
                raise ValueError('saved learning scale provenance is inconsistent')
            result.pending[day] = json.loads(json.dumps(row, allow_nan=False))
        for row in value['rows']:
            day = date.fromisoformat(row['day'])
            released = date.fromisoformat(row['released_origin'])
            if result.last_origin is None or day > released - timedelta(days=2) or released > result.last_origin:
                raise ValueError('learning row not released by D-2')
            _vector(row['truth'], len(day_hours(day)), 'learning truth')
            result.rows[day] = json.loads(json.dumps(row, allow_nan=False))
            result.pending.pop(day)
        consumed = [date.fromisoformat(d) for d in value['consumed']]
        if len(consumed) != len(set(consumed)) or set(consumed) != set(result.rows):
            raise ValueError('learning consumed set differs from rows')
        result.consumed = set(consumed)
        return result


class ReplayState:
    """Persistent state throughout warm-up, searches, failures and restart."""
    def __init__(self, identity):
        if not identity:
            raise ValueError('replay requires a frozen identity')
        self.identity = json.loads(json.dumps(identity, allow_nan=False))
        self.policies, self.ledgers, self.last_origin = {}, {}, None

    def policy(self, route, policy, layer='H'):
        key = f'{route}/{policy}'
        if key not in self.policies:
            self.policies[key] = PolicyState(policy, route, self.identity, layer)
        if self.policies[key].layer != layer:
            raise ValueError('cannot change a policy interval recipe')
        return self.policies[key]

    def learning(self, route):
        if route not in self.ledgers:
            self.ledgers[route] = LearningLedger(route, self.identity)
        return self.ledgers[route]

    def release(self, origin, truth):
        if self.last_origin is not None and origin < self.last_origin:
            raise ValueError('replay origin cannot move backwards')
        for policy in self.policies.values():
            policy.release(origin, truth)
        for ledger in self.ledgers.values():
            ledger.release(origin, truth)
        self.last_origin = origin

    def to_dict(self):
        result = {'schema': SCHEMA, 'identity': self.identity, 'last_origin': str(self.last_origin) if self.last_origin else None,
                  'policies': {k: p.to_dict() for k, p in sorted(self.policies.items())},
                  'learning': {k: p.to_dict() for k, p in sorted(self.ledgers.items())}}
        result = json.loads(json.dumps(result, allow_nan=False))
        result['content_sha256'] = digest(result)
        return result

    @classmethod
    def from_dict(cls, value, expected_identity=None):
        if value.get('schema') != SCHEMA or expected_identity is not None and value['identity'] != expected_identity \
                or value.get('content_sha256') != digest({k: v for k, v in value.items() if k != 'content_sha256'}):
            raise ValueError('replay saved identity/schema mismatch')
        result = cls(value['identity'])
        result.last_origin = date.fromisoformat(value['last_origin']) if value['last_origin'] else None
        result.policies = {k: PolicyState.from_dict(v) for k, v in value['policies'].items()}
        result.ledgers = {k: LearningLedger.from_dict(v) for k, v in value['learning'].items()}
        if any(k != f'{p.route}/{p.policy}' or p.identity != result.identity or p.last_origin != result.last_origin
               for k, p in result.policies.items()):
            raise ValueError('saved route/policy identity mismatch')
        if any(k != p.route or p.identity != result.identity or p.last_origin != result.last_origin for k, p in result.ledgers.items()):
            raise ValueError('saved learning route identity mismatch')
        return result

    def save(self, path):
        _save(path, self.to_dict())

    @classmethod
    def load(cls, path, expected_identity=None):
        return cls.from_dict(json.loads(Path(path).read_text()), expected_identity)
