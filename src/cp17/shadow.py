"""The fallbacks' own interval states, kept current by daily shadow issuances (the Owner's decision 14).

Every delivery day, whether or not A1 succeeds, the issuance also runs the fallbacks as **shadow issuances**:

- **v4 and v2** are fitted under the issuing host, each in its own process. Their native central and issued
  scale are registered before the outcome (`cp16.residuals.SharedResidualState.issue`). Each policy's own errors
  from complete delivery days enter its own buffer once, after D-2's release (`release`).
- **B0** fits nothing. Its central is the similar-day naive of published prices, on B0's own forecastable keys,
  and it is registered the same way (`cp26.state.RawResidualState`). A rebuild from prices alone does not match
  CP-26: it would also take 36 days whose B0 keys the frozen design left partial (CP-17b's shadow check), so B0
  is shadow-issued like the others.

A shadow is never published, displayed or scored as the product's forecast. A fallback issued by the chain is
the same fit as its shadow, so it is fitted once. The intake conditions are CP-26's (`cp26.state.PolicyState
.issue`):
- only a complete canonical issuance (23, 24 or 25 keys) is registered;
- an H policy needs a finite central, a scale floored at 1, and the single-central identity;
- a failed fit registers nothing.

The states are the landed classes, so the arithmetic is CP-26's. The freeze-origin state is CP-26's own, rebuilt
from its ledger (`cp17.tools.states`).
"""
from __future__ import annotations

from datetime import date
import json

import numpy as np

H_POLICIES = ('v4', 'v2')
POLICIES = H_POLICIES + ('B0',)
SCHEMA = 'cp17b-shadow-states-v1'


def _complete(day: date, index) -> bool:
    from .issue import canonical_hours
    return [t.isoformat() for t in index] == canonical_hours(day)


class ShadowStates:
    """v4's and v2's H states and B0's raw-error state, carried day by day."""

    def __init__(self, h: dict, b0):
        self.h, self.b0 = h, b0

    @classmethod
    def from_buffers(cls, origin: date, buffers: dict, pending: dict | None = None):
        """States at `origin`, after its release: the released buffers `buffers[policy]` (CP-26's list of
        {'day', 'errors'}) and the issuances registered but not yet released, `pending[policy]` (a list of
        {'day', 'index', 'central'[, 'scale']}, the central as registered)."""
        from cp16.residuals import SharedResidualState
        from cp26.state import RawResidualState
        pending = pending or {}
        h = {p: SharedResidualState.from_dict({'schema': 'cp16-shared-residuals-v1', 'last_origin': str(origin),
                                               'pending': pending.get(p, []), 'consumed': [b['day'] for b in buffers[p]],
                                               'buffer': buffers[p], 'trace': []}) for p in H_POLICIES}
        b0 = RawResidualState.from_dict({'last_origin': str(origin), 'pending': pending.get('B0', []), 'trace': [],
                                         'consumed': [b['day'] for b in buffers['B0']], 'buffer': buffers['B0']})
        return cls(h, b0)

    def release(self, origin: date, truth) -> None:
        """Release every pending issuance of a complete delivery day <= origin - 2 into its own buffer, once."""
        for state in (*self.h.values(), self.b0):
            state.release(origin, truth)

    def issue(self, policy: str, day: date, result: dict | None) -> str:
        """Register a shadow (or issued fallback) fit before its outcome; returns what happened."""
        import pandas as pd
        if result is None:
            return 'no intake: the fit failed'
        index = pd.DatetimeIndex(pd.to_datetime(result['index'], utc=True))
        if not _complete(day, index):
            return 'no intake: not a complete canonical issuance'
        central = np.asarray(result['central'], float)
        if not np.isfinite(central).all():
            return 'no intake: nonfinite central'
        if policy == 'B0':
            self.b0.issue(day, index, central)
            return 'registered'
        scale = np.asarray(result['scale'], float)
        if not np.isfinite(scale).all() or (scale < 1).any():
            return 'no intake: invalid issued scale'
        if not np.array_equal(central / 2 + central / 2, central):
            return 'no intake: single-central identity underflowed'
        self.h[policy].issue(day, index, central, central, scale)
        return 'registered'

    def buffer(self, policy: str) -> list[dict]:
        """The released buffer a payload ships (`dailyfit`'s `h_buffer` or `b0_buffer`)."""
        if policy == 'B0':
            return [{'day': str(d), 'errors': e.tolist()} for d, e in self.b0.buffer]
        return [{'day': str(d), 'errors': e.tolist()} for d, e in self.h[policy]._buffer]

    def to_dict(self) -> dict:
        return {'schema': SCHEMA, 'h': {p: s.to_dict() for p, s in self.h.items()}, 'b0': self.b0.to_dict()}

    @classmethod
    def from_dict(cls, value: dict):
        from cp16.residuals import SharedResidualState
        from cp26.state import RawResidualState
        if value.get('schema') != SCHEMA:
            raise ValueError('not a shadow-states record')
        return cls({p: SharedResidualState.from_dict(v) for p, v in value['h'].items()}, RawResidualState.from_dict(value['b0']))

    def digest(self) -> dict:
        """Each policy's released buffer as one SHA-256, for comparison with a recorded state."""
        import hashlib
        return {p: hashlib.sha256(json.dumps(self.buffer(p), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                for p in POLICIES}
