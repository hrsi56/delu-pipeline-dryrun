"""HG's A1_w/B2_w LEAR components, refitted for CP-21's controls and daily-cycle diagnostic.

The inherited CP-15 `fit_day` on CP-20's weather-augmented design (`cp20.components.augment`),
unchanged; every component-day attempt and every primitive Lasso call (continuations included)
is charged to the CP-21 ledger before it runs. CP-21 never uses a refit in place of the verified
CP-20 cache: a refit is a check against it.
"""
from __future__ import annotations

import contextlib

from cp20.components import augment  # noqa: F401  (re-exported: the weather-augmented LEAR design)

POLICIES = ('A1', 'B2')


@contextlib.contextmanager
def counted_lear(budget, purpose: str):
    import cp15.models as m
    original_day, original_lasso, original_fit = m.fit_day, m._lasso, m.Lasso.fit
    state = {'ordinal': 0, 'kind': None}

    def day(*args, **kwargs):
        if args[2] not in POLICIES:
            raise ValueError('only the two inherited A1/B2 components are refitted')
        budget.reserve(component_attempts=1, **{f'component_attempts_{purpose}': 1})
        state['ordinal'], state['kind'] = 0, None
        return original_day(*args, **kwargs)

    def lasso(*args, **kwargs):
        state['kind'] = 'lasso_final_fits' if state['ordinal'] % 5 == 4 else 'lasso_inner_fits'
        state['ordinal'] += 1
        return original_lasso(*args, **kwargs)

    def fit(self, *args, **kwargs):
        if state['kind'] is None:
            raise RuntimeError('uncategorised estimator attempt')
        budget.reserve(primitive_fits=1, **{state['kind']: 1})
        return original_fit(self, *args, **kwargs)

    m.fit_day, m._lasso, m.Lasso.fit = day, lasso, fit
    try:
        yield m.fit_day
    finally:
        m.fit_day, m._lasso, m.Lasso.fit = original_day, original_lasso, original_fit


def refit(budget, purpose: str, data_aug, present, day, rows):
    """Fresh A1_w and B2_w central forecasts for `rows` of `day` (EUR/MWh), exactly as CP-20's
    `fit_hg` computed them (same design, same weather-record check, same inherited `fit_day`)."""
    from cp20.components import training_rows
    out = {}
    with counted_lear(budget, purpose) as fit_day:
        for policy in POLICIES:
            train = training_rows(data_aug, day, policy)
            if not present[train].all() or not present[rows].all():
                raise ValueError(f'{day}: a training/forecast row lacks a frozen weather record')
            pred, _ = fit_day(data_aug, day, policy, rows=rows)
            out[policy] = pred
    return out
