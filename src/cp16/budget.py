"""Persistent pre-call ceilings and monitored local execution, shared with review."""
from __future__ import annotations
import contextlib
import fcntl
import json
import os
from pathlib import Path
import time

CAPS = {'component_attempts': 2000, 'main_component_attempts': 1500,
        'primitive_fits': 240000, 'inner_fits': 192000, 'final_fits': 48000,
        'policy_days': 9000, 'reference_passes': 3, 'analysis_passes': 3,
        'machine_seconds': 86400, 'active_seconds': 144000,
        'rss_bytes': 10 * 1024**3, 'additional_disk_bytes': 20 * 1024**3}


def atomic(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=str)+'\n')
    os.replace(temp, path)


class Budget:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def transaction(self):
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.path.read_text()) if self.path.exists() else {
                'caps': CAPS, 'counts': {}, 'events': [], 'jobs': [], 'created_epoch': time.time()}
            if state['caps'] != CAPS: raise ValueError('resource contract changed')
            yield state
            atomic(self.path, state)

    def reserve(self, **increments):
        with self.transaction() as s:
            for k, v in increments.items():
                if v < 0 or k not in CAPS: raise ValueError('invalid resource counter')
                if s['counts'].get(k, 0) + v > CAPS[k]:
                    raise RuntimeError('CP-16 hard cap refuses next operation: '+k)
            for k, v in increments.items(): s['counts'][k] = s['counts'].get(k, 0) + v

    def event(self, name, **fields):
        with self.transaction() as s: s['events'].append({'event': name, 'epoch': time.time(), **fields})

    def read(self): return json.loads(self.path.read_text())


@contextlib.contextmanager
def counted_fits(budget, main=False):
    """Count every primitive solver continuation, including failures, before it runs."""
    import cp15.models as m
    original_day, original_lasso, original_fit = m.fit_day, m._lasso, m.Lasso.fit
    state = {'ordinal': 0, 'kind': None}
    def day(*args, **kwargs):
        if args[2] not in ('A1','B2'): raise ValueError('only the two inherited components allowed')
        increments = {'component_attempts': 1}
        if main: increments['main_component_attempts'] = 1
        budget.reserve(**increments); state['ordinal'] = 0
        return original_day(*args, **kwargs)
    def lasso(*args, **kwargs):
        state['kind'] = 'final_fits' if state['ordinal'] % 5 == 4 else 'inner_fits'
        state['ordinal'] += 1
        return original_lasso(*args, **kwargs)
    def fit(self, *args, **kwargs):
        if state['kind'] is None: raise RuntimeError('uncategorized estimator attempt')
        budget.reserve(primitive_fits=1, **{state['kind']: 1})
        return original_fit(self, *args, **kwargs)
    m.fit_day, m._lasso, m.Lasso.fit = day, lasso, fit
    try: yield m.fit_day
    finally: m.fit_day, m._lasso, m.Lasso.fit = original_day, original_lasso, original_fit
