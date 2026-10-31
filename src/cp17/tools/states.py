"""The interval-layer states at any development origin, rebuilt from CP-26's stored ledger and checked.

CP-26 stored only its final replay state (2026-09-06). The state at an earlier origin D is a deterministic
function of the ledger, under `cp16.residuals.SharedResidualState` (H policies) and
`cp26.state.RawResidualState` (B0):

- an issuance s enters a policy's errors only if CP-26 recorded it `error_eligible`;
- its errors are (y - native central) / issued scale (H), or y - central (B0), with y CP-26's truth
  (`ctx.data.y` at the issuance keys);
- it is consumed at the first origin >= s + 2 (complete delivery day D-2, consumed once), which the final
  state's trace records per day;
- the buffer at D is the latest 28 consumed days, in day order.

Every rebuilt day's error hash must equal the hash the final state's consumption trace recorded for it, and
the buffer at D must match the buffer metadata CP-26's ledger recorded at D (days, hours, oldest and latest
day, span). A mismatch raises; nothing is approximated.
"""
from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np

from cp15.data import array_hash
from . import ledger

H_POLICIES = ('v2', 'v4', 'v5')
POLICIES = H_POLICIES + ('B0',)
LOOKBACK_DAYS = 120
TRACES = 'artifacts/cp-17/state-traces.json'


def traces(root) -> dict:
    """The final replay state's consumption traces, extracted once from the 357 MB file (and refreshed if absent)."""
    path = ledger.project_root(root) / '.local' / TRACES
    expected = ledger.verify_tag_file(root, ledger.REPLAY_STATE)
    if path.exists():
        value = json.loads(path.read_text())
        if value.get('replay_state_sha256') == expected:
            return value
    state = ledger.load_replay_state(root)
    value = {'replay_state_sha256': expected, 'content_sha256': state['content_sha256'],
             'last_origin': state['last_origin'], 'policies': {}}
    for policy in POLICIES:
        p = state['policies'][f'nominal/{policy}']
        value['policies'][f'nominal/{policy}'] = {
            'layer': p['layer'], 'last_origin': p['last_origin'],
            'consumed_traces': [t for t in p['errors']['trace'] if t['status'] == 'consumed']}
    path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
    return value


def _by_day(trace_rows: list[dict], layer: str) -> dict:
    key = 'feedback_day' if layer == 'H' else 'day'
    return {date.fromisoformat(t[key]): t for t in trace_rows}


def rebuild(root, ctx, origins) -> dict:
    """{policy: {origin: buffer}} for the requested origins, each buffer CP-26's list of {'day', 'errors'}."""
    origins = sorted(origins)
    tr = traces(root)
    consumed = {p: _by_day(tr['policies'][f'nominal/{p}']['consumed_traces'], 'B0' if p == 'B0' else 'H') for p in POLICIES}
    days = {p: {} for p in POLICIES}          # policy -> day -> (errors, consumed origin)
    meta = {p: {} for p in POLICIES}          # policy -> origin -> ledger buffer metadata at that origin
    data = ctx.data
    first = origins[0] - timedelta(days=LOOKBACK_DAYS)
    checked = {p: 0 for p in POLICIES}
    for d, value in ledger.iter_days(root, first=first, last=origins[-1]):
        for p in POLICIES:
            record = value['policies'][f'nominal/{p}']
            if d in origins:
                meta[p][d] = record['buffer']
            if not record['error_eligible']:
                continue
            trace = consumed[p].get(d)
            if trace is None:
                continue          # issued but never consumed by the end of the run (the last two days)
            index = ledger.keys(record['index'])
            pos = data.index.get_indexer(index)
            if (pos < 0).any():
                raise ValueError(f'{p} {d}: an issuance key is absent from the prepared inputs')
            y = data.y[pos]
            central = np.asarray(record['native_central'], float)
            if p == 'B0':
                errors = y - central
                expected = trace['errors_sha256']
            else:
                errors = (y - central) / np.asarray(record['issued_scale'], float)
                expected = trace['error_sha256']
            if not np.isfinite(errors).all() or array_hash(errors) != expected:
                raise ValueError(f'{p} {d}: the rebuilt errors differ from the hash CP-26 consumed')
            days[p][d] = (errors, date.fromisoformat(trace['origin']))
            checked[p] += 1
    out = {p: {} for p in POLICIES}
    for p in POLICIES:
        for origin in origins:
            released = sorted((d, e) for d, (e, at) in days[p].items() if at <= origin)[-28:]
            if len(released) != 28:
                raise ValueError(f'{p} at {origin}: {len(released)} released days, not 28')
            m = meta[p][origin]
            got = {'days': 28, 'hours': int(sum(len(e) for _, e in released)), 'oldest_day': str(released[0][0]),
                   'latest_day': str(released[-1][0]), 'calendar_span_days': (released[-1][0] - released[0][0]).days + 1}
            if any(m[k] != v for k, v in got.items()):
                raise ValueError(f'{p} at {origin}: rebuilt buffer {got} differs from the ledger record {m}')
            if any(d > origin - timedelta(days=2) for d, _ in released):
                raise ValueError(f'{p} at {origin}: a buffer day is not released under D-2')
            out[p][origin] = [{'day': str(d), 'errors': e.tolist()} for d, e in released]
    return {'buffers': out, 'checked_days': checked, 'replay_state_sha256': tr['replay_state_sha256'],
            'replay_state_content_sha256': tr['content_sha256']}


def buffer_sha256(buffer: list[dict]) -> str:
    return hashlib.sha256(json.dumps(buffer, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
