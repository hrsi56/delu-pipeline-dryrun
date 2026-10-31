"""Item 2's B0 check: does B0's raw-error buffer, rebuilt from published prices alone, equal CP-26's?

The Owner's decision 14 says B0 needs no fit: its raw-error buffer comes from published prices. The brief asks to
show that a rebuild this way equals CP-26's `RawResidualState` exactly, and to shadow-issue B0 too if it does
not.

**The rebuild from prices.** For every delivery day d whose canonical hours (23, 24 or 25) all have a published
price and a similar-day naive (`delu_forecast.baselines.similar_day_naive`: D-1 Tue-Fri, D-7 Mon/Sat/Sun), the
day's raw errors are price - naive on those hours. The buffer at origin D is the latest 28 such days <= D-2.

**CP-26's state.** The days its final replay state consumed into B0's buffer (the consumption trace of
`nominal/B0`), each with the hash of its errors and the origin that consumed it (`cp17.tools.states.traces`).

The check compares the two day by day (the error hashes on common days; the days only one side holds, with
CP-26's own record of why) and origin by origin (the 28-day buffers at every origin CP-26 ran).

    OPENBLAS_NUM_THREADS=1 ... PYTHONPATH=src python -m cp17.tools.b0check --out reports/cp17b/b0-rebuild.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cp15.data import array_hash, day_hours
from . import ledger, payloads, states


def from_prices(data) -> dict[date, np.ndarray]:
    """{day: price - naive on its canonical hours} for every day both are complete."""
    from delu_forecast.baselines import similar_day_naive
    frame = pd.DataFrame({'timestamp_utc': data.index, 'price_eur_mwh': np.asarray(data.y, float)})
    frame = frame[np.isfinite(frame['price_eur_mwh'])]
    naive = similar_day_naive(frame)
    price = pd.Series(frame['price_eur_mwh'].to_numpy(), index=pd.DatetimeIndex(frame['timestamp_utc']))
    first, last = data.dates.min().astype('datetime64[D]').item(), data.dates.max().astype('datetime64[D]').item()
    out = {}
    for d in pd.date_range(first, last, freq='D').date:
        hours = day_hours(d).as_unit('ns')
        y = price.reindex(hours).to_numpy(float)
        c = naive.reindex(hours).to_numpy(float)
        if np.isfinite(y).all() and np.isfinite(c).all():
            out[d] = y - c
    return out


def check(root: Path) -> dict:
    ctx = payloads.context(root)
    rebuilt = from_prices(ctx.data)
    traces = states.traces(root)['policies']['nominal/B0']['consumed_traces']
    consumed = {date.fromisoformat(t['day']): (t['errors_sha256'], date.fromisoformat(t['origin'])) for t in traces}
    days = {d: v for d, v in ledger.iter_days(root)}
    ledger_days = sorted(days)
    span = (ledger_days[0], ledger_days[-1] - timedelta(days=2))
    common = sorted(set(consumed) & set(rebuilt))
    differing = [str(d) for d in common if array_hash(rebuilt[d]) != consumed[d][0]]
    only_prices = sorted(d for d in rebuilt if span[0] <= d <= span[1] and d not in consumed)
    only_cp26 = sorted(d for d in consumed if d not in rebuilt)

    def why(d):
        r = days.get(d, {}).get('policies', {}).get('nominal/B0')
        if r is None:
            return 'no CP-26 ledger day'
        n, full = len(r['index']), len(day_hours(d))
        if n == 0:
            return 'CP-26 issued no B0 key (no forecastable hour)'
        if n < full:
            return f'CP-26 issued {n} of {full} keys (a partial issuance is not taken into the buffer)'
        return f"CP-26 issued all keys; error_eligible={r['error_eligible']}"
    reasons = {str(d): why(d) for d in only_prices}
    origins = [d for d in ledger_days if d >= ledger_days[0] + timedelta(days=60)]
    differ = []
    for o in origins:
        mine = sorted(d for d in rebuilt if d <= o - timedelta(days=2))[-28:]
        theirs = sorted(d for d, (_, at) in consumed.items() if at <= o)[-28:]
        if mine != theirs:
            differ.append(str(o))
    equal = not differing and not only_prices and not only_cp26 and not differ
    return {'schema': 'cp17b-b0-rebuild-v1',
            'question': "does B0's raw-error buffer rebuilt from published prices equal CP-26's RawResidualState?",
            'cp26_consumed_days': len(consumed), 'price_rebuild_days_in_span': sum(span[0] <= d <= span[1] for d in rebuilt),
            'span': [str(span[0]), str(span[1])], 'common_days': len(common),
            'common_days_bitwise_equal': len(common) - len(differing), 'common_days_differing': differing,
            'days_only_in_price_rebuild': [str(d) for d in only_prices], 'their_cp26_record': reasons,
            'reasons_counted': dict(Counter(v.split(' (')[0] for v in reasons.values())),
            'days_only_in_cp26': [str(d) for d in only_cp26],
            'origins_compared': len(origins), 'origins_whose_28_day_buffer_differs': len(differ),
            'first_differing_origins': differ[:10],
            'equal': equal,
            'conclusion': ('equal: B0 needs no shadow' if equal else
                           'not equal: B0 is shadow-issued too, as the brief prescribes (cp17.shadow, cp17.daily)'),
            'replay_state_sha256': states.traces(root)['replay_state_sha256']}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    record = check(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True) + '\n')
    print(record['conclusion'], {k: record[k] for k in ('common_days', 'common_days_bitwise_equal', 'reasons_counted',
                                                         'origins_whose_28_day_buffer_differs')})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
