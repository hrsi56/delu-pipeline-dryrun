"""Metered throughput probe (feasibility only): non-target byte ranges from the two named
endpoints, single connection then concurrent, so no target-message attempt is used."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time

from .budget import Budget, GIB
from .gfs import aws_url, ncar_url
from .net import Fetcher

RANGE = 6_000_000


def main():
    out = Path(sys.argv[1])
    budget = Budget(os.environ['CP20_LEDGER'])
    fetcher = Fetcher(budget, out / 'probe-requests.jsonl', int(150 * GIB))
    run = dt.date(2023, 5, 1)
    results = []

    def one(label, url, offset):
        began = time.time()
        _, _, body = fetcher.get(url, f'throughput probe {label}', byte_range=(offset, offset + RANGE - 1))
        return len(body) / (time.time() - began) / 1e6

    for label, url in (('aws', aws_url(run, 24)), ('ncar', ncar_url(run, 24))):
        single = one(f'{label} single', url, 0)
        began = time.time()
        with ThreadPoolExecutor(3) as pool:
            rates = list(pool.map(lambda k: one(f'{label} parallel {k}', url, (k + 1) * RANGE), range(3)))
        wall = time.time() - began
        results.append({'endpoint': label, 'single_connection_MBps': single, 'parallel_per_connection_MBps': rates,
                        'parallel_aggregate_MBps': 3 * RANGE / wall / 1e6})
        print(json.dumps(results[-1]), flush=True)
    (out / 'throughput-probe.json').write_text(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                                           'range_bytes': RANGE, 'results': results}, indent=1))


if __name__ == '__main__':
    main()
