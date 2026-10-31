"""Ratio intervals at 97.5% and 95% for every §23.8 contrast (capstone v21-r11 §23.8), from the stored draws only.

§23.8 asks that ratio intervals be "reported at 97.5%, the decision level, and at 95% for comparison with CP-21 to
CP-23". The ratio is §17.5's R = S_candidate / S_comparator − 1 on the equal-fold B0-normalised scores. Attempt 1's
frozen scoring stored the 95% ratio interval in `uncertainty.csv` and the 2,000 equal-fold ratio draws in
`replicates.parquet`, but it computed its 97.5% intervals only for the score differences.

This module reads the committed equal-fold ratio draws and writes `attempt-<k>/ratio-intervals.csv` beside them,
with no rescoring and no bootstrap or reference pass. It changes no file of the frozen protocol and no committed
scoring table.

- **The 95% interval** is recomputed exactly as the scoring code computes it (`np.quantile` at 0.025 and 0.975,
  linear), and must equal the committed `ratio_ci_lower` and `ratio_ci_upper`.
- **The 97.5% interval** uses the levels `cp24.scoring.with_level` uses for the 97.5% difference intervals:
  (1 − 0.975)/2 and its complement, the 1.25% and 98.75% percentiles.
- **A contrast without a defined interval forecast** stays undefined: D2 − L in WIS, since L is point-only.

Run as ``python -m cp24.ratios --attempt <k>`` (writes) or ``--check``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from .scoring import LEVEL_DECISION

OUT = Path('reports/ddnn2')
COLUMNS = ['candidate', 'baseline', 'metric', 'role', 'defined', 'replicates', 'ratio', 'ratio_ci95_lower', 'ratio_ci95_upper',
           'ratio_ci97.5_lower', 'ratio_ci97.5_upper']


def build(root: Path, k: int) -> pd.DataFrame:
    ad = Path(root) / OUT / f'attempt-{k}'
    unc = pd.read_csv(ad / 'uncertainty.csv', float_precision='round_trip')
    eq = unc.loc[unc.scope.eq('equal_fold')]
    draws = pd.read_parquet(ad / 'replicates.parquet', filters=[('scope', '==', 'equal_fold')])
    keyed = {key: g.sort_values('replicate')['ratio'].to_numpy(float)
             for key, g in draws.groupby(['candidate', 'baseline', 'metric'])}
    lo = (1 - LEVEL_DECISION) / 2
    rows = []
    for _, r in eq.iterrows():
        d = keyed[(r.candidate, r.baseline, r.metric)]
        defined = bool(r.defined) and np.isfinite(r.ratio) and len(d) == int(r.replicates) and bool(np.isfinite(d).all())
        row = {'candidate': r.candidate, 'baseline': r.baseline, 'metric': r.metric, 'role': r.role, 'defined': defined,
               'replicates': int(len(d)), 'ratio': float(r.ratio) if defined else np.nan}
        if defined:
            c95 = tuple(float(x) for x in np.quantile(d, [.025, .975], method='linear'))
            if c95 != (float(r.ratio_ci_lower), float(r.ratio_ci_upper)):
                raise ValueError(f'{r.candidate}-{r.baseline} {r.metric}: the stored draws do not give the committed 95% ratio interval')
            c975 = tuple(float(x) for x in np.quantile(d, [lo, 1 - lo], method='linear'))
        else:
            c95 = c975 = (np.nan, np.nan)
        row.update({'ratio_ci95_lower': c95[0], 'ratio_ci95_upper': c95[1], 'ratio_ci97.5_lower': c975[0],
                    'ratio_ci97.5_upper': c975[1]})
        rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def text(frame: pd.DataFrame) -> str:
    return frame.to_csv(index=False, float_format='%.17g', lineterminator='\n')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', type=int, required=True, choices=(1, 2))
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args(argv)
    root = Path.cwd()
    path = root / OUT / f'attempt-{args.attempt}' / 'ratio-intervals.csv'
    out = text(build(root, args.attempt))
    if args.check:
        same = path.exists() and path.read_text() == out
        print(json.dumps({'ratio_intervals_identical_to_committed': same}))
        return 0 if same else 12
    path.write_text(out)
    print(f'wrote {path.relative_to(root)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
