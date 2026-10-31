"""The step-0.01 simplex grid and its fits: A3/A6 globally, A4 per block (protocol §4).

Points are enumerated i = 0..100, then j = 0..100−i, with w = (i, j, 100−i−j)/100 in the fixed
member order (c_HG, L, D2); the zero-based index is the anchor's lexicographic (w_HG, w_L) order.

Arithmetic, fixed so that exact ties are reproducible: a weighted central is
`(w_HG·c_HG + w_L·L) + w_D2·D2`, left to right in float64, and its error against the outcome is
taken in absolute value. A date's loss for every grid point is the sequential sum of those hourly
errors in ascending canonical-UTC order, divided by the date's (or block's) actual hour count.
The objective at an origin is the sequential sum of the window dates' losses in ascending
delivery-date order, divided by the number of dates. Each date therefore weighs equally,
including DST dates. An objective tie is exact float equality, broken by the integer squared
distance to v5's weights `(6i−400)² + (6j−100)² + (6(100−i−j)−100)²`, then by the lowest index.
"""
from __future__ import annotations

import numpy as np

STEP = 100
I, J = np.array([(i, j) for i in range(STEP + 1) for j in range(STEP + 1 - i)], dtype=np.int64).T
K = STEP - I - J
POINTS = len(I)                                   # 5,151
W_HG, W_L, W_D2 = I / STEP, J / STEP, K / STEP    # float64 coordinates, never rounded
WEIGHTS = np.stack([W_HG, W_L, W_D2], axis=1)
W0 = np.array([2 / 3, 1 / 6, 1 / 6])
DISTANCE = (6 * I - 400) ** 2 + (6 * J - 100) ** 2 + (6 * K - 100) ** 2   # ∝ squared distance to W0
BLOCKS = ('night', 'solar', 'shoulder_peak')
BLOCK_HOURS = {'night': (22, 23, 0, 1, 2, 3, 4, 5), 'solar': tuple(range(10, 17)),
               'shoulder_peak': (6, 7, 8, 9, 17, 18, 19, 20, 21)}
HOUR_BLOCK = {hour: b for b, hours in BLOCK_HOURS.items() for hour in hours}
assert POINTS == 5151 and sorted(HOUR_BLOCK) == list(range(24))


def weighted_central(weights, c_hg, l, d2):
    """The fixed combination arithmetic, shared by the fit objective and the deployed arm."""
    w = np.asarray(weights, float)
    if w.ndim == 1:
        return (w[0] * c_hg + w[1] * l) + w[2] * d2
    return (w[:, 0] * c_hg + w[:, 1] * l) + w[:, 2] * d2          # per-hour weights, shape (H, 3)


def date_losses(c_hg, l, d2, y, blocks):
    """Every grid point's mean absolute central error for one date, overall and per block.

    ``blocks`` holds each hour's block name in canonical UTC order. Returns (overall (5151,),
    {block: (5151,)}). Hours are accumulated one at a time, in order.
    """
    total = np.zeros(POINTS)
    per_block = {b: np.zeros(POINTS) for b in BLOCKS}
    counts = dict.fromkeys(BLOCKS, 0)
    for h in range(len(y)):
        error = np.abs((W_HG * c_hg[h] + W_L * l[h]) + W_D2 * d2[h] - y[h])
        total += error
        per_block[blocks[h]] += error
        counts[blocks[h]] += 1
    if any(count == 0 for count in counts.values()):
        raise ValueError('a complete date must contain every block')
    return total / len(y), {b: per_block[b] / counts[b] for b in BLOCKS}


def choose(objective):
    """Exact-minimum point, ties by integer distance to W0 then lowest index."""
    objective = np.asarray(objective, float)
    if objective.shape != (POINTS,) or not np.isfinite(objective).all():
        raise FloatingPointError('grid objective is not finite')
    best = objective.min()
    tied = np.flatnonzero(objective == best)
    nearest = tied[DISTANCE[tied] == DISTANCE[tied].min()]
    index = int(nearest[0])
    return {'index': index, 'i': int(I[index]), 'j': int(J[index]), 'k': int(K[index]),
            'weights': WEIGHTS[index].copy(), 'objective': float(best),
            'objective_ties': int(len(tied)), 'distance_ties': int(len(nearest))}


def window_objective(losses, rows):
    """Sequential ascending-date sum of the window rows' date losses, divided by their count."""
    acc = np.zeros(POINTS)
    for r in rows:
        acc += losses[r]
    return acc / len(rows)
