"""DDNN-2's training-only search procedure: the space, the seeded sampler, successive halving, the
ranking and the fold's ensemble (capstone v21-r11 §23.4). NumPy and the standard library only
(§23.3: "including to the search"); the import audit covers this module.

**The validation batches** of fold f tile backward in consecutive 28-day blocks from D0_f - 57, the day
before the gate window. A block that meets any fold's warm-up or evaluation day, or that starts before
2020-01-01, is skipped; B_f = min(11, the blocks kept), the most recent.

**The sampler.** Seeded random search: each trial draws one configuration from the space below with
a PCG64 stream seeded by (round, fold). Successive halving: every trial runs on the min(4, B_f) most
recent batches; the best third (ceil(N/3)) of them, ranked over those batches, runs on all B_f. The
ranking metric is the mean seven-level pinball loss in EUR/MWh, after inversion, pooled over the
scored hours of the batches each trial ran; a failed fit ranks last. The fold's ensemble is the four
best distinct configurations among the trials that ran all B_f batches, ranked over those batches;
ties go to the network with fewer parameters, then to the earlier trial.
"""
from __future__ import annotations

from datetime import date, timedelta
import math

import numpy as np

from . import ddnn2 as M

BATCH_DAYS = 28
MAX_BATCHES = 11
HALVING_BATCHES = 4
EARLIEST_BATCH_START = date(2020, 1, 1)
ENSEMBLE_CONFIGS = 4
ENSEMBLE_SEEDS = 2
OPTIONAL_GROUPS = ('price_d2', 'price_d3', 'price_d7', 'load_d1', 'load_d7', 'gfs', 'stats', 'calendar')

#: Round 1's space, within §23.4's bounds. A later round may change it only as §23.6 allows.
SPACE_ROUND_1 = {
    'hidden_layers': {'choices': [1, 2]},
    'width': {'log_uniform': [16, 512], 'per_layer': True, 'round': 'nearest integer'},
    'activation': {'choices': list(M.ACTIVATIONS)},
    'input_dropout': {'off_probability': 0.5, 'uniform': [0.05, 0.5]},
    'l1': {'off_probability': 0.5, 'log_uniform': [1e-7, 1e-3]},
    'l2': {'off_probability': 0.5, 'log_uniform': [1e-6, 1e-2]},
    'lr': {'log_uniform': [1e-4, 1e-2]},
    'batch_size': {'choices': [32, 64, 128]},
    'kappa': {'choices': [1.0, 0.5, 0.0]},
    'half_life': {'choices': [None, 365, 180]},
    'groups': {'inclusion_probability': 0.5, 'flags': list(OPTIONAL_GROUPS)},
    'transform': {'choices': list(M.TRANSFORMS)},
}


def gate_window(d0: date) -> tuple[date, date]:
    """G_f = [D0_f - 56, D0_f), as (first day, last day)."""
    return d0 - timedelta(days=56), d0 - timedelta(days=1)


def batches(d0: date, fold_days: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Fold f's validation batches, most recent first: (first day, last day) of each kept block."""
    kept = []
    end = d0 - timedelta(days=57)
    while True:
        start = end - timedelta(days=BATCH_DAYS - 1)
        if start < EARLIEST_BATCH_START:
            break
        meets = any(not (end < a or start > b) for a, b in fold_days)
        if not meets:
            kept.append((start, end))
            if len(kept) == MAX_BATCHES:
                break
        end = start - timedelta(days=1)
    return kept


def _log_uniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample(rng: np.random.Generator, space: dict, trial_id: str) -> dict:
    """One configuration, drawn in a fixed order so a seed reproduces it exactly."""
    layers = int(rng.choice(space['hidden_layers']['choices']))
    lo, hi = space['width']['log_uniform']
    hidden = tuple(int(round(_log_uniform(rng, lo, hi))) for _ in range(layers))
    activation = str(rng.choice(space['activation']['choices']))

    def maybe(spec, draw):
        off = rng.uniform() < spec['off_probability']
        value = draw()
        return None if off else value

    dropout = maybe(space['input_dropout'], lambda: float(rng.uniform(*space['input_dropout']['uniform'])))
    l1 = maybe(space['l1'], lambda: _log_uniform(rng, *space['l1']['log_uniform']))
    l2 = maybe(space['l2'], lambda: _log_uniform(rng, *space['l2']['log_uniform']))
    lr = _log_uniform(rng, *space['lr']['log_uniform'])
    batch = int(rng.choice(space['batch_size']['choices']))
    kappa = float(space['kappa']['choices'][int(rng.integers(len(space['kappa']['choices'])))])
    half = space['half_life']['choices'][int(rng.integers(len(space['half_life']['choices'])))]
    flags = rng.uniform(size=len(space['groups']['flags'])) < space['groups']['inclusion_probability']
    groups = tuple(g for g, f in zip(space['groups']['flags'], flags) if f)
    transform = str(rng.choice(space['transform']['choices']))
    return {'id': trial_id, 'hidden': list(hidden), 'activation': activation, 'input_dropout': dropout, 'l1': l1,
            'l2': l2, 'lr': lr, 'batch_size': batch, 'kappa': kappa, 'half_life': half, 'groups': list(groups),
            'transform': transform}


def sampler_seed(round_number: int, fold_index: int) -> np.random.SeedSequence:
    return np.random.SeedSequence([24, int(round_number), int(fold_index), 0x5A3])


def trials(round_number: int, fold_index: int, n: int, space: dict) -> list[dict]:
    rng = np.random.Generator(np.random.PCG64(sampler_seed(round_number, fold_index)))
    return [sample(rng, space, f'r{round_number}-f{fold_index}-t{t:03d}') for t in range(n)]


def fit_seed(round_number: int, fold_index: int, trial: int, batch: int) -> int:
    return 24_000_000 + round_number * 1_000_000 + fold_index * 100_000 + trial * 100 + batch


def member_seeds(round_number: int, fold_index: int) -> list[int]:
    """The two seeds of each ensemble rank: 8 distinct member seeds per fold and round."""
    return [24_000_000 + round_number * 1_000_000 + fold_index * 100_000 + 90_000 + 10 * j + s
            for j in range(1, ENSEMBLE_CONFIGS + 1) for s in range(1, ENSEMBLE_SEEDS + 1)]


def n_params(config: dict, n_inputs: int) -> int:
    return M.n_parameters(n_inputs, tuple(config['hidden']))


def rank(results: dict, size: dict, order: list[str]) -> list[str]:
    """Trial ids from best to worst: metric, then fewer parameters, then trial order. `results[id]`
    is the pooled metric (inf for a failed trial); `size[id]` the network's parameter count."""
    def key(tid):
        metric = results[tid]
        return (metric if math.isfinite(metric) else math.inf, size[tid], order.index(tid))
    return sorted(results, key=key)


def halving_survivors(n_trials: int) -> int:
    return int(math.ceil(n_trials / 3))


def pooled(sums: dict, counts: dict, batch_ids) -> float:
    """Pooled mean over the given batches; inf if any batch failed."""
    total, count = 0.0, 0
    for b in batch_ids:
        if b not in sums or not math.isfinite(sums[b]):
            return math.inf
        total += sums[b]
        count += counts[b]
    return total / count if count else math.inf
