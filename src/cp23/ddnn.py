"""DDNN: a feed-forward network with a Johnson SU distributional head, in NumPy only.

Capstone v21-r10 §21.2-§21.3 and §18.2. This module is DDNN's single implementation: the network,
the Johnson SU head and its likelihood, the analytic gradients, the optimizer, regularisation, the
training loop, early stopping, seeding, ensembling, the training-only configuration choice and the
emission of quantiles and the p50. It imports NumPy and the Python standard library only; the
import audit in `tests/cp23/test_numpy_only.py` enforces that, with a fixture that fails it.

**Network.** Inputs -> hidden layers (ELU) -> four linear outputs per row. One row is one delivery
hour (the representation is fixed in `cp23.features`). The outputs map to the Johnson SU parameters

    xi = o1,  lambda = softplus(o2) + LAMBDA_FLOOR,  gamma = o3,  delta = softplus(o4) + DELTA_FLOOR,

for the variable  Y = xi + lambda * sinh((Z - gamma) / delta),  Z ~ N(0, 1)  (Johnson, 1949), so
the quantile at level q is  xi + lambda * sinh((Phi^-1(q) - gamma) / delta)  and is strictly
increasing in q.

**Loss.** The mean Johnson SU negative log-likelihood of §4's normalised target over the batch, plus
`l2 * sum(W**2)` over every weight matrix (biases unpenalised). Gradients are analytic; finite
differences check them in `tests/cp23/test_ddnn_gradients.py`, and PyTorch checks the forward pass,
the likelihood, its gradients and short optimizer trajectories in the explicitly invoked
`tests/cp23/torch_reference_checks.py` (§21.3).

**Optimizer.** Adam (Kingma and Ba, 2015) with PyTorch's update order: `m += (1 - b1)(g - m)`,
`v = b2 v + (1 - b2) g^2`, `p -= (lr / (1 - b1^t)) * m / (sqrt(v) / sqrt(1 - b2^t) + eps)`.

**Training.** Glorot-uniform weights and zero biases from a seeded PCG64 generator, except the output
biases, which start lambda and delta at 1. Mini-batches of `batch_size` rows in a fresh seeded
permutation every epoch. After every epoch the validation negative log-likelihood (no penalty) is
computed on the inner-validation rows -- the window's last 28 calendar delivery days, training data
only -- and the weights of the best epoch are kept. Training stops after `patience` epochs without a
strict improvement, or at `max_epochs`. The kept weights are the member: one fit per member.

**Ensemble and emission.** A fixed number of seeds per origin. Members combine by averaging their
quantiles at each level (Lichtendahl et al., 2013), which keeps every row's quantiles ordered. The
emitted p50 and the central forecast D are the ensemble's median (the average of the members'
medians). Quantiles are inverted to EUR/MWh with each row's own origin level and scale (§4).
Crossings, which averaging cannot create, would be restored by sorting and every one recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from statistics import NormalDist
import time

import numpy as np

#: The seven CP-15 levels; the p50 is the fourth.
LEVELS = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
MEDIAN = LEVELS.index(0.50)
Z_LEVELS = np.array([NormalDist().inv_cdf(q) for q in LEVELS])
LOG_2PI = math.log(2.0 * math.pi)
LAMBDA_FLOOR = 1e-3
DELTA_FLOOR = 0.05

#: The frozen configuration set, smallest to largest (parameter count strictly increasing for any
#: input width). "Smaller" for the selection tie rule is this order.
CONFIGS: tuple[dict, ...] = (
    {'id': 'C1', 'hidden': (32,)},
    {'id': 'C2', 'hidden': (32, 32)},
    {'id': 'C3', 'hidden': (64, 64)},
    {'id': 'C4', 'hidden': (128, 128)},
)
#: The fixed ensemble: four seeds at every origin (§21.2's maximum).
SEEDS = (42, 43, 44, 45)
#: The frozen training recipe (no setting is chosen from outcomes).
TRAINING = {'lr': 1e-3, 'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8, 'batch_size': 256,
            'max_epochs': 400, 'patience': 25, 'l2': 1e-4}


class TrainingFailure(RuntimeError):
    """A nonfinite loss or forecast. Recorded as a failed fit, never substituted."""


# ------------------------------------------------------------------ elementary functions
def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


def elu_grad(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, 1.0, np.exp(np.minimum(x, 0.0)))


OUTPUT_BIAS = np.array([0.0, inverse_softplus(1.0 - LAMBDA_FLOOR), 0.0, inverse_softplus(1.0 - DELTA_FLOOR)])


# ------------------------------------------------------------------ the network
def layer_sizes(n_inputs: int, hidden: tuple[int, ...]) -> list[int]:
    return [int(n_inputs), *map(int, hidden), 4]


def init_params(n_inputs: int, hidden: tuple[int, ...], rng: np.random.Generator) -> list[np.ndarray]:
    """[W1, b1, ..., W_out, b_out]; Glorot-uniform weights, zero biases, output biases at OUTPUT_BIAS."""
    sizes = layer_sizes(n_inputs, hidden)
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        params.append(rng.uniform(-bound, bound, size=(fan_in, fan_out)))
        params.append(np.zeros(fan_out))
    params[-1] = OUTPUT_BIAS.copy()
    return params


def n_parameters(n_inputs: int, hidden: tuple[int, ...]) -> int:
    sizes = layer_sizes(n_inputs, hidden)
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


def forward(params: list[np.ndarray], x: np.ndarray):
    """Raw outputs (n x 4) and the cache the backward pass needs."""
    h = x
    inputs, pre = [x], []
    for k in range(len(params) // 2 - 1):
        a = h @ params[2 * k] + params[2 * k + 1]
        pre.append(a)
        h = elu(a)
        inputs.append(h)
    out = h @ params[-2] + params[-1]
    return out, (inputs, pre)


def head(out: np.ndarray):
    """The Johnson SU parameters (xi, lambda, gamma, delta) of every row."""
    return out[:, 0], softplus(out[:, 1]) + LAMBDA_FLOOR, out[:, 2], softplus(out[:, 3]) + DELTA_FLOOR


def jsu_nll(y, xi, lam, gamma, delta):
    """Per-row Johnson SU negative log-likelihood."""
    u = (y - xi) / lam
    r = gamma + delta * np.arcsinh(u)
    return -np.log(delta) + np.log(lam) + 0.5 * LOG_2PI + 0.5 * np.log1p(u * u) + 0.5 * r * r


def jsu_nll_grad(y, xi, lam, gamma, delta):
    """Per-row negative log-likelihood and its analytic partial derivatives in (xi, lambda, gamma, delta)."""
    u = (y - xi) / lam
    asinh_u = np.arcsinh(u)
    r = gamma + delta * asinh_u
    one_u2 = 1.0 + u * u
    nll = -np.log(delta) + np.log(lam) + 0.5 * LOG_2PI + 0.5 * np.log1p(u * u) + 0.5 * r * r
    d_u = u / one_u2 + r * delta / np.sqrt(one_u2)
    return nll, (-d_u / lam, (1.0 - u * d_u) / lam, r, -1.0 / delta + r * asinh_u)


def penalty(params: list[np.ndarray], l2: float) -> float:
    return l2 * sum(float(np.sum(w * w)) for w in params[0::2])


def loss_and_grads(params: list[np.ndarray], x: np.ndarray, y: np.ndarray, l2: float):
    """Mean negative log-likelihood plus the L2 penalty, and its gradient for every parameter."""
    out, (inputs, pre) = forward(params, x)
    xi, lam, gamma, delta = head(out)
    nll, (g_xi, g_lam, g_gamma, g_delta) = jsu_nll_grad(y, xi, lam, gamma, delta)
    n = len(y)
    d_out = np.empty_like(out)
    d_out[:, 0] = g_xi / n
    d_out[:, 1] = g_lam * sigmoid(out[:, 1]) / n
    d_out[:, 2] = g_gamma / n
    d_out[:, 3] = g_delta * sigmoid(out[:, 3]) / n
    grads: list[np.ndarray] = [None] * len(params)  # type: ignore[list-item]
    grads[-2] = inputs[-1].T @ d_out + 2.0 * l2 * params[-2]
    grads[-1] = d_out.sum(axis=0)
    dh = d_out @ params[-2].T
    for k in range(len(params) // 2 - 2, -1, -1):
        da = dh * elu_grad(pre[k])
        grads[2 * k] = inputs[k].T @ da + 2.0 * l2 * params[2 * k]
        grads[2 * k + 1] = da.sum(axis=0)
        if k:
            dh = da @ params[2 * k].T
    return float(np.mean(nll)) + penalty(params, l2), grads


def mean_nll(params: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> float:
    out, _ = forward(params, x)
    return float(np.mean(jsu_nll(y, *head(out))))


def jsu_quantiles(params: list[np.ndarray], x: np.ndarray) -> np.ndarray:
    """One member's quantiles of the normalised target at the seven levels (n x 7)."""
    out, _ = forward(params, x)
    xi, lam, gamma, delta = head(out)
    return xi[:, None] + lam[:, None] * np.sinh((Z_LEVELS[None, :] - gamma[:, None]) / delta[:, None])


class Adam:
    """Adam with PyTorch's update order (torch.optim.Adam, no weight decay, no amsgrad)."""

    def __init__(self, params: list[np.ndarray], lr: float, beta1: float, beta2: float, eps: float):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1 ** self.t
        bias2_sqrt = math.sqrt(1.0 - self.beta2 ** self.t)
        step_size = self.lr / bias1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m += (g - m) * (1.0 - self.beta1)
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            p -= step_size * (m / (np.sqrt(v) / bias2_sqrt + self.eps))


def params_sha256(params: list[np.ndarray]) -> str:
    h = hashlib.sha256()
    for p in params:
        h.update(np.ascontiguousarray(p, dtype='<f8').tobytes())
    return h.hexdigest()


@dataclass
class Member:
    """One trained network: the best-epoch weights and how it was trained."""
    config: str
    seed: int
    params: list[np.ndarray]
    best_epoch: int
    epochs_run: int
    best_validation_nll: float
    history: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0

    def record(self) -> dict:
        return {'config': self.config, 'seed': self.seed, 'best_epoch': self.best_epoch, 'epochs_run': self.epochs_run,
                'best_validation_nll': self.best_validation_nll, 'params_sha256': params_sha256(self.params),
                'fit_wall_seconds': self.wall_seconds, 'fit_cpu_seconds': self.cpu_seconds,
                'validation_nll_history': self.history, 'train_loss_history': self.train_loss}


def train_member(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, config: dict,
                 seed: int, settings: dict | None = None) -> Member:
    """Train one network with early stopping on the inner-validation rows; keep the best epoch."""
    s = dict(TRAINING if settings is None else settings)
    if not (np.isfinite(x_train).all() and np.isfinite(y_train).all() and np.isfinite(x_val).all() and np.isfinite(y_val).all()):
        raise TrainingFailure('nonfinite training or validation input')
    if not len(y_train) or not len(y_val):
        raise TrainingFailure('empty training or validation rows')
    wall, cpu = time.perf_counter(), time.process_time()
    rng = np.random.Generator(np.random.PCG64(seed))
    params = init_params(x_train.shape[1], tuple(config['hidden']), rng)
    opt = Adam(params, s['lr'], s['beta1'], s['beta2'], s['eps'])
    n, bs = len(y_train), int(s['batch_size'])
    best, best_params, best_epoch, wait = math.inf, [p.copy() for p in params], 0, 0
    history, train_loss = [], []
    epoch = 0
    for epoch in range(1, int(s['max_epochs']) + 1):
        order = rng.permutation(n)
        total = 0.0
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            loss, grads = loss_and_grads(params, x_train[idx], y_train[idx], s['l2'])
            if not math.isfinite(loss):
                raise TrainingFailure(f'nonfinite training loss at epoch {epoch}')
            opt.step(params, grads)
            total += loss * len(idx)
        val = mean_nll(params, x_val, y_val)
        if not math.isfinite(val):
            raise TrainingFailure(f'nonfinite validation loss at epoch {epoch}')
        history.append(val)
        train_loss.append(total / n)
        if val < best:
            best, best_params, best_epoch, wait = val, [p.copy() for p in params], epoch, 0
        else:
            wait += 1
            if wait >= int(s['patience']):
                break
    return Member(config['id'], int(seed), best_params, best_epoch, epoch, float(best), history, train_loss,
                  time.perf_counter() - wall, time.process_time() - cpu)


# ------------------------------------------------------------------ inputs (§15.3, on training rows only)
class Preprocessor:
    """§15.3 fitted on one training partition: weather medians (all-null -> 0) with one missing
    indicator per weather column, then a StandardScaler over every input column (a zero-variance
    column keeps scale 1). Nothing is fitted on validation or forecast rows."""

    def __init__(self, x: np.ndarray, wx: np.ndarray):
        with np.errstate(all='ignore'):
            fill = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else np.nan for c in wx.T])
        self.fill = np.where(np.isfinite(fill), fill, 0.0)
        full = self._raw(x, wx)
        self.mean = full.mean(axis=0)
        sd = full.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)

    def _raw(self, x: np.ndarray, wx: np.ndarray) -> np.ndarray:
        missing = ~np.isfinite(wx)
        return np.column_stack((x, np.where(missing, self.fill, wx), missing.astype(float)))

    def transform(self, x: np.ndarray, wx: np.ndarray) -> np.ndarray:
        out = (self._raw(x, wx) - self.mean) / self.sd
        if not np.isfinite(out).all():
            raise TrainingFailure('nonfinite transformed input')
        return out

    def record(self) -> dict:
        return {'weather_fill': self.fill.tolist(), 'mean_sha256': hashlib.sha256(self.mean.tobytes()).hexdigest(),
                'sd_sha256': hashlib.sha256(self.sd.tobytes()).hexdigest()}


# ------------------------------------------------------------------ ensemble and emission
def ensemble_quantiles(members: list[Member], x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantile-averaged ensemble quantiles in z (n x 7) and each member's own (members x n x 7)."""
    each = np.stack([jsu_quantiles(m.params, x) for m in members])
    return each.mean(axis=0), each


def emit(z_quantiles: np.ndarray, level: np.ndarray, scale: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """EUR/MWh quantiles, the central forecast (the ensemble median) and the number of rows whose
    quantiles had to be rearranged by sorting (recorded; averaging cannot create a crossing)."""
    eur = level[:, None] + scale[:, None] * z_quantiles
    if not np.isfinite(eur).all():
        raise TrainingFailure('nonfinite emitted quantile')
    crossed = np.any(np.diff(eur, axis=1) < 0, axis=1)
    if crossed.any():
        eur = np.sort(eur, axis=1)
    return eur, eur[:, MEDIAN].copy(), int(crossed.sum())


def fit_ensemble(x_train, y_train, x_val, y_val, config: dict, seeds=SEEDS, *, charge=None,
                 settings: dict | None = None) -> list[Member]:
    """One member per seed, each charged (`charge()`) before it trains."""
    members = []
    for seed in seeds:
        if charge is not None:
            charge()
        members.append(train_member(x_train, y_train, x_val, y_val, config, seed, settings))
    return members


def select_configuration(candidates: dict[str, tuple[np.ndarray, np.ndarray]], y_holdout: np.ndarray) -> tuple[str, dict]:
    """§21.2's training-only selection: the lowest holdout MAE (EUR/MWh) of each configuration's
    ensemble median; an exact tie goes to the smaller configuration (CONFIGS order).

    `candidates[id] = (median forecast on the holdout rows, unused)`."""
    order = [c['id'] for c in CONFIGS if c['id'] in candidates]
    losses = {cid: float(np.mean(np.abs(candidates[cid][0] - y_holdout))) for cid in order}
    if not all(math.isfinite(v) for v in losses.values()):
        raise TrainingFailure('nonfinite selection loss')
    chosen = min(order, key=lambda cid: (losses[cid], order.index(cid)))
    ranked = sorted(losses.values())
    margin = (ranked[1] - ranked[0]) / ranked[0] if len(ranked) > 1 and ranked[0] > 0 else None
    return chosen, {'holdout_mae': losses, 'selected': chosen, 'tie': sum(v == losses[chosen] for v in losses.values()) > 1,
                    'winner_margin_relative': margin}
