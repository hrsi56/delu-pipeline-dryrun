"""DDNN-2: a day-level feed-forward network with 24 Johnson SU heads, in NumPy only.

Capstone v21-r11 §23.3 (with §18.2, §18.3 and §21.3). This module is DDNN-2's single model
implementation: the network, the Johnson SU head and its masked likelihood, the pinball loss through
the Johnson SU quantile function, L1 and L2, input dropout, the analytic gradients, the optimizer,
the training loop with random whole-week early stopping, seeding, the target transforms and their
inverses, the guards and the emission of quantiles. It imports NumPy and the Python standard
library only; the import audit (`tests/cp24/test_numpy_only.py`) enforces that, with a fixture that
fails it.

**Network.** Inputs -> optional input dropout -> one or two hidden layers (ELU, ReLU, softplus or
tanh) -> 96 linear outputs, laid out as four blocks of 24 local-hour slots: o1 (xi), o2 (lambda),
o3 (gamma), o4 (delta). Per slot, CP-23's parameterisation (`src/cp23/ddnn.py`):

    xi = o1,  lambda = softplus(o2) + 1e-3,  gamma = o3,  delta = softplus(o4) + 0.05,

for Y = xi + lambda * sinh((Z - gamma) / delta), Z ~ N(0, 1) (Johnson, 1949): the quantile at level
tau is xi + lambda * sinh((Phi^-1(tau) - gamma) / delta), strictly increasing in tau.

**Loss** (per batch, over the present target slots only; a 23-hour day masks its missing slot):

    L = k * NLL + (1 - k) * PIN + l1 * sum|W| + l2 * sum W^2   (weights only; biases unpenalised)

* NLL: the mean Johnson SU negative log-likelihood of the transformed target t over the present
  slots, each row weighted by its recency weight (weights have mean 1 over the training rows);
* PIN: the mean pinball loss over the 19-level grid `GRID` (which contains the seven scored levels)
  of the Johnson SU quantiles in t, with the same weights;
* the first `WARM_EPOCHS` epochs use k = 1 (NLL only); then k = kappa, the searched loss weight.

**Training.** Glorot-uniform weights and zero biases from the member's seeded PCG64 stream, except
the output biases, which start lambda and delta at 1. Adam in PyTorch's update order. Mini-batches in
a fresh seeded permutation every epoch; input-dropout masks from the same stream. After every epoch
from `WARM_EPOCHS + 1` on, the stopping metric -- the mean pinball loss over the seven scored levels
in EUR/MWh, after inversion and the cap, on the member's held-out whole calendar weeks -- is computed;
the best epoch's weights are kept (strict improvement); training stops after `PATIENCE` epochs
without improvement or at `MAX_EPOCHS`. A nonfinite training loss or gradient ends training before
the step is applied and keeps the best epoch (recorded); without a best epoch it is a failed fit,
never substituted.

**Guards** (§23.3), each activation recorded: winsorisation of continuous inputs (in
`cp24.design`), the cap of every emitted quantile at +-`CAP_MULTIPLE` x the largest |z| on the
member's training rows (z = the normalised target before any asinh), and restoration of crossings
by sorting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from statistics import NormalDist
import time

import numpy as np

#: The seven scored CP-15 levels; the p50 is the fourth.
LEVELS = (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)
MEDIAN = LEVELS.index(0.50)
#: The 19-level training grid: symmetric about 0.5 and containing the seven scored levels.
GRID = (0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
        0.975)
Z_LEVELS = np.array([NormalDist().inv_cdf(q) for q in LEVELS])
Z_GRID = np.array([NormalDist().inv_cdf(q) for q in GRID])
TAU_LEVELS = np.array(LEVELS)
TAU_GRID = np.array(GRID)
SCORED_IN_GRID = tuple(GRID.index(q) for q in LEVELS)
LOG_2PI = math.log(2.0 * math.pi)
LAMBDA_FLOOR = 1e-3
DELTA_FLOOR = 0.05
SLOTS = 24
WARM_EPOCHS = 20
PATIENCE = 50
MAX_EPOCHS = 1000
CAP_MULTIPLE = 1.25
ACTIVATIONS = ('elu', 'relu', 'softplus', 'tanh')
ADAM = {'beta1': 0.9, 'beta2': 0.999, 'eps': 1e-8}


class TrainingFailure(RuntimeError):
    """A nonfinite input or forecast, or a nonfinite loss before any kept epoch. Recorded as a failed
    fit, never substituted."""


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


def activate(name: str, a: np.ndarray) -> np.ndarray:
    if name == 'elu':
        return np.where(a > 0, a, np.expm1(np.minimum(a, 0.0)))
    if name == 'relu':
        return np.maximum(a, 0.0)
    if name == 'softplus':
        return softplus(a)
    if name == 'tanh':
        return np.tanh(a)
    raise ValueError(f'unknown activation {name}')


def activate_grad(name: str, a: np.ndarray, h: np.ndarray) -> np.ndarray:
    """d activation / d a, given the pre-activation `a` and the activation `h`."""
    if name == 'elu':
        return np.where(a > 0, 1.0, h + 1.0)
    if name == 'relu':
        return (a > 0).astype(a.dtype)
    if name == 'softplus':
        return sigmoid(a)
    if name == 'tanh':
        return 1.0 - h * h
    raise ValueError(f'unknown activation {name}')


OUTPUT_BIAS = np.concatenate([np.zeros(SLOTS), np.full(SLOTS, inverse_softplus(1.0 - LAMBDA_FLOOR)), np.zeros(SLOTS),
                              np.full(SLOTS, inverse_softplus(1.0 - DELTA_FLOOR))])


# ------------------------------------------------------------------ the network
def layer_sizes(n_inputs: int, hidden: tuple[int, ...]) -> list[int]:
    return [int(n_inputs), *map(int, hidden), 4 * SLOTS]


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


def forward(params: list[np.ndarray], x: np.ndarray, activation: str, dropout_mask: np.ndarray | None = None):
    """Raw outputs (n x 96) and the cache the backward pass needs."""
    h = x if dropout_mask is None else x * dropout_mask
    inputs, pre = [h], []
    for k in range(len(params) // 2 - 1):
        a = h @ params[2 * k] + params[2 * k + 1]
        pre.append(a)
        h = activate(activation, a)
        inputs.append(h)
    out = h @ params[-2] + params[-1]
    return out, (inputs, pre)


def head(out: np.ndarray):
    """The Johnson SU parameters (xi, lambda, gamma, delta), each n x 24."""
    s = SLOTS
    return (out[:, 0:s], softplus(out[:, s:2 * s]) + LAMBDA_FLOOR, out[:, 2 * s:3 * s],
            softplus(out[:, 3 * s:4 * s]) + DELTA_FLOOR)


def jsu_nll(t, xi, lam, gamma, delta):
    """Elementwise Johnson SU negative log-likelihood."""
    u = (t - xi) / lam
    r = gamma + delta * np.arcsinh(u)
    return -np.log(delta) + np.log(lam) + 0.5 * LOG_2PI + 0.5 * np.log1p(u * u) + 0.5 * r * r


def jsu_nll_grad(t, xi, lam, gamma, delta):
    """Elementwise negative log-likelihood and its partial derivatives in (xi, lambda, gamma, delta)."""
    u = (t - xi) / lam
    asinh_u = np.arcsinh(u)
    r = gamma + delta * asinh_u
    one_u2 = 1.0 + u * u
    nll = -np.log(delta) + np.log(lam) + 0.5 * LOG_2PI + 0.5 * np.log1p(u * u) + 0.5 * r * r
    d_u = u / one_u2 + r * delta / np.sqrt(one_u2)
    return nll, (-d_u / lam, (1.0 - u * d_u) / lam, r, -1.0 / delta + r * asinh_u)


def jsu_quantiles(xi, lam, gamma, delta, z_levels=Z_LEVELS) -> np.ndarray:
    """Quantiles of t at the given standard-normal levels: n x 24 x L."""
    with np.errstate(over='ignore'):
        return xi[..., None] + lam[..., None] * np.sinh((z_levels - gamma[..., None]) / delta[..., None])


def pinball_grad(t, xi, lam, gamma, delta, tau=TAU_GRID, z_levels=Z_GRID):
    """Elementwise pinball losses rho_tau(t - q_tau) (n x 24 x L) and their partial derivatives in
    (xi, lambda, gamma, delta), summed over the levels (n x 24 each). The subgradient at t == q is
    the right derivative of rho in q: 1{t < q} - tau, so -tau there."""
    with np.errstate(over='ignore', invalid='ignore'):
        w = (z_levels - gamma[..., None]) / delta[..., None]
        sh, ch = np.sinh(w), np.cosh(w)
        q = xi[..., None] + lam[..., None] * sh
        diff = t[..., None] - q
        loss = np.maximum(tau * diff, (tau - 1.0) * diff)
        g = (diff < 0).astype(float) - tau                      # d rho / d q
        g_xi = g.sum(axis=-1)
        g_lam = (g * sh).sum(axis=-1)
        g_gamma = -(g * ch).sum(axis=-1) * lam / delta
        g_delta = -(g * ch * w).sum(axis=-1) * lam / delta
    return loss, (g_xi, g_lam, g_gamma, g_delta)


def penalty(params: list[np.ndarray], l1: float, l2: float) -> float:
    total = 0.0
    for w in params[0::2]:
        if l1:
            total += l1 * float(np.sum(np.abs(w)))
        if l2:
            total += l2 * float(np.sum(w * w))
    return total


def loss_and_grads(params: list[np.ndarray], x: np.ndarray, t: np.ndarray, mask: np.ndarray, weight: np.ndarray,
                   activation: str, kappa: float, l1: float, l2: float, dropout_mask: np.ndarray | None = None):
    """The batch loss (module docstring) and its gradient for every parameter.

    `t` is n x 24 (any value where `mask` is False), `mask` n x 24 boolean, `weight` the n row weights."""
    out, (inputs, pre) = forward(params, x, activation, dropout_mask)
    xi, lam, gamma, delta = head(out)
    m = mask.astype(float) * weight[:, None]
    count = float(mask.sum())
    if count <= 0:
        raise TrainingFailure('a batch without any present target slot')
    tt = np.where(mask, t, xi)  # absent slots never enter: weight 0, finite placeholder
    g = [np.zeros_like(xi) for _ in range(4)]
    loss = 0.0
    if kappa > 0:
        nll, gn = jsu_nll_grad(tt, xi, lam, gamma, delta)
        loss += kappa * float(np.sum(m * nll)) / count
        for i in range(4):
            g[i] += kappa * m * gn[i] / count
    if kappa < 1:
        pin, gp = pinball_grad(tt, xi, lam, gamma, delta)
        denom = count * len(GRID)
        loss += (1.0 - kappa) * float(np.sum(m[..., None] * pin)) / denom
        for i in range(4):
            g[i] += (1.0 - kappa) * m * gp[i] / denom
    s = SLOTS
    d_out = np.empty_like(out)
    d_out[:, 0:s] = g[0]
    d_out[:, s:2 * s] = g[1] * sigmoid(out[:, s:2 * s])
    d_out[:, 2 * s:3 * s] = g[2]
    d_out[:, 3 * s:4 * s] = g[3] * sigmoid(out[:, 3 * s:4 * s])
    grads: list[np.ndarray] = [None] * len(params)  # type: ignore[list-item]
    grads[-2] = inputs[-1].T @ d_out
    grads[-1] = d_out.sum(axis=0)
    dh = d_out @ params[-2].T
    for k in range(len(params) // 2 - 2, -1, -1):
        da = dh * activate_grad(activation, pre[k], inputs[k + 1])
        grads[2 * k] = inputs[k].T @ da
        grads[2 * k + 1] = da.sum(axis=0)
        if k:
            dh = da @ params[2 * k].T
    for k in range(0, len(params), 2):
        if l1:
            grads[k] = grads[k] + l1 * np.sign(params[k])
        if l2:
            grads[k] = grads[k] + 2.0 * l2 * params[k]
    return loss + penalty(params, l1, l2), grads


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


# ------------------------------------------------------------------ target transforms
#: The four target forms (§23.4): z centred and scaled by §4's level and scale ('s4') or by the
#: median and MAD of the last 168 hours with a floor ('mad'); then z or asinh(z).
TRANSFORMS = ('z-s4', 'asinh-s4', 'z-mad', 'asinh-mad')


def forward_transform(form: str, z: np.ndarray) -> np.ndarray:
    return np.arcsinh(z) if form.startswith('asinh') else z


def inverse_transform(form: str, t: np.ndarray) -> np.ndarray:
    if form.startswith('asinh'):
        with np.errstate(over='ignore'):
            return np.sinh(t)
    return t


def statistics_of(form: str) -> str:
    return form.split('-')[1]


# ------------------------------------------------------------------ emission and the stopping metric
def member_quantiles_z(params, x, activation, form, cap, levels=Z_LEVELS):
    """One member's quantiles of z (n x 24 x L) after the inverse transform and the cap, and the cap's
    activations (a boolean array of the same shape: |z| above the cap before clipping)."""
    out, _ = forward(params, x, activation)
    q_t = jsu_quantiles(*head(out), z_levels=levels)
    z = inverse_transform(form, q_t)
    if np.isnan(z).any():
        raise TrainingFailure('nonfinite member quantile')
    active = np.abs(z) > cap
    return np.clip(z, -cap, cap), active


def pinball_eur(q_eur: np.ndarray, y: np.ndarray, mask: np.ndarray, tau=TAU_LEVELS) -> float:
    """Mean pinball loss over the present slots and the levels, in EUR/MWh."""
    diff = y[..., None] - q_eur
    loss = np.maximum(tau * diff, (tau - 1.0) * diff)
    return float(loss[mask].mean())


@dataclass
class Member:
    """One trained network: the best-epoch weights and how it was trained."""
    config_id: str
    seed: int
    params: list[np.ndarray]
    best_epoch: int
    epochs_run: int
    best_metric: float
    stop_reason: str
    history: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    cap: float = math.inf
    events: list[dict] = field(default_factory=list)

    def record(self) -> dict:
        return {'config': self.config_id, 'seed': self.seed, 'best_epoch': self.best_epoch, 'epochs_run': self.epochs_run,
                'best_stopping_metric_eur': self.best_metric, 'stop_reason': self.stop_reason,
                'params_sha256': params_sha256(self.params), 'cap_z': self.cap,
                'fit_wall_seconds': self.wall_seconds, 'fit_cpu_seconds': self.cpu_seconds,
                'stopping_metric_history': self.history, 'train_loss_history': self.train_loss, 'events': self.events}


def train(x_train: np.ndarray, t_train: np.ndarray, mask_train: np.ndarray, weight: np.ndarray,
          x_hold: np.ndarray, y_hold_eur: np.ndarray, mask_hold: np.ndarray, center_hold: np.ndarray,
          scale_hold: np.ndarray, config: dict, seed: int, cap: float, *, max_epochs: int = MAX_EPOCHS,
          patience: int = PATIENCE, warm_epochs: int = WARM_EPOCHS, trace: list | None = None) -> Member:
    """Train one member; early-stop on the held-out weeks' seven-level pinball loss (EUR/MWh).

    `trace`, if given, receives a copy of the parameters after every epoch (the PyTorch reference
    compares the trajectory); it never changes the training."""
    for name, a in (('x_train', x_train), ('x_hold', x_hold), ('weight', weight), ('center_hold', center_hold),
                    ('scale_hold', scale_hold)):
        if not np.isfinite(a).all():
            raise TrainingFailure(f'nonfinite {name}')
    if not np.isfinite(t_train[mask_train]).all() or not np.isfinite(y_hold_eur[mask_hold]).all():
        raise TrainingFailure('nonfinite target')
    if not mask_train.any() or not mask_hold.any():
        raise TrainingFailure('empty training or held-out target slots')
    wall, cpu = time.perf_counter(), time.process_time()
    rng = np.random.Generator(np.random.PCG64(seed))
    activation, form = config['activation'], config['transform']
    params = init_params(x_train.shape[1], tuple(config['hidden']), rng)
    opt = Adam(params, float(config['lr']), **ADAM)
    n, bs = len(x_train), int(config['batch_size'])
    dropout = float(config.get('input_dropout') or 0.0)
    l1, l2, kappa = float(config.get('l1') or 0.0), float(config.get('l2') or 0.0), float(config['kappa'])
    best, best_params, best_epoch, wait = math.inf, None, 0, 0
    history, train_loss, events = [], [], []
    stop_reason, epoch = 'max_epochs', 0
    for epoch in range(1, int(max_epochs) + 1):
        order = rng.permutation(n)
        k_now = 1.0 if epoch <= warm_epochs else kappa
        total, bad = 0.0, False
        for start in range(0, n, bs):
            idx = order[start:start + bs]
            xb = x_train[idx]
            mask_b = mask_train[idx]
            if not mask_b.any():
                continue
            dm = None
            if dropout > 0:
                dm = (rng.random(xb.shape) >= dropout) / (1.0 - dropout)
            loss, grads = loss_and_grads(params, xb, t_train[idx], mask_b, weight[idx], activation, k_now, l1, l2, dm)
            if not math.isfinite(loss) or not all(np.isfinite(g).all() for g in grads):
                bad = True
                break
            opt.step(params, grads)
            total += loss * len(idx)
        if bad:
            events.append({'event': 'nonfinite_training_loss', 'epoch': epoch})
            stop_reason = 'nonfinite_loss'
            break
        train_loss.append(total / n)
        if trace is not None:
            trace.append([p.copy() for p in params])
        if epoch <= warm_epochs:
            continue
        zq, _ = member_quantiles_z(params, x_hold, activation, form, cap)
        metric = pinball_eur(center_hold[:, None, None] + scale_hold[:, None, None] * zq, y_hold_eur, mask_hold)
        if not math.isfinite(metric):
            events.append({'event': 'nonfinite_stopping_metric', 'epoch': epoch})
            stop_reason = 'nonfinite_metric'
            break
        history.append(metric)
        if metric < best:
            best, best_params, best_epoch, wait = metric, [p.copy() for p in params], epoch, 0
        else:
            wait += 1
            if wait >= int(patience):
                stop_reason = 'patience'
                break
    if best_params is None:
        raise TrainingFailure(f'no kept epoch ({stop_reason} at epoch {epoch})')
    return Member(config['id'], int(seed), best_params, best_epoch, epoch, float(best), stop_reason, history, train_loss,
                  time.perf_counter() - wall, time.process_time() - cpu, float(cap), events)


# ------------------------------------------------------------------ the ensemble
def ensemble_median(member_eur: np.ndarray) -> tuple[np.ndarray, int]:
    """Per-level median of the members' EUR/MWh quantiles (members x n x 24 x L): with an even number
    of members, the mean of the two middle values. Crossings are restored by sorting and counted
    (a per-level median of ordered vectors cannot cross)."""
    med = np.median(member_eur, axis=0)
    crossed = np.any(np.diff(med, axis=-1) < 0, axis=-1)
    if crossed.any():
        med = np.sort(med, axis=-1)
    return med, int(crossed.sum())
