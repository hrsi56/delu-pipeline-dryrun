"""A5/A7's NumPy meta-learner: 16 → 8 tanh → 3 softmax, refit daily from scratch (protocol §5).

Inputs, in this column order: HG 7-date MAE, HG 28-date MAE, L 7-date MAE, L 28-date MAE,
D2 7-date MAE, D2 28-date MAE, high-volatility, negative-price-risk, volatility-unknown,
DJF/MAM/JJA/SON, night/solar/shoulder-peak. One row per window date and block; the row weight
`H_sb/(n·H_s)` makes the data loss the mean daily central MAE:

    (1/n) · Σ_s Σ_b (H_sb/H_s) · mean_{h∈b} |Σ_m p_sbm·c_smh − y_sh|

All sixteen inputs are standardized with the window's weighted mean and population SD (ddof=0),
scale 1 for an exactly zero SD. W1 (16,8) then W2 (8,3) are drawn in C order from
Normal(0, 0.1) by `default_rng(SeedSequence([28, D.toordinal(), 0x4D4C]))`; biases start at zero.
Objective = data loss + 1e-3·(ΣW1² + ΣW2²) (not divided by rows; biases unpenalized). The MAE
subgradient is sign(error), 0 at exactly zero. Full-batch Adam, exactly 300 updates, lr 0.01,
β1 0.9, β2 0.999, ε 1e-8 outside the square root, bias-corrected; the parameters after update 300
are deployed. Any nonfinite quantity or shape error is a solver failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib

import numpy as np

INPUTS = 16
HIDDEN = 8
OUTPUTS = 3
EPOCHS = 300
LEARNING_RATE = 0.01
BETA1, BETA2, EPSILON = 0.9, 0.999, 1e-8
L2 = 1e-3
INIT_SD = 0.1
SEED_TAG = 0x4D4C
COLUMNS = ('HG_mae7', 'HG_mae28', 'L_mae7', 'L_mae28', 'D2_mae7', 'D2_mae28',
           'high_volatility', 'negative_price_risk', 'volatility_unknown',
           'DJF', 'MAM', 'JJA', 'SON', 'night', 'solar', 'shoulder_peak')
assert len(COLUMNS) == INPUTS


class SolverFailure(FloatingPointError):
    """A nonfinite input, loss, gradient, moment, parameter or output, or a shape error."""


def _finite(name, *arrays):
    for array in arrays:
        if not np.all(np.isfinite(array)):
            raise SolverFailure(f'nonfinite {name}')


def seed_sequence(day: date) -> np.random.SeedSequence:
    return np.random.SeedSequence([28, day.toordinal(), SEED_TAG])


def initial_parameters(day: date) -> dict:
    rng = np.random.default_rng(seed_sequence(day))
    w1 = rng.normal(0.0, INIT_SD, size=(INPUTS, HIDDEN))
    w2 = rng.normal(0.0, INIT_SD, size=(HIDDEN, OUTPUTS))
    return {'W1': w1, 'b1': np.zeros(HIDDEN), 'W2': w2, 'b2': np.zeros(OUTPUTS)}


def standardize_fit(x, row_weight):
    mean = np.average(x, axis=0, weights=row_weight)
    sd = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=row_weight))
    scale = np.where(sd == 0.0, 1.0, sd)
    return mean, sd, scale


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(params, xs):
    a = np.tanh(xs @ params['W1'] + params['b1'])
    return a, softmax(a @ params['W2'] + params['b2'])


def combine(p, c):
    """Σ_m p_m·c_m in the fixed member order, left to right: c has shape (rows, hours, 3)."""
    return (p[:, None, 0] * c[:, :, 0] + p[:, None, 1] * c[:, :, 1]) + p[:, None, 2] * c[:, :, 2]


@dataclass
class Fit:
    params: dict
    mean: np.ndarray
    sd: np.ndarray
    scale: np.ndarray
    seed: tuple
    initial_sha256: str
    final_sha256: str
    data_loss: float
    penalty: float

    def predict(self, x):
        xs = (np.asarray(x, float) - self.mean) / self.scale
        _finite('deployment input', xs)
        _, p = forward(self.params, xs)
        _finite('deployment output', p)
        if (p < 0).any() or np.any(np.abs(p.sum(axis=1) - 1.0) > 1e-12):
            raise SolverFailure('softmax weights must be nonnegative and sum to one')
        return p


def parameter_hash(params) -> str:
    digest = hashlib.sha256()
    for name in ('W1', 'b1', 'W2', 'b2'):
        digest.update(np.ascontiguousarray(params[name], dtype='<f8').tobytes())
    return digest.hexdigest()


def gradients(params, xs, c, y, mask, q):
    """Data loss and gradients. ``q`` holds 1/(n·H_s) per row; ``mask`` marks the block's actual hours."""
    a, p = forward(params, xs)
    error = combine(p, c) - y
    data_loss = float(np.sum(q[:, None] * np.abs(error) * mask))
    g_error = np.sign(error) * mask * q[:, None]
    g_p = np.stack([np.sum(g_error * c[:, :, m], axis=1) for m in range(OUTPUTS)], axis=1)
    g_z2 = p * (g_p - np.sum(p * g_p, axis=1, keepdims=True))
    g_w2 = a.T @ g_z2 + 2 * L2 * params['W2']
    g_b2 = g_z2.sum(axis=0)
    g_a = g_z2 @ params['W2'].T
    g_z1 = g_a * (1 - a * a)
    g_w1 = xs.T @ g_z1 + 2 * L2 * params['W1']
    g_b1 = g_z1.sum(axis=0)
    return data_loss, {'W1': g_w1, 'b1': g_b1, 'W2': g_w2, 'b2': g_b2}


def fit(day: date, x, c, y, mask, hours_per_date, n_dates) -> Fit:
    """Train on rows (date, block). ``hours_per_date`` is H_s for each row's date."""
    x, c, y, mask = (np.asarray(v, float) for v in (x, c, y, mask))
    hours_per_date = np.asarray(hours_per_date, float)
    rows = len(x)
    if x.shape != (rows, INPUTS) or c.shape[:2] != y.shape or c.shape[2] != OUTPUTS or mask.shape != y.shape \
            or hours_per_date.shape != (rows,) or rows != 3 * n_dates:
        raise SolverFailure('meta-learner input shapes are inconsistent')
    c = np.where(mask[:, :, None] > 0, c, 0.0)
    y = np.where(mask > 0, y, 0.0)
    _finite('training input', x, c, y)
    block_hours = mask.sum(axis=1)
    row_weight = block_hours / (n_dates * hours_per_date)            # H_sb / (n·H_s)
    q = 1.0 / (n_dates * hours_per_date)                              # per-hour factor (H_sb/(n·H_s))/H_sb
    mean, sd, scale = standardize_fit(x, row_weight)
    xs = (x - mean) / scale
    _finite('standardized input', xs)
    params = initial_parameters(day)
    initial = parameter_hash(params)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    for t in range(1, EPOCHS + 1):
        data_loss, grads = gradients(params, xs, c, y, mask, q)
        if not np.isfinite(data_loss):
            raise SolverFailure('nonfinite loss')
        for name in ('W1', 'b1', 'W2', 'b2'):
            g = grads[name]
            _finite('gradient', g)
            m[name] = BETA1 * m[name] + (1 - BETA1) * g
            v[name] = BETA2 * v[name] + (1 - BETA2) * g * g
            m_hat = m[name] / (1 - BETA1 ** t)
            v_hat = v[name] / (1 - BETA2 ** t)
            params[name] = params[name] - LEARNING_RATE * m_hat / (np.sqrt(v_hat) + EPSILON)
            _finite('Adam state', m[name], v[name], params[name])
    final_loss, _ = gradients(params, xs, c, y, mask, q)
    penalty = L2 * float(np.sum(params['W1'] ** 2) + np.sum(params['W2'] ** 2))
    return Fit(params=params, mean=mean, sd=sd, scale=scale, seed=(28, day.toordinal(), SEED_TAG),
               initial_sha256=initial, final_sha256=parameter_hash(params), data_loss=final_loss, penalty=penalty)
