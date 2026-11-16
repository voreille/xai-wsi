"""Cox proportional hazards and Harrell's concordance, in numpy.

``cox_fit`` maximises the Breslow partial log-likelihood (ties handled by Breslow's
approximation) minus ``0.5 * penalizer * ||beta||^2`` by Newton-Raphson with step halving.
Covariates should be standardised by the caller; there is no intercept (the baseline hazard
absorbs it). The covariance is the inverse of the penalised observed information, so Wald
tests are exact for ``penalizer=0`` and approximate otherwise.

``concordance`` is Harrell's C of a risk score (higher = earlier event): over comparable pairs
(the shorter time is an event), the share where the shorter time has the higher risk, ties in
risk counting one half.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class CoxResult:
    beta: np.ndarray
    se: np.ndarray
    z: np.ndarray
    p: np.ndarray
    loglik: float
    iterations: int

    def risk(self, x: np.ndarray) -> np.ndarray:
        return x @ self.beta


def _loglik_terms(x: np.ndarray, time: np.ndarray, event: np.ndarray, beta: np.ndarray):
    """Partial log-likelihood, gradient and observed information (Breslow), on time-sorted data."""
    eta = x @ beta
    shift = eta.max()
    w = np.exp(eta - shift)
    # Risk sets: all subjects with time >= t. Data are sorted by descending time, so cumulative
    # sums give the risk-set sums; tied times share the sum at the last row of their block.
    s0 = np.cumsum(w)
    s1 = np.cumsum(w[:, None] * x, axis=0)
    s2 = np.cumsum(w[:, None, None] * x[:, :, None] * x[:, None, :], axis=0)
    ends = np.flatnonzero(np.r_[time[1:] != time[:-1], True])
    idx = ends[np.searchsorted(ends, np.flatnonzero(event))]
    s0e, s1e, s2e = s0[idx], s1[idx], s2[idx]
    mean = s1e / s0e[:, None]
    loglik = float((eta[event] - shift - np.log(s0e)).sum())
    grad = (x[event] - mean).sum(axis=0)
    info = (s2e / s0e[:, None, None] - mean[:, :, None] * mean[:, None, :]).sum(axis=0)
    return loglik, grad, info


def cox_fit(
    x: np.ndarray, time: np.ndarray, event: np.ndarray, penalizer: float = 0.0, max_iter: int = 100, tol: float = 1e-9
) -> CoxResult:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    time, event = np.asarray(time, dtype=np.float64), np.asarray(event).astype(bool)
    if not (np.isfinite(x).all() and np.isfinite(time).all()):
        raise ValueError("non-finite covariates or times")
    if event.sum() == 0:
        raise ValueError("no events")
    order = np.argsort(-time, kind="stable")
    x, time, event = x[order], time[order], event[order]
    beta = np.zeros(x.shape[1])
    ridge = penalizer * np.eye(x.shape[1])

    def objective(b):
        ll, g, h = _loglik_terms(x, time, event, b)
        return ll - 0.5 * penalizer * b @ b, g - penalizer * b, h + ridge

    value, grad, info = objective(beta)
    for iteration in range(1, max_iter + 1):
        step = np.linalg.solve(info, grad)
        for _ in range(30):
            candidate = beta + step
            new_value, new_grad, new_info = objective(candidate)
            if new_value >= value - 1e-12:
                break
            step /= 2
        else:
            raise RuntimeError("Cox fit: step halving failed")
        converged = abs(new_value - value) < tol * (1 + abs(value))
        beta, value, grad, info = candidate, new_value, new_grad, new_info
        if converged:
            break
    else:
        raise RuntimeError(f"Cox fit did not converge in {max_iter} iterations")
    cov = np.linalg.inv(info)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    return CoxResult(beta, se, z, 2 * norm.sf(np.abs(z)), value, iteration)


def concordance(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    time, event, risk = np.asarray(time, float), np.asarray(event).astype(bool), np.asarray(risk, float)
    earlier = time[:, None] < time[None, :]
    comparable = earlier & event[:, None]
    higher = risk[:, None] > risk[None, :]
    tied = risk[:, None] == risk[None, :]
    n = comparable.sum()
    if n == 0:
        raise ValueError("no comparable pairs")
    return float((comparable & higher).sum() + 0.5 * (comparable & tied).sum()) / n
