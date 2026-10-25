"""Split-conformal prediction intervals for multi-output regression (one interval per output).

Plain NumPy. Arrays are (N cases, P outputs). An interval is ``pred ± q * scale``: with
``scale = 1`` the width is constant per output; with a per-case scale (e.g. the spread of
an ensemble) it adapts to the case. ``q`` is the conformal quantile of the calibration
scores ``|y - pred| / scale``, which gives marginal coverage ≥ 1 − alpha per output when
the calibration and test cases are exchangeable. Under a cohort shift there is no
guarantee; coverage on an external cohort is an empirical question.
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> np.ndarray:
    """Per-output quantile of calibration scores (N, P) at level ceil((N+1)(1−alpha))/N.

    Returns inf where N is too small for that level (the interval is then unbounded)."""
    n = scores.shape[0]
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return np.full(scores.shape[1], np.inf)
    return np.sort(scores, axis=0)[rank - 1]


def fit_intervals(
    y_cal: np.ndarray, pred_cal: np.ndarray, scale_cal: np.ndarray | float, alpha: float
) -> np.ndarray:
    """Quantiles q (P,) such that pred ± q * scale covers 1 − alpha of calibration-like cases."""
    scale_cal = np.broadcast_to(scale_cal, y_cal.shape)
    if (scale_cal <= 0).any():
        raise ValueError("Interval scales must be positive.")
    return conformal_quantile(np.abs(y_cal - pred_cal) / scale_cal, alpha)


def interval_metrics(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> dict[str, np.ndarray]:
    """Per-output coverage, mean width and mean interval score (Gneiting & Raftery 2007:
    width plus 2/alpha times the distance by which y falls outside; lower is better)."""
    below = np.clip(lower - y, 0, None)
    above = np.clip(y - upper, 0, None)
    width = upper - lower
    return {
        "coverage": ((y >= lower) & (y <= upper)).mean(axis=0),
        "width": width.mean(axis=0),
        "interval_score": (width + 2 / alpha * (below + above)).mean(axis=0),
    }
