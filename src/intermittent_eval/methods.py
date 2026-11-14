"""Point forecasting methods for intermittent demand.

Every method takes the history of one series (a 1-D array of non-negative demands, oldest first) and
returns a single number: the flat forecast for every future period. The conventions (initialisation in
particular) are the ones used in the Causal64 research note
https://causal64.com/research/intermittent-demand-forecasting/ so that the published figures can be
reproduced exactly.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "zero",
    "naive",
    "moving_average",
    "ses",
    "croston",
    "sba",
    "tsb",
    "default_methods",
    "Method",
]

Method = Callable[[np.ndarray], float]
"""A forecasting method: history of one series in, flat point forecast out."""


def _as_history(y: ArrayLike) -> np.ndarray:
    a = np.asarray(y, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"expected a 1-D history, got shape {a.shape}")
    if a.size == 0:
        raise ValueError("history is empty")
    if np.isnan(a).any():
        raise ValueError("history contains NaN; fill or drop missing periods first")
    if (a < 0).any():
        raise ValueError("demand must be non-negative")
    return a


def _check_rate(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}")


def zero(y: ArrayLike) -> float:
    """Forecast zero. A control: a metric that ranks it first says more about the metric than the methods."""
    _as_history(y)
    return 0.0


def naive(y: ArrayLike) -> float:
    """Last observed value."""
    return float(_as_history(y)[-1])


def moving_average(y: ArrayLike, window: int = 12) -> float:
    """Mean of the last ``window`` periods (all periods if the history is shorter)."""
    if window < 1:
        raise ValueError("window must be at least 1")
    return float(_as_history(y)[-window:].mean())


def ses(y: ArrayLike, alpha: float = 0.1, init: Optional[float] = None) -> float:
    """Simple exponential smoothing.

    The level starts at ``init`` or, if ``init`` is None, at the mean of the first ``min(12, len(y))``
    periods, then is updated on every period: ``level += alpha * (y_t - level)``.
    """
    a = _as_history(y)
    _check_rate("alpha", alpha)
    level = float(a[: min(12, a.size)].mean()) if init is None else float(init)
    for v in a:
        level += alpha * (v - level)
    return float(level)


def croston(y: ArrayLike, alpha: float = 0.1, alpha_interval: Optional[float] = None) -> float:
    """Croston (1972): smoothed non-zero size divided by smoothed inter-demand interval.

    Both parts are updated only in periods with a demand. The size starts at the first non-zero demand,
    the interval at the number of periods up to and including it. ``alpha_interval`` defaults to
    ``alpha``. Returns 0 when the history has no demand.
    """
    a = _as_history(y)
    _check_rate("alpha", alpha)
    b = alpha if alpha_interval is None else alpha_interval
    _check_rate("alpha_interval", b)
    nz = np.flatnonzero(a > 0)
    if nz.size == 0:
        return 0.0
    size = float(a[nz[0]])
    interval = float(nz[0] + 1)
    q = 1
    for t in range(int(nz[0]) + 1, a.size):
        if a[t] > 0:
            size += alpha * (a[t] - size)
            interval += b * (q - interval)
            q = 1
        else:
            q += 1
    return size / interval


def sba(y: ArrayLike, alpha: float = 0.1, alpha_interval: Optional[float] = None) -> float:
    """Syntetos-Boylan approximation (2005): Croston times ``1 - alpha_interval / 2``.

    Corrects the upward bias of Croston's ratio. ``alpha_interval`` defaults to ``alpha``.
    """
    b = alpha if alpha_interval is None else alpha_interval
    return croston(y, alpha, b) * (1.0 - b / 2.0)


def tsb(
    y: ArrayLike,
    alpha: float = 0.1,
    beta: float = 0.1,
    size_init: Optional[float] = None,
    prob_init: Optional[float] = None,
) -> float:
    """Teunter, Syntetos and Babai (2011): smoothed demand probability times smoothed non-zero size.

    The probability is updated every period (towards 1 after a demand, towards 0 otherwise) with ``beta``;
    the size only after a demand, with ``alpha``. When demand stops, the forecast decays towards zero.
    Initial size: mean of the first three non-zero demands (fewer if the history has fewer). Initial
    probability: share of non-zero periods among the first ``min(12, len(y))``. ``size_init`` and
    ``prob_init`` override them (statsforecast, for instance, starts from the first non-zero demand and from
    the indicator of the first period). Returns 0 when the history has no demand.
    """
    a = _as_history(y)
    _check_rate("alpha", alpha)
    _check_rate("beta", beta)
    nz = np.flatnonzero(a > 0)
    if nz.size == 0:
        return 0.0
    size = float(a[nz[:3]].mean()) if size_init is None else float(size_init)
    prob = float((a[: min(12, a.size)] > 0).mean()) if prob_init is None else float(prob_init)
    for v in a:
        if v > 0:
            size += alpha * (v - size)
            prob += beta * (1.0 - prob)
        else:
            prob += beta * (0.0 - prob)
    return prob * size


def default_methods(
    alpha: float = 0.1,
    beta: float = 0.1,
    window: int = 12,
    include_controls: bool = True,
) -> Dict[str, Method]:
    """The comparison set of the research note: controls (zero, naive, moving average), SES, Croston, SBA, TSB.

    One ``alpha`` is used for SES, Croston, SBA and the TSB size; ``beta`` is the TSB probability rate.
    Choose these on a development window, not on the window you report.
    """
    m: Dict[str, Method] = {}
    if include_controls:
        m["Zero"] = zero
        m["Naive"] = naive
        m["Moving average"] = lambda y: moving_average(y, window)
    m["SES"] = lambda y: ses(y, alpha)
    m["Croston"] = lambda y: croston(y, alpha)
    m["SBA"] = lambda y: sba(y, alpha)
    m["TSB"] = lambda y: tsb(y, alpha, beta)
    return m
