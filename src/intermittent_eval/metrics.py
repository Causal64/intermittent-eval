"""Error measures that stay finite and meaningful when most periods are zero.

Per-series measures take the actual values over the forecast horizon, the forecasts for the same periods,
and the in-sample history (data available at the forecast origin) used to scale the error. Scaling by the
in-sample history only is what keeps the evaluation free of look-ahead.

MAPE is deliberately absent: it divides by actual values that are zero in most intermittent periods.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "scales",
    "mase",
    "rmsse",
    "bias",
    "cumulative_error",
    "pinball",
    "scaled_pinball",
    "coverage",
    "classify",
    "ADI_CUTOFF",
    "CV2_CUTOFF",
]

ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


def _pair(actual: ArrayLike, forecast: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    f = np.broadcast_to(np.asarray(forecast, dtype=float), a.shape)
    return a, f


def scales(insample: ArrayLike) -> Tuple[float, float]:
    """Mean absolute and mean squared one-period change of the in-sample history (MASE and RMSSE scales)."""
    y = np.asarray(insample, dtype=float)
    if y.size < 2:
        raise ValueError("need at least two in-sample periods to compute a scale")
    d = np.diff(y)
    return float(np.abs(d).mean()), float((d ** 2).mean())


def _scale_or_raise(insample: ArrayLike, which: int) -> float:
    s = scales(insample)[which]
    if s == 0:
        raise ValueError("in-sample history is constant: the scaled error is undefined for this series")
    return s


def mase(actual: ArrayLike, forecast: ArrayLike, insample: ArrayLike) -> float:
    """Mean absolute scaled error (Hyndman and Koehler, 2006), non-seasonal scale.

    Minimised by the median of future demand, which is zero when more than half the periods are zero, so
    a zero forecast often wins it. Report it, do not rank intermittent forecasts on it.
    """
    a, f = _pair(actual, forecast)
    return float(np.abs(a - f).mean() / _scale_or_raise(insample, 0))


def rmsse(actual: ArrayLike, forecast: ArrayLike, insample: ArrayLike) -> float:
    """Root mean squared scaled error (the M5 accuracy measure). Minimised by the mean of future demand."""
    a, f = _pair(actual, forecast)
    return float(np.sqrt(((a - f) ** 2).mean() / _scale_or_raise(insample, 1)))


def bias(actual: ArrayLike, forecast: ArrayLike) -> float:
    """Relative bias of the total: ``sum(forecast) / sum(actual) - 1`` (0.18 means 18 % over-forecast).

    Pass all series and periods at once to get the pooled bias of a catalogue (mean forecast over mean
    actual). Returns NaN when the actual total is zero.
    """
    a, f = _pair(actual, forecast)
    tot = a.sum()
    return float(f.sum() / tot - 1.0) if tot > 0 else float("nan")


def cumulative_error(
    actual: ArrayLike,
    forecast: ArrayLike,
    lead_time: int | None = None,
    insample: ArrayLike | None = None,
) -> float:
    """Error on cumulative demand over the lead time: ``sum(forecast[:L]) - sum(actual[:L])``.

    This is the quantity a reorder decision depends on: positive means stock would have been over-bought.
    ``lead_time`` defaults to the full horizon. With ``insample`` the error is divided by the MASE scale
    times the lead time, so it can be averaged across items of different sizes.
    """
    a, f = _pair(actual, forecast)
    if a.ndim != 1:
        raise ValueError("cumulative_error works on one series at a time")
    L = a.size if lead_time is None else int(lead_time)
    if not 1 <= L <= a.size:
        raise ValueError(f"lead_time must be between 1 and the horizon ({a.size})")
    err = float(f[:L].sum() - a[:L].sum())
    if insample is None:
        return err
    return err / (_scale_or_raise(insample, 0) * L)


def pinball(actual: ArrayLike, quantile_forecast: ArrayLike, tau: float) -> np.ndarray:
    """Pinball (quantile) loss per period: ``tau`` per unit of unmet demand, ``1 - tau`` per unit of excess."""
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie in (0, 1)")
    a, q = _pair(actual, quantile_forecast)
    d = a - q
    return np.maximum(tau * d, (tau - 1.0) * d)


def scaled_pinball(actual: ArrayLike, quantile_forecast: ArrayLike, insample: ArrayLike, tau: float) -> float:
    """Mean pinball loss divided by the MASE scale of the in-sample history."""
    return float(pinball(actual, quantile_forecast, tau).mean() / _scale_or_raise(insample, 0))


def coverage(actual: ArrayLike, quantile_forecast: ArrayLike) -> float:
    """Share of periods with demand at or below the quantile forecast. Coarse on counts: see the README."""
    a, q = _pair(actual, quantile_forecast)
    return float((a <= q).mean())


def classify(y: ArrayLike) -> Tuple[float, float, str]:
    """ADI, CV² and demand class (Syntetos, Boylan and Croston, 2005).

    ADI is the number of periods divided by the number of non-zero periods; CV² is the squared coefficient
    of variation (population standard deviation) of the non-zero sizes, 0 with fewer than two demands.
    Class: Smooth, Erratic, Intermittent or Lumpy with cut-offs 1.32 (ADI) and 0.49 (CV²).
    """
    a = np.asarray(y, dtype=float)
    nz = a[a > 0]
    if nz.size == 0:
        return float("inf"), 0.0, "No demand"
    adi = a.size / nz.size
    cv2 = float((nz.std() / nz.mean()) ** 2) if nz.size > 1 else 0.0
    if adi < ADI_CUTOFF:
        cls = "Smooth" if cv2 < CV2_CUTOFF else "Erratic"
    else:
        cls = "Intermittent" if cv2 < CV2_CUTOFF else "Lumpy"
    return float(adi), cv2, cls
