"""Rolling-origin evaluation without look-ahead, and the comparison table.

At each forecast origin ``o`` a method only sees ``y[:o]``; the error over periods ``o .. o+h-1`` is
scaled with the same ``y[:o]``. Nothing computed after the origin enters a forecast or a scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .methods import Method, default_methods
from .metrics import classify

__all__ = ["rolling_origin", "evaluate", "compare", "paired_bootstrap", "ComparisonTable", "SeriesScores"]


def _as_panel(Y: ArrayLike) -> np.ndarray:
    a = np.asarray(Y, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError("expected a 2-D array of shape (n_series, n_periods)")
    if np.isnan(a).any():
        raise ValueError("data contains NaN; fill or drop incomplete series first")
    if (a < 0).any():
        raise ValueError("demand must be non-negative")
    return a


def _check_origins(origins: Optional[Sequence[int]], n_periods: int, horizon: int) -> List[int]:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if origins is None:
        origins = [n_periods - horizon]
    out = [int(o) for o in origins]
    for o in out:
        if o < 2 or o + horizon > n_periods:
            raise ValueError(
                f"origin {o} invalid: need 2 <= origin and origin + horizon ({horizon}) <= n_periods ({n_periods})"
            )
    return out


def rolling_origin(
    Y: ArrayLike,
    methods: Mapping[str, Method],
    origins: Optional[Sequence[int]] = None,
    horizon: int = 6,
) -> Dict[str, np.ndarray]:
    """Forecasts from each origin, each fitted on the data before that origin only.

    ``Y`` has shape ``(n_series, n_periods)``. ``origins`` are 0-based indices of the first forecast period
    (equivalently, the number of periods the method sees); the default is a single origin at
    ``n_periods - horizon``. A method returns either one number (a flat forecast) or ``horizon`` numbers.
    Returns, per method, an array of shape ``(n_series, n_origins, horizon)``.
    """
    P = _as_panel(Y)
    org = _check_origins(origins, P.shape[1], horizon)
    out: Dict[str, np.ndarray] = {}
    for name, fn in methods.items():
        F = np.empty((P.shape[0], len(org), horizon))
        for i, y in enumerate(P):
            for j, o in enumerate(org):
                f = np.asarray(fn(y[:o].copy()), dtype=float)
                if f.ndim == 0:
                    F[i, j, :] = f
                elif f.shape == (horizon,):
                    F[i, j, :] = f
                else:
                    raise ValueError(f"method {name!r} returned shape {f.shape}, expected () or ({horizon},)")
        out[name] = F
    return out


@dataclass
class SeriesScores:
    """Per-series scores, averaged over origins. Arrays are indexed by the kept series."""

    mase: Dict[str, np.ndarray]
    rmsse: Dict[str, np.ndarray]
    lead_time_error: Dict[str, np.ndarray]
    forecast_total: Dict[str, np.ndarray]
    actual_total: np.ndarray
    kept: np.ndarray
    classes: np.ndarray


def evaluate(
    Y: ArrayLike,
    methods: Optional[Mapping[str, Method]] = None,
    origins: Optional[Sequence[int]] = None,
    horizon: int = 6,
    lead_time: Optional[int] = None,
) -> SeriesScores:
    """Score every method on every series with rolling origins.

    Series whose in-sample history is constant before some origin (scale zero) cannot be scaled and are
    dropped; ``kept`` marks the series that remain. ``lead_time`` (default: the horizon) sets the window of
    the cumulative error, scaled by the MASE scale times the lead time. Classes (ADI / CV²) are computed on
    the data before the first origin.
    """
    P = _as_panel(Y)
    methods = dict(default_methods() if methods is None else methods)
    org = _check_origins(origins, P.shape[1], horizon)
    L = horizon if lead_time is None else int(lead_time)
    if not 1 <= L <= horizon:
        raise ValueError("lead_time must be between 1 and horizon")

    s1 = np.empty((P.shape[0], len(org)))
    s2 = np.empty_like(s1)
    for j, o in enumerate(org):
        d = np.diff(P[:, :o], axis=1)
        s1[:, j] = np.abs(d).mean(1)
        s2[:, j] = (d ** 2).mean(1)
    kept = (s1 > 0).all(1) & (s2 > 0).all(1)
    Pk, s1, s2 = P[kept], s1[kept], s2[kept]
    A = np.stack([Pk[:, o:o + horizon] for o in org], axis=1)  # (n, n_origins, h)

    fc = rolling_origin(Pk, methods, org, horizon)
    res = SeriesScores({}, {}, {}, {}, A.sum((1, 2)), kept,
                       np.array([classify(y[: org[0]])[2] for y in Pk], dtype=object))
    for name, F in fc.items():
        e = A - F
        res.mase[name] = (np.abs(e).mean(2) / s1).mean(1)
        res.rmsse[name] = np.sqrt((e ** 2).mean(2) / s2).mean(1)
        res.lead_time_error[name] = ((F[:, :, :L].sum(2) - A[:, :, :L].sum(2)) / (s1 * L)).mean(1)
        res.forecast_total[name] = F.sum((1, 2))
    return res


def paired_bootstrap(d: ArrayLike, n_boot: int = 4000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Mean of per-series differences ``d`` with a percentile interval from resampling series."""
    x = np.asarray(d, dtype=float)
    n = x.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = max(1, 2_000_000 // n)
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        means[s:s + k] = x[rng.integers(0, n, size=(k, n))].mean(1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(x.mean()), float(lo), float(hi)


@dataclass
class ComparisonTable:
    """Result of :func:`compare`. ``rows`` is a list of dicts, one per (group, method)."""

    rows: List[Dict[str, Any]]
    reference: str
    n_series: int
    n_dropped: int
    scores: SeriesScores = field(repr=False)

    COLUMNS = ("group", "method", "n", "MASE", "RMSSE", "RMSSE_vs_ref_%", "ci_low_%", "ci_high_%", "bias_%",
               "lead_time_error")

    def to_pandas(self):  # pragma: no cover - thin wrapper
        """Return the rows as a pandas DataFrame (requires pandas)."""
        import pandas as pd

        return pd.DataFrame(self.rows, columns=list(self.COLUMNS))

    def to_markdown(self) -> str:
        """Markdown table, rounded for reading."""
        head = "| group | method | n | MASE | RMSSE | RMSSE vs " + self.reference + " (95 % CI) | bias | lead-time error |"
        lines = [head, "|" + "---|" * 8]
        for r in self.rows:
            if r["method"] == self.reference:
                vs = "reference"
            else:
                vs = f"{r['RMSSE_vs_ref_%']:+.1f} % [{r['ci_low_%']:+.1f}; {r['ci_high_%']:+.1f}]"
            lines.append(
                f"| {r['group']} | {r['method']} | {r['n']} | {r['MASE']:.3f} | {r['RMSSE']:.3f} | {vs} | "
                f"{r['bias_%']:+.0f} % | {r['lead_time_error']:+.3f} |"
            )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_markdown()


def compare(
    Y: ArrayLike,
    methods: Optional[Mapping[str, Method]] = None,
    origins: Optional[Sequence[int]] = None,
    horizon: int = 6,
    reference: str = "SES",
    lead_time: Optional[int] = None,
    n_boot: int = 4000,
    seed: int = 0,
    by_class: bool = False,
    min_class_size: int = 30,
) -> ComparisonTable:
    """Compare methods on a panel of series and return the table of the research note.

    Columns: mean MASE, mean RMSSE, mean RMSSE change against ``reference`` in % of the reference's mean
    RMSSE with a 95 % paired-bootstrap interval across series, pooled bias of the forecast total, and the
    mean scaled error on cumulative demand over ``lead_time``. With ``by_class`` the table is repeated for
    each ADI / CV² class holding at least ``min_class_size`` series.

    Choose method settings on an earlier window than the one passed here; this function does not tune.
    """
    methods = dict(default_methods() if methods is None else methods)
    if reference not in methods:
        raise ValueError(f"reference {reference!r} is not one of the methods {list(methods)}")
    sc = evaluate(Y, methods, origins, horizon, lead_time)
    n = int(sc.kept.sum())
    groups: List[tuple[str, np.ndarray]] = [("All", np.ones(n, bool))]
    if by_class:
        for c in ("Smooth", "Erratic", "Intermittent", "Lumpy"):
            m = sc.classes == c
            if m.sum() >= min_class_size:
                groups.append((c, m))
    rows: List[Dict[str, Any]] = []
    for g, sel in groups:
        base = sc.rmsse[reference][sel]
        for name in methods:
            r = sc.rmsse[name][sel]
            if name == reference:
                rel = (0.0, 0.0, 0.0)
            else:
                mu, lo, hi = paired_bootstrap(r - base, n_boot, seed)
                rel = tuple(100 * v / base.mean() for v in (mu, lo, hi))
            tot_a = sc.actual_total[sel].sum()
            rows.append({
                "group": g,
                "method": name,
                "n": int(sel.sum()),
                "MASE": float(sc.mase[name][sel].mean()),
                "RMSSE": float(r.mean()),
                "RMSSE_vs_ref_%": float(rel[0]),
                "ci_low_%": float(rel[1]),
                "ci_high_%": float(rel[2]),
                "bias_%": float(100 * (sc.forecast_total[name][sel].sum() / tot_a - 1)) if tot_a > 0 else float("nan"),
                "lead_time_error": float(sc.lead_time_error[name][sel].mean()),
            })
    return ComparisonTable(rows, reference, n, int((~sc.kept).sum()), sc)
