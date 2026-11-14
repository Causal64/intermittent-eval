"""Cross-check against statsforecast (skipped when it is not installed).

statsforecast starts each smoothed quantity from its first observation. With the same starting values our
methods must return the same forecasts.
"""
import numpy as np
import pytest

sf = pytest.importorskip("statsforecast.models")

from intermittent_eval import croston, sba, ses, tsb  # noqa: E402

RNG = np.random.default_rng(11)
SERIES = [RNG.poisson(0.35, 48).astype(float) * RNG.integers(1, 6, 48) for _ in range(25)]
SERIES = [y for y in SERIES if (y > 0).sum() >= 2]


def _mean(model, y):
    return float(model.forecast(y=y, h=1)["mean"][0])


def test_croston_classic_matches():
    for y in SERIES:
        assert croston(y, 0.1) == pytest.approx(_mean(sf.CrostonClassic(), y), rel=1e-6)


def test_croston_sba_matches():
    for y in SERIES:
        assert sba(y, 0.1) == pytest.approx(_mean(sf.CrostonSBA(), y), rel=1e-6)


@pytest.mark.parametrize("a,b", [(0.1, 0.1), (0.3, 0.05), (0.2, 0.4)])
def test_tsb_matches(a, b):
    for y in SERIES:
        nz = y[y > 0]
        ours = tsb(y, a, b, size_init=nz[0], prob_init=float(y[0] > 0))
        assert ours == pytest.approx(_mean(sf.TSB(alpha_d=a, alpha_p=b), y), rel=1e-6)


@pytest.mark.parametrize("a", [0.1, 0.3])
def test_ses_matches(a):
    for y in SERIES:
        assert ses(y, a, init=y[0]) == pytest.approx(_mean(sf.SimpleExponentialSmoothing(alpha=a), y), rel=1e-6)
