"""Hand-computed values and structural properties of the forecasting methods."""
import numpy as np
import pytest

from intermittent_eval import croston, default_methods, moving_average, naive, sba, ses, tsb, zero

Y = np.array([0, 2, 0, 0, 3, 0], dtype=float)


# ---------------------------------------------------------------- hand-computed (alpha = beta = 0.5)
def test_croston_by_hand():
    # size 2, interval 2 (first demand at period 2); at period 5: size 2.5, interval 2 + 0.5 * (3 - 2) = 2.5
    assert croston(Y, 0.5) == pytest.approx(1.0)


def test_sba_by_hand():
    assert sba(Y, 0.5) == pytest.approx(0.75)


def test_tsb_by_hand():
    # size starts at mean(2, 3) = 2.5, probability at 2/6; final probability 55/192, final size 2.625
    assert tsb(Y, 0.5, 0.5) == pytest.approx(55 / 192 * 2.625)


def test_ses_by_hand():
    # level starts at mean(Y) = 5/6, then 5/12, 29/24, 29/48, 29/96, 317/192, 317/384
    assert ses(Y, 0.5) == pytest.approx(317 / 384)


def test_controls_by_hand():
    assert zero(Y) == 0.0
    assert naive(Y) == 0.0
    assert moving_average(Y, 3) == pytest.approx(1.0)
    assert moving_average(Y, 100) == pytest.approx(5 / 6)


# ---------------------------------------------------------------- properties
RNG = np.random.default_rng(0)
SERIES = [RNG.poisson(0.4, 40).astype(float) * RNG.integers(1, 5, 40) for _ in range(30)]
SERIES = [s for s in SERIES if (s > 0).sum() >= 2]
RATES = [0.05, 0.1, 0.3, 0.7]


@pytest.mark.parametrize("a", RATES)
def test_sba_is_croston_times_correction(a):
    for y in SERIES:
        assert sba(y, a) == pytest.approx(croston(y, a) * (1 - a / 2))


@pytest.mark.parametrize("a", RATES)
@pytest.mark.parametrize("b", RATES)
def test_tsb_without_zeros_is_ses(a, b):
    # every period has a demand: the probability stays at 1 and TSB smooths the sizes like SES
    for y in SERIES:
        pos = y + 1.0
        assert tsb(pos, a, b) == pytest.approx(ses(pos, a, init=pos[:3].mean()))


@pytest.mark.parametrize("a", RATES)
def test_tsb_with_frozen_probability_is_ses_on_sizes(a):
    # beta = 0: the probability never moves, TSB = initial probability x SES of the non-zero sizes
    for y in SERIES:
        nz = y[y > 0]
        p0 = (y[:12] > 0).mean()
        assert tsb(y, a, 0.0) == pytest.approx(p0 * ses(nz, a, init=nz[:3].mean()))


@pytest.mark.parametrize("a", RATES)
def test_croston_without_zeros_is_ses(a):
    for y in SERIES:
        pos = y + 1.0
        assert croston(pos, a) == pytest.approx(ses(pos[1:], a, init=pos[0]))


def test_tsb_decays_after_demand_stops_croston_does_not():
    y = np.array([0, 3, 0, 2, 0, 4, 0, 3] + [0] * 24, dtype=float)
    assert tsb(y, 0.3, 0.1) < 0.1 * tsb(y[:8], 0.3, 0.1)
    assert croston(y, 0.3) == pytest.approx(croston(y[:8], 0.3))


def test_rate_zero_keeps_initial_state():
    assert ses(Y, 0.0) == pytest.approx(Y.mean())
    assert croston(Y, 0.0) == pytest.approx(2 / 2)


def test_no_demand_returns_zero():
    y = np.zeros(10)
    assert croston(y) == 0.0 and sba(y) == 0.0 and tsb(y) == 0.0 and ses(y) == 0.0


def test_forecasts_are_non_negative():
    for y in SERIES:
        for name, f in default_methods(0.3, 0.2).items():
            assert f(y) >= 0, name


@pytest.mark.parametrize("bad", [[], [1.0, np.nan], [1.0, -1.0], [[1.0, 2.0]]])
def test_invalid_history_raises(bad):
    with pytest.raises(ValueError):
        croston(np.array(bad, dtype=float))


def test_invalid_rate_raises():
    with pytest.raises(ValueError):
        ses(Y, 1.5)
    with pytest.raises(ValueError):
        tsb(Y, 0.1, -0.1)
