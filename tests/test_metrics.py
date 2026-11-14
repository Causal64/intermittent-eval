"""Hand-computed values of the error measures."""
import math

import numpy as np
import pytest

from intermittent_eval import (
    bias,
    classify,
    coverage,
    cumulative_error,
    mase,
    pinball,
    rmsse,
    scaled_pinball,
    scales,
)

INSAMPLE = np.array([0, 2, 0, 0, 3, 0], dtype=float)  # changes 2, -2, 0, 3, -3
ACTUAL = np.array([1, 0, 2], dtype=float)
FORECAST = 1.0


def test_scales_by_hand():
    s1, s2 = scales(INSAMPLE)
    assert s1 == pytest.approx(10 / 5)
    assert s2 == pytest.approx(26 / 5)


def test_mase_by_hand():
    # errors 0, -1, 1: mean absolute error 2/3, scale 2
    assert mase(ACTUAL, FORECAST, INSAMPLE) == pytest.approx(1 / 3)


def test_rmsse_by_hand():
    assert rmsse(ACTUAL, FORECAST, INSAMPLE) == pytest.approx(math.sqrt((2 / 3) / (26 / 5)))


def test_zero_forecast_wins_mase_when_most_periods_are_zero():
    # the median of future demand is zero, the mean is not: MASE prefers zero, RMSSE prefers the mean
    insample = np.array([0, 0, 0, 4, 0, 0, 0, 4], dtype=float)
    actual = np.array([0, 0, 0, 4], dtype=float)
    assert mase(actual, 0.0, insample) < mase(actual, 1.0, insample)
    assert rmsse(actual, 1.0, insample) < rmsse(actual, 0.0, insample)


def test_bias_by_hand():
    assert bias(ACTUAL, FORECAST) == pytest.approx(0.0)
    assert bias(ACTUAL, 1.5) == pytest.approx(0.5)
    assert bias(ACTUAL, 0.0) == pytest.approx(-1.0)
    assert math.isnan(bias(np.zeros(3), 1.0))


def test_cumulative_error_by_hand():
    # lead time 2: forecast total 2, actual total 1
    assert cumulative_error(ACTUAL, FORECAST, 2) == pytest.approx(1.0)
    assert cumulative_error(ACTUAL, FORECAST, 2, INSAMPLE) == pytest.approx(1.0 / (2 * 2))
    assert cumulative_error(ACTUAL, FORECAST) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        cumulative_error(ACTUAL, FORECAST, 4)


def test_pinball_by_hand():
    # errors 0, -1, 1 at tau 0.9: losses 0, 0.1, 0.9
    np.testing.assert_allclose(pinball(ACTUAL, 1.0, 0.9), [0.0, 0.1, 0.9])
    assert scaled_pinball(ACTUAL, 1.0, INSAMPLE, 0.9) == pytest.approx((1.0 / 3) / 2)


def test_coverage_by_hand():
    assert coverage(ACTUAL, 1.0) == pytest.approx(2 / 3)


def test_classify_by_hand():
    adi, cv2, cls = classify(INSAMPLE)
    assert adi == pytest.approx(3.0)
    assert cv2 == pytest.approx((0.5 / 2.5) ** 2)
    assert cls == "Intermittent"
    assert classify([1, 1, 1, 1])[2] == "Smooth"
    assert classify([1, 9, 1, 9])[2] == "Erratic"
    assert classify([0, 0, 1, 0, 0, 9])[2] == "Lumpy"


def test_constant_history_raises():
    with pytest.raises(ValueError):
        mase(ACTUAL, FORECAST, np.ones(5))
