"""Rolling-origin evaluation: no look-ahead, hand-computed scores, comparison table."""
import numpy as np
import pytest

from intermittent_eval import compare, default_methods, evaluate, naive, paired_bootstrap, rolling_origin, ses


def panel(n=60, t=36, seed=1):
    rng = np.random.default_rng(seed)
    Y = rng.poisson(0.5, (n, t)) * rng.integers(1, 4, (n, t))
    return Y.astype(float)


def test_methods_only_see_the_past():
    seen = []

    def spy(y):
        seen.append(len(y))
        return 0.0

    rolling_origin(panel(2, 20), {"spy": spy}, origins=[10, 14], horizon=3)
    assert seen == [10, 14, 10, 14]


def test_future_values_do_not_change_forecasts():
    Y = panel(20, 30)
    Z = Y.copy()
    Z[:, 24:] = 999.0
    a = rolling_origin(Y, default_methods(), origins=[18, 24], horizon=6)
    b = rolling_origin(Z, default_methods(), origins=[18, 24], horizon=6)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_periods_after_the_horizon_do_not_change_scores():
    # origin 18, horizon 6 scores periods 18..23; forecasts and scales must ignore periods 24 and later
    Y = panel(20, 30)
    Z = Y.copy()
    Z[:, 24:] = 999.0
    s_y = evaluate(Y, origins=[18], horizon=6)
    s_z = evaluate(Z, origins=[18], horizon=6)
    np.testing.assert_array_equal(s_y.kept, s_z.kept)
    for k in s_y.rmsse:
        np.testing.assert_array_equal(s_y.rmsse[k], s_z.rmsse[k])
        np.testing.assert_array_equal(s_y.mase[k], s_z.mase[k])


def test_evaluate_by_hand():
    # history 0 2 0 0 3 0 (scale 2), naive forecasts 0, actual 1 0 2: MASE (1 + 0 + 2) / 3 / 2 = 0.5
    Y = np.array([[0, 2, 0, 0, 3, 0, 1, 0, 2]], dtype=float)
    s = evaluate(Y, {"Naive": naive}, origins=[6], horizon=3)
    assert s.mase["Naive"][0] == pytest.approx(0.5)
    assert s.rmsse["Naive"][0] == pytest.approx(np.sqrt((5 / 3) / (26 / 5)))
    # cumulative error over the horizon: (0 - 3) / (2 * 3)
    assert s.lead_time_error["Naive"][0] == pytest.approx(-0.5)


def test_scores_average_over_origins():
    Y = panel(10, 30)
    m = {"SES": lambda y: ses(y, 0.2)}
    both = evaluate(Y, m, origins=[20, 24], horizon=6)
    one = evaluate(Y, m, origins=[20], horizon=6)
    two = evaluate(Y, m, origins=[24], horizon=6)
    keep = both.kept[one.kept & two.kept]
    assert keep.all()
    np.testing.assert_allclose(both.rmsse["SES"], (one.rmsse["SES"] + two.rmsse["SES"]) / 2)


def test_constant_series_are_dropped():
    Y = panel(10, 30)
    Y[3] = 0.0
    t = compare(Y, horizon=6, n_boot=200)
    assert t.n_dropped == 1 and t.n_series == 9


def test_compare_table():
    Y = panel()
    t = compare(Y, horizon=6, n_boot=500, seed=3)
    names = [r["method"] for r in t.rows]
    assert names == list(default_methods())
    ref = next(r for r in t.rows if r["method"] == "SES")
    assert ref["RMSSE_vs_ref_%"] == 0.0
    zero = next(r for r in t.rows if r["method"] == "Zero")
    assert zero["bias_%"] == pytest.approx(-100.0)
    for r in t.rows:
        assert r["ci_low_%"] <= r["RMSSE_vs_ref_%"] <= r["ci_high_%"]
    assert "| All | SBA |" in t.to_markdown()


def test_compare_is_reproducible():
    Y = panel()
    assert compare(Y, n_boot=300, seed=5).rows == compare(Y, n_boot=300, seed=5).rows


def test_compare_by_class():
    Y = panel(80, 36)
    t = compare(Y, n_boot=200, by_class=True, min_class_size=1)
    groups = {r["group"] for r in t.rows}
    assert "All" in groups and len(groups) >= 2


def test_bad_arguments():
    Y = panel(5, 20)
    with pytest.raises(ValueError):
        compare(Y, reference="nope")
    with pytest.raises(ValueError):
        rolling_origin(Y, {"Naive": naive}, origins=[18], horizon=6)
    with pytest.raises(ValueError):
        rolling_origin(Y, {"bad": lambda y: np.zeros(2)}, origins=[10], horizon=3)


def test_paired_bootstrap_interval_contains_mean():
    d = np.random.default_rng(0).normal(0.1, 1, 500)
    mu, lo, hi = paired_bootstrap(d, 1000, seed=0)
    assert lo < mu < hi
    assert mu == pytest.approx(d.mean())
