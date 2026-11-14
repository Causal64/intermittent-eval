# intermittent-eval

Croston, SBA and TSB forecasts for intermittent demand, with a rolling-origin evaluation that never looks
ahead and the error measures that stay meaningful when most periods are zero.

This is the companion code of the Causal64 research note
[Intermittent demand forecasting, measured](https://causal64.com/research/intermittent-demand-forecasting/).
It reproduces the classical-method results of that note exactly (see [Measured result](#measured-result)).

## What it does

- **Methods.** `croston`, `sba` (Syntetos-Boylan approximation), `tsb` (Teunter-Syntetos-Babai), `ses`, and
  three controls: `zero`, `naive`, `moving_average`. Each takes the history of one series and returns a flat
  point forecast. Any function with the same signature can be added to a comparison.
- **Leak-free evaluation.** `rolling_origin` and `evaluate` fit each method at each origin on the data before
  that origin only, and scale each error with the same data.
- **Metrics.** `mase`, `rmsse`, `bias` (forecast total over actual total), `cumulative_error` (error on the
  cumulative demand over the lead time, optionally scaled), `pinball`, `scaled_pinball`, `coverage`, and
  `classify` (ADI, CV² and the Syntetos-Boylan-Croston class). MAPE is left out on purpose: it divides by
  actual values that are zero in most periods.
- **Comparison table.** `compare` returns mean MASE, mean RMSSE, the RMSSE change against a reference method
  with a 95 % paired-bootstrap interval across series, the pooled bias and the lead-time error, optionally
  per demand class.

The only dependency is NumPy. `ComparisonTable.to_pandas()` needs pandas if you want a DataFrame.

## Installation

```
pip install intermittent-eval
```

Python 3.10 or later.

## Example

```python
import numpy as np

from intermittent_eval import compare, croston, default_methods, sba, tsb

# 200 simulated items over 48 months: a sale in 25 % of months, 1 to 4 units each time
rng = np.random.default_rng(0)
Y = rng.binomial(1, 0.25, (200, 48)) * rng.integers(1, 5, (200, 48))

# one item, three forecasts of the mean demand per month
print(round(croston(Y[0], 0.1), 3), round(sba(Y[0], 0.1), 3), round(tsb(Y[0], 0.1, 0.1), 3))

# three forecast origins, six months ahead, each forecast fitted on the past only
methods = default_methods(alpha=0.1, beta=0.1, window=12)
table = compare(Y, methods, origins=[30, 36, 42], horizon=6, reference="SES", seed=0)
print(table)
```

Output (NumPy 2.4):

```
0.54 0.513 0.525
| group | method | n | MASE | RMSSE | RMSSE vs SES (95 % CI) | bias | lead-time error |
|---|---|---|---|---|---|---|---|
| All | Zero | 200 | 0.667 | 0.681 | -1.0 % [-3.1; +1.3] | -100 % | -0.667 |
| All | Naive | 200 | 1.065 | 0.859 | +24.8 % [+19.6; +30.3] | -1 % | -0.040 |
| All | Moving average | 200 | 0.984 | 0.699 | +1.5 % [+0.9; +2.2] | +4 % | -0.041 |
| All | SES | 200 | 0.978 | 0.688 | reference | +3 % | -0.050 |
| All | Croston | 200 | 1.061 | 0.708 | +2.9 % [+0.8; +5.3] | +23 % | +0.096 |
| All | SBA | 200 | 1.040 | 0.701 | +1.9 % [-0.1; +4.1] | +17 % | +0.058 |
| All | TSB | 200 | 0.979 | 0.688 | +0.0 % [-0.4; +0.5] | +3 % | -0.053 |
```

`Y` has shape `(n_series, n_periods)`. An origin is the number of periods a method sees; the default is a
single origin at `n_periods - horizon`. The forecast of zero has the lowest MASE in this simulation, which is
the reason not to rank intermittent forecasts on MASE.

## Measured result

The note compares the methods on the car parts data of Hyndman, Koehler, Ord and Snyder (2008): 2,404 monthly
series kept out of 2,674, 74.0 % of months with zero demand. Settings were chosen on months 40 to 45 and the
results read once on months 46 to 51. Sealed window, figures from the note:

| Method | Mean RMSSE | RMSSE against SES, 95 % paired interval | Bias | Mean MASE |
|---|---|---|---|---|
| Zero | 0.526 | -6.4 % [-8.1; -4.7] | -100 % | 0.625 |
| Moving average, 12 months | 0.557 | -0.9 % [-1.4; -0.3] | +18 % | 0.900 |
| TSB | 0.561 | -0.2 % [-0.3; +0.0] | +21 % | 0.922 |
| SES | 0.562 | reference | +23 % | 0.921 |
| SBA | 0.590 | +5.0 % [+3.9; +6.0] | +16 % | 0.990 |
| Croston | 0.615 | +9.5 % [+8.4; +10.6] | +37 % | 1.060 |
| Naive (last month) | 0.618 | +10.1 % [+7.2; +13.1] | -10 % | 0.872 |

What the note concludes, in short: SBA beats Croston (RMSSE 0.590 against 0.615, bias +16 % against +37 %);
the 12-month moving average, TSB and SES finish within 1 % of each other; the forecast of zero wins on MASE
(0.625) and, because demand fell over the period, also on RMSSE, which is a warning about the window rather
than advice. The note also covers a pooled gradient-boosting model and 90 % quantile forecasts, which are not
part of this package.

`examples/carparts.py` reruns the protocol with this package. On the same data it returns the same mean MASE,
mean RMSSE and bias for every method in the table above, to the third decimal, and the same class counts
(2,090 intermittent, 309 lumpy). The bootstrap intervals agree to within 0.1 point (different random draws).
The data is distributed with the R package expsmooth on CRAN under the GPL and is not bundled here; the
script explains how to obtain it.

## Conventions

The initialisations follow the note, so that its figures can be reproduced:

- SES starts from the mean of the first 12 periods (or `init`).
- Croston and SBA start from the first non-zero demand and the number of periods up to it; sizes and
  intervals are updated only in periods with a demand. SBA multiplies Croston by `1 - alpha_interval / 2`.
- TSB starts its size from the mean of the first three non-zero demands and its probability from the share of
  non-zero periods among the first 12 (or `size_init`, `prob_init`).

With the starting values of statsforecast (`ses(y, a, init=y[0])`,
`tsb(y, a, b, size_init=first_nonzero, prob_init=float(y[0] > 0))`) the forecasts match statsforecast's
`SimpleExponentialSmoothing`, `CrostonClassic`, `CrostonSBA` and `TSB`; the test suite checks this when
statsforecast is installed (checked with statsforecast 2.1.1).

## Limits

- The methods return flat point forecasts. There are no prediction intervals and no quantile models; for a
  service level you need a distribution on top (the note compares four ways of getting a 90 % quantile).
- Nothing is tuned for you. Choose smoothing constants on an earlier window than the one you report, as the
  note does; `compare` only scores.
- The loops are plain Python over series and origins. Tens of thousands of series with many origins will be
  slow; statsforecast is the better tool for forecasting at that scale.
- The result above comes from one public dataset (1998 to 2002), one sealed window of six months and a
  declining catalogue. A different catalogue can rank the methods differently.
- Series whose history is constant before an origin cannot be scaled and are dropped (`n_dropped`).

## Citation

If you use this package, please cite the note:

```bibtex
@misc{causal64_intermittent_2026,
  author       = {{Causal64}},
  title        = {Intermittent demand forecasting, measured},
  year         = {2026},
  howpublished = {\url{https://causal64.com/research/intermittent-demand-forecasting/}},
  note         = {Software: intermittent-eval, https://github.com/causal64/intermittent-eval}
}
```

and the method papers: Croston (1972), doi:10.1057/jors.1972.50; Syntetos and Boylan (2005),
doi:10.1016/j.ijforecast.2004.10.001; Teunter, Syntetos and Babai (2011), doi:10.1016/j.ejor.2011.05.018;
Syntetos, Boylan and Croston (2005), doi:10.1057/palgrave.jors.2601841; Hyndman and Koehler (2006),
doi:10.1016/j.ijforecast.2006.03.001.

## License

MIT. See [LICENSE](LICENSE).
