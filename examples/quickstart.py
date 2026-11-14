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
