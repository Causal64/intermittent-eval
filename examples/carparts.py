"""Reproduce the classical-method rows of the Causal64 note on the carparts data.

The carparts data (Hyndman, Koehler, Ord and Snyder, 2008) ships with the R package expsmooth on CRAN
under the GPL, so it is not bundled here. Download expsmooth_2.3.tar.gz from CRAN, unpack it, then:

    pip install pyreadr
    python examples/carparts.py path/to/expsmooth/data/carparts.rda

Protocol of the note: drop series with missing months and series with fewer than two demands (or a
constant history) in months 1-39; settings chosen on months 40-45; report months 46-51 from origin 45.
"""
from __future__ import annotations

import sys

import numpy as np

from intermittent_eval import compare, croston, moving_average, naive, sba, ses, tsb, zero


def load(path: str) -> np.ndarray:
    import pyreadr

    df = pyreadr.read_r(path)["carparts"]
    return df.to_numpy(dtype=float).T  # (series, 51 months)


def main(path: str) -> None:
    Y = load(path)
    Y = Y[~np.isnan(Y).any(1)]
    dev = Y[:, :39]
    Y = Y[((dev > 0).sum(1) >= 2) & (np.abs(np.diff(dev, axis=1)).mean(1) > 0)]
    print(f"series kept: {len(Y)}, share of zero months: {(Y == 0).mean():.3f}")

    # settings chosen on the development window in the note
    methods = {
        "Zero": zero,
        "Naive": naive,
        "Moving average": lambda y: moving_average(y, 12),
        "SES": lambda y: ses(y, 0.1),
        "Croston": lambda y: croston(y, 0.3),
        "SBA": lambda y: sba(y, 0.3),
        "TSB": lambda y: tsb(y, 0.3, 0.1),
    }
    table = compare(Y[:, :51], methods, origins=[45], horizon=6, reference="SES", seed=2026, by_class=True)
    print(table)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
