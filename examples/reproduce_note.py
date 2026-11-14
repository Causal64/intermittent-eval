"""Recompute every number of the research note "Intermittent demand forecasting, measured"
(https://causal64.com/research/intermittent-demand-forecasting/).

Two parts:
  first_reading()  the comparison as first run: settings chosen on the development window (months 40-45),
                   then the test window (months 46-51) scored with those settings.
  post_hoc()       the checks added after that first reading. They change no setting and no number of it.
                   It asserts that the first-reading numbers are reproduced before computing anything else.

Data: carparts (Hyndman, Koehler, Ord and Snyder, 2008), from the CRAN package expsmooth (GPL >= 2). It is not
redistributed here. Download https://cran.r-project.org/src/contrib/expsmooth_2.3.tar.gz, unpack it, and pass the
path of expsmooth/data/carparts.rda.

Requirements beyond the package: scipy, lightgbm, pyreadr, pandas. Numbers in the note were produced with
Python 3.12, numpy 2.4.4, scipy 1.18.1, lightgbm 4.6.0 (2 threads, deterministic, fixed seeds). Bootstrap
intervals can move in the last digit with other versions.

Usage: python examples/reproduce_note.py path/to/carparts.rda
Writes results_first_reading.json and results_post_hoc.json in the current directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("carparts.rda")
OUT = {}


def boot_ci(d: np.ndarray, rng, b: int = 4000):
    """Paired bootstrap across series of the mean of d (one value per series)."""
    n = len(d)
    idx = rng.integers(0, n, size=(b, n))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def load_carparts() -> np.ndarray:
    import pyreadr
    r = pyreadr.read_r(str(DATA))
    df = r["carparts"]
    return df.to_numpy(dtype=float)  # (51 months, 2674 series)


def ses(y, a):
    lvl = y[: min(12, len(y))].mean()
    for v in y:
        lvl = lvl + a * (v - lvl)
    return lvl


def croston(y, a, variant="croston"):
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0
    z = y[nz[0]]
    p = nz[0] + 1.0
    q = 1
    for t in range(nz[0] + 1, len(y)):
        if y[t] > 0:
            z = z + a * (y[t] - z)
            p = p + a * (q - p)
            q = 1
        else:
            q += 1
    f = z / p
    return f * (1 - a / 2) if variant == "sba" else f


def tsb(y, a, b):
    nz = np.flatnonzero(y > 0)
    if len(nz) == 0:
        return 0.0
    z = y[nz].mean() if len(nz) < 3 else y[nz[:3]].mean()
    p = (y[: min(12, len(y))] > 0).mean()
    for v in y:
        if v > 0:
            z = z + a * (v - z)
            p = p + b * (1 - p)
        else:
            p = p + b * (0 - p)
    return p * z


def point_forecasts(Y, origin, params):
    """Y: (series, months). Flat forecasts from data up to `origin` (exclusive index)."""
    n = Y.shape[0]
    out = {}
    hist = Y[:, :origin]
    out["Zero"] = np.zeros(n)
    out["Naive"] = hist[:, -1].copy()
    out["Moving average"] = hist[:, -params["ma"]:].mean(1)
    out["SES"] = np.array([ses(y, params["ses"]) for y in hist])
    out["Croston"] = np.array([croston(y, params["croston"]) for y in hist])
    out["SBA"] = np.array([croston(y, params["sba"], "sba") for y in hist])
    out["TSB"] = np.array([tsb(y, *params["tsb"]) for y in hist])
    return out


def scales(Y, origin):
    d = np.diff(Y[:, :origin], axis=1)
    return np.abs(d).mean(1), (d ** 2).mean(1)


def mase_rmsse(F, Yt, s1, s2):
    e = Yt - F[:, None]
    return np.abs(e).mean(1) / s1, np.sqrt((e ** 2).mean(1) / s2)


def gbm_frames(Y, last_origin, horizons=6, first_origin=12, target=True):
    """Direct multi-horizon pooled design. One row per (series, origin, h)."""
    rows, ys, keys = [], [], []
    n, T = Y.shape
    origins = range(first_origin, last_origin + 1)
    for o in origins:
        hist = Y[:, :o]
        sc = hist.mean(1)
        sc = np.where(sc > 0, sc, 1.0)
        h_ = hist / sc[:, None]
        lags = h_[:, -12:][:, ::-1]
        nzpos = np.where(hist > 0, np.arange(o)[None, :], -1).max(1)
        since = np.where(nzpos >= 0, o - nzpos, o + 12)
        feats = np.column_stack([
            lags, h_[:, -3:].mean(1), h_[:, -6:].mean(1), h_[:, -12:].mean(1), h_.mean(1),
            (hist[:, -12:] > 0).mean(1), (hist > 0).mean(1), since, np.log1p(sc),
        ])
        for h in range(1, horizons + 1):
            if target and o + h - 1 >= T:
                continue
            month = (o + h - 1) % 12
            rows.append(np.column_stack([feats, np.full(n, h), np.full(n, month)]))
            keys.append(np.column_stack([np.arange(n), np.full(n, o), np.full(n, h), sc]))
            if target:
                ys.append(Y[:, o + h - 1] / sc)
    X = np.vstack(rows)
    K = np.vstack(keys)
    yv = np.concatenate(ys) if target else None
    return X, yv, K


def gbm_forecast(Y, origin, objective, alpha=None, seed=7):
    import lightgbm as lgb
    # training rows: origins whose 6 targets all lie before `origin`
    Xtr, ytr, Ktr = gbm_frames(Y[:, :origin], origin - 1)
    keep = (Ktr[:, 1] + Ktr[:, 2] - 1) < origin
    Xtr, ytr = Xtr[keep], ytr[keep]
    params = dict(objective=objective, learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=seed, verbose=-1,
                  num_threads=2, deterministic=True)
    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.3
    if objective == "quantile":
        params["alpha"] = alpha
    model = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=300)
    Xte, _, Kte = gbm_frames(Y[:, :origin], origin, target=False, first_origin=origin)
    pred = model.predict(Xte) * Kte[:, 3]
    pred = np.maximum(pred, 0)
    out = np.zeros((Y.shape[0], 6))
    out[Kte[:, 0].astype(int), Kte[:, 2].astype(int) - 1] = pred
    return out


def classify(Y, origin):
    adi, cv2 = np.zeros(len(Y)), np.zeros(len(Y))
    for i, y in enumerate(Y[:, :origin]):
        nz = np.flatnonzero(y > 0)
        adi[i] = origin / len(nz)
        s = y[nz]
        cv2[i] = (s.std() / s.mean()) ** 2 if len(s) > 1 else 0.0
    cls = np.where(adi < 1.32, np.where(cv2 < 0.49, "Smooth", "Erratic"),
                   np.where(cv2 < 0.49, "Intermittent", "Lumpy"))
    return adi, cv2, cls


def pinball(q, y, tau):
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def first_reading():
    rng = np.random.default_rng(2026)
    Yall = load_carparts().T  # (series, 51)
    complete = ~np.isnan(Yall).any(1)
    Y = Yall[complete]
    # eligibility decided on the development training months only (1..39)
    nzdev = (Y[:, :39] > 0).sum(1)
    s1d, _ = scales(Y, 39)
    elig = (nzdev >= 2) & (s1d > 0)
    Y = Y[elig]
    n = len(Y)
    print(f"[demand] series {Yall.shape[0]}, complete {complete.sum()}, eligible {n}")
    zero_share = (Y == 0).mean()
    print(f"[demand] share of zero months in eligible series: {zero_share:.3f}")
    OUT["n_total"], OUT["n_complete"], OUT["n_elig"] = int(Yall.shape[0]), int(complete.sum()), int(n)
    OUT["zero_share"] = float(zero_share)

    # ---------- development window: choose settings (origin 39, months 40-45)
    s1, s2 = scales(Y, 39)
    Yt = Y[:, 39:45]
    grid = [0.05, 0.1, 0.2, 0.3]
    hist = Y[:, :39]

    def dev_score(F):
        return mase_rmsse(F, Yt, s1, s2)[1].mean()

    best = {}
    best["ma"] = min([3, 6, 12, 24], key=lambda w: dev_score(hist[:, -w:].mean(1)))
    best["ses"] = min(grid, key=lambda a: dev_score(np.array([ses(y, a) for y in hist])))
    best["croston"] = min(grid, key=lambda a: dev_score(np.array([croston(y, a) for y in hist])))
    best["sba"] = min(grid, key=lambda a: dev_score(np.array([croston(y, a, "sba") for y in hist])))
    best["tsb"] = min([(a, b) for a in grid for b in grid],
                      key=lambda ab: dev_score(np.array([tsb(y, *ab) for y in hist])))
    dev_gbm = {}
    for obj in ("regression", "tweedie"):
        G = gbm_forecast(Y, 39, obj)
        e = Yt - G
        dev_gbm[obj] = float(np.sqrt((e ** 2).mean(1) / s2).mean())
    best["gbm"] = min(dev_gbm, key=dev_gbm.get)
    print(f"[demand] settings chosen on development window: {best}; gbm dev RMSSE {dev_gbm}")
    OUT["settings"] = {k: v for k, v in best.items()}

    # ---------- test window: origin 45, months 46-51 (settings frozen above)
    s1, s2 = scales(Y, 45)
    Yt = Y[:, 45:51]
    adi, cv2, cls = classify(Y, 45)
    classes = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
    cnt = {c: int((cls == c).sum()) for c in classes}
    print(f"[demand] classes on months 1-45: {cnt}")
    OUT["classes"] = cnt
    P = point_forecasts(Y, 45, best)
    G = gbm_forecast(Y, 45, best["gbm"])
    mase, rmsse = {}, {}
    for k, F in P.items():
        mase[k], rmsse[k] = mase_rmsse(F, Yt, s1, s2)
    e = Yt - G
    mase["Pooled GBM"] = np.abs(e).mean(1) / s1
    rmsse["Pooled GBM"] = np.sqrt((e ** 2).mean(1) / s2)
    methods = ["Zero", "Naive", "Moving average", "SES", "Croston", "SBA", "TSB", "Pooled GBM"]
    tab = {}
    for m in methods:
        row = {"MASE": float(mase[m].mean()), "RMSSE": float(rmsse[m].mean())}
        for c in classes:
            row["RMSSE_" + c] = float(rmsse[m][cls == c].mean())
            row["MASE_" + c] = float(mase[m][cls == c].mean())
        tab[m] = row
        print(f"[demand] {m:15s} MASE {row['MASE']:.3f}  RMSSE {row['RMSSE']:.3f}  "
              + "  ".join(f"{c[:4]} {row['RMSSE_' + c]:.3f}" for c in classes))
    OUT["table"] = tab
    # paired bootstrap: each method vs naive and vs SES, on RMSSE (relative change of the mean)
    rel = {}
    for ref in ("Naive", "SES"):
        for m in methods:
            if m == ref:
                continue
            for c in ["All"] + classes:
                sel = np.ones(n, bool) if c == "All" else cls == c
                d = rmsse[m][sel] - rmsse[ref][sel]
                mu, lo, hi = boot_ci(d, rng)
                base = rmsse[ref][sel].mean()
                rel[(m, ref, c)] = (100 * mu / base, 100 * lo / base, 100 * hi / base)
    for m in methods:
        if m != "SES":
            print(f"[demand] RMSSE vs SES, all: {m:15s} {rel[(m, 'SES', 'All')][0]:+.1f}% "
                  f"[{rel[(m, 'SES', 'All')][1]:+.1f}; {rel[(m, 'SES', 'All')][2]:+.1f}]  per class "
                  + "  ".join(f"{c[:4]} {rel[(m, 'SES', c)][0]:+.1f} [{rel[(m, 'SES', c)][1]:+.1f};"
                              f"{rel[(m, 'SES', c)][2]:+.1f}]" for c in classes))
    for m in methods:
        if m != "Naive":
            r = rel[(m, "Naive", "All")]
            print(f"[demand] RMSSE vs Naive, all: {m:15s} {r[0]:+.1f}% [{r[1]:+.1f}; {r[2]:+.1f}]")
    OUT["rel"] = {" | ".join(k): v for k, v in rel.items()}
    # MASE trap: zero forecast vs others, share of series where zero has the best MASE
    wins_mase = np.mean(mase["Zero"] <= np.min(np.vstack([mase[m] for m in methods if m != "Zero"]), 0))
    print(f"[demand] zero forecast has the lowest MASE on {wins_mase:.3f} of series; "
          f"ranks MASE {sorted(methods, key=lambda m: tab[m]['MASE'])}; "
          f"ranks RMSSE {sorted(methods, key=lambda m: tab[m]['RMSSE'])}")
    OUT["zero_best_mase_share"] = float(wins_mase)
    # MAPE breaks: share of test months equal to zero
    print(f"[demand] share of zero months in the sealed window: {(Yt == 0).mean():.3f}; "
          f"series with at least one zero test month: {(Yt == 0).any(1).mean():.3f}")
    OUT["zero_test"] = float((Yt == 0).mean())
    OUT["zero_test_series"] = float((Yt == 0).any(1).mean())
    # demand drift: mean demand per series-month by year, and forecast bias in the sealed window
    for a_, b_ in ((0, 12), (12, 24), (24, 36), (36, 45), (45, 51)):
        print(f"[demand] months {a_ + 1}-{b_}: mean demand {Y[:, a_:b_].mean():.3f}, share non-zero {(Y[:, a_:b_] > 0).mean():.3f}")
    for k in methods:
        F = G.mean(1) if k == "Pooled GBM" else P[k]
        print(f"[demand] bias {k:15s} mean forecast {F.mean():.3f} vs mean actual {Yt.mean():.3f} "
              f"({100 * (F.mean() / Yt.mean() - 1):+.0f} %)")

    # ---------- 90 % quantile (service level)
    tau = 0.9
    sba = P["SBA"]
    q = {}
    q["Poisson around SBA"] = np.repeat(stats.poisson.ppf(tau, np.maximum(sba, 1e-9))[:, None], 6, 1)
    # negative binomial: variance from in-sample one-step SBA errors (months 25-45)
    var = np.zeros(n)
    for i, y in enumerate(Y):
        errs = [y[t] - croston(y[:t], best["sba"], "sba") for t in range(24, 45)]
        var[i] = np.mean(np.square(errs))
    mu = np.maximum(sba, 1e-6)
    var = np.maximum(var, mu * 1.0001)
    pnb = mu / var
    rnb = mu * pnb / (1 - pnb)
    q["Negative binomial around SBA"] = np.repeat(stats.nbinom.ppf(tau, rnb, pnb)[:, None], 6, 1)
    q["Empirical, last 24 months"] = np.repeat(np.quantile(Y[:, 21:45], tau, axis=1, method="inverted_cdf")[:, None], 6, 1)
    q["Pooled GBM, quantile loss"] = gbm_forecast(Y, 45, "quantile", alpha=tau)
    qtab = {}
    for k, Q in q.items():
        cov = (Yt <= Q).mean(1)
        spl = pinball(Q, Yt, tau).mean(1) / s1
        row = {"coverage": float(cov.mean()), "spl": float(spl.mean())}
        for c in classes:
            row["coverage_" + c] = float(cov[cls == c].mean())
            row["spl_" + c] = float(spl[cls == c].mean())
        qtab[k] = row
        print(f"[demand] q90 {k:30s} coverage {row['coverage']:.3f} scaled pinball {row['spl']:.3f}  "
              + "  ".join(f"{c[:4]} cov {row['coverage_' + c]:.3f} spl {row['spl_' + c]:.3f}" for c in classes))
    # bootstrap of pinball difference NB vs GBM
    spl_nb = pinball(q["Negative binomial around SBA"], Yt, tau).mean(1) / s1
    spl_gb = pinball(q["Pooled GBM, quantile loss"], Yt, tau).mean(1) / s1
    spl_po = pinball(q["Poisson around SBA"], Yt, tau).mean(1) / s1
    spl_em = pinball(q["Empirical, last 24 months"], Yt, tau).mean(1) / s1
    for nm, a_, b_ in (("GBM - NB", spl_gb, spl_nb), ("NB - Poisson", spl_nb, spl_po), ("GBM - Empirical", spl_gb, spl_em),
                       ("Empirical - Poisson", spl_em, spl_po)):
        mu_, lo_, hi_ = boot_ci(a_ - b_, rng)
        print(f"[demand] scaled pinball {nm}: {100 * mu_ / b_.mean():+.1f}% "
              f"[{100 * lo_ / b_.mean():+.1f}; {100 * hi_ / b_.mean():+.1f}]")
    OUT["qtab"] = qtab
    Path("results_first_reading.json").write_text(json.dumps(OUT, indent=1))


def boot_ratio(num: np.ndarray, den: np.ndarray, rng, b: int = 4000):
    """Bootstrap across series of sum(num) / sum(den) - 1 (the pooled bias)."""
    n = len(num)
    idx = rng.integers(0, n, size=(b, n))
    r = num[idx].sum(1) / den[idx].sum(1) - 1
    return float(num.sum() / den.sum() - 1), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def post_hoc() -> int:
    rng = np.random.default_rng(20261114)
    OUT.clear()
    Yall = load_carparts().T
    complete = ~np.isnan(Yall).any(1)

    # ---- (g) the 165 series dropped for missing months: where are the gaps?
    miss = Yall[~complete]
    first_obs = np.array([np.flatnonzero(~np.isnan(y))[0] for y in miss])
    last_obs = np.array([np.flatnonzero(~np.isnan(y))[-1] for y in miss])
    inner = np.array([np.isnan(y[f:l + 1]).any() for y, f, l in zip(miss, first_obs, last_obs)])
    OUT["dropped_missing"] = {"n": int(len(miss)), "start_late": int((first_obs > 0).sum()),
                              "end_early": int((last_obs < 50).sum()), "inner_gap": int(inner.sum()),
                              "median_first_month": float(np.median(first_obs) + 1),
                              "median_missing": float(np.median(np.isnan(miss).sum(1)))}
    print("[posthoc] dropped for missing months:", OUT["dropped_missing"])

    Y = Yall[complete]
    nzdev = (Y[:, :39] > 0).sum(1)
    s1d, _ = scales(Y, 39)
    one = (nzdev == 1).sum(), (nzdev == 0).sum(), ((nzdev >= 2) & (s1d == 0)).sum()
    print("[posthoc] complete series with exactly 1 demand in months 1-39:", int(one[0]), "; with none:", int(one[1]),
          "; >=2 demands but zero scale:", int(one[2]))
    OUT["dropped_sparse"] = {"one_demand": int(one[0]), "no_demand": int(one[1])}
    Y = Y[(nzdev >= 2) & (s1d > 0)]
    n = len(Y)
    assert n == 2404, n

    # ---- development window, original grid (must give the published settings) and extended grid
    s1, s2 = scales(Y, 39)
    Yt = Y[:, 39:45]
    hist = Y[:, :39]

    def dev(F):
        return mase_rmsse(F, Yt, s1, s2)[1].mean()

    def choose(grid):
        b = {"ma": min([3, 6, 12, 24], key=lambda w: dev(hist[:, -w:].mean(1))),
             "ses": min(grid, key=lambda a: dev(np.array([ses(y, a) for y in hist]))),
             "croston": min(grid, key=lambda a: dev(np.array([croston(y, a) for y in hist]))),
             "sba": min(grid, key=lambda a: dev(np.array([croston(y, a, "sba") for y in hist]))),
             "tsb": min([(a, c) for a in grid for c in grid], key=lambda ab: dev(np.array([tsb(y, *ab) for y in hist])))}
        return b

    base = choose([0.05, 0.1, 0.2, 0.3])
    assert base == {"ma": 12, "ses": 0.1, "croston": 0.3, "sba": 0.3, "tsb": (0.3, 0.1)}, base
    wide = choose([0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    print("[posthoc] settings, published grid:", base, "| grid extended to 0.5:", wide)
    OUT["settings_wide"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in wide.items()}
    dev_curve = {a: float(dev(np.array([croston(y, a, "sba") for y in hist]))) for a in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)}
    print("[posthoc] SBA development RMSSE by alpha:", {k: round(v, 4) for k, v in dev_curve.items()})
    OUT["sba_dev_curve"] = dev_curve

    # ---- test window (post hoc from here)
    s1, s2 = scales(Y, 45)
    Yt = Y[:, 45:51]
    _, _, cls = classify(Y, 45)
    P = point_forecasts(Y, 45, base)
    G = gbm_forecast(Y, 45, "tweedie")
    rm = {k: mase_rmsse(F, Yt, s1, s2)[1] for k, F in P.items()}
    e = Yt - G
    rm["Pooled GBM"] = np.sqrt((e ** 2).mean(1) / s2)
    assert abs(rm["SES"].mean() - 0.562) < 5e-4 and abs(rm["Pooled GBM"].mean() - 0.574) < 5e-4, "published numbers not reproduced"

    # (e) extended grid on the test window
    Pw = point_forecasts(Y, 45, wide)
    rmw = {k: mase_rmsse(F, Yt, s1, s2)[1] for k, F in Pw.items()}
    OUT["wide_grid_test"] = {k: {"rmsse": float(rmw[k].mean()), "published": float(rm[k].mean()),
                                 "bias": float(Pw[k].mean() / Yt.mean() - 1)} for k in ("SES", "Croston", "SBA", "TSB")}
    print("[posthoc] test RMSSE with the extended grid:", {k: (round(v["rmsse"], 3), round(v["published"], 3), f"{100 * v['bias']:+.0f}%")
                                                           for k, v in OUT["wide_grid_test"].items()})

    # (b) pooled model against each plain method, paired, by class
    pair = {}
    for ref in ("Moving average", "TSB", "SES"):
        for c in ("All", "Intermittent", "Lumpy"):
            sel = np.ones(n, bool) if c == "All" else cls == c
            d = rm["Pooled GBM"][sel] - rm[ref][sel]
            mu, lo, hi = boot_ci(d, rng)
            b_ = rm[ref][sel].mean()
            pair[f"Pooled GBM vs {ref} | {c}"] = [100 * mu / b_, 100 * lo / b_, 100 * hi / b_]
            print(f"[posthoc] pooled vs {ref:15s} {c:12s} {100 * mu / b_:+.1f}% [{100 * lo / b_:+.1f}; {100 * hi / b_:+.1f}]")
    OUT["pairs"] = pair

    # (c) one-parameter control: SES times the catalogue-wide change measured before the origin.
    # g = total demand of the last 6 months before the origin / total of the 6 months before those (all series).
    g = Y[:, 39:45].sum() / Y[:, 33:39].sum()
    ctrl = P["SES"] * g
    r_ctrl = mase_rmsse(ctrl, Yt, s1, s2)[1]
    OUT["control"] = {"g": float(g), "rmsse": float(r_ctrl.mean()), "bias": float(ctrl.mean() / Yt.mean() - 1)}
    print(f"[posthoc] control SES x g, g = {g:.3f}: RMSSE {r_ctrl.mean():.3f}, bias {100 * (ctrl.mean() / Yt.mean() - 1):+.0f}%")
    for c in ("All", "Lumpy"):
        sel = np.ones(n, bool) if c == "All" else cls == c
        for nm, a_ in (("control vs SES", r_ctrl - rm["SES"]), ("pooled vs control", rm["Pooled GBM"] - r_ctrl)):
            mu, lo, hi = boot_ci(a_[sel], rng)
            b_ = (rm["SES"] if nm.endswith("SES") else r_ctrl)[sel].mean()
            OUT["control"][f"{nm} | {c}"] = [100 * mu / b_, 100 * lo / b_, 100 * hi / b_]
            print(f"[posthoc] {nm:18s} {c:6s} {100 * mu / b_:+.1f}% [{100 * lo / b_:+.1f}; {100 * hi / b_:+.1f}]")

    # (d) bias with an interval, and the six-month cumulative error
    bias = {}
    tot = Yt.sum(1)
    for k in ("Zero", "Naive", "Moving average", "SES", "Croston", "SBA", "TSB", "Pooled GBM"):
        F6 = G.sum(1) if k == "Pooled GBM" else P[k] * 6
        b0, lo, hi = boot_ratio(F6, tot, rng)
        cum = np.abs(F6 - tot) / (6 * s1)
        bias[k] = {"bias": b0, "lo": lo, "hi": hi, "cum_err": float(cum.mean())}
        print(f"[posthoc] bias {k:15s} {100 * b0:+.0f}% [{100 * lo:+.0f}; {100 * hi:+.0f}]  scaled 6-month cumulative error {cum.mean():.3f}")
    OUT["bias"] = bias
    # unit-weighted bias is driven by large parts: the same ratio for the median series
    med = {k: float(np.median((G.mean(1) if k == "Pooled GBM" else P[k]) - Yt.mean(1))) for k in bias}
    OUT["median_error_per_series"] = med
    share_top = float(np.sort(tot)[::-1][: n // 10].sum() / tot.sum())
    OUT["top_decile_share_of_demand"] = share_top
    print(f"[posthoc] the 10 % largest series hold {100 * share_top:.0f}% of test demand")

    # (f) quantiles: P(Y < q) and P(Y <= q) must bracket 0.9; Poisson around other bases
    tau = 0.9
    q = {}
    for nm, F in (("SBA", P["SBA"]), ("TSB", P["TSB"]), ("moving average", P["Moving average"]), ("SES", P["SES"])):
        q[f"Poisson around {nm}"] = np.repeat(stats.poisson.ppf(tau, np.maximum(F, 1e-9))[:, None], 6, 1)
    q["Poisson around pooled GBM"] = stats.poisson.ppf(tau, np.maximum(G, 1e-9))
    var = np.zeros(n)
    for i, y in enumerate(Y):
        errs = [y[t] - croston(y[:t], base["sba"], "sba") for t in range(24, 45)]
        var[i] = np.mean(np.square(errs))
    mu = np.maximum(P["SBA"], 1e-6)
    floored = var <= mu * 1.0001
    var = np.maximum(var, mu * 1.0001)
    pnb = mu / var
    q["Negative binomial around SBA"] = np.repeat(stats.nbinom.ppf(tau, mu * pnb / (1 - pnb), pnb)[:, None], 6, 1)
    q["Empirical, last 24 months"] = np.repeat(np.quantile(Y[:, 21:45], tau, axis=1, method="inverted_cdf")[:, None], 6, 1)
    q["Pooled GBM, quantile loss"] = gbm_forecast(Y, 45, "quantile", alpha=tau)
    qt = {}
    spl = {}
    for k, Q in q.items():
        spl[k] = pinball(Q, Yt, tau).mean(1) / s1
        qt[k] = {"below": float((Yt < Q).mean()), "at_or_below": float((Yt <= Q).mean()), "spl": float(spl[k].mean()),
                 "mean_q": float(Q.mean()), "share_q0": float((Q == 0).mean())}
        print(f"[posthoc] q90 {k:30s} P(Y<q) {qt[k]['below']:.3f}  P(Y<=q) {qt[k]['at_or_below']:.3f}  "
              f"scaled pinball {qt[k]['spl']:.3f}  mean q {qt[k]['mean_q']:.2f}  share q=0 {qt[k]['share_q0']:.3f}")
    OUT["quantiles"] = qt
    diff = float((q["Negative binomial around SBA"] != q["Poisson around SBA"]).mean())
    OUT["nb"] = {"variance_floored_share": float(floored.mean()), "q_differs_from_poisson_share": diff,
                 "ratio_of_means": float(spl["Negative binomial around SBA"].mean() / spl["Poisson around SBA"].mean() - 1)}
    print(f"[posthoc] negative binomial: variance floored at the mean for {100 * floored.mean():.1f}% of series; "
          f"its quantile differs from the Poisson one on {100 * diff:.1f}% of series-months; ratio of mean losses "
          f"{100 * OUT['nb']['ratio_of_means']:+.2f}%")
    emp = spl["Empirical, last 24 months"]
    for k in q:
        if k != "Empirical, last 24 months":
            m_, lo, hi = boot_ci(emp - spl[k], rng)
            OUT.setdefault("empirical_vs", {})[k] = [100 * m_ / spl[k].mean(), 100 * lo / spl[k].mean(), 100 * hi / spl[k].mean()]
            print(f"[posthoc] empirical vs {k:30s} {100 * m_ / spl[k].mean():+.1f}% [{100 * lo / spl[k].mean():+.1f}; {100 * hi / spl[k].mean():+.1f}]")

    out = Path("results_post_hoc.json")
    out.write_text(json.dumps(OUT, indent=1), encoding="utf-8")
    print("[posthoc] written", out)
    return 0


if __name__ == "__main__":
    if not DATA.exists():
        sys.exit(__doc__)
    first_reading()
    sys.exit(post_hoc())
