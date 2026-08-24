"""Signal-diversity loop — iteration 4 (S4): does the RESIDUAL PATH carry structure our contemporaneous
features miss?

Our model sees 14 contemporaneous price/vol transforms and does no sequence modelling. SOTA stat arb applies
a convolutional/attention model to the TIME SERIES of residuals — the claim being that the residual path
(how a name's alpha has evolved) holds information a snapshot does not. The repo has touched this only at two
points (`resid_rev_2`/`resid_rev_3` = 8h/12h past-residual sums, in the long book only).

This tests the residual TERM STRUCTURE systematically: past-residual sums at 4h/8h/12h/1d/2d/7d/14d, each
added individually and all together, through the real per-symbol Ridge pipeline. If the residual path has
exploitable shape (e.g. short-horizon reversal flipping to longer-horizon continuation), a linear model given
the whole term structure will find it — and if a linear model given every lag finds nothing, a sequence model
on the same information is not the missing piece.

All PIT: pa_k(t) = sum of alpha_A over bars t-k..t-1, i.e. .shift(1).rolling(k).sum() — the label at t covers
t..t+4h, so there is no overlap. Same construction as the validated resid_rev features.

Gate: paired Δ rank-IC vs V0_LEAN, day-clustered CI>0 in BOTH eras. Falsifier: none clears -> S4 null and the
loop's owned-data agenda is exhausted.
Run: python3 -u -m live.sd_iter4_respath
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.cost_loop_harness import CACHE, ERAS, CUTS, build_panel
from live.v0_feature_ablation import V0, gen
from live.build_alpha_beta_decomp import x6

RNG = np.random.default_rng(29)
LAGS = {"pa_4h": 1, "pa_8h": 2, "pa_12h": 3, "pa_1d": 6, "pa_2d": 12, "pa_7d": 42, "pa_14d": 84}


def build():
    PAN = build_panel().sort_values(["symbol", "open_time"])
    a = PAN.groupby("symbol")["alpha_vs_btc_realized"]
    for name, k in LAGS.items():
        PAN[name] = a.transform(lambda s: s.shift(1).rolling(k, min_periods=max(1, k // 2)).sum())
    x6.HEAVY_TAIL = set(x6.HEAVY_TAIL) | set(LAGS)
    return PAN.reset_index(drop=True)


def perbar_ic(P, col="pred", tgt="alpha_A"):
    return P.groupby("open_time").apply(
        lambda g: spearmanr(g[col], g[tgt]).correlation if len(g) >= 10 else np.nan).dropna()


def day_ci_diff(a, b, nb=3000):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    j["d"] = j["b"] - j["a"]
    gg = [x["d"].to_numpy() for _, x in j.groupby(pd.to_datetime(j.index, utc=True).floor("1D"))]
    boot = [np.concatenate([gg[k] for k in RNG.integers(0, len(gg), len(gg))]).mean() for _ in range(nb)]
    return float(j["d"].mean()), *np.percentile(boot, [2.5, 97.5])


def day_ci(s, nb=3000):
    gg = [x.to_numpy() for _, x in s.groupby(pd.to_datetime(s.index, utc=True).floor("1D"))]
    b = [np.concatenate([gg[k] for k in RNG.integers(0, len(gg), len(gg))]).mean() for _ in range(nb)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def cached(PAN, feats, cuts, tag, era):
    fp = CACHE / f"sd4_{tag}_{era}.parquet"
    if fp.exists():
        d = pd.read_parquet(fp); d["open_time"] = pd.to_datetime(d["open_time"], utc=True); return d
    P = gen(PAN, feats, cuts)
    if P.empty:
        return P
    P["open_time"] = pd.to_datetime(P["open_time"], utc=True)
    P.to_parquet(fp, index=False)
    return P


def main():
    PAN = build()
    print("residual-path term structure — standalone rank-IC (sign shows reversal<0 / continuation>0)\n",
          flush=True)
    for era in ERAS:
        c0, c1 = CUTS[era][0], CUTS[era][-1]
        d = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)].rename(
            columns={"alpha_vs_btc_realized": "alpha_A"})
        print(f"----- {era} -----", flush=True)
        for name in LAGS:
            s = d.dropna(subset=[name, "alpha_A"]).copy()
            s["r"] = s.groupby("open_time")[name].rank(pct=True)
            ic = perbar_ic(s, "r"); lo, hi = day_ci(ic)
            print(f"  {name:<8} IC {ic.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {'SIG' if (lo>0 or hi<0) else 'spans0'}",
                  flush=True)
        print("", flush=True)

    print("============ incremental through the real pipeline ============", flush=True)
    res = {}
    for era in ERAS:
        cuts = CUTS[era]
        base = perbar_ic(cached(PAN, list(V0), cuts, "base", era))
        res[("base", era)] = base
        print(f"\n----- {era} -----  baseline V0_LEAN {base.mean():+.4f}", flush=True)
        for name in LAGS:
            P = cached(PAN, list(V0) + [name], cuts, name, era)
            if P.empty:
                continue
            ic = perbar_ic(P); res[(name, era)] = ic
            dd, lo, hi = day_ci_diff(base, ic)
            tag = "ADDS" if lo > 0 else ("hurts" if hi < 0 else "within noise")
            print(f"  +{name:<9} IC {ic.mean():+.4f}  Δ {dd:+.4f} [{lo:+.4f},{hi:+.4f}]  {tag}", flush=True)
        P = cached(PAN, list(V0) + list(LAGS), cuts, "allpath", era)
        if not P.empty:
            ic = perbar_ic(P); res[("allpath", era)] = ic
            dd, lo, hi = day_ci_diff(base, ic)
            tag = "ADDS" if lo > 0 else ("hurts" if hi < 0 else "within noise")
            print(f"  +{'FULL PATH (7)':<9} IC {ic.mean():+.4f}  Δ {dd:+.4f} [{lo:+.4f},{hi:+.4f}]  {tag}",
                  flush=True)

    print("\n============ GATE READ ============", flush=True)
    win = []
    for name in list(LAGS) + ["allpath"]:
        ok = all((name, e) in res and day_ci_diff(res[("base", e)], res[(name, e)])[1] > 0 for e in ERAS)
        print(f"  {name:<12}{'PASS both eras' if ok else 'fail'}", flush=True)
        if ok:
            win.append(name)
    print(f"\n  survivors: {win if win else 'NONE -> S4 null; owned-data agenda exhausted'}", flush=True)
    print("\nSDITER4DONE", flush=True)


if __name__ == "__main__":
    main()
