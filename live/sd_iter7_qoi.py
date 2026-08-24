"""Signal-diversity loop — iteration 7: clock-phase order imbalance (the workflow's only survivor).

Source: Kim & Hansen, arXiv 2607.09426 (2026) — order imbalance in the first seconds of each quarter-hour
boundary predicts subsequent returns, attributed to scheduled/algorithmic execution clustering at the
boundary. Adversarially screened GENUINELY_NEW: every flow feature in this repo averages over that phase
(fl_tfi/fl_vpin are 5-min resamples, OB-flow 5-min, bookDepth 30s), so a statistic confined to seconds 0-10
of minutes {0,15,30,45} is destroyed by all of them. Screener's prior: 0.15 for the IC gate, 0.05 for a book
improvement.

RUN ORDER IS DELIBERATE — the cheapest, most likely killer runs FIRST.

  A1  TS -> XS TRANSLATION (the likely death). The paper is pure TIME SERIES: contract i's own boundary flow
      predicts contract i's own return, six contracts, no cross-section. Our target is the CROSS-SECTIONAL z
      of the BTC-beta residual, which demeans the panel and strips beta. If boundary flow is largely
      market-wide, the common component is exactly what our target removes. So:
        (a) regress raw 4h return on qOI  -> must replicate the paper's sign/magnitude, else our
            construction is wrong and nothing downstream means anything
        (b) regress the cross-sectionally demeaned BTC-beta residual on qOI -> if this dies, STOP.
  A2  instrument mismatch: all comparisons use an incumbent run on the SAME 31 symbols, never the 176-name one.
  G1  standalone rank-IC vs the residual target, same sign both eras, day-clustered CI excluding 0.
      Look-ahead sanity: IC against the +1-bar-shifted return must not exceed +0.10 (CLAUDE.md rule).
  G2  incremental over V0_LEAN through the real per-symbol RidgeCV, paired delta, CI on the DELTA, both eras.

G3 (hard split, all free parameters frozen on 2023-06..2024-12) and G4 (book level) only run if G2 passes.
Falsifier: A1(b) dies, or G1 sign-flips across eras, or G2 spans 0 in either era -> close the
flow-as-alpha family permanently.
Run: python3 -u -m live.sd_iter7_qoi
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.cost_loop_harness import CACHE, ERAS, CUTS, build_panel
from live.v0_feature_ablation import V0, gen
from live.build_alpha_beta_decomp import x6, FULL

RNG = np.random.default_rng(71)
QOI = CACHE / "qoi_windows.parquet"
WCOL = "qoi_10s"                 # frozen a priori as the paper's primary window; 30s/60s only as robustness
NLAG = 12                        # same-phase lags for the persistence feature


def build():
    Q = pd.read_parquet(QOI)
    Q["wtime"] = pd.to_datetime(Q["wtime"], utc=True)
    Q = Q.sort_values(["symbol", "wtime"])
    # the 4h bar opening at t may use only windows STRICTLY BEFORE t
    Q["bar"] = Q["wtime"].dt.ceil("4h")
    Q.loc[Q["wtime"] == Q["bar"], "bar"] = Q["wtime"] + pd.Timedelta(hours=4)
    g = Q.groupby("symbol")[WCOL]
    Q["qoi_pers"] = g.transform(lambda s: s.shift(1).rolling(NLAG, min_periods=6).mean())
    last = Q.groupby(["symbol", "bar"]).tail(1)[["symbol", "bar", WCOL, "qoi_pers"]].rename(
        columns={"bar": "open_time", WCOL: "qoi_last"})
    mean4 = Q.groupby(["symbol", "bar"])[WCOL].mean().rename("qoi_mean4h").reset_index().rename(
        columns={"bar": "open_time"})
    F = last.merge(mean4, on=["symbol", "open_time"], how="outer")

    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    # keep BOTH names: v0_feature_ablation.gen() reads `alpha_vs_btc_realized` internally and its bare
    # `except Exception: pass` would silently swallow a KeyError and return an empty frame.
    PAN["alpha_A"] = PAN["alpha_vs_btc_realized"]
    PAN = PAN.merge(RP, on=["symbol", "open_time"], how="left")
    PAN = PAN[PAN["symbol"].isin(set(F["symbol"].unique()))].copy()
    PAN = PAN.merge(F, on=["symbol", "open_time"], how="left")
    x6.HEAVY_TAIL = set(x6.HEAVY_TAIL) | {"qoi_last", "qoi_pers", "qoi_mean4h"}
    return PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)


def perbar_ic(P, col, tgt, nmin=8):
    return P.groupby("open_time").apply(
        lambda g: spearmanr(g[col], g[tgt]).correlation if len(g) >= nmin else np.nan).dropna()


def day_ci(s, nb=3000):
    gg = [x.to_numpy() for _, x in s.groupby(pd.to_datetime(s.index, utc=True).floor("1D"))]
    b = [np.concatenate([gg[k] for k in RNG.integers(0, len(gg), len(gg))]).mean() for _ in range(nb)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def day_ci_diff(a, b, nb=3000):
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    j["d"] = j["b"] - j["a"]
    gg = [x["d"].to_numpy() for _, x in j.groupby(pd.to_datetime(j.index, utc=True).floor("1D"))]
    boot = [np.concatenate([gg[k] for k in RNG.integers(0, len(gg), len(gg))]).mean() for _ in range(nb)]
    return float(j["d"].mean()), *np.percentile(boot, [2.5, 97.5])


def clustered_ols(y, x, times):
    """Univariate OLS slope in bps with day-clustered SE."""
    m = np.isfinite(y) & np.isfinite(x)
    y, x, t = y[m], x[m], pd.DatetimeIndex(times[m]).floor("1D")
    x = x - x.mean()
    b = float((x @ y) / (x @ x))
    r = y - b * x
    df = pd.DataFrame({"xr": x * r, "d": t})
    s = df.groupby("d")["xr"].sum().to_numpy()
    var = (s ** 2).sum() / ((x @ x) ** 2)
    se = float(np.sqrt(var))
    return b * 1e4, se * 1e4, b / se if se > 0 else np.nan


def cached(PAN, feats, cuts, tag, era):
    fp = CACHE / f"sd7_{tag}_{era}.parquet"
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
    print(f"universe {PAN.symbol.nunique()} symbols, {PAN.open_time.nunique()} bars", flush=True)
    for c in ("qoi_last", "qoi_pers", "qoi_mean4h"):
        print(f"  {c:<12} coverage {PAN[c].notna().mean()*100:.1f}%  "
              f"mean {PAN[c].mean():+.4f}  sd {PAN[c].std():.4f}", flush=True)

    print("\n================ A1 — does the effect survive our TARGET? ================", flush=True)
    print("  (a) raw 4h return ~ qOI  [must replicate the paper]", flush=True)
    print("  (b) xs-demeaned BTC-beta residual ~ qOI  [if this dies, STOP]\n", flush=True)
    for era in ERAS:
        c0, c1 = CUTS[era][0], CUTS[era][-1]
        d = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)].dropna(
            subset=["qoi_last", "return_pct", "alpha_A"]).copy()
        d["res_dm"] = d["alpha_A"] - d.groupby("open_time")["alpha_A"].transform("mean")
        d["raw_dm"] = d["return_pct"] - d.groupby("open_time")["return_pct"].transform("mean")
        t = d["open_time"].to_numpy()
        rows = [("raw return", d["return_pct"].to_numpy()),
                ("raw, xs-demeaned", d["raw_dm"].to_numpy()),
                ("BTC-beta residual", d["alpha_A"].to_numpy()),
                ("residual, xs-demeaned (OUR TARGET)", d["res_dm"].to_numpy())]
        print(f"----- {era} ({len(d):,} obs) -----", flush=True)
        for lbl, y in rows:
            b, se, tstat = clustered_ols(y, d["qoi_last"].to_numpy(), t)
            sig = "SIG" if abs(tstat) > 1.96 else "ns"
            print(f"  {lbl:<38} slope {b:+8.2f} bps  (se {se:5.2f}, t {tstat:+5.2f}) {sig}", flush=True)
        print("", flush=True)

    print("================ G1 — standalone cross-sectional rank-IC ================", flush=True)
    for era in ERAS:
        c0, c1 = CUTS[era][0], CUTS[era][-1]
        d = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)].copy()
        d["fwd1"] = d.groupby("symbol")["alpha_A"].shift(-1)
        print(f"----- {era} -----", flush=True)
        for c in ("qoi_last", "qoi_pers", "qoi_mean4h"):
            s = d.dropna(subset=[c, "alpha_A"]).copy()
            if len(s) < 3000:
                print(f"  {c:<12}(insufficient)", flush=True); continue
            s["r"] = s.groupby("open_time")[c].rank(pct=True)
            ic = perbar_ic(s, "r", "alpha_A"); lo, hi = day_ci(ic)
            la = perbar_ic(s.dropna(subset=["fwd1"]), "r", "fwd1").mean()
            tag = "SIG" if (lo > 0 or hi < 0) else "spans0"
            flag = "  LOOK-AHEAD?" if abs(la) > 0.10 else ""
            print(f"  {c:<12} IC {ic.mean():+.4f} [{lo:+.4f},{hi:+.4f}] {tag}   "
                  f"(IC vs +1bar {la:+.4f}){flag}", flush=True)
        print("", flush=True)

    print("================ G2 — incremental over V0_LEAN (same 31 symbols) ================", flush=True)
    res = {}
    for era in ERAS:
        cuts = CUTS[era]
        base = perbar_ic(cached(PAN, list(V0), cuts, "base", era), "pred", "alpha_A")
        res[("base", era)] = base
        print(f"\n----- {era} -----  matched-universe baseline {base.mean():+.4f} ({len(base)} bars)",
              flush=True)
        for c in ("qoi_last", "qoi_pers", "qoi_mean4h"):
            P = cached(PAN, list(V0) + [c], cuts, c, era)
            if P.empty:
                continue
            ic = perbar_ic(P, "pred", "alpha_A"); res[(c, era)] = ic
            dd, lo, hi = day_ci_diff(base, ic)
            tag = "ADDS" if lo > 0 else ("hurts" if hi < 0 else "within noise")
            print(f"  +{c:<12} IC {ic.mean():+.4f}  Δ {dd:+.4f} [{lo:+.4f},{hi:+.4f}]  {tag}", flush=True)
        P = cached(PAN, list(V0) + ["qoi_last", "qoi_pers"], cuts, "qoi_both", era)
        if not P.empty:
            ic = perbar_ic(P, "pred", "alpha_A"); res[("qoi_both", era)] = ic
            dd, lo, hi = day_ci_diff(base, ic)
            tag = "ADDS" if lo > 0 else ("hurts" if hi < 0 else "within noise")
            print(f"  +{'both':<12} IC {ic.mean():+.4f}  Δ {dd:+.4f} [{lo:+.4f},{hi:+.4f}]  {tag}",
                  flush=True)

    print("\n================ GATE READ ================", flush=True)
    win = []
    for c in ("qoi_last", "qoi_pers", "qoi_mean4h", "qoi_both"):
        ok = all((c, e) in res and day_ci_diff(res[("base", e)], res[(c, e)])[1] > 0 for e in ERAS)
        print(f"  {c:<14}{'PASS both eras' if ok else 'fail'}", flush=True)
        if ok:
            win.append(c)
    print(f"\n  G2 survivors: {win if win else 'NONE -> close the flow-as-alpha family'}", flush=True)
    print("\nSDITER7DONE", flush=True)


if __name__ == "__main__":
    main()
