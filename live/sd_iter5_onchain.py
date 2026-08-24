"""Signal-diversity loop — iteration 5 (S2): does ON-CHAIN activity add anything price/volume cannot?

This is the only direction left with genuinely orthogonal information: active-address and transaction counts
are not derivable from price or volume. Every other lead this cycle failed because it turned out to be the
same one factor the model already prices — on-chain data cannot fail that way by construction.

Coverage is the binding constraint: the free CoinMetrics community feed covers 27 of our 176 base assets with
2023+ history. They are majors, i.e. the deployable universe, but the cross-section is thin (~15-27 names per
bar) so the BASELINE MUST BE RECOMPUTED ON THE SAME RESTRICTED UNIVERSE — comparing against the full-universe
+0.030/+0.021 would be meaningless.

Features (PIT: the daily metric for date D is only applied to 4h bars from D+1 onward):
  adr_growth_7d   log change in the 7d-mean active-address count vs 7d earlier   (network growth)
  adr_z_30d       (AdrActCnt - 30d mean) / 30d std                               (abnormal on-chain activity)
  adr_per_dvol    log(7d-mean AdrActCnt) - log(7d-mean dollar volume)            (chain activity per $ traded —
                  the scale-free cousin of the literature's new-address-to-price ratio)
  tx_growth_7d    same as adr_growth_7d for transaction count

Gates: G1 standalone same-sign IC, day-clustered CI excludes 0 in BOTH eras. G2 incremental Δ rank-IC vs the
RESTRICTED-universe V0_LEAN baseline, CI>0 in BOTH eras. G3 survives residualization on vol rank.
Falsifier: G1 or G2 fails -> S2 is a null on free data and the loop's agenda is exhausted.
Run: python3 -u -m live.sd_iter5_onchain
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.cost_loop_harness import CACHE, ERAS, CUTS, REPO, build_panel, pit_adv
from live.v0_feature_ablation import V0, gen
from live.build_alpha_beta_decomp import x6
from live.sd_onchain_fetch import base_of

RNG = np.random.default_rng(37)
ONC = ["adr_growth_7d", "adr_z_30d", "adr_per_dvol", "tx_growth_7d"]


def build():
    D = pd.read_parquet(CACHE / "onchain_daily.parquet")
    D["date"] = pd.to_datetime(D["date"], utc=True)
    D = D.sort_values(["asset", "date"])
    g = D.groupby("asset")
    adr7 = g["AdrActCnt"].transform(lambda s: s.rolling(7, min_periods=4).mean())
    tx7 = g["TxCnt"].transform(lambda s: s.rolling(7, min_periods=4).mean())
    D["adr7"] = adr7
    D["adr_growth_7d"] = np.log(adr7) - np.log(g["AdrActCnt"].transform(
        lambda s: s.rolling(7, min_periods=4).mean()).groupby(D["asset"]).shift(7))
    D["tx_growth_7d"] = np.log(tx7) - np.log(tx7.groupby(D["asset"]).shift(7))
    m30 = g["AdrActCnt"].transform(lambda s: s.rolling(30, min_periods=15).mean())
    s30 = g["AdrActCnt"].transform(lambda s: s.rolling(30, min_periods=15).std())
    D["adr_z_30d"] = (D["AdrActCnt"] - m30) / s30.replace(0, np.nan)
    D["date"] = D["date"] + pd.Timedelta(days=1)          # PIT: metric for D is usable from D+1
    D = D.replace([np.inf, -np.inf], np.nan)

    PAN = build_panel()
    PAN["base"] = [base_of(s) for s in PAN["symbol"]]
    keep = set(D["asset"].unique())
    PAN = PAN[PAN["base"].isin(keep)].copy()
    PAN["date"] = PAN["open_time"].dt.floor("1D")
    PAN = PAN.merge(D[["asset", "date", "adr7", "adr_growth_7d", "tx_growth_7d", "adr_z_30d"]],
                    left_on=["base", "date"], right_on=["asset", "date"], how="left")
    A = pit_adv()
    PAN = PAN.merge(A, on=["symbol", "date"], how="left")
    PAN["adr_per_dvol"] = np.log(PAN["adr7"]) - np.log(PAN["tadv"].replace(0, np.nan))
    PAN = PAN.replace([np.inf, -np.inf], np.nan)
    x6.HEAVY_TAIL = set(x6.HEAVY_TAIL) | set(ONC)
    return PAN.sort_values(["symbol", "open_time"]).reset_index(drop=True)


def perbar_ic(P, col="pred", tgt="alpha_A", nmin=8):
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


def xs_resid(d, y, x):
    def f(g):
        xv = g[x].to_numpy(); yv = g[y].to_numpy()
        if len(g) < 8 or np.std(xv) == 0:
            return pd.Series(np.nan, index=g.index)
        b = np.polyfit(xv, yv, 1)[0]
        r = yv - b * xv
        return pd.Series(r - r.mean(), index=g.index)
    return d.groupby("open_time", group_keys=False).apply(f)


def cached(PAN, feats, cuts, tag, era):
    fp = CACHE / f"sd5_{tag}_{era}.parquet"
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
    print(f"restricted universe: {PAN.symbol.nunique()} symbols "
          f"({sorted(PAN.symbol.unique())[:8]}...)", flush=True)
    print(f"  bars {PAN.open_time.nunique()}, median names/bar "
          f"{PAN.groupby('open_time').size().median():.0f}", flush=True)
    for c in ONC:
        print(f"  {c:<16}coverage {PAN[c].notna().mean()*100:.1f}%", flush=True)

    print("\n============ G1/G3 — standalone on-chain IC, and vs the vol factor ============", flush=True)
    for era in ERAS:
        c0, c1 = CUTS[era][0], CUTS[era][-1]
        d = PAN[(PAN.open_time >= c0) & (PAN.open_time < c1)].rename(
            columns={"alpha_vs_btc_realized": "alpha_A"}).dropna(subset=["alpha_A", "rvol_7d"])
        d["vol_rank"] = d.groupby("open_time")["rvol_7d"].rank(pct=True)
        print(f"\n----- {era} -----", flush=True)
        print(f"  {'feature':<16}{'own IC':<26}{'IC | vol-resid':<26}{'corr w/ vol':<12}", flush=True)
        for c in ONC:
            s = d.dropna(subset=[c]).copy()
            if len(s) < 3000:
                print(f"  {c:<16}(insufficient rows: {len(s)})", flush=True); continue
            s["r"] = s.groupby("open_time")[c].rank(pct=True)
            ic = perbar_ic(s, "r"); lo, hi = day_ci(ic)
            s["res"] = xs_resid(s, "r", "vol_rank")
            ic2 = perbar_ic(s.dropna(subset=["res"]), "res"); lo2, hi2 = day_ci(ic2)
            rho = s.groupby("open_time").apply(
                lambda g: spearmanr(g["r"], g["vol_rank"]).correlation).dropna().mean()
            t1 = "SIG" if (lo > 0 or hi < 0) else "spans0"
            t2 = "SIG" if (lo2 > 0 or hi2 < 0) else "spans0"
            print(f"  {c:<16}{f'{ic.mean():+.4f}[{lo:+.4f},{hi:+.4f}] {t1}':<26}"
                  f"{f'{ic2.mean():+.4f}[{lo2:+.4f},{hi2:+.4f}] {t2}':<26}{rho:<+12.3f}", flush=True)

    print("\n============ G2 — incremental vs the RESTRICTED-universe V0_LEAN baseline ============",
          flush=True)
    res = {}
    for era in ERAS:
        cuts = CUTS[era]
        base = perbar_ic(cached(PAN, list(V0), cuts, "base", era))
        res[("base", era)] = base
        print(f"\n----- {era} -----  restricted-universe baseline {base.mean():+.4f} "
              f"({len(base)} bars)", flush=True)
        for c in ONC:
            P = cached(PAN, list(V0) + [c], cuts, c, era)
            if P.empty:
                print(f"  +{c:<16}(failed)", flush=True); continue
            ic = perbar_ic(P); res[(c, era)] = ic
            dd, lo, hi = day_ci_diff(base, ic)
            tag = "ADDS" if lo > 0 else ("hurts" if hi < 0 else "within noise")
            print(f"  +{c:<16}IC {ic.mean():+.4f}  Δ {dd:+.4f} [{lo:+.4f},{hi:+.4f}]  {tag}", flush=True)
        P = cached(PAN, list(V0) + ONC, cuts, "allonc", era)
        if not P.empty:
            ic = perbar_ic(P); res[("allonc", era)] = ic
            dd, lo, hi = day_ci_diff(base, ic)
            tag = "ADDS" if lo > 0 else ("hurts" if hi < 0 else "within noise")
            print(f"  +{'ALL 4 on-chain':<16}IC {ic.mean():+.4f}  Δ {dd:+.4f} [{lo:+.4f},{hi:+.4f}]  {tag}",
                  flush=True)

    print("\n============ GATE READ ============", flush=True)
    win = []
    for c in ONC + ["allonc"]:
        ok = all((c, e) in res and day_ci_diff(res[("base", e)], res[(c, e)])[1] > 0 for e in ERAS)
        print(f"  {c:<18}{'PASS both eras' if ok else 'fail'}", flush=True)
        if ok:
            win.append(c)
    print(f"\n  survivors: {win if win else 'NONE -> S2 null on free on-chain data'}", flush=True)
    print("\nSDITER5DONE", flush=True)


if __name__ == "__main__":
    main()
