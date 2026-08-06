"""Can the leftover beta be FARMED as a market-TIMING signal? User's idea: many high-vol names pumping
=> bull regime => go long beta.

Test: does a trailing high-vol-breadth signal predict the FORWARD MARKET return (mean raw forward return
across names)? Directional/time-series, both eras, block-bootstrap CI (overlap-aware — few independent
windows, so honest CIs matter). Sign: + = bull/momentum (user's hypothesis); − = contrarian froth-top.
Run: python3 -u -m live.build_beta_timing
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FULL = "outputs/vBTC_features/panel_expanded_v0_clean.parquet"
CUT = pd.Timestamp("2025-10-01", tz="UTC")


def block_corr_ci(a, b, block=12, nboot=3000, seed=3):
    a = np.asarray(a, float); b = np.asarray(b, float)
    n = len(a)
    if n < block * 3:
        return (np.nan, np.nan, np.nan)
    base = pd.Series(a).corr(pd.Series(b))
    rng = np.random.default_rng(seed); nb = int(np.ceil(n / block)); hi = n - block
    out = np.empty(nboot)
    for i in range(nboot):
        idx = (rng.integers(0, hi + 1, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        out[i] = pd.Series(a[idx]).corr(pd.Series(b[idx]))
    lo, up = np.nanpercentile(out, [2.5, 97.5])
    return float(base), float(lo), float(up)


def main():
    P = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct", "ret_3d", "idio_vol_to_btc_1d"])
    P["open_time"] = pd.to_datetime(P["open_time"], utc=True)
    P = P[(P["open_time"].dt.hour % 4 == 0) & (P["open_time"].dt.minute == 0)]
    P = P.dropna(subset=["return_pct", "ret_3d", "idio_vol_to_btc_1d"])
    r = P.groupby("open_time")["idio_vol_to_btc_1d"].rank(pct=True)
    P["volq"] = np.clip((r * 5).astype(int), 0, 4)

    # per-bar signals (trailing, PIT) + target (forward market return)
    g = P.groupby("open_time")
    bar = pd.DataFrame({
        "mkt_fwd": g["return_pct"].mean(),                                    # market forward return (target)
        "hv_ret": P[P.volq == 4].groupby("open_time")["ret_3d"].mean(),        # high-vol recent momentum
        "hv_breadth": P[P.volq == 4].assign(u=P["ret_3d"] > 0).groupby("open_time")["u"].mean(),
        "hvlv": (P[P.volq == 4].groupby("open_time")["ret_3d"].mean()
                 - P[P.volq == 0].groupby("open_time")["ret_3d"].mean()),     # speculative-minus-safe leading
    }).dropna()
    bar["era"] = np.where(bar.index < CUT, "OOS", "REC")
    print(f"bars {len(bar)} | testing: trailing high-vol-breadth -> forward MARKET return "
          f"(+ = bull/momentum, − = contrarian)\n", flush=True)

    for sig in ("hv_ret", "hv_breadth", "hvlv"):
        print(f"  signal = {sig}", flush=True)
        for era in ("OOS", "REC"):
            d = bar[bar.era == era]
            c, lo, hi = block_corr_ci(d[sig].to_numpy(), d["mkt_fwd"].to_numpy())
            tag = "off-0" if (lo > 0 or hi < 0) else "spans 0"
            print(f"    {era}: corr(signal, mkt_fwd) {c:+.3f} [{lo:+.3f},{hi:+.3f}] {tag}", flush=True)
        print("", flush=True)

    # regime framing: does high-vol breadth being HIGH predict a positive forward market? (bull call)
    print("=== bull-call framing: forward market return when high-vol breadth is HIGH vs LOW "
          "(era-locked median split) ===", flush=True)
    med = bar.loc[bar.era == "OOS", "hv_breadth"].median()
    for era in ("OOS", "REC"):
        d = bar[bar.era == era]
        hi_m = d.loc[d.hv_breadth >= med, "mkt_fwd"].mean() * 1e4
        lo_m = d.loc[d.hv_breadth < med, "mkt_fwd"].mean() * 1e4
        print(f"    {era}: mkt_fwd | high-vol-breadth HIGH {hi_m:+.1f}bps vs LOW {lo_m:+.1f}bps "
              f"→ diff {hi_m-lo_m:+.1f}bps", flush=True)
    print("\nTIMINGDONE", flush=True)


if __name__ == "__main__":
    main()
