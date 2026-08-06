"""DEEP: is the low-vol edge a BETA-RESIDUALIZATION ARTIFACT?
If the "BTC-residual" target isn't fully beta-clean, high-vol (=high-beta) names keep leftover market
beta in their residual → the vol effect flips sign with market direction (would explain the era-
inconsistency: OOS high-vol-mean wins, REC loses).

Test: reconstruct market forward direction = cross-sectional mean of the RAW forward return (return_pct).
Then:
  1. Does the high−low vol quintile forward-RESIDUAL spread co-move with the market's forward return?
     corr>0 → leftover beta (high-vol residual inflated in up-markets).
  2. Residual "beta" per vol quintile: regress residual on market_fwd within quintile; is it larger for
     high-vol? (clean residual → ~0 for all).
  3. Era market direction (was OOS an up-market vs REC?).
Run: python3 -u -m live.build_why_beta
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FULL = "outputs/vBTC_features/panel_expanded_v0_clean.parquet"
CUT = pd.Timestamp("2025-10-01", tz="UTC")


def main():
    P = pd.read_parquet(FULL, columns=["symbol", "open_time", "alpha_vs_btc_realized",
                                        "return_pct", "return_1d", "idio_vol_to_btc_1d"])
    P["open_time"] = pd.to_datetime(P["open_time"], utc=True)
    P = P[(P["open_time"].dt.hour % 4 == 0) & (P["open_time"].dt.minute == 0)]
    P = P.dropna(subset=["alpha_vs_btc_realized", "return_pct", "idio_vol_to_btc_1d"])

    # verify return_pct is the FORWARD raw return (alpha is its BTC-residual)
    s = P.sample(min(300000, len(P)), random_state=0)
    c_fwd = s["return_pct"].corr(s["alpha_vs_btc_realized"])
    c_trail = s["return_pct"].corr(s["return_1d"])
    print(f"return_pct vs alpha_vs_btc_realized (forward residual): corr {c_fwd:+.2f}  "
          f"| vs return_1d (trailing): corr {c_trail:+.2f}", flush=True)
    print("  (high corr with forward-residual + low with trailing => return_pct is the RAW forward return)\n",
          flush=True)

    P["mkt_fwd"] = P.groupby("open_time")["return_pct"].transform("mean")  # market forward direction
    r = P.groupby("open_time")["idio_vol_to_btc_1d"].rank(pct=True)
    P["volq"] = np.clip((r * 5).astype(int), 0, 4)
    P["era"] = np.where(P["open_time"] < CUT, "OOS", "REC")

    print("=== (1) does high−low vol RESIDUAL spread co-move with market forward return? "
          "(corr>0 = leftover beta) ===", flush=True)
    for era in ("OOS", "REC"):
        g = P[P.era == era]
        per_bar = g.groupby("open_time").apply(
            lambda x: pd.Series({
                "spread": x.loc[x.volq == 4, "alpha_vs_btc_realized"].mean()
                          - x.loc[x.volq == 0, "alpha_vs_btc_realized"].mean(),
                "mkt": x["mkt_fwd"].iloc[0]}), include_groups=False).dropna()
        cc = per_bar["spread"].corr(per_bar["mkt"])
        print(f"  {era}: corr(hi−lo vol residual spread, market fwd) = {cc:+.2f}  "
              f"| mean market fwd {per_bar['mkt'].mean()*1e4:+.1f}bps (era direction)", flush=True)

    print("\n=== (2) leftover residual-beta by vol quintile "
          "(slope of residual on market_fwd; clean => ~0, larger for hi-vol => leftover beta) ===", flush=True)
    for era in ("OOS", "REC"):
        g = P[P.era == era]
        print(f"  [{era}]", flush=True)
        for q in (0, 2, 4):
            d = g[g.volq == q]
            x = d["mkt_fwd"].to_numpy(); y = d["alpha_vs_btc_realized"].to_numpy()
            beta = np.polyfit(x, y, 1)[0] if len(d) > 100 and x.std() > 0 else np.nan
            lab = "Q0 lowvol" if q == 0 else ("Q4 hivol" if q == 4 else "Q2 mid")
            print(f"    {lab:<10} residual-beta to market = {beta:+.2f}", flush=True)
    print("\nBETADONE", flush=True)


if __name__ == "__main__":
    main()
