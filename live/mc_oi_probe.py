"""Does OI relate to market cap, and is ADV an adequate MC proxy for this repo's purposes?

Market cap is NOT in owned data (Binance Vision has no supply/float) and CoinGecko's free tier caps historical
market_chart at 365 days, so a PIT market-cap PANEL back through the OOS era (2023-06+) is not obtainable
without a paid plan. What IS obtainable free is a CURRENT cross-sectional market-cap snapshot — enough to
measure how well the size proxies we DO own (trailing ADV, OI value) stand in for market cap.

Snapshot MC is used here ONLY to validate the proxy relationship. It is never used as a backtest signal:
today's market caps are the survivors' market caps, so ranking history by them would be look-ahead.

Outputs: cross-sectional log-log correlations between MC, ADV and OI value; and the OI/MC ratio distribution
(the crowding/leverage metric the question is really about).
Run: python3 -u -m live.mc_oi_probe
"""
from __future__ import annotations

import glob
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path("/home/yuqing/ctaNew")
CACHE = REPO / "live/state/cost_loop"
CACHE.mkdir(parents=True, exist_ok=True)
MCFILE = CACHE / "cg_mcap_snapshot.json"
MULT = re.compile(r"^(\d+)(.+)$")


def base_of(sym: str) -> tuple[str, float]:
    """BTCUSDT -> (BTC, 1); 1000PEPEUSDT -> (PEPE, 1000). Returns (base ticker, contract multiplier)."""
    s = sym.replace("USDT", "").replace("USDC", "")
    m = MULT.match(s)
    if m and m.group(1) in ("1000", "10000", "1000000", "100"):
        return m.group(2), float(m.group(1))
    return s, 1.0


def fetch_mcap() -> dict:
    if MCFILE.exists():
        return json.loads(MCFILE.read_text())
    out = {}
    for page in (1, 2, 3):
        r = requests.get("https://api.coingecko.com/api/v3/coins/markets",
                         params=dict(vs_currency="usd", order="market_cap_desc", per_page=250,
                                     page=page, sparkline="false"), timeout=45)
        if r.status_code != 200:
            print(f"  page {page}: HTTP {r.status_code} — stopping", flush=True)
            break
        for c in r.json():
            t = (c.get("symbol") or "").upper()
            mc = c.get("market_cap")
            if t and mc and t not in out:          # first hit = highest mcap for that ticker
                out[t] = dict(mcap=float(mc), id=c.get("id"), price=c.get("current_price"),
                              vol24=c.get("total_volume"))
        print(f"  page {page}: {len(out)} tickers cumulative", flush=True)
        time.sleep(3)
    MCFILE.write_text(json.dumps(out))
    return out


def owned_size() -> pd.DataFrame:
    """Trailing-30d ADV (USD/day) and mean OI value (USD) per symbol over the last 90 days of owned data."""
    rows = []
    for f in glob.glob(str(REPO / "data/ml/cache/metrics_*.parquet")):
        sym = Path(f).stem.replace("metrics_", "")
        try:
            m = pd.read_parquet(f, columns=["sum_open_interest_value", "sum_open_interest"])
            m = m[m.index >= m.index.max() - pd.Timedelta(days=90)]
            if m.empty:
                continue
            rows.append(dict(symbol=sym, oi_usd=float(m["sum_open_interest_value"].mean()),
                             oi_base=float(m["sum_open_interest"].mean()),
                             oi_last=m.index.max()))
        except Exception:
            pass
    OI = pd.DataFrame(rows)
    advs = []
    for f in glob.glob(str(REPO / "data/ml/cache/flow_*.parquet")):
        sym = Path(f).stem.replace("flow_", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            if not isinstance(d.index, pd.DatetimeIndex):
                continue
            dv = (d["total_volume"] * d["vwap"]).sort_index()
            dv = dv[dv.index >= dv.index.max() - pd.Timedelta(days=90)]
            advs.append(dict(symbol=sym, adv=float(dv.resample("1D").sum().mean())))
        except Exception:
            pass
    return OI.merge(pd.DataFrame(advs), on="symbol", how="outer")


def main():
    print("fetching CoinGecko market-cap snapshot (validation only, never a backtest signal)...", flush=True)
    mc = fetch_mcap()
    S = owned_size()
    S["base"] = [base_of(s)[0] for s in S["symbol"]]
    S["mcap"] = S["base"].map(lambda b: mc.get(b, {}).get("mcap", np.nan))
    n_map = S["mcap"].notna().sum()
    print(f"\nuniverse {len(S)} perp symbols | market-cap mapped for {n_map} "
          f"({100*n_map/len(S):.0f}%) | unmapped = delisted / renamed / ticker collision", flush=True)
    print(f"  OI data through {S['oi_last'].max()}", flush=True)

    D = S.dropna(subset=["mcap", "adv", "oi_usd"]).copy()
    D = D[(D.adv > 0) & (D.oi_usd > 0) & (D.mcap > 0)]
    for c in ("mcap", "adv", "oi_usd"):
        D["l_" + c] = np.log(D[c])
    print(f"\n=== cross-sectional log-log correlation (n={len(D)}) ===", flush=True)
    C = D[["l_mcap", "l_adv", "l_oi_usd"]].corr(method="pearson")
    Sp = D[["l_mcap", "l_adv", "l_oi_usd"]].corr(method="spearman")
    print("  pearson (log):\n" + C.round(3).to_string(), flush=True)
    print("  spearman (rank):\n" + Sp.round(3).to_string(), flush=True)

    D["oi_mc"] = D["oi_usd"] / D["mcap"]
    D["oi_adv"] = D["oi_usd"] / D["adv"]
    D["adv_mc"] = D["adv"] / D["mcap"]
    print(f"\n=== OI / MC  (the leverage-vs-size ratio) ===", flush=True)
    q = D["oi_mc"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print("  quantiles: " + "  ".join(f"p{int(k*100)} {v*100:.2f}%" for k, v in q.items()), flush=True)
    print(f"  corr(log OI/MC, log MC) = {np.corrcoef(np.log(D['oi_mc']), D['l_mcap'])[0,1]:+.3f}"
          "   (negative => small caps carry proportionally MORE futures OI)", flush=True)
    print(f"  corr(log OI/ADV, log MC) = {np.corrcoef(np.log(D['oi_adv']), D['l_mcap'])[0,1]:+.3f}", flush=True)
    print(f"  corr(log OI/MC, log OI/ADV) = "
          f"{np.corrcoef(np.log(D['oi_mc']), np.log(D['oi_adv']))[0,1]:+.3f}"
          "   (how well the owned ratio stands in for the MC one)", flush=True)

    print(f"\n=== top / bottom 10 by OI/MC ===", flush=True)
    D["mc_bn"] = D["mcap"] / 1e9
    cols = ["symbol", "mc_bn", "oi_usd", "oi_mc", "oi_adv"]
    top = D.nlargest(10, "oi_mc")[cols].copy(); bot = D.nsmallest(10, "oi_mc")[cols].copy()
    for t, nm in ((top, "HIGHEST OI/MC (most levered vs size)"), (bot, "LOWEST OI/MC")):
        t["oi_usd"] = (t["oi_usd"] / 1e6).round(1); t["oi_mc"] = (t["oi_mc"] * 100).round(2)
        t["mc_bn"] = t["mc_bn"].round(2); t["oi_adv"] = t["oi_adv"].round(2)
        print(f"  {nm}:\n" + t.rename(columns={"mc_bn": "mcap$bn", "oi_usd": "OI$m", "oi_mc": "OI/MC%",
                                               "oi_adv": "OI/ADV"}).to_string(index=False), flush=True)

    D.to_csv(CACHE / "mc_oi_snapshot.csv", index=False)
    print("\nMCOIDONE", flush=True)


if __name__ == "__main__":
    main()
