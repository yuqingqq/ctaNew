"""QNT 2026-09 case study: computes every number quoted in docs/QNT_2026-09_CASE_STUDY.md.

  python3 -m research.qnt_case_study  -> data/case_qnt/summary.json

Venues: Binance USDM (Vision, T+1), Binance spot, Bybit linear (archive, T+1),
Bitget perp + spot, Coinbase, KuCoin, MEXC. OKX QNT-USDT-SWAP is EXCLUDED:
it is an equity perp (instCategory 3, ~$48) that shares the ticker.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from research.fetch_1m import (binance_spot, binance_usdm, binance_usdm_metrics, bitget, bybit_linear,
                               coinbase, kucoin, mexc)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "case_qnt"
S = datetime(2026, 9, 20, tzinfo=timezone.utc)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    E = datetime.now(timezone.utc)
    V = {"binance_usdm": binance_usdm("QNTUSDT", S, E), "binance_spot": binance_spot("QNTUSDT", S, E),
         "bybit_linear": bybit_linear("QNTUSDT", S, E), "bitget_fut": bitget("QNTUSDT", S, E, futures=True),
         "bitget_spot": bitget("QNTUSDT", S, E), "coinbase": coinbase("QNT-USD", S, E),
         "kucoin": kucoin("QNT-USDT", S, E), "mexc": mexc("QNTUSDT", S, E)}
    C = pd.DataFrame({k: v["close"] for k, v in V.items()})
    H = pd.DataFrame({k: v["high"] for k, v in V.items()})
    out: dict = {"as_of": E.isoformat(timespec="minutes"), "window_start": S.isoformat()}

    # --- volume by venue over the common complete window (T+1 archives end 09-27 23:59)
    w0, w1 = "2026-09-20", "2026-09-27 23:59"
    qv = {k: float(v["quote_volume"].loc[w0:w1].sum()) for k, v in V.items()}
    tot = sum(qv.values())
    out["quote_volume_usd_0920_0927"] = {k: round(x / 1e6, 1) for k, x in sorted(qv.items(), key=lambda x: -x[1])}
    out["perp_share"] = round(sum(qv[k] for k in ("binance_usdm", "bybit_linear", "bitget_fut")) / tot, 3)

    # --- price path milestones (Binance spot = the venue with a live current day)
    sp = V["binance_spot"]["close"]
    out["price"] = {
        "pre_move_0921_close": round(float(sp.loc["2026-09-21"].iloc[-1]), 2),
        "news_day_0924_1159": round(float(sp.loc["2026-09-24 11:59"]), 2),
        "news_day_0924_1600": round(float(sp.loc["2026-09-24 16:00"]), 2),
        "peak_high": round(float(H.loc["2026-09-27"].max().max()), 2),
        "peak_minute": str(H.loc["2026-09-27", "binance_spot"].idxmax()),
        "last": round(float(sp.iloc[-1]), 2), "last_ts": str(sp.index[-1]),
    }
    out["price"]["peak_over_pre_move"] = round(out["price"]["peak_high"] / out["price"]["pre_move_0921_close"], 2)

    # --- cross-venue lockstep: corr of 1m returns vs Binance USDM, lag 0 and +-1 min
    win = C.loc["2026-09-26 12:00":"2026-09-27 23:59"]
    r = np.log(win).diff()
    ref = r["binance_usdm"]
    cc = {}
    for k in r.columns:
        if k == "binance_usdm":
            continue
        cc[k] = {"lag0": round(float(ref.corr(r[k])), 3),
                 "venue_leads_1m": round(float(ref.corr(r[k].shift(1))), 3),
                 "venue_lags_1m": round(float(ref.corr(r[k].shift(-1))), 3)}
    out["corr_1m_returns_vs_binance_usdm_0926_12_to_0927_24"] = cc

    # --- basis perp vs spot (bps), open interest, top-trader L/S (5m metrics)
    fu = V["binance_usdm"]["close"]
    basis = ((fu / sp - 1) * 1e4).loc["2026-09-21":"2026-09-27"].resample("1D").mean()
    out["basis_bps_daily_mean"] = {str(k.date()): round(float(v), 1) for k, v in basis.items()}
    m = binance_usdm_metrics("QNTUSDT", S, E)
    oi = m["sum_open_interest_value"]
    out["open_interest_usd_m"] = {
        "0924_0000": round(float(oi.loc["2026-09-24 00:00"]) / 1e6, 2),
        "0925_0700": round(float(oi.loc["2026-09-25 07:00"]) / 1e6, 2),
        "0927_peak": round(float(oi.loc["2026-09-27"].max()) / 1e6, 2),
        "0927_peak_ts": str(oi.loc["2026-09-27"].idxmax()),
        "0927_2355": round(float(oi.loc["2026-09-27 23:55"]) / 1e6, 2),
    }
    ls = m["sum_toptrader_long_short_ratio"]
    out["top_trader_long_short"] = {"0924_0000": round(float(ls.loc["2026-09-24 00:00"]), 2),
                                    "0925_0700": round(float(ls.loc["2026-09-25 07:00"]), 2),
                                    "0927_2200": round(float(ls.loc["2026-09-27 22:00"]), 2)}

    # --- funding (Bitget QNTUSDT history; Binance fapi is geo-blocked here)
    try:
        fr = requests.get("https://api.bitget.com/api/v2/mix/market/history-fund-rate",
                          params={"symbol": "QNTUSDT", "productType": "USDT-FUTURES", "pageSize": 100},
                          timeout=15).json()["data"]
        f = pd.Series({pd.to_datetime(int(x["fundingTime"]), unit="ms", utc=True): float(x["fundingRate"])
                       for x in fr}).sort_index().loc["2026-09-21":]
        out["bitget_funding_rate_pct"] = {str(k): round(v * 100, 4) for k, v in f.items()}
    except Exception as e:  # noqa: BLE001 -- recorded, not dropped
        out["bitget_funding_rate_pct"] = f"unavailable: {e}"

    # --- hourly volume multiple at key hours (Binance USDM vs median hour 09-20..21)
    hq = V["binance_usdm"]["quote_volume"].resample("1h").sum()
    base = float(hq.loc["2026-09-20":"2026-09-21"].median())
    out["perp_hourly_volume_multiple"] = {
        "baseline_hour_usd_m": round(base / 1e6, 3),
        **{str(k): round(float(hq.loc[k] / base), 1) for k in
           ["2026-09-22 12:00", "2026-09-24 14:00", "2026-09-25 07:00", "2026-09-26 22:00",
            "2026-09-27 01:00", "2026-09-27 21:00", "2026-09-27 22:00"]}}
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
