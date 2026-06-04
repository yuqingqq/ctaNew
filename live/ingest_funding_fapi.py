"""Top up funding caches with CURRENT settled funding from FAPI (Vision monthly lags into the live month).
GET /fapi/v1/fundingRate?symbol=&limit= → recent settled fundingTime/fundingRate; merge into the cache so
funding_rate_z_7d stays real for the live bars. Weight-1/call, throttled. Run each cycle before the panel.
"""
import sys, time
from pathlib import Path
import requests, pandas as pd
REPO = Path("/home/yuqing/ctaNew"); CACHE = REPO/"data/ml/cache"
FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"


def main():
    syms = sorted(p.name for p in (REPO/"data/ml/test/parquet/klines").iterdir() if p.name.endswith("USDT"))
    n_ok = n_new = 0
    for sym in syms:
        try:
            r = requests.get(FAPI, params={"symbol": sym, "limit": 40}, timeout=15)
            if r.status_code != 200:
                time.sleep(0.25); continue
            j = r.json()
            if not j: time.sleep(0.2); continue
            new = pd.DataFrame([{"calc_time": pd.to_datetime(x["fundingTime"], unit="ms", utc=True),
                                 "funding_rate": float(x["fundingRate"])} for x in j])
            fp = CACHE/f"funding_{sym}.parquet"
            if fp.exists():
                old = pd.read_parquet(fp)
                iv = old["interval_hours"].iloc[-1] if "interval_hours" in old.columns and len(old) else 8
                new["interval_hours"] = iv
                comb = pd.concat([old, new], ignore_index=True)
                before = len(old)
            else:
                new["interval_hours"] = 8; comb = new; before = 0
            comb = comb.drop_duplicates("calc_time").sort_values("calc_time").reset_index(drop=True)
            comb.to_parquet(fp); n_ok += 1; n_new += len(comb) - before
        except Exception:
            pass
        time.sleep(0.2)                       # throttle — avoid the 418 IP ban
    print(f"[ingest_funding_fapi] topped up {n_ok}/{len(syms)} caches, +{n_new} rows")


if __name__ == "__main__":
    main()
