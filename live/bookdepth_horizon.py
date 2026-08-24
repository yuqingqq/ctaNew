"""Where does the order-book-imbalance signal actually live? The 4h/1d tests were all null, but book imbalance is a
known SHORT-horizon predictor. Aggregate imbalance to 5-MIN bars (not 4h) for a few liquid names and measure the
per-name time-series IC of imbalance vs forward return at horizons 5m / 15m / 30m / 1h / 4h. If IC is positive at
5-15m and DECAYS toward 0 by 1-4h, the idea is real but lives below this strategy's horizon (unusable for a multi-day
book, and a different HFT game net of spread/impact). PIT: imbalance during bar B is known at B's close -> predicts
forward from B's close. Recent window, liquid names (demonstration of the mechanism/horizon, not a both-era claim).
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_loader import _fetch_day
KD = Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines")
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "AVAXUSDT"]
DAYS = pd.date_range("2026-06-08", "2026-06-30", freq="D")
HZ = {"5m": 1, "15m": 3, "30m": 6, "1h": 12, "2h": 24, "4h": 48}   # in 5-min steps

def imb_5m(sym):
    from concurrent.futures import ThreadPoolExecutor
    parts = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        for r in ex.map(lambda d: _fetch_day(sym, d), DAYS):
            if r is not None and len(r): parts.append(r["imb1"])
    if not parts: return None
    s = pd.concat(parts).sort_index()
    s = s[~s.index.duplicated()]
    return s.groupby(s.index.floor("5min")).mean()          # imbalance per 5-min bar

def close_5m(sym):
    fs = [f for f in glob.glob(str(KD / sym / "5m" / "*.parquet")) if "2026-06" in Path(f).stem]
    if not fs: return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(fs)], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.drop_duplicates("open_time").set_index("open_time")["close"].sort_index()

def main():
    print(f"fine-horizon imbalance IC, {len(SYMS)} liquid names, {DAYS[0].date()}..{DAYS[-1].date()} (recent demo)\n")
    per_sym = {}
    for sym in SYMS:
        im = imb_5m(sym); c = close_5m(sym)
        if im is None or c is None: continue
        d = pd.DataFrame({"imb": im}).join(pd.DataFrame({"c": c}), how="inner")
        ics = {}
        for h, k in HZ.items():
            fwd = d["c"].shift(-k) / d["c"] - 1
            ok = d["imb"].notna() & fwd.notna()
            ics[h] = d.loc[ok, "imb"].corr(fwd[ok], method="spearman")
        per_sym[sym] = ics
        print(f"  {sym:10s} " + "  ".join(f"{h}:{ics[h]:+.3f}" for h in HZ))
    A = pd.DataFrame(per_sym).T
    print(f"\n  {'MEAN':10s} " + "  ".join(f"{h}:{A[h].mean():+.3f}" for h in HZ))
    print(f"  {'(t-stat)':10s} " + "  ".join(f"{h}:{A[h].mean()/ (A[h].std()/len(A)**0.5):+.1f}" for h in HZ))
    print("\nread: if IC is clearly>0 at 5-15m and decays to ~0 by 1-4h => real but SHORT-horizon (below this strategy's grid). HZDONE")

if __name__ == "__main__":
    main()
