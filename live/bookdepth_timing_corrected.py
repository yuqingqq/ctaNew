"""CORRECTED backtest addressing the review. Fixes:
 (1) FIXED PIT universe — only symbols present from <=2023-07 through recent (removes the expanding-universe /
     composition-drift artifact; agg_ob() averaged whatever was cached each bar, 53->175 names).
 (2) PIT-clean z — build agg_imb from OB observations over the FULL history, compute the rolling z there, THEN join
     to outcomes (previously rows were dropped for missing future returns BEFORE the z, contaminating the window).
 (3) REAL tradeable instruments — BTC and ETH (liquid, one perp), plus a FIXED liquid-major basket, plus the
     fixed-universe equal-weight for reference. (The old backtest traded the survivor equal-weight alt basket.)
 (4) HONEST stats — Sharpe with 10-day block-bootstrap CI, both eras. Does anything survive on a real instrument?
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_persist import close_4h
rng = np.random.default_rng(97)
CACHE = "/home/yuqing/ctaNew/data/ml/cache"
PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet"
CUT = pd.Timestamp("2025-10-01", tz="UTC")
FIXED_BEFORE = pd.Timestamp("2023-07-01", tz="UTC")
MAJORS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT",
          "DOTUSDT", "LTCUSDT", "BCHUSDT", "ATOMUSDT", "UNIUSDT", "ETCUSDT", "FILUSDT"]

def fixed_universe():
    syms = []
    for f in glob.glob(CACHE + "/l2_*.parquet"):
        if "BTCUSDT" in f: continue
        try:
            ix = pd.to_datetime(pd.read_parquet(f, columns=["l2_imb1"]).index, utc=True)
        except Exception:
            continue
        if len(ix) and ix.min() <= FIXED_BEFORE and ix.max() >= pd.Timestamp("2026-06-01", tz="UTC"):
            syms.append(Path(f).stem[3:])
    return sorted(syms)

def agg_z(syms):
    parts = []
    for s in syms:
        d = pd.read_parquet(f"{CACHE}/l2_{s}.parquet", columns=["l2_imb1"])
        d.index = pd.to_datetime(d.index, utc=True) + pd.Timedelta("4h")
        parts.append(d["l2_imb1"][~d.index.duplicated()].rename(s))
    agg = pd.concat(parts, axis=1).mean(axis=1).sort_index()                 # PIT equal-weight agg imbalance
    return ((agg - agg.rolling(90, min_periods=45).mean()) / agg.rolling(90, min_periods=45).std()).rename("z")

def instruments(syms):
    out = {}
    for name, sym in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
        c = close_4h(sym, "2023-01-01", "2026-07-15")
        if c is not None: out[name] = (c.shift(-1) / c - 1).rename(name)     # forward 4h return over [T,T+4h)
    p = pd.read_parquet(PANEL, columns=["symbol", "open_time", "return_pct"])
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    mj = [m for m in MAJORS if m in syms]
    out["MAJOR-basket(fix)"] = p[p.symbol.isin(mj)].groupby("open_time")["return_pct"].mean()
    out["ALT-EW(fixed univ)"] = p[p.symbol.isin(syms)].groupby("open_time")["return_pct"].mean()
    return out, mj

def daily(bar):
    return bar.groupby(bar.index.floor("1D")).apply(lambda x: (1 + x).prod() - 1)

def block_sharpe(dr, block=10, n=1500):
    d = dr.dropna().values
    if len(d) < 40: return (np.nan, np.nan, np.nan)
    base = d.mean() / d.std() * np.sqrt(365) if d.std() > 0 else np.nan
    nb = int(np.ceil(len(d) / block)); boot = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.integers(0, max(1, len(d) - block), nb)])[:len(d)]
        x = d[idx]
        if x.std() > 0: boot.append(x.mean() / x.std() * np.sqrt(365))
    lo, up = np.nanpercentile(boot, [2.5, 97.5]); return (base, lo, up)

def main():
    syms = fixed_universe(); z = agg_z(syms); insts, mj = instruments(syms)
    print(f"FIXED universe {len(syms)} syms (present <=2023-07 through recent) | major basket {len(mj)} | PIT z on {z.notna().sum()} bars\n")
    for HL, hlab in [(42, "7d"), (60, "10d")]:
        pos = (-z).clip(-1.5, 1.5).ewm(halflife=HL).mean()
        print(f"=== holding {hlab}, net 10bps | Sharpe [10d-block-boot 95% CI] ===")
        print(f"{'instrument':18s} | {'OOS Sharpe [CI]':27s} | {'RECENT Sharpe [CI]':27s}")
        for name, ret in insts.items():
            r = ret.reindex(pos.index)
            sd = daily((pos * r - 0.0010 * pos.diff().abs().fillna(0)).dropna())
            bo, br = block_sharpe(sd[sd.index < CUT]), block_sharpe(sd[sd.index >= CUT])
            print(f"{name:18s} | {bo[0]:+.2f} [{bo[1]:+.2f},{bo[2]:+.2f}] | {br[0]:+.2f} [{br[1]:+.2f},{br[2]:+.2f}]")
        print()
    print("read: does the signal survive on a REAL instrument (BTC/ETH) with a FIXED universe + honest CI? CORRECTEDDONE")

if __name__ == "__main__":
    main()
