"""Enrich every pump-state entry (pump_both.csv) with a RICH PIT feature set for a proper multivariate squeeze-vs-
dump model (user: the univariate splits are too trivial). Features (all past-only at entry):
 price/vol: climax, climax_build (trailing-max climax), runup_10d(have), runup_3d, runup_1d(accel), rvol_7d,
            dist_ath (distance below running ATH), parab (accel ratio 1d/3d).
 flow: taker (24h taker-buy fraction).
 funding: funding(have), funding_chg_3d, funding_z (vs trailing 30d).
 positioning (RECENT-ONLY, metrics start 2025-09): oi_chg_3d, oi_lvl_z, tt_ls (smart$), ls (crowd), taker_ls.
 age: age_d (days since first kline).
Targets already in pump_both: fwd_ret, fwd_dd, fwd_maxrise. Writes pump_enriched.csv.
"""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import warnings; warnings.filterwarnings("ignore")
KD = Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines")
FC = Path("/home/yuqing/ctaNew/data/ml/cache")
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
WANT = ["open_time", "high", "close", "quote_volume", "taker_buy_quote_volume"]

def _read1(f):
    """defensive per-file read: intersect requested cols with the file's actual schema (old parquets vary),
    fill any missing with NaN so a single odd file can't crash the whole concat."""
    try:
        avail = set(pq.ParquetFile(f).schema.names)
    except Exception:
        return None
    cols = [c for c in WANT if c in avail]
    if "open_time" not in cols or "close" not in cols:
        return None
    try:
        d = pd.read_parquet(f, columns=cols)
    except Exception:
        return None
    for c in WANT:
        if c not in d.columns:
            d[c] = np.nan
    return d

def load1h(sym):
    fs = sorted(glob.glob(str(KD / sym / "5m" / "*.parquet")))
    if not fs: return None
    parts = [d for d in (_read1(f) for f in fs) if d is not None and len(d)]
    if not parts: return None
    df = pd.concat(parts, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    return pd.DataFrame({"close": df["close"].resample("1h").last(), "high": df["high"].resample("1h").max(),
                         "qv": df["quote_volume"].resample("1h").sum(), "tbq": df["taker_buy_quote_volume"].resample("1h").sum()}).dropna()

def kfeats(h):
    c = h["close"]; qv = h["qv"]; tbq = h["tbq"]; ret = c.pct_change()
    vol24 = qv.rolling(24).sum(); med = vol24.rolling(30 * 24, min_periods=7 * 24).median()
    climax = vol24 / med.replace(0, np.nan)
    r3 = c / c.shift(72) - 1; r1 = c / c.shift(24) - 1
    return pd.DataFrame({
        "climax": climax, "climax_build": climax.rolling(72).max(),
        "runup_3d": r3, "runup_1d": r1, "parab": (r1) / (r3.abs() + 1e-9),
        "rvol_7d": ret.rolling(168).std() * np.sqrt(24 * 365),
        "dist_ath": c / c.cummax() - 1,
        "taker": tbq.rolling(24).sum() / vol24.replace(0, np.nan),
        "age_d": (c.index - c.index[0]).days,
    }, index=c.index)

def ser(path, col):
    if not Path(path).exists(): return None
    d = pd.read_parquet(path)
    if not isinstance(d.index, pd.DatetimeIndex):
        tc = "create_time" if "create_time" in d.columns else ("calc_time" if "calc_time" in d.columns else "open_time")
        if tc not in d.columns or col not in d.columns: return None
        d = d.set_index(pd.to_datetime(d[tc], utc=True))
    if col not in d.columns: return None
    s = d[col].sort_index(); return s[~s.index.duplicated()]

def asof(s, t): return s.reindex(t, method="ffill").values if s is not None else np.full(len(t), np.nan)

def main():
    e = pd.read_csv(SD / "pump_both.csv"); e["t"] = pd.to_datetime(e["t"], utc=True)
    syms = sorted(e.sym.unique()); print(f"enriching {len(e)} entries across {len(syms)} symbols...", flush=True)
    out = []; nfail = 0
    for i, (sym, g) in enumerate(e.groupby("sym"), 1):
        g = g.sort_values("t").copy()
        try:
            h = load1h(sym)
            if h is not None:
                f = kfeats(h)
                for col in f.columns: g[col] = f[col].reindex(g["t"], method="ffill").values
            fr = ser(FC / f"funding_{sym}.parquet", "funding_rate")
            if fr is not None:
                fn = asof(fr, g["t"]); f3 = asof(fr, g["t"] - pd.Timedelta(days=3))
                mu = fr.rolling(90).mean(); sd = fr.rolling(90).std()
                g["funding_chg"] = fn - f3
                g["funding_z"] = (fn - asof(mu, g["t"])) / (asof(sd, g["t"]) + 1e-12)
            for col, out_c in [("sum_open_interest", "oi"), ("sum_toptrader_long_short_ratio", "tt_ls"),
                               ("count_long_short_ratio", "ls"), ("sum_taker_long_short_vol_ratio", "taker_ls")]:
                s = ser(FC / f"metrics_{sym}.parquet", col)
                if col == "sum_open_interest" and s is not None:
                    g["oi_chg"] = asof(s, g["t"]) / asof(s, g["t"] - pd.Timedelta(days=3)) - 1
                else:
                    g[out_c] = asof(s, g["t"])
        except Exception as ex:
            nfail += 1; print(f"  [skip-enrich {sym}: {type(ex).__name__} {ex}]", flush=True)
        out.append(g)
        if i % 20 == 0: print(f"  {i}/{len(syms)} (fails={nfail})", flush=True)
    en = pd.concat(out, ignore_index=True)
    en.to_csv(SD / "pump_enriched.csv", index=False)
    fc = [c for c in en.columns if c not in ("sym", "t", "fwd_ret", "fwd_dd", "fwd_maxrise")]
    print(f"wrote pump_enriched.csv: {len(en)} rows, {len(fc)} features: {fc}")
    print(f"  recent (2025-10+) {len(en[en.t>=pd.Timestamp('2025-10-01',tz='UTC')])} | OOS {len(en[en.t<pd.Timestamp('2025-10-01',tz='UTC')])}")
    print(f"  metrics-covered rows (tt_ls non-null): {en['tt_ls'].notna().sum()}")
    print("ENRICHDONE")

if __name__ == "__main__":
    main()
