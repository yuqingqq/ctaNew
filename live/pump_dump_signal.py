"""Decisive test of the pump->dump signal: is the volume-climax blow-off PIT-PREDICTIVE of the dump, with usable
precision, in BOTH eras? (medians looked promising: dumped vol_climax 13.8 vs survived 5.2 — but 14% base rate +
this session's rule that everything dies on the both-eras test.) For each symbol, sample DAILY, keep PUMP-STATE bars
(prior-10d run-up>=+50%), compute PIT features (vol_climax=24h$vol/trailing-30d-median, 3d accel — all past-only) and
the FORWARD-7d return + drawdown (outcome). Tercile by vol_climax within each ERA; report forward return, dump-rate
(<=-40%), and a naive short-the-signal PnL. Split RECENT (2025-10+) vs OOS (2023-25).
"""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
KD = Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines")
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
RUNUP_W = 240; FWD = 168; PUMP_THR = 0.50   # 10d run-up, 7d forward, +50%

def load1h(sym, path=None):
    if path:
        df = pd.read_parquet(path).rename(columns={"t": "open_time", "quote_volume": "qv"})
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True); df = df.set_index("open_time")
        return pd.DataFrame({"close": df["close"], "qv": df["qv"]}).dropna()
    fs = sorted(glob.glob(str(KD / sym / "5m" / "*.parquet")))
    if not fs: return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close", "quote_volume"]) for f in fs], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    return pd.DataFrame({"close": df["close"].resample("1h").last(), "qv": df["quote_volume"].resample("1h").sum()}).dropna()

def rows_for(sym, h):
    if h is None or len(h) < RUNUP_W + FWD + 24: return None
    c = h["close"]; qv = h["qv"]
    runup = c / c.shift(RUNUP_W) - 1                                  # PIT
    vol24 = qv.rolling(24).sum(); med = vol24.rolling(30 * 24, min_periods=7 * 24).median()
    climax = vol24 / med.replace(0, np.nan)                            # PIT
    accel = c / c.shift(72) - 1                                        # PIT (3d)
    fwd_ret = c.shift(-FWD) / c - 1                                    # forward 7d return
    fwd_dd = c.iloc[::-1].rolling(FWD, min_periods=FWD // 2).min().iloc[::-1].shift(-1) / c - 1
    df = pd.DataFrame({"t": c.index, "runup": runup.values, "climax": climax.values, "accel": accel.values,
                       "fwd_ret": fwd_ret.values, "fwd_dd": fwd_dd.values}).set_index("t")
    df = df[df.index.hour == 0]                                        # 1 sample/day (avoid overlap)
    df = df[(df.runup >= PUMP_THR) & df.fwd_ret.notna() & df.climax.notna()]
    df["sym"] = sym
    return df.reset_index()

def report(e, era):
    if len(e) < 30:
        print(f"\n===== {era}: only {len(e)} pump-state days — too few to tercile (see caveat) ====="); return
    print(f"\n===== {era}: {len(e)} pump-state days | base dump-rate(fwd_dd<=-40%) {(e.fwd_dd<=-0.40).mean()*100:.0f}% =====")
    e = e.copy(); e["ct"] = pd.qcut(e["climax"].rank(method="first"), 3, labels=["lo", "mid", "HI"], duplicates="drop")
    for t in ["lo", "mid", "HI"]:
        x = e[e.ct == t]
        print(f"  climax {t:3s} (median {x.climax.median():5.1f}): fwd-7d return {x.fwd_ret.median()*100:+5.1f}% (mean {x.fwd_ret.mean()*100:+5.1f}%) | dump-rate {(x.fwd_dd<=-0.40).mean()*100:3.0f}% | short-PnL(-fwd_ret) mean {(-x.fwd_ret.mean())*100:+.1f}%")
    hi = e[e.ct == "HI"]; sh = -hi["fwd_ret"]
    dd = hi.assign(day=hi["t"].dt.date).groupby("day").apply(lambda g: (-g["fwd_ret"]).mean())
    print(f"  >> SHORT top-climax tercile: n={len(hi)}, mean short-PnL {sh.mean()*100:+.1f}%, median {sh.median()*100:+.1f}%, win% {(sh>0).mean()*100:.0f}%")

def main():
    syms = sorted(os.path.basename(p) for p in glob.glob(str(KD / "*")) if os.path.isdir(p))
    print(f"scanning {len(syms)} symbols for pump-state days...", flush=True)
    parts = []
    for i, s in enumerate(syms, 1):
        try:
            r = rows_for(s, load1h(s))
            if r is not None and len(r): parts.append(r)
        except Exception: pass
        if i % 50 == 0: print(f"  {i}/{len(syms)}", flush=True)
    if (SD / "LABUSDT_1h.parquet").exists():
        r = rows_for("LABUSDT", load1h("LABUSDT", SD / "LABUSDT_1h.parquet"))
        if r is not None: parts.append(r)
    e = pd.concat(parts, ignore_index=True)
    e["t"] = pd.to_datetime(e["t"], utc=True)
    e.to_csv(SD / "pump_signal.csv", index=False)
    rec = e[e.t >= pd.Timestamp("2025-10-01", tz="UTC")]; oos = e[e.t < pd.Timestamp("2025-10-01", tz="UTC")]
    report(oos, "OOS 2023-25"); report(rec, "RECENT 2025-10+")
    print("\n  (short-PnL = -forward-7d return; a real short also pays cost + faces squeeze/execution risk on these illiquid froth names)")
    print("PUMPSIGNALDONE")

if __name__ == "__main__":
    main()
