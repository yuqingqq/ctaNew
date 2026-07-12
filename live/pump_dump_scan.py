"""Pump-then-dump pattern scan (user question: is there a COMMON, actionable pattern? LABUSDT = +6515% pump over
228d then -71% dump in 8d with a 98x volume climax at the peak). For every local symbol: find PUMP PEAKS (local
max after a >=+50% 10-day run-up), label each DUMPED (next-7d drawdown <=-40%) vs SURVIVED, and record peak
features (volume-climax ratio, 3d acceleration, days-since-listing). The decisive test is DUMPED-vs-SURVIVED: if a
pre/at-peak feature separates them, the blow-off is actionable; if not, the dump is unpredictable (the session's
prior). Resolution 1h. Also loads LABUSDT (fetched to scratchpad) as the motivating example.
"""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
KD = Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines")
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
PUMP_W = 10 * 24; PUMP_THR = 0.50      # >=+50% over prior 10 days
DUMP_H = 7 * 24;  DUMP_DD = -0.40      # <=-40% over next 7 days
LOCALMAX_W = 3 * 24                     # local max over +/-3 days

def load1h(sym, path=None):
    if path:
        df = pd.read_parquet(path).rename(columns={"t": "open_time", "quote_volume": "qv"})
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.set_index("open_time")
        return pd.DataFrame({"close": df["close"], "qv": df["qv"]}).dropna()
    fs = sorted(glob.glob(str(KD / sym / "5m" / "*.parquet")))
    if not fs: return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close", "quote_volume"]) for f in fs], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    return pd.DataFrame({"close": df["close"].resample("1h").last(),
                         "qv": df["quote_volume"].resample("1h").sum()}).dropna()

def find_pumps(h, sym):
    if h is None or len(h) < PUMP_W + DUMP_H: return []
    c = h["close"]; qv = h["qv"]
    runup = c / c.shift(PUMP_W) - 1
    fwd_min = c.iloc[::-1].rolling(DUMP_H, min_periods=DUMP_H // 2).min().iloc[::-1].shift(-1)
    fdd = fwd_min / c - 1
    localmax = c >= c.rolling(LOCALMAX_W * 2, center=True, min_periods=LOCALMAX_W).max()
    daily_med = (qv.rolling(24).sum()).rolling(30 * 24, min_periods=7 * 24).median()   # trailing median 24h $vol
    ev = []; last = None
    for t in c.index[(localmax & (runup >= PUMP_THR)).fillna(False)]:
        if last is not None and (t - last).total_seconds() < 7 * 86400: continue
        last = t
        dd = fdd.loc[t]
        if not np.isfinite(dd): continue
        vol24 = qv.loc[:t].iloc[-24:].sum(); med = daily_med.loc[t]
        v3 = (c.loc[t] / c.shift(3 * 24).loc[t] - 1) if (t - pd.Timedelta(days=3)) >= c.index[0] else np.nan
        ev.append(dict(sym=sym, t=t, runup=float(runup.loc[t]), fdd=float(dd),
                       dumped=bool(dd <= DUMP_DD), vol_climax=float(vol24 / med) if med and np.isfinite(med) and med > 0 else np.nan,
                       accel3d=float(v3), age_d=int((t - c.index[0]).days)))
    return ev

def main():
    syms = sorted(os.path.basename(p) for p in glob.glob(str(KD / "*")) if os.path.isdir(p))
    print(f"scanning {len(syms)} local symbols for pump peaks (runup>=+50%/10d, local max)...", flush=True)
    allev = []
    for i, s in enumerate(syms, 1):
        try: allev += find_pumps(load1h(s), s)
        except Exception: pass
        if i % 40 == 0: print(f"  {i}/{len(syms)} ({len(allev)} pump peaks so far)", flush=True)
    # LABUSDT (fetched) as the example
    if (SD / "LABUSDT_1h.parquet").exists():
        allev += find_pumps(load1h("LABUSDT", SD / "LABUSDT_1h.parquet"), "LABUSDT")
    e = pd.DataFrame(allev)
    e.to_csv(SD / "pump_events.csv", index=False)
    print(f"\n===== {len(e)} pump peaks across {e.sym.nunique()} symbols =====")
    nd = int(e.dumped.sum())
    print(f"DUMPED (next-7d <= -40%): {nd} ({nd/len(e)*100:.0f}%)  |  SURVIVED: {len(e)-nd}")
    print(f"  median pump run-up (prior 10d): {e.runup.median()*100:+.0f}%  | median forward-7d move: {e.fdd.median()*100:+.0f}%")
    print("\n--- DECISIVE TEST: do DUMPED pumps look different AT THE PEAK vs SURVIVED? ---")
    D = e[e.dumped]; S = e[~e.dumped]
    for feat, lab in [("vol_climax", "volume climax (24h/median)"), ("accel3d", "3-day acceleration"),
                      ("age_d", "days since listing"), ("runup", "prior-10d run-up")]:
        dm, sm = D[feat].median(), S[feat].median()
        print(f"  {lab:32s}: DUMPED median {dm:+.2f}  vs SURVIVED median {sm:+.2f}   (ratio {dm/sm if sm not in (0,) and np.isfinite(sm) else float('nan'):.2f})")
    print("\n--- LABUSDT peaks ---")
    print(e[e.sym == "LABUSDT"][["t", "runup", "fdd", "dumped", "vol_climax", "age_d"]].to_string(index=False))
    print("PUMPDUMPDONE")

if __name__ == "__main__":
    main()
