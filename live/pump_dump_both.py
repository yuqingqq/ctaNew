"""Two follow-ups on the pump->dump risk-ranker (user: try both):
 IDEA 1 STOP-MANAGED SHORT — short the high-climax blow-off with a hard stop at +S adverse (cap the squeeze tail).
   Does capping the squeeze turn the negative-mean OOS short positive? Sweep S in {20,30,50}% (pre-registered).
 IDEA 2 FUNDING-AT-PEAK — high positive funding = crowded longs = dump-prone. Within the high-climax tercile, does
   funding separate the dumps from the squeezes?
Discipline (reviewer note 1): NON-OVERLAPPING entries (>=7d apart per symbol) so obs are independent; bootstrap CI
on the mean. Both eras. Short-PnL gross of cost (froth names are expensive/illiquid — a real drag, noted).
"""
import glob, os
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
KD = Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines")
FC = Path("/home/yuqing/ctaNew/data/ml/cache")
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
RUNUP_W = 240; FWD = 168; PUMP_THR = 0.50
rng = np.random.default_rng(13)

def load1h(sym):
    fs = sorted(glob.glob(str(KD / sym / "5m" / "*.parquet")))
    if not fs: return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close", "quote_volume"]) for f in fs], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    return pd.DataFrame({"close": df["close"].resample("1h").last(), "qv": df["quote_volume"].resample("1h").sum()}).dropna()

def funding_at(sym, times):
    f = FC / f"funding_{sym}.parquet"
    if not f.exists(): return np.full(len(times), np.nan)
    fd = pd.read_parquet(f); tc = "calc_time" if "calc_time" in fd.columns else "open_time"
    if tc not in fd.columns or "funding_rate" not in fd.columns: return np.full(len(times), np.nan)
    fd[tc] = pd.to_datetime(fd[tc], utc=True)
    fr = fd.set_index(tc)["funding_rate"].sort_index(); fr = fr[~fr.index.duplicated()]
    return fr.reindex(times, method="ffill").values

def entries_for(sym, h):
    if h is None or len(h) < RUNUP_W + FWD + 24: return None
    c = h["close"]; qv = h["qv"]
    runup = c / c.shift(RUNUP_W) - 1
    vol24 = qv.rolling(24).sum(); med = vol24.rolling(30 * 24, min_periods=7 * 24).median()
    climax = vol24 / med.replace(0, np.nan)
    fwd_ret = c.shift(-FWD) / c - 1
    fwd_maxrise = c.iloc[::-1].rolling(FWD, min_periods=FWD // 2).max().iloc[::-1].shift(-1) / c - 1   # adverse for short
    fwd_dd = c.iloc[::-1].rolling(FWD, min_periods=FWD // 2).min().iloc[::-1].shift(-1) / c - 1
    df = pd.DataFrame({"t": c.index, "runup": runup.values, "climax": climax.values, "fwd_ret": fwd_ret.values,
                       "fwd_maxrise": fwd_maxrise.values, "fwd_dd": fwd_dd.values}).set_index("t")
    df = df[(df.index.hour == 0) & (df.runup >= PUMP_THR) & df.fwd_ret.notna() & df.climax.notna()]
    if not len(df): return None
    df["funding"] = funding_at(sym, df.index)
    keep = []; last = None                       # non-overlapping: >=7d apart
    for t in df.index:
        if last is None or (t - last).days >= 7: keep.append(t); last = t
    df = df.loc[keep].copy(); df["sym"] = sym
    return df.reset_index()

def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 10: return (np.nan, np.nan)
    b = [rng.choice(x, len(x), replace=True).mean() for _ in range(2000)]
    return np.percentile(b, [2.5, 97.5])

def stopped(hi, S):
    return np.where(hi["fwd_maxrise"].values >= S, -S, -hi["fwd_ret"].values)

def main():
    syms = sorted(os.path.basename(p) for p in glob.glob(str(KD / "*")) if os.path.isdir(p))
    print(f"collecting non-overlapping pump-state entries from {len(syms)} symbols...", flush=True)
    parts = []
    for i, s in enumerate(syms, 1):
        try:
            r = entries_for(s, load1h(s))
            if r is not None and len(r): parts.append(r)
        except Exception: pass
        if i % 60 == 0: print(f"  {i}/{len(syms)}", flush=True)
    e = pd.concat(parts, ignore_index=True); e["t"] = pd.to_datetime(e["t"], utc=True)
    e.to_csv(SD / "pump_both.csv", index=False)
    for era, sub in [("OOS 2023-25", e[e.t < pd.Timestamp("2025-10-01", tz="UTC")]),
                     ("RECENT 2025-10+", e[e.t >= pd.Timestamp("2025-10-01", tz="UTC")])]:
        if len(sub) < 30: print(f"\n{era}: {len(sub)} entries — too few"); continue
        sub = sub.copy(); sub["ct"] = pd.qcut(sub["climax"].rank(method="first"), 3, labels=["lo", "mid", "HI"], duplicates="drop")
        hi = sub[sub.ct == "HI"].copy()
        print(f"\n========== {era}: {len(sub)} non-overlap entries, {len(hi)} high-climax ==========")
        unst = -hi["fwd_ret"].values
        lo, up = ci(unst)
        print(f"  IDEA 1 — naive short (no stop): mean {unst.mean()*100:+5.1f}% [CI {lo*100:+.1f},{up*100:+.1f}] median {np.median(unst)*100:+.1f}% win {(unst>0).mean()*100:.0f}%")
        for S in [0.20, 0.30, 0.50]:
            st = stopped(hi, S); lo, up = ci(st); hit = (hi["fwd_maxrise"].values >= S).mean()
            print(f"  IDEA 1 — stop +{int(S*100)}%: mean {st.mean()*100:+5.1f}% [CI {lo*100:+.1f},{up*100:+.1f}] median {np.median(st)*100:+.1f}% win {(st>0).mean()*100:.0f}% | stop-hit {hit*100:.0f}%")
        # IDEA 2 — funding within high-climax
        hf = hi[hi["funding"].notna()].copy()
        if len(hf) >= 30:
            hf["ff"] = pd.qcut(hf["funding"].rank(method="first"), 2, labels=["loF", "HIF"], duplicates="drop")
            print(f"  IDEA 2 — funding split within high-climax (n={len(hf)} with funding):")
            for f in ["loF", "HIF"]:
                x = hf[hf.ff == f]; s = -x["fwd_ret"].values; l, u = ci(s)
                print(f"      {f} (median funding {x.funding.median()*100:+.3f}%): short mean {s.mean()*100:+5.1f}% [CI {l*100:+.1f},{u*100:+.1f}] median {np.median(s)*100:+.1f}% dump-rate {(x.fwd_dd<=-0.40).mean()*100:.0f}%")
        else:
            print(f"  IDEA 2 — only {len(hf)} high-climax entries have funding (froth names often lack local funding cache)")
    print("\n  (all gross of cost; froth names carry high spread/funding cost + execution risk on the short)")
    print("PUMPBOTHDONE")

if __name__ == "__main__":
    main()
