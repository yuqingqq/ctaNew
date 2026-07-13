"""Squeeze-vs-dump classifier from FREE positioning proxies (user: build it). Metrics cache (OI, top-trader
long/short ratio, crowd long/short ratio) is RECENT-ONLY (2025-09+) -> the positioning classifier is recent-only
(cannot both-eras-validate; funding/climax/taker span both eras, done in addendum 67). Question: do positioning
signals separate the DUMP tail from the SQUEEZE tail at a blow-off, better than funding+climax alone?
Per-feature median-split of the high-climax blow-off short-EV, WEEK-CLUSTERED bootstrap CI (froth waves cluster
across symbols), gross + a cost haircut. All features PIT (as-of entry).
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

def load1h(sym, since="2025-07-01"):   # metrics/classifier are RECENT-only -> load only recent daily files (fast)
    fs = sorted(f for f in glob.glob(str(KD / sym / "5m" / "*.parquet")) if os.path.basename(f)[:-8] >= since)
    if not fs: return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close", "quote_volume", "taker_buy_quote_volume"]) for f in fs], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
    return pd.DataFrame({"close": df["close"].resample("1h").last(),
                         "qv": df["quote_volume"].resample("1h").sum(),
                         "tbq": df["taker_buy_quote_volume"].resample("1h").sum()}).dropna()

def series_at(path, col, times, kind="ffill"):
    if not Path(path).exists(): return np.full(len(times), np.nan)
    d = pd.read_parquet(path)
    if col not in d.columns: return np.full(len(times), np.nan)
    if not isinstance(d.index, pd.DatetimeIndex):
        tc = "create_time" if "create_time" in d.columns else ("calc_time" if "calc_time" in d.columns else "open_time")
        if tc not in d.columns: return np.full(len(times), np.nan)
        d = d.set_index(pd.to_datetime(d[tc], utc=True))
    s = d[col].sort_index(); s = s[~s.index.duplicated()]
    return s.reindex(times, method="ffill").values

def entries_for(sym, h):
    if h is None or len(h) < RUNUP_W + FWD + 24: return None
    c = h["close"]; qv = h["qv"]; tbq = h["tbq"]
    runup = c / c.shift(RUNUP_W) - 1
    vol24 = qv.rolling(24).sum(); med = vol24.rolling(30 * 24, min_periods=7 * 24).median()
    climax = vol24 / med.replace(0, np.nan)
    taker = tbq.rolling(24).sum() / vol24.replace(0, np.nan)          # taker-buy quote fraction, trailing 24h (PIT)
    fwd_ret = c.shift(-FWD) / c - 1
    fwd_dd = c.iloc[::-1].rolling(FWD, min_periods=FWD // 2).min().iloc[::-1].shift(-1) / c - 1
    df = pd.DataFrame({"t": c.index, "runup": runup.values, "climax": climax.values, "taker": taker.values,
                       "fwd_ret": fwd_ret.values, "fwd_dd": fwd_dd.values}).set_index("t")
    df = df[(df.index.hour == 0) & (df.runup >= PUMP_THR) & df.fwd_ret.notna() & df.climax.notna()]
    if not len(df): return None
    df["funding"] = series_at(FC / f"funding_{sym}.parquet", "funding_rate", df.index)
    mp = FC / f"metrics_{sym}.parquet"
    df["oi_chg"] = df["tt_ls"] = df["ls"] = np.nan
    if mp.exists():
        md = pd.read_parquet(mp)                                              # read metrics ONCE
        if not isinstance(md.index, pd.DatetimeIndex) and "create_time" in md.columns:
            md = md.set_index(pd.to_datetime(md["create_time"], utc=True))
        md = md[~md.index.duplicated()].sort_index()
        def mat(col, tt): return md[col].reindex(tt, method="ffill").values if col in md.columns else np.full(len(tt), np.nan)
        oi = mat("sum_open_interest", df.index); oi3 = mat("sum_open_interest", df.index - pd.Timedelta(days=3))
        df["oi_chg"] = oi / oi3 - 1
        df["tt_ls"] = mat("sum_toptrader_long_short_ratio", df.index)         # smart-money long/short
        df["ls"] = mat("count_long_short_ratio", df.index)                    # crowd long/short
    keep = []; last = None
    for t in df.index:
        if last is None or (t - last).days >= 7: keep.append(t); last = t
    df = df.loc[keep].copy(); df["sym"] = sym
    return df.reset_index()

def wk_boot(df, val="short"):    # week-clustered bootstrap of the mean (resample froth-week blocks)
    df = df.copy(); df["wk"] = df["t"].dt.to_period("W").astype(str)
    grps = [g[val].values for _, g in df.groupby("wk")]
    if len(grps) < 5: return (np.nan, np.nan)
    out = []
    for _ in range(2000):
        pick = rng.integers(0, len(grps), len(grps))
        out.append(np.concatenate([grps[i] for i in pick]).mean())
    return np.percentile(out, [2.5, 97.5])

def split(hi, feat, label):
    x = hi[hi[feat].notna()].copy()
    if len(x) < 30: print(f"    {label:26s}: n={len(x)} too few"); return
    x["g"] = pd.qcut(x[feat].rank(method="first"), 2, labels=["LO", "HI"], duplicates="drop")
    parts = []
    for g in ["LO", "HI"]:
        s = x[x.g == g]; sh = s["short"].values; lo, up = wk_boot(s)
        parts.append(f"{g}(med {s[feat].median():+.3f}) short {sh.mean()*100:+5.1f}% [wkCI {lo*100:+.0f},{up*100:+.0f}] dump {(s.fwd_dd<=-0.40).mean()*100:.0f}%")
    print(f"    {label:26s}: " + "  |  ".join(parts))

def main():
    metsyms = set(os.path.basename(x)[8:-8] for x in glob.glob(str(FC / "metrics_*.parquet")))
    syms = sorted(os.path.basename(p) for p in glob.glob(str(KD / "*")) if os.path.isdir(p) and os.path.basename(p) in metsyms)
    print(f"classifier over {len(syms)} metrics-covered symbols (positioning is recent-only)", flush=True)
    parts = []
    for i, s in enumerate(syms, 1):
        try:
            r = entries_for(s, load1h(s))
            if r is not None and len(r): parts.append(r)
        except Exception: pass
        if i % 60 == 0: print(f"  {i}/{len(syms)}", flush=True)
    e = pd.concat(parts, ignore_index=True); e["t"] = pd.to_datetime(e["t"], utc=True)
    e["short"] = -e["fwd_ret"]
    rec = e[e.t >= pd.Timestamp("2025-09-17", tz="UTC")].copy()   # metrics coverage start
    rec["ct"] = pd.qcut(rec["climax"].rank(method="first"), 3, labels=["lo", "mid", "HI"], duplicates="drop")
    hi = rec[rec.ct == "HI"].copy()
    e.to_csv(SD / "pump_classifier.csv", index=False)
    print(f"\n===== RECENT high-climax blow-offs, n={len(hi)} (metrics-era; positioning is recent-ONLY) =====")
    print(f"  naive short EV: {(-hi.fwd_ret).mean()*100:+.1f}%  |  base dump-rate {(hi.fwd_dd<=-0.40).mean()*100:.0f}%")
    print("  PER-FEATURE median-split short-EV (which positioning proxy separates squeeze<-LO / dump<-HI?):")
    for feat, lab in [("funding", "funding (low=dump 67)"), ("tt_ls", "top-trader long/short"),
                      ("ls", "crowd long/short"), ("oi_chg", "OI change 3d"), ("taker", "taker-buy fraction"),
                      ("climax", "volume climax")]:
        split(hi, feat, lab)
    # simple combined rule: SHORT only low-funding + (whichever positioning separated). Report after seeing splits.
    hf = hi[hi.funding.notna() & hi.tt_ls.notna()].copy()
    if len(hf) >= 30:
        lofund = hf.funding <= hf.funding.median()
        for name, mask in [("low-funding only", lofund),
                           ("low-funding & low-ttLS", lofund & (hf.tt_ls <= hf.tt_ls.median())),
                           ("low-funding & high-ttLS", lofund & (hf.tt_ls > hf.tt_ls.median()))]:
            s = hf[mask]; sh = s["short"].values; lo, up = wk_boot(s)
            print(f"  COMBO {name:26s}: n={len(s)} short {sh.mean()*100:+.1f}% [wkCI {lo*100:+.0f},{up*100:+.0f}] median {np.median(sh)*100:+.0f}% dump {(s.fwd_dd<=-0.40).mean()*100:.0f}%")
    print("\n  (gross; a real froth short pays ~0.5-1% round-trip + funding drag over 7d — apply a haircut to any positive EV)")
    print("CLASSIFIERDONE")

if __name__ == "__main__":
    main()
