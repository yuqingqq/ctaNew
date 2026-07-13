"""Squeeze-vs-dump positioning classifier (lightweight): reuse the recent pump-state entries already in
pump_both.csv (climax/funding/forward outcomes) and ENRICH each with the positioning metrics (OI change,
smart-money top-trader long/short, crowd long/short) — read each metrics file once per symbol (no kline re-scan).
RECENT-ONLY (metrics start 2025-09). Per-feature median-split short-EV with WEEK-CLUSTERED bootstrap CI; combined
rule. Question: do positioning signals separate the dump tail from the squeeze tail at a high-climax blow-off?
"""
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
FC = Path("/home/yuqing/ctaNew/data/ml/cache")
SD = Path("/tmp/claude-1001/-home-yuqing-ctaNew/ecbd8f4c-236c-426c-85e5-e1f6b6edd11d/scratchpad")
rng = np.random.default_rng(13)

def enrich(sym, g):
    g = g.sort_values("t").copy()
    mp = FC / f"metrics_{sym}.parquet"
    g["oi_chg"] = g["tt_ls"] = g["ls"] = np.nan
    if mp.exists():
        md = pd.read_parquet(mp)
        if "create_time" in md.columns and not isinstance(md.index, pd.DatetimeIndex):
            md = md.set_index(pd.to_datetime(md["create_time"], utc=True))
        md = md[~md.index.duplicated()].sort_index()
        def at(col, tt): return md[col].reindex(tt, method="ffill").values if col in md.columns else np.full(len(tt), np.nan)
        g["tt_ls"] = at("sum_toptrader_long_short_ratio", g["t"])
        g["ls"] = at("count_long_short_ratio", g["t"])
        oi = at("sum_open_interest", g["t"]); oi3 = at("sum_open_interest", g["t"] - pd.Timedelta(days=3))
        g["oi_chg"] = oi / oi3 - 1
    return g

def wk_boot(df, val="short"):
    df = df.copy(); df["wk"] = df["t"].dt.to_period("W").astype(str)
    grps = [x[val].values for _, x in df.groupby("wk")]
    if len(grps) < 4: return (np.nan, np.nan)
    out = [np.concatenate([grps[i] for i in rng.integers(0, len(grps), len(grps))]).mean() for _ in range(2000)]
    return np.percentile(out, [2.5, 97.5])

def split(hi, feat, label):
    x = hi[hi[feat].notna()].copy()
    if len(x) < 24: print(f"    {label:28s}: n={len(x)} too few"); return
    x["g"] = pd.qcut(x[feat].rank(method="first"), 2, labels=["LO", "HI"], duplicates="drop")
    out = []
    for g in ["LO", "HI"]:
        s = x[x.g == g]; lo, up = wk_boot(s)
        out.append(f"{g}(med {s[feat].median():+.3f}) short {s['short'].mean()*100:+5.1f}% [wk {lo*100:+.0f},{up*100:+.0f}] dump{(s.fwd_dd<=-0.40).mean()*100:.0f}%")
    print(f"    {label:28s}: " + "   |   ".join(out))

def main():
    e = pd.read_csv(SD / "pump_both.csv"); e["t"] = pd.to_datetime(e["t"], utc=True)
    rec = e[e.t >= pd.Timestamp("2025-09-17", tz="UTC")].copy()
    rec = pd.concat([enrich(s, g) for s, g in rec.groupby("sym")], ignore_index=True)
    rec["short"] = -rec["fwd_ret"]
    rec["ct"] = pd.qcut(rec["climax"].rank(method="first"), 3, labels=["lo", "mid", "HI"], duplicates="drop")
    hi = rec[rec.ct == "HI"].copy()
    nmet = hi["tt_ls"].notna().sum()
    print(f"===== RECENT high-climax blow-offs n={len(hi)} ({nmet} with metrics); metrics recent-ONLY =====")
    print(f"  naive short EV {hi['short'].mean()*100:+.1f}% | base dump-rate {(hi.fwd_dd<=-0.40).mean()*100:.0f}% | n symbols {hi.sym.nunique()}, weeks {hi['t'].dt.to_period('W').nunique()}")
    print("  PER-FEATURE median split (does the proxy separate SQUEEZE=low-short vs DUMP=high-short?):")
    for f, l in [("funding", "funding (addendum67: low=dump)"), ("tt_ls", "top-trader long/short (smart$)"),
                 ("ls", "crowd long/short"), ("oi_chg", "OI change 3d")]:
        split(hi, f, l)
    hf = hi[hi.funding.notna() & hi.tt_ls.notna()].copy()
    if len(hf) >= 24:
        print("  COMBINED rules (short-EV, week-clustered CI):")
        lofund = hf.funding <= hf.funding.median()
        for name, m in [("low-funding", lofund),
                        ("low-funding & low-crowd-LS", lofund & (hf.ls <= hf.ls.median())),
                        ("low-funding & high-crowd-LS", lofund & (hf.ls > hf.ls.median())),
                        ("low-funding & low-smart-LS", lofund & (hf.tt_ls <= hf.tt_ls.median())),
                        ("low-funding & high-smart-LS", lofund & (hf.tt_ls > hf.tt_ls.median()))]:
            s = hf[m]
            if len(s) < 8: print(f"    {name:30s}: n={len(s)} too few"); continue
            lo, up = wk_boot(s)
            print(f"    {name:30s}: n={len(s)} short {s['short'].mean()*100:+5.1f}% [wk {lo*100:+.0f},{up*100:+.0f}] med {np.median(s['short'])*100:+.0f}% dump{(s.fwd_dd<=-0.40).mean()*100:.0f}%")
    print("\n  (gross; froth short pays ~0.5-1% round-trip + funding drag over 7d; RECENT-ONLY, cannot both-eras-validate)")
    print("CLASSIFIER2DONE")

if __name__ == "__main__":
    main()
