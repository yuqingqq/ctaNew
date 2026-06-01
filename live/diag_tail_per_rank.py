"""Per-rank tail-edge diagnostic. For each cycle, compute the mean realized return for
the top-N (highest-pred) and bot-N (lowest-pred) symbols, for N=1,2,3,5,7,10, vs the
all-syms mean. If top-1 or top-2 still has positive edge in H2 even though top-5 doesn't,
the long signal is just compressed into the most-extreme tail and reducing K_long would
recover it. If top-1 is also negative in H2, the long signal is truly dead.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"

print("loading preds...")
d = pd.read_parquet(PREDS)
d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]

H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))

def per_cycle_per_rank(g):
    if len(g) < 20: return None
    g = g.sort_values("pred")
    n = len(g)
    out = dict(all_mean=g["return_pct"].mean(), n=n)
    for k in [1,2,3,5,7,10]:
        if n < 2*k: continue
        out[f"top{k}_mean"] = g.tail(k)["return_pct"].mean()
        out[f"bot{k}_mean"] = g.head(k)["return_pct"].mean()
        out[f"top{k}_edge"] = g.tail(k)["return_pct"].mean() - g["return_pct"].mean()
        out[f"bot{k}_edge"] = g["return_pct"].mean() - g.head(k)["return_pct"].mean()
    return pd.Series(out)

print("computing per-cycle per-rank tail returns...")
cyc = d.groupby("open_time").apply(per_cycle_per_rank).dropna(how="all")
cyc.index = cyc.index.get_level_values(0)
cyc = cyc.reset_index()
cyc.rename(columns={"index":"open_time"}, inplace=True)
cyc["open_time"] = pd.to_datetime(cyc["open_time"], utc=True)

for label,(s,e) in [("FULL OOS",(H1[0],H2[1])), ("H1",H1), ("H2",H2)]:
    sub = cyc[(cyc["open_time"]>=s) & (cyc["open_time"]<e)]
    print(f"\n=== {label} (n_cycles={len(sub)}) ===")
    print(f"  market mean: {sub['all_mean'].mean()*1e4:+.1f} bps")
    print(f"\n  {'K':>3}  {'top-K edge':>14}  {'%pos':>6}  {'bot-K edge':>14}  {'%pos':>6}  {'L-S spread':>12}")
    for k in [1,2,3,5,7,10]:
        if f"top{k}_edge" not in sub.columns: continue
        te = sub[f"top{k}_edge"].dropna()
        be = sub[f"bot{k}_edge"].dropna()
        spread = sub[f"top{k}_mean"].dropna() - sub[f"bot{k}_mean"].dropna()
        print(f"  {k:>3}  {te.mean()*1e4:+10.1f} bps  {100*(te>0).mean():>5.1f}%  {be.mean()*1e4:+10.1f} bps  {100*(be>0).mean():>5.1f}%  {spread.mean()*1e4:+8.1f} bps")

# Per-month top-1 and top-5 trajectory
print("\n=== Monthly top-1 vs top-5 long edge trajectory ===")
cyc["month"] = cyc["open_time"].dt.to_period("M").astype(str)
mm = cyc.groupby("month").agg(n=("top1_edge","count"),
    top1=("top1_edge",lambda x:x.mean()*1e4),
    top2=("top2_edge",lambda x:x.mean()*1e4),
    top5=("top5_edge",lambda x:x.mean()*1e4),
    bot1=("bot1_edge",lambda x:x.mean()*1e4),
    bot5=("bot5_edge",lambda x:x.mean()*1e4),
    mkt =("all_mean",lambda x:x.mean()*1e4))
print(mm.round(1).tail(20).to_string())
