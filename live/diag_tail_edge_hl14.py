"""Did hl=14d fix the long side? Compare per-rank tail edge on H2 (Jan22→May11)
for ORIGINAL preds vs hl=14d preds."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
REPO = Path("/home/yuqing/ctaNew")

PREDS_ORIG = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PREDS_HL14 = REPO/"live/state/convexity/x132_recw_hl14_h2_preds.parquet"

H2_START = pd.Timestamp("2026-01-22", tz="UTC")
H2_END = pd.Timestamp("2026-05-11", tz="UTC")
K = [1, 2, 3, 5]

def load_preds(path):
    d = pd.read_parquet(path)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]
    d = d[(d["open_time"]>=H2_START)&(d["open_time"]<=H2_END)]
    return d

def per_cycle(g):
    if len(g) < 2*max(K): return None
    g = g.sort_values("pred")
    out = dict(all_mean=g["return_pct"].mean())
    for k in K:
        out[f"top{k}_edge"] = g.tail(k)["return_pct"].mean() - g["return_pct"].mean()
        out[f"bot{k}_edge"] = g["return_pct"].mean() - g.head(k)["return_pct"].mean()
        out[f"top{k}_abs"]  = g.tail(k)["return_pct"].mean()
        out[f"bot{k}_abs"]  = g.head(k)["return_pct"].mean()
    return pd.Series(out)

for label, path in [("ORIGINAL preds (static walk-forward)", PREDS_ORIG),
                     ("hl=14d preds (recency-weighted)",      PREDS_HL14)]:
    d = load_preds(path)
    cyc = d.groupby("open_time").apply(per_cycle).dropna(how="all")
    cyc.index = cyc.index.get_level_values(0); cyc = cyc.reset_index().rename(columns={"index":"open_time"})
    print(f"\n=== {label} on H2 (n_cycles={len(cyc)}, market mean {cyc['all_mean'].mean()*1e4:+.1f} bps) ===")
    print(f"  {'K':>3}  {'top-K edge':>14}  {'top-K absolute':>15}  {'bot-K edge':>14}  {'bot-K absolute':>15}  {'L-S spread':>13}")
    for k in K:
        te = cyc[f"top{k}_edge"]; be = cyc[f"bot{k}_edge"]
        ta = cyc[f"top{k}_abs"];  ba = cyc[f"bot{k}_abs"]
        spread = ta - ba
        print(f"  {k:>3}  {te.mean()*1e4:+10.1f} bps  {ta.mean()*1e4:+11.1f} bps  "
              f"{be.mean()*1e4:+10.1f} bps  {ba.mean()*1e4:+11.1f} bps  {spread.mean()*1e4:+9.1f} bps")
