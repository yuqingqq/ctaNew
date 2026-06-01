"""Sanity check: is the bot-K short edge real cross-sectional alpha, or is it
covertly beta-correlated? Test by conditioning on BTC's realized direction in the
same cycle: if shorts edge holds when BTC is up vs down, the signal is robust."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"

def load_close_4h(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return pd.Series(dtype="float64")
    dfs=[pd.read_parquet(f, columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    if not dfs: return pd.Series(dtype="float64")
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    c=df.set_index("open_time")["close"].astype(np.float64)
    return c[(c.index.hour%4==0)&(c.index.minute==0)]

d = pd.read_parquet(PREDS)
d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]

# realized 4h BTC return per cycle (concurrent, not lookback)
btc = load_close_4h("BTCUSDT")
btc_ret_4h = btc.pct_change().shift(-1)   # forward 4h ret to match strategy horizon
btc_4h = btc_ret_4h.rename("btc_fwd_4h").reset_index()
btc_4h["open_time"] = pd.to_datetime(btc_4h["open_time"], utc=True)

H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))
K = [1,2,3,5]

# per-cycle: get bot-K return + cycle's BTC realized 4h
def per_cycle(g):
    if len(g)<2*max(K): return None
    g = g.sort_values("pred")
    out = dict(all_mean=g["return_pct"].mean(), n=len(g))
    for k in K:
        out[f"bot{k}_ret"] = g.head(k)["return_pct"].mean()
        out[f"bot{k}_edge"] = g["return_pct"].mean() - g.head(k)["return_pct"].mean()
        out[f"top{k}_ret"] = g.tail(k)["return_pct"].mean()
        out[f"top{k}_edge"] = g.tail(k)["return_pct"].mean() - g["return_pct"].mean()
    return pd.Series(out)

cyc = d.groupby("open_time").apply(per_cycle).dropna(how="all")
cyc.index = cyc.index.get_level_values(0); cyc = cyc.reset_index().rename(columns={"index":"open_time"})
cyc["open_time"] = pd.to_datetime(cyc["open_time"], utc=True)
cyc = cyc.merge(btc_4h, on="open_time", how="left")

for label,(s,e) in [("FULL OOS",(H1[0],H2[1])), ("H1",H1), ("H2",H2)]:
    sub = cyc[(cyc["open_time"]>=s)&(cyc["open_time"]<e)].dropna(subset=["btc_fwd_4h"])
    if len(sub)==0: continue
    print(f"\n=== {label} (n={len(sub)}) — conditional bot-K short edge ===")
    print(f"  market 4h ret distribution: mean {sub['btc_fwd_4h'].mean()*1e4:+.1f}  median {sub['btc_fwd_4h'].median()*1e4:+.1f}")
    up = sub[sub["btc_fwd_4h"]>0]; down = sub[sub["btc_fwd_4h"]<0]; flat = sub[sub["btc_fwd_4h"].abs()<0.001]
    print(f"  cycles: up={len(up)}  down={len(down)}  near-flat(<10bps)={len(flat)}")
    print(f"\n  {'K':>3}  {'bot-K edge all':>15}  {'bot-K edge UP':>15}  {'bot-K edge DOWN':>17}  {'top-K edge all':>15}  {'top-K edge UP':>15}  {'top-K edge DOWN':>17}")
    for k in K:
        bk = sub[f"bot{k}_edge"]
        bu = up[f"bot{k}_edge"]; bd = down[f"bot{k}_edge"]
        tk = sub[f"top{k}_edge"]
        tu = up[f"top{k}_edge"]; td = down[f"top{k}_edge"]
        print(f"  {k:>3}  {bk.mean()*1e4:+11.1f} bps  {bu.mean()*1e4:+11.1f} bps  {bd.mean()*1e4:+13.1f} bps  "
              f"{tk.mean()*1e4:+11.1f} bps  {tu.mean()*1e4:+11.1f} bps  {td.mean()*1e4:+13.1f} bps")
