"""iter-002 — is the corr-DD mechanism visible in the MODEL's own within-cycle pred
spread, and does the realized cross-sectional return dispersion of the PICKS (not just
the throttle timing) explain the loss? If the long-short pred SPREAD (top-K minus bot-K
mean pred) collapses when corr is high, then a selection-confidence signal is the
construction-layer instrument -- and crucially it is per-cycle skill, not a magnitude knob.

We rebuild per-cycle: top-K vs bot-K mean pred (the selected spread), the realized
alpha spread of the picks, and the within-cycle pred std. Then check correlation with
corr7d and with next-cycle pnl, and whether gating on LOW selected-spread beats a
matched placebo any better than the corr throttle did.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np

REPO=Path("/home/yuqing/ctaNew")
RC=REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT=REPO/"research/convexity_portable_2026-05-20/results"
PREDS=RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
KLINES=REPO/"data/ml/test/parquet/klines"
K=5

def load_close(sym):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)
def ann(x):
    x=pd.Series(x).dropna(); return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan

d=pd.read_parquet(PREDS,columns=["symbol","open_time","pred","alpha_A"])
d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy()
btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
btc30=(b4/b4.shift(180)-1)
d=d.merge(btc30.to_frame("b30").reset_index().assign(open_time=lambda x:pd.to_datetime(x["open_time"],utc=True)),on="open_time",how="left").dropna(subset=["b30"])
d["regime"]=np.where(d["b30"]>0.10,"bull",np.where(d["b30"]<-0.10,"bear","side"))

ctx=pd.read_parquet(OUT/"iter002_hl70_context.parquet")

rows=[]
for ot,g in d.groupby("open_time"):
    rg=g["regime"].iloc[0]
    gg=g.dropna(subset=["pred"])
    if len(gg)<2*K:
        rows.append(dict(open_time=ot,regime=rg,pred_spread=np.nan,pred_std=np.nan)); continue
    gs=gg.sort_values("pred")
    top=gs.tail(K)["pred"].mean(); bot=gs.head(K)["pred"].mean()
    rows.append(dict(open_time=ot,regime=rg,pred_spread=top-bot,pred_std=gg["pred"].std()))
ps=pd.DataFrame(rows).set_index("open_time")
m=ps.join(ctx[["pnl","corr7d","regime"]].rename(columns={"regime":"rg2"}),how="inner")

print("=== within-cycle selected pred spread (top-K mean - bot-K mean) ===")
sidem=m[m["regime"]=="side"].dropna(subset=["pred_spread","corr7d"])
print(f"corr(pred_spread, corr7d) in side: {sidem['pred_spread'].corr(sidem['corr7d']):+.3f}")
print(f"corr(pred_spread, next pnl) in side: {sidem['pred_spread'].corr(sidem['pnl']):+.3f}")
print(f"corr(pred_std, corr7d) in side: {sidem['pred_std'].corr(sidem['corr7d']):+.3f}\n")

# does LOW pred_spread (model not differentiated) predict loss?
sidem=sidem.copy(); sidem["q"]=pd.qcut(sidem["pred_spread"],5,labels=False,duplicates="drop")
print("side-regime book PnL by pred_spread quintile (q0=narrowest spread):")
for q,gg in sidem.groupby("q"):
    print(f"   q{q}: spread {gg['pred_spread'].mean():.3f}  pnl {gg['pnl'].mean():+6.1f} bps  Sh {ann(gg['pnl']/1e4):+.2f}  n={len(gg)}")

# is pred_spread (a per-cycle model-confidence quantity) PIT? It uses only current-cycle preds
# (which use trailing features) -> it is point-in-time at decision time. Good.
# Quick: does it ADD over corr7d? 2D
print("\n2D side-regime PnL by (corr7d tercile, pred_spread tercile):")
sm=sidem.dropna(subset=["corr7d","pred_spread"]).copy()
sm["cq"]=pd.qcut(sm["corr7d"],3,labels=["lo","mid","hi"],duplicates="drop")
sm["sq"]=pd.qcut(sm["pred_spread"],3,labels=["narrow","mid","wide"],duplicates="drop")
print(sm.pivot_table(index="cq",columns="sq",values="pnl",aggfunc="mean").round(1).to_string())
