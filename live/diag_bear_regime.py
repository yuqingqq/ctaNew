"""Is flat-in-bear optimal? Count regime cycles in OOS + counterfactual: what would the K=3 mean-rev L/S have made
in BEAR cycles (currently FLAT). Uses production preds. Regime = btc_ret_30d with N=3 entry hysteresis (bull>+.10,
bear<-.10, else side)."""
import sys; from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); sys.path.insert(0,str(REPO))
import live.convexity_paper_bot as bot
ann=np.sqrt(6*365)
# btc_ret_30d per cycle = BTC 4h close / close 180 bars (30d) ago - 1  (the bot's regime input)
c=bot.load_close_4h("BTCUSDT").sort_index()
btc=(c/c.shift(180)-1).dropna()
btc=btc[btc.index>="2025-10-04"]
raw=[bot.regime_for_cycle(x) for x in btc.values]
eff=bot.apply_hysteresis(raw, n=3)
reg=pd.Series(eff,index=btc.index)
print("OOS 2025-10-04 -> ",btc.index.max().date())
print("regime cycle counts:", reg.value_counts().to_dict())
print(f"current btc_ret_30d (latest panel) = {btc.iloc[-1]:+.3f}  -> regime {eff[-1]}")
# counterfactual bear PnL: K=3 L/S raw return per bear cycle (long top-3 pred_long, short bottom-3 pred_short)
L=pd.read_parquet(REPO/"live/state/convexity/hl_residrev/v0full_hl60.parquet"); L["open_time"]=pd.to_datetime(L["open_time"],utc=True)
S=pd.read_parquet(REPO/"live/state/convexity/hl/v0full_hl60.parquet"); S["open_time"]=pd.to_datetime(S["open_time"],utc=True)
def leg_ret(d,t,k,largest):
    x=d[d.open_time==t].dropna(subset=["pred"])
    if len(x)<k: return np.nan
    sel=x.nlargest(k,"pred") if largest else x.nsmallest(k,"pred")
    return sel["return_pct"].mean()
rows=[]
for t in btc.index:
    r=reg.loc[t]
    lr=leg_ret(L,t,3,True); sr=leg_ret(S,t,3,False)
    if np.isfinite(lr) and np.isfinite(sr): rows.append((t,r,(lr-sr)*1e4))  # L-S bps, beta-neut approx equal-wt
df=pd.DataFrame(rows,columns=["t","reg","ls_bps"])
print("\ncounterfactual equal-wt K=3 L/S RAW return by regime (bps/cycle, gross):")
for r in ["bull","side","bear"]:
    g=df[df.reg==r]
    if len(g): print(f"  {r:5s}: n={len(g):4d}  mean {g.ls_bps.mean():+.1f} bps  Sharpe {g.ls_bps.mean()/g.ls_bps.std()*ann:+.2f}  totPnL {g.ls_bps.sum():+.0f}")
# bear sub-analysis: long-only, short-only
b=df[df.reg=="bear"]
if len(b):
    bl=[leg_ret(L,t,3,True) for t in b.t]; bs=[leg_ret(S,t,3,False) for t in b.t]
    bl=np.array(bl)*1e4; bs=np.array(bs)*1e4
    print(f"\nBEAR leg detail: long3 mean {np.nanmean(bl):+.1f} bps  short3 mean {np.nanmean(bs):+.1f} bps  (short edge = {-np.nanmean(bs):+.1f})")
