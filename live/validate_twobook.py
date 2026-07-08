"""Validate the two-book design empirically (v4 residual-target books). Use EACH book to rank BOTH legs and measure
forward 24h residual alpha. Tests the claims:
  (1) base and residrev preds actually differ (correlation).
  (2) residrev (V0_LEAN+resid_rev) improves the LONG leg vs base (higher fwd on top-K longs).
  (3) residrev does NOT help (or hurts) the SHORT leg vs base (short wants LOW fwd on bottom-K).
If (2)&(3) hold, the split earns its keep: each leg wants a different feature set.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
b=pd.read_parquet(f"{R}/live/state/convexity/hl_tgt_res_base/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"base"})
l=pd.read_parquet(f"{R}/live/state/convexity/hl_tgt_res_long/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"rr"})
for x in (b,l): x["open_time"]=pd.to_datetime(x["open_time"],utc=True)
d=b.merge(l,on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])
print(f"rows {len(d)}, cycles {d.open_time.nunique()}")
# (1) how different are the two book preds?
print(f"\n(1) corr(base pred, residrev pred): spearman {d['base'].corr(d['rr'],method='spearman'):.3f}  (1.0=identical books)")
print(f"    per-cycle mean |rank diff| top/bottom overlap check below")
# (2)&(3) each book ranking each leg, forward alpha (bps)
def leg(col,which,K):
    vals=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<2*K: continue
        pick = g.nlargest(K,col) if which=="long" else g.nsmallest(K,col)
        vals.append(pick["fwd"].mean())
    return np.nanmean(vals)
print(f"\n{'K':>2s} | {'LONG leg fwd (higher=better)':32s} | {'SHORT leg fwd (lower=better short)':34s}")
print(f"   | {'base':>10s} {'residrev':>10s} {'Δrr-base':>9s} | {'base':>10s} {'residrev':>10s} {'Δrr-base':>9s}")
for K in [1,2,3]:
    lb,lr=leg("base","long",K),leg("rr","long",K)
    sb,sr=leg("base","short",K),leg("rr","short",K)
    print(f"  {K} | {lb:+10.1f} {lr:+10.1f} {lr-lb:+9.1f} | {sb:+10.1f} {sr:+10.1f} {sr-sb:+9.1f}")
print("\nInterpretation: LONG wants residrev>base (Δ>0). SHORT wants LOW fwd; residrev HELPS short only if Δ<0 (more negative).")
print("VTBDONE")
