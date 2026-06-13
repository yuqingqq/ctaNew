"""Capacity-tiered performance: bucket traded symbols by HL orderbook impact, then for each tier measure how much
the strategy trades it, its paper PnL contribution, and its REALISTIC (impact-adjusted) PnL. Answers: which capacity
band carries the paper edge vs the REALIZABLE edge at size?
"""
import sys; import numpy as np, pandas as pd
sys.path.insert(0,"/home/yuqing/ctaNew")
ROOT="/home/yuqing/ctaNew"; NOT=[10e3,50e3,100e3,250e3,500e3]
cap=pd.read_csv(f"{ROOT}/live/state/v3loop/capacity_hl.csv")
P=pd.read_parquet(f"{ROOT}/live/state/v3loop/iter5_tilt0/predictions.parquet"); P["open_time"]=pd.to_datetime(P["open_time"],utc=True)
pan=pd.read_parquet(f"{ROOT}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","return_pct"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True)
P=P.merge(pan,on=["symbol","open_time"],how="left")
# per-pick signed gross contribution (equal-attribution proxy): long->+ret, short->-ret
P["contrib_bps"]=(np.where(P["selected_long"],P["return_pct"],0.0)+np.where(P["selected_short"],-P["return_pct"],0.0))*1e4
P["is_pick"]=P["selected_long"]|P["selected_short"]
pk=P[P["is_pick"]].copy()
# per-symbol rollup
sym=pk.groupby("symbol").agg(n_picks=("is_pick","sum"), gross_bps=("contrib_bps","sum")).reset_index()
sym=sym.merge(cap[["sym","book_depth_usd","spread_bps","imp_50k","imp_100k"]].rename(columns={"sym":"symbol"}),on="symbol",how="left")
sym["imp_50k"]=sym["imp_50k"].fillna(sym["imp_50k"].median())
# realistic net per pick at $1M AUM (per-name trade ~$56k -> use imp_50k one-way; RT ~ 2x as taker enter+exit, sleeve-amortized so ~1.5x)
RT=1.5
sym["net_per_pick"]=sym["gross_bps"]/sym["n_picks"] - RT*sym["imp_50k"]
sym["realistic_bps"]=sym["net_per_pick"]*sym["n_picks"]

BINS=[0,10,20,40,80,1e9]; LBL=["<10 (deep)","10-20","20-40","40-80",">80 (thin)"]
sym["tier"]=pd.cut(sym["imp_50k"],BINS,labels=LBL)
print(f"{'tier(imp@50k)':14s} {'#sym':>4} {'#picks':>7} {'medImp':>7} {'medDepth$M':>10} {'grossPnL':>9} {'g/pick':>7} {'realPnL':>9} {'r/pick':>7}")
tot_g=tot_r=0
for t in LBL:
    g=sym[sym["tier"]==t]
    if not len(g): continue
    gp=g["gross_bps"].sum(); rp=g["realistic_bps"].sum(); np_=g["n_picks"].sum()
    tot_g+=gp; tot_r+=rp
    print(f"{t:14s} {len(g):4d} {int(np_):7d} {g['imp_50k'].median():7.1f} {g['book_depth_usd'].median()/1e6:10.2f} {gp:9.0f} {gp/max(np_,1):7.1f} {rp:9.0f} {g['net_per_pick'].mean():7.1f}")
print(f"{'TOTAL':14s} {len(sym):4d} {int(sym['n_picks'].sum()):7d} {'':7s} {'':10s} {tot_g:9.0f} {'':7s} {tot_r:9.0f}")
print(f"\n=> paper gross {tot_g:.0f} bps; realistic (RT={RT}x taker impact@$1M) {tot_r:.0f} bps. Which tiers are net-POSITIVE realizable?")
for t in LBL:
    g=sym[sym["tier"]==t]
    if len(g): print(f"   {t:14s}: realPnL {g['realistic_bps'].sum():+8.0f}  ({'KEEP' if g['realistic_bps'].sum()>0 else 'DRAG'})")
# sample symbols per tier
print("\nsample symbols per tier:")
for t in LBL:
    g=sym[sym["tier"]==t].nlargest(6,"gross_bps")
    print(f"   {t:14s}: {[(r['symbol'].replace('USDT',''),int(r['gross_bps']),round(r['imp_50k'])) for _,r in g.iterrows()]}")
print("DONE capacity_tiers")
