"""W4a — DECISIVE capacity check: does K=2's backtest edge survive depth-aware market impact?

The replay costs a flat 4.5 bps/leg (notional-proportional but DEPTH-BLIND). K=2 concentrates the same book gross
into 2 names/side vs K=3's 3 (1.5x per-name notional); square-root impact => total impact ~ 1/sqrt(K), so K=2 pays
~22% more impact at equal AUM. This overlay reprices K=2 vs K=3 NET of a PIT depth-aware impact model.

Model: reconstruct the net book per cycle from sleeves.csv (deque of last HOLD sleeves, net[s]=sum wt/HOLD).
Per cycle per symbol turnover_notional = |net[s]-prev[s]|*AUM. bar_ADV = trailing-30d mean DAILY $vol / 6 (4h bars).
impact_bps[s] = HALF_SPREAD + IMPACT_K * sqrt(turnover_notional / bar_ADV). cost_bps_cycle = sum_s turnover_notional*
impact_bps[s] / (AUM*book_gross). net_pnl = gross_pnl_bps - cost. IMPACT_K calibrated so ~$1M gives ~25 bps median
leg impact (matches the prior live-book capacity finding). Reports daily Sharpe across an AUM ladder + the crossover.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
REPO=Path("/home/yuqing/ctaNew"); OUT=REPO/"live/state/v3loop/opt2025"; ANN=np.sqrt(365); HOLD=6
HALF_SPREAD=2.0   # bps, half-spread floor per leg
IMPACT_K=float(sys.argv[1]) if len(sys.argv)>1 else 60.0   # bps at 100% bar-participation (sqrt model)
AUMS=[50e3,156e3,500e3,1e6,5e6]

dv=pd.read_parquet(OUT/"_dvol_pit.parquet"); dv["date"]=pd.to_datetime(dv["date"],utc=True)
# per-symbol date-indexed series for asof lookups
DVOL={s:g.set_index("date")["dvol30"].sort_index() for s,g in dv.groupby("symbol")}
def bar_adv(sym, ot):
    s=DVOL.get(sym)
    if s is None or not len(s): return np.nan
    v=s.asof(ot); return (float(v)/6.0) if v==v and v>0 else np.nan

def net_books(tag):
    sl=pd.read_csv(OUT/tag/"sleeves.csv"); sl=sl[sl.event=="enter"].copy()
    sl["open_time"]=pd.to_datetime(sl["open_time"],utc=True); sl=sl.sort_values("open_time")
    from collections import deque
    active=deque(maxlen=HOLD); books={}
    for _,r in sl.iterrows():
        w=json.loads(r["weights_json"]); active.append(w)
        net={}
        for sw in active:
            for s,wt in sw.items(): net[s]=net.get(s,0.0)+wt/HOLD
        books[r["open_time"]]=net
    return books

def capacity_curve(tag):
    books=net_books(tag)
    cyc=pd.read_csv(OUT/tag/"cycles.csv"); cyc["open_time"]=pd.to_datetime(cyc["open_time"],utc=True)
    cyc=cyc.sort_values("open_time").set_index("open_time")
    times=sorted(books.keys()); prev={}
    # per-cycle per-symbol turnover (book-weight units) + bar_adv
    turn_rows=[]
    for ot in times:
        net=books[ot]; allk=set(net)|set(prev)
        for s in allk:
            dt=net.get(s,0.0)-prev.get(s,0.0)
            if abs(dt)>1e-9: turn_rows.append((ot,s,abs(dt)))
        prev=net
    T=pd.DataFrame(turn_rows,columns=["open_time","symbol","turn_w"])
    T["adv"]=[bar_adv(s,ot) for ot,s in zip(T.open_time,T.symbol)]
    res={}
    for AUM in AUMS:
        T["turn_notional"]=T["turn_w"]*AUM
        part=T["turn_notional"]/T["adv"]
        T["impact_bps"]=HALF_SPREAD+IMPACT_K*np.sqrt(part.clip(lower=0).fillna(0))
        T.loc[T["adv"].isna(),"impact_bps"]=HALF_SPREAD+IMPACT_K*0.5  # missing depth => penalize moderately
        # cost in book-bps for the cycle: sum(turn_notional*impact)/(AUM*gross). gross ~ sum|net| ~ 2.0
        T["cost_contrib"]=T["turn_notional"]*T["impact_bps"]/1e4   # $ cost
        ccost=T.groupby("open_time")["cost_contrib"].sum()/AUM*1e4   # bps of AUM
        med_leg=float(T["impact_bps"].median())
        g=cyc["gross_pnl_bps"].copy() if "gross_pnl_bps" in cyc.columns else (cyc["pnl_bps"]+cyc.get("cost_bps",0))
        net_pnl=g.add(-ccost.reindex(g.index).fillna(0))
        dense=net_pnl.loc['2025-01-01':'2026-06-04']; y25=net_pnl.loc['2025-01-01':'2025-12-31']
        def dsh(x): d=(x.fillna(0)/1e4).resample("1D").sum(); return float(d.mean()/d.std()*ANN) if d.std()>0 else np.nan
        res[AUM]=(dsh(dense),dsh(y25),med_leg)
    return res

print(f"IMPACT_K={IMPACT_K} HALF_SPREAD={HALF_SPREAD}  (gross-pnl minus depth-aware impact; flat-cost replay shown as AUM->0)")
print(f"{'AUM':>8} | {'K=2 dense':>9} {'K=2 2025':>9} {'K=2 medLeg':>10} | {'K=3 dense':>9} {'K=3 2025':>9} {'K=3 medLeg':>10}")
c2=capacity_curve("ks2_kl2"); c3=capacity_curve("baseline")
for AUM in AUMS:
    d2,y2,m2=c2[AUM]; d3,y3,m3=c3[AUM]
    print(f"{AUM/1e3:>7.0f}k | {d2:+9.3f} {y2:+9.3f} {m2:>9.1f}b | {d3:+9.3f} {y3:+9.3f} {m3:>9.1f}b   {'<-- K3>K2' if d3>d2 else ''}")
print("\nNote: flat-cost replay (depth-blind) gave K2 dense +2.02 / K3 +1.33. Crossover AUM = where K=3 dense overtakes K=2.")
