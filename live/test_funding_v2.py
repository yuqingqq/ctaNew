"""#177 funding-cost realism on the v2 candidate (assembled equal-weight + stop-off-bear).
Reconstruct the held net positions (6-sleeve blend of the bot's actual selected legs) and charge realized funding:
funding 8h rate repeated on 2x 4h bars -> charge rate*0.5 per 4h bar. Sign: funding>0 => LONGS pay (PnL = -pos*rate).
Aggregate funding PnL by regime; compare to v2 gross PnL — does funding erode or help the bear edge?
"""
import pandas as pd, numpy as np, sys; sys.path.insert(0,'.')
import live.convexity_paper_bot as bot
RUN="live/state/opt_loop/aud_bearfix_sideeq/monthly/stateB"
K=3; HOLD=6
# regime
c4=bot.load_close_4h("BTCUSDT").sort_index(); btc=(c4/c4.shift(180)-1).dropna(); btc=btc[btc.index>="2025-10-04"]
reg=pd.Series(bot.apply_hysteresis([bot.regime_for_cycle(x) for x in btc.values],n=3),index=btc.index)
# per-cycle selected legs
pr=pd.read_parquet(f"{RUN}/predictions.parquet"); pr["open_time"]=pd.to_datetime(pr["open_time"],utc=True)
pr["w"]=(pr["selected_long"].astype(int)-pr["selected_short"].astype(int))/K   # entry basket weight per sym
times=sorted(pr["open_time"].unique())
# funding per (symbol, open_time)
fund=pd.read_parquet("outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","funding_rate"])
fund["open_time"]=pd.to_datetime(fund["open_time"],utc=True)
fmap=fund.dropna(subset=["funding_rate"]).set_index(["open_time","symbol"])["funding_rate"]
# sleeve net position per cycle (blend last 6 entry baskets), then funding PnL
from collections import deque
sleeves=deque(maxlen=HOLD); rows=[]
bycyc={t:g for t,g in pr.groupby("open_time")}
for t in times:
    g=bycyc[t]; basket={s:w for s,w in zip(g["symbol"],g["w"]) if w!=0}
    sleeves.append(basket)
    net={}
    for sl in sleeves:
        for s,w in sl.items(): net[s]=net.get(s,0)+w/HOLD
    # funding pnl this 4h bar = -sum pos*rate*0.5 (0.5 = 4h/8h); funding>0 -> long pays
    fp=0.0
    for s,pos in net.items():
        r=fmap.get((t,s),np.nan)
        if np.isfinite(r): fp+= -pos*r*0.5
    rows.append((t,fp*1e4))   # bps
F=pd.DataFrame(rows,columns=["open_time","fund_bps"]).set_index("open_time"); F["reg"]=F.index.map(reg)
# v2 gross PnL by regime
c=pd.read_csv(f"{RUN}/cycles.csv"); c["open_time"]=pd.to_datetime(c["open_time"],utc=True)
col="pnl_bps" if "pnl_bps" in c else [x for x in c.columns if "pnl" in x][0]
c["pb"]=c[col] if col=="pnl_bps" else c[col]*1e4; c["reg"]=c["open_time"].map(reg)
ann=np.sqrt(6*365)
print("FUNDING CARRY on v2 candidate (negative = cost, positive = carry gain), by regime:")
for r in ["bull","side","bear","ALL"]:
    fr=F if r=="ALL" else F[F.reg==r]; pr_=c if r=="ALL" else c[c.reg==r]
    net_after=pr_["pb"].sum()+fr["fund_bps"].sum()
    print(f"  {r:5s}: v2 PnL {pr_['pb'].sum():+7.0f}  funding {fr['fund_bps'].sum():+6.0f} ({fr['fund_bps'].mean():+.3f}/cyc)  -> net {net_after:+7.0f}")
# net Sharpe with funding
m=c.set_index("open_time")[["pb","reg"]].join(F["fund_bps"]).fillna({"fund_bps":0})
m["net"]=m["pb"]+m["fund_bps"]
print(f"\nv2 overall: gross Sharpe {c['pb'].mean()/c['pb'].std()*ann:+.3f} -> NET-of-funding Sharpe {m['net'].mean()/m['net'].std()*ann:+.3f}")
print(f"  totPnL gross {c['pb'].sum():+.0f} -> net {m['net'].sum():+.0f}  (production flat baseline +9657)")
