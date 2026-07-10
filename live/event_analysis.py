"""EVENT FORENSICS: the edge is event-driven — identify the actual events, quantify them, characterize
the market state during them (user idea, 2026-07-10). NOT prediction (that fails, R²=0.005) — descriptive:
what KIND of event pays, so we recognize a similar setup. Per-day 1L/2S book net (clean, non-deep-bull),
top-day concentration, and for the top events: BTC 30d/day-move/realized-vol, cross-sectional dispersion,
the symbols that drove the PnL. Then: is the winning-day SETUP distinctive vs ordinary days?
"""
import numpy as np, pandas as pd, glob
from pathlib import Path
import sys; sys.path.insert(0,"live")
from attribution_v4_regime import btc_reg, load, COST
import warnings; warnings.filterwarnings("ignore")
WL=WS=0.5; KD=Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines")

def btc_ctx():
    fs=sorted(glob.glob(str(KD/"BTCUSDT"/"5m"/"*.parquet")))
    b=pd.concat([pd.read_parquet(f,columns=["open_time","close"]) for f in fs],ignore_index=True)
    b["open_time"]=pd.to_datetime(b["open_time"],utc=True); b=b.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")["close"]
    d=b.resample("1D").last(); ret=d.pct_change()
    out=pd.DataFrame({"btc_day":ret*100,"btc_30d":(d/d.shift(30)-1)*100,"btc_rvol7":ret.rolling(7).std()*100}).reset_index()
    out.columns=["day"]+list(out.columns[1:]); out["day"]=pd.to_datetime(out["day"],utc=True).dt.normalize()
    return out

def build(base, long, reg):
    lg=long.groupby("open_time"); rows=[]; prevL=None; prevS=set()
    for t,g in base.groupby("open_time"):
        if len(g)<5 or reg.get(t) in (None,"deepbull"): continue
        try: gl=lg.get_group(t)
        except KeyError: continue
        Lp=gl.nlargest(1,"pred"); S=g.nsmallest(2,"pred")
        if len(Lp)<1 or len(S)<2: continue
        la=Lp.iloc[0]["alpha_A"]*1e4; sa=S["alpha_A"].mean()*1e4
        disp=float(g["return_pct"].std()*1e4)              # realized cross-sectional dispersion (bps)
        Ln,Ss=Lp.iloc[0]["symbol"],set(S["symbol"])
        lt=1.0 if (prevL is None or Ln!=prevL) else 0.0; st=(len(Ss-prevS)/2.0) if prevS else 1.0
        net=WL*la-WS*sa-lt*0.5*WL*COST/0.5-st*0.5*WS*COST/0.5
        # which names drove it: short winners (most-negative alpha = best short PnL), long winner
        sw=", ".join(f"{r['symbol']}({-r['alpha_A']*1e4:+.0f})" for _,r in S.nsmallest(2,"alpha_A").iterrows())
        rows.append((t,net,disp,Ln,la,sw)); prevL,prevS=Ln,Ss
    return pd.DataFrame(rows,columns=["t","net","disp","longsym","longpnl","shortwin"])

def main():
    reg=btc_reg(); ctx=btc_ctx()
    for era,bp,lp in (("RECENT","hl_tgt_res_base_cleanfix","hl_tgt_res_long_cleanfix"),
                     ("OOS","hl_v4base_oos_cleanfix","hl_v4long_oos_cleanfix")):
        base,long=load(bp,lp); d=build(base,long,reg)
        d["day"]=pd.to_datetime(d["t"]).dt.normalize()
        day=d.groupby("day").agg(net=("net","sum"),disp=("disp","mean"),ncyc=("net","size")).reset_index()
        day=day.merge(ctx,on="day",how="left").sort_values("net",ascending=False)
        tot=day.net.sum(); topN=day.head(8)
        print(f"\n{'='*92}\n{era}: {len(day)} trading days, total net {tot:+.0f} bps | top-8 days = {topN.net.sum()/tot*100 if tot!=0 else float('nan'):.0f}% of total")
        print(f"  {'date':<12}{'net':>7} | {'xs_disp':>7} {'BTC_day':>8} {'BTC_30d':>8} {'BTC_rvol7':>9} | drivers")
        for _,r in topN.iterrows():
            ex=d[d.day==r.day]; sw=ex.loc[ex.net.idxmax(),"shortwin"] if len(ex) else ""
            print(f"  {str(r.day.date()):<12}{r.net:>+7.0f} | {r.disp:>7.0f} {r.btc_day:>+7.1f}% {r.btc_30d:>+7.1f}% {r.btc_rvol7:>8.2f}% | {sw}")
        # SETUP distinctiveness: winning-day median vs ordinary-day median
        win=day.head(max(5,len(day)//10)); ordn=day.tail(len(day)-len(win))
        print(f"  SETUP (top-decile winning days vs rest): xs_disp {win.disp.median():.0f} vs {ordn.disp.median():.0f} | "
              f"BTC|day| {win.btc_day.abs().median():.1f}% vs {ordn.btc_day.abs().median():.1f}% | "
              f"BTC_rvol7 {win.btc_rvol7.median():.2f}% vs {ordn.btc_rvol7.median():.2f}%")
    print("EVENTDONE")

if __name__=="__main__":
    main()
