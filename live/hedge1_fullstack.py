"""HEDGE1 full-stack-lite replay (RESEARCH_LOOP_20260707 addendum 19c).

Applies the DOMINANT KEEPSET4 overlays — bull-gross-0, the faithful VolNormStop DD-stop
(exact params k=2.0/g_floor=0.40/win=180/warmup60/heal0.5/timeout90), and GLOBAL_GROSS_MULT=0.5 —
to both long arms (A=alt-top1, B_beta=BTC beta-matched). Answers: does BTC-long's ~30% maxDD
advantage SURVIVE the DD-stop, or does the stop already capture it? (Regime-gate + inv_sqrt_vol
omitted — symmetric to both arms, affect absolute level not the A-vs-B DD delta.)
"""
import numpy as np, pandas as pd, glob
from collections import deque
from pathlib import Path
D=Path("/home/yuqing/ctaNew/live/state/convexity"); rng=np.random.default_rng(19)
COST=9.0; WL=0.5; WS=0.5; INIT=10000.0; GCAP=0.5

class VolNormStop:   # verbatim from convexity_paper_bot.py
    def __init__(self,k=2.0,g_floor=0.40,sigma_win=180,warmup=60,heal=0.5,timeout=90):
        self.k=k;self.g_floor=g_floor;self.sigma_win=sigma_win;self.warmup=warmup;self.heal=heal;self.timeout=timeout
        self.eq_hist=deque(maxlen=sigma_win+1);self.peak=INIT;self.engaged=False;self.engage_dd=0.0;self.engage_age=0;self.trough=INIT
    def update(self,eq_pre,eq_post,bar_idx):
        if len(self.eq_hist)>=self.warmup:
            inc=np.diff(np.array(self.eq_hist));sigma=float(np.std(inc)) if len(inc)>=2 else 0.0
        else: sigma=0.0
        thr=self.k*sigma*np.sqrt(self.sigma_win);self.peak=max(self.peak,eq_pre);dd=self.peak-eq_pre
        if not self.engaged:
            if bar_idx>=self.warmup and dd>=thr and thr>0:
                self.engaged=True;self.engage_dd=dd;self.engage_age=0;self.trough=eq_pre
            gm=1.0
        else:
            self.engage_age+=1;self.trough=min(self.trough,eq_pre)
            healed=(self.peak-eq_pre)<=self.engage_dd*self.heal;timed=self.engage_age>=self.timeout;above=eq_pre>self.trough
            gm=1.0 if ((healed or timed) and above) else self.g_floor
            if gm==1.0: self.engaged=False
        self.eq_hist.append(eq_post);return gm

def btc_series():
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(glob.glob("/home/yuqing/ctaNew/data/ml/test/parquet/klines/BTCUSDT/5m/*.parquet"))]
    b=pd.concat(dfs).drop_duplicates("open_time").sort_values("open_time");b["open_time"]=pd.to_datetime(b["open_time"],utc=True)
    b=b.set_index("open_time")["close"];b4=b[(b.index.hour%4==0)&(b.index.minute==0)]
    ret=(b4.shift(-1)/b4-1)*1e4;reg=(b4/b4.shift(180)-1)
    R={t:("bull" if v>0.10 else ("bear" if v<-0.10 else "side")) for t,v in reg.items() if np.isfinite(v)}
    return ret,R

def sharpe(x): x=np.array(x); return x.mean()/x.std(ddof=1)*np.sqrt(365) if x.std(ddof=1)>0 else np.nan
def maxdd_eq(eq): eq=np.array(eq); return float((eq-np.maximum.accumulate(eq)).min())

def replay(cyc_ret, regs):
    """apply bull-gross-0 + VolNormStop + 0.5x cap to a per-cycle bps return series -> equity path."""
    stop=VolNormStop(); eq=INIT; eqs=[INIT]; net=[]
    for i,(r,rg) in enumerate(zip(cyc_ret,regs)):
        bull0 = 0.0 if rg=="bull" else 1.0
        gm_stop = stop.update(eq, eq, i)         # PIT gross mult from stop (pre-bar)
        g = bull0 * gm_stop * GCAP
        pnl = g * r/1e4 * eq
        eq_post = eq + pnl
        stop.eq_hist[-1] = eq_post if stop.eq_hist else None
        net.append(pnl/INIT*1e4); eq=eq_post; eqs.append(eq)
    return np.array(net), np.array(eqs)

def main():
    btc_ret,R=btc_series()
    for win,(bk,lk) in (("REC",("hl_tgt_res_base_clean","hl_tgt_res_long_clean")),
                        ("OOS",("hl_v4base_oos_clean","hl_v4long_oos_clean"))):
        def L(x): d=pd.read_parquet(D/f"{x}/v0full_hl60.parquet",columns=["symbol","open_time","pred","return_pct"]);d["open_time"]=pd.to_datetime(d["open_time"],utc=True);d["r"]=d["return_pct"]*1e4;return d
        bb=L(bk);ll=L(lk);gl=ll.groupby("open_time");rows=[];prevL=None
        for t,g in bb.groupby("open_time"):
            if len(g)<25 or t not in btc_ret.index or not np.isfinite(btc_ret[t]): continue
            try: glt=gl.get_group(t)
            except KeyError: continue
            Lp=glt.nlargest(1,"pred").iloc[0];S=g.nsmallest(2,"pred")
            Lchg=1 if (prevL is None or Lp["symbol"]!=prevL) else 0
            rows.append((t,R.get(t,"unk"),Lp["r"],btc_ret[t],S["r"].mean(),Lchg));prevL=Lp["symbol"]
        df=pd.DataFrame(rows,columns=["t","reg","alt","btc","s","Lchg"])
        beta_alt=np.cov(df.alt,df.btc)[0,1]/np.var(df.btc)
        # vanilla per-cycle book return (bps) for each arm
        A_v  = WL*df.alt          - WS*df.s - WL*COST*df.Lchg
        Bb_v = WL*df.btc*beta_alt - WS*df.s - WL*beta_alt*COST*0.02
        print(f"\n===== {win}: FULL-STACK-LITE (bull0 + VolNormStop + 0.5x cap); alt beta {beta_alt:.2f} =====")
        for nm,v in (("A alt-top1",A_v),("B_beta BTC",Bb_v)):
            net,eqs=replay(v.values, df.reg.values)
            # vanilla (no overlays) for reference
            print(f"  {nm:14s}: FULLSTACK Sharpe {sharpe(net):+.2f} maxDD {maxdd_eq(eqs):+8.0f} | vanilla-ref Sharpe {sharpe(v.values):+.2f} maxDD {maxdd_eq(np.cumsum(v.values)):+8.0f}")
    print("HEDGE1FSDONE")

if __name__=="__main__":
    main()
