"""HEDGE1: BTC-long vs alt-long hedge leg (RESEARCH_LOOP_20260707 addendum 19).

Book-level, overlays-off (estimator law), naked returns, fixed 0.5/0.5 gross, cost 9 bps/leg.
Short leg (bottom-2 base pred) common to all arms. Arms: A=alt top-1, B=BTC matched-notional,
B_beta=BTC beta-matched to A's realized net book-beta, HYB=BTC + alt-in-bear (diagnostic).
Reports net Sharpe, maxDD, paired day-block CI, regime split, realized net book-beta.
"""
import numpy as np, pandas as pd, glob
from pathlib import Path
D=Path("/home/yuqing/ctaNew/live/state/convexity"); rng=np.random.default_rng(19)
COST=9.0; WL=0.5; WS=0.5

def btc_series():
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(glob.glob("/home/yuqing/ctaNew/data/ml/test/parquet/klines/BTCUSDT/5m/*.parquet"))]
    b=pd.concat(dfs).drop_duplicates("open_time").sort_values("open_time"); b["open_time"]=pd.to_datetime(b["open_time"],utc=True)
    b=b.set_index("open_time")["close"]; b4=b[(b.index.hour%4==0)&(b.index.minute==0)]
    ret=(b4.shift(-1)/b4-1)*1e4; reg30=(b4/b4.shift(180)-1)
    R={t:("bull" if v>0.10 else ("bear" if v<-0.10 else "side")) for t,v in reg30.items() if np.isfinite(v)}
    return ret,R

def load(bk,lk):
    def L(x): d=pd.read_parquet(D/f"{x}/v0full_hl60.parquet",columns=["symbol","open_time","pred","return_pct"]); d["open_time"]=pd.to_datetime(d["open_time"],utc=True); d["r"]=d["return_pct"]*1e4; return d
    return L(bk),L(lk)

def sharpe(x): return x.mean()/x.std(ddof=1)*np.sqrt(365) if x.std(ddof=1)>0 else np.nan
def maxdd(x): eq=np.cumsum(x); return float((eq-np.maximum.accumulate(eq)).min())
def dayci(x,days,n=2000):
    per=[g.values for _,g in pd.Series(x,index=days).groupby(level=0)]
    ms=[np.concatenate([per[i] for i in rng.integers(0,len(per),len(per))]).mean() for _ in range(n)]
    return np.percentile(ms,[2.5,97.5])

def main():
    btc_ret,R=btc_series()
    for win,(bk,lk) in (("REC",("hl_tgt_res_base_clean","hl_tgt_res_long_clean")),
                        ("OOS",("hl_v4base_oos_clean","hl_v4long_oos_clean"))):
        bb,ll=load(bk,lk); gl=ll.groupby("open_time")
        rows=[]; prevL=None; prevS=set()
        for t,g in bb.groupby("open_time"):
            if len(g)<25 or t not in btc_ret.index or not np.isfinite(btc_ret[t]): continue
            try: glt=gl.get_group(t)
            except KeyError: continue
            L=glt.nlargest(1,"pred").iloc[0]; S=g.nsmallest(2,"pred")
            altret=L["r"]; btcr=btc_ret[t]; sret=S["r"].mean()   # short PnL = -sret
            Lchg=1 if (prevL is None or L["symbol"]!=prevL) else 0
            Sset=set(S["symbol"]); Schg=len(Sset-prevS)
            rows.append((t,R.get(t,"unk"),altret,btcr,sret,Lchg,Schg)); prevL=L["symbol"]; prevS=Sset
        df=pd.DataFrame(rows,columns=["t","reg","alt","btc","s","Lchg","Schg"]); days=df.t.dt.date
        scost=COST*df.Schg/2.0   # short-side turnover cost (2 legs, per-name)
        # realized alt-long beta to BTC (for beta-matching)
        beta_alt=np.cov(df.alt,df.btc)[0,1]/np.var(df.btc)
        # book net per cycle: WL*long - WS*short_avg - long_cost - short_cost
        def book(long_ret,long_turn_cost,lw=WL):
            return lw*long_ret - WS*df.s - long_turn_cost - WS*scost
        A   = book(df.alt, WL*COST*df.Lchg)
        B   = book(df.btc, WL*COST*0.02)
        Bb  = book(df.btc*beta_alt, WL*beta_alt*COST*0.02, lw=WL)   # beta-matched: scale BTC exposure
        hyb_long=np.where(df.reg=="bear", df.alt, df.btc)
        hyb_cost=np.where(df.reg=="bear", WL*COST*df.Lchg, WL*COST*0.02)
        HYB = WL*hyb_long - WS*df.s - hyb_cost - WS*scost
        def netbeta(bk_ret): return np.cov(bk_ret,df.btc)[0,1]/np.var(df.btc)
        print(f"\n===== {win}: BTC-long hedge test (book-level, overlays-off, net 9bps) — realized alt beta {beta_alt:.2f} =====")
        for nm,bkr in (("A alt-top1",A),("B btc-notional",B),("B_beta btc-matched",Bb),("HYB btc+alt-bear",HYB)):
            print(f"  {nm:20s} Sharpe {sharpe(bkr):+.2f}  mean {bkr.mean():+5.1f}  maxDD {maxdd(bkr):+7.0f}  netβ {netbeta(bkr):+.2f}")
        for arm,bkr in (("B−A",B-A),("Bβ−A",Bb-A),("HYB−A",HYB-A)):
            lo,hi=dayci(bkr.values,days)
            print(f"  paired Δ {arm:7s} {bkr.mean():+5.1f} bps/cyc CI[{lo:+.1f},{hi:+.1f}] {'>0' if lo>0 else ('<0' if hi<0 else 'crosses')}")
        print(f"  regime net (A/B/Bβ):", end=" ")
        for rg in ("side","bear","bull"):
            m=df.reg==rg
            if m.sum()<30: continue
            print(f"{rg} {A[m].mean():+.0f}/{B[m].mean():+.0f}/{Bb[m].mean():+.0f}", end="  ")
        print()
    print("HEDGE1DONE")

if __name__=="__main__":
    main()
