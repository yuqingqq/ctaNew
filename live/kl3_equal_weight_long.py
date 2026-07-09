"""KL3: K_long=3 equal-weight vs K_long=1 (RESEARCH_LOOP_20260707 addendum 20).

Book-level, overlays-off, naked, matched total long gross / short leg / costs. Turnover charged
honestly (ew-top3 basket turnover = |membership change|/3). Endpoints dual-era with day-block CI:
long-leg net mean, std, CVaR5%, top-decile (jackpot) contribution, paired mean-Δ CI. Reports
realized net beta of each long leg.
"""
import numpy as np, pandas as pd, glob
from pathlib import Path
D=Path("/home/yuqing/ctaNew/live/state/convexity"); rng=np.random.default_rng(20)
COST=9.0; WL=0.5; WS=0.5

def btc_ret():
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(glob.glob("/home/yuqing/ctaNew/data/ml/test/parquet/klines/BTCUSDT/5m/*.parquet"))]
    b=pd.concat(dfs).drop_duplicates("open_time").sort_values("open_time");b["open_time"]=pd.to_datetime(b["open_time"],utc=True)
    b=b.set_index("open_time")["close"];b4=b[(b.index.hour%4==0)&(b.index.minute==0)]
    return (b4.shift(-1)/b4-1)*1e4

def dayci(x,days,n=2000,seed=1):
    r=np.random.default_rng(seed)
    per=[g.values for _,g in pd.Series(x,index=days).groupby(level=0)]
    ms=[np.concatenate([per[i] for i in r.integers(0,len(per),len(per))]).mean() for _ in range(n)]
    return np.percentile(ms,[2.5,97.5])

def main():
    br=btc_ret()
    for win,(bk,lk) in (("REC",("hl_tgt_res_base_clean","hl_tgt_res_long_clean")),
                        ("OOS",("hl_v4base_oos_clean","hl_v4long_oos_clean"))):
        def L(x): d=pd.read_parquet(D/f"{x}/v0full_hl60.parquet",columns=["symbol","open_time","pred","alpha_A","return_pct"]);d["open_time"]=pd.to_datetime(d["open_time"],utc=True);d["a"]=d["alpha_A"]*1e4;d["r"]=d["return_pct"]*1e4;return d
        bb=L(bk);ll=L(lk);gl=ll.groupby("open_time")
        rows=[];prev1=None;prev3=set()
        for t,g in bb.groupby("open_time"):
            if len(g)<25 or t not in br.index or not np.isfinite(br[t]): continue
            try: glt=gl.get_group(t)
            except KeyError: continue
            top=glt.nlargest(3,"pred")
            if len(top)<3: continue
            r1=top.iloc[0]["a"]; ew3=top["a"].mean()          # long-leg outcome (residual alpha)
            rr1=top.iloc[0]["r"]; rew3=top["r"].mean()         # naked (for beta)
            s1=set([top.iloc[0]["symbol"]]); s3=set(top["symbol"])
            tc1=1.0 if (prev1 is None or top.iloc[0]["symbol"]!=prev1) else 0.0
            tc3=len(s3-prev3)/3.0 if prev3 else 1.0            # basket turnover = frac of names changed
            rows.append((t,r1,ew3,rr1,rew3,tc1,tc3,br[t]));prev1=top.iloc[0]["symbol"];prev3=s3
        df=pd.DataFrame(rows,columns=["t","r1","ew3","nr1","new3","tc1","tc3","btc"]);days=df.t.dt.date
        # net long-leg (gross alpha - turnover cost); jackpot = outcome>+500
        net1=df.r1 - COST*df.tc1; net3=df.ew3 - COST*df.tc3
        def cvar(x,q=5): x=np.sort(x); return x[:max(1,len(x)*q//100)].mean()
        def jkc(x): return x[x>=np.percentile(x,90)].sum()/len(x)   # top-decile contribution to mean
        b1=np.cov(df.nr1,df.btc)[0,1]/np.var(df.btc); b3=np.cov(df.new3,df.btc)[0,1]/np.var(df.btc)
        print(f"\n===== {win}: KL3 long-leg (net {COST}bps, matched gross) — beta1 {b1:.2f} ew3 {b3:.2f} =====")
        print(f"  top-1   : mean {net1.mean():+6.1f}  std {net1.std():6.0f}  CVaR5% {cvar(net1.values):+7.0f}  jkpot-contrib {jkc(df.r1.values):+.1f}  turnover {df.tc1.mean()*100:.0f}%")
        print(f"  ew-top3 : mean {net3.mean():+6.1f}  std {net3.std():6.0f}  CVaR5% {cvar(net3.values):+7.0f}  jkpot-contrib {jkc(df.ew3.values):+.1f}  turnover {df.tc3.mean()*100:.0f}%")
        dm=net3-net1; lo,hi=dayci(dm.values,days,seed=1)
        print(f"  paired mean-Δ (ew3−top1) {dm.mean():+.1f} bps/cyc  day-block CI [{lo:+.1f},{hi:+.1f}]  "
              f"{'preserves mean (CI crosses 0/positive)' if hi>=0 else 'SIG WORSE'}")
        print(f"  BARS: mean {'PASS' if hi>=0 else 'FAIL'} | var {'PASS' if net3.std()<net1.std() else 'FAIL'} "
              f"({net3.std()/net1.std()*100:.0f}%) | CVaR {'PASS' if cvar(net3.values)>cvar(net1.values) else 'FAIL'} | "
              f"jackpot {'PASS' if jkc(df.ew3.values)>=jkc(df.r1.values)*0.9 else 'FAIL'}")
    print("KL3DONE")

if __name__=="__main__":
    main()
