"""iter-020 pre-check — Absorption Ratio (Kritzman et al. 2010) as a market-fragility
LEADING signal for the HL70 held-book drawdown.

SOTA idea: the Absorption Ratio = fraction of total cross-sectional variance absorbed by the
top eigenvector(s) of the trailing universe return-correlation matrix. Published as a LEADING
indicator of equity drawdowns (Kritzman/Li/Page/Rigobon 2010). It measures the iter-006 DD
root cause directly: a correlated alt deleverage = the *market mode* (first eigenvalue) of the
correlation matrix swelling. This is structurally DIFFERENT from every prior rejected free
observable (price/vol/positioning LEVELS, iters 5/7/8/9/10) — it is a *correlation-STRUCTURE
concentration* measure, not a level.

This pre-check answers, cheaply, the only question that matters before building:
  (A) Does AR LEAD the HL70 book PnL at the 24h trade horizon (the test that killed iters 5/9/10)?
      i.e. is IC(AR_past, fwd_book_PnL) stronger than IC(AR_past, past_book_PnL)? Or is it
      coincident/lagging like everything else?
  (B) PRE-CHECK-G4/R4: does an AR-fragility de-gross cut the LEFT TAIL better than a matched
      random de-gross of equal %-time? (If only ~p<95, it's "run smaller", same i1/i9 wall.)

Outputs decide READY vs NO-CANDIDATE. No build unless AR clears both.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"
K=5; HOLD=6; COST=4.5e-4
import os
AR_WIN=int(os.environ.get("AR_WIN","180"))          # trailing 4h bars for the correlation matrix
HZN=6               # book trade horizon in 4h bars (24h)
rng=np.random.default_rng(0)


def load_close(sym):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def ann(x):
    x=pd.Series(x).dropna(); return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan


def main():
    t0=time.time()
    print("=== iter-020 Absorption-Ratio pre-check ===\n", flush=True)
    d=pd.read_parquet(PREDS, columns=["symbol","open_time","pred","return_pct"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy()
    syms=sorted(d["symbol"].unique())

    # ---- build baseline held-book (regime hybrid, K=5, 6 sleeves) ----
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    mom_rows=[]; beta_map={}; ret4_cols={}
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":c4.index,"mom30":(c4/c4.shift(180)-1).shift(1).values}))
        r=np.log(c4/c4.shift(1)); ret4_cols[sym]=r
        ri,bi=r.align(br,join="inner")
        beta_map[sym]=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    betas=pd.concat([s.rename(k) for k,s in beta_map.items()],axis=1)
    d=d.merge(mom,on=["symbol","open_time"],how="left")
    btc30=(b4/b4.shift(180)-1).to_frame("b30").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left").dropna(subset=["b30"])
    d["regime"]=np.where(d["b30"]>0.10,"bull",np.where(d["b30"]<-0.10,"bear","side"))
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}

    cyc_w=[]
    for ot in times:
        g=by_t[ot]; rg=g["regime"].iloc[0]
        if rg=="bear": cyc_w.append({}); continue
        key="mom30" if rg=="bull" else "pred"; gg=g.dropna(subset=[key])
        if len(gg)<2*K: cyc_w.append({}); continue
        gg=gg.sort_values(key); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
        a=b=1.0
        if rg=="side":
            brow=betas.loc[ot] if ot in betas.index else None
            if brow is not None:
                mbL=np.nanmean([brow.get(s,np.nan) for s in L]); mbS=np.nanmean([brow.get(s,np.nan) for s in S])
                if np.isfinite(mbL) and np.isfinite(mbS) and mbL>0 and mbS>0: a=2*mbS/(mbL+mbS); b=2*mbL/(mbL+mbS)
        w={}
        for s in L: w[s]=w.get(s,0)+a/K
        for s in S: w[s]=w.get(s,0)-b/K
        cyc_w.append(w)

    # net target weights per cycle (cost-independent), and a gross-scaler hook
    def heldbook(gross=None):
        # gross: optional dict-or-array of per-cycle gross multiplier in [0,1]; None=1
        prev={}; pnl=[]
        for t in range(len(cyc_w)):
            active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
            for w in active:
                for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
            gm=1.0 if gross is None else float(gross[t])
            net={s:wt*gm for s,wt in net.items()}
            alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
            rl=by_t[times[t]]; rmap=dict(zip(rl["symbol"],rl["return_pct"]))
            pnl.append(sum(net.get(s,0)*rmap.get(s,0.0) for s in net if np.isfinite(rmap.get(s,0.0)))-turn*0.5*COST); prev=net
        return pd.Series(pnl,index=pd.to_datetime(times))

    base=heldbook()
    pb=base*1e4; eq=pb.cumsum(); dd=(eq-eq.cummax())
    print(f"baseline: Sharpe {ann(base):+.2f}  totPnL {eq.iloc[-1]:+.0f}  maxDD {dd.min():+.0f}  "
          f"Calmar {pb.mean()*6*365/abs(dd.min()):+.2f}  ({len(base)} cycles)\n", flush=True)

    # ---- build PIT Absorption Ratio on the 4h universe returns ----
    R=pd.DataFrame(ret4_cols)                       # index=4h times, cols=syms
    R=R.reindex(pd.to_datetime(times))              # align to book cycles
    arr=R.values
    n=arr.shape[0]
    ar1=np.full(n,np.nan); ar_n5=np.full(n,np.nan)
    for t in range(n):
        lo=t-AR_WIN+1
        if lo<0: continue
        win=arr[lo:t+1]                              # uses data up to & including bar t
        # require columns with enough non-nan
        ok=np.isfinite(win).sum(0) >= int(AR_WIN*0.6)
        w=win[:,ok]
        if w.shape[1]<8: continue
        # fill remaining nans per-col with col mean (rare)
        cm=np.nanmean(w,0); idx=np.where(~np.isfinite(w))
        w[idx]=np.take(cm,idx[1])
        C=np.corrcoef(w,rowvar=False)
        C=np.nan_to_num(C, nan=0.0)
        ev=np.linalg.eigvalsh(C)                     # ascending
        ev=ev[::-1]; tot=ev.sum()
        if tot<=0: continue
        ar1[t]=ev[0]/tot
        ar_n5[t]=ev[:max(1,len(ev)//5)].sum()/tot     # Kritzman N≈n/5
    AR=pd.Series(ar1,index=pd.to_datetime(times),name="AR1")
    ARn=pd.Series(ar_n5,index=pd.to_datetime(times),name="AR_n5")
    # PIT lag: trigger at t uses AR through t-1
    AR_l=AR.shift(1); ARn_l=ARn.shift(1)
    # z-score / standardized shift over trailing 252 (Kritzman delta-AR), PIT
    AR_z=((AR - AR.rolling(252,min_periods=120).mean())/AR.rolling(252,min_periods=120).std()).shift(1)
    dAR=(AR - AR.shift(15)).shift(1)                 # 15-bar (~2.5d) change, lagged

    print(f"AR1 coverage {AR_l.notna().mean()*100:.0f}%  mean {AR.mean():.3f}  "
          f"range [{AR.min():.3f},{AR.max():.3f}]\n", flush=True)

    # ================= (A) LEAD-LAG TEST =================
    # forward book PnL over next HZN cycles (the trade horizon) vs past HZN
    fwd=base.shift(-1).rolling(HZN).sum().shift(-(HZN-1))   # sum of cycles t+1..t+HZN
    past=base.rolling(HZN).sum()                            # sum of cycles t-HZN+1..t
    fwd1=base.shift(-1)                                     # next single cycle
    def ic(a,b):
        m=pd.concat([a,b],axis=1).dropna()
        if len(m)<50: return np.nan,len(m)
        return m.iloc[:,0].corr(m.iloc[:,1],method="spearman"), len(m)
    print("=== (A) LEAD-LAG: Spearman IC of lagged AR feature vs book PnL ===")
    print(f"  {'feature':<10}{'vs PAST PnL':>13}{'vs FWD(24h)':>13}{'vs FWD(1cyc)':>14}   verdict")
    for nm,feat in [("AR1",AR_l),("AR_n5",ARn_l),("AR_z",AR_z),("dAR15",dAR)]:
        ip,_=ic(feat,past); iff,nf=ic(feat,fwd); i1,_=ic(feat,fwd1)
        # high AR should predict LOWER future PnL -> negative IC is the "leads DD" sign
        lead = (abs(iff)>abs(ip)+0.02) and (iff<0)
        print(f"  {nm:<10}{ip:>+13.3f}{iff:>+13.3f}{i1:>+14.3f}   "
              f"{'LEADS(neg)' if lead else 'coincident/none'}  n={nf}", flush=True)

    # ================= (B) PRE-CHECK-G4/R4 =================
    # AR-fragility de-gross: when AR (lagged) in top tercile -> de-gross book to g_floor.
    # Compare LEFT-TAIL cap vs matched random de-gross of equal %-time (200 seeds).
    print("\n=== (B) PRE-CHECK-G4/R4: AR-fragility de-gross vs matched random de-gross ===")
    g_floor=0.40
    for q in [0.70, 0.80]:
        thr=AR_l.quantile(q)
        flag=(AR_l>=thr).fillna(False).values            # de-gross when fragile
        pct=flag.mean()
        gross=np.where(flag,g_floor,1.0).astype(float)
        p_real=heldbook(gross); eqr=(p_real*1e4).cumsum(); ddr=(eqr-eqr.cummax()).min(); shr=ann(p_real)
        # matched random: de-gross the same NUMBER of cycles, random which
        kfire=flag.sum(); dds=[]
        for s in range(200):
            gi=np.ones(len(cyc_w)); pick=rng.choice(len(cyc_w),size=kfire,replace=False); gi[pick]=g_floor
            ps=heldbook(gi); es=(ps*1e4).cumsum(); dds.append((es-es.cummax()).min())
        dds=np.array(dds)
        # percentile: how many random de-gross have a WORSE (more negative) maxDD than real?
        # real is "good" if its maxDD is shallower (less negative) than random -> high rank
        rank=(dds<ddr).mean()*100   # frac of random with deeper DD than real => real better
        print(f"  AR-tercile q={q:.2f} (fire {pct*100:.0f}%, g_floor={g_floor}):  "
              f"Sharpe {shr:+.2f}  maxDD {ddr:+.0f} (base {dd.min():+.0f})  "
              f"random-degross maxDD mean {dds.mean():+.0f} [{dds.min():+.0f},{dds.max():+.0f}]  "
              f"real ranks p{rank:.0f}", flush=True)
    print("\n  R4/G4 read: p>=95 => AR de-gross caps the LEFT TAIL better than 'run smaller'."
          " p<95 => same wall as iters 1/9 (proportional, no edge).")

    print(f"\nDone [{time.time()-t0:.0f}s]")


if __name__=="__main__":
    main()
