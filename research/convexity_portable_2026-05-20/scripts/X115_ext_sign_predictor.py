"""X115 — Per-symbol LEADING sign predictor on the EXTENDED 2021-2026 panel (task #109 payoff).

X111 failed on the 44-sym/2023+ panel because the efficacy never flipped in training (1 event).
X114 shows the extended 23-sym 2021-2026 panel has MANY flips (6 quarterly). Now retrain the
walk-forward classifier — it sees both + and - regimes (2021 bull, 2022 bear, etc.) — and test
whether a LEADING model flip can finally beat the LAGGING ic120 flip OOS.

Label  y[i,t]=1 if forward-90 per-sym IC(pred,alpha)>0. Feats (PIT): trailing ic30/ic90,
return_1d, ret_3d, atr_pct, funding_rate, funding_z, idio_vol_1d, beta_chg, corr_btc,
rvol_7d, btc_rvol_7d, btc_ret_30d. Walk-forward 6-mo blocks from 2023-01 (training incl 2021-22).
Eval: hit-rate (model/lagging/oracle) + held-book hybrid (base/lagging/model) per-year, esp 2026.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np
import lightgbm as lgb

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
PREDS = RC/"x113_ext_v0_preds.parquet"
PANEL = REPO/"outputs/vBTC_features/panel_ext2021_v0.parquet"
KLINES = REPO/"data/ml/test/parquet/klines"
COST=4.5e-4; K=5; HOLD=6; W=90


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
    print("=== X115 extended-panel per-symbol leading sign predictor ===\n", flush=True)
    p=pd.read_parquet(PREDS, columns=["symbol","open_time","pred","alpha_A","return_pct"])
    p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
    p=p[(p["open_time"].dt.hour%4==0)&(p["open_time"].dt.minute==0)]
    fc=["return_1d","ret_3d","atr_pct","funding_rate","funding_rate_z_7d","idio_vol_to_btc_1d",
        "beta_to_btc_change_5d","corr_to_btc_1d","rvol_7d","btc_rvol_7d"]
    feats=pd.read_parquet(PANEL, columns=["symbol","open_time"]+fc)
    feats["open_time"]=pd.to_datetime(feats["open_time"],utc=True)
    d=p.merge(feats,on=["symbol","open_time"],how="left").sort_values(["symbol","open_time"])
    # btc_ret_30d
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left")

    d["ic_nt"]=d.groupby("symbol",group_keys=False).apply(lambda g:g["pred"].rolling(W,min_periods=W//2).corr(g["alpha_A"]))
    d["f_ic30"]=d.groupby("symbol",group_keys=False).apply(lambda g:g["pred"].rolling(30,min_periods=15).corr(g["alpha_A"]).shift(HOLD))
    d["f_ic90"]=d.groupby("symbol")["ic_nt"].shift(HOLD)
    d["y"]=(d.groupby("symbol")["ic_nt"].shift(-W)>0).astype(float)
    d["fwd_sign"]=np.sign(d.groupby("symbol")["ic_nt"].shift(-W))
    FEATS=["f_ic30","f_ic90"]+fc+["btc_ret_30d"]

    blocks=[("2023-01-01","2023-07-01"),("2023-07-01","2024-01-01"),("2024-01-01","2024-07-01"),
            ("2024-07-01","2025-01-01"),("2025-01-01","2025-07-01"),("2025-07-01","2026-01-01"),
            ("2026-01-01","2026-07-01")]
    d["shat"]=np.nan
    for lo,hi in blocks:
        lo=pd.Timestamp(lo,tz="UTC"); hi=pd.Timestamp(hi,tz="UTC")
        tr=d[d["open_time"]<lo].dropna(subset=FEATS+["y"]); te=d[(d["open_time"]>=lo)&(d["open_time"]<hi)].dropna(subset=FEATS)
        if len(tr)<5000 or len(te)==0: continue
        mdl=lgb.LGBMClassifier(n_estimators=200,num_leaves=31,learning_rate=0.03,subsample=0.8,
                               colsample_bytree=0.8,min_child_samples=200,verbosity=-1)
        mdl.fit(tr[FEATS],tr["y"]); d.loc[te.index,"shat"]=np.where(mdl.predict_proba(te[FEATS])[:,1]>=0.5,1.0,-1.0)
    imp=pd.Series(mdl.feature_importances_,index=FEATS).sort_values(ascending=False)

    ev=d.dropna(subset=["shat","fwd_sign"]); ev=ev[ev["fwd_sign"]!=0]
    el=d.dropna(subset=["f_ic90","fwd_sign"]).query("fwd_sign!=0")
    h_m=(ev["shat"]==ev["fwd_sign"]).mean()*100; h_l=(np.sign(el["f_ic90"])==el["fwd_sign"]).mean()*100
    ev26=ev[ev["open_time"]>=pd.Timestamp("2026-01-01",tz="UTC")]
    print(f"hit-rate vs realized fwd sign: MODEL {h_m:.1f}% (2026 {(ev26['shat']==ev26['fwd_sign']).mean()*100:.1f}%)  |  lagging ic90 {h_l:.1f}%")
    print(f"top feats: " + ", ".join(f"{k}={v}" for k,v in imp.head(6).items()))

    # held-book hybrid (mom-bull/mean-rev-side-BN/flat-bear, K=5) base/lagging/model
    print("\nmom+beta...", flush=True)
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    syms=sorted(d["symbol"].unique()); mom_rows=[]; beta_map={}
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":c4.index,"mom30":(c4/c4.shift(180)-1).shift(1).values}))
        r=np.log(c4/c4.shift(1)); ri,bi=r.align(br,join="inner")
        beta_map[sym]=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    betas=pd.concat([s.rename(k) for k,s in beta_map.items()],axis=1)
    d=d.merge(mom,on=["symbol","open_time"],how="left")
    d["regime"]=np.where(d["btc_ret_30d"]>0.10,"bull",np.where(d["btc_ret_30d"]<-0.10,"bear","side"))
    d["ic120"]=d.groupby("symbol",group_keys=False).apply(lambda g:g["pred"].rolling(120,min_periods=60).corr(g["alpha_A"]).shift(HOLD))
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}

    def hb(mode):
        ws=[]
        for ot in times:
            g=by_t[ot]; rg=g["regime"].iloc[0]
            if rg=="bear": ws.append({}); continue
            key="mom30" if rg=="bull" else "pred"; gg=g.dropna(subset=[key]).copy()
            if rg=="side" and mode!="base":
                fs=np.where(gg["ic120"].fillna(0.0)<0,-1.0,1.0) if mode=="lag" else np.where(gg["shat"].fillna(1.0)<0,-1.0,1.0)
                gg["score"]=gg["pred"]*fs; sc="score"
            else: sc=key
            if len(gg)<2*K: ws.append({}); continue
            gg=gg.sort_values(sc); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
            a=b=1.0
            if rg=="side":
                brow=betas.loc[ot] if ot in betas.index else None
                if brow is not None:
                    mbL=np.nanmean([brow.get(s,np.nan) for s in L]); mbS=np.nanmean([brow.get(s,np.nan) for s in S])
                    if np.isfinite(mbL) and np.isfinite(mbS) and mbL>0 and mbS>0: a=2*mbS/(mbL+mbS); b=2*mbL/(mbL+mbS)
            w={}
            for s in L: w[s]=w.get(s,0)+a/K
            for s in S: w[s]=w.get(s,0)-b/K
            ws.append(w)
        prev={}; pnl=[]
        for t in range(len(ws)):
            active=ws[max(0,t-HOLD+1):t+1]; net={}
            for w in active:
                for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
            alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
            rl=by_t[times[t]]; rmap=dict(zip(rl["symbol"],rl["return_pct"]))
            pnl.append(sum(net.get(s,0)*rmap.get(s,0.0) for s in net if np.isfinite(rmap.get(s,0.0)))-turn*0.5*COST); prev=net
        return pd.Series(pnl,index=pd.to_datetime(times))

    def row(name,p):
        pb=p*1e4; dd=(pb.cumsum()-pb.cumsum().cummax()).min(); yr={y:ann(g/1e4) for y,g in pb.groupby(pb.index.year)}
        s=" ".join(f"{y}:{yr.get(y,float('nan')):+.2f}" for y in [2023,2024,2025,2026])
        print(f"  {name:<20}{ann(p):>+7.2f}{dd:>+9.0f}   {s}", flush=True)
    print(f"\n  {'variant':<20}{'Sharpe':>7}{'maxDD':>9}   per-year (predictor active 2023+)")
    row("base", hb("base")); row("lagging ic120 flip", hb("lag")); row("MODEL flip", hb("model"))
    print(f"\nPASS iff MODEL beats lagging on hit-rate AND 2026 (leading predictor now learnable w/ multi-regime). Done [{time.time()-t0:.0f}s]")


if __name__=="__main__":
    main()
