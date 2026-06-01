"""iter-019 PRE-CHECK — meta-labeling feasibility on the HL70 held-book.

Mechanism question (the "can't squeeze the same orange twice" test from Baldisserri):
Does a SECONDARY model have access to information the PRIMARY (per-symbol Ridge `pred`)
does NOT, that predicts whether a SELECTED LEG realizes a profitable trade — AT THE 4h
TRADE HORIZON, conditional on `pred`?

Primary blind spots (candidate meta-features, all PIT/cross-sectional at selection time):
  - pred_rank_gap : leg's pred minus the K-cutoff pred (decisiveness of selection)
  - xs_pred_disp  : cross-sectional std of pred that cycle (signal breadth/agreement)
  - xs_pred_skew  : cross-sectional skew of pred (one-sided crowding)
  - n_elig        : number of eligible names (competition)
  - is_long       : side flag (known asymmetry: long 47.8% vs short 57.4% correct)
  - abs_pred      : |pred| (magnitude/confidence, primary HAS this but worth a baseline)

Per-leg label: leg_pnl = side_sign * leg_alpha_residual (alpha_A), i.e. realized signed
alpha contribution of that leg over the next 4h (the thing the book actually earns).

PRE-CHECK passes if (a) at least one meta-feature has |conditional IC| on leg_pnl that
is non-trivial AFTER residualizing out pred (the orange isn't already squeezed), AND
(b) it is not just the long/short asymmetry (which is structural, not skill).
If everything reduces to pred or to the side flag, meta-labeling is dead before building.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
PREDS = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
K=5

def load_close(sym):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)

def main():
    t0=time.time()
    print("=== iter-019 meta-label PRE-CHECK (HL70) ===\n",flush=True)
    d=pd.read_parquet(PREDS,columns=["symbol","open_time","pred","alpha_A","return_pct"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy()

    # regime (BTC trailing-30d), to restrict to side+bull (book is flat in bear)
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("b30").reset_index()
    btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left").dropna(subset=["b30"])
    d["regime"]=np.where(d["b30"]>0.10,"bull",np.where(d["b30"]<-0.10,"bear","side"))

    # mom30 for bull ranking
    rows=[]
    for sym in sorted(d["symbol"].unique()):
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        rows.append(pd.DataFrame({"symbol":sym,"open_time":c4.index,
                                  "mom30":(c4/c4.shift(180)-1).shift(1).values}))
    mom=pd.concat(rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    d=d.merge(mom,on=["symbol","open_time"],how="left")

    # Build per-LEG records as the engine does
    legs=[]
    for ot,g in d.groupby("open_time"):
        rg=g["regime"].iloc[0]
        if rg=="bear": continue
        key="mom30" if rg=="bull" else "pred"
        gg=g.dropna(subset=[key,"alpha_A","pred"])
        if len(gg)<2*K: continue
        gg=gg.sort_values(key)
        disp=gg["pred"].std(); skew=stats.skew(gg["pred"].values) if len(gg)>3 else 0.0
        n=len(gg)
        L=gg.tail(K); S=gg.head(K)
        # cutoff pred values (the marginal name just outside K)
        cutL=gg.iloc[-(K+1)][key] if n>K else gg.iloc[-1][key]
        cutS=gg.iloc[K][key] if n>K else gg.iloc[0][key]
        for _,r in L.iterrows():
            legs.append((ot,rg,+1,r["pred"],r[key]-cutL,disp,skew,n,r["alpha_A"]))
        for _,r in S.iterrows():
            legs.append((ot,rg,-1,r["pred"],cutS-r[key],disp,skew,n,r["alpha_A"]))
    lf=pd.DataFrame(legs,columns=["open_time","regime","side","pred","rank_gap",
                                  "xs_disp","xs_skew","n_elig","alpha_res"])
    lf["leg_pnl"]=lf["side"]*lf["alpha_res"]        # signed realized alpha contribution
    lf["abs_pred"]=lf["pred"].abs()
    lf["is_long"]=(lf["side"]>0).astype(float)
    print(f"legs: {len(lf)}  (long {int(lf['is_long'].sum())} / short {int((1-lf['is_long']).sum())})")
    print(f"leg_pnl mean {lf['leg_pnl'].mean()*1e4:+.2f}bps  hitrate {(lf['leg_pnl']>0).mean()*100:.1f}%")
    print(f"  LONG : pnl {lf[lf.side>0]['leg_pnl'].mean()*1e4:+.2f}bps hit {(lf[lf.side>0]['leg_pnl']>0).mean()*100:.1f}%")
    print(f"  SHORT: pnl {lf[lf.side<0]['leg_pnl'].mean()*1e4:+.2f}bps hit {(lf[lf.side<0]['leg_pnl']>0).mean()*100:.1f}%\n",flush=True)

    feats=["rank_gap","xs_disp","xs_skew","n_elig","abs_pred","is_long"]
    y=lf["leg_pnl"].values

    print("=== Raw Spearman IC of meta-features vs leg_pnl ===")
    for f in feats:
        ic,_=stats.spearmanr(lf[f],y)
        print(f"  {f:>10}: IC {ic:+.4f}")

    # Conditional test: residualize leg_pnl on a rank-spline of pred (what primary KNOWS),
    # then test meta-features on the RESIDUAL. This is the "second squeeze" test.
    # Use pred-decile means as the primary's monetizable info.
    lf["pred_dec"]=pd.qcut(lf["pred"],10,labels=False,duplicates="drop")
    lf["y_hat_pred"]=lf.groupby("pred_dec")["leg_pnl"].transform("mean")
    lf["y_resid"]=lf["leg_pnl"]-lf["y_hat_pred"]
    print("\n=== Conditional IC vs leg_pnl RESIDUAL (after removing pred-decile mean) ===")
    print("    (this is the orange-squeeze test: info beyond what pred already gives)")
    for f in feats:
        ic,_=stats.spearmanr(lf[f],lf["y_resid"])
        print(f"  {f:>10}: residIC {ic:+.4f}")

    # Also residualize on BOTH pred-decile AND side (remove the structural long/short asym),
    # to see if anything CROSS-SECTIONAL survives that isn't just "short more".
    lf["ps"]=lf["pred_dec"].astype(str)+"_"+lf["side"].astype(str)
    lf["y_hat_ps"]=lf.groupby("ps")["leg_pnl"].transform("mean")
    lf["y_resid2"]=lf["leg_pnl"]-lf["y_hat_ps"]
    print("\n=== Conditional IC vs residual after pred-decile × SIDE (removes side asym) ===")
    for f in ["rank_gap","xs_disp","xs_skew","n_elig"]:
        ic,_=stats.spearmanr(lf[f],lf["y_resid2"])
        print(f"  {f:>10}: residIC {ic:+.4f}")

    # Decile lift of the single best meta-feature on the residual (eyeball monetizability)
    for f in ["xs_disp","rank_gap","n_elig"]:
        lf["fb"]=pd.qcut(lf[f],5,labels=False,duplicates="drop")
        m=lf.groupby("fb")["y_resid2"].mean()*1e4
        print(f"\n  {f} quintile residual-pnl (bps): "+" ".join(f"{v:+.1f}" for v in m.values))

    print(f"\nDone [{time.time()-t0:.0f}s]")

if __name__=="__main__":
    main()
