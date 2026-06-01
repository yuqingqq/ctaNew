"""iter-007 — quick STRUCTURAL-FORM data check for the 2-axis alt-bear side gate.

We are NOT tuning a threshold. We compare candidate PARAMETER-FREE structural forms of
"alt complex is in a bear" and ask which one best separates the bad side cycles (the ones
that supply the side-regime drawdown), on BOTH:
  - HL70 (production, one big DD episode)
  - ext 23-sym 2021-26 panel (multiple episodes: 2022 LUNA/FTX, 2024 selloffs)

Candidate structural forms (alt_index_30d = PIT trailing-30d cum log-ret of the equal-weight
TRADED universe, ex-BTC, ex-ETH, lagged by .shift(1) — exactly the iter006 alt-index):
  F0  alt30 < 0                        absolute alt drawdown (structural 0 boundary)
  F1  alt30 < btc30                    alts UNDERPERFORM BTC (relative, structural 0 boundary)
  F2  alt30 < btc30  AND  alt30 < 0    relative AND absolute (both structural)
  F3  alt_breadth < 0.5                <50% of alts up over 30d (structural majority boundary)

Reference (NOT a candidate — it is the tuned one we are replacing):
  T   alt30 < -0.10                    the swept scalar that failed G3

For each form, on side-regime cycles only, we report:
  - n_flag (how many side cycles it flags), gated fraction
  - mean side held-book PnL on FLAGGED vs UNFLAGGED cycles (does it isolate the losers?)
  - the side-regime DD removed if we FLAT all flagged side cycles
  - separation = mean(unflagged) - mean(flagged)  (bps/cycle; bigger = cleaner separation)
  - per-EPISODE breakdown on the ext panel (does the form flag the bad cycles in EACH episode,
    not just one?)

All PIT for the trading decision; the held-book side PnL we score against is the base book's
realized per-cycle PnL on side cycles (computed by the same X117 engine). No future info.
"""
from __future__ import annotations
import time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
RC = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
HL70 = RC/"x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet"
EXT = RC/"x113_ext_v0_preds.parquet"
K=5; HOLD=6; COST=4.5e-4


def load_close(sym):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def build(preds_path, label):
    d=pd.read_parquet(preds_path, columns=["symbol","open_time","pred","return_pct","fold"])
    d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d=d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)].copy()
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    syms=sorted(d["symbol"].unique()); mom_rows=[]; beta_map={}; ret_map={}
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":c4.index,"mom30":(c4/c4.shift(180)-1).shift(1).values}))
        r=np.log(c4/c4.shift(1)); ri,bi=r.align(br,join="inner")
        beta_map[sym]=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
        ret_map[sym]=r
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    betas=pd.concat([s.rename(k) for k,s in beta_map.items()],axis=1)
    ret4=pd.concat([s.rename(k) for k,s in ret_map.items()],axis=1).sort_index()
    d=d.merge(mom,on=["symbol","open_time"],how="left")
    btc30=(b4/b4.shift(180)-1).to_frame("b30").reset_index(); btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    d=d.merge(btc30,on="open_time",how="left").dropna(subset=["b30"])
    d["regime"]=np.where(d["b30"]>0.10,"bull",np.where(d["b30"]<-0.10,"bear","side"))
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}
    fold_by={ot:int(g["fold"].iloc[0]) for ot,g in by_t.items()}

    # --- PIT alt-index from the SAME traded universe (ex BTC/ETH), .shift(1) lagged ---
    altcols=[c for c in ret4.columns if c not in ("BTCUSDT","ETHUSDT")]
    altidx=ret4[altcols].mean(axis=1)                 # eq-wt alt 4h log-ret
    alt_cum=altidx.cumsum()
    alt30=(alt_cum-alt_cum.shift(180)).shift(1)        # ~30d cum log-ret, LAGGED (PIT)
    # alt breadth: fraction of alts with positive trailing-30d cum log-ret, lagged
    per_cum={c:ret4[c].cumsum() for c in altcols}
    per30=pd.DataFrame({c:(per_cum[c]-per_cum[c].shift(180)) for c in altcols})
    breadth=(per30>0).mean(axis=1).shift(1)
    alt30=alt30.reindex(pd.DatetimeIndex(times)); breadth=breadth.reindex(pd.DatetimeIndex(times))
    b30s=btc30.set_index("open_time")["b30"].reindex(pd.DatetimeIndex(times))

    # --- build base-arm cycle weights (X117) and held book per-cycle PnL ---
    cyc_w=[]; regimes=[]; rs=[]
    for ot in times:
        g=by_t[ot]; rg=g["regime"].iloc[0]; regimes.append(rg); rs.append(dict(zip(g["symbol"],g["return_pct"])))
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
    prev={}; pnl=[]
    for t in range(len(cyc_w)):
        active=cyc_w[max(0,t-HOLD+1):t+1]; net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
        rl=rs[t]; cyc=sum(net.get(s,0)*rl.get(s,0.0) for s in net if np.isfinite(rl.get(s,0.0)))
        if not np.isfinite(cyc): cyc=0.0
        pnl.append((cyc-turn*0.5*COST)*1e4)
        prev=net
    df=pd.DataFrame({"open_time":pd.DatetimeIndex(times),"regime":regimes,
                     "pnl":pnl,"fold":[fold_by.get(t,-1) for t in times],
                     "alt30":alt30.values,"btc30":b30s.values,"breadth":breadth.values})
    return df


# fixed structural episodes on the ext panel (calendar windows of known alt bears)
EXT_EPISODES = [
    ("2021_blowoff",  "2021-05-01","2021-07-31"),
    ("2022_luna",     "2022-05-01","2022-07-31"),
    ("2022_ftx",      "2022-11-01","2023-01-31"),
    ("2024_summer",   "2024-06-01","2024-09-30"),
    ("2025_q4",       "2025-09-01","2025-12-31"),
]


def eval_forms(df, label, episodes=None):
    side=df[df["regime"]=="side"].dropna(subset=["alt30","btc30","breadth"]).copy()
    print(f"\n===== {label}: side cycles={len(side)}  mean side PnL {side['pnl'].mean():+.2f} bps =====", flush=True)
    forms={
        "F0 alt30<0":            side["alt30"]<0,
        "F1 alt30<btc30":        side["alt30"]<side["btc30"],
        "F2 alt30<btc30 & <0":   (side["alt30"]<side["btc30"])&(side["alt30"]<0),
        "F3 breadth<0.5":        side["breadth"]<0.5,
        "T  alt30<-0.10 (tuned)":side["alt30"]<-0.10,
    }
    print(f"  {'form':<24}{'n_flag':>7}{'frac':>7}{'mean_flag':>11}{'mean_unflag':>12}{'sep':>9}{'DDremoved%':>11}", flush=True)
    eq=side["pnl"].cumsum(); base_dd=(eq-eq.cummax()).min()
    for name,flag in forms.items():
        mf=side.loc[flag,"pnl"]; mu=side.loc[~flag,"pnl"]
        # FLAT the flagged side cycles: keep unflagged side PnL only (rest of book unchanged
        # is approximated here at side-subbook level for the data check)
        kept=side["pnl"].where(~flag,0.0)
        eqk=kept.cumsum(); dd_k=(eqk-eqk.cummax()).min()
        ddrem=(1-abs(dd_k)/abs(base_dd))*100 if base_dd<0 else np.nan
        sep=mu.mean()-mf.mean()
        print(f"  {name:<24}{int(flag.sum()):>7}{flag.mean():>7.2f}{mf.mean():>+11.2f}{mu.mean():>+12.2f}{sep:>+9.2f}{ddrem:>+11.1f}", flush=True)

    if episodes:
        print(f"\n  --- per-episode separation (mean side PnL flagged vs unflagged), key forms ---", flush=True)
        print(f"  {'episode':<16}{'n_side':>7}{'F1_flag':>8}{'F1_flagPnL':>11}{'F1_unflPnL':>11}"
              f"{'F2_flag':>8}{'F2_flagPnL':>11}", flush=True)
        for ename,a,b in episodes:
            m=(side["open_time"]>=pd.Timestamp(a,tz="UTC"))&(side["open_time"]<=pd.Timestamp(b,tz="UTC"))
            sub=side[m]
            if len(sub)<5:
                print(f"  {ename:<16}{len(sub):>7}  (too few side cycles)", flush=True); continue
            f1=sub["alt30"]<sub["btc30"]; f2=(sub["alt30"]<sub["btc30"])&(sub["alt30"]<0)
            print(f"  {ename:<16}{len(sub):>7}{int(f1.sum()):>8}"
                  f"{sub.loc[f1,'pnl'].mean():>+11.2f}{sub.loc[~f1,'pnl'].mean():>+11.2f}"
                  f"{int(f2.sum()):>8}{sub.loc[f2,'pnl'].mean():>+11.2f}", flush=True)


def main():
    t0=time.time()
    print("=== iter-007 structural-form data check ===", flush=True)
    dfh=build(HL70,"HL70"); eval_forms(dfh,"HL70 (1 episode)")
    dfe=build(EXT,"EXT");   eval_forms(dfe,"EXT 23-sym 2021-26 (multi-episode)", EXT_EPISODES)
    print(f"\nDone [{time.time()-t0:.0f}s]", flush=True)


if __name__=="__main__":
    main()
