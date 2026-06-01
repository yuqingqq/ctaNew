"""X88 — 3-regime PIT gate on the sleeve: does separating BEAR (flat or momentum)
beat the 2-regime hybrid (X85, where bear got mean-rev and bled)?

PIT gate (BTC trailing-30d return): bull >+0.10, bear <-0.10, else sideways.
Variants (V3.1 sleeve, NET, 3yr + 12mo):
  H2  = 2-regime hybrid (X85): bull→mom30, else→V0  (bear gets mean-rev)   [baseline +1.19]
  H3f = 3-regime flat-bear   : bull→mom30, side→V0, bear→FLAT (drop bear cycles)
  H3m = 3-regime mom-bear    : bull→mom30, side→V0, bear→mom30 (momentum catch)
Also: gate quality — when gate=bear, what is mean-rev's realized PnL? (does the
PIT gate reliably flag the regime where mean-rev bleeds?)
"""
from __future__ import annotations
import sys, importlib.util, time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
RCACHE = REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
spec = importlib.util.spec_from_file_location("x6", REPO/"research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def load_close(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def per_cycle_z(s):
    return s.groupby(level=0).transform(lambda x:(x-x.mean())/(x.std()+1e-9))


def run(label, apd):
    p=RCACHE/f"x88_{label}_preds.parquet"; apd.to_parquet(p,index=False)
    return x6.run_sleeve_on_preds(p,f"x88_{label}")


def main():
    t0=time.time()
    print("=== X88 3-regime PIT gate (sleeve net) ===\n", flush=True)
    apd=pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
    apd["exit_time"]=pd.to_datetime(apd["exit_time"],utc=True)
    syms=sorted(apd["symbol"].unique())

    print("mom_30d...", flush=True)
    mr=[]
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        mom=(c/c.shift(8640)-1).shift(1)
        mr.append(pd.DataFrame({"symbol":sym,"open_time":mom.index,"mom30":mom.values}))
    mom=pd.concat(mr,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    apd=apd.merge(mom,on=["symbol","open_time"],how="left")

    # PIT regime via merge_asof (5m grid)
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index()
    btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True).astype("datetime64[ns, UTC]")
    apd["open_time"]=apd["open_time"].astype("datetime64[ns, UTC]")
    apd["exit_time"]=apd["exit_time"].astype("datetime64[ns, UTC]")
    apd=apd.sort_values("open_time"); btc30=btc30.sort_values("open_time")
    apd=pd.merge_asof(apd, btc30, on="open_time", direction="backward")
    apd["regime"]=np.where(apd["btc_ret_30d"]>0.10,"bull",np.where(apd["btc_ret_30d"]<-0.10,"bear","side"))

    apd=apd.set_index("open_time")
    apd["v0_z"]=per_cycle_z(apd["pred"]); apd["mom_z"]=per_cycle_z(apd["mom30"].fillna(0))
    apd=apd.reset_index().dropna(subset=["alpha_A","return_pct"])

    fr=apd.groupby("open_time")["regime"].first().value_counts(normalize=True)*100
    print(f"regime mix: {dict(fr.round(1))}\n")

    def variants(sub, tag):
        out={}
        # H2: bull→mom, else→v0 (bear=mean-rev)
        a=sub.copy(); a["pred"]=np.where(a["regime"]=="bull", a["mom_z"], a["v0_z"])
        out["H2_bear_meanrev"]=run(f"{tag}_H2", a[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]])
        # H3f: bull→mom, side→v0, bear→FLAT (drop)
        b=sub[sub["regime"]!="bear"].copy(); b["pred"]=np.where(b["regime"]=="bull", b["mom_z"], b["v0_z"])
        out["H3_flat_bear"]=run(f"{tag}_H3f", b[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]])
        # H3m: bull→mom, side→v0, bear→mom
        c=sub.copy(); c["pred"]=np.where(c["regime"].isin(["bull","bear"]), c["mom_z"], c["v0_z"])
        out["H3_mom_bear"]=run(f"{tag}_H3m", c[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]])
        return out

    for tag, sub in [("3yr", apd), ("12mo", apd[apd["open_time"]>=pd.Timestamp("2025-05-01",tz="UTC")])]:
        print(f"--- {tag} (sleeve net) ---")
        for k,m in variants(sub, tag).items():
            print(f"  {k:<18} Sharpe={m.get('sharpe',0):+.2f} folds={m.get('folds_pos','?')} conc={m.get('concentration','?')} PnL={m.get('totPnL','?')}", flush=True)

    # gate quality: when gate=bear, mean-rev realized cohort PnL (4h spread)
    print(f"\n--- PIT gate quality: mean-rev PnL when gate flags bear ---")
    g4=apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)]
    for rg in ["bull","side","bear"]:
        rows=[]
        for ot,grp in g4[g4["regime"]==rg].groupby("open_time"):
            grp=grp.dropna(subset=["pred","return_pct"])
            if len(grp)<6: continue
            gg=grp.sort_values("pred"); rows.append(gg.tail(3)["return_pct"].mean()-gg.head(3)["return_pct"].mean())
        r=np.array(rows)
        if len(r)<5: continue
        print(f"  gate={rg:<5}: mean-rev mean {r.mean()*1e4:+.2f}bps, % cycles negative {(r<0).mean()*100:.0f}%  (n={len(r)})", flush=True)

    print(f"\nDone [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
