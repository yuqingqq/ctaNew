"""SYMAGE1 (pre-reg addendum 51): does the short edge concentrate in YOUNG (freshly-listed) symbols,
era-stably? Age = cycle_t − first_kline_date (PIT-leading). Short PnL = residual −alpha_A, non-deep-bull.
Per age-tier daily-agg Sharpe, both eras + young−old delta block-bootstrap CI. CI-gated (short-only lesson).
"""
import numpy as np, pandas as pd, glob
from pathlib import Path
import sys; sys.path.insert(0,"live")
from attribution_v4_regime import btc_reg, load
import warnings; warnings.filterwarnings("ignore")
KD=Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines"); rng=np.random.default_rng(51)

def first_listing(symbols):
    fl={}
    for s in symbols:
        fs=sorted(glob.glob(str(KD/s/"5m"/"*.parquet")))
        if not fs: continue
        try:
            t=pd.read_parquet(fs[0],columns=["open_time"])["open_time"]
            fl[s]=pd.to_datetime(t,utc=True).min()
        except Exception: pass
    return fl

def short_names(base, reg):
    rows=[]
    for t,g in base.groupby("open_time"):
        if len(g)<5 or reg.get(t) in (None,"deepbull"): continue
        S=g.nsmallest(2,"pred")
        for _,r in S.iterrows(): rows.append((t, r["symbol"], -r["alpha_A"]*1e4))  # residual short PnL
    return pd.DataFrame(rows,columns=["open_time","symbol","pnl"])

def tier_sharpe(d, tier):
    g=d[d.tier==tier]; day=pd.to_datetime(g["open_time"]).dt.date
    dr=g.assign(day=day).groupby("day")["pnl"].sum()
    return (dr.mean()/dr.std()*np.sqrt(365) if dr.std()>0 else np.nan), dr

def boot_delta(dy, do, L=10, n=2000):  # young - old Sharpe delta CI on shared days
    idx=dy.index.union(do.index); y=dy.reindex(idx).fillna(0).values; o=do.reindex(idx).fillna(0).values
    m=len(y); nb=int(np.ceil(m/L)); out=[]
    for _ in range(n):
        st=rng.integers(0,max(1,m-L+1),nb); ii=np.concatenate([np.arange(x,x+L) for x in st])[:m]%m
        yy,oo=y[ii],o[ii]
        if yy.std()>0 and oo.std()>0: out.append(yy.mean()/yy.std()*np.sqrt(365)-oo.mean()/oo.std()*np.sqrt(365))
    return np.percentile(out,[2.5,50,97.5]) if out else (np.nan,)*3

def main():
    reg=btc_reg()
    picks={}
    for era,bp in (("RECENT","hl_tgt_res_base_cleanfix"),("OOS","hl_v4base_oos_cleanfix")):
        base,_=load(bp,bp); picks[era]=short_names(base,reg)
    syms=sorted(set(picks["RECENT"].symbol)|set(picks["OOS"].symbol))
    print(f"first-listing dates for {len(syms)} short-name symbols...",flush=True)
    fl=first_listing(syms)
    res={}
    for era in ("RECENT","OOS"):
        d=picks[era].copy()
        d["age"]=[ (t-fl[s]).days if s in fl else np.nan for s,t in zip(d.symbol,d.open_time)]
        d=d.dropna(subset=["age"])
        d["tier"]=np.where(d.age<120,"young",np.where(d.age<400,"mid","old"))
        print(f"\n===== {era}: SHORT PnL by symbol age (n={len(d)} short-names) =====")
        drs={}
        for t in ["young","mid","old"]:
            sS,dr=tier_sharpe(d,t); drs[t]=dr
            g=d[d.tier==t]
            print(f"  {t:>5} (<120d/120-400/>400) n={len(g):5d}: Sharpe {sS:+.2f}  mean PnL/name {g.pnl.mean():+.1f}  medPnL {g.pnl.median():+.1f}")
        lo,md,hi=boot_delta(drs["young"],drs["old"])
        res[era]=(tier_sharpe(d,"young")[0], tier_sharpe(d,"old")[0], (lo,hi))
        print(f"  young−old Sharpe delta: point {res[era][0]-res[era][1]:+.2f}  CI [{lo:+.2f},{hi:+.2f}]  {'SIG' if lo>0 else 'NOT SIG (crosses 0)'}")
    both_pt=all(y>o for y,o,_ in res.values())
    both_sig=all(ci[0]>0 for _,_,ci in res.values())
    print(f"\n  >> GATE (young>old BOTH eras + delta CI excludes 0 BOTH): point {'PASS' if both_pt else 'FAIL'} | CI {'PASS -> real young-tilt lever' if both_sig else 'FAIL (noise)'}")
    print("SYMAGEDONE")

if __name__=="__main__":
    main()
