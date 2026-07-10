"""Short-only candidate — the cheap decisive screen (reviewer aacdc47 flag 3): paired block-bootstrap CI
on the Sharpe delta + per-quarter breakdown, BOTH eras. Every prior "breakthrough" here died on this
(thin OOS deltas concentrate in 1-2 folds). If the delta CI crosses 0 or it's 1-2-quarter-driven, it's
an artifact — kill it before the expensive full-stack replay.
"""
import numpy as np, pandas as pd
import sys; sys.path.insert(0,"live")
from shortonly_test import build
from attribution_v4_regime import btc_reg, load
import warnings; warnings.filterwarnings("ignore")
rng=np.random.default_rng(47)

def daily(d, c):
    dd=pd.to_datetime(d["t"]).dt.date
    return d[[c]].groupby(dd).sum()[c]
def sharpe(s): return s.mean()/s.std()*np.sqrt(365) if s.std()>0 else np.nan

def boot_delta_ci(db, ds, L=10, n=2000):
    """paired block-bootstrap of Sharpe(short_only) - Sharpe(1L2S) over shared days."""
    idx=db.index.intersection(ds.index); b=db.reindex(idx).values; s=ds.reindex(idx).values
    m=len(b); nb=int(np.ceil(m/L)); out=[]
    for _ in range(n):
        st=rng.integers(0,max(1,m-L+1),nb); ii=np.concatenate([np.arange(x,x+L) for x in st])[:m]%m
        bb,ss=b[ii],s[ii]
        if bb.std()>0 and ss.std()>0:
            out.append(ss.mean()/ss.std()*np.sqrt(365) - bb.mean()/bb.std()*np.sqrt(365))
    return np.percentile(out,[2.5,50,97.5]) if out else (np.nan,)*3

def main():
    reg=btc_reg()
    for era,bp,lp in (("RECENT","hl_tgt_res_base_cleanfix","hl_tgt_res_long_cleanfix"),
                     ("OOS","hl_v4base_oos_cleanfix","hl_v4long_oos_cleanfix")):
        base,long=load(bp,lp); d=build(base,long,reg)
        db,ds=daily(d,"b"),daily(d,"s")
        dS=sharpe(ds)-sharpe(db)
        lo,md,hi=boot_delta_ci(db,ds)
        print(f"\n===== {era}: short-only − 1L/2S Sharpe delta =====")
        print(f"  point Δ {dS:+.2f}  | block-bootstrap CI [{lo:+.2f}, {hi:+.2f}]  {'SIG (excludes 0)' if lo>0 else 'NOT SIG (CI crosses 0)'}")
        # per-quarter delta (concentration): Sharpe of each book per quarter
        d2=d.copy(); d2["q"]=pd.to_datetime(d2["t"]).dt.to_period("Q").astype(str)
        print("  per-quarter Sharpe (1L2S -> short-only, Δ):")
        for q,g in d2.groupby("q"):
            if len(g)<20: continue
            gb=daily(g,"b"); gs=daily(g,"s")
            print(f"     {q}: {sharpe(gb):+.2f} -> {sharpe(gs):+.2f}  (Δ {sharpe(gs)-sharpe(gb):+.2f})")
    print("\nSHORTONLYCIDONE")

if __name__=="__main__":
    main()
