"""LOOP iter-11 — nested-OOS over K: is K=3 genuinely robust or window-fit?
For each month m (from the 3rd onward), pick the K that maximized Sharpe on months < m (strictly
past), apply it to month m. Compare nested-selected-K Sharpe vs fixed K=3 vs fixed K=5.
Uses the xsz60 cycles already replayed at K=3/5/7/10 (/tmp/cyc_xsz60_K{K}.csv).
"""
import sys
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
Ks=[3,5,7,10]
def load(K):
    d=pd.read_csv(f"/tmp/cyc_xsz60_K{K}.csv",parse_dates=["open_time"]); d["open_time"]=pd.to_datetime(d["open_time"],utc=True)
    d["m"]=d["open_time"].dt.strftime("%Y-%m"); return d
def msh(d,m):
    p=d[d.m==m]["pnl_bps"].values/1e4; return p.mean()/p.std()*np.sqrt(6*365) if len(p)>1 and p.std()>0 else np.nan
def full_sh(pnl):
    p=np.array(pnl)/1e4; return p.mean()/p.std()*np.sqrt(6*365) if len(p)>1 and p.std()>0 else np.nan

def main():
    print("=== LOOP iter-11: nested-OOS over K ===\n",flush=True)
    dat={K:load(K) for K in Ks}
    months=sorted(dat[3]["m"].unique())
    # past-Sharpe per K up to month i-1, pick best K for month i
    nested_pnl=[]; picks=[]
    for i,m in enumerate(months):
        if i<2:  # warmup: use K=5 default
            bestK=5
        else:
            past=months[:i]
            scoreK={}
            for K in Ks:
                d=dat[K]; p=d[d.m.isin(past)]["pnl_bps"].values/1e4
                scoreK[K]=p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else -9
            bestK=max(scoreK,key=scoreK.get)
        picks.append((m,bestK))
        nested_pnl.extend(dat[bestK][dat[bestK].m==m]["pnl_bps"].values)
    print("  month   nested-pickK")
    for m,k in picks: print(f"  {m}     K={k}")
    print(f"\n  nested-OOS-K Sharpe : {full_sh(nested_pnl):+.2f}")
    for K in Ks: print(f"  fixed K={K} Sharpe   : {full_sh(dat[K]['pnl_bps'].values):+.2f}")
    print(f"\n  K=3 robust if fixed-K=3 >= nested and >= other fixed K (discrete, no past-fit needed).")

if __name__=="__main__": main()
