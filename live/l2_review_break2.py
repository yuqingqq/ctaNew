"""BREAK attempt 2: single-name (ZEC) dependence + neighbor-corner fragility of the
RECENT $10k/0.1% gating win. Scratch only."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"/home/yuqing/ctaNew/live")
import warnings; warnings.filterwarnings("ignore")
from bookdepth_impact import impact_pct
from l2_exec_backtest import build_selection, load_depth, attach_depth, gated, equal_dollar, sharpe, PRED

def load_era(era):
    sel0=build_selection(PRED[era]); syms=pd.unique(sel0[["long","s2","s3"]].values.ravel())
    depth=load_depth(syms); sel,_,_=attach_depth(sel0,depth); return sel

def gated_excl(sel,S,X,exclude=()):
    """gated() but any bar whose surviving legs include an excluded name is skipped."""
    iL=np.vectorize(impact_pct)(S,sel.d02_L,sel.d1_L,sel.d5_L)
    iA=np.vectorize(impact_pct)(S/2,sel.d02_A,sel.d1_A,sel.d5_A)
    iB=np.vectorize(impact_pct)(S/2,sel.d02_B,sel.d1_B,sel.d5_B)
    net=np.full(len(sel),np.nan); ex=set(exclude)
    L=sel["long"].values; A=sel["s2"].values; B=sel["s3"].values
    aL=sel["a_long"].values; aA=sel["a_s2"].values; aB=sel["a_s3"].values
    for i in range(len(sel)):
        if iL[i]>X or L[i] in ex: continue
        kA,kB=iA[i]<=X and A[i] not in ex, iB[i]<=X and B[i] not in ex
        if not kA and not kB: continue
        if kA and kB:
            paper=0.5*aL[i]-0.25*aA[i]-0.25*aB[i]; imp=iL[i]+0.5*iA[i]+0.5*iB[i]
        else:
            aS=aA[i] if kA else aB[i]; iS=iA[i] if kA else iB[i]
            paper=0.5*aL[i]-0.5*aS; imp=iL[i]+iS
        net[i]=paper-imp
    return net

sel=load_era("RECENT")
print("BREAK: RECENT $10k/0.1% single-name dependence")
for excl in [(), ("ZECUSDT",), ("ZECUSDT","MONUSDT","ZORAUSDT"), ("ZECUSDT","MONUSDT","ZORAUSDT","AVNTUSDT","IPUSDT","MERLUSDT","RESOLVUSDT","SUPERUSDT")]:
    g=gated_excl(sel,10_000,0.001,excl); m=~np.isnan(g)
    sh=sharpe(g[m],sel["open_time"][m])
    tag="none" if not excl else f"{len(excl)} names (ZEC+{len(excl)-1})"
    print(f"  exclude {tag:24}: gated Sharpe {sh:+.2f}  bars={m.sum()}")

print("\nBREAK: neighbor-corner fragility around ($10k, 0.1%)  [Sharpe (bars-kept%)]")
print(f"{'size\\thr':>10} "+" ".join(f"{x*100:>9.2f}%" for x in [0.0008,0.001,0.0012,0.0015]))
for S in [8_000,10_000,12_000,15_000]:
    cells=[]
    for X in [0.0008,0.001,0.0012,0.0015]:
        g=gated(sel,S,X); m=~np.isnan(g); sh=sharpe(g[m],sel["open_time"][m])
        cells.append(f"{sh:+5.2f}({m.mean()*100:2.0f}%)")
    print(f"${S/1e3:>7.0f}k "+" ".join(f"{c:>10}" for c in cells))

# OOS: why is gating inert at $10-25k? show impact distribution of covered legs
print("\nOOS: one-way impact distribution at $10k (covered legs) -> is there any tail to gate?")
selo=load_era("OOS")
imp=np.concatenate([np.vectorize(impact_pct)(10_000,selo[f"d02_{t}"],selo[f"d1_{t}"],selo[f"d5_{t}"]) for t in ("L","A","B")])
for q in [50,75,90,95,99]:
    print(f"  p{q}: {np.percentile(imp,q)*100:.3f}%")
print(f"  legs with >0.1% impact at $10k: {(imp>0.001).mean()*100:.1f}%  | >0.2%: {(imp>0.002).mean()*100:.1f}%")
print("RECENT same:")
impr=np.concatenate([np.vectorize(impact_pct)(10_000,sel[f"d02_{t}"],sel[f"d1_{t}"],sel[f"d5_{t}"]) for t in ("L","A","B")])
for q in [50,75,90,95,99]:
    print(f"  p{q}: {np.percentile(impr,q)*100:.3f}%")
print(f"  legs with >0.1% impact at $10k: {(impr>0.001).mean()*100:.1f}%  | >0.2%: {(impr>0.002).mean()*100:.1f}%")
