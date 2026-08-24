"""SKEPTICAL review of l2_exec_backtest.py: try to BREAK liquidity gating +
verify capacity ceiling. Scratch only. Run from live/."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/yuqing/ctaNew/live")
import warnings; warnings.filterwarnings("ignore")
from bookdepth_impact import impact_pct
from l2_exec_backtest import (build_selection, load_depth, attach_depth,
    equal_dollar, gated, sharpe, PRED, L2_DIR, _ns, BAR)

pd.set_option("display.width", 200)

# ============================================================ CHECK 1: impact sanity
def check1():
    print("="*90, "\nCHECK 1  IMPACT MODEL SANITY (recent-median depth; round-trip=2x one-way)\n", "="*90)
    # use POST-2026-01 window where l2_touch is real (not the 0.14 fallback)
    rec = pd.Timestamp("2026-02-01", tz="UTC")
    names = ["BTCUSDT","ETHUSDT","SOLUSDT","RUNEUSDT","WLDUSDT","SEIUSDT"]
    print(f"{'sym':9} {'L(1%)$M':>8} {'d02$k':>7} {'d1$k':>7} {'d5$k':>8} | "
          f"{'imp100k':>8} {'imp1M':>8} | {'RT100k_bps':>10} {'touch':>6} {'slope':>6}")
    for sym in names:
        d = pd.read_parquet(L2_DIR/f"l2_{sym}.parquet", columns=["l2_liq1","l2_touch","l2_slope"])
        d = d[pd.DatetimeIndex(d.index) >= rec]
        L = np.exp(d["l2_liq1"]).median()
        touch = d["l2_touch"].median(); slope = d["l2_slope"].median()
        d1=L/2; d02=touch*L/2; d5=slope*L/2
        i100=impact_pct(100_000,d02,d1,d5); i1m=impact_pct(1_000_000,d02,d1,d5)
        print(f"{sym:9} {L/1e6:8.1f} {d02/1e3:7.0f} {d1/1e3:7.0f} {d5/1e3:8.0f} | "
              f"{i100*100:7.3f}% {i1m*100:7.3f}% | {2*i100*1e4:10.1f} {touch:6.3f} {slope:6.2f}")
    # unit / monotonicity torture: tiny trade below d02 must be ~linear <0.2%; RT=2x
    d02,d1,d5 = 50_000, 200_000, 1_000_000
    for N in [1_000, 10_000, 50_000, 200_000, 1_000_000, 5_000_000]:
        oneway=impact_pct(N,d02,d1,d5)
        print(f"  torture d02=50k d1=200k d5=1M: N=${N:>9,} one-way={oneway*100:7.3f}%  RT={2*oneway*100:7.3f}%")
    print("  (expect: N<=d02 -> <0.2%; N=d1 -> 1.0%; N=d5 -> 5.0%; monotone; RT exactly 2x)")

# ============================================================ full gating grid both eras
def gating_grid(sel):
    SIZES=[10_000,25_000,50_000,100_000]; THR=[0.003,0.002,0.001]
    out={}
    for S in SIZES:
        base=sharpe(equal_dollar(sel,S), sel["open_time"])
        row={"no_gate":base}
        for X in THR:
            g=gated(sel,S,X); m=~np.isnan(g)
            sh=sharpe(g[m],sel["open_time"][m]) if m.sum()>5 else np.nan
            row[f"g{X}"]=(sh, m.mean())
        out[S]=row
    return out

def print_grid(name, grid):
    print(f"\n--- {name} liquidity-gating grid (Sharpe, %bars-kept) ---")
    print(f"{'S':>8} {'no_gate':>8} | {'0.3%':>16} {'0.2%':>16} {'0.1%':>16}")
    for S,row in grid.items():
        cells=[]
        for X in [0.003,0.002,0.001]:
            sh,frac=row[f"g{X}"]; cells.append(f"{sh:+6.2f} ({frac*100:2.0f}%)")
        print(f"${S/1e3:>6.0f}k {row['no_gate']:+8.2f} | "+"   ".join(f"{c:>14}" for c in cells))

# ============================================================ CHECK 4: name composition
def name_composition(sel, S, X):
    """Which legs survive gating at (S,X)? Return surviving-leg name counts."""
    iL=np.vectorize(impact_pct)(S,sel.d02_L,sel.d1_L,sel.d5_L)
    iA=np.vectorize(impact_pct)(S/2,sel.d02_A,sel.d1_A,sel.d5_A)
    iB=np.vectorize(impact_pct)(S/2,sel.d02_B,sel.d1_B,sel.d5_B)
    kept_legs=[]; kept_bars=0
    for i in range(len(sel)):
        if iL[i]>X: continue
        kA,kB = iA[i]<=X, iB[i]<=X
        if not kA and not kB: continue
        kept_bars+=1
        kept_legs.append(sel["long"].values[i])         # long always kept
        if kA: kept_legs.append(sel["s2"].values[i])
        if kB: kept_legs.append(sel["s3"].values[i])
    vc=pd.Series(kept_legs).value_counts()
    return kept_bars, vc

# ============================================================ CHECK 4b: block bootstrap CI
def block_bootstrap_ci(net, index, n=2000, seed=0):
    """Daily-block bootstrap CI on Sharpe*sqrt(365)."""
    s=pd.Series(net,index=index); daily=s.groupby(s.index.floor("D")).sum().dropna()
    d=daily.values; rng=np.random.default_rng(seed); shs=[]
    for _ in range(n):
        b=rng.choice(d,size=len(d),replace=True)
        if b.std()>0: shs.append(b.mean()/b.std()*np.sqrt(365))
    shs=np.array(shs)
    return np.percentile(shs,[2.5,50,97.5]), (shs>0).mean()

# ============================================================ CHECK 5: OOS survivorship
def survivorship(sel_all_before_drop, kept_sel, depth):
    """Compare median 1% depth of legs in KEPT vs DROPPED bars."""
    def med_depth(names):
        ds=[]
        for s in names:
            if s in depth and len(depth[s]): ds.append(depth[s]["d1"].median())
        return np.median(ds) if ds else np.nan
    kept_names=pd.unique(kept_sel[["long","s2","s3"]].values.ravel())
    all_names=pd.unique(sel_all_before_drop[["long","s2","s3"]].values.ravel())
    dropped=set(all_names)-set(kept_names)
    print(f"  covered legs: {len(kept_names)} names  median 1%-depth ${med_depth(kept_names)/1e6:.2f}M")
    print(f"  DROPPED (no-PIT-depth) legs: {len(dropped)} names  median 1%-depth "
          f"${med_depth(list(dropped))/1e6:.2f}M" if dropped else "  (none dropped)")

# ============================================================ run
def load_era(era):
    sel0=build_selection(PRED[era])
    syms=pd.unique(sel0[["long","s2","s3"]].values.ravel())
    depth=load_depth(syms)
    sel,n0,nk=attach_depth(sel0,depth)
    return sel0, sel, depth, n0, nk

def main():
    check1()
    results={}
    for era in ["RECENT","OOS"]:
        print("\n"+"#"*90+f"\n# {era}\n"+"#"*90)
        sel0,sel,depth,n0,nk=load_era(era)
        cov=sel["open_time"].agg(["min","max"])
        paper=sharpe(sel["paper"].values, sel["open_time"])
        print(f"bars total={n0} with-PIT-depth={nk} ({nk/n0*100:.0f}%) "
              f"window {cov['min'].date()}..{cov['max'].date()}  PAPER Sharpe={paper:+.3f}")
        grid=gating_grid(sel); print_grid(era,grid); results[era]=(sel,depth,grid,paper)

    # CHECK 4: name composition + bootstrap at the headline RECENT corner and OOS equiv
    print("\n"+"="*90+"\nCHECK 4  ROBUSTNESS of the RECENT +1.6 corner ($10k, 0.1%)\n"+"="*90)
    sel,depth,grid,paper=results["RECENT"]
    kb,vc=name_composition(sel,10_000,0.001)
    print(f"RECENT $10k/0.1%: {kb} bars survive; {len(vc)} distinct surviving-leg names")
    print("top-12 surviving legs:\n", vc.head(12).to_string())
    print(f"top-3 names = {vc.head(3).sum()}/{vc.sum()} leg-slots ({vc.head(3).sum()/vc.sum()*100:.0f}%); "
          f"top-8 = {vc.head(8).sum()/vc.sum()*100:.0f}%")
    g=gated(sel,10_000,0.001); m=~np.isnan(g)
    ci,ppos=block_bootstrap_ci(g[m],sel["open_time"][m])
    print(f"RECENT $10k/0.1% gated Sharpe daily-block-bootstrap: median {ci[1]:+.2f} "
          f"95%CI [{ci[0]:+.2f}, {ci[2]:+.2f}]  P(Sharpe>0)={ppos:.2f}")
    # also $25k/0.2% (the other headline)
    g2=gated(sel,25_000,0.002); m2=~np.isnan(g2)
    ci2,ppos2=block_bootstrap_ci(g2[m2],sel["open_time"][m2])
    print(f"RECENT $25k/0.2% gated Sharpe bootstrap: median {ci2[1]:+.2f} "
          f"95%CI [{ci2[0]:+.2f}, {ci2[2]:+.2f}]  P>0={ppos2:.2f}")

    # OOS best corner bootstrap
    sel_o,depth_o,grid_o,paper_o=results["OOS"]
    best=None
    for S,row in grid_o.items():
        for X in [0.003,0.002,0.001]:
            sh,frac=row[f"g{X}"]
            if frac>0.05 and (best is None or sh>best[0]): best=(sh,S,X,frac)
    print(f"\nOOS best gating corner: Sharpe {best[0]:+.2f} at S=${best[1]/1e3:.0f}k thr={best[2]*100:.1f}% ({best[3]*100:.0f}% bars)")
    gob=gated(sel_o,best[1],best[2]); mob=~np.isnan(gob)
    cio,pposo=block_bootstrap_ci(gob[mob],sel_o["open_time"][mob])
    print(f"OOS best-corner bootstrap: median {cio[1]:+.2f} 95%CI [{cio[0]:+.2f},{cio[2]:+.2f}] P>0={pposo:.2f}")

    # CHECK 5: OOS survivorship
    print("\n"+"="*90+"\nCHECK 5  OOS SURVIVORSHIP (are covered bars biased to liquid names?)\n"+"="*90)
    for era in ["RECENT","OOS"]:
        sel0,sel,depth,n0,nk=load_era(era)
        print(f"{era}:")
        survivorship(sel0, sel, depth)

if __name__=="__main__":
    main()
