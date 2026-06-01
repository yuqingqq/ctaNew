"""X118 — Shorter-horizon test (#4): is the alpha stronger at 1h/2h than 4h?

Current strategy targets 4h-forward alpha-residual (HORIZON=48 5m-bars). Rebuild the V0
target at HORIZON in {12 (1h), 24 (2h), 48 (4h baseline)} on the 44-sym universe, retrain V0
walk-forward, and compare per-cycle cross-sec IC + a matched held-book Sharpe. Reuses cached
xs_feats + funding (horizon-independent); only target/return_pct/exit_time change.

Held-book matched to horizon: entry cadence = horizon, hold = 6 sleeves (so 1h→6h, 2h→12h,
4h→24h), cost 4.5bps/leg. Annualization uses the horizon's cycles/year.
"""
from __future__ import annotations
import time, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
spec=importlib.util.spec_from_file_location("x70mod", REPO/"research/convexity_portable_2026-05-20/scripts/X70_build_3yr_and_regime_test.py")
X70=importlib.util.module_from_spec(spec); spec.loader.exec_module(X70)
x6,x6b=X70.x6,X70.x6b
RC=REPO/"research/convexity_portable_2026-05-20/results/_cache"
KLINES=REPO/"data/ml/test/parquet/klines"
COST=4.5e-4; K=5; HOLD=6
CANDS=sorted(pd.read_parquet(REPO/"outputs/vBTC_features/panel_3yr_v0.parquet",columns=["symbol"])["symbol"].unique())


def ann_h(x, bars):  # bars = 5m-bars per cycle; cycles/yr = 365*288/bars
    x=pd.Series(x).dropna(); cyc=365*288/bars
    return x.mean()/x.std()*np.sqrt(cyc) if len(x)>2 and x.std()>0 else np.nan


def build_for_horizon(H, btc_close):
    """Rebuild per-sym panel rows with target at horizon H (reuse cached xs_feats + funding)."""
    X70.HORIZON=H  # monkeypatch so target_alpha + build_sym use H
    sdfs=[]
    for sym in CANDS:
        if sym=="BTCUSDT": continue
        try:
            sdf=X70.build_sym(sym, btc_close)
            if sdf is not None and len(sdf)>0: sdfs.append(sdf)
        except Exception: pass
    panel=pd.concat(sdfs,ignore_index=True)
    panel["open_time"]=pd.to_datetime(panel["open_time"],utc=True)
    panel["exit_time"]=pd.to_datetime(panel["exit_time"],utc=True)
    panel=panel.dropna(subset=["alpha_vs_btc_realized"])
    panel=x6b.build_cohort_fixed(panel)
    x6.HORIZON=H; panel=x6.build_target_z(panel)
    x6.HEAVY_TAIL.discard("rvol_7d"); x6.HEAVY_TAIL.discard("ret_3d"); x6.HEAVY_TAIL.discard("btc_rvol_7d")
    panel["bars_since_high_xs_rank"]=panel.groupby("open_time")["bars_since_high"].rank(pct=True).astype("float32")
    return panel


def heldbook(d, cadence_bars):
    """entry every `cadence_bars` 5m-bars; hold 6 sleeves; mean-rev long-top-pred."""
    d=d[(d["open_time"].dt.hour*12+d["open_time"].dt.minute//5) % cadence_bars==0]  # subsample to cadence
    times=sorted(d["open_time"].unique()); by_t={ot:g for ot,g in d.groupby("open_time")}
    ws=[]
    for ot in times:
        g=by_t[ot].dropna(subset=["pred"])
        if len(g)<2*K: ws.append({}); continue
        gg=g.sort_values("pred"); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
        w={}
        for s in L: w[s]=w.get(s,0)+1.0/K
        for s in S: w[s]=w.get(s,0)-1.0/K
        ws.append(w)
    prev={}; pnl=[]
    for t in range(len(ws)):
        active=ws[max(0,t-HOLD+1):t+1]; net={}
        for w in active:
            for s,wt in w.items(): net[s]=net.get(s,0)+wt/HOLD
        alls=set(net)|set(prev); turn=sum(abs(net.get(s,0)-prev.get(s,0)) for s in alls)
        rl=by_t[times[t]]; rmap=dict(zip(rl["symbol"],rl["return_pct"]))
        pnl.append(sum(net.get(s,0)*rmap.get(s,0.0) for s in net if np.isfinite(rmap.get(s,0.0)))-turn*0.5*COST); prev=net
    return np.array(pnl)


def main():
    t0=time.time()
    print("=== X118 shorter-horizon test (1h/2h/4h) ===\n", flush=True)
    btc=X70.load_closes("BTCUSDT")
    print(f"  {'H(5m-bars)':<12}{'horizon':>8}{'#preds':>10}{'meanIC':>9}{'heldbook Sh':>13}{'totPnL':>9}", flush=True)
    for H in [12,24,48]:
        panel=build_for_horizon(H, btc)
        folds=x6.get_folds(panel)
        feats=[f for f in x6.BASE+x6.COHORT_EXTRAS if f in panel.columns]
        apd=x6.train_per_sym_ridge(panel, folds, feats, label=f"x118_h{H}")
        apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
        # per-cycle cross-sec IC (at horizon cadence to avoid overlap inflation)
        sub=apd[(apd["open_time"].dt.hour*12+apd["open_time"].dt.minute//5)%H==0]
        ics=sub.groupby("open_time").apply(lambda g: g["pred"].corr(g["alpha_A"],method="spearman") if len(g.dropna(subset=["pred","alpha_A"]))>=6 else np.nan)
        meanic=ics.mean()
        p=heldbook(apd, H)
        print(f"  {H:<12}{H*5:>6}m{len(apd):>10}{meanic:>+9.4f}{ann_h(p,H):>+13.2f}{p.sum()*1e4:>+9.0f}", flush=True)
    print(f"\nREAD: higher meanIC / Sharpe at shorter H = shorter-horizon alpha stronger. Done [{time.time()-t0:.0f}s]")


if __name__=="__main__":
    main()
