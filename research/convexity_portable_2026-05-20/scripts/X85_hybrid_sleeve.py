"""X85 — Hybrid regime-switched strategy on the V3.1 SLEEVE (held-book cost).

The make-or-break deployability test. Scoping used harsh full-rotation 4h cost
(9-18 bps/cycle) which kills the mean-rev core. The real engine is the V3.1
overlapping sleeve (6 sleeves × 24h hold) which amortizes turnover to ≤~9 bps.
This tests whether the hybrid clears cost on the REAL book.

Hybrid prediction series (fed to the sleeve):
  - BULL cycle (BTC trailing-30d > +0.10): pred = cross-sectional z of sym mom_30d
    (trend-following: long high-momentum / short low-momentum)
  - else (sideways/bear): pred = V0 mean-rev model prediction
Both legs z-scored per cycle so the sleeve sees comparable magnitudes.

Variants compared (net Sharpe via sleeve, 12mo + 3yr):
  A. V0 mean-rev everywhere (baseline)
  B. V0 in non-bull + FLAT in bull (current best)
  C. HYBRID: V0 in non-bull + mom_30d trend-follow in bull
"""
from __future__ import annotations
import sys, importlib.util, time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scripts"))
OUT = REPO/"research/convexity_portable_2026-05-20/results"; RCACHE = OUT/"_cache"
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
    return s.groupby(level=0).transform(lambda x: (x-x.mean())/(x.std()+1e-9))


def run(label, apd):
    p = RCACHE/f"x85_{label}_preds.parquet"; apd.to_parquet(p, index=False)
    m = x6.run_sleeve_on_preds(p, f"x85_{label}")
    return m


def summarize(apd, syms, tag):
    """Run the 3 variants on a given apd (already has v0_pred, mom30_z, bull flag)."""
    res={}
    # A. V0 everywhere
    a=apd.copy(); a["pred"]=a["v0_pred"]
    res["A_v0_everywhere"]=run(f"{tag}_A", a[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]])
    # B. V0 non-bull + flat bull (drop bull rows so sleeve holds nothing there)
    b=apd[~apd["bull"]].copy(); b["pred"]=b["v0_pred"]
    res["B_flat_bull"]=run(f"{tag}_B", b[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]])
    # C. HYBRID: V0 non-bull + mom30 trend-follow bull
    c=apd.copy()
    c["pred"]=np.where(c["bull"], c["mom30_z"], c["v0_pred_z"])
    res["C_hybrid"]=run(f"{tag}_C", c[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]])
    return res


def main():
    t0=time.time()
    print("=== X85 hybrid regime-switched sleeve backtest ===\n", flush=True)
    apd=pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
    apd["exit_time"]=pd.to_datetime(apd["exit_time"],utc=True)
    syms=sorted(apd["symbol"].unique())
    print(f"V0 preds: {len(apd):,} rows, {len(syms)} syms")

    # mom_30d per sym (PIT, 5m→ shift), aligned to pred timestamps
    print("computing mom_30d per sym...", flush=True)
    mom_rows=[]
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        mom=(c/c.shift(8640)-1).shift(1)  # 30d trailing, PIT
        mom_rows.append(pd.DataFrame({"symbol":sym,"open_time":mom.index,"mom30":mom.values}))
    mom=pd.concat(mom_rows,ignore_index=True); mom["open_time"]=pd.to_datetime(mom["open_time"],utc=True)
    apd=apd.merge(mom,on=["symbol","open_time"],how="left")

    # BTC 30d regime (PIT) — align 4h btc30 to 5m pred grid via merge_asof (backward = PIT)
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc30=(b4/b4.shift(180)-1).to_frame("btc_ret_30d").reset_index()
    btc30["open_time"]=pd.to_datetime(btc30["open_time"],utc=True)
    btc30["open_time"]=btc30["open_time"].astype("datetime64[ns, UTC]")
    apd["open_time"]=apd["open_time"].astype("datetime64[ns, UTC]")
    apd["exit_time"]=apd["exit_time"].astype("datetime64[ns, UTC]")
    btc30=btc30.sort_values("open_time")
    apd=apd.sort_values("open_time")
    apd=pd.merge_asof(apd, btc30, on="open_time", direction="backward")
    apd["bull"]=apd["btc_ret_30d"]>0.10

    # per-cycle z-scores
    apd=apd.set_index("open_time")
    apd["v0_pred_z"]=per_cycle_z(apd["pred"])
    apd["mom30_z"]=per_cycle_z(apd["mom30"].fillna(0))
    apd["v0_pred"]=apd["pred"]
    apd=apd.reset_index()
    apd=apd.dropna(subset=["alpha_A","return_pct"])

    bull_frac=apd.groupby("open_time")["bull"].first().mean()
    print(f"bull cycle fraction (3yr): {bull_frac*100:.1f}%\n")

    # 3yr
    print("--- 3-YEAR (sleeve, held book) ---", flush=True)
    r3=summarize(apd, syms, "3yr")
    for k,m in r3.items():
        print(f"  {k:<20} Sharpe={m.get('sharpe',0):+.2f} folds={m.get('folds_pos','?')} conc={m.get('concentration','?')} PnL={m.get('totPnL','?')}", flush=True)

    # recent 12mo
    print("\n--- RECENT 12mo (open_time >= 2025-05-01) ---", flush=True)
    apd12=apd[apd["open_time"]>=pd.Timestamp("2025-05-01",tz="UTC")].copy()
    # re-fold for the 12mo subset so sleeve folds are sane
    r12=summarize(apd12, syms, "12mo")
    for k,m in r12.items():
        print(f"  {k:<20} Sharpe={m.get('sharpe',0):+.2f} folds={m.get('folds_pos','?')} conc={m.get('concentration','?')} PnL={m.get('totPnL','?')}", flush=True)

    print(f"\nVERDICT: does HYBRID (C) beat FLAT-in-bull (B) and v0-everywhere (A) NET on the")
    print(f"held sleeve book, on BOTH 3yr and recent-12mo? If C>B robustly → bull overlay deployable.")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
