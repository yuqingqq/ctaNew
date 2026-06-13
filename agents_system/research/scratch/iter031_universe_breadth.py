"""iter-031 — DEPLOYMENT decision: rule-based deploy universe for the HL70 champion.

Runs the FULL champion (baseline regime-hybrid held-book + iter-012 vol-norm reactive stop, k=2.0)
on:
  (1) liquidity ranking from 5m-klines quote_volume (median daily $vol),
  (2) top-N-by-liquidity subsets (N=20,30,40,50,70) on HL70 (production) and EXT (transport),
  (3) random-subset composition stress (drop k random names, ~25 draws) at recommended N and full,
  (4) liquidity-top-N vs random-N vs full comparison (is liquidity selection >= random / full?).

Reuses VERBATIM: X123.build_universe (preds+klines pipeline, X117 base book) and X125's
run_volnorm_heldbook (the champion-with-stop engine), gross_unit (base book, gross=1), metrics.
To subset the universe we filter the preds frame to a symbol whitelist by writing a temp preds
parquet and calling build_universe on it (so the per-cycle book is RE-RANKED within the subset —
the correct way to simulate deploying on a smaller universe).
"""
from __future__ import annotations
import sys, time, json, tempfile
from pathlib import Path
import numpy as np, pandas as pd
import importlib.util as _ilu

REPO=Path("/home/yuqing/ctaNew")
SCRIPTS=REPO/"research/convexity_portable_2026-05-20/scripts"
RC=REPO/"research/convexity_portable_2026-05-20/results/_cache"
OUT=REPO/"outputs/iter031"; OUT.mkdir(parents=True, exist_ok=True)

_s123=_ilu.spec_from_file_location("x123", SCRIPTS/"X123_altbear_short_probe.py")
x123=_ilu.module_from_spec(_s123); _s123.loader.exec_module(x123)
_s125=_ilu.spec_from_file_location("x125", SCRIPTS/"X125_volnorm_stop.py")
x125=_ilu.module_from_spec(_s125); _s125.loader.exec_module(x125)

build_universe=x123.build_universe
gross_unit=x125.gross_unit
metrics=x125.metrics
run_volnorm=x125.run_volnorm_heldbook
REC_K=x125.REC_K            # 2.0
PRIMARY_COST=x125.PRIMARY_COST  # 4.5e-4
GFLOOR=x125.GFLOOR

HL70_PREDS=x123.HL70_PREDS
EXT_PREDS=x123.EXT_PREDS

SEED=12345
N_DRAWS=25                  # random-subset composition draws

# ----- liquidity ranking (median daily quote_volume from 5m klines) -----
KLINES=REPO/"data/ml/test/parquet/klines"
def daily_qvol_med(sym, start, end):
    sd=KLINES/sym/"5m"
    if not sd.exists(): return np.nan, 0
    files=[f for f in sorted(sd.glob("*.parquet")) if start<=f.stem<=end]
    if not files: return np.nan, 0
    dfs=[pd.read_parquet(f, columns=["open_time","quote_volume"]) for f in files]
    df=pd.concat(dfs, ignore_index=True)
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    dv=df.groupby(df["open_time"].dt.date)["quote_volume"].sum()
    if len(dv)<10: return np.nan, len(dv)
    return float(dv.median()), int(len(dv))

def liquidity_rank(syms, start, end):
    rows=[]
    for s in syms:
        med,nd=daily_qvol_med(s,start,end)
        rows.append({"symbol":s,"med_dayvol_usd":med,"ndays":nd})
    r=pd.DataFrame(rows).sort_values("med_dayvol_usd", ascending=False).reset_index(drop=True)
    return r

# ----- run champion (base + stop) on a symbol subset -----
_tmpfiles=[]
def build_subset(preds_path, label, syms):
    """Filter preds to `syms`, write temp parquet, build_universe (re-ranks within subset)."""
    cols=["symbol","open_time","pred","return_pct","fold"]
    d=pd.read_parquet(preds_path, columns=cols)
    d=d[d["symbol"].isin(syms)].copy()
    tf=tempfile.NamedTemporaryFile(suffix=".parquet", delete=False, dir="/tmp")
    d.to_parquet(tf.name, index=False); _tmpfiles.append(tf.name)
    return build_universe(Path(tf.name), f"{label}_N{len(syms)}")

def champion_metrics(U, cost=PRIMARY_COST, k=REC_K):
    """Return (base_metrics, stop_metrics) for the champion held-book."""
    base=gross_unit(U["cyc"]["base"], U["rs"], cost)*1e4
    bm=metrics(base)
    pnl,gross,stop,rt,_=run_volnorm(U["cyc"]["base"], U["rs"], cost, k)
    sm=metrics(pnl*1e4); sm["pct_stop"]=float(stop.mean()*100); sm["rt"]=int(rt)
    return bm, sm

def fmt(m):
    return (f"Sh {m['Sharpe']:+.2f} maxDD {m['maxDD']:+.0f} Calmar {m['Calmar']:+.2f} "
            f"totPnL {m['tot']:+.0f}")

def main():
    t0=time.time()
    rng=np.random.default_rng(SEED)
    hl70=sorted(pd.read_parquet(HL70_PREDS, columns=["symbol"])["symbol"].unique())
    ext=sorted(pd.read_parquet(EXT_PREDS, columns=["symbol"])["symbol"].unique())

    print("="*100); print("STEP 1 — LIQUIDITY RANKING (median daily $quote_volume, 5m klines)"); print("="*100, flush=True)
    lr_hl=liquidity_rank(hl70, "2025-03-30","2026-05-10")
    lr_ext=liquidity_rank(ext, "2021-08-03","2026-05-06")
    lr_hl.to_csv(OUT/"liq_rank_hl70.csv", index=False)
    lr_ext.to_csv(OUT/"liq_rank_ext.csv", index=False)
    print(f"  HL70: {len(lr_hl)} syms; top5 {lr_hl.symbol.head(5).tolist()}; "
          f"bottom5 {lr_hl.symbol.tail(5).tolist()}")
    print(f"  EXT:  {len(lr_ext)} syms; top5 {lr_ext.symbol.head(5).tolist()}; "
          f"bottom5 {lr_ext.symbol.tail(5).tolist()}", flush=True)

    results=[]

    # ---------------- STEP 2: breadth-N sweep (top-N by liquidity) ----------------
    print("\n"+"="*100); print("STEP 2 — BREADTH-N SWEEP (top-N by liquidity)  [champion = base + iter-012 stop k=2.0]"); print("="*100, flush=True)
    for univ, lr, preds, Ns in [("HL70", lr_hl, HL70_PREDS, [20,30,40,50,70]),
                                 ("EXT",  lr_ext, EXT_PREDS, [10,15,20,23])]:
        print(f"\n--- {univ} ---", flush=True)
        print(f"{'N':>4} {'base_Sh':>8}{'base_DD':>9}{'base_Cal':>9}  | {'stop_Sh':>8}{'stop_DD':>9}{'stop_Cal':>9}{'%stop':>7}", flush=True)
        for N in Ns:
            if N>len(lr): continue
            syms=lr.symbol.head(N).tolist()
            U=build_subset(preds, univ, syms)
            bm,sm=champion_metrics(U)
            print(f"{N:>4} {bm['Sharpe']:>8.2f}{bm['maxDD']:>9.0f}{bm['Calmar']:>9.2f}  | "
                  f"{sm['Sharpe']:>8.2f}{sm['maxDD']:>9.0f}{sm['Calmar']:>9.2f}{sm['pct_stop']:>7.1f}", flush=True)
            results.append(dict(univ=univ, kind="liq_topN", N=N,
                base_Sharpe=bm["Sharpe"], base_maxDD=bm["maxDD"], base_Calmar=bm["Calmar"], base_totPnL=bm["tot"],
                stop_Sharpe=sm["Sharpe"], stop_maxDD=sm["maxDD"], stop_Calmar=sm["Calmar"], stop_totPnL=sm["tot"],
                pct_stop=sm["pct_stop"]))

    # ---------------- STEP 3: composition stress (random subsets) ----------------
    # at recommended N (we'll use N=50 for HL70 and N=full=70; for EXT N=full=23 and N=15)
    print("\n"+"="*100); print(f"STEP 3 — COMPOSITION STRESS: {N_DRAWS} random N-subset draws (base book @4.5bps)"); print("="*100, flush=True)
    def random_stress(univ, preds, full_syms, N):
        bs=[]
        for i in range(N_DRAWS):
            pick=sorted(rng.choice(full_syms, size=N, replace=False).tolist())
            U=build_subset(preds, f"{univ}_rand", pick)
            base=gross_unit(U["cyc"]["base"], U["rs"], PRIMARY_COST)*1e4
            m=metrics(base)
            bs.append(dict(univ=univ, N=N, draw=i, Sharpe=m["Sharpe"], maxDD=m["maxDD"],
                           Calmar=m["Calmar"], totPnL=m["tot"], syms=";".join(pick)))
        return bs
    stress_rows=[]
    for univ, preds, full, Ns in [("HL70", HL70_PREDS, hl70, [50, 40, 30]),
                                   ("EXT",  EXT_PREDS, ext, [15, 18])]:
        for N in Ns:
            print(f"\n  [{univ}] random {N}-of-{len(full)} ({N_DRAWS} draws)...", flush=True)
            sr=random_stress(univ, preds, full, N)
            stress_rows+=sr
            sh=np.array([r["Sharpe"] for r in sr]); dd=np.array([r["maxDD"] for r in sr])
            cal=np.array([r["Calmar"] for r in sr])
            print(f"    Sharpe mean {sh.mean():+.2f} std {sh.std():.2f} min {sh.min():+.2f} "
                  f"max {sh.max():+.2f} | maxDD mean {dd.mean():+.0f} worst {dd.min():+.0f} | "
                  f"Calmar mean {cal.mean():+.2f}", flush=True)
            results.append(dict(univ=univ, kind="random_subset", N=N,
                base_Sharpe=float(sh.mean()), base_Sharpe_std=float(sh.std()),
                base_Sharpe_min=float(sh.min()), base_Sharpe_max=float(sh.max()),
                base_maxDD=float(dd.mean()), base_maxDD_worst=float(dd.min()),
                base_Calmar=float(cal.mean())))
    pd.DataFrame(stress_rows).to_csv(OUT/"composition_stress.csv", index=False)

    # ---------------- STEP 4: liquidity-top-N vs random-N (already have both) ----
    # comparison printed in analysis; full-set reference rows captured above (N=70 / N=23)

    pd.DataFrame(results).to_csv(OUT/"iter031_summary.csv", index=False)
    print(f"\nSaved -> {OUT}/iter031_summary.csv, composition_stress.csv, liq_rank_*.csv")
    print(f"Done [{time.time()-t0:.0f}s]", flush=True)
    for f in _tmpfiles:
        try: Path(f).unlink()
        except Exception: pass

if __name__=="__main__":
    main()
