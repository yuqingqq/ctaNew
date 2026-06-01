"""LONG-PRED iter-005 — per-sym signal decay map

Tests hypothesis #4: maybe a subset of syms still have working per-sym Ridge
models in H2 (model predicts their own forward residual correctly), while others
have decayed. If we restrict universe to "still-working" syms, the H2 fresh-signal
edge might recover from -0.40 to positive.

(a) Per-sym Spearman IC (pred → return_pct) in H1 and H2 separately
(b) Classify: working (IC>0.02), broken (<-0.02), noise (|IC|<0.02) — in BOTH H1 and H2
(c) In-sample H2 check: top-K=5 edge restricted to H2-working syms only
(d) OOS deploy test: classify on H1 only, test if H1-working syms still beat in H2
    (the honest forward-realistic test)
(e) If H1-working AND H2-working overlap is meaningful, run bot replay with that allowlist
"""
import sys, time, os, subprocess
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
S = REPO/"live/state/convexity"
H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))

def sharpe(p_bps):
    p = p_bps/1e4
    return p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float("nan")

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-005: per-sym signal decay map ===\n", flush=True)

    print("loading preds...", flush=True)
    d = pd.read_parquet(PREDS)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]

    # (a) per-sym IC in H1 and H2
    print("computing per-sym IC...", flush=True)
    rows = []
    for sym in d["symbol"].unique():
        d_sym = d[d["symbol"]==sym]
        for label,(s,e) in [("h1",H1),("h2",H2)]:
            sub = d_sym[(d_sym["open_time"]>=s)&(d_sym["open_time"]<e)].dropna(subset=["pred","return_pct"])
            if len(sub) < 30: continue
            ic, _ = spearmanr(sub["pred"], sub["return_pct"])
            rows.append(dict(sym=sym, period=label, ic=ic, n=len(sub)))
    ic_df = pd.DataFrame(rows).pivot(index="sym", columns="period", values="ic").reset_index()
    ic_df.columns.name = None
    print(f"\nper-sym IC summary:")
    print(f"  H1 mean {ic_df['h1'].mean():+.4f}  median {ic_df['h1'].median():+.4f}")
    print(f"  H2 mean {ic_df['h2'].mean():+.4f}  median {ic_df['h2'].median():+.4f}")

    # (b) classify
    def classify(ic):
        if ic > 0.02: return "working"
        if ic < -0.02: return "broken"
        return "noise"
    ic_df["cls_h1"] = ic_df["h1"].apply(classify)
    ic_df["cls_h2"] = ic_df["h2"].apply(classify)
    print(f"\nH1 classification: {ic_df['cls_h1'].value_counts().to_dict()}")
    print(f"H2 classification: {ic_df['cls_h2'].value_counts().to_dict()}")
    # transition matrix
    print(f"\nH1 → H2 transition matrix:")
    print(pd.crosstab(ic_df["cls_h1"], ic_df["cls_h2"]).to_string())

    h2_working = set(ic_df[ic_df["cls_h2"]=="working"]["sym"])
    h1_working = set(ic_df[ic_df["cls_h1"]=="working"]["sym"])
    overlap = h1_working & h2_working
    print(f"\nH2 working syms: {len(h2_working)}")
    print(f"H1 working syms: {len(h1_working)}")
    print(f"OVERLAP (H1-working AND H2-working): {len(overlap)} syms")
    print(f"H1-working stays-working in H2: {len(overlap)/max(1,len(h1_working))*100:.0f}%")
    print(f"Random baseline (if independent): {len(h2_working)/len(ic_df)*100:.0f}%")

    # (c) in-sample H2 check: top-K edge restricted to H2-working syms
    print(f"\n=== (c) IN-SAMPLE H2 edge — restrict universe to H2-working syms only ===")
    h2_sub = d[(d["open_time"]>=H2[0])&(d["open_time"]<H2[1])]
    h2_restricted = h2_sub[h2_sub["symbol"].isin(h2_working)]
    for k in [1, 3, 5]:
        edges_all = []; edges_restr = []
        for ot, g in h2_sub.groupby("open_time"):
            if len(g) < 2*k: continue
            g = g.sort_values("pred"); m = g["return_pct"].median()
            edges_all.append(g.tail(k)["return_pct"].mean() - m)
        for ot, g in h2_restricted.groupby("open_time"):
            if len(g) < 2*k: continue
            g = g.sort_values("pred"); m = g["return_pct"].median()
            edges_restr.append(g.tail(k)["return_pct"].mean() - m)
        ea = np.mean(edges_all)*1e4 if edges_all else float("nan")
        er = np.mean(edges_restr)*1e4 if edges_restr else float("nan")
        print(f"  K={k}: full universe top-K edge vs median = {ea:+.2f} bps  "
              f"H2-working only = {er:+.2f} bps  Δ {er-ea:+.2f}")

    # (d) OOS deploy test: classify on H1 ONLY, test in H2
    print(f"\n=== (d) OOS DEPLOY TEST — classify by H1 IC, test in H2 (honest) ===")
    for k in [1, 3, 5]:
        edges_all = []; edges_h1w = []
        for ot, g in h2_sub.groupby("open_time"):
            if len(g) < 2*k: continue
            g = g.sort_values("pred"); m = g["return_pct"].median()
            edges_all.append(g.tail(k)["return_pct"].mean() - m)
            g_h1w = g[g["symbol"].isin(h1_working)]
            if len(g_h1w) >= 2*k:
                g_h1w = g_h1w.sort_values("pred"); m2 = g_h1w["return_pct"].median()
                edges_h1w.append(g_h1w.tail(k)["return_pct"].mean() - m2)
        ea = np.mean(edges_all)*1e4 if edges_all else float("nan")
        eh = np.mean(edges_h1w)*1e4 if edges_h1w else float("nan")
        print(f"  K={k}: full universe top-K vs median = {ea:+.2f} bps  "
              f"H1-working only = {eh:+.2f} bps  Δ {eh-ea:+.2f}")

    # (e) if H1-working subset has meaningfully better H2 edge, run bot replay
    h1_working_str = ",".join(sorted(h1_working))
    print(f"\n=== (e) Bot replay on H1-working subset (OOS-honest filter) ===")
    print(f"H1-working subset size: {len(h1_working)} syms")
    if len(h1_working) < 30:
        print("  too few syms in H1-working subset to run a meaningful replay (need ≥30); skipping bot")
    else:
        env = os.environ.copy()
        env.update({"PYTHONPATH":str(REPO),"BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3",
                    "SIDE_MODE":"default","STRAT_HOLD":"6","COST_BPS_LEG":"4.5",
                    "SYM_ALLOWLIST": h1_working_str})
        res = subprocess.run(["python3","-m","live.convexity_paper_bot",
                              "--replay-from","2025-10-04","--replay-end","2026-05-26"],
                             capture_output=True, text=True, env=env)
        if res.returncode != 0:
            print(f"  REPLAY FAILED: {res.stderr[-500:]}")
        else:
            c = pd.read_csv(S/"cycles.csv"); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
            h2c = c[(c["open_time"]>=H2[0])&(c["open_time"]<=H2[1])]
            h1c = c[c["open_time"]<H2[0]]
            print(f"\n  RESTRICTED REPLAY (cost=4.5, HOLD=6, H1-working subset):")
            print(f"    full OOS Sh {sharpe(c['pnl_bps']):+.3f}  totPnL {int(c['pnl_bps'].sum()):+d}")
            print(f"    H1 Sh {sharpe(h1c['pnl_bps']):+.3f}")
            print(f"    H2 Sh {sharpe(h2c['pnl_bps']):+.3f}")
            print(f"    avg n_universe {c['n_universe'].mean():.0f}")
            print(f"  vs baseline (full universe): full +1.30, H1 +2.70, H2 -2.57")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
