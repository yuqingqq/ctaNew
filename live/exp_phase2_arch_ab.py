"""Phase 2 architectural A/B: short_btc_hedge construction × hl=14d recency.

Tests 4 variants on FULL OOS (Oct04 → May26), with both PREDICTION QUALITY metrics
(IC, leg edge, hit rate) and downstream Sharpe — so we can attribute lift to
prediction improvement vs construction improvement separately.

  A: static preds + default 5L/5S beta-neutral  (current spec)
  B: hl=14d preds + default 5L/5S beta-neutral  (recency only)
  C: static preds + short_btc_hedge             (construction only)
  D: hl=14d preds + short_btc_hedge             (both)
"""
import subprocess, sys, os, json, time
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
S = REPO/"live/state/convexity"
PREDS_DIR = REPO/"research/convexity_portable_2026-05-20/results/_cache"

STATIC_PREDS = PREDS_DIR/"x132_expanded_v0_preds.parquet"

# hl=14 was trained through 2026-01-22 — too short to test on full OOS without overlap.
# Build a NEW hl=14 trained through 2025-10-02 (= original training cutoff) so the OOS
# window 2025-10-04 → 2026-05-26 is honest OOS for both static and hl=14d.
HL14_TAG = "p2_hl14_full"
HL14_ARTIFACT = REPO/f"live/models/convexity_portable_{HL14_TAG}.pkl"
HL14_PREDS = S/f"x132_{HL14_TAG}_fullOOS_preds.parquet"

def run(cmd, label):
    env = os.environ.copy(); env["PYTHONPATH"] = str(REPO)
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"  !!! {label} FAILED: {res.stderr[-500:]}", flush=True); return False
    return True

def pred_quality_metrics(preds_path, period_start, period_end, label):
    """Compute prediction-quality metrics for a preds file on a given period."""
    d = pd.read_parquet(preds_path)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]
    d = d[(d["open_time"]>=pd.Timestamp(period_start,tz="UTC"))&(d["open_time"]<=pd.Timestamp(period_end,tz="UTC"))]
    rows = []
    for ot, g in d.groupby("open_time"):
        if len(g) < 15: continue
        g = g.sort_values("pred")
        ic, _ = spearmanr(g["pred"], g["return_pct"])
        # per-K edge
        out = dict(n=len(g), ic_xs=ic)
        for k in [1, 3, 5]:
            if len(g) < 2*k: continue
            out[f"top{k}_edge"] = g.tail(k)["return_pct"].mean() - g["return_pct"].mean()
            out[f"bot{k}_edge"] = g["return_pct"].mean() - g.head(k)["return_pct"].mean()
            out[f"spread{k}"] = g.tail(k)["return_pct"].mean() - g.head(k)["return_pct"].mean()
        rows.append(out)
    df = pd.DataFrame(rows).dropna()
    summary = dict(
        label=label, n_cycles=len(df),
        ic_xs_mean=df["ic_xs"].mean(),
        ic_pct_positive=100*(df["ic_xs"]>0).mean(),
    )
    for k in [1,3,5]:
        summary[f"top{k}_edge_bps"] = df[f"top{k}_edge"].mean()*1e4
        summary[f"bot{k}_edge_bps"] = df[f"bot{k}_edge"].mean()*1e4
        summary[f"spread{k}_bps"] = df[f"spread{k}"].mean()*1e4
    return summary

def replay(preds_path, side_mode, label):
    env = os.environ.copy()
    env.update({"PYTHONPATH":str(REPO), "CONVEXITY_PREDS_PATH":str(preds_path),
                "BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3","SIDE_MODE":side_mode})
    res = subprocess.run(["python3","-m","live.convexity_paper_bot",
                          "--replay-from","2025-10-04","--replay-end","2026-05-26"],
                         capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"  REPLAY FAILED {label}: {res.stderr[-300:]}"); return None
    import shutil
    out = S/f"phase2_{label}.csv"; shutil.copy(S/"cycles.csv", out)
    return out

def cycles_stats(path, label):
    c = pd.read_csv(path); c["open_time"] = pd.to_datetime(c["open_time"],utc=True)
    p = c["pnl_bps"]/1e4
    sh = p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float("nan")
    cum = pd.Series(c["pnl_bps"]).cumsum(); dd = (cum-cum.cummax()).min()
    # H1 vs H2 split
    h1 = c[c["open_time"]<pd.Timestamp("2026-01-22",tz="UTC")]
    h2 = c[c["open_time"]>=pd.Timestamp("2026-01-22",tz="UTC")]
    def sh_(x):
        p = x["pnl_bps"]/1e4
        return p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float("nan")
    return dict(label=label, n=len(c),
                Sharpe=round(sh,3), totPnL=int(c["pnl_bps"].sum()), maxDD=int(dd),
                H1_Sharpe=round(sh_(h1),3), H2_Sharpe=round(sh_(h2),3),
                stop_pct=round(100*c["stop_engaged"].mean(),1))

def main():
    t0 = time.time()
    print(f"=== PHASE 2 ARCHITECTURAL A/B ===\n", flush=True)

    # Build hl=14d artifact trained through 2025-10-02 (= original artifact training cutoff)
    if not HL14_ARTIFACT.exists():
        print(f"--- training hl=14d artifact (full-OOS-honest) ---", flush=True)
        ok = run(["python3", str(REPO/"live/train_convexity_artifact.py"),
                  "--train-end", "2025-10-02", "--tag", HL14_TAG, "--halflife-days", "14"],
                 "train hl14 full")
        if not ok: sys.exit(2)
        # Generate preds on full OOS
        ok = run(["python3", str(REPO/"live/predict_with_artifact.py"),
                  "--artifact", HL14_TAG, "--from", "2025-10-04", "--to", "2026-05-26",
                  "--out-tag", f"{HL14_TAG}_fullOOS"],
                 "predict hl14 fullOOS")
        if not ok: sys.exit(2)
    else:
        print(f"--- hl=14d artifact already exists, reusing ---", flush=True)

    # === Prediction quality comparison ===
    print(f"\n=== PREDICTION QUALITY (pred quality is the strategy's foundation) ===")
    print(f"\n--- FULL OOS (2025-10-04 → 2026-05-26) ---")
    static_qual = pred_quality_metrics(STATIC_PREDS, "2025-10-04", "2026-05-26", "STATIC")
    hl14_qual = pred_quality_metrics(HL14_PREDS, "2025-10-04", "2026-05-26", "HL=14d")
    qual_df = pd.DataFrame([static_qual, hl14_qual])
    print(qual_df.round(3).to_string(index=False))

    for period_label, ps, pe in [
        ("H1 (Oct→Jan22)", "2025-10-04", "2026-01-22"),
        ("H2 (Jan22→May26)", "2026-01-22", "2026-05-26"),
    ]:
        print(f"\n--- {period_label} ---")
        s = pred_quality_metrics(STATIC_PREDS, ps, pe, "STATIC")
        h = pred_quality_metrics(HL14_PREDS, ps, pe, "HL=14d")
        print(pd.DataFrame([s,h]).round(3).to_string(index=False))

    # === Architectural A/B (4 variants) ===
    print(f"\n\n=== ARCHITECTURAL A/B — 4 variants, full OOS ===\n")
    variants = [
        ("A_static_default",        STATIC_PREDS, "default"),
        ("B_hl14_default",          HL14_PREDS,   "default"),
        ("C_static_short_hedge",    STATIC_PREDS, "short_btc_hedge"),
        ("D_hl14_short_hedge",      HL14_PREDS,   "short_btc_hedge"),
    ]
    results = []
    for label, preds, mode in variants:
        print(f"--- replay {label} ---", flush=True)
        out = replay(preds, mode, label)
        if out: results.append(cycles_stats(out, label))
    print(f"\n=== SHARPE COMPARISON ===")
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__ == "__main__": main()
