"""LONG-PRED iter-006 — Rolling per-sym IC filter (deploy-realistic)

iter-005 found that filtering universe to H1-working syms recovers H2 Sharpe from
-2.57 to -0.12. But H1-classification uses look-ahead (we know H1 ICs only after
H1 is done). For PRODUCTION DEPLOY we need a ROLLING per-sym IC filter — compute
each sym's trailing IC PIT and exclude bad-IC syms each cycle.

Sweep: window W ∈ {60, 90, 180 days} × threshold τ ∈ {0.0, 0.02, 0.05}
For each (W, τ): generate per-cycle allowlist parquet, run bot replay on full OOS,
compare H2 Sharpe.

ADOPT criteria: rolling-90d at τ=0 recovers H2 Sharpe ≥ -1.0 AND full Sharpe ≥ +1.0.
"""
import sys, time, os, subprocess
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
S = REPO/"live/state/convexity"
ALLOW_DIR = S/"dyn_allow"; ALLOW_DIR.mkdir(parents=True, exist_ok=True)
H1 = (pd.Timestamp("2025-10-04",tz="UTC"), pd.Timestamp("2026-01-22",tz="UTC"))
H2 = (pd.Timestamp("2026-01-22",tz="UTC"), pd.Timestamp("2026-05-26",tz="UTC"))
CYCLES_PER_DAY = 6

def sharpe(p_bps):
    p = p_bps/1e4
    return p.mean()/p.std()*np.sqrt(6*365) if p.std()>0 else float("nan")

def precompute_rolling_ic(d: pd.DataFrame, w_days: int) -> pd.Series:
    """For each (sym, open_time), compute rolling Pearson correlation of pred and
    return_pct over the trailing w_days × 6 cycles, then shift by 1 cycle so the
    IC at time t uses only data through t-1 (PIT).
    Returns a Series indexed (open_time, symbol) with the IC value.
    """
    bars = w_days * CYCLES_PER_DAY
    out_frames = []
    for sym, g in d.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)
        ic = g["pred"].rolling(bars, min_periods=bars//3).corr(g["return_pct"]).shift(1)
        df_sym = pd.DataFrame({"symbol": sym, "open_time": g["open_time"].values, "ic": ic.values})
        out_frames.append(df_sym)
    return pd.concat(out_frames, ignore_index=True)

def build_allowlist(ic_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Allow rows where rolling IC > threshold."""
    return ic_df[(ic_df["ic"] > threshold) & ic_df["ic"].notna()][["open_time","symbol"]].copy()

def replay(allowlist_path: Path, label: str) -> dict:
    env = os.environ.copy()
    env.update({"PYTHONPATH":str(REPO),"BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3",
                "SIDE_MODE":"default","STRAT_HOLD":"6","COST_BPS_LEG":"4.5",
                "CONVEXITY_DYNAMIC_ALLOWLIST_PATH":str(allowlist_path)})
    res = subprocess.run(["python3","-m","live.convexity_paper_bot",
                          "--replay-from","2025-10-04","--replay-end","2026-05-26"],
                         capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"  REPLAY FAILED {label}: {res.stderr[-300:]}"); return None
    c = pd.read_csv(S/"cycles.csv"); c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
    h2c = c[(c["open_time"]>=H2[0])&(c["open_time"]<=H2[1])]
    h1c = c[c["open_time"]<H2[0]]
    return dict(label=label, n=len(c),
                full_Sh=round(sharpe(c["pnl_bps"]),3),
                H1_Sh=round(sharpe(h1c["pnl_bps"]),3),
                H2_Sh=round(sharpe(h2c["pnl_bps"]),3),
                full_totPnL=int(c["pnl_bps"].sum()),
                H2_totPnL=int(h2c["pnl_bps"].sum()),
                avg_n_universe=round(c["n_universe"].mean(),0))

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-006: Rolling per-sym IC filter ===\n", flush=True)

    print("loading preds...", flush=True)
    d = pd.read_parquet(PREDS)
    d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    d = d[(d["open_time"].dt.hour%4==0)&(d["open_time"].dt.minute==0)]

    results = []
    for W in [60, 90, 180]:
        print(f"\n=== computing rolling IC for W={W}d ===", flush=True)
        ic_df = precompute_rolling_ic(d, W)
        # diagnostic: how many syms have IC available + positive at, say, 2026-04-01?
        snap_t = pd.Timestamp("2026-04-01", tz="UTC")
        snap = ic_df[ic_df["open_time"]==snap_t]
        if len(snap)>0:
            print(f"  W={W}d snapshot at {snap_t}: {snap['ic'].notna().sum()} have IC, "
                  f"{(snap['ic']>0).sum()} positive, {(snap['ic']>0.02).sum()} > 0.02")
        for tau in [0.0, 0.02, 0.05]:
            allowlist = build_allowlist(ic_df, tau)
            label = f"W{W}_t{tau}"
            allowlist_path = ALLOW_DIR/f"allow_{label}.parquet"
            allowlist.to_parquet(allowlist_path)
            n_unique_syms = allowlist["symbol"].nunique()
            avg_per_cycle = allowlist.groupby("open_time").size().mean() if len(allowlist) else 0
            print(f"  W={W}d τ={tau}: {n_unique_syms} unique syms ever allowed, "
                  f"avg {avg_per_cycle:.0f} syms/cycle", flush=True)
            r = replay(allowlist_path, label)
            if r:
                r["W"] = W; r["tau"] = tau; r["n_syms_total"] = n_unique_syms; r["avg_n_syms_per_cycle"] = round(avg_per_cycle,0)
                results.append(r)
                print(f"     → full Sh {r['full_Sh']:+.3f}  H1 {r['H1_Sh']:+.3f}  H2 {r['H2_Sh']:+.3f}  "
                      f"totPnL {r['full_totPnL']:+d}", flush=True)

    df = pd.DataFrame(results)
    print(f"\n=== ROLLING PER-SYM IC FILTER SWEEP — FULL OOS ===")
    print(df[["label","W","tau","avg_n_syms_per_cycle","full_Sh","H1_Sh","H2_Sh","full_totPnL","H2_totPnL"]].to_string(index=False))

    # Verdict
    print(f"\n=== VERDICT ===")
    print(f"Baseline (no filter): full +1.30, H1 +2.70, H2 -2.57")
    print(f"iter-005 fixed-H1-filter (lookahead!): full +1.52, H1 +2.26, H2 -0.12")
    # rolling-90d τ=0
    best = df[(df["W"]==90)&(df["tau"]==0.0)]
    if len(best):
        r = best.iloc[0]
        if r["H2_Sh"] >= -1.0 and r["full_Sh"] >= 1.0:
            print(f"  ✓ rolling-90d τ=0: full {r['full_Sh']:+.2f}, H2 {r['H2_Sh']:+.2f} — MEETS ADOPT criteria (+1.5 H2 lift, full ≥ 1.0)")
        else:
            print(f"  rolling-90d τ=0: full {r['full_Sh']:+.2f}, H2 {r['H2_Sh']:+.2f}")
    # overall best variant
    best_h2 = df.sort_values("H2_Sh", ascending=False).iloc[0]
    print(f"  Best H2 Sharpe across sweep: W={best_h2['W']} τ={best_h2['tau']} → H2 {best_h2['H2_Sh']:+.3f}, full {best_h2['full_Sh']:+.3f}")
    best_full = df.sort_values("full_Sh", ascending=False).iloc[0]
    print(f"  Best FULL Sharpe across sweep: W={best_full['W']} τ={best_full['tau']} → full {best_full['full_Sh']:+.3f}, H2 {best_full['H2_Sh']:+.3f}")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__ == "__main__": main()
