"""LONG-PRED iter-018 — Random-pick benchmark — does the model actually pick better than random?

Test: at each cycle, sample K=5 names from the universe at random (with seed),
measure their realized returns. Compare to model's top-K=5 selection.

If random ≈ model: V0's "long alpha" is mostly universe lottery, not predictive skill
If model >> random: model is adding real predictive value

Configurations:
  (A) Full universe (120 syms) — V0's playing field
  (B) Filtered universe (~30 syms) — V_FULL's playing field

Run 50 random seeds per cycle for confidence intervals.
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PREDS_STATIC = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PREDS_HL14 = REPO/"live/state/convexity/x132_p2_hl14_full_fullOOS_preds.parquet"
ALLOWLIST = REPO/"live/state/convexity/dyn_allow/allow_W180_t0.02.parquet"

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
K = 5
N_SEEDS = 50

def main():
    t0 = time.time()
    print("=== iter-018: Random-pick benchmark ===\n", flush=True)

    # Load preds + allowlist
    print("loading data...", flush=True)
    static = pd.read_parquet(PREDS_STATIC, columns=["symbol","open_time","pred","return_pct"])
    static["open_time"] = pd.to_datetime(static["open_time"], utc=True)
    static = static[(static["open_time"].dt.hour%4==0) & (static["open_time"].dt.minute==0)]

    hl14 = pd.read_parquet(PREDS_HL14, columns=["symbol","open_time","pred","return_pct"])
    hl14["open_time"] = pd.to_datetime(hl14["open_time"], utc=True)
    hl14 = hl14[(hl14["open_time"].dt.hour%4==0) & (hl14["open_time"].dt.minute==0)]

    allowlist = pd.read_parquet(ALLOWLIST)
    allowlist["open_time"] = pd.to_datetime(allowlist["open_time"], utc=True)
    allow_set = set(zip(allowlist["open_time"], allowlist["symbol"]))

    print(f"  static preds: {len(static):,}, hl14 preds: {len(hl14):,}", flush=True)

    def measure(preds_df, use_filter, name):
        """For each cycle, compute model top-K mean return, and N_SEEDS random K-picks."""
        if use_filter:
            mask = preds_df.apply(lambda r: (r["open_time"], r["symbol"]) in allow_set, axis=1)
            preds_df = preds_df[mask].copy()
        rows = []
        rng = np.random.default_rng(42)
        for ot, g in preds_df.groupby("open_time"):
            if len(g) < 2*K: continue
            # Model top-K
            g_sorted = g.sort_values("pred")
            model_top_K = g_sorted.tail(K)["return_pct"].mean()
            # Random K from same universe
            random_picks = []
            for _ in range(N_SEEDS):
                random_K = rng.choice(g["return_pct"].values, size=K, replace=False).mean()
                random_picks.append(random_K)
            row = dict(open_time=ot, n=len(g), model_top=model_top_K,
                       random_mean=np.mean(random_picks),
                       random_p5=np.percentile(random_picks, 5),
                       random_p95=np.percentile(random_picks, 95))
            rows.append(row)
        return pd.DataFrame(rows)

    print("\nmeasuring V0 (static preds + full universe, no filter)...", flush=True)
    df_v0 = measure(static, use_filter=False, name="V0")
    df_v0["open_time"] = pd.to_datetime(df_v0["open_time"], utc=True)
    print(f"  {len(df_v0):,} cycles, avg universe={df_v0['n'].mean():.0f}", flush=True)

    print("\nmeasuring V_FULL (hl14 preds + filtered universe)...", flush=True)
    df_vf = measure(hl14, use_filter=True, name="V_FULL")
    df_vf["open_time"] = pd.to_datetime(df_vf["open_time"], utc=True)
    print(f"  {len(df_vf):,} cycles, avg universe={df_vf['n'].mean():.0f}", flush=True)

    print("\nmeasuring HL14-no-filter (hl14 preds + full universe)...", flush=True)
    df_hl14f = measure(hl14, use_filter=False, name="HL14_full")
    df_hl14f["open_time"] = pd.to_datetime(df_hl14f["open_time"], utc=True)
    print(f"  {len(df_hl14f):,} cycles, avg universe={df_hl14f['n'].mean():.0f}", flush=True)

    # H1 vs H2 analysis
    print(f"\n=== RESULTS — top-K=5 long edge (absolute realized return per cycle, bps) ===\n")
    print(f"{'config':<28} {'period':<3} {'model_top':>12} {'random_mean':>13} {'random_p5':>12} {'random_p95':>13} {'avg univ':>9}")
    print("-"*100)
    for label, df in [("V0 (static, full)", df_v0),
                       ("HL14 (hl14, full)", df_hl14f),
                       ("V_FULL (hl14, filtered)", df_vf)]:
        for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
            sub = df[(df["open_time"]>=s) & (df["open_time"]<e)]
            if len(sub)==0: continue
            mt = sub["model_top"].mean()*1e4
            rm = sub["random_mean"].mean()*1e4
            rp5 = sub["random_p5"].mean()*1e4
            rp95 = sub["random_p95"].mean()*1e4
            avg_n = sub["n"].mean()
            print(f"  {label:<28} {period_label:<3} {mt:>+10.2f}  {rm:>+10.2f}    {rp5:>+10.2f}  {rp95:>+10.2f}  {avg_n:>7.0f}")
        print()

    # Verdict
    print(f"=== VERDICT ===\n")
    for label, df in [("V0 (static, full)", df_v0),
                       ("HL14 (hl14, full)", df_hl14f),
                       ("V_FULL (hl14, filtered)", df_vf)]:
        for period_label, (s, e) in [("H1", (H1_START, H2_START)), ("H2", (H2_START, H2_END))]:
            sub = df[(df["open_time"]>=s) & (df["open_time"]<e)]
            if len(sub)==0: continue
            mt = sub["model_top"].mean()*1e4
            rm = sub["random_mean"].mean()*1e4
            delta = mt - rm
            # Is model significantly different from random? bootstrap-like test
            diffs = (sub["model_top"] - sub["random_mean"]).values * 1e4
            se = diffs.std()/np.sqrt(len(diffs))
            t = diffs.mean()/se if se>0 else float("nan")
            sig = "★" if abs(t)>1.96 else " "
            print(f"  {label} {period_label}: model={mt:+.2f} vs random={rm:+.2f}, Δ={delta:+.2f} bps (t={t:+.2f}) {sig}")
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
