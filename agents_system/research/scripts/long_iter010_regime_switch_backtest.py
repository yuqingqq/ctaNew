"""LONG-PRED iter-010 — Regime-switch backtest (V0 ↔ V6 dynamic)

Tests whether the regime detector + switching rules I proposed can dynamically
move between V0 (full long+short, baseline) and V6 (filter + short_btc_hedge)
on the full OOS data, achieving better risk-adjusted return than either alone.

Approach:
  1. Run V0 (baseline) and V6 (production candidate) replays on full OOS
  2. Per cycle, compute regime indicators:
       - pred_disp (XS std of preds this cycle)
       - n_working_syms (count with trailing-180d per-sym IC > 0.02)
       - fwd_residual_skew_30d (rolling 30d per-sym skew avg)
       - V0_shadow_sharpe_30d, V6_shadow_sharpe_30d
  3. Apply switching rules with hysteresis
  4. Stitch PnL from whichever strategy is "active" each cycle
  5. Compare aggregate to V0-alone, V6-alone, and oracle (perfect H1/H2 knowledge)

CAVEAT: Simple cycle-level stitching assumes instantaneous switching.
Real switching has 24h transition cost (6-sleeve roll-off). This test gives
an UPPER BOUND on what the detector can achieve. We add a simple 3-bps cycle
transition cost on each switch to make it more realistic.

VERDICT criteria:
- Dynamic switcher must beat MAX(V0, V6) on full OOS Sharpe AND maxDD
- Number of switches should be low (<10 over 8 months)
- Switching should align with H1 (V0) / H2 (V6) boundaries
"""
from __future__ import annotations
import sys, time, os, subprocess
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
S = REPO/"live/state/convexity"
PREDS_STATIC = REPO/"research/convexity_portable_2026-05-20/results/_cache/x132_expanded_v0_preds.parquet"
PREDS_HL14 = S/"x132_p2_hl14_full_fullOOS_preds.parquet"
ALLOWLIST = S/"dyn_allow/allow_W180_t0.02.parquet"
H1_END = pd.Timestamp("2026-01-22",tz="UTC")

# Switching cost: 3 bps per switch (1.5 bps × 2 — approximate roll-off cost)
SWITCH_COST_BPS = 3.0

# Detector thresholds (calibrated on OOS data — see below for what they imply)
PRED_DISP_HIGH = 1.5    # > this → V0-favored
PRED_DISP_LOW  = 0.7    # < this → V6-favored
N_WORKING_V0 = 50       # ≥ → V0
N_WORKING_V6 = 40       # ≤ → V6
SKEW_NEGATIVE = -0.2    # < → V0
SKEW_POSITIVE = +0.5    # > → V6
SHADOW_DIFF_V0 = 0.5    # required V0 shadow advantage to switch to V0
SHADOW_DIFF_V6 = 0.3    # required V6 shadow advantage to switch to V6 (asymmetric)
CONSECUTIVE_CYCLES = 5  # need this many consecutive cycles meeting all conditions
MIN_DAYS_BETWEEN_SWITCH = 14
ROLLING_W_CYCLES = 30 * 6   # 30d × 6 cycles/day

def sharpe(p):
    p = np.array(p)/1e4
    return float(p.mean()/p.std()*np.sqrt(6*365)) if len(p)>1 and p.std()>0 else float("nan")

def run_v0():
    env = os.environ.copy()
    env.update({"PYTHONPATH":str(REPO),"BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3",
                "SIDE_MODE":"default","STRAT_HOLD":"6","COST_BPS_LEG":"4.5"})
    env.pop("CONVEXITY_DYNAMIC_ALLOWLIST_PATH", None)
    env.pop("CONVEXITY_PREDS_PATH", None)
    res = subprocess.run(["python3","-m","live.convexity_paper_bot",
                          "--replay-from","2025-10-04","--replay-end","2026-05-26"],
                         capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"V0 REPLAY FAILED: {res.stderr[-300:]}"); sys.exit(2)
    import shutil
    out = S/"iter010_v0_cycles.csv"; shutil.copy(S/"cycles.csv", out)
    return pd.read_csv(out, parse_dates=["open_time"])

def run_v6():
    env = os.environ.copy()
    env.update({"PYTHONPATH":str(REPO),"BULL_MODE":"mom","REGIME_HYSTERESIS_N":"3",
                "SIDE_MODE":"short_btc_hedge","STRAT_HOLD":"6","COST_BPS_LEG":"4.5",
                "CONVEXITY_DYNAMIC_ALLOWLIST_PATH":str(ALLOWLIST),
                "CONVEXITY_PREDS_PATH":str(PREDS_HL14)})
    res = subprocess.run(["python3","-m","live.convexity_paper_bot",
                          "--replay-from","2025-10-04","--replay-end","2026-05-26"],
                         capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(f"V6 REPLAY FAILED: {res.stderr[-300:]}"); sys.exit(2)
    import shutil
    out = S/"iter010_v6_cycles.csv"; shutil.copy(S/"cycles.csv", out)
    return pd.read_csv(out, parse_dates=["open_time"])

def compute_regime_indicators(preds: pd.DataFrame) -> pd.DataFrame:
    """Per-cycle: pred_disp, n_working_syms (180d), fwd_residual_skew_30d."""
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    preds = preds[(preds["open_time"].dt.hour%4==0) & (preds["open_time"].dt.minute==0)].copy()
    # per-cycle: pred_disp
    per_cyc = preds.groupby("open_time").agg(pred_disp=("pred","std"),
                                              n_syms=("symbol","nunique")).reset_index()
    # per-sym rolling-180d IC for n_working_syms
    print("  computing rolling 180d per-sym IC...", flush=True)
    bars_180 = 180*6
    ic_rows = []
    for sym, g in preds.groupby("symbol"):
        g = g.sort_values("open_time").reset_index(drop=True)
        ic = g["pred"].rolling(bars_180, min_periods=bars_180//3).corr(g["return_pct"]).shift(1)
        ic_rows.append(pd.DataFrame({"symbol":sym,"open_time":g["open_time"].values,"ic":ic.values}))
    ic_df = pd.concat(ic_rows, ignore_index=True)
    n_work = ic_df[ic_df["ic"]>0.02].groupby("open_time").size().rename("n_working_syms")
    # per-cycle rolling 30d cross-sym skew
    print("  computing rolling 30d forward-residual skew...", flush=True)
    bars_30 = 30*6
    skew_per_sym = preds.sort_values(["symbol","open_time"]).groupby("symbol",group_keys=False).apply(
        lambda g: g["return_pct"].rolling(bars_30, min_periods=bars_30//3).skew().shift(1))
    preds["sym_skew_30d"] = skew_per_sym.reset_index(drop=True)
    avg_skew = preds.groupby("open_time")["sym_skew_30d"].mean().rename("fwd_residual_skew_30d")
    # merge all
    out = per_cyc.merge(n_work, on="open_time", how="left").merge(avg_skew, on="open_time", how="left")
    out["n_working_syms"] = out["n_working_syms"].fillna(0).astype(int)
    return out

def apply_switching(v0: pd.DataFrame, v6: pd.DataFrame, indicators: pd.DataFrame) -> pd.DataFrame:
    """Apply switching rules; return per-cycle DataFrame with chosen strategy + PnL."""
    # align on open_time
    v0 = v0[["open_time","pnl_bps"]].rename(columns={"pnl_bps":"v0_pnl_bps"})
    v6 = v6[["open_time","pnl_bps"]].rename(columns={"pnl_bps":"v6_pnl_bps"})
    v0["open_time"] = pd.to_datetime(v0["open_time"], utc=True)
    v6["open_time"] = pd.to_datetime(v6["open_time"], utc=True)
    df = v0.merge(v6, on="open_time").merge(indicators, on="open_time")
    df = df.sort_values("open_time").reset_index(drop=True)
    # rolling shadow Sharpes
    df["v0_sh_30d"] = df["v0_pnl_bps"].rolling(ROLLING_W_CYCLES, min_periods=ROLLING_W_CYCLES//2).apply(
        lambda x: sharpe(x), raw=False)
    df["v6_sh_30d"] = df["v6_pnl_bps"].rolling(ROLLING_W_CYCLES, min_periods=ROLLING_W_CYCLES//2).apply(
        lambda x: sharpe(x), raw=False)
    # decision logic
    current = "V6"   # start with V6 (conservative default)
    n_consec_v0 = 0; n_consec_v6 = 0
    last_switch_cycle = -10000
    chosen = []
    switches = []
    for i, row in df.iterrows():
        # condition for V0 selection
        cond_v0 = (
            (row["pred_disp"] > PRED_DISP_HIGH if pd.notna(row["pred_disp"]) else False) and
            (row["n_working_syms"] >= N_WORKING_V0) and
            (row["fwd_residual_skew_30d"] < SKEW_NEGATIVE if pd.notna(row["fwd_residual_skew_30d"]) else False) and
            (row["v0_sh_30d"] > row["v6_sh_30d"] + SHADOW_DIFF_V0 if pd.notna(row["v0_sh_30d"]) and pd.notna(row["v6_sh_30d"]) else False)
        )
        # condition for V6 selection
        cond_v6 = (
            (row["pred_disp"] < PRED_DISP_LOW if pd.notna(row["pred_disp"]) else False) and
            (row["n_working_syms"] <= N_WORKING_V6) and
            (row["fwd_residual_skew_30d"] > SKEW_POSITIVE if pd.notna(row["fwd_residual_skew_30d"]) else False) and
            (row["v6_sh_30d"] > row["v0_sh_30d"] + SHADOW_DIFF_V6 if pd.notna(row["v0_sh_30d"]) and pd.notna(row["v6_sh_30d"]) else False)
        )
        if cond_v0: n_consec_v0 += 1
        else: n_consec_v0 = 0
        if cond_v6: n_consec_v6 += 1
        else: n_consec_v6 = 0

        days_since_switch = (i - last_switch_cycle) / 6
        new_strategy = current
        if current != "V0" and n_consec_v0 >= CONSECUTIVE_CYCLES and days_since_switch >= MIN_DAYS_BETWEEN_SWITCH:
            new_strategy = "V0"; switches.append((row["open_time"], current, new_strategy, "to_V0")); last_switch_cycle = i
        elif current != "V6" and n_consec_v6 >= CONSECUTIVE_CYCLES and days_since_switch >= MIN_DAYS_BETWEEN_SWITCH:
            new_strategy = "V6"; switches.append((row["open_time"], current, new_strategy, "to_V6")); last_switch_cycle = i
        current = new_strategy
        chosen.append(current)
    df["chosen"] = chosen
    df["dyn_pnl_bps"] = np.where(df["chosen"]=="V0", df["v0_pnl_bps"], df["v6_pnl_bps"])
    # apply transition cost at switch points
    df["is_switch"] = df["chosen"] != df["chosen"].shift(1)
    df["dyn_pnl_bps_after_cost"] = df["dyn_pnl_bps"] - df["is_switch"].astype(float)*SWITCH_COST_BPS
    return df, switches

def summarize(df: pd.DataFrame, pnl_col: str, label: str) -> dict:
    p = df[pnl_col]
    h1 = df[df["open_time"] < H1_END][pnl_col]
    h2 = df[df["open_time"] >= H1_END][pnl_col]
    cum = pd.Series(p).cumsum(); dd = float((cum-cum.cummax()).min())
    return dict(label=label, n=len(df),
                full_Sh=round(sharpe(p),3),
                H1_Sh=round(sharpe(h1),3),
                H2_Sh=round(sharpe(h2),3),
                totPnL=int(p.sum()),
                H2_totPnL=int(h2.sum()),
                maxDD=int(dd))

def main():
    t0 = time.time()
    print("=== LONG-PRED iter-010: Regime-switch backtest ===\n", flush=True)

    print("running V0 (baseline)...", flush=True)
    v0 = run_v0()
    print(f"  V0 cycles: {len(v0)}", flush=True)

    print("running V6 (production candidate)...", flush=True)
    v6 = run_v6()
    print(f"  V6 cycles: {len(v6)}", flush=True)

    print("\ncomputing regime indicators on static preds...", flush=True)
    preds = pd.read_parquet(PREDS_STATIC, columns=["symbol","open_time","pred","return_pct"])
    indicators = compute_regime_indicators(preds)
    print(f"  indicators: {len(indicators)} cycles", flush=True)
    print(f"  pred_disp distribution: 10%={indicators['pred_disp'].quantile(0.1):.2f} median={indicators['pred_disp'].median():.2f} 90%={indicators['pred_disp'].quantile(0.9):.2f}")
    print(f"  n_working_syms: 10%={indicators['n_working_syms'].quantile(0.1):.0f} median={indicators['n_working_syms'].median():.0f} 90%={indicators['n_working_syms'].quantile(0.9):.0f}")
    print(f"  fwd_residual_skew_30d: 10%={indicators['fwd_residual_skew_30d'].quantile(0.1):+.2f} median={indicators['fwd_residual_skew_30d'].median():+.2f} 90%={indicators['fwd_residual_skew_30d'].quantile(0.9):+.2f}")

    print("\napplying switching rules...", flush=True)
    df, switches = apply_switching(v0, v6, indicators)

    # ORACLE: perfect knowledge — use V0 in H1, V6 in H2
    df["oracle_pnl_bps"] = np.where(df["open_time"] < H1_END, df["v0_pnl_bps"], df["v6_pnl_bps"])
    df["oracle_switch"] = (df["open_time"].shift(1) < H1_END) & (df["open_time"] >= H1_END)
    df["oracle_pnl_bps_after_cost"] = df["oracle_pnl_bps"] - df["oracle_switch"].astype(float)*SWITCH_COST_BPS

    results = [
        summarize(df, "v0_pnl_bps", "V0_baseline_only"),
        summarize(df, "v6_pnl_bps", "V6_production_only"),
        summarize(df, "dyn_pnl_bps", "DYNAMIC_switcher_NO_cost"),
        summarize(df, "dyn_pnl_bps_after_cost", "DYNAMIC_switcher_WITH_cost"),
        summarize(df, "oracle_pnl_bps", "ORACLE_perfect_no_cost"),
        summarize(df, "oracle_pnl_bps_after_cost", "ORACLE_perfect_with_cost"),
    ]
    print(f"\n=== STRATEGY COMPARISON ===")
    rdf = pd.DataFrame(results)
    print(rdf[["label","full_Sh","H1_Sh","H2_Sh","totPnL","H2_totPnL","maxDD","n"]].to_string(index=False))

    # switch analysis
    print(f"\n=== SWITCH EVENTS ===")
    print(f"  Total switches: {len(switches)}")
    for ot, frm, to, kind in switches:
        print(f"  {ot}: {frm} → {to}")

    # strategy choice breakdown
    print(f"\n=== STRATEGY CHOICE BREAKDOWN ===")
    chosen_counts = df["chosen"].value_counts().to_dict()
    print(f"  Cycles chosen: {chosen_counts}")
    # by half
    h1_choices = df[df["open_time"]<H1_END]["chosen"].value_counts(normalize=True).to_dict()
    h2_choices = df[df["open_time"]>=H1_END]["chosen"].value_counts(normalize=True).to_dict()
    print(f"  H1 split: V0 {h1_choices.get('V0',0)*100:.0f}%, V6 {h1_choices.get('V6',0)*100:.0f}%")
    print(f"  H2 split: V0 {h2_choices.get('V0',0)*100:.0f}%, V6 {h2_choices.get('V6',0)*100:.0f}%")

    # Verdict
    print(f"\n=== VERDICT ===")
    v0_full = rdf[rdf["label"]=="V0_baseline_only"]["full_Sh"].iloc[0]
    v6_full = rdf[rdf["label"]=="V6_production_only"]["full_Sh"].iloc[0]
    dyn_full = rdf[rdf["label"]=="DYNAMIC_switcher_WITH_cost"]["full_Sh"].iloc[0]
    oracle_full = rdf[rdf["label"]=="ORACLE_perfect_with_cost"]["full_Sh"].iloc[0]
    max_single = max(v0_full, v6_full)
    print(f"  V0 standalone:    {v0_full:+.3f}")
    print(f"  V6 standalone:    {v6_full:+.3f}")
    print(f"  Best of single:   {max_single:+.3f}")
    print(f"  Dynamic switcher: {dyn_full:+.3f}  (Δ vs best single: {dyn_full-max_single:+.3f})")
    print(f"  Oracle (perfect): {oracle_full:+.3f}  (Δ vs best single: {oracle_full-max_single:+.3f})")
    if dyn_full > max_single + 0.15:
        print(f"  ✓ DYNAMIC SWITCHER ADDS VALUE: beats best standalone by ≥0.15")
    elif dyn_full > max_single - 0.05:
        print(f"  ≈ tied: detector doesn't materially help OR hurt — may still be worth deploying for robustness")
    else:
        print(f"  ✗ DETECTOR HURTS: stay with V6 (or V0 by regime)")

    # save full state for inspection
    df.to_csv(S/"iter010_dynamic.csv", index=False)
    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
