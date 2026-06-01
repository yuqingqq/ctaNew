"""LONG-PRED iter-028 — Why does the short side work in H2?

Hypothesis competition:
  H_A: Mean-reversion symmetric → shorts shouldn't work consistently
  H_B: H2 is a BEAR market → market beta drives short edge (everything down)
  H_C: Asymmetric mean-reversion → overheated names crash but oversold names don't bounce

Tests:
  1. Aggregate H2 market behavior: BTC return, alt median return, cross-sym dispersion
  2. Decompose short edge into beta vs selection:
     short_edge = (market beta contribution) + (cross-sym selection alpha)
     If beta dominant → H_B (it's just a bear market)
     If selection dominant → H_C (overheated names crash specifically)
  3. Compare short picks' characteristics vs long picks:
     - Funding rate, vol, momentum at entry
     - Realized return distribution (do shorts crash hard? do longs slowly drift down?)
  4. Symmetry test: pick BOT-K of momentum signal vs random; pick TOP-K random short candidates
     - If shorts work because of beta: random short = real short
     - If shorts work because of selection: real short >> random short
"""
import sys, time
from pathlib import Path
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")

REPO = Path("/home/yuqing/ctaNew")
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
PREDS = REPO/"agents_system/research/outputs/iter025/preds_L1_V0_pooled.parquet"

H1_START = pd.Timestamp("2025-10-04",tz="UTC")
H2_START = pd.Timestamp("2026-01-22",tz="UTC")
H2_END   = pd.Timestamp("2026-05-26",tz="UTC")
K = 5

def main():
    t0 = time.time()
    print("=== iter-028: Why does the short side work in H2? ===\n", flush=True)

    # Load predictions from best LGBM model (iter-025 L1_V0_pooled)
    print("loading predictions + panel...", flush=True)
    preds = pd.read_parquet(PREDS)
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    print(f"  preds: {len(preds):,} rows", flush=True)

    # Load panel for characteristic features
    cols_need = ["symbol","open_time","return_pct","return_1d","atr_pct","funding_rate",
                  "rvol_7d","btc_rvol_7d","bars_since_high","idio_vol_to_btc_1d","ret_3d"]
    panel = pd.read_parquet(PANEL, columns=cols_need)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[(panel["open_time"].dt.hour%4==0) & (panel["open_time"].dt.minute==0)]
    print(f"  panel: {len(panel):,} rows", flush=True)

    # Run analysis for BOTH H1 and H2
    for PERIOD_LABEL, S, E in [("H1", H1_START, H2_START), ("H2", H2_START, H2_END)]:
        print(f"\n\n{'='*70}\n=== ANALYSIS FOR {PERIOD_LABEL} ===\n{'='*70}", flush=True)
        analyze_period(preds, panel, PERIOD_LABEL, S, E)
    return

def analyze_period(preds, panel, PERIOD_LABEL, S, E):
    h2_preds = preds[(preds["open_time"] >= S) & (preds["open_time"] < E)].copy()
    h2_panel = panel[(panel["open_time"] >= S) & (panel["open_time"] < E)].copy()

    # ============ TEST 1: Aggregate H2 market behavior ============
    print(f"\n=== TEST 1: H2 aggregate market behavior ===\n", flush=True)
    cycle_stats = h2_panel.groupby("open_time").agg(
        median_ret=("return_pct", "median"),
        mean_ret=("return_pct", "mean"),
        std_ret=("return_pct", "std"),
        n_syms=("symbol", "size")
    )
    print(f"  Total H2 cycles: {len(cycle_stats)}")
    print(f"  Mean of cross-sym MEAN return per cycle: {cycle_stats['mean_ret'].mean()*1e4:+.2f} bps")
    print(f"  Mean of cross-sym MEDIAN return per cycle: {cycle_stats['median_ret'].mean()*1e4:+.2f} bps")
    print(f"  Mean cross-sym std per cycle: {cycle_stats['std_ret'].mean()*1e4:.2f} bps")
    # Cumulative basket return over H2 (compound the mean returns)
    cum_mean = (1 + cycle_stats["mean_ret"]).prod() - 1
    cum_med = (1 + cycle_stats["median_ret"]).prod() - 1
    print(f"  Cumulative basket return (equal-weighted mean): {cum_mean*100:+.2f}%")
    print(f"  Cumulative basket return (equal-weighted median): {cum_med*100:+.2f}%")
    if cum_med < -0.10:
        print(f"  ★ H2 IS A BEAR MARKET — median alt down >10% cumulatively")
    elif cum_med < -0.05:
        print(f"  H2 is moderately bearish")
    else:
        print(f"  H2 is roughly flat or up")

    # ============ TEST 2: Decompose short edge into beta vs selection ============
    print(f"\n=== TEST 2: Decompose short edge into beta vs selection ===\n", flush=True)
    # Beta contribution = -1 × market median return (shorting the median)
    # Selection alpha = picked shorts' return - market median (how much MORE they decline)
    rows = []
    for ot, grp in h2_preds.groupby("open_time"):
        if len(grp) < 2*K: continue
        market_ret = grp["return_pct"].median()
        market_mean = grp["return_pct"].mean()
        # Top-K longs
        top = grp.nlargest(K, "pred")
        long_ret = top["return_pct"].mean()
        # Bot-K shorts
        bot = grp.nsmallest(K, "pred")
        short_ret_picked = bot["return_pct"].mean()
        # Short edge (we make money when picked names go down)
        short_edge = -short_ret_picked
        # Beta component: shorting the market (median)
        beta_short = -market_ret
        # Selection: alpha beyond shorting the market
        selection_short = (market_ret - short_ret_picked)
        # Long side
        long_edge = long_ret
        beta_long = market_ret
        selection_long = long_ret - market_ret
        rows.append({
            "open_time": ot, "market_ret": market_ret,
            "short_picks_ret": short_ret_picked, "short_edge": short_edge,
            "short_beta": beta_short, "short_selection": selection_short,
            "long_picks_ret": long_ret, "long_edge": long_edge,
            "long_beta": beta_long, "long_selection": selection_long,
        })
    df = pd.DataFrame(rows)

    def stat(arr):
        a = arr.values * 1e4
        m = a.mean(); s = a.std()/np.sqrt(len(a)); t = m/s if s>0 else float("nan")
        return m, t
    print(f"  {'component':<22} {'mean bps':>10} {'t-stat':>8}  signif")
    print("-"*60)
    for label, col in [
        ("Long total edge",      "long_edge"),
        ("  Long beta",          "long_beta"),
        ("  Long selection",     "long_selection"),
        ("Short total edge",     "short_edge"),
        ("  Short beta",         "short_beta"),
        ("  Short selection",    "short_selection"),
    ]:
        m, t = stat(df[col])
        sig = "★" if abs(t)>1.96 else " "
        print(f"  {label:<22} {m:>+8.2f}  {t:>+6.2f}   {sig}")

    print(f"\n  Interpretation:")
    short_beta_m, _ = stat(df["short_beta"])
    short_sel_m, _ = stat(df["short_selection"])
    if abs(short_beta_m) > abs(short_sel_m) * 2:
        print(f"  ★ SHORT EDGE DRIVEN BY MARKET BETA — it's a bear market, shorts harvest beta")
    elif abs(short_sel_m) > abs(short_beta_m) * 2:
        print(f"  ★ SHORT EDGE DRIVEN BY SELECTION — picked names underperform specifically")
    else:
        print(f"  Mixed: {short_beta_m:.2f} bps beta + {short_sel_m:.2f} bps selection")

    # ============ TEST 3: Characteristics of short picks vs long picks ============
    print(f"\n=== TEST 3: Characteristics of long vs short picks ===\n", flush=True)
    merged = h2_preds.merge(h2_panel[["symbol","open_time","return_1d","funding_rate",
                                        "rvol_7d","atr_pct","ret_3d","bars_since_high"]],
                             on=["symbol","open_time"], how="left")
    short_chars = []
    long_chars = []
    for ot, grp in merged.groupby("open_time"):
        if len(grp) < 2*K: continue
        top = grp.nlargest(K, "pred")
        bot = grp.nsmallest(K, "pred")
        for c in ["return_1d","funding_rate","rvol_7d","atr_pct","ret_3d","bars_since_high"]:
            short_chars.append({"feature": c, "value": bot[c].mean()})
            long_chars.append({"feature": c, "value": top[c].mean()})
    short_df = pd.DataFrame(short_chars).groupby("feature")["value"].mean()
    long_df = pd.DataFrame(long_chars).groupby("feature")["value"].mean()
    print(f"  {'feature':<22} {'long picks':>12} {'short picks':>13} {'diff':>10}")
    print("-"*60)
    for f in ["return_1d","funding_rate","rvol_7d","atr_pct","ret_3d","bars_since_high"]:
        l = long_df[f]; s = short_df[f]
        print(f"  {f:<22} {l:>+10.4f}  {s:>+10.4f}  {l-s:>+8.4f}")

    # ============ TEST 4: Distribution of picked shorts' realized returns ============
    print(f"\n=== TEST 4: Distribution of picked shorts' realized returns ===\n", flush=True)
    picked_short_rets = []
    for ot, grp in h2_preds.groupby("open_time"):
        if len(grp) < 2*K: continue
        bot = grp.nsmallest(K, "pred")
        picked_short_rets.extend(bot["return_pct"].tolist())
    sr = np.array(picked_short_rets) * 1e4
    print(f"  n picked-short observations: {len(sr):,}")
    print(f"  mean:    {sr.mean():+.2f} bps  (negative = short wins)")
    print(f"  median:  {np.median(sr):+.2f} bps")
    print(f"  p10:     {np.percentile(sr, 10):+.2f} bps  (worst declines: short wins big)")
    print(f"  p25:     {np.percentile(sr, 25):+.2f} bps")
    print(f"  p75:     {np.percentile(sr, 75):+.2f} bps")
    print(f"  p90:     {np.percentile(sr, 90):+.2f} bps  (best gains: short loses)")
    pct_negative = (sr < 0).mean() * 100
    print(f"  % observations negative: {pct_negative:.1f}%  (= % short positions that win)")
    if pct_negative > 55:
        print(f"  ★ Most short positions WIN (>55% win rate) — directional signal")
    elif pct_negative < 50:
        print(f"  Most short positions LOSE — must be carry/big-wins asymmetry")
    else:
        print(f"  ~50% wins; alpha from asymmetric magnitudes")

    # ============ TEST 5: Long picks' return distribution ============
    print(f"\n=== TEST 5: Distribution of picked longs' realized returns ===\n", flush=True)
    picked_long_rets = []
    for ot, grp in h2_preds.groupby("open_time"):
        if len(grp) < 2*K: continue
        top = grp.nlargest(K, "pred")
        picked_long_rets.extend(top["return_pct"].tolist())
    lr = np.array(picked_long_rets) * 1e4
    print(f"  n picked-long observations: {len(lr):,}")
    print(f"  mean:    {lr.mean():+.2f} bps  (positive = long wins)")
    print(f"  median:  {np.median(lr):+.2f} bps")
    print(f"  p10:     {np.percentile(lr, 10):+.2f} bps")
    print(f"  p90:     {np.percentile(lr, 90):+.2f} bps")
    pct_pos_long = (lr > 0).mean() * 100
    print(f"  % observations positive: {pct_pos_long:.1f}%")
    print(f"\n  ASYMMETRY: short p10={np.percentile(sr,10):+.0f} vs long p90={np.percentile(lr,90):+.0f}")
    print(f"  ASYMMETRY: short mean={sr.mean():+.0f} vs long mean={lr.mean():+.0f}")

    print(f"\nDONE [{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
