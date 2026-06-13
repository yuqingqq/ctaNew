"""Phase CAL: per-symbol calibration layer (A4).

Pre-registered (single rule, no fit):
  For each (symbol, cycle_t):
    1. Compute trailing IC = Pearson(pred, alpha_A) over the past 540 cycles
       (~90 days × 6 cycles/day), using only samples with exit_time ≤ current
       open_time (PIT, via shift=1 in 4h-aligned grid).
    2. pred_cal = pred × sign(ic_trail)  if |ic_trail| > 0
                = 0                       otherwise (no opinion)

Rationale: A2 segmentation failed because the per-symbol heterogeneity isn't
cleanly addressable by binary split. A4 adapts CONTINUOUSLY per symbol — each
symbol gets its own multiplier based on its trailing predictive performance.
ETH/BNB/etc. with negative trailing IC get sign-flipped; symbols with positive
trailing IC pass through unchanged.

Same V3.1 sleeve machinery downstream. 6-gate validation.
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psl)
spec2 = importlib.util.spec_from_file_location(
    "svar", REPO / "scripts/phase_ah_sleeve_variants.py")
svar = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(svar)

OUT = REPO / "outputs/vBTC_phase_CAL"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 48
TRAIL_WINDOW = 540  # ~90 days × 6 cycles/day
MIN_PERIODS = 30
OOS_FOLDS = list(range(1, 10))
N_SLEEVES = 6
HOLD_BARS = 288
V31_REF_SHARPE = 2.23
CYCLES_PER_YEAR = (288 * 365) / 48


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def block_bootstrap_ci(x, stat=_sharpe, block_size=7, n_boot=2000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < block_size + 2: return stat(x), stat(x), stat(x)
    rng = np.random.RandomState(seed)
    n = len(x); nb = n // block_size + 1
    boots = []
    for _ in range(n_boot):
        starts = rng.randint(0, n - block_size + 1, size=nb)
        blocks = np.concatenate([x[s:s+block_size] for s in starts])[:n]
        boots.append(stat(blocks))
    boots = np.array(boots)
    return float(stat(x)), float(np.percentile(boots, 100 * alpha / 2)), \
           float(np.percentile(boots, 100 * (1 - alpha / 2)))


def main():
    print("=== Phase CAL: per-symbol calibration ===\n", flush=True)
    print(f"  Pre-registered: pred_cal = pred × sign(trailing_IC_90d)", flush=True)
    print(f"  Window = {TRAIL_WINDOW} cycles, min_periods = {MIN_PERIODS}, PIT shift = 1\n",
          flush=True)

    # Load audit panel
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet")
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    print(f"  loaded {len(apd):,} predictions", flush=True)

    # Subset to 4h-aligned cycle times (every 48 bars within OOS folds)
    df_oos = apd[apd["fold"].isin(OOS_FOLDS)].sort_values(["open_time", "symbol"])
    times = sorted(df_oos["open_time"].unique())
    cycle_times = set(times[::HORIZON])
    apd_cyc = apd[apd["open_time"].isin(cycle_times)].copy()
    apd_cyc = apd_cyc.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print(f"  4h-aligned cycle rows: {len(apd_cyc):,} ({len(cycle_times)} cycles × ~{apd_cyc['symbol'].nunique()} syms)",
          flush=True)

    # Per-symbol trailing IC (Pearson on shifted preds for PIT)
    print(f"  computing per-symbol trailing IC...", flush=True)
    t0 = time.time()
    def add_trail_ic(g):
        g = g.sort_values("open_time").copy()
        # Shift by 1 cycle so trailing window excludes current sample (PIT)
        g["ic_trail"] = (g["pred"].shift(1)
                            .rolling(TRAIL_WINDOW, min_periods=MIN_PERIODS)
                            .corr(g["alpha_A"].shift(1)))
        return g
    apd_cyc = apd_cyc.groupby("symbol", group_keys=False).apply(add_trail_ic)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Calibrated prediction
    apd_cyc["pred_raw"] = apd_cyc["pred"]
    apd_cyc["pred_cal"] = apd_cyc["pred"] * np.sign(apd_cyc["ic_trail"].fillna(0))
    apd_cyc.to_parquet(OUT / "all_predictions_calibrated.parquet", index=False)

    # Stats on calibration coverage
    has_ic = apd_cyc["ic_trail"].notna()
    pos_ic = (apd_cyc["ic_trail"] > 0).sum()
    neg_ic = (apd_cyc["ic_trail"] < 0).sum()
    zero = (apd_cyc["ic_trail"] == 0).sum() + (~has_ic).sum()
    total = len(apd_cyc)
    print(f"\n  Calibration stats:", flush=True)
    print(f"    IC > 0 (kept):   {pos_ic:,} ({pos_ic/total*100:.1f}%)", flush=True)
    print(f"    IC < 0 (flipped): {neg_ic:,} ({neg_ic/total*100:.1f}%)", flush=True)
    print(f"    no opinion (zeroed): {zero:,} ({zero/total*100:.1f}%)", flush=True)

    # Per-cycle IC comparison: raw pred vs calibrated pred
    print(f"\n  Per-cycle IC (across 50-sym cross-section):", flush=True)
    raw_ics = []
    cal_ics = []
    apd_oos = apd_cyc[apd_cyc["fold"].isin(OOS_FOLDS)].copy()
    for t, g in apd_oos.groupby("open_time"):
        g = g.dropna(subset=["alpha_A"])
        if len(g) < 10: continue
        raw_ic = g["pred_raw"].rank().corr(g["alpha_A"].rank())
        cal_ic = g["pred_cal"].rank().corr(g["alpha_A"].rank())
        if not pd.isna(raw_ic): raw_ics.append(raw_ic)
        if not pd.isna(cal_ic): cal_ics.append(cal_ic)
    raw_ics = np.array(raw_ics); cal_ics = np.array(cal_ics)
    print(f"    Raw pred:    mean = {raw_ics.mean():+.4f}  median = {np.median(raw_ics):+.4f}  pct_pos = {(raw_ics>0).mean()*100:.1f}%",
          flush=True)
    print(f"    Calibrated:  mean = {cal_ics.mean():+.4f}  median = {np.median(cal_ics):+.4f}  pct_pos = {(cal_ics>0).mean()*100:.1f}%",
          flush=True)
    print(f"    Δ mean IC: {cal_ics.mean() - raw_ics.mean():+.4f}", flush=True)

    if cal_ics.mean() <= raw_ics.mean() - 0.001:
        print(f"\n  → IC dropped. Skipping basket rebuild (would fail).", flush=True)
        return

    # Build new audit panel with pred = pred_cal for the production protocol
    apd_for_protocol = apd_cyc.copy()
    apd_for_protocol["pred"] = apd_for_protocol["pred_cal"]
    # Also need return_pct (already in audit panel)
    if "return_pct" not in apd_for_protocol.columns:
        rp = apd[["symbol", "open_time", "return_pct"]]
        apd_for_protocol = apd_for_protocol.merge(rp, on=["symbol", "open_time"], how="left")

    # Build rolling-IC universe using CALIBRATED predictions
    print(f"\n  building rolling-IC universe (180/90) on calibrated preds...",
          flush=True)
    t0 = time.time()
    listings = psl.get_listings()
    panel_first_obs = apd_for_protocol.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            listings[sym] = t
    def elig_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in apd_for_protocol["symbol"].unique()
                  if listings.get(s) and listings[s] <= cutoff}
    sampled = sorted(apd_for_protocol[apd_for_protocol["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    # We already subsampled to 4h cycles, so use sampled as-is
    universe = psl.build_rolling_ic_universe(apd_for_protocol, sampled, psl.TOP_N, elig_at)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Build sleeves
    print(f"  building sleeves...", flush=True)
    t0 = time.time()
    sleeves_cal = psl.run_production_protocol_save_sleeves(apd_for_protocol, universe)
    print(f"  done: {len(sleeves_cal)} cycles, traded={sleeves_cal['traded'].sum()} ({time.time()-t0:.0f}s)",
          flush=True)
    sleeves_cal["time"] = pd.to_datetime(sleeves_cal["time"], utc=True)
    sleeves_cal.to_parquet(OUT / "production_sleeves_cal.parquet", index=False)

    # Load close prices
    all_syms = sorted(apd["symbol"].unique())
    print(f"  loading close prices...", flush=True)
    t0 = time.time()
    close_wide = svar.load_close_wide(all_syms)
    fwd_rets_4h = (close_wide.shift(-HORIZON) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Baseline V3.1 (WINNER_21)
    sleeves_w21 = pd.read_parquet(svar.SLEEVES_PATH)
    sleeves_w21["time"] = pd.to_datetime(sleeves_w21["time"], utc=True)
    print(f"\n  V3.1 aggregation on baseline...", flush=True)
    df_w21 = svar.aggregate_sleeves_variant(sleeves_w21, fwd_rets_4h, N_SLEEVES, HOLD_BARS,
                                                sleeve_weights=[1/6]*6)
    print(f"  V3.1 aggregation on calibrated...", flush=True)
    df_cal = svar.aggregate_sleeves_variant(sleeves_cal, fwd_rets_4h, N_SLEEVES, HOLD_BARS,
                                                sleeve_weights=[1/6]*6)
    df_cal.to_csv(OUT / "per_cycle_cal_v31.csv", index=False)

    sh_w21 = _sharpe(df_w21["net_pnl_bps"])
    sh_cal = _sharpe(df_cal["net_pnl_bps"])
    dd_w21 = _max_dd(df_w21["net_pnl_bps"])
    dd_cal = _max_dd(df_cal["net_pnl_bps"])
    npos_w21 = sum(1 for _, g in df_w21.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)
    npos_cal = sum(1 for _, g in df_cal.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)
    print(f"\n  V3.1 baseline:    Sharpe={sh_w21:+.3f}  maxDD={dd_w21:+.0f}  PnL={df_w21['net_pnl_bps'].sum():+.0f}  folds+={npos_w21}/9",
          flush=True)
    print(f"  V3.1 calibrated:  Sharpe={sh_cal:+.3f}  maxDD={dd_cal:+.0f}  PnL={df_cal['net_pnl_bps'].sum():+.0f}  folds+={npos_cal}/9",
          flush=True)
    lift = sh_cal - sh_w21
    print(f"  Static lift: {lift:+.3f}", flush=True)

    # Per-fold breakdown
    print(f"\n  Per-fold breakdown:", flush=True)
    print(f"  {'fold':>4}  {'W21':>8}  {'CAL':>8}  {'Δ':>7}", flush=True)
    fold_diffs = {}
    for f in OOS_FOLDS:
        a = df_w21[df_w21["fold"] == f]["net_pnl_bps"].sum()
        b = df_cal[df_cal["fold"] == f]["net_pnl_bps"].sum()
        d = b - a
        fold_diffs[f] = d
        print(f"  {f:>4}  {a:>+8.0f}  {b:>+8.0f}  {d:>+7.0f}", flush=True)
    pos_lift = sum(v for v in fold_diffs.values() if v > 0)
    max_fold_contribution = (max(fold_diffs.values()) / pos_lift * 100) if pos_lift > 0 else 0
    print(f"\n  Max single fold contribution: {max_fold_contribution:.0f}%", flush=True)

    # Paired bootstrap
    paired = df_w21[["time", "fold", "net_pnl_bps"]].rename(
        columns={"net_pnl_bps": "w21"}).merge(
        df_cal[["time", "net_pnl_bps"]].rename(columns={"net_pnl_bps": "cal"}),
        on="time")
    paired["diff"] = paired["cal"] - paired["w21"]
    def _mean(x): return float(np.mean(x))
    mu, lo, hi = block_bootstrap_ci(paired["diff"].to_numpy(), stat=_mean,
                                        block_size=7, n_boot=2000)
    print(f"\n  Paired diff: {mu:+.3f} bps/cycle CI [{lo:+.3f}, {hi:+.3f}]", flush=True)
    diff_sig = (lo > 0) or (hi < 0)
    print(f"  Paired diff CI excludes 0: {'YES' if diff_sig else 'NO'}", flush=True)

    # Verdict
    print(f"\n=== Phase CAL Verdict ===\n", flush=True)
    g1 = lift >= 0.10
    g4 = diff_sig
    g5 = npos_cal >= 6
    g6 = max_fold_contribution <= 40
    print(f"  [{'PASS' if g1 else 'FAIL'}]  Static lift ≥ +0.10  ({lift:+.2f})", flush=True)
    print(f"  [{'PASS' if g4 else 'FAIL'}]  Paired diff CI excludes 0  ([{lo:+.3f}, {hi:+.3f}])",
          flush=True)
    print(f"  [{'PASS' if g5 else 'FAIL'}]  ≥ 6/9 folds positive  ({npos_cal}/9)", flush=True)
    print(f"  [{'PASS' if g6 else 'FAIL'}]  Max fold contribution ≤ 40%  ({max_fold_contribution:.0f}%)",
          flush=True)


if __name__ == "__main__":
    main()
