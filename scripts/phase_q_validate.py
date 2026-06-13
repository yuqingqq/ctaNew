"""Phase Q step 3-4: rebuild sleeves from WINNER_23 predictions + validate.

After phase_q_winner23_retrain.py produces all_predictions_w23.parquet:
  Step 3: rebuild sleeves with same production protocol (rolling-IC 180/90 +
          filter_refill_90d_mean + conv_gate 252/p30 + flat_real + K=3).
  Step 4: V3.1 equal-weight 6-sleeve aggregation.
  Step 5: paired bootstrap vs current V3.1 (WINNER_21).

Validation gates (same as iter loop):
  Static lift ≥ +0.10 over V3.1 WINNER_21
  Paired diff vs V3.1 CI excludes 0
  ≥ 6/9 folds positive
  Beats matched-basket placebo p95
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
from collections import deque, defaultdict
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

OUT = REPO / "outputs/vBTC_phase_Q"

HORIZON_ENTRY = 48
HOLD_BARS = 288
N_SLEEVES = 6
COST_PER_UNIT_ABS_DELTA = 2.25
CYCLES_PER_YEAR = (288 * 365) / HORIZON_ENTRY
OOS_FOLDS = list(range(1, 10))
V31_REF_SHARPE = 2.23


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
    print("=== Phase Q step 3-4: rebuild sleeves + validate ===\n", flush=True)

    # Load predictions
    pred_path = OUT / "all_predictions_w23.parquet"
    if not pred_path.exists():
        print(f"ABORT: {pred_path} does not exist. Run phase_q_winner23_retrain.py first.",
              flush=True)
        return
    apd = pd.read_parquet(pred_path)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    if "return_pct" not in apd.columns:
        # Need return_pct for the protocol — re-derive from panel
        panel = pd.read_parquet(OUT / "panel_w23.parquet",
                                  columns=["symbol", "open_time", "return_pct"])
        panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
        apd = apd.merge(panel, on=["symbol", "open_time"], how="left")
    print(f"  loaded {len(apd):,} WINNER_23 predictions", flush=True)

    # Build rolling-IC universe (same machinery)
    listings = psl.get_listings()
    panel_first_obs = apd.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            listings[sym] = t

    def elig_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in apd["symbol"].unique()
                  if listings.get(s) and listings[s] <= cutoff}

    tgt = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled = tgt[::HORIZON_ENTRY]
    print(f"  building rolling-IC universe (180/90) on {len(sampled)} target times...",
          flush=True)
    t0 = time.time()
    universe = psl.build_rolling_ic_universe(apd, sampled, psl.TOP_N, elig_at)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Build sleeves (Phase M K=3 production protocol)
    print(f"  building sleeves with K=3 + filter_refill + conv_gate + flat_real...",
          flush=True)
    t0 = time.time()
    sleeves_w23 = psl.run_production_protocol_save_sleeves(apd, universe)
    print(f"  done: {len(sleeves_w23)} cycles ({time.time()-t0:.0f}s)", flush=True)
    sleeves_w23["time"] = pd.to_datetime(sleeves_w23["time"], utc=True)
    sleeves_w23.to_parquet(OUT / "production_sleeves_w23.parquet", index=False)

    n_traded = sleeves_w23["traded"].sum()
    print(f"  traded cycles: {n_traded} / {len(sleeves_w23)}", flush=True)

    # Load close prices for fwd_rets
    apd_orig = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet",
                                  columns=["symbol"])
    all_syms = sorted(apd_orig["symbol"].unique())
    print(f"  loading close prices...", flush=True)
    t0 = time.time()
    close_wide = svar.load_close_wide(all_syms)
    fwd_rets_4h = (close_wide.shift(-HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    # Load current V3.1 baseline (WINNER_21) sleeves
    sleeves_w21 = pd.read_parquet(svar.SLEEVES_PATH)
    sleeves_w21["time"] = pd.to_datetime(sleeves_w21["time"], utc=True)

    # Run V3.1 aggregation on both
    print(f"\n  V3.1 aggregation on WINNER_21 baseline...", flush=True)
    t0 = time.time()
    df_w21 = svar.aggregate_sleeves_variant(sleeves_w21, fwd_rets_4h,
                                                  N_SLEEVES, HOLD_BARS,
                                                  sleeve_weights=[1/6] * 6)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    print(f"  V3.1 aggregation on WINNER_23...", flush=True)
    t0 = time.time()
    df_w23 = svar.aggregate_sleeves_variant(sleeves_w23, fwd_rets_4h,
                                                  N_SLEEVES, HOLD_BARS,
                                                  sleeve_weights=[1/6] * 6)
    print(f"  done ({time.time()-t0:.0f}s)\n", flush=True)

    df_w23.to_csv(OUT / "per_cycle_w23_v31.csv", index=False)

    # Stats
    sh_w21 = _sharpe(df_w21["net_pnl_bps"])
    sh_w23 = _sharpe(df_w23["net_pnl_bps"])
    dd_w21 = _max_dd(df_w21["net_pnl_bps"])
    dd_w23 = _max_dd(df_w23["net_pnl_bps"])
    npos_w21 = sum(1 for _, g in df_w21.groupby("fold")
                     if _sharpe(g["net_pnl_bps"]) > 0)
    npos_w23 = sum(1 for _, g in df_w23.groupby("fold")
                     if _sharpe(g["net_pnl_bps"]) > 0)
    print(f"  V3.1 WINNER_21:  Sharpe={sh_w21:+.3f}  maxDD={dd_w21:+.0f}  "
          f"PnL={df_w21['net_pnl_bps'].sum():+.0f}  folds+={npos_w21}/9", flush=True)
    print(f"  V3.1 WINNER_23:  Sharpe={sh_w23:+.3f}  maxDD={dd_w23:+.0f}  "
          f"PnL={df_w23['net_pnl_bps'].sum():+.0f}  folds+={npos_w23}/9", flush=True)
    lift = sh_w23 - sh_w21
    print(f"  Static lift (W23 - W21): {lift:+.3f}\n", flush=True)

    # Per-fold breakdown
    print(f"  Per-fold breakdown:", flush=True)
    print(f"  {'fold':>4}  {'W21':>8}  {'W23':>8}  {'Δ':>7}", flush=True)
    fold_diffs = {}
    for f in OOS_FOLDS:
        a = df_w21[df_w21["fold"] == f]["net_pnl_bps"].sum()
        b = df_w23[df_w23["fold"] == f]["net_pnl_bps"].sum()
        d = b - a
        fold_diffs[f] = d
        print(f"  {f:>4}  {a:>+8.0f}  {b:>+8.0f}  {d:>+7.0f}", flush=True)
    pos_lift = sum(v for v in fold_diffs.values() if v > 0)
    max_fold_contribution = (max(fold_diffs.values()) / pos_lift * 100) if pos_lift > 0 else 0
    print(f"\n  Max single fold contribution: {max_fold_contribution:.0f}%", flush=True)

    # Paired bootstrap
    print(f"\n--- Paired W21 vs W23 bootstrap ---", flush=True)
    paired = df_w21[["time", "fold", "net_pnl_bps"]].rename(
        columns={"net_pnl_bps": "w21"}).merge(
        df_w23[["time", "net_pnl_bps"]].rename(columns={"net_pnl_bps": "w23"}),
        on="time")
    paired["diff"] = paired["w23"] - paired["w21"]
    def _mean(x): return float(np.mean(x))
    mu, lo, hi = block_bootstrap_ci(paired["diff"].to_numpy(), stat=_mean,
                                        block_size=7, n_boot=2000)
    print(f"  Mean diff: {mu:+.3f} bps/cycle  CI [{lo:+.3f}, {hi:+.3f}]", flush=True)
    diff_sig = (lo > 0) or (hi < 0)
    print(f"  Paired diff CI excludes 0: {'YES' if diff_sig else 'NO'}", flush=True)

    # Matched-basket placebo on W23 (same machinery as V3.1 placebo)
    print(f"\n--- Matched-basket placebo (W23 universe, 100 seeds) ---", flush=True)
    t0 = time.time()
    placebo_sh = []
    for seed in range(100):
        df_p = svar.aggregate_sleeves_variant(sleeves_w23, fwd_rets_4h,
                                                  N_SLEEVES, HOLD_BARS,
                                                  sleeve_weights=[1/6] * 6,
                                                  placebo_universe=universe,
                                                  placebo_seed=seed)
        placebo_sh.append(_sharpe(df_p["net_pnl_bps"]))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/100  ({time.time()-t0:.0f}s)", flush=True)
    pdf = pd.DataFrame({"seed": range(100), "sharpe": placebo_sh})
    pdf.to_csv(OUT / "matched_placebo_w23.csv", index=False)
    p_sh = pdf["sharpe"].to_numpy()
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < sh_w23).mean() * 100)
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}  p50={np.median(p_sh):+.2f}  "
          f"p95={p95:+.2f}  max={p_sh.max():+.2f}", flush=True)
    print(f"  W23 ({sh_w23:+.2f}) ranks p{rank:.0f}  "
          f"beats_p95={'PASS' if sh_w23 > p95 else 'FAIL'}", flush=True)

    # Verdict
    print(f"\n=== Phase Q Verdict ===\n", flush=True)
    g1 = lift >= 0.10
    g2 = True  # WINNER_23 is the pre-registered feature set
    g3 = sh_w23 > p95
    g4 = diff_sig
    g5 = npos_w23 >= 6
    g6 = max_fold_contribution <= 40
    gates = [
        ("Static lift ≥ +0.10", g1, f"{sh_w23:+.2f} - {sh_w21:+.2f} = {lift:+.2f}"),
        ("Pre-registered features (no fit)", g2, "WINNER_23 = WINNER_21 + 2 fixed"),
        ("Beats matched-basket placebo p95", g3, f"{sh_w23:+.2f} vs p95 {p95:+.2f}"),
        ("Paired diff CI excludes 0", g4, f"[{lo:+.3f}, {hi:+.3f}]"),
        ("≥ 6/9 folds positive", g5, f"{npos_w23}/9"),
        ("Max fold contribution ≤ 40%", g6, f"{max_fold_contribution:.0f}%"),
    ]
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}  ({detail})", flush=True)
    all_pass = all(g[1] for g in gates)
    print(f"\n  Verdict: {'ACCEPT' if all_pass else 'REJECT'} WINNER_23 retrain",
          flush=True)


if __name__ == "__main__":
    main()
