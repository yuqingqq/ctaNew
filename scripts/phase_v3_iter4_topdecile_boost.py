"""Iteration 4: asymmetric top-decile boost on btc_rvol_7d.

Pre-registered single rule (no fit, no tuning, never reduces below 1.0):
  scale = 1.5  if  rvol_7d_pctile_PIT >= 0.90
        = 1.0  otherwise

Only the top decile of the strongest cohort predictor is boosted. The bet is
that high-vol cohort PnL (+229 bps mean per cohort) is more capture-able when
concentrated by size than diluted across a continuous scaling rule.

Same 6 gates as iter 1/3.
"""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "iter3", REPO / "scripts/phase_v3_iter3_rvol7d_scaling.py")
i3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(i3)

OUT = REPO / "outputs/vBTC_iter_loop"

V31_REF_SHARPE = 2.23


def main():
    print("=== Iteration 4: top-decile rvol_7d boost (asymmetric, one-sided) ===",
          flush=True)
    print(f"  Pre-registered rule: scale = 1.5 if rvol_7d_pctile_PIT >= 0.90 else 1.0",
          flush=True)
    print(f"  V3.1 reference Sharpe = {V31_REF_SHARPE}\n", flush=True)

    records = pd.read_parquet(i3.svar.SLEEVES_PATH)
    records["time"] = pd.to_datetime(records["time"], utc=True)
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet",
                            columns=["symbol"])
    all_syms = sorted(apd["symbol"].unique())
    print(f"  loading close prices...", flush=True)
    t0 = time.time()
    close_wide = i3.svar.load_close_wide(all_syms)
    fwd_rets_4h = (close_wide.shift(-i3.HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)

    print(f"  loading BTC 7d rvol + computing PIT scale...", flush=True)
    t0 = time.time()
    btc_rvol = i3.load_btc_rvol_7d()
    records = i3.attach_scale(records, btc_rvol)
    # Override sleeve_scale with the iter 4 rule
    records["sleeve_scale"] = np.where(records["rvol_pctile"] >= 0.90, 1.5, 1.0)
    print(f"  done ({time.time()-t0:.0f}s)", flush=True)
    boost_pct = (records["sleeve_scale"] > 1.0).mean() * 100
    print(f"  Boost fires on {boost_pct:.1f}% of cycles\n", flush=True)

    # V3.1 baseline
    rec_v31 = records.copy(); rec_v31["sleeve_scale"] = 1.0
    df_v31 = i3.aggregate_sleeves_scaled(rec_v31, fwd_rets_4h, base_weight=1/6)
    v31_sh = i3._sharpe(df_v31["net_pnl_bps"])
    v31_dd = i3._max_dd(df_v31["net_pnl_bps"])
    v31_npos = sum(1 for _, g in df_v31.groupby("fold")
                    if i3._sharpe(g["net_pnl_bps"]) > 0)
    print(f"  V3.1 baseline:        Sharpe={v31_sh:+.3f}  maxDD={v31_dd:+.0f}  "
          f"PnL={df_v31['net_pnl_bps'].sum():+.0f}  folds+={v31_npos}/9", flush=True)

    # Iter 4
    df_b = i3.aggregate_sleeves_scaled(records, fwd_rets_4h, base_weight=1/6)
    b_sh = i3._sharpe(df_b["net_pnl_bps"])
    b_dd = i3._max_dd(df_b["net_pnl_bps"])
    b_npos = sum(1 for _, g in df_b.groupby("fold")
                  if i3._sharpe(g["net_pnl_bps"]) > 0)
    print(f"  Iter 4 top-dec boost: Sharpe={b_sh:+.3f}  maxDD={b_dd:+.0f}  "
          f"PnL={df_b['net_pnl_bps'].sum():+.0f}  folds+={b_npos}/9", flush=True)
    print(f"  Static lift: {b_sh - v31_sh:+.3f}\n", flush=True)
    df_b.to_csv(OUT / "per_cycle_iter4_topdec_boost.csv", index=False)

    # Per-fold
    print(f"  Per-fold breakdown:", flush=True)
    print(f"  {'fold':>4}  {'V3.1':>8}  {'Iter4':>8}  {'Δ':>7}  {'n_boost':>8}",
          flush=True)
    fold_diffs = {}
    for f in i3.OOS_FOLDS:
        v = df_v31[df_v31["fold"] == f]["net_pnl_bps"].sum()
        s = df_b[df_b["fold"] == f]["net_pnl_bps"].sum()
        d = s - v
        fold_diffs[f] = d
        n_boost_f = (df_b[df_b["fold"] == f]["scale"] > 1.0).sum()
        print(f"  {f:>4}  {v:>+8.0f}  {s:>+8.0f}  {d:>+7.0f}  {n_boost_f:>8d}",
              flush=True)
    pos_lift = sum(v for v in fold_diffs.values() if v > 0)
    max_fold_contribution = (max(fold_diffs.values()) / pos_lift * 100) if pos_lift > 0 else 0
    print(f"\n  Max single fold contribution: {max_fold_contribution:.0f}%", flush=True)

    # Paired bootstrap
    print(f"\n--- Paired V3.1 vs Iter4 bootstrap ---", flush=True)
    paired = df_v31[["time", "fold", "net_pnl_bps"]].rename(
        columns={"net_pnl_bps": "v31"}).merge(
        df_b[["time", "net_pnl_bps"]].rename(columns={"net_pnl_bps": "iter4"}),
        on="time")
    paired["diff"] = paired["iter4"] - paired["v31"]
    def _mean(x): return float(np.mean(x))
    mu, lo, hi = i3.block_bootstrap_ci(paired["diff"].to_numpy(), stat=_mean,
                                          block_size=7, n_boot=2000)
    print(f"  Mean diff: {mu:+.3f} bps/cycle  CI [{lo:+.3f}, {hi:+.3f}]", flush=True)
    diff_sig = (lo > 0) or (hi < 0)
    print(f"  Paired diff CI excludes 0: {'YES' if diff_sig else 'NO'}", flush=True)

    # Matched-boost-time placebo: shuffle WHICH cycles get the boost
    print(f"\n--- Matched-boost-time placebo (100 seeds, shuffled boost cycles) ---",
          flush=True)
    n_boost_total = int((records["sleeve_scale"] > 1.0).sum())
    placebo_sh = []
    t0 = time.time()
    for seed in range(100):
        rng = np.random.RandomState(seed)
        rec_p = records.copy()
        # Randomly assign boost to same number of traded cycles
        traded_idx = rec_p.index[rec_p["traded"]].tolist()
        if len(traded_idx) < n_boost_total:
            continue
        chosen = rng.choice(traded_idx, size=n_boost_total, replace=False)
        rec_p["sleeve_scale"] = 1.0
        rec_p.loc[chosen, "sleeve_scale"] = 1.5
        df_p = i3.aggregate_sleeves_scaled(rec_p, fwd_rets_4h, base_weight=1/6)
        placebo_sh.append(i3._sharpe(df_p["net_pnl_bps"]))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/100  ({time.time()-t0:.0f}s)", flush=True)
    pdf = pd.DataFrame({"seed": range(len(placebo_sh)), "sharpe": placebo_sh})
    pdf.to_csv(OUT / "iter4_matched_placebo.csv", index=False)
    p_sh = pdf["sharpe"].to_numpy()
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < b_sh).mean() * 100)
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}  p50={np.median(p_sh):+.2f}  "
          f"p95={p95:+.2f}  max={p_sh.max():+.2f}", flush=True)
    print(f"  Iter4 ({b_sh:+.2f}) ranks p{rank:.0f}  "
          f"beats_p95={'PASS' if b_sh > p95 else 'FAIL'}", flush=True)

    # Verdict
    print(f"\n=== Iteration 4 Verdict ===\n", flush=True)
    g1 = b_sh >= V31_REF_SHARPE + 0.10
    g2 = True  # pre-registered, no fit
    g3 = b_sh > p95
    g4 = diff_sig
    g5 = b_npos >= 6
    g6 = max_fold_contribution <= 40
    gates = [
        ("Static lift ≥ +0.10", g1, f"{b_sh:+.2f} - {V31_REF_SHARPE:+.2f} = {b_sh-V31_REF_SHARPE:+.2f}"),
        ("Pre-registered formula (no fit)", g2, "1.5 if pctile≥0.90 else 1.0, fixed"),
        ("Beats matched-boost placebo p95", g3, f"{b_sh:+.2f} vs p95 {p95:+.2f}"),
        ("Paired diff CI excludes 0", g4, f"[{lo:+.3f}, {hi:+.3f}]"),
        ("≥ 6/9 folds positive", g5, f"{b_npos}/9"),
        ("Max fold contribution ≤ 40%", g6, f"{max_fold_contribution:.0f}%"),
    ]
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name}  ({detail})", flush=True)
    all_pass = all(g[1] for g in gates)
    print(f"\n  Verdict: {'ACCEPT' if all_pass else 'REJECT'} top-decile boost",
          flush=True)


if __name__ == "__main__":
    main()
