"""DD anatomy: where does the variance/drawdown come from?

Uses the validated rolling-IC cycle data (with hold + Test C overlay applied).

Analyses:
  1. Cumulative PnL trajectory and worst DD episode
  2. Worst-30 cycles attribution (% of total adverse PnL)
  3. Per-fold drawdown distribution
  4. Drawdown clustering — are bad cycles consecutive or scattered?
  5. Compare baseline vs overlay-adjusted DD
  6. Time of worst DD episode + duration

This is a fast post-process on existing cycle CSVs.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CYCLE_PATH = REPO / "outputs/vBTC_current_validation/cycles_rolling_ic_universe.csv"
OUT_DIR = REPO / "outputs/vBTC_dd_anatomy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd_with_episode(net):
    cum = np.cumsum(net)
    peak = np.maximum.accumulate(cum)
    dd_series = cum - peak
    max_dd_idx = int(np.argmin(dd_series))
    max_dd = float(dd_series[max_dd_idx])
    # Find episode start (last time DD was 0 before max)
    ep_start = max_dd_idx
    while ep_start > 0 and dd_series[ep_start - 1] < 0:
        ep_start -= 1
    # Find episode end (first time after max where DD recovers to 0)
    ep_end = max_dd_idx
    while ep_end < len(net) - 1 and dd_series[ep_end + 1] < 0:
        ep_end += 1
    return max_dd, ep_start, max_dd_idx, ep_end


def apply_dd_overlay(net_bps, threshold_dd=0.20, size_drawdown=0.3):
    net = np.asarray(net_bps, dtype=float)
    sizes = np.ones_like(net)
    cum = np.cumsum(net)
    peak = -np.inf
    for i in range(len(net)):
        peak = max(peak, cum[i] if i > 0 else 0)
        if peak > 0:
            dd_pct = (peak - cum[i]) / peak
            sizes[i] = size_drawdown if dd_pct > threshold_dd else 1.0
    return sizes * net, sizes


def main():
    print(f"Loading cycle data: {CYCLE_PATH.name}", flush=True)
    df = pd.read_csv(CYCLE_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"  n={len(df)}, mean={df['net_bps'].mean():+.2f}, "
          f"std={df['net_bps'].std():.1f}, baseline_Sharpe={_sharpe(df['net_bps'].to_numpy()):+.2f}",
          flush=True)

    net = df["net_bps"].to_numpy()
    overlay_net, sizes = apply_dd_overlay(net)
    df["net_with_overlay"] = overlay_net

    print(f"\n=== Distribution analysis ===", flush=True)
    for label, arr in [("baseline (no overlay)", net), ("with Test C overlay", overlay_net)]:
        print(f"\n  {label}:", flush=True)
        print(f"    mean={arr.mean():+.2f}, std={arr.std():.1f}, "
              f"Sharpe={_sharpe(arr):+.2f}", flush=True)
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            print(f"    p{p:>2}: {np.percentile(arr, p):+7.0f}", flush=True)

    print(f"\n=== Bottom-N cycles attribution ===", flush=True)
    print(f"  {'slice':<12}  {'count':>5}  {'sum_net':>9}  {'% of total adverse':>20}", flush=True)
    sorted_net = np.sort(net)
    total_adverse = sorted_net[sorted_net < 0].sum()
    for n_slice in [5, 10, 25, 50, 100]:
        slice_sum = sorted_net[:n_slice].sum()
        pct = (slice_sum / total_adverse) * 100 if total_adverse < 0 else 0
        print(f"  bottom {n_slice:>2}    {n_slice:>5}  {slice_sum:>+9.0f}  {pct:>19.1f}%", flush=True)

    print(f"\n=== Worst DD episode (baseline) ===", flush=True)
    max_dd, ep_start, ep_max, ep_end = _max_dd_with_episode(net)
    ep_data = df.iloc[ep_start:ep_end + 1]
    print(f"  max_DD: {max_dd:+.0f} bps", flush=True)
    print(f"  start: {df.iloc[ep_start]['time']}", flush=True)
    print(f"  trough: {df.iloc[ep_max]['time']} (cycle index {ep_max})", flush=True)
    print(f"  end: {df.iloc[ep_end]['time']}", flush=True)
    print(f"  duration: {ep_end - ep_start + 1} cycles "
          f"({(ep_end - ep_start + 1) * 4} hours = {(ep_end - ep_start + 1) / 6:.1f} days)",
          flush=True)
    print(f"  drawdown phase ({ep_max - ep_start + 1} cycles): "
          f"sum_net={df.iloc[ep_start:ep_max+1]['net_bps'].sum():+.0f}", flush=True)
    print(f"  recovery phase ({ep_end - ep_max} cycles): "
          f"sum_net={df.iloc[ep_max+1:ep_end+1]['net_bps'].sum():+.0f}", flush=True)
    if "fold" in df.columns:
        ep_folds = df.iloc[ep_start:ep_end+1]["fold"].dropna().unique()
        print(f"  spans folds: {sorted(ep_folds)}", flush=True)

    print(f"\n=== Worst DD episode (with overlay) ===", flush=True)
    max_dd_o, ep_start_o, ep_max_o, ep_end_o = _max_dd_with_episode(overlay_net)
    print(f"  max_DD: {max_dd_o:+.0f} bps (vs baseline {max_dd:+.0f}, "
          f"reduction {(1 - max_dd_o/max_dd) * 100:+.0f}%)", flush=True)
    print(f"  start: {df.iloc[ep_start_o]['time']}", flush=True)
    print(f"  trough: {df.iloc[ep_max_o]['time']}", flush=True)
    print(f"  end: {df.iloc[ep_end_o]['time']}", flush=True)
    print(f"  duration: {ep_end_o - ep_start_o + 1} cycles", flush=True)

    print(f"\n=== Per-fold breakdown ===", flush=True)
    print(f"  {'fold':>5}  {'n':>4}  {'mean':>6}  {'std':>6}  "
          f"{'Sharpe':>7}  {'maxDD':>7}  {'pos_pct':>7}", flush=True)
    fold_summary = []
    for fid in sorted(df["fold"].dropna().unique()):
        fdat = df[df["fold"] == fid]["net_bps"].to_numpy()
        if len(fdat) < 3: continue
        sh = _sharpe(fdat)
        cum = np.cumsum(fdat)
        peak = np.maximum.accumulate(cum)
        max_dd_f = float((cum - peak).min())
        pos_pct = (fdat > 0).sum() / len(fdat) * 100
        fold_summary.append({"fold": int(fid), "n": len(fdat), "mean": fdat.mean(),
                              "std": fdat.std(), "sharpe": sh, "max_dd": max_dd_f,
                              "pos_pct": pos_pct})
        print(f"  {int(fid):>5}  {len(fdat):>4}  {fdat.mean():>+6.2f}  {fdat.std():>6.1f}  "
              f"{sh:>+7.2f}  {max_dd_f:>+7.0f}  {pos_pct:>6.1f}%", flush=True)

    print(f"\n=== Cluster analysis: are bad cycles consecutive? ===", flush=True)
    # For each cycle, count consecutive losing cycles INCLUDING current
    consec = np.zeros(len(net), dtype=int)
    for i in range(len(net)):
        if net[i] < 0:
            consec[i] = consec[i-1] + 1 if i > 0 else 1
    print(f"  Loss streaks distribution:", flush=True)
    streak_max = consec.max()
    print(f"    max streak: {streak_max} consecutive losing cycles", flush=True)
    print(f"    streaks >= 3: {(consec >= 3).sum()} positions", flush=True)
    print(f"    streaks >= 5: {(consec >= 5).sum()} positions", flush=True)
    print(f"    streaks >= 10: {(consec >= 10).sum()} positions", flush=True)
    # Find longest losing streak episodes
    in_streak = False
    streaks = []
    streak_start = 0
    for i in range(len(net)):
        if net[i] < 0:
            if not in_streak:
                streak_start = i
                in_streak = True
        else:
            if in_streak:
                streaks.append((streak_start, i - 1, df.iloc[streak_start]["time"],
                                  df.iloc[i-1]["time"], i - streak_start))
                in_streak = False
    if in_streak:
        streaks.append((streak_start, len(net) - 1, df.iloc[streak_start]["time"],
                          df.iloc[-1]["time"], len(net) - streak_start))
    streaks.sort(key=lambda s: -s[4])
    print(f"  Top-5 longest losing streaks:", flush=True)
    for s in streaks[:5]:
        s_start_idx, s_end_idx, t_s, t_e, length = s
        sum_loss = net[s_start_idx:s_end_idx+1].sum()
        print(f"    {length:>3} cycles ({t_s} to {t_e}): sum {sum_loss:+.0f} bps",
              flush=True)

    print(f"\n=== Variance source decomposition ===", flush=True)
    # Variance contribution from extreme cycles
    abs_net = np.abs(net - net.mean())
    sorted_idx = np.argsort(-abs_net)  # largest deviations first
    total_var = abs_net.sum()  # actually mean abs deviation; close enough for breakdown
    top_5_pct = abs_net[sorted_idx[:int(len(net) * 0.05)]].sum() / total_var * 100
    top_10_pct = abs_net[sorted_idx[:int(len(net) * 0.10)]].sum() / total_var * 100
    top_25_pct = abs_net[sorted_idx[:int(len(net) * 0.25)]].sum() / total_var * 100
    print(f"  Top-5% extreme cycles contribute {top_5_pct:.1f}% of mean-abs-deviation", flush=True)
    print(f"  Top-10% extreme cycles contribute {top_10_pct:.1f}%", flush=True)
    print(f"  Top-25% extreme cycles contribute {top_25_pct:.1f}%", flush=True)

    pd.DataFrame(fold_summary).to_csv(OUT_DIR / "fold_summary.csv", index=False)
    df.to_csv(OUT_DIR / "cycles_with_overlay.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
