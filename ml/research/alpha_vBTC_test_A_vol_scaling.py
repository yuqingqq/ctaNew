"""Test A: Vol-scaling.

Apply trailing-N-cycle vol scaling to the existing strategy PnL.
For each cycle, scale position size by target_vol / trailing_std.

Configurations:
  baseline (no scaling)
  target_vol = 80 bps   (most aggressive scaling, reduces ~70% of cycles)
  target_vol = 100 bps  (medium, my recommendation)
  target_vol = 150 bps  (lighter, only scales extremes)

Output:
  Sharpe, std, max DD, per-fold breakdown for each config.
  Compares to baseline.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs import block_bootstrap_ci

CYCLE_PATH = REPO / "outputs/vBTC_dd_analysis/cycle_pnl.csv"
OUT_DIR = REPO / "outputs/vBTC_test_A_vol_scaling"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
TRAILING_WINDOW = 60   # 60 cycles trailing for std estimation
WARMUP = 30   # cycles before scaling kicks in
SIZE_CAP = (0.25, 4.0)


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def vol_scale(net_bps: np.ndarray, target_vol: float, window: int = TRAILING_WINDOW,
                warmup: int = WARMUP, cap: tuple = SIZE_CAP) -> np.ndarray:
    """Apply trailing-window vol-scaling to per-cycle net_bps."""
    net = np.asarray(net_bps)
    scaled = np.zeros_like(net)
    sizes = np.ones_like(net)
    for i in range(len(net)):
        if i < warmup:
            sizes[i] = 1.0
        else:
            past = net[max(0, i - window):i]
            past_clean = past[~np.isnan(past)]
            if len(past_clean) < 10:
                sizes[i] = 1.0
            else:
                std = float(past_clean.std())
                if std < 1e-6:
                    sizes[i] = cap[1]
                else:
                    sizes[i] = np.clip(target_vol / std, cap[0], cap[1])
        scaled[i] = sizes[i] * net[i]
    return scaled, sizes


def assign_fold(times, prod_folds_dates):
    """Map each time to its fold by date range."""
    folds = []
    for t in times:
        t = pd.Timestamp(t)
        fid = None
        for f, (start, end) in prod_folds_dates.items():
            if start <= t < end:
                fid = f; break
        folds.append(fid)
    return folds


def main():
    print(f"Loading cycle data...", flush=True)
    df = pd.read_csv(CYCLE_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    print(f"  {len(df)} cycles, total PnL={df['net_bps'].sum():+.0f} bps", flush=True)

    # Define fold boundaries from times
    prod_folds_dates = {
        5: (pd.Timestamp("2025-11-24", tz="UTC"), pd.Timestamp("2025-12-26", tz="UTC")),
        6: (pd.Timestamp("2025-12-26", tz="UTC"), pd.Timestamp("2026-01-27", tz="UTC")),
        7: (pd.Timestamp("2026-01-27", tz="UTC"), pd.Timestamp("2026-02-28", tz="UTC")),
        8: (pd.Timestamp("2026-02-28", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
        9: (pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")),
    }

    print(f"\n=== Vol-scaling test ===", flush=True)
    print(f"  {'config':<22}  {'Sharpe':>7}  {'std':>6}  {'max_DD':>7}  {'mean':>6}  {'mean_size':>9}",
          flush=True)
    results = []
    for label, target_vol in [
        ("baseline_no_scale", None),
        ("target_vol_80", 80.0),
        ("target_vol_100", 100.0),
        ("target_vol_150", 150.0),
    ]:
        net = df["net_bps"].to_numpy()
        if target_vol is not None:
            scaled, sizes = vol_scale(net, target_vol)
        else:
            scaled, sizes = net.copy(), np.ones_like(net)
        sh, lo, hi = block_bootstrap_ci(scaled, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd(scaled)
        mean_size = float(np.mean(sizes))
        std = float(np.std(scaled))

        # Per-fold Sharpe
        df_temp = df.copy()
        df_temp["scaled"] = scaled
        df_temp["fold"] = assign_fold(df_temp["time"].tolist(), prod_folds_dates)
        per_fold = {}
        for fid in sorted(prod_folds_dates.keys()):
            n_f = df_temp[df_temp["fold"] == fid]["scaled"].to_numpy()
            if len(n_f) >= 3:
                per_fold[fid] = _sharpe(n_f)

        results.append({"config": label, "target_vol": target_vol or 0,
                          "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "std_bps": std, "max_dd": max_dd,
                          "mean_net": scaled.mean(), "mean_size": mean_size,
                          **{f"sh_f{f}": v for f, v in per_fold.items()}})
        print(f"  {label:<22}  {sh:>+7.2f}  {std:>6.1f}  {max_dd:>+7.0f}  "
              f"{scaled.mean():>+6.2f}  {mean_size:>9.2f}", flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    print(f"  {'config':<22}  " + " ".join(f"{'fold' + str(f):>8}" for f in [5, 6, 7, 8, 9]),
          flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in [5, 6, 7, 8, 9])
        print(f"  {r['config']:<22}  " + cells, flush=True)

    # Stats: variance reduction
    print(f"\n=== Variance reduction summary ===", flush=True)
    base = next(r for r in results if r["config"] == "baseline_no_scale")
    for r in results:
        if r["config"] == "baseline_no_scale": continue
        std_pct = r["std_bps"] / base["std_bps"] * 100
        sh_diff = r["sharpe"] - base["sharpe"]
        dd_pct = r["max_dd"] / base["max_dd"] * 100 if base["max_dd"] != 0 else 0
        print(f"  {r['config']:<22}  std={std_pct:.0f}% of baseline  "
              f"Δsharpe={sh_diff:+.2f}  DD={dd_pct:.0f}% of baseline", flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "vol_scaling_results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
