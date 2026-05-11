"""Test C: PnL-trailing-based deleveraging.

Mechanism: monitor recent PnL. When trailing-K-cycle Sharpe drops below
threshold, scale position size down. Recover when trailing performance
improves.

This is fundamentally different from vol-scaling (which targets total vol)
or dispersion-sizing (which targets cycle-level confidence). This targets
REGIME bad periods directly via PnL signal.

Key trade-off:
  - Pro: cuts losses during bad streaks
  - Con: can be procyclical (sells low, buys high) — may miss recoveries
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
OUT_DIR = REPO / "outputs/vBTC_test_C_dd_deleverage"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def dd_deleverage(net_bps: np.ndarray, mode: str, **kwargs) -> tuple:
    """Apply drawdown-based deleveraging.

    Modes:
      'trailing_sharpe' — scale by trailing-K-cycle Sharpe vs threshold
      'cum_dd_pct'      — scale based on % drawdown from peak
      'consecutive_loss' — full size after consecutive losses
    """
    net = np.asarray(net_bps)
    sizes = np.ones_like(net)
    cum = np.cumsum(net)

    if mode == "trailing_sharpe":
        window = kwargs.get("window", 20)
        threshold_off = kwargs.get("threshold_off", -1.0)   # below this → scale down
        threshold_on = kwargs.get("threshold_on", 0.5)      # above this → scale up
        size_low = kwargs.get("size_low", 0.3)              # size when trailing bad
        size_high = kwargs.get("size_high", 1.0)
        current_size = 1.0
        for i in range(len(net)):
            past = net[max(0, i-window):i]
            if len(past) >= window // 2:
                trailing_sh = _sharpe(past)
                if trailing_sh < threshold_off:
                    current_size = size_low
                elif trailing_sh > threshold_on:
                    current_size = size_high
                # else hysteresis: keep current_size
            sizes[i] = current_size
    elif mode == "cum_dd_pct":
        peak = -np.inf
        threshold_dd = kwargs.get("threshold_dd", 0.20)
        size_drawdown = kwargs.get("size_drawdown", 0.5)
        for i in range(len(net)):
            peak = max(peak, cum[i] if i > 0 else 0)
            if peak > 0:
                dd_pct = (peak - cum[i]) / peak
                if dd_pct > threshold_dd:
                    sizes[i] = size_drawdown
                else:
                    sizes[i] = 1.0
            else:
                sizes[i] = 1.0
    elif mode == "consecutive_loss":
        max_consec = kwargs.get("max_consec", 5)
        scale_down = kwargs.get("scale_down", 0.5)
        consec = 0
        for i in range(len(net)):
            if i > 0 and net[i-1] < 0:
                consec += 1
            else:
                consec = 0
            sizes[i] = scale_down if consec >= max_consec else 1.0
    else:
        sizes[:] = 1.0

    scaled = sizes * net
    return scaled, sizes


def main():
    print(f"Loading cycle data...", flush=True)
    df = pd.read_csv(CYCLE_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    net = df["net_bps"].to_numpy()
    print(f"  {len(df)} cycles, total PnL={net.sum():+.0f} bps, "
          f"baseline Sharpe={_sharpe(net):+.2f}", flush=True)

    prod_folds_dates = {
        5: (pd.Timestamp("2025-11-24", tz="UTC"), pd.Timestamp("2025-12-26", tz="UTC")),
        6: (pd.Timestamp("2025-12-26", tz="UTC"), pd.Timestamp("2026-01-27", tz="UTC")),
        7: (pd.Timestamp("2026-01-27", tz="UTC"), pd.Timestamp("2026-02-28", tz="UTC")),
        8: (pd.Timestamp("2026-02-28", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
        9: (pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")),
    }
    df["fold"] = [next((f for f, (s, e) in prod_folds_dates.items()
                          if s <= pd.Timestamp(t) < e), None)
                   for t in df["time"]]

    print(f"\n=== Test C: Drawdown-based deleveraging ===", flush=True)
    print(f"  {'config':<35}  {'Sharpe':>7}  {'std':>6}  {'max_DD':>7}  {'mean':>6}  {'mean_size':>9}",
          flush=True)
    results = []

    configs = [
        ("baseline", "none", {}),
        ("trail_sh<-1.0_size=0.3", "trailing_sharpe",
            {"window": 20, "threshold_off": -1.0, "threshold_on": 0.5, "size_low": 0.3}),
        ("trail_sh<0.0_size=0.3",  "trailing_sharpe",
            {"window": 20, "threshold_off": 0.0, "threshold_on": 1.0, "size_low": 0.3}),
        ("trail_sh<-1.0_size=0.5", "trailing_sharpe",
            {"window": 20, "threshold_off": -1.0, "threshold_on": 0.5, "size_low": 0.5}),
        ("trail_W=40_sh<-0.5",     "trailing_sharpe",
            {"window": 40, "threshold_off": -0.5, "threshold_on": 0.5, "size_low": 0.4}),
        ("dd_pct>20%_size=0.5",    "cum_dd_pct",
            {"threshold_dd": 0.20, "size_drawdown": 0.5}),
        ("consec_loss>=5_size=0.5", "consecutive_loss",
            {"max_consec": 5, "scale_down": 0.5}),
    ]
    for label, mode, params in configs:
        if mode == "none":
            scaled = net.copy()
            sizes = np.ones_like(net)
        else:
            scaled, sizes = dd_deleverage(net, mode, **params)
        sh, lo, hi = block_bootstrap_ci(scaled, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd(scaled)

        df_temp = df.copy(); df_temp["scaled"] = scaled
        per_fold = {}
        for fid in sorted(prod_folds_dates.keys()):
            n_f = df_temp[df_temp["fold"] == fid]["scaled"].to_numpy()
            if len(n_f) >= 3:
                per_fold[fid] = _sharpe(n_f)
        results.append({"config": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "std_bps": scaled.std(), "max_dd": max_dd,
                          "mean_net": scaled.mean(), "mean_size": sizes.mean(),
                          **{f"sh_f{f}": v for f, v in per_fold.items()}})
        print(f"  {label:<35}  {sh:>+7.2f}  {scaled.std():>6.1f}  {max_dd:>+7.0f}  "
              f"{scaled.mean():>+6.2f}  {sizes.mean():>9.2f}", flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    print(f"  {'config':<35}  " + " ".join(f"{'fold' + str(f):>8}" for f in [5, 6, 7, 8, 9]),
          flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in [5, 6, 7, 8, 9])
        print(f"  {r['config']:<35}  " + cells, flush=True)

    # Variance analysis
    print(f"\n=== Per-fold variance reduction ===", flush=True)
    base = next(r for r in results if r["config"] == "baseline")
    base_per_fold = [base[f"sh_f{f}"] for f in [5, 6, 7, 8, 9]]
    base_per_fold_std = float(np.std(base_per_fold))
    print(f"  Baseline per-fold Sharpe std: {base_per_fold_std:.2f}", flush=True)
    for r in results:
        if r["config"] == "baseline": continue
        per_fold = [r[f"sh_f{f}"] for f in [5, 6, 7, 8, 9]]
        std_change = float(np.std(per_fold)) / base_per_fold_std * 100
        print(f"  {r['config']:<35}  per-fold std = {float(np.std(per_fold)):.2f}  "
              f"({std_change:.0f}% of baseline)", flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "test_C_results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
