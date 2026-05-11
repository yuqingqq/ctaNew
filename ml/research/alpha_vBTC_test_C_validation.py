"""Validate Test C (trailing-DD deleveraging) on the corrected pipeline.

Loads per-cycle PnL from current_validation (rolling-IC universe, 180/90 cadence,
expanding training, walk-forward 9 folds) and applies overlay variants.

Variants tested:
  1. baseline           — no overlay
  2. dd_pct>10%_size=0.5
  3. dd_pct>15%_size=0.5
  4. dd_pct>20%_size=0.5  (the Test C winner from earlier)
  5. dd_pct>30%_size=0.5
  6. trail_sh<-1.0_size=0.3 (alternative mechanism)
  7. consec_loss>=5_size=0.5
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ml.research.alpha_v4_xs import block_bootstrap_ci

CYCLE_DIR = REPO / "outputs/vBTC_current_validation"
OUT_DIR = REPO / "outputs/vBTC_test_C_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
OOS_FOLDS = list(range(1, 10))
PROD_FOLDS = [5, 6, 7, 8, 9]


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def overlay(net_bps, mode, **kwargs):
    net = np.asarray(net_bps, dtype=float)
    sizes = np.ones_like(net)

    if mode == "none":
        return net.copy(), sizes

    if mode == "cum_dd_pct":
        cum = np.cumsum(net)
        peak = -np.inf
        threshold_dd = kwargs["threshold_dd"]
        size_drawdown = kwargs["size_drawdown"]
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
    elif mode == "trailing_sharpe":
        window = kwargs.get("window", 20)
        threshold_off = kwargs.get("threshold_off", -1.0)
        threshold_on = kwargs.get("threshold_on", 0.5)
        size_low = kwargs.get("size_low", 0.3)
        size_high = kwargs.get("size_high", 1.0)
        cur_size = 1.0
        for i in range(len(net)):
            past = net[max(0, i-window):i]
            if len(past) >= window // 2:
                tr_sh = _sharpe(past)
                if tr_sh < threshold_off: cur_size = size_low
                elif tr_sh > threshold_on: cur_size = size_high
            sizes[i] = cur_size
    elif mode == "consecutive_loss":
        max_consec = kwargs["max_consec"]
        scale_down = kwargs["scale_down"]
        consec = 0
        for i in range(len(net)):
            if i > 0 and net[i-1] < 0:
                consec += 1
            else:
                consec = 0
            sizes[i] = scale_down if consec >= max_consec else 1.0

    scaled = sizes * net
    return scaled, sizes


def main():
    for cfg in ["static_universe", "rolling_ic_universe"]:
        path = CYCLE_DIR / f"cycles_{cfg}.csv"
        if not path.exists():
            print(f"Missing {path}; skip {cfg}", flush=True)
            continue
        df = pd.read_csv(path)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)
        net = df["net_bps"].to_numpy()

        print(f"\n{'=' * 90}", flush=True)
        print(f"{cfg.upper()}: n={len(net)}, mean={net.mean():+.2f}, "
              f"baseline Sharpe={_sharpe(net):+.2f}, max_DD={_max_dd(net):+.0f}", flush=True)
        print(f"{'=' * 90}", flush=True)

        configs = [
            ("baseline",                "none", {}),
            ("dd_pct>10%_size=0.5",     "cum_dd_pct", {"threshold_dd": 0.10, "size_drawdown": 0.5}),
            ("dd_pct>15%_size=0.5",     "cum_dd_pct", {"threshold_dd": 0.15, "size_drawdown": 0.5}),
            ("dd_pct>20%_size=0.5",     "cum_dd_pct", {"threshold_dd": 0.20, "size_drawdown": 0.5}),
            ("dd_pct>30%_size=0.5",     "cum_dd_pct", {"threshold_dd": 0.30, "size_drawdown": 0.5}),
            ("dd_pct>20%_size=0.3",     "cum_dd_pct", {"threshold_dd": 0.20, "size_drawdown": 0.3}),
            ("trail_sh<-1.0_size=0.3",  "trailing_sharpe",
                {"window": 20, "threshold_off": -1.0, "threshold_on": 0.5, "size_low": 0.3}),
            ("trail_W=40_sh<-0.5_s=0.4","trailing_sharpe",
                {"window": 40, "threshold_off": -0.5, "threshold_on": 0.5, "size_low": 0.4}),
            ("consec_loss>=5_size=0.5", "consecutive_loss", {"max_consec": 5, "scale_down": 0.5}),
            ("consec_loss>=3_size=0.5", "consecutive_loss", {"max_consec": 3, "scale_down": 0.5}),
        ]

        rows = []
        print(f"  {'config':<28}  {'Sharpe':>7}  {'CI':>17}  {'max_DD':>7}  "
              f"{'std':>6}  {'mean':>6}  {'mean_size':>9}", flush=True)
        for label, mode, params in configs:
            scaled, sizes = overlay(net, mode, **params)
            sh, lo, hi = block_bootstrap_ci(scaled, statistic=_sharpe, block_size=7, n_boot=2000)
            max_dd = _max_dd(scaled)
            df_temp = df.copy(); df_temp["scaled"] = scaled
            per_fold = {}
            for fid in OOS_FOLDS:
                fdat = df_temp[df_temp["fold"] == fid]["scaled"].to_numpy()
                if len(fdat) >= 3:
                    per_fold[fid] = _sharpe(fdat)
            prod_mask = df_temp["fold"].isin(PROD_FOLDS)
            prod_sh = _sharpe(df_temp.loc[prod_mask, "scaled"].to_numpy())
            rows.append({"config": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "max_dd": max_dd, "std": scaled.std(), "mean": scaled.mean(),
                          "mean_size": sizes.mean(), "prod_sharpe": prod_sh,
                          **{f"sh_f{f}": v for f, v in per_fold.items()}})
            print(f"  {label:<28}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
                  f"{max_dd:>+7.0f}  {scaled.std():>6.1f}  {scaled.mean():>+6.2f}  "
                  f"{sizes.mean():>9.2f}", flush=True)

        # Per-fold breakdown
        print(f"\n  Per-fold Sharpe:", flush=True)
        print(f"  {'config':<28}  " + " ".join(f"{'f' + str(f):>6}" for f in OOS_FOLDS), flush=True)
        for r in rows:
            cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in OOS_FOLDS)
            print(f"  {r['config']:<28}  " + cells, flush=True)

        # Prod folds 5-9 isolation
        print(f"\n  Prod-folds-5-9 Sharpe:", flush=True)
        for r in rows:
            base_change = ""
            if r['config'] != "baseline":
                base = next(rr for rr in rows if rr['config'] == 'baseline')
                base_change = f" (Δ={r['prod_sharpe'] - base['prod_sharpe']:+.2f})"
            print(f"  {r['config']:<28}  prod_Sh={r['prod_sharpe']:+.2f}{base_change}",
                  flush=True)

        out_path = OUT_DIR / f"test_C_validated_{cfg}.csv"
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\n  saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
