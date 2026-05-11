"""Test F: basket-level per-cycle stop-loss (PnL floor).

If a cycle's PnL drops below -X bps, cap it at -X. This simulates a hard
stop-loss exit at the portfolio level. The lost upside on cap-hit cycles
is the cost.

Mechanism is naive but tells us:
  (a) what fraction of total drawdown comes from a small set of extreme cycles
  (b) whether truncating those cycles improves Sharpe or just reshuffles tail risk

Uses existing cycle PnL data (no retraining). Quick.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ml.research.alpha_v4_xs import block_bootstrap_ci

CYCLE_PATH = REPO / "outputs/vBTC_dd_analysis/cycle_pnl.csv"
OUT_DIR = REPO / "outputs/vBTC_test_F_stop_loss"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


PROD_FOLDS_DATES = {
    5: (pd.Timestamp("2025-11-24", tz="UTC"), pd.Timestamp("2025-12-26", tz="UTC")),
    6: (pd.Timestamp("2025-12-26", tz="UTC"), pd.Timestamp("2026-01-27", tz="UTC")),
    7: (pd.Timestamp("2026-01-27", tz="UTC"), pd.Timestamp("2026-02-28", tz="UTC")),
    8: (pd.Timestamp("2026-02-28", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
    9: (pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")),
}


def main():
    df = pd.read_csv(CYCLE_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    net = df["net_bps"].to_numpy()

    df["fold"] = [next((f for f, (s, e) in PROD_FOLDS_DATES.items()
                          if s <= pd.Timestamp(t) < e), None)
                   for t in df["time"]]

    print(f"=== Cycle distribution ===", flush=True)
    print(f"  N = {len(net)}, mean = {net.mean():+.2f}, std = {net.std():.1f}", flush=True)
    print(f"  total = {net.sum():+.0f} bps, baseline Sharpe = {_sharpe(net):+.2f}", flush=True)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  Percentiles:")
    for p in pcts:
        print(f"    {p:>3}%: {np.percentile(net, p):+7.0f} bps")

    # How much of total adverse PnL comes from extreme cycles?
    sorted_net = np.sort(net)
    n_neg = (net < 0).sum()
    print(f"\n  Negative cycles: {n_neg} ({n_neg/len(net)*100:.1f}%)", flush=True)
    print(f"  Bottom 1%: total {sorted_net[:max(1, len(net)//100)].sum():+.0f} bps "
          f"(n={max(1, len(net)//100)})", flush=True)
    print(f"  Bottom 5%: total {sorted_net[:max(1, len(net)//20)].sum():+.0f} bps "
          f"(n={max(1, len(net)//20)})", flush=True)
    print(f"  Bottom 10%: total {sorted_net[:max(1, len(net)//10)].sum():+.0f} bps "
          f"(n={max(1, len(net)//10)})", flush=True)

    print(f"\n=== Test F: per-cycle PnL floor ===", flush=True)
    print(f"  {'config':<25}  {'Sharpe':>7}  {'std':>6}  {'max_DD':>7}  {'mean':>6}  {'caps_hit':>9}", flush=True)
    results = []
    base_sh = _sharpe(net)
    base_dd = _max_dd(net)

    floors = [None, -100, -200, -300, -400, -500, -700, -1000, -1500]
    for f in floors:
        if f is None:
            scaled = net.copy()
            label = "baseline"
            caps_hit = 0
        else:
            scaled = np.maximum(net, f)
            label = f"floor_{f}"
            caps_hit = (net < f).sum()

        sh, lo, hi = block_bootstrap_ci(scaled, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd(scaled)
        per_fold = {}
        df_temp = df.copy(); df_temp["scaled"] = scaled
        for fid in sorted(PROD_FOLDS_DATES.keys()):
            n_f = df_temp[df_temp["fold"] == fid]["scaled"].to_numpy()
            if len(n_f) >= 3:
                per_fold[fid] = _sharpe(n_f)

        results.append({"config": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "std_bps": scaled.std(), "max_dd": max_dd,
                          "mean_net": scaled.mean(), "caps_hit": caps_hit,
                          **{f"sh_f{f}": v for f, v in per_fold.items()}})
        print(f"  {label:<25}  {sh:>+7.2f}  {scaled.std():>6.1f}  {max_dd:>+7.0f}  "
              f"{scaled.mean():>+6.2f}  {caps_hit:>9}", flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    print(f"  {'config':<25}  " + " ".join(f"{'fold' + str(f):>8}" for f in [5, 6, 7, 8, 9]),
          flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in [5, 6, 7, 8, 9])
        print(f"  {r['config']:<25}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "test_F_results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
