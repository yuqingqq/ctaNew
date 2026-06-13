"""V3.1 cost sensitivity analysis.

V3.1's lift over K=3 (+0.25 Sharpe) is primarily through smooth turnover that
amortizes cost. This script tests how that lift evolves across a realistic cost
range:

  - 1.0 bps/leg : tight maker scenario (Hyperliquid maker after rebates)
  - 2.0 bps/leg : aggressive maker
  - 3.0 bps/leg : HL taker (no slippage)
  - 4.5 bps/leg : current production calibration
  - 6.0 bps/leg : HL taker with slippage
  - 9.0 bps/leg : Binance VIP-0 taker
  - 12.0 bps/leg : worst-case execution

Reports:
  Sharpe, PnL, maxDD for V3.1 and K=3 at each cost level
  Sharpe lift V3.1 - K=3 across costs
  Break-even cost (where lift → 0)
  Cost as % of gross
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
    "svar", REPO / "scripts/phase_ah_sleeve_variants.py")
svar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svar)

OUT = REPO / "outputs/vBTC_cost_sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON_ENTRY = 48
HOLD_BARS = 288
N_SLEEVES = 6
CYCLES_PER_YEAR = (288 * 365) / HORIZON_ENTRY
OOS_FOLDS = list(range(1, 10))


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def aggregate_v31(records, fwd_rets_4h, cost_per_leg):
    """V3.1 equal-weight 6-sleeve aggregation with parameterizable cost."""
    cost_per_unit = 0.5 * cost_per_leg
    bar_freq = pd.Timedelta(minutes=5)
    sleeve_queue = deque(maxlen=N_SLEEVES)
    prev_weights = {}
    sleeve_weights = [1/6] * N_SLEEVES
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]
        fold = rec["fold"]
        long_b = list(rec["long_basket"])
        short_b = list(rec["short_basket"])
        if rec["traded"] and len(long_b) > 0 and len(short_b) > 0:
            sleeve_queue.append({"entry_time": t, "longs": long_b, "shorts": short_b})
        max_age = HOLD_BARS * bar_freq
        sleeve_queue = deque(
            [s for s in sleeve_queue if (t - s["entry_time"]) < max_age],
            maxlen=N_SLEEVES
        )
        active_list = sorted(list(sleeve_queue),
                                key=lambda s: s["entry_time"], reverse=True)
        target_weights = defaultdict(float)
        for i, s in enumerate(active_list):
            if i >= len(sleeve_weights): break
            w = sleeve_weights[i]
            n_long = len(s["longs"]); n_short = len(s["shorts"])
            if n_long == 0 or n_short == 0: continue
            for sym in s["longs"]:
                target_weights[sym] += w * (1.0 / n_long)
            for sym in s["shorts"]:
                target_weights[sym] -= w * (1.0 / n_short)
        gross_pnl_bps = 0.0
        if t in fwd_rets_4h.index:
            rets_at_t = fwd_rets_4h.loc[t]
            for sym, w in prev_weights.items():
                if sym in rets_at_t.index and not pd.isna(rets_at_t[sym]):
                    gross_pnl_bps += w * rets_at_t[sym] * 1e4
        all_syms = set(target_weights.keys()) | set(prev_weights.keys())
        total_abs_delta = sum(abs(target_weights.get(s, 0.0) -
                                    prev_weights.get(s, 0.0))
                                for s in all_syms)
        cost_bps = total_abs_delta * cost_per_unit
        net_pnl_bps = gross_pnl_bps - cost_bps
        rows.append({"time": t, "fold": fold,
                      "gross_pnl_bps": gross_pnl_bps,
                      "cost_bps": cost_bps,
                      "net_pnl_bps": net_pnl_bps,
                      "gross_exposure": sum(abs(w) for w in target_weights.values())})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def aggregate_k3(records, fwd_rets_4h, cost_per_leg):
    """K=3 single-shot baseline: enter K=3 basket per cycle, hold 4h, exit, re-enter.

    Equivalent to V3.1 with N_SLEEVES=1 (no overlap).
    """
    cost_per_unit = 0.5 * cost_per_leg
    bar_freq = pd.Timedelta(minutes=5)
    sleeve_queue = deque(maxlen=1)  # ← key difference: no overlap
    prev_weights = {}
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]
        fold = rec["fold"]
        long_b = list(rec["long_basket"])
        short_b = list(rec["short_basket"])
        if rec["traded"] and len(long_b) > 0 and len(short_b) > 0:
            sleeve_queue.append({"entry_time": t, "longs": long_b, "shorts": short_b})
        # 4h hold for K=3 single-shot
        max_age = HORIZON_ENTRY * bar_freq
        sleeve_queue = deque(
            [s for s in sleeve_queue if (t - s["entry_time"]) < max_age],
            maxlen=1
        )
        target_weights = defaultdict(float)
        for s in sleeve_queue:
            n_long = len(s["longs"]); n_short = len(s["shorts"])
            if n_long == 0 or n_short == 0: continue
            for sym in s["longs"]:
                target_weights[sym] += 1.0 * (1.0 / n_long)
            for sym in s["shorts"]:
                target_weights[sym] -= 1.0 * (1.0 / n_short)
        gross_pnl_bps = 0.0
        if t in fwd_rets_4h.index:
            rets_at_t = fwd_rets_4h.loc[t]
            for sym, w in prev_weights.items():
                if sym in rets_at_t.index and not pd.isna(rets_at_t[sym]):
                    gross_pnl_bps += w * rets_at_t[sym] * 1e4
        all_syms = set(target_weights.keys()) | set(prev_weights.keys())
        total_abs_delta = sum(abs(target_weights.get(s, 0.0) -
                                    prev_weights.get(s, 0.0))
                                for s in all_syms)
        cost_bps = total_abs_delta * cost_per_unit
        net_pnl_bps = gross_pnl_bps - cost_bps
        rows.append({"time": t, "fold": fold,
                      "gross_pnl_bps": gross_pnl_bps,
                      "cost_bps": cost_bps,
                      "net_pnl_bps": net_pnl_bps,
                      "gross_exposure": sum(abs(w) for w in target_weights.values())})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def summarize(df, label, cost_per_leg):
    net = df["net_pnl_bps"].to_numpy()
    gross = df["gross_pnl_bps"].sum()
    cost = df["cost_bps"].sum()
    cost_share = cost / max(abs(gross), 1) * 100
    sh = _sharpe(net)
    dd = _max_dd(net)
    npos = sum(1 for _, g in df.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)
    return {"label": label, "cost_per_leg": cost_per_leg, "sharpe": sh,
             "maxDD": dd, "pnl": net.sum(), "gross_pnl": gross, "cost": cost,
             "cost_share_pct": cost_share, "folds_pos": npos}


def main():
    print("=== V3.1 vs K=3 cost sensitivity sweep ===\n", flush=True)
    records = pd.read_parquet(svar.SLEEVES_PATH)
    records["time"] = pd.to_datetime(records["time"], utc=True)
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet",
                            columns=["symbol"])
    all_syms = sorted(apd["symbol"].unique())
    print(f"  loading close prices...", flush=True)
    t0 = time.time()
    close_wide = svar.load_close_wide(all_syms)
    fwd_rets_4h = (close_wide.shift(-HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)\n", flush=True)

    costs = [1.0, 2.0, 3.0, 4.5, 6.0, 9.0, 12.0]
    rows_v31 = []
    rows_k3 = []
    print(f"  cost  | V3.1 Sh  V3.1 PnL  V3.1 DD  cost%  | K=3 Sh  K=3 PnL  K=3 DD  cost%  |  Δ Sharpe", flush=True)
    print(f"  ------|---------|---------|--------|-------|---------|--------|--------|-------|----------",
          flush=True)
    for c in costs:
        t0 = time.time()
        df_v31 = aggregate_v31(records, fwd_rets_4h, c)
        df_k3 = aggregate_k3(records, fwd_rets_4h, c)
        v31 = summarize(df_v31, "V3.1", c)
        k3 = summarize(df_k3, "K=3", c)
        rows_v31.append(v31); rows_k3.append(k3)
        lift = v31["sharpe"] - k3["sharpe"]
        print(f"  {c:>4.1f}  | {v31['sharpe']:+7.2f}  {v31['pnl']:+7.0f}  {v31['maxDD']:+7.0f}  "
              f"{v31['cost_share_pct']:>5.1f}  | {k3['sharpe']:+6.2f}  {k3['pnl']:+6.0f}  "
              f"{k3['maxDD']:+7.0f}  {k3['cost_share_pct']:>5.1f}  |  {lift:+7.2f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    df_summary = pd.DataFrame(rows_v31 + rows_k3)
    df_summary.to_csv(OUT / "cost_sensitivity_summary.csv", index=False)

    # Break-even analysis
    print(f"\n  --- Break-even analysis ---", flush=True)
    lifts = [(v["cost_per_leg"], v["sharpe"] - k["sharpe"])
              for v, k in zip(rows_v31, rows_k3)]
    print(f"  cost (bps/leg) | V3.1 - K=3 Sharpe", flush=True)
    for c, lift in lifts:
        print(f"      {c:>4.1f}      |  {lift:+.2f}", flush=True)

    # Find approximate break-even by linear interpolation
    lifts_arr = np.array(lifts)
    if np.any(lifts_arr[:, 1] > 0) and np.any(lifts_arr[:, 1] < 0):
        # Find where lift crosses zero
        for i in range(len(lifts_arr) - 1):
            if (lifts_arr[i, 1] > 0) != (lifts_arr[i+1, 1] > 0):
                c1, l1 = lifts_arr[i]
                c2, l2 = lifts_arr[i+1]
                breakeven = c1 + (c2 - c1) * (-l1) / (l2 - l1)
                print(f"\n  Break-even cost ≈ {breakeven:.1f} bps/leg", flush=True)
                break
    else:
        if all(l[1] > 0 for l in lifts):
            print(f"\n  V3.1 beats K=3 at ALL tested cost levels — robust structural edge", flush=True)
        elif all(l[1] < 0 for l in lifts):
            print(f"\n  K=3 beats V3.1 at all costs — V3.1 lift is fragile", flush=True)

    # Implications
    print(f"\n=== Implications ===\n", flush=True)
    prod_v31 = rows_v31[3]  # 4.5 bps current production
    prod_k3 = rows_k3[3]
    print(f"  At current production cost (4.5 bps/leg):", flush=True)
    print(f"    V3.1 Sharpe = {prod_v31['sharpe']:+.2f}  (cost share {prod_v31['cost_share_pct']:.1f}% of gross)",
          flush=True)
    print(f"    K=3  Sharpe = {prod_k3['sharpe']:+.2f}  (cost share {prod_k3['cost_share_pct']:.1f}% of gross)",
          flush=True)
    print(f"    Lift V3.1 - K=3 = {prod_v31['sharpe'] - prod_k3['sharpe']:+.2f}\n",
          flush=True)
    print(f"  At HL taker (~3.0 bps/leg):", flush=True)
    hl_v31 = rows_v31[2]
    hl_k3 = rows_k3[2]
    print(f"    V3.1 Sharpe = {hl_v31['sharpe']:+.2f}", flush=True)
    print(f"    K=3  Sharpe = {hl_k3['sharpe']:+.2f}", flush=True)
    print(f"    Lift = {hl_v31['sharpe'] - hl_k3['sharpe']:+.2f}\n", flush=True)
    print(f"  At HL maker (~1.0 bps/leg):", flush=True)
    mk_v31 = rows_v31[0]
    mk_k3 = rows_k3[0]
    print(f"    V3.1 Sharpe = {mk_v31['sharpe']:+.2f}", flush=True)
    print(f"    K=3  Sharpe = {mk_k3['sharpe']:+.2f}", flush=True)
    print(f"    Lift = {mk_v31['sharpe'] - mk_k3['sharpe']:+.2f}", flush=True)


if __name__ == "__main__":
    main()
