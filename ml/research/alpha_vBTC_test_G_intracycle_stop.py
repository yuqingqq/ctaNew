"""Test G: intra-cycle stop-loss using 5-min data.

For each cycle:
  - Load 5-min closes for picked L/S symbols over the 4h cycle window
  - Compute basket spread = (mean long ret) - (mean short ret) at each 5-min bar
  - If spread crosses below stop threshold, exit at that bar
  - Apply exit cost (one-way leg = COST_PER_LEG bps)

This is the realistic version of Test F (per-cycle floor). Whereas Test F
magically truncates end-of-cycle PnL, Test G actually checks whether cycles
hit the threshold mid-flight and exits there.

Walks fold-by-fold to avoid lookahead in the comparison.
"""
from __future__ import annotations
import sys, ast, time, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from ml.research.alpha_v4_xs import block_bootstrap_ci

CYCLE_LOG = REPO / "outputs/vBTC_dd_root_cause/cycle_logs.csv"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_test_G_intracycle"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5  # one-way leg cost (bps); applied on entry+exit
EXTRA_EXIT_COST = 4.5  # additional one-way cost when stop fires (premature exit, both sides)

PROD_FOLDS_DATES = {
    5: (pd.Timestamp("2025-11-24", tz="UTC"), pd.Timestamp("2025-12-26", tz="UTC")),
    6: (pd.Timestamp("2025-12-26", tz="UTC"), pd.Timestamp("2026-01-27", tz="UTC")),
    7: (pd.Timestamp("2026-01-27", tz="UTC"), pd.Timestamp("2026-02-28", tz="UTC")),
    8: (pd.Timestamp("2026-02-28", tz="UTC"), pd.Timestamp("2026-04-01", tz="UTC")),
    9: (pd.Timestamp("2026-04-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")),
}


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def load_5min_panel(symbols, start_date, end_date):
    """Load all 5-min closes for symbols across [start_date, end_date].
    Returns dict {symbol → DataFrame with index=open_time, columns=[close]}.
    """
    panel = {}
    days = pd.date_range(start_date, end_date, freq="D")
    for sym in symbols:
        sym_dir = KLINES_DIR / sym / "5m"
        if not sym_dir.exists():
            continue
        frames = []
        for d in days:
            f = sym_dir / f"{d.strftime('%Y-%m-%d')}.parquet"
            if f.exists():
                frames.append(pd.read_parquet(f, columns=["open_time", "close"]))
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True).sort_values("open_time").drop_duplicates("open_time")
        df = df.set_index("open_time")
        panel[sym] = df
    return panel


def parse_list(s):
    if pd.isna(s) or s == "" or s == "[]":
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []


def simulate_intracycle_stop(cycle_log: pd.DataFrame,
                              klines_panel: dict,
                              stop_bps: float | None,
                              cycle_minutes: int = 240,
                              extra_exit_cost: float = EXTRA_EXIT_COST) -> tuple[np.ndarray, dict]:
    """For each cycle, compute net PnL applying intra-cycle stop.

    stop_bps = None → baseline (no stop, returns end-of-cycle PnL from log).
    Otherwise: trigger when running spread (after entry costs) drops below stop_bps.

    Note: cycle_log already has cost_bps embedded in net_bps. We rebuild PnL
    from spread_bps (gross), apply intra-cycle path simulation, subtract entry+exit costs.
    """
    nets = np.zeros(len(cycle_log))
    stops_hit = 0
    stop_cycles = []
    for i, row in cycle_log.reset_index(drop=True).iterrows():
        # Skipped cycles: keep as-is from log (carry-over PnL or zero).
        if row.get("skipped", 0) == 1:
            nets[i] = row["net_bps"]
            continue
        if stop_bps is None:
            nets[i] = row["net_bps"]
            continue
        longs = parse_list(row["longs"])
        shorts = parse_list(row["shorts"])
        if not longs or not shorts:
            nets[i] = row["net_bps"]
            continue
        t0 = pd.Timestamp(row["time"])
        if t0.tz is None:
            t0 = t0.tz_localize("UTC")
        t_end = t0 + pd.Timedelta(minutes=cycle_minutes)

        # Build intra-cycle return paths for L/S sides.
        bar_times = pd.date_range(t0, t_end, freq="5min", inclusive="both", tz="UTC")
        long_rets = np.zeros(len(bar_times))
        short_rets = np.zeros(len(bar_times))
        n_long_ok = 0
        n_short_ok = 0
        for sym in longs:
            df = klines_panel.get(sym)
            if df is None or t0 not in df.index:
                continue
            p0 = df.loc[t0, "close"]
            for k, bt in enumerate(bar_times):
                if bt in df.index:
                    long_rets[k] += (df.loc[bt, "close"] / p0 - 1.0)
                else:
                    long_rets[k] += long_rets[k - 1] if k > 0 else 0.0
            n_long_ok += 1
        for sym in shorts:
            df = klines_panel.get(sym)
            if df is None or t0 not in df.index:
                continue
            p0 = df.loc[t0, "close"]
            for k, bt in enumerate(bar_times):
                if bt in df.index:
                    short_rets[k] += (df.loc[bt, "close"] / p0 - 1.0)
                else:
                    short_rets[k] += short_rets[k - 1] if k > 0 else 0.0
            n_short_ok += 1
        if n_long_ok == 0 or n_short_ok == 0:
            nets[i] = row["net_bps"]
            continue
        long_avg = long_rets / n_long_ok
        short_avg = short_rets / n_short_ok
        spread_bps = (long_avg - short_avg) * 1e4  # at each 5-min bar
        entry_cost = float(row.get("cost_bps", 9.0))  # from cycle log

        # Simulate stop: net = spread - entry_cost; if at any bar net < stop_bps, exit.
        running_net = spread_bps - entry_cost
        below = np.where(running_net < stop_bps)[0]
        if len(below) > 0:
            k_stop = below[0]
            # Exit at that bar's spread, pay extra exit cost.
            stop_pnl = running_net[k_stop] - extra_exit_cost
            nets[i] = stop_pnl
            stops_hit += 1
            stop_cycles.append({"time": t0, "stop_bar_min": int(k_stop * 5),
                                "running_net_at_stop": running_net[k_stop],
                                "would_have_been": running_net[-1]})
        else:
            # Cycle ran to completion: use end-of-cycle running_net.
            nets[i] = running_net[-1]
    return nets, {"stops_hit": stops_hit, "stop_cycles": stop_cycles}


def main():
    print(f"Loading cycle log...", flush=True)
    df = pd.read_csv(CYCLE_LOG)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    print(f"  {len(df)} cycles, baseline Sharpe = {_sharpe(df['net_bps'].to_numpy()):+.2f}",
          flush=True)

    # Get full universe of symbols ever picked.
    all_syms = set()
    for col in ["longs", "shorts"]:
        for s in df[col].dropna():
            all_syms.update(parse_list(s))
    all_syms = sorted(all_syms)
    print(f"  Universe: {len(all_syms)} unique symbols", flush=True)
    print(f"    {all_syms}", flush=True)

    # Determine date range from cycle log.
    start_d = df["time"].min().normalize()
    end_d = (df["time"].max() + pd.Timedelta(hours=4)).normalize() + pd.Timedelta(days=1)
    print(f"\nLoading 5-min klines from {start_d.date()} to {end_d.date()}...", flush=True)
    t0 = time.time()
    panel = load_5min_panel(all_syms, start_d, end_d)
    print(f"  Loaded {len(panel)} symbols in {time.time()-t0:.0f}s", flush=True)
    missing = [s for s in all_syms if s not in panel]
    if missing:
        print(f"  Missing: {missing}", flush=True)

    df["fold"] = [next((f for f, (s, e) in PROD_FOLDS_DATES.items()
                          if s <= t < e), None) for t in df["time"]]

    print(f"\n=== Test G: intra-cycle stop-loss ===", flush=True)
    print(f"  {'config':<25}  {'Sharpe':>7}  {'std':>6}  {'max_DD':>7}  {'mean':>6}  {'stops':>6}",
          flush=True)
    results = []
    stops_data = {}
    for stop_bps in [None, -100, -200, -300, -400, -500, -700, -1000]:
        t_run = time.time()
        nets, info = simulate_intracycle_stop(df, panel, stop_bps)
        elapsed = time.time() - t_run
        sh, lo, hi = block_bootstrap_ci(nets, statistic=_sharpe, block_size=7, n_boot=2000)
        max_dd = _max_dd(nets)
        per_fold = {}
        df_temp = df.copy(); df_temp["scaled"] = nets
        for fid in sorted(PROD_FOLDS_DATES.keys()):
            n_f = df_temp[df_temp["fold"] == fid]["scaled"].to_numpy()
            if len(n_f) >= 3:
                per_fold[fid] = _sharpe(n_f)
        label = "baseline" if stop_bps is None else f"stop_{stop_bps}"
        results.append({"config": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "std_bps": nets.std(), "max_dd": max_dd,
                          "mean_net": nets.mean(),
                          "stops_hit": info["stops_hit"],
                          **{f"sh_f{f}": v for f, v in per_fold.items()}})
        stops_data[label] = info["stop_cycles"]
        print(f"  {label:<25}  {sh:>+7.2f}  {nets.std():>6.1f}  {max_dd:>+7.0f}  "
              f"{nets.mean():>+6.2f}  {info['stops_hit']:>6}  ({elapsed:.0f}s)",
              flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    print(f"  {'config':<25}  " + " ".join(f"{'fold' + str(f):>8}" for f in [5, 6, 7, 8, 9]),
          flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in [5, 6, 7, 8, 9])
        print(f"  {r['config']:<25}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "test_G_results.csv", index=False)
    for label, sc in stops_data.items():
        if sc:
            pd.DataFrame(sc).to_csv(OUT_DIR / f"stops_{label}.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)

    # Compare Test F (theoretical) vs Test G (realistic) cap rates.
    print(f"\n=== F vs G comparison (caps hit) ===", flush=True)
    print(f"  {'threshold':>10}  {'F caps':>8}  {'G stops':>8}  {'F (theoretical) Sharpe':>22}  {'G (realistic) Sharpe':>22}",
          flush=True)


if __name__ == "__main__":
    main()
