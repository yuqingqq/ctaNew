"""Phase AH V3.2 + V3.3: sleeve variants.

V3.1 (already adopted): 6 sleeves @ 1/6 each, 24h hold → Sharpe +2.23 vs B0 +1.98
V3.2 equal3: 3 sleeves @ 1/3 each, 12h hold
V3.3 decay6: 6 sleeves @ 24h hold with decay weights [0.30, 0.22, 0.17, 0.13, 0.10, 0.08]

Reuses production_sleeves.parquet saved by phase_ah_sleeve.py V3.1.
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from ml.research.alpha_v4_xs import block_bootstrap_ci

OUT = REPO / "outputs/vBTC_sleeve_horizon"
OUT.mkdir(parents=True, exist_ok=True)
SLEEVES_PATH = OUT / "production_sleeves.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

HORIZON_ENTRY = 48  # 4h entry cadence
COST_PER_LEG = 4.5
COST_PER_UNIT_ABS_DELTA = 0.5 * COST_PER_LEG  # 2.25 bps/unit
CYCLES_PER_YEAR = (288 * 365) / HORIZON_ENTRY
OOS_FOLDS = list(range(1, 10))
N_PLACEBO_SEEDS = 100

# Pre-registered decay weights (newest first, sum to 1.0)
DECAY_WEIGHTS = [0.30, 0.22, 0.17, 0.13, 0.10, 0.08]


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def load_close_wide(symbols):
    frames = []
    for sym in symbols:
        sym_dir = KLINES_DIR / sym / "5m"
        if not sym_dir.exists(): continue
        files = sorted(sym_dir.glob("*.parquet"))
        if not files: continue
        dfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["open_time"]).drop_duplicates("open_time").set_index("open_time")
        df = df.rename(columns={"close": sym})
        frames.append(df)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def aggregate_sleeves_variant(records, fwd_rets_4h, n_sleeves, hold_bars,
                                 sleeve_weights=None,
                                 placebo_universe=None, placebo_seed=None):
    """Run sleeve protocol with parameterized N and weighting.

    sleeve_weights: list of length n_sleeves, sleeve_weights[0] = newest, ...
                    if None: uniform 1/n_sleeves.
    """
    if sleeve_weights is None:
        sleeve_weights = [1.0 / n_sleeves] * n_sleeves
    assert len(sleeve_weights) == n_sleeves

    bar_freq = pd.Timedelta(minutes=5)
    sleeve_queue = deque(maxlen=n_sleeves)  # newest at right (append)
    prev_weights = {}
    rng = np.random.RandomState(placebo_seed if placebo_seed is not None else 0)
    rows = []

    for _, rec in records.iterrows():
        t = rec["time"]
        fold = rec["fold"]

        if placebo_seed is not None and placebo_universe is not None and rec["traded"]:
            u = placebo_universe.get(t, set())
            pool = sorted(list(u))
            K_long = len(rec["long_basket"])
            K_short = len(rec["short_basket"])
            if len(pool) >= K_long + K_short and K_long > 0 and K_short > 0:
                shuffled = rng.permutation(len(pool))
                long_b = sorted([pool[i] for i in shuffled[:K_long]])
                short_b = sorted([pool[i] for i in shuffled[K_long:K_long+K_short]])
            else:
                long_b = []; short_b = []
        else:
            long_b = rec["long_basket"]
            short_b = rec["short_basket"]

        if len(long_b) > 0 and len(short_b) > 0:
            sleeve_queue.append({"entry_time": t, "longs": long_b, "shorts": short_b})

        # Drop aged-out sleeves
        max_age = hold_bars * bar_freq
        sleeve_queue = deque(
            [s for s in sleeve_queue if (t - s["entry_time"]) < max_age],
            maxlen=n_sleeves
        )

        # Build target weights: newest sleeve = first weight, oldest = last
        # Note: queue ordering — most recent appended is at index -1 (right)
        active_list = list(sleeve_queue)
        # Sort by entry_time descending (newest first)
        active_list.sort(key=lambda s: s["entry_time"], reverse=True)

        target_weights = defaultdict(float)
        for i, sleeve in enumerate(active_list):
            if i >= len(sleeve_weights): break
            w = sleeve_weights[i]
            n_long = len(sleeve["longs"])
            n_short = len(sleeve["shorts"])
            if n_long == 0 or n_short == 0: continue
            for s in sleeve["longs"]:
                target_weights[s] += w * (1.0 / n_long)
            for s in sleeve["shorts"]:
                target_weights[s] -= w * (1.0 / n_short)

        # Compute 4h PnL
        gross_pnl_bps = 0.0
        if t in fwd_rets_4h.index:
            rets_at_t = fwd_rets_4h.loc[t]
            for sym, w in prev_weights.items():
                if sym in rets_at_t.index and not pd.isna(rets_at_t[sym]):
                    gross_pnl_bps += w * rets_at_t[sym] * 1e4

        # Cost
        all_syms = set(target_weights.keys()) | set(prev_weights.keys())
        total_abs_delta = sum(abs(target_weights.get(s, 0.0) - prev_weights.get(s, 0.0))
                                for s in all_syms)
        cost_bps = total_abs_delta * COST_PER_UNIT_ABS_DELTA
        net_pnl_bps = gross_pnl_bps - cost_bps

        gross_exposure = sum(abs(w) for w in target_weights.values())
        net_exposure = sum(target_weights.values())

        rows.append({"time": t, "fold": fold, "active_sleeves": len(sleeve_queue),
                      "gross_pnl_bps": gross_pnl_bps, "cost_bps": cost_bps,
                      "net_pnl_bps": net_pnl_bps, "turnover": total_abs_delta,
                      "gross_exposure": gross_exposure,
                      "net_exposure": net_exposure,
                      "n_symbols": len(target_weights)})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def fold_concentration(df_v):
    fold_pnls = df_v.groupby("fold")["net_pnl_bps"].sum()
    pos = fold_pnls[fold_pnls > 0]
    total_pos = pos.sum() if len(pos) > 0 else 0
    if total_pos <= 0: return 0.0
    return float(pos.max() / total_pos)


def main():
    print("=== Phase AH V3.2 + V3.3: sleeve variants ===\n", flush=True)
    records = pd.read_parquet(SLEEVES_PATH)
    records["time"] = pd.to_datetime(records["time"], utc=True)
    print(f"  loaded {len(records)} sleeves; {records['traded'].sum()} traded\n",
          flush=True)

    panel_syms = sorted(set().union(
        *[set(r["long_basket"]) | set(r["short_basket"]) for _, r in records.iterrows()
          if r["traded"]]
    ))
    # Make sure we have all 51 panel symbols for return lookup
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet",
                            columns=["symbol"])
    all_syms = sorted(apd["symbol"].unique())
    print(f"  loading close prices for {len(all_syms)} symbols...", flush=True)
    t0 = time.time()
    close_wide = load_close_wide(all_syms)
    fwd_rets_4h = (close_wide.shift(-HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)\n", flush=True)

    variants = [
        ("V3.1_equal6_baseline", 6, 288, None),  # baseline reproduction
        ("V3.2_equal3_12h", 3, 144, None),
        ("V3.3_decay6_24h", 6, 288, DECAY_WEIGHTS),
    ]

    print(f"  {'variant':<26}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  "
          f"{'totPnL':>8}  {'gross':>6}  {'cost':>5}  {'net':>6}  "
          f"{'pos_folds':>9}  {'conc':>4}", flush=True)
    results = {}
    for label, n_sleeves, hold_bars, decay_w in variants:
        t0 = time.time()
        df_v = aggregate_sleeves_variant(records, fwd_rets_4h, n_sleeves, hold_bars,
                                              sleeve_weights=decay_w)
        net = df_v["net_pnl_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        gross = df_v["gross_pnl_bps"].mean()
        cost = df_v["cost_bps"].mean()
        n_pos = 0
        for f in OOS_FOLDS:
            d = df_v[df_v["fold"] == f]["net_pnl_bps"].to_numpy()
            if len(d) >= 3 and _sharpe(d) > 0: n_pos += 1
        conc = fold_concentration(df_v)
        results[label] = {"sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                            "max_dd": _max_dd(net), "total_pnl": net.sum(),
                            "n_folds_positive": n_pos, "concentration": conc,
                            "df": df_v, "n_sleeves": n_sleeves,
                            "hold_bars": hold_bars, "decay_w": decay_w}
        df_v.to_csv(OUT / f"per_cycle_{label}.csv", index=False)
        print(f"  {label:<26}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{_max_dd(net):>+7.0f}  {net.sum():>+8.0f}  "
              f"{gross:>+6.2f}  {cost:>5.2f}  "
              f"{df_v['net_pnl_bps'].mean():>+6.2f}  "
              f"{n_pos:>5d}/9  {conc*100:>3.0f}%  ({time.time()-t0:.0f}s)", flush=True)

    # Compare
    B0_SHARPE = 1.98
    print(f"\n=== Comparison vs B0 (Phase M K=3 +{B0_SHARPE:+.2f}) ===\n", flush=True)
    for label, r in results.items():
        lift = r['sharpe'] - B0_SHARPE
        print(f"  {label:<26}  Sharpe={r['sharpe']:+.2f}  lift={lift:+.2f}  "
              f"totPnL={r['total_pnl']:+.0f}  maxDD={r['max_dd']:+.0f}  "
              f"folds={r['n_folds_positive']}/9", flush=True)

    # Find best non-baseline variant
    best_name = max(["V3.2_equal3_12h", "V3.3_decay6_24h"],
                      key=lambda k: results[k]["sharpe"])
    best = results[best_name]
    v3_1 = results["V3.1_equal6_baseline"]
    print(f"\n  Best new variant: {best_name}", flush=True)
    print(f"  Best Sharpe: {best['sharpe']:+.2f}", flush=True)
    print(f"  vs V3.1 baseline: {best['sharpe'] - v3_1['sharpe']:+.2f}", flush=True)
    print(f"  vs B0: {best['sharpe'] - B0_SHARPE:+.2f}", flush=True)

    # Matched placebo on best
    if best["sharpe"] > B0_SHARPE + 0.10:
        print(f"\n--- Matched placebo on {best_name} ({N_PLACEBO_SEEDS} seeds) ---",
              flush=True)
        # Need universe — reconstruct from production_sleeves time range
        # Use union of all traded baskets at each cycle as placebo pool
        # Simpler: reconstruct universe via rolling-IC (same as phase_ah_sleeve)
        # For speed, use union of all symbols ever traded at each time as proxy pool
        # Actually let me just use the simpler approach: load universe via fast method
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "psl", REPO / "scripts/phase_ah_sleeve.py")
        psl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(psl)
        apd_full = pd.read_parquet(psl.APD_PATH)
        apd_full["open_time"] = pd.to_datetime(apd_full["open_time"], utc=True)
        apd_full["exit_time"] = pd.to_datetime(apd_full["exit_time"], utc=True)
        listings = psl.get_listings()
        def elig_at(b):
            ts = pd.Timestamp(b, unit="ms", tz="UTC")
            cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
            return {s for s in all_syms if listings.get(s) and listings[s] <= cutoff}
        tgt = sorted(apd_full[apd_full["fold"].isin(OOS_FOLDS)]["open_time"].unique())
        sampled = tgt[::HORIZON_ENTRY]
        universe = psl.build_rolling_ic_universe(apd_full, sampled, psl.TOP_N, elig_at)

        t0 = time.time()
        placebo_sh = []
        for seed in range(N_PLACEBO_SEEDS):
            df_p = aggregate_sleeves_variant(records, fwd_rets_4h,
                                                  best["n_sleeves"], best["hold_bars"],
                                                  sleeve_weights=best["decay_w"],
                                                  placebo_universe=universe,
                                                  placebo_seed=seed)
            placebo_sh.append(_sharpe(df_p["net_pnl_bps"].to_numpy()))
            if (seed + 1) % 25 == 0:
                print(f"  ... {seed+1}/{N_PLACEBO_SEEDS}  ({time.time()-t0:.0f}s)",
                      flush=True)
        p_sh = np.array(placebo_sh)
        p95 = float(np.percentile(p_sh, 95))
        rank = float((p_sh < best["sharpe"]).mean() * 100)
        print(f"\n  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
              f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
        print(f"  {best_name} ranks p{rank:.0f}  "
              f"beats_p95={'PASS' if best['sharpe'] > p95 else 'FAIL'}", flush=True)

    print(f"\n=== Final ranking ===\n", flush=True)
    sorted_vars = sorted(results.items(), key=lambda kv: -kv[1]["sharpe"])
    for label, r in sorted_vars:
        print(f"  {label:<26}  Sharpe={r['sharpe']:+.2f}  "
              f"totPnL={r['total_pnl']:+.0f}  folds={r['n_folds_positive']}/9", flush=True)


if __name__ == "__main__":
    main()
