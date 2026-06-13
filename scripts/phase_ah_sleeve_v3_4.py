"""Phase AH V3.4: early SL/TP exit on V3.3 decay-weighted base.

Each sleeve tracks entry close prices. At every 4h tick, compute unrealized
sleeve PnL = (long_basket avg return - short_basket avg return) × 1e4 bps
relative to entry. If SL or TP breached, retire sleeve (weight → 0 next cycle).

Variants (each on V3.3 decay weights [0.30, 0.22, 0.17, 0.13, 0.10, 0.08]):
  V3.3_baseline (no SL/TP — for reference)
  V3.4a_SL_only_40bps  (SL=-40, no TP)
  V3.4b_TP_only_40bps  (TP=+40, no SL)
  V3.4c_SL40_TP80      (SL=-40, TP=+80, asymmetric)
  V3.4d_SL60_TP60      (SL=-60, TP=+60, symmetric)

Pass: Sharpe ≥ V3.3 + 0.10 OR maxDD improves >15% with Sharpe neutral, AND
beats matched placebo p95 (random sleeves with same SL/TP rule).
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
SLEEVES_PATH = OUT / "production_sleeves.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

HORIZON_ENTRY = 48
HOLD_BARS = 288
N_SLEEVES = 6
COST_PER_LEG = 4.5
COST_PER_UNIT_ABS_DELTA = 0.5 * COST_PER_LEG  # 2.25 bps/unit
CYCLES_PER_YEAR = (288 * 365) / HORIZON_ENTRY
OOS_FOLDS = list(range(1, 10))
N_PLACEBO_SEEDS = 100

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
    return pd.concat(frames, axis=1).sort_index()


def get_close_at(close_wide, t, syms):
    """Get close prices for syms at time t (nearest bar). Returns dict[sym] → price (or NaN)."""
    if t > close_wide.index.max():
        return {s: np.nan for s in syms}
    try:
        idx = close_wide.index.get_indexer([t], method="nearest")[0]
        row = close_wide.iloc[idx]
    except Exception:
        return {s: np.nan for s in syms}
    return {s: (row[s] if s in row.index else np.nan) for s in syms}


def aggregate_sleeves_v3_4(records, close_wide, fwd_rets_4h, sleeve_weights,
                              sl_bps=None, tp_bps=None,
                              placebo_universe=None, placebo_seed=None):
    """Sleeve protocol with optional SL/TP early exit.

    sl_bps: if set, retire sleeve when unrealized PnL < sl_bps (negative number)
    tp_bps: if set, retire sleeve when unrealized PnL > tp_bps (positive number)
    Tracks entry-close per sleeve to compute unrealized PnL at each 4h tick.
    """
    bar_freq = pd.Timedelta(minutes=5)
    # Each sleeve: {entry_time, longs, shorts, entry_close: dict[sym]→price}
    sleeve_queue = deque(maxlen=N_SLEEVES)
    prev_weights = {}
    rng = np.random.RandomState(placebo_seed if placebo_seed is not None else 0)
    rows = []

    for _, rec in records.iterrows():
        t = rec["time"]
        fold = rec["fold"]

        if placebo_seed is not None and placebo_universe is not None and rec["traded"]:
            u = placebo_universe.get(t, set())
            pool = sorted(list(u))
            K_l = len(rec["long_basket"]); K_s = len(rec["short_basket"])
            if len(pool) >= K_l + K_s and K_l > 0 and K_s > 0:
                shuffled = rng.permutation(len(pool))
                long_b = sorted([pool[i] for i in shuffled[:K_l]])
                short_b = sorted([pool[i] for i in shuffled[K_l:K_l+K_s]])
            else:
                long_b = []; short_b = []
        else:
            long_b = list(rec["long_basket"])
            short_b = list(rec["short_basket"])

        if len(long_b) > 0 and len(short_b) > 0:
            # New sleeve: capture entry close prices
            all_syms_this_sleeve = long_b + short_b
            entry_close = get_close_at(close_wide, t, all_syms_this_sleeve)
            sleeve_queue.append({
                "entry_time": t, "longs": long_b, "shorts": short_b,
                "entry_close": entry_close,
            })

        # Drop aged-out sleeves (24h max hold)
        max_age = HOLD_BARS * bar_freq
        sleeve_queue = deque(
            [s for s in sleeve_queue if (t - s["entry_time"]) < max_age],
            maxlen=N_SLEEVES
        )

        # Apply SL/TP early exit
        if sl_bps is not None or tp_bps is not None:
            # Compute current close for all symbols in active sleeves
            all_active_syms = set()
            for s in sleeve_queue:
                all_active_syms |= set(s["longs"]) | set(s["shorts"])
            now_close = get_close_at(close_wide, t, list(all_active_syms))

            # Compute each sleeve's unrealized PnL
            retained_sleeves = []
            for s in sleeve_queue:
                long_rets = []
                short_rets = []
                ok = True
                for sym in s["longs"]:
                    p0 = s["entry_close"].get(sym)
                    p1 = now_close.get(sym)
                    if p0 is None or p1 is None or pd.isna(p0) or pd.isna(p1) or p0 <= 0:
                        ok = False; break
                    long_rets.append((p1 - p0) / p0)
                if not ok:
                    retained_sleeves.append(s)
                    continue
                for sym in s["shorts"]:
                    p0 = s["entry_close"].get(sym)
                    p1 = now_close.get(sym)
                    if p0 is None or p1 is None or pd.isna(p0) or pd.isna(p1) or p0 <= 0:
                        ok = False; break
                    short_rets.append((p1 - p0) / p0)
                if not ok:
                    retained_sleeves.append(s)
                    continue
                pnl_bps = (np.mean(long_rets) - np.mean(short_rets)) * 1e4
                retire = False
                if sl_bps is not None and pnl_bps < sl_bps:
                    retire = True
                if tp_bps is not None and pnl_bps > tp_bps:
                    retire = True
                if not retire:
                    retained_sleeves.append(s)
            sleeve_queue = deque(retained_sleeves, maxlen=N_SLEEVES)

        # Build target weights — sorted newest first, apply decay weights
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

        # 4h PnL
        gross_pnl_bps = 0.0
        if t in fwd_rets_4h.index:
            rets_at_t = fwd_rets_4h.loc[t]
            for sym, w in prev_weights.items():
                if sym in rets_at_t.index and not pd.isna(rets_at_t[sym]):
                    gross_pnl_bps += w * rets_at_t[sym] * 1e4

        all_syms = set(target_weights.keys()) | set(prev_weights.keys())
        total_abs_delta = sum(abs(target_weights.get(s, 0.0) - prev_weights.get(s, 0.0))
                                for s in all_syms)
        cost_bps = total_abs_delta * COST_PER_UNIT_ABS_DELTA
        net_pnl_bps = gross_pnl_bps - cost_bps

        rows.append({"time": t, "fold": fold,
                      "active_sleeves": len(sleeve_queue),
                      "gross_pnl_bps": gross_pnl_bps, "cost_bps": cost_bps,
                      "net_pnl_bps": net_pnl_bps,
                      "turnover": total_abs_delta,
                      "gross_exposure": sum(abs(w) for w in target_weights.values()),
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
    print("=== Phase AH V3.4: SL/TP early exit on V3.3 decay base ===\n", flush=True)
    records = pd.read_parquet(SLEEVES_PATH)
    records["time"] = pd.to_datetime(records["time"], utc=True)
    apd = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet",
                            columns=["symbol"])
    all_syms = sorted(apd["symbol"].unique())
    print(f"  loading close prices...", flush=True)
    t0 = time.time()
    close_wide = load_close_wide(all_syms)
    fwd_rets_4h = (close_wide.shift(-HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  done ({time.time()-t0:.0f}s)\n", flush=True)

    variants = [
        ("V3.3_baseline", None, None),
        ("V3.4a_SL40", -40.0, None),
        ("V3.4b_TP40", None, +40.0),
        ("V3.4c_SL40_TP80", -40.0, +80.0),
        ("V3.4d_SL60_TP60", -60.0, +60.0),
    ]

    print(f"  {'variant':<22}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  "
          f"{'totPnL':>8}  {'gross':>6}  {'cost':>5}  {'net':>6}  "
          f"{'pos_folds':>9}  {'conc':>4}", flush=True)
    results = {}
    for label, sl, tp in variants:
        t0 = time.time()
        df_v = aggregate_sleeves_v3_4(records, close_wide, fwd_rets_4h,
                                            DECAY_WEIGHTS, sl_bps=sl, tp_bps=tp)
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
                            "sl": sl, "tp": tp, "df": df_v}
        df_v.to_csv(OUT / f"per_cycle_{label}.csv", index=False)
        print(f"  {label:<22}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{_max_dd(net):>+7.0f}  {net.sum():>+8.0f}  "
              f"{gross:>+6.2f}  {cost:>5.2f}  "
              f"{df_v['net_pnl_bps'].mean():>+6.2f}  "
              f"{n_pos:>5d}/9  {conc*100:>3.0f}%  ({time.time()-t0:.0f}s)", flush=True)

    base = results["V3.3_baseline"]
    cand = {k: v for k, v in results.items() if k != "V3.3_baseline"}
    best_name = max(cand, key=lambda k: cand[k]["sharpe"])
    best = cand[best_name]
    print(f"\n  V3.3 baseline:    Sharpe={base['sharpe']:+.2f}  "
          f"maxDD={base['max_dd']:+.0f}", flush=True)
    print(f"  Best ({best_name}): Sharpe={best['sharpe']:+.2f}  "
          f"maxDD={best['max_dd']:+.0f}", flush=True)
    lift = best['sharpe'] - base['sharpe']
    dd_change = (base['max_dd'] - best['max_dd']) / abs(base['max_dd']) * 100
    print(f"  Lift over V3.3:   {lift:+.2f}", flush=True)
    print(f"  DD improvement:   {dd_change:+.1f}% "
          f"({'better' if dd_change > 0 else 'worse'})", flush=True)

    # Matched placebo on best non-baseline (if interesting)
    pass_sharpe = lift >= 0.10
    pass_dd_neutral = dd_change > 15 and abs(lift) < 0.1
    if pass_sharpe or pass_dd_neutral:
        print(f"\n--- Matched placebo on {best_name} ({N_PLACEBO_SEEDS} seeds) ---",
              flush=True)
        # Load universe for placebo
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
            df_p = aggregate_sleeves_v3_4(records, close_wide, fwd_rets_4h,
                                                DECAY_WEIGHTS, sl_bps=best["sl"],
                                                tp_bps=best["tp"],
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
    else:
        print(f"\n  → No variant clears Sharpe +0.10 or DD>15% improvement gate. Skip placebo.",
              flush=True)

    print(f"\n=== Final ranking ===\n", flush=True)
    sorted_vars = sorted(results.items(), key=lambda kv: -kv[1]["sharpe"])
    for label, r in sorted_vars:
        print(f"  {label:<22}  Sharpe={r['sharpe']:+.2f}  maxDD={r['max_dd']:+.0f}  "
              f"totPnL={r['total_pnl']:+.0f}  folds={r['n_folds_positive']}/9",
              flush=True)


if __name__ == "__main__":
    main()
