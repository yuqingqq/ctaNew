"""Phase 1b: Symbol-side filter applied INSIDE the evaluator loop.

Addresses the high-priority finding from the SS sweep review:
  Post-processed filter does NOT alter PM state. Filtering needs to happen
  BEFORE PM persistence so that cur_long/cur_short, flat/active state,
  and churn-based cost all evolve from the filtered set.

Per-cycle order of operations:
  1. universe restriction (PIT rolling-IC, pre-built)
  2. compute top-K long candidates, bottom-K short candidates from universe
  3. dispersion + conv_gate (binary skip)
  4. **NEW: apply SS filter to top-K / bottom-K candidates**
     trailing metric uses ONLY past picks accumulated by THIS filtered simulation
     (not baseline picks — self-consistent)
  5. PM persistence on FILTERED candidates
  6. flat_real state machine on FILTERED basket
  7. compute spread, churn cost from FILTERED basket
  8. append picks to picks_history (for future trailing metrics)

Variants:
  baseline_inline      no filter (sanity: should reproduce baseline pipeline)
  ss_filter_90d_sharpe trailing 90d Sharpe < -0.5, min N picks → exclude
  ss_filter_90d_mean   trailing 90d mean_contrib < 0, min N picks → exclude
  ss_filter_180d_mean  same with 180d
  ss_worst_k3          exclude worst 3 (sym, side) by trailing 90d mean
  placebo_rand_K_seed{N}  random exclusion matched to ss_worst_k3's K per cycle

Placebo: 100 independent seeds, deterministic (numpy seeds), reported as
distribution. The real filter must beat at least the 95th percentile of placebo
on Sharpe to be considered meaningful.

PIT contract:
  - rolling_universe at time t built from data with exit_time <= boundary(t)
  - SS filter trailing metric at time t uses picks with exit_time <= t

Inputs:
  outputs/vBTC_audit_panel/audit_panel.parquet  (per-cycle per-symbol pred + return)
  (universe is rebuilt deterministically from the panel here)

Output:
  outputs/vBTC_ss_filter_inline/<variant>_per_cycle.csv
  outputs/vBTC_ss_filter_inline/results.csv
  outputs/vBTC_ss_filter_inline/placebo_distribution.csv
"""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
from collections import deque, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs import block_bootstrap_ci

AUDIT_DIR = REPO / "outputs/vBTC_audit_panel"
OUT_DIR = REPO / "outputs/vBTC_ss_filter_inline"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
PM_BAND = 1.0
K = 4
MIN_PICKS_FOR_FILTER = 30
OOS_FOLDS = list(range(1, 10))
N_PLACEBO_SEEDS = 100


def _sharpe(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def trailing_metric_from_history(history_arr, window_days, decision_t):
    """history_arr: list of (time, exit_time, contrib_bps).
    Returns dict with mean, sharpe, n.
    """
    if not history_arr:
        return {"mean": 0.0, "sharpe": 0.0, "n": 0}
    cutoff = decision_t - pd.Timedelta(days=window_days)
    vals = [c for (t, et, c) in history_arr
            if et <= decision_t and t >= cutoff]
    if not vals:
        return {"mean": 0.0, "sharpe": 0.0, "n": 0}
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(arr.mean()), "sharpe": _sharpe(arr), "n": len(arr)}


def filter_decision(history_arr, window_days, mode, decision_t):
    """Return True if (sym, side) PASSES (keep), False if filtered out."""
    m = trailing_metric_from_history(history_arr, window_days, decision_t)
    if m["n"] < MIN_PICKS_FOR_FILTER:
        return True  # not enough data — keep
    if mode == "sharpe":
        return m["sharpe"] >= -0.5
    if mode == "mean":
        return m["mean"] >= 0
    return True


def worst_k_blacklist(picks_history, window_days, decision_t, top_k):
    """Build blacklist of worst-K (sym, side) pairs by trailing window mean."""
    cutoff = decision_t - pd.Timedelta(days=window_days)
    grouped = defaultdict(list)
    for (sym, side), arr in picks_history.items():
        vals = [c for (t, et, c) in arr if et <= decision_t and t >= cutoff]
        if len(vals) >= MIN_PICKS_FOR_FILTER:
            grouped[(sym, side)] = np.mean(vals)
    if len(grouped) <= top_k:
        return set()
    sorted_pairs = sorted(grouped.items(), key=lambda x: x[1])  # ascending = worst first
    return {pair for pair, _ in sorted_pairs[:top_k]}


def evaluate_inline(audit, rolling_universe, variant, oos_folds=OOS_FOLDS):
    """Run flat_real evaluator with SS filter applied INSIDE the loop.

    PM, flat_real, and churn all evolve from FILTERED basket.
    """
    df = audit.sort_values(["time", "symbol"]).copy()
    times = sorted(df["time"].unique())
    fold_lookup = df.groupby("time")["fold"].first().to_dict()

    history_dispersion = deque(maxlen=GATE_LOOKBACK)
    history_basket = []   # list of {long: set, short: set} for PM persistence
    cur_long, cur_short = set(), set()
    is_flat = False

    # Per-(sym, side) historic picks: list of (time, exit_time, contrib_bps_actual)
    picks_history = defaultdict(list)

    rows = []
    rng = np.random.RandomState(variant.get("placebo_seed", 0))

    for t in times:
        g = df[df["time"] == t]
        u = rolling_universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                          "skipped": 1, "spread_bps": 0.0, "cost_bps": 0.0,
                          "net_bps": 0.0, "n_long": 0, "n_short": 0,
                          "n_excluded_long": 0, "n_excluded_short": 0})
            continue

        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_lookup = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_lookup = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))

        idx_top = np.argpartition(-pred_arr, K - 1)[:K]
        idx_bot = np.argpartition(pred_arr, K - 1)[:K]
        dispersion = float(pred_arr[idx_top].mean() - pred_arr[idx_bot].mean())

        skip = False
        if len(history_dispersion) >= 30:
            thr = float(np.quantile(list(history_dispersion), GATE_PCTILE))
            if dispersion < thr:
                skip = True
        history_dispersion.append(dispersion)

        # PM history (top/bot bands) computed on the unfiltered candidates for PM gate logic
        band_k = max(K, int(round(PM_BAND * K)))
        bk = min(band_k, len(g_u))
        idx_top_band = np.argpartition(-pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        idx_bot_band = np.argpartition(pred_arr, bk - 1)[:bk] if bk < len(g_u) else np.arange(len(g_u))
        history_basket.append({"long": set(sym_arr[idx_top_band]),
                                 "short": set(sym_arr[idx_bot_band])})
        if len(history_basket) > PM_M:
            history_basket = history_basket[-PM_M:]

        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "skipped": 1, "spread_bps": 0.0,
                              "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0,
                              "n_excluded_long": 0, "n_excluded_short": 0})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "skipped": 1, "spread_bps": 0.0, "cost_bps": 0.0,
                              "net_bps": 0.0, "n_long": 0, "n_short": 0,
                              "n_excluded_long": 0, "n_excluded_short": 0})
            continue

        cand_long = set(sym_arr[idx_top])
        cand_short = set(sym_arr[idx_bot])
        cand_long_orig_size = len(cand_long)
        cand_short_orig_size = len(cand_short)

        # ===== SS FILTER (applied BEFORE PM persistence) =====
        kind = variant["kind"]
        if kind == "filter":
            mode = variant["mode"]
            window = variant["window_days"]
            cand_long = {s for s in cand_long
                         if filter_decision(picks_history.get((s, "long"), []),
                                            window, mode, t)}
            cand_short = {s for s in cand_short
                          if filter_decision(picks_history.get((s, "short"), []),
                                             window, mode, t)}
        elif kind == "worst_k":
            window = variant["window_days"]
            black = worst_k_blacklist(picks_history, window, t, variant["k"])
            cand_long = {s for s in cand_long if (s, "long") not in black}
            cand_short = {s for s in cand_short if (s, "short") not in black}
        elif kind == "placebo":
            # Match the number of excluded pairs to ss_worst_k3 distribution.
            # Sample variant["k"] random (sym, side) pairs from those with enough
            # history; deterministic per-seed RNG.
            cutoff = t - pd.Timedelta(days=variant["window_days"])
            eligible = [(s, sd) for (s, sd), arr in picks_history.items()
                        if sum(1 for (tt, et, _) in arr
                               if et <= t and tt >= cutoff) >= MIN_PICKS_FOR_FILTER]
            if len(eligible) > variant["k"]:
                idx = rng.choice(len(eligible), size=variant["k"], replace=False)
                black = {eligible[i] for i in idx}
                cand_long = {s for s in cand_long if (s, "long") not in black}
                cand_short = {s for s in cand_short if (s, "short") not in black}
        # else: "none" — no filter

        n_excl_l = cand_long_orig_size - len(cand_long)
        n_excl_s = cand_short_orig_size - len(cand_short)

        # ===== PM PERSISTENCE on filtered candidates =====
        if len(history_basket) >= PM_M:
            past_long = [h["long"] for h in history_basket[-PM_M:][:PM_M - 1]]
            past_short = [h["short"] for h in history_basket[-PM_M:][:PM_M - 1]]
            new_long = cur_long & cand_long
            new_short = cur_short & cand_short
            for s in cand_long - cur_long:
                if all(s in p for p in past_long):
                    new_long.add(s)
            for s in cand_short - cur_short:
                if all(s in p for p in past_short):
                    new_short.add(s)
            if len(new_long) > K:
                # rank by pred
                ranked = sorted(new_long, key=lambda s: -pred_arr[np.where(sym_arr == s)[0][0]])[:K]
                new_long = set(ranked)
            if len(new_short) > K:
                ranked = sorted(new_short, key=lambda s: pred_arr[np.where(sym_arr == s)[0][0]])[:K]
                new_short = set(ranked)
        else:
            new_long, new_short = cand_long, cand_short

        if not new_long or not new_short:
            # No survivors on at least one side — treat as flat (with close cost
            # if we were holding)
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "skipped": 1, "spread_bps": 0.0,
                              "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0,
                              "n_excluded_long": n_excl_l,
                              "n_excluded_short": n_excl_s})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "skipped": 0, "spread_bps": 0.0, "cost_bps": 0.0,
                              "net_bps": 0.0, "n_long": 0, "n_short": 0,
                              "n_excluded_long": n_excl_l,
                              "n_excluded_short": n_excl_s})
            continue

        # ===== Compute spread + cost on FINAL filtered basket =====
        long_rets = [ret_lookup[s] for s in new_long]
        short_rets = [ret_lookup[s] for s in new_short]
        long_mean = float(np.mean(long_rets))
        short_mean = float(np.mean(short_rets))
        spread = (long_mean - short_mean) * 1e4

        if is_flat:
            cost = 2 * COST_PER_LEG
            is_flat = False
        else:
            churn_long = (len(new_long.symmetric_difference(cur_long)) /
                          max(len(new_long | cur_long), 1))
            churn_short = (len(new_short.symmetric_difference(cur_short)) /
                           max(len(new_short | cur_short), 1))
            cost = (churn_long + churn_short) * COST_PER_LEG
        net = spread - cost

        # ===== Append to picks_history (PIT: exit_time recorded for future filtering) =====
        for s in new_long:
            contrib = ret_lookup[s] * 1e4 / len(new_long)
            picks_history[(s, "long")].append((t, exit_lookup[s], contrib))
        for s in new_short:
            contrib = -ret_lookup[s] * 1e4 / len(new_short)
            picks_history[(s, "short")].append((t, exit_lookup[s], contrib))

        rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                      "skipped": 0, "spread_bps": spread, "cost_bps": cost,
                      "net_bps": net,
                      "n_long": len(new_long), "n_short": len(new_short),
                      "n_excluded_long": n_excl_l, "n_excluded_short": n_excl_s})
        cur_long, cur_short = new_long, new_short

    return pd.DataFrame(rows)


def summarize(df_v, label):
    net = df_v["net_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    per_fold = {}
    for fid in OOS_FOLDS:
        fdat = df_v[df_v["fold"] == fid]["net_bps"].to_numpy()
        if len(fdat) >= 3:
            per_fold[fid] = _sharpe(fdat)
    return {
        "variant": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
        "max_dd": _max_dd(net), "total_pnl": net.sum(), "mean_bps": net.mean(),
        "n_cycles": len(df_v),
        "n_active": int((df_v["n_long"] > 0).sum()),
        "avg_n_excl_l": float(df_v["n_excluded_long"].mean()),
        "avg_n_excl_s": float(df_v["n_excluded_short"].mean()),
        **{f"sh_f{f}": v for f, v in per_fold.items()},
    }


def main():
    print(f"=== Phase 1b: Inline SS filter sweep ===\n", flush=True)
    audit = pd.read_parquet(AUDIT_DIR / "audit_panel.parquet")
    audit["time"] = pd.to_datetime(audit["time"])
    audit["exit_time"] = pd.to_datetime(audit["exit_time"])
    print(f"  audit panel: {len(audit):,} rows, {audit['symbol'].nunique()} symbols, "
          f"{audit['time'].nunique()} cycles", flush=True)

    # Build rolling_universe dict {time → set of universe symbols at that cycle}
    rolling_universe = (audit[audit["in_universe"] == 1]
                       .groupby("time")["symbol"].apply(set).to_dict())
    print(f"  rolling_universe cycles: {len(rolling_universe)}", flush=True)

    # Variants
    variants = [
        {"label": "baseline_inline", "kind": "none"},
        {"label": "ss_filter_90d_sharpe", "kind": "filter", "mode": "sharpe",
            "window_days": 90},
        {"label": "ss_filter_90d_mean", "kind": "filter", "mode": "mean",
            "window_days": 90},
        {"label": "ss_filter_180d_mean", "kind": "filter", "mode": "mean",
            "window_days": 180},
        {"label": "ss_worst_k3", "kind": "worst_k", "k": 3, "window_days": 90},
    ]

    print(f"\n=== Real variants ===\n", flush=True)
    print(f"  {'variant':<26}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  {'totPnL':>7}  "
          f"{'n_excl_l':>8}  {'n_excl_s':>8}", flush=True)
    results = []
    for v in variants:
        t0 = time.time()
        df_v = evaluate_inline(audit, rolling_universe, v)
        elapsed = time.time() - t0
        res = summarize(df_v, v["label"])
        results.append(res)
        df_v.to_csv(OUT_DIR / f"{v['label']}_per_cycle.csv", index=False)
        print(f"  {res['variant']:<26}  {res['sharpe']:>+7.2f}  "
              f"[{res['ci_lo']:>+5.2f},{res['ci_hi']:>+5.2f}]  "
              f"{res['max_dd']:>+7.0f}  {res['total_pnl']:>+7.0f}  "
              f"{res['avg_n_excl_l']:>8.2f}  {res['avg_n_excl_s']:>8.2f}  "
              f"({elapsed:.0f}s)", flush=True)

    print(f"\n=== Placebo distribution ({N_PLACEBO_SEEDS} seeds) ===\n", flush=True)
    placebo_results = []
    t0 = time.time()
    for seed in range(N_PLACEBO_SEEDS):
        v = {"label": f"placebo_seed_{seed}", "kind": "placebo", "k": 3,
              "window_days": 90, "placebo_seed": seed}
        df_v = evaluate_inline(audit, rolling_universe, v)
        res = summarize(df_v, v["label"])
        placebo_results.append(res)
        if (seed + 1) % 20 == 0:
            print(f"  ... placebo {seed + 1}/{N_PLACEBO_SEEDS} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
    placebo_df = pd.DataFrame(placebo_results)
    placebo_df.to_csv(OUT_DIR / "placebo_distribution.csv", index=False)
    print(f"\n  Placebo Sharpe distribution:", flush=True)
    p_sh = placebo_df["sharpe"].values
    print(f"    mean: {p_sh.mean():+.2f}", flush=True)
    print(f"    p5:   {np.percentile(p_sh, 5):+.2f}", flush=True)
    print(f"    p50:  {np.percentile(p_sh, 50):+.2f}", flush=True)
    print(f"    p95:  {np.percentile(p_sh, 95):+.2f}", flush=True)
    print(f"    min:  {p_sh.min():+.2f}", flush=True)
    print(f"    max:  {p_sh.max():+.2f}", flush=True)

    # Decision: does real filter beat 95th percentile of placebo?
    base_sharpe = next(r["sharpe"] for r in results if r["variant"] == "baseline_inline")
    p95 = np.percentile(p_sh, 95)
    print(f"\n=== Falsification: real filter vs placebo p95 ===", flush=True)
    print(f"  baseline_inline Sharpe:   {base_sharpe:+.2f}", flush=True)
    print(f"  placebo p95 Sharpe:        {p95:+.2f}", flush=True)
    for r in results:
        if r["variant"] == "baseline_inline": continue
        rank = (p_sh < r["sharpe"]).mean() * 100
        beats_p95 = r["sharpe"] > p95
        d_sh = r["sharpe"] - base_sharpe
        print(f"  {r['variant']:<26}  Sharpe={r['sharpe']:+.2f}  ΔSh={d_sh:+.2f}  "
              f"placebo_rank={rank:>5.1f}%  beats_p95={'✓' if beats_p95 else '✗'}",
              flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    print(f"  {'variant':<26}  " + " ".join(f"{'f' + str(f):>6}" for f in OOS_FOLDS),
          flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in OOS_FOLDS)
        print(f"  {r['variant']:<26}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
