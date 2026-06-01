"""Phase 1: Symbol-side filter & sizing sweep on the audit panel.

Tests 7 variants + placebo, with strict PIT discipline.

Variants:
  SS_FILTER_90D_MEAN     exclude (sym,side) if trailing 90d mean_contrib < 0, min N picks
  SS_FILTER_180D_MEAN    same with 180d
  SS_FILTER_90D_SHARPE   exclude if trailing Sharpe < -0.5, min N picks
  SS_FILTER_90D_TAIL     exclude if trailing p10 contrib < -75 bps, min N picks
  SS_SIZE_90D_SHARPE     scale weight by tiered trailing Sharpe (0.25/0.5/1.0)
  SS_SIZE_180D_SHARPE    same with 180d
  SS_WORST_K3            exclude worst 3 (sym,side) by trailing 90d contrib at each cycle
  PLACEBO_RAND_K3        exclude 3 random (sym,side) pairs (sanity check)

Inputs:
  outputs/vBTC_audit_panel/audit_panel.parquet
  outputs/vBTC_audit_panel/cycle_meta.parquet

Output:
  outputs/vBTC_ss_filter_sweep/results.csv
  outputs/vBTC_ss_filter_sweep/per_cycle_<variant>.csv

PIT contract:
  At decision time t, trailing metric for (sym, side) only uses past PICKED rows
  with exit_time <= t (the realized contribution is known by t).
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

AUDIT_DIR = REPO / "outputs/vBTC_audit_panel"
OUT_DIR = REPO / "outputs/vBTC_ss_filter_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5
OOS_FOLDS = list(range(1, 10))
MIN_PICKS_FOR_FILTER = 30  # need at least N past picks before acting on metric


def _sharpe(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def build_picked_history(audit):
    """Long-format table of actual picks with realized contribution and exit_time.

    Columns: time, exit_time, symbol, side, contrib_bps
    """
    long_picks = audit[audit["picked_long"] == 1][["time", "exit_time", "symbol",
                                                       "long_contrib_bps_actual"]].copy()
    long_picks = long_picks.rename(columns={"long_contrib_bps_actual": "contrib_bps"})
    long_picks["side"] = "long"

    short_picks = audit[audit["picked_short"] == 1][["time", "exit_time", "symbol",
                                                         "short_contrib_bps_actual"]].copy()
    short_picks = short_picks.rename(columns={"short_contrib_bps_actual": "contrib_bps"})
    short_picks["side"] = "short"

    picks = pd.concat([long_picks, short_picks], ignore_index=True)
    picks = picks.sort_values("exit_time").reset_index(drop=True)
    return picks


def trailing_metric(picks_history, sym, side, decision_t, window_days):
    """Trailing per-(sym, side) metrics using only past picks with exit_time <= decision_t.

    Returns dict with mean, sharpe, hit_rate, p10, n_count.
    """
    cutoff = decision_t - pd.Timedelta(days=window_days)
    mask = ((picks_history["symbol"] == sym) &
              (picks_history["side"] == side) &
              (picks_history["exit_time"] <= decision_t) &
              (picks_history["time"] >= cutoff))
    sub = picks_history.loc[mask, "contrib_bps"]
    if len(sub) == 0:
        return {"mean": 0.0, "sharpe": 0.0, "hit_rate": 0.5, "p10": 0.0, "n": 0}
    return {
        "mean": float(sub.mean()),
        "sharpe": _sharpe(sub.to_numpy()),
        "hit_rate": float((sub > 0).mean()),
        "p10": float(sub.quantile(0.10)),
        "n": int(len(sub)),
    }


def filter_predicate(metric, mode):
    """Return True if (sym, side) PASSES the filter (i.e., keep it)."""
    if metric["n"] < MIN_PICKS_FOR_FILTER:
        return True  # not enough data; keep by default
    if mode == "mean":
        return metric["mean"] >= 0
    if mode == "sharpe":
        return metric["sharpe"] >= -0.5
    if mode == "tail":
        return metric["p10"] >= -75.0
    if mode == "hit_rate":
        return metric["hit_rate"] >= 0.45
    return True


def size_multiplier(metric, mode):
    """Return sizing multiplier in [0.25, 1.25]."""
    if metric["n"] < MIN_PICKS_FOR_FILTER:
        return 1.0
    if mode == "tier_sharpe":
        sh = metric["sharpe"]
        if sh < -1.0: return 0.25
        if sh < 0.0: return 0.5
        return 1.0
    return 1.0


def simulate_variant(audit, cycle_meta, picks_history, variant):
    """Simulate one variant. Returns per-cycle DataFrame with net_bps, fold, etc."""
    rng = np.random.RandomState(42)
    # Group picks by cycle
    pick_long_by_t = audit[audit["picked_long"] == 1].groupby("time")["symbol"].apply(list).to_dict()
    pick_short_by_t = audit[audit["picked_short"] == 1].groupby("time")["symbol"].apply(list).to_dict()
    ret_lookup = audit.set_index(["time", "symbol"])["return_pct"].to_dict()
    fold_lookup = audit.groupby("time")["fold"].first().to_dict()

    cycles = sorted(cycle_meta["time"].unique())
    prev_long, prev_short = set(), set()
    is_flat = False
    rows = []

    for t in cycles:
        meta_row = cycle_meta[cycle_meta["time"] == t].iloc[0]
        if meta_row["skipped"] == 1:
            # Honor original skip behavior (flat_real)
            net = meta_row["spread_bps"] - meta_row["cost_bps"]
            rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                          "spread_bps": meta_row["spread_bps"],
                          "cost_bps": meta_row["cost_bps"],
                          "net_bps": net, "skipped": 1,
                          "n_long_filtered": 0, "n_short_filtered": 0})
            if meta_row["cost_bps"] > 0:
                is_flat = True
                prev_long, prev_short = set(), set()
            continue

        raw_long = pick_long_by_t.get(t, [])
        raw_short = pick_short_by_t.get(t, [])
        if not raw_long or not raw_short:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                          "spread_bps": 0.0, "cost_bps": 0.0, "net_bps": 0.0,
                          "skipped": 0, "n_long_filtered": 0, "n_short_filtered": 0})
            continue

        # Apply filter or sizing per (sym, side)
        long_kept = []   # list of (sym, weight_mult)
        short_kept = []
        if variant["kind"] == "filter":
            mode = variant["mode"]
            window = variant["window_days"]
            for s in raw_long:
                m = trailing_metric(picks_history, s, "long", t, window)
                if filter_predicate(m, mode):
                    long_kept.append((s, 1.0))
            for s in raw_short:
                m = trailing_metric(picks_history, s, "short", t, window)
                if filter_predicate(m, mode):
                    short_kept.append((s, 1.0))
        elif variant["kind"] == "size":
            mode = variant["mode"]
            window = variant["window_days"]
            for s in raw_long:
                m = trailing_metric(picks_history, s, "long", t, window)
                w = size_multiplier(m, mode)
                if w > 0:
                    long_kept.append((s, w))
            for s in raw_short:
                m = trailing_metric(picks_history, s, "short", t, window)
                w = size_multiplier(m, mode)
                if w > 0:
                    short_kept.append((s, w))
        elif variant["kind"] == "worst_k":
            # At time t, find the K worst (sym, side) by trailing window contrib across ALL eligible
            window = variant["window_days"]
            top_k = variant["k"]
            # Compute trailing mean for each (sym, side) ever picked
            cutoff = t - pd.Timedelta(days=window)
            past = picks_history[(picks_history["exit_time"] <= t) &
                                   (picks_history["time"] >= cutoff)]
            grp = past.groupby(["symbol", "side"])["contrib_bps"].agg(["mean", "count"]).reset_index()
            grp = grp[grp["count"] >= MIN_PICKS_FOR_FILTER].sort_values("mean")
            blacklist = set(zip(grp["symbol"].head(top_k).tolist(),
                                  grp["side"].head(top_k).tolist()))
            for s in raw_long:
                if (s, "long") not in blacklist:
                    long_kept.append((s, 1.0))
            for s in raw_short:
                if (s, "short") not in blacklist:
                    short_kept.append((s, 1.0))
        elif variant["kind"] == "placebo":
            # Random exclusion of K (sym, side) pairs (PIT-deterministic seed by cycle)
            top_k = variant["k"]
            # Build same candidate set as worst_k
            cutoff = t - pd.Timedelta(days=variant["window_days"])
            past = picks_history[(picks_history["exit_time"] <= t) &
                                   (picks_history["time"] >= cutoff)]
            grp = past.groupby(["symbol", "side"])["contrib_bps"].agg(["count"]).reset_index()
            grp = grp[grp["count"] >= MIN_PICKS_FOR_FILTER]
            pairs = list(zip(grp["symbol"].tolist(), grp["side"].tolist()))
            if len(pairs) > top_k:
                local_rng = np.random.RandomState(hash(str(t)) & 0xFFFFFFFF)
                idx = local_rng.choice(len(pairs), size=top_k, replace=False)
                blacklist = {pairs[i] for i in idx}
            else:
                blacklist = set()
            for s in raw_long:
                if (s, "long") not in blacklist:
                    long_kept.append((s, 1.0))
            for s in raw_short:
                if (s, "short") not in blacklist:
                    short_kept.append((s, 1.0))
        else:
            for s in raw_long:
                long_kept.append((s, 1.0))
            for s in raw_short:
                short_kept.append((s, 1.0))

        if not long_kept or not short_kept:
            # No survivors on at least one side — treat as flat
            if not is_flat and (prev_long or prev_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "spread_bps": 0.0, "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG, "skipped": 1,
                              "n_long_filtered": 0, "n_short_filtered": 0})
                is_flat = True
                prev_long, prev_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "spread_bps": 0.0, "cost_bps": 0.0, "net_bps": 0.0,
                              "skipped": 0,
                              "n_long_filtered": 0, "n_short_filtered": 0})
            continue

        # Compute spread = weighted mean long return - weighted mean short return
        long_w_sum = sum(w for _, w in long_kept)
        short_w_sum = sum(w for _, w in short_kept)
        long_ret_w = sum(ret_lookup[(t, s)] * w for s, w in long_kept) / long_w_sum
        short_ret_w = sum(ret_lookup[(t, s)] * w for s, w in short_kept) / short_w_sum
        spread = (long_ret_w - short_ret_w) * 1e4

        # Cost based on churn of filtered set
        cur_long_syms = set(s for s, _ in long_kept)
        cur_short_syms = set(s for s, _ in short_kept)
        if is_flat:
            cost = 2 * COST_PER_LEG
            is_flat = False
        else:
            churn_l = (len(cur_long_syms.symmetric_difference(prev_long)) /
                         max(len(cur_long_syms | prev_long), 1))
            churn_s = (len(cur_short_syms.symmetric_difference(prev_short)) /
                         max(len(cur_short_syms | prev_short), 1))
            cost = (churn_l + churn_s) * COST_PER_LEG
        net = spread - cost
        rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                      "spread_bps": spread, "cost_bps": cost, "net_bps": net,
                      "skipped": 0,
                      "n_long_filtered": len(cur_long_syms),
                      "n_short_filtered": len(cur_short_syms)})
        prev_long, prev_short = cur_long_syms, cur_short_syms

    return pd.DataFrame(rows)


def evaluate_variant(df_v, label):
    net = df_v["net_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    max_dd = _max_dd(net)
    per_fold = {}
    for fid in OOS_FOLDS:
        fdat = df_v[df_v["fold"] == fid]["net_bps"].to_numpy()
        if len(fdat) >= 3:
            per_fold[fid] = _sharpe(fdat)
    n_active = int((df_v["n_long_filtered"] > 0).sum())
    return {
        "variant": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
        "max_dd": max_dd, "mean_bps": net.mean(), "total_pnl_bps": net.sum(),
        "n_cycles": len(df_v), "n_active": n_active,
        "avg_n_long": float(df_v["n_long_filtered"].mean()),
        "avg_n_short": float(df_v["n_short_filtered"].mean()),
        **{f"sh_f{f}": v for f, v in per_fold.items()},
    }


def main():
    print(f"=== Phase 1: Symbol-side filter sweep ===\n", flush=True)
    audit = pd.read_parquet(AUDIT_DIR / "audit_panel.parquet")
    cycle_meta = pd.read_parquet(AUDIT_DIR / "cycle_meta.parquet")
    audit["time"] = pd.to_datetime(audit["time"])
    audit["exit_time"] = pd.to_datetime(audit["exit_time"])
    cycle_meta["time"] = pd.to_datetime(cycle_meta["time"])
    print(f"  audit panel: {len(audit):,} rows", flush=True)
    print(f"  cycle meta: {len(cycle_meta):,} rows", flush=True)

    picks_history = build_picked_history(audit)
    print(f"  picked-history (long+short): {len(picks_history):,}", flush=True)

    # Variants
    variants = [
        {"label": "BASELINE_NO_FILTER",  "kind": "none"},
        {"label": "SS_FILTER_90D_MEAN",  "kind": "filter", "mode": "mean", "window_days": 90},
        {"label": "SS_FILTER_180D_MEAN", "kind": "filter", "mode": "mean", "window_days": 180},
        {"label": "SS_FILTER_90D_SHARPE","kind": "filter", "mode": "sharpe", "window_days": 90},
        {"label": "SS_FILTER_90D_TAIL",  "kind": "filter", "mode": "tail", "window_days": 90},
        {"label": "SS_SIZE_90D_SHARPE",  "kind": "size", "mode": "tier_sharpe", "window_days": 90},
        {"label": "SS_SIZE_180D_SHARPE", "kind": "size", "mode": "tier_sharpe", "window_days": 180},
        {"label": "SS_WORST_K3",         "kind": "worst_k", "k": 3, "window_days": 90},
        {"label": "PLACEBO_RAND_K3",     "kind": "placebo", "k": 3, "window_days": 90},
    ]

    print(f"\n=== Simulating {len(variants)} variants ===\n", flush=True)
    print(f"  {'variant':<24}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  "
          f"{'totPnL':>7}  {'mean':>6}  {'avgL':>5}  {'avgS':>5}", flush=True)
    results = []
    for v in variants:
        df_v = simulate_variant(audit, cycle_meta, picks_history, v)
        res = evaluate_variant(df_v, v["label"])
        results.append(res)
        df_v.to_csv(OUT_DIR / f"per_cycle_{v['label']}.csv", index=False)
        print(f"  {res['variant']:<24}  {res['sharpe']:>+7.2f}  "
              f"[{res['ci_lo']:>+5.2f},{res['ci_hi']:>+5.2f}]  {res['max_dd']:>+7.0f}  "
              f"{res['total_pnl_bps']:>+7.0f}  {res['mean_bps']:>+6.2f}  "
              f"{res['avg_n_long']:>5.2f}  {res['avg_n_short']:>5.2f}", flush=True)

    print(f"\n  Per-fold Sharpe:", flush=True)
    print(f"  {'variant':<24}  " + " ".join(f"{'f' + str(f):>6}" for f in OOS_FOLDS), flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in OOS_FOLDS)
        print(f"  {r['variant']:<24}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "results.csv", index=False)

    # Pass-condition evaluation
    print(f"\n=== Pass-condition assessment vs BASELINE ===", flush=True)
    base = next(r for r in results if r["variant"] == "BASELINE_NO_FILTER")
    base_per_fold = [base.get(f"sh_f{f}", 0) for f in OOS_FOLDS]
    for r in results:
        if r["variant"] == "BASELINE_NO_FILTER": continue
        d_sh = r["sharpe"] - base["sharpe"]
        per_fold = [r.get(f"sh_f{f}", 0) for f in OOS_FOLDS]
        n_better = sum(1 for a, b in zip(per_fold, base_per_fold) if a > b)
        dd_ok = r["max_dd"] >= base["max_dd"]  # less negative is better
        ci_pos = r["ci_lo"] > 0
        pass_count = sum([d_sh >= 0.3, n_better >= 6, dd_ok, ci_pos])
        print(f"  {r['variant']:<24}  ΔSh={d_sh:+.2f}  folds_better={n_better}/9  "
              f"DD_ok={'✓' if dd_ok else '✗'}  CI_pos={'✓' if ci_pos else '✗'}  "
              f"score={pass_count}/4", flush=True)

    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
