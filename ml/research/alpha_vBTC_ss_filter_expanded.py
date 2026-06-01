"""Phase 2a: SS filter with expanded universe + refill semantics.

Tests whether broadening the IC-ranked trade universe (N=15/25/35/all-eligible)
and refilling after the symbol-side filter (keep walking down the ranks until
K names pass per side) recovers active leg count and improves Sharpe.

Per-cycle pipeline (inline):
  1. PIT eligibility (listing_date + 60d ≤ t)
  2. Rolling-IC universe of size N (trailing 180d IC over PIT-eligible)
  3. Top-K long candidates by descending pred (within universe N)
  4. Bottom-K short candidates by ascending pred (within universe N)
  5. Apply SS filter (90d_mean, min 30 picks) to each candidate
  6. **REFILL**: if fewer than K pass, walk down the next-ranked candidates
     within the universe until K survive (or universe exhausted)
  7. PM persistence on the filtered+refilled set
  8. conv_gate / flat_real state machine
  9. Spread + churn cost
  10. Append picks to picks_history

Variants:
  N=15 (current) | N=25 | N=35 | N=all_eligible
  best N + 100-seed placebo (matches K=3 random exclusion on the refilled set)

Inputs:
  outputs/vBTC_audit_panel/audit_panel.parquet  (pred, return, exit_time, alpha_A
                                                    for ALL 51 symbols per cycle)

PIT contract:
  - Rolling-IC universe at t built from past predictions with exit_time ≤ boundary
  - SS filter trailing metric at t uses picks with exit_time ≤ t
  - All from self-consistent filtered simulation
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
OUT_DIR = REPO / "outputs/vBTC_ss_filter_expanded"
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
MIN_OBS_PER_SYM = 100
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
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


def build_rolling_ic_universe(audit, target_times, top_n,
                                  ic_window_days=IC_WINDOW_DAYS,
                                  update_days=IC_UPDATE_DAYS):
    """Build PIT rolling-IC universe with arbitrary top_n.

    top_n = None → return all PIT-eligible symbols at each cycle.
    """
    bar_ms = 5 * 60 * 1000
    window_ms = ic_window_days * 288 * bar_ms
    update_ms = update_days * 288 * bar_ms

    # audit['time'] and audit['exit_time'] need int conversion (force ms-precision int64)
    audit = audit.copy()
    def to_ms_int(s):
        ts = pd.to_datetime(s)
        if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
        return ts.astype("datetime64[ms]").astype("int64").to_numpy()
    audit["t_int"] = to_ms_int(audit["time"])
    audit["exit_t_int"] = to_ms_int(audit["exit_time"])

    if not target_times: return {}
    t0_ms = int(pd.Timestamp(target_times[0]).timestamp() * 1000)
    boundaries = []
    for t in target_times:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        n = (t_ms - t0_ms) // update_ms
        b = t0_ms + n * update_ms
        boundaries.append((t, b))

    unique_b = sorted(set(b for _, b in boundaries))
    boundary_to_universe = {}
    audit_clean = audit.dropna(subset=["alpha_A"])
    for b in unique_b:
        past = audit_clean[(audit_clean["t_int"] >= b - window_ms) &
                             (audit_clean["t_int"] < b) &
                             (audit_clean["exit_t_int"] <= b) &
                             (audit_clean["eligible_pit"] == 1)]
        if len(past) < 1000:
            boundary_to_universe[b] = set()
            continue
        ics = past.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank())
            if len(g) >= MIN_OBS_PER_SYM else np.nan
        )
        ics_sorted = ics.dropna().sort_values(ascending=False)
        if top_n is None:
            # all PIT-eligible with sufficient observations
            boundary_to_universe[b] = set(ics_sorted.index.tolist())
        else:
            boundary_to_universe[b] = set(ics_sorted.head(top_n).index.tolist())

    return {t: boundary_to_universe[b] for t, b in boundaries}


def filter_decision(history_arr, window_days, mode, decision_t):
    if not history_arr:
        return True
    cutoff = decision_t - pd.Timedelta(days=window_days)
    vals = [c for (t, et, c) in history_arr if et <= decision_t and t >= cutoff]
    if len(vals) < MIN_PICKS_FOR_FILTER:
        return True
    arr = np.asarray(vals, dtype=float)
    if mode == "mean":
        return arr.mean() >= 0
    if mode == "sharpe":
        return _sharpe(arr) >= -0.5
    return True


def select_with_refill(ranked_syms, side, k, picks_history, filter_window, filter_mode, t):
    """Walk down ranked_syms in order, keep names that pass filter, up to k."""
    kept = []
    for s in ranked_syms:
        if len(kept) >= k:
            break
        if filter_decision(picks_history.get((s, side), []), filter_window, filter_mode, t):
            kept.append(s)
    return kept


def select_with_placebo_refill(ranked_syms, side, k, picks_history, filter_window,
                                  blacklist, t):
    """Refill but exclude pairs in blacklist (used for placebo)."""
    kept = []
    for s in ranked_syms:
        if len(kept) >= k:
            break
        if (s, side) in blacklist:
            continue
        kept.append(s)
    return kept


def evaluate_expanded(audit, rolling_universe, variant, oos_folds=OOS_FOLDS):
    """Inline simulator with universe expansion + refill."""
    df = audit.sort_values(["time", "symbol"]).copy()
    times = sorted(df["time"].unique())
    fold_lookup = df.groupby("time")["fold"].first().to_dict()

    history_dispersion = deque(maxlen=GATE_LOOKBACK)
    history_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_history = defaultdict(list)
    rng = np.random.RandomState(variant.get("placebo_seed", 0))

    rows = []
    # Pre-index audit by time for speed
    audit_by_t = {t: g for t, g in df.groupby("time")}

    for t in times:
        g = audit_by_t.get(t)
        if g is None:
            continue
        u = rolling_universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                          "skipped": 1, "spread_bps": 0.0, "cost_bps": 0.0,
                          "net_bps": 0.0, "n_long": 0, "n_short": 0,
                          "n_universe": len(g_u)})
            continue

        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_lookup = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_lookup = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))

        # Conv gate uses top-K vs bottom-K within universe
        idx_top = np.argpartition(-pred_arr, K - 1)[:K]
        idx_bot = np.argpartition(pred_arr, K - 1)[:K]
        dispersion = float(pred_arr[idx_top].mean() - pred_arr[idx_bot].mean())
        skip = False
        if len(history_dispersion) >= 30:
            thr = float(np.quantile(list(history_dispersion), GATE_PCTILE))
            if dispersion < thr:
                skip = True
        history_dispersion.append(dispersion)

        # PM band tracking (still applied on top-K candidates, for hysteresis)
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
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u)})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "skipped": 1, "spread_bps": 0.0, "cost_bps": 0.0,
                              "net_bps": 0.0, "n_long": 0, "n_short": 0,
                              "n_universe": len(g_u)})
            continue

        # Build full long-ranked and short-ranked lists (within universe)
        order_desc = np.argsort(-pred_arr)
        order_asc = np.argsort(pred_arr)
        long_ranked = [sym_arr[i] for i in order_desc]
        short_ranked = [sym_arr[i] for i in order_asc]

        # SS filter with refill
        kind = variant["kind"]
        if kind == "filter_refill":
            window = variant["window_days"]
            mode = variant["mode"]
            cand_long = select_with_refill(long_ranked, "long", K, picks_history,
                                              window, mode, t)
            cand_short = select_with_refill(short_ranked, "short", K, picks_history,
                                                window, mode, t)
        elif kind == "no_filter":
            cand_long = long_ranked[:K]
            cand_short = short_ranked[:K]
        elif kind == "placebo":
            # Build blacklist (matched K random pairs from eligible-history pairs)
            cutoff = t - pd.Timedelta(days=variant["window_days"])
            eligible_pairs = [(s, sd) for (s, sd), arr in picks_history.items()
                              if sum(1 for (tt, et, _) in arr
                                     if et <= t and tt >= cutoff) >= MIN_PICKS_FOR_FILTER]
            if len(eligible_pairs) > variant["k"]:
                idx = rng.choice(len(eligible_pairs), size=variant["k"], replace=False)
                blacklist = {eligible_pairs[i] for i in idx}
            else:
                blacklist = set()
            cand_long = select_with_placebo_refill(long_ranked, "long", K, picks_history,
                                                       variant["window_days"], blacklist, t)
            cand_short = select_with_placebo_refill(short_ranked, "short", K,
                                                        picks_history,
                                                        variant["window_days"], blacklist, t)
        else:
            cand_long = long_ranked[:K]
            cand_short = short_ranked[:K]

        cand_long_set = set(cand_long)
        cand_short_set = set(cand_short)

        # PM persistence on filtered+refilled set
        if len(history_basket) >= PM_M:
            past_long = [h["long"] for h in history_basket[-PM_M:][:PM_M - 1]]
            past_short = [h["short"] for h in history_basket[-PM_M:][:PM_M - 1]]
            new_long = cur_long & cand_long_set
            new_short = cur_short & cand_short_set
            for s in cand_long_set - cur_long:
                if all(s in p for p in past_long):
                    new_long.add(s)
            for s in cand_short_set - cur_short:
                if all(s in p for p in past_short):
                    new_short.add(s)
            if len(new_long) > K:
                new_long = set(sorted(new_long,
                                        key=lambda s: -pred_arr[np.where(sym_arr == s)[0][0]])[:K])
            if len(new_short) > K:
                new_short = set(sorted(new_short,
                                         key=lambda s: pred_arr[np.where(sym_arr == s)[0][0]])[:K])
        else:
            new_long, new_short = cand_long_set, cand_short_set

        if not new_long or not new_short:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "skipped": 1, "spread_bps": 0.0,
                              "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u)})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "skipped": 0, "spread_bps": 0.0, "cost_bps": 0.0,
                              "net_bps": 0.0, "n_long": 0, "n_short": 0,
                              "n_universe": len(g_u)})
            continue

        long_rets = [ret_lookup[s] for s in new_long]
        short_rets = [ret_lookup[s] for s in new_short]
        long_mean = float(np.mean(long_rets))
        short_mean = float(np.mean(short_rets))
        spread = (long_mean - short_mean) * 1e4

        if is_flat:
            cost = 2 * COST_PER_LEG
            is_flat = False
        else:
            churn_l = (len(new_long.symmetric_difference(cur_long)) /
                       max(len(new_long | cur_long), 1))
            churn_s = (len(new_short.symmetric_difference(cur_short)) /
                       max(len(new_short | cur_short), 1))
            cost = (churn_l + churn_s) * COST_PER_LEG
        net = spread - cost

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
                      "n_universe": len(g_u)})
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
        "avg_n_long": float(df_v["n_long"].mean()),
        "avg_n_short": float(df_v["n_short"].mean()),
        "avg_n_universe": float(df_v["n_universe"].mean()),
        **{f"sh_f{f}": v for f, v in per_fold.items()},
    }


def main():
    print(f"=== Phase 2a: SS filter expanded-universe sweep ===\n", flush=True)
    audit = pd.read_parquet(AUDIT_DIR / "audit_panel.parquet")
    audit["time"] = pd.to_datetime(audit["time"])
    audit["exit_time"] = pd.to_datetime(audit["exit_time"])
    print(f"  audit panel: {len(audit):,} rows, {audit['symbol'].nunique()} symbols, "
          f"{audit['time'].nunique()} cycles", flush=True)

    target_times = sorted(audit["time"].unique())

    # Universe variants: top N from rolling-IC ranking (or all eligible)
    n_variants = [
        (15, "N=15 (current)"),
        (25, "N=25"),
        (35, "N=35"),
        (None, "N=all_eligible"),
    ]

    # SS filter spec (winner from inline run)
    SS_FILTER = {"kind": "filter_refill", "window_days": 90, "mode": "mean"}
    NO_FILTER = {"kind": "no_filter"}

    print(f"  Building rolling-IC universes for each N...", flush=True)
    universes = {}
    for n, label in n_variants:
        t0 = time.time()
        u = build_rolling_ic_universe(audit, target_times, n)
        universes[label] = u
        avg_size = np.mean([len(v) for v in u.values()])
        print(f"    {label}: avg universe size = {avg_size:.1f} ({time.time()-t0:.0f}s)",
              flush=True)

    # Run no-filter baseline + filter+refill for each universe size
    print(f"\n=== Variants ===\n", flush=True)
    print(f"  {'variant':<40}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  "
          f"{'totPnL':>7}  {'avgL':>5}  {'avgS':>5}  {'avgU':>5}  {'act':>5}",
          flush=True)
    results = []
    cycle_dfs = {}
    for n, label in n_variants:
        for filt_spec, filt_label in [(NO_FILTER, "no_filter"),
                                          (SS_FILTER, "ss_filter_90d_mean_refill")]:
            t0 = time.time()
            full_label = f"{label} | {filt_label}"
            df_v = evaluate_expanded(audit, universes[label], filt_spec)
            res = summarize(df_v, full_label)
            results.append(res)
            cycle_dfs[full_label] = df_v
            print(f"  {full_label:<40}  {res['sharpe']:>+7.2f}  "
                  f"[{res['ci_lo']:>+5.2f},{res['ci_hi']:>+5.2f}]  "
                  f"{res['max_dd']:>+7.0f}  {res['total_pnl']:>+7.0f}  "
                  f"{res['avg_n_long']:>5.2f}  {res['avg_n_short']:>5.2f}  "
                  f"{res['avg_n_universe']:>5.1f}  {res['n_active']:>5}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # Identify best filter variant for placebo
    filter_results = [r for r in results if "ss_filter_90d_mean_refill" in r["variant"]]
    best = max(filter_results, key=lambda r: r["sharpe"])
    print(f"\n  Best filter variant: {best['variant']}", flush=True)

    # Find the corresponding universe size
    best_label = best["variant"].split(" |")[0]
    best_univ = universes[best_label]

    # 100-seed placebo for the best universe size
    print(f"\n=== Placebo distribution for best universe ({N_PLACEBO_SEEDS} seeds) ===\n",
          flush=True)
    t0 = time.time()
    placebo_results = []
    for seed in range(N_PLACEBO_SEEDS):
        v = {"label": f"placebo_seed_{seed}", "kind": "placebo", "k": 3,
              "window_days": 90, "placebo_seed": seed}
        df_v = evaluate_expanded(audit, best_univ, v)
        res = summarize(df_v, v["label"])
        placebo_results.append(res)
        if (seed + 1) % 25 == 0:
            print(f"  ... placebo {seed + 1}/{N_PLACEBO_SEEDS} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    placebo_df = pd.DataFrame(placebo_results)
    placebo_df.to_csv(OUT_DIR / "placebo_distribution.csv", index=False)
    p_sh = placebo_df["sharpe"].values
    print(f"\n  Placebo Sharpe (best universe={best_label}):", flush=True)
    print(f"    mean: {p_sh.mean():+.2f}", flush=True)
    print(f"    p5:   {np.percentile(p_sh, 5):+.2f}", flush=True)
    print(f"    p50:  {np.percentile(p_sh, 50):+.2f}", flush=True)
    print(f"    p95:  {np.percentile(p_sh, 95):+.2f}", flush=True)
    print(f"    max:  {p_sh.max():+.2f}", flush=True)

    # Falsification check
    print(f"\n=== Falsification vs placebo p95 ===\n", flush=True)
    p95 = np.percentile(p_sh, 95)
    print(f"  placebo p95: {p95:+.2f}", flush=True)
    for r in filter_results:
        rank = (p_sh < r["sharpe"]).mean() * 100
        beats = r["sharpe"] > p95
        print(f"  {r['variant']:<48}  Sharpe={r['sharpe']:+.2f}  "
              f"placebo_rank={rank:>5.1f}%  beats_p95={'✓' if beats else '✗'}",
              flush=True)

    # Per-fold
    print(f"\n=== Per-fold Sharpe ===", flush=True)
    print(f"  {'variant':<48}  " + " ".join(f"{'f' + str(f):>6}" for f in OOS_FOLDS),
          flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in OOS_FOLDS)
        print(f"  {r['variant']:<48}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "results.csv", index=False)
    for label, df_v in cycle_dfs.items():
        safe_label = label.replace(" ", "_").replace("|", "")
        df_v.to_csv(OUT_DIR / f"per_cycle_{safe_label}.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
