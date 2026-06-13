"""Phase E5b: run Phase 2b v3 protocol on EXPANDED universe with $10M PIT volume gate.

Inputs:
  outputs/vBTC_audit_panel_expanded/all_predictions.parquet (folds 0-9, 111 syms)
  outputs/vBTC_features_expanded/volume_pit_table.parquet (per-day volume table)

PIT eligibility:
  listing_date (from klines) + 60d ≤ t
  AND trailing_30d_median_qvol (lookup) ≥ $10M

Variants:
  expanded_N=15_no_filter
  expanded_N=15_filter_refill_90d_mean
  expanded_N=25_filter_refill_90d_mean
  expanded_N=35_filter_refill_90d_mean
  expanded_N=all_filter_refill_90d_mean
  + 100-seed matched placebo on best variant

Compare to 51-panel baselines:
  51_N=15 no_filter (corrected final_sim) Sharpe ~+0.19
  51_N=15 filter_refill (Phase 2b v3) Sharpe +1.16, placebo rank 51%
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

AUDIT_DIR = REPO / "outputs/vBTC_audit_panel_expanded"
VOL_TABLE = REPO / "outputs/vBTC_features_expanded/volume_pit_table.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_expanded_protocol"
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
MIN_HISTORY_DAYS = 60
MIN_VOLUME_USD = 10_000_000  # $10M trailing 30d median


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def to_ms_int(s):
    ts = pd.to_datetime(s)
    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return ts.astype("datetime64[ms]").astype("int64").to_numpy()


def get_listing_dates_from_klines():
    listings = {}
    for sym_dir in KLINES_DIR.iterdir():
        if not sym_dir.is_dir(): continue
        m5 = sym_dir / "5m"
        if not m5.exists(): continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        try:
            listings[sym_dir.name] = pd.Timestamp(files[0].stem, tz="UTC")
        except Exception:
            continue
    return listings


def build_rolling_ic_universe_pit(all_pred_df, target_times, top_n,
                                       eligibility_at_t,
                                       ic_window_days=IC_WINDOW_DAYS,
                                       update_days=IC_UPDATE_DAYS):
    bar_ms = 5 * 60 * 1000
    window_ms = ic_window_days * 288 * bar_ms
    update_ms = update_days * 288 * bar_ms

    df = all_pred_df.copy()
    df["t_int"] = to_ms_int(df["open_time"])
    df["exit_t_int"] = to_ms_int(df["exit_time"])
    df_clean = df.dropna(subset=["alpha_A"])

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
    for b in unique_b:
        eligible = eligibility_at_t(b)
        past = df_clean[(df_clean["t_int"] >= b - window_ms) &
                          (df_clean["t_int"] < b) &
                          (df_clean["exit_t_int"] <= b) &
                          (df_clean["symbol"].isin(eligible))]
        if len(past) < 1000:
            boundary_to_universe[b] = set()
            continue
        ics = past.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank())
            if len(g) >= MIN_OBS_PER_SYM else np.nan
        )
        ics_sorted = ics.dropna().sort_values(ascending=False)
        if top_n is None:
            boundary_to_universe[b] = set(ics_sorted.index.tolist())
        else:
            boundary_to_universe[b] = set(ics_sorted.head(top_n).index.tolist())

    return {t: boundary_to_universe[b] for t, b in boundaries}


def filter_decision(history_arr, window_days, mode, decision_t):
    if not history_arr: return True
    cutoff = decision_t - pd.Timedelta(days=window_days)
    vals = [c for (t, et, c) in history_arr if et <= decision_t and t >= cutoff]
    if len(vals) < MIN_PICKS_FOR_FILTER: return True
    arr = np.asarray(vals, dtype=float)
    if mode == "mean": return arr.mean() >= 0
    if mode == "sharpe": return _sharpe(arr) >= -0.5
    return True


def select_with_refill(ranked_syms, side, k, picks_history, filter_window, filter_mode, t):
    kept = []
    n_excluded = 0
    for s in ranked_syms:
        if len(kept) >= k: break
        if filter_decision(picks_history.get((s, side), []), filter_window, filter_mode, t):
            kept.append(s)
        else:
            n_excluded += 1
    return kept, n_excluded


def select_with_matched_placebo(ranked_syms, side, k, n_to_exclude, picks_history,
                                  filter_window, rng, t):
    if n_to_exclude <= 0:
        return ranked_syms[:k]
    cutoff = t - pd.Timedelta(days=filter_window)
    filterable = []
    for i, s in enumerate(ranked_syms):
        history_arr = picks_history.get((s, side), [])
        vals = [c for (tt, et, c) in history_arr if et <= t and tt >= cutoff]
        if len(vals) >= MIN_PICKS_FOR_FILTER:
            filterable.append(i)
    if len(filterable) <= n_to_exclude:
        skip_positions = set(filterable)
    else:
        skip_positions = set(rng.choice(filterable, size=n_to_exclude, replace=False))
    kept = []
    for i, s in enumerate(ranked_syms):
        if len(kept) >= k: break
        if i in skip_positions: continue
        kept.append(s)
    return kept


def evaluate_inline(all_pred_df, rolling_universe, variant, real_exclusion_counts=None):
    df = all_pred_df.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    # Sample every HORIZON
    keep_times = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_times)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()

    history_dispersion = deque(maxlen=GATE_LOOKBACK)
    history_basket_filtered = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_history = defaultdict(list)
    rng = np.random.RandomState(variant.get("placebo_seed", 0))

    audit_by_t = {t: g for t, g in df.groupby("open_time")}
    rows = []
    real_excl_track = []

    for cycle_idx, t in enumerate(times):
        g = audit_by_t.get(t)
        if g is None:
            continue
        u = rolling_universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                          "skipped": 1, "spread_bps": 0.0, "cost_bps": 0.0,
                          "net_bps": 0.0, "n_long": 0, "n_short": 0,
                          "n_universe": len(g_u), "n_excl_long": 0, "n_excl_short": 0})
            real_excl_track.append((0, 0))
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
            if dispersion < thr: skip = True
        history_dispersion.append(dispersion)

        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "skipped": 1, "spread_bps": 0.0,
                              "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": 0, "n_excl_short": 0})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "skipped": 1, "spread_bps": 0.0, "cost_bps": 0.0,
                              "net_bps": 0.0, "n_long": 0, "n_short": 0,
                              "n_universe": len(g_u), "n_excl_long": 0, "n_excl_short": 0})
            real_excl_track.append((0, 0))
            continue

        order_desc = np.argsort(-pred_arr)
        order_asc = np.argsort(pred_arr)
        long_ranked = [sym_arr[i] for i in order_desc]
        short_ranked = [sym_arr[i] for i in order_asc]

        kind = variant["kind"]
        n_excl_l = 0
        n_excl_s = 0
        if kind == "no_filter":
            cand_long = long_ranked[:K]
            cand_short = short_ranked[:K]
        elif kind == "filter_refill":
            window = variant["window_days"]; mode = variant["mode"]
            cand_long, n_excl_l = select_with_refill(long_ranked, "long", K,
                                                       picks_history, window, mode, t)
            cand_short, n_excl_s = select_with_refill(short_ranked, "short", K,
                                                         picks_history, window, mode, t)
        elif kind == "matched_placebo":
            if real_exclusion_counts and cycle_idx < len(real_exclusion_counts):
                target_l, target_s = real_exclusion_counts[cycle_idx]
            else:
                target_l, target_s = 0, 0
            cand_long = select_with_matched_placebo(long_ranked, "long", K, target_l,
                                                       picks_history, variant["window_days"],
                                                       rng, t)
            cand_short = select_with_matched_placebo(short_ranked, "short", K, target_s,
                                                         picks_history, variant["window_days"],
                                                         rng, t)
            n_excl_l = target_l; n_excl_s = target_s
        else:
            cand_long = long_ranked[:K]
            cand_short = short_ranked[:K]

        real_excl_track.append((n_excl_l, n_excl_s))

        cand_long_set = set(cand_long)
        cand_short_set = set(cand_short)

        history_basket_filtered.append({"long": cand_long_set, "short": cand_short_set})
        if len(history_basket_filtered) > PM_M:
            history_basket_filtered = history_basket_filtered[-PM_M:]

        if len(history_basket_filtered) >= PM_M:
            past_long = [h["long"] for h in history_basket_filtered[-PM_M:][:PM_M - 1]]
            past_short = [h["short"] for h in history_basket_filtered[-PM_M:][:PM_M - 1]]
            new_long = cur_long & cand_long_set
            new_short = cur_short & cand_short_set
            for s in cand_long_set - cur_long:
                if all(s in p for p in past_long): new_long.add(s)
            for s in cand_short_set - cur_short:
                if all(s in p for p in past_short): new_short.add(s)
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
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": n_excl_l, "n_excl_short": n_excl_s})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "skipped": 0, "spread_bps": 0.0, "cost_bps": 0.0,
                              "net_bps": 0.0, "n_long": 0, "n_short": 0,
                              "n_universe": len(g_u),
                              "n_excl_long": n_excl_l, "n_excl_short": n_excl_s})
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
                      "net_bps": net, "n_long": len(new_long),
                      "n_short": len(new_short), "n_universe": len(g_u),
                      "n_excl_long": n_excl_l, "n_excl_short": n_excl_s})
        cur_long, cur_short = new_long, new_short

    return pd.DataFrame(rows), real_excl_track


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
        "n_active": int((df_v["n_long"] > 0).sum()),
        "avg_n_long": float(df_v["n_long"].mean()),
        "avg_n_short": float(df_v["n_short"].mean()),
        "avg_n_universe": float(df_v["n_universe"].mean()),
        "avg_excl_l": float(df_v["n_excl_long"].mean()),
        "avg_excl_s": float(df_v["n_excl_short"].mean()),
        **{f"sh_f{f}": v for f, v in per_fold.items()},
    }


def main():
    print(f"=== Phase E5b: Expanded universe protocol with $10M volume gate ===\n", flush=True)

    # Load predictions
    apd = pd.read_parquet(AUDIT_DIR / "all_predictions.parquet")
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    print(f"  predictions: {len(apd):,} rows, {apd['symbol'].nunique()} symbols, "
          f"folds {sorted(apd['fold'].unique())}", flush=True)

    # Load volume table
    vol = pd.read_parquet(VOL_TABLE)
    vol["date"] = pd.to_datetime(vol["date"]).dt.date
    # Build lookup: (sym, date) -> trailing_30d_median_qvol
    vol_dict = vol.set_index(["symbol", "date"])["trailing_30d_median_qvol"].to_dict()
    print(f"  volume table: {len(vol):,} rows", flush=True)

    # Listing dates
    listings = get_listing_dates_from_klines()
    print(f"  listing dates: {len(listings)} symbols", flush=True)

    panel_syms = set(apd["symbol"].unique())

    def eligibility_at(b_ms):
        if isinstance(b_ms, (int, np.integer)):
            ts = pd.Timestamp(b_ms, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(b_ms)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        # PIT volume: use trailing-30d-median as of date BEFORE b (so it's actually known)
        vol_date = (ts - pd.Timedelta(days=1)).date()
        eligible = set()
        for s in panel_syms:
            # Listing date check
            if not listings.get(s) or listings[s] > cutoff:
                continue
            # Volume check
            v = vol_dict.get((s, vol_date), 0)
            if v < MIN_VOLUME_USD:
                continue
            eligible.add(s)
        return eligible

    target_times = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    target_times_sampled = target_times[::HORIZON]
    print(f"  Rebalance times: {len(target_times_sampled)}", flush=True)

    # Quick check: eligibility at first OOS time
    first_t = target_times_sampled[0]
    elig0 = eligibility_at(first_t)
    print(f"  Eligible symbols at first OOS ({first_t}): {len(elig0)}", flush=True)

    # Build universes
    n_specs = [(15, "N=15"), (25, "N=25"), (35, "N=35"), (None, "N=all")]
    universes = {}
    for n, lbl in n_specs:
        t0 = time.time()
        u = build_rolling_ic_universe_pit(apd, target_times_sampled, n, eligibility_at)
        universes[lbl] = u
        avg_sz = float(np.mean([len(v) for v in u.values()]))
        print(f"    {lbl}: avg universe size = {avg_sz:.1f} ({time.time()-t0:.0f}s)",
              flush=True)

    # Variants
    real_variants = [
        ("expanded_N=15_no_filter", {"kind": "no_filter"}, "N=15"),
        ("expanded_N=15_filter_refill", {"kind": "filter_refill", "mode": "mean", "window_days": 90}, "N=15"),
        ("expanded_N=25_filter_refill", {"kind": "filter_refill", "mode": "mean", "window_days": 90}, "N=25"),
        ("expanded_N=35_filter_refill", {"kind": "filter_refill", "mode": "mean", "window_days": 90}, "N=35"),
        ("expanded_N=all_filter_refill", {"kind": "filter_refill", "mode": "mean", "window_days": 90}, "N=all"),
    ]

    print(f"\n=== Real variants ===\n", flush=True)
    print(f"  {'variant':<35}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  {'totPnL':>7}  "
          f"{'avgL':>4}  {'avgS':>4}  {'avgU':>5}", flush=True)
    results = []
    real_excl_track_by_label = {}
    for label, v_spec, u_label in real_variants:
        t0 = time.time()
        df_v, excl_track = evaluate_inline(apd, universes[u_label], v_spec)
        res = summarize(df_v, label)
        results.append(res)
        real_excl_track_by_label[label] = excl_track
        df_v.to_csv(OUT_DIR / f"per_cycle_{label}.csv", index=False)
        print(f"  {label:<35}  {res['sharpe']:>+7.2f}  "
              f"[{res['ci_lo']:>+5.2f},{res['ci_hi']:>+5.2f}]  "
              f"{res['max_dd']:>+7.0f}  {res['total_pnl']:>+7.0f}  "
              f"{res['avg_n_long']:>4.2f}  {res['avg_n_short']:>4.2f}  "
              f"{res['avg_n_universe']:>5.1f}  ({time.time()-t0:.0f}s)", flush=True)

    # Best filter variant for matched placebo
    filter_results = [r for r in results if "filter_refill" in r["variant"]]
    best = max(filter_results, key=lambda r: r["sharpe"])
    best_u_label = next(u for lbl, _, u in real_variants if lbl == best["variant"])
    best_excl = real_excl_track_by_label[best["variant"]]
    print(f"\n  Best filter variant: {best['variant']}", flush=True)

    # Matched placebo
    print(f"\n=== Matched placebo ({N_PLACEBO_SEEDS} seeds) ===\n", flush=True)
    t0 = time.time()
    placebo_results = []
    for seed in range(N_PLACEBO_SEEDS):
        v = {"kind": "matched_placebo", "window_days": 90, "placebo_seed": seed}
        df_v, _ = evaluate_inline(apd, universes[best_u_label], v,
                                       real_exclusion_counts=best_excl)
        res = summarize(df_v, f"placebo_seed_{seed}")
        placebo_results.append(res)
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_PLACEBO_SEEDS} ({time.time()-t0:.0f}s)", flush=True)
    placebo_df = pd.DataFrame(placebo_results)
    placebo_df.to_csv(OUT_DIR / "matched_placebo.csv", index=False)
    p_sh = placebo_df["sharpe"].values
    print(f"\n  Matched-placebo Sharpe: mean={p_sh.mean():+.2f}, "
          f"p50={np.percentile(p_sh, 50):+.2f}, p95={np.percentile(p_sh, 95):+.2f}, "
          f"max={p_sh.max():+.2f}", flush=True)

    # Falsification
    print(f"\n=== Falsification vs placebo p95 ===\n", flush=True)
    p95 = np.percentile(p_sh, 95)
    for r in filter_results:
        rank = (p_sh < r["sharpe"]).mean() * 100
        beats = r["sharpe"] > p95
        print(f"  {r['variant']:<35}  Sharpe={r['sharpe']:+.2f}  rank={rank:>5.1f}%  "
              f"beats_p95={'✓' if beats else '✗'}", flush=True)

    print(f"\n=== Per-fold Sharpe ===", flush=True)
    print(f"  {'variant':<35}  " + " ".join(f"{'f'+str(f):>6}" for f in OOS_FOLDS),
          flush=True)
    for r in results:
        cells = " ".join(f"{r.get(f'sh_f{f}', 0):+5.2f}" for f in OOS_FOLDS)
        print(f"  {r['variant']:<35}  " + cells, flush=True)

    print(f"\n=== Comparison vs 51-panel Phase 2b v3 baseline ===\n", flush=True)
    print(f"  51_N=15 no_filter            Sharpe ~+0.34 (placebo 26%)", flush=True)
    print(f"  51_N=15 filter_refill        Sharpe +1.16 (placebo 51%, fails p95)", flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "results.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
