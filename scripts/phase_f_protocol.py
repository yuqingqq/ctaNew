"""Phase F5/F6: Phase 2b v3 protocol on dedup_23_fund vs dedup_26_fund_sector,
matched-placebo on the better variant.

Compared against the 51-panel baseline at Sharpe +1.16 (p51 of placebo).

Pass conditions (BOTH must hold to adopt sector features):
  1. real Sharpe lift >= +0.3 over dedup_23_fund on the same protocol
  2. real Sharpe beats matched-placebo p95

Outputs:
  outputs/vBTC_sector_features/per_cycle_<label>.csv
  outputs/vBTC_sector_features/results.csv
  outputs/vBTC_sector_features/matched_placebo.csv
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

OUT_DIR = REPO / "outputs/vBTC_sector_features"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
LABELS = ["dedup_23_fund", "dedup_26_fund_sector"]

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
K = 4
MIN_PICKS_FOR_FILTER = 30
MIN_OBS_PER_SYM = 100
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
OOS_FOLDS = list(range(1, 10))
N_PLACEBO_SEEDS = 100
MIN_HISTORY_DAYS = 60
TOP_N = 15


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


def get_listing_dates():
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


def build_rolling_ic_universe(all_pred_df, target_times, top_n, eligibility_at_t):
    bar_ms = 5 * 60 * 1000
    window_ms = IC_WINDOW_DAYS * 288 * bar_ms
    update_ms = IC_UPDATE_DAYS * 288 * bar_ms

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


def filter_decision(history_arr, window_days, decision_t):
    if not history_arr: return True
    cutoff = decision_t - pd.Timedelta(days=window_days)
    vals = [c for (t, et, c) in history_arr if et <= decision_t and t >= cutoff]
    if len(vals) < MIN_PICKS_FOR_FILTER: return True
    arr = np.asarray(vals, dtype=float)
    return arr.mean() >= 0


def select_with_refill(ranked_syms, side, k, picks_history, filter_window, t):
    kept = []; n_excl = 0
    for s in ranked_syms:
        if len(kept) >= k: break
        if filter_decision(picks_history.get((s, side), []), filter_window, t):
            kept.append(s)
        else:
            n_excl += 1
    return kept, n_excl


def select_with_matched_placebo(ranked_syms, side, k, n_excl, picks_history,
                                  filter_window, rng, t):
    if n_excl <= 0: return ranked_syms[:k]
    cutoff = t - pd.Timedelta(days=filter_window)
    filterable = []
    for i, s in enumerate(ranked_syms):
        ha = picks_history.get((s, side), [])
        vals = [c for (tt, et, c) in ha if et <= t and tt >= cutoff]
        if len(vals) >= MIN_PICKS_FOR_FILTER:
            filterable.append(i)
    skip = set(rng.choice(filterable, size=min(n_excl, len(filterable)), replace=False)) if filterable else set()
    kept = []
    for i, s in enumerate(ranked_syms):
        if len(kept) >= k: break
        if i in skip: continue
        kept.append(s)
    return kept


def evaluate(all_pred_df, rolling_universe, variant, real_excl_counts=None):
    df = all_pred_df.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
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
    rows = []; excl_track = []

    for cycle_idx, t in enumerate(times):
        g = audit_by_t.get(t)
        if g is None: continue
        u = rolling_universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                          "spread_bps": 0.0, "cost_bps": 0.0, "net_bps": 0.0,
                          "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                          "n_excl_long": 0, "n_excl_short": 0})
            excl_track.append((0, 0))
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
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "spread_bps": 0.0, "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": 0, "n_excl_short": 0})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "spread_bps": 0.0, "cost_bps": 0.0, "net_bps": 0.0,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": 0, "n_excl_short": 0})
            excl_track.append((0, 0))
            continue

        order_desc = np.argsort(-pred_arr)
        order_asc = np.argsort(pred_arr)
        long_ranked = [sym_arr[i] for i in order_desc]
        short_ranked = [sym_arr[i] for i in order_asc]

        kind = variant["kind"]
        n_excl_l = 0; n_excl_s = 0
        if kind == "no_filter":
            cand_long = long_ranked[:K]
            cand_short = short_ranked[:K]
        elif kind == "filter_refill":
            window = variant["window_days"]
            cand_long, n_excl_l = select_with_refill(long_ranked, "long", K,
                                                       picks_history, window, t)
            cand_short, n_excl_s = select_with_refill(short_ranked, "short", K,
                                                         picks_history, window, t)
        elif kind == "matched_placebo":
            if real_excl_counts and cycle_idx < len(real_excl_counts):
                target_l, target_s = real_excl_counts[cycle_idx]
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
            cand_long = long_ranked[:K]; cand_short = short_ranked[:K]

        excl_track.append((n_excl_l, n_excl_s))

        cand_long_set = set(cand_long); cand_short_set = set(cand_short)
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
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "spread_bps": 0.0, "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0,
                              "n_universe": len(g_u), "n_excl_long": n_excl_l,
                              "n_excl_short": n_excl_s})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                              "spread_bps": 0.0, "cost_bps": 0.0, "net_bps": 0.0,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": n_excl_l, "n_excl_short": n_excl_s})
            continue

        long_rets = [ret_lookup[s] for s in new_long]
        short_rets = [ret_lookup[s] for s in new_short]
        spread = (np.mean(long_rets) - np.mean(short_rets)) * 1e4
        if is_flat:
            cost = 2 * COST_PER_LEG; is_flat = False
        else:
            cl = (len(new_long.symmetric_difference(cur_long)) / max(len(new_long | cur_long), 1))
            cs = (len(new_short.symmetric_difference(cur_short)) / max(len(new_short | cur_short), 1))
            cost = (cl + cs) * COST_PER_LEG
        net = spread - cost

        for s in new_long:
            picks_history[(s, "long")].append((t, exit_lookup[s], ret_lookup[s] * 1e4 / len(new_long)))
        for s in new_short:
            picks_history[(s, "short")].append((t, exit_lookup[s], -ret_lookup[s] * 1e4 / len(new_short)))

        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                      "spread_bps": spread, "cost_bps": cost, "net_bps": net,
                      "n_long": len(new_long), "n_short": len(new_short),
                      "n_universe": len(g_u), "n_excl_long": n_excl_l,
                      "n_excl_short": n_excl_s})
        cur_long, cur_short = new_long, new_short

    return pd.DataFrame(rows), excl_track


def summarize(df_v, label):
    net = df_v["net_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    pf = {}
    for fid in OOS_FOLDS:
        fd = df_v[df_v["fold"] == fid]["net_bps"].to_numpy()
        if len(fd) >= 3: pf[fid] = _sharpe(fd)
    return {
        "variant": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
        "max_dd": _max_dd(net), "total_pnl": net.sum(), "mean_bps": net.mean(),
        "avg_L": float(df_v["n_long"].mean()), "avg_S": float(df_v["n_short"].mean()),
        "avg_excl_L": float(df_v["n_excl_long"].mean()),
        "avg_excl_S": float(df_v["n_excl_short"].mean()),
        **{f"sh_f{f}": v for f, v in pf.items()},
    }


def run_for_label(label, listings, panel_syms):
    apd_path = OUT_DIR / f"audit_{label}" / "all_predictions.parquet"
    apd = pd.read_parquet(apd_path)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

    def eligibility_at(b):
        if isinstance(b, (int, np.integer)):
            ts = pd.Timestamp(b, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(b)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_times = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    target_times_sampled = target_times[::HORIZON]
    u = build_rolling_ic_universe(apd, target_times_sampled, TOP_N, eligibility_at)

    out = {}
    print(f"\n--- {label} ---", flush=True)
    print(f"  {'variant':<22}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  {'totPnL':>7}  "
          f"{'L/S':>5}  {'excl':>5}", flush=True)
    for v_label, v_spec in [
        ("no_filter", {"kind": "no_filter"}),
        ("filter_refill", {"kind": "filter_refill", "window_days": 90}),
    ]:
        t0 = time.time()
        df_v, excl_track = evaluate(apd, u, v_spec)
        res = summarize(df_v, f"{label}_{v_label}")
        out[v_label] = (res, df_v, excl_track, apd, u)
        df_v.to_csv(OUT_DIR / f"per_cycle_{label}_{v_label}.csv", index=False)
        print(f"  {v_label:<22}  {res['sharpe']:>+7.2f}  "
              f"[{res['ci_lo']:>+5.2f},{res['ci_hi']:>+5.2f}]  "
              f"{res['max_dd']:>+7.0f}  {res['total_pnl']:>+7.0f}  "
              f"{res['avg_L']:>2.1f}/{res['avg_S']:>2.1f}  "
              f"{res['avg_excl_L']:>2.1f}/{res['avg_excl_S']:>2.1f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    print(f"=== Phase F5/F6: dedup_23_fund vs dedup_26_fund_sector ===\n", flush=True)

    listings = get_listing_dates()
    # Use union of panel syms (same for both)
    apd0 = pd.read_parquet(OUT_DIR / f"audit_{LABELS[0]}/all_predictions.parquet",
                              columns=["symbol"])
    panel_syms = set(apd0["symbol"].unique())
    print(f"  Panel syms: {len(panel_syms)}, listings: {len(listings)}", flush=True)

    results_by_label = {}
    for label in LABELS:
        results_by_label[label] = run_for_label(label, listings, panel_syms)

    # Summary table
    print(f"\n=== Real variant comparison ===\n", flush=True)
    summary_rows = []
    for label in LABELS:
        for v_label, (res, _, _, _, _) in results_by_label[label].items():
            summary_rows.append(res)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "results.csv", index=False)

    # Choose better filter_refill variant for matched placebo
    bs_rows = [r for r in summary_rows if "filter_refill" in r["variant"]]
    best = max(bs_rows, key=lambda r: r["sharpe"])
    best_label_full = best["variant"]
    base_label = best_label_full.split("_filter_refill")[0]
    print(f"  Better filter_refill: {best_label_full}  Sharpe={best['sharpe']:+.2f}", flush=True)
    baseline = next(r for r in bs_rows if r["variant"] != best_label_full)
    print(f"  Other filter_refill : {baseline['variant']}  Sharpe={baseline['sharpe']:+.2f}", flush=True)
    print(f"  Δsharpe (sector vs dedup_23_fund) = "
          f"{best['sharpe'] - baseline['sharpe']:+.2f}", flush=True)

    # Pass gate 1: lift >= +0.3
    lift = next(r for r in bs_rows if r["variant"] == "dedup_26_fund_sector_filter_refill")["sharpe"] - \
            next(r for r in bs_rows if r["variant"] == "dedup_23_fund_filter_refill")["sharpe"]
    print(f"  Sector lift over baseline: {lift:+.2f}  "
          f"{'PASS' if lift >= 0.3 else 'FAIL'} (threshold +0.30)", flush=True)

    # Matched placebo on best variant
    print(f"\n=== Matched placebo ({N_PLACEBO_SEEDS} seeds on {best_label_full}) ===\n",
          flush=True)
    _, _, best_excl, best_apd, best_u = results_by_label[base_label]["filter_refill"]
    t0 = time.time()
    placebo_rows = []
    for seed in range(N_PLACEBO_SEEDS):
        v = {"kind": "matched_placebo", "window_days": 90, "placebo_seed": seed}
        df_v, _ = evaluate(best_apd, best_u, v, real_excl_counts=best_excl)
        placebo_rows.append(summarize(df_v, f"placebo_seed_{seed}"))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_PLACEBO_SEEDS} ({time.time()-t0:.0f}s)", flush=True)
    placebo_df = pd.DataFrame(placebo_rows)
    placebo_df.to_csv(OUT_DIR / "matched_placebo.csv", index=False)
    p_sh = placebo_df["sharpe"].values
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={np.percentile(p_sh,95):+.2f}, max={p_sh.max():+.2f}", flush=True)
    rank = (p_sh < best["sharpe"]).mean() * 100
    beats = best["sharpe"] > np.percentile(p_sh, 95)
    print(f"\n  {best_label_full}  Sharpe={best['sharpe']:+.2f}  rank={rank:.1f}%  "
          f"beats_p95={'PASS' if beats else 'FAIL'}", flush=True)

    # Final verdict
    print(f"\n=== Phase F verdict ===\n", flush=True)
    if lift >= 0.3 and beats:
        print(f"  ADOPT sector features  (lift {lift:+.2f}, beats p95)", flush=True)
    elif lift >= 0.3 and not beats:
        print(f"  REJECT — lift {lift:+.2f} but fails matched-placebo p95 = "
              f"indistinguishable from random exclusion", flush=True)
    elif lift < 0.3 and beats:
        print(f"  REJECT — beats p95 but lift only {lift:+.2f} < +0.3 threshold", flush=True)
    else:
        print(f"  REJECT — lift {lift:+.2f} and fails matched p95", flush=True)

    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
