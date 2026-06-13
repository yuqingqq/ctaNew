"""Phase K: cost-aware swap rule sweep on WINNER_21 production stack.

Tests rank-hysteresis: keep an incumbent in the basket if its current rank is
within top-(K+B), not just top-K. New names enter only when there's room.
This makes the basket sticky and reduces turnover beyond PM persistence alone.

Also tests cost-margin swap: a name only enters when its predicted pred-lift
over the incumbent it would displace exceeds the swap cost (4.5 bps) by a
margin. Calibration: regress realized spread on pred-difference per cycle to
get bps-per-pred-unit.

Variants (all with production conv_gate, filter_refill_90d_mean, PM_M=2):

  K0 production              — production baseline
  K1 hysteresis_B2           — keep incumbents in top-6 (B=2)
  K2 hysteresis_B4           — keep incumbents in top-8 (B=4)
  K3 hysteresis_B6           — keep incumbents in top-10 (B=6)
  K4 cost_margin_swap        — swap only if alpha-lift > 9 bps (round-trip cost)
  K5 hyst_B4_+cost_margin    — combined: hysteresis B=4 + cost margin

After identifying best, run 100-seed matched churn-placebo: randomly drop
the same proportion of "real" swaps to match the realized turnover of the
best variant. Tests whether the lift is genuine signal or just exposure
reduction.

Output: outputs/vBTC_swap_rule/{results.csv, per_cycle_*.csv,
                                matched_placebo.csv, cost_calibration.csv}
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

OUT = REPO / "outputs/vBTC_swap_rule"
OUT.mkdir(parents=True, exist_ok=True)
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"

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


def get_listings():
    L = {}
    for d in KLINES_DIR.iterdir():
        if not d.is_dir(): continue
        m5 = d / "5m"
        if not m5.exists(): continue
        f = sorted(m5.glob("*.parquet"))
        if not f: continue
        try: L[d.name] = pd.Timestamp(f[0].stem, tz="UTC")
        except Exception: pass
    return L


def build_rolling_ic_universe(all_pred_df, target_times, top_n, eligibility_at_t):
    bar_ms = 5 * 60 * 1000
    window_ms = IC_WINDOW_DAYS * 288 * bar_ms
    update_ms = IC_UPDATE_DAYS * 288 * bar_ms
    df = all_pred_df.copy()
    df["t_int"] = to_ms_int(df["open_time"])
    df["exit_t_int"] = to_ms_int(df["exit_time"])
    df_clean = df.dropna(subset=["alpha_A"])
    t0_ms = int(pd.Timestamp(target_times[0]).timestamp() * 1000)
    boundaries = []
    for t in target_times:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        n = (t_ms - t0_ms) // update_ms
        b = t0_ms + n * update_ms
        boundaries.append((t, b))
    unique_b = sorted(set(b for _, b in boundaries))
    b2u = {}
    for b in unique_b:
        elig = eligibility_at_t(b)
        past = df_clean[(df_clean["t_int"] >= b - window_ms) &
                          (df_clean["t_int"] < b) &
                          (df_clean["exit_t_int"] <= b) &
                          (df_clean["symbol"].isin(elig))]
        if len(past) < 1000:
            b2u[b] = set(); continue
        ics = past.groupby("symbol").apply(
            lambda g: g["pred"].rank().corr(g["alpha_A"].rank())
            if len(g) >= MIN_OBS_PER_SYM else np.nan
        )
        ics_sorted = ics.dropna().sort_values(ascending=False)
        b2u[b] = set(ics_sorted.head(top_n).index.tolist()) if top_n else set(ics_sorted.index)
    return {t: b2u[b] for t, b in boundaries}


def filter_decision(hist, window_days, t):
    if not hist: return True
    cutoff = t - pd.Timedelta(days=window_days)
    vals = [c for (tt, et, c) in hist if et <= t and tt >= cutoff]
    if len(vals) < MIN_PICKS_FOR_FILTER: return True
    return float(np.mean(vals)) >= 0


def select_refill(ranked, side, k, hist, window, t):
    kept = []; n_excl = 0
    for s in ranked:
        if len(kept) >= k: break
        if filter_decision(hist.get((s, side), []), window, t):
            kept.append(s)
        else: n_excl += 1
    return kept, n_excl


def build_basket_hysteresis(sym_arr, pred_arr, side, cur_basket, picks_hist,
                              t, K_target, buffer_size,
                              cost_margin_pred_units=None):
    """Build basket with rank hysteresis.

    side: "long" or "short" (controls sort direction)
    K_target: target basket size
    buffer_size: incumbent kept if rank within top-(K+buffer)
    cost_margin_pred_units: if set, new name only enters if its pred lift over
        the WEAKEST KEPT incumbent exceeds this in pred-unit space (None disables).

    Returns: set of selected symbols, list of n_excluded by SS filter
    """
    # Apply SS filter refill first
    if side == "long":
        order = np.argsort(-pred_arr)
    else:
        order = np.argsort(pred_arr)
    ranked = [sym_arr[i] for i in order]
    ranked_passing, n_excl = select_refill(ranked, side, K_target + buffer_size,
                                              picks_hist, 90, t)
    # ranked_passing has up to K_target+buffer_size names that pass SS filter,
    # ordered by pred strength (best first)

    # Get pred for each ranked name (post-filter)
    sym_to_pred = dict(zip(sym_arr, pred_arr))
    ranked_with_pred = [(s, sym_to_pred[s]) for s in ranked_passing]

    if not cur_basket:
        # No incumbents — take top K_target from filtered ranking
        return set(ranked_passing[:K_target]), n_excl

    # Hysteresis: keep incumbents that are within top-(K+B) of filtered ranking
    extended_top = set(ranked_passing)  # symbols that survived to top-(K+B)
    kept = cur_basket & extended_top

    # If kept already ≥ K_target, prune to top-K by pred among kept
    if len(kept) >= K_target:
        kept_sorted = sorted(kept,
                              key=lambda s: -sym_to_pred[s] if side == "long" else sym_to_pred[s])
        return set(kept_sorted[:K_target]), n_excl

    # Need to add (K_target - len(kept)) new names from ranked_passing
    n_to_add = K_target - len(kept)
    candidates_to_add = [s for s in ranked_passing if s not in kept]

    if cost_margin_pred_units is not None and kept:
        # Compute weakest kept incumbent's pred
        if side == "long":
            weakest_kept_pred = min(sym_to_pred[s] for s in kept)
        else:
            weakest_kept_pred = max(sym_to_pred[s] for s in kept)
        # Filter candidates: only add if pred-lift > cost_margin in pred units
        filtered_candidates = []
        for s in candidates_to_add:
            p = sym_to_pred[s]
            if side == "long":
                lift = p - weakest_kept_pred
                if lift > cost_margin_pred_units:
                    filtered_candidates.append(s)
            else:
                lift = weakest_kept_pred - p
                if lift > cost_margin_pred_units:
                    filtered_candidates.append(s)
        candidates_to_add = filtered_candidates

    new_names = candidates_to_add[:n_to_add]
    return kept | set(new_names), n_excl


def evaluate(apd, universe, buffer_size, cost_margin_pred_units=None):
    """Run protocol with hysteresis swap rule."""
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()

    hist_disp = deque(maxlen=GATE_LOOKBACK)
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    by_t = {t: g for t, g in df.groupby("open_time")}
    rows = []

    for cycle_idx, t in enumerate(times):
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                          "spread_bps": 0, "cost_bps": 0, "net_bps": 0,
                          "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                          "n_excl_long": 0, "n_excl_short": 0,
                          "churn_l": 0, "churn_s": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))

        # Production conv_gate
        idx_t = np.argpartition(-pred_arr, K - 1)[:K]
        idx_b = np.argpartition(pred_arr, K - 1)[:K]
        disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
        skip = False
        if len(hist_disp) >= 30:
            thr = float(np.quantile(list(hist_disp), GATE_PCTILE))
            if disp < thr: skip = True
        hist_disp.append(disp)

        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "spread_bps": 0, "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": 0, "n_excl_short": 0,
                              "churn_l": 0, "churn_s": 0})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "spread_bps": 0, "cost_bps": 0, "net_bps": 0,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": 0, "n_excl_short": 0,
                              "churn_l": 0, "churn_s": 0})
            continue

        # Build baskets with hysteresis
        nl, n_el = build_basket_hysteresis(sym_arr, pred_arr, "long",
                                              cur_long, picks_hist, t,
                                              K, buffer_size, cost_margin_pred_units)
        ns, n_es = build_basket_hysteresis(sym_arr, pred_arr, "short",
                                              cur_short, picks_hist, t,
                                              K, buffer_size, cost_margin_pred_units)

        # Still apply PM persistence on top of hysteresis (preserves entry-time bar)
        c_ls = nl; c_ss = ns
        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > PM_M:
            hist_basket = hist_basket[-PM_M:]
        if len(hist_basket) >= PM_M:
            p_l = [h["long"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            p_s = [h["short"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            nl_pm = cur_long & c_ls
            ns_pm = cur_short & c_ss
            for s_ in c_ls - cur_long:
                if all(s_ in p for p in p_l): nl_pm.add(s_)
            for s_ in c_ss - cur_short:
                if all(s_ in p for p in p_s): ns_pm.add(s_)
            if len(nl_pm) > K:
                nl_pm = set(sorted(nl_pm, key=lambda s_: -pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
            if len(ns_pm) > K:
                ns_pm = set(sorted(ns_pm, key=lambda s_: pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
            nl, ns = nl_pm, ns_pm

        if not nl or not ns:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "spread_bps": 0, "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0,
                              "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": n_el, "n_excl_short": n_es,
                              "churn_l": 0, "churn_s": 0})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                              "spread_bps": 0, "cost_bps": 0, "net_bps": 0,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": n_el, "n_excl_short": n_es,
                              "churn_l": 0, "churn_s": 0})
            continue

        lr = [ret_l[s_] for s_ in nl]; sr_ = [ret_l[s_] for s_ in ns]
        spread = (np.mean(lr) - np.mean(sr_)) * 1e4
        if is_flat:
            cost = 2 * COST_PER_LEG; is_flat = False
            cl = 0; cs = 0
        else:
            cl = len(nl.symmetric_difference(cur_long)) / max(len(nl | cur_long), 1)
            cs = len(ns.symmetric_difference(cur_short)) / max(len(ns | cur_short), 1)
            cost = (cl + cs) * COST_PER_LEG
        net = spread - cost
        for s_ in nl:
            picks_hist[(s_, "long")].append((t, exit_l[s_], ret_l[s_] * 1e4 / len(nl)))
        for s_ in ns:
            picks_hist[(s_, "short")].append((t, exit_l[s_], -ret_l[s_] * 1e4 / len(ns)))
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                      "spread_bps": spread, "cost_bps": cost, "net_bps": net,
                      "n_long": len(nl), "n_short": len(ns), "n_universe": len(g_u),
                      "n_excl_long": n_el, "n_excl_short": n_es,
                      "churn_l": cl, "churn_s": cs})
        cur_long, cur_short = nl, ns
    return pd.DataFrame(rows)


def summarize(df_v, label):
    net = df_v["net_bps"].to_numpy()
    if len(net) < 10:
        return {"variant": label, "sharpe": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                 "max_dd": 0.0, "total_pnl": 0.0,
                 "avg_L": 0.0, "avg_S": 0.0,
                 "avg_churn_l": 0.0, "avg_churn_s": 0.0,
                 "avg_cost_bps": 0.0, "avg_spread_bps": 0.0,
                 "n_cycles": len(net)}
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    return {
        "variant": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
        "max_dd": _max_dd(net), "total_pnl": net.sum(),
        "avg_L": float(df_v["n_long"].mean()), "avg_S": float(df_v["n_short"].mean()),
        "avg_churn_l": float(df_v["churn_l"].mean()),
        "avg_churn_s": float(df_v["churn_s"].mean()),
        "avg_cost_bps": float(df_v["cost_bps"].mean()),
        "avg_spread_bps": float(df_v["spread_bps"].mean()),
        "n_cycles": len(net),
    }


def calibrate_cost_margin(apd):
    """Compute pred-unit equivalent of 9 bps swap cost via cycle-level regression."""
    df = apd[apd["fold"].isin(OOS_FOLDS)].sort_values(["open_time", "symbol"])
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    rows = []
    for t, g in df.groupby("open_time"):
        if len(g) < 9: continue
        p = g["pred"].values
        rl = g["return_pct"].values
        idx_t = np.argpartition(-p, K - 1)[:K]
        idx_b = np.argpartition(p, K - 1)[:K]
        pred_disp = float(p[idx_t].mean() - p[idx_b].mean())
        realized = (rl[idx_t].mean() - rl[idx_b].mean()) * 1e4
        rows.append({"pred_disp": pred_disp, "realized_bps": realized})
    df_calib = pd.DataFrame(rows)
    if len(df_calib) > 100:
        slope = np.polyfit(df_calib["pred_disp"], df_calib["realized_bps"], 1)[0]
        pred_per_bps = 1.0 / abs(slope) if slope != 0 else 1.0
    else:
        pred_per_bps = 1.0  # fallback
    margin_9bps = 9.0 * pred_per_bps
    return slope, margin_9bps, df_calib


def main():
    print("=== Phase K: cost-aware swap rule sweep ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    print(f"  apd: {len(apd):,} rows, {apd.symbol.nunique()} syms", flush=True)

    listings = get_listings()
    panel_syms = set(apd["symbol"].unique())

    def eligibility_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON]
    print(f"\n  Building N=15 rolling-IC universe...", flush=True)
    universe = build_rolling_ic_universe(apd, sampled_t, TOP_N, eligibility_at)
    print(f"  done.\n", flush=True)

    # Calibration for cost-margin
    slope, margin_9bps, df_calib = calibrate_cost_margin(apd)
    print(f"  Cost calibration: slope={slope:+.3f} bps/pred-unit; "
          f"9bps margin = {margin_9bps:.3f} pred-units\n", flush=True)
    df_calib.to_csv(OUT / "cost_calibration.csv", index=False)

    variants = [
        ("K0_production",       0, None),
        ("K1_hysteresis_B2",    2, None),
        ("K2_hysteresis_B4",    4, None),
        ("K3_hysteresis_B6",    6, None),
        ("K4_cost_margin",      0, margin_9bps),
        ("K5_hyst_B4_+cost",    4, margin_9bps),
    ]

    print(f"  {'variant':<28}  {'Sharpe':>7}  {'CI':>17}  "
          f"{'avg_churn':>9}  {'avg_cost':>8}  {'totPnL':>7}  {'L/S':>6}",
          flush=True)
    results = []
    excl_track = {}
    for label, buf, cost_margin in variants:
        t0 = time.time()
        df_v = evaluate(apd, universe, buf, cost_margin)
        res = summarize(df_v, label)
        results.append(res)
        df_v.to_csv(OUT / f"per_cycle_{label}.csv", index=False)
        avg_churn = (res["avg_churn_l"] + res["avg_churn_s"]) / 2
        print(f"  {label:<28}  {res['sharpe']:>+7.2f}  "
              f"[{res['ci_lo']:>+5.2f},{res['ci_hi']:>+5.2f}]  "
              f"{avg_churn:>9.3f}  {res['avg_cost_bps']:>+8.2f}  "
              f"{res['total_pnl']:>+7.0f}  {res['avg_L']:.1f}/{res['avg_S']:.1f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)

    # Best non-production
    cands = [r for r in results if r["variant"] != "K0_production"]
    best = max(cands, key=lambda r: r["sharpe"])
    prod = next(r for r in results if r["variant"] == "K0_production")
    print(f"\n=== Best replacement: {best['variant']}  Sharpe={best['sharpe']:+.2f} ===",
          flush=True)
    print(f"  vs production           Sharpe={prod['sharpe']:+.2f}", flush=True)
    print(f"  Δsharpe                = {best['sharpe']-prod['sharpe']:+.2f}", flush=True)

    # Phase J-style verdict
    print(f"\n=== Phase K verdict ===", flush=True)
    if best["sharpe"] > prod["sharpe"] + 0.20:
        verdict = f"PROMISING — {best['variant']} lifts Sharpe by +{best['sharpe']-prod['sharpe']:.2f}; run placebo"
    elif best["sharpe"] > prod["sharpe"] - 0.10:
        verdict = f"NEUTRAL — {best['variant']} matches production within noise; defer placebo"
    else:
        verdict = f"NO IMPROVEMENT — keep production swap rule"
    print(f"  {verdict}", flush=True)
    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
