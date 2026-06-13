"""Phase K2: validate the K4 cost-aware swap rule before adoption.

Three checks:
  (1) Per-fold Sharpe — does K4 lift hold across all 9 OOS folds, or
      concentrated in 1-2 lucky folds?
  (2) Margin sweep at {0, 4.5, 9.0, 13.5, 18.0, 22.5} bps round-trip cost.
      For each margin level, compute Sharpe + matched-basket-size placebo.
      Pass if lift is monotone/stable AND beats p95 at multiple margins.
  (3) Out-of-sample calibration — derive slope (bps per pred-unit) using only
      past folds at each cutoff, check stability vs full-sample slope.

Output: outputs/vBTC_swap_rule/k2_robustness/
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

OUT = REPO / "outputs/vBTC_swap_rule/k2_robustness"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

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


def build_basket_cost_aware(sym_arr, pred_arr, side, cur_basket, picks_hist,
                              t, K_target, cost_margin_pred_units):
    if side == "long":
        order = np.argsort(-pred_arr)
    else:
        order = np.argsort(pred_arr)
    ranked = [sym_arr[i] for i in order]
    ranked_passing, n_excl = select_refill(ranked, side, K_target, picks_hist, 90, t)
    sym_to_pred = dict(zip(sym_arr, pred_arr))
    if not cur_basket:
        return set(ranked_passing[:K_target]), n_excl
    extended_top = set(ranked_passing)
    kept = cur_basket & extended_top
    if len(kept) >= K_target:
        kept_sorted = sorted(kept, key=lambda s: -sym_to_pred[s] if side == "long" else sym_to_pred[s])
        return set(kept_sorted[:K_target]), n_excl
    n_to_add = K_target - len(kept)
    candidates_to_add = [s for s in ranked_passing if s not in kept]
    if cost_margin_pred_units is not None and kept:
        if side == "long":
            weakest_kept = min(sym_to_pred[s] for s in kept)
        else:
            weakest_kept = max(sym_to_pred[s] for s in kept)
        filtered = []
        for s in candidates_to_add:
            p = sym_to_pred[s]
            lift = (p - weakest_kept) if side == "long" else (weakest_kept - p)
            if lift > cost_margin_pred_units:
                filtered.append(s)
        candidates_to_add = filtered
    return kept | set(candidates_to_add[:n_to_add]), n_excl


def evaluate(apd, universe, cost_margin_pred_units):
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
    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                          "cost_bps": 0, "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
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
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 2 * COST_PER_LEG, "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 0, "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        nl, n_el = build_basket_cost_aware(sym_arr, pred_arr, "long", cur_long,
                                              picks_hist, t, K, cost_margin_pred_units)
        ns, n_es = build_basket_cost_aware(sym_arr, pred_arr, "short", cur_short,
                                              picks_hist, t, K, cost_margin_pred_units)
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
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 2 * COST_PER_LEG, "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": 0,
                              "cost_bps": 0, "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        lr = [ret_l[s_] for s_ in nl]; sr = [ret_l[s_] for s_ in ns]
        spread = (np.mean(lr) - np.mean(sr)) * 1e4
        if is_flat:
            cost = 2 * COST_PER_LEG; is_flat = False
        else:
            cl = len(nl.symmetric_difference(cur_long)) / max(len(nl | cur_long), 1)
            cs = len(ns.symmetric_difference(cur_short)) / max(len(ns | cur_short), 1)
            cost = (cl + cs) * COST_PER_LEG
        net = spread - cost
        for s_ in nl:
            picks_hist[(s_, "long")].append((t, exit_l[s_], ret_l[s_] * 1e4 / len(nl)))
        for s_ in ns:
            picks_hist[(s_, "short")].append((t, exit_l[s_], -ret_l[s_] * 1e4 / len(ns)))
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "spread_bps": spread,
                      "cost_bps": cost, "net_bps": net,
                      "n_long": len(nl), "n_short": len(ns)})
        cur_long, cur_short = nl, ns
    return pd.DataFrame(rows)


def evaluate_matched_placebo(apd, universe, target_n_l, target_n_s, seed):
    rng = np.random.RandomState(seed)
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    hist_disp = deque(maxlen=252)
    cur_long, cur_short = set(), set()
    is_flat = False
    by_t = {t: g for t, g in df.groupby("open_time")}
    rows = []
    cycle_id = 0
    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        sym_arr = g_u["symbol"].to_numpy()
        ret_l_map = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        if len(g_u) >= 2 * K + 1:
            pred_arr = g_u["pred"].to_numpy()
            idx_t = np.argpartition(-pred_arr, K - 1)[:K]
            idx_b = np.argpartition(pred_arr, K - 1)[:K]
            disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
            skip = False
            if len(hist_disp) >= 30:
                thr = float(np.quantile(list(hist_disp), GATE_PCTILE))
                if disp < thr: skip = True
            hist_disp.append(disp)
        else:
            skip = True
        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "net_bps": 0, "n_long": 0, "n_short": 0})
            cycle_id += 1
            continue
        tl = target_n_l[cycle_id] if cycle_id < len(target_n_l) else 0
        ts = target_n_s[cycle_id] if cycle_id < len(target_n_s) else 0
        if tl == 0 or ts == 0 or len(sym_arr) < tl + ts:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "net_bps": 0, "n_long": 0, "n_short": 0})
            cycle_id += 1
            continue
        shuffled = rng.permutation(len(sym_arr))
        nl = set(sym_arr[shuffled[:tl]])
        ns = set(sym_arr[shuffled[tl:tl + ts]])
        lr = [ret_l_map[s] for s in nl]
        sr = [ret_l_map[s] for s in ns]
        spread = (np.mean(lr) - np.mean(sr)) * 1e4
        if is_flat:
            cost = 2 * COST_PER_LEG; is_flat = False
        else:
            cl = len(nl.symmetric_difference(cur_long)) / max(len(nl | cur_long), 1)
            cs = len(ns.symmetric_difference(cur_short)) / max(len(ns | cur_short), 1)
            cost = (cl + cs) * COST_PER_LEG
        net = spread - cost
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "net_bps": net,
                      "n_long": len(nl), "n_short": len(ns)})
        cur_long, cur_short = nl, ns
        cycle_id += 1
    return pd.DataFrame(rows)


def calibrate_full_sample(apd):
    df = apd[apd["fold"].isin(OOS_FOLDS)].sort_values(["open_time", "symbol"])
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    rows = []
    for t, g in df.groupby("open_time"):
        if len(g) < 9: continue
        p = g["pred"].values; rl = g["return_pct"].values
        idx_t = np.argpartition(-p, K - 1)[:K]
        idx_b = np.argpartition(p, K - 1)[:K]
        pd_t = float(p[idx_t].mean() - p[idx_b].mean())
        realized = (rl[idx_t].mean() - rl[idx_b].mean()) * 1e4
        rows.append({"pred_disp": pd_t, "realized_bps": realized})
    cal = pd.DataFrame(rows)
    slope = np.polyfit(cal["pred_disp"], cal["realized_bps"], 1)[0]
    return slope


def calibrate_per_fold(apd):
    """Per-fold slope: at each fold f, use folds < f to calibrate slope."""
    df = apd[apd["fold"].isin(OOS_FOLDS)].sort_values(["open_time", "symbol"])
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    out = []
    for cutoff_fold in [2, 3, 4, 5, 6, 7, 8, 9]:
        past = df[df["fold"] < cutoff_fold]
        rows = []
        for t, g in past.groupby("open_time"):
            if len(g) < 9: continue
            p = g["pred"].values; rl = g["return_pct"].values
            idx_t = np.argpartition(-p, K - 1)[:K]
            idx_b = np.argpartition(p, K - 1)[:K]
            pd_t = float(p[idx_t].mean() - p[idx_b].mean())
            realized = (rl[idx_t].mean() - rl[idx_b].mean()) * 1e4
            rows.append({"pred_disp": pd_t, "realized_bps": realized})
        if len(rows) < 100:
            out.append({"cutoff_fold": cutoff_fold, "n_obs": len(rows),
                         "slope": np.nan})
            continue
        cal = pd.DataFrame(rows)
        slope = np.polyfit(cal["pred_disp"], cal["realized_bps"], 1)[0]
        out.append({"cutoff_fold": cutoff_fold, "n_obs": len(rows),
                     "slope": slope, "pred_unit_9bps": 9.0 / abs(slope)})
    return pd.DataFrame(out)


def main():
    print("=== Phase K2: K4 robustness validation ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

    listings = get_listings()
    panel_syms = set(apd["symbol"].unique())

    def eligibility_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON]
    print(f"  Building universe...", flush=True)
    universe = build_rolling_ic_universe(apd, sampled_t, TOP_N, eligibility_at)

    # (1) Full-sample calibration
    slope_full = calibrate_full_sample(apd)
    print(f"\n  Full-sample slope: {slope_full:+.3f} bps/pred-unit", flush=True)

    # (2) Per-fold OOS calibration
    per_fold_cal = calibrate_per_fold(apd)
    per_fold_cal.to_csv(OUT / "per_fold_calibration.csv", index=False)
    print(f"\n--- (3) Per-fold OOS calibration (past-only) ---", flush=True)
    print(per_fold_cal.to_string(index=False), flush=True)

    # (3) Margin sweep with placebos
    margins_bps = [0, 4.5, 9.0, 13.5, 18.0]
    print(f"\n--- (2) Margin sweep + matched-basket-size placebo ---", flush=True)
    print(f"  {'margin_bps':>10}  {'pred_unit':>10}  {'real_Sh':>7}  "
          f"{'placebo_p95':>11}  {'totPnL':>8}  {'avg_L/S':>8}  {'beats_p95':>10}",
          flush=True)
    sweep_rows = []
    for m_bps in margins_bps:
        m_pred = m_bps / abs(slope_full) if m_bps > 0 else None
        t0 = time.time()
        df_v = evaluate(apd, universe, m_pred)
        real_sh = _sharpe(df_v["net_bps"].to_numpy())
        real_pnl = float(df_v["net_bps"].sum())
        avg_l = float(df_v["n_long"].mean()); avg_s = float(df_v["n_short"].mean())
        # Matched placebo
        target_l = df_v["n_long"].tolist(); target_s = df_v["n_short"].tolist()
        placebo_sh = []
        for seed in range(N_PLACEBO_SEEDS):
            df_p = evaluate_matched_placebo(apd, universe, target_l, target_s, seed)
            placebo_sh.append(_sharpe(df_p["net_bps"].to_numpy()))
        p95 = float(np.percentile(placebo_sh, 95))
        beats = real_sh > p95
        rank = float((np.array(placebo_sh) < real_sh).mean() * 100)
        print(f"  {m_bps:>10.1f}  {(m_pred or 0):>10.3f}  {real_sh:>+7.2f}  "
              f"{p95:>+11.2f}  {real_pnl:>+8.0f}  "
              f"{avg_l:.2f}/{avg_s:.2f}  {'PASS' if beats else 'FAIL':>10}  ({time.time()-t0:.0f}s)",
              flush=True)
        sweep_rows.append({
            "margin_bps": m_bps, "margin_pred_unit": m_pred,
            "real_sharpe": real_sh, "real_pnl": real_pnl,
            "placebo_mean": float(np.mean(placebo_sh)),
            "placebo_p50": float(np.percentile(placebo_sh, 50)),
            "placebo_p95": p95,
            "placebo_max": float(np.max(placebo_sh)),
            "rank_vs_placebo": rank, "beats_p95": beats,
            "avg_L": avg_l, "avg_S": avg_s,
        })
        df_v.to_csv(OUT / f"per_cycle_m{m_bps:.1f}.csv", index=False)
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUT / "margin_sweep.csv", index=False)

    # (4) Per-fold Sharpe breakdown at margin=9bps (K4 baseline)
    print(f"\n--- (1) Per-fold Sharpe at margin=9.0 bps (K4) ---", flush=True)
    m_pred = 9.0 / abs(slope_full)
    df_k4 = evaluate(apd, universe, m_pred)
    # Compare against production (margin=None)
    df_prod = evaluate(apd, universe, None)
    rows_pf = []
    print(f"  {'fold':>4}  {'K4_Sh':>6}  {'prod_Sh':>7}  {'K4_pnl':>7}  "
          f"{'prod_pnl':>7}  {'K4_L/S':>8}", flush=True)
    for f in OOS_FOLDS:
        d4 = df_k4[df_k4["fold"] == f]["net_bps"].to_numpy()
        dp = df_prod[df_prod["fold"] == f]["net_bps"].to_numpy()
        if len(d4) < 3 or len(dp) < 3: continue
        sh4 = _sharpe(d4); shp = _sharpe(dp)
        pnl4 = d4.sum(); pnlp = dp.sum()
        avg_l = df_k4[df_k4["fold"] == f]["n_long"].mean()
        avg_s = df_k4[df_k4["fold"] == f]["n_short"].mean()
        print(f"  {f:>4}  {sh4:>+6.2f}  {shp:>+7.2f}  {pnl4:>+7.0f}  "
              f"{pnlp:>+7.0f}  {avg_l:.2f}/{avg_s:.2f}", flush=True)
        rows_pf.append({"fold": f, "k4_sharpe": sh4, "prod_sharpe": shp,
                         "k4_pnl": pnl4, "prod_pnl": pnlp,
                         "k4_avg_L": avg_l, "k4_avg_S": avg_s,
                         "lift": sh4 - shp})
    pd.DataFrame(rows_pf).to_csv(OUT / "per_fold_lift.csv", index=False)

    print(f"\n=== Phase K2 verdict ===\n", flush=True)
    n_beats = sum(1 for r in sweep_rows if r["beats_p95"] and r["margin_bps"] > 0)
    n_lift = sum(1 for r in rows_pf if r["lift"] > 0)
    print(f"  Margin sweep: {n_beats}/4 non-zero margins beat placebo p95", flush=True)
    print(f"  Per-fold: K4 outperforms production in {n_lift}/9 folds", flush=True)
    if n_beats >= 3 and n_lift >= 6:
        verdict = "ROBUST — adopt K4 (margin=9bps) for production after live forward test"
    elif n_beats >= 2 and n_lift >= 5:
        verdict = "MIXED — lift exists but fragile to margin choice; pick a conservative margin"
    else:
        verdict = "FRAGILE — lift is concentrated; do not adopt without more validation"
    print(f"  {verdict}", flush=True)
    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
