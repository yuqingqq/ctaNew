"""Phase H2: protocol comparison — WINNER_21 vs WINNER_16 on 51-panel.

Reads:
  outputs/vBTC_audit_panel/all_predictions.parquet  (WINNER_21 baseline)
  outputs/vBTC_audit_winner16/all_predictions.parquet  (WINNER_16 treatment)

Runs Phase 2b v3 protocol (kline-listing PIT eligibility, N=15 rolling-IC,
filter_refill_90d_mean) on each. Then 100-seed matched placebo on the
better filter_refill variant.

Pass conditions to ADOPT WINNER_16:
  1. Sharpe within ±0.2 of WINNER_21 (or higher)
  2. Beats matched-placebo p95 OR matches WINNER_21's rank
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

OUT = REPO / "outputs/vBTC_winner16_protocol"
OUT.mkdir(parents=True, exist_ok=True)
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
LABELS = [
    ("WINNER_21_baseline", REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"),
    ("WINNER_16_pruned",   REPO / "outputs/vBTC_audit_winner16/all_predictions.parquet"),
]

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
    if not target_times: return {}
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


def select_placebo(ranked, side, k, n_excl, hist, window, rng, t):
    if n_excl <= 0: return ranked[:k]
    cutoff = t - pd.Timedelta(days=window)
    filterable = []
    for i, s in enumerate(ranked):
        ha = hist.get((s, side), [])
        vals = [c for (tt, et, c) in ha if et <= t and tt >= cutoff]
        if len(vals) >= MIN_PICKS_FOR_FILTER: filterable.append(i)
    skip = set(rng.choice(filterable, size=min(n_excl, len(filterable)), replace=False)) if filterable else set()
    kept = []
    for i, s in enumerate(ranked):
        if len(kept) >= k: break
        if i in skip: continue
        kept.append(s)
    return kept


def evaluate(apd, universe, variant, real_excl=None):
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
    rng = np.random.RandomState(variant.get("placebo_seed", 0))
    by_t = {t: g for t, g in df.groupby("open_time")}
    rows = []; excl_track = []
    for cycle_idx, t in enumerate(times):
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                          "spread_bps": 0, "cost_bps": 0, "net_bps": 0,
                          "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                          "n_excl_long": 0, "n_excl_short": 0})
            excl_track.append((0, 0))
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
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "spread_bps": 0, "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": 0, "n_excl_short": 0})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "spread_bps": 0, "cost_bps": 0, "net_bps": 0,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": 0, "n_excl_short": 0})
            excl_track.append((0, 0))
            continue
        order_d = np.argsort(-pred_arr); order_a = np.argsort(pred_arr)
        long_r = [sym_arr[i] for i in order_d]
        short_r = [sym_arr[i] for i in order_a]
        kind = variant["kind"]
        n_el = 0; n_es = 0
        if kind == "no_filter":
            cand_l = long_r[:K]; cand_s = short_r[:K]
        elif kind == "filter_refill":
            w = variant["window_days"]
            cand_l, n_el = select_refill(long_r, "long", K, picks_hist, w, t)
            cand_s, n_es = select_refill(short_r, "short", K, picks_hist, w, t)
        elif kind == "matched_placebo":
            if real_excl and cycle_idx < len(real_excl):
                tl, ts2 = real_excl[cycle_idx]
            else:
                tl, ts2 = 0, 0
            cand_l = select_placebo(long_r, "long", K, tl, picks_hist,
                                       variant["window_days"], rng, t)
            cand_s = select_placebo(short_r, "short", K, ts2, picks_hist,
                                       variant["window_days"], rng, t)
            n_el = tl; n_es = ts2
        else:
            cand_l = long_r[:K]; cand_s = short_r[:K]
        excl_track.append((n_el, n_es))
        c_ls = set(cand_l); c_ss = set(cand_s)
        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > PM_M:
            hist_basket = hist_basket[-PM_M:]
        if len(hist_basket) >= PM_M:
            p_l = [h["long"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            p_s = [h["short"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            nl = cur_long & c_ls
            ns = cur_short & c_ss
            for s in c_ls - cur_long:
                if all(s in p for p in p_l): nl.add(s)
            for s in c_ss - cur_short:
                if all(s in p for p in p_s): ns.add(s)
            if len(nl) > K:
                nl = set(sorted(nl, key=lambda s: -pred_arr[np.where(sym_arr == s)[0][0]])[:K])
            if len(ns) > K:
                ns = set(sorted(ns, key=lambda s: pred_arr[np.where(sym_arr == s)[0][0]])[:K])
        else:
            nl, ns = c_ls, c_ss
        if not nl or not ns:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "spread_bps": 0, "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0,
                              "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": n_el, "n_excl_short": n_es})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                              "spread_bps": 0, "cost_bps": 0, "net_bps": 0,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": n_el, "n_excl_short": n_es})
            continue
        lr = [ret_l[s] for s in nl]; sr = [ret_l[s] for s in ns]
        spread = (np.mean(lr) - np.mean(sr)) * 1e4
        if is_flat:
            cost = 2 * COST_PER_LEG; is_flat = False
        else:
            cl = len(nl.symmetric_difference(cur_long)) / max(len(nl | cur_long), 1)
            cs = len(ns.symmetric_difference(cur_short)) / max(len(ns | cur_short), 1)
            cost = (cl + cs) * COST_PER_LEG
        net = spread - cost
        for s in nl:
            picks_hist[(s, "long")].append((t, exit_l[s], ret_l[s] * 1e4 / len(nl)))
        for s in ns:
            picks_hist[(s, "short")].append((t, exit_l[s], -ret_l[s] * 1e4 / len(ns)))
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                      "spread_bps": spread, "cost_bps": cost, "net_bps": net,
                      "n_long": len(nl), "n_short": len(ns), "n_universe": len(g_u),
                      "n_excl_long": n_el, "n_excl_short": n_es})
        cur_long, cur_short = nl, ns
    return pd.DataFrame(rows), excl_track


def summarize(df_v, label):
    net = df_v["net_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    return {
        "variant": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
        "max_dd": _max_dd(net), "total_pnl": net.sum(),
        "avg_L": float(df_v["n_long"].mean()), "avg_S": float(df_v["n_short"].mean()),
    }


def main():
    print("=== Phase H2: WINNER_21 vs WINNER_16 on 51-panel ===\n", flush=True)
    listings = get_listings()

    state = {}
    summary_rows = []
    for label, path in LABELS:
        print(f"\n--- {label} ---", flush=True)
        apd = pd.read_parquet(path)
        apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
        apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
        panel_syms = set(apd["symbol"].unique())

        def eligibility_at(b, panel_syms=panel_syms):
            if isinstance(b, (int, np.integer)):
                ts = pd.Timestamp(b, unit="ms", tz="UTC")
            else:
                ts = pd.Timestamp(b)
                if ts.tz is None: ts = ts.tz_localize("UTC")
            cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
            return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

        target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
        sampled_t = target_t[::HORIZON]
        t0 = time.time()
        u = build_rolling_ic_universe(apd, sampled_t, TOP_N, eligibility_at)
        avg_sz = float(np.mean([len(v) for v in u.values()]))
        print(f"  N=15 universe built ({time.time()-t0:.0f}s), avg size={avg_sz:.1f}",
              flush=True)
        for v_label, v_spec in [
            ("no_filter", {"kind": "no_filter"}),
            ("filter_refill", {"kind": "filter_refill", "window_days": 90}),
        ]:
            t0 = time.time()
            df_v, excl = evaluate(apd, u, v_spec)
            res = summarize(df_v, f"{label}_{v_label}")
            summary_rows.append(res)
            df_v.to_csv(OUT / f"per_cycle_{label}_{v_label}.csv", index=False)
            print(f"  {v_label:<14}  Sharpe={res['sharpe']:>+6.2f}  "
                  f"[{res['ci_lo']:>+5.2f},{res['ci_hi']:>+5.2f}]  "
                  f"maxDD={res['max_dd']:>+7.0f}  totPnL={res['total_pnl']:>+7.0f}  "
                  f"L/S={res['avg_L']:.1f}/{res['avg_S']:.1f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if v_label == "filter_refill":
                state[label] = (apd, u, excl)

    pd.DataFrame(summary_rows).to_csv(OUT / "results.csv", index=False)

    base = next(r for r in summary_rows if r["variant"] == "WINNER_21_baseline_filter_refill")
    treat = next(r for r in summary_rows if r["variant"] == "WINNER_16_pruned_filter_refill")
    lift = treat["sharpe"] - base["sharpe"]
    print(f"\n=== WINNER_16 vs WINNER_21 (filter_refill) ===", flush=True)
    print(f"  WINNER_21 baseline    Sharpe={base['sharpe']:+.2f}", flush=True)
    print(f"  WINNER_16 (pruned)    Sharpe={treat['sharpe']:+.2f}", flush=True)
    print(f"  Δsharpe (lift)       = {lift:+.2f}", flush=True)
    if lift >= 0.2:
        verdict_h2 = "ADOPT (pruning lifts ≥+0.2)"
    elif lift >= -0.1:
        verdict_h2 = "ADOPT (parsimony — neutral within noise)"
    else:
        verdict_h2 = "REJECT (pruning hurts >0.1)"
    print(f"  decision (signal):    {verdict_h2}", flush=True)

    # Matched placebo on the BETTER of the two filter_refill variants
    better_label = "WINNER_16_pruned" if treat["sharpe"] >= base["sharpe"] else "WINNER_21_baseline"
    better_row = treat if treat["sharpe"] >= base["sharpe"] else base
    print(f"\n=== Matched placebo (100 seeds on {better_label}_filter_refill) ===", flush=True)
    apd_b, u_b, excl_b = state[better_label]
    t0 = time.time()
    placebo_rows = []
    for seed in range(N_PLACEBO_SEEDS):
        v = {"kind": "matched_placebo", "window_days": 90, "placebo_seed": seed}
        df_v, _ = evaluate(apd_b, u_b, v, real_excl=excl_b)
        placebo_rows.append(summarize(df_v, f"placebo_{seed}"))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/100 ({time.time()-t0:.0f}s)", flush=True)
    placebo_df = pd.DataFrame(placebo_rows)
    placebo_df.to_csv(OUT / "matched_placebo.csv", index=False)
    p_sh = placebo_df["sharpe"].values
    p95 = np.percentile(p_sh, 95)
    rank = (p_sh < better_row["sharpe"]).mean() * 100
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
    print(f"\n  {better_label}_filter_refill  Sharpe={better_row['sharpe']:+.2f}  "
          f"rank={rank:.0f}%  beats_p95={'PASS' if better_row['sharpe'] > p95 else 'FAIL'}",
          flush=True)

    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
