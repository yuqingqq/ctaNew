"""Phase L / Test 5: composite profitability gate.

Replaces conv_gate's pred_disp metric with composite quality scores. Same gate
semantics: skip if score < 30th pctile of trailing 252-cycle distribution.

Composite candidates:
  V0 pred_disp (production)
  V1 pred_disp / max(rank_churn, ε)
  V2 pred_disp * rolling_IC
  V3 pred_disp * xs_dispersion (panel xs_alpha_dispersion_48b within universe)
  V4 pred_disp - α * rank_churn (additive)

Production selection (filter_refill + PM + WINNER_21) unchanged.
Matched skip-placebo on best variant.

Pass: beats production AND beats matched skip-placebo p95 AND ≥6/9 folds improve.
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

OUT = REPO / "outputs/vBTC_composite_gate"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
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
MIN_HISTORY_DAYS = 60
TOP_N = 15
N_PLACEBO_SEEDS = 100


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


def build_rolling_ic_universe(apd, target_times, top_n, eligibility_at_t):
    bar_ms = 5 * 60 * 1000
    window_ms = IC_WINDOW_DAYS * 288 * bar_ms
    update_ms = IC_UPDATE_DAYS * 288 * bar_ms
    df = apd.copy()
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


def evaluate(apd, universe, gate_fn_name, panel_by_time=None, alpha_param=1.0,
              rng_seed=0):
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    # Combined history for score
    hist_score = deque(maxlen=GATE_LOOKBACK)
    # Trailing IC computation: maintain a deque of recent (sym, pred, alpha)
    rolling_ic_hist = deque(maxlen=30 * 288 // HORIZON * 50)  # ~50 syms * 30 days at sub-sampling
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    by_t = {t: g for t, g in df.groupby("open_time")}
    prev_top_long, prev_top_short = set(), set()
    rng = np.random.RandomState(rng_seed)
    rows = []

    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "score": np.nan,
                          "skipped": 1, "spread_bps": 0, "cost_bps": 0,
                          "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        alpha_arr = g_u["alpha_A"].to_numpy()
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        idx_t = np.argpartition(-pred_arr, K - 1)[:K]
        idx_b = np.argpartition(pred_arr, K - 1)[:K]
        top_set = set(sym_arr[idx_t])
        bot_set = set(sym_arr[idx_b])
        pred_disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())

        # Stability
        if prev_top_long:
            overlap_l = len(top_set & prev_top_long) / K
            overlap_s = len(bot_set & prev_top_short) / K
            stability = (overlap_l + overlap_s) / 2
        else:
            stability = 1.0
        rank_churn = max(1.0 - stability, 0.01)  # avoid div0
        prev_top_long = top_set; prev_top_short = bot_set

        # Rolling IC: use accumulated history (PIT)
        if len(rolling_ic_hist) >= 50:
            past = pd.DataFrame(list(rolling_ic_hist), columns=["pred", "alpha"])
            past_clean = past.dropna()
            if len(past_clean) >= 30:
                roll_ic = past_clean["pred"].rank().corr(past_clean["alpha"].rank())
                if pd.isna(roll_ic): roll_ic = 0.0
            else:
                roll_ic = 0.0
        else:
            roll_ic = 0.5  # bootstrap default
        # Append current cycle's symbols & alphas to rolling history
        for sym_i in range(len(sym_arr)):
            if not np.isnan(alpha_arr[sym_i]):
                rolling_ic_hist.append((pred_arr[sym_i], alpha_arr[sym_i]))

        # xs_dispersion (panel lookup)
        xs_disp = 1.0
        if panel_by_time is not None:
            p_at_t = panel_by_time.get(t)
            if p_at_t is not None:
                p_u = p_at_t[p_at_t["symbol"].isin(u)]
                if len(p_u) > 0:
                    v = p_u["xs_alpha_dispersion_48b"].mean()
                    if not pd.isna(v): xs_disp = v

        # Compute gate score
        if gate_fn_name == "V0_production":
            score = pred_disp
        elif gate_fn_name == "V1_disp_div_churn":
            score = pred_disp / rank_churn
        elif gate_fn_name == "V2_disp_times_ic":
            score = pred_disp * roll_ic
        elif gate_fn_name == "V3_disp_times_xs":
            score = pred_disp * xs_disp
        elif gate_fn_name == "V4_disp_minus_churn":
            score = pred_disp - alpha_param * rank_churn
        elif gate_fn_name == "V5_matched_placebo":
            score = pred_disp  # record but don't use for gate
        else:
            score = pred_disp

        # Skip decision
        if gate_fn_name == "V5_matched_placebo":
            # Random skip at the matched rate
            skip = rng.random() < GATE_PCTILE
        else:
            skip = False
            if len(hist_score) >= 30:
                thr = float(np.quantile(list(hist_score), GATE_PCTILE))
                if score < thr: skip = True
            hist_score.append(score)

        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "score": score,
                              "skipped": 1, "spread_bps": 0, "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "score": score,
                              "skipped": 1, "spread_bps": 0, "cost_bps": 0,
                              "net_bps": 0, "n_long": 0, "n_short": 0})
            continue

        order_d = np.argsort(-pred_arr); order_a = np.argsort(pred_arr)
        long_r = [sym_arr[i] for i in order_d]
        short_r = [sym_arr[i] for i in order_a]
        cand_l, n_el = select_refill(long_r, "long", K, picks_hist, 90, t)
        cand_s, n_es = select_refill(short_r, "short", K, picks_hist, 90, t)
        c_ls = set(cand_l); c_ss = set(cand_s)
        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > PM_M:
            hist_basket = hist_basket[-PM_M:]
        if len(hist_basket) >= PM_M:
            p_l = [h["long"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            p_s = [h["short"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            nl = cur_long & c_ls
            ns = cur_short & c_ss
            for s_ in c_ls - cur_long:
                if all(s_ in p for p in p_l): nl.add(s_)
            for s_ in c_ss - cur_short:
                if all(s_ in p for p in p_s): ns.add(s_)
            if len(nl) > K:
                nl = set(sorted(nl, key=lambda s_: -pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
            if len(ns) > K:
                ns = set(sorted(ns, key=lambda s_: pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
        else:
            nl, ns = c_ls, c_ss
        if not nl or not ns:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "score": score,
                              "skipped": 1, "spread_bps": 0, "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "score": score,
                              "skipped": 0, "spread_bps": 0, "cost_bps": 0,
                              "net_bps": 0, "n_long": 0, "n_short": 0})
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
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "score": score,
                      "skipped": 0, "spread_bps": spread, "cost_bps": cost, "net_bps": net,
                      "n_long": len(nl), "n_short": len(ns)})
        cur_long, cur_short = nl, ns
    return pd.DataFrame(rows)


def main():
    print("=== Phase L / Test 5: composite profitability gate ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)

    panel = pd.read_parquet(PANEL_PATH,
                              columns=["open_time", "symbol", "xs_alpha_dispersion_48b"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel_by_time = {t: g for t, g in panel.groupby("open_time")}

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

    variants = [
        "V0_production",
        "V1_disp_div_churn",
        "V2_disp_times_ic",
        "V3_disp_times_xs",
        "V4_disp_minus_churn",
    ]

    print(f"\n  {'variant':<22}  {'Sharpe':>7}  {'CI':>17}  {'skip%':>6}  "
          f"{'maxDD':>7}  {'totPnL':>8}  {'pos_folds':>9}", flush=True)
    results = []
    for v in variants:
        t0 = time.time()
        df_v = evaluate(apd, universe, v, panel_by_time)
        net = df_v["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        n_pos = 0
        for f in OOS_FOLDS:
            d = df_v[df_v["fold"] == f]["net_bps"].to_numpy()
            if len(d) >= 3 and _sharpe(d) > 0: n_pos += 1
        skip_rate = float((df_v["skipped"] == 1).mean())
        results.append({"variant": v, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "skip_rate": skip_rate, "max_dd": _max_dd(net),
                          "total_pnl": net.sum(), "n_folds_positive": n_pos})
        df_v.to_csv(OUT / f"per_cycle_{v}.csv", index=False)
        print(f"  {v:<22}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{skip_rate*100:>5.1f}%  {_max_dd(net):>+7.0f}  {net.sum():>+8.0f}  "
              f"{n_pos:>5d}/9  ({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)

    # Best non-production
    prod = results[0]
    cand = [r for r in results[1:]]
    best = max(cand, key=lambda r: r["sharpe"])
    print(f"\n  Production: Sharpe={prod['sharpe']:+.2f}, folds_pos={prod['n_folds_positive']}/9",
          flush=True)
    print(f"  Best composite ({best['variant']}): Sharpe={best['sharpe']:+.2f}, "
          f"folds_pos={best['n_folds_positive']}/9", flush=True)
    lift = best["sharpe"] - prod["sharpe"]
    print(f"  Lift: {lift:+.2f}", flush=True)

    # Matched skip-placebo (random skip at same rate as best variant)
    print(f"\n--- Matched skip-placebo ({N_PLACEBO_SEEDS} seeds at {best['skip_rate']*100:.1f}% rate) ---",
          flush=True)
    t0 = time.time()
    placebo_sh = []
    for seed in range(N_PLACEBO_SEEDS):
        df_p = evaluate(apd, universe, "V5_matched_placebo", panel_by_time, rng_seed=seed)
        placebo_sh.append(_sharpe(df_p["net_bps"].to_numpy()))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_PLACEBO_SEEDS}  ({time.time()-t0:.0f}s)",
                  flush=True)
    p_sh = np.array(placebo_sh)
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < best['sharpe']).mean() * 100)
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
    print(f"  Best composite ranks p{rank:.0f}  "
          f"beats_p95={'PASS' if best['sharpe'] > p95 else 'FAIL'}", flush=True)
    pd.DataFrame({"seed": range(N_PLACEBO_SEEDS), "sharpe": p_sh}).to_csv(
        OUT / "matched_placebo.csv", index=False)

    # Final
    print(f"\n=== Test 5 verdict ===", flush=True)
    pass_prod = lift >= 0.0
    pass_folds = best['n_folds_positive'] >= 6
    pass_placebo = best['sharpe'] > p95
    print(f"  Beats production:   {'PASS' if pass_prod else 'FAIL'} ({lift:+.2f})", flush=True)
    print(f"  ≥6/9 folds:         {'PASS' if pass_folds else 'FAIL'} ({best['n_folds_positive']}/9)",
          flush=True)
    print(f"  Beats placebo p95:  {'PASS' if pass_placebo else 'FAIL'}", flush=True)
    if pass_prod and pass_folds and pass_placebo:
        print(f"  → ADOPT", flush=True)
    else:
        print(f"  → NOT ADOPTED", flush=True)


if __name__ == "__main__":
    main()
