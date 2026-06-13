"""Phase S2: alternative universe selectors on 51-panel with K=3 (Phase M winner).

After correlation diagnostic identified s5/s4/s7/s6 as the non-redundant
selectors (s2/s3 ≈ raw IC, s8 ≈ s7 under 60d eligibility), test these four
selectors as universe builders.

For each selector at each 90d refresh boundary:
  - Compute per-symbol score from past 180d data
  - Pick top-15 by score
  - Run protocol (K=3, filter_refill, PM, conv_gate)

Pass condition: Sharpe ≥ B1 (+1.98) + 0.30, ≥6/9 folds positive,
not concentrated in ≤2 folds (max single-fold contribution < 40% of total
positive PnL), beats matched basket-size placebo p95.
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

OUT = REPO / "outputs/vBTC_selector_test"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5
COST_HIT_THRESHOLD_BPS = 9.0
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
K = 3  # Phase M winner
MIN_PICKS_FOR_FILTER = 30
MIN_OBS_PER_SYM = 100
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
TOP_N = 15
SHRINK_LAMBDA = 20.0
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


def compute_score_for_symbol(gg, selector):
    """Return score for one symbol given its past (pred, alpha_A, return_pct) data."""
    if len(gg) < MIN_OBS_PER_SYM: return None
    gg = gg.dropna(subset=["pred", "alpha_A", "return_pct"])
    if len(gg) < MIN_OBS_PER_SYM: return None
    ic = gg["pred"].rank().corr(gg["alpha_A"].rank())
    if pd.isna(ic): return None
    n = len(gg)
    se = np.sqrt((1 - ic ** 2) / max(n - 2, 1))

    pred_z = gg["pred"] - gg["pred"].mean()
    signed_alpha = np.sign(pred_z) * gg["alpha_A"]
    past_dir_sharpe = float(signed_alpha.mean() / max(signed_alpha.std(), 1e-6))
    signed_ret_bps = np.sign(pred_z) * gg["return_pct"] * 1e4
    hit_rate = float((signed_ret_bps > COST_HIT_THRESHOLD_BPS).mean())
    pred_scale = float(gg["pred"].abs().mean())
    pred_vol = float(gg["pred"].std())
    rank_churn = pred_vol / max(pred_scale, 1e-6)

    return {"ic": ic, "se": se, "past_dir_sharpe": past_dir_sharpe,
            "hit_rate": hit_rate, "rank_churn": rank_churn, "n": n}


def build_universe(apd, sampled_t, eligibility_at_t, selector):
    """Build per-cycle universe using selected score."""
    bar_ms = 5 * 60 * 1000
    window_ms = IC_WINDOW_DAYS * 288 * bar_ms
    update_ms = IC_UPDATE_DAYS * 288 * bar_ms
    df = apd.copy()
    df["t_int"] = to_ms_int(df["open_time"])
    df["exit_t_int"] = to_ms_int(df["exit_time"])
    df_clean = df.dropna(subset=["alpha_A"])
    t0_ms = int(pd.Timestamp(sampled_t[0]).timestamp() * 1000)
    boundaries = []
    seen = set()
    for t in sampled_t:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        n = (t_ms - t0_ms) // update_ms
        b_ms = t0_ms + n * update_ms
        boundaries.append((t, b_ms))
        seen.add(b_ms)
    unique_b = sorted(seen)
    b2u = {}
    for b_ms in unique_b:
        elig = eligibility_at_t(b_ms)
        past = df_clean[(df_clean["t_int"] >= b_ms - window_ms) &
                          (df_clean["t_int"] < b_ms) &
                          (df_clean["exit_t_int"] <= b_ms) &
                          (df_clean["symbol"].isin(elig))]
        if len(past) < 1000:
            b2u[b_ms] = set(); continue
        sym_scores = {}
        for sym, gg in past.groupby("symbol"):
            r = compute_score_for_symbol(gg, selector)
            if r is None: continue
            sym_scores[sym] = r
        if not sym_scores:
            b2u[b_ms] = set(); continue
        # z-score normalize the components within boundary
        ics = np.array([s["ic"] for s in sym_scores.values()])
        shrunk_ics = np.array([s["ic"] - SHRINK_LAMBDA * s["se"] for s in sym_scores.values()])
        past_sharpes = np.array([s["past_dir_sharpe"] for s in sym_scores.values()])
        hit_rates = np.array([s["hit_rate"] for s in sym_scores.values()])
        rank_churns = np.array([s["rank_churn"] for s in sym_scores.values()])
        syms = list(sym_scores.keys())
        z_shrunk = (shrunk_ics - shrunk_ics.mean()) / max(shrunk_ics.std(), 1e-6)
        z_past_sh = (past_sharpes - past_sharpes.mean()) / max(past_sharpes.std(), 1e-6)
        z_churn = (rank_churns - rank_churns.mean()) / max(rank_churns.std(), 1e-6)

        if selector == "s1_raw_IC":
            scores = ics
        elif selector == "s4_past_dir_sharpe":
            scores = past_sharpes
        elif selector == "s5_hit_rate":
            scores = hit_rates
        elif selector == "s6_hybrid":
            scores = z_shrunk + z_past_sh
        elif selector == "s7_stable_hybrid":
            scores = z_shrunk + z_past_sh - z_churn
        else:
            scores = ics
        order = np.argsort(-scores)
        top = set([syms[i] for i in order[:TOP_N]])
        b2u[b_ms] = top
    return {t: b2u[b_ms] for t, b_ms in boundaries}


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


def evaluate(apd, universe, K_target=K, placebo_seed=None):
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
    rng = np.random.RandomState(placebo_seed if placebo_seed is not None else 0)
    rows = []
    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K_target + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "net_bps": 0,
                          "n_long": 0, "n_short": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        idx_t = np.argpartition(-pred_arr, K_target - 1)[:K_target]
        idx_b = np.argpartition(pred_arr, K_target - 1)[:K_target]
        disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
        skip = False
        if len(hist_disp) >= 30:
            thr = float(np.quantile(list(hist_disp), GATE_PCTILE))
            if disp < thr: skip = True
        hist_disp.append(disp)
        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "net_bps": 0,
                              "n_long": 0, "n_short": 0})
            continue
        if placebo_seed is not None:
            shuffled = rng.permutation(len(sym_arr))
            cand_l = sym_arr[shuffled[:K_target]].tolist()
            cand_s = sym_arr[shuffled[K_target:2*K_target]].tolist()
        else:
            order_d = np.argsort(-pred_arr); order_a = np.argsort(pred_arr)
            long_r = [sym_arr[i] for i in order_d]
            short_r = [sym_arr[i] for i in order_a]
            cand_l, _ = select_refill(long_r, "long", K_target, picks_hist, 90, t)
            cand_s, _ = select_refill(short_r, "short", K_target, picks_hist, 90, t)
        c_ls = set(cand_l); c_ss = set(cand_s)
        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > PM_M:
            hist_basket = hist_basket[-PM_M:]
        if len(hist_basket) >= PM_M:
            p_l = [h["long"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            p_s = [h["short"] for h in hist_basket[-PM_M:][:PM_M - 1]]
            nl = cur_long & c_ls; ns = cur_short & c_ss
            for s_ in c_ls - cur_long:
                if all(s_ in p for p in p_l): nl.add(s_)
            for s_ in c_ss - cur_short:
                if all(s_ in p for p in p_s): ns.add(s_)
            if len(nl) > K_target:
                nl = set(sorted(nl, key=lambda s_: -pred_arr[np.where(sym_arr == s_)[0][0]])[:K_target])
            if len(ns) > K_target:
                ns = set(sorted(ns, key=lambda s_: pred_arr[np.where(sym_arr == s_)[0][0]])[:K_target])
        else:
            nl, ns = c_ls, c_ss
        if not nl or not ns:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
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
        if placebo_seed is None:
            for s_ in nl:
                picks_hist[(s_, "long")].append((t, exit_l[s_], ret_l[s_] * 1e4 / len(nl)))
            for s_ in ns:
                picks_hist[(s_, "short")].append((t, exit_l[s_], -ret_l[s_] * 1e4 / len(ns)))
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "net_bps": net,
                      "n_long": len(nl), "n_short": len(ns)})
        cur_long, cur_short = nl, ns
    return pd.DataFrame(rows)


def fold_concentration(df_v):
    """Return max single-fold positive PnL contribution as fraction of total positive."""
    fold_pnls = df_v.groupby("fold")["net_bps"].sum()
    pos = fold_pnls[fold_pnls > 0]
    total_pos = pos.sum() if len(pos) > 0 else 0
    if total_pos <= 0: return 0.0
    return float(pos.max() / total_pos)


def main():
    print("=== Phase S2: alternative selectors on 51-panel K=3 ===\n", flush=True)
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

    selectors = ["s1_raw_IC", "s4_past_dir_sharpe", "s5_hit_rate",
                  "s6_hybrid", "s7_stable_hybrid"]

    print(f"  {'selector':<22}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  {'totPnL':>8}  "
          f"{'pos_folds':>9}  {'max_concentration':>17}", flush=True)
    results = {}
    for sel in selectors:
        t0 = time.time()
        u = build_universe(apd, sampled_t, eligibility_at, sel)
        df_v = evaluate(apd, u, K)
        net = df_v["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        n_pos = 0
        for f in OOS_FOLDS:
            d = df_v[df_v["fold"] == f]["net_bps"].to_numpy()
            if len(d) >= 3 and _sharpe(d) > 0: n_pos += 1
        conc = fold_concentration(df_v)
        results[sel] = {"u": u, "df": df_v, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                          "max_dd": _max_dd(net), "total_pnl": net.sum(),
                          "n_folds_positive": n_pos, "concentration": conc}
        df_v.to_csv(OUT / f"per_cycle_{sel}.csv", index=False)
        print(f"  {sel:<22}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{_max_dd(net):>+7.0f}  {net.sum():>+8.0f}  {n_pos:>5d}/9   "
              f"{conc*100:>15.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    # Best non-baseline
    baseline = results["s1_raw_IC"]
    candidates = {k: v for k, v in results.items() if k != "s1_raw_IC"}
    best_name = max(candidates, key=lambda k: candidates[k]["sharpe"])
    best = candidates[best_name]
    print(f"\n  Baseline (s1_raw_IC):  Sharpe={baseline['sharpe']:+.2f}, "
          f"folds={baseline['n_folds_positive']}/9, conc={baseline['concentration']*100:.0f}%",
          flush=True)
    print(f"  Best candidate ({best_name}):  Sharpe={best['sharpe']:+.2f}, "
          f"folds={best['n_folds_positive']}/9, conc={best['concentration']*100:.0f}%",
          flush=True)
    lift = best["sharpe"] - baseline["sharpe"]
    print(f"  Lift: {lift:+.2f}", flush=True)

    if lift > 0.2:
        # Run matched-basket-size placebo on best
        print(f"\n--- Matched basket-size placebo on {best_name} ({N_PLACEBO_SEEDS} seeds) ---",
              flush=True)
        t0 = time.time()
        placebo_sh = []
        for seed in range(N_PLACEBO_SEEDS):
            df_p = evaluate(apd, best["u"], K, placebo_seed=seed)
            placebo_sh.append(_sharpe(df_p["net_bps"].to_numpy()))
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
        pd.DataFrame({"seed": range(N_PLACEBO_SEEDS), "sharpe": p_sh}).to_csv(
            OUT / f"matched_placebo_{best_name}.csv", index=False)

    print(f"\n=== Phase S2 verdict ===", flush=True)
    pass_lift = lift >= 0.3
    pass_folds = best["n_folds_positive"] >= 6
    pass_concentration = best["concentration"] <= 0.40
    print(f"  Lift ≥ +0.30:                 {'PASS' if pass_lift else 'FAIL'} ({lift:+.2f})",
          flush=True)
    print(f"  ≥6/9 folds positive:           {'PASS' if pass_folds else 'FAIL'} "
          f"({best['n_folds_positive']}/9)", flush=True)
    print(f"  Concentration ≤ 40%:           {'PASS' if pass_concentration else 'FAIL'} "
          f"({best['concentration']*100:.0f}%)", flush=True)
    if pass_lift and pass_folds and pass_concentration:
        print(f"  → CANDIDATE for nested validation: {best_name}", flush=True)
    else:
        print(f"  → REJECT (need nested + placebo passes)", flush=True)


if __name__ == "__main__":
    main()
