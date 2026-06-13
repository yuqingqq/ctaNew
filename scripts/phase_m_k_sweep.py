"""Phase M: K-sweep honest validation.

Re-tests K ∈ {2, 3, 4, 5, 6} on the current production stack with matched
basket-size placebo per K-value. The pre-timing-audit Phase 6 result said
K=4 won, but that was pre-audit and pre-placebo discipline. This is the
honest re-validation.

For each K, the protocol is unchanged (filter_refill + PM + conv_gate) except
K_target = K. Matched placebo uses K random picks from the N=15 universe.
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

OUT = REPO / "outputs/vBTC_k_sweep"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
MIN_PICKS_FOR_FILTER = 30
MIN_OBS_PER_SYM = 100
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
TOP_N = 15
N_PLACEBO_SEEDS = 100

K_SWEEP = [2, 3, 4, 5, 6]


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


def evaluate(apd, universe, K_target, placebo_seed=None):
    """Run protocol with K_target picks per side. If placebo_seed is set,
    pick K_target random names per side from universe (matched-basket placebo)."""
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
                rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        if placebo_seed is not None:
            # Random selection: pick K_target longs and K_target shorts randomly from universe
            shuffled = rng.permutation(len(sym_arr))
            cand_l = sym_arr[shuffled[:K_target]].tolist()
            cand_s = sym_arr[shuffled[K_target:2 * K_target]].tolist()
            n_el = 0; n_es = 0
        else:
            order_d = np.argsort(-pred_arr); order_a = np.argsort(pred_arr)
            long_r = [sym_arr[i] for i in order_d]
            short_r = [sym_arr[i] for i in order_a]
            cand_l, n_el = select_refill(long_r, "long", K_target, picks_hist, 90, t)
            cand_s, n_es = select_refill(short_r, "short", K_target, picks_hist, 90, t)
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


def main():
    print("=== Phase M: K-sweep honest validation ===\n", flush=True)
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

    print(f"\n--- (1) Real K-sweep ---", flush=True)
    print(f"  {'K':>3}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  {'totPnL':>8}  "
          f"{'avg_L/S':>8}  {'pos_folds':>9}  per-fold", flush=True)
    real = {}
    for K in K_SWEEP:
        t0 = time.time()
        df_v = evaluate(apd, universe, K)
        net = df_v["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        n_pos = 0
        per_fold = []
        for f in OOS_FOLDS:
            d = df_v[df_v["fold"] == f]["net_bps"].to_numpy()
            if len(d) >= 3:
                sh_f = _sharpe(d)
                per_fold.append(sh_f)
                if sh_f > 0: n_pos += 1
        real[K] = {"sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                    "max_dd": _max_dd(net), "total_pnl": net.sum(),
                    "avg_L": float(df_v["n_long"].mean()),
                    "avg_S": float(df_v["n_short"].mean()),
                    "n_folds_positive": n_pos, "df": df_v}
        df_v.to_csv(OUT / f"per_cycle_K{K}.csv", index=False)
        pf_str = " ".join(f"{x:+.1f}" for x in per_fold)
        print(f"  {K:>3}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {_max_dd(net):>+7.0f}  "
              f"{net.sum():>+8.0f}  {real[K]['avg_L']:.1f}/{real[K]['avg_S']:.1f}  "
              f"{n_pos:>5d}/9  {pf_str}  ({time.time()-t0:.0f}s)", flush=True)

    # Matched basket-size placebo per K
    print(f"\n--- (2) Matched basket-size placebo ({N_PLACEBO_SEEDS} seeds per K) ---",
          flush=True)
    print(f"  {'K':>3}  {'real_Sh':>7}  {'placebo_p50':>12}  {'placebo_p95':>12}  "
          f"{'rank':>6}  {'beats_p95':>10}", flush=True)
    placebo_summary = []
    for K in K_SWEEP:
        t0 = time.time()
        ps = []
        for seed in range(N_PLACEBO_SEEDS):
            df_p = evaluate(apd, universe, K, placebo_seed=seed)
            ps.append(_sharpe(df_p["net_bps"].to_numpy()))
        ps_arr = np.array(ps)
        p95 = float(np.percentile(ps_arr, 95))
        p50 = float(np.percentile(ps_arr, 50))
        rank = float((ps_arr < real[K]["sharpe"]).mean() * 100)
        placebo_summary.append({"K": K, "real_sharpe": real[K]["sharpe"],
                                  "placebo_mean": float(ps_arr.mean()),
                                  "placebo_p50": p50, "placebo_p95": p95,
                                  "rank_vs_placebo": rank,
                                  "beats_p95": real[K]["sharpe"] > p95})
        print(f"  {K:>3}  {real[K]['sharpe']:>+7.2f}  {p50:>+12.2f}  {p95:>+12.2f}  "
              f"p{rank:>3.0f}  {'PASS' if real[K]['sharpe'] > p95 else 'FAIL':>10}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    pd.DataFrame(placebo_summary).to_csv(OUT / "placebo_summary.csv", index=False)
    pd.DataFrame([{**{k: v for k, v in r.items() if k != "df"}, "K": K}
                    for K, r in real.items()]).to_csv(OUT / "real_summary.csv", index=False)

    # Verdict
    print(f"\n=== Phase M verdict ===", flush=True)
    best_K = max(K_SWEEP, key=lambda k: real[k]["sharpe"])
    pass_p95_Ks = [r["K"] for r in placebo_summary if r["beats_p95"]]
    print(f"  Best Sharpe K: K={best_K} (Sharpe={real[best_K]['sharpe']:+.2f})", flush=True)
    print(f"  K values beating placebo p95: {pass_p95_Ks if pass_p95_Ks else 'NONE'}", flush=True)
    print(f"  Current production: K=4 (Sharpe={real[4]['sharpe']:+.2f})", flush=True)
    if best_K != 4 and best_K in pass_p95_Ks:
        print(f"  → Consider switching to K={best_K}", flush=True)
    else:
        print(f"  → K=4 remains best honest choice (or all equivalent in noise)", flush=True)
    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
