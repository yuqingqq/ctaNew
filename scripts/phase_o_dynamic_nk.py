"""Phase O: dynamic (N, K) selection on 111-symbol expanded panel.

Combines:
  - Phase E5b's 111-panel + $10M volume PIT eligibility
  - Phase M's K-sweep finding that K=3 may be better than K=4
  - Phase K3's nested-fold parameter-selection discipline

Grid: N ∈ {15, 25, 35, all_eligible} × K ∈ {2, 3, 4, 5, 6} = 20 variants.

For each (N, K), run Phase 2b v3 protocol with $10M volume PIT and matched
basket-size placebo.

Then nested-fold selection: at fold f, pick best (N, K) from cumulative
Sharpe on folds < f. Apply to fold f. Default for early folds: (N=15, K=4).
Stitch fold-by-fold predictions into nested-OOS curve. Matched-basket-size
placebo on the nested curve.

Pass conditions for nested-OOS to ADOPT:
  - Nested Sharpe > 51-panel K=3 honest +1.98
  - Beats matched basket-size placebo p95
  - ≥6/9 folds positive

Output: outputs/vBTC_dynamic_nk_111/
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

OUT = REPO / "outputs/vBTC_dynamic_nk_111"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel_expanded/all_predictions.parquet"
VOL_TABLE_PATH = REPO / "outputs/vBTC_features_expanded/volume_pit_table.parquet"
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
MIN_VOLUME_USD = 10_000_000
N_PLACEBO_SEEDS = 100

N_SWEEP = [15, 25, 35, None]  # None = all eligible
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
            shuffled = rng.permutation(len(sym_arr))
            cand_l = sym_arr[shuffled[:K_target]].tolist()
            cand_s = sym_arr[shuffled[K_target:2 * K_target]].tolist()
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


def main():
    print("=== Phase O: dynamic (N, K) on 111-panel ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    print(f"  111-panel apd: {len(apd):,} rows, {apd.symbol.nunique()} syms", flush=True)

    vol = pd.read_parquet(VOL_TABLE_PATH)
    vol["date"] = pd.to_datetime(vol["date"]).dt.date
    vol_dict = vol.set_index(["symbol", "date"])["trailing_30d_median_qvol"].to_dict()

    listings = get_listings()
    panel_syms = set(apd["symbol"].unique())

    def eligibility_at(b):
        if isinstance(b, (int, np.integer)):
            ts = pd.Timestamp(b, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(b)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        vol_date = (ts - pd.Timedelta(days=1)).date()
        eligible = set()
        for s in panel_syms:
            if not listings.get(s) or listings[s] > cutoff: continue
            v = vol_dict.get((s, vol_date), 0)
            if v < MIN_VOLUME_USD: continue
            eligible.add(s)
        return eligible

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON]

    # Build universes for each N
    universes = {}
    for N in N_SWEEP:
        n_label = "all" if N is None else str(N)
        t0 = time.time()
        u = build_rolling_ic_universe(apd, sampled_t, N, eligibility_at)
        avg_sz = float(np.mean([len(v) for v in u.values()]))
        universes[N] = u
        print(f"  N={n_label}: built (avg size {avg_sz:.1f}, {time.time()-t0:.0f}s)",
              flush=True)

    # Run all (N, K) combinations
    print(f"\n--- (1) (N, K) grid ---", flush=True)
    print(f"  {'(N, K)':<12}  {'Sharpe':>7}  {'CI':>17}  {'maxDD':>7}  {'totPnL':>8}  "
          f"{'avg_L/S':>8}  {'pos_folds':>9}", flush=True)
    grid_results = {}
    for N in N_SWEEP:
        n_label = "all" if N is None else str(N)
        for K_t in K_SWEEP:
            t0 = time.time()
            df_v = evaluate(apd, universes[N], K_t)
            net = df_v["net_bps"].to_numpy()
            sh = _sharpe(net)
            n_pos = 0
            per_fold = {}
            for f in OOS_FOLDS:
                d = df_v[df_v["fold"] == f]["net_bps"].to_numpy()
                if len(d) >= 3:
                    sh_f = _sharpe(d)
                    per_fold[f] = sh_f
                    if sh_f > 0: n_pos += 1
            grid_results[(N, K_t)] = {
                "df": df_v, "sharpe": sh, "max_dd": _max_dd(net),
                "total_pnl": net.sum(), "avg_L": float(df_v["n_long"].mean()),
                "avg_S": float(df_v["n_short"].mean()),
                "n_folds_positive": n_pos, "per_fold": per_fold
            }
            print(f"  N={n_label:>3},K={K_t}    {sh:>+7.2f}  "
                  f"{_max_dd(net):>+7.0f}  {net.sum():>+8.0f}  "
                  f"{grid_results[(N, K_t)]['avg_L']:.1f}/{grid_results[(N, K_t)]['avg_S']:.1f}  "
                  f"{n_pos:>5d}/9  ({time.time()-t0:.0f}s)", flush=True)

    # Save grid results
    grid_summary = []
    for (N, K_t), r in grid_results.items():
        n_label = "all" if N is None else str(N)
        grid_summary.append({"N": n_label, "K": K_t, "sharpe": r["sharpe"],
                                "max_dd": r["max_dd"], "total_pnl": r["total_pnl"],
                                "avg_L": r["avg_L"], "avg_S": r["avg_S"],
                                "n_folds_positive": r["n_folds_positive"]})
    pd.DataFrame(grid_summary).to_csv(OUT / "grid_summary.csv", index=False)

    # Identify best in-sample variant
    best_global = max(grid_results.items(), key=lambda kv: kv[1]["sharpe"])
    print(f"\n  In-sample best (N, K): {best_global[0]} Sharpe={best_global[1]['sharpe']:+.2f}",
          flush=True)

    # Nested-fold selection
    print(f"\n--- (2) Nested-fold (N, K) selection ---", flush=True)
    default = (15, 4)
    nested_per_cycle = []
    selected = {}
    for f in OOS_FOLDS:
        past_folds = [pf for pf in OOS_FOLDS if pf < f]
        if len(past_folds) < 2:
            chosen = default
        else:
            scores = {}
            for nk, r in grid_results.items():
                d = r["df"][r["df"]["fold"].isin(past_folds)]["net_bps"].to_numpy()
                if len(d) < 10: continue
                scores[nk] = _sharpe(d)
            chosen = max(scores, key=scores.get) if scores else default
        selected[f] = chosen
        df_chosen = grid_results[chosen]["df"]
        nested_per_cycle.append(df_chosen[df_chosen["fold"] == f].copy())
        n_label = "all" if chosen[0] is None else str(chosen[0])
        print(f"  fold {f}: chose (N={n_label}, K={chosen[1]})", flush=True)
    nested_df = pd.concat(nested_per_cycle, ignore_index=True)
    nested_df.to_csv(OUT / "nested_per_cycle.csv", index=False)

    sh_nested = _sharpe(nested_df["net_bps"].to_numpy())
    sh_lo, sh_hi = block_bootstrap_ci(nested_df["net_bps"].to_numpy(),
                                          statistic=_sharpe, block_size=7, n_boot=2000)[1:]
    n_pos_nested = 0
    print(f"\n  Nested-OOS Sharpe: {sh_nested:+.2f} [{sh_lo:+.2f}, {sh_hi:+.2f}]", flush=True)
    print(f"  totPnL: {nested_df['net_bps'].sum():+.0f}", flush=True)
    print(f"  maxDD:  {_max_dd(nested_df['net_bps'].to_numpy()):+.0f}", flush=True)
    print(f"  avg L/S: {nested_df['n_long'].mean():.2f}/{nested_df['n_short'].mean():.2f}",
          flush=True)
    print(f"\n  Per-fold nested:", flush=True)
    for f in OOS_FOLDS:
        d = nested_df[nested_df["fold"] == f]["net_bps"].to_numpy()
        if len(d) < 3: continue
        sh_f = _sharpe(d)
        if sh_f > 0: n_pos_nested += 1
        chose = selected[f]
        n_label = "all" if chose[0] is None else str(chose[0])
        print(f"    fold {f} (chose N={n_label},K={chose[1]}): "
              f"Sharpe={sh_f:+.2f}  pnl={d.sum():+.0f}", flush=True)
    print(f"\n  Folds positive: {n_pos_nested}/9", flush=True)

    # Matched basket-size placebo on the nested curve
    print(f"\n--- (3) Matched basket-size placebo ({N_PLACEBO_SEEDS} seeds) ---", flush=True)
    # For each cycle, randomly pick n_long + n_short symbols from the chosen
    # universe at the matching size.
    target_l = nested_df["n_long"].tolist()
    target_s = nested_df["n_short"].tolist()
    # Build placebo: at each fold, use the same chosen (N, K) universe
    # but random selection
    t0 = time.time()
    placebo_sh = []
    for seed in range(N_PLACEBO_SEEDS):
        per_fold_dfs = []
        for f in OOS_FOLDS:
            chose = selected[f]
            df_p_fold = evaluate(apd, universes[chose[0]], chose[1],
                                    placebo_seed=seed * 10 + f)
            per_fold_dfs.append(df_p_fold[df_p_fold["fold"] == f])
        df_p = pd.concat(per_fold_dfs, ignore_index=True)
        placebo_sh.append(_sharpe(df_p["net_bps"].to_numpy()))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_PLACEBO_SEEDS}  ({time.time()-t0:.0f}s)",
                  flush=True)
    p_sh = np.array(placebo_sh)
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < sh_nested).mean() * 100)
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
    print(f"  Nested-OOS Sharpe={sh_nested:+.2f} ranks p{rank:.0f}  "
          f"beats_p95={'PASS' if sh_nested > p95 else 'FAIL'}", flush=True)
    pd.DataFrame({"seed": range(N_PLACEBO_SEEDS), "sharpe": p_sh}).to_csv(
        OUT / "matched_placebo.csv", index=False)

    # Compare to references
    print(f"\n=== Phase O verdict ===", flush=True)
    print(f"  References:", flush=True)
    print(f"    51-panel K=4 (production):  Sharpe +1.16", flush=True)
    print(f"    51-panel K=3 (Phase M):      Sharpe +1.98", flush=True)
    print(f"    111-panel nested-OOS:        Sharpe {sh_nested:+.2f}", flush=True)
    print(f"    111-panel in-sample best:    Sharpe {best_global[1]['sharpe']:+.2f}",
          flush=True)
    lift_vs_51K3 = sh_nested - 1.98
    pass_lift = lift_vs_51K3 >= 0.3
    pass_folds = n_pos_nested >= 6
    pass_placebo = sh_nested > p95
    print(f"\n  Beats 51-panel K=3 by +0.3:  {'PASS' if pass_lift else 'FAIL'} "
          f"(Δ={lift_vs_51K3:+.2f})", flush=True)
    print(f"  ≥6/9 folds:                   {'PASS' if pass_folds else 'FAIL'} "
          f"({n_pos_nested}/9)", flush=True)
    print(f"  Beats placebo p95:            {'PASS' if pass_placebo else 'FAIL'}",
          flush=True)
    if pass_lift and pass_folds and pass_placebo:
        print(f"  → ADOPT 111-panel dynamic (N, K)", flush=True)
    else:
        print(f"  → KEEP 51-panel K=3 (Phase M winner)", flush=True)
    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
