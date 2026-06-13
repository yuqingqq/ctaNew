"""Phase U protocol: K=3 evaluation on bps-target predictions with
magnitude-aware gate (only trade when predicted spread > cost threshold).

Variants:
  V0 baseline_K3_z_target     — production: z-target model + conv_gate
  V1 bps_K3_no_gate           — bps target, no gate, just K=3 selection
  V2 bps_K3_conv_gate         — bps target + conv_gate (same logic as prod)
  V3 bps_K3_mag_gate_9bps     — bps target + magnitude gate: trade if pred_spread > 9 bps
  V4 bps_K3_mag_gate_18bps    — bps target + magnitude gate: pred_spread > 18 bps
  V5 bps_K3_mag_gate_27bps    — bps target + magnitude gate: pred_spread > 27 bps

Each compared with matched skip-placebo.
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

OUT = REPO / "outputs/vBTC_bps_protocol"
OUT.mkdir(parents=True, exist_ok=True)
APD_BPS_PATH = REPO / "outputs/vBTC_audit_bps_target/all_predictions.parquet"
APD_Z_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"  # reference baseline
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
GATE_PCTILE = 0.30
PM_M = 2
K = 3
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


def evaluate(apd, universe, gate_kind, mag_threshold_bps=0.0, K_target=K,
              placebo_seed=None, matched_skip_rate=None):
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
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                          "pred_spread": np.nan, "net_bps": 0,
                          "n_long": 0, "n_short": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        idx_t = np.argpartition(-pred_arr, K_target - 1)[:K_target]
        idx_b = np.argpartition(pred_arr, K_target - 1)[:K_target]
        # pred_spread in same units as pred (bps if bps-target, z-score if prod)
        pred_spread = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())

        # Skip decision based on gate_kind
        if matched_skip_rate is not None:
            skip = rng.random() < matched_skip_rate
            hist_disp.append(pred_spread)
        elif gate_kind == "no_gate":
            skip = False
            hist_disp.append(pred_spread)
        elif gate_kind == "conv_gate":
            skip = False
            if len(hist_disp) >= 30:
                thr = float(np.quantile(list(hist_disp), GATE_PCTILE))
                if pred_spread < thr: skip = True
            hist_disp.append(pred_spread)
        elif gate_kind == "mag_gate":
            # Trade only if predicted spread > threshold (absolute bps)
            skip = pred_spread < mag_threshold_bps
            hist_disp.append(pred_spread)
        else:
            skip = False

        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "pred_spread": pred_spread, "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "pred_spread": pred_spread, "net_bps": 0,
                              "n_long": 0, "n_short": 0})
            continue
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
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "pred_spread": pred_spread, "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                              "pred_spread": pred_spread, "net_bps": 0,
                              "n_long": 0, "n_short": 0})
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
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                      "pred_spread": pred_spread, "spread_bps": spread,
                      "cost_bps": cost, "net_bps": net,
                      "n_long": len(nl), "n_short": len(ns)})
        cur_long, cur_short = nl, ns
    return pd.DataFrame(rows)


def fold_concentration(df_v):
    fold_pnls = df_v.groupby("fold")["net_bps"].sum()
    pos = fold_pnls[fold_pnls > 0]
    total_pos = pos.sum() if len(pos) > 0 else 0
    if total_pos <= 0: return 0.0
    return float(pos.max() / total_pos)


def main():
    print("=== Phase U protocol: bps-target + magnitude gate ===\n", flush=True)
    apd_bps = pd.read_parquet(APD_BPS_PATH)
    apd_bps["open_time"] = pd.to_datetime(apd_bps["open_time"], utc=True)
    apd_bps["exit_time"] = pd.to_datetime(apd_bps["exit_time"], utc=True)
    apd_z = pd.read_parquet(APD_Z_PATH)
    apd_z["open_time"] = pd.to_datetime(apd_z["open_time"], utc=True)
    apd_z["exit_time"] = pd.to_datetime(apd_z["exit_time"], utc=True)
    print(f"  bps-target apd: {len(apd_bps):,} rows; pred std={apd_bps['pred'].std():.2f} bps",
          flush=True)
    print(f"  z-target apd:   {len(apd_z):,} rows; pred std={apd_z['pred'].std():.4f}",
          flush=True)

    # Inspect predicted spread distribution under bps target
    df_t = apd_bps[apd_bps["fold"].isin(OOS_FOLDS)].copy()
    times = sorted(df_t["open_time"].unique())[::HORIZON]
    df_sub = df_t[df_t["open_time"].isin(times)]
    spreads = []
    for t, g in df_sub.groupby("open_time"):
        if len(g) >= 2 * K + 1:
            p = g["pred"].values
            it = np.argpartition(-p, K-1)[:K]
            ib = np.argpartition(p, K-1)[:K]
            spreads.append(p[it].mean() - p[ib].mean())
    sp = np.array(spreads)
    print(f"\n  Predicted spread distribution (bps target, K={K}):", flush=True)
    print(f"    n={len(sp)} cycles", flush=True)
    print(f"    mean: {sp.mean():+.2f} bps", flush=True)
    print(f"    p25:  {np.quantile(sp, 0.25):+.2f} bps", flush=True)
    print(f"    p50:  {np.median(sp):+.2f} bps", flush=True)
    print(f"    p75:  {np.quantile(sp, 0.75):+.2f} bps", flush=True)
    print(f"    p95:  {np.quantile(sp, 0.95):+.2f} bps", flush=True)
    pct_above_9 = (sp > 9).mean()
    pct_above_18 = (sp > 18).mean()
    pct_above_27 = (sp > 27).mean()
    print(f"    %>9bps: {pct_above_9:.1%}", flush=True)
    print(f"    %>18bps: {pct_above_18:.1%}", flush=True)
    print(f"    %>27bps: {pct_above_27:.1%}", flush=True)

    listings = get_listings()
    panel_syms = set(apd_bps["symbol"].unique())
    def eligibility_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd_bps[apd_bps["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON]

    # Build universes using each apd's pred (rolling-IC selection)
    print(f"\n  Building universes...", flush=True)
    universe_bps = build_rolling_ic_universe(apd_bps, sampled_t, TOP_N, eligibility_at)
    universe_z = build_rolling_ic_universe(apd_z, sampled_t, TOP_N, eligibility_at)

    variants = [
        ("V0_z_target_conv_gate", apd_z, universe_z, "conv_gate", 0.0),
        ("V1_bps_no_gate", apd_bps, universe_bps, "no_gate", 0.0),
        ("V2_bps_conv_gate", apd_bps, universe_bps, "conv_gate", 0.0),
        ("V3_bps_mag_9bps", apd_bps, universe_bps, "mag_gate", 9.0),
        ("V4_bps_mag_18bps", apd_bps, universe_bps, "mag_gate", 18.0),
        ("V5_bps_mag_27bps", apd_bps, universe_bps, "mag_gate", 27.0),
    ]

    print(f"\n  {'variant':<26}  {'Sharpe':>7}  {'CI':>17}  {'skip%':>6}  "
          f"{'maxDD':>7}  {'totPnL':>8}  {'pos_folds':>9}  {'conc':>4}", flush=True)
    results = {}
    for label, apd_use, u_use, gate, mag in variants:
        t0 = time.time()
        df_v = evaluate(apd_use, u_use, gate, mag)
        net = df_v["net_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
        skip_rate = float((df_v["skipped"] == 1).mean())
        n_pos = 0
        for f in OOS_FOLDS:
            d = df_v[df_v["fold"] == f]["net_bps"].to_numpy()
            if len(d) >= 3 and _sharpe(d) > 0: n_pos += 1
        conc = fold_concentration(df_v)
        results[label] = {"sharpe": sh, "ci_lo": lo, "ci_hi": hi,
                            "skip_rate": skip_rate, "max_dd": _max_dd(net),
                            "total_pnl": net.sum(), "n_folds_positive": n_pos,
                            "concentration": conc, "apd": apd_use, "u": u_use,
                            "gate": gate, "mag": mag}
        df_v.to_csv(OUT / f"per_cycle_{label}.csv", index=False)
        print(f"  {label:<26}  {sh:>+7.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  "
              f"{skip_rate*100:>5.1f}%  {_max_dd(net):>+7.0f}  {net.sum():>+8.0f}  "
              f"{n_pos:>5d}/9  {conc*100:>3.0f}%  ({time.time()-t0:.0f}s)", flush=True)

    # Best non-V0 (against z-target baseline)
    prod = results["V0_z_target_conv_gate"]
    cand = {k: v for k, v in results.items() if k != "V0_z_target_conv_gate"}
    best_name = max(cand, key=lambda k: cand[k]["sharpe"])
    best = cand[best_name]
    print(f"\n  Production (V0 z-target): Sharpe={prod['sharpe']:+.2f}, "
          f"folds={prod['n_folds_positive']}/9", flush=True)
    print(f"  Best bps-target ({best_name}): Sharpe={best['sharpe']:+.2f}, "
          f"folds={best['n_folds_positive']}/9, conc={best['concentration']*100:.0f}%",
          flush=True)
    lift = best['sharpe'] - prod['sharpe']
    print(f"  Lift: {lift:+.2f}", flush=True)

    # Matched skip-placebo on best
    print(f"\n--- Matched skip-placebo on {best_name} ({N_PLACEBO_SEEDS} seeds, "
          f"skip {best['skip_rate']*100:.1f}%) ---", flush=True)
    t0 = time.time()
    placebo_sh = []
    for seed in range(N_PLACEBO_SEEDS):
        df_p = evaluate(best["apd"], best["u"], "no_gate", K_target=K,
                          placebo_seed=seed, matched_skip_rate=best["skip_rate"])
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

    print(f"\n=== Phase U verdict ===", flush=True)
    pass_lift = lift >= 0.3
    pass_folds = best['n_folds_positive'] >= 6
    pass_conc = best['concentration'] <= 0.40
    pass_placebo = best['sharpe'] > p95
    print(f"  Lift ≥ +0.30:           {'PASS' if pass_lift else 'FAIL'} ({lift:+.2f})",
          flush=True)
    print(f"  ≥6/9 folds positive:    {'PASS' if pass_folds else 'FAIL'} "
          f"({best['n_folds_positive']}/9)", flush=True)
    print(f"  Concentration ≤ 40%:    {'PASS' if pass_conc else 'FAIL'} "
          f"({best['concentration']*100:.0f}%)", flush=True)
    print(f"  Beats placebo p95:      {'PASS' if pass_placebo else 'FAIL'}",
          flush=True)
    if pass_lift and pass_folds and pass_conc and pass_placebo:
        print(f"  → ADOPT {best_name}", flush=True)
    else:
        print(f"  → NOT ADOPTED", flush=True)


if __name__ == "__main__":
    main()
