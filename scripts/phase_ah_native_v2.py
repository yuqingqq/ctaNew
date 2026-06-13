"""Phase AH-native v2: clean rebuild with precomputed multi-horizon returns.

Bug in v1: compute_basket_return_bps used get_indexer(method='nearest') which
silently bias-aligned to wrong bars when symbols had data gaps.

Fix: precompute forward returns at each horizon as close.pct_change(h).shift(-h),
giving a clean (time × symbol) DataFrame. Then protocol just looks up
forward_ret[time][symbol] directly.

Verifies by checking h=48 baseline reproduces production K=3 Sharpe (+1.98).

Variants:
  h=48 K=3 (baseline; should match Phase M +1.98)
  h=96 K=3
  h=288 K=3

All with PM=1 (no persistence beyond native horizon) and PM=2 (hold 2 cycles).
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

OUT = REPO / "outputs/vBTC_native_horizon_v2"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

COST_PER_LEG = 4.5
GATE_PCTILE = 0.30
K = 3
MIN_PICKS_FOR_FILTER = 30
MIN_OBS_PER_SYM = 100
IC_WINDOW_DAYS = 180
IC_UPDATE_DAYS = 90
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
TOP_N = 15
N_PLACEBO_SEEDS = 100


def _sharpe(x, cycles_per_year):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(cycles_per_year))


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


def load_close_wide(symbols):
    frames = []
    for sym in symbols:
        sym_dir = KLINES_DIR / sym / "5m"
        if not sym_dir.exists(): continue
        files = sorted(sym_dir.glob("*.parquet"))
        if not files: continue
        df_list = []
        for f in files:
            try:
                df = pd.read_parquet(f, columns=["open_time", "close"])
                df_list.append(df)
            except Exception: continue
        if not df_list: continue
        df = pd.concat(df_list, ignore_index=True)
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["open_time"]).drop_duplicates("open_time").set_index("open_time")
        df = df.rename(columns={"close": sym})
        frames.append(df)
    if not frames: return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def compute_forward_returns(close_wide, h_bars):
    """Return DataFrame: index=time, cols=symbol, values=forward h-bar return.

    forward_return[t, s] = (close[t+h, s] - close[t, s]) / close[t, s]
    """
    shifted = close_wide.shift(-h_bars)
    return (shifted - close_wide) / close_wide


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


def evaluate_native(apd, universe, fwd_rets, h_bars, pm_m, gate_lookback,
                     placebo_seed=None):
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::h_bars])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    bar_freq = pd.Timedelta(minutes=5)
    hist_disp = deque(maxlen=gate_lookback)
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
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                          "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        # Forward returns at h_bars horizon, from precomputed fwd_rets
        if t not in fwd_rets.index:
            # Skip if forward return not available
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                          "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        rets_at_t = fwd_rets.loc[t]
        ret_l = {s: rets_at_t[s] for s in sym_arr if s in rets_at_t.index}
        # Compute spreads using only symbols with valid returns
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
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        if placebo_seed is not None:
            shuffled = rng.permutation(len(sym_arr))
            cand_l = sym_arr[shuffled[:K]].tolist()
            cand_s = sym_arr[shuffled[K:2*K]].tolist()
        else:
            order_d = np.argsort(-pred_arr); order_a = np.argsort(pred_arr)
            long_r = [sym_arr[i] for i in order_d]
            short_r = [sym_arr[i] for i in order_a]
            cand_l, _ = select_refill(long_r, "long", K, picks_hist, 90, t)
            cand_s, _ = select_refill(short_r, "short", K, picks_hist, 90, t)
        c_ls = set(cand_l); c_ss = set(cand_s)
        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > pm_m:
            hist_basket = hist_basket[-pm_m:]
        if pm_m > 1 and len(hist_basket) >= pm_m:
            p_l = [h["long"] for h in hist_basket[-pm_m:][:pm_m - 1]]
            p_s = [h["short"] for h in hist_basket[-pm_m:][:pm_m - 1]]
            nl = cur_long & c_ls; ns = cur_short & c_ss
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
        # Filter out symbols with NaN returns
        nl_valid = {s for s in nl if s in ret_l and not pd.isna(ret_l[s])}
        ns_valid = {s for s in ns if s in ret_l and not pd.isna(ret_l[s])}
        if not nl_valid or not ns_valid:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                              "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        nl = nl_valid; ns = ns_valid
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
            t_exit = t + h_bars * bar_freq
            for s_ in nl:
                picks_hist[(s_, "long")].append((t, t_exit, ret_l[s_] * 1e4 / len(nl)))
            for s_ in ns:
                picks_hist[(s_, "short")].append((t, t_exit, -ret_l[s_] * 1e4 / len(ns)))
        rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                      "spread_bps": spread, "cost_bps": cost, "net_bps": net,
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
    print("=== Phase AH-native v2: precomputed multi-horizon returns ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())

    listings = get_listings()
    def eligibility_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    print(f"  loading close prices for {len(panel_syms)} symbols...", flush=True)
    t0 = time.time()
    close_wide = load_close_wide(panel_syms)
    print(f"  loaded: {close_wide.shape} ({time.time()-t0:.0f}s)", flush=True)

    # Precompute forward returns at multiple horizons
    print(f"  computing forward returns h ∈ {{48, 96, 288}}...", flush=True)
    fwd_rets = {}
    for h in [48, 96, 288]:
        t0 = time.time()
        fwd_rets[h] = compute_forward_returns(close_wide, h)
        print(f"    h={h}: {fwd_rets[h].shape} ({time.time()-t0:.0f}s); "
              f"sample non-NaN frac: {(~fwd_rets[h].isna()).mean().mean():.0%}",
              flush=True)

    target_t_all = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())

    variants = [
        ("h48_PM2_baseline", 48, 2),
        ("h96_PM1", 96, 1),
        ("h96_PM2", 96, 2),
        ("h288_PM1", 288, 1),
        ("h288_PM2", 288, 2),
    ]

    results = {}
    print(f"\n  {'variant':<22}  {'Sharpe':>7}  {'cycles':>6}  {'maxDD':>7}  {'totPnL':>8}  "
          f"{'gross':>6}  {'cost':>5}  {'net':>6}  {'pos_folds':>9}  {'conc':>4}",
          flush=True)
    for label, h_bars, pm_m in variants:
        t0 = time.time()
        cycles_per_year = (288 * 365) / h_bars
        gate_lookback = int(cycles_per_year)
        sampled_t = target_t_all[::h_bars]
        universe = build_rolling_ic_universe(apd, sampled_t, TOP_N, eligibility_at)
        df_v = evaluate_native(apd, universe, fwd_rets[h_bars], h_bars, pm_m,
                                  gate_lookback)
        net = df_v["net_bps"].to_numpy()
        sh = _sharpe(net, cycles_per_year)
        non_skip = df_v[df_v["skipped"] == 0]
        gross = non_skip["spread_bps"].mean() if "spread_bps" in non_skip.columns and len(non_skip) > 0 else 0
        cost = non_skip["cost_bps"].mean() if "cost_bps" in non_skip.columns and len(non_skip) > 0 else 0
        n_pos = 0
        for f in OOS_FOLDS:
            d = df_v[df_v["fold"] == f]["net_bps"].to_numpy()
            if len(d) >= 2 and _sharpe(d, cycles_per_year) > 0: n_pos += 1
        conc = fold_concentration(df_v)
        results[label] = {"sharpe": sh, "df": df_v, "max_dd": _max_dd(net),
                            "total_pnl": net.sum(), "n_cycles": len(df_v),
                            "h_bars": h_bars, "pm_m": pm_m, "universe": universe,
                            "n_folds_positive": n_pos, "concentration": conc,
                            "cycles_per_year": cycles_per_year}
        df_v.to_csv(OUT / f"per_cycle_{label}.csv", index=False)
        print(f"  {label:<22}  {sh:>+7.2f}  {len(df_v):>6}  {_max_dd(net):>+7.0f}  "
              f"{net.sum():>+8.0f}  {gross:>+6.1f}  {cost:>5.1f}  "
              f"{non_skip['net_bps'].mean() if len(non_skip) > 0 else 0:>+6.1f}  "
              f"{n_pos:>5d}/9  {conc*100:>3.0f}%  ({time.time()-t0:.0f}s)", flush=True)

    base = results["h48_PM2_baseline"]
    cands = {k: v for k, v in results.items() if k != "h48_PM2_baseline"}
    best_name = max(cands, key=lambda k: cands[k]["sharpe"])
    best = cands[best_name]
    print(f"\n  Baseline (h=48 PM=2): Sharpe={base['sharpe']:+.2f}, "
          f"folds={base['n_folds_positive']}/9", flush=True)
    print(f"  Best alternative ({best_name}): Sharpe={best['sharpe']:+.2f}, "
          f"folds={best['n_folds_positive']}/9", flush=True)
    lift = best['sharpe'] - base['sharpe']
    print(f"  Lift: {lift:+.2f}", flush=True)

    # Matched basket-size placebo on best
    if lift > 0.10 or best["sharpe"] > 1.0:
        print(f"\n--- Matched basket-size placebo on {best_name} ({N_PLACEBO_SEEDS} seeds) ---",
              flush=True)
        t0 = time.time()
        placebo_sh = []
        h_bars = best["h_bars"]; pm_m = best["pm_m"]
        gate_lookback = int(best["cycles_per_year"])
        for seed in range(N_PLACEBO_SEEDS):
            df_p = evaluate_native(apd, best["universe"], fwd_rets[h_bars],
                                       h_bars, pm_m, gate_lookback,
                                       placebo_seed=seed)
            placebo_sh.append(_sharpe(df_p["net_bps"].to_numpy(), best["cycles_per_year"]))
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

    print(f"\n=== Phase AH-native v2 verdict ===", flush=True)
    pass_baseline = abs(base['sharpe'] - 1.98) < 0.3  # sanity: baseline should match production
    pass_lift = lift >= 0.30
    pass_folds = best['n_folds_positive'] >= 6
    print(f"  Sanity (baseline ≈ +1.98): {'PASS' if pass_baseline else 'WARN'} "
          f"(got {base['sharpe']:+.2f})", flush=True)
    print(f"  Lift ≥ +0.30:              {'PASS' if pass_lift else 'FAIL'} ({lift:+.2f})",
          flush=True)
    print(f"  ≥6/9 folds:                 {'PASS' if pass_folds else 'FAIL'} "
          f"({best['n_folds_positive']}/9)", flush=True)


if __name__ == "__main__":
    main()
