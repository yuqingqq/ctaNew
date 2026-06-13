"""Phase AH-native v4: full production machinery (SS filter + PM) at native horizon.

Fix v2/v3 issues:
  - v2 NaN-filtering corrupted picks_history; v3 removed SS filter+PM entirely.
  - v4: precompute fwd_rets[h] for h ∈ {48, 96, 144, 288}, then run protocol
    matching Phase M K=3 logic EXACTLY, using fwd_rets[h_bars] as the return source.
  - Skip whole cycle if ANY selected symbol has NaN return at the chosen horizon
    (this is the only safe handling — otherwise picks_history diverges).

Verification: h=48 baseline MUST reproduce Phase M K=3 Sharpe (+1.98). If
it doesn't, the script has a residual bug. If it does, h=96/144/288 results
answer the real question: does the production stack benefit from longer hold?

Variants: h=48 (baseline), h=96, h=144, h=288 — all with SS filter + PM=2.
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

OUT = REPO / "outputs/vBTC_native_horizon_v4"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

COST_PER_LEG = 4.5
GATE_PCTILE = 0.30
GATE_LOOKBACK_FIXED = 252  # fixed cycle count, matches Phase M
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


def _sharpe(x, cpy):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(cpy))


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
    vals = [c for (tt, et, c) in hist if et <= t and tt >= cutoff and not pd.isna(c)]
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


def evaluate_full_machinery(apd, universe, fwd_rets, h_bars, gate_lookback,
                              placebo_seed=None):
    """Production-equivalent protocol: SS filter + PM=2 + conv_gate + K=3 selection,
    at native h_bars cadence using fwd_rets[h_bars] for realized returns.
    """
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
        if t not in fwd_rets.index:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                          "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        rets_at_t = fwd_rets.loc[t]
        # ret_l: pick fwd_rets value for each symbol (can be NaN)
        ret_l = {s: (rets_at_t[s] if s in rets_at_t.index else np.nan)
                  for s in sym_arr}

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
        # PM persistence
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
            if len(nl) > K:
                nl = set(sorted(nl, key=lambda s_: -pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
            if len(ns) > K:
                ns = set(sorted(ns, key=lambda s_: pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
        else:
            nl, ns = c_ls, c_ss
        if not nl or not ns:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0, "n_short": 0})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                              "net_bps": 0, "n_long": 0, "n_short": 0})
            continue
        # Check if all basket members have non-NaN return; if any NaN, SKIP CYCLE (don't trade, don't update history)
        nl_has_nan = any(pd.isna(ret_l.get(s, np.nan)) for s in nl)
        ns_has_nan = any(pd.isna(ret_l.get(s, np.nan)) for s in ns)
        if nl_has_nan or ns_has_nan:
            # Treat as flat (preserve state from prior cycle to avoid spurious churn)
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                          "spread_bps": np.nan, "cost_bps": np.nan,
                          "net_bps": 0, "n_long": 0, "n_short": 0,
                          "reason": "nan_return"})
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
        # Update picks_history WITH FULL CONTRIBUTIONS — matches Phase M exactly
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
    print("=== Phase AH-native v4: full production machinery at native horizon ===\n",
          flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())

    listings = get_listings()
    def eligibility_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    print(f"  loading close prices...", flush=True)
    t0 = time.time()
    close_wide = load_close_wide(panel_syms)
    print(f"  loaded: {close_wide.shape} ({time.time()-t0:.0f}s)", flush=True)

    horizons = [48, 96, 144, 288]
    print(f"  computing fwd_rets for h ∈ {horizons}...", flush=True)
    fwd_rets = {h: compute_forward_returns(close_wide, h) for h in horizons}

    target_t_all = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())

    results = {}
    print(f"\n  {'h':>4}  {'Sharpe':>7}  {'cycles':>6}  {'maxDD':>7}  {'totPnL':>8}  "
          f"{'gross':>6}  {'cost':>5}  {'net':>6}  {'pos_folds':>9}  {'conc':>4}",
          flush=True)
    for h in horizons:
        t0 = time.time()
        cpy = (288 * 365) / h
        gate_lookback = GATE_LOOKBACK_FIXED  # fixed at 252 cycles to match Phase M
        sampled_t = target_t_all[::h]
        universe = build_rolling_ic_universe(apd, sampled_t, TOP_N, eligibility_at)
        df_v = evaluate_full_machinery(apd, universe, fwd_rets[h], h, gate_lookback)
        net = df_v["net_bps"].to_numpy()
        sh = _sharpe(net, cpy)
        non_skip = df_v[df_v["skipped"] == 0]
        gross = non_skip["spread_bps"].mean() if "spread_bps" in non_skip.columns and len(non_skip) > 0 else 0
        cost = non_skip["cost_bps"].mean() if "cost_bps" in non_skip.columns and len(non_skip) > 0 else 0
        n_pos = 0
        for f in OOS_FOLDS:
            d = df_v[df_v["fold"] == f]["net_bps"].to_numpy()
            if len(d) >= 2 and _sharpe(d, cpy) > 0: n_pos += 1
        conc = fold_concentration(df_v)
        results[h] = {"sharpe": sh, "df": df_v, "max_dd": _max_dd(net),
                        "total_pnl": net.sum(), "universe": universe,
                        "n_folds_positive": n_pos, "concentration": conc,
                        "cpy": cpy}
        df_v.to_csv(OUT / f"per_cycle_h{h}.csv", index=False)
        print(f"  h={h:>3}    {sh:>+7.2f}  {len(df_v):>6}  {_max_dd(net):>+7.0f}  "
              f"{net.sum():>+8.0f}  {gross:>+6.1f}  {cost:>5.1f}  "
              f"{non_skip['net_bps'].mean() if len(non_skip) > 0 else 0:>+6.1f}  "
              f"{n_pos:>5d}/9  {conc*100:>3.0f}%  ({time.time()-t0:.0f}s)", flush=True)

    base = results[48]
    print(f"\n  Sanity check: h=48 should match Phase M K=3 (+1.98)", flush=True)
    print(f"  h=48 v4 actual: Sharpe={base['sharpe']:+.2f}", flush=True)
    if abs(base['sharpe'] - 1.98) > 0.3:
        print(f"  WARNING: baseline diverges from production by {abs(base['sharpe']-1.98):.2f}",
              flush=True)
    else:
        print(f"  → baseline matches production (within ±0.3); h>48 results are reliable",
              flush=True)

    cands = {h: r for h, r in results.items() if h != 48}
    best_h = max(cands, key=lambda h: cands[h]["sharpe"])
    best = cands[best_h]
    print(f"\n  Best horizon: h={best_h} Sharpe={best['sharpe']:+.2f} "
          f"(lift {best['sharpe'] - base['sharpe']:+.2f})", flush=True)

    if best["sharpe"] > base["sharpe"] + 0.20:
        print(f"\n--- Matched basket-size placebo on h={best_h} ({N_PLACEBO_SEEDS} seeds) ---",
              flush=True)
        t0 = time.time()
        placebo_sh = []
        gate_lookback = GATE_LOOKBACK_FIXED
        for seed in range(N_PLACEBO_SEEDS):
            df_p = evaluate_full_machinery(apd, best["universe"], fwd_rets[best_h],
                                              best_h, gate_lookback, placebo_seed=seed)
            placebo_sh.append(_sharpe(df_p["net_bps"].to_numpy(), best["cpy"]))
            if (seed + 1) % 25 == 0:
                print(f"  ... {seed+1}/{N_PLACEBO_SEEDS}  ({time.time()-t0:.0f}s)",
                      flush=True)
        p_sh = np.array(placebo_sh)
        p95 = float(np.percentile(p_sh, 95))
        rank = float((p_sh < best["sharpe"]).mean() * 100)
        print(f"\n  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
              f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
        print(f"  h={best_h} ranks p{rank:.0f}  "
              f"beats_p95={'PASS' if best['sharpe'] > p95 else 'FAIL'}", flush=True)
        pd.DataFrame({"seed": range(N_PLACEBO_SEEDS), "sharpe": p_sh}).to_csv(
            OUT / f"matched_placebo_h{best_h}.csv", index=False)


if __name__ == "__main__":
    main()
