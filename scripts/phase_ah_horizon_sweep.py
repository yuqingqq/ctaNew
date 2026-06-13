"""Phase AH0+AH1: adaptive horizon test.

AH0: For each entry cycle (at production 4h cadence), replay the selected
     basket at horizons h ∈ {12, 24, 48, 96, 288}. Compute gross spread,
     full-turnover cost (2 × COST_PER_LEG per cycle as approx for non-h=48
     cadences; per-cycle churn cost for h=48 native).
AH1: Oracle upper bound — for each cycle, pick the h that gives best
     realized net_bps. Compute Sharpe. This is the diagnostic upper bound.

Decision gate (per user plan):
  - AH0 PASS: some fixed h has Sharpe > 4h baseline +0.30, ≥6/9 folds non-worse
  - AH1 PASS: oracle Sharpe > 4h baseline +1.00, ≥6/9 folds improve, not 1-fold
  - If neither, STOP — adaptive horizon has no theoretical room.

Implementation:
  1. Reuse production protocol (K=3, conv_gate, filter_refill, PM, flat_real).
     Record selected basket symbols per cycle.
  2. Load wide close prices for all 51 symbols.
  3. For each basket at time t, compute basket return at h ∈ {12,24,48,96,288}.
  4. Apply cost model per horizon.
  5. Compute Sharpe per horizon and oracle.
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

OUT = REPO / "outputs/vBTC_adaptive_horizon"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

HORIZON_BASE = 48  # production cycle cadence
HORIZONS = [12, 24, 48, 96, 288]
CYCLES_PER_YEAR_BASE = (288 * 365) / HORIZON_BASE
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


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR_BASE))


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


def load_close_wide(symbols, time_range=None):
    """Load close prices into wide DataFrame: rows=open_time, cols=symbol."""
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
    wide = pd.concat(frames, axis=1).sort_index()
    return wide


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


def run_protocol_save_baskets(apd, universe):
    """Run production protocol; save (time, fold, long_basket, short_basket) per cycle."""
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON_BASE])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    hist_disp = deque(maxlen=GATE_LOOKBACK)
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    by_t = {t: g for t, g in df.groupby("open_time")}
    records = []
    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "skipped": True})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
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
                records.append({"time": t, "fold": fold_lookup.get(t, 0),
                                  "long_basket": [], "short_basket": [], "skipped": True})
                is_flat = True; cur_long, cur_short = set(), set()
            else:
                records.append({"time": t, "fold": fold_lookup.get(t, 0),
                                  "long_basket": [], "short_basket": [], "skipped": True})
            continue
        order_d = np.argsort(-pred_arr); order_a = np.argsort(pred_arr)
        long_r = [sym_arr[i] for i in order_d]
        short_r = [sym_arr[i] for i in order_a]
        cand_l, _ = select_refill(long_r, "long", K, picks_hist, 90, t)
        cand_s, _ = select_refill(short_r, "short", K, picks_hist, 90, t)
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
            if len(nl) > K:
                nl = set(sorted(nl, key=lambda s_: -pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
            if len(ns) > K:
                ns = set(sorted(ns, key=lambda s_: pred_arr[np.where(sym_arr == s_)[0][0]])[:K])
        else:
            nl, ns = c_ls, c_ss
        if not nl or not ns:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "skipped": True})
            if not is_flat and (cur_long or cur_short):
                is_flat = True; cur_long, cur_short = set(), set()
            continue
        # Trade — update PM state and picks history
        for s_ in nl:
            picks_hist[(s_, "long")].append((t, exit_l[s_], ret_l[s_] * 1e4 / len(nl)))
        for s_ in ns:
            picks_hist[(s_, "short")].append((t, exit_l[s_], -ret_l[s_] * 1e4 / len(ns)))
        records.append({"time": t, "fold": fold_lookup.get(t, 0),
                          "long_basket": sorted(list(nl)),
                          "short_basket": sorted(list(ns)),
                          "skipped": False})
        cur_long, cur_short = nl, ns
        is_flat = False
    return pd.DataFrame(records)


def compute_horizon_pnl(baskets_df, close_wide, h_bars):
    """For each basket, compute net_bps at horizon h (bars)."""
    bar_freq = pd.Timedelta(minutes=5)
    rows = []
    closes_idx = close_wide.index
    for _, row in baskets_df.iterrows():
        t = row["time"]
        nl = row["long_basket"]
        ns = row["short_basket"]
        if row["skipped"] or len(nl) == 0 or len(ns) == 0:
            rows.append({"time": t, "fold": row["fold"], "skipped": True,
                          "gross_bps": 0, "net_bps": 0})
            continue
        t_exit = t + h_bars * bar_freq
        if t_exit > closes_idx.max():
            rows.append({"time": t, "fold": row["fold"], "skipped": True,
                          "gross_bps": 0, "net_bps": 0})
            continue
        try:
            close_t = close_wide.loc[t]
            close_exit = close_wide.loc[t_exit]
        except KeyError:
            # If exact time not in index, use nearest
            try:
                t_idx = close_wide.index.get_indexer([t], method="nearest")[0]
                exit_idx = close_wide.index.get_indexer([t_exit], method="nearest")[0]
                close_t = close_wide.iloc[t_idx]
                close_exit = close_wide.iloc[exit_idx]
            except Exception:
                rows.append({"time": t, "fold": row["fold"], "skipped": True,
                              "gross_bps": 0, "net_bps": 0})
                continue
        # Returns per symbol
        long_rets = []
        for s in nl:
            if s in close_t.index and s in close_exit.index:
                p0 = close_t[s]; p1 = close_exit[s]
                if not pd.isna(p0) and not pd.isna(p1) and p0 > 0:
                    long_rets.append((p1 - p0) / p0)
        short_rets = []
        for s in ns:
            if s in close_t.index and s in close_exit.index:
                p0 = close_t[s]; p1 = close_exit[s]
                if not pd.isna(p0) and not pd.isna(p1) and p0 > 0:
                    short_rets.append((p1 - p0) / p0)
        if not long_rets or not short_rets:
            rows.append({"time": t, "fold": row["fold"], "skipped": True,
                          "gross_bps": 0, "net_bps": 0})
            continue
        gross_bps = (np.mean(long_rets) - np.mean(short_rets)) * 1e4
        # Cost model: full close+reopen at each entry cycle = 2 * COST_PER_LEG
        cost_bps = 2 * COST_PER_LEG
        net_bps = gross_bps - cost_bps
        rows.append({"time": t, "fold": row["fold"], "skipped": False,
                      "gross_bps": gross_bps, "cost_bps": cost_bps, "net_bps": net_bps})
    return pd.DataFrame(rows)


def main():
    print("=== Phase AH0+AH1: adaptive horizon test ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    print(f"  panel syms: {len(panel_syms)}", flush=True)

    listings = get_listings()
    def eligibility_at(b):
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}
    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON_BASE]
    print(f"  cycles: {len(sampled_t)}", flush=True)

    # Build production universe
    print(f"  building production universe...", flush=True)
    universe = build_rolling_ic_universe(apd, sampled_t, TOP_N, eligibility_at)

    # Run protocol to save basket selections
    print(f"  running production protocol to save baskets...", flush=True)
    t0 = time.time()
    baskets_df = run_protocol_save_baskets(apd, universe)
    baskets_df.to_parquet(OUT / "production_baskets.parquet", index=False)
    n_traded = (~baskets_df["skipped"]).sum()
    print(f"  baskets saved: {len(baskets_df)} cycles, "
          f"{n_traded} traded ({n_traded/len(baskets_df):.1%})  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # Load wide close prices
    print(f"\n  loading wide close prices for {len(panel_syms)} symbols...", flush=True)
    t0 = time.time()
    close_wide = load_close_wide(panel_syms)
    print(f"  loaded: {close_wide.shape}  ({time.time()-t0:.0f}s)", flush=True)

    # Compute multi-horizon PnL
    print(f"\n=== AH0: Fixed-horizon Sharpe ===\n", flush=True)
    print(f"  {'horizon':>8}  {'Sharpe':>7}  {'maxDD':>7}  {'totPnL':>8}  "
          f"{'gross_avg':>9}  {'cost':>5}  {'net_avg':>7}  per-fold (1-9)", flush=True)
    per_horizon = {}
    per_cycle_pnls = {}  # for oracle
    for h in HORIZONS:
        t0 = time.time()
        df_h = compute_horizon_pnl(baskets_df, close_wide, h)
        per_horizon[h] = df_h
        per_cycle_pnls[h] = df_h.set_index("time")["net_bps"]
        net = df_h["net_bps"].to_numpy()
        non_skipped = df_h[~df_h["skipped"]]
        gross_avg = non_skipped["gross_bps"].mean() if len(non_skipped) > 0 else 0
        cost_avg = non_skipped["cost_bps"].mean() if len(non_skipped) > 0 else 0
        per_fold_sh = []
        for f in OOS_FOLDS:
            d = df_h[df_h["fold"] == f]["net_bps"].to_numpy()
            if len(d) >= 3:
                per_fold_sh.append(_sharpe(d))
            else:
                per_fold_sh.append(0.0)
        pf_str = " ".join(f"{x:+.1f}" for x in per_fold_sh)
        print(f"  h={h:>3}    {_sharpe(net):>+7.2f}  {_max_dd(net):>+7.0f}  "
              f"{net.sum():>+8.0f}  {gross_avg:>+9.2f}  {cost_avg:>5.1f}  "
              f"{non_skipped['net_bps'].mean():>+7.2f}  {pf_str}  ({time.time()-t0:.0f}s)",
              flush=True)
        df_h.to_csv(OUT / f"per_cycle_h{h}.csv", index=False)

    # AH1: Oracle — for each cycle, pick max h by net_bps
    print(f"\n=== AH1: Oracle upper bound (pick best h per cycle) ===\n", flush=True)
    # Build a matrix: rows = cycle, cols = horizon
    all_cycles = sorted(set.union(*[set(s.index) for s in per_cycle_pnls.values()]))
    oracle_rows = []
    horizon_chosen_counts = defaultdict(int)
    for t in all_cycles:
        pnls = {h: per_cycle_pnls[h].get(t, 0.0) for h in HORIZONS}
        best_h = max(pnls, key=pnls.get)
        best_pnl = pnls[best_h]
        # Get fold from baskets_df
        fold = baskets_df[baskets_df["time"] == t]["fold"].iloc[0] if len(baskets_df[baskets_df["time"] == t]) > 0 else 0
        oracle_rows.append({"time": t, "fold": int(fold), "best_h": best_h,
                              "best_net_bps": best_pnl})
        horizon_chosen_counts[best_h] += 1
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(OUT / "oracle_per_cycle.csv", index=False)

    oracle_sh = _sharpe(oracle_df["best_net_bps"].to_numpy())
    print(f"  Oracle Sharpe: {oracle_sh:+.2f}", flush=True)
    print(f"  Oracle totPnL: {oracle_df['best_net_bps'].sum():+.0f}", flush=True)
    print(f"  Oracle maxDD:  {_max_dd(oracle_df['best_net_bps'].to_numpy()):+.0f}", flush=True)
    print(f"\n  Horizon chosen by oracle:", flush=True)
    total = sum(horizon_chosen_counts.values())
    for h in HORIZONS:
        cnt = horizon_chosen_counts[h]
        print(f"    h={h:>3}: {cnt:>4} cycles ({cnt/total:.1%})", flush=True)
    print(f"\n  Per-fold oracle:", flush=True)
    n_pos_oracle = 0
    for f in OOS_FOLDS:
        d = oracle_df[oracle_df["fold"] == f]["best_net_bps"].to_numpy()
        if len(d) >= 3:
            sh_f = _sharpe(d)
            if sh_f > 0: n_pos_oracle += 1
            print(f"    fold {f}: Sharpe={sh_f:+.2f}  pnl={d.sum():+.0f}", flush=True)

    # Verdict
    print(f"\n=== Decision gate ===\n", flush=True)
    base_sh = _sharpe(per_horizon[48]["net_bps"].to_numpy())
    print(f"  h=48 baseline Sharpe: {base_sh:+.2f}", flush=True)
    best_fixed = max(HORIZONS, key=lambda h: _sharpe(per_horizon[h]["net_bps"].to_numpy()))
    best_fixed_sh = _sharpe(per_horizon[best_fixed]["net_bps"].to_numpy())
    print(f"  Best fixed h: h={best_fixed} Sharpe={best_fixed_sh:+.2f} "
          f"(lift {best_fixed_sh - base_sh:+.2f})", flush=True)
    print(f"  Oracle Sharpe: {oracle_sh:+.2f} (lift over h=48: {oracle_sh - base_sh:+.2f})",
          flush=True)
    print(f"  Folds positive (oracle): {n_pos_oracle}/9", flush=True)

    ah0_pass = best_fixed_sh > base_sh + 0.30
    ah1_pass = (oracle_sh > base_sh + 1.00) and (n_pos_oracle >= 6)
    print(f"\n  AH0 (fixed h beats baseline +0.30): {'PASS' if ah0_pass else 'FAIL'}",
          flush=True)
    print(f"  AH1 (oracle beats baseline +1.00 AND ≥6/9 folds): "
          f"{'PASS' if ah1_pass else 'FAIL'}", flush=True)
    if not ah1_pass:
        print(f"\n  → STOP per plan: oracle does not show enough theoretical room.",
              flush=True)
    else:
        print(f"\n  → PROCEED to AH2 (PIT regime rules)", flush=True)
    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
