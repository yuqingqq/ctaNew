"""Phase AH V3.1: 6-sleeve overlapping portfolio.

Entry cadence: every 4h (production K=3 stack).
Each entry → new sleeve at 1/6 capital, held 24h (6 cycles).
Active sleeves at time t: most recent 6 entries (or fewer during warmup).
Net portfolio weight per symbol = sum over active sleeves of:
    sleeve_weight × side_weight_in_sleeve
where side_weight = +1/n_long_sleeve if long member, -1/n_short_sleeve if short.

Cost = sum(|delta_net_weight|) × 0.5 × COST_PER_LEG (= 2.25 bps per unit abs delta).
This calibrates to production: full K=3 replacement = 4.0 abs delta × 2.25 = 9 bps,
matching production's "9 bps for 100% churn on both sides".

Mark-to-market every 4h on net portfolio. Compare to B0 Phase M K=3 +1.98.
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

OUT = REPO / "outputs/vBTC_sleeve_horizon"
OUT.mkdir(parents=True, exist_ok=True)
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"

HORIZON_ENTRY = 48  # 4h entry cadence
HOLD_BARS = 288  # 24h hold per sleeve
N_SLEEVES = HOLD_BARS // HORIZON_ENTRY  # 6
COST_PER_LEG = 4.5
COST_PER_UNIT_ABS_DELTA = 0.5 * COST_PER_LEG  # = 2.25 bps, matches production
CYCLES_PER_YEAR = (288 * 365) / HORIZON_ENTRY
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


def run_production_protocol_save_sleeves(apd, universe, placebo_seed=None):
    """Run Phase M K=3 production protocol, save (time, fold, long_basket, short_basket, traded)."""
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON_ENTRY])
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
    records = []
    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
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
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            if not is_flat and (cur_long or cur_short):
                is_flat = True; cur_long, cur_short = set(), set()
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
                              "long_basket": [], "short_basket": [], "traded": False})
            if not is_flat and (cur_long or cur_short):
                is_flat = True; cur_long, cur_short = set(), set()
            continue
        # Update picks_hist with realized returns (same as production)
        if placebo_seed is None:
            for s_ in nl:
                picks_hist[(s_, "long")].append((t, exit_l[s_], ret_l[s_] * 1e4 / len(nl)))
            for s_ in ns:
                picks_hist[(s_, "short")].append((t, exit_l[s_], -ret_l[s_] * 1e4 / len(ns)))
        records.append({"time": t, "fold": fold_lookup.get(t, 0),
                          "long_basket": sorted(list(nl)),
                          "short_basket": sorted(list(ns)),
                          "traded": True})
        cur_long, cur_short = nl, ns
        is_flat = False
    return pd.DataFrame(records)


def aggregate_sleeves(records, fwd_rets_4h, placebo_universe=None,
                       placebo_seed=None):
    """Run 6-sleeve overlap portfolio.

    Active sleeves at time t = last N_SLEEVES traded baskets within HOLD_BARS ago.
    Net weight per symbol = sum over active sleeves of (sleeve_weight × side_weight).
    Cost = sum |delta_net_weight| × COST_PER_UNIT_ABS_DELTA bps.
    Gross PnL = sum (prev_net_w[sym] × return_4h[sym, t]) × 1e4 bps.
    """
    bar_freq = pd.Timedelta(minutes=5)
    sleeve_queue = deque(maxlen=N_SLEEVES)  # holds active sleeves' (entry_time, longs, shorts)
    prev_weights = {}  # symbol -> net weight
    rows = []
    rng = np.random.RandomState(placebo_seed if placebo_seed is not None else 0)

    for _, rec in records.iterrows():
        t = rec["time"]
        fold = rec["fold"]
        if placebo_seed is not None and placebo_universe is not None:
            # For placebo: replace long_basket/short_basket with random picks
            # but only when production traded
            if rec["traded"]:
                u = placebo_universe.get(t, set())
                pool = sorted(list(u))
                if len(pool) >= 2 * K:
                    shuffled = rng.permutation(len(pool))
                    long_b = sorted([pool[i] for i in shuffled[:K]])
                    short_b = sorted([pool[i] for i in shuffled[K:2*K]])
                else:
                    long_b = []; short_b = []
            else:
                long_b = []; short_b = []
        else:
            long_b = rec["long_basket"]
            short_b = rec["short_basket"]

        # Add new sleeve to queue (only if traded)
        if long_b and short_b:
            sleeve_queue.append({"entry_time": t, "longs": long_b, "shorts": short_b})
        else:
            # Production skipped this cycle — no new sleeve added (gap)
            # Old sleeves continue to expire
            pass

        # Drop sleeves that have aged out (entry_time + HOLD_BARS ≤ t)
        max_age = HOLD_BARS * bar_freq
        sleeve_queue = deque(
            [s for s in sleeve_queue if (t - s["entry_time"]) < max_age],
            maxlen=N_SLEEVES
        )

        # Build target portfolio: equal sleeve weight = 1/N_SLEEVES (full capital deployed
        # only when 6 sleeves active; during warmup, less deployed)
        target_weights = defaultdict(float)
        active_count = len(sleeve_queue)
        sleeve_weight = 1.0 / N_SLEEVES  # always 1/6 — empty slots = zero contribution
        for sleeve in sleeve_queue:
            n_long = len(sleeve["longs"])
            n_short = len(sleeve["shorts"])
            if n_long == 0 or n_short == 0: continue
            for s in sleeve["longs"]:
                target_weights[s] += sleeve_weight * (1.0 / n_long)
            for s in sleeve["shorts"]:
                target_weights[s] -= sleeve_weight * (1.0 / n_short)

        # Compute realized 4h PnL using prev_weights × return_4h[sym, t]
        gross_pnl_bps = 0.0
        if t in fwd_rets_4h.index:
            rets_at_t = fwd_rets_4h.loc[t]
            for sym, w in prev_weights.items():
                if sym in rets_at_t.index and not pd.isna(rets_at_t[sym]):
                    gross_pnl_bps += w * rets_at_t[sym] * 1e4
        # Compute turnover and cost
        all_syms = set(target_weights.keys()) | set(prev_weights.keys())
        total_abs_delta = sum(abs(target_weights.get(s, 0.0) - prev_weights.get(s, 0.0))
                                for s in all_syms)
        cost_bps = total_abs_delta * COST_PER_UNIT_ABS_DELTA
        net_pnl_bps = gross_pnl_bps - cost_bps

        # Stats
        gross_exposure = sum(abs(w) for w in target_weights.values())
        net_exposure = sum(target_weights.values())

        rows.append({"time": t, "fold": fold, "active_sleeves": active_count,
                      "gross_pnl_bps": gross_pnl_bps, "cost_bps": cost_bps,
                      "net_pnl_bps": net_pnl_bps,
                      "turnover": total_abs_delta,
                      "gross_exposure": gross_exposure,
                      "net_exposure": net_exposure,
                      "n_symbols": len(target_weights)})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def fold_concentration(df_v):
    fold_pnls = df_v.groupby("fold")["net_pnl_bps"].sum()
    pos = fold_pnls[fold_pnls > 0]
    total_pos = pos.sum() if len(pos) > 0 else 0
    if total_pos <= 0: return 0.0
    return float(pos.max() / total_pos)


def main():
    print("=== Phase AH V3.1: 6-sleeve overlapping portfolio ===\n", flush=True)
    print(f"  Entry cadence: {HORIZON_ENTRY} bars (4h)", flush=True)
    print(f"  Hold per sleeve: {HOLD_BARS} bars (24h)", flush=True)
    print(f"  Sleeves: {N_SLEEVES} × {1.0/N_SLEEVES:.3f} capital each", flush=True)
    print(f"  Cost: {COST_PER_UNIT_ABS_DELTA:.2f} bps per unit absolute weight delta\n",
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

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON_ENTRY]
    print(f"  building universe...", flush=True)
    universe = build_rolling_ic_universe(apd, sampled_t, TOP_N, eligibility_at)

    print(f"  running production K=3 protocol, saving sleeves...", flush=True)
    t0 = time.time()
    records = run_production_protocol_save_sleeves(apd, universe)
    print(f"  saved {len(records)} cycles, {records['traded'].sum()} traded "
          f"({time.time()-t0:.0f}s)", flush=True)
    records.to_parquet(OUT / "production_sleeves.parquet", index=False)

    # Build fwd_rets at 4h (HORIZON_ENTRY) for mark-to-market
    print(f"  loading close prices for fwd_rets at h=48 (4h MtM)...", flush=True)
    t0 = time.time()
    # Reuse fast-path from prior scripts by constructing wide
    frames = []
    for sym in panel_syms:
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
    close_wide = pd.concat(frames, axis=1).sort_index()
    fwd_rets_4h = (close_wide.shift(-HORIZON_ENTRY) - close_wide) / close_wide
    print(f"  close_wide: {close_wide.shape}, fwd_rets_4h ready ({time.time()-t0:.0f}s)",
          flush=True)

    # Aggregate sleeves
    print(f"\n  aggregating sleeves...", flush=True)
    t0 = time.time()
    df_sleeve = aggregate_sleeves(records, fwd_rets_4h)
    df_sleeve.to_csv(OUT / "per_cycle_equal6.csv", index=False)

    net = df_sleeve["net_pnl_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    gross_avg = df_sleeve["gross_pnl_bps"].mean()
    cost_avg = df_sleeve["cost_bps"].mean()
    net_avg = df_sleeve["net_pnl_bps"].mean()
    turnover_avg = df_sleeve["turnover"].mean()
    gross_exp_avg = df_sleeve["gross_exposure"].mean()
    net_exp_avg = df_sleeve["net_exposure"].mean()

    print(f"\n=== V3.1 results ===", flush=True)
    print(f"  Sharpe:        {sh:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
    print(f"  totPnL:        {net.sum():+.0f} bps", flush=True)
    print(f"  maxDD:         {_max_dd(net):+.0f} bps", flush=True)
    print(f"  gross_avg:     {gross_avg:+.2f} bps/cycle", flush=True)
    print(f"  cost_avg:      {cost_avg:+.2f} bps/cycle", flush=True)
    print(f"  net_avg:       {net_avg:+.2f} bps/cycle", flush=True)
    print(f"  cost/gross:    {cost_avg/max(abs(gross_avg),1e-6):.2%}", flush=True)
    print(f"  turnover_avg:  {turnover_avg:.3f} (1.0 unit = one full leg replacement)",
          flush=True)
    print(f"  gross_exp_avg: {gross_exp_avg:.3f} (= total |w|, target 1.0 at full deploy)",
          flush=True)
    print(f"  net_exp_avg:   {net_exp_avg:+.4f} (long-short imbalance)", flush=True)

    n_pos = 0
    print(f"\n  Per-fold Sharpe:", flush=True)
    fold_sharpes = []
    for f in OOS_FOLDS:
        d = df_sleeve[df_sleeve["fold"] == f]["net_pnl_bps"].to_numpy()
        if len(d) >= 3:
            sh_f = _sharpe(d)
            fold_sharpes.append(sh_f)
            if sh_f > 0: n_pos += 1
            print(f"    fold {f}: Sharpe={sh_f:+.2f}, pnl={d.sum():+.0f}", flush=True)
    conc = fold_concentration(df_sleeve)
    print(f"\n  Folds positive: {n_pos}/9", flush=True)
    print(f"  Concentration:  {conc*100:.0f}%", flush=True)

    # B0 baseline reference
    B0_SHARPE = 1.98
    B0_TOTPNL = 9167
    B0_MAXDD = -4414
    lift = sh - B0_SHARPE
    print(f"\n=== vs B0 (Phase M K=3 +1.98 Sharpe) ===", flush=True)
    print(f"  V3.1 Sharpe:    {sh:+.2f}", flush=True)
    print(f"  Lift:           {lift:+.2f}", flush=True)
    print(f"  V3.1 totPnL:    {net.sum():+.0f}  vs B0 {B0_TOTPNL:+.0f}", flush=True)
    print(f"  V3.1 maxDD:     {_max_dd(net):+.0f}  vs B0 {B0_MAXDD:+.0f}", flush=True)

    # Pass gate before placebo: only run placebo if lift >= 0.30
    if lift < 0.30:
        print(f"\n  → V3.1 does NOT beat B0 by ≥+0.30. Skip placebo.", flush=True)
        print(f"  Per the plan: stop here unless need to diagnose.", flush=True)
        # Still report gates
        print(f"\n=== Gates ===", flush=True)
        print(f"  Sharpe ≥ B0 + 0.30:                  FAIL ({lift:+.2f})", flush=True)
        print(f"  folds positive ≥ 6/9:                {'PASS' if n_pos>=6 else 'FAIL'} ({n_pos}/9)",
              flush=True)
        print(f"  maxDD not worse by >20%:             {'PASS' if _max_dd(net)>=B0_MAXDD*1.2 else 'FAIL'} "
              f"({_max_dd(net):+.0f} vs B0×1.2={B0_MAXDD*1.2:+.0f})", flush=True)
        print(f"  concentration ≤ 40%:                 {'PASS' if conc<=0.40 else 'FAIL'} "
              f"({conc*100:.0f}%)", flush=True)
        return

    # Matched sleeve placebo
    print(f"\n--- Matched sleeve placebo ({N_PLACEBO_SEEDS} seeds) ---", flush=True)
    print(f"  At each entry cycle (when production traded), randomly select K=3 longs",
          flush=True)
    print(f"  and K=3 shorts from the same N=15 universe. Same hold/netting/cost.",
          flush=True)
    t0 = time.time()
    placebo_sh = []
    for seed in range(N_PLACEBO_SEEDS):
        df_p = aggregate_sleeves(records, fwd_rets_4h,
                                       placebo_universe=universe,
                                       placebo_seed=seed)
        placebo_sh.append(_sharpe(df_p["net_pnl_bps"].to_numpy()))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_PLACEBO_SEEDS}  ({time.time()-t0:.0f}s)", flush=True)
    p_sh = np.array(placebo_sh)
    p95 = float(np.percentile(p_sh, 95))
    rank = float((p_sh < sh).mean() * 100)
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
    print(f"  V3.1 ranks p{rank:.0f}  beats_p95={'PASS' if sh > p95 else 'FAIL'}",
          flush=True)
    pd.DataFrame({"seed": range(N_PLACEBO_SEEDS), "sharpe": p_sh}).to_csv(
        OUT / "matched_placebo.csv", index=False)

    print(f"\n=== Gates ===", flush=True)
    g1 = lift >= 0.30
    g2 = n_pos >= 6
    g3 = _max_dd(net) >= B0_MAXDD * 1.2
    g4 = sh > p95
    g5 = conc <= 0.40
    print(f"  Sharpe ≥ B0 + 0.30:                  {'PASS' if g1 else 'FAIL'} ({lift:+.2f})",
          flush=True)
    print(f"  folds positive ≥ 6/9:                {'PASS' if g2 else 'FAIL'} ({n_pos}/9)",
          flush=True)
    print(f"  maxDD not worse by >20%:             {'PASS' if g3 else 'FAIL'}", flush=True)
    print(f"  beats matched placebo p95:           {'PASS' if g4 else 'FAIL'}", flush=True)
    print(f"  concentration ≤ 40%:                 {'PASS' if g5 else 'FAIL'} "
          f"({conc*100:.0f}%)", flush=True)
    if g1 and g2 and g3 and g4 and g5:
        print(f"\n  → ADOPT V3.1 (6-sleeve overlapping portfolio)", flush=True)
    else:
        print(f"\n  → NOT ADOPTED", flush=True)


if __name__ == "__main__":
    main()
