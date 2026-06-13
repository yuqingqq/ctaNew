"""Phase J: gate-replacement audit on WINNER_21 production stack.

Tests 5 gate variants vs production conv_gate (pred_disp <30th pctile skip).
All variants run with filter_refill_90d_mean on the existing all_predictions
audit panel (outputs/vBTC_audit_panel/all_predictions.parquet).

Variants:
  V0 production_conv_gate    — pred_disp < 30th pctile skip (current baseline)
  V1 no_gate                 — never skip
  V2 inverted_conv_gate      — pred_disp > 70th pctile skip (test the inversion)
  V3 rolling_realized_IC     — skip if trailing-90d IC of universe < threshold
  V4 rank_instability        — skip if current top-K overlap with prior < threshold
  V5 random_skip_30pct       — random 30% skip rate (sanity-check placebo)

For V3 and V4, threshold tuned to match production skip rate (~33%).
For V1, no skipping at all.
For V5, 30% random skip.

After identifying the best variant, run 100-seed matched skip-placebo:
randomly skip the SAME number of cycles as the best variant, measure Sharpe
distribution. Pass condition: best variant beats matched-placebo p95.

Output: outputs/vBTC_gate_audit/{results.csv, per_cycle_*.csv, matched_placebo.csv}
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

OUT = REPO / "outputs/vBTC_gate_audit"
OUT.mkdir(parents=True, exist_ok=True)
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
APD_PATH = REPO / "outputs/vBTC_audit_panel/all_predictions.parquet"

HORIZON = 48
CYCLES_PER_YEAR = (288 * 365) / HORIZON
COST_PER_LEG = 4.5
GATE_LOOKBACK = 252
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
TARGET_SKIP_RATE = 0.30
PROD_SKIP_PCTILE = 0.30  # production: skip below 30th pctile of trailing 252


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


def evaluate_with_gate(apd, universe, gate_fn, seed=0):
    """Run filter_refill protocol with a custom skip-gate function.

    gate_fn(state, t, g_u, pred_arr, sym_arr, prev_top_set) → (skip: bool, signal: float)
      state is a dict (persists across cycles; use to maintain history/state)
      signal is the gate's raw measurement (for diagnostics)
    """
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()

    state = {"hist_dispersion": deque(maxlen=GATE_LOOKBACK),
              "hist_ic": deque(maxlen=GATE_LOOKBACK),
              "prev_top_long": set(), "prev_top_short": set(),
              "hist_universe_pairs": deque(maxlen=GATE_LOOKBACK),
              "rng": np.random.RandomState(seed),
              "first_n_warmup": 30}
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    by_t = {t: g for t, g in df.groupby("open_time")}
    rows = []

    for cycle_idx, t in enumerate(times):
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2 * K + 1:
            rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                          "gate_signal": np.nan, "skip_reason": "empty_univ",
                          "spread_bps": 0, "cost_bps": 0, "net_bps": 0,
                          "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                          "n_excl_long": 0, "n_excl_short": 0})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        ret_l = dict(zip(sym_arr, g_u["return_pct"].to_numpy()))
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))

        # Compute top-K / bot-K candidates (before gate)
        idx_t = np.argpartition(-pred_arr, K - 1)[:K]
        idx_b = np.argpartition(pred_arr, K - 1)[:K]
        top_set = set(sym_arr[idx_t])
        bot_set = set(sym_arr[idx_b])

        # Call custom gate function
        skip, signal = gate_fn(state, t, g_u, pred_arr, sym_arr, idx_t, idx_b,
                                cur_long, cur_short)

        if skip:
            if not is_flat and (cur_long or cur_short):
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "gate_signal": signal, "skip_reason": "gate",
                              "spread_bps": 0, "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": 0, "n_excl_short": 0})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "gate_signal": signal, "skip_reason": "gate",
                              "spread_bps": 0, "cost_bps": 0, "net_bps": 0,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": 0, "n_excl_short": 0})
            # Still update state's prev_top sets (lookback continuity)
            state["prev_top_long"] = top_set
            state["prev_top_short"] = bot_set
            continue

        # filter_refill
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
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 1,
                              "gate_signal": signal, "skip_reason": "empty_basket",
                              "spread_bps": 0, "cost_bps": 2 * COST_PER_LEG,
                              "net_bps": -2 * COST_PER_LEG, "n_long": 0,
                              "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": n_el, "n_excl_short": n_es})
                is_flat = True
                cur_long, cur_short = set(), set()
            else:
                rows.append({"time": t, "fold": fold_lookup.get(t, 0), "skipped": 0,
                              "gate_signal": signal, "skip_reason": "",
                              "spread_bps": 0, "cost_bps": 0, "net_bps": 0,
                              "n_long": 0, "n_short": 0, "n_universe": len(g_u),
                              "n_excl_long": n_el, "n_excl_short": n_es})
            state["prev_top_long"] = top_set
            state["prev_top_short"] = bot_set
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
                      "gate_signal": signal, "skip_reason": "",
                      "spread_bps": spread, "cost_bps": cost, "net_bps": net,
                      "n_long": len(nl), "n_short": len(ns), "n_universe": len(g_u),
                      "n_excl_long": n_el, "n_excl_short": n_es})
        cur_long, cur_short = nl, ns
        state["prev_top_long"] = top_set
        state["prev_top_short"] = bot_set
    return pd.DataFrame(rows)


# Gate functions ------------------------------------------------------------

def gate_production(state, t, g_u, pred_arr, sym_arr, idx_t, idx_b, cur_long, cur_short):
    """Production conv_gate: skip if pred_disp < 30th pctile of trailing 252."""
    disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
    skip = False
    if len(state["hist_dispersion"]) >= state["first_n_warmup"]:
        thr = float(np.quantile(list(state["hist_dispersion"]), PROD_SKIP_PCTILE))
        if disp < thr: skip = True
    state["hist_dispersion"].append(disp)
    return skip, disp


def gate_no_gate(state, t, g_u, pred_arr, sym_arr, idx_t, idx_b, cur_long, cur_short):
    """Never skip."""
    disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
    state["hist_dispersion"].append(disp)
    return False, disp


def gate_inverted(state, t, g_u, pred_arr, sym_arr, idx_t, idx_b, cur_long, cur_short):
    """Inverted: skip if pred_disp > 70th pctile (high conviction = mean-reverts)."""
    disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
    skip = False
    if len(state["hist_dispersion"]) >= state["first_n_warmup"]:
        thr = float(np.quantile(list(state["hist_dispersion"]), 1 - TARGET_SKIP_RATE))
        if disp > thr: skip = True
    state["hist_dispersion"].append(disp)
    return skip, disp


def gate_rolling_ic(state, t, g_u, pred_arr, sym_arr, idx_t, idx_b, cur_long, cur_short):
    """Skip if trailing realized IC of current universe < 30th pctile of past 252 cycles."""
    # Compute realized IC at THIS cycle for the universe: rank-corr of pred to actual return
    rl = g_u["return_pct"].to_numpy()
    if len(g_u) < 5:
        cur_ic = 0.0
    else:
        cur_ic = pd.Series(pred_arr).rank().corr(pd.Series(rl).rank())
        if np.isnan(cur_ic): cur_ic = 0.0
    # Skip decision uses HISTORY (not current — would be look-ahead)
    skip = False
    if len(state["hist_ic"]) >= state["first_n_warmup"]:
        thr = float(np.quantile(list(state["hist_ic"]), PROD_SKIP_PCTILE))
        # NEED PIT signal: use the LAST observed IC (state["hist_ic"][-1])
        last_observed_ic = state["hist_ic"][-1] if state["hist_ic"] else cur_ic
        if last_observed_ic < thr: skip = True
    state["hist_ic"].append(cur_ic)
    return skip, cur_ic


def gate_rank_instability(state, t, g_u, pred_arr, sym_arr, idx_t, idx_b, cur_long, cur_short):
    """Skip if top-K/bot-K overlap with previous cycle's top/bot is below threshold.
    Use overlap as continuous signal; threshold via trailing 252 distribution.
    """
    top_set = set(sym_arr[idx_t])
    bot_set = set(sym_arr[idx_b])
    # Overlap with prev cycle's top/bot
    overlap_l = len(top_set & state["prev_top_long"]) / K
    overlap_s = len(bot_set & state["prev_top_short"]) / K
    overlap = (overlap_l + overlap_s) / 2
    # Track in history
    skip = False
    state["hist_dispersion"].append(overlap)  # reuse hist queue
    if len(state["hist_dispersion"]) >= state["first_n_warmup"]:
        # Skip when overlap < threshold (high instability)
        thr = float(np.quantile(list(state["hist_dispersion"]), PROD_SKIP_PCTILE))
        if overlap < thr: skip = True
    return skip, overlap


def gate_random_30pct(state, t, g_u, pred_arr, sym_arr, idx_t, idx_b, cur_long, cur_short):
    """Random 30% skip — sanity-check placebo."""
    disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
    state["hist_dispersion"].append(disp)
    skip = state["rng"].random() < TARGET_SKIP_RATE
    return skip, disp


GATES = {
    "V0_production": gate_production,
    "V1_no_gate": gate_no_gate,
    "V2_inverted": gate_inverted,
    "V3_rolling_ic": gate_rolling_ic,
    "V4_rank_instability": gate_rank_instability,
    "V5_random_30pct": gate_random_30pct,
}


def summarize(df_v, label):
    net = df_v["net_bps"].to_numpy()
    if len(net) < 10:
        return {"variant": label, "sharpe": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                 "max_dd": 0.0, "total_pnl": 0.0, "skip_rate": 0.0,
                 "avg_L": 0.0, "avg_S": 0.0, "n_cycles": len(net)}
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    return {
        "variant": label, "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
        "max_dd": _max_dd(net), "total_pnl": net.sum(),
        "skip_rate": float((df_v["skipped"] == 1).mean()),
        "avg_L": float(df_v["n_long"].mean()),
        "avg_S": float(df_v["n_short"].mean()),
        "n_cycles": len(net),
    }


def main():
    print("=== Phase J: gate-replacement audit ===\n", flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    print(f"  apd: {len(apd):,} rows, {apd.symbol.nunique()} syms\n", flush=True)

    listings = get_listings()
    panel_syms = set(apd["symbol"].unique())

    def eligibility_at(b):
        if isinstance(b, (int, np.integer)):
            ts = pd.Timestamp(b, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(b)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON]
    print(f"  Building N=15 rolling-IC universe...", flush=True)
    universe = build_rolling_ic_universe(apd, sampled_t, TOP_N, eligibility_at)
    print(f"  done. Running {len(GATES)} gate variants...\n", flush=True)

    print(f"  {'variant':<28}  {'Sharpe':>7}  {'CI':>17}  {'skip%':>6}  "
          f"{'maxDD':>7}  {'totPnL':>7}  {'avgL/S':>8}", flush=True)
    results = []
    excl_track = {}
    for label, gate_fn in GATES.items():
        t0 = time.time()
        df_v = evaluate_with_gate(apd, universe, gate_fn, seed=0)
        res = summarize(df_v, label)
        results.append(res)
        df_v.to_csv(OUT / f"per_cycle_{label}.csv", index=False)
        excl_track[label] = df_v["skipped"].to_numpy()
        print(f"  {label:<28}  {res['sharpe']:>+7.2f}  "
              f"[{res['ci_lo']:>+5.2f},{res['ci_hi']:>+5.2f}]  "
              f"{res['skip_rate']*100:>5.1f}%  "
              f"{res['max_dd']:>+7.0f}  {res['total_pnl']:>+7.0f}  "
              f"{res['avg_L']:>3.1f}/{res['avg_S']:>3.1f}  ({time.time()-t0:.0f}s)",
              flush=True)

    pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)

    # Identify best non-trivial variant (not the random or production)
    candidates = [r for r in results
                  if r["variant"] not in ("V0_production", "V5_random_30pct")]
    best = max(candidates, key=lambda r: r["sharpe"])
    prod = next(r for r in results if r["variant"] == "V0_production")
    print(f"\n=== Best replacement variant: {best['variant']}  Sharpe={best['sharpe']:+.2f} ===",
          flush=True)
    print(f"  vs production            Sharpe={prod['sharpe']:+.2f}",
          flush=True)
    print(f"  Δsharpe                 = {best['sharpe']-prod['sharpe']:+.2f}",
          flush=True)

    # Matched skip-placebo on best replacement variant
    print(f"\n=== Matched skip-placebo ({N_PLACEBO_SEEDS} seeds, matched skip pattern) ===",
          flush=True)
    best_skips = excl_track[best["variant"]].astype(bool)
    n_skips_real = int(best_skips.sum())
    n_total = len(best_skips)
    skip_rate_real = n_skips_real / n_total
    print(f"  Real variant skips {n_skips_real}/{n_total} cycles "
          f"({skip_rate_real*100:.1f}%); placebo will randomly skip same count.",
          flush=True)

    def gate_random_matched(state, t, g_u, pred_arr, sym_arr, idx_t, idx_b, cur_long, cur_short):
        disp = float(pred_arr[idx_t].mean() - pred_arr[idx_b].mean())
        state["hist_dispersion"].append(disp)
        skip = state["rng"].random() < skip_rate_real
        return skip, disp

    t0 = time.time()
    placebo_rows = []
    for seed in range(N_PLACEBO_SEEDS):
        df_v = evaluate_with_gate(apd, universe, gate_random_matched, seed=seed)
        placebo_rows.append(summarize(df_v, f"placebo_{seed}"))
        if (seed + 1) % 25 == 0:
            print(f"  ... {seed+1}/{N_PLACEBO_SEEDS}  ({time.time()-t0:.0f}s)", flush=True)
    placebo_df = pd.DataFrame(placebo_rows)
    placebo_df.to_csv(OUT / "matched_placebo.csv", index=False)
    p_sh = placebo_df["sharpe"].values
    p95 = np.percentile(p_sh, 95)
    rank = (p_sh < best["sharpe"]).mean() * 100
    print(f"\n  Placebo: mean={p_sh.mean():+.2f}, p50={np.percentile(p_sh,50):+.2f}, "
          f"p95={p95:+.2f}, max={p_sh.max():+.2f}", flush=True)
    print(f"\n  {best['variant']}  Sharpe={best['sharpe']:+.2f}  "
          f"rank={rank:.0f}%  beats_p95={'PASS' if best['sharpe'] > p95 else 'FAIL'}",
          flush=True)

    print(f"\n=== Phase J verdict ===\n", flush=True)
    if best["sharpe"] > prod["sharpe"] + 0.20 and best["sharpe"] > p95:
        verdict = f"ADOPT {best['variant']} (lift +{best['sharpe']-prod['sharpe']:.2f}, beats p95)"
    elif best["sharpe"] > prod["sharpe"] + 0.20 and best["sharpe"] <= p95:
        verdict = f"INTERESTING but FAILS p95 — lift is matched by random skipping"
    elif best["sharpe"] > prod["sharpe"] - 0.10:
        verdict = f"NEUTRAL — replacement gate doesn't change Sharpe vs production"
    else:
        verdict = f"NO IMPROVEMENT — keep production conv_gate"
    print(f"  {verdict}", flush=True)

    print(f"\n  saved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
