"""Step 4: Backtest Ridge predictions with threshold-bps gating.

Pipeline:
  1. Load Ridge predictions (pred_z) + σ_idio
  2. pred_bps = pred_z × σ_idio × 1e4
  3. Build rolling-IC top-15 universe (same as production)
  4. Per cycle: within universe, rank by pred_bps
     longs:  symbols with pred_bps > +threshold (up to K=3)
     shorts: symbols with pred_bps < -threshold (up to K=3)
  5. PM_M2 persistence filter (same as production)
  6. filter_refill 90d trailing PnL discipline (same as production)
  7. V3.1 6-sleeve overlay, variable basket sizes
  8. β-hedged MTM on α_β

Sweep threshold ∈ {0, 4.5, 9, 15, 25} bps.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

PREDS    = REPO / "linear_model/results/predictions.parquet"
SIGMA    = REPO / "linear_model/data/sigma_idio.csv"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
RESULTS  = REPO / "linear_model/results"

K = 3
N_SLEEVES = 6
COST_PER_UNIT_ABS_DELTA = 2.25
CAPITAL = 100.0
CYCLES_PER_YEAR = 2190  # 4h cycles × 365 days
THRESHOLDS_BPS = [0, 4.5, 9, 15, 25]
OOS_FOLDS = list(range(1, 10))
HORIZON_ENTRY = 48
TOP_N = 15
PM_M = 2
MIN_HISTORY_DAYS = 60


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


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


def select_refill(ranked, side, k, picks_hist, window_days, t):
    """Same logic as production: prefer symbols with positive trailing PnL."""
    eligible = []
    for s in ranked:
        if len(eligible) >= k: break
        hist = picks_hist.get((s, side), [])
        cutoff = t - pd.Timedelta(days=window_days)
        recent = [r for (et, ex, r) in hist if et >= cutoff]
        if not recent or sum(recent) >= 0:
            eligible.append(s)
    refill_used = []
    if len(eligible) < k:
        for s in ranked:
            if s in eligible: continue
            if len(eligible) >= k: break
            eligible.append(s); refill_used.append(s)
    return eligible[:k], refill_used


def run_threshold_protocol(apd, universe, threshold_bps):
    """V3.1 protocol but with threshold-bps gate replacing conv_gate.

    Variable basket sizes — picks below threshold are dropped.
    """
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::HORIZON_ENTRY])
    df = df[df["open_time"].isin(keep_t)]
    times = sorted(df["open_time"].unique())
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    hist_basket = []
    cur_long, cur_short = set(), set()
    picks_hist = defaultdict(list)
    by_t = {t: g for t, g in df.groupby("open_time")}
    records = []

    for t in times:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g.iloc[0:0]
        if len(g_u) < 2:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                            "long_basket": [], "short_basket": [], "traded": False})
            continue

        sym_arr = g_u["symbol"].to_numpy()
        pred_bps_arr = g_u["pred_bps"].to_numpy()
        exit_l = dict(zip(sym_arr, g_u["exit_time"].to_numpy()))
        ret_l = dict(zip(sym_arr, g_u["alpha_beta"].to_numpy()))   # for picks_hist (bps)

        # Threshold-bps filter at PICK level
        long_mask = pred_bps_arr > threshold_bps
        short_mask = pred_bps_arr < -threshold_bps

        # Top-K longs above threshold, ranked desc by pred
        if long_mask.sum() == 0:
            cand_l = []
        else:
            l_idx = np.where(long_mask)[0]
            l_idx_sorted = l_idx[np.argsort(-pred_bps_arr[l_idx])]
            cand_l = sym_arr[l_idx_sorted].tolist()

        # Bottom-K shorts below -threshold, ranked asc by pred
        if short_mask.sum() == 0:
            cand_s = []
        else:
            s_idx = np.where(short_mask)[0]
            s_idx_sorted = s_idx[np.argsort(pred_bps_arr[s_idx])]
            cand_s = sym_arr[s_idx_sorted].tolist()

        # Apply filter_refill discipline (same as production)
        cand_l, _ = select_refill(cand_l, "long", K, picks_hist, 90, t)
        cand_s, _ = select_refill(cand_s, "short", K, picks_hist, 90, t)
        c_ls = set(cand_l); c_ss = set(cand_s)

        # PM_M2 persistence (same as production)
        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > PM_M:
            hist_basket = hist_basket[-PM_M:]
        if len(hist_basket) >= PM_M:
            p_l = [h["long"] for h in hist_basket[-PM_M:][:PM_M-1]]
            p_s = [h["short"] for h in hist_basket[-PM_M:][:PM_M-1]]
            nl = cur_long & c_ls; ns = cur_short & c_ss
            for s_ in c_ls - cur_long:
                if all(s_ in p for p in p_l): nl.add(s_)
            for s_ in c_ss - cur_short:
                if all(s_ in p for p in p_s): ns.add(s_)
            if len(nl) > K:
                nl = set(sorted(nl, key=lambda s_: -pred_bps_arr[
                    np.where(sym_arr == s_)[0][0]])[:K])
            if len(ns) > K:
                ns = set(sorted(ns, key=lambda s_: pred_bps_arr[
                    np.where(sym_arr == s_)[0][0]])[:K])
        else:
            nl, ns = c_ls, c_ss

        # Require BOTH sides non-empty for β-hedged trading
        if not nl or not ns:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                            "long_basket": [], "short_basket": [], "traded": False})
            if cur_long or cur_short:
                cur_long, cur_short = set(), set()
            continue

        # Update picks_hist with realized returns
        n_l_actual = len(nl); n_s_actual = len(ns)
        for s_ in nl:
            r = ret_l.get(s_, np.nan)
            if not pd.isna(r):
                picks_hist[(s_, "long")].append((t, exit_l[s_],
                                                  r * 1e4 / n_l_actual))
        for s_ in ns:
            r = ret_l.get(s_, np.nan)
            if not pd.isna(r):
                picks_hist[(s_, "short")].append((t, exit_l[s_],
                                                   -r * 1e4 / n_s_actual))
        records.append({"time": t, "fold": fold_lookup.get(t, 0),
                        "long_basket": sorted(list(nl)),
                        "short_basket": sorted(list(ns)),
                        "traded": True})
        cur_long, cur_short = nl, ns

    return pd.DataFrame(records)


def aggregate_alpha(records, alpha_wide):
    """V3.1 6-sleeve, variable basket size, β-hedged MTM on α_β."""
    sleeve_queue = deque(maxlen=N_SLEEVES)
    prev_weights = {}
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"entry_time": t, "longs": list(rec["long_basket"]),
                                  "shorts": list(rec["short_basket"])})
        else:
            sleeve_queue.append({"entry_time": t, "longs": [], "shorts": []})
        target_weights = defaultdict(float)
        sleeve_weight = 1.0 / N_SLEEVES
        for sleeve in sleeve_queue:
            n_long = len(sleeve["longs"]); n_short = len(sleeve["shorts"])
            if n_long == 0 or n_short == 0: continue
            for s in sleeve["longs"]:
                target_weights[s] += sleeve_weight * (1.0 / n_long)
            for s in sleeve["shorts"]:
                target_weights[s] -= sleeve_weight * (1.0 / n_short)
        gross = 0.0
        if t in alpha_wide.index:
            alphas = alpha_wide.loc[t]
            for sym, w in prev_weights.items():
                if sym in alphas.index and not pd.isna(alphas[sym]):
                    gross += w * alphas[sym] * 1e4
        all_syms = set(target_weights.keys()) | set(prev_weights.keys())
        abs_delta = sum(abs(target_weights.get(s, 0.0) - prev_weights.get(s, 0.0))
                         for s in all_syms)
        cost = abs_delta * COST_PER_UNIT_ABS_DELTA
        rows.append({"time": t, "fold": fold,
                      "gross_pnl_bps": gross, "cost_bps": cost,
                      "net_pnl_bps": gross - cost, "turnover": abs_delta,
                      "gross_exposure": sum(abs(w) for w in target_weights.values()),
                      "n_long_total": sum(1 for w in target_weights.values() if w > 0),
                      "n_short_total": sum(1 for w in target_weights.values() if w < 0),
                      "n_symbols": len(target_weights)})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def main():
    print("=== Step 4: Threshold-gated Ridge backtest ===\n", flush=True)
    t0 = time.time()

    print("  Loading Ridge predictions...", flush=True)
    apd = pd.read_parquet(PREDS)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    # Need 'pred' column name + alpha_A for the sleeve aggregator
    apd = apd.rename(columns={"pred_z": "pred"})
    # Recover pred_bps = pred_z × σ_idio × 1e4
    apd["pred_bps"] = apd["pred"] * apd["sigma_idio_ref"] * 1e4
    apd["alpha_A"] = apd["alpha_beta"]
    print(f"    apd: {len(apd):,} rows", flush=True)

    sigma = pd.read_csv(SIGMA, index_col=0)["sigma_idio"]

    # Universe filter
    print("\n  Building rolling-IC universe (top-15)...", flush=True)
    listings = get_listings()
    panel_syms = set(apd["symbol"].unique())
    panel_first = apd.groupby("symbol")["open_time"].min()
    for s, t in panel_first.items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    def elig_pit(b):
        if isinstance(b, pd.Timestamp): ts = b
        else: ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON_ENTRY]
    universe = psl.build_rolling_ic_universe(apd, sampled_t, TOP_N, elig_pit)
    print(f"    universe built for {len(universe)} time points "
          f"({time.time()-t0:.0f}s elapsed)", flush=True)

    # Alpha wide pivot for MTM
    alpha_wide = apd.pivot_table(index="open_time", columns="symbol",
                                  values="alpha_A", aggfunc="first").sort_index()

    # Threshold sweep
    print(f"\n  Threshold sweep on pred_bps...", flush=True)
    results = []
    for thr in THRESHOLDS_BPS:
        ts = time.time()
        records = run_threshold_protocol(apd, universe, threshold_bps=thr)
        df_v = aggregate_alpha(records, alpha_wide)
        net = df_v["net_pnl_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe,
                                         block_size=7, n_boot=1000)
        total_d = net.sum() / 1e4 * CAPITAL
        end_eq = CAPITAL + total_d
        # Avg basket sizes
        long_actual = df_v["n_long_total"].mean()
        short_actual = df_v["n_short_total"].mean()
        n_traded = int(records["traded"].sum())
        n_cycles = len(records)
        r = {
            "threshold_bps": thr,
            "sharpe": sh, "sh_lo": lo, "sh_hi": hi,
            "end_eq": end_eq, "totPnL_bps": float(net.sum()),
            "maxDD": _max_dd(net),
            "gross_cycle": float(df_v["gross_pnl_bps"].mean()),
            "cost_cycle": float(df_v["cost_bps"].mean()),
            "avg_long_basket": float(long_actual),
            "avg_short_basket": float(short_actual),
            "folds_pos": folds_positive(df_v),
            "n_traded": n_traded, "n_cycles": n_cycles,
            "pct_traded": n_traded / n_cycles if n_cycles else 0,
        }
        results.append(r)
        print(f"    threshold ≥ {thr:>4} bps: Sharpe={sh:+.2f} [{lo:+.2f},{hi:+.2f}], "
              f"end-eq=${end_eq:.2f}, traded={n_traded}/{n_cycles} ({r['pct_traded']*100:.0f}%), "
              f"avg L/S={long_actual:.1f}/{short_actual:.1f}, "
              f"folds+={r['folds_pos']}/9 ({time.time()-ts:.0f}s)", flush=True)
        # Save per-cycle CSV
        df_v.to_csv(RESULTS / f"v31_thresh_{thr}bps.csv", index=False)

    # Summary
    print("\n" + "="*100, flush=True)
    print(f"  RIDGE THRESHOLD-GATE SWEEP — full V3.1 β-hedged", flush=True)
    print("="*100, flush=True)
    print(f"  {'thresh':>6} {'Sharpe':>8} {'CI':>20} {'end-eq':>10} "
          f"{'gross':>7} {'cost':>7} {'avg_L/S':>9} {'traded%':>8} {'folds+':>6}",
          flush=True)
    for r in results:
        ci = f"[{r['sh_lo']:+.2f},{r['sh_hi']:+.2f}]"
        print(f"  {r['threshold_bps']:>5} bps {r['sharpe']:+8.2f} "
              f"{ci:>20} ${r['end_eq']:>8.2f} "
              f"{r['gross_cycle']:>+7.2f} {r['cost_cycle']:>7.2f} "
              f"{r['avg_long_basket']:>4.1f}/{r['avg_short_basket']:>3.1f} "
              f"{r['pct_traded']*100:>7.1f}% {r['folds_pos']:>3}/9", flush=True)

    print(f"\n  Reference: LGBM WINNER_17 + β-residual + conv_gate + V3.1 = Sharpe +0.74",
          flush=True)
    best = max(results, key=lambda r: r["sharpe"])
    print(f"\n  Ridge best: threshold = {best['threshold_bps']} bps → "
          f"Sharpe = {best['sharpe']:+.2f} (Δ vs LGBM: {best['sharpe']-0.74:+.2f})",
          flush=True)

    pd.DataFrame(results).to_csv(RESULTS / "threshold_sweep.csv", index=False)
    print(f"\n  Results saved to {RESULTS}/threshold_sweep.csv", flush=True)
    print(f"  Total time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
