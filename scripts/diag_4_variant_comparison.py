"""4-variant A/B/C/D comparison for β-hedged α-residual strategy.

  A. Phase 1D predictions (WINNER_21 + sym_id) + IC top-15 universe
  B. Phase 1D predictions (WINNER_21 + sym_id) + liquidity top-30 universe
  C. BTC-only predictions (WINNER_BTC, no sym_id)   + IC top-15 universe
  D. BTC-only predictions (WINNER_BTC, no sym_id)   + liquidity top-30 universe

For each variant:
  - V3.1 β-hedged execution (MTM on α_β), full gates (conv_gate + PM_M2 + filter_refill)
  - Sharpe + end-equity on $100 capital
  - Random-symbol-drop K=5 stress test (20 draws each, same seed → comparable)

Liquidity universe definition (universe-portable alternative to IC top-15):
  - PIT trailing 90d median log_quote_volume (proxy from each symbol's klines)
  - Refresh every 90 days; take top-30 by rank
  - Listing-age ≥ 180d filter (eligibility)
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

OUT = REPO / "outputs/vBTC_4variant_comparison"
OUT.mkdir(parents=True, exist_ok=True)

APD_OLD = REPO / "outputs/vBTC_phase1d_rolling_beta/all_predictions.parquet"
APD_NEW = REPO / "outputs/vBTC_audit_panel_btc_only/all_predictions.parquet"
PANEL_BTC = REPO / "outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet"

OOS_FOLDS = list(range(1, 10))
CAPITAL = 100.0
K = psl.K
PM_M = psl.PM_M
GATE_LOOKBACK = psl.GATE_LOOKBACK
GATE_PCTILE = psl.GATE_PCTILE
N_DRAWS = 20
K_DROP = 5
TOP_N_IC = 15
TOP_N_LIQ = 30
LIQ_REFRESH_DAYS = 90


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def build_liquidity_universe(panel_btc, sampled_t, top_n, refresh_days, panel_syms,
                              listings, min_age_days=180):
    """Build PIT liquidity universe: top-N by trailing log_quote_volume_90d, refresh
    every refresh_days, with listing-age ≥ min_age filter."""
    bar_ms = 5 * 60 * 1000
    update_ms = refresh_days * 288 * bar_ms
    t0_ms = int(pd.Timestamp(sampled_t[0]).timestamp() * 1000)
    boundaries = []
    for t in sampled_t:
        t_ms = int(pd.Timestamp(t).timestamp() * 1000)
        n = (t_ms - t0_ms) // update_ms
        b = t0_ms + n * update_ms
        boundaries.append((t, b))
    unique_b = sorted(set(b for _, b in boundaries))
    b2u = {}
    panel_indexed = panel_btc.set_index(["open_time","symbol"])
    for b in unique_b:
        ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff_age = ts - pd.Timedelta(days=min_age_days)
        # eligible: listed before cutoff_age
        elig = {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff_age}
        # log_quote_volume_90d sampled at the boundary (or nearest prior)
        nearest_t = panel_btc[panel_btc["open_time"] <= ts]["open_time"].max()
        if pd.isna(nearest_t):
            b2u[b] = set(); continue
        snap = panel_btc[panel_btc["open_time"] == nearest_t][["symbol","log_quote_volume_90d"]]
        snap = snap.dropna(subset=["log_quote_volume_90d"])
        snap = snap[snap["symbol"].isin(elig)]
        if len(snap) < top_n:
            b2u[b] = set(snap["symbol"]); continue
        top = snap.nlargest(top_n, "log_quote_volume_90d")["symbol"].tolist()
        b2u[b] = set(top)
    return {t: b2u[b] for t, b in boundaries}


def run_protocol(apd, universe_map, fold_lookup,
                  use_conv_gate=True, use_pm=True, use_filter_refill=True):
    """V3.1 production protocol with gates. Picks by pred (model)."""
    df = apd.sort_values(["open_time", "symbol"]).copy()
    df = df[df["fold"].isin(OOS_FOLDS)]
    times = sorted(df["open_time"].unique())
    keep_t = set(times[::psl.HORIZON_ENTRY])
    df = df[df["open_time"].isin(keep_t)]
    by_t = {t: g for t, g in df.groupby("open_time")}
    hist_disp = deque(maxlen=GATE_LOOKBACK)
    hist_basket = []
    cur_long, cur_short = set(), set()
    is_flat = False
    picks_hist = defaultdict(list)
    records = []
    for t in sorted(by_t.keys()):
        g = by_t[t]
        u = universe_map.get(t, set())
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
        if use_conv_gate and len(hist_disp) >= 30:
            thr = float(np.quantile(list(hist_disp), GATE_PCTILE))
            if disp < thr: skip = True
        hist_disp.append(disp)
        if skip:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                              "long_basket": [], "short_basket": [], "traded": False})
            if not is_flat and (cur_long or cur_short):
                is_flat = True; cur_long, cur_short = set(), set()
            continue
        order_d = np.argsort(-pred_arr); order_a = np.argsort(pred_arr)
        long_r = [sym_arr[i] for i in order_d]
        short_r = [sym_arr[i] for i in order_a]
        if use_filter_refill:
            cand_l, _ = psl.select_refill(long_r, "long", K, picks_hist, 90, t)
            cand_s, _ = psl.select_refill(short_r, "short", K, picks_hist, 90, t)
        else:
            cand_l = long_r[:K]; cand_s = short_r[:K]
        c_ls = set(cand_l); c_ss = set(cand_s)
        hist_basket.append({"long": c_ls, "short": c_ss})
        if len(hist_basket) > PM_M:
            hist_basket = hist_basket[-PM_M:]
        if use_pm and len(hist_basket) >= PM_M:
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
        if use_filter_refill:
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


def aggregate_alpha(records, alpha_wide):
    """V3.1 sleeve aggregator, β-hedged MTM (on α_β instead of return)."""
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
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
        sleeve_weight = 1.0 / psl.N_SLEEVES
        for sleeve in sleeve_queue:
            n_long = len(sleeve["longs"]); n_short = len(sleeve["shorts"])
            if n_long == 0 or n_short == 0: continue
            for s in sleeve["longs"]:
                target_weights[s] += sleeve_weight * (1.0 / n_long)
            for s in sleeve["shorts"]:
                target_weights[s] -= sleeve_weight * (1.0 / n_short)
        gross_pnl_bps = 0.0
        if t in alpha_wide.index:
            alphas_at_t = alpha_wide.loc[t]
            for sym, w in prev_weights.items():
                if sym in alphas_at_t.index and not pd.isna(alphas_at_t[sym]):
                    gross_pnl_bps += w * alphas_at_t[sym] * 1e4
        all_syms = set(target_weights.keys()) | set(prev_weights.keys())
        total_abs_delta = sum(abs(target_weights.get(s, 0.0) - prev_weights.get(s, 0.0))
                                for s in all_syms)
        cost_bps = total_abs_delta * psl.COST_PER_UNIT_ABS_DELTA
        net_pnl_bps = gross_pnl_bps - cost_bps
        rows.append({"time": t, "fold": fold,
                      "gross_pnl_bps": gross_pnl_bps, "cost_bps": cost_bps,
                      "net_pnl_bps": net_pnl_bps,
                      "turnover": total_abs_delta,
                      "gross_exposure": sum(abs(w) for w in target_weights.values()),
                      "n_symbols": len(target_weights)})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def run_single(apd, universe_map, fold_lookup, alpha_wide):
    records = run_protocol(apd, universe_map, fold_lookup,
                            use_conv_gate=True, use_pm=True, use_filter_refill=True)
    df_v = aggregate_alpha(records, alpha_wide)
    return records, df_v


def main():
    print("=== 4-variant A/B/C/D comparison (β-hedged α-residual strategy) ===\n",
          flush=True)
    t_start = time.time()

    # Load predictions
    print("Loading predictions panels...", flush=True)
    apd_old = pd.read_parquet(APD_OLD)
    apd_old["open_time"] = pd.to_datetime(apd_old["open_time"], utc=True)
    apd_old["exit_time"] = pd.to_datetime(apd_old["exit_time"], utc=True)
    apd_new = pd.read_parquet(APD_NEW)
    apd_new["open_time"] = pd.to_datetime(apd_new["open_time"], utc=True)
    apd_new["exit_time"] = pd.to_datetime(apd_new["exit_time"], utc=True)
    panel_syms = sorted(apd_old["symbol"].unique())
    listings = psl.get_listings()
    print(f"  old preds: {len(apd_old):,} rows; new preds: {len(apd_new):,} rows", flush=True)

    # Load panel_btc for liquidity universe building
    panel_btc = pd.read_parquet(PANEL_BTC, columns=["open_time","symbol","log_quote_volume_90d"])
    panel_btc["open_time"] = pd.to_datetime(panel_btc["open_time"], utc=True)

    # Build α_β wide panel for β-hedged MTM (use ALL predictions' alpha_A which == alpha_β)
    print("Building α_β wide panel...", flush=True)
    alpha_wide_old = apd_old.pivot_table(index="open_time", columns="symbol",
                                          values="alpha_A", aggfunc="first").sort_index()
    alpha_wide_new = apd_new.pivot_table(index="open_time", columns="symbol",
                                          values="alpha_A", aggfunc="first").sort_index()

    def elig_pit(b):
        if isinstance(b, pd.Timestamp): ts = b
        else: ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t_old = sorted(apd_old[apd_old["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    target_t_new = sorted(apd_new[apd_new["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t_old = target_t_old[::psl.HORIZON_ENTRY]
    sampled_t_new = target_t_new[::psl.HORIZON_ENTRY]
    fold_lookup_old = apd_old[apd_old["open_time"].isin(set(sampled_t_old))].groupby("open_time")["fold"].first().to_dict()
    fold_lookup_new = apd_new[apd_new["open_time"].isin(set(sampled_t_new))].groupby("open_time")["fold"].first().to_dict()

    # Build universes
    print("Building universes...", flush=True)
    universe_ic_old = psl.build_rolling_ic_universe(apd_old, sampled_t_old, TOP_N_IC, elig_pit)
    universe_ic_new = psl.build_rolling_ic_universe(apd_new, sampled_t_new, TOP_N_IC, elig_pit)
    universe_liq = build_liquidity_universe(panel_btc, sampled_t_old, TOP_N_LIQ,
                                              LIQ_REFRESH_DAYS, panel_syms, listings, min_age_days=180)
    # Liquidity universe ~same for both (same panel)
    print(f"  IC top-{TOP_N_IC}: built", flush=True)
    print(f"  Liquidity top-{TOP_N_LIQ}: built", flush=True)
    # Sample liq universe at one boundary to show
    sample_b = list(universe_liq.keys())[0]
    print(f"  Example liq universe at {sample_b}: {sorted(list(universe_liq[sample_b]))[:15]}...",
          flush=True)

    # Run 4 variants
    variants = [
        ("A_old_features_IC_universe",  apd_old, universe_ic_old, fold_lookup_old, alpha_wide_old),
        ("B_old_features_LIQ_universe", apd_old, universe_liq,    fold_lookup_old, alpha_wide_old),
        ("C_BTC_features_IC_universe",  apd_new, universe_ic_new, fold_lookup_new, alpha_wide_new),
        ("D_BTC_features_LIQ_universe", apd_new, universe_liq,    fold_lookup_new, alpha_wide_new),
    ]

    results = {}
    for label, apd, universe, fold_lookup, alpha_wide in variants:
        print(f"\n{'='*80}\n  {label}\n{'='*80}", flush=True)
        records, df_v = run_single(apd, universe, fold_lookup, alpha_wide)
        net = df_v["net_pnl_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
        total_d = net.sum() / 1e4 * CAPITAL
        end_eq = CAPITAL + total_d
        print(f"  Sharpe         : {sh:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
        print(f"  total PnL      : {net.sum():+.0f} bps = ${total_d:+.2f}", flush=True)
        print(f"  END-EQUITY     : ${end_eq:.2f}  ({total_d/CAPITAL*100:+.1f}%)", flush=True)
        print(f"  maxDD          : {_max_dd(net):+.0f} bps", flush=True)
        print(f"  gross/cycle    : {df_v['gross_pnl_bps'].mean():+.2f} bps", flush=True)
        print(f"  cost/cycle     : {df_v['cost_bps'].mean():+.2f} bps", flush=True)
        print(f"  turnover       : {df_v['turnover'].mean():.3f}", flush=True)
        print(f"  cycles traded  : {records['traded'].sum()}/{len(records)}", flush=True)
        print(f"  folds positive : {folds_positive(df_v)}/9", flush=True)
        results[label] = {
            "sharpe": sh, "totPnL_d": total_d, "end_eq": end_eq,
            "maxDD_bps": _max_dd(net),
            "gross": df_v['gross_pnl_bps'].mean(),
            "cost": df_v['cost_bps'].mean(),
            "turnover": df_v['turnover'].mean(),
            "n_traded": int(records['traded'].sum()),
            "folds_pos": folds_positive(df_v),
        }
        df_v.to_csv(OUT / f"{label}.csv", index=False)

    print("\n" + "="*100)
    print(f"  HEAD-TO-HEAD — β-hedged Sharpe + end-equity, $100 capital, all gates ON")
    print("="*100)
    print(f"  {'variant':<32} {'Sharpe':>10} {'end-eq':>10} {'pnl%':>8} {'traded':>10} {'folds+':>7}",
          flush=True)
    for label, _, _, _, _ in variants:
        r = results[label]
        pct = r['totPnL_d'] / CAPITAL * 100
        print(f"  {label:<32} {r['sharpe']:+10.2f} ${r['end_eq']:>8.2f} "
              f"{pct:+7.1f}% {r['n_traded']:>9}  {r['folds_pos']:>4}/9", flush=True)

    # Save summary
    summary_df = pd.DataFrame([{"variant": label, **results[label]} for label, _, _, _, _ in variants])
    summary_df.to_csv(OUT / "summary.csv", index=False)
    print(f"\nSaved CSVs to {OUT}/", flush=True)
    print(f"Total runtime: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
