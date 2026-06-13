"""α-residual strategy: baseline + incremental gate ablations.

Strategy structure:
  1. Phase 1D's β-residual predictions (target = z-scored (return − β_PIT × BTC_ret))
  2. Rolling-IC top-15 universe (180d window, 90d refresh) — "predictability" filter
  3. K=3 long / K=3 short by pred-rank within universe
  4. β-hedged execution → portfolio MTM is on α_β (not raw return)
  5. V3.1 6-sleeve overlay (4h entry, 24h hold, 1/6 weight per sleeve)
  6. Cost model: production V3.1 turnover × 2.25 bps

Variants:
  V0  baseline: universe + picks + hedge + sleeves (NO conv_gate / PM_M2 / filter_refill)
  V1  + conv_gate
  V2  + conv_gate + PM_M2
  V3  + conv_gate + PM_M2 + filter_refill   (= full Phase 1D stack with β-hedged exec)

Reports: Sharpe, end-equity on $100 capital, per-fold, cycles traded.
Saves: per-cycle CSVs + final summary to docs/vBTC_ALPHA_RESIDUAL_PROGRESS.md.
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

APD_PATH = REPO / "outputs/vBTC_phase1d_rolling_beta/all_predictions.parquet"
OUT = REPO / "outputs/vBTC_alpha_residual_gates"
OUT.mkdir(parents=True, exist_ok=True)
OOS_FOLDS = list(range(1, 10))
CAPITAL = 100.0
K = psl.K  # = 3
PM_M = psl.PM_M  # = 2
GATE_LOOKBACK = psl.GATE_LOOKBACK
GATE_PCTILE = psl.GATE_PCTILE


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


def _max_dd(net):
    cum = np.cumsum(net)
    return float((cum - np.maximum.accumulate(cum)).min())


def folds_positive(df_v):
    return sum(1 for _, g in df_v.groupby("fold") if _sharpe(g["net_pnl_bps"]) > 0)


def run_protocol_with_toggles(apd, universe_map, fold_lookup,
                                use_conv_gate, use_pm, use_filter_refill):
    """Production protocol with toggleable gates. Picks by pred (model)."""
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
        g = by_t.get(t)
        if g is None: continue
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


def aggregate_sleeves_alpha(records, alpha_wide):
    """V3.1 sleeve aggregator but MTM on alpha_β (β-hedged exec) instead of raw return."""
    sleeve_queue = deque(maxlen=psl.N_SLEEVES)
    prev_weights = {}
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]
        fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"entry_time": t,
                                  "longs": list(rec["long_basket"]),
                                  "shorts": list(rec["short_basket"])})
        else:
            if not sleeve_queue or sleeve_queue[-1].get("placeholder", False):
                sleeve_queue.append({"entry_time": t, "longs": [], "shorts": [],
                                     "placeholder": True})
            else:
                sleeve_queue.append({"entry_time": t, "longs": [], "shorts": [],
                                     "placeholder": True})
        target_weights = defaultdict(float)
        active_count = len(sleeve_queue)
        sleeve_weight = 1.0 / psl.N_SLEEVES
        for sleeve in sleeve_queue:
            n_long = len(sleeve["longs"])
            n_short = len(sleeve["shorts"])
            if n_long == 0 or n_short == 0: continue
            for s in sleeve["longs"]:
                target_weights[s] += sleeve_weight * (1.0 / n_long)
            for s in sleeve["shorts"]:
                target_weights[s] -= sleeve_weight * (1.0 / n_short)
        # MTM on α_β  (β-hedged execution)
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
        gross_exposure = sum(abs(w) for w in target_weights.values())
        rows.append({"time": t, "fold": fold, "active_sleeves": active_count,
                      "gross_pnl_bps": gross_pnl_bps, "cost_bps": cost_bps,
                      "net_pnl_bps": net_pnl_bps,
                      "turnover": total_abs_delta,
                      "gross_exposure": gross_exposure,
                      "n_symbols": len(target_weights)})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def main():
    print("=== α-residual strategy: baseline + gate ablations (β-hedged exec) ===\n",
          flush=True)
    apd = pd.read_parquet(APD_PATH)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    panel_syms = sorted(apd["symbol"].unique())
    listings = psl.get_listings()
    print(f"Panel: {len(apd):,} rows, {len(panel_syms)} symbols, capital ${CAPITAL:.0f}",
          flush=True)
    print(f"β-hedged exec → MTM on α_β (alpha_A column = realized β-residual)\n",
          flush=True)

    def elig_pit(b):
        if isinstance(b, pd.Timestamp): ts = b
        else: ts = pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=psl.MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    # Build α_β wide panel for β-hedged MTM
    print("Building α_β wide panel (time × symbol) for β-hedged MTM...", flush=True)
    t0 = time.time()
    alpha_wide = apd.pivot_table(index="open_time", columns="symbol",
                                   values="alpha_A", aggfunc="first")
    alpha_wide = alpha_wide.sort_index()
    print(f"  alpha_wide {alpha_wide.shape}, {time.time()-t0:.0f}s\n", flush=True)

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    fold_lookup = apd[apd["open_time"].isin(set(sampled_t))].groupby("open_time")["fold"].first().to_dict()

    print("Building rolling-IC top-15 universe (predictability filter)...", flush=True)
    universe_map = psl.build_rolling_ic_universe(apd, sampled_t, psl.TOP_N, elig_pit)

    # Run 4 variants
    variants = [
        ("V0_baseline_no_gates",          dict(use_conv_gate=False, use_pm=False, use_filter_refill=False)),
        ("V1_conv_gate",                  dict(use_conv_gate=True,  use_pm=False, use_filter_refill=False)),
        ("V2_conv_gate_PM",               dict(use_conv_gate=True,  use_pm=True,  use_filter_refill=False)),
        ("V3_conv_gate_PM_filter_refill", dict(use_conv_gate=True,  use_pm=True,  use_filter_refill=True)),
    ]

    results = {}
    for label, gates in variants:
        print(f"\n{'='*80}\n  {label}  (gates: {gates})\n{'='*80}", flush=True)
        records = run_protocol_with_toggles(apd, universe_map, fold_lookup, **gates)
        df_v = aggregate_sleeves_alpha(records, alpha_wide)
        net = df_v["net_pnl_bps"].to_numpy()
        sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
        total_pnl_d = net.sum() / 1e4 * CAPITAL
        end_equity = CAPITAL + total_pnl_d
        print(f"\n  cycles traded   : {records['traded'].sum()}/{len(records)}", flush=True)
        print(f"  Sharpe          : {sh:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
        print(f"  totPnL          : {net.sum():+.0f} bps = ${total_pnl_d:+.2f}", flush=True)
        print(f"  END-EQUITY      : ${end_equity:.2f} (from $100, {total_pnl_d/CAPITAL*100:+.1f}%)",
              flush=True)
        print(f"  maxDD           : {_max_dd(net):+.0f} bps", flush=True)
        print(f"  gross/cycle     : {df_v['gross_pnl_bps'].mean():+.2f} bps", flush=True)
        print(f"  cost/cycle      : {df_v['cost_bps'].mean():+.2f} bps", flush=True)
        print(f"  turnover/cycle  : {df_v['turnover'].mean():.3f}", flush=True)
        print(f"  folds positive  : {folds_positive(df_v)}/9", flush=True)
        # per-fold
        per_fold = []
        for fid in OOS_FOLDS:
            g = df_v[df_v["fold"] == fid]["net_pnl_bps"].to_numpy()
            per_fold.append((fid, _sharpe(g), g.sum() / 1e4 * CAPITAL))
        results[label] = {
            "sharpe": sh, "sh_lo": lo, "sh_hi": hi,
            "totPnL_dollars": total_pnl_d, "end_equity": end_equity,
            "maxDD_bps": _max_dd(net),
            "gross_per_cycle": df_v['gross_pnl_bps'].mean(),
            "cost_per_cycle": df_v['cost_bps'].mean(),
            "turnover_per_cycle": df_v['turnover'].mean(),
            "n_traded": int(records["traded"].sum()), "n_cycles": len(records),
            "folds_pos": folds_positive(df_v),
            "per_fold": per_fold,
        }
        df_v.to_csv(OUT / f"{label}.csv", index=False)

    print("\n" + "="*100)
    print(f"  SUMMARY — α-residual β-hedged strategy, capital ${CAPITAL:.0f}")
    print("="*100)
    print(f"  {'variant':<32} {'Sharpe':>10} {'end-eq':>10} {'pnl%':>8} {'traded':>10} {'folds+':>7}", flush=True)
    for label, _ in variants:
        r = results[label]
        pct = r['totPnL_dollars'] / CAPITAL * 100
        traded_pct = r['n_traded'] / r['n_cycles'] * 100
        print(f"  {label:<32} {r['sharpe']:+10.2f} ${r['end_equity']:>9.2f} "
              f"{pct:+7.1f}% {traded_pct:>9.0f}% {r['folds_pos']:>4}/9", flush=True)

    # Save summary for doc
    summary_rows = []
    for label, _ in variants:
        r = results[label]
        summary_rows.append({
            "variant": label,
            "sharpe": round(r['sharpe'], 2),
            "sh_lo": round(r['sh_lo'], 2),
            "sh_hi": round(r['sh_hi'], 2),
            "end_equity_$": round(r['end_equity'], 2),
            "pnl_pct": round(r['totPnL_dollars'] / CAPITAL * 100, 1),
            "maxDD_bps": round(r['maxDD_bps'], 0),
            "gross_per_cycle_bps": round(r['gross_per_cycle'], 2),
            "cost_per_cycle_bps": round(r['cost_per_cycle'], 2),
            "turnover_per_cycle": round(r['turnover_per_cycle'], 3),
            "n_traded": r['n_traded'],
            "n_cycles": r['n_cycles'],
            "folds_pos": r['folds_pos'],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / "summary.csv", index=False)
    print(f"\nSaved per-cycle CSVs + summary to {OUT}/", flush=True)
    return results


if __name__ == "__main__":
    main()
