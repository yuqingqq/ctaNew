"""Step 8: Ridge model-only test WITHOUT V3.1 sleeve overlap.

Pure execution:
  - Entry every 4h (48 bars)
  - Hold 4h (one period, no overlap)
  - K=3 long / K=3 short, equal weight (1/K per side)
  - β-hedged MTM on α_β
  - NO gates, NO sleeve, NO universe filter (or optionally rolling-IC)

Compare:
  Ridge clean-PIT, ALL universe
  Ridge clean-PIT, rolling-IC universe
  LGBM clean-PIT, ALL universe
  LGBM clean-PIT, rolling-IC universe

Sleeve smoothing typically adds 0.5-1.0 Sharpe via turnover cost reduction.
This test isolates pure prediction quality.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("psl", REPO / "scripts/phase_ah_sleeve.py")
psl = importlib.util.module_from_spec(spec); spec.loader.exec_module(psl)
from ml.research.alpha_v4_xs import block_bootstrap_ci

RIDGE_PREDS  = REPO / "linear_model/results/predictions.parquet"
LGBM_SHIFT49 = REPO / "linear_model/results/lgbm_shift49_predictions.parquet"
KLINES_DIR   = REPO / "data/ml/test/parquet/klines"
OUT          = REPO / "linear_model/results"

K = 3
COST = 2.25
CAPITAL = 100.0
HORIZON_ENTRY = 48
TOP_N = 15
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
CYCLES_PER_YEAR = psl.CYCLES_PER_YEAR


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


def run_no_sleeve(apd, universe_mode, listings):
    """At each 4h sample: pick top-K + bot-K, hold 4h, then exit.

    Returns DataFrame with one row per 4h cycle:
      time, fold, gross_pnl_bps, cost_bps, net_pnl_bps, n_long, n_short,
      btc_hedge_signed (signed β-weighted sum, in units of portfolio capital)
    """
    panel_syms = set(apd["symbol"].unique())
    for s, t in apd.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    def elig_pit(b):
        ts = b if isinstance(b, pd.Timestamp) else pd.Timestamp(b, unit="ms", tz="UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::HORIZON_ENTRY]

    if universe_mode == "rolling_ic":
        universe = psl.build_rolling_ic_universe(apd, sampled_t, TOP_N, elig_pit)
    else:
        universe = {t: elig_pit(t) for t in sampled_t}

    df = apd.sort_values(["open_time","symbol"])
    df = df[df["fold"].isin(OOS_FOLDS) & df["open_time"].isin(set(sampled_t))]
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    by_t = {t: g for t, g in df.groupby("open_time")}

    prev_weights = {}
    rows = []
    for t in sampled_t:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g
        long_picks, short_picks = [], []
        if len(g_u) >= 2 * K:
            sym_arr = g_u["symbol"].to_numpy()
            pred_arr = g_u["pred"].to_numpy()
            idx_top = np.argpartition(-pred_arr, K-1)[:K]
            idx_bot = np.argpartition(pred_arr,  K-1)[:K]
            long_picks = sym_arr[idx_top].tolist()
            short_picks = sym_arr[idx_bot].tolist()

        # Compute target weights for THIS cycle (no overlap)
        target_weights = {}
        if long_picks and short_picks:
            for s in long_picks:
                target_weights[s] = +1.0 / K  # +1/3
            for s in short_picks:
                target_weights[s] = -1.0 / K  # -1/3

        # MTM on PREVIOUS weights × current α_β realized over past 4h
        gross = 0.0
        alpha_lookup = dict(zip(g["symbol"], g["alpha_A"]))
        for sym, w in prev_weights.items():
            a = alpha_lookup.get(sym, np.nan)
            if not pd.isna(a):
                gross += w * a * 1e4

        # Turnover cost from rebalance to new weights
        all_syms = set(target_weights) | set(prev_weights)
        abs_d = sum(abs(target_weights.get(s, 0) - prev_weights.get(s, 0)) for s in all_syms)
        cost = abs_d * COST

        # BTC hedge sizing (diagnostic only — already in α_β math)
        beta_lookup = dict(zip(g["symbol"], g["beta_pit"])) if "beta_pit" in g.columns else {}
        btc_hedge = sum(w * beta_lookup.get(s, np.nan) for s, w in target_weights.items()
                        if not pd.isna(beta_lookup.get(s, np.nan)))

        rows.append({"time": t, "fold": fold_lookup.get(t, 0),
                     "gross_pnl_bps": gross, "cost_bps": cost,
                     "net_pnl_bps": gross - cost,
                     "turnover": abs_d,
                     "n_long": len(long_picks), "n_short": len(short_picks),
                     "btc_hedge_signed": btc_hedge,
                     "long_basket": long_picks, "short_basket": short_picks})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def evaluate(apd, label, universe_mode, listings, has_beta=True):
    apd = apd.copy()
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    apd_s = apd[apd["fold"].isin(OOS_FOLDS)].copy()
    cyc_ic = apd_s.dropna(subset=["alpha_A"]).groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank())
        if len(g) >= 5 else np.nan).dropna()
    per_cycle_ic = float(cyc_ic.mean())

    df_v = run_no_sleeve(apd, universe_mode, listings)
    net = df_v["net_pnl_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
    total_d = net.sum() / 1e4 * CAPITAL
    end_eq = CAPITAL + total_d
    btc_hedge_mean = df_v["btc_hedge_signed"].mean()
    btc_hedge_abs_mean = df_v["btc_hedge_signed"].abs().mean()

    print(f"\n  >>> {label} ({universe_mode}):", flush=True)
    print(f"        Sharpe={sh:+.2f} [{lo:+.2f},{hi:+.2f}]  IC={per_cycle_ic:+.4f}  "
          f"end-eq=${end_eq:.2f}", flush=True)
    print(f"        totPnL={net.sum():+.0f} bps  maxDD={_max_dd(net):+.0f}  "
          f"folds+={folds_positive(df_v)}/9", flush=True)
    print(f"        gross/cyc={df_v['gross_pnl_bps'].mean():+.2f}  "
          f"cost/cyc={df_v['cost_bps'].mean():+.2f}  "
          f"turnover/cyc={df_v['turnover'].mean():.2f}", flush=True)
    print(f"        BTC hedge: mean={btc_hedge_mean:+.3f}, "
          f"avg |β-residual|={btc_hedge_abs_mean:.3f} of capital", flush=True)

    df_v.to_csv(OUT / f"no_sleeve_{label.replace(' ','_')}_{universe_mode}.csv", index=False)
    return {"label": label, "universe": universe_mode, "sharpe": sh,
            "ci_lo": lo, "ci_hi": hi, "ic": per_cycle_ic, "end_eq": end_eq,
            "folds_pos": folds_positive(df_v), "maxDD": _max_dd(net),
            "gross_cycle": float(df_v["gross_pnl_bps"].mean()),
            "cost_cycle": float(df_v["cost_bps"].mean()),
            "turnover_cycle": float(df_v["turnover"].mean()),
            "btc_hedge_abs_mean": btc_hedge_abs_mean}


def main():
    print("=== Step 8: Ridge model-only, NO sleeve, native 4h entry/exit ===\n",
          flush=True)
    t0 = time.time()
    listings = get_listings()

    ridge = pd.read_parquet(RIDGE_PREDS)
    ridge = ridge.rename(columns={"pred_z": "pred"})
    ridge["alpha_A"] = ridge["alpha_beta"]
    print(f"  Ridge: {len(ridge):,} rows (β_pit available)", flush=True)

    lgbm49 = pd.read_parquet(LGBM_SHIFT49)
    # LGBM predictions don't have beta_pit in the parquet — merge from ridge
    if "beta_pit" not in lgbm49.columns:
        merge = ridge[["symbol","open_time","beta_pit"]].copy()
        merge["open_time"] = pd.to_datetime(merge["open_time"], utc=True)
        lgbm49["open_time"] = pd.to_datetime(lgbm49["open_time"], utc=True)
        lgbm49 = lgbm49.merge(merge, on=["symbol","open_time"], how="left")
    print(f"  LGBM clean-PIT: {len(lgbm49):,} rows", flush=True)

    results = []
    for universe in ("rolling_ic", "all"):
        results.append(evaluate(ridge, "Ridge_shift49", universe, listings))
        results.append(evaluate(lgbm49, "LGBM_shift49", universe, listings))

    print("\n" + "="*108, flush=True)
    print("  SUMMARY — Model-only NO GATES, NO SLEEVE, native 4h entry/exit",
          flush=True)
    print("="*108, flush=True)
    print(f"  {'model':<18} {'universe':<14} {'Sharpe':>8} {'CI':>20} "
          f"{'IC':>10} {'end-eq':>10} {'turn/cyc':>9} {'fold+':>6}", flush=True)
    for r in results:
        ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(f"  {r['label']:<18} {r['universe']:<14} {r['sharpe']:+8.2f} "
              f"{ci:>20} {r['ic']:+10.4f} ${r['end_eq']:>8.2f} "
              f"{r['turnover_cycle']:>9.2f} {r['folds_pos']:>3}/9", flush=True)

    print("\n  References:", flush=True)
    print(f"    LGBM full production (sleeve + all gates, shift 1): Sharpe +0.74", flush=True)
    print(f"    LGBM clean-PIT no gates, V3.1 sleeve, all universe: Sharpe +0.59", flush=True)
    print(f"    Ridge production protocol (sleeve + gates):         Sharpe -1.62", flush=True)

    pd.DataFrame(results).to_csv(OUT / "no_sleeve_comparison.csv", index=False)
    print(f"\n  Saved: {OUT / 'no_sleeve_comparison.csv'}", flush=True)
    print(f"  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
