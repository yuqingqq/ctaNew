"""Step 7: Run Ridge (and LGBM for comparison) with NO gates at all.

Pure model-only diagnostic. At each 4h cycle:
  1. Rank symbols by pred
  2. Top K=3 long, Bot K=3 short
  3. V3.1 6-sleeve overlay (24h hold)
  4. β-hedged MTM on α_β

NO: conv_gate, PM_M, filter_refill, threshold gate.

Two universe variants:
  A. rolling-IC top-15 (matches production universe)
  B. ALL symbols (no universe filter at all)

Compare Ridge vs LGBM clean-PIT under both setups. Apples-to-apples.
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

RIDGE_PREDS  = REPO / "linear_model/results/predictions.parquet"
LGBM_SHIFT49 = REPO / "linear_model/results/lgbm_shift49_predictions.parquet"
LGBM_SHIFT1  = REPO / "linear_model/results/lgbm_shift1_predictions.parquet"
KLINES_DIR   = REPO / "data/ml/test/parquet/klines"
OUT          = REPO / "linear_model/results"

K = 3
N_SLEEVES = 6
COST = 2.25
CAPITAL = 100.0
HORIZON_ENTRY = 48
TOP_N = 15
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60


def _sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2 or x.std() == 0: return 0.0
    return float(x.mean() / x.std() * np.sqrt(psl.CYCLES_PER_YEAR))


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


def run_no_gates(apd, universe_mode, listings):
    """Pure model-only: rank pred, take top-K / bot-K, no gates."""
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
    else:  # "all"
        universe = {t: elig_pit(t) for t in sampled_t}

    df = apd.sort_values(["open_time","symbol"])
    df = df[df["fold"].isin(OOS_FOLDS) & df["open_time"].isin(set(sampled_t))]
    fold_lookup = df.groupby("open_time")["fold"].first().to_dict()
    by_t = {t: g for t, g in df.groupby("open_time")}
    records = []
    for t in sampled_t:
        g = by_t.get(t)
        if g is None: continue
        u = universe.get(t, set())
        g_u = g[g["symbol"].isin(u)] if u else g
        if len(g_u) < 2 * K:
            records.append({"time": t, "fold": fold_lookup.get(t, 0),
                            "long_basket": [], "short_basket": [], "traded": False})
            continue
        sym_arr = g_u["symbol"].to_numpy()
        pred_arr = g_u["pred"].to_numpy()
        idx_top = np.argpartition(-pred_arr, K-1)[:K]
        idx_bot = np.argpartition(pred_arr,  K-1)[:K]
        cand_l = sym_arr[idx_top].tolist()
        cand_s = sym_arr[idx_bot].tolist()
        records.append({"time": t, "fold": fold_lookup.get(t, 0),
                        "long_basket": cand_l, "short_basket": cand_s, "traded": True})
    return pd.DataFrame(records)


def aggregate_alpha(records, alpha_wide):
    sleeve_queue = deque(maxlen=N_SLEEVES)
    prev_weights = {}
    rows = []
    for _, rec in records.iterrows():
        t = rec["time"]; fold = rec["fold"]
        if rec["traded"]:
            sleeve_queue.append({"longs": list(rec["long_basket"]),
                                  "shorts": list(rec["short_basket"])})
        else:
            sleeve_queue.append({"longs": [], "shorts": []})
        target_weights = defaultdict(float)
        sw = 1.0 / N_SLEEVES
        for sl in sleeve_queue:
            nL, nS = len(sl["longs"]), len(sl["shorts"])
            if nL == 0 or nS == 0: continue
            for s in sl["longs"]: target_weights[s] += sw * (1.0 / nL)
            for s in sl["shorts"]: target_weights[s] -= sw * (1.0 / nS)
        gross = 0.0
        if t in alpha_wide.index:
            a = alpha_wide.loc[t]
            for sym, w in prev_weights.items():
                if sym in a.index and not pd.isna(a[sym]):
                    gross += w * a[sym] * 1e4
        syms = set(target_weights.keys()) | set(prev_weights.keys())
        abs_d = sum(abs(target_weights.get(s, 0) - prev_weights.get(s, 0)) for s in syms)
        cost = abs_d * COST
        rows.append({"time": t, "fold": fold, "gross_pnl_bps": gross,
                     "cost_bps": cost, "net_pnl_bps": gross - cost,
                     "turnover": abs_d,
                     "long_basket": list(rec.get("long_basket", [])),
                     "short_basket": list(rec.get("short_basket", []))})
        prev_weights = dict(target_weights)
    return pd.DataFrame(rows)


def evaluate(apd, label, universe_mode, listings):
    apd = apd.copy()
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["exit_time"] = pd.to_datetime(apd["exit_time"], utc=True)
    # IC at every 4h sample
    apd_s = apd[apd["fold"].isin(OOS_FOLDS)].copy()
    cyc_ic = apd_s.dropna(subset=["alpha_A"]).groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank())
        if len(g) >= 5 else np.nan).dropna()
    per_cycle_ic = float(cyc_ic.mean())

    records = run_no_gates(apd, universe_mode, listings)
    alpha_wide = apd.pivot_table(index="open_time", columns="symbol",
                                  values="alpha_A", aggfunc="first").sort_index()
    df_v = aggregate_alpha(records, alpha_wide)
    net = df_v["net_pnl_bps"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=1000)
    total_d = net.sum() / 1e4 * CAPITAL
    end_eq = CAPITAL + total_d
    print(f"\n  >>> {label} ({universe_mode}):", flush=True)
    print(f"        Sharpe={sh:+.2f} [{lo:+.2f},{hi:+.2f}]  IC={per_cycle_ic:+.4f}  "
          f"end-eq=${end_eq:.2f}", flush=True)
    print(f"        totPnL={net.sum():+.0f} bps  maxDD={_max_dd(net):+.0f}  "
          f"folds+={folds_positive(df_v)}/9", flush=True)
    print(f"        gross/cyc={df_v['gross_pnl_bps'].mean():+.2f}  "
          f"cost/cyc={df_v['cost_bps'].mean():+.2f}", flush=True)
    return {"label": label, "universe": universe_mode, "sharpe": sh,
            "ci_lo": lo, "ci_hi": hi, "ic": per_cycle_ic, "end_eq": end_eq,
            "folds_pos": folds_positive(df_v), "maxDD": _max_dd(net),
            "gross_cycle": float(df_v["gross_pnl_bps"].mean())}


def main():
    print("=== Step 7: Model-only (no gates) Ridge vs LGBM ===\n", flush=True)
    t0 = time.time()
    listings = get_listings()

    print("  Loading Ridge predictions...", flush=True)
    ridge = pd.read_parquet(RIDGE_PREDS)
    ridge = ridge.rename(columns={"pred_z": "pred"})
    ridge["alpha_A"] = ridge["alpha_beta"]
    print(f"    Ridge: {len(ridge):,} rows", flush=True)

    print("\n  Loading LGBM clean-PIT (shift 49) predictions...", flush=True)
    if not LGBM_SHIFT49.exists():
        print(f"    MISSING: {LGBM_SHIFT49} — run 05_lgbm_clean_pit_baseline.py first",
              flush=True)
        return
    lgbm49 = pd.read_parquet(LGBM_SHIFT49)
    print(f"    LGBM49: {len(lgbm49):,} rows", flush=True)

    results = []
    for universe in ("rolling_ic", "all"):
        results.append(evaluate(ridge, "Ridge_shift49", universe, listings))
        results.append(evaluate(lgbm49, "LGBM_shift49", universe, listings))

    print("\n" + "="*100, flush=True)
    print("  SUMMARY — Model-only (NO gates), V3.1 6-sleeve, β-hedged MTM", flush=True)
    print("="*100, flush=True)
    print(f"  {'model':<20} {'universe':<14} {'Sharpe':>8} {'CI':>20} "
          f"{'IC':>10} {'end-eq':>10} {'fold+':>6}", flush=True)
    for r in results:
        ci = f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}]"
        print(f"  {r['label']:<20} {r['universe']:<14} {r['sharpe']:+8.2f} "
              f"{ci:>20} {r['ic']:+10.4f} ${r['end_eq']:>8.2f} "
              f"{r['folds_pos']:>3}/9", flush=True)

    print("\n  Production reference (LGBM shift(1) + full gates): Sharpe +0.74",
          flush=True)
    pd.DataFrame(results).to_csv(OUT / "no_gates_comparison.csv", index=False)
    print(f"\n  Saved: {OUT / 'no_gates_comparison.csv'}", flush=True)
    print(f"  Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
