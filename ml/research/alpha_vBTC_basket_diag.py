"""Diagnose Tier 1 underperformance vs v6_clean baseline +2.47.

Two parallel checks:
  A. ORIG25 subset run — restrict the 51-panel to v6_clean's 25 symbols.
     If this recovers ~+2.47, universe expansion is the binding issue.
  B. Per-symbol PnL attribution on the 51-name run — which symbols dragged?
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

PANEL_PATH = REPO / "outputs/vBTC_features/panel_with_btc_features.parquet"
OUT_DIR = REPO / "outputs/vBTC_basket_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIG25 = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT", "AVAXUSDT",
          "DOTUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT", "INJUSDT",
          "TIAUSDT", "SEIUSDT", "BCHUSDT", "LTCUSDT", "FILUSDT",
          "ARBUSDT", "OPUSDT", "LINKUSDT", "UNIUSDT", "RUNEUSDT",
          "DOGEUSDT", "WLDUSDT", "XRPUSDT"]

FAST_SEEDS = (42,)
FAST_N_FOLDS = 3
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def run_universe(panel: pd.DataFrame, name: str) -> tuple[pd.DataFrame, list]:
    """Train + evaluate on a given subset of the panel.
    Returns (cycles_df with per_cycle_per_name attribution, summary list)."""
    print(f"\n==== {name}: {panel['symbol'].nunique()} syms, {len(panel):,} rows ====", flush=True)
    feat_set = list(XS_FEATURE_COLS_V6_CLEAN)
    all_folds = _multi_oos_splits(panel)
    fold_idx = [len(all_folds) // 5, len(all_folds) // 2, 4 * len(all_folds) // 5]
    folds = [all_folds[i] for i in fold_idx if i < len(all_folds)]

    cycles_summary = []   # per cycle aggregate
    per_name_pnl = {}     # {symbol: {fold: list of name PnL contributions}}

    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue

        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        Xtest = test[feat_set].to_numpy(np.float32)
        yt = tr["demeaned_target"].to_numpy(np.float32)
        yc = ca["demeaned_target"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
        if mask_t.sum() < 1000 or mask_c.sum() < 200: continue

        models = [_train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
                  for s in FAST_SEEDS]
        pred = np.mean([m.predict(Xtest, num_iteration=m.best_iteration) for m in models], axis=0)

        # Aggregate evaluation
        df = evaluate_stacked(test, pred, use_conv_gate=True, use_pm_gate=True)
        for _, r in df.iterrows():
            cycles_summary.append({"fold": fold["fid"], "time": r["time"],
                                   "net": r["net_bps"], "cost": r["cost_bps"],
                                   "skipped": r["skipped"]})

        # Per-name attribution: for each cycle, look at top-K longs and bot-K shorts
        # and split PnL by name. We need to re-run the cycle picks ourselves.
        test_t = test.copy()
        test_t["pred"] = pred
        TOP_K = 7
        for t, g in test_t.groupby("open_time"):
            if len(g) < 2 * TOP_K + 1: continue
            sym_arr = g["symbol"].to_numpy()
            pred_arr = g["pred"].to_numpy()
            ret_arr = g["return_pct"].to_numpy()
            idx_top = np.argpartition(-pred_arr, TOP_K - 1)[:TOP_K]
            idx_bot = np.argpartition(pred_arr, TOP_K - 1)[:TOP_K]
            for i in idx_top:
                s = sym_arr[i]
                per_name_pnl.setdefault(s, []).append(("long", fold["fid"], ret_arr[i]))
            for i in idx_bot:
                s = sym_arr[i]
                per_name_pnl.setdefault(s, []).append(("short", fold["fid"], -ret_arr[i]))

        df_t = pd.DataFrame(cycles_summary)
        df_t = df_t[df_t["fold"] == fold["fid"]]
        n = df_t["net"].to_numpy() if not df_t.empty else np.array([])
        sh = _sharpe(n) if len(n) else 0.0
        mn = n.mean() if len(n) else 0.0
        print(f"  fold {fold['fid']:>2}: basket_xs={mn:+.2f}({sh:+.1f})  ({time.time()-t0:.0f}s)", flush=True)

    return pd.DataFrame(cycles_summary), per_name_pnl


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    # === Run A: ORIG25 subset ===
    panel_orig = panel[panel["symbol"].isin(ORIG25)].copy()
    cycles_orig, per_name_orig = run_universe(panel_orig, "ORIG25")
    if not cycles_orig.empty:
        net_o = cycles_orig["net"].to_numpy()
        sh_o, lo_o, hi_o = block_bootstrap_ci(net_o, statistic=_sharpe, block_size=7, n_boot=2000)
        print(f"  AGGREGATE: n={len(net_o)}  mean_net={net_o.mean():+.2f}  "
              f"Sharpe={sh_o:+.2f} CI=[{lo_o:+.2f},{hi_o:+.2f}]", flush=True)

    # === Run B: per-symbol attribution on full 51-name run (re-do for consistency) ===
    print(f"\nLoading 51-name attribution from re-run...", flush=True)
    cycles_full, per_name_full = run_universe(panel, "FULL51")
    if not cycles_full.empty:
        net_f = cycles_full["net"].to_numpy()
        sh_f, lo_f, hi_f = block_bootstrap_ci(net_f, statistic=_sharpe, block_size=7, n_boot=2000)
        print(f"  AGGREGATE: n={len(net_f)}  mean_net={net_f.mean():+.2f}  "
              f"Sharpe={sh_f:+.2f} CI=[{lo_f:+.2f},{hi_f:+.2f}]", flush=True)

    # === Per-symbol attribution analysis ===
    print(f"\n{'=' * 100}", flush=True)
    print(f"PER-SYMBOL CONTRIBUTION (FULL51 run)", flush=True)
    print(f"{'=' * 100}", flush=True)

    rows = []
    for s, contribs in per_name_full.items():
        long_list = [c[2] for c in contribs if c[0] == "long"]
        short_list = [c[2] for c in contribs if c[0] == "short"]
        long_mean = np.mean(long_list) * 1e4 if long_list else np.nan
        short_mean = np.mean(short_list) * 1e4 if short_list else np.nan
        n_long = len(long_list)
        n_short = len(short_list)
        total_pnl = (sum(long_list) + sum(short_list)) * 1e4
        rows.append({
            "symbol": s,
            "n_long": n_long, "n_short": n_short,
            "long_mean_bps": long_mean, "short_mean_bps": short_mean,
            "total_pnl_bps": total_pnl,
            "in_orig25": s in ORIG25,
        })
    df_attr = pd.DataFrame(rows).sort_values("total_pnl_bps")

    print(f"\n  Worst 15 contributors (most negative total PnL):", flush=True)
    print(f"  {'symbol':<14} {'in_orig25':<10} {'n_long':>7} {'n_short':>7} "
          f"{'long_bps':>10} {'short_bps':>10} {'total_bps':>10}", flush=True)
    for _, r in df_attr.head(15).iterrows():
        print(f"  {r['symbol']:<14} {str(r['in_orig25']):<10} "
              f"{r['n_long']:>7} {r['n_short']:>7} "
              f"{r['long_mean_bps']:>+10.2f} {r['short_mean_bps']:>+10.2f} "
              f"{r['total_pnl_bps']:>+10.0f}", flush=True)

    print(f"\n  Best 15 contributors (most positive total PnL):", flush=True)
    print(f"  {'symbol':<14} {'in_orig25':<10} {'n_long':>7} {'n_short':>7} "
          f"{'long_bps':>10} {'short_bps':>10} {'total_bps':>10}", flush=True)
    for _, r in df_attr.tail(15).iloc[::-1].iterrows():
        print(f"  {r['symbol']:<14} {str(r['in_orig25']):<10} "
              f"{r['n_long']:>7} {r['n_short']:>7} "
              f"{r['long_mean_bps']:>+10.2f} {r['short_mean_bps']:>+10.2f} "
              f"{r['total_pnl_bps']:>+10.0f}", flush=True)

    # Aggregate by ORIG25 vs new
    is_orig = df_attr["in_orig25"]
    pnl_orig = df_attr.loc[is_orig, "total_pnl_bps"].sum()
    pnl_new = df_attr.loc[~is_orig, "total_pnl_bps"].sum()
    print(f"\n  Total PnL by group:", flush=True)
    print(f"    ORIG25 names ({is_orig.sum()}): {pnl_orig:+.0f} bps", flush=True)
    print(f"    NEW names ({(~is_orig).sum()}): {pnl_new:+.0f} bps", flush=True)

    df_attr.to_csv(OUT_DIR / "per_symbol_attribution.csv", index=False)
    cycles_orig.to_csv(OUT_DIR / "cycles_orig25.csv", index=False)
    cycles_full.to_csv(OUT_DIR / "cycles_full51.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
