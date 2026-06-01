"""X10 — Apply C1 winning regularization (normalized sym_id + wider α) to all Ridge Pool+symid cells.

C1 from X8c gave +1.38 vs X6's +1.22 on best cell (+0.12 lift). C1 only affects
Pool+symid architecture (Per-sym and Pool-nosym don't have sym_id dummies).

Re-runs 6 cells: Ridge Pool+symid × {BASE, +aggT, +cohort, +v3, +crossX, +ALL}
with normalized sym_id dummies + RidgeCV alphas={0.001 .. 300} (12 values).

Memory-lite: float32, sparse for original (we use dense for normalized since
normalization makes them non-sparse anyway), gc per fold.
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)

WIDER_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())


def load_panel_with_all_features():
    """Load HL-50 panel with all feature groups merged."""
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE + x6.AGGT_EXTRAS + x6.V3_EXTRAS)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()
    # cohort
    panel = x6b.build_cohort_fixed(panel)
    # crossX
    cross_path = REPO / "data/ml/cache/cross_exchange_features.parquet"
    cross_df = pd.read_parquet(cross_path)
    cross_z_cols = [c for c in cross_df.columns if c.endswith("_basis_z")]
    panel = panel.merge(cross_df[["symbol", "open_time"] + cross_z_cols],
                        on=["symbol", "open_time"], how="left")
    panel = x6.build_target_z(panel)
    for c in panel.columns:
        if panel[c].dtype in ("float64",):
            panel[c] = panel[c].astype("float32")
    for c in cross_z_cols + x6.COHORT_EXTRAS:
        x6.HEAVY_TAIL.add(c)
    return panel, cross_z_cols


def train_ridge_c1(panel, folds, feats):
    """Ridge Pool+symid with NORMALIZED sym_id dummies + wider α grid (C1 winner)."""
    syms_sorted = sorted(panel["symbol"].unique())
    sym_idx = {s: i for i, s in enumerate(syms_sorted)}
    sym_codes = panel["symbol"].map(sym_idx).to_numpy(np.int32)
    n_rows = len(panel)
    n_dum = len(syms_sorted) - 1   # drop first

    # Build dense sym dummies once (5M × 49 float32 = ~960MB — fits)
    dum_all = np.zeros((n_rows, n_dum), dtype=np.float32)
    mask = sym_codes > 0
    dum_all[mask, sym_codes[mask] - 1] = 1.0

    all_preds = []
    alpha_log = []
    sstats0, hstats0 = None, None
    for f, ts, te, ec in folds:
        train_mask = ((panel["exit_time"] < ec).to_numpy()
                      & panel["target_z"].notna().to_numpy())
        test_mask = ((panel["open_time"] >= ts)
                     & (panel["open_time"] <= te)).to_numpy()
        train = panel.iloc[train_mask]
        test = panel.iloc[test_mask]
        if len(train) < 5000 or len(test) < 1000: continue

        if sstats0 is None:
            sstats0, hstats0 = x6.fit_preproc(train, feats)
        Xtr = x6.apply_preproc(train, feats, sstats0, hstats0).astype(np.float32)
        Xte = x6.apply_preproc(test, feats, sstats0, hstats0).astype(np.float32)
        ytr = train["target_z"].to_numpy(np.float32)

        # NORMALIZED sym_id dummies (C1 winner: standardize to unit std)
        dum_tr = dum_all[train_mask]
        dum_te = dum_all[test_mask]
        mean = dum_tr.mean(axis=0)
        std = dum_tr.std(axis=0)
        std[std == 0] = 1.0
        dum_tr_n = ((dum_tr - mean) / std).astype(np.float32)
        dum_te_n = ((dum_te - mean) / std).astype(np.float32)

        X_train = np.hstack([Xtr, dum_tr_n])
        X_test = np.hstack([Xte, dum_te_n])

        m = RidgeCV(alphas=WIDER_GRID).fit(X_train, ytr)
        pred = m.predict(X_test).astype(np.float32)
        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred
        out["fold"] = f
        all_preds.append(out)
        alpha_log.append(float(m.alpha_))
        print(f"      fold {f}: α={m.alpha_:.3f}, n_test={len(test)}", flush=True)
        del Xtr, Xte, X_train, X_test, dum_tr, dum_te, dum_tr_n, dum_te_n, ytr, m
        gc.collect()

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd, alpha_log


def main():
    t0 = time.time()
    print("=== X10 apply C1 normalized sym_id to all Ridge Pool+symid cells ===\n", flush=True)
    print("Winning recipe: NORMALIZED sym_id dummies + wider α grid (0.001..300)\n", flush=True)

    panel, cross_z_cols = load_panel_with_all_features()
    folds = x6.get_folds(panel)
    print(f"  panel loaded: {len(panel):,} rows", flush=True)

    feature_sets = {
        "BASE":    x6.BASE,
        "paggT":   x6.BASE + x6.AGGT_EXTRAS,
        "pcohort": x6.BASE + x6.COHORT_EXTRAS,
        "pv3":     x6.BASE + x6.V3_EXTRAS,
        "pcrossX": x6.BASE + cross_z_cols,
        "pall":    x6.BASE + x6.AGGT_EXTRAS + x6.COHORT_EXTRAS + x6.V3_EXTRAS + cross_z_cols,
    }

    results = []
    for fs_label, feats in feature_sets.items():
        cell_label = f"X10_Ridge_pool+symid_{fs_label}_c1"
        print(f"\n[{fs_label}] features={len(feats)}", flush=True)
        tf = time.time()
        try:
            apd, alpha_log = train_ridge_c1(panel, folds, feats)
            pred_path = CACHE / f"x10_{cell_label}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} "
                  f"α med={np.median(alpha_log):.3f}, [{time.time()-tf:.0f}s]",
                  flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            results.append({"cell": cell_label, "feature_set": fs_label, "error": str(e)})
            continue

        m = x6.run_sleeve_on_preds(pred_path, cell_label)
        row = {"cell": cell_label,
               "model": "Ridge", "arch": "pool+symid_C1norm",
               "feature_set": fs_label, "n_feats": len(feats),
               "train_ic": round(ic, 4),
               "alpha_median": float(np.median(alpha_log)),
               "train_time_s": round(time.time()-tf, 0), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}",
                  flush=True)
        gc.collect()

    keys = ["cell", "model", "arch", "feature_set", "n_feats",
            "train_ic", "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD",
            "folds_pos", "concentration", "net_bps_cycle",
            "alpha_median", "train_time_s", "error"]
    out_csv = OUT / "X10_c1_norm_symid_results.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} cells → {out_csv} [{time.time()-t0:.0f}s]")

    # Also append to master matrix CSV
    master_csv = OUT / "X6_controlled_matrix.csv"
    master_keys = ["cell", "model", "arch", "feature_set", "n_feats",
                   "train_ic", "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD",
                   "folds_pos", "concentration", "net_bps_cycle",
                   "train_time_s", "error"]
    with open(master_csv, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=master_keys, extrasaction="ignore")
        for r in results: w.writerow(r)

    print(f"\n=== X10 results ===")
    print(f"{'features':<10} {'Sharpe':>8} {'X6 baseline':>14} {'Δ':>7} {'folds':>7} {'conc':>6}")
    x6_baselines = {"BASE": 0.38, "paggT": 1.22, "pcohort": -0.72, "pv3": -0.04,
                    "pcrossX": 0.43, "pall": -1.11}
    for r in results:
        if "sharpe" not in r: print(f"{r['feature_set']:<10} ERR"); continue
        base = x6_baselines.get(r["feature_set"], 0)
        delta = r["sharpe"] - base
        print(f"{r['feature_set']:<10} {r['sharpe']:>+8.2f} {base:>+14.2f} {delta:>+7.2f} "
              f"{str(r.get('folds_pos','?')):>7} {str(r.get('concentration','?')):>6}")


if __name__ == "__main__":
    main()
