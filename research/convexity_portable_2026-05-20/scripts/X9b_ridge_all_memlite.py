"""X9b — memory-efficient Ridge × {pool+symid, pool-nosym, per-sym} × +ALL.

Skip the 3 LGBM cells (already done via X9), train only the 3 Ridge cells with:
  - float32 panel & design matrix
  - sparse one-hot sym_id (scipy.sparse.csr_matrix)
  - explicit gc.collect() between folds
  - per-fold prediction writes to a temporary list, only concat at end

Resume: skip cells whose pred parquet already exists in _cache/x9_*.parquet
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc
from pathlib import Path
import pandas as pd, numpy as np
from scipy import sparse
from sklearn.linear_model import RidgeCV, Ridge

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


def load_panel_lite():
    """Load HL-50 panel with float32 features, cohort + crossX merged."""
    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE + x6.AGGT_EXTRAS + x6.V3_EXTRAS)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()
    # cohort (with BTC fix)
    panel = x6b.build_cohort_fixed(panel)
    # crossX
    cross_path = REPO / "data/ml/cache/cross_exchange_features.parquet"
    cross_df = pd.read_parquet(cross_path)
    cross_z_cols = [c for c in cross_df.columns if c.endswith("_basis_z")]
    panel = panel.merge(cross_df[["symbol", "open_time"] + cross_z_cols],
                        on=["symbol", "open_time"], how="left")
    # target
    panel = x6.build_target_z(panel)
    # downcast all numeric to float32
    numeric_cols = [c for c in panel.columns if panel[c].dtype in ("float64", "float32")]
    for c in numeric_cols:
        panel[c] = panel[c].astype("float32")
    # mark heavy-tail for the new features
    for c in cross_z_cols + x6.COHORT_EXTRAS:
        x6.HEAVY_TAIL.add(c)
    print(f"  panel loaded: {len(panel):,} rows, dtypes={set(panel.dtypes.values)}",
          flush=True)
    return panel, cross_z_cols


def train_ridge_memlite(panel, folds, feats, arch):
    """Ridge with float32 + sparse sym_id (when applicable) + gc per fold."""
    syms_sorted = sorted(panel["symbol"].unique())
    print(f"    arch={arch}, n_feats={len(feats)}, n_syms={len(syms_sorted)}", flush=True)

    # Precompute sparse sym_id one-hot (CSR) for pool+symid
    use_symid_dummies = (arch == "pool+symid")
    if use_symid_dummies:
        sym_idx = {s: i for i, s in enumerate(syms_sorted)}
        sym_codes = panel["symbol"].map(sym_idx).to_numpy(np.int32)
        # CSR: rows = panel rows, cols = sym indices (drop first to avoid collinearity with intercept)
        rows = np.arange(len(panel))
        # drop first sym
        mask = sym_codes > 0
        cols_dum = sym_codes[mask] - 1   # shift to 0-indexed
        data_dum = np.ones(mask.sum(), dtype=np.float32)
        sym_sparse = sparse.csr_matrix(
            (data_dum, (rows[mask], cols_dum)),
            shape=(len(panel), len(syms_sorted) - 1)
        )
    else:
        sym_sparse = None

    all_preds = []
    sstats0, hstats0 = None, None
    for f, ts, te, ec in folds:
        train_mask = (panel["exit_time"] < ec).to_numpy() & panel["target_z"].notna().to_numpy()
        test_mask = ((panel["open_time"] >= ts) & (panel["open_time"] <= te)).to_numpy()
        train = panel.iloc[train_mask]
        test = panel.iloc[test_mask]
        if len(train) < 5000 or len(test) < 1000:
            continue

        if sstats0 is None:
            sstats0, hstats0 = x6.fit_preproc(train, feats)
        Xtr_dense = x6.apply_preproc(train, feats, sstats0, hstats0).astype(np.float32)
        Xte_dense = x6.apply_preproc(test, feats, sstats0, hstats0).astype(np.float32)

        if use_symid_dummies:
            # sparse hstack: dense features + sparse sym_id dummies
            Xtr = sparse.hstack([sparse.csr_matrix(Xtr_dense), sym_sparse[train_mask]]).tocsr()
            Xte = sparse.hstack([sparse.csr_matrix(Xte_dense), sym_sparse[test_mask]]).tocsr()
        else:
            Xtr = Xtr_dense
            Xte = Xte_dense

        ytr = train["target_z"].to_numpy(np.float32)

        # RidgeCV with explicit alphas, sparse support
        m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
        pred = m.predict(Xte)

        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred.astype(np.float32)
        out["fold"] = f
        all_preds.append(out)
        print(f"      fold {f}: n_tr={len(train):>6,} n_te={len(test):>6,} α={m.alpha_}",
              flush=True)

        # Cleanup before next fold
        del Xtr_dense, Xte_dense, Xtr, Xte, ytr, train, test, m
        gc.collect()

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd


def train_persym_ridge_memlite(panel, folds, feats):
    """Per-symbol Ridge, float32, gc per (sym, fold)."""
    all_preds = []
    for f, ts, te, ec in folds:
        train_all = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test_all = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        out_frames = []
        for sym, gtr in train_all.groupby("symbol"):
            if len(gtr) < 300: continue
            gte = test_all[test_all["symbol"] == sym]
            if len(gte) < 30: continue
            sstats, hstats = x6.fit_preproc(gtr, feats)
            Xtr = x6.apply_preproc(gtr, feats, sstats, hstats).astype(np.float32)
            Xte = x6.apply_preproc(gte, feats, sstats, hstats).astype(np.float32)
            ytr = gtr["target_z"].to_numpy(np.float32)
            try:
                m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
                pred = m.predict(Xte).astype(np.float32)
            except Exception:
                continue
            o = gte[["symbol", "open_time", "alpha_vs_btc_realized",
                     "return_pct", "exit_time"]].copy()
            o.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
            o["pred"] = pred
            o["fold"] = f
            out_frames.append(o)
            del Xtr, Xte, ytr, m
        if out_frames:
            all_preds.append(pd.concat(out_frames, ignore_index=True))
        gc.collect()
        print(f"      fold {f}: {sum(len(o) for o in out_frames):,} test rows", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd


def main():
    t0 = time.time()
    print("=== X9b memory-lite Ridge cells with +ALL features ===\n", flush=True)

    panel, cross_z_cols = load_panel_lite()
    folds = x6.get_folds(panel)
    feats_all = x6.BASE + x6.AGGT_EXTRAS + x6.COHORT_EXTRAS + x6.V3_EXTRAS + cross_z_cols
    feats_all = list(dict.fromkeys(feats_all))
    print(f"  feats_all ({len(feats_all)}): {feats_all[:8]}... + {len(feats_all)-8} more",
          flush=True)

    cells = [
        ("Ridge", "pool+symid", "Ridge_pool+symid_pall"),
        ("Ridge", "pool-nosym", "Ridge_pool-nosym_pall"),
        ("Ridge", "per-sym",    "Ridge_per-sym_pall"),
    ]

    new_rows = []
    for i, (model, arch, cell_label) in enumerate(cells, 1):
        pred_path = CACHE / f"x9_{cell_label}_preds.parquet"
        print(f"\n[{i}/3] {model} | {arch} | +ALL", flush=True)
        tf = time.time()
        if pred_path.exists():
            print(f"    cached: {pred_path.name}, skipping train", flush=True)
            apd = pd.read_parquet(pred_path)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
        else:
            try:
                if arch == "per-sym":
                    apd = train_persym_ridge_memlite(panel, folds, feats_all)
                else:
                    apd = train_ridge_memlite(panel, folds, feats_all, arch)
                apd.to_parquet(pred_path, index=False)
                ic = float(apd["pred"].corr(apd["alpha_A"]))
                print(f"    trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]",
                      flush=True)
                gc.collect()
            except Exception as e:
                print(f"    TRAIN ERR: {type(e).__name__}: {e}", flush=True)
                new_rows.append({"cell": cell_label, "model": model, "arch": arch,
                                  "feature_set": "+ALL", "error": str(e)})
                continue

        m = x6.run_sleeve_on_preds(pred_path, cell_label)
        row = {"cell": cell_label, "model": model, "arch": arch,
               "feature_set": "+ALL", "n_feats": len(feats_all),
               "train_ic": round(ic, 4),
               "train_time_s": round(time.time()-tf, 0), **m}
        new_rows.append(row)
        if "sharpe" in m:
            print(f"    sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)

    # Append to matrix CSV
    out_csv = OUT / "X6_controlled_matrix.csv"
    keys = ["cell", "model", "arch", "feature_set", "n_feats",
            "train_ic", "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD",
            "folds_pos", "concentration", "net_bps_cycle",
            "train_time_s", "error"]
    with open(out_csv, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        for r in new_rows: w.writerow(r)
    print(f"\nAppended {len(new_rows)} Ridge +ALL cells to {out_csv} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
