"""X8c — sym_id normalization sweep (HIGH priority, addressing 50x over-penalty issue).

Tests 4 variants on the best cell (Ridge Pool+symid +aggT, X6 baseline +1.22):
  C1: Normalize sym_id dummies (subtract mean, div by std) + RidgeCV wider grid
      Equalizes scale → uniform α applied to uniformly-scaled features.
  C2: Group α — α_main from RidgeCV, α_sym=0 (no penalty on sym_id intercepts).
      Implementation: separate Ridge on demeaned-by-symid target, then sym_id as additional
      free coefs (effectively per-sym intercepts).
  C3: Group α — α_main from RidgeCV, α_sym = α_main / 50 (compensate 50× scale).
  C4: Drop sym_id entirely + wider α grid (Option A reference).

Memory-lite: float32 + sparse sym_id (where applicable) + gc per fold.
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

WIDER_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
FEATS = x6.BASE + x6.AGGT_EXTRAS
HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())


def load_panel():
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + FEATS)
    p = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                        columns=list(set(needed)))
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p["exit_time"] = pd.to_datetime(p["exit_time"], utc=True)
    p = p[p["symbol"].isin(HL_SYMS) & (p["symbol"] != "BTCUSDT")].copy()
    p = x6.build_target_z(p)
    for c in p.columns:
        if p[c].dtype in ("float64",):
            p[c] = p[c].astype("float32")
    return p


def make_sym_dum(panel, normalize=False, mode="dense"):
    """Create sym_id dummy matrix.
    normalize: if True, subtract mean and divide by std (fold-0 normalization happens
               at call site since we want fold-0 stats; here we just create raw)
    mode: 'dense' (np.ndarray float32) or 'sparse' (csr)
    Returns: dense array or sparse matrix
    """
    syms_sorted = sorted(panel["symbol"].unique())
    sym_idx = {s: i for i, s in enumerate(syms_sorted)}
    sym_codes = panel["symbol"].map(sym_idx).to_numpy(np.int32)
    n_rows = len(panel)
    n_dum = len(syms_sorted) - 1  # drop first to avoid collinearity with intercept

    if mode == "sparse":
        mask = sym_codes > 0
        rows = np.arange(n_rows)[mask]
        cols = sym_codes[mask] - 1
        data = np.ones(mask.sum(), dtype=np.float32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, n_dum))
    else:
        dum = np.zeros((n_rows, n_dum), dtype=np.float32)
        mask = sym_codes > 0
        dum[mask, sym_codes[mask] - 1] = 1.0
        return dum


def variant_train(panel, folds, variant_name):
    """Train one variant across folds, return predictions df + α picked log."""
    syms_sorted = sorted(panel["symbol"].unique())
    all_preds = []
    alpha_log = []
    sstats0, hstats0 = None, None

    for f, ts, te, ec in folds:
        train_mask = (panel["exit_time"] < ec).to_numpy() & panel["target_z"].notna().to_numpy()
        test_mask = ((panel["open_time"] >= ts) & (panel["open_time"] <= te)).to_numpy()
        train = panel.iloc[train_mask]
        test = panel.iloc[test_mask]
        if len(train) < 5000 or len(test) < 1000: continue

        if sstats0 is None:
            sstats0, hstats0 = x6.fit_preproc(train, FEATS)
        Xtr = x6.apply_preproc(train, FEATS, sstats0, hstats0).astype(np.float32)
        Xte = x6.apply_preproc(test, FEATS, sstats0, hstats0).astype(np.float32)
        ytr = train["target_z"].to_numpy(np.float32)

        # Per-variant: sym_id handling
        if variant_name == "C1_norm_symid":
            # Normalize sym_id dummies using fold-0 train stats
            dum_full = make_sym_dum(panel, mode="dense")  # 5M × 49 float32 = ~960MB
            dum_train = dum_full[train_mask]
            dum_test = dum_full[test_mask]
            mean = dum_train.mean(axis=0)
            std = dum_train.std(axis=0)
            std[std == 0] = 1.0
            dum_train_n = ((dum_train - mean) / std).astype(np.float32)
            dum_test_n = ((dum_test - mean) / std).astype(np.float32)
            del dum_full
            X_train = np.hstack([Xtr, dum_train_n])
            X_test = np.hstack([Xte, dum_test_n])
            y_train = ytr
            del dum_train, dum_test, dum_train_n, dum_test_n

            m = RidgeCV(alphas=WIDER_GRID).fit(X_train, y_train)
            pred = m.predict(X_test)
            picked_alpha = float(m.alpha_)

        elif variant_name == "C2_free_symid":
            # Demean target per-symbol on train (free sym intercept), then RidgeCV
            sym_means = train.groupby("symbol")["target_z"].mean().astype(np.float32)
            y_train = ytr - train["symbol"].map(sym_means).to_numpy(np.float32)
            m = RidgeCV(alphas=WIDER_GRID).fit(Xtr, y_train)
            pred_z = m.predict(Xte).astype(np.float32)
            # Add back per-symbol mean for OOS
            pred = pred_z + test["symbol"].map(sym_means).fillna(0).to_numpy(np.float32)
            picked_alpha = float(m.alpha_)

        elif variant_name == "C3_group_alpha_50x":
            # α_main from CV on main feats; α_sym = α_main/50.
            # Implementation: solve weighted Ridge.
            # Use closed-form: β = (X.T@X + diag(α_per_col))^{-1} @ X.T@y
            dum_sparse_full = make_sym_dum(panel, mode="sparse")
            dum_train = dum_sparse_full[train_mask].toarray().astype(np.float32)
            dum_test = dum_sparse_full[test_mask].toarray().astype(np.float32)
            del dum_sparse_full
            # First pick α_main via RidgeCV on main features only
            m_main = RidgeCV(alphas=WIDER_GRID).fit(Xtr, ytr)
            α_main = float(m_main.alpha_)
            α_sym = α_main / 50.0
            # Now solve combined Ridge with per-coef penalty
            X_combined = np.hstack([Xtr, dum_train]).astype(np.float32)
            X_test_combined = np.hstack([Xte, dum_test]).astype(np.float32)
            n_main = Xtr.shape[1]
            n_sym = dum_train.shape[1]
            penalty = np.concatenate([
                np.full(n_main, α_main, dtype=np.float32),
                np.full(n_sym, α_sym, dtype=np.float32)
            ])
            # β = (X.T@X + diag(penalty))^-1 @ X.T@y
            XtX = X_combined.T @ X_combined
            np.fill_diagonal(XtX, np.diag(XtX) + penalty)
            Xty = X_combined.T @ ytr
            beta = np.linalg.solve(XtX, Xty)
            pred = X_test_combined @ beta
            picked_alpha = α_main
            del dum_train, dum_test, X_combined, X_test_combined, XtX, Xty, beta, m_main

        elif variant_name == "C4_drop_symid":
            m = RidgeCV(alphas=WIDER_GRID).fit(Xtr, ytr)
            pred = m.predict(Xte).astype(np.float32)
            picked_alpha = float(m.alpha_)

        else:
            raise ValueError(f"Unknown variant {variant_name}")

        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred.astype(np.float32)
        out["fold"] = f
        all_preds.append(out)
        alpha_log.append({"fold": f, "alpha": picked_alpha})
        print(f"      fold {f}: α={picked_alpha:.3f}, n_test={len(test)}", flush=True)
        del Xtr, Xte, ytr, train, test
        gc.collect()

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd, alpha_log


def main():
    t0 = time.time()
    print("=== X8c sym_id normalization sweep ===\n", flush=True)
    print("Test cell: Ridge Pool+symid +aggT (BASE+aggT, 19 + 49 sym dummies)")
    print("Baselines: X6 +1.22 (narrow α), X8 A2 +1.26 (wider α)\n", flush=True)

    panel = load_panel()
    folds = x6.get_folds(panel)
    print(f"Panel loaded: {len(panel):,} rows, {panel['symbol'].nunique()} syms",
          flush=True)

    variants = [
        ("C1_norm_symid",       "Normalize sym_id dummies + wider α"),
        ("C2_free_symid",       "Free sym_id intercepts (target demean)"),
        ("C3_group_alpha_50x",  "Group α: α_sym = α_main/50"),
        ("C4_drop_symid",       "Drop sym_id entirely (reference)"),
    ]

    results = []
    for v_name, v_desc in variants:
        tf = time.time()
        print(f"\n[{v_name}] {v_desc}", flush=True)
        try:
            apd, alpha_log = variant_train(panel, folds, v_name)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            pred_path = CACHE / f"x8c_{v_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            alpha_vals = [a["alpha"] for a in alpha_log]
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} "
                  f"α∈[{min(alpha_vals):.3f}, {max(alpha_vals):.3f}] "
                  f"med={np.median(alpha_vals):.3f}, [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            results.append({"variant": v_name, "error": str(e)})
            continue

        m = x6.run_sleeve_on_preds(pred_path, f"x8c_{v_name}")
        row = {"variant": v_name, "desc": v_desc,
               "train_ic": round(ic, 4),
               "alpha_median": float(np.median(alpha_vals)),
               "alpha_min": float(min(alpha_vals)),
               "alpha_max": float(max(alpha_vals)),
               "train_time_s": round(time.time()-tf, 0),
               **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        gc.collect()

    keys = ["variant", "desc", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration",
            "alpha_median", "alpha_min", "alpha_max",
            "train_time_s", "error"]
    out_csv = OUT / "X8c_symid_norm_sweep.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} variants → {out_csv} [{time.time()-t0:.0f}s]")

    print(f"\n=== X8c sym_id normalization results ===")
    print(f"{'variant':<24} {'Sharpe':>8} {'folds':>7} {'conc':>6} {'IC':>9} {'α-med':>8}")
    print(f"{'X8 A2 baseline (control)':<24} {'+1.26':>8} {'5/9':>7} {'74%':>6} "
          f"{'+0.0070':>9} {'10':>8}")
    for r in results:
        if "sharpe" not in r:
            print(f"{r['variant']:<24} ERR")
            continue
        print(f"{r['variant']:<24} {r['sharpe']:>+8.2f} {str(r.get('folds_pos','?')):>7} "
              f"{str(r.get('concentration','?')):>6} {r['train_ic']:>+9.4f} "
              f"{r['alpha_median']:>8.3f}")


if __name__ == "__main__":
    main()
