"""X8b — Retry ElasticNet variants with numerical fixes.

Original X8 A4/A5/A6 failed with Gram matrix instability when using
ElasticNetCV with sym_id one-hot dummies (50 highly-collinear binary columns).

Fixes applied here:
  - precompute=False (use coordinate descent directly, no Gram matrix shortcut)
  - dtype=float64 instead of float32 (higher precision for collinear features)
  - selection='cyclic' (deterministic)
  - max_iter increased
  - explicit alpha grid (skip auto-derivation)

Variants:
  B1: ElasticNet l1_ratio=0.1 (mostly L2 with light L1)
  B2: ElasticNet l1_ratio=0.5 (balanced)
  B3: ElasticNet l1_ratio=0.9 (mostly L1, feature selection)
  B4: Pure Lasso (l1_ratio=1.0)

All on the same best cell: Ridge Pool+symid +aggT (BASE+aggT, 19 features + 50 sym_id one-hot dummies).
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import ElasticNetCV, Lasso, LassoCV

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
    return p


def train_enet(panel, folds, l1_ratio, alpha_grid):
    """ElasticNet with numerical-safe settings, pooled+symid arch."""
    feats = FEATS
    syms_sorted = sorted(panel["symbol"].unique())
    panel = panel.copy()
    sym_dum = pd.get_dummies(panel["symbol"], prefix="sym", drop_first=True).astype(np.float64)
    sym_dum.index = panel.index

    all_preds = []
    alpha_log = []
    sstats0, hstats0 = None, None
    for f, ts, te, ec in folds:
        train = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        if len(train) < 5000 or len(test) < 1000: continue

        if sstats0 is None:
            sstats0, hstats0 = x6.fit_preproc(train, feats)
        # float64 throughout for numerical stability
        Xtr = x6.apply_preproc(train, feats, sstats0, hstats0).astype(np.float64)
        Xte = x6.apply_preproc(test, feats, sstats0, hstats0).astype(np.float64)
        Xtr = np.hstack([Xtr, sym_dum.loc[train.index].to_numpy()])
        Xte = np.hstack([Xte, sym_dum.loc[test.index].to_numpy()])
        ytr = train["target_z"].to_numpy(dtype=np.float64)

        # Subsample for speed (5M rows × 12 alphas × 5-fold inner CV is heavy)
        if len(Xtr) > 200_000:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(Xtr), size=200_000, replace=False)
            Xtr_fit = Xtr[idx]; ytr_fit = ytr[idx]
        else:
            Xtr_fit, ytr_fit = Xtr, ytr

        m = ElasticNetCV(
            l1_ratio=l1_ratio,
            alphas=alpha_grid,
            precompute=False,         # FIX: no Gram matrix
            cv=3,                     # inner CV folds (faster)
            max_iter=3000,
            tol=1e-3,
            selection="cyclic",       # deterministic
            n_jobs=1,
        ).fit(Xtr_fit, ytr_fit)

        pred = m.predict(Xte)
        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred
        out["fold"] = f
        all_preds.append(out)

        # Sparsity check
        n_nonzero = int(np.sum(np.abs(m.coef_) > 1e-8))
        alpha_log.append({"fold": f, "alpha": float(m.alpha_),
                          "l1_ratio": float(m.l1_ratio_),
                          "n_nonzero_coef": n_nonzero,
                          "n_total_coef": int(len(m.coef_))})

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd, alpha_log


def main():
    t0 = time.time()
    print("=== X8b ElasticNet retry with numerical fixes ===\n", flush=True)
    print("Best cell: Ridge Pool+symid +aggT (BASE+aggT)")
    print("X6 baseline (RidgeCV narrow grid): Sharpe +1.22")
    print("X8 A2 (RidgeCV wider grid): Sharpe +1.26")
    print("Now testing ElasticNet variants...\n", flush=True)

    p = load_panel()
    folds = x6.get_folds(p)

    variants = [
        ("B1_enet_01", 0.1),
        ("B2_enet_05", 0.5),
        ("B3_enet_09", 0.9),
        ("B4_lasso",   1.0),   # pure L1
    ]
    results = []
    for variant_name, l1r in variants:
        tf = time.time()
        print(f"\n[{variant_name}] l1_ratio={l1r}", flush=True)
        try:
            apd, alpha_log = train_enet(p, folds, l1r, WIDER_GRID)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            pred_path = CACHE / f"x8b_{variant_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            avg_nonzero = np.mean([a["n_nonzero_coef"] for a in alpha_log])
            total_coef = alpha_log[0]["n_total_coef"]
            print(f"  trained {len(apd):,} rows, IC={ic:+.4f} "
                  f"[{time.time()-tf:.0f}s]", flush=True)
            print(f"  α per fold: {[round(a['alpha'], 4) for a in alpha_log]}", flush=True)
            print(f"  features used (non-zero coef): mean {avg_nonzero:.0f}/{total_coef} "
                  f"= {avg_nonzero/total_coef*100:.0f}% sparsity", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {type(e).__name__}: {e}", flush=True)
            results.append({"variant": variant_name, "l1_ratio": l1r, "error": str(e)})
            continue

        m = x6.run_sleeve_on_preds(pred_path, f"x8b_{variant_name}")
        row = {"variant": variant_name, "l1_ratio": l1r,
               "train_ic": round(ic, 4),
               "train_time_s": round(time.time()-tf, 0),
               "alpha_median": float(np.median([a["alpha"] for a in alpha_log])),
               "avg_nonzero_coef": round(avg_nonzero, 0),
               "total_coef": total_coef, **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)

    # Save
    keys = ["variant", "l1_ratio", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration",
            "alpha_median", "avg_nonzero_coef", "total_coef",
            "train_time_s", "error"]
    out_csv = OUT / "X8b_elasticnet_results.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} variants → {out_csv} [{time.time()-t0:.0f}s]")

    # Print table
    print(f"\n=== X8b ElasticNet results on best cell ===")
    print(f"{'variant':<14} {'l1_ratio':>8} {'Sharpe':>8} {'folds':>7} "
          f"{'IC':>9} {'α-med':>8} {'sparsity':>9}")
    for r in results:
        if "sharpe" not in r:
            print(f"{r['variant']:<14} {r['l1_ratio']:>8} ERR")
            continue
        spars = f"{r['avg_nonzero_coef']:.0f}/{r['total_coef']}"
        print(f"{r['variant']:<14} {r['l1_ratio']:>8} {r['sharpe']:>+8.2f} "
              f"{str(r.get('folds_pos','?')):>7} {r['train_ic']:>+9.4f} "
              f"{r['alpha_median']:>8.3f} {spars:>9}")


if __name__ == "__main__":
    main()
