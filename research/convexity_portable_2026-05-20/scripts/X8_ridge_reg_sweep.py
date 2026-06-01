"""X8 — Ridge regularization sweep on the BEST X6 cell (Pool+symid +aggT).

Tests 6 variants on Ridge Pool+symid +aggT (X6 baseline +1.22 Sharpe):
  A1: baseline — α grid {0.01, 0.1, 1, 10, 100}, sym_id penalized
  A2: wider α grid {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300}
  A3: A2 + free sym_id intercepts (target demean per-sym before Ridge)
  A4: ElasticNet l1_ratio=0.1, wider α grid
  A5: ElasticNet l1_ratio=0.5, wider α grid
  A6: ElasticNet l1_ratio=0.9, wider α grid

For each variant: train 9 folds, log α (or l1_ratio) picked per fold, run sleeve,
report Sharpe / folds+ / conc / IC.
"""
from __future__ import annotations
import csv, sys, time, warnings, io, contextlib, importlib.util
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV, ElasticNetCV, Ridge, ElasticNet

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"
CACHE.mkdir(parents=True, exist_ok=True)

# Import X6 module
spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

# Variant alpha grids
NARROW_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
WIDER_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]

FEATS = x6.BASE + x6.AGGT_EXTRAS   # the best cell's features (19 total)
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


def train_variant(panel, folds, variant_name, alpha_grid, free_symid_intercept,
                   l1_ratio=None):
    """Train Ridge or ElasticNet across folds, with optional free per-sym intercepts."""
    feats = FEATS
    syms_sorted = sorted(panel["symbol"].unique())
    sym_idx = {s: i for i, s in enumerate(syms_sorted)}
    panel = panel.copy()
    panel["sym_id"] = panel["symbol"].map(sym_idx).astype("int32")

    if not free_symid_intercept:
        sym_dum = pd.get_dummies(panel["symbol"], prefix="sym", drop_first=True).astype(np.float32)
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
        Xtr = x6.apply_preproc(train, feats, sstats0, hstats0)
        Xte = x6.apply_preproc(test, feats, sstats0, hstats0)

        if free_symid_intercept:
            # Compute per-sym mean of target_z on training data
            sym_means_tr = train.groupby("symbol")["target_z"].mean()
            train_demean = train["target_z"].to_numpy() - train["symbol"].map(sym_means_tr).to_numpy()
            ytr_use = train_demean
            sym_means_for_pred = sym_means_tr.to_dict()
        else:
            # Append sym_id one-hot dummies
            Xtr = np.hstack([Xtr, sym_dum.loc[train.index].to_numpy()])
            Xte = np.hstack([Xte, sym_dum.loc[test.index].to_numpy()])
            ytr_use = train["target_z"].to_numpy()
            sym_means_for_pred = None

        # Train
        if l1_ratio is None:
            m = RidgeCV(alphas=alpha_grid).fit(Xtr, ytr_use)
            picked_alpha = float(m.alpha_)
        else:
            m = ElasticNetCV(l1_ratio=l1_ratio, alphas=alpha_grid,
                              max_iter=2000, tol=1e-3, n_jobs=1).fit(Xtr, ytr_use)
            picked_alpha = float(m.alpha_)

        pred = m.predict(Xte)
        if free_symid_intercept:
            # Add back per-symbol mean for OOS prediction
            test_syms = test["symbol"].to_numpy()
            pred = pred + np.array([sym_means_for_pred.get(s, 0.0) for s in test_syms])

        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred
        out["fold"] = f
        all_preds.append(out)
        alpha_log.append({"fold": f, "alpha": picked_alpha, "l1_ratio": l1_ratio,
                          "n_train": int(len(train)), "n_test": int(len(test))})

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd, alpha_log


def main():
    t0 = time.time()
    print("=== X8 Ridge regularization sweep ===\n", flush=True)
    print(f"Test cell: Ridge | Pool+symid | +aggT (BASE+5 aggTrades, {len(FEATS)} features)")
    print(f"X6 baseline: Sharpe +1.22, 5/9 folds, 76% conc\n", flush=True)

    p = load_panel()
    folds = x6.get_folds(p)
    print(f"Panel: {len(p):,} rows, {p['symbol'].nunique()} syms, 9 folds", flush=True)

    variants = [
        ("A1_baseline",     NARROW_GRID, False, None),
        ("A2_wider_alpha",  WIDER_GRID,  False, None),
        ("A3_wider_freesymid", WIDER_GRID, True, None),
        ("A4_enet_01",      WIDER_GRID,  False, 0.1),
        ("A5_enet_05",      WIDER_GRID,  False, 0.5),
        ("A6_enet_09",      WIDER_GRID,  False, 0.9),
    ]

    results = []
    for variant_name, alpha_grid, free_int, l1r in variants:
        tf = time.time()
        print(f"\n[{variant_name}]", flush=True)
        try:
            apd, alpha_log = train_variant(p, folds, variant_name, alpha_grid, free_int, l1r)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            pred_path = CACHE / f"x8_{variant_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            print(f"  trained {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]")
            print(f"  α picked per fold: {[round(a['alpha'], 4) for a in alpha_log]}")
        except Exception as e:
            print(f"  TRAIN ERR: {type(e).__name__}: {e}")
            results.append({"variant": variant_name, "error": str(e)})
            continue

        # Sleeve
        m = x6.run_sleeve_on_preds(pred_path, f"x8_{variant_name}")
        row = {"variant": variant_name,
               "alpha_grid_size": len(alpha_grid),
               "free_symid_intercept": free_int,
               "l1_ratio": l1r,
               "train_ic": round(ic, 4),
               "train_time_s": round(time.time()-tf, 0),
               "alpha_picks_min": min(a["alpha"] for a in alpha_log),
               "alpha_picks_max": max(a["alpha"] for a in alpha_log),
               "alpha_picks_median": np.median([a["alpha"] for a in alpha_log]),
               **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}")
        else:
            print(f"  sleeve ERR: {m.get('error','?')}")

    # Save
    keys = ["variant", "alpha_grid_size", "free_symid_intercept", "l1_ratio",
            "train_ic", "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD",
            "folds_pos", "concentration",
            "alpha_picks_min", "alpha_picks_max", "alpha_picks_median",
            "train_time_s", "error"]
    out_csv = OUT / "X8_ridge_reg_sweep.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} variants → {out_csv} [{time.time()-t0:.0f}s]")

    # Print table
    print(f"\n=== X8 Ridge regularization sweep — best cell (Pool+symid +aggT) ===")
    print(f"{'variant':<26} {'Sharpe':>8} {'folds':>7} {'conc':>6} {'IC':>9} "
          f"{'α-med':>8} {'α-min':>8} {'α-max':>8}")
    for r in results:
        if "sharpe" not in r:
            print(f"{r['variant']:<26} ERR")
            continue
        print(f"{r['variant']:<26} {r['sharpe']:>+8.2f} {str(r.get('folds_pos','?')):>7} "
              f"{str(r.get('concentration','?')):>6} {r['train_ic']:>+9.4f} "
              f"{r['alpha_picks_median']:>8.3f} {r['alpha_picks_min']:>8.3f} "
              f"{r['alpha_picks_max']:>8.3f}")


if __name__ == "__main__":
    main()
