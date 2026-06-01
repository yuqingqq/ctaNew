"""X8e — LGBM regularization sweep on best LGBM cell (Pool+symid +aggT, X6 -0.63).

Test 4 variants:
  E1: Early stopping (10% train→val split, stop_rounds=30)
  E2: Adaptive min_data_in_leaf (max(100, n_train // 100))
  E3: Higher reg_alpha=1.0, reg_lambda=1.0 (vs default 0.1)
  E4: Combined E1+E2+E3
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc, resource
from pathlib import Path
import pandas as pd, numpy as np
import lightgbm as lgb


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)
    return rss_mb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

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
        if p[c].dtype in ("float64",): p[c] = p[c].astype("float32")
    syms_sorted = sorted(p["symbol"].unique())
    sym_map = {s: i for i, s in enumerate(syms_sorted)}
    p["sym_id"] = p["symbol"].map(sym_map).astype("int32")
    return p


def train_lgbm_variant(panel, folds, variant):
    feats_with_symid = FEATS + ["sym_id"]
    all_preds = []
    for f, ts, te, ec in folds:
        train = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        if len(train) < 5000 or len(test) < 1000: continue

        # Variant-specific params
        params = dict(
            objective="regression", metric="rmse", learning_rate=0.03,
            num_leaves=31, max_depth=6,
            feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
            reg_alpha=0.1, reg_lambda=0.1, verbose=-1, n_estimators=400,
        )
        if variant == "E1_early_stop":
            params["min_data_in_leaf"] = 300
        elif variant == "E2_adaptive_leaf":
            params["min_data_in_leaf"] = max(100, len(train) // 100)
        elif variant == "E3_higher_reg":
            params["min_data_in_leaf"] = 300
            params["reg_alpha"] = 1.0
            params["reg_lambda"] = 1.0
        elif variant == "E4_combined":
            params["min_data_in_leaf"] = max(100, len(train) // 100)
            params["reg_alpha"] = 1.0
            params["reg_lambda"] = 1.0

        Xtr = train[feats_with_symid]
        ytr = train["target_z"].to_numpy(np.float32)
        Xte = test[feats_with_symid]

        if variant in ("E1_early_stop", "E4_combined"):
            # split off last 10% of train as val for early stopping
            n_train = len(Xtr)
            val_start = int(n_train * 0.9)
            Xv = Xtr.iloc[val_start:]
            yv = ytr[val_start:]
            Xt = Xtr.iloc[:val_start]
            yt = ytr[:val_start]
            m = lgb.LGBMRegressor(random_state=20260520, **params)
            m.fit(Xt, yt, eval_set=[(Xv, yv)], categorical_feature=["sym_id"],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        else:
            m = lgb.LGBMRegressor(random_state=20260520, **params)
            m.fit(Xtr, ytr, categorical_feature=["sym_id"])

        pred = m.predict(Xte).astype(np.float32)
        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred; out["fold"] = f
        all_preds.append(out)
        print(f"      fold {f}: best_iter={m.best_iteration_ or params['n_estimators']}, "
              f"n_train={len(train):,}", flush=True)
        del m, Xtr, Xte, ytr; gc.collect()
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd


def main():
    t0 = time.time()
    print("=== X8e LGBM regularization sweep ===\n", flush=True)
    print("Best LGBM cell: Pool+symid +aggT, X6 Sharpe -0.63\n", flush=True)
    panel = load_panel()
    folds = x6.get_folds(panel)
    print(f"  panel: {len(panel):,} rows", flush=True)

    variants = [
        ("E1_early_stop",   "Early stopping (10% val, stop_rounds=30)"),
        ("E2_adaptive_leaf", "Adaptive min_data_in_leaf = max(100, n_train//100)"),
        ("E3_higher_reg",   "reg_alpha=1.0, reg_lambda=1.0 (vs default 0.1)"),
        ("E4_combined",     "E1 + E2 + E3 combined"),
    ]

    results = []
    for v_name, desc in variants:
        tf = time.time()
        log_mem(f"before {v_name}")
        print(f"\n[{v_name}] {desc}", flush=True)
        try:
            apd = train_lgbm_variant(panel, folds, v_name)
            pred_path = CACHE / f"x8e_{v_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {type(e).__name__}: {e}"); results.append({"variant": v_name, "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x8e_{v_name}")
        row = {"variant": v_name, "desc": desc, "train_ic": round(ic, 4),
               "train_time_s": round(time.time()-tf, 0), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        del apd
        gc.collect()
        log_mem(f"after {v_name}")

    keys = ["variant", "desc", "train_ic", "sharpe", "ci_lo", "ci_hi", "totPnL",
            "maxDD", "folds_pos", "concentration", "train_time_s", "error"]
    out_csv = OUT / "X8e_lgbm_reg_sweep.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} variants → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\nReference: X6 LGBM Pool+symid +aggT = -0.63")
    for r in results:
        if "sharpe" in r:
            print(f"  {r['variant']:<22} Sharpe={r['sharpe']:+.2f}  IC={r['train_ic']:+.4f}")


if __name__ == "__main__":
    main()
