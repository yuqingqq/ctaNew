"""X8d — Cohort collinearity diagnosis: WHY does cohort hurt Ridge Pool+symid?

Original X6b Ridge Pool+symid +cohort = -0.72 (vs BASE +0.38, lift -1.10).
X10 C1 normalized sym_id partially fixed: -0.10 (lift +0.62 vs X6b).

Hypothesis: btc_rvol_7d is BROADCAST (same value across all syms at each cycle)
→ collinear with intercept and with sym_id one-hot dummies.

Test 3 variants on Ridge Pool+symid + BASE + cohort:
  D1: Normalize sym_id + ONLY add rvol_7d, ret_3d (per-sym features); DROP btc_rvol_7d
      Tests whether per-sym cohort features alone help (no broadcast).
  D2: Drop sym_id entirely + cohort (Pool-nosym +cohort baseline for reference)
  D3: Per-sym time-varying btc_rvol_7d instead of broadcast (each sym gets its own
      rolling-vol-vs-btc proxy, breaking the broadcast collinearity)
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc, resource
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    # ru_maxrss is in KB on Linux
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
spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)

WIDER_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())


def load_panel():
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()
    panel = x6b.build_cohort_fixed(panel)
    panel = x6.build_target_z(panel)
    for c in panel.columns:
        if panel[c].dtype in ("float64",): panel[c] = panel[c].astype("float32")
    for c in x6.COHORT_EXTRAS:
        x6.HEAVY_TAIL.add(c)
    return panel


def add_per_sym_volatility(panel):
    """Add per-symbol-time-varying analog of btc_rvol_7d: each symbol's own 7d rvol."""
    # We already have idio_vol_to_btc_1d (BTC-residual vol, per-sym). Use that as the
    # per-sym time-varying proxy for the broadcast btc_rvol_7d.
    panel["per_sym_vol_proxy"] = panel["idio_vol_to_btc_1d"].copy()
    return panel


def get_sym_dum_normalized(panel):
    syms_sorted = sorted(panel["symbol"].unique())
    sym_idx = {s: i for i, s in enumerate(syms_sorted)}
    sym_codes = panel["symbol"].map(sym_idx).to_numpy(np.int32)
    n_dum = len(syms_sorted) - 1
    dum = np.zeros((len(panel), n_dum), dtype=np.float32)
    m = sym_codes > 0
    dum[m, sym_codes[m] - 1] = 1.0
    return dum


def train_ridge(panel, folds, feats, use_symid_normalized):
    dum_all = get_sym_dum_normalized(panel) if use_symid_normalized else None
    all_preds, alpha_log = [], []
    sstats0 = hstats0 = None
    for f, ts, te, ec in folds:
        train_mask = ((panel["exit_time"] < ec).to_numpy()
                      & panel["target_z"].notna().to_numpy())
        test_mask = ((panel["open_time"] >= ts)
                     & (panel["open_time"] <= te)).to_numpy()
        train = panel.iloc[train_mask]; test = panel.iloc[test_mask]
        if len(train) < 5000 or len(test) < 1000: continue
        if sstats0 is None:
            sstats0, hstats0 = x6.fit_preproc(train, feats)
        Xtr = x6.apply_preproc(train, feats, sstats0, hstats0).astype(np.float32)
        Xte = x6.apply_preproc(test, feats, sstats0, hstats0).astype(np.float32)
        ytr = train["target_z"].to_numpy(np.float32)
        if use_symid_normalized:
            dum_tr = dum_all[train_mask]; dum_te = dum_all[test_mask]
            mean = dum_tr.mean(axis=0); std = dum_tr.std(axis=0); std[std == 0] = 1.0
            dum_tr_n = ((dum_tr - mean) / std).astype(np.float32)
            dum_te_n = ((dum_te - mean) / std).astype(np.float32)
            X_train = np.hstack([Xtr, dum_tr_n]); X_test = np.hstack([Xte, dum_te_n])
            del dum_tr, dum_te, dum_tr_n, dum_te_n
        else:
            X_train, X_test = Xtr, Xte
        m = RidgeCV(alphas=WIDER_GRID).fit(X_train, ytr)
        pred = m.predict(X_test).astype(np.float32)
        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred; out["fold"] = f
        all_preds.append(out); alpha_log.append(float(m.alpha_))
        del Xtr, Xte, X_train, X_test, ytr, m; gc.collect()
    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    return apd, alpha_log


def main():
    t0 = time.time()
    print("=== X8d cohort collinearity diagnosis ===\n", flush=True)
    print("X6b Ridge Pool+symid +cohort: -0.72  | X10 C1 +cohort: -0.10\n", flush=True)

    panel = load_panel()
    panel = add_per_sym_volatility(panel)
    folds = x6.get_folds(panel)
    print(f"  panel: {len(panel):,} rows", flush=True)

    variants = [
        ("D1_persym_only", x6.BASE + ["rvol_7d", "ret_3d"], True,
         "BASE + rvol_7d + ret_3d (drop broadcast btc_rvol_7d) + normalized sym_id"),
        ("D2_nosym_full",  x6.BASE + x6.COHORT_EXTRAS, False,
         "BASE + cohort (drop sym_id entirely, reference)"),
        ("D3_persym_vol",  x6.BASE + ["rvol_7d", "ret_3d", "per_sym_vol_proxy"], True,
         "BASE + rvol_7d + ret_3d + per_sym_vol_proxy (replace broadcast with per-sym) + norm sym_id"),
    ]

    results = []
    for v_name, feats, use_symid, desc in variants:
        tf = time.time()
        log_mem(f"before {v_name}")
        print(f"\n[{v_name}] {desc} ({len(feats)} feats, use_symid={use_symid})", flush=True)
        try:
            apd, alpha_log = train_ridge(panel, folds, feats, use_symid)
            pred_path = CACHE / f"x8d_{v_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained IC={ic:+.4f} α med={np.median(alpha_log):.1f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {e}"); results.append({"variant": v_name, "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x8d_{v_name}")
        row = {"variant": v_name, "desc": desc, "n_feats": len(feats),
               "use_symid": use_symid, "train_ic": round(ic, 4),
               "alpha_median": float(np.median(alpha_log)),
               "train_time_s": round(time.time()-tf, 0), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        # release variant artifacts before next iteration
        del apd, alpha_log
        gc.collect()
        log_mem(f"after {v_name}")

    keys = ["variant", "desc", "n_feats", "use_symid", "train_ic",
            "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD", "folds_pos", "concentration",
            "alpha_median", "train_time_s", "error"]
    out_csv = OUT / "X8d_cohort_collinearity.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} variants → {out_csv} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
