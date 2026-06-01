"""X6b — fill the 6 +cohort cells that errored in X6 due to BTC closes bug.

Fix: load BTC closes separately for btc_rvol_7d computation, not from HL-50 closes.

After X6 main run completes (or even during), runs the same 6 architectures
on BASE + cohort features and appends to X6_controlled_matrix.csv.
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
KLINES = REPO / "data/ml/test/parquet/klines"

def build_cohort_fixed(panel):
    """Compute rvol_7d (per-sym), ret_3d (per-sym), btc_rvol_7d (broadcast).
    KEY FIX: load BTC closes separately even though BTC isn't in panel."""
    # Per-symbol from panel syms
    syms_in_panel = list(panel["symbol"].unique())
    # ALSO include BTCUSDT for the broadcast feature
    syms_to_load = sorted(set(syms_in_panel) | {"BTCUSDT"})

    closes = {}
    for sym in syms_to_load:
        sd = KLINES / sym / "5m"
        if not sd.exists(): continue
        dfs = []
        for f in sorted(sd.glob("*.parquet")):
            try:
                df = pd.read_parquet(f, columns=["open_time", "close"])
                dfs.append(df)
            except Exception: pass
        if not dfs: continue
        c = pd.concat(dfs, ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
        c["open_time"] = pd.to_datetime(c["open_time"], utc=True)
        closes[sym] = c.set_index("open_time")["close"]
    print(f"  loaded closes for {len(closes)} syms (incl BTC={('BTCUSDT' in closes)})", flush=True)

    rows_rvol, rows_ret = [], []
    for sym, c in closes.items():
        if sym == "BTCUSDT": continue  # handled separately for broadcast
        logret = np.log(c / c.shift(1))
        rv = logret.rolling(288*7, min_periods=288).std().shift(1)
        rt = c.pct_change(288*3).shift(1)
        df_r = rv.rename("rvol_7d").reset_index(); df_r["symbol"] = sym
        df_t = rt.rename("ret_3d").reset_index(); df_t["symbol"] = sym
        rows_rvol.append(df_r); rows_ret.append(df_t)
    rvol = pd.concat(rows_rvol, ignore_index=True)
    ret3 = pd.concat(rows_ret, ignore_index=True)

    btc = closes.get("BTCUSDT")
    if btc is None:
        raise RuntimeError("BTCUSDT closes not available — fix loading first")
    btc_rvol_series = np.log(btc / btc.shift(1)).rolling(288*7, min_periods=288).std().shift(1)
    btc_rvol = btc_rvol_series.rename("btc_rvol_7d").reset_index()

    panel = panel.merge(rvol, on=["symbol", "open_time"], how="left")
    panel = panel.merge(ret3, on=["symbol", "open_time"], how="left")
    panel = panel.merge(btc_rvol, on="open_time", how="left")
    print(f"  cohort merge done: rvol_7d non-null={panel['rvol_7d'].notna().mean()*100:.1f}%, "
          f"ret_3d {panel['ret_3d'].notna().mean()*100:.1f}%, "
          f"btc_rvol_7d {panel['btc_rvol_7d'].notna().mean()*100:.1f}%", flush=True)
    return panel


def main():
    t0 = time.time()
    print("=== X6b cohort fill (6 cells) ===\n", flush=True)

    # Import X6 module to reuse functions
    spec = importlib.util.spec_from_file_location("x6",
        REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
    x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

    # Build HL-50 panel with cohort features (fixed)
    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())
    needed_cols = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
                   + x6.BASE)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed_cols)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()
    print(f"  HL-50 panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    print("  computing cohort features (BTC fix)...", flush=True)
    panel = build_cohort_fixed(panel)
    panel = x6.build_target_z(panel)
    folds = x6.get_folds(panel)

    feats_cohort = x6.BASE + x6.COHORT_EXTRAS
    print(f"\n  cohort feature set: {len(feats_cohort)} features", flush=True)

    archs = [
        ("LGBM", "pool+symid", lambda p, f, fs: x6.train_pooled_lgbm(p, f, fs, with_symid=True)),
        ("LGBM", "pool-nosym", lambda p, f, fs: x6.train_pooled_lgbm(p, f, fs, with_symid=False)),
        ("LGBM", "per-sym", lambda p, f, fs: x6.train_per_sym_lgbm(p, f, fs)),
        ("Ridge", "pool+symid", lambda p, f, fs: x6.train_pooled_ridge(p, f, fs, with_symid=True)),
        ("Ridge", "pool-nosym", lambda p, f, fs: x6.train_pooled_ridge(p, f, fs, with_symid=False)),
        ("Ridge", "per-sym", lambda p, f, fs: x6.train_per_sym_ridge(p, f, fs)),
    ]

    new_rows = []
    for i, (model, arch, fn) in enumerate(archs, 1):
        cell_label = f"{model}_{arch}_pcohort"
        print(f"\n[cohort {i}/{len(archs)}] {model} | {arch}", flush=True)
        tf = time.time()
        try:
            apd = fn(panel, folds, feats_cohort)
            pred_path = REPO / f"research/convexity_portable_2026-05-20/results/_cache/x6b_{cell_label}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"    trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"    TRAIN ERR: {e}", flush=True)
            new_rows.append({"cell": cell_label, "model": model, "arch": arch,
                              "feature_set": "+cohort", "error": str(e)})
            continue
        m = x6.run_sleeve_on_preds(pred_path, cell_label)
        row = {"cell": cell_label, "model": model, "arch": arch,
               "feature_set": "+cohort", "n_feats": len(feats_cohort),
               "train_ic": round(ic, 4), "train_time_s": round(time.time()-tf, 0), **m}
        new_rows.append(row)
        if "sharpe" in m:
            print(f"    sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)

    out_csv = OUT / "X6_controlled_matrix.csv"
    keys = ["cell", "model", "arch", "feature_set", "n_feats",
            "train_ic", "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD",
            "folds_pos", "concentration", "net_bps_cycle",
            "train_time_s", "error"]
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        if write_header: w.writeheader()
        for r in new_rows: w.writerow(r)
    print(f"\nAppended {len(new_rows)} cohort cells [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
