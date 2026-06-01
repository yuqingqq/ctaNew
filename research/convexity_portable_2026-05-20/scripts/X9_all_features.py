"""X9 — All features combined: BASE + aggT + cohort + v3 + crossX = 31 features.

Train 6 architectures (LGBM × 3 + Ridge × 3) on the full feature stack and
see if combining all helps or if redundancy/noise hurts.
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

# Import X6 and X6b modules
spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)


def main():
    t0 = time.time()
    print("=== X9 ALL features combined ===\n", flush=True)

    # Load HL-50 panel with all base + aggT + v3 cols
    HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
    HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE + x6.AGGT_EXTRAS + x6.V3_EXTRAS)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()
    print(f"  HL-50 panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    # Cohort features (fixed version)
    print("  computing cohort features (BTC fix)...", flush=True)
    panel = x6b.build_cohort_fixed(panel)

    # CrossX features (already saved from X7)
    cross_path = REPO / "data/ml/cache/cross_exchange_features.parquet"
    cross_df = pd.read_parquet(cross_path)
    cross_z_cols = [c for c in cross_df.columns if c.endswith("_basis_z")]
    print(f"  merging crossX features ({len(cross_z_cols)}): {cross_z_cols}", flush=True)
    panel = panel.merge(cross_df[["symbol", "open_time"] + cross_z_cols],
                        on=["symbol", "open_time"], how="left")

    # Target
    panel = x6.build_target_z(panel)
    folds = x6.get_folds(panel)

    # Heavy-tail: add cross-exchange and cohort heavy-tails
    for c in cross_z_cols:
        x6.HEAVY_TAIL.add(c)
    # cohort features may have heavy tails too
    for c in x6.COHORT_EXTRAS:
        x6.HEAVY_TAIL.add(c)

    # ALL features combined
    feats_all = x6.BASE + x6.AGGT_EXTRAS + x6.COHORT_EXTRAS + x6.V3_EXTRAS + cross_z_cols
    feats_all = list(dict.fromkeys(feats_all))  # de-dup just in case
    print(f"  ALL features = {len(feats_all)}: {feats_all}", flush=True)

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
        cell_label = f"{model}_{arch}_pall"
        pred_path = REPO / f"research/convexity_portable_2026-05-20/results/_cache/x9_{cell_label}_preds.parquet"
        print(f"\n[ALL {i}/6] {model} | {arch} (features={len(feats_all)})", flush=True)
        tf = time.time()
        # Resume: skip train if predictions already saved
        if pred_path.exists():
            print(f"    pred parquet exists, skipping train: {pred_path.name}", flush=True)
            apd = pd.read_parquet(pred_path)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"    cached: {len(apd):,} rows, IC={ic:+.4f}", flush=True)
        else:
            try:
                apd = fn(panel, folds, feats_all)
                apd.to_parquet(pred_path, index=False)
                ic = float(apd["pred"].corr(apd["alpha_A"]))
                print(f"    trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
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
        else:
            print(f"    sleeve ERR: {m.get('error','?')}", flush=True)

    out_csv = OUT / "X6_controlled_matrix.csv"
    keys = ["cell", "model", "arch", "feature_set", "n_feats",
            "train_ic", "sharpe", "ci_lo", "ci_hi", "totPnL", "maxDD",
            "folds_pos", "concentration", "net_bps_cycle",
            "train_time_s", "error"]
    with open(out_csv, "a", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        for r in new_rows: w.writerow(r)
    print(f"\nAppended {len(new_rows)} ALL-features cells to {out_csv} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
