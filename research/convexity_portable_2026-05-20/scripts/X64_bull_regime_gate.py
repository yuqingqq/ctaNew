"""X64 — Bull-regime gate on V5_minus_v3_7cx.

Hypothesis from X55 LOFO: fold 8 (BULL +18% BTC) HURTS V5_minus_v3_7cx —
removing it lifts Sharpe by +0.10 (HL-50) and +0.43 (HL-70).

Test: skip trading when BTC 30d return > threshold (using PIT-only data).
Variants:
  - Bull-skip @ thresholds {0.05, 0.10, 0.15, 0.20, 0.30}
  - Zero-out predictions in bull regime (sleeve picks "no signal")
  - Compare to baseline V5_minus_v3_7cx

Both HL-50 and HL-70.
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"
KLINES = REPO / "data/ml/test/parquet/klines"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def load_btc_30d_return():
    """Load BTC 5m closes, compute PIT 30d return at each bar."""
    files = sorted((KLINES / "BTCUSDT" / "5m").glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in files],
                     ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.set_index("open_time")["close"].astype(np.float32)
    # 30d = 288*30 = 8640 5m bars
    btc_30d_ret = (btc / btc.shift(8640) - 1).astype(np.float32)
    return btc_30d_ret


def test_bull_gate(preds_path, btc_30d_ret, label, thresholds):
    """Apply bull-skip gate at various thresholds and compute Sharpe."""
    apd = pd.read_parquet(preds_path)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)

    # Baseline (no gate)
    m_base = x6.run_sleeve_on_preds(preds_path, f"x64_{label}_base")
    base_sh = m_base.get("sharpe", 0) or 0
    base_folds = m_base.get("folds_pos", "?")
    print(f"\n[{label}] baseline (no gate): Sharpe={base_sh:+.2f} folds={base_folds}")
    print(f"  {'threshold':<12} {'mode':<10} {'Sharpe':>8} {'cycles kept':>12} {'folds+':>8}")

    # Merge btc_30d_ret onto preds
    btc_30d_df = btc_30d_ret.to_frame("btc_30d_ret").reset_index()
    btc_30d_df["open_time"] = pd.to_datetime(btc_30d_df["open_time"], utc=True)
    apd_with_btc = apd.merge(btc_30d_df, on="open_time", how="left")

    results = []
    for thr in thresholds:
        for mode in ["filter", "zero"]:
            apd_gated = apd_with_btc.copy()
            in_bull = apd_gated["btc_30d_ret"] > thr
            if mode == "filter":
                # Drop bull-regime rows
                apd_gated = apd_gated[~in_bull].drop(columns=["btc_30d_ret"])
            else:  # zero
                # Set predictions to 0 (sleeve will pick near-zero = no clear signal)
                apd_gated.loc[in_bull, "pred"] = 0.0
                apd_gated = apd_gated.drop(columns=["btc_30d_ret"])
            tmp = CACHE / f"x64_{label}_{mode}_t{thr}_preds.parquet"
            apd_gated.to_parquet(tmp, index=False)
            m = x6.run_sleeve_on_preds(tmp, f"x64_{label}_{mode}_t{thr}")
            sh = m.get("sharpe", 0) or 0
            fp = m.get("folds_pos", "?")
            n_cycles = apd_gated["open_time"].nunique() if mode == "filter" else apd["open_time"].nunique()
            print(f"  {thr:<12.2f} {mode:<10} {sh:>+8.2f} {n_cycles:>12,} {str(fp):>8}", flush=True)
            results.append({"label": label, "threshold": thr, "mode": mode,
                             "sharpe": sh, "delta_vs_base": sh - base_sh,
                             "folds_pos": fp, "n_cycles": n_cycles})
    return results


def main():
    print("=== X64 bull-regime gate test ===\n")
    btc_30d = load_btc_30d_return()
    print(f"BTC 30d return range: {btc_30d.min():+.2f} → {btc_30d.max():+.2f}")
    print(f"  median: {btc_30d.median():+.2f}")
    print(f"  pct > 0.15: {(btc_30d > 0.15).mean()*100:.1f}%")
    print(f"  pct > 0.20: {(btc_30d > 0.20).mean()*100:.1f}%")

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.30]
    cases = [
        ("HL50_V5mv3_7cx", CACHE / "x54_V5_minus_v3_7cx_preds.parquet"),
        ("HL70_V5mv3_7cx", CACHE / "x53_V5_minus_v3_preds.parquet"),
    ]
    all_results = []
    for label, path in cases:
        all_results.extend(test_bull_gate(path, btc_30d, label, thresholds))

    import csv
    out_csv = OUT / "X64_bull_gate.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["label","threshold","mode","sharpe","delta_vs_base","folds_pos","n_cycles"])
        w.writeheader()
        for r in all_results: w.writerow(r)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
