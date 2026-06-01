"""X66 — Regime-conditional ensemble: switch between V5_minus_v3 and V0 by regime.

V5_minus_v3_7cx: thrives in SIDEWAYS (folds 1,2,3 are alpha contributors)
V0 (BASE+cohort): point Sharpe +2.01 but uncertain bootstrap

Test: use V5_minus_v3 in non-bull, V0 in bull (or vice versa).
Also test: V5_minus_v3 in sideways/low-vol, V0 in everything else.
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


def load_btc_metrics():
    files = sorted((KLINES / "BTCUSDT" / "5m").glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in files],
                     ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.set_index("open_time")["close"].astype(np.float32)
    ret_30d = (btc / btc.shift(8640) - 1).astype(np.float32)
    log_ret = np.log(btc / btc.shift(1))
    rv_30d = (log_ret.rolling(8640, min_periods=288).std() * np.sqrt(8640)).astype(np.float32)
    return pd.DataFrame({"btc_30d_ret": ret_30d, "btc_rv_30d": rv_30d}).reset_index()


def main():
    print("=== X66 regime-conditional ensemble ===\n")
    btc_metrics = load_btc_metrics()
    btc_metrics["open_time"] = pd.to_datetime(btc_metrics["open_time"], utc=True)

    # Load both V0 and V5_minus_v3_7cx predictions on canonical HL-50
    v0 = pd.read_parquet(CACHE / "x29_V0_BASE_cohort_v2_preds.parquet")
    v5mv3 = pd.read_parquet(CACHE / "x54_V5_minus_v3_7cx_preds.parquet")
    v0["open_time"] = pd.to_datetime(v0["open_time"], utc=True)
    v5mv3["open_time"] = pd.to_datetime(v5mv3["open_time"], utc=True)
    print(f"V0: {len(v0):,} rows")
    print(f"V5mv3: {len(v5mv3):,} rows")

    # Normalize predictions per-fold for fair switching
    v0["pred_n"] = v0.groupby("fold")["pred"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    v5mv3["pred_n"] = v5mv3.groupby("fold")["pred"].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

    merged = v5mv3[["symbol","open_time","alpha_A","return_pct","exit_time","fold","pred_n"]].rename(
        columns={"pred_n": "pred_v5mv3"})
    merged = merged.merge(v0[["symbol","open_time","pred_n"]].rename(columns={"pred_n": "pred_v0"}),
                            on=["symbol","open_time"], how="left")
    merged = merged.merge(btc_metrics, on="open_time", how="left")
    print(f"  merged: {len(merged):,} rows")

    # Baselines
    m_v0 = x6.run_sleeve_on_preds(CACHE / "x29_V0_BASE_cohort_v2_preds.parquet", "x66_v0")
    m_v5 = x6.run_sleeve_on_preds(CACHE / "x54_V5_minus_v3_7cx_preds.parquet", "x66_v5mv3")
    print(f"\nBaselines:")
    print(f"  V0 alone: Sharpe={m_v0.get('sharpe',0):+.2f}")
    print(f"  V5_minus_v3_7cx alone: Sharpe={m_v5.get('sharpe',0):+.2f}")

    # Variants: switch by BTC trend
    print(f"\n{'config':<30} {'Sharpe':>8} {'folds+':>8} {'conc':>6}")
    results = []
    for thr_bull in [0.10, 0.15, 0.20]:
        for cond_name, condition_func in [
            ("V5_sideways_V0_bull", lambda m: m["btc_30d_ret"] > thr_bull),  # use V0 in bull
            ("V0_sideways_V5_bull", lambda m: m["btc_30d_ret"] <= thr_bull),  # use V5 in bull (opposite)
        ]:
            m2 = merged.copy()
            cond = condition_func(m2)
            # When cond=True, use V0; else V5mv3
            m2["pred"] = np.where(cond, m2["pred_v0"], m2["pred_v5mv3"]).astype(np.float32)
            apd = m2[["symbol","open_time","alpha_A","return_pct","exit_time","pred","fold"]].copy()
            label = f"{cond_name}_t{thr_bull}"
            tmp = CACHE / f"x66_{label}_preds.parquet"
            apd.to_parquet(tmp, index=False)
            mm = x6.run_sleeve_on_preds(tmp, f"x66_{label}")
            sh = mm.get("sharpe", 0) or 0
            fp = mm.get("folds_pos", "?")
            conc = mm.get("concentration", "?")
            print(f"  {label:<30} {sh:>+8.2f} {str(fp):>8} {str(conc):>6}", flush=True)
            results.append({"config": label, "sharpe": sh, "folds_pos": fp, "concentration": conc})

    import csv
    out_csv = OUT / "X66_regime_conditional_ensemble.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["config","sharpe","folds_pos","concentration"])
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
