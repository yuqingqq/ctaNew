"""X72 — Apply bull-regime gate to the 3-year V0 predictions.

X70 found V0 unconditional over 3 years = +0.12 (bull-dominated sample).
Question: does gating out bull regimes (BTC 30d > threshold) rescue it to a
tradeable Sharpe, confirming the strategy is a viable sideways/bear play with
a regime filter?

Tests:
  - bull-zero gate (set pred=0 when BTC 30d ret > thr) at {0.05, 0.10, 0.15, 0.20}
  - Reports Sharpe + cycles-gated fraction
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


def main():
    print("=== X72 bull-regime gate on 3-year V0 ===\n")
    # BTC 30d return (PIT)
    files = sorted((KLINES/"BTCUSDT"/"5m").glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in files],
                     ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.set_index("open_time")["close"].astype(np.float32)
    btc_30d = (btc/btc.shift(8640)-1).astype(np.float32)
    btc_df = btc_30d.to_frame("btc_30d_ret").reset_index()
    btc_df["open_time"] = pd.to_datetime(btc_df["open_time"], utc=True)

    apd = pd.read_parquet(CACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    print(f"3-year V0 preds: {len(apd):,} rows, {apd['fold'].nunique()} folds")

    m0 = x6.run_sleeve_on_preds(CACHE/"x70_v0_3yr_preds.parquet", "x72_base")
    base = m0.get("sharpe",0) or 0
    print(f"Baseline (no gate): Sharpe={base:+.2f} folds={m0.get('folds_pos','?')} conc={m0.get('concentration','?')}\n")

    apd_b = apd.merge(btc_df, on="open_time", how="left")
    print(f"{'thr':<8} {'mode':<8} {'Sharpe':>8} {'folds+':>8} {'gated%':>8}")
    results = []
    for thr in [0.05, 0.10, 0.15, 0.20]:
        for mode in ["zero","filter"]:
            a = apd_b.copy()
            in_bull = a["btc_30d_ret"] > thr
            gated_pct = in_bull.mean()*100
            if mode == "zero":
                a.loc[in_bull, "pred"] = 0.0
                a = a.drop(columns=["btc_30d_ret"])
            else:
                a = a[~in_bull].drop(columns=["btc_30d_ret"])
            tmp = CACHE/f"x72_{mode}_t{thr}_preds.parquet"
            a.to_parquet(tmp, index=False)
            m = x6.run_sleeve_on_preds(tmp, f"x72_{mode}_t{thr}")
            sh = m.get("sharpe",0) or 0
            fp = m.get("folds_pos","?")
            print(f"{thr:<8.2f} {mode:<8} {sh:>+8.2f} {str(fp):>8} {gated_pct:>7.1f}%", flush=True)
            results.append({"thr":thr,"mode":mode,"sharpe":sh,"folds":fp,"gated_pct":gated_pct})

    print(f"\nBaseline 3yr V0 = {base:+.2f}. If gate lifts to >+1, strategy is a")
    print(f"viable sideways/bear play with a regime filter (key production insight).")


if __name__ == "__main__":
    main()
