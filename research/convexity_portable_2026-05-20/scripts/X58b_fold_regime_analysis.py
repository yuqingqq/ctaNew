"""X58b — Classify each of the 9 folds by BTC regime + map LOFO results.

For each fold's OOS window:
  - BTC return (start → end)
  - BTC realized vol (annualized)
  - BTC max drawdown within fold
  - Funding regime average

Then cross-reference with X55 LOFO findings to see which regimes are
load-bearing for V5_minus_v3_7cx.
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "research/convexity_portable_2026-05-20/results"
KLINES = REPO / "data/ml/test/parquet/klines"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def main():
    print("=== X58b fold regime analysis ===\n")
    # Load BTC closes
    files = sorted((KLINES / "BTCUSDT" / "5m").glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in files],
                     ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.set_index("open_time")["close"].astype(np.float32)

    # Get folds from x6
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet",
                            columns=["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"])
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"] != "BTCUSDT"]
    panel = x6.build_target_z(panel)
    folds = x6.get_folds(panel)

    print(f"\n{'Fold':<5} {'Start':<22} {'End':<22} {'BTC ret':>9} {'BTC ann vol':>13} {'Max DD':>9} {'Trend':>10} {'Vol':>6}")
    print("-" * 110)
    fold_info = []
    for f, ts, te, ec in folds:
        # BTC return over fold window
        btc_fold = btc.loc[ts:te]
        if len(btc_fold) < 10: continue
        ret = (btc_fold.iloc[-1] / btc_fold.iloc[0] - 1)
        # annualized vol
        log_ret = np.log(btc_fold / btc_fold.shift(1)).dropna()
        ann_vol = log_ret.std() * np.sqrt(288 * 365)  # 5m bars × 365 days
        # max DD within fold
        cum = btc_fold / btc_fold.cummax()
        max_dd = cum.min() - 1
        # Classify
        trend = "BULL" if ret > 0.10 else "BEAR" if ret < -0.10 else "SIDE"
        vol = "HIGH" if ann_vol > 0.60 else "LOW" if ann_vol < 0.35 else "MID"
        print(f"  {f:<3} {str(ts)[:19]:<22} {str(te)[:19]:<22} "
              f"{ret*100:>+8.1f}% {ann_vol*100:>11.0f}% {max_dd*100:>+8.1f}% "
              f"{trend:>10} {vol:>6}")
        fold_info.append({
            "fold": f, "start": ts, "end": te,
            "btc_ret": ret, "btc_ann_vol": ann_vol, "max_dd": max_dd,
            "trend": trend, "vol_regime": vol,
        })

    df = pd.DataFrame(fold_info)
    df.to_csv(OUT / "X58b_fold_regimes.csv", index=False)

    # Cross-reference with V5_minus_v3_7cx LOFO findings
    print("\n\n=== Cross-reference with V5_minus_v3_7cx LOFO ===")
    print("\nHL-50 LOFO (baseline +1.74):")
    hl50_lofo = {
        0: None,  # not yet computed
        1: None,
        2: None,
        3: 0.26,   # Δ -1.48
        4: None,
        5: 1.03,   # Δ -0.71
        6: None,
        7: 1.26,   # Δ -0.48
        8: None,
    }
    for _, row in df.iterrows():
        f = row["fold"]
        sh_after_drop = hl50_lofo.get(f)
        delta = (sh_after_drop - 1.74) if sh_after_drop is not None else None
        delta_str = f"{delta:+.2f}" if delta is not None else "?"
        print(f"  Fold {f} ({row['trend']:>4} / {row['vol_regime']:>4}, "
              f"BTC ret {row['btc_ret']*100:+.1f}%, vol {row['btc_ann_vol']*100:.0f}%): "
              f"drop → Sharpe {sh_after_drop if sh_after_drop else '?':>5}, Δ {delta_str}")

    print("\nHL-70 LOFO (baseline +1.67):")
    hl70_lofo = {
        0: None,
        1: None,
        2: -0.67,  # Δ -2.34
        3: -1.23,  # Δ -2.90
        4: -0.63,  # Δ -2.30
        5: None,
        6: 0.38,   # Δ -1.29
        7: 1.41,   # Δ -0.26
        8: None,
    }
    for _, row in df.iterrows():
        f = row["fold"]
        sh_after_drop = hl70_lofo.get(f)
        delta = (sh_after_drop - 1.67) if sh_after_drop is not None else None
        delta_str = f"{delta:+.2f}" if delta is not None else "?"
        print(f"  Fold {f} ({row['trend']:>4} / {row['vol_regime']:>4}, "
              f"BTC ret {row['btc_ret']*100:+.1f}%, vol {row['btc_ann_vol']*100:.0f}%): "
              f"drop → Sharpe {sh_after_drop if sh_after_drop else '?':>5}, Δ {delta_str}")

    print(f"\nSaved → {OUT / 'X58b_fold_regimes.csv'}")


if __name__ == "__main__":
    main()
