"""X59 — Per-regime Sharpe for V5_minus_v3_7cx on current sample.

Uses X58_regime_labels.parquet to segment cycles by regime.
Then runs sleeve eval separately per regime to see model performance.
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
spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def main():
    print("=== X59 per-regime evaluation ===\n")
    # Load champion predictions on canonical HL-50
    preds = pd.read_parquet(CACHE / "x54_V5_minus_v3_7cx_preds.parquet")
    preds["open_time"] = pd.to_datetime(preds["open_time"], utc=True)
    print(f"V5_minus_v3_7cx HL-50 preds: {len(preds):,} rows")

    regimes = pd.read_parquet(OUT / "X58_regime_labels.parquet")
    regimes["open_time"] = pd.to_datetime(regimes["open_time"], utc=True)
    print(f"Regime labels: {len(regimes):,} cycles")

    # Merge regime onto preds by open_time (round to 4h)
    preds_4h = preds[(preds["open_time"].dt.hour % 4 == 0) & (preds["open_time"].dt.minute == 0)].copy()
    print(f"Preds at 4h-aligned bars: {len(preds_4h):,} rows")
    m = preds_4h.merge(regimes[["open_time", "trend_30d", "vol_regime", "drawdown", "month"]],
                        on="open_time", how="left")

    # Per-regime sleeve eval
    # We'll just subset preds by regime and compute simple PnL stats
    # Full sleeve is expensive — use lightweight per-regime mean PnL × pred sign
    print(f"\nLightweight per-regime alpha (pred × alpha_A mean by regime):")
    for col, label in [("trend_30d", "Trend"), ("vol_regime", "Vol"),
                         ("drawdown", "DD"), ("month", "Month")]:
        print(f"\n  by {label}:")
        if col not in m.columns:
            print(f"    {col} missing")
            continue
        grp = m.dropna(subset=[col]).groupby(col)
        for k, g in grp:
            if len(g) < 100: continue
            ic = g["pred"].corr(g["alpha_A"]) if g["pred"].std() > 0 else np.nan
            mean_pnl = (np.sign(g["pred"]) * g["alpha_A"]).mean() * 10000  # bps
            n_cycles = g["open_time"].nunique()
            print(f"    {k}: n_cycles={n_cycles}, IC={ic:+.4f}, mean_signed_alpha={mean_pnl:+.2f} bps")

    # Save merged
    out_path = OUT / "X59_per_regime_preds.parquet"
    m.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # Full per-regime sleeve via prediction subsetting (run sleeve on regime subsets)
    # We can do this for the main regime split (trend_30d) since it has enough data per regime
    print(f"\n=== Full sleeve per trend_30d regime ===")
    for trend in ["bull", "sideways", "bear"]:
        sub = m[m["trend_30d"] == trend]
        if len(sub) < 1000: continue
        # need full pred file with fold and exit_time
        sub_full = preds[preds["open_time"].isin(sub["open_time"])]
        if len(sub_full) < 100: continue
        tmp = CACHE / f"x59_regime_{trend}_preds.parquet"
        sub_full.to_parquet(tmp, index=False)
        sm = x6.run_sleeve_on_preds(tmp, f"x59_regime_{trend}")
        sh = sm.get("sharpe")
        fp = sm.get("folds_pos", "?")
        conc = sm.get("concentration", "?")
        sh_str = f"{sh:+.2f}" if isinstance(sh, (int, float)) else "?"
        n_cycles = sub["open_time"].nunique()
        print(f"  {trend}: n_cycles={n_cycles}, Sharpe={sh_str}, folds={fp}, conc={conc}")


if __name__ == "__main__":
    main()
