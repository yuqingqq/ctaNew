"""Diagnose why 111-panel β-residual setup still fails.

Per-symbol IC analysis: split 111-panel predictions into (51 original) and (60 new)
symbol sets. Compare per-symbol pred quality. Is the model worse on the originals
under the bigger panel, or are the new memes the noisy ones?

Also: was the 51-panel WINNER_17 model's preds for the 51 originals better than
the 111-panel WINNER_17 model's preds for the SAME 51 originals?
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

APD_51 = REPO / "outputs/vBTC_winner17_b_residual/51-panel_predictions.parquet"
APD_111 = REPO / "outputs/vBTC_winner17_b_residual/111-panel_predictions.parquet"

OOS_FOLDS = list(range(1, 10))


def per_symbol_ic(apd):
    rows = []
    apd_clean = apd[apd["fold"].isin(OOS_FOLDS)].dropna(subset=["alpha_A","pred"])
    for sym, g in apd_clean.groupby("symbol"):
        if len(g) < 1000: continue
        ic = g["pred"].rank().corr(g["alpha_A"].rank())
        rows.append({"symbol": sym, "n": len(g), "ic": float(ic),
                      "alpha_mean": float(g["alpha_A"].mean()),
                      "alpha_std": float(g["alpha_A"].std()),
                      "pred_mean": float(g["pred"].mean()),
                      "pred_std": float(g["pred"].std())})
    return pd.DataFrame(rows).sort_values("ic", ascending=False)


def main():
    print("=== Diagnose why 111-panel β-residual fails ===\n", flush=True)
    apd51 = pd.read_parquet(APD_51)
    apd111 = pd.read_parquet(APD_111)
    apd51["open_time"] = pd.to_datetime(apd51["open_time"], utc=True)
    apd111["open_time"] = pd.to_datetime(apd111["open_time"], utc=True)

    syms_51 = set(apd51["symbol"].unique())
    syms_111 = set(apd111["symbol"].unique())
    overlap = syms_51 & syms_111
    new_syms = syms_111 - syms_51
    print(f"51-panel: {len(syms_51)} symbols", flush=True)
    print(f"111-panel: {len(syms_111)} symbols", flush=True)
    print(f"overlap (51 originals): {len(overlap)}", flush=True)
    print(f"new in 111: {len(new_syms)}", flush=True)

    # === Per-symbol IC, 51-panel model ===
    print("\n--- Per-symbol IC on 51-panel (WINNER_17 + β-residual) ---", flush=True)
    ic51 = per_symbol_ic(apd51)
    print(f"  51-panel mean IC: {ic51['ic'].mean():+.4f}", flush=True)
    print(f"  median IC: {ic51['ic'].median():+.4f}", flush=True)
    print(f"  positive IC: {(ic51['ic'] > 0).sum()}/{len(ic51)}", flush=True)

    # === Per-symbol IC, 111-panel model ===
    print("\n--- Per-symbol IC on 111-panel (WINNER_17 + β-residual) ---", flush=True)
    ic111 = per_symbol_ic(apd111)
    ic111["is_new"] = ic111["symbol"].isin(new_syms)
    print(f"  111-panel mean IC: {ic111['ic'].mean():+.4f}", flush=True)
    print(f"  median IC: {ic111['ic'].median():+.4f}", flush=True)
    print(f"  positive IC: {(ic111['ic'] > 0).sum()}/{len(ic111)}", flush=True)

    # Split: original 51 vs new 60
    ic111_orig = ic111[~ic111["is_new"]]
    ic111_new = ic111[ic111["is_new"]]
    print(f"\n  111-panel split:")
    print(f"    Original 51 symbols (in both panels): mean IC = {ic111_orig['ic'].mean():+.4f}, "
          f"positive = {(ic111_orig['ic']>0).sum()}/{len(ic111_orig)}", flush=True)
    print(f"    New 60 symbols (in 111 only):         mean IC = {ic111_new['ic'].mean():+.4f}, "
          f"positive = {(ic111_new['ic']>0).sum()}/{len(ic111_new)}", flush=True)

    # === Cross-panel comparison: same 51 symbols, 51-model vs 111-model ===
    print("\n--- Same 51 symbols, 51-model vs 111-model (which model is better per-symbol) ---",
          flush=True)
    merged = ic51[["symbol","ic"]].rename(columns={"ic":"ic_51_model"}).merge(
        ic111_orig[["symbol","ic"]].rename(columns={"ic":"ic_111_model"}),
        on="symbol", how="inner")
    merged["delta_111_minus_51"] = merged["ic_111_model"] - merged["ic_51_model"]
    print(f"\n  Mean Δ IC (111-model − 51-model) on overlapping symbols: "
          f"{merged['delta_111_minus_51'].mean():+.4f}", flush=True)
    print(f"  Symbols where 111-model is BETTER: "
          f"{(merged['delta_111_minus_51']>0).sum()}/{len(merged)}", flush=True)
    print(f"  Symbols where 111-model is WORSE:  "
          f"{(merged['delta_111_minus_51']<0).sum()}/{len(merged)}", flush=True)

    # Top 10 IC drops
    pd.set_option("display.width", 200)
    print(f"\n  Top 10 IC drops (51-model → 111-model):", flush=True)
    print(merged.nsmallest(10, "delta_111_minus_51").to_string(index=False,
          float_format=lambda x: f"{x:+.4f}"), flush=True)

    # Top 10 IC gains
    print(f"\n  Top 10 IC gains (51-model → 111-model):", flush=True)
    print(merged.nlargest(10, "delta_111_minus_51").to_string(index=False,
          float_format=lambda x: f"{x:+.4f}"), flush=True)

    # === Top 10 new symbols by IC (the memes the ranker picks) ===
    print(f"\n  Top 20 new symbols by IC on 111-panel (these are what IC ranker picks):",
          flush=True)
    print(ic111_new.head(20).to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
          flush=True)

    # === Bottom 10 new symbols ===
    print(f"\n  Bottom 10 new symbols by IC on 111-panel:",
          flush=True)
    print(ic111_new.tail(10).to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
          flush=True)

    # Summary
    print("\n" + "="*80)
    print("  SYNTHESIS")
    print("="*80)
    n_orig_pos = (ic111_orig["ic"] > 0).sum()
    n_new_pos = (ic111_new["ic"] > 0).sum()
    print(f"  Original 51 symbols on 111-panel: {n_orig_pos}/{len(ic111_orig)} positive IC, "
          f"mean = {ic111_orig['ic'].mean():+.4f}", flush=True)
    print(f"  New 60 symbols on 111-panel:      {n_new_pos}/{len(ic111_new)} positive IC, "
          f"mean = {ic111_new['ic'].mean():+.4f}", flush=True)
    print(f"  IC redistribution: {(merged['delta_111_minus_51']<0).sum()} of original 51 LOST IC, "
          f"{(merged['delta_111_minus_51']>0).sum()} GAINED IC under 111-model", flush=True)


if __name__ == "__main__":
    main()
