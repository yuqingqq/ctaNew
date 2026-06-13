"""Phase E4 v2: build expanded panel STREAMING (one symbol at a time).

Key insight: keep the existing 51-symbol basket FIXED. Compute basket-relative
features for the new 60 against the existing basket. This is also operationally
correct: in production, the reference basket shouldn't morph each time we
consider adding a new token.

Process:
  1. Reconstruct basket from existing 51 (close-only, low memory)
  2. For each of 60 new symbols:
     - Load xs_feats cache
     - Add basket-relative features (vs existing basket)
     - Add funding features
     - Compute alpha_A target
     - Reduce to needed columns
     - Save per-symbol intermediate parquet
  3. Load existing 51-panel; concat 60 new; save expanded panel.

Memory: only one symbol's features in memory at a time.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import gc

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2] if __file__.endswith("expansion_phase_e4_v2_streaming.py") else Path(__file__).resolve().parents[1]
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import (
    build_kline_features, build_basket, add_basket_features,
)
from features_ml.funding_features import add_funding_features

OUT_DIR = REPO / "outputs/vBTC_features_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)
INTERMEDIATE_DIR = OUT_DIR / "per_symbol_new"
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48


def main():
    print(f"=== Phase E4 v2: streaming expanded panel build ===\n", flush=True)

    # Existing panel
    existing_panel = pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
    )
    existing_syms = sorted(existing_panel["symbol"].unique())
    print(f"  Existing 51-panel: {len(existing_syms)} symbols, "
          f"{len(existing_panel):,} rows, {existing_panel.shape[1]} cols", flush=True)

    # New 60 candidates
    new_candidates = pd.read_csv(REPO / "outputs/vBTC_universe_expansion/final_candidates.csv")
    new_syms = sorted(new_candidates["symbol"].tolist())
    print(f"  New 60 candidates from E3", flush=True)

    # Step 1: rebuild basket from existing 51 close prices (low memory)
    print(f"\n--- Step 1: Reconstruct basket from existing 51 ---", flush=True)
    t0 = time.time()
    closes_dict = {}
    for sym in existing_syms:
        f = build_kline_features(sym)  # cached → fast
        if not f.empty:
            closes_dict[sym] = f["close"]
        # No need to keep features around
        del f
    closes = pd.DataFrame(closes_dict).sort_index()
    basket_ret, basket_close = build_basket(closes)
    print(f"  Basket built: {len(basket_close):,} bars in {time.time()-t0:.0f}s", flush=True)
    del closes_dict, closes
    gc.collect()

    # Save basket for reuse
    pd.DataFrame({"basket_close": basket_close, "basket_ret": basket_ret}).to_parquet(
        OUT_DIR / "basket.parquet"
    )

    # Step 2: per-symbol build for new 60
    print(f"\n--- Step 2: Build features for new 60 symbols (streaming) ---", flush=True)
    schema_cols = set(existing_panel.columns)
    print(f"  Target schema has {len(schema_cols)} columns", flush=True)
    success = 0
    fail = 0
    for i, sym in enumerate(new_syms, 1):
        t0 = time.time()
        try:
            f = build_kline_features(sym)
            if f.empty:
                fail += 1
                print(f"  [{i:>2}/60] {sym:<14}  EMPTY, skipped", flush=True)
                continue
            # Reindex to basket time grid (handles late listings)
            f = f.reindex(basket_close.index)
            # Add basket-relative features (vs existing 51-basket)
            f = add_basket_features(f, basket_close, basket_ret)
            # Add funding features
            try:
                f = add_funding_features(f, sym)
            except Exception:
                pass
            # Compute alpha target like make_xs_alpha_labels
            my_close = f["close"]
            my_fwd = my_close.pct_change(HORIZON).shift(-HORIZON)
            bk_fwd = basket_close.pct_change(HORIZON).shift(-HORIZON)
            beta = f["beta_short_vs_bk"] if "beta_short_vs_bk" in f.columns else 1.0
            alpha = my_fwd - beta * bk_fwd
            rmean = alpha.expanding(min_periods=288).mean().shift(HORIZON)
            rstd = alpha.rolling(288 * 7, min_periods=288).std().shift(HORIZON)
            target = (alpha - rmean) / rstd.replace(0, np.nan)
            exit_time = my_close.index.to_series().shift(-HORIZON)

            f["return_pct"] = my_fwd
            f["basket_fwd"] = bk_fwd
            f["alpha_realized"] = alpha
            f["alpha_A"] = alpha
            f["target_A"] = target
            f["exit_time"] = exit_time
            f["symbol"] = sym
            f = f.reset_index().rename(columns={"index": "open_time"})

            # Keep only columns matching existing panel schema
            keep = [c for c in f.columns if c in schema_cols]
            f = f[keep]

            # Save per-symbol
            f.to_parquet(INTERMEDIATE_DIR / f"{sym}.parquet", index=False)
            success += 1
            elapsed = time.time() - t0
            print(f"  [{i:>2}/60] {sym:<14}  rows={len(f):>7,}  cols={f.shape[1]:>3}  "
                  f"({elapsed:.0f}s)", flush=True)
            del f
            gc.collect()
        except Exception as e:
            fail += 1
            print(f"  [{i:>2}/60] {sym:<14}  ERROR: {e}", flush=True)

    print(f"\n  Success: {success}/60, Failed: {fail}", flush=True)

    # Step 3: concatenate
    print(f"\n--- Step 3: Concatenate with existing panel ---", flush=True)
    new_frames = []
    for sym in new_syms:
        p = INTERMEDIATE_DIR / f"{sym}.parquet"
        if p.exists():
            new_frames.append(pd.read_parquet(p))
    if not new_frames:
        print("  No new symbol data, skipping", flush=True)
        return
    new_panel = pd.concat(new_frames, ignore_index=True)
    print(f"  New rows: {len(new_panel):,}", flush=True)

    # Align columns: keep only those present in BOTH
    common_cols = [c for c in existing_panel.columns if c in new_panel.columns]
    print(f"  Common columns: {len(common_cols)}", flush=True)
    new_panel_aligned = new_panel[common_cols].copy()
    existing_aligned = existing_panel[common_cols].copy()

    expanded = pd.concat([existing_aligned, new_panel_aligned], ignore_index=True)
    expanded = expanded.sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print(f"  Combined panel: {len(expanded):,} rows × {expanded.shape[1]} cols, "
          f"{expanded['symbol'].nunique()} symbols", flush=True)

    out_path = OUT_DIR / "panel_variants_with_funding.parquet"
    expanded.to_parquet(out_path, compression="zstd", index=False)
    print(f"\n  saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
