"""Phase E4: rebuild feature panel with EXPANDED universe (51 + 60 = 111 symbols).

Uses existing features_ml.cross_sectional.assemble_universe() — handles everything:
  - Reloads per-symbol kline features (cached for existing 51, fresh for new 60)
  - Rebuilds basket from ALL 111 closes
  - Recomputes basket-relative features for all symbols against expanded basket
  - Assigns new sym_id cardinality

Output: outputs/vBTC_features_expanded/panel_variants_with_funding.parquet

Skips xs_rank features for now (we'll add separately if needed).
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import assemble_universe, make_xs_alpha_labels
from features_ml.cross_sectional import add_xs_rank_features

OUT_DIR = REPO / "outputs/vBTC_features_expanded"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48


def main():
    print(f"=== Phase E4: Rebuild feature panel with expanded universe ===\n", flush=True)

    # Existing 51
    existing = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                                  columns=["symbol"])
    existing_syms = sorted(existing["symbol"].unique())
    print(f"  Existing 51-panel: {len(existing_syms)} symbols", flush=True)

    # New 60 from E3 download
    new_candidates = pd.read_csv(REPO / "outputs/vBTC_universe_expansion/final_candidates.csv")
    new_syms = sorted(new_candidates["symbol"].tolist())
    print(f"  New candidates from E3: {len(new_syms)} symbols", flush=True)

    expanded_syms = sorted(set(existing_syms) | set(new_syms))
    print(f"  Expanded universe total: {len(expanded_syms)} symbols", flush=True)

    print(f"\n  Calling assemble_universe()...", flush=True)
    t0 = time.time()
    pkg = assemble_universe(expanded_syms, horizon=HORIZON)
    print(f"  assemble_universe done in {time.time()-t0:.0f}s", flush=True)
    print(f"  Returned: {len(pkg['feats_by_sym'])} symbols enriched", flush=True)

    # Compute alpha labels
    print(f"\n  Computing alpha labels...", flush=True)
    labels = make_xs_alpha_labels(pkg["feats_by_sym"], pkg["basket_close"], HORIZON)

    # Assemble final per-symbol panels with feat + label columns + sym_id + symbol
    print(f"\n  Concatenating per-symbol panels...", flush=True)
    frames = []
    for sym, feat_df in pkg["feats_by_sym"].items():
        if sym not in labels: continue
        lab = labels[sym]
        # Join on index (open_time)
        combined = feat_df.join(lab, how="left")
        combined["symbol"] = sym
        combined = combined.reset_index().rename(columns={"index": "open_time"})
        # Ensure open_time is a column not index
        if "open_time" not in combined.columns:
            combined["open_time"] = combined.index
        frames.append(combined)
    panel = pd.concat(frames, ignore_index=True).sort_values(["symbol", "open_time"])
    print(f"  Combined panel: {len(panel):,} rows × {panel.shape[1]} cols", flush=True)

    # Add cross-sectional rank features
    print(f"\n  Adding xs_rank features...", flush=True)
    panel = add_xs_rank_features(panel)
    print(f"  Done, panel shape: {panel.shape}", flush=True)

    # Save
    out_path = OUT_DIR / "panel_variants_with_funding.parquet"
    panel.to_parquet(out_path, compression="zstd", index=False)
    print(f"\n  saved → {out_path}", flush=True)
    print(f"  Symbols: {panel['symbol'].nunique()}", flush=True)
    print(f"  Time range: {panel['open_time'].min()} → {panel['open_time'].max()}", flush=True)


if __name__ == "__main__":
    main()
