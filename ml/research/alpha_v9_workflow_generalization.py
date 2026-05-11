"""Workflow generalization test: does conv+PM produce similar Sharpe on
different 25-name baskets, or is ORIG25's +2.75 specific to that universe?

Tests:
  1. ORIG25 (reference, validated +2.75)
  2. ORIG25_swap5: drop 5 lowest-volume ORIG25, add 5 highest-volume NEW
  3. ORIG25_swap10: drop 10 lowest-volume ORIG25, add 10 highest-volume NEW
  4. Top25_by_volume: 25 highest-volume of all 39 (likely ≈ORIG25)
  5. Random25_seed1: 25 random from FULL39 (different mix)
  6. Random25_seed2: 25 random from FULL39 (different mix)

If 4-6 cluster around +2.0 to +3.0 → workflow generalizes, ORIG25 not special.
If 4-6 scatter widely → result is basket-specific.

Each universe re-runs the full conv+PM workflow:
  - Build wide panel for that universe (re-derive _xs_rank features from new basket)
  - Train LGBM ensemble per fold (with universe-specific cross-sectional structure)
  - Apply conv_gate + PM_M2_b1 + per-name=1/7 weighting
"""
from __future__ import annotations
import sys, time, warnings, json
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import list_universe, build_kline_features
from ml.research.alpha_v9_universe_expand import build_wide_panel_for, run_config

OUT_DIR = REPO / "outputs/workflow_generalization"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
               "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
               "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}


def get_volume_ranking() -> dict[str, float]:
    """Return {symbol: median_24h_volume_USD} for ranking."""
    universe = sorted(list_universe(min_days=200))
    vols = {}
    for s in universe:
        f = build_kline_features(s)
        if f.empty: continue
        f["dollar_vol_5m"] = f["close"] * f["volume"]
        daily_vol = f["dollar_vol_5m"].rolling(288).sum()
        vols[s] = float(daily_vol.iloc[-1000:].median())
    return vols


def main():
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    new14 = sorted([s for s in universe_full if s in NEW_SYMBOLS])

    print(f"FULL39: {len(universe_full)} symbols")
    print(f"ORIG25: {len(orig25)} symbols")
    print(f"NEW14:  {len(new14)} symbols")
    print()

    # Volume ranking
    print("Computing volume rankings...")
    vols = get_volume_ranking()
    orig25_by_vol = sorted(orig25, key=lambda s: vols.get(s, 0))   # ascending
    new14_by_vol = sorted(new14, key=lambda s: vols.get(s, 0), reverse=True)  # descending

    # Construct universes
    universes = {
        "ORIG25 (reference)": orig25,
    }

    # Swap 5: drop 5 lowest-vol ORIG25, add 5 highest-vol NEW
    swap5_drop = set(orig25_by_vol[:5])
    swap5_add = set(new14_by_vol[:5])
    universes["ORIG25_swap5"] = sorted((set(orig25) - swap5_drop) | swap5_add)
    print(f"  swap5: drop={sorted(swap5_drop)}, add={sorted(swap5_add)}")

    # Swap 10: drop 10 lowest-vol ORIG25, add 10 highest-vol NEW
    swap10_drop = set(orig25_by_vol[:10])
    swap10_add = set(new14_by_vol[:10])
    universes["ORIG25_swap10"] = sorted((set(orig25) - swap10_drop) | swap10_add)

    # Top25 by volume (all 39, take top 25)
    top25_by_vol = sorted(sorted(vols.keys(), key=lambda s: -vols.get(s, 0))[:25])
    universes["Top25_by_volume"] = top25_by_vol
    n_orig_in_top25 = len(set(top25_by_vol) & set(orig25))
    print(f"  Top25_by_volume: {n_orig_in_top25}/25 are ORIG25 names")

    # Random 25 (no replacement) from FULL39
    rng1 = np.random.default_rng(42)
    universes["Random25_seed42"] = sorted(rng1.choice(universe_full, 25, replace=False).tolist())
    rng2 = np.random.default_rng(2026)
    universes["Random25_seed2026"] = sorted(rng2.choice(universe_full, 25, replace=False).tolist())

    print(f"\nTesting {len(universes)} universes:")
    for name, u in universes.items():
        n_orig = len(set(u) & set(orig25))
        n_new = len(set(u) & set(new14))
        print(f"  {name}: {len(u)} syms ({n_orig} ORIG25 + {n_new} NEW)")
    print()

    # Run conv+PM on each
    results = []
    for name, u in universes.items():
        print(f"\n--- Building panel + running conv+PM on {name} ---")
        panel = build_wide_panel_for(u)
        r = run_config(panel, top_k=7, label=name)
        r["universe"] = u
        r["n_orig25"] = len(set(u) & set(orig25))
        r["n_new14"] = len(set(u) & set(new14))
        results.append(r)

    # Headline
    print("\n" + "=" * 110)
    print("WORKFLOW GENERALIZATION TEST (conv+PM, K=7, multi-OOS, 4.5 bps/leg)")
    print("=" * 110)
    print(f"  {'universe':<25}  {'N':>3}  {'ORIG25/NEW':>10}  "
          f"{'baseline_Sh':>11}  {'stacked_Sh':>10}  {'CI_lo':>7}  {'CI_hi':>7}")
    for r in results:
        composition = f"{r['n_orig25']}/{r['n_new14']}"
        print(f"  {r['label']:<25}  {r['n_universe']:>3}  {composition:>10}  "
              f"{r['baseline_sharpe']:>+11.2f}  {r['stacked_sharpe']:>+10.2f}  "
              f"{r['stacked_ci'][0]:>+7.2f}  {r['stacked_ci'][1]:>+7.2f}")

    # Statistical analysis
    sh_values = [r['stacked_sharpe'] for r in results]
    print(f"\n  Stacked Sharpe distribution:")
    print(f"    Mean:   {np.mean(sh_values):+.2f}")
    print(f"    Std:    {np.std(sh_values):+.2f}")
    print(f"    Range:  [{min(sh_values):+.2f}, {max(sh_values):+.2f}]")
    print(f"    ORIG25: {results[0]['stacked_sharpe']:+.2f}  (rank "
          f"{sorted(sh_values, reverse=True).index(results[0]['stacked_sharpe']) + 1}/{len(sh_values)})")

    # Save
    summary = pd.DataFrame([{k: v for k, v in r.items() if k != "universe"} for r in results])
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    with open(OUT_DIR / "universes.json", "w") as f:
        json.dump({r["label"]: r["universe"] for r in results}, f, indent=2)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
