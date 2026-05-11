"""ORIG25 + curated NEW additions test (conv+PM stack).

Prior FULL39 result: adding all 14 new perps drops Sharpe +2.75 → +0.88.
This test isolates which NEW_SYMBOLS subsets help vs hurt.

Configs:
  A. ORIG25 (baseline, 25 syms) — validated +2.75
  B. +DeFi3:        ORIG25 + {AAVE, MKR, LDO}                      (28 syms)
  C. +L1_3:         ORIG25 + {ETC, HBAR, TON}                       (28 syms)
  D. +Quality6:     ORIG25 + {AAVE, MKR, LDO, ETC, HBAR, TON}       (31 syms)
  E. +NonMeme10:    ORIG25 + 10 non-meme NEW (DeFi + L1 + AXS+ICP+TRB+GMX) (35 syms)
  F. +MemesOnly4:   ORIG25 + {1000PEPE, 1000SHIB, ORDI, WIF}        (29 syms)
                    (sanity check that memes are the drag)
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN, list_universe
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_universe_expand import build_wide_panel_for, run_config

HORIZON = 48
COST_PER_LEG = 4.5
CYCLES_PER_YEAR = (288 * 365) / HORIZON
OUT_DIR = REPO / "outputs/universe_curated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
               "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
               "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}
DEFI3 = {"AAVEUSDT", "MKRUSDT", "LDOUSDT"}
L1_3 = {"ETCUSDT", "HBARUSDT", "TONUSDT"}
QUALITY6 = DEFI3 | L1_3
MEMES4 = {"1000PEPEUSDT", "1000SHIBUSDT", "ORDIUSDT", "WIFUSDT"}
NONMEME10 = NEW_SYMBOLS - MEMES4


def main():
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])

    configs = [
        ("ORIG25 (baseline)",       orig25,                       7),
        ("+DeFi3 (AAVE,MKR,LDO)",   sorted(orig25 + list(DEFI3)), 7),
        ("+L1_3 (ETC,HBAR,TON)",    sorted(orig25 + list(L1_3)),  7),
        ("+Quality6 (DeFi+L1)",     sorted(orig25 + list(QUALITY6)), 7),
        ("+NonMeme10",              sorted(orig25 + list(NONMEME10)), 7),
        ("+MemesOnly4 (sanity)",    sorted(orig25 + list(MEMES4)), 7),
    ]
    print(f"Testing {len(configs)} configs:")
    for label, u, k in configs:
        print(f"  {label}: N={len(u)} K={k}")
    print()

    results = []
    panels_cache: dict[tuple, pd.DataFrame] = {}
    for label, universe, k in configs:
        key = tuple(universe)
        if key not in panels_cache:
            print(f"\n--- Building panel for {label} ({len(universe)} syms) ---")
            panels_cache[key] = build_wide_panel_for(universe)
        panel = panels_cache[key]
        r = run_config(panel, top_k=k, label=label)
        results.append(r)

    print("\n" + "=" * 110)
    print("ORIG25 + QUALITY-FILTERED NEW ADDITIONS  (conv+PM, multi-OOS, 4.5 bps/leg)")
    print("=" * 110)
    print(f"  {'config':<32}  {'N':>3}  {'baseline_Sh':>11}  {'stacked_Sh':>10}  "
          f"{'Δsh':>6}  {'stacked_net':>11}  {'CI':>16}  {'K_avg':>6}")
    base_sh = None
    for r in results:
        marker = "  ← baseline" if r['label'].startswith("ORIG25") else ""
        if r['label'].startswith("ORIG25"): base_sh = r['stacked_sharpe']
        delta = r['stacked_sharpe'] - base_sh if base_sh is not None else 0
        if not r['label'].startswith("ORIG25"):
            marker = f"  Δ vs ORIG25={delta:+.2f}"
        print(f"  {r['label']:<32}  {r['n_universe']:>3}  "
              f"{r['baseline_sharpe']:>+11.2f}  {r['stacked_sharpe']:>+10.2f}  "
              f"{r['delta_sh']:>+6.2f}  {r['stacked_net']:>+11.2f}  "
              f"[{r['stacked_ci'][0]:+.2f},{r['stacked_ci'][1]:+.2f}]  "
              f"{r['K_avg']:>6.2f}{marker}")

    pd.DataFrame(results).to_csv(OUT_DIR / "curated_summary.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
