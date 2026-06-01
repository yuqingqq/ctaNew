"""X38 — Audit per-fold training availability for new-listing syms.

Checks HYPEUSDT (June 2025), ASTERUSDT (Sept 2025), PUMPUSDT (April 2025)
and other newer syms. In each of the 9 folds:
  - When is each sym's first appearance?
  - How many training rows per sym per fold?
  - Which folds have insufficient training data?
"""
from __future__ import annotations
import sys, importlib.util
from pathlib import Path
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)


def main():
    print("=== X38 New-listing fold availability audit ===\n")
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding_v2.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    panel = panel[panel["symbol"] != "BTCUSDT"].copy()

    # First appearance
    first_seen = panel.groupby("symbol")["open_time"].min().sort_values()
    NEW_SYMS = first_seen[first_seen > "2025-04-01"].index.tolist()
    print(f"Syms first seen after 2025-04-01 ({len(NEW_SYMS)} syms):")
    for sym in NEW_SYMS:
        n = (panel["symbol"] == sym).sum()
        print(f"  {sym}: first={first_seen[sym]}, total_rows={n:,}")

    # Compute target_z to get folds
    panel = x6.build_target_z(panel)
    folds = x6.get_folds(panel)

    print(f"\n=== Per-fold training row counts for NEW syms ===")
    print(f"{'Fold':<5} {'OOS start':<22}", end="")
    for sym in NEW_SYMS:
        print(f" {sym[:8]:>10}", end="")
    print()

    for f, ts, te, ec in folds:
        train = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        print(f"  {f:<3} {str(ts)[:19]:<22}", end="")
        for sym in NEW_SYMS:
            n = (train["symbol"] == sym).sum()
            print(f" {n:>10,}", end="")
        print()

    # Also check: are these "new" syms in the AI cluster?
    import json
    with open(REPO / "config/clusters_v1.json") as f:
        clusters = json.load(f)
    AI_SYMS = set(clusters.get("ai", []))
    print(f"\nAI cluster: {sorted(AI_SYMS)}")
    new_AI_overlap = [s for s in NEW_SYMS if s in AI_SYMS]
    print(f"NEW syms in AI cluster: {new_AI_overlap}")


if __name__ == "__main__":
    main()
