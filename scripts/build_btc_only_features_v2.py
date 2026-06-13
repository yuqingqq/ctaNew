"""Fix the BTC-only panel: drop merge duplicates, normalize column names, define WINNER_BTC.

Uses the existing panel's pre-computed BTC features (validated, presumably PIT-safe)
where available, falls back to my new computations for genuinely-new features.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

PANEL_IN = REPO / "outputs/vBTC_features_btc_only/panel_btc_only.parquet"
PANEL_OUT = REPO / "outputs/vBTC_features_btc_only/panel_btc_only_clean.parquet"


# Canonical 25-feature BTC-only set
WINNER_BTC = [
    # (1) BTC residual momentum (3 horizons)
    "idio_ret_to_btc_12b",
    "idio_ret_to_btc_48b",
    "idio_ret_to_btc_288b",
    # (2) BTC residual price level (3) — all new
    "dom_btc_z_1d",
    "dom_btc_change_48b",
    "dom_btc_change_288b",
    # (3) BTC β / corr state (4)
    "beta_to_btc",
    "beta_to_btc_change_5d",
    "corr_to_btc_1d",
    "corr_to_btc_change_3d",
    # (4) BTC residual risk (3)
    "idio_vol_to_btc_1h",
    "idio_vol_to_btc_1d",
    "idio_vol_ratio_to_btc",
    # (5) BTC market regime (3 — same for all syms at t)
    "btc_ret_48b",
    "btc_realized_vol_1d",
    "btc_realized_vol_30d",
    # (6) Single-name flow / funding (6)
    "atr_pct",
    "obv_z_1d",
    "vwap_slope_96",
    "funding_rate",
    "funding_rate_z_7d",
    "funding_rate_1d_change",
    # (7) Stable per-symbol context (3 — replaces sym_id)
    "listing_age_days",
    "log_quote_volume_90d",
    "residual_vol_90d_own_pctile",
]


def main():
    print("=== Build clean BTC-only panel ===\n", flush=True)
    panel = pd.read_parquet(PANEL_IN)
    print(f"Input: {len(panel):,} rows × {len(panel.columns)} cols", flush=True)

    # Strategy: prefer _x suffixed (existing/canonical) over _y (my recomputation).
    # If only one variant exists, rename to canonical (no suffix).
    duplicates = {}
    for col in list(panel.columns):
        if col.endswith("_x"):
            stem = col[:-2]
            other = stem + "_y"
            duplicates.setdefault(stem, {})["x"] = col
            if other in panel.columns:
                duplicates[stem]["y"] = other
        elif col.endswith("_y"):
            stem = col[:-2]
            other = stem + "_x"
            duplicates.setdefault(stem, {})["y"] = col
            if other in panel.columns:
                duplicates[stem]["x"] = other

    print(f"\nFound {len(duplicates)} duplicated stems:", flush=True)
    for stem, parts in duplicates.items():
        print(f"  {stem}: has {list(parts.keys())}", flush=True)

    # Resolution: keep _x (existing), drop _y, rename _x → stem
    drop_cols = []
    rename_map = {}
    for stem, parts in duplicates.items():
        if "x" in parts:
            rename_map[parts["x"]] = stem
            if "y" in parts:
                drop_cols.append(parts["y"])
        elif "y" in parts:
            # only _y exists, rename
            rename_map[parts["y"]] = stem
    panel = panel.drop(columns=drop_cols)
    panel = panel.rename(columns=rename_map)
    print(f"\nDropped {len(drop_cols)} _y duplicates; renamed {len(rename_map)} _x → stem",
          flush=True)
    print(f"Cleaned panel: {len(panel):,} rows × {len(panel.columns)} cols", flush=True)

    # Verify WINNER_BTC features all present
    print("\nChecking WINNER_BTC feature availability:", flush=True)
    missing = []
    for f in WINNER_BTC:
        present = f in panel.columns
        n_valid = panel[f].notna().sum() if present else 0
        n_total = len(panel)
        mark = "✓" if present and n_valid > n_total * 0.8 else ("⚠" if present else "✗")
        print(f"  {mark} {f:<35} {'(missing)' if not present else f'{n_valid:,}/{n_total:,} valid'}",
              flush=True)
        if not present: missing.append(f)
    if missing:
        print(f"\nERROR: {len(missing)} features missing. Need to compute them.", flush=True)
        sys.exit(1)

    # Save clean panel
    panel.to_parquet(PANEL_OUT, index=False)
    print(f"\nSaved: {PANEL_OUT} ({len(panel):,} rows × {len(panel.columns)} cols)",
          flush=True)

    # IC check on clean WINNER_BTC features against alpha_beta target
    print(f"\nPer-cycle IC vs alpha_beta target (WINNER_BTC, 25 features):", flush=True)
    times = sorted(panel["open_time"].unique())
    keep_t = set(times[::48])
    samp = panel[panel["open_time"].isin(keep_t)].dropna(subset=["alpha_beta"])
    rows = []
    for feat in WINNER_BTC:
        ics = []
        for t, g in samp.dropna(subset=[feat]).groupby("open_time"):
            if len(g) < 10: continue
            ic = g[feat].rank().corr(g["alpha_beta"].rank())
            if not pd.isna(ic): ics.append(ic)
        if ics:
            ics = np.array(ics)
            rows.append({"feature": feat,
                          "mean_ic": float(ics.mean()),
                          "median_ic": float(np.median(ics)),
                          "abs_mean_ic": abs(float(ics.mean())),
                          "n_cycles": len(ics)})
    df_ic = pd.DataFrame(rows).sort_values("abs_mean_ic", ascending=False)
    pd.set_option("display.width", 200)
    print(df_ic[["feature","mean_ic","median_ic","n_cycles"]].to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"), flush=True)
    df_ic.to_csv(PANEL_OUT.parent / "winner_btc_ic_check.csv", index=False)


if __name__ == "__main__":
    main()
