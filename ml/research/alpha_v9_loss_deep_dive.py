"""Deep dive: WHY did we lose money in recent months?

Three-axis decomposition:
  1. Per-feature univariate IC stability — has any feature's IC decayed in recent months?
  2. Per-symbol cumulative contribution — which symbols dominate recent losses?
  3. Per-feature category breakdown — kline vs flow vs basket vs xs_rank features.
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v8_h48_audit import build_wide_panel

# Categorize v6_clean features by mechanism
FEATURE_CATEGORIES = {
    "return": ["return_1d", "return_4h", "return_8h", "return_12h", "return_24h", "return_48h",
                "ema_slope_20_1h"],
    "momentum_kline": ["bars_since_high", "bars_since_low", "atr_pct", "obv_z_1d", "obv_signal",
                        "vwap_zscore", "vwap_slope_96", "mfi", "price_volume_corr_10",
                        "price_volume_corr_20", "volume_ma_50"],
    "basket_relative": ["dom_level_vs_bk", "dom_change_48b_vs_bk", "dom_change_288b_vs_bk",
                         "dom_z_7d_vs_bk", "bk_ret_48b", "bk_ema_slope_4h",
                         "idio_ret_12b_vs_bk", "idio_ret_48b_vs_bk",
                         "corr_change_3d_vs_bk"],
    "xs_rank": [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")],
}


def per_bar_ic(s: pd.Series, y: pd.Series, bar_ids: pd.Series) -> float:
    df = pd.DataFrame({"f": s, "y": y, "b": bar_ids})
    return df.groupby("b").apply(
        lambda g: g["f"].rank().corr(g["y"].rank()) if len(g) >= 5 else np.nan
    ).mean()


def main():
    panel = build_wide_panel()
    panel["open_time"] = pd.to_datetime(panel["open_time"])
    if panel["open_time"].dt.tz is None:
        panel["open_time"] = panel["open_time"].dt.tz_localize("UTC")
    print(f"Panel: {panel['open_time'].min().date()} → {panel['open_time'].max().date()}", flush=True)

    # Build month tags
    panel["month"] = panel["open_time"].dt.to_period("M").astype(str)

    # Use only filtered (rc=0.50) panel for IC calc
    panel_f = panel[panel["autocorr_pctile_7d"] >= 0.50].copy()

    # ===== 1. PER-FEATURE UNIVARIATE IC over time =====
    print("\n" + "=" * 110, flush=True)
    print("1. PER-FEATURE UNIVARIATE IC vs alpha_realized — by month", flush=True)
    print("=" * 110, flush=True)

    months = sorted(panel_f["month"].unique())
    feat_cols = [c for c in XS_FEATURE_COLS_V6_CLEAN if c in panel_f.columns]
    print(f"  {len(feat_cols)} features × {len(months)} months", flush=True)

    # IC matrix: feature × month
    ic_data = []
    for m in months:
        m_data = panel_f[panel_f["month"] == m]
        if len(m_data) < 100: continue
        bar_ids = m_data["open_time"]
        y = m_data["alpha_realized"]
        for f in feat_cols:
            if f not in m_data.columns: continue
            ic = per_bar_ic(m_data[f], y, bar_ids)
            ic_data.append({"month": m, "feature": f, "ic": ic})
    ic_df = pd.DataFrame(ic_data)
    ic_pivot = ic_df.pivot(index="feature", columns="month", values="ic")

    # Rank features by IC decay: compare last 3 months avg vs prior 6 months avg
    if len(months) >= 6:
        recent_mo = months[-3:]
        prior_mo = months[-9:-3] if len(months) >= 9 else months[:-3]
        ic_pivot["mean_prior"] = ic_pivot[prior_mo].mean(axis=1)
        ic_pivot["mean_recent"] = ic_pivot[recent_mo].mean(axis=1)
        ic_pivot["abs_prior"] = ic_pivot[prior_mo].abs().mean(axis=1)
        ic_pivot["abs_recent"] = ic_pivot[recent_mo].abs().mean(axis=1)
        ic_pivot["ic_decay"] = ic_pivot["abs_prior"] - ic_pivot["abs_recent"]
        ic_pivot["sign_flip"] = (np.sign(ic_pivot["mean_prior"])
                                  != np.sign(ic_pivot["mean_recent"]))

        print(f"\n  Features with biggest IC decay (|prior_IC| - |recent_IC|):", flush=True)
        decay_sorted = ic_pivot.sort_values("ic_decay", ascending=False)
        print(f"  {'feature':<35} {'prior_|IC|':>11} {'recent_|IC|':>12} {'decay':>8} {'sign_flip':>10}", flush=True)
        for f, row in decay_sorted.head(10).iterrows():
            sf = "YES" if row["sign_flip"] else "no"
            print(f"  {f:<35} {row['abs_prior']:>+10.4f}  {row['abs_recent']:>+11.4f}  "
                  f"{row['ic_decay']:>+7.4f}  {sf:>10}", flush=True)

        print(f"\n  Features that REVERSED sign (prior + recent):", flush=True)
        flipped = ic_pivot[ic_pivot["sign_flip"]].sort_values("abs_prior", ascending=False)
        for f, row in flipped.head(10).iterrows():
            print(f"    {f:<35} prior {row['mean_prior']:>+.4f} → recent {row['mean_recent']:>+.4f}", flush=True)
        if len(flipped) == 0:
            print(f"    (none)", flush=True)

    # ===== 2. PER-SYMBOL recent attribution =====
    print("\n" + "=" * 110, flush=True)
    print("2. PER-SYMBOL: RECENT 3-MONTH alpha_realized × prediction-direction analysis", flush=True)
    print("=" * 110, flush=True)

    # Look at recent 3 months
    recent_panel = panel_f[panel_f["month"].isin(months[-3:])]
    print(f"  Recent 3 months: {months[-3:]}", flush=True)
    print(f"  n_cycles in recent panel: {recent_panel['open_time'].nunique()}", flush=True)

    # Per-symbol: avg alpha realized vs typical
    sym_recent = recent_panel.groupby("symbol")["alpha_realized"].agg(
        ["mean", "std", "count"]).reset_index()
    sym_recent["mean_bps"] = sym_recent["mean"] * 1e4
    sym_recent = sym_recent.sort_values("mean_bps")

    # Per-symbol: prior period alpha for comparison
    if len(months) >= 6:
        prior_panel = panel_f[panel_f["month"].isin(prior_mo)]
        sym_prior = prior_panel.groupby("symbol")["alpha_realized"].agg(
            ["mean", "std"]).reset_index()
        sym_prior["mean_bps_prior"] = sym_prior["mean"] * 1e4
        sym_recent = sym_recent.merge(
            sym_prior[["symbol", "mean_bps_prior"]], on="symbol")
        sym_recent["alpha_shift"] = sym_recent["mean_bps"] - sym_recent["mean_bps_prior"]

    print(f"\n  Per-symbol alpha_realized (recent 3 months, sorted by mean):", flush=True)
    print(f"  {'symbol':<10} {'mean_bps':>10} {'std_bps':>9} {'n_cycles':>9} "
          f"{'prior_mean_bps':>16} {'alpha_shift':>12}", flush=True)
    for _, r in sym_recent.iterrows():
        shift = r.get("alpha_shift", float("nan"))
        prior = r.get("mean_bps_prior", float("nan"))
        print(f"  {r['symbol']:<10} {r['mean_bps']:>+9.2f}  {r['std']*1e4:>+8.2f}  "
              f"{r['count']:>8d}  {prior:>+15.2f}  {shift:>+11.2f}", flush=True)

    # ===== 3. CATEGORY-LEVEL IC summary =====
    print("\n" + "=" * 110, flush=True)
    print("3. FEATURE CATEGORY: aggregate IC by category, prior vs recent", flush=True)
    print("=" * 110, flush=True)
    if len(months) >= 6:
        for cat, feats in FEATURE_CATEGORIES.items():
            in_cat = [f for f in feats if f in ic_pivot.index]
            if not in_cat: continue
            prior_mean = ic_pivot.loc[in_cat, "abs_prior"].mean()
            recent_mean = ic_pivot.loc[in_cat, "abs_recent"].mean()
            decay = prior_mean - recent_mean
            n_flipped = int(ic_pivot.loc[in_cat, "sign_flip"].sum())
            print(f"  {cat:<22}: {len(in_cat):>3d} features  "
                  f"prior_|IC|={prior_mean:>+.4f}  recent_|IC|={recent_mean:>+.4f}  "
                  f"decay={decay:>+.4f}  sign_flips={n_flipped}/{len(in_cat)}", flush=True)

    # ===== 4. Recent IC time series — was there a sudden break? =====
    print("\n" + "=" * 110, flush=True)
    print("4. AGGREGATE alpha_realized strength by month (sign of dispersion)", flush=True)
    print("=" * 110, flush=True)
    print(f"  {'month':>10} {'mean_alpha_bps':>14} {'std_alpha_bps':>13} {'n_cycles':>9}", flush=True)
    for m in months:
        m_panel = panel_f[panel_f["month"] == m]
        if m_panel.empty: continue
        mean_a = m_panel["alpha_realized"].mean() * 1e4
        std_a = m_panel["alpha_realized"].std() * 1e4
        n_c = m_panel["open_time"].nunique()
        marker = "← recent 3" if m in months[-3:] else ""
        print(f"  {m:>10} {mean_a:>+13.2f}  {std_a:>+12.2f}  {n_c:>8d}  {marker}", flush=True)

    # Save full IC pivot for reference
    out = REPO / "outputs/h48_loss_deep_dive"
    out.mkdir(parents=True, exist_ok=True)
    ic_pivot.to_csv(out / "feature_ic_by_month.csv")
    sym_recent.to_csv(out / "symbol_alpha_recent_3mo.csv", index=False)
    print(f"\n  saved → {out}", flush=True)


if __name__ == "__main__":
    main()
