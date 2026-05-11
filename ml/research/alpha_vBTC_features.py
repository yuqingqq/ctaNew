"""Phase 3: engineer BTC-target-specific features.

Categories (motivated by Phase 1 + 1.5 EDA findings):

  A. Cross-asset / factor features (capture PC1 = 44% of variance):
     - xs_alpha_dispersion_12b   std of recent factor-orthogonal returns
     - xs_alpha_mean_12b         mean (basket-vs-BTC drift)
     - xs_alpha_iqr_12b          robust dispersion (q75 - q25)

  B. Per-name BTC-relative:
     - beta_to_btc                (already in panel)
     - beta_to_btc_change_5d
     - corr_to_btc_1d
     - idio_vol_to_btc_1d
     - idio_ret_to_btc_12b        recent residual return
     - idio_ret_to_btc_48b

  C. Per-name factor-loading (PC1 proxy via rolling correlation):
     - name_factor_loading_1d    corr(name's residual return, xs mean)
     - name_idio_share           1 - factor_loading²

  D. BTC microstructure / regime:
     - btc_vol_30d                annualized 30d BTC vol
     - btc_ret_12b                BTC recent return
     - btc_ret_48b
     - btc_ema_slope_4h           BTC trend

  E. Tail / robust features (heavy tails noted in EDA):
     - idio_max_abs_12b           recent extreme move
     - idio_skew_1d
     - idio_kurt_1d

  F. Cross-sectional rank versions of selected features (IC boosters)

After computation, single-feature IC against btc_target on training data.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_vBTC_panel import build_panel_51, BTC_SYMBOL

OUT_DIR = REPO / "outputs/vBTC_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_WITH_FEATURES = OUT_DIR / "panel_with_btc_features.parquet"
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC


def add_btc_target_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Engineer all Phase 3 features per (symbol, open_time) and merge."""
    panel = panel.sort_values(["symbol", "open_time"]).reset_index(drop=True)

    # We need raw 5-min returns per symbol. The panel has return_pct = forward
    # h-bar return (NOT the 5-min return). So we need to compute 5-min returns
    # via a derived close series. Since panel has return_pct (forward h=48),
    # we use a different approach: reconstruct 5-min closes per symbol from
    # the kline cache.
    #
    # Actually: it's cleaner to compute features in the per-symbol kline-feature
    # space (where 5-min closes ARE available) and merge back. But that would
    # duplicate the panel-build pipeline. Compromise: compute features that
    # rely on 5-min data using rolling operations on the existing panel.

    # 5-min approximation: use the bar-level return_pct interpreted at 5min
    # resolution would be wrong (it's h-forward). Instead, we approximate
    # using the basket_fwd column (BTC's forward return per cycle) and compose.
    # For this script, we assume the panel is sampled at 5-min cadence.

    # The clean approach: load per-symbol close series and compute 5-min ret + features.
    print("  Loading raw close series per symbol...")
    from features_ml.cross_sectional import build_kline_features
    closes = {}
    btc_close = None
    for s in sorted(panel["symbol"].unique()):
        f = build_kline_features(s)
        if f.empty: continue
        c = f["close"].copy()
        if c.index.tz is None:
            c.index = c.index.tz_localize("UTC")
        closes[s] = c
        if s == BTC_SYMBOL:
            btc_close = c

    if btc_close is None:
        raise RuntimeError("BTC close series not found")

    btc_ret = btc_close.pct_change()

    # ====== Cross-sectional features (computed at each timestamp across all syms) ======
    print("  Computing cross-sectional residual matrix (factor features)...")
    # For each timestamp t, compute factor-orthogonal returns of each symbol
    #   res[s, t] = ret[s, t] - β[s, t] × btc_ret[t]
    # Then xs features = stats across symbols at that t.
    sym_list = sorted(closes.keys())
    sym_list = [s for s in sym_list if s != BTC_SYMBOL]
    # Align all to BTC's index
    ret_mat = pd.DataFrame(index=btc_close.index)
    for s in sym_list:
        ret_mat[s] = closes[s].pct_change().reindex(btc_close.index, method="ffill")
    ret_mat["BTC"] = btc_ret

    # We need β per (sym, t). Use the panel's beta_to_btc column (already PIT-shifted).
    print("  Building rolling β matrix from panel...")
    beta_pivot = panel.pivot_table(index="open_time", columns="symbol",
                                     values="beta_to_btc", aggfunc="first")
    beta_pivot = beta_pivot.reindex(ret_mat.index, method="ffill")

    # Residuals per (sym, t)
    print("  Computing residual matrix...")
    res_mat = pd.DataFrame(index=ret_mat.index)
    for s in sym_list:
        if s in beta_pivot.columns:
            res_mat[s] = ret_mat[s] - beta_pivot[s].fillna(1.0) * ret_mat["BTC"]

    # XS factor features (per timestamp, computed on rolling 12-bar windows of residuals)
    print("  Cross-sectional features (12-bar rolling)...")
    # xs_alpha_mean: rolling mean across symbols of residuals (cross-sectional avg per bar, then rolled)
    xs_per_bar_mean = res_mat.mean(axis=1)
    xs_per_bar_std = res_mat.std(axis=1)
    xs_per_bar_iqr = res_mat.quantile(0.75, axis=1) - res_mat.quantile(0.25, axis=1)

    # Roll over 12 bars (1h) for the per-bar metrics
    xs_features = pd.DataFrame({
        "xs_alpha_mean_12b": xs_per_bar_mean.rolling(12).mean(),
        "xs_alpha_dispersion_12b": xs_per_bar_std.rolling(12).mean(),
        "xs_alpha_iqr_12b": xs_per_bar_iqr.rolling(12).mean(),
        "xs_alpha_mean_48b": xs_per_bar_mean.rolling(48).mean(),
        "xs_alpha_dispersion_48b": xs_per_bar_std.rolling(48).mean(),
    })
    # Shift by 1 to be PIT (use only past info)
    xs_features = xs_features.shift(1)
    xs_features = xs_features.reset_index().rename(columns={"index": "open_time"})

    # ====== BTC microstructure / regime ======
    print("  BTC microstructure features...")
    btc_features = pd.DataFrame({
        "btc_ret_12b": btc_close.pct_change(12),
        "btc_ret_48b": btc_close.pct_change(48),
        "btc_ret_288b": btc_close.pct_change(288),
        "btc_realized_vol_1h": btc_ret.rolling(12).std(),
        "btc_realized_vol_1d": btc_ret.rolling(288).std(),
        "btc_realized_vol_30d": btc_ret.rolling(8640).std(),
    })
    btc_ema_long = btc_close.ewm(span=48, adjust=False).mean()
    btc_features["btc_ema_slope_4h"] = (btc_ema_long - btc_ema_long.shift(12)) / btc_close
    btc_features = btc_features.shift(1).reset_index().rename(columns={"index": "open_time"})

    # ====== Per-name features (loop over symbols) ======
    print("  Per-name BTC-relative features...")
    BETA_WINDOW = 288
    name_rows = []
    for s in sym_list:
        c = closes[s]
        # Build a single aligned frame so all derived series share index/length
        df_n = pd.DataFrame({
            "my_close": c,
            "my_ret": c.pct_change(),
            "btc_close": btc_close.reindex(c.index, method="ffill"),
            "btc_ret": btc_ret.reindex(c.index, method="ffill"),
            "xs_mean": xs_per_bar_mean.reindex(c.index, method="ffill"),
        }).dropna(subset=["my_ret", "btc_ret"])
        if len(df_n) < BETA_WINDOW + 100: continue

        # β + β change + corr (PIT shift)
        cov = (df_n["my_ret"] * df_n["btc_ret"]).rolling(BETA_WINDOW).mean() - \
              df_n["my_ret"].rolling(BETA_WINDOW).mean() * df_n["btc_ret"].rolling(BETA_WINDOW).mean()
        var = df_n["btc_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        beta_pit = (cov / var).clip(-5, 5).shift(1)
        std_my = df_n["my_ret"].rolling(BETA_WINDOW).std()
        std_btc = df_n["btc_ret"].rolling(BETA_WINDOW).std()
        corr_pit = (cov / (std_my * std_btc).replace(0, np.nan)).clip(-1, 1).shift(1)

        # idio (residual) + idio vol + tail features
        idio = df_n["my_ret"] - beta_pit * df_n["btc_ret"]
        idio_vol_1h = idio.rolling(12).std()
        idio_vol_1d = idio.rolling(288).std()
        idio_max_abs_12b = idio.rolling(12).apply(lambda x: np.max(np.abs(x)), raw=True)
        idio_skew_1d = idio.rolling(288).skew()
        idio_kurt_1d = idio.rolling(288).kurt()

        # Recent residual returns over 12b and 48b windows
        my_ret_12b = df_n["my_close"].pct_change(12)
        btc_ret_12b = df_n["btc_close"].pct_change(12)
        my_ret_48b = df_n["my_close"].pct_change(48)
        btc_ret_48b = df_n["btc_close"].pct_change(48)
        idio_ret_12b = my_ret_12b - beta_pit * btc_ret_12b
        idio_ret_48b = my_ret_48b - beta_pit * btc_ret_48b

        # β change over 5d (1440 bars)
        beta_change_5d = beta_pit - beta_pit.shift(1440)

        # Factor loading: corr(idio, xs_mean) over rolling window
        xs_mean = df_n["xs_mean"]
        cov_factor = (idio * xs_mean).rolling(BETA_WINDOW).mean() - \
                      idio.rolling(BETA_WINDOW).mean() * xs_mean.rolling(BETA_WINDOW).mean()
        var_factor = xs_mean.rolling(BETA_WINDOW).var().replace(0, np.nan)
        var_idio = idio.rolling(BETA_WINDOW).var().replace(0, np.nan)
        factor_loading = (cov_factor / np.sqrt(var_factor * var_idio).replace(0, np.nan)).clip(-1, 1).shift(1)
        idio_share = (1 - factor_loading**2).clip(0, 1)

        # All series have same length and index = df_n.index
        sub = pd.DataFrame({
            "open_time": df_n.index,
            "symbol": s,
            "beta_to_btc_change_5d": beta_change_5d.values,
            "corr_to_btc_1d": corr_pit.values,
            "idio_vol_to_btc_1h": idio_vol_1h.values,
            "idio_vol_to_btc_1d": idio_vol_1d.values,
            "idio_ret_to_btc_12b": idio_ret_12b.values,
            "idio_ret_to_btc_48b": idio_ret_48b.values,
            "idio_max_abs_12b": idio_max_abs_12b.values,
            "idio_skew_1d": idio_skew_1d.values,
            "idio_kurt_1d": idio_kurt_1d.values,
            "name_factor_loading_1d": factor_loading.values,
            "name_idio_share_1d": idio_share.values,
        })
        name_rows.append(sub)
    name_features_df = pd.concat(name_rows, ignore_index=True)

    # Merge all into panel
    print("  Merging features into panel...")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    name_features_df["open_time"] = pd.to_datetime(name_features_df["open_time"], utc=True)
    xs_features["open_time"] = pd.to_datetime(xs_features["open_time"], utc=True)
    btc_features["open_time"] = pd.to_datetime(btc_features["open_time"], utc=True)

    panel = panel.merge(name_features_df, on=["symbol", "open_time"], how="left")
    panel = panel.merge(xs_features, on="open_time", how="left")
    panel = panel.merge(btc_features, on="open_time", how="left")

    return panel


def main():
    print("Loading 51-name panel...")
    panel = build_panel_51(force_rebuild=False)

    print("\nEngineering BTC-target features...")
    panel = add_btc_target_features(panel)
    print(f"  Final panel: {len(panel):,} rows × {panel.shape[1]} cols")

    # New feature names
    NEW_FEATS = [
        "xs_alpha_mean_12b", "xs_alpha_dispersion_12b", "xs_alpha_iqr_12b",
        "xs_alpha_mean_48b", "xs_alpha_dispersion_48b",
        "btc_ret_12b", "btc_ret_48b", "btc_ret_288b",
        "btc_realized_vol_1h", "btc_realized_vol_1d", "btc_realized_vol_30d",
        "btc_ema_slope_4h",
        "beta_to_btc_change_5d", "corr_to_btc_1d",
        "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
        "idio_ret_to_btc_12b", "idio_ret_to_btc_48b",
        "idio_max_abs_12b", "idio_skew_1d", "idio_kurt_1d",
        "name_factor_loading_1d", "name_idio_share_1d",
    ]
    NEW_FEATS = [f for f in NEW_FEATS if f in panel.columns]
    print(f"\n  New features added ({len(NEW_FEATS)}): {NEW_FEATS}")

    # Save
    panel.to_parquet(PANEL_WITH_FEATURES, compression="zstd")
    print(f"  saved → {PANEL_WITH_FEATURES}")

    # Single-feature IC test on training subset
    print("\nSingle-feature IC against btc_target...")
    folds = _multi_oos_splits(panel)
    fold0 = folds[0]
    train, _, _ = _slice(panel, fold0)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    tr = tr.dropna(subset=["btc_target"])
    print(f"  Training subset: {len(tr):,} rows")

    rows = []
    for f in NEW_FEATS:
        sub = tr[[f, "btc_target"]].dropna()
        if len(sub) < 1000: continue
        ic = float(sub[f].rank().corr(sub["btc_target"].rank()))
        rows.append({"feature": f, "ic_btc": ic, "abs_ic": abs(ic), "n": len(sub)})
    ic_df = pd.DataFrame(rows).sort_values("abs_ic", ascending=False)

    print("\n" + "=" * 90)
    print("PHASE 3: NEW FEATURES — IC vs btc_target (sorted by |IC|)")
    print("=" * 90)
    print(f"  {'rank':>4}  {'feature':<32}  {'IC_btc':>9}  {'recommendation':>14}")
    keep = []
    for i, r in enumerate(ic_df.itertuples(), 1):
        if r.abs_ic > 0.04: rec = "KEEP (strong)"
        elif r.abs_ic > 0.02: rec = "CONSIDER"
        elif r.abs_ic > 0.005: rec = "WEAK keep"
        else: rec = "DROP"
        if r.abs_ic > 0.005: keep.append(r.feature)
        print(f"  {i:>4}  {r.feature:<32}  {r.ic_btc:>+9.4f}  {rec:>14}")
    print(f"\n  Kept ({len(keep)}/{len(NEW_FEATS)}): {keep}")
    ic_df.to_csv(OUT_DIR / "new_features_ic.csv", index=False)
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
