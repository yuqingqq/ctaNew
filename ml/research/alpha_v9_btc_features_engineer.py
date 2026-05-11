"""Phase 2: engineer + IC-test BTC-specific features.

Adds candidate BTC-specific features that weren't in v6_clean (which was
curated for basket-target). Tests single-feature IC against btc_beta_target
and decides keep / drop.

Candidates:
  beta_to_btc                    — name's rolling β to BTC (already computed)
  beta_to_btc_change_5d          — β shift over 5d (regime indicator)
  corr_to_btc_1d                 — rolling 1d corr name vs BTC
  corr_to_btc_change_3d          — corr regime shift
  idio_vol_to_btc_1d             — std of (return − β × btc_return) over 1d
  idio_vol_to_btc_1h             — same, 1h scale
  dom_btc_level                  — log(my_close / btc_close)
  dom_btc_change_48b             — change in dom over 48b
  dom_btc_change_288b            — change in dom over 1d
  dom_btc_z_1d                   — z-score of dom over 1d
  idio_ret_to_btc_48b            — beta-adjusted residual return over 48b
  idio_ret_to_btc_12b            — same, 12b

For each: compute, then Spearman IC against btc_beta_target. Keep features
with |IC| > 0.005 (basket-target threshold convention).
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import list_universe, build_kline_features
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice
from ml.research.alpha_v8_h48_audit import build_wide_panel
from ml.research.alpha_v9_btc_beta_target import add_btc_beta_target

OUT_DIR = REPO / "outputs/btc_features_engineer"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RC = 0.50
THRESHOLD = 1 - RC
BETA_WINDOW = 288
BTC_SYMBOL = "BTCUSDT"


def compute_btc_specific_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute name-vs-BTC features per (sym, time) and merge into panel."""
    print("  Loading per-symbol kline data for BTC-specific features...")
    btc_feats = build_kline_features(BTC_SYMBOL)
    btc_close = btc_feats["close"].copy()
    btc_close.index = pd.to_datetime(btc_close.index, utc=True)
    btc_ret = btc_close.pct_change()

    syms = sorted(panel["symbol"].unique())
    rows = []
    for s in syms:
        f = build_kline_features(s)
        if f.empty: continue
        my_close = f["close"].copy()
        my_close.index = pd.to_datetime(my_close.index, utc=True)
        my_ret = my_close.pct_change()
        joined = pd.DataFrame({"my_ret": my_ret, "btc_ret": btc_ret,
                                 "my_close": my_close, "btc_close": btc_close.reindex(my_close.index, method="ffill")})
        joined = joined.dropna(subset=["my_ret", "btc_ret"])
        if len(joined) < BETA_WINDOW + 50: continue

        # Beta to BTC (rolling 1d, PIT shift)
        cov = (joined["my_ret"] * joined["btc_ret"]).rolling(BETA_WINDOW).mean() - \
              joined["my_ret"].rolling(BETA_WINDOW).mean() * joined["btc_ret"].rolling(BETA_WINDOW).mean()
        var = joined["btc_ret"].rolling(BETA_WINDOW).var().replace(0, np.nan)
        beta_btc = (cov / var).clip(-5, 5).shift(1)

        # β change over 5d (1440 bars)
        beta_change_5d = beta_btc - beta_btc.shift(1440)

        # Correlation to BTC (rolling 1d)
        std_my = joined["my_ret"].rolling(BETA_WINDOW).std()
        std_btc = joined["btc_ret"].rolling(BETA_WINDOW).std()
        corr_btc = (cov / (std_my * std_btc).replace(0, np.nan)).clip(-1, 1).shift(1)
        corr_change_3d = corr_btc - corr_btc.shift(864)

        # Idiosyncratic-to-BTC residual: my_ret - beta × btc_ret
        beta_pit = beta_btc
        idio_1bar = joined["my_ret"] - beta_pit * joined["btc_ret"]
        idio_vol_1h = idio_1bar.rolling(12).std()
        idio_vol_1d = idio_1bar.rolling(288).std()

        # Idio returns over 48b and 12b
        my_fwd_48b = joined["my_close"].pct_change(48)
        btc_fwd_48b = joined["btc_close"].pct_change(48)
        idio_ret_48b = my_fwd_48b - beta_pit * btc_fwd_48b
        my_fwd_12b = joined["my_close"].pct_change(12)
        btc_fwd_12b = joined["btc_close"].pct_change(12)
        idio_ret_12b = my_fwd_12b - beta_pit * btc_fwd_12b

        # Dominance vs BTC: log(my_close / btc_close)
        dom = np.log(joined["my_close"] / joined["btc_close"])
        dom_change_48b = dom - dom.shift(48)
        dom_change_288b = dom - dom.shift(288)
        dom_rmean_1d = dom.rolling(288, min_periods=48).mean()
        dom_rstd_1d = dom.rolling(288, min_periods=48).std().replace(0, np.nan)
        dom_z_1d = ((dom - dom_rmean_1d) / dom_rstd_1d).clip(-5, 5)

        sub = pd.DataFrame({
            "open_time": joined.index,
            "symbol": s,
            "beta_to_btc": beta_btc.values,
            "beta_to_btc_change_5d": beta_change_5d.values,
            "corr_to_btc_1d": corr_btc.values,
            "corr_to_btc_change_3d": corr_change_3d.values,
            "idio_vol_to_btc_1h": idio_vol_1h.values,
            "idio_vol_to_btc_1d": idio_vol_1d.values,
            "idio_ret_to_btc_48b": idio_ret_48b.values,
            "idio_ret_to_btc_12b": idio_ret_12b.values,
            "dom_btc_level": dom.values,
            "dom_btc_change_48b": dom_change_48b.values,
            "dom_btc_change_288b": dom_change_288b.values,
            "dom_btc_z_1d": dom_z_1d.values,
        })
        rows.append(sub)
    btc_feats_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(f"    {len(btc_feats_df):,} (sym, time) BTC-feature rows")

    # Merge into panel by (symbol, open_time)
    panel = panel.copy()
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    btc_feats_df["open_time"] = pd.to_datetime(btc_feats_df["open_time"], utc=True)
    panel = panel.merge(btc_feats_df, on=["symbol", "open_time"], how="left",
                          suffixes=("", "_dup"))
    # Drop duplicated columns from earlier merges
    panel = panel.loc[:, ~panel.columns.str.endswith("_dup")]
    return panel, list(btc_feats_df.columns[2:])  # skip open_time, symbol


def main():
    print("Building panel + BTC target + BTC features...")
    panel = build_wide_panel()
    panel = add_btc_beta_target(panel)
    panel, new_features = compute_btc_specific_features(panel)
    print(f"\n  Added {len(new_features)} new BTC-specific features:")
    for f in new_features:
        non_nan = panel[f].notna().sum()
        print(f"    {f:<32}  non-NaN: {non_nan:,}")

    # Compute IC on training subset
    folds = _multi_oos_splits(panel)
    fold0 = folds[0]
    train, _, _ = _slice(panel, fold0)
    tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
    tr = tr.dropna(subset=["btc_beta_target"])
    print(f"\n  IC computation on {len(tr):,} training rows")

    rows = []
    for feat in new_features:
        sub = tr[[feat, "btc_beta_target", "demeaned_target"]].dropna()
        if len(sub) < 1000:
            print(f"    {feat}: insufficient non-NaN ({len(sub)}) — skip")
            continue
        ic_btc = float(sub[feat].rank().corr(sub["btc_beta_target"].rank()))
        ic_basket = float(sub[feat].rank().corr(sub["demeaned_target"].rank()))
        rows.append({
            "feature": feat,
            "ic_btc": ic_btc,
            "ic_basket": ic_basket,
            "abs_ic_btc": abs(ic_btc),
        })
    df = pd.DataFrame(rows).sort_values("abs_ic_btc", ascending=False)

    print("\n" + "=" * 90)
    print("PHASE 2: NEW BTC-SPECIFIC FEATURES — IC RANKING")
    print("=" * 90)
    print(f"  {'rank':>4}  {'feature':<32}  {'IC_btc':>9}  {'IC_basket':>10}  {'recommendation':>14}")
    keep = []
    for i, r in enumerate(df.itertuples(), 1):
        if r.abs_ic_btc > 0.04: rec = "KEEP (strong)"
        elif r.abs_ic_btc > 0.02: rec = "CONSIDER"
        elif r.abs_ic_btc > 0.005: rec = "WEAK keep"
        else: rec = "DROP"
        if r.abs_ic_btc > 0.005: keep.append(r.feature)
        print(f"  {i:>4}  {r.feature:<32}  {r.ic_btc:>+9.4f}  {r.ic_basket:>+10.4f}  {rec:>14}")

    print(f"\n  Features kept (|IC| > 0.005): {len(keep)}")
    print(f"  → {keep}")

    df.to_csv(OUT_DIR / "new_features_ic.csv", index=False)
    with open(OUT_DIR / "kept_new_features.txt", "w") as f:
        for feat in keep:
            f.write(f"{feat}\n")
    print(f"\n  saved → {OUT_DIR}")


if __name__ == "__main__":
    main()
