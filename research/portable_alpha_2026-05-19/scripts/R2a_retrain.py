"""R2a — add rvol_7d + ret_3d + btc_rvol_7d as MODEL features (clean A/B).

Replicates the EXACT production harness that built
outputs/vBTC_audit_panel/all_predictions.parquet
(ml/research/alpha_vBTC_build_audit_panel.py: WINNER_21 incl. sym_id,
train_fold_restricted = autocorr>=0.5 filter + 60d listing eligibility +
target_A + 5-seed _train), changing ONLY the feature set by appending the
3 pre-registered PIT features. Output schema == cached APD so the V3.1
sleeve machinery consumes it unchanged.

Locked feature spec (PLAN.md):
  rvol_7d     = std(log 5m-returns over 288*7 bars).shift(1)   [per symbol]
  ret_3d      = close.pct_change(288*3).shift(1)                [per symbol]
  btc_rvol_7d = BTCUSDT rvol_7d, broadcast by open_time
  winsor +-5 robust-sigma using FOLD-0 TRAIN rows only (PIT-safe).
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
import ml.research.alpha_vBTC_build_audit_panel as BA  # exact prod harness

PANEL = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
CW = REPO / "research/portable_alpha_2026-05-19/results/_cache/close_wide.parquet"
OUTP = REPO / "research/portable_alpha_2026-05-19/results/_cache/all_predictions_R2a.parquet"
NEW = ["rvol_7d", "ret_3d", "btc_rvol_7d"]


def build_new_features(panel):
    cw = pd.read_parquet(CW)
    px = cw[[c for c in cw.columns if c.startswith("c_")]].rename(columns=lambda x: x[2:])
    px = px.sort_index()
    logret = np.log(px / px.shift(1))
    rvol = logret.rolling(288 * 7, min_periods=288).std().shift(1)
    ret3 = px.pct_change(288 * 3).shift(1)
    btc = rvol["BTCUSDT"] if "BTCUSDT" in rvol.columns else rvol.iloc[:, 0]
    # long-format frames keyed (symbol, open_time)
    rv = rvol.reset_index().melt("open_time", var_name="symbol", value_name="rvol_7d")
    r3 = ret3.reset_index().melt("open_time", var_name="symbol", value_name="ret_3d")
    feat = rv.merge(r3, on=["open_time", "symbol"], how="outer")
    btc_df = btc.rename("btc_rvol_7d").reset_index()
    feat = feat.merge(btc_df, on="open_time", how="left")
    panel = panel.merge(feat, on=["open_time", "symbol"], how="left")
    return panel


def winsor_fold0(panel, folds0_train_mask):
    for c in NEW:
        tr = panel.loc[folds0_train_mask, c].dropna()
        if len(tr) < 100:
            continue
        med = float(tr.median())
        mad = float((tr - med).abs().median()) * 1.4826
        if mad <= 0:
            mad = float(tr.std()) or 1.0
        lo, hi = med - 5 * mad, med + 5 * mad
        panel[c] = panel[c].clip(lo, hi)
    return panel


def main():
    t0 = time.time()
    print("R2a — retrain WINNER_21 + [rvol_7d, ret_3d, btc_rvol_7d]", flush=True)
    panel = pd.read_parquet(PANEL)
    panel = build_new_features(panel)
    folds_all = BA._multi_oos_splits(panel)
    # fold-0 train rows (earliest; PIT-safe winsor reference)
    tr0, _, _ = BA._slice(panel, folds_all[0])
    panel = winsor_fold0(panel, panel.index.isin(tr0.index))
    na = {c: float(panel[c].isna().mean()) for c in NEW}
    print(f"  features built; NaN frac {na}", flush=True)

    feat_set = [f for f in BA.WINNER_21 if f in panel.columns] + NEW
    print(f"  feat_set n={len(feat_set)} (WINNER_21={len(BA.WINNER_21)} + {NEW})",
          flush=True)
    listings = BA.get_listing_dates_from_klines()
    pf = panel.groupby("symbol")["open_time"].min()
    for s, t in pf.items():
        if s not in listings:
            listings[s] = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
    syms = set(panel["symbol"].unique())

    def elig(ts_in):
        ts = (pd.Timestamp(ts_in, unit="ms", tz="UTC")
              if isinstance(ts_in, (int, np.integer))
              else pd.Timestamp(ts_in))
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        cut = ts - pd.Timedelta(days=BA.MIN_HISTORY_DAYS)
        return {s for s in syms if listings.get(s) and listings[s] <= cut}

    all_preds = []
    for fid in BA.ALL_FOLDS:
        if fid >= len(folds_all):
            continue
        tf = time.time()
        eligible = elig(folds_all[fid]["cal_start"])
        td, p = BA.train_fold_restricted(panel, folds_all[fid], feat_set, eligible)
        if td is None:
            print(f"  fold {fid}: skipped", flush=True)
            continue
        cols = ["symbol", "open_time", "alpha_A", "return_pct"]
        if "exit_time" in td.columns:
            cols.append("exit_time")
        df = td[cols].copy()
        df["pred"] = p
        df["fold"] = fid
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=BA.HORIZON * 5)
        all_preds.append(df)
        print(f"  fold {fid}: n={len(td):,} ({time.time()-tf:.0f}s)", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    apd.to_parquet(OUTP, index=False)
    print(f"\nR2a predictions saved: {len(apd):,} rows -> {OUTP} "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
