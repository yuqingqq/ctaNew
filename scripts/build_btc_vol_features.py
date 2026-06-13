"""Add PIT-correct VOLUME / liquidity / order-flow features to the full-PIT
btc-only panel (the panel itself keeps no raw volume; only derived obv_z_1d /
vol_zscore exist). Source = the same 5m klines the original builder used.

Produces a NEW augmented panel (does NOT overwrite the original — that file
underpins every Steps 76-80a result and the production comparison):
  outputs/vBTC_features_btc_only_111_volaug/panel_btc_only_111_volaug.parquet
  = existing full-PIT panel  LEFT-joined  with the new volume columns on
    (symbol, open_time). The existing panel is the SPINE so rows / order /
    folds / sigma_idio / alpha_beta are byte-identical -> Steps 76-80a stay
    directly comparable; volaug is a strict column superset.

PIT discipline (identical to build_btc_features_111_full_pit.py / obv_z_1d):
every rolling stat is computed on bars up to t then `.shift(1)`, so the value
at row t reflects a window ending at bar t-1 — known when a position is opened
at open_time t. No look-ahead. (Same shift level as the existing obv_z_1d,
the closest analog; NOT the .shift(49) used only for beta and the two Step-29
features.)

New features (qv=quote_volume, v=volume, tbqv=taker_buy_quote_volume,
cnt=count, r=close.pct_change()):
  qvol_z_1d / qvol_z_7d / qvol_z_30d  : z(qv) over 288 / 7d / 30d
  qvol_surge_1h_over_1d               : mean(qv,12) / mean(qv,288)
  dollar_vol_log_1d                   : log1p(mean(qv,288))   (liquidity scale)
  amihud_illiq_1d                     : log1p(mean(|r|/qv,288)) (illiquidity)
  taker_buy_frac_z_1d                 : z(tbqv/qv, 288)        (flow imbalance)
  signed_qvol_1h                      : sum(sign(r)*qv,12)/mean(qv,288)
  trade_size_z_1d                     : z(qv/cnt, 288)         (whale proxy)
"""
from __future__ import annotations
import sys, time, warnings, gc
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

PANEL = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT_DIR = REPO / "outputs/vBTC_features_btc_only_111_volaug"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAY = 288
NEW_COLS = ["qvol_z_1d", "qvol_z_7d", "qvol_z_30d", "qvol_surge_1h_over_1d",
            "dollar_vol_log_1d", "amihud_illiq_1d", "taker_buy_frac_z_1d",
            "signed_qvol_1h", "trade_size_z_1d"]
WANT = ["open_time", "close", "quote_volume", "volume",
        "taker_buy_quote_volume", "count"]


def load_klines(sym):
    sd = KLINES_DIR / sym / "5m"
    if not sd.exists():
        return None
    files = sorted(sd.glob("*.parquet"))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        for c in WANT:
            if c not in df.columns:
                df[c] = np.nan
        dfs.append(df[WANT])
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
    df = (df.dropna(subset=["open_time"]).drop_duplicates("open_time")
            .sort_values("open_time").reset_index(drop=True))
    return df


def _z(s, w):
    return (s - s.rolling(w).mean()) / s.rolling(w).std().replace(0, np.nan)


def vol_features(df):
    qv = df["quote_volume"].astype(float)
    cnt = df["count"].astype(float)
    tbqv = df["taker_buy_quote_volume"].astype(float)
    r = df["close"].astype(float).pct_change()
    out = pd.DataFrame(index=df.index)
    out["qvol_z_1d"] = _z(qv, DAY).shift(1)
    out["qvol_z_7d"] = _z(qv, 7 * DAY).shift(1)
    out["qvol_z_30d"] = _z(qv, 30 * DAY).shift(1)
    out["qvol_surge_1h_over_1d"] = (
        qv.rolling(12).mean() / qv.rolling(DAY).mean().replace(0, np.nan)
    ).shift(1)
    out["dollar_vol_log_1d"] = np.log1p(qv.rolling(DAY).mean()).shift(1)
    amihud = (r.abs() / qv.replace(0, np.nan)).rolling(DAY).mean()
    out["amihud_illiq_1d"] = np.log1p(amihud).shift(1)
    frac = tbqv / qv.replace(0, np.nan)
    out["taker_buy_frac_z_1d"] = _z(frac, DAY).shift(1)
    out["signed_qvol_1h"] = (
        (np.sign(r.fillna(0)) * qv).rolling(12).sum()
        / qv.rolling(DAY).mean().replace(0, np.nan)
    ).shift(1)
    ts = qv / cnt.replace(0, np.nan)
    out["trade_size_z_1d"] = _z(ts, DAY).shift(1)
    out["open_time"] = df["open_time"].values
    return out


def main():
    print("=== Build PIT volume features (volaug panel) ===\n", flush=True)
    t0 = time.time()
    panel = pd.read_parquet(PANEL)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    syms = sorted(panel["symbol"].unique())
    n0, c0 = len(panel), len(panel.columns)
    print(f"spine panel: {n0:,} rows x {c0} cols, {len(syms)} symbols", flush=True)

    parts, skipped = [], []
    for i, sym in enumerate(syms):
        kd = load_klines(sym)
        if kd is None or len(kd) < 1000:
            skipped.append(sym)
            continue
        vf = vol_features(kd)
        vf["symbol"] = sym
        parts.append(vf)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(syms)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  feature build done ({time.time()-t0:.0f}s); "
          f"skipped {len(skipped)}: {skipped[:6]}", flush=True)

    allv = pd.concat(parts, ignore_index=True)
    allv["open_time"] = pd.to_datetime(allv["open_time"], utc=True)
    allv["symbol"] = allv["symbol"].astype("category")
    panel["symbol"] = panel["symbol"].astype("category")
    merged = panel.merge(allv[["symbol", "open_time"] + NEW_COLS],
                         on=["symbol", "open_time"], how="left")
    del allv, parts
    gc.collect()

    assert len(merged) == n0, f"ROW COUNT CHANGED {len(merged)} != {n0}"
    assert all(c in merged.columns for c in panel.columns), "lost a spine col"
    print(f"\nmerged: {len(merged):,} rows (spine preserved) x "
          f"{len(merged.columns)} cols (+{len(NEW_COLS)} vol)", flush=True)
    print("\nNaN coverage of new vol features:", flush=True)
    for c in NEW_COLS:
        nn = merged[c].notna().mean()
        print(f"  {c:24s} non-NaN {nn*100:5.1f}%  "
              f"mean={merged[c].mean():+.3g} std={merged[c].std():.3g}",
              flush=True)

    out = OUT_DIR / "panel_btc_only_111_volaug.parquet"
    merged.to_parquet(out, index=False)
    print(f"\nSaved: {out}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
