"""Build PIT OI / positioning features from cached Binance Vision metrics.

Source: data/ml/cache/metrics_<SYM>.parquet (5-min create_time snapshots):
  sum_open_interest, sum_open_interest_value,
  count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
  count_long_short_ratio, sum_taker_long_short_vol_ratio

23 panel∩cached symbols with full clean history (ETHUSDT dropped: ~8d only).

PIT alignment (the #1 hazard, resolved):
  metrics create_time=t is the OI snapshot *at instant t* (:05 grid, end of
  the t-5→t interval). Panel open_time=t is the bar opening at t (:00 grid).
  Recipe: floor create_time to 5min, last-per-bucket, reindex onto the panel
  open_time grid (ffill limit 2 bars = 10min, OI is a slow stock), then
  EVERY feature .shift(1) per symbol -> decision bar t uses only the
  snapshot from t-5min. Strictly PIT, identical convention to obv_z_1d /
  build_btc_vol_features.

Output: outputs/vBTC_features_oi/oi_panel.parquet  (symbol, open_time, +OI
features), to be merged onto the existing panel at eval time. Built-in audit:
independent recompute exact-match + shift proof + look-ahead IC<0.10.
"""
from __future__ import annotations
import sys, time, warnings, glob, os
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

PANEL = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"
CACHE = REPO / "data/ml/cache"
OUT_DIR = REPO / "outputs/vBTC_features_oi"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DAY = 288
DROP = {"ETHUSDT"}                       # only ~8d of OI cached


def _z(s, w):
    return (s - s.rolling(w).mean()) / s.rolling(w).std().replace(0, np.nan)


def _pchg(s, n):
    """pct_change with inf (zero-base / near-zero OI) -> NaN."""
    return s.pct_change(n).replace([np.inf, -np.inf], np.nan)


def oi_features(m: pd.Series, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """m = metrics DataFrame (create_time index). Align to panel grid, build
    PIT OI features (each .shift(1)). grid = this symbol's panel open_times."""
    mm = m.copy()
    mm.index = mm.index.floor("5min")
    mm = mm[~mm.index.duplicated(keep="last")].sort_index()
    a = mm.reindex(grid, method="ffill", limit=2)        # snap to panel grid
    oi = a["sum_open_interest"].astype(float)
    oiv = a["sum_open_interest_value"].astype(float)
    lsc = a["count_long_short_ratio"].astype(float)
    lst = a["sum_toptrader_long_short_ratio"].astype(float)
    lstk = a["sum_taker_long_short_vol_ratio"].astype(float)
    out = pd.DataFrame(index=grid)
    out["oi_chg_1h"] = _pchg(oi, 12).shift(1)
    out["oi_chg_4h"] = _pchg(oi, 48).shift(1)
    out["oi_chg_1d"] = _pchg(oi, DAY).shift(1)
    out["oi_z_1d"] = _z(oi, DAY).shift(1)
    out["oi_z_7d"] = _z(oi, 7 * DAY).shift(1)
    out["oiv_z_1d"] = _z(oiv, DAY).shift(1)
    out["ls_count_z_1d"] = _z(lsc, DAY).shift(1)
    out["ls_count_chg_4h"] = _pchg(lsc, 48).shift(1)
    out["ls_top_z_1d"] = _z(lst, DAY).shift(1)
    out["ls_taker_z_1d"] = _z(lstk, DAY).shift(1)
    out["ls_taker_chg_4h"] = _pchg(lstk, 48).shift(1)
    return out


FEATS = ["oi_chg_1h", "oi_chg_4h", "oi_chg_1d", "oi_z_1d", "oi_z_7d",
         "oiv_z_1d", "ls_count_z_1d", "ls_count_chg_4h", "ls_top_z_1d",
         "ls_taker_z_1d", "ls_taker_chg_4h"]


def main():
    print("=== Build PIT OI/positioning features (23 syms) ===\n", flush=True)
    t0 = time.time()
    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "alpha_beta"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    cached = {os.path.basename(f)[8:-8] for f in glob.glob(str(CACHE / "metrics_*.parquet"))}
    syms = sorted(s for s in pan["symbol"].unique()
                  if s in cached and s not in DROP)
    print(f"symbols: {len(syms)} (panel∩cached, ETHUSDT dropped)", flush=True)

    parts = []
    for s in syms:
        m = pd.read_parquet(CACHE / f"metrics_{s}.parquet")
        if len(m) < 50_000:                              # guard short history
            print(f"  SKIP {s}: only {len(m)} metrics rows", flush=True)
            continue
        if m.index.tz is None:
            m.index = m.index.tz_localize("UTC")
        grid = (pan[pan.symbol == s]["open_time"].sort_values()
                .drop_duplicates().reset_index(drop=True))
        gi = pd.DatetimeIndex(grid)
        f = oi_features(m, gi)
        f.insert(0, "symbol", s)
        f = f.reset_index().rename(columns={"index": "open_time"})
        parts.append(f)
    oip = pd.concat(parts, ignore_index=True)
    oip["open_time"] = pd.to_datetime(oip["open_time"], utc=True)
    print(f"\noi_panel: {len(oip):,} rows, {oip['symbol'].nunique()} syms, "
          f"{len(FEATS)} feats", flush=True)
    for c in FEATS:
        print(f"  {c:18s} non-NaN {oip[c].notna().mean()*100:5.1f}%  "
              f"mean={oip[c].mean():+.3g} std={oip[c].std():.3g}", flush=True)

    # ---- AUDIT A: independent recompute + shift proof (oi_chg_1d) ----
    print("\n--- AUDIT A: PIT recompute + shift proof (oi_chg_1d) ---", flush=True)
    okA = True
    for s in ["SOLUSDT", "ADAUSDT"]:
        m = pd.read_parquet(CACHE / f"metrics_{s}.parquet")
        if m.index.tz is None:
            m.index = m.index.tz_localize("UTC")
        mm = m.copy(); mm.index = mm.index.floor("5min")
        mm = mm[~mm.index.duplicated(keep="last")].sort_index()
        grid = pd.DatetimeIndex(pan[pan.symbol == s]["open_time"]
                                .sort_values().drop_duplicates())
        oi = mm["sum_open_interest"].astype(float).reindex(grid, method="ffill",
                                                           limit=2)
        ref_shift1 = oi.pct_change(DAY).shift(1)
        ref_unshift = oi.pct_change(DAY)
        st = oip[oip.symbol == s].set_index("open_time")["oi_chg_1d"].reindex(grid)
        v = pd.DataFrame({"st": st, "s1": ref_shift1, "uns": ref_unshift,
                          "oi_t": oi}).dropna(subset=["st", "s1"])
        c1 = float(v["st"].corr(v["s1"])); md = float((v["st"]-v["s1"]).abs().max())
        cu = float(v["st"].corr(v["uns"])); co = float(v["st"].corr(v["oi_t"]))
        good = c1 > 0.9999 and md < 1e-9
        okA &= good
        print(f"  {s}: n={len(v):,} corr(stored,indep_SHIFT1)={c1:.6f} "
              f"maxdiff={md:.1e}  corr(stored,UNSHIFTED)={cu:.3f} "
              f"corr(stored,oi[t])={co:+.3f} -> "
              f"{'OK (= t-1 snapshot, not t)' if good else 'MISMATCH'}",
              flush=True)

    # ---- AUDIT B: look-ahead IC vs fwd alpha_beta (OOS decision cadence) ----
    print("\n--- AUDIT B: look-ahead IC vs fwd alpha_beta ---", flush=True)
    full = pd.read_parquet(PANEL, columns=["symbol", "open_time", "alpha_beta",
                                           "exit_time"])
    full["open_time"] = pd.to_datetime(full["open_time"], utc=True)
    j = full.merge(oip, on=["symbol", "open_time"], how="inner")
    folds = _multi_oos_splits(j)
    j["fold"] = -1
    for fid in range(len(folds)):
        te = _slice(j, folds[fid])[2]
        j.loc[te.index, "fold"] = fid
    oos = j[j["fold"].between(1, 9)]
    grid = sorted(oos["open_time"].unique())[::48]
    dec = oos[oos["open_time"].isin(set(grid))]
    okB = True
    for c in FEATS:
        ics = []
        for _, g in dec.groupby("open_time"):
            vv = g[[c, "alpha_beta"]].dropna()
            if len(vv) >= 5 and vv[c].std() > 1e-12 and vv["alpha_beta"].std() > 1e-12:
                ics.append(vv[c].corr(vv["alpha_beta"], method="spearman"))
        mu = float(np.mean(ics)) if ics else np.nan
        flag = abs(mu) > 0.10
        okB &= not flag
        print(f"  {c:18s} mean cycle-IC={mu:+.4f} n={len(ics)} "
              f"{'<<< SUSPECT' if flag else 'ok'}", flush=True)

    out = OUT_DIR / "oi_panel.parquet"
    oip.to_parquet(out, index=False)
    print(f"\nPIT AUDIT: {'PASS' if (okA and okB) else 'FAIL'}", flush=True)
    print(f"Saved: {out} ({len(oip):,} rows x {oip.shape[1]} cols)\n"
          f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
