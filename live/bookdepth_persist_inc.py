"""DECISIVE test: is the sustained-imbalance signal (imb_ewma / imb_mean12) INCREMENTAL to the full V0 feature set, or
redundant? Partial cross-sectional rank-IC vs alpha_vs_btc, controlling for ALL 17 V0 features, both eras, day-boot CI.
If partial ~ raw and CI-off-zero both eras => a genuine NEW book feature. If partial collapses => already captured.
(Same test that killed l2_liq1.) No klines needed — target is the panel's alpha_vs_btc.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_persist import persist_feats
rng = np.random.default_rng(29)
V0 = ["return_1d", "atr_pct", "vwap_slope_96", "bars_since_high", "autocorr_pctile_7d", "obv_z_1d", "corr_to_btc_1d",
      "beta_to_btc_change_5d", "idio_vol_to_btc_1h", "idio_vol_to_btc_1d", "funding_rate", "funding_rate_z_7d",
      "funding_rate_1d_change", "rvol_7d", "ret_3d", "btc_rvol_7d", "bars_since_high_xs_rank"]
FEATS = ["imb_ewma", "imb_mean12", "imb_run", "imb_z30"]
TGT = "alpha_vs_btc_realized"

def xrank(df, cols): return df.groupby("open_time")[cols].rank(pct=True)
def resid(y, X):
    A = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(A, y, rcond=None); return y - A @ b
def day_boot(rx, ry, days):
    d = pd.DataFrame({"rx": rx, "ry": ry, "day": days}); g = [x for _, x in d.groupby("day")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [pd.concat([g[i] for i in rng.integers(0, len(g), len(g))]).pipe(lambda s: s["rx"].corr(s["ry"])) for _ in range(1500)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))
def partial(sub, feat):
    sub = sub.dropna(subset=[feat, TGT] + V0).copy()
    if len(sub) < 500: return (np.nan,) * 4
    R = xrank(sub, V0 + [feat, TGT]).fillna(0.5)
    ry = resid(R[TGT].values, R[V0].values); rx = resid(R[feat].values, R[V0].values)
    raw = np.corrcoef(R[feat].values, R[TGT].values)[0, 1]; part = np.corrcoef(rx, ry)[0, 1]
    lo, up = day_boot(rx, ry, sub["open_time"].dt.floor("1D").values)
    return raw, part, lo, up

def main():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]
        d0 = pd.read_parquet(f)[["l2_imb1"]]; d0.index = pd.to_datetime(d0.index, utc=True) + pd.Timedelta("4h")
        pf = persist_feats(d0["l2_imb1"].sort_index())
        pf["symbol"] = sym; pf["open_time"] = pf.index; rows.append(pf.reset_index(drop=True))
    m = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", TGT] + V0)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    m = m.merge(pan, on=["symbol", "open_time"], how="inner")
    cut = pd.Timestamp("2025-10-01", tz="UTC"); eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"merged {len(m)} rows | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])} | target={TGT}, control=17 V0 feats\n")
    print(f"{'feature':11s} | RECENT raw->partial [CI]        | OOS raw->partial [CI]           | incremental both?")
    for feat in FEATS:
        (rr, rp, rl, ru) = partial(eras["RECENT"], feat); (orr, op, ol, ou) = partial(eras["OOS"], feat)
        inc = "YES" if (np.sign(rp) == np.sign(op) and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
        print(f"{feat:11s} | {rr:+.3f}->{rp:+.3f} [{rl:+.3f},{ru:+.3f}] | {orr:+.3f}->{op:+.3f} [{ol:+.3f},{ou:+.3f}] | {inc}")
    print("\nincremental YES (partial CI-off-zero both eras) => genuine new book feature. INCDONE")

if __name__ == "__main__":
    main()
