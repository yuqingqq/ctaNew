"""Can imbstd (order-book quote instability) PREDICT forward VOLATILITY — and INCREMENTALLY to the model's existing
trailing-vol features? (User's flip: OB's redundant-for-alpha vol content might still be a useful LEADING vol
indicator for risk/sizing.) Cross-sectional rank-IC of imbstd vs forward realized vol (next-1d) + |next-4h return|,
BOTH eras, then PARTIAL-IC controlling for trailing rvol_7d/atr_pct/idio_vol (is it a LEAD, or just measuring current
vol the model already has?). If partial-IC is both-era positive, imbstd is genuinely useful on the risk side.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
rng = np.random.default_rng(13)
CTRL = ["rvol_7d", "atr_pct", "idio_vol_to_btc_1d"]   # the model's trailing-vol features

def build():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]; d = pd.read_parquet(f)
        if "l2_imbstd" not in d.columns: continue
        s = d["l2_imbstd"].sort_index(); s.index = pd.to_datetime(s.index, utc=True) + pd.Timedelta("4h")  # PIT
        rows.append(pd.DataFrame({"imbstd": s, "symbol": sym, "open_time": s.index}).reset_index(drop=True))
    L = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "return_pct"] + CTRL)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan = pan.sort_values(["symbol", "open_time"])
    # forward realized vol (next ~1d = 6 bars of 4h fwd returns) + next-4h abs move
    pan["fwd_vol1d"] = pan.groupby("symbol")["return_pct"].transform(lambda s: s.rolling(6).std().shift(-6))
    pan["fwd_absret"] = pan["return_pct"].abs()                      # next-4h realized magnitude (PIT: return_pct is fwd)
    return pan.merge(L, on=["symbol", "open_time"], how="inner")

def xrank(df, cols): return df.groupby("open_time")[cols].rank(pct=True)
def resid(y, X): A = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(A, y, rcond=None); return y - A @ b
def dayci(rx, ry, days):
    d = pd.DataFrame({"rx": rx, "ry": ry, "day": days}); g = [x for _, x in d.groupby("day")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [pd.concat([g[i] for i in rng.integers(0, len(g), len(g))]).pipe(lambda s: s["rx"].corr(s["ry"])) for _ in range(2000)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))

def test(sub, tgt):
    sub = sub.dropna(subset=["imbstd", tgt] + CTRL).copy()
    if len(sub) < 500: return None
    R = xrank(sub, ["imbstd", tgt] + CTRL).fillna(0.5)
    raw = R["imbstd"].corr(R[tgt])
    rx = resid(R["imbstd"].values, R[CTRL].values); ry = resid(R[tgt].values, R[CTRL].values)
    part = np.corrcoef(rx, ry)[0, 1]
    lo, up = dayci(rx, ry, sub["open_time"].dt.floor("1D").values)
    return raw, part, lo, up

def main():
    m = build(); cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"merged {len(m)} | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}")
    print("does imbstd predict forward vol? raw x-sec IC, then PARTIAL controlling trailing vol (=incremental/lead)\n")
    for tgt, lab in [("fwd_vol1d", "forward 1d realized vol"), ("fwd_absret", "next-4h |return| (realized magnitude)")]:
        print(f"### target = {lab} ###")
        for era, sub in eras.items():
            r = test(sub, tgt)
            if r is None: print(f"  {era}: n/a"); continue
            raw, part, lo, up = r
            f = "INCREMENTAL (CI>0)" if lo > 0 else ("neg" if up < 0 else "redundant (CI~0)")
            print(f"  {era}: raw IC {raw:+.3f} -> partial(|trailing-vol) {part:+.3f} [{lo:+.3f},{up:+.3f}] -> {f}")
        print()
    print("read: high raw IC = imbstd tracks vol; partial CI>0 both eras = it LEADS vol beyond trailing feats (useful). VOLPREDDONE")

if __name__ == "__main__":
    main()
