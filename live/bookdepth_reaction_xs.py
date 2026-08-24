"""DECISIVE cross-sectional test of the price-validated absorption/reaction idea, done with the review's rigor.
Per eligible alt, PIT: ob_z = per-symbol trailing-z of ±1% imbalance; price_z = vol-normalized COMPLETED 1d return
(momentum). reaction = price_z - lambda*ob_z (strongly + when price rises despite ask-heavy book = 'asks failing';
strongly - when price falls despite bid-heavy book = 'bids failing'). Forward returns 4h/12h/1d.

THE make-or-break question is INCREMENTAL: does the book term add to PURE MOMENTUM?  So we report, per horizon, both
eras, day-clustered CI:
  price_z            cross-sectional rank-IC  (pure momentum BASELINE)
  reaction lambda=1/2 rank-IC                 (does adding -lambda*ob_z beat momentum?)
  partial ob_z|price_z rank-IC                (lambda-FREE: does ob_z predict fwd AFTER controlling for momentum?)
  pure ob_z          rank-IC                  (contrarian reference)
Fixed PIT universe (both-era), per-symbol standardization. If partial ob_z|price_z is ~0 / era-unstable -> the book
adds nothing to momentum = same wall as every prior OB test. If it's same-sign CI-off-zero BOTH eras -> real, escalate.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from live.bookdepth_timing_corrected import fixed_universe
CACHE = "/home/yuqing/ctaNew/data/ml/cache"
PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet"
CUT = pd.Timestamp("2025-10-01", tz="UTC")
rng = np.random.default_rng(101)

def build(syms):
    obz = []
    for s in syms:
        d = pd.read_parquet(f"{CACHE}/l2_{s}.parquet", columns=["l2_imb1"])
        d.index = pd.to_datetime(d.index, utc=True) + pd.Timedelta("4h")
        imb = d["l2_imb1"][~d.index.duplicated()].sort_index()
        z = (imb - imb.rolling(90, min_periods=45).mean()) / imb.rolling(90, min_periods=45).std()
        obz.append(pd.DataFrame({"symbol": s, "open_time": z.index, "ob_z": z.values}))
    OBZ = pd.concat(obz, ignore_index=True)
    p = pd.read_parquet(PANEL, columns=["symbol", "open_time", "return_pct"])
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p = p[p.symbol.isin(syms)].sort_values(["symbol", "open_time"])
    rows = []
    for s, g in p.groupby("symbol"):
        g = g.reset_index(drop=True); lr = np.log1p(g["return_pct"])
        trail = lr.shift(1).rolling(6, min_periods=6).sum()                       # COMPLETED 1d return (known at T)
        vol = lr.shift(1).rolling(30, min_periods=15).std() * np.sqrt(6)
        g["price_z"] = trail / vol.replace(0, np.nan)
        g["fwd_4h"] = g["return_pct"]
        for h, nm in [(3, "fwd_12h"), (6, "fwd_1d")]:
            g[nm] = np.expm1(lr[::-1].rolling(h, min_periods=h).sum()[::-1].values)  # forward h-bar return
        g["symbol"] = s
        rows.append(g)
    R = pd.concat(rows, ignore_index=True)
    return OBZ.merge(R, on=["symbol", "open_time"], how="inner")

def xic(df, feat, tgt):
    return df.groupby("open_time").apply(lambda g: spearmanr(g[feat], g[tgt]).correlation
                                         if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()

def partial_xic(df, feat, ctrl, tgt):
    def pb(g):
        gg = g[[feat, ctrl, tgt]].dropna()
        if len(gg) < 10: return np.nan
        X = np.column_stack([np.ones(len(gg)), gg[ctrl].values])
        resid = gg[feat].values - X @ np.linalg.lstsq(X, gg[feat].values, rcond=None)[0]
        return spearmanr(resid, gg[tgt].values).correlation
    return df.groupby("open_time").apply(pb).dropna()

def ci(ic):
    if len(ic) < 5: return (np.nan, np.nan, np.nan)
    s = pd.DataFrame({"v": ic.values}, index=pd.to_datetime(ic.index, utc=True)); s["d"] = s.index.floor("1D")
    g = [x["v"].values for _, x in s.groupby("d")]
    b = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(2000)]
    return (ic.mean(), *np.nanpercentile(b, [2.5, 97.5]))

def row(df, feat, tgt, ctrl=None):
    e = {}
    for era, m in [("OOS", df.open_time < CUT), ("REC", df.open_time >= CUT)]:
        sub = df[m]
        icv = partial_xic(sub, feat, ctrl, tgt) if ctrl else xic(sub, feat, tgt)
        e[era] = ci(icv)
    (ra, rl, ru), (oa, ol, ou) = e["OOS"], e["REC"]
    both = "BOTH✓" if (np.sign(ra) == np.sign(oa) and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
    return f"{ra:+.4f}[{rl:+.4f},{ru:+.4f}] | {oa:+.4f}[{ol:+.4f},{ou:+.4f}] | {both}"

def main():
    syms = fixed_universe(); D = build(syms)
    print(f"fixed universe {len(syms)} syms | panel {len(D)} rows | {D.open_time.min().date()}..{D.open_time.max().date()}\n")
    for tgt in ["fwd_4h", "fwd_12h", "fwd_1d"]:
        d = D.copy(); d["react1"] = d["price_z"] - 1.0 * d["ob_z"]; d["react2"] = d["price_z"] - 2.0 * d["ob_z"]
        print(f"### target = {tgt}  (rank-IC: OOS [CI] | RECENT [CI] | both-era?) ###")
        print(f"  price_z (momentum base) : {row(d, 'price_z', tgt)}")
        print(f"  reaction lambda=1       : {row(d, 'react1', tgt)}")
        print(f"  reaction lambda=2       : {row(d, 'react2', tgt)}")
        print(f"  ob_z | price_z (PARTIAL): {row(d, 'ob_z', tgt, ctrl='price_z')}   <- does book ADD to momentum?")
        print(f"  pure ob_z (contrarian)  : {row(d, 'ob_z', tgt)}")
        print()
    print("read: reaction beats price_z AND partial ob_z|price_z is same-sign CI-off-zero BOTH eras -> book validates")
    print("momentum = real, escalate. Else -> book adds nothing to momentum, same wall. REACTXSDONE")

if __name__ == "__main__":
    main()
