"""User's refinement: each symbol's book signal AGAINST BTC's. Feature = symbol's imbalance RELATIVE to BTC's
(rel_imb = imb_sym - imb_BTC), plus its dynamics; target = symbol's return RELATIVE to BTC (alpha_vs_btc, and a
simple fwd_sym - fwd_BTC). So: when a symbol's book leans more bid than BTC's (and stays/strengthens), does the symbol
OUTPERFORM BTC forward? Both eras, day-clustered CI. This is the beta-neutral / relative reading of the idea.

features: rel_lvl (imb_sym-imb_BTC), rel_d1 (its 1-bar change), rel_mom3 (3-bar), rel_absd (|.| build)
targets:  alpha_vs_btc_realized (panel, beta-adjusted vs BTC); fwd_rel_4h / fwd_rel_1d (sym fwd - BTC fwd, raw relative)
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
KD = Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines")
CACHE = Path("/home/yuqing/ctaNew/data/ml/cache")
rng = np.random.default_rng(9)

def close_4h(sym, lo, hi):
    fs = [f for f in glob.glob(str(KD / sym / "5m" / "*.parquet")) if lo <= Path(f).stem <= hi]
    if not fs: return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(fs)], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    s = df.drop_duplicates("open_time").set_index("open_time")["close"].sort_index()
    grid = pd.date_range(s.index.min().floor("4h"), s.index.max().ceil("4h"), freq="4h", tz="UTC")
    return s.reindex(grid, method="ffill")

def imb_series(sym):
    d = pd.read_parquet(CACHE / f"l2_{sym}.parquet")[["l2_imb1"]]
    d.index = pd.to_datetime(d.index, utc=True) + pd.Timedelta("4h")     # obs bar -> decision bar (PIT)
    return d["l2_imb1"].sort_index()

def day_boot(ic):
    s = pd.DataFrame({"ic": ic.values}, index=pd.to_datetime(ic.index, utc=True)); s["d"] = s.index.floor("1D")
    g = [x["ic"].values for _, x in s.groupby("d")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(2500)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))

def pooled_ic(df, feat, tgt):
    return df.groupby("open_time").apply(
        lambda g: g[feat].corr(g[tgt], method="spearman") if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()

def main():
    btc_imb = imb_series("BTCUSDT")
    btc_c = None
    rows = []
    files = glob.glob(str(CACHE / "l2_*.parquet"))
    for i, f in enumerate(files):
        sym = Path(f).stem[3:]
        im = imb_series(sym)
        rel = (im - btc_imb.reindex(im.index)).dropna()
        if not len(rel): continue
        d = pd.DataFrame(index=rel.index)
        d["rel_lvl"] = rel
        d["rel_d1"] = rel.diff()
        d["rel_mom3"] = rel - rel.shift(3)
        d["rel_absd"] = rel.abs() - rel.abs().shift(1)
        lo, hi = str(d.index.min().date()), str((d.index.max() + pd.Timedelta("2D")).date())
        c = close_4h(sym, lo, hi)
        if c is None: continue
        if btc_c is None: btc_c = close_4h("BTCUSDT", lo, "2027-01-01")
        c = c.reindex(d.index.union(c.index)).sort_index()
        bc = btc_c.reindex(c.index, method="ffill")
        fwd4 = (c.shift(-1) / c - 1); fwd1d = (c.shift(-6) / c - 1)
        bf4 = (bc.shift(-1) / bc - 1); bf1d = (bc.shift(-6) / bc - 1)
        d["fwd_rel_4h"] = (fwd4 - bf4).reindex(d.index)
        d["fwd_rel_1d"] = (fwd1d - bf1d).reindex(d.index)
        d["symbol"] = sym; d["open_time"] = d.index
        rows.append(d.reset_index(drop=True))
        if (i + 1) % 40 == 0: print(f"  built {i+1}/{len(files)}", flush=True)
    m = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    m = m.merge(pan, on=["symbol", "open_time"], how="left")
    cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"\nmerged {len(m)} rows | {m.symbol.nunique()} syms | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}")
    print("feature = symbol imbalance RELATIVE to BTC; IC>0 = more-bid-than-BTC -> OUTPERFORMS BTC forward\n")
    for tgt in ["fwd_rel_4h", "fwd_rel_1d", "alpha_vs_btc_realized"]:
        print(f"### target = {tgt} (symbol return vs BTC) ###")
        print(f"{'feature':9s} | {'RECENT IC [CI]':26s} | {'OOS IC [CI]':26s} | both-era?")
        for feat in ["rel_lvl", "rel_d1", "rel_mom3", "rel_absd"]:
            cells = {}
            for era, sub in eras.items():
                ic = pooled_ic(sub, feat, tgt); lo, up = day_boot(ic)
                cells[era] = (ic.mean(), lo, up)
            (ra, rl, ru), (oa, ol, ou) = cells["RECENT"], cells["OOS"]
            both = "YES" if (np.sign(ra) == np.sign(oa) and abs(ra) > 0.02 and abs(oa) > 0.02
                             and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
            print(f"{feat:9s} | {ra:+.3f} [{rl:+.3f},{ru:+.3f}] | {oa:+.3f} [{ol:+.3f},{ou:+.3f}] | {both}")
        print()
    print("VSBTCDONE")

if __name__ == "__main__":
    main()
