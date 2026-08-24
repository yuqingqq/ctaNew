"""User's hypothesis: does the CHANGE of order-book imbalance carry a directional signal? (bid-heavy & staying so ->
hold long; imbalance flips bid->ask -> flip short). Tests the DYNAMICS of imbalance (not the static level I tested
before), vs RAW forward direction (not beta-neutral rank), at SHORT horizons (next-4h, next-1d) -- all three ways the
pilot was the wrong test for this idea. Both eras, day-clustered CI. PIT: imbalance @T uses book over [T-4h,T); its
change uses only past bars; forward return is [T, T+h].

features (per name, from the cached 4h imb1 series):
  imb_lvl   imb1 now (bid-heavy>0)                     persistence: does current imbalance predict same-dir move?
  imb_d1    imb1 - imb1.shift(1)   (1-bar change)      strengthening/weakening
  imb_mom3  imb1 - imb1.shift(3)   (3-bar momentum)    sustained lean
  imb_absd  |imb1| - |imb1.shift(1)|                   is the imbalance building or decaying
targets: fwd_4h, fwd_1d = RAW forward return (directional, keeps beta); alpha = beta-neutral (control).
IC>0 on imb_lvl/imb_mom => momentum (book lean predicts continuation -> the 'hold' rule works);
IC<0 => contrarian (book lean fades -> you'd fade it). ~0 both eras => no tradeable book-dynamics signal.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
KD = Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines")
CACHE = Path("/home/yuqing/ctaNew/data/ml/cache")
rng = np.random.default_rng(9)

def close_4h(sym, lo, hi):
    fs = [f for f in glob.glob(str(KD / sym / "5m" / "*.parquet"))
          if lo <= Path(f).stem <= hi]
    if not fs: return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(fs)], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    s = df.drop_duplicates("open_time").set_index("open_time")["close"].sort_index()
    grid = pd.date_range(s.index.min().floor("4h"), s.index.max().ceil("4h"), freq="4h", tz="UTC")
    return s.reindex(grid, method="ffill")

def day_boot(ic):
    s = pd.DataFrame({"ic": ic.values}, index=pd.to_datetime(ic.index, utc=True)); s["d"] = s.index.floor("1D")
    g = [x["ic"].values for _, x in s.groupby("d")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(2500)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))

def pooled_ic(df, feat, tgt):
    """per-bar pooled Spearman(feature, target) across names -> series over bars (day-clustered later)."""
    return df.groupby("open_time").apply(
        lambda g: g[feat].corr(g[tgt], method="spearman") if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()

def main():
    # build imbalance dynamics from cache + forward raw returns from klines
    rows = []
    files = glob.glob(str(CACHE / "l2_*.parquet"))
    for i, f in enumerate(files):
        sym = Path(f).stem[3:]
        d = pd.read_parquet(f)[["l2_imb1"]].copy()
        d.index = pd.to_datetime(d.index, utc=True)
        d["open_time"] = d.index + pd.Timedelta("4h")           # PIT: obs bar -> decision bar
        d = d.set_index("open_time").sort_index()
        d["imb_lvl"] = d["l2_imb1"]
        d["imb_d1"] = d["l2_imb1"].diff()
        d["imb_mom3"] = d["l2_imb1"] - d["l2_imb1"].shift(3)
        d["imb_absd"] = d["l2_imb1"].abs() - d["l2_imb1"].abs().shift(1)
        lo, hi = str(d.index.min().date()), str((d.index.max() + pd.Timedelta("2D")).date())
        c = close_4h(sym, lo, hi)
        if c is None: continue
        c = c.reindex(d.index.union(c.index)).sort_index()
        d["fwd_4h"] = (c.shift(-1) / c - 1).reindex(d.index)
        d["fwd_1d"] = (c.shift(-6) / c - 1).reindex(d.index)
        d["symbol"] = sym
        rows.append(d.reset_index())
        if (i + 1) % 40 == 0: print(f"  built {i+1}/{len(files)}", flush=True)
    m = pd.concat(rows, ignore_index=True)
    # attach beta-neutral alpha from the panel for control
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    m = m.merge(pan, on=["symbol", "open_time"], how="left")
    cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"\nmerged {len(m)} rows | {m.symbol.nunique()} syms | RECENT {len(eras['RECENT'])} OOS {len(eras['OOS'])}")
    print("IC>0 = book-lean predicts CONTINUATION (the 'hold when imbalanced' rule works); <0 = fades; ~0 = nothing\n")
    for tgt in ["fwd_4h", "fwd_1d", "alpha_vs_btc_realized"]:
        print(f"### target = {tgt} (raw direction)" + ("  [beta-neutral control]" if "alpha" in tgt else "") + " ###")
        print(f"{'feature':9s} | {'RECENT IC [CI]':26s} | {'OOS IC [CI]':26s} | both-era?")
        for feat in ["imb_lvl", "imb_d1", "imb_mom3", "imb_absd"]:
            cells = {}
            for era, sub in eras.items():
                ic = pooled_ic(sub, feat, tgt); lo, up = day_boot(ic)
                cells[era] = (ic.mean(), lo, up)
            (ra, rl, ru), (oa, ol, ou) = cells["RECENT"], cells["OOS"]
            both = "YES" if (np.sign(ra) == np.sign(oa) and abs(ra) > 0.02 and abs(oa) > 0.02
                             and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
            print(f"{feat:9s} | {ra:+.3f} [{rl:+.3f},{ru:+.3f}] | {oa:+.3f} [{ol:+.3f},{ou:+.3f}] | {both}")
        print()
    print("DYNDONE")

if __name__ == "__main__":
    main()
