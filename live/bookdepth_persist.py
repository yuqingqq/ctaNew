"""User's sharpened idea: not the imbalance CHANGE (transient, decays in minutes) but SUSTAINED imbalance — if a
book HOLDS bid-heavy for a long time (high z-score, long run), does that slow accumulated pressure predict a 4h+ move?
Persistence features (PIT, from the cached 4h imb1 series; trailing windows known at T):
  imb_z30    (imb1 - roll30 mean)/roll30 std   how extreme vs its own recent history (the z-score you asked for)
  imb_ewma   EWMA(imb1, halflife=12bars=2d)    smoothed persistent lean
  imb_mean12 trailing 12-bar (2d) mean          sustained level
  imb_run    signed consecutive same-sign bars  HOW LONG it has held (duration), capped +/-20
  imb_frac12 frac of last 12 bars bid-heavy-0.5 directional persistence
targets: fwd_4h / fwd_1d / fwd_2d (raw direction) + alpha_vs_btc (beta-neutral). Cross-sectional rank-IC, both eras,
day-clustered CI. If a SUSTAINED feature is same-sign + CI-off-zero in BOTH eras where the transient ones were null,
persistent imbalance is a real 4h signal.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
KD = Path("/home/yuqing/ctaNew/data/ml/test/parquet/klines")
CACHE = Path("/home/yuqing/ctaNew/data/ml/cache")
rng = np.random.default_rng(17)

def close_4h(sym, lo, hi):
    fs = [f for f in glob.glob(str(KD / sym / "5m" / "*.parquet")) if lo <= Path(f).stem <= hi]
    if not fs: return None
    df = pd.concat([pd.read_parquet(f, columns=["open_time", "close"]) for f in sorted(fs)], ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    s = df.drop_duplicates("open_time").set_index("open_time")["close"].sort_index()
    grid = pd.date_range(s.index.min().floor("4h"), s.index.max().ceil("4h"), freq="4h", tz="UTC")
    return s.reindex(grid, method="ffill")

def _persist_seg(s):
    d = pd.DataFrame(index=s.index)
    mu = s.rolling(30, min_periods=10).mean(); sd = s.rolling(30, min_periods=10).std()
    d["imb_z30"] = (s - mu) / sd.replace(0, np.nan)
    d["imb_ewma"] = s.ewm(halflife=12).mean()
    d["imb_mean12"] = s.rolling(12, min_periods=6).mean()
    sign = np.sign(s.fillna(0))
    grp = (sign != sign.shift()).cumsum()
    run = sign.groupby(grp).cumcount() + 1
    d["imb_run"] = (sign * run).clip(-20, 20)
    d["imb_frac12"] = (s > 0).rolling(12, min_periods=6).mean() - 0.5
    return d

def persist_feats(imb):
    """gap-aware: reset rolling/run at breaks >8h so trailing windows never span a data gap (e.g. the 2024->2026 jump)."""
    imb = imb.sort_index()
    seg = (imb.index.to_series().diff() > pd.Timedelta("8h")).cumsum().values
    return pd.concat([_persist_seg(imb[seg == g]) for g in pd.unique(seg)])

def day_boot(ic):
    s = pd.DataFrame({"ic": ic.values}, index=pd.to_datetime(ic.index, utc=True)); s["d"] = s.index.floor("1D")
    g = [x["ic"].values for _, x in s.groupby("d")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(2500)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))

def pooled_ic(df, feat, tgt):
    return df.groupby("open_time").apply(
        lambda g: g[feat].corr(g[tgt], method="spearman") if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()

FEATS = ["imb_z30", "imb_ewma", "imb_mean12", "imb_run", "imb_frac12"]

def main():
    rows = []
    files = [f for f in glob.glob(str(CACHE / "l2_*.parquet")) if "BTCUSDT" not in f]
    for i, f in enumerate(files):
        sym = Path(f).stem[3:]
        d0 = pd.read_parquet(f)[["l2_imb1"]]
        d0.index = pd.to_datetime(d0.index, utc=True) + pd.Timedelta("4h")   # PIT decision bar
        imb = d0["l2_imb1"].sort_index()
        d = persist_feats(imb)
        lo, hi = str(d.index.min().date()), str((d.index.max() + pd.Timedelta("3D")).date())
        c = close_4h(sym, lo, hi)
        if c is None: continue
        c = c.reindex(d.index.union(c.index)).sort_index()
        d["fwd_4h"] = (c.shift(-1) / c - 1).reindex(d.index)
        d["fwd_1d"] = (c.shift(-6) / c - 1).reindex(d.index)
        d["fwd_2d"] = (c.shift(-12) / c - 1).reindex(d.index)
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
    print("SUSTAINED imbalance -> forward return; IC>0 = persistent-bid-lean predicts UP (the '4h signal' hypothesis)\n")
    for tgt in ["fwd_4h", "fwd_1d", "fwd_2d", "alpha_vs_btc_realized"]:
        print(f"### target = {tgt} ###")
        print(f"{'feature':11s} | {'RECENT IC [CI]':26s} | {'OOS IC [CI]':26s} | both-era?")
        for feat in FEATS:
            cells = {}
            for era, sub in eras.items():
                ic = pooled_ic(sub, feat, tgt); lo, up = day_boot(ic); cells[era] = (ic.mean(), lo, up)
            (ra, rl, ru), (oa, ol, ou) = cells["RECENT"], cells["OOS"]
            both = "YES" if (np.sign(ra) == np.sign(oa) and abs(ra) > 0.02 and abs(oa) > 0.02
                             and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
            print(f"{feat:11s} | {ra:+.3f} [{rl:+.3f},{ru:+.3f}] | {oa:+.3f} [{ol:+.3f},{ou:+.3f}] | {both}")
        print()
    print("PERSISTDONE")

if __name__ == "__main__":
    main()
