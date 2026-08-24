"""Is the faint 1d-2d persistence signal just MOMENTUM? A persistently bid-heavy book ~ a name in a sustained uptrend,
and momentum predicts raw drift. Partial cross-sectional rank-IC of imb_run/imb_frac12 vs fwd_1d/fwd_2d CONTROLLING
for the panel's momentum features (return_1d, ret_3d). If partial ~0, persistence is redundant with momentum (already
in the model, and raw-directional not beta-neutral). Also shows raw vs partial side by side, both eras.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_persist import persist_feats, close_4h
rng = np.random.default_rng(23)
CTRL = ["return_1d", "ret_3d"]

def xrank(df, cols): return df.groupby("open_time")[cols].rank(pct=True)
def resid(y, X):
    A = np.column_stack([np.ones(len(X)), X]); b, *_ = np.linalg.lstsq(A, y, rcond=None); return y - A @ b
def day_boot(rx, ry, days):
    d = pd.DataFrame({"rx": rx, "ry": ry, "day": days}); g = [x for _, x in d.groupby("day")]
    if len(g) < 5: return (np.nan, np.nan)
    o = [pd.concat([g[i] for i in rng.integers(0, len(g), len(g))]).pipe(lambda s: s["rx"].corr(s["ry"])) for _ in range(1500)]
    return tuple(np.nanpercentile(o, [2.5, 97.5]))

def partial(sub, feat, tgt):
    sub = sub.dropna(subset=[feat, tgt] + CTRL).copy()
    if len(sub) < 300: return (np.nan, np.nan, np.nan, np.nan)
    R = xrank(sub, CTRL + [feat, tgt]).fillna(0.5)
    ry = resid(R[tgt].values, R[CTRL].values); rx = resid(R[feat].values, R[CTRL].values)
    raw = np.corrcoef(R[feat].values, R[tgt].values)[0, 1]; part = np.corrcoef(rx, ry)[0, 1]
    lo, up = day_boot(rx, ry, sub["open_time"].dt.floor("1D").values)
    return raw, part, lo, up

def main():
    rows = []
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        sym = Path(f).stem[3:]
        d0 = pd.read_parquet(f)[["l2_imb1"]]; d0.index = pd.to_datetime(d0.index, utc=True) + pd.Timedelta("4h")
        pf = persist_feats(d0["l2_imb1"].sort_index())
        lo, hi = str(pf.index.min().date()), str((pf.index.max() + pd.Timedelta("3D")).date())
        c = close_4h(sym, lo, hi)
        if c is None: continue
        c = c.reindex(pf.index.union(c.index)).sort_index()
        pf["fwd_1d"] = (c.shift(-6) / c - 1).reindex(pf.index); pf["fwd_2d"] = (c.shift(-12) / c - 1).reindex(pf.index)
        pf["symbol"] = sym; pf["open_time"] = pf.index; rows.append(pf.reset_index(drop=True))
    m = pd.concat(rows, ignore_index=True)
    pan = pd.read_parquet("/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet",
                          columns=["symbol", "open_time"] + CTRL)
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    m = m.merge(pan, on=["symbol", "open_time"], how="inner")
    cut = pd.Timestamp("2025-10-01", tz="UTC"); eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    print(f"controlling for momentum {CTRL}; raw->partial rank-IC. If partial~0 => persistence is redundant with momentum\n")
    for tgt in ["fwd_1d", "fwd_2d"]:
        print(f"### target {tgt} ###   {'RECENT raw->partial [CI]':32s} | OOS raw->partial [CI]")
        for feat in ["imb_run", "imb_frac12"]:
            (rr, rp, rl, ru) = partial(eras["RECENT"], feat, tgt); (orr, op, ol, ou) = partial(eras["OOS"], feat, tgt)
            print(f"  {feat:11s} {rr:+.3f}->{rp:+.3f} [{rl:+.3f},{ru:+.3f}] | {orr:+.3f}->{op:+.3f} [{ol:+.3f},{ou:+.3f}]")
        print()
    print("MOMDONE")

if __name__ == "__main__":
    main()
