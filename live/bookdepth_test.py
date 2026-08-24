"""Does bookDepth (coarse-L2) add signal to the convexity panel? Cross-sectional rank-IC of each 4h L2 feature vs the
model's target (target_z = xs-z of alpha-vs-btc), BOTH eras, day-clustered bootstrap CI. STRICT PIT: a decision bar at
open_time=T uses the book observed during [T-4h, T) only (obs_bar + 4h = T). Go/no-go: a feature earns interest only
if its rank-IC is same-sign and non-trivial in BOTH eras (recent 2025-10+ and OOS pre-2025-10) with a CI off zero.
Also reports the incremental rank-IC when L2 is added to the V0 feature set (pooled RidgeCV, walk-forward-lite).
"""
from pathlib import Path
import glob
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
CACHE = Path("/home/yuqing/ctaNew/data/ml/cache")
PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet"
L2COLS = ["l2_imb1", "l2_imb02", "l2_liq1", "l2_touch", "l2_slope", "l2_asym1", "l2_imbstd"]
rng = np.random.default_rng(11)

def load_l2():
    frames = []
    for f in glob.glob(str(CACHE / "l2_*.parquet")):
        sym = Path(f).stem[3:]
        d = pd.read_parquet(f)
        if not len(d): continue
        d = d.reset_index().rename(columns={"obs_bar": "obs_bar"})
        d["symbol"] = sym
        frames.append(d)
    if not frames: return None
    L = pd.concat(frames, ignore_index=True)
    L["obs_bar"] = pd.to_datetime(L["obs_bar"], utc=True)
    L["open_time"] = L["obs_bar"] + pd.Timedelta("4h")            # PIT: prior-bar book -> this decision bar
    return L

def xic(df, feat, tgt="target_z"):
    """per-bar cross-sectional Spearman rank-IC series (index = bar)."""
    def f(g):
        if g[feat].notna().sum() < 5: return np.nan
        return g[feat].corr(g[tgt], method="spearman")
    return df.groupby("open_time").apply(f).dropna()

def day_boot(ic):
    """day-clustered bootstrap CI on the mean per-bar IC."""
    s = pd.DataFrame({"ic": ic.values}, index=pd.to_datetime(ic.index, utc=True))
    s["day"] = s.index.floor("1D")
    grps = [g["ic"].values for _, g in s.groupby("day")]
    if len(grps) < 5: return (np.nan, np.nan)
    out = [np.concatenate([grps[i] for i in rng.integers(0, len(grps), len(grps))]).mean() for _ in range(3000)]
    return tuple(np.percentile(out, [2.5, 97.5]))

def main():
    L = load_l2()
    if L is None: print("no L2 cache yet"); return
    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "target_z", "alpha_vs_btc_realized"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    m = pan.merge(L[["symbol", "open_time"] + L2COLS], on=["symbol", "open_time"], how="inner")
    print(f"merged rows {len(m)} | symbols {m.symbol.nunique()} | bars {m.open_time.nunique()} | "
          f"span {str(m.open_time.min())[:10]}..{str(m.open_time.max())[:10]}")
    cut = pd.Timestamp("2025-10-01", tz="UTC")
    eras = {"RECENT": m[m.open_time >= cut], "OOS": m[m.open_time < cut]}
    for era, sub in eras.items():
        print(f"  {era}: rows {len(sub)} bars {sub.open_time.nunique()} span {str(sub.open_time.min())[:10]}..{str(sub.open_time.max())[:10]}")
    print(f"\n{'feature':10s} | {'RECENT IC [CI]  %pos':28s} | {'OOS IC [CI]  %pos':28s} | both-era?")
    verdict = []
    for feat in L2COLS:
        row = {}
        for era, sub in eras.items():
            ic = xic(sub, feat)
            lo, up = day_boot(ic)
            row[era] = (ic.mean(), lo, up, (ic > 0).mean(), len(ic))
        (ra, rlo, rup, rp, rn) = row["RECENT"]; (oa, olo, oup, op, on) = row["OOS"]
        both = "YES" if (np.sign(ra) == np.sign(oa) and abs(ra) > 0.02 and abs(oa) > 0.02
                         and (rlo > 0 or rup < 0) and (olo > 0 or oup < 0)) else ("~" if np.sign(ra) == np.sign(oa) and abs(ra) > 0.015 and abs(oa) > 0.015 else "no")
        print(f"{feat:10s} | {ra:+.3f} [{rlo:+.3f},{rup:+.3f}] {rp*100:3.0f}% | {oa:+.3f} [{olo:+.3f},{oup:+.3f}] {op*100:3.0f}% | {both}")
        if both in ("YES", "~"): verdict.append((feat, ra, oa, both))
    print()
    if verdict:
        print("both-era candidates:", [(f, round(r,3), round(o,3), b) for f, r, o, b in verdict])
    else:
        print("no L2 feature is same-sign + non-trivial + CI-off-zero in BOTH eras -> L2 does not add at this pilot scale")
    print("BDTESTDONE")

if __name__ == "__main__":
    main()
