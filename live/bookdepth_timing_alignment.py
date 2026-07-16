"""User: cross-DEPTH-LEVEL structure of the AGGREGATE book — do all levels ALIGN (uniform one-sided = clean crowding)
or MISALIGN (near vs deep disagree = e.g. distribution: bid-heavy touch, ask-heavy deep)? Test on the aggregate:
 (A) MISALIGNMENT as signal: near-minus-deep z (z(imb1)-z(imb5)) -> forward 2d market return, both eras.
 (B) ALIGNMENT as confidence: split the fade by cross-level dispersion (std of the 4 level z's); is the crowding-fade
     STRONGER when levels agree (low dispersion = aligned) than when they disagree?
 (C) CONSENSUS vs SINGLE: does the all-level mean-z fade beat single-level imb1?
CAVEAT: single market series, few independent episodes -> low power, wide CIs; suggestive only.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_crowding_horizon import market_series, block_ic
CUT = pd.Timestamp("2025-10-01", tz="UTC")
LV = ["imb1", "imb2", "imb3", "imb5"]

def agg_levels():
    acc = {lv: [] for lv in LV}
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue
        d.index = pd.to_datetime(d.index, utc=True) + pd.Timedelta("4h"); d = d[~d.index.duplicated()]
        for lv in LV:
            if "l2_" + lv in d.columns: acc[lv].append(d["l2_" + lv].dropna())
    return pd.DataFrame({"agg_" + lv: pd.concat(acc[lv]).groupby(level=0).mean() for lv in LV}).sort_index()

def zc(s, W=90):
    return (s - s.rolling(W, min_periods=W // 2).mean()) / s.rolling(W, min_periods=W // 2).std()

def main():
    A = agg_levels()
    Z = pd.DataFrame({lv: zc(A["agg_" + lv]) for lv in LV})
    F = pd.DataFrame({"consensus": Z.mean(axis=1), "disp": Z.std(axis=1), "near_deep": Z["imb1"] - Z["imb5"], "z1": Z["imb1"]})
    F = F.join(market_series()[["mkt_2d"]]).dropna()
    eras = {"OOS": F[F.index < CUT], "RECENT": F[F.index >= CUT]}

    print("=== (A) MISALIGNMENT: IC( near_deep = z(imb1)-z(imb5) -> fwd 2d market ), both eras ===")
    for e, s in eras.items():
        a, lo, up = block_ic(s["near_deep"], s["mkt_2d"], 48)
        print(f"  {e:7s}: IC {a:+.4f} [{lo:+.4f},{up:+.4f}]  (near-deep disagreement predicts market?)")

    print("\n=== (C) CONSENSUS (all-level mean z) vs SINGLE (imb1 z): contrarian IC -> fwd 2d ===")
    for e, s in eras.items():
        ca, cl, cu = block_ic(s["consensus"], s["mkt_2d"], 48); za, zl, zu = block_ic(s["z1"], s["mkt_2d"], 48)
        print(f"  {e:7s}: consensus {ca:+.4f} [{cl:+.4f},{cu:+.4f}] | imb1 {za:+.4f} [{zl:+.4f},{zu:+.4f}]")

    print("\n=== (B) ALIGNMENT as confidence: fade edge (-sign(consensus)*fwd2d) by cross-level dispersion tercile ===")
    print(f"{'dispersion':16s} | {'OOS fade':10s} | {'RECENT fade':11s} | median disp  (LOW disp = levels ALIGNED)")
    for e in ["OOS", "RECENT"]:
        eras[e]["dt"] = pd.qcut(eras[e]["disp"], 3, labels=False, duplicates="drop")
        eras[e]["fade"] = -np.sign(eras[e]["consensus"]) * eras[e]["mkt_2d"]
    for b in range(3):
        o = eras["OOS"].loc[eras["OOS"]["dt"] == b]; r = eras["RECENT"].loc[eras["RECENT"]["dt"] == b]
        tag = {0: "T1 ALIGNED", 2: "T3 MISALIGNED"}.get(b, "T2 mid")
        print(f"{tag:16s} | {o['fade'].mean()*100:+9.2f}% | {r['fade'].mean()*100:+10.2f}% | {o['disp'].median():.2f}")
    print("\nread: (A) does misalignment predict? (B) is the fade cleaner when ALIGNED (T1) vs MISALIGNED (T3)?")
    print("(C) does using all levels beat imb1 alone? All on a single series -> wide CIs, suggestive. ALIGNDONE")

if __name__ == "__main__":
    main()
