"""Two user hypotheses on the aggregate crowding-fade market-timing signal:
 (1) DEPTH LEVELS: is the contrarian edge the same at every book depth? near-touch (+/-0.2%, "top of book" = maybe
     informed/institutional/trending) vs deep (+/-5% = maybe crowding)? Aggregate imbalance at each level, contrarian
     IC to forward 2d market return, both eras. (+/-0.2% is recent-only -> recent only.)
 (2) PRICE x OB: does conditioning the fade on recent PRICE action sharpen it? 3x3 grid of trailing-1d market return
     tercile x aggregate-imbalance z tercile -> forward 2d market return. E.g. does "price dumped + bid-heavy book"
     bounce (fade works) or keep falling (falling knife)?
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from live.bookdepth_crowding_horizon import market_series, block_ic
from live.bookdepth_timing_backtest import market_ret
CUT = pd.Timestamp("2025-10-01", tz="UTC")
LV = ["imb02", "imb1", "imb2", "imb3", "imb5"]

def agg_levels():
    acc = {lv: [] for lv in LV}
    for f in [x for x in glob.glob("/home/yuqing/ctaNew/data/ml/cache/l2_*.parquet") if "BTCUSDT" not in x]:
        try:
            d = pd.read_parquet(f)
        except Exception:
            continue
        d.index = pd.to_datetime(d.index, utc=True) + pd.Timedelta("4h"); d = d[~d.index.duplicated()]
        for lv in LV:
            c = "l2_" + lv
            if c in d.columns: acc[lv].append(d[c].dropna())
    out = {}
    for lv in LV:
        if acc[lv]:
            s = pd.concat(acc[lv]); out["agg_" + lv] = s.groupby(s.index).mean()
    return pd.DataFrame(out).sort_index()

def zc(s, W=90):
    return (s - s.rolling(W, min_periods=W // 2).mean()) / s.rolling(W, min_periods=W // 2).std()

def main():
    A = agg_levels()
    M = market_series()[["mkt_1d", "mkt_2d"]]
    mbar = market_ret()
    lr = np.log1p(mbar); trail1d = lr.rolling(6, min_periods=6).sum().shift(1).rename("trail1d")  # past 1d market return
    D = A.join(M).join(trail1d).dropna(subset=["agg_imb1", "mkt_2d"])
    eras = {"OOS": D[D.index < CUT], "RECENT": D[D.index >= CUT]}

    print("=== (1) DEPTH LEVELS: contrarian IC( agg_imb_LEVEL -> fwd 2d market ), both eras (more NEG = stronger fade) ===")
    print(f"{'level':8s} | {'OOS IC [CI]':26s} | {'RECENT IC [CI]':26s} | coverage")
    for lv in LV:
        col = "agg_" + lv
        if col not in D: continue
        oa, ol, ou = block_ic(eras["OOS"][col], eras["OOS"]["mkt_2d"], 48) if eras["OOS"][col].notna().sum() > 200 else (np.nan,)*3
        ra, rl, ru = block_ic(eras["RECENT"][col], eras["RECENT"]["mkt_2d"], 48) if eras["RECENT"][col].notna().sum() > 100 else (np.nan,)*3
        cov = "recent-only" if lv == "imb02" else "both-era"
        print(f"{lv:8s} | {oa:+.4f} [{ol:+.4f},{ou:+.4f}] | {ra:+.4f} [{rl:+.4f},{ru:+.4f}] | {cov}")

    print("\n=== (2) PRICE x OB: mean fwd 2d market return by [trailing-1d market ret] x [agg_imb1 z] tercile ===")
    for e in ["OOS", "RECENT"]:
        sub = eras[e].dropna(subset=["trail1d", "agg_imb1", "mkt_2d"]).copy()
        sub["pt"] = pd.qcut(sub["trail1d"], 3, labels=["price DOWN", "price flat", "price UP"])
        sub["it"] = pd.qcut(zc(sub["agg_imb1"]).reindex(sub.index), 3, labels=["ask-heavy", "mid", "bid-heavy"])
        print(f"  [{e}]  fwd2d market return (%), rows: contrarian short-bid-heavy => bid-heavy row should be most NEGATIVE")
        tab = sub.groupby(["it", "pt"])["mkt_2d"].mean().unstack() * 100
        print(tab.round(2).to_string())
        print()
    print("read: (1) which level fades strongest? near(imb02)-vs-deep(imb5). (2) does price DOWN + bid-heavy behave")
    print("differently than price UP + bid-heavy (a real interaction to exploit)? DEPTHPRICEDONE")

if __name__ == "__main__":
    main()
