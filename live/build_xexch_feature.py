"""#185 — build cross-exchange premium features and test univariate cross-sectional IC
against the model target (xs_z = XS z-score of forward return_pct).

Premium = log(other_venue_close) - log(binance_close), then cross-sectionally demeaned per
4h bar to strip the common USDT-USD / spot-perp basis -> isolates per-symbol dislocation.
Features are PIT (.shift(1)). IC = mean over time of per-bar Spearman(feature_rank, xs_z_rank).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/yuqing/ctaNew")
import live.convexity_paper_bot as bot
from scipy.stats import spearmanr

REPO = Path("/home/yuqing/ctaNew")
XE = REPO/"data/ml/cache/xexch"
PANEL = REPO/"outputs/vBTC_features/panel_expanded_v0.parquet"
OOS = pd.Timestamp("2025-10-04", tz="UTC")


def load_venue(venue, sym):
    f = XE/venue/f"{sym}.parquet"
    if not f.exists(): return None
    d = pd.read_parquet(f); d["open_time"] = pd.to_datetime(d["open_time"], utc=True)
    return d.set_index("open_time")["close"].astype(float)


def main():
    u = json.load(open(REPO/"live/models/convexity_v1_universe.json"))
    syms = u["tradeable_low_vol"]
    # target: xs_z per (open_time, symbol)
    pan = pd.read_parquet(PANEL, columns=["symbol","open_time","return_pct"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan = pan[(pan.open_time.dt.hour%4==0)&(pan.open_time.dt.minute==0)]
    g = pan.groupby("open_time"); mu = g["return_pct"].transform("mean"); sd = g["return_pct"].transform("std").replace(0,np.nan)
    pan["xs_z"] = ((pan["return_pct"]-mu)/sd).clip(-10,10)
    tgt = pan[["symbol","open_time","xs_z"]]

    rows = []
    for venue in ("coinbase","okx"):
        prem = {}
        for s in syms:
            cb = load_venue(venue, s)
            if cb is None: continue
            bn = bot.load_close_4h(s)
            if bn is None or not len(bn): continue
            j = pd.concat([np.log(cb).rename("o"), np.log(bn).rename("b")], axis=1).dropna()
            if len(j) < 100: continue
            prem[s] = (j["o"] - j["b"]).rename(s)        # log premium level
        if not prem:
            print(f"{venue}: no symbols"); continue
        P = pd.concat(prem.values(), axis=1)              # index=open_time, cols=symbols
        Pxs = P.sub(P.median(axis=1), axis=0)             # XS-demean per bar -> strip common basis
        feats = {
            f"{venue}_level": Pxs.shift(1),
            f"{venue}_chg1":  (Pxs - Pxs.shift(1)).shift(1),
            f"{venue}_chg3":  (Pxs - Pxs.shift(3)).shift(1),
        }
        for fname, Fdf in feats.items():
            long = Fdf.reset_index().melt(id_vars="open_time", var_name="symbol", value_name="feat").dropna()
            m = long.merge(tgt, on=["symbol","open_time"], how="inner").dropna()
            # full-sample + OOS-only XS IC
            for lbl, sub in [("all", m), ("oos", m[m.open_time>=OOS])]:
                ics = sub.groupby("open_time").apply(
                    lambda d: spearmanr(d["feat"], d["xs_z"]).correlation if len(d) >= 5 else np.nan).dropna()
                rows.append({"feature": fname, "window": lbl, "IC_mean": ics.mean(),
                             "IC_t": ics.mean()/ics.std()*np.sqrt(len(ics)) if ics.std()>0 else np.nan,
                             "n_bars": len(ics), "n_syms": sub.symbol.nunique()})
    R = pd.DataFrame(rows)
    print("\n=== cross-exchange premium — univariate XS IC vs xs_z target ===")
    print(R.round(4).to_string(index=False))
    R.to_csv(REPO/"outputs/convexity_v1_xexch_feature.csv", index=False)
    print(f"\nsaved -> outputs/convexity_v1_xexch_feature.csv")
    print("sanity bar: |IC|>0.02 with |t|>2 and stable sign across all/oos -> worth a full model test")


if __name__ == "__main__":
    main()
