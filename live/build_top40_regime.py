"""Does the top-40 low-vol alpha depend on the market REGIME (the 'quiet market' hypothesis)? Mechanism = fade the
over-demanded volatile majors; it should be strong when there's speculative excess to fade and weak/negative when
the volatile names deliver. Condition the top-40 beta-neutral alpha on: (a) market vol (btc_rvol_7d terciles),
(b) cross-sectional vol dispersion (std of return_1d across top-40). Pooled + by era. Run: python3 -u -m live.build_top40_regime
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

NTOP = 40


def adv_rank():
    out = {}
    for f in glob.glob("data/ml/cache/flow_*.parquet"):
        sym = f.split("/")[-1].replace("flow_", "").replace(".parquet", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            out[sym] = float((d["total_volume"] * d["vwap"]).mean())
        except Exception:
            pass
    return pd.Series(out).sort_values(ascending=False).index.tolist()


def perbar(d):
    bt = d["open_time"].to_numpy("datetime64[ns]"); rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(bt, sort=True); k = len(uniq)
    r = pd.Series(d["pred"].to_numpy()).groupby(codes).rank(pct=True).to_numpy()
    lo = r >= 0.8; sh = r <= 0.2
    nl = np.bincount(codes, lo.astype(float), k); ns = np.bincount(codes, sh.astype(float), k)
    sl = np.bincount(codes, np.where(lo, rp, 0.0), k); ss = np.bincount(codes, np.where(sh, rp, 0.0), k)
    na = np.bincount(codes, minlength=k); sa = np.bincount(codes, rp, k)
    disp = pd.Series(d["return_1d"].to_numpy()).groupby(codes).std().to_numpy()
    ok = (nl >= 2) & (ns >= 2)
    ls = sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1)
    mkt = sa[ok] / np.maximum(na[ok], 1)
    return pd.DataFrame({"t": uniq[ok], "ls": ls, "mkt": mkt, "disp": disp[ok]})


def main():
    top = set(adv_rank()[:NTOP])
    PAN = build_panel()
    if "return_1d" not in PAN.columns:
        ex = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_1d"])
        ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True)
        PAN = PAN.merge(ex, on=["symbol", "open_time"], how="left")
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    btcvol = PAN.groupby("open_time")["btc_rvol_7d"].first()
    frames = []
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").merge(
            PAN[["symbol", "open_time", "return_1d"]], on=["symbol", "open_time"], how="left")
        d = d[d["symbol"].isin(top)].dropna(subset=["pred", "return_pct"])
        pb = perbar(d); pb["era"] = era
        frames.append(pb)
    A = pd.concat(frames, ignore_index=True)
    A["t"] = pd.to_datetime(A["t"], utc=True)
    A["btcvol"] = A["t"].map(btcvol)
    beta = np.polyfit(A["mkt"], A["ls"], 1)[0]
    A["alpha"] = A["ls"] - beta * A["mkt"]              # beta-neutral alpha per bar
    print(f"top-{NTOP} beta-neutral alpha (net beta hedged, β={beta:+.3f}); alpha in bps/bar by regime tercile\n",
          flush=True)
    for name, col in (("MARKET VOL (btc_rvol_7d)", "btcvol"), ("XS DISPERSION (std return_1d)", "disp")):
        q = A[col].quantile([1/3, 2/3]).to_numpy()
        A["b"] = np.digitize(A[col].to_numpy(), q)
        print(f"  {name}:", flush=True)
        for b, lab in [(0, "LOW/quiet"), (1, "MID"), (2, "HIGH/frothy")]:
            s = A[A.b == b]
            print(f"    {lab:<12} alpha {s['alpha'].mean()*1e4:+.2f}bps  (n={len(s)})  "
                  f"[OOS {s[s.era=='OOS']['alpha'].mean()*1e4:+.2f} | REC {s[s.era=='RECENT']['alpha'].mean()*1e4:+.2f}]",
                  flush=True)
        print("", flush=True)
    print("TOP40REGIMEDONE", flush=True)


if __name__ == "__main__":
    main()
