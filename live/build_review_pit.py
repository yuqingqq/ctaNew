"""Review the 'validated OOS top-40 alpha' for the universe look-ahead. The prior tests picked top-N by FULL-SAMPLE
ADV (forward-looking: names that BECAME big). Honest PIT test: rank names by TRAILING 30d ADV at each bar, trade the
PIT top-N. If the OOS beta-neutral alpha survives (still significant), the claim is robust; if it drops, the earlier
'validated' number was inflated by look-ahead. In-era beta-neutralization (attribution, no hedge look-ahead),
day-clustered CI, both eras. Run: python3 -u -m live.build_review_pit
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

PYR = 6 * 365.0
TIERS = [20, 40]
RNG = np.random.default_rng(7)


def trailing_adv():
    frames = []
    for f in glob.glob("data/ml/cache/flow_*.parquet"):
        sym = f.split("/")[-1].replace("flow_", "").replace(".parquet", "")
        try:
            d = pd.read_parquet(f, columns=["total_volume", "vwap"])
            if not isinstance(d.index, pd.DatetimeIndex):
                continue
            dv = (d["total_volume"] * d["vwap"]).sort_index()
            daily = dv.resample("1D").sum()
            tadv = daily.rolling(30, min_periods=10).mean().shift(1)      # PIT trailing-30d ADV
            frames.append(pd.DataFrame({"symbol": sym, "date": tadv.index, "tadv": tadv.values}))
        except Exception:
            pass
    A = pd.concat(frames, ignore_index=True)
    A["date"] = pd.to_datetime(A["date"], utc=True)
    return A.dropna(subset=["tadv"])


def ls_mkt(d):
    bt = d["open_time"].to_numpy("datetime64[ns]"); rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(bt, sort=True); k = len(uniq)
    r = pd.Series(d["pred"].to_numpy()).groupby(codes).rank(pct=True).to_numpy()
    lo = r >= 0.8; sh = r <= 0.2
    nl = np.bincount(codes, lo.astype(float), k); ns = np.bincount(codes, sh.astype(float), k)
    sl = np.bincount(codes, np.where(lo, rp, 0.0), k); ss = np.bincount(codes, np.where(sh, rp, 0.0), k)
    na = np.bincount(codes, minlength=k); sa = np.bincount(codes, rp, k)
    ok = (nl >= 2) & (ns >= 2)
    return uniq[ok], sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1), sa[ok] / np.maximum(na[ok], 1)


def day_ci(vals, times, stat="mean", nb=3000):
    d = pd.DatetimeIndex(times).floor("1D")
    g = [x for _, x in pd.Series(np.arange(len(times))).groupby(d.values)]
    out = np.empty(nb)
    for i in range(nb):
        idx = np.concatenate([g[k].to_numpy() for k in RNG.integers(0, len(g), len(g))])
        s = vals[idx]
        out[i] = s.mean() * 1e4 if stat == "mean" else s.mean() / s.std() * np.sqrt(PYR)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    tadv = trailing_adv()
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    print("PIT universe (trailing-30d ADV, ranked per bar) vs the full-sample-ADV universe used before.\n"
          "beta-neutral alpha (IN-ERA beta), day-clustered 95% CI:\n", flush=True)
    print(f"  {'N':<5}{'era':<8}{'alpha bps [CI]':<26}{'Sharpe [CI]':<24}{'avg #names'}", flush=True)
    for era, cuts in (("OOS", OOS_CUTS), ("RECENT", RECENT_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts); pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna()
        d["date"] = d["open_time"].dt.floor("1D")
        d = d.merge(tadv, on=["symbol", "date"], how="left").dropna(subset=["tadv"])
        d["advrank"] = d.groupby("open_time")["tadv"].rank(ascending=False, method="first")
        for N in TIERS:
            sub = d[d["advrank"] <= N]
            nnames = sub.groupby("open_time")["symbol"].size().mean()
            t, ls, mkt = ls_mkt(sub)
            beta = np.polyfit(mkt, ls, 1)[0]
            al = ls - beta * mkt
            mlo, mhi = day_ci(al, t, "mean"); slo, shi = day_ci(al, t, "sharpe")
            print(f"  {N:<5}{era:<8}{f'{al.mean()*1e4:+.2f} [{mlo:+.2f},{mhi:+.2f}]':<26}"
                  f"{f'{al.mean()/al.std()*np.sqrt(PYR):+.2f} [{slo:+.2f},{shi:+.2f}]':<24}{nnames:.0f}", flush=True)
        print("", flush=True)
    print("REVIEWPITDONE", flush=True)


if __name__ == "__main__":
    main()
