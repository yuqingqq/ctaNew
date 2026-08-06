"""Careful validation of the load-bearing conclusions, with day-clustered bootstrap CIs + robustness across
universe size. Answers, honestly:
 (1) DEPLOYABILITY: is the top-N beta-neutral alpha > 0 (OOS) and what in RECENT — CI, robust across N=20/30/40/50?
 (2) BETA-NEUTRAL: net beta small (CI)? does return survive hedging?
 (3) MECHANISM: is the structural low-vol (BETWEEN-rvol) IC the robust carrier, both eras (day-clustered CI)?
gen_pred once per era (full), filter to each tier. Era-locked hedge. Run: python3 -u -m live.build_validate_conclusion
"""
from __future__ import annotations

import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL

PYR = 6 * 365.0
TIERS = [20, 30, 40, 50]
RNG = np.random.default_rng(42)


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


def ls_mkt(d):
    bt = d["open_time"].to_numpy("datetime64[ns]"); rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(bt, sort=True); k = len(uniq)
    r = pd.Series(d["pred"].to_numpy()).groupby(codes).rank(pct=True).to_numpy()
    lo = r >= 0.8; sh = r <= 0.2
    nl = np.bincount(codes, lo.astype(float), k); ns = np.bincount(codes, sh.astype(float), k)
    sl = np.bincount(codes, np.where(lo, rp, 0.0), k); ss = np.bincount(codes, np.where(sh, rp, 0.0), k)
    na = np.bincount(codes, minlength=k); sa = np.bincount(codes, rp, k)
    ok = (nl >= 2) & (ns >= 2)
    return pd.DataFrame({"t": uniq[ok], "ls": sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1),
                         "mkt": sa[ok] / np.maximum(na[ok], 1)})


def day_groups(times):
    d = pd.DatetimeIndex(times).floor("1D")
    order = np.argsort(d.values)
    return [g for _, g in pd.Series(np.arange(len(times))[order]).groupby(d.values[order])]


def day_ci(vals, times, stat="mean", nb=3000):
    g = day_groups(times); out = np.empty(nb)
    for i in range(nb):
        idx = np.concatenate([g[k] for k in RNG.integers(0, len(g), len(g))])
        s = vals[idx]
        out[i] = s.mean() * 1e4 if stat == "mean" else s.mean() / s.std() * np.sqrt(PYR)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    ranked = adv_rank()
    PAN = build_panel()
    if "rvol_7d" not in PAN.columns:
        ex = pd.read_parquet(FULL, columns=["symbol", "open_time", "rvol_7d"])
        ex["open_time"] = pd.to_datetime(ex["open_time"], utc=True); PAN = PAN.merge(ex, on=["symbol", "open_time"], how="left")
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    pe = {}
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts); pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        pe[era] = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna()
    other = {"RECENT": "OOS", "OOS": "RECENT"}
    print("(1) DEPLOYABILITY — top-N beta-neutral alpha (era-locked hedge); day-clustered 95% CI\n", flush=True)
    print(f"  {'N':<5}{'era':<8}{'net beta':<10}{'alpha bps [CI]':<26}{'alpha Sharpe [CI]':<24}", flush=True)
    for N in TIERS:
        uni = set(ranked[:N])
        bk = {e: ls_mkt(pe[e][pe[e]["symbol"].isin(uni)]) for e in pe}
        beta = {e: np.polyfit(bk[e]["mkt"], bk[e]["ls"], 1)[0] for e in bk}
        for era in ("OOS", "RECENT"):
            b = bk[era]; al = (b["ls"] - beta[other[era]] * b["mkt"]).to_numpy(); t = b["t"].to_numpy()
            mlo, mhi = day_ci(al, t, "mean"); slo, shi = day_ci(al, t, "sharpe")
            v = "netbeta " + f"{beta[era]:+.3f}"
            print(f"  {N:<5}{era:<8}{beta[era]:<+10.3f}"
                  f"{f'{al.mean()*1e4:+.2f} [{mlo:+.2f},{mhi:+.2f}]':<26}"
                  f"{f'{al.mean()/al.std()*np.sqrt(PYR):+.2f} [{slo:+.2f},{shi:+.2f}]':<24}", flush=True)
        print("", flush=True)
    # (3) mechanism: BETWEEN-rvol IC in top-40, day-clustered CI
    print("(3) MECHANISM — structural low-vol (BETWEEN-rvol) IC in top-40, day-clustered CI:", flush=True)
    top40 = set(ranked[:40]); P = PAN.sort_values(["symbol", "open_time"]).copy()
    P["btw"] = P.groupby("symbol")["rvol_7d"].transform(lambda s: s.rolling(360, min_periods=60).mean().shift(1))
    for era, cutoff, ge in (("OOS", pd.Timestamp("2025-10-01", tz="UTC"), False),
                            ("RECENT", pd.Timestamp("2025-10-01", tz="UTC"), True)):
        d = P[(P["open_time"] >= cutoff) if ge else (P["open_time"] < cutoff)]
        d = d[d["symbol"].isin(top40)].dropna(subset=["btw", "alpha_vs_btc_realized"])
        ic = d.groupby("open_time").apply(
            lambda g: spearmanr(g["btw"], g["alpha_vs_btc_realized"]).correlation if len(g) >= 8 else np.nan,
            include_groups=False).dropna()
        lo, hi = day_ci(ic.to_numpy(), ic.index.to_numpy(), "mean")
        print(f"  {era}: BETWEEN-rvol IC {ic.mean():+.4f} [day-CI {lo/1e4:+.4f},{hi/1e4:+.4f}] "
              f"({'CI excl 0' if (lo>0 or hi<0) else 'spans 0'})", flush=True)
    print("\nVALIDATEDONE", flush=True)


if __name__ == "__main__":
    main()
