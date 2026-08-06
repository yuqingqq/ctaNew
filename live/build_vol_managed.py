"""Research cycle 3: VOLATILITY-MANAGED book (Barroso & Santa-Clara 2015 "Momentum Has Its Moments";
Moreira-Muir 2017 JF). Scale the L/S book by inverse trailing realized variance x target constant.
Canonical claim: kills crashes (skew/kurt/worst/drawdown) AND raises Sharpe.

Critique to honor (Cederburg 2020; Barroso-Detzel 2021): OOS estimation error + turnover cost erode the
gain. So: scaling weight is STRICTLY PIT (trailing var, shift(1)); test BOTH eras; block-bootstrap CI on
the Sharpe change; report the EXTRA turnover the re-levering costs. Sharpe is invariant to the target
constant, so only the time-varying 1/var part is tested. Shape metrics (worst/drawdown) computed at
MATCHED unit vol so we compare tail SHAPE not leverage.

Books: (A) quintile L/S (clean mechanism), (B) deployed top-K=3 band + era-locked beta-hedge (actionable).
Run: python3 -u -m live.build_vol_managed
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from live.v0_feature_ablation import build_panel, V0, RECENT_CUTS, OOS_CUTS
from live.build_alpha_beta_decomp import gen_pred, FULL
from live.build_deployed_band import band_topk, turnover

PYR = 6 * 365.0
K, M = 3, 8
WINS = [30, 120, 240]          # ~5d / 20d / 40d at 4h bars
CAP = 3.0
RNG = np.random.default_rng(0)


def perbar(d, pos, min_leg=2):
    """Return sorted (times, ls, mkt) per bar for a position vector aligned to d's rows."""
    bt = d["open_time"].to_numpy("datetime64[ns]"); rp = d["return_pct"].to_numpy()
    codes, uniq = pd.factorize(bt, sort=True); k = len(uniq)
    nl = np.bincount(codes, (pos == 1).astype(float), k); ns = np.bincount(codes, (pos == -1).astype(float), k)
    sl = np.bincount(codes, np.where(pos == 1, rp, 0.0), k); ss = np.bincount(codes, np.where(pos == -1, rp, 0.0), k)
    na = np.bincount(codes, minlength=k); sa = np.bincount(codes, rp, k)
    ok = (nl >= min_leg) & (ns >= min_leg)
    ls = sl[ok] / np.maximum(nl[ok], 1) - ss[ok] / np.maximum(ns[ok], 1)
    mkt = sa[ok] / np.maximum(na[ok], 1)
    return uniq[ok], ls, mkt


def pos_quintile(d, k=0.2):
    r = d.groupby("open_time")["pred"].rank(pct=True).to_numpy()
    pos = np.zeros(len(d)); pos[r >= 1 - k] = 1; pos[r <= k] = -1
    return pos


def pos_band(d):
    d["rhi"] = d.groupby("open_time")["pred"].rank(ascending=False, method="first")
    d["n"] = d.groupby("open_time")["pred"].transform("size"); d["rlo"] = d["n"] + 1 - d["rhi"]
    return np.concatenate([band_topk(g["rhi"].to_numpy(), g["rlo"].to_numpy(), K, M)
                           for _, g in d.groupby("symbol", sort=False)])


def vol_scale(ls, win, cap=CAP):
    """PIT inverse-vol leverage: w_t = median(sig)/sig_{t-1}, capped. Warmup -> 1.0 (no scaling)."""
    s = pd.Series(ls)
    sig = s.rolling(win).std().shift(1)
    med = np.nanmedian(sig)
    w = (med / sig).clip(upper=cap)
    w = w.fillna(1.0).to_numpy()
    return w * ls, w


def shape(x):
    """Sharpe (own scale) + skew, excess-kurt, worst-bar & max-drawdown at MATCHED unit vol."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    sh = x.mean() / x.std() * np.sqrt(PYR)
    xu = x / x.std()
    cum = np.cumsum(xu); dd = float((np.maximum.accumulate(cum) - cum).max())
    return sh, float(skew(x)), float(kurtosis(x)), float(xu.min()), dd


def block_sharpe_ci(raw, scaled, block=30, nb=3000):
    """Paired block-bootstrap CI on Sharpe(scaled) - Sharpe(raw) (per-bar aligned)."""
    n = len(raw); nblk = int(np.ceil(n / block)); diffs = np.empty(nb)
    for i in range(nb):
        starts = RNG.integers(0, max(n - block + 1, 1), nblk)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        r = raw[idx]; sc = scaled[idx]
        diffs[i] = sc.mean() / sc.std() - r.mean() / r.std()
    d = diffs * np.sqrt(PYR)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run_book(label, series, betas):
    """series: {era: (times, ls, mkt)}; betas: era-locked hedge betas keyed by era to APPLY."""
    print(f"===== {label} =====", flush=True)
    scaled_store = {}
    for era in ("RECENT", "OOS"):
        t, ls, mkt = series[era]
        book = ls - betas[era] * mkt if betas else ls          # beta-hedged if requested
        sh0, sk0, ku0, w0, dd0 = shape(book)
        print(f"  {era}: RAW      Sh {sh0:+.2f} | skew {sk0:+.2f} kurt {ku0:5.1f} | worst {w0:+.2f} maxDD {dd0:6.1f}",
              flush=True)
        best = None
        for win in WINS:
            sc, w = vol_scale(book, win)
            sh, sk, ku, wo, dd = shape(sc)
            xturn = np.nanmean(np.abs(np.diff(w)))             # extra book turnover from re-levering
            print(f"       volmgd w={win:<3} Sh {sh:+.2f} | skew {sk:+.2f} kurt {ku:5.1f} | worst {wo:+.2f} "
                  f"maxDD {dd:6.1f} | dSh {sh - sh0:+.2f} | xturn {xturn:.2f}", flush=True)
            if best is None or win == 120:                     # pick 20d window a priori (Barroso-robust)
                best = (win, book, sc)
        scaled_store[era] = best
    # honest CI on the a-priori 120-bar window, both eras
    for era in ("RECENT", "OOS"):
        win, raw, sc = scaled_store[era]
        lo, hi = block_sharpe_ci(raw[~np.isnan(raw)], sc[~np.isnan(sc)])
        v = "improves" if lo > 0 else ("hurts" if hi < 0 else "CI spans 0 (null)")
        print(f"  {era}: dSharpe(w={win}) 95% CI [{lo:+.2f}, {hi:+.2f}] -> {v}", flush=True)
    print("", flush=True)


def main():
    PAN = build_panel()
    RP = pd.read_parquet(FULL, columns=["symbol", "open_time", "return_pct"])
    RP["open_time"] = pd.to_datetime(RP["open_time"], utc=True)
    qseries, bseries = {}, {}
    for era, cuts in (("RECENT", RECENT_CUTS), ("OOS", OOS_CUTS)):
        pred = gen_pred(PAN, list(V0), cuts)
        pred["open_time"] = pd.to_datetime(pred["open_time"], utc=True)
        d = pred.merge(RP, on=["symbol", "open_time"], how="inner").dropna().sort_values(["symbol", "open_time"])
        qseries[era] = perbar(d, pos_quintile(d))
        bseries[era] = perbar(d, pos_band(d))
    # era-locked hedge betas for the deployed book (fit on the OTHER era)
    other = {"RECENT": "OOS", "OOS": "RECENT"}
    b_in = {e: np.polyfit(bseries[e][2], bseries[e][1], 1)[0] for e in bseries}
    b_apply = {e: b_in[other[e]] for e in bseries}
    print(f"deployed L/S beta (in-era): RECENT {b_in['RECENT']:+.3f} OOS {b_in['OOS']:+.3f}\n", flush=True)
    run_book("A) quintile L/S (clean mechanism, unhedged)", qseries, None)
    run_book("B) deployed top-K=3 band + era-locked beta-hedge", bseries, b_apply)
    print("VOLMGDDONE", flush=True)


if __name__ == "__main__":
    main()
