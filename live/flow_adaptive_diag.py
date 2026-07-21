"""Adaptive-feasibility diagnostic — is the NON-STATIONARY signal's IC forecastable from its OWN past?

Any adaptive scheme (rolling recalibration / EWMA coef / "trade the trailing-window sign") is a bet that
sign(IC over trailing window) predicts sign(IC over the NEXT window). Test it honestly:
  * the non-stationary object = signed_pressure PARTIAL-IC vs returns+vol (iter4 showed THIS flips sign;
    the RAW IC does not). Trade-relevant = the vol-neutralized signal.
  * NON-OVERLAPPING calendar windows (overlapping windows fake positive autocorrelation).
  * per window: mean partial-IC; then lag-1 autocorrelation rho(IC_t, IC_{t+1}), sign hit-rate, and an
    adaptive-vs-static value check: mean(sign(IC_t)*IC_{t+1}) [trade trailing sign] vs mean(IC_{t+1})
    [static fixed sign] vs mean(|IC_{t+1}|) [oracle who knows next sign].
Small N (=span/W windows) => report SE(rho) ~ 1/sqrt(N-1). If rho ~0/neg and sign-hit ~50% and
adaptive <= static, adaptive cannot rescue it. Prior (vBTC IC-selector): rho(past->future IC) ~ +0.11, value-negative.
"""
import os, glob
import numpy as np, pandas as pd
os.environ.setdefault("MALLOC_ARENA_MAX", "2"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys; sys.path.insert(0, "/home/yuqing/ctaNew")
from live.flow_harness import partial_xsic
from live.flow_iter2_vol import _load_sym, RET, VOL

FEAT = "signed_pressure_5min"


def window_ics(D, tgt, controls, W, min_ts=20):
    bt = D["bar_time"]
    t0 = bt.min()
    wid = ((bt - t0).dt.days // W).to_numpy()
    out = {}
    for w in np.unique(wid):
        ic = partial_xsic(D, FEAT, controls, tgt, row_mask=(wid == w))
        if len(ic) >= min_ts:
            out[int(w)] = float(ic.mean())
    return pd.Series(out).sort_index()


def lag1(wic):
    v = wic.dropna().to_numpy()
    if len(v) < 4:
        return dict(n=len(v), rho=np.nan, se=np.nan, hit=np.nan,
                    adaptive=np.nan, static=np.nan, oracle=np.nan)
    a, b = v[:-1], v[1:]
    rho = float(np.corrcoef(a, b)[0, 1])
    return dict(n=len(v), rho=rho, se=1 / np.sqrt(len(a)),
                hit=float(np.mean(np.sign(a) == np.sign(b))),
                adaptive=float(np.mean(np.sign(a) * b)),   # trade trailing sign
                static=float(np.mean(b)),                  # always same fixed sign
                oracle=float(np.mean(np.abs(b))),          # knows next sign
                lo=float(v.min()), hi=float(v.max()), mean=float(v.mean()))


def main():
    syms = sorted(p.split("/")[-1][:-8] for p in glob.glob(
        "/home/yuqing/ctaNew/data/ml/cache/research/flow_slim_v3/*.parquet"))
    D = pd.concat([x for x in (_load_sym(s) for s in syms) if x is not None], ignore_index=True)
    for c in D.columns:
        if D[c].dtype == np.float64:
            D[c] = D[c].astype(np.float32)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)
    print(f"panel {len(D):,} rows | {D.bar_time.min().date()}..{D.bar_time.max().date()}\n")
    print(f"OBJECT: {FEAT} PARTIAL-IC vs returns+vol (the non-stationary component)\n")

    for tgt in ["fwd_5m", "fwd_1h"]:
        print(f"### target {tgt}")
        for W in [30, 60, 90]:
            r = lag1(window_ics(D, tgt, RET + VOL, W))
            if not np.isfinite(r["rho"]):
                print(f"  W={W:>3}d: too few windows (n={r['n']})"); continue
            verdict = ("FORECASTABLE" if (r["rho"] - r["se"] > 0 and r["hit"] > 0.5
                       and r["adaptive"] > abs(r["static"])) else "not forecastable")
            print(f"  W={W:>3}d N={r['n']:>2} | IC/window [{r['lo']:+.4f},{r['hi']:+.4f}] mean {r['mean']:+.4f}")
            print(f"          rho(IC_t,IC_t+1)={r['rho']:+.2f} (SE~{r['se']:.2f}) | "
                  f"sign-hit {r['hit']:.0%} | adaptive {r['adaptive']:+.4f} vs static {r['static']:+.4f} "
                  f"vs oracle {r['oracle']:+.4f}  => {verdict}", flush=True)
        print()
    print("read: adaptive can work ONLY if rho robustly >0 (beyond SE), sign-hit >>50%, AND adaptive>|static|.")
    print("ADAPTIVEDIAGDONE", flush=True)


if __name__ == "__main__":
    main()
